# Connection-point regressor (Option B): per-vertex 7-channel head + losses +
# training loop. Predicts (heatmap, offset xyz, direction xyz) per vertex; the
# decoder in cp_targets turns that into discrete terminal-block detections.
#
# Two backbones are provided:
#   * build_diffusionnet_regressor() – the production head: a DiffusionNet with
#     C_out=7 (reuses diffusionnet.precompute_operators / _forward). Needs the
#     `diffusion_net` package.
#   * CPMLP – a light per-vertex MLP on normalised xyz that needs only torch.
#     Used for smoke-tests / environments without the diffusion_net compile
#     chain. It has no geometric receptive field, so it is NOT the production
#     model, but it exercises the full target/loss/decode pipeline.
#
# torch is imported lazily (like diffusionnet.py) so the data utilities remain
# importable without it.

import logging
import numpy as np

import cp_targets as ct

logger = logging.getLogger(__name__)

N_CHANNELS = ct.N_CHANNELS  # 7


def _require_torch():
    try:
        import torch
        return torch
    except ImportError as exc:
        raise ImportError("'torch' is required for cp_regressor "
                          "(pip install torch)") from exc


# ---------------------------------------------------------------------------
# normalisation: center + scale vertices so the MLP/optimiser see ~unit coords
# ---------------------------------------------------------------------------

def normalize_vertices(V):
    """Center on centroid and scale by bounding-box diagonal. Returns (Vn, center, scale)."""
    V = np.asarray(V, dtype=np.float64)
    center = V.mean(0)
    diag = float(np.linalg.norm(V.max(0) - V.min(0)))
    scale = diag if diag > 0 else 1.0
    return (V - center) / scale, center, scale


# ---------------------------------------------------------------------------
# light MLP backbone (smoke-test / no diffusion_net)
# ---------------------------------------------------------------------------

def build_cpmlp(width=128, depth=4, c_in=3):
    """Per-vertex MLP: c_in -> ... -> 7. torch.nn.Module."""
    _require_torch()
    import torch.nn as nn

    class CPMLP(nn.Module):
        def __init__(self):
            super().__init__()
            layers = [nn.Linear(c_in, width), nn.ReLU()]
            for _ in range(depth - 1):
                layers += [nn.Linear(width, width), nn.ReLU()]
            layers += [nn.Linear(width, N_CHANNELS)]
            self.net = nn.Sequential(*layers)

        def forward(self, x):
            return self.net(x)

    return CPMLP()


# ---------------------------------------------------------------------------
# production backbone: DiffusionNet with a 7-channel regression head
# ---------------------------------------------------------------------------

def build_diffusionnet_regressor(config=None):
    """Build a DiffusionNet with C_out=7 (regression). Returns (model, meta)."""
    import diffusionnet as dnmod
    _require_torch()
    dn = dnmod._require("diffusion_net", "pip install git+https://github.com/nmwsharp/diffusion-net")
    cfg = {**dnmod.DEFAULTS, **(config or {})}
    c_in = 3 if cfg.get("input_features", "xyz") == "xyz" else 16
    model = dn.layers.DiffusionNet(
        C_in=c_in, C_out=N_CHANNELS,
        C_width=int(cfg.get("c_width", 128)),
        N_block=int(cfg.get("n_diffusion_blocks", 4)),
        last_activation=None,            # raw outputs; we apply sigmoid/normalise in post
        outputs_at="vertices",
        dropout=float(cfg.get("dropout", 0.0)) > 0.0,
    )
    meta = {"input_features": cfg.get("input_features", "xyz"),
            "k_eig": int(cfg.get("n_eig", 128)), "hks_count": 16,
            # structural params -> a checkpoint is self-describing and rebuildable
            "c_in": c_in, "c_width": int(cfg.get("c_width", 128)),
            "n_block": int(cfg.get("n_diffusion_blocks", 4)),
            "dropout": float(cfg.get("dropout", 0.0))}
    return model, meta


# ---------------------------------------------------------------------------
# checkpointing: persist / restore a trained DiffusionNet regressor
# ---------------------------------------------------------------------------
# A checkpoint is {state_dict, meta}. meta carries the structural params
# (c_in/c_width/n_block/dropout/input_features/k_eig) so load_diffusionnet can
# rebuild the exact architecture before loading weights -- no separate config
# needed to run inference later.

def save_diffusionnet(model, meta, path):
    """Save a trained DiffusionNet regressor (weights + meta) to `path`."""
    torch = _require_torch()
    import os
    d = os.path.dirname(os.path.abspath(path))
    if d:
        os.makedirs(d, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "meta": meta}, path)
    logger.info("saved DiffusionNet checkpoint -> %s", path)


def load_diffusionnet(path, device="cpu"):
    """Rebuild a DiffusionNet regressor from a checkpoint. Returns (model, meta)."""
    torch = _require_torch()
    ckpt = torch.load(path, map_location=device, weights_only=False)
    meta = ckpt["meta"]
    cfg = {"input_features": meta["input_features"], "c_width": meta["c_width"],
           "n_diffusion_blocks": meta["n_block"], "n_eig": meta["k_eig"],
           "dropout": meta.get("dropout", 0.0)}
    model, _ = build_diffusionnet_regressor(cfg)
    model.load_state_dict(ckpt["state_dict"])
    model = model.to(device)
    model.eval()
    logger.info("loaded DiffusionNet checkpoint <- %s", path)
    return model, meta


# ---------------------------------------------------------------------------
# loss: heatmap BCE + masked offset L1 + masked direction cosine
# ---------------------------------------------------------------------------

def cp_loss(pred, target, mask, w_heat=1.0, w_off=1.0, w_dir=1.0,
            heat_pos_weight=50.0, heat_loss="bce", focal_gamma=2.0):
    """Combined regression loss.

    pred   : (N,7) raw model output (heatmap channel is a logit).
    target : (N,7) from prepare_sample (heatmap in [0,1]; offset in NORMALISED
             units, i.e. mm/scale -- see prepare_sample). Because the offset is
             already scale-normalised, no per-mesh rescaling happens here.
    mask   : (N,) bool, vertices near a CP (offset/direction supervised there).
    heat_loss : "bce"   -> per-vertex BCE weighted by (1 + heat_pos_weight*h) so
                           the few high-heat vertices are not drowned out;
                "focal" -> quality-focal loss |h - sigmoid(logit)|^gamma * BCE,
                           which down-weights the easy ~0 background and tends to
                           improve precision on the sparse-keypoint heatmap.
    """
    torch = _require_torch()
    import torch.nn.functional as Fnn
    heat_logit = pred[:, ct.HEATMAP]
    heat_tgt = target[:, ct.HEATMAP]
    if heat_loss == "focal":
        bce = Fnn.binary_cross_entropy_with_logits(heat_logit, heat_tgt,
                                                   reduction="none")
        mod = (heat_tgt - torch.sigmoid(heat_logit)).abs().pow(focal_gamma)
        loss_heat = (mod * bce).mean()
    else:
        heat_w = 1.0 + heat_pos_weight * heat_tgt
        loss_heat = Fnn.binary_cross_entropy_with_logits(
            heat_logit, heat_tgt, weight=heat_w)

    if mask.any():
        m = mask
        loss_off = Fnn.l1_loss(pred[m][:, ct.OFFSET], target[m][:, ct.OFFSET])
        dir_pred = Fnn.normalize(pred[m][:, ct.DIRECTION], dim=-1, eps=1e-8)
        dir_tgt = Fnn.normalize(target[m][:, ct.DIRECTION], dim=-1, eps=1e-8)
        loss_dir = (1.0 - (dir_pred * dir_tgt).sum(-1)).mean()
    else:
        loss_off = pred.sum() * 0.0
        loss_dir = pred.sum() * 0.0

    total = w_heat * loss_heat + w_off * loss_off + w_dir * loss_dir
    return total, {"heat": float(loss_heat.detach()), "off": float(loss_off.detach()),
                   "dir": float(loss_dir.detach()), "total": float(total.detach())}


def pred_to_array(pred, offset_scale=1.0):
    """Convert raw model output -> (N,7) numpy ready for cp_targets.decode_predictions
    (sigmoid on heatmap, unit-normalise direction).

    offset_scale : the model predicts offsets in NORMALISED units; multiply by the
    mesh scale (mm) here so decode_predictions, which adds the offset to the raw
    mm vertices, sees millimetres. Pass the sample's 'scale'.
    """
    torch = _require_torch()
    import torch.nn.functional as Fnn
    out = pred.detach().clone()
    out[:, ct.HEATMAP] = torch.sigmoid(out[:, ct.HEATMAP])
    out[:, ct.OFFSET] = out[:, ct.OFFSET] * float(offset_scale)
    out[:, ct.DIRECTION] = Fnn.normalize(out[:, ct.DIRECTION], dim=-1, eps=1e-8)
    return out.cpu().numpy()


# ---------------------------------------------------------------------------
# MLP training loop (smoke-test path). Samples: dicts with verts, target, mask, scale.
# ---------------------------------------------------------------------------

def train_cpmlp(samples, epochs=300, lr=1e-3, width=128, depth=4,
                device="cpu", log_every=50):
    """Overfit/smoke train the MLP backbone on prepared samples.

    samples: list of {'verts_norm' (N,3), 'target' (N,7), 'mask' (N,), 'scale'}.
    Returns the trained model.
    """
    torch = _require_torch()
    model = build_cpmlp(width=width, depth=depth).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    tensors = []
    for s in samples:
        tensors.append({
            "x": torch.tensor(s["verts_norm"], dtype=torch.float32, device=device),
            "t": torch.tensor(s["target"], dtype=torch.float32, device=device),
            "m": torch.tensor(s["mask"], dtype=torch.bool, device=device),
            "scale": float(s["scale"]),
        })
    for ep in range(epochs):
        model.train()
        tot = 0.0
        for s in tensors:
            opt.zero_grad(set_to_none=True)
            out = model(s["x"])
            loss, parts = cp_loss(out, s["t"], s["m"])
            loss.backward()
            opt.step()
            tot += parts["total"]
        if log_every and (ep % log_every == 0 or ep == epochs - 1):
            logger.info("epoch %3d  loss=%.5f", ep, tot / max(1, len(tensors)))
    return model


def prepare_sample(part, dedup=True):
    """Part -> training sample dict (normalised verts, encoded target, mask, scale).

    Uses deduped terminal-block locations as the regression targets. Carries the
    raw verts/faces too so the DiffusionNet backbone can build mesh operators.
    """
    import json_dataset as jd
    if dedup:
        _, bp, bd = jd.dedup_connection_points(part)
    else:
        bp, bd = part.cp_points, part.cp_directions
    Vn, center, scale = normalize_vertices(part.vertices)
    target, mask, sigma = ct.encode_targets(part.vertices, bp, bd)
    # #3: regress offsets in NORMALISED units (mm/scale) so the target is
    # scale-invariant and matches the scale-normalised input features. The mm
    # offset is recovered at decode via pred_to_array(offset_scale=scale).
    target = target.copy()
    target[:, ct.OFFSET] = target[:, ct.OFFSET] / scale
    return {"verts_norm": Vn, "verts": np.asarray(part.vertices, dtype=np.float64),
            "faces": np.asarray(part.faces, dtype=np.int64),
            "target": target, "mask": mask, "scale": scale,
            "center": center, "sigma": sigma, "gt_points": bp, "gt_directions": bd}


# ---------------------------------------------------------------------------
# production backbone: DiffusionNet training / inference (needs diffusion_net)
# ---------------------------------------------------------------------------
# The eigenbasis (mesh operators) is the expensive step at 429-file scale; it is
# computed ONCE per part and cached to op_cache_dir (diffusion_net hashes the
# vertices to key the cache), then reused across epochs and runs. Operators are
# built on the raw mm mesh; the *input features* are the normalised xyz so the
# first layer is well-conditioned regardless of part size. Offset targets are
# scale-normalised (mm/scale) to match the normalised input; pred_to_array
# multiplies them back by the mesh scale before decoding.

def _cp_operators(verts, faces, k_eig=128, op_cache_dir=None):
    """Mesh operators (mass/Laplacian/eigenbasis/gradients) for one part.

    Cached to op_cache_dir when given (recommended for the 429-file corpus).
    """
    torch = _require_torch()
    import diffusionnet as dnmod
    dn = dnmod._require(
        "diffusion_net",
        "clone https://github.com/nmwsharp/diffusion-net and add its src/ to "
        "PYTHONPATH; needs robust_laplacian + potpourri3d + scikit-learn")
    k_eig = int(min(int(k_eig), dnmod.MAX_K_EIG))
    V = torch.tensor(np.asarray(verts), dtype=torch.float32)
    F = torch.tensor(np.asarray(faces), dtype=torch.long)
    _, mass, L, evals, evecs, gradX, gradY = dn.geometry.get_operators(
        V, F, k_eig=k_eig, op_cache_dir=op_cache_dir)
    return {"verts": V, "faces": F, "mass": mass, "L": L, "evals": evals,
            "evecs": evecs, "gradX": gradX, "gradY": gradY}


def _to_device_ops(ops, device):
    return {k: (v.to(device) if hasattr(v, "to") else v) for k, v in ops.items()}


def train_diffusionnet_regressor(train_samples, config=None, epochs=200, lr=1e-3,
                                 device="cpu", op_cache_dir=None, log_every=1,
                                 eval_fn=None, eval_every=5,
                                 w_heat=1.0, w_off=1.0, w_dir=1.0,
                                 heat_pos_weight=50.0, heat_loss="bce",
                                 focal_gamma=2.0, max_gpu_verts=60000,
                                 ckpt_path=None, ckpt_every=0, seed=0,
                                 lr_decay_every=0, lr_decay_rate=0.5,
                                 accum_steps=1, low_memory=False):
    """Train the production DiffusionNet regressor (C_out=7).

    Each sample needs 'verts','faces','verts_norm','target','mask','scale'
    (see prepare_sample). Operators are precomputed/cached once per part, each
    part is moved to the device for its step then freed. Returns (model, meta).

    seed                 : seeds torch/np/random for reproducible runs (#4).
    lr_decay_every/rate  : StepLR schedule (epochs / gamma); 0 disables it (#4).
    accum_steps          : gradient accumulation over N parts before opt.step.
    eval_fn(model,meta,ep): optional callback run every `eval_every` epochs; if it
                           returns a truthy value training stops early (#5). It is
                           expected to own best-by-metric checkpointing.
    ckpt_path/ckpt_every : crash-resilience snapshot of the *latest* weights to
                           "<ckpt_path>.latest" every N epochs; the final/best at
                           ckpt_path is written here only when no eval_fn drives it.
    max_gpu_verts        : parts above this run on CPU (4 GB GPU OOMs on big
                           meshes); grads from the temp CPU copy are added back to
                           the GPU optimiser, identical to a GPU step.
    low_memory           : do NOT hold every part's eigenbasis resident; reload it
                           from op_cache_dir each step (bounded RAM for corpora
                           larger than memory, at the cost of cache I/O) (#7).
    """
    import random
    import copy
    torch = _require_torch()
    import diffusionnet as dnmod
    # #4: reproducibility -- the data split was already seeded; seed the optimiser/
    # shuffle/dropout too so runs are comparable.
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)

    model, meta = build_diffusionnet_regressor(config)
    model = model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sched = (torch.optim.lr_scheduler.StepLR(opt, step_size=int(lr_decay_every),
                                             gamma=float(lr_decay_rate))
             if lr_decay_every and lr_decay_every > 0 else None)
    accum = max(1, int(accum_steps))

    def _accumulate(ops, d, run_device, src_device, loss_div):
        """Forward+backward for one part, accumulating gradient into `model`
        (no zero_grad / no step here -- the caller steps at accumulation
        boundaries). Oversized/OOM parts run on a temporary copy on run_device
        and copy their gradient back to the model on src_device."""
        ops_d = _to_device_ops(ops, run_device)
        x = (d["x"].to(run_device) if meta["input_features"] == "xyz"
             else dnmod._model_input(ops_d, meta))
        t = d["t"].to(run_device); m = d["m"].to(run_device)
        if run_device == src_device:
            out = dnmod._forward(model, ops_d, x)
            loss, parts = cp_loss(out, t, m, w_heat=w_heat, w_off=w_off,
                                  w_dir=w_dir, heat_pos_weight=heat_pos_weight,
                                  heat_loss=heat_loss, focal_gamma=focal_gamma)
            (loss / loss_div).backward()
        else:
            cm = copy.deepcopy(model).to(run_device)
            out = dnmod._forward(cm, ops_d, x)
            loss, parts = cp_loss(out, t, m, w_heat=w_heat, w_off=w_off,
                                  w_dir=w_dir, heat_pos_weight=heat_pos_weight,
                                  heat_loss=heat_loss, focal_gamma=focal_gamma)
            (loss / loss_div).backward()
            for pg, pc in zip(model.parameters(), cm.parameters()):
                if pc.grad is not None:
                    g = pc.grad.detach().to(src_device)
                    pg.grad = g if pg.grad is None else (pg.grad + g)
            del cm
        del ops_d, x, t, m, out, loss
        return parts

    # #1: precompute operators with a per-part guard -- one malformed mesh
    # (degenerate faces, eigensolver non-convergence) must not kill the whole run.
    logger.info("DiffusionNet regressor: preparing operators for %d parts "
                "(k_eig=%d, cache=%s, low_memory=%s) ...", len(train_samples),
                meta["k_eig"], op_cache_dir, low_memory)
    prepared, skipped = [], 0
    for s in train_samples:
        item = {"x": torch.tensor(s["verts_norm"], dtype=torch.float32),
                "t": torch.tensor(s["target"], dtype=torch.float32),
                "m": torch.tensor(s["mask"], dtype=torch.bool),
                "scale": float(s["scale"]),
                "verts": s["verts"], "faces": s["faces"]}
        try:
            ops = _cp_operators(s["verts"], s["faces"], meta["k_eig"], op_cache_dir)
        except Exception as exc:                      # noqa: BLE001 (skip bad mesh)
            skipped += 1
            logger.warning("skipping part (operator build failed, %d verts): %s",
                           len(np.asarray(s["verts"])), exc)
            continue
        if not low_memory:
            item["ops"] = ops                          # keep resident
        del ops                                        # low_memory: only warmed cache
        prepared.append(item)
    if skipped:
        logger.warning("skipped %d/%d parts with bad geometry", skipped,
                       len(train_samples))
    if not prepared:
        raise RuntimeError("no usable parts after operator precompute")

    order = list(range(len(prepared)))
    for ep in range(epochs):
        model.train()
        random.shuffle(order)
        opt.zero_grad(set_to_none=True)
        tot = 0.0
        for k, i in enumerate(order, 1):
            d = prepared[i]
            ops = (_cp_operators(d["verts"], d["faces"], meta["k_eig"], op_cache_dir)
                   if low_memory else d["ops"])
            n = d["x"].shape[0]
            big = device != "cpu" and max_gpu_verts and n > max_gpu_verts
            try:
                parts = _accumulate(ops, d, "cpu" if big else device, device, accum)
            except (torch.cuda.OutOfMemoryError, RuntimeError) as exc:
                if "out of memory" not in str(exc).lower():
                    raise
                torch.cuda.empty_cache()
                logger.warning("CUDA OOM on %d-vertex part -> CPU fallback", n)
                parts = _accumulate(ops, d, "cpu", device, accum)
            if low_memory:
                del ops
            if k % accum == 0 or k == len(order):
                opt.step()
                opt.zero_grad(set_to_none=True)
            tot += parts["total"]
        if sched is not None:
            sched.step()
        if log_every and (ep % log_every == 0 or ep == epochs - 1):
            logger.info("epoch %3d  loss=%.5f", ep, tot / max(1, len(prepared)))
        if eval_fn is not None and (ep % eval_every == 0 or ep == epochs - 1):
            if eval_fn(model, meta, ep):               # truthy -> early stop (#5)
                logger.info("early stopping at epoch %d", ep)
                break
        # crash-resilience snapshot of the latest weights (separate from best)
        if ckpt_path and ckpt_every and ep % ckpt_every == 0 and ep > 0:
            save_diffusionnet(model, meta, ckpt_path + ".latest")
    # if no eval_fn is driving best-checkpointing, persist the final weights
    if ckpt_path and eval_fn is None:
        save_diffusionnet(model, meta, ckpt_path)
    return model, meta


def infer_diffusionnet(model, meta, verts, faces, verts_norm,
                       op_cache_dir=None, device="cpu", max_gpu_verts=60000,
                       offset_scale=1.0):
    """Run a trained DiffusionNet regressor on one part -> (N,7) numpy array
    ready for cp_targets.decode_predictions.

    offset_scale : mesh scale (mm); the model emits normalised offsets, so this
    converts them back to millimetres for the decoder (pass the sample 'scale').
    Like training, oversized meshes (or a CUDA OOM) fall back to the CPU so a
    4 GB GPU does not crash on the largest parts; results are identical.
    """
    torch = _require_torch()
    import diffusionnet as dnmod
    import copy
    n = len(np.asarray(verts))
    run_device = ("cpu" if (device != "cpu" and max_gpu_verts and n > max_gpu_verts)
                  else device)

    def _run(rd):
        ops = _to_device_ops(_cp_operators(verts, faces, meta["k_eig"], op_cache_dir), rd)
        x = (torch.tensor(np.asarray(verts_norm), dtype=torch.float32, device=rd)
             if meta["input_features"] == "xyz" else dnmod._model_input(ops, meta))
        mdl = model if rd == device else copy.deepcopy(model).to(rd)
        mdl.eval()
        with torch.no_grad():
            out = dnmod._forward(mdl, ops, x)
        return pred_to_array(out, offset_scale=offset_scale)

    try:
        return _run(run_device)
    except (torch.cuda.OutOfMemoryError, RuntimeError) as exc:
        if "out of memory" not in str(exc).lower():
            raise
        torch.cuda.empty_cache()
        logger.warning("CUDA OOM during inference on %d-vertex part -> CPU", n)
        return _run("cpu")


# ===========================================================================
# kNN-graph backbone (native-Windows: NO robust_laplacian / potpourri3d)
# ===========================================================================
# A torch-only geometric backbone for environments where the DiffusionNet native
# wheels segfault (Windows). The "operator" is a k-nearest-neighbour graph built
# with scipy.spatial.cKDTree; message passing is plain torch (EdgeConv: each
# vertex aggregates an MLP of (h_i, h_j - h_i) over its neighbours, max-pooled).
# Stacking blocks with residuals grows a real geometric receptive field, so --
# unlike the bare MLP -- it generalises across parts. It reuses the SAME target
# encoding / cp_loss / decode / metrics as the other backbones; only the network
# and the operator differ. Deps: torch + scipy + numpy (all clean Windows wheels).

KNN_DEFAULTS = {"c_width": 128, "n_layers": 4, "k": 16}


def _knn_graph(verts, k=16):
    """(N,k) neighbour-index tensor from a point set, via scipy cKDTree.

    Built on the (uniformly) normalised coords -- uniform scale/translation
    preserves nearest neighbours, so the graph is scale-invariant. k is clamped
    to N-1 for tiny meshes. Needs no mesh faces (works on point clouds too).
    """
    torch = _require_torch()
    from scipy.spatial import cKDTree
    V = np.asarray(verts, dtype=np.float64)
    n = len(V)
    kq = int(min(k + 1, n))                  # +1: query returns the point itself
    tree = cKDTree(V)
    _, idx = tree.query(V, k=kq)
    idx = np.atleast_2d(idx)
    if idx.shape[1] > 1:
        idx = idx[:, 1:]                     # drop self-neighbour (column 0)
    return torch.tensor(np.ascontiguousarray(idx), dtype=torch.long)


def build_knngraph_regressor(config=None):
    """Build the EdgeConv kNN-graph regressor (C_out=7). Returns (model, meta)."""
    _require_torch()
    import torch
    import torch.nn as nn
    cfg = {**KNN_DEFAULTS, **(config or {})}
    width = int(cfg["c_width"]); n_layers = int(cfg["n_layers"]); c_in = 3

    class EdgeConv(nn.Module):
        """h_i' = max_j MLP([h_i, h_j - h_i]) over the kNN neighbours j of i."""
        def __init__(self, c_in, c_out):
            super().__init__()
            self.mlp = nn.Sequential(nn.Linear(2 * c_in, c_out), nn.ReLU(),
                                     nn.Linear(c_out, c_out))

        def forward(self, h, nbr_idx):
            hj = h[nbr_idx]                              # (N, k, C)
            hi = h.unsqueeze(1).expand_as(hj)           # (N, k, C)
            edge = torch.cat([hi, hj - hi], dim=-1)     # (N, k, 2C)
            return self.mlp(edge).amax(dim=1)           # (N, C') max aggregation

    class KNNGraphNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.inp = nn.Sequential(nn.Linear(c_in, width), nn.ReLU())
            self.blocks = nn.ModuleList([EdgeConv(width, width) for _ in range(n_layers)])
            self.head = nn.Linear(width, N_CHANNELS)

        def forward(self, x, nbr_idx):
            h = self.inp(x)
            for blk in self.blocks:
                h = h + blk(h, nbr_idx)                  # residual -> stable depth
            return self.head(h)

    meta = {"backbone": "knngraph", "input_features": "xyz", "c_in": c_in,
            "c_width": width, "n_layers": n_layers, "k": int(cfg["k"])}
    return KNNGraphNet(), meta


def save_knngraph(model, meta, path):
    """Save a trained kNN-graph regressor (weights + meta) to `path`."""
    torch = _require_torch()
    import os
    d = os.path.dirname(os.path.abspath(path))
    if d:
        os.makedirs(d, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "meta": meta}, path)
    logger.info("saved kNN-graph checkpoint -> %s", path)


def load_knngraph(path, device="cpu"):
    """Rebuild a kNN-graph regressor from a checkpoint. Returns (model, meta)."""
    torch = _require_torch()
    ckpt = torch.load(path, map_location=device, weights_only=False)
    meta = ckpt["meta"]
    model, _ = build_knngraph_regressor(
        {"c_width": meta["c_width"], "n_layers": meta["n_layers"], "k": meta["k"]})
    model.load_state_dict(ckpt["state_dict"])
    model = model.to(device); model.eval()
    logger.info("loaded kNN-graph checkpoint <- %s", path)
    return model, meta


def train_knngraph_regressor(train_samples, config=None, epochs=200, lr=1e-3,
                             device="cpu", log_every=1, eval_fn=None, eval_every=5,
                             w_heat=1.0, w_off=1.0, w_dir=1.0, heat_pos_weight=50.0,
                             heat_loss="bce", focal_gamma=2.0, max_gpu_verts=60000,
                             ckpt_path=None, ckpt_every=0, seed=0,
                             lr_decay_every=0, lr_decay_rate=0.5, accum_steps=1):
    """Train the kNN-graph regressor. Mirrors train_diffusionnet_regressor (seed,
    StepLR, gradient accumulation, GPU/CPU-hybrid for big meshes, eval_fn early
    stop, best/latest checkpoints) but the per-part operator is a cheap kNN graph
    held in RAM (no eigenbasis, no op cache). Returns (model, meta)."""
    import random
    import copy
    torch = _require_torch()
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)

    model, meta = build_knngraph_regressor(config)
    model = model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sched = (torch.optim.lr_scheduler.StepLR(opt, step_size=int(lr_decay_every),
                                             gamma=float(lr_decay_rate))
             if lr_decay_every and lr_decay_every > 0 else None)
    accum = max(1, int(accum_steps))

    def _accumulate(d, run_device, src_device, loss_div):
        x = d["x"].to(run_device); nb = d["nbr"].to(run_device)
        t = d["t"].to(run_device); m = d["m"].to(run_device)
        if run_device == src_device:
            out = model(x, nb)
            loss, parts = cp_loss(out, t, m, w_heat=w_heat, w_off=w_off, w_dir=w_dir,
                                  heat_pos_weight=heat_pos_weight, heat_loss=heat_loss,
                                  focal_gamma=focal_gamma)
            (loss / loss_div).backward()
        else:
            cm = copy.deepcopy(model).to(run_device)
            out = cm(x, nb)
            loss, parts = cp_loss(out, t, m, w_heat=w_heat, w_off=w_off, w_dir=w_dir,
                                  heat_pos_weight=heat_pos_weight, heat_loss=heat_loss,
                                  focal_gamma=focal_gamma)
            (loss / loss_div).backward()
            for pg, pc in zip(model.parameters(), cm.parameters()):
                if pc.grad is not None:
                    g = pc.grad.detach().to(src_device)
                    pg.grad = g if pg.grad is None else (pg.grad + g)
            del cm
        del x, nb, t, m, out, loss
        return parts

    logger.info("kNN-graph regressor: building graphs for %d parts (k=%d) ...",
                len(train_samples), meta["k"])
    prepared, skipped = [], 0
    for s in train_samples:
        try:
            nbr = _knn_graph(s["verts_norm"], meta["k"])
        except Exception as exc:                       # noqa: BLE001
            skipped += 1
            logger.warning("skipping part (kNN graph failed): %s", exc)
            continue
        prepared.append({"nbr": nbr,
                         "x": torch.tensor(s["verts_norm"], dtype=torch.float32),
                         "t": torch.tensor(s["target"], dtype=torch.float32),
                         "m": torch.tensor(s["mask"], dtype=torch.bool)})
    if skipped:
        logger.warning("skipped %d/%d parts", skipped, len(train_samples))
    if not prepared:
        raise RuntimeError("no usable parts for kNN-graph training")

    order = list(range(len(prepared)))
    for ep in range(epochs):
        model.train()
        random.shuffle(order)
        opt.zero_grad(set_to_none=True)
        tot = 0.0
        for k, i in enumerate(order, 1):
            d = prepared[i]
            n = d["x"].shape[0]
            big = device != "cpu" and max_gpu_verts and n > max_gpu_verts
            try:
                parts = _accumulate(d, "cpu" if big else device, device, accum)
            except (torch.cuda.OutOfMemoryError, RuntimeError) as exc:
                if "out of memory" not in str(exc).lower():
                    raise
                torch.cuda.empty_cache()
                logger.warning("CUDA OOM on %d-vertex part -> CPU fallback", n)
                parts = _accumulate(d, "cpu", device, accum)
            if k % accum == 0 or k == len(order):
                opt.step()
                opt.zero_grad(set_to_none=True)
            tot += parts["total"]
        if sched is not None:
            sched.step()
        if log_every and (ep % log_every == 0 or ep == epochs - 1):
            logger.info("epoch %3d  loss=%.5f", ep, tot / max(1, len(prepared)))
        if eval_fn is not None and (ep % eval_every == 0 or ep == epochs - 1):
            if eval_fn(model, meta, ep):
                logger.info("early stopping at epoch %d", ep)
                break
        if ckpt_path and ckpt_every and ep % ckpt_every == 0 and ep > 0:
            save_knngraph(model, meta, ckpt_path + ".latest")
    if ckpt_path and eval_fn is None:
        save_knngraph(model, meta, ckpt_path)
    return model, meta


def infer_knngraph(model, meta, verts_norm, device="cpu", max_gpu_verts=60000,
                   offset_scale=1.0):
    """Run a trained kNN-graph regressor on one part -> (N,7) numpy array ready
    for cp_targets.decode_predictions. Needs only normalised vertices (no faces).
    Oversized meshes / CUDA OOM fall back to CPU, identical result."""
    torch = _require_torch()
    import copy
    n = len(np.asarray(verts_norm))
    run_device = ("cpu" if (device != "cpu" and max_gpu_verts and n > max_gpu_verts)
                  else device)

    def _run(rd):
        nbr = _knn_graph(verts_norm, meta["k"]).to(rd)
        x = torch.tensor(np.asarray(verts_norm), dtype=torch.float32, device=rd)
        mdl = model if rd == device else copy.deepcopy(model).to(rd)
        mdl.eval()
        with torch.no_grad():
            out = mdl(x, nbr)
        return pred_to_array(out, offset_scale=offset_scale)

    try:
        return _run(run_device)
    except (torch.cuda.OutOfMemoryError, RuntimeError) as exc:
        if "out of memory" not in str(exc).lower():
            raise
        torch.cuda.empty_cache()
        logger.warning("CUDA OOM during inference on %d-vertex part -> CPU", n)
        return _run("cpu")


def _selftest():
    """Overfit the MLP on one synthetic plate; check it recovers the CPs."""
    torch = _require_torch()
    torch.manual_seed(0)
    np.random.seed(0)
    import json_dataset as jd
    # synthetic plate part
    xs, ys = np.meshgrid(np.linspace(0, 100, 40), np.linspace(0, 100, 40))
    V = np.column_stack([xs.ravel(), ys.ravel(), np.zeros(xs.size)])
    F = np.zeros((0, 3), dtype=np.int64)
    cp_pts = np.array([[25, 25, 0], [75, 30, 0], [50, 80, 0]], float)
    cp_dir = np.array([[0, 0, 1.0]] * 3)
    part = jd.Part("SYN", V, F, cp_pts, cp_dir, ["a", "b", "c"])
    s = prepare_sample(part, dedup=False)
    model = train_cpmlp([s], epochs=400, lr=1e-3, log_every=0)
    model.eval()
    with torch.no_grad():
        out = model(torch.tensor(s["verts_norm"], dtype=torch.float32))
    arr = pred_to_array(out, offset_scale=s["scale"])
    got = ct.decode_predictions(V, arr, heatmap_thresh=0.5, nms_radius_mm=15.0)
    print("cp_regressor selftest: recovered", len(got), "of 3 CPs")
    # each ground-truth CP should be matched by some prediction (recall=1)
    matched = 0
    for cp in cp_pts:
        d = min(np.linalg.norm(g["point"] - cp) for g in got) if got else 1e9
        if d <= 10.0:
            matched += 1
    for g in got:
        d = np.linalg.norm(cp_pts - g["point"], axis=1).min()
        assert g["direction"][2] > 0.9, "direction not outward (+z)"
        print("   point %s  locErr=%.2fmm  dir=%s" %
              (np.round(g["point"], 1), d, np.round(g["direction"], 2)))
    assert matched == 3, f"overfit MLP recovered only {matched}/3 GT CPs"
    print("cp_regressor selftest OK")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    _selftest()
