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
            "k_eig": int(cfg.get("n_eig", 128)), "hks_count": 16}
    return model, meta


# ---------------------------------------------------------------------------
# loss: heatmap BCE + masked offset L1 + masked direction cosine
# ---------------------------------------------------------------------------

def cp_loss(pred, target, mask, w_heat=1.0, w_off=1.0, w_dir=1.0, scale=1.0,
            heat_pos_weight=50.0):
    """Combined regression loss.

    pred   : (N,7) raw model output (heatmap channel is a logit).
    target : (N,7) from cp_targets.encode_targets (heatmap in [0,1], offset mm).
    mask   : (N,) bool, vertices near a CP (offset/direction supervised there).
    scale  : mesh scale (mm) used to make the offset L1 term unitless.
    heat_pos_weight : the heatmap is dominated by ~0 background; per-vertex BCE
        is weighted by (1 + heat_pos_weight * target_heat) so the few high-heat
        vertices around each connection point are not drowned out (the analogue
        of the segmentation head's class weights).
    """
    _require_torch()
    import torch.nn.functional as Fnn
    heat_logit = pred[:, ct.HEATMAP]
    heat_tgt = target[:, ct.HEATMAP]
    heat_w = 1.0 + heat_pos_weight * heat_tgt
    loss_heat = Fnn.binary_cross_entropy_with_logits(
        heat_logit, heat_tgt, weight=heat_w)

    if mask.any():
        m = mask
        off_pred = pred[m][:, ct.OFFSET] / scale
        off_tgt = target[m][:, ct.OFFSET] / scale
        loss_off = Fnn.l1_loss(off_pred, off_tgt)
        dir_pred = Fnn.normalize(pred[m][:, ct.DIRECTION], dim=-1, eps=1e-8)
        dir_tgt = Fnn.normalize(target[m][:, ct.DIRECTION], dim=-1, eps=1e-8)
        loss_dir = (1.0 - (dir_pred * dir_tgt).sum(-1)).mean()
    else:
        loss_off = pred.sum() * 0.0
        loss_dir = pred.sum() * 0.0

    total = w_heat * loss_heat + w_off * loss_off + w_dir * loss_dir
    return total, {"heat": float(loss_heat), "off": float(loss_off),
                   "dir": float(loss_dir), "total": float(total)}


def pred_to_array(pred):
    """Convert raw model output -> (N,7) numpy ready for cp_targets.decode_predictions
    (sigmoid on heatmap, unit-normalise direction)."""
    torch = _require_torch()
    import torch.nn.functional as Fnn
    out = pred.detach().clone()
    out[:, ct.HEATMAP] = torch.sigmoid(out[:, ct.HEATMAP])
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
            loss, parts = cp_loss(out, s["t"], s["m"], scale=s["scale"])
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
# first layer is well-conditioned regardless of part size. Targets stay in mm
# and cp_loss divides the offset term by the mesh scale.

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
                                 heat_pos_weight=50.0):
    """Train the production DiffusionNet regressor (C_out=7).

    Each sample needs 'verts','faces','verts_norm','target','mask','scale'
    (see prepare_sample). Mirrors diffusionnet.train_diffusionnet: operators are
    precomputed once per part (cached), each part is moved to the device for its
    step then freed. eval_fn(model, meta, epoch) is an optional callback.
    Returns (model, meta).
    """
    import random
    torch = _require_torch()
    import diffusionnet as dnmod
    model, meta = build_diffusionnet_regressor(config)
    model = model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    logger.info("DiffusionNet regressor: precomputing operators for %d parts "
                "(k_eig=%d, cache=%s) ...", len(train_samples), meta["k_eig"],
                op_cache_dir)
    prepared = []
    for s in train_samples:
        ops = _cp_operators(s["verts"], s["faces"], meta["k_eig"], op_cache_dir)
        prepared.append({
            "ops": ops,
            "x": torch.tensor(s["verts_norm"], dtype=torch.float32),
            "t": torch.tensor(s["target"], dtype=torch.float32),
            "m": torch.tensor(s["mask"], dtype=torch.bool),
            "scale": float(s["scale"]),
        })

    order = list(range(len(prepared)))
    for ep in range(epochs):
        model.train()
        random.shuffle(order)
        tot = 0.0
        for i in order:
            d = prepared[i]
            ops = _to_device_ops(d["ops"], device)
            x = (d["x"].to(device) if meta["input_features"] == "xyz"
                 else dnmod._model_input(ops, meta))
            t = d["t"].to(device); m = d["m"].to(device)
            opt.zero_grad(set_to_none=True)
            out = dnmod._forward(model, ops, x)
            loss, parts = cp_loss(out, t, m, scale=d["scale"], w_heat=w_heat,
                                  w_off=w_off, w_dir=w_dir,
                                  heat_pos_weight=heat_pos_weight)
            loss.backward()
            opt.step()
            tot += parts["total"]
            del ops, x, t, m, out, loss
        if log_every and (ep % log_every == 0 or ep == epochs - 1):
            logger.info("epoch %3d  loss=%.5f", ep, tot / max(1, len(prepared)))
        if eval_fn is not None and (ep % eval_every == 0 or ep == epochs - 1):
            eval_fn(model, meta, ep)
    return model, meta


def infer_diffusionnet(model, meta, verts, faces, verts_norm,
                       op_cache_dir=None, device="cpu"):
    """Run a trained DiffusionNet regressor on one part -> (N,7) numpy array
    ready for cp_targets.decode_predictions."""
    torch = _require_torch()
    import diffusionnet as dnmod
    ops = _to_device_ops(_cp_operators(verts, faces, meta["k_eig"], op_cache_dir),
                         device)
    x = (torch.tensor(np.asarray(verts_norm), dtype=torch.float32, device=device)
         if meta["input_features"] == "xyz" else dnmod._model_input(ops, meta))
    model.eval()
    with torch.no_grad():
        out = dnmod._forward(model, ops, x)
    return pred_to_array(out)


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
    arr = pred_to_array(out)
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
