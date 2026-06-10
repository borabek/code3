# DiffusionNet training and inference (§5.3.3 / §5.4). Lazy torch import.
#
# Reference: Scheffler (2022), §5.3.3, §5.4.1, §5.4.2, §6.2.1, §6.2.2, §6.2.3, §2.2.3
#   §5.3.3 / Abb.40  – DiffusionNet: MLP + diffusion layer + spatial gradient;
#                       NLL/CE loss; k_eig eigenvectors; HKS vs XYZ input
#   §5.4.1 / Tabelle 3 – HPO search space (input_features, lr, decay_steps, loss,
#                         n_blocks, width, k_eig, dropout)
#   §6.2.1 / Tabelle 4 – class loss weights: Housing=0.1, Contact=2.0, SnapPoint=2.5,
#                         CableEntry=2.0, LabelSurface=2.5
#   §6.2.2 / Tabelle 5 – best DiffusionNet hyperparameters
#   §6.2.3             – hybrid inference: DiffusionNet for all classes except SnapPoint
#   §2.2.3             – NLL loss (Negative Log-Likelihood), weighted cross-entropy

import logging
from dataclasses import dataclass
import numpy as np
import metrics

# ---------------------------------------------------------------------------
# HPO search space and best hyperparameters (merged from hpo_ray.py)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Param:
    kind: str
    low: float | None = None
    high: float | None = None
    choices: tuple | None = None

    def sample(self, rng):
        if self.kind == "loguniform":
            import math
            assert self.low is not None and self.high is not None
            lo, hi = math.log(self.low), math.log(self.high)
            return math.exp(rng.uniform(lo, hi))
        if self.kind == "uniform":
            assert self.low is not None and self.high is not None
            return rng.uniform(self.low, self.high)
        if self.kind == "randint":
            assert self.low is not None and self.high is not None
            return rng.randint(int(self.low), int(self.high) - 1)
        if self.kind == "choice":
            assert self.choices is not None
            return rng.choice(list(self.choices))
        raise ValueError(f"unknown param kind: {self.kind}")

    def to_ray(self):
        from ray import tune  # type: ignore[import]
        if self.kind == "loguniform":
            assert self.low is not None and self.high is not None
            return tune.loguniform(self.low, self.high)
        if self.kind == "uniform":
            assert self.low is not None and self.high is not None
            return tune.uniform(self.low, self.high)
        if self.kind == "randint":
            assert self.low is not None and self.high is not None
            return tune.randint(int(self.low), int(self.high))
        if self.kind == "choice":
            assert self.choices is not None
            return tune.choice(list(self.choices))
        raise ValueError(f"unknown param kind: {self.kind}")


DIFFUSIONNET_SPACE = {
    "input_features":     Param("choice", choices=("xyz", "hks")),
    "learning_rate":      Param("loguniform", 1e-5, 1e-1),
    "lr_decay_every":     Param("choice", choices=(25, 50, 100, 250, 500)),
    "lr_decay_rate":      Param("uniform", 0.0, 1.0),
    "loss":               Param("choice", choices=("nll", "ce")),
    "n_diffusion_blocks": Param("choice", choices=(1, 2, 3, 4, 5)),
    "c_width":            Param("choice", choices=(32, 64, 128, 256, 512, 1024)),
    "n_eig":              Param("choice", choices=(32, 64, 128, 256, 512, 1024)),
    "dropout":            Param("uniform", 0.0, 1.0),
    "tversky_weight":     Param("choice", choices=(0.0, 0.25, 0.5)),
}

LITERATURE_DEFAULTS = {
    "meshcnn": {"learning_rate": 0.0002, "n_conv_filters": (64, 128, 256, 256),
                "pool_resolutions": (600, 450, 300, 180), "batch_size": 16},
    "yolov6":  {"learning_rate": 0.01, "img_size": 512, "batch_size": 32, "epochs": 300},
}

LOSS_WEIGHTS = {
    "housing": 46.4605, "contact": 221.5453, "snap_point": 1156.4556,
    "cable_entry": 546.8523, "label_surface": 514.3766,
}
_CLASS_ORDER = ("housing", "contact", "snap_point", "cable_entry", "label_surface")
LOSS_WEIGHTS_BY_ID = [LOSS_WEIGHTS[k] for k in _CLASS_ORDER]

BEST_DIFFUSIONNET = {
    "input_features": "xyz", "learning_rate": 1e-3, "lr_decay_every": 100,
    "lr_decay_rate": 0.75, "loss": "nll", "n_diffusion_blocks": 3,
    "c_width": 64, "n_eig": 64, "dropout": 0.3, "tversky_weight": 0.0,
}


def to_ray_space(space):
    return {k: p.to_ray() for k, p in space.items()}


def default_hparams(model):
    key = model.lower()
    if key not in LITERATURE_DEFAULTS:
        raise KeyError(f"no literature defaults for {model!r}")
    return dict(LITERATURE_DEFAULTS[key])


def build_asha_scheduler(max_epochs=100, grace_period=10, reduction_factor=3):
    from ray.tune.schedulers import ASHAScheduler  # type: ignore[import]
    return ASHAScheduler(max_t=max_epochs, grace_period=grace_period,
                         reduction_factor=reduction_factor)


def build_pbt_scheduler(perturbation_interval=5, hyperparam_mutations=None):
    from ray.tune.schedulers import PopulationBasedTraining  # type: ignore[import]
    return PopulationBasedTraining(time_attr="training_iteration",
                                   perturbation_interval=perturbation_interval,
                                   hyperparam_mutations=hyperparam_mutations or {})


def run_hpo(trainable, space, num_samples=20, metric="val_iou", mode="max",
            scheduler="asha", max_epochs=100, gpus_per_trial=1, cpus_per_trial=2,
            storage_path=None, experiment_name="diffusionnet_hpo"):
    try:
        import ray  # type: ignore[import]
        from ray import tune  # type: ignore[import]
        from ray.tune import TuneConfig  # type: ignore[import]
    except ImportError as exc:
        raise ImportError("Ray Tune not installed. pip install 'ray[tune]'") from exc
    sched = (build_asha_scheduler(max_epochs=max_epochs) if scheduler == "asha"
             else build_pbt_scheduler() if scheduler == "pbt" else None)
    trainable_with_resources = tune.with_resources(
        trainable, resources={"cpu": cpus_per_trial, "gpu": gpus_per_trial})
    tuner = tune.Tuner(
        trainable_with_resources, param_space=to_ray_space(space),
        tune_config=TuneConfig(metric=metric, mode=mode, scheduler=sched,
                               num_samples=num_samples),
        run_config=ray.train.RunConfig(name=experiment_name, storage_path=storage_path))
    results = tuner.fit()
    best = results.get_best_result(metric=metric, mode=mode)
    logging.getLogger(__name__).info("Best result: %s=%.4f  config=%s",
                                     metric, best.metrics.get(metric, float("nan")), best.config)
    return best


def local_random_search(objective, space, num_samples=20, mode="max", seed=0):
    import random
    rng = random.Random(seed)
    trials = [(cfg := {k: p.sample(rng) for k, p in space.items()}, objective(cfg))
              for _ in range(num_samples)]
    better = max if mode == "max" else min
    best_cfg, best_val = better(trials, key=lambda t: t[1])
    return best_cfg, best_val, trials


def _hpo_demo():
    print("DiffusionNet HPO search space (Table 3):")
    for k, p in DIFFUSIONNET_SPACE.items():
        rng = f"[{p.low}, {p.high}]" if p.choices is None else f"choices={p.choices}"
        print(f"   - {k:18s} {p.kind:11s} {rng}")
    try:
        import ray  # type: ignore[import]  # noqa: F401
        have_ray = True
    except ImportError:
        have_ray = False
    print(f"\n[backend] Ray Tune {'available' if have_ray else 'NOT installed -> local random search'}")

# ---------------------------------------------------------------------------

logger = logging.getLogger(__name__)

N_CLASSES = 5  # housing, contact, snap-point, cable-entry, label-surface

# central defaults (§4: single defaulting point instead of scattered config.get).
# base = best network from Table 5 plus new training options.
DEFAULTS = {
    **BEST_DIFFUSIONNET,
    "tversky_weight": 0.0,   # 0 = pure (weighted) NLL/CE as in the thesis
    "tversky_alpha":  0.3,   # weight false-positives
    "tversky_beta":   0.7,   # weight false-negatives (>alpha boosts rare classes)
    "accum_steps":    1,     # gradient accumulation (1 = one opt.step per mesh)
}


def _require(pkg, hint):
    """Import a package, raising a clear message if missing."""
    import importlib
    try:
        return importlib.import_module(pkg)
    except ImportError as exc:
        raise ImportError(
            f"'{pkg}' is required but not installed.\n{hint}"
        ) from exc


# §5.3.3 / Abb.40 – DiffusionNet model builder: MLP + diffusion layer + spatial gradient
def build_diffusionnet(config, n_classes=N_CLASSES, hks_count=16):
    """Build a DiffusionNet model from an HPO configuration (Table 3/5).

    Config keys: input_features, c_width, n_diffusion_blocks, n_eig,
    dropout, loss. Returns (model, meta) where meta holds input info.
    """
    torch = _require("torch", "pip install torch")
    dn = _require("diffusion_net",
                  "pip install git+https://github.com/nmwsharp/diffusion-net "
                  "(or project extra: pip install -e '.[diffnet]')")

    c_in = 3 if config.get("input_features", "xyz") == "xyz" else hks_count
    use_ce = config.get("loss", "nll") == "ce"

    # NLL needs log-softmax at output; CE uses raw logits
    last = (None if use_ce
            else (lambda x: torch.nn.functional.log_softmax(x, dim=-1)))

    drop_rate = float(config.get("dropout", 0.0))
    model = dn.layers.DiffusionNet(
        C_in=c_in,
        C_out=n_classes,
        C_width=int(config.get("c_width", 128)),
        N_block=int(config.get("n_diffusion_blocks", 4)),
        last_activation=last,
        outputs_at="vertices",
        # diffusion_net's `dropout` is only an on/off switch with FIXED internal
        # rate -- the HPO-found rate (Table 5: 0.3) would be lost otherwise.
        dropout=drop_rate > 0.0,
    )
    if drop_rate > 0.0:
        set_dropout_rate(model, drop_rate)  # apply real rate (§1.2)

    meta = {"input_features": config.get("input_features", "xyz"),
            "k_eig": int(config.get("n_eig", 128)),
            "hks_count": hks_count, "use_ce": use_ce}
    return model, meta


def set_dropout_rate(model, p):
    """Set real dropout rate on all nn.Dropout modules (§1.2).

    diffusion_net.DiffusionNet(dropout=...) is just a boolean switch with a
    fixed internal rate. to actually use the HPO-found rate, we turn dropout
    on and patch the rate here. returns: number of modules patched.
    """
    torch = _require("torch", "pip install torch")
    import torch.nn as nn
    n = 0
    for m in model.modules():
        if isinstance(m, nn.Dropout):
            m.p = float(p)
            n += 1
    return n


# Hardware cap: more than 128 eigenvectors rarely improves segmentation but
# scales eigen-decomposition cost/VRAM steeply (problematic for big meshes on
# an 11 GB GPU). Keep k_eig bounded for the robotic deployment target.
MAX_K_EIG = 128


# §5.3.3 – cotangent Laplacian, mass matrix, eigenbasis (k_eig eigenvectors)
def precompute_operators(verts, faces, k_eig, op_cache_dir=None):
    """Laplace/mass matrices, eigenbasis and gradient operators.

    `k_eig` is capped at MAX_K_EIG (128) to bound eigen-decomposition cost and
    GPU memory on large meshes. When `op_cache_dir` is given the eigenbasis is
    cached/loaded by diffusion_net (keyed on the mesh) -- strongly recommended
    so the expensive decomposition is not repeated every run.
    """
    torch = _require("torch", "pip install torch")
    dn = _require("diffusion_net", "pip install git+https://github.com/nmwsharp/diffusion-net")
    k_eig = int(min(int(k_eig), MAX_K_EIG))
    V = torch.tensor(np.asarray(verts), dtype=torch.float32)
    F = torch.tensor(np.asarray(faces), dtype=torch.long)
    frames, mass, L, evals, evecs, gradX, gradY = dn.geometry.get_operators(
        V, F, k_eig=k_eig, op_cache_dir=op_cache_dir)
    return {"verts": V, "faces": F, "mass": mass, "L": L, "evals": evals,
            "evecs": evecs, "gradX": gradX, "gradY": gradY}


# §5.3.3 / Abb.40 – HKS vs XYZ input features for DiffusionNet
def _model_input(ops, meta):
    torch = _require("torch", "pip install torch")
    dn = _require("diffusion_net", "pip install git+https://github.com/nmwsharp/diffusion-net")
    if meta["input_features"] == "xyz":
        return ops["verts"]
    # heat-kernel signature as alternative input (non-rigid invariance)
    return dn.geometry.compute_hks_autoscale(ops["evals"], ops["evecs"], meta["hks_count"])


def _forward(model, ops, x):
    return model(x, ops["mass"], L=ops["L"], evals=ops["evals"], evecs=ops["evecs"],
                 gradX=ops["gradX"], gradY=ops["gradY"], faces=ops["faces"])


# ---------------------------------------------------------------------------
# §2.2.3 – NLL loss (Negative Log-Likelihood), weighted cross-entropy
# loss: weighted NLL/CE + optional Tversky overlap term (§1.1 / §3)
# ---------------------------------------------------------------------------
# the network trains on NLL but is evaluated on IoU/Dice -- a poor surrogate
# with ~70% housing. the Tversky term directly optimizes region overlap and
# boosts rare classes (snap-points).

def tversky_loss(probs, target, alpha=0.3, beta=0.7, class_w=None, eps=1e-6):
    """Multi-class Tversky loss on probabilities (probs: (V, C)).

    alpha weights false-positives, beta false-negatives. beta>alpha penalizes
    missed vertices more strongly -> boosts rare classes. if class_w is set,
    per-class overlaps are weighted additionally.
    """
    torch = _require("torch", "pip install torch")
    import torch.nn.functional as Fnn
    t = Fnn.one_hot(target, probs.shape[-1]).to(probs.dtype)
    tp = (probs * t).sum(0)
    fp = (probs * (1.0 - t)).sum(0)
    fn = ((1.0 - probs) * t).sum(0)
    tv = (tp + eps) / (tp + alpha * fp + beta * fn + eps)   # (C,)
    if class_w is not None:
        tv = tv * (class_w / class_w.sum() * len(class_w))
    return 1.0 - tv.mean()


def _compute_loss(out, lab, w, meta, cfg):
    """Base loss (weighted NLL or CE) + optional Tversky term."""
    torch = _require("torch", "pip install torch")
    import torch.nn.functional as Fnn
    base = (Fnn.cross_entropy(out, lab, weight=w) if meta["use_ce"]
            else Fnn.nll_loss(out, lab, weight=w))
    tw = float(cfg.get("tversky_weight", 0.0))
    if tw > 0.0:
        # `out` is log-softmax (NLL) or logits (CE) -> convert to probabilities
        probs = out.softmax(dim=-1) if meta["use_ce"] else out.exp()
        base = base + tw * tversky_loss(
            probs, lab, alpha=float(cfg.get("tversky_alpha", 0.3)),
            beta=float(cfg.get("tversky_beta", 0.7)), class_w=w)
    return base


# ---------------------------------------------------------------------------
# dataset pre-computation on CPU (§2.1 / §2.2)
# ---------------------------------------------------------------------------
# the eigenbasis is the most expensive step and is computed ONCE per sample;
# for HKS the model input is cached together (otherwise recomputed per epoch/step).
# all tensors stay on CPU and are moved to GPU individually during the training
# step -> the full operator cache does not live permanently in VRAM (avoids OOM
# with many meshes / large k_eig on 11 GB).

def precompute_dataset(samples, meta, op_cache_dir=None):
    """Pre-compute operators + model input per sample on CPU.

    Returns list of dicts {ops, x_in, lab} (all CPU tensors). A part whose
    operators fail to build (degenerate geometry, eigensolver error) is logged
    and skipped rather than killing the whole run.
    """
    torch = _require("torch", "pip install torch")
    out, skipped = [], 0
    for s in samples:
        try:
            ops = precompute_operators(s["verts"], s["faces"], meta["k_eig"], op_cache_dir)
        except Exception as exc:                       # noqa: BLE001 (skip bad mesh)
            skipped += 1
            logger.warning("skipping part (operator build failed): %s", exc)
            continue
        x_in = ops["verts"] if meta["input_features"] == "xyz" else _model_input(ops, meta)
        lab = torch.as_tensor(np.asarray(s["labels"]), dtype=torch.long)
        out.append({"ops": ops, "x_in": x_in, "lab": lab})
    if skipped:
        logger.warning("skipped %d/%d parts with bad geometry", skipped, len(samples))
    return out


def _to_device(sample, device):
    """Move a pre-computed sample (CPU) to the target device."""
    ops = {k: (v.to(device) if hasattr(v, "to") else v)
           for k, v in sample["ops"].items()}
    return ops, sample["x_in"].to(device), sample["lab"].to(device)


def train_diffusionnet(config, train_samples, val_samples, epochs=200,
                       device="cpu", weights=None, report=None, eval_every=5,
                       checkpoint_path=None, op_cache_dir=None):
    """Train DiffusionNet and report validation metrics per epoch.

    train_samples / val_samples: lists of dicts with 'verts','faces','labels'.
    weights: class weights (default Table 4). report(dict): callback per
    eval epoch (e.g. ray.tune.report). lr schedule = step-decay per
    lr_decay_every / lr_decay_rate (Table 3). checkpoint_path: if set,
    saves weights of best epoch state (by mean_iou) together with config,
    so load_checkpoint()/predict() can infer without retraining.
    Returns best validation metrics.
    """
    import random
    torch = _require("torch", "pip install torch")

    cfg = {**DEFAULTS, **config}  # §4: single defaulting point instead of scattered .get
    model, meta = build_diffusionnet(cfg)
    model = model.to(device)

    # §6.2.1 / Tabelle 4 – class loss weights (Housing=low, SnapPoint/LabelSurface=high)
    # §3: normalize class weights to mean 1 -> the lr scale is interpretable
    # across trials/datasets (the absolute level of the weighted loss would
    # otherwise depend on per-mesh class mixture).
    w = torch.tensor(weights if weights is not None else LOSS_WEIGHTS_BY_ID,
                     dtype=torch.float32, device=device)
    w = w / w.mean()

    opt = torch.optim.Adam(model.parameters(), lr=float(cfg["learning_rate"]))
    sched = torch.optim.lr_scheduler.StepLR(
        opt, step_size=int(cfg["lr_decay_every"]), gamma=float(cfg["lr_decay_rate"]))
    accum = max(1, int(cfg.get("accum_steps", 1)))

    # §2.1/§2.2: pre-compute operators (incl. HKS) ONCE on CPU; move each
    # sample to GPU during the step, then free it.
    logger.info("DiffusionNet: precomputing operators (%d train, %d val, k_eig=%d) ...",
                len(train_samples), len(val_samples), meta["k_eig"])
    train = precompute_dataset(train_samples, meta, op_cache_dir)
    val = precompute_dataset(val_samples, meta, op_cache_dir)

    best = {"mean_iou": -1.0}
    best_state = None
    order = list(range(len(train)))
    for ep in range(epochs):
        model.train()
        random.shuffle(order)                      # §2.3: avoid order bias
        opt.zero_grad(set_to_none=True)
        for n, i in enumerate(order, 1):
            ops, x_in, lab = _to_device(train[i], device)   # one mesh to GPU
            out = _forward(model, ops, x_in)
            loss = _compute_loss(out, lab, w, meta, cfg) / accum
            loss.backward()
            # §2.3: gradient accumulation. one opt.step every `accum` meshes; the
            # last (possibly smaller) remainder at epoch end also gets stepped,
            # but is still divided by `accum` -> slightly under-weighted.
            # deliberately accepted approximation (avoids per-remainder scaling).
            if n % accum == 0 or n == len(order):
                opt.step()
                opt.zero_grad(set_to_none=True)
            del ops, x_in, lab, out, loss          # §2.1: free VRAM before next mesh
        sched.step()

        if ep % eval_every == 0 or ep == epochs - 1:
            model.eval()
            preds, trues = [], []
            with torch.no_grad():
                for sample in val:
                    ops, x_in, lab = _to_device(sample, device)
                    out = _forward(model, ops, x_in)
                    preds.append(out.argmax(dim=-1).cpu().numpy())
                    trues.append(lab.cpu().numpy())
                    del ops, x_in, lab, out
            rep = metrics.segmentation_report(np.concatenate(preds),
                                              np.concatenate(trues), n_classes=N_CLASSES)
            rep["epoch"] = ep
            if rep["mean_iou"] > best["mean_iou"]:
                best = rep
                # §2.4: snapshot onto CPU -> no second model in VRAM
                best_state = {k: v.detach().cpu().clone()
                              for k, v in model.state_dict().items()}
            if report is not None:
                report(rep)
            logger.info("epoch %3d  val mean_iou=%.4f  weighted_iou=%.4f",
                        ep, rep["mean_iou"], rep["weighted_iou"])

    # persist best model -> usable via predict() directly later
    if checkpoint_path is not None and best_state is not None:
        save_checkpoint_state(best_state, cfg, meta, checkpoint_path)
        logger.info("best model saved: %s (val mean_iou=%.4f)",
                    checkpoint_path, best["mean_iou"])
    return best


# ---------------------------------------------------------------------------
# checkpoints + inference (§5.3.7: (V, 5) -> argmax -> 1D label tensor)
# ---------------------------------------------------------------------------

def save_checkpoint_state(state_dict, config, meta, path):
    """Save a pre-extracted state_dict + config/meta."""
    torch = _require("torch", "pip install torch")
    torch.save({"state_dict": state_dict, "config": dict(config), "meta": meta}, path)
    return path


def save_checkpoint(model, config, meta, path):
    """Save model weights + config (Table 3/5) + meta together.

    load_checkpoint() can then reconstruct the model exactly and predict()
    can run inference directly on it.
    """
    return save_checkpoint_state(model.state_dict(), config, meta, path)


def load_checkpoint(path, device="cpu", n_classes=N_CLASSES):
    """Restore a DiffusionNet saved with save_checkpoint().

    Returns (model, meta, config). model is on `device` and in eval mode.
    """
    torch = _require("torch", "pip install torch")
    ckpt = torch.load(path, map_location=device)
    config = ckpt["config"]
    model, meta = build_diffusionnet(config, n_classes=n_classes)
    model.load_state_dict(ckpt["state_dict"])
    meta = ckpt.get("meta", meta) or meta   # prefer meta from checkpoint
    model = model.to(device)
    model.eval()
    return model, meta, config


# §5.3.7 / Abb.45 – per-vertex segmentation: (V, 5) probability -> argmax -> 1D label tensor
def predict(model, meta, verts, faces, device="cpu", return_probs=False,
            op_cache_dir=None):
    """Per-vertex segmentation of a single part (§5.3.7 / Figure 45).

    Runs forward pass and reduces the (V, 5) probability distribution via
    argmax to a 1D label tensor -- exactly the post-processing from §5.3.7.
    Returns numpy array (V,) with class ids {0..4}; with return_probs also
    returns the (V, 5) array of class probabilities.
    """
    torch = _require("torch", "pip install torch")
    model = model.to(device)
    model.eval()
    ops = precompute_operators(verts, faces, meta["k_eig"], op_cache_dir)
    ops = {k: (v.to(device) if hasattr(v, "to") else v) for k, v in ops.items()}
    with torch.no_grad():
        out = _forward(model, ops, _model_input(ops, meta))
        labels = out.argmax(dim=-1).cpu().numpy()
        probs_np = None
        if return_probs:
            # for NLL, out is log-softmax -> convert back to probability
            probs = (out.exp() if not meta.get("use_ce") else out.softmax(dim=-1))
            probs_np = probs.cpu().numpy()
    # free the per-part operators/activations before the next stage (segmentation
    # of large meshes is the main VRAM consumer in the pipeline).
    del ops, out
    free_gpu_memory()
    if return_probs:
        return labels, probs_np
    return labels


def free_gpu_memory():
    """Aggressively release cached GPU memory between pipeline stages.

    No-op when torch / CUDA are unavailable, so it is safe to call on CPU-only
    deployments and in the selftests.
    """
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
    except Exception:
        pass


def make_ray_trainable(train_samples, val_samples, epochs=200, weights=None):
    """Build a Ray-Tune trainable that feeds the search-space config into
    train_diffusionnet and reports val_iou. pass to hpo_ray.run_hpo(trainable, ...)."""
    def trainable(config):
        from ray import tune  # type: ignore[import]
        train_diffusionnet(
            config, train_samples, val_samples, epochs=epochs, weights=weights,
            report=lambda rep: tune.report({"val_iou": rep["mean_iou"],
                                            "weighted_iou": rep["weighted_iou"],
                                            "accuracy": rep["accuracy"]}))
    return trainable


def _demo():
    print("DiffusionNet integration (§5.3.3 / §5.4)")
    print("best config (Table 5):", BEST_DIFFUSIONNET)
    try:
        import importlib
        importlib.import_module("torch")
        importlib.import_module("diffusion_net")
        print("torch + diffusion_net available -> can build_diffusionnet(BEST_DIFFUSIONNET)")
        model, meta = build_diffusionnet(BEST_DIFFUSIONNET)
        n_params = sum(p.numel() for p in model.parameters())
        print(f"model built: {n_params} parameters, input={meta['input_features']}, k_eig={meta['k_eig']}")
        n_drop = set_dropout_rate(model, float(BEST_DIFFUSIONNET.get("dropout", 0.0)))
        print(f"dropout : real rate set on {n_drop} modules (§1.2 -- not just on/off)")
        print("loss    : config['tversky_weight']>0 adds a Tversky")
        print("          overlap-term (§1.1) that optimizes IoU/Dice directly and boosts")
        print("          rare classes (snap-points).")
        print("inference: train_diffusionnet(..., checkpoint_path='best.pt') saves best")
        print("          weights; load_checkpoint('best.pt') + predict(model, meta, V, F)")
        print("          then gives per-vertex labels for the e2e pipeline (see infer_pipeline.py).")
    except ImportError as exc:
        print("note:", exc)
        print("-> without torch/diffusion_net only the integration is provided;")
        print("   train/predict/checkpoint work once torch + diffusion_net are installed.")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
    _demo()
