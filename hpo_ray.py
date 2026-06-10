# Hyperparameter optimization (HPO) with Ray Tune.
#
# Context:
#   * MeshCNN and YOLOv6 were trained with recommended defaults from the
#     literature -> no custom HPO (see LITERATURE_DEFAULTS below).
#   * DiffusionNet hyperparameters were searched. The tunable knobs are in
#     DIFFUSIONNET_SPACE, including learning rate and number of diffusion blocks.
#
# Why Ray Tune:
#   Training takes a long time. Ray Tune tests many configurations in parallel
#   across multiple GPUs instead of one by one. The ASHA scheduler kills bad
#   trials early (early stopping); PBT copies settings from successful trials
#   and mutates them, and resources are assigned automatically.
#
# Ray is a heavy optional dependency. This module imports it only when actually
# tuning (lazy). Without Ray, the demo mode falls back to a simple local random
# search -- only as a stand-in for trying things out, NOT a real substitute for
# distributed Ray Tune.
#
# Reference: Scheffler (2022), §5.4.1, §5.4.2, §6.2.1, §6.2.2
#   §5.4.1 / Tabelle 3 – HPO search space: input_features (XYZ/HKS),
#                         lr [1e-5,1e-1], decay_steps {25,50,100,250,500},
#                         decay_rate (0,1), loss {NLL,CE}, n_blocks {1-5},
#                         width {32-1024}, k_eig {32-1024}, dropout (0,1)
#   §5.4.2             – Ray Tune + ASHA scheduler for distributed HPO
#   §6.2.1 / Tabelle 4 – class loss weights: Housing, Contact, SnapPoint,
#                         CableEntry, LabelSurface
#   §6.2.2 / Tabelle 5 – best DiffusionNet hyperparameters

import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# backend-neutral search space description
# ---------------------------------------------------------------------------
# Each parameter is described neutrally and then translated either to a
# Ray Tune config dict (to_ray_space) or sampled locally (sample). This way
# the search space works with and without Ray.

@dataclass(frozen=True)
class Param:
    kind: str                 # "loguniform" | "uniform" | "choice" | "randint"
    low: float = None
    high: float = None
    choices: tuple = None

    def sample(self, rng):
        if self.kind == "loguniform":
            import math
            lo, hi = math.log(self.low), math.log(self.high)
            return math.exp(rng.uniform(lo, hi))
        if self.kind == "uniform":
            return rng.uniform(self.low, self.high)
        if self.kind == "randint":
            return rng.randint(int(self.low), int(self.high) - 1)
        if self.kind == "choice":
            return rng.choice(list(self.choices))
        raise ValueError(f"unknown param kind: {self.kind}")

    def to_ray(self):
        from ray import tune
        if self.kind == "loguniform":
            return tune.loguniform(self.low, self.high)
        if self.kind == "uniform":
            return tune.uniform(self.low, self.high)
        if self.kind == "randint":
            return tune.randint(int(self.low), int(self.high))
        if self.kind == "choice":
            return tune.choice(list(self.choices))
        raise ValueError(f"unknown param kind: {self.kind}")


# §5.4.1 / Tabelle 3 – DiffusionNet HPO search space (reproduced exactly from Table 3)
DIFFUSIONNET_SPACE = {
    "input_features":     Param("choice", choices=("xyz", "hks")),                # input feature (categorical)
    "learning_rate":      Param("loguniform", 1e-5, 1e-1),                        # learning rate (log-uniform)
    "lr_decay_every":     Param("choice", choices=(25, 50, 100, 250, 500)),       # decay frequency (categorical)
    "lr_decay_rate":      Param("uniform", 0.0, 1.0),                             # decay rate (uniform)
    "loss":               Param("choice", choices=("nll", "ce")),                 # loss function (categorical)
    "n_diffusion_blocks": Param("choice", choices=(1, 2, 3, 4, 5)),               # diffusion blocks (categorical)
    "c_width":            Param("choice", choices=(32, 64, 128, 256, 512, 1024)), # block width (categorical)
    "n_eig":              Param("choice", choices=(32, 64, 128, 256, 512, 1024)), # number of eigenvectors (categorical)
    "dropout":            Param("uniform", 0.0, 1.0),                             # dropout rate (uniform)
    # extra beyond the thesis: weight of the Tversky overlap term in the loss
    # (0 = pure NLL/CE as in the thesis). The overlap term optimizes IoU/Dice
    # directly and boosts rare classes (snap points).
    "tversky_weight":     Param("choice", choices=(0.0, 0.25, 0.5)),
}


# ---- literature defaults (NOT searched) ----
# These networks were trained with recommended defaults, no custom HPO.
LITERATURE_DEFAULTS = {
    "meshcnn": {
        "learning_rate": 0.0002,
        "n_conv_filters": (64, 128, 256, 256),
        "pool_resolutions": (600, 450, 300, 180),
        "batch_size": 16,
    },
    "yolov6": {
        "learning_rate": 0.01,        # SGD start-lr from the YOLOv6 repo
        "img_size": 512,              # matches the 512x512 render views
        "batch_size": 32,
        "epochs": 300,
    },
}


# §6.2.1 / Tabelle 4 – class loss weights: Housing=0.1, Contact=2.0, SnapPoint=2.5,
#                       CableEntry=2.0, LabelSurface=2.5 (compensates label imbalance)
LOSS_WEIGHTS = {
    "housing":       46.4605,
    "contact":       221.5453,
    "snap_point":    1156.4556,
    "cable_entry":   546.8523,
    "label_surface": 514.3766,
}

# class order = label id (see connector_constants.py):
# 0=Housing, 1=Contact, 2=SnapPoint, 3=CableEntry, 4=LabelSurface
_CLASS_ORDER = ("housing", "contact", "snap_point", "cable_entry", "label_surface")

# same weights as a list (index = class id), derived from LOSS_WEIGHTS --
# single source of truth so both never drift apart.
LOSS_WEIGHTS_BY_ID = [LOSS_WEIGHTS[k] for k in _CLASS_ORDER]


# §6.2.2 / Tabelle 5 – best DiffusionNet hyperparameters from HPO run
BEST_DIFFUSIONNET = {
    "input_features":     "xyz",
    "learning_rate":      1e-3,
    "lr_decay_every":     100,
    "lr_decay_rate":      0.75,
    "loss":               "nll",
    "n_diffusion_blocks": 3,
    "c_width":            64,
    "n_eig":              64,
    "dropout":            0.3,
    "tversky_weight":     0.0,   # pure weighted NLL (overlap term off)
}


def to_ray_space(space):
    """Translate a neutral search space (dict of Param) into a Ray Tune config dict."""
    return {k: p.to_ray() for k, p in space.items()}


def default_hparams(model):
    """Get literature defaults for a non-tuned model (meshcnn / yolov6)."""
    key = model.lower()
    if key not in LITERATURE_DEFAULTS:
        raise KeyError(f"no literature defaults for {model!r} "
                       f"(known: {list(LITERATURE_DEFAULTS)})")
    return dict(LITERATURE_DEFAULTS[key])


# ---------------------------------------------------------------------------
# scheduler (early stopping)
# ---------------------------------------------------------------------------

# §5.4.2 – Ray Tune + ASHA scheduler for distributed HPO (early-stop bad trials)
def build_asha_scheduler(max_epochs=100, grace_period=10, reduction_factor=3):
    """Build an ASHA scheduler: bad trials are stopped early.

    grace_period = minimum epochs before a trial can be killed.
    reduction_factor = how aggressively to prune (higher = more aggressive).
    """
    from ray.tune.schedulers import ASHAScheduler
    return ASHAScheduler(
        max_t=max_epochs,
        grace_period=grace_period,
        reduction_factor=reduction_factor,
    )


def build_pbt_scheduler(perturbation_interval=5, hyperparam_mutations=None):
    """Population Based Training: copies settings from successful trials
    onto weaker ones and mutates them."""
    from ray.tune.schedulers import PopulationBasedTraining
    return PopulationBasedTraining(
        time_attr="training_iteration",
        perturbation_interval=perturbation_interval,
        hyperparam_mutations=hyperparam_mutations or {},
    )


# ---------------------------------------------------------------------------
# actual tuning via Ray Tune
# ---------------------------------------------------------------------------

def run_hpo(trainable, space, num_samples=20, metric="val_iou", mode="max",
            scheduler="asha", max_epochs=100, gpus_per_trial=1, cpus_per_trial=2,
            storage_path=None, experiment_name="diffusionnet_hpo"):
    """Run HPO with Ray Tune.

    trainable      : function config -> training. Must report metrics per epoch,
                     e.g. with:
                         from ray import tune
                         tune.report({"val_iou": iou, "loss": loss})
    space          : neutral search space (dict of Param), e.g. DIFFUSIONNET_SPACE.
    num_samples    : how many configurations to sample.
    gpus_per_trial : GPUs per trial -> Ray distributes trials across available
                     GPUs (parallel instead of sequential).

    Returns: best result (ray.tune.Result).
    """
    try:
        import ray
        from ray import tune
        from ray.tune import TuneConfig
    except ImportError as exc:
        raise ImportError(
            "Ray Tune is not installed. Install with:\n"
            "  pip install 'ray[tune]'\n"
            "or the project extra:  pip install -e '.[hpo]'"
        ) from exc

    if scheduler == "asha":
        sched = build_asha_scheduler(max_epochs=max_epochs)
    elif scheduler == "pbt":
        sched = build_pbt_scheduler()
    elif scheduler is None:
        sched = None
    else:
        raise ValueError(f"unknown scheduler: {scheduler!r} (asha / pbt / None)")

    trainable_with_resources = tune.with_resources(
        trainable, resources={"cpu": cpus_per_trial, "gpu": gpus_per_trial}
    )

    tuner = tune.Tuner(
        trainable_with_resources,
        param_space=to_ray_space(space),
        tune_config=TuneConfig(
            metric=metric,
            mode=mode,
            scheduler=sched,
            num_samples=num_samples,
        ),
        run_config=ray.train.RunConfig(name=experiment_name, storage_path=storage_path),
    )

    logger.info("Starting Ray Tune: %d samples, %s, %d GPU/trial",
                num_samples, scheduler, gpus_per_trial)
    results = tuner.fit()
    best = results.get_best_result(metric=metric, mode=mode)
    logger.info("Best result: %s=%.4f  config=%s",
                metric, best.metrics.get(metric, float("nan")), best.config)
    return best


# ---------------------------------------------------------------------------
# offline fallback: simple local random search (no Ray, no GPU)
# ---------------------------------------------------------------------------

def local_random_search(objective, space, num_samples=20, mode="max", seed=0):
    """Simple random search in the space -- for trying things out without Ray.

    objective : config -> scalar (higher or lower = better, depending on mode).
    Returns   : (best_config, best_value, all_trials).
    """
    import random
    rng = random.Random(seed)
    trials = []
    for i in range(num_samples):
        cfg = {k: p.sample(rng) for k, p in space.items()}
        val = objective(cfg)
        trials.append((cfg, val))
        logger.debug("trial %2d: value=%.4f config=%s", i, val, cfg)

    better = max if mode == "max" else min
    best_cfg, best_val = better(trials, key=lambda t: t[1])
    return best_cfg, best_val, trials


# ---------------------------------------------------------------------------
# demo: synthetic objective with known optimum
# ---------------------------------------------------------------------------

def _toy_objective(config):
    # cheap stand-in for real DiffusionNet training. the pseudo val_iou in [0,1)
    # has its optimum at the best network from Table 5 (lr=1e-3, 3 diffusion
    # blocks, dropout=0.3) so the demo converges in the same direction as the thesis.
    import math
    lr = config["learning_rate"]
    blocks = config["n_diffusion_blocks"]
    lr_term = math.exp(-((math.log10(lr) - (-3.0)) ** 2) / 0.5)   # optimum lr=1e-3
    block_term = math.exp(-((blocks - 3) ** 2) / 2.0)             # optimum 3 blocks
    dropout_term = 1.0 - abs(config["dropout"] - 0.3)             # optimum dropout 0.3
    return 0.5 * lr_term + 0.3 * block_term + 0.2 * dropout_term


def _demo():
    print("DiffusionNet HPO search space (Table 3):")
    for k, p in DIFFUSIONNET_SPACE.items():
        rng = f"[{p.low}, {p.high}]" if p.choices is None else f"choices={p.choices}"
        print(f"   - {k:18s} {p.kind:11s} {rng}")

    print("\nLiterature defaults (not tuned):")
    for model, d in LITERATURE_DEFAULTS.items():
        print(f"   - {model}: {d}")

    print("\nLoss function weights (Table 4):")
    for cls, w in LOSS_WEIGHTS.items():
        print(f"   - {cls:22s} {w}")

    print("\nBest DiffusionNet (Table 5):")
    for k, v in BEST_DIFFUSIONNET.items():
        print(f"   - {k:18s} {v}")

    try:
        import ray  # noqa: F401
        have_ray = True
    except ImportError:
        have_ray = False

    print(f"\n[backend] Ray Tune {'available' if have_ray else 'NOT installed -> local random search'}")
    best_cfg, best_val, trials = local_random_search(
        _toy_objective, DIFFUSIONNET_SPACE, num_samples=40, mode="max", seed=0
    )
    print(f"[search]  {len(trials)} configurations tested")
    print(f"[best]    pseudo-val_iou={best_val:.4f}")
    print(f"          learning_rate     = {best_cfg['learning_rate']:.2e}")
    print(f"          n_diffusion_blocks= {best_cfg['n_diffusion_blocks']}")
    print(f"          dropout           = {best_cfg['dropout']:.3f}")
    print("\nnote: for real DiffusionNet training use run_hpo(...),")
    print("      the trainable reports tune.report({'val_iou': ...}) per epoch.")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
    _demo()
