# Training entry point for connection-point detection on ABB JSON data (Option B).
#
# Ties together:
#   json_dataset  – stream parts from the corpus (1 GB single-array file OR a
#                   directory of per-part files)
#   cp_targets    – encode terminal-block locations -> per-vertex targets;
#                   decode predictions -> discrete connection points
#   cp_regressor  – the per-vertex 7-channel head + combined loss + train loop
#   metrics_cp    – localisation (mm) / angular (deg) / precision / recall
#
# Three backbones:
#   --backbone mlp           : light per-vertex MLP on xyz (only needs torch).
#                              Memorises geometry -> good for overfit smoke-tests,
#                              but has no receptive field so it does NOT generalise.
#   --backbone diffusionnet  : the production model (needs the diffusion_net pkg
#                              + robust_laplacian/potpourri3d -> Linux/WSL).
#   --backbone knngraph      : torch+scipy EdgeConv on a kNN graph. Has a real
#                              geometric receptive field (generalises) but NO
#                              native geometry deps -> runs on native Windows.
#
# Memory / 1 GB scale: parts are streamed; with --max-parts you cap how many are
# materialised. The eigenbasis for the diffusionnet backbone is the expensive
# step and should be cached per part (see diffusionnet.precompute_operators).

import json
import logging
import argparse

import numpy as np

import json_dataset as jd
import cp_targets as ct
import cp_regressor as cpr
import metrics as mcp

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# split a corpus by PartNr (deterministic hash -> stable across runs)
# ---------------------------------------------------------------------------

def split_part_ids(ids, val_frac=0.2, seed=0):
    """Deterministic train/val split by PartNr (hash-based, stable)."""
    import hashlib
    val = set()
    train = []
    for pid in ids:
        h = int(hashlib.sha256((str(seed) + ":" + pid).encode()).hexdigest(), 16)
        if (h % 1000) / 1000.0 < val_frac:
            val.add(pid)
        else:
            train.append(pid)
    return set(train), val


# ---------------------------------------------------------------------------
# evaluate a trained model (any backbone) on prepared samples
# ---------------------------------------------------------------------------

def _nms_radius(s, clearance_mm=5.0):
    """Deployment-realistic NMS radius from part GEOMETRY only -- not ground truth.

    #2: the previous version shrank the radius below the closest GT-block pair,
    which leaks the labels of the sample being scored (you have no GT at
    inference). sigma is a fixed fraction of the bbox diagonal
    (cp_targets.encode_targets), so max(2*sigma, clearance) needs no GT and is
    exactly what production decoding would use.
    """
    return max(2.0 * s["sigma"], clearance_mm)


def evaluate(samples, parts_meta, infer_arr, dist_thresh_mm=5.0,
             heatmap_thresh=0.3, min_votes=2, nms_clearance_mm=5.0):
    """Decode predictions for each sample and aggregate keypoint metrics.

    infer_arr(sample, part_meta) -> (N,7) numpy array of per-vertex predictions
    (heatmap already sigmoid'd, direction unit). This abstracts the backbone so
    the same decode+metrics path serves both the MLP and DiffusionNet models.
    min_votes: a decoded cluster needs at least this many voting vertices (#6:
    >1 suppresses lone-vertex false positives -> higher precision).
    """
    reports = []
    for s, pm in zip(samples, parts_meta):
        arr = infer_arr(s, pm)
        preds = ct.decode_predictions(pm["vertices"], arr,
                                      heatmap_thresh=heatmap_thresh,
                                      nms_radius_mm=_nms_radius(s, nms_clearance_mm),
                                      min_votes=min_votes)
        rep = mcp.keypoint_report(preds, s["gt_points"], s["gt_directions"],
                                  dist_thresh_mm=dist_thresh_mm)
        rep["part_nr"] = pm["part_nr"]
        reports.append(rep)
    return reports


def aggregate(reports):
    """Mean of the per-part metrics."""
    if not reports:
        return {}
    keys = ["precision", "recall", "f1", "mean_loc_err_mm", "mean_ang_err_deg"]
    agg = {}
    for k in keys:
        vals = [r[k] for r in reports if not (isinstance(r[k], float) and np.isnan(r[k]))]
        agg[k] = float(np.mean(vals)) if vals else float("nan")
    agg["n_parts"] = len(reports)
    agg["total_gt"] = sum(r["n_gt"] for r in reports)
    agg["total_tp"] = sum(r["tp"] for r in reports)
    return agg


def _best_ckpt_eval_fn(val_s, val_m, infer_builder, save_fn, ckpt_path,
                       dist_thresh_mm, min_votes, patience, state):
    """Build eval_fn(model, meta, epoch) shared by the graph backbones: scores
    val F1, saves the best model via save_fn(model, meta, ckpt_path), and returns
    True to early-stop after `patience` stale evals. `state` is a mutable dict
    {f1, since, saved} carried across calls (#5)."""
    def eval_fn(m, mt, ep):
        if not val_s:
            return False
        agg = aggregate(evaluate(val_s, val_m, infer_builder(m, mt),
                                 dist_thresh_mm=dist_thresh_mm, min_votes=min_votes))
        f1 = agg.get("f1", float("nan"))
        if isinstance(f1, float) and np.isnan(f1):
            return False
        if f1 > state["f1"]:
            state["f1"], state["since"] = f1, 0
            if ckpt_path:
                save_fn(m, mt, ckpt_path); state["saved"] = True
            logger.info("  epoch %d  val F1=%.4f (best) -> checkpoint", ep, f1)
        else:
            state["since"] += 1
            logger.info("  epoch %d  val F1=%.4f (no improve %d)", ep, f1, state["since"])
        return bool(patience) and state["since"] >= patience
    return eval_fn


# ---------------------------------------------------------------------------
# main train+eval
# ---------------------------------------------------------------------------

def train_and_eval(source, backbone="mlp", epochs=300, val_frac=0.2,
                   max_parts=None, device="cpu", dist_thresh_mm=5.0, seed=0,
                   op_cache_dir=None, max_gpu_verts=60000,
                   ckpt_path=None, ckpt_every=0, min_votes=2, heat_loss="bce",
                   focal_gamma=2.0, lr_decay_every=0, lr_decay_rate=0.5,
                   accum_steps=1, low_memory=False, eval_every=10, patience=0):
    """Load parts, prepare targets, train, and evaluate. Returns (train_agg, val_agg)."""
    parts = []
    for i, part in enumerate(jd.iter_parts(source)):
        parts.append(part)
        if max_parts and len(parts) >= max_parts:
            break
    if not parts:
        raise SystemExit(f"no parts found in {source}")
    logger.info("loaded %d parts", len(parts))

    ids = [p.part_nr for p in parts]
    train_ids, val_ids = split_part_ids(ids, val_frac=val_frac, seed=seed)
    # guarantee at least one part trains even on a tiny corpus
    if not train_ids:
        train_ids = {ids[0]}
        val_ids.discard(ids[0])

    train_s, train_m, val_s, val_m = [], [], [], []
    for p in parts:
        if p.n_cps == 0:
            continue
        s = cpr.prepare_sample(p, dedup=True)
        meta = {"part_nr": p.part_nr, "vertices": p.vertices}
        if p.part_nr in val_ids:
            val_s.append(s); val_m.append(meta)
        else:
            train_s.append(s); train_m.append(meta)
    logger.info("train parts=%d  val parts=%d", len(train_s), len(val_s))

    if backbone == "mlp":
        import torch
        model = cpr.train_cpmlp(train_s, epochs=epochs, device=device,
                                log_every=max(1, epochs // 6))

        def infer_arr(s, pm):
            model.eval()
            with torch.no_grad():
                x = torch.tensor(s["verts_norm"], dtype=torch.float32, device=device)
                return cpr.pred_to_array(model(x), offset_scale=s["scale"])
    elif backbone == "knngraph":
        # torch-only backbone (no native geometry deps) -> runs on native Windows.
        def _infer_builder(m, mt):
            def f(s, pm):
                return cpr.infer_knngraph(m, mt, s["verts_norm"], device=device,
                                          max_gpu_verts=max_gpu_verts,
                                          offset_scale=s["scale"])
            return f

        state = {"f1": -1.0, "saved": False, "since": 0}
        eval_fn = _best_ckpt_eval_fn(val_s, val_m, _infer_builder, cpr.save_knngraph,
                                     ckpt_path, dist_thresh_mm, min_votes, patience, state)
        model, meta = cpr.train_knngraph_regressor(
            train_s, epochs=epochs, device=device,
            log_every=max(1, epochs // 10), max_gpu_verts=max_gpu_verts,
            ckpt_path=ckpt_path, ckpt_every=ckpt_every, seed=seed,
            eval_fn=(eval_fn if val_s else None), eval_every=eval_every,
            heat_loss=heat_loss, focal_gamma=focal_gamma,
            lr_decay_every=lr_decay_every, lr_decay_rate=lr_decay_rate,
            accum_steps=accum_steps)
        if ckpt_path and val_s and not state["saved"]:
            cpr.save_knngraph(model, meta, ckpt_path)
        infer_arr = _infer_builder(model, meta)
    else:  # diffusionnet
        def _infer_builder(m, mt):
            def f(s, pm):
                return cpr.infer_diffusionnet(
                    m, mt, pm["vertices"], s["faces"], s["verts_norm"],
                    op_cache_dir=op_cache_dir, device=device,
                    max_gpu_verts=max_gpu_verts, offset_scale=s["scale"])
            return f

        # #5: validation-driven best checkpoint + early stopping.
        state = {"f1": -1.0, "saved": False, "since": 0}
        eval_fn = _best_ckpt_eval_fn(val_s, val_m, _infer_builder, cpr.save_diffusionnet,
                                     ckpt_path, dist_thresh_mm, min_votes, patience, state)
        model, meta = cpr.train_diffusionnet_regressor(
            train_s, epochs=epochs, device=device, op_cache_dir=op_cache_dir,
            log_every=max(1, epochs // 10), max_gpu_verts=max_gpu_verts,
            ckpt_path=ckpt_path, ckpt_every=ckpt_every, seed=seed,
            eval_fn=(eval_fn if val_s else None), eval_every=eval_every,
            heat_loss=heat_loss, focal_gamma=focal_gamma,
            lr_decay_every=lr_decay_every, lr_decay_rate=lr_decay_rate,
            accum_steps=accum_steps, low_memory=low_memory)
        if ckpt_path and val_s and not state["saved"]:
            cpr.save_diffusionnet(model, meta, ckpt_path)
        infer_arr = _infer_builder(model, meta)

    train_reports = evaluate(train_s, train_m, infer_arr,
                             dist_thresh_mm=dist_thresh_mm, min_votes=min_votes)
    val_reports = (evaluate(val_s, val_m, infer_arr,
                            dist_thresh_mm=dist_thresh_mm, min_votes=min_votes)
                   if val_s else [])
    return aggregate(train_reports), aggregate(val_reports), train_reports, val_reports


def main(argv=None):
    ap = argparse.ArgumentParser(description="Train connection-point detector on ABB JSON data")
    ap.add_argument("source", help="corpus: a directory, a single big JSON array, or one part file")
    ap.add_argument("--backbone", choices=["mlp", "diffusionnet", "knngraph"],
                    default="mlp")
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--val-frac", type=float, default=0.2)
    ap.add_argument("--max-parts", type=int, default=None)
    ap.add_argument("--dist-thresh-mm", type=float, default=5.0)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--op-cache-dir", default=None,
                    help="cache dir for DiffusionNet mesh operators "
                         "(strongly recommended for the 429-file corpus)")
    ap.add_argument("--max-gpu-verts", type=int, default=60000,
                    help="parts larger than this run on CPU even with --device cuda "
                         "(avoids OOM on small GPUs); 0 forces GPU for all parts")
    ap.add_argument("--ckpt", default=None,
                    help="path to save the trained DiffusionNet model (weights+meta); "
                         "reload later with cp_regressor.load_diffusionnet")
    ap.add_argument("--ckpt-every", type=int, default=0,
                    help="also checkpoint every N epochs (crash-resilient); 0 = only at end")
    ap.add_argument("--out", default=None,
                    help="write aggregate + per-part metrics (with timing) to this JSON file")
    ap.add_argument("--min-votes", type=int, default=2,
                    help="min voting vertices per decoded CP (>1 raises precision)")
    ap.add_argument("--heat-loss", choices=["bce", "focal"], default="bce",
                    help="heatmap loss: weighted BCE or quality-focal")
    ap.add_argument("--focal-gamma", type=float, default=2.0)
    ap.add_argument("--lr-decay-every", type=int, default=0,
                    help="StepLR step size in epochs (0 disables the schedule)")
    ap.add_argument("--lr-decay-rate", type=float, default=0.5)
    ap.add_argument("--accum-steps", type=int, default=1,
                    help="gradient accumulation over N parts before opt.step")
    ap.add_argument("--low-memory", action="store_true",
                    help="reload eigenbasis from cache per step instead of holding "
                         "all in RAM (for corpora larger than memory)")
    ap.add_argument("--eval-every", type=int, default=10,
                    help="run val eval + best-checkpoint every N epochs (diffusionnet)")
    ap.add_argument("--patience", type=int, default=0,
                    help="early stop after this many stale val evals (0 disables)")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    import time
    t0 = time.time()
    tr, va, tr_reps, va_reps = train_and_eval(
        args.source, backbone=args.backbone, epochs=args.epochs,
        val_frac=args.val_frac, max_parts=args.max_parts,
        device=args.device, dist_thresh_mm=args.dist_thresh_mm, seed=args.seed,
        op_cache_dir=args.op_cache_dir, max_gpu_verts=args.max_gpu_verts,
        ckpt_path=args.ckpt, ckpt_every=args.ckpt_every, min_votes=args.min_votes,
        heat_loss=args.heat_loss, focal_gamma=args.focal_gamma,
        lr_decay_every=args.lr_decay_every, lr_decay_rate=args.lr_decay_rate,
        accum_steps=args.accum_steps, low_memory=args.low_memory,
        eval_every=args.eval_every, patience=args.patience)
    elapsed = time.time() - t0
    print("\n=== TRAIN (aggregate) ===");  print(json.dumps(tr, indent=2))
    print("\n=== VAL (aggregate) ===");    print(json.dumps(va, indent=2))
    print("\nelapsed: %.0f s (%.2f h)" % (elapsed, elapsed / 3600))

    if args.out:
        with open(args.out, "w") as fh:
            json.dump({"train_aggregate": tr, "val_aggregate": va,
                       "train_per_part": tr_reps, "val_per_part": va_reps,
                       "elapsed_sec": elapsed}, fh, indent=2)
        print("metrics written to", args.out)


if __name__ == "__main__":
    main()
