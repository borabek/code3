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

import os
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
    """Per-part means (macro) + pooled detection counts (micro) in one dict.

    macro keys (accuracy/precision/recall/f1/loc/ang) average each part's
    score; micro_* pool TP/FP/FN over all parts first, which weights every
    ground-truth point equally and is what 'overall corpus accuracy' means.
    """
    if not reports:
        return {}
    keys = ["accuracy", "precision", "recall", "f1",
            "mean_loc_err_mm", "mean_ang_err_deg"]
    agg = {}
    for k in keys:
        vals = [r[k] for r in reports if not (isinstance(r[k], float) and np.isnan(r[k]))]
        agg[k] = float(np.mean(vals)) if vals else float("nan")
    agg["n_parts"] = len(reports)
    agg["total_gt"] = sum(r["n_gt"] for r in reports)
    agg["total_tp"] = sum(r["tp"] for r in reports)
    agg["total_fp"] = sum(r["fp"] for r in reports)
    agg["total_fn"] = sum(r["fn"] for r in reports)
    tp, fp, fn = agg["total_tp"], agg["total_fp"], agg["total_fn"]
    agg["micro_precision"] = tp / (tp + fp) if (tp + fp) else 0.0
    agg["micro_recall"] = tp / (tp + fn) if (tp + fn) else 0.0
    pr = agg["micro_precision"] + agg["micro_recall"]
    agg["micro_f1"] = (2 * agg["micro_precision"] * agg["micro_recall"] / pr
                       if pr else 0.0)
    agg["micro_accuracy"] = tp / (tp + fp + fn) if (tp + fp + fn) else 1.0
    return agg


def _val_metrics_fn(val_s, val_m, infer_builder, dist_thresh_mm, min_votes,
                    heatmap_thresh=0.3, nms_clearance_mm=5.0):
    """Build metrics_fn(model, meta) for the training bookkeeper: decode and
    score the validation parts, returning the aggregate keypoint metrics
    (accuracy / precision / recall / f1 / loc / ang). None when no val set --
    the loop then skips val evaluation and best-by-F1 tracking.

    heatmap_thresh / nms_clearance_mm are the decode knobs that govern the
    precision/recall trade-off; they must match the values used for the final
    train/val evaluate() so the best-by-F1 checkpoint is selected under the same
    decoding the report uses."""
    if not val_s:
        return None

    def metrics_fn(m, mt):
        return aggregate(evaluate(val_s, val_m, infer_builder(m, mt),
                                  dist_thresh_mm=dist_thresh_mm,
                                  heatmap_thresh=heatmap_thresh,
                                  min_votes=min_votes,
                                  nms_clearance_mm=nms_clearance_mm))
    return metrics_fn


# ---------------------------------------------------------------------------
# main train+eval
# ---------------------------------------------------------------------------

def train_and_eval(source, backbone="mlp", epochs=300, val_frac=0.2,
                   max_parts=None, device="cpu", dist_thresh_mm=5.0, seed=0,
                   op_cache_dir=None, max_gpu_verts=60000,
                   best_path=None, last_path=None, history_path=None,
                   save_every=1, resume_from=None,
                   min_votes=2, heat_loss="bce",
                   focal_gamma=2.0, lr_decay_every=0, lr_decay_rate=0.5,
                   accum_steps=1, low_memory=False, eval_every=10, patience=0,
                   heatmap_thresh=0.3, nms_clearance_mm=5.0, heat_pos_weight=50.0):
    """Load parts, prepare targets, train, and evaluate. Returns (train_agg, val_agg).

    best_path / last_path : full-state .ckpt files (best-by-val-F1, rolling
    resumable snapshot). resume_from continues a previous run from its saved
    epoch toward `epochs` total -- including on another machine after a git
    pull. history_path collects the per-eval val metrics as JSON.
    """
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

        def _infer_builder(m, mt):
            def f(s, pm):
                m.eval()
                with torch.no_grad():
                    x = torch.tensor(s["verts_norm"], dtype=torch.float32,
                                     device=device)
                    return cpr.pred_to_array(m(x), offset_scale=s["scale"])
            return f

        metrics_fn = _val_metrics_fn(val_s, val_m, _infer_builder,
                                     dist_thresh_mm, min_votes,
                                     heatmap_thresh=heatmap_thresh,
                                     nms_clearance_mm=nms_clearance_mm)
        model = cpr.train_cpmlp(
            train_s, epochs=epochs, device=device,
            log_every=max(1, epochs // 6),
            metrics_fn=metrics_fn, eval_every=eval_every, patience=patience,
            best_path=best_path, last_path=last_path, save_every=save_every,
            history_path=history_path, resume_from=resume_from, seed=seed)
        infer_arr = _infer_builder(model, None)
    elif backbone == "knngraph":
        # torch-only backbone (no native geometry deps) -> runs on native Windows.
        def _infer_builder(m, mt):
            def f(s, pm):
                return cpr.infer_knngraph(m, mt, s["verts_norm"], device=device,
                                          max_gpu_verts=max_gpu_verts,
                                          offset_scale=s["scale"])
            return f

        metrics_fn = _val_metrics_fn(val_s, val_m, _infer_builder,
                                     dist_thresh_mm, min_votes,
                                     heatmap_thresh=heatmap_thresh,
                                     nms_clearance_mm=nms_clearance_mm)
        model, meta = cpr.train_knngraph_regressor(
            train_s, epochs=epochs, device=device,
            log_every=max(1, epochs // 10), max_gpu_verts=max_gpu_verts,
            metrics_fn=metrics_fn, eval_every=eval_every, patience=patience,
            best_path=best_path, last_path=last_path, save_every=save_every,
            history_path=history_path, resume_from=resume_from, seed=seed,
            heat_pos_weight=heat_pos_weight,
            heat_loss=heat_loss, focal_gamma=focal_gamma,
            lr_decay_every=lr_decay_every, lr_decay_rate=lr_decay_rate,
            accum_steps=accum_steps)
        infer_arr = _infer_builder(model, meta)
    else:  # diffusionnet
        def _infer_builder(m, mt):
            def f(s, pm):
                return cpr.infer_diffusionnet(
                    m, mt, pm["vertices"], s["faces"], s["verts_norm"],
                    op_cache_dir=op_cache_dir, device=device,
                    max_gpu_verts=max_gpu_verts, offset_scale=s["scale"])
            return f

        metrics_fn = _val_metrics_fn(val_s, val_m, _infer_builder,
                                     dist_thresh_mm, min_votes,
                                     heatmap_thresh=heatmap_thresh,
                                     nms_clearance_mm=nms_clearance_mm)
        model, meta = cpr.train_diffusionnet_regressor(
            train_s, epochs=epochs, device=device, op_cache_dir=op_cache_dir,
            log_every=max(1, epochs // 10), max_gpu_verts=max_gpu_verts,
            metrics_fn=metrics_fn, eval_every=eval_every, patience=patience,
            best_path=best_path, last_path=last_path, save_every=save_every,
            history_path=history_path, resume_from=resume_from, seed=seed,
            heat_pos_weight=heat_pos_weight,
            heat_loss=heat_loss, focal_gamma=focal_gamma,
            lr_decay_every=lr_decay_every, lr_decay_rate=lr_decay_rate,
            accum_steps=accum_steps, low_memory=low_memory)
        infer_arr = _infer_builder(model, meta)

    train_reports = evaluate(train_s, train_m, infer_arr,
                             dist_thresh_mm=dist_thresh_mm,
                             heatmap_thresh=heatmap_thresh, min_votes=min_votes,
                             nms_clearance_mm=nms_clearance_mm)
    val_reports = (evaluate(val_s, val_m, infer_arr,
                            dist_thresh_mm=dist_thresh_mm,
                            heatmap_thresh=heatmap_thresh, min_votes=min_votes,
                            nms_clearance_mm=nms_clearance_mm)
                   if val_s else [])
    return aggregate(train_reports), aggregate(val_reports), train_reports, val_reports


def _print_agg(title, agg):
    """Readable metric block: per-part means + pooled (micro) values."""
    print(f"\n=== {title} ===")
    if not agg:
        print("  (no parts)")
        return
    print("  parts=%d  GT=%d  TP=%d  FP=%d  FN=%d"
          % (agg["n_parts"], agg["total_gt"], agg["total_tp"],
             agg["total_fp"], agg["total_fn"]))
    for k in ("accuracy", "precision", "recall", "f1"):
        print("  %-9s : %.2f%%   (pooled %.2f%%)"
              % (k, agg[k] * 100, agg["micro_" + k] * 100))
    print("  loc error : %.3f mm    ang error : %.2f deg"
          % (agg["mean_loc_err_mm"], agg["mean_ang_err_deg"]))


def main(argv=None):
    ap = argparse.ArgumentParser(description="Train connection-point detector on ABB JSON data")
    ap.add_argument("source", help="corpus: a directory, a single big JSON array, or one part file")
    ap.add_argument("--backbone", choices=["mlp", "diffusionnet", "knngraph"],
                    default="mlp")
    ap.add_argument("--epochs", type=int, default=300,
                    help="TOTAL epochs to reach; with --resume, training continues "
                         "from the checkpoint's epoch up to this target")
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
    ap.add_argument("--ckpt-dir", default="checkpoints",
                    help="directory for the .ckpt files (default: checkpoints/, "
                         "tracked in git so other machines get them on clone/pull)")
    ap.add_argument("--run-name", default=None,
                    help="basename for the checkpoint files "
                         "(default: cp_<backbone>)")
    ap.add_argument("--resume", action="store_true",
                    help="continue training from <ckpt-dir>/<run-name>_last.ckpt "
                         "(falls back to _best.ckpt); restores model+optimizer+"
                         "scheduler+epoch, so a run can hop between machines")
    ap.add_argument("--resume-from", default=None,
                    help="explicit checkpoint path to resume from (overrides --resume)")
    ap.add_argument("--ckpt", default=None,
                    help="explicit path for the BEST checkpoint "
                         "(default <ckpt-dir>/<run-name>_best.ckpt); reload with "
                         "cp_regressor.load_model")
    ap.add_argument("--ckpt-every", type=int, default=1,
                    help="write the resumable last.ckpt every N epochs "
                         "(default 1; 0 = only at the end)")
    ap.add_argument("--out", default=None,
                    help="write aggregate + per-part metrics (with timing) to this JSON file")
    ap.add_argument("--export-model", default=None,
                    help="after training, write a lean inference-only .pt from the "
                         "best checkpoint (weights+meta+backbone+decode params, no "
                         "optimizer/history). Load it with predict.py")
    ap.add_argument("--min-votes", type=int, default=2,
                    help="min voting vertices per decoded CP (>1 raises precision)")
    ap.add_argument("--heatmap-thresh", type=float, default=0.3,
                    help="decode: min sigmoid heat for a vertex to vote for a CP. "
                         "Higher -> fewer false positives, but too high fires "
                         "nothing. This is just the training-time eval point; pick "
                         "the deployment operating point with sweep_decode.py")
    ap.add_argument("--nms-clearance-mm", type=float, default=5.0,
                    help="decode: votes are merged within max(2*sigma, this) mm; "
                         "larger collapses scattered votes into fewer detections")
    ap.add_argument("--heat-pos-weight", type=float, default=50.0,
                    help="heatmap BCE positive weight (1 + w*h). The heat target is "
                         "~99%% background, so this must stay high (default 50) or "
                         "the model collapses to ~0 heat everywhere (recall 0). To "
                         "raise precision prefer --heat-loss focal over lowering it "
                         "(knngraph/diffusionnet only)")
    ap.add_argument("--heat-loss", choices=["bce", "focal", "centernet"],
                    default="bce",
                    help="heatmap loss. 'centernet' is the penalty-reduced focal "
                         "loss normalised by #keypoints -- the right choice for this "
                         "sparse heatmap; 'bce'/'focal' average over all vertices and "
                         "either over-fire or collapse on the 99%% background")
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

    # checkpoint file layout: best (by val F1), last (resume point), history
    run_name = args.run_name or f"cp_{args.backbone}"
    if args.ckpt:
        best_path = args.ckpt
        stem = os.path.splitext(args.ckpt)[0]
        last_path = stem + "_last.ckpt"
        history_path = stem + "_history.json"
    else:
        best_path = os.path.join(args.ckpt_dir, run_name + "_best.ckpt")
        last_path = os.path.join(args.ckpt_dir, run_name + "_last.ckpt")
        history_path = os.path.join(args.ckpt_dir, run_name + "_history.json")

    resume_from = args.resume_from
    if resume_from is None and args.resume:
        if os.path.exists(last_path):
            resume_from = last_path
        elif os.path.exists(best_path):
            resume_from = best_path
        else:
            logger.warning("--resume: no checkpoint at %s or %s -- starting fresh",
                           last_path, best_path)
    if resume_from:
        logger.info("resuming from %s", resume_from)

    import time
    t0 = time.time()
    tr, va, tr_reps, va_reps = train_and_eval(
        args.source, backbone=args.backbone, epochs=args.epochs,
        val_frac=args.val_frac, max_parts=args.max_parts,
        device=args.device, dist_thresh_mm=args.dist_thresh_mm, seed=args.seed,
        op_cache_dir=args.op_cache_dir, max_gpu_verts=args.max_gpu_verts,
        best_path=best_path, last_path=last_path, history_path=history_path,
        save_every=args.ckpt_every, resume_from=resume_from,
        min_votes=args.min_votes, heatmap_thresh=args.heatmap_thresh,
        nms_clearance_mm=args.nms_clearance_mm, heat_pos_weight=args.heat_pos_weight,
        heat_loss=args.heat_loss, focal_gamma=args.focal_gamma,
        lr_decay_every=args.lr_decay_every, lr_decay_rate=args.lr_decay_rate,
        accum_steps=args.accum_steps, low_memory=args.low_memory,
        eval_every=args.eval_every, patience=args.patience)
    elapsed = time.time() - t0
    _print_agg("TRAIN metrics", tr)
    _print_agg("VAL metrics", va)
    print("\nelapsed: %.0f s (%.2f h)" % (elapsed, elapsed / 3600))

    print("\ncheckpoints:")
    for tag, p in (("best", best_path), ("last", last_path),
                   ("history", history_path)):
        print("  %-7s %s%s" % (tag, p,
                               "" if os.path.exists(p) else "  (not written)"))
    print("\nto continue this run later (same or another machine):")
    print("  git add %s && git commit -m 'training checkpoint' && git push"
          % (args.ckpt_dir if not args.ckpt else os.path.dirname(best_path) or "."))
    print("  # then on the other machine: git pull && \\")
    print("  python train_cp.py %s --backbone %s --resume --epochs <target>"
          % (args.source, args.backbone))

    if args.out:
        with open(args.out, "w") as fh:
            json.dump({"train_aggregate": tr, "val_aggregate": va,
                       "train_per_part": tr_reps, "val_per_part": va_reps,
                       "elapsed_sec": elapsed}, fh, indent=2)
        print("\nmetrics written to", args.out)

    if args.export_model:
        # bundle the decode operating point used for this run so the exported
        # model is self-contained (the weights alone are ambiguous for keypoints)
        decode = {"heatmap_thresh": args.heatmap_thresh, "min_votes": args.min_votes,
                  "nms_clearance_mm": args.nms_clearance_mm,
                  "dist_thresh_mm": args.dist_thresh_mm}
        src = best_path if os.path.exists(best_path) else last_path
        if os.path.exists(src):
            cpr.export_inference_checkpoint(src, args.export_model, decode=decode)
            print("\nexported inference model -> %s\n  (from %s, decode=%s)"
                  % (args.export_model, src, decode))
            print("  run it:  python predict.py %s <mesh-or-corpus> --out preds.json"
                  % args.export_model)
        else:
            print("\n--export-model: no checkpoint found to export (%s)" % src)


if __name__ == "__main__":
    main()
