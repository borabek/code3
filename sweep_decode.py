# Decode-parameter sweep for a trained connection-point regressor.
#
# Why this exists: the keypoint metrics (precision / F1) are *decode-bound*, not
# model-bound -- the network's per-vertex (heatmap, offset, direction) output is
# fixed once trained, but how many discrete connection points you read out of it
# depends entirely on three knobs in cp_targets.decode_predictions:
#
#   heatmap_thresh   min sigmoid heat for a vertex to vote   (higher -> fewer FP)
#   min_votes        min voting vertices per accepted cluster (higher -> fewer FP)
#   nms_clearance_mm vote-merge radius = max(2*sigma, this)   (larger -> fewer FP)
#
# This tool runs inference ONCE per part (the expensive step), caches the (N,7)
# arrays, then sweeps a grid of those three knobs and reports the configuration
# that maximises F1 on the validation split -- no retraining. Point it at the
# same corpus / --val-frac / --seed / --max-parts you trained with so the val
# split matches exactly.
#
# Example:
#   python sweep_decode.py checkpoints/cp_knngraph_best.ckpt parts/ \
#       --val-frac 0.2 --seed 0
#
# Take the winning row's --heatmap-thresh / --min-votes / --nms-clearance-mm and
# pass them to train_cp.py (or your inference) to lock the operating point in.

import os
import json
import argparse
import logging
import itertools

import numpy as np

import json_dataset as jd
import cp_targets as ct
import cp_regressor as cpr
import metrics as mcp
from train_cp import split_part_ids, aggregate, _nms_radius

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# per-backbone inference: sample -> (N,7) numpy (heatmap sigmoid'd, dir unit)
# ---------------------------------------------------------------------------

def _build_infer(backbone, model, meta, device, op_cache_dir, max_gpu_verts):
    """Mirror the inference path train_cp.train_and_eval builds for each backbone."""
    if backbone == "knngraph":
        def f(s, pm):
            return cpr.infer_knngraph(model, meta, s["verts_norm"], device=device,
                                      max_gpu_verts=max_gpu_verts,
                                      offset_scale=s["scale"])
    elif backbone == "diffusionnet":
        def f(s, pm):
            return cpr.infer_diffusionnet(model, meta, pm["vertices"], s["faces"],
                                          s["verts_norm"], op_cache_dir=op_cache_dir,
                                          device=device, max_gpu_verts=max_gpu_verts,
                                          offset_scale=s["scale"])
    elif backbone == "mlp":
        import torch

        def f(s, pm):
            model.eval()
            with torch.no_grad():
                x = torch.tensor(s["verts_norm"], dtype=torch.float32, device=device)
                return cpr.pred_to_array(model(x), offset_scale=s["scale"])
    else:
        raise ValueError(f"unknown backbone {backbone!r}")
    return f


# ---------------------------------------------------------------------------
# load + split the corpus the same way train_cp does, keep one split
# ---------------------------------------------------------------------------

def _load_split(source, split, val_frac, seed, max_parts):
    """Return (samples, metas) for the requested split ('val' | 'train' | 'all'),
    skipping parts with no connection points (exactly as training does)."""
    parts = []
    for part in jd.iter_parts(source):
        parts.append(part)
        if max_parts and len(parts) >= max_parts:
            break
    if not parts:
        raise SystemExit(f"no parts found in {source}")

    ids = [p.part_nr for p in parts]
    train_ids, val_ids = split_part_ids(ids, val_frac=val_frac, seed=seed)
    if not train_ids:                       # mirror train_cp's tiny-corpus guard
        train_ids = {ids[0]}
        val_ids.discard(ids[0])

    samples, metas = [], []
    for p in parts:
        if p.n_cps == 0:
            continue
        in_val = p.part_nr in val_ids
        if split == "val" and not in_val:
            continue
        if split == "train" and in_val:
            continue
        samples.append(cpr.prepare_sample(p, dedup=True))
        metas.append({"part_nr": p.part_nr, "vertices": p.vertices})
    return samples, metas


# ---------------------------------------------------------------------------
# score one decode configuration over the cached prediction arrays
# ---------------------------------------------------------------------------

def _score(arrs, samples, metas, thr, min_votes, nms_clear, dist_thresh_mm):
    reports = []
    for arr, s, pm in zip(arrs, samples, metas):
        preds = ct.decode_predictions(pm["vertices"], arr, heatmap_thresh=thr,
                                       nms_radius_mm=_nms_radius(s, nms_clear),
                                       min_votes=min_votes)
        rep = mcp.keypoint_report(preds, s["gt_points"], s["gt_directions"],
                                  dist_thresh_mm=dist_thresh_mm)
        reports.append(rep)
    return aggregate(reports)


def _parse_floats(text):
    return [float(x) for x in text.split(",") if x.strip()]


def _parse_ints(text):
    return [int(x) for x in text.split(",") if x.strip()]


def main(argv=None):
    ap = argparse.ArgumentParser(
        description="Sweep decode params on a trained CP regressor (no retraining)")
    ap.add_argument("ckpt", help="trained checkpoint (e.g. checkpoints/cp_knngraph_best.ckpt)")
    ap.add_argument("source", help="same corpus you trained on (dir / big JSON / part file)")
    ap.add_argument("--split", choices=["val", "train", "all"], default="val",
                    help="which split to tune on (default val -- the honest one)")
    ap.add_argument("--val-frac", type=float, default=0.2,
                    help="must match the training run so the val split is identical")
    ap.add_argument("--seed", type=int, default=0, help="must match the training run")
    ap.add_argument("--max-parts", type=int, default=None,
                    help="must match the training run (caps parts materialised)")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--dist-thresh-mm", type=float, default=5.0,
                    help="a prediction matches a GT point within this radius")
    ap.add_argument("--op-cache-dir", default=None,
                    help="DiffusionNet operator cache (diffusionnet backbone only)")
    ap.add_argument("--max-gpu-verts", type=int, default=60000)
    ap.add_argument("--thresholds", default="0.3,0.4,0.5,0.6,0.7,0.8",
                    help="comma-separated heatmap thresholds to try")
    ap.add_argument("--min-votes-list", default="1,2,3,4,6",
                    help="comma-separated min-votes values to try")
    ap.add_argument("--nms-list", default="5,10,15",
                    help="comma-separated nms clearance (mm) values to try")
    ap.add_argument("--metric", choices=["micro_f1", "f1"], default="micro_f1",
                    help="rank configs by pooled (micro) or per-part-mean (macro) F1")
    ap.add_argument("--top", type=int, default=15, help="how many rows to print")
    ap.add_argument("--out", default=None, help="write the full sweep table to this JSON")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    model, meta, backbone = cpr.load_model(args.ckpt, device=args.device)
    logger.info("loaded %s checkpoint <- %s", backbone, args.ckpt)

    samples, metas = _load_split(args.source, args.split, args.val_frac,
                                 args.seed, args.max_parts)
    if not samples:
        raise SystemExit(f"no parts in the '{args.split}' split (check --val-frac/--seed)")
    total_gt = sum(len(s["gt_points"]) for s in samples)
    logger.info("%s split: %d parts, %d ground-truth connection points",
                args.split, len(samples), total_gt)

    # the expensive step, done exactly once: cache each part's (N,7) prediction
    infer = _build_infer(backbone, model, meta, args.device,
                         args.op_cache_dir, args.max_gpu_verts)
    logger.info("running inference once per part ...")
    arrs = [infer(s, pm) for s, pm in zip(samples, metas)]

    thresholds = _parse_floats(args.thresholds)
    min_votes_list = _parse_ints(args.min_votes_list)
    nms_list = _parse_floats(args.nms_list)
    grid = list(itertools.product(thresholds, min_votes_list, nms_list))
    logger.info("sweeping %d decode configurations ...", len(grid))

    rows = []
    for thr, mv, nms in grid:
        agg = _score(arrs, samples, metas, thr, mv, nms, args.dist_thresh_mm)
        rows.append({
            "heatmap_thresh": thr, "min_votes": mv, "nms_clearance_mm": nms,
            "micro_f1": agg["micro_f1"], "micro_precision": agg["micro_precision"],
            "micro_recall": agg["micro_recall"], "f1": agg["f1"],
            "precision": agg["precision"], "recall": agg["recall"],
            "tp": agg["total_tp"], "fp": agg["total_fp"], "fn": agg["total_fn"],
        })

    rows.sort(key=lambda r: r[args.metric], reverse=True)

    hdr = ("  thr  votes   nms |   F1(micro)  prec   recall |    TP    FP   FN")
    print("\n=== top %d decode configs on '%s' (ranked by %s) ==="
          % (min(args.top, len(rows)), args.split, args.metric))
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))
    for r in rows[:args.top]:
        print("  %.2f   %3d  %5.1f | %9.4f  %.4f  %.4f | %5d %5d %4d"
              % (r["heatmap_thresh"], r["min_votes"], r["nms_clearance_mm"],
                 r["micro_f1"], r["micro_precision"], r["micro_recall"],
                 r["tp"], r["fp"], r["fn"]))

    # baseline = the old hard-coded decode (thr=0.3, min_votes=2, nms=5)
    base = next((r for r in rows if r["heatmap_thresh"] == 0.3
                 and r["min_votes"] == 2 and r["nms_clearance_mm"] == 5.0), None)
    best = rows[0]
    if base:
        print("\n  old default (thr=0.30 votes=2 nms=5.0): "
              "F1=%.4f  prec=%.4f  recall=%.4f  (FP=%d)"
              % (base["micro_f1"], base["micro_precision"],
                 base["micro_recall"], base["fp"]))
    print("  BEST  (thr=%.2f votes=%d nms=%.1f):              "
          "F1=%.4f  prec=%.4f  recall=%.4f  (FP=%d)"
          % (best["heatmap_thresh"], best["min_votes"], best["nms_clearance_mm"],
             best["micro_f1"], best["micro_precision"], best["micro_recall"],
             best["fp"]))
    print("\nlock it in:")
    print("  python train_cp.py %s --backbone %s --heatmap-thresh %.2f "
          "--min-votes %d --nms-clearance-mm %.1f --resume --epochs <target>"
          % (args.source, backbone, best["heatmap_thresh"],
             best["min_votes"], best["nms_clearance_mm"]))

    if args.out:
        with open(args.out, "w", encoding="utf-8") as fh:
            json.dump({"backbone": backbone, "split": args.split,
                       "n_parts": len(samples), "total_gt": total_gt,
                       "rows": rows}, fh, indent=2)
        print("\nfull sweep table written to", args.out)


if __name__ == "__main__":
    main()
