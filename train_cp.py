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
# Two backbones:
#   --backbone mlp           : light per-vertex MLP on xyz (only needs torch).
#                              Memorises geometry -> good for overfit smoke-tests,
#                              but has no receptive field so it does NOT generalise.
#   --backbone diffusionnet  : the production model (needs the diffusion_net pkg).
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
import metrics_cp as mcp

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

def _nms_radius(s):
    """NMS radius: comfortably below the closest pair of GT blocks."""
    if len(s["gt_points"]) > 1:
        D = np.linalg.norm(s["gt_points"][:, None, :] - s["gt_points"][None, :, :], axis=2)
        np.fill_diagonal(D, np.inf)
        return min(max(2 * s["sigma"], 5.0), 0.5 * float(D.min()))
    return max(2 * s["sigma"], 5.0)


def evaluate(samples, parts_meta, infer_arr, dist_thresh_mm=5.0, heatmap_thresh=0.3):
    """Decode predictions for each sample and aggregate keypoint metrics.

    infer_arr(sample, part_meta) -> (N,7) numpy array of per-vertex predictions
    (heatmap already sigmoid'd, direction unit). This abstracts the backbone so
    the same decode+metrics path serves both the MLP and DiffusionNet models.
    """
    reports = []
    for s, pm in zip(samples, parts_meta):
        arr = infer_arr(s, pm)
        preds = ct.decode_predictions(pm["vertices"], arr,
                                      heatmap_thresh=heatmap_thresh,
                                      nms_radius_mm=_nms_radius(s))
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


# ---------------------------------------------------------------------------
# main train+eval
# ---------------------------------------------------------------------------

def train_and_eval(source, backbone="mlp", epochs=300, val_frac=0.2,
                   max_parts=None, device="cpu", dist_thresh_mm=5.0, seed=0,
                   op_cache_dir=None):
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
                return cpr.pred_to_array(model(x))
    else:
        model, meta = cpr.train_diffusionnet_regressor(
            train_s, epochs=epochs, device=device, op_cache_dir=op_cache_dir,
            log_every=max(1, epochs // 10))

        def infer_arr(s, pm):
            return cpr.infer_diffusionnet(model, meta, pm["vertices"], s["faces"],
                                          s["verts_norm"], op_cache_dir=op_cache_dir,
                                          device=device)

    train_reports = evaluate(train_s, train_m, infer_arr, dist_thresh_mm=dist_thresh_mm)
    val_reports = (evaluate(val_s, val_m, infer_arr, dist_thresh_mm=dist_thresh_mm)
                   if val_s else [])
    return aggregate(train_reports), aggregate(val_reports), train_reports, val_reports


def main(argv=None):
    ap = argparse.ArgumentParser(description="Train connection-point detector on ABB JSON data")
    ap.add_argument("source", help="corpus: a directory, a single big JSON array, or one part file")
    ap.add_argument("--backbone", choices=["mlp", "diffusionnet"], default="mlp")
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--val-frac", type=float, default=0.2)
    ap.add_argument("--max-parts", type=int, default=None)
    ap.add_argument("--dist-thresh-mm", type=float, default=5.0)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--op-cache-dir", default=None,
                    help="cache dir for DiffusionNet mesh operators "
                         "(strongly recommended for the 429-file corpus)")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    tr, va, tr_reps, va_reps = train_and_eval(
        args.source, backbone=args.backbone, epochs=args.epochs,
        val_frac=args.val_frac, max_parts=args.max_parts,
        device=args.device, dist_thresh_mm=args.dist_thresh_mm, seed=args.seed,
        op_cache_dir=args.op_cache_dir)
    print("\n=== TRAIN (aggregate) ===");  print(json.dumps(tr, indent=2))
    print("\n=== VAL (aggregate) ===");    print(json.dumps(va, indent=2))


if __name__ == "__main__":
    main()
