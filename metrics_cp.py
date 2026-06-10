# Keypoint metrics for connection-point detection (Option B regression).
#
# Segmentation metrics (metrics.py) measure per-vertex IoU/Dice and do not
# apply to discrete keypoints. This module evaluates the decoded connection
# points against ground-truth terminal blocks:
#
#   * localization error (mm)   – distance from a matched prediction to its GT
#   * direction angular error   – angle (deg) between predicted and GT directions
#   * precision / recall / F1    – at a distance threshold, via greedy one-to-one
#                                  matching (a prediction matches the nearest
#                                  unused GT within `dist_thresh_mm`)
#
# Pure numpy; no torch dependency so it is testable without a trained model.

import numpy as np


def match_predictions(pred_points, gt_points, dist_thresh_mm):
    """Greedy one-to-one matching: nearest unused GT within threshold.

    Returns (matches, unmatched_pred, unmatched_gt) where matches is a list of
    (pred_idx, gt_idx, dist). Predictions are consumed in input order (caller
    should pass them sorted by descending score for standard detection AP-style
    matching).
    """
    pred_points = np.asarray(pred_points, dtype=float).reshape(-1, 3)
    gt_points = np.asarray(gt_points, dtype=float).reshape(-1, 3)
    used_gt = np.zeros(len(gt_points), dtype=bool)
    matches = []
    unmatched_pred = []
    for pi in range(len(pred_points)):
        if len(gt_points) == 0:
            unmatched_pred.append(pi)
            continue
        d = np.linalg.norm(gt_points - pred_points[pi], axis=1)
        d[used_gt] = np.inf
        gj = int(np.argmin(d))
        if d[gj] <= dist_thresh_mm:
            used_gt[gj] = True
            matches.append((pi, gj, float(d[gj])))
        else:
            unmatched_pred.append(pi)
    unmatched_gt = [j for j in range(len(gt_points)) if not used_gt[j]]
    return matches, unmatched_pred, unmatched_gt


def _angle_deg(a, b):
    a = np.asarray(a, float); b = np.asarray(b, float)
    na = np.linalg.norm(a); nb = np.linalg.norm(b)
    if na == 0 or nb == 0:
        return float("nan")
    c = np.clip(np.dot(a, b) / (na * nb), -1.0, 1.0)
    return float(np.degrees(np.arccos(c)))


def keypoint_report(preds, gt_points, gt_directions, dist_thresh_mm=5.0):
    """Evaluate decoded connection points against ground-truth blocks.

    preds : list of dicts {'point','direction','score'} (decode_predictions).
    gt_points / gt_directions : (G,3) arrays.
    Returns a dict with precision/recall/F1, mean & max localization error (mm,
    over matched), and mean & max direction angular error (deg, over matched).
    """
    preds = sorted(preds, key=lambda p: -p.get("score", 0.0))
    pred_pts = np.array([p["point"] for p in preds], dtype=float).reshape(-1, 3)
    matches, unmatched_pred, unmatched_gt = match_predictions(
        pred_pts, gt_points, dist_thresh_mm)

    tp = len(matches)
    fp = len(unmatched_pred)
    fn = len(unmatched_gt)
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * precision * recall / (precision + recall)
          if (precision + recall) else 0.0)

    loc_errs = [d for _, _, d in matches]
    ang_errs = [_angle_deg(preds[pi]["direction"], gt_directions[gj])
                for pi, gj, _ in matches]

    def _stat(xs, fn_):
        return float(fn_(xs)) if xs else float("nan")

    return {
        "n_pred": len(preds), "n_gt": len(gt_points),
        "tp": tp, "fp": fp, "fn": fn,
        "precision": precision, "recall": recall, "f1": f1,
        "mean_loc_err_mm": _stat(loc_errs, np.mean),
        "max_loc_err_mm": _stat(loc_errs, np.max),
        "mean_ang_err_deg": _stat(ang_errs, np.mean),
        "max_ang_err_deg": _stat(ang_errs, np.max),
        "dist_thresh_mm": dist_thresh_mm,
    }


def print_keypoint_report(rep):
    print("Connection-point metrics (thresh=%.1f mm)" % rep["dist_thresh_mm"])
    print("  pred=%d gt=%d  TP=%d FP=%d FN=%d"
          % (rep["n_pred"], rep["n_gt"], rep["tp"], rep["fp"], rep["fn"]))
    print("  precision=%.3f recall=%.3f F1=%.3f"
          % (rep["precision"], rep["recall"], rep["f1"]))
    print("  loc err (mm): mean=%.3f max=%.3f"
          % (rep["mean_loc_err_mm"], rep["max_loc_err_mm"]))
    print("  dir err (deg): mean=%.3f max=%.3f"
          % (rep["mean_ang_err_deg"], rep["max_ang_err_deg"]))


def _selftest():
    gt_pts = np.array([[0, 0, 0], [10, 0, 0], [0, 10, 0]], float)
    gt_dir = np.array([[0, 0, 1.0]] * 3)
    # perfect predictions + one false positive
    preds = [
        {"point": np.array([0.1, 0, 0]), "direction": np.array([0, 0, 1.0]), "score": 0.9},
        {"point": np.array([10, 0.2, 0]), "direction": np.array([0, 0.1, 1.0]), "score": 0.8},
        {"point": np.array([0, 10, 0]), "direction": np.array([0, 0, 1.0]), "score": 0.7},
        {"point": np.array([99, 99, 99]), "direction": np.array([1, 0, 0.0]), "score": 0.2},
    ]
    rep = keypoint_report(preds, gt_pts, gt_dir, dist_thresh_mm=2.0)
    assert rep["tp"] == 3 and rep["fp"] == 1 and rep["fn"] == 0, rep
    assert rep["recall"] == 1.0 and abs(rep["precision"] - 0.75) < 1e-9, rep
    assert rep["max_loc_err_mm"] < 0.3, rep
    print_keypoint_report(rep)
    print("metrics_cp selftest OK")


if __name__ == "__main__":
    _selftest()
