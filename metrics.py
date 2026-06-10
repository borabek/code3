# Evaluation metrics for segmentation.
#
# The models are evaluated not just on accuracy (which is misleading with
# heavily imbalanced labels) but also on Dice coefficient and Jaccard index
# (IoU) -- per class, averaged, and additionally weighted. This module
# computes all of these from predictions and labels (per vertex, edge, or
# face -- they are all just integer class ids).
#
# Reference: Scheffler (2022), §6.2.1
#   §6.2.1 – class-weighted IoU/Dice evaluation (compensates housing ~70% imbalance)

import numpy as np
from connector_constants import CLASS_NAMES


def confusion_matrix(pred, true, n_classes):
    """Confusion matrix (n_classes x n_classes), row = true, column = predicted."""
    pred = np.asarray(pred, dtype=int).ravel()
    true = np.asarray(true, dtype=int).ravel()
    m = np.zeros((n_classes, n_classes), dtype=np.int64)
    np.add.at(m, (true, pred), 1)
    return m


def _tp_fp_fn(cm):
    tp = np.diag(cm).astype(float)
    fp = cm.sum(axis=0).astype(float) - tp   # column sum minus diagonal
    fn = cm.sum(axis=1).astype(float) - tp   # row sum minus diagonal
    return tp, fp, fn


def per_class_dice(pred, true, n_classes):
    """Dice per class = 2TP / (2TP + FP + FN). Empty classes -> nan."""
    tp, fp, fn = _tp_fp_fn(confusion_matrix(pred, true, n_classes))
    denom = 2 * tp + fp + fn
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(denom > 0, 2 * tp / denom, np.nan)


def per_class_iou(pred, true, n_classes):
    """IoU / Jaccard per class = TP / (TP + FP + FN). Empty classes -> nan."""
    tp, fp, fn = _tp_fp_fn(confusion_matrix(pred, true, n_classes))
    denom = tp + fp + fn
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(denom > 0, tp / denom, np.nan)


def _dice_iou_from_cm(cm):
    """Compute per-class dice and IoU from a pre-built confusion matrix."""
    tp, fp, fn = _tp_fp_fn(cm)
    with np.errstate(invalid="ignore", divide="ignore"):
        denom_d = 2 * tp + fp + fn
        denom_j = tp + fp + fn
        dice = np.where(denom_d > 0, 2 * tp / denom_d, np.nan)
        iou  = np.where(denom_j > 0, tp / denom_j, np.nan)
    return dice, iou


def accuracy(pred, true):
    pred = np.asarray(pred, dtype=int).ravel()
    true = np.asarray(true, dtype=int).ravel()
    return float((pred == true).mean())


def mean_iou(pred, true, n_classes):
    return float(np.nanmean(per_class_iou(pred, true, n_classes)))


def mean_dice(pred, true, n_classes):
    return float(np.nanmean(per_class_dice(pred, true, n_classes)))


def weighted_iou(pred, true, n_classes, _iou=None, _true=None):
    """IoU weighted by class frequency (support in labels).

    Compensates for the label imbalance (housing ~70%) -- this is the
    weighted Jaccard index. Accepts pre-computed iou/_true to avoid
    rebuilding the confusion matrix inside segmentation_report.
    """
    if _iou is None:
        _true = np.asarray(true, dtype=int).ravel()
        _iou = per_class_iou(pred, true, n_classes)
    support = np.array([(_true == c).sum() for c in range(n_classes)], dtype=float)
    valid = ~np.isnan(_iou) & (support > 0)
    if support[valid].sum() == 0:
        return float("nan")
    return float((_iou[valid] * support[valid]).sum() / support[valid].sum())


def segmentation_report(pred, true, n_classes=5, class_names=None):
    """All metrics at once as a dict.

    Contains accuracy, per-class/mean Dice + IoU, weighted IoU, and the
    confusion matrix -- ready for logging or a ray-tune report callback.
    Builds the confusion matrix exactly once.
    """
    names = class_names or (CLASS_NAMES if n_classes == len(CLASS_NAMES)
                            else [f"class_{i}" for i in range(n_classes)])
    cm = confusion_matrix(pred, true, n_classes)
    dice, iou = _dice_iou_from_cm(cm)
    true_arr = np.asarray(true, dtype=int).ravel()
    return {
        "accuracy":     accuracy(pred, true),
        "mean_dice":    float(np.nanmean(dice)),
        "mean_iou":     float(np.nanmean(iou)),
        "weighted_iou": weighted_iou(None, None, n_classes, _iou=iou, _true=true_arr),
        "dice_per_class": {names[i]: (None if np.isnan(dice[i]) else float(dice[i]))
                           for i in range(n_classes)},
        "iou_per_class": {names[i]: (None if np.isnan(iou[i]) else float(iou[i]))
                          for i in range(n_classes)},
        "confusion": cm.tolist(),
    }


def print_report(report):
    print("Segmentation metrics")
    print(f"  accuracy      : {report['accuracy']:.4f}")
    print(f"  mean Dice     : {report['mean_dice']:.4f}")
    print(f"  mean IoU      : {report['mean_iou']:.4f}")
    print(f"  weighted IoU  : {report['weighted_iou']:.4f}")
    print("  per class (Dice / IoU):")
    for name in report["dice_per_class"]:
        d = report["dice_per_class"][name]
        j = report["iou_per_class"][name]
        ds = "  -  " if d is None else f"{d:.4f}"
        js = "  -  " if j is None else f"{j:.4f}"
        print(f"     {name:22s} {ds} / {js}")


def _demo():
    # synthetic example with 5 classes and slight label imbalance
    rng = np.random.default_rng(0)
    true = np.concatenate([np.zeros(700), np.full(140, 1), np.full(60, 2),
                           np.full(60, 3), np.full(40, 4)]).astype(int)
    pred = true.copy()
    flip = rng.random(len(true)) < 0.15        # introduce 15% errors
    pred[flip] = rng.integers(0, 5, size=flip.sum())

    rep = segmentation_report(pred, true, n_classes=5)
    print_report(rep)
    assert 0.0 <= rep["mean_iou"] <= 1.0
    assert 0.0 <= rep["weighted_iou"] <= 1.0
    # perfect prediction -> all metrics 1.0
    perfect = segmentation_report(true, true, n_classes=5)
    assert abs(perfect["mean_iou"] - 1.0) < 1e-9 and abs(perfect["accuracy"] - 1.0) < 1e-9
    print("\n[ok] metrics self-test passed (perfect prediction -> IoU/Dice/acc = 1.0).")


if __name__ == "__main__":
    _demo()
