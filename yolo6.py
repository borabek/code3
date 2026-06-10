# YOLOv6 integration for the 2D->3D branch of the pipeline.
#
# The thesis uses the official meituan/YOLOv6 repo, trained on the 6 grayscale
# views (512x512) with bounding-box labels in COCO/YOLO format. The actual
# training and inference of the network runs through that external repo with
# trained weights -- it is not reimplemented here.
#
# What this module provides is the glue around YOLOv6, connecting all the
# already-tested building blocks into a complete chain:
#   visualization.render_all_views  -> the 6 input images
#   (external YOLOv6)               -> bounding boxes per view
#   projection_2d3d.load_detections / detections_to_crops -> 3D crops
# Including the thesis constraint of only using YOLO for snap points.
#
# Reference: Scheffler (2022), §5.3.4, §6.2.3
#   §5.3.4 / Abb.41,42 – YOLOv6 with RepVGG backbone, 6 grayscale 512x512 views,
#                         VariFocal + IoU loss
#   §6.2.3              – hybrid inference: YOLOv6 for SnapPoint only

import os
import logging

import numpy as np

from visualization import render_all_views, save_views
from projection_2d3d import (
    world_aabb, load_detections, detections_to_crops, crop_labels,
    YOLO_PIPELINE_CLASSES,
)
from coco_labels import YOLO2D_TO_CONNECTOR

logger = logging.getLogger(__name__)

# Detection classes in the 3D/connector convention (housing is not detected).
# YOLO/COCO files use the 2D convention (Code 4) and are translated via
# YOLO2D_TO_CONNECTOR when loaded.
DETECTION_CLASSES = {1: "Contact", 2: "SnapPoint", 3: "CableEntry", 4: "LabelSurface"}


# §5.3.4 / Abb.41 – render 6 grayscale 512x512 views as YOLOv6 input images
def render_for_yolo(vertices, faces, out_dir=None, res=512, pad_frac=0.05):
    """Render the 6 input images for YOLOv6 (proper triangle rasterization).

    Returns: (views_dict, bounds). bounds must be passed to detections_to_crops
    later so the box->3D projection matches the render convention. If out_dir
    is set, images are saved as PNG/PGM (top/bottom/front/...).
    """
    bounds = world_aabb(vertices, pad_frac=pad_frac)
    views = render_all_views(vertices, faces, bounds=bounds, res=res)
    if out_dir:
        paths = save_views(out_dir, views, res=res)
        logger.info("render_for_yolo: %d views saved to %s", len(paths), out_dir)
    return views, bounds


# §6.2.3 – hybrid inference: process only SnapPoint from YOLO, other classes from DiffusionNet
def yolo_to_crops(vertices, faces, detections, bounds, res=512,
                  conf_thresh=0.25, snap_point_only=True, face_mode="any",
                  front_band_frac=None, min_depth_extent=0.0):
    """Convert YOLOv6 detections to 3D crops.

    snap_point_only=True reflects the final pipeline from the thesis: only
    snap-point detections are processed (other classes come from DiffusionNet).
    front_band_frac optionally limits the slab to the front depth layer --
    leave as None for snap points (which run the full depth).
    Returns: list of Crop.
    """
    only = YOLO_PIPELINE_CLASSES if snap_point_only else None
    return detections_to_crops(
        vertices, faces, detections, bounds=bounds, res=res,
        conf_thresh=conf_thresh, face_mode=face_mode, only_classes=only,
        front_band_frac=front_band_frac, min_depth_extent=min_depth_extent)


def load_yolo_predictions(path, res=512, class_map=YOLO2D_TO_CONNECTOR):
    """Load YOLOv6 output (COCO JSON or YOLO txt) as a list of Detection objects.

    Expected format: see projection_2d3d.load_detections -- each detection has
    view, bbox/bbox_norm, conf, cls.

    File ids use the 2D convention (Code 4); by default they are translated via
    YOLO2D_TO_CONNECTOR into the 3D/connector convention so the snap-point
    filter (3D id 2) works correctly. Pass class_map=None to skip translation
    (e.g. if the file already uses connector ids).
    """
    return load_detections(path, res=res, class_map=class_map)


def run_external_yolo(image_dir, weights, out_json=None):
    """Placeholder for real YOLOv6 inference via the external repo.

    Deliberately NOT reimplemented -- the thesis uses meituan/YOLOv6 with
    trained weights. Typical call from that repo:
        python tools/infer.py --weights best.pt --source <image_dir> --save-txt
    Then load the output YOLO txt/JSON with load_yolo_predictions().
    """
    raise NotImplementedError(
        "YOLOv6 inference runs via the external repo meituan/YOLOv6 with "
        "trained weights:\n"
        "  python tools/infer.py --weights <best.pt> --source <image_dir> --save-txt\n"
        "Load the output with load_yolo_predictions() and pass to yolo_to_crops()."
    )


def _demo():
    # Fully testable chain WITHOUT real YOLO: render a part, fake a snap-point
    # detection (projected), and crop back to 3D.
    from projection_2d3d import project_points, Detection
    from connector3d import SNAP_POINT as AP, CONTACT

    # flat plate with two snap-point strips at the bottom + one contact at the top
    res = 60
    xs = np.linspace(0, 1, res)
    V = np.array([(x, y, 0.0) for y in xs for x in xs], dtype=float)
    F = []
    for iy in range(res - 1):
        for ix in range(res - 1):
            a = iy * res + ix
            b, c, d = a + 1, a + res, a + res + 1
            F.append((a, b, d)); F.append((a, d, c))
    F = np.array(F, dtype=int)
    x, y = V[:, 0], V[:, 1]
    L = np.zeros(len(V), dtype=int)
    L[(y < 0.12) & (x > 0.15) & (x < 0.4)] = AP
    L[(y < 0.12) & (x > 0.6) & (x < 0.85)] = AP
    L[(y > 0.85) & (x > 0.4) & (x < 0.6)] = CONTACT

    views, bounds = render_for_yolo(V, F, res=256)
    covered = int((views["z+"] != 255).sum())
    print(f"[render] z+ view: {covered} part pixels (proper rasterization)")

    # fake YOLO: pixel bbox per feature in the top view (z+)
    dets = []
    for cls in (AP, CONTACT):
        for half in ([0.15, 0.4], [0.6, 0.85]) if cls == AP else ([0.4, 0.6],):
            sel = (L == cls) & (x >= half[0]) & (x <= half[1])
            if not sel.any():
                continue
            px = project_points("z+", V[sel], bounds, res=256)
            x0, y0 = px.min(axis=0); x1, y1 = px.max(axis=0)
            dets.append(Detection("z+", (x0, y0, x1, y1), conf=0.9, cls=int(cls)))
    print(f"[yolo]   {len(dets)} detections faked ({sum(d.cls==AP for d in dets)} snap-point, "
          f"{sum(d.cls==CONTACT for d in dets)} contact)")

    crops = yolo_to_crops(V, F, dets, bounds, res=256, conf_thresh=0.25, snap_point_only=True)
    print(f"[crops]  {len(crops)} crops (only snap-point expected -> 2)")
    for cr in crops:
        print(f"         snap-point crop: center={np.round(cr.cog['center'],2)}  "
              f"({len(cr.vertices)} verts)")
    assert all(cr.detection.cls == AP for cr in crops), "only snap-point should pass through"
    assert len(crops) == 2

    # --- regression: 2D->3D class id mapping (Code 4 -> connector) ---
    # A real YOLO file carries 2D ids (Code 4: snap-point=3, label-surface=2).
    # Without the remap, the snap-point filter (3D id 2) would reject the actual
    # snap points (file id 3) and wrongly let the label-surface (file id 2) through.
    import json
    import tempfile

    def _bbox_2d(sel):
        px = project_points("z+", V[sel], bounds, res=256)
        x0, y0 = px.min(axis=0); x1, y1 = px.max(axis=0)
        return [float(x0), float(y0), float(x1), float(y1)]

    file_dets = [
        {"view": "z+", "bbox": _bbox_2d((y < 0.12) & (x > 0.15) & (x < 0.4)), "conf": 0.9, "cls": 3},
        {"view": "z+", "bbox": _bbox_2d((y < 0.12) & (x > 0.6) & (x < 0.85)), "conf": 0.9, "cls": 3},
        {"view": "z+", "bbox": _bbox_2d((x > 0.4) & (x < 0.6) & (y > 0.4) & (y < 0.6)), "conf": 0.9, "cls": 2},
    ]
    tmp = tempfile.NamedTemporaryFile(suffix=".json", delete=False, mode="w")
    json.dump(file_dets, tmp); tmp.close()

    def ap_recall(c):
        return int((crop_labels(L, c.vertex_index) == AP).sum())

    fixed = load_yolo_predictions(tmp.name, res=256)                       # remap on (default)
    buggy = load_yolo_predictions(tmp.name, res=256, class_map=None)       # remap off (broken)
    os.remove(tmp.name)
    print(f"[id-map] file ids (Code 4)=[3,3,2] -> connector={[d.cls for d in fixed]} "
          f"(snap-point 3->2, label-surface 2->4)")

    crops_fixed = yolo_to_crops(V, F, fixed, bounds, res=256, snap_point_only=True)
    crops_buggy = yolo_to_crops(V, F, buggy, bounds, res=256, snap_point_only=True)
    print(f"[id-map] with remap: {len(crops_fixed)} real snap-point crops; "
          f"without remap: {len(crops_buggy)} crop(s), real snap-points "
          f"{sum(ap_recall(c) > 0 for c in crops_buggy)} -> bug without fix confirmed")
    assert len(crops_fixed) == 2 and all(ap_recall(c) > 0 for c in crops_fixed), \
        "with remap, exactly 2 real snap-point crops must be produced"
    assert not (len(crops_buggy) == 2 and all(ap_recall(c) > 0 for c in crops_buggy)), \
        "without remap, 2 real snap-point crops must NOT be produced (the bug)"

    print("\n[ok] YOLOv6 bridge: 6-view render -> detection -> 3D crop (snap-point only),")
    print("     incl. 2D->3D class id mapping (Code 4 -> connector) verified.")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
    _demo()
