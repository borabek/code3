# 2D bounding-box labels -> COCO format.
#
# The grayscale images (6 views per part) are labeled in Label Studio with
# bounding boxes and exported as YOLO pseudo-labels (Code 4):
# one line per box "cls xc yc w h", all values normalized to [0,1].
# This module converts that format to COCO JSON for object detection training.
#
# COCO bbox convention: [x_min, y_min, width, height] in absolute pixels.
# YOLO uses normalized center + size. The conversion happens here.
#
# Class ids follow Code 4:
#   0 = CableEntry, 1 = Contact, 2 = LabelSurface, 3 = SnapPoint
# (Housing is not detected in 2D.) Override via `categories`.
#
# Reference: Scheffler (2022), §5.2.2
#   §5.2.2 / Code 4 – YOLO/COCO bounding-box labels, 2D convention:
#                     CableEntry=0, Contact=1, LabelSurface=2, SnapPoint=3

import os
import json
import glob
import logging

logger = logging.getLogger(__name__)

# §5.2.2 / Code 4 – YOLO/COCO bounding-box class ids (2D convention)
# id -> name, exactly as in Code 4
DEFAULT_CATEGORIES = {
    0: "CableEntry",
    1: "Contact",
    2: "LabelSurface",
    3: "SnapPoint",
}

# Robotic-vocabulary display names for the same 2D ids (for robot-facing tooling
# and reports). IDs are unchanged; only the human-readable name differs.
ROBOTIC_CATEGORIES = {
    0: "CableEntry",
    1: "TerminalContact",
    2: "IdentificationSurface",
    3: "RailMount",
}

# ---------------------------------------------------------------------------
# class id bridge between the TWO conventions used in the thesis
# ---------------------------------------------------------------------------
# The thesis intentionally uses two different id schemes:
#   2D (YOLO/COCO, Code 4):   0=CableEntry  1=Contact
#                             2=LabelSurface 3=SnapPoint
#   3D (connector3d/labels):  0=Housing 1=Contact 2=SnapPoint
#                             3=CableEntry  4=LabelSurface
# When loading YOLO detections we MUST translate to the 3D convention,
# otherwise the snap-point filter (3D id 2) would catch the wrong class
# (2D id 2 = LabelSurface). This table is the single source of that mapping.
YOLO2D_TO_CONNECTOR = {0: 3, 1: 1, 2: 4, 3: 2}
CONNECTOR_TO_YOLO2D = {v: k for k, v in YOLO2D_TO_CONNECTOR.items()}


def remap_class(cls, class_map):
    """Translate a class id via class_map; unknown ids pass through unchanged."""
    if class_map is None:
        return cls
    return class_map.get(cls, cls)


# ---------------------------------------------------------------------------
# YOLO (normalized) -> COCO bbox (absolute pixels)
# ---------------------------------------------------------------------------

# §5.2.2 / Code 4 – normalized YOLO box -> absolute-pixel COCO bbox conversion
def yolo_to_coco_bbox(xc, yc, w, h, img_w, img_h):
    """Convert a normalized YOLO box (center+size) to a COCO pixel box.

    Returns: [x_min, y_min, width, height] in pixels (floats, clipped to image).
    """
    bw = w * img_w
    bh = h * img_h
    x_min = (xc * img_w) - bw / 2.0
    y_min = (yc * img_h) - bh / 2.0
    # clamp to image bounds (loose boxes at the edge)
    x_min = max(0.0, min(x_min, img_w))
    y_min = max(0.0, min(y_min, img_h))
    bw = max(0.0, min(bw, img_w - x_min))
    bh = max(0.0, min(bh, img_h - y_min))
    return [round(x_min, 3), round(y_min, 3), round(bw, 3), round(bh, 3)]


# ---------------------------------------------------------------------------
# read YOLO pseudo-label lines (Code 4)
# ---------------------------------------------------------------------------

def parse_yolo_lines(text):
    """Parse YOLO text lines "cls xc yc w h [conf]" -> list of (cls, xc, yc, w, h)."""
    out = []
    for ln in text.splitlines():
        s = ln.strip()
        if not s or s.startswith("#"):
            continue
        p = s.split()
        if len(p) < 5:
            logger.warning("line skipped (too few values): %r", s)
            continue
        cls = int(float(p[0]))
        xc, yc, w, h = (float(p[1]), float(p[2]), float(p[3]), float(p[4]))
        out.append((cls, xc, yc, w, h))
    return out


def load_yolo_file(path):
    """Load a YOLO pseudo-label file (Code 4)."""
    with open(path, encoding="utf-8") as fh:
        return parse_yolo_lines(fh.read())


# ---------------------------------------------------------------------------
# build COCO document
# ---------------------------------------------------------------------------

def build_coco(images, categories=None, img_size=512):
    """Build a COCO JSON dict from multiple labeled images.

    images: list of dicts per image:
        {"file_name": "part_top.png",
         "boxes": [(cls, xc, yc, w, h), ...],     # normalized YOLO boxes
         "width": 512, "height": 512}             # optional, defaults to img_size
    categories: dict {id -> name} (default Code 4). Only referenced classes
                appear in the COCO categories list.
    Returns: COCO dict with images / annotations / categories.
    """
    categories = categories or DEFAULT_CATEGORIES
    coco_images = []
    coco_annotations = []
    used_cats = set()
    ann_id = 1

    for img_id, img in enumerate(images, start=1):
        W = int(img.get("width", img_size))
        H = int(img.get("height", img_size))
        coco_images.append({"id": img_id, "file_name": img["file_name"],
                            "width": W, "height": H})
        for (cls, xc, yc, w, h) in img.get("boxes", []):
            bbox = yolo_to_coco_bbox(xc, yc, w, h, W, H)
            used_cats.add(cls)
            coco_annotations.append({
                "id": ann_id,
                "image_id": img_id,
                "category_id": cls,
                "bbox": bbox,
                "area": round(bbox[2] * bbox[3], 3),
                "iscrowd": 0,
            })
            ann_id += 1

    coco_categories = [{"id": cid, "name": categories.get(cid, f"class_{cid}")}
                       for cid in sorted(used_cats)]
    return {"images": coco_images, "annotations": coco_annotations,
            "categories": coco_categories}


# ---------------------------------------------------------------------------
# convert an entire directory (one .txt per image)
# ---------------------------------------------------------------------------

def yolo_dir_to_coco(label_dir, image_ext=".png", img_size=512, categories=None,
                     image_dir=None):
    """Combine all YOLO .txt files in a directory into one COCO document.

    One image entry is created per .txt; the image filename uses the same stem
    with `image_ext`. If image_dir is set and PIL is available, the actual image
    size is read; otherwise img_size is assumed.
    """
    images = []
    for txt in sorted(glob.glob(os.path.join(label_dir, "*.txt"))):
        stem = os.path.splitext(os.path.basename(txt))[0]
        boxes = load_yolo_file(txt)
        W = H = img_size
        if image_dir:
            img_path = os.path.join(image_dir, stem + image_ext)
            wh = _image_size(img_path)
            if wh:
                W, H = wh
        images.append({"file_name": stem + image_ext, "boxes": boxes,
                       "width": W, "height": H})
    coco = build_coco(images, categories=categories, img_size=img_size)
    logger.info("yolo_dir_to_coco: %d images, %d annotations, %d categories",
                len(coco["images"]), len(coco["annotations"]), len(coco["categories"]))
    return coco


def _image_size(path):
    """Read image size (w, h) via PIL if available, else return None."""
    try:
        from PIL import Image
    except ImportError:
        return None
    try:
        with Image.open(path) as im:
            return im.size
    except Exception:
        return None


def save_coco(coco, path):
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(coco, fh, indent=2, ensure_ascii=False)
    return path


# ---------------------------------------------------------------------------
# self-test
# ---------------------------------------------------------------------------

def _demo():
    # Code 4 example (1 Contact, 1 SnapPoint, 1 CableEntry, 1 LabelSurface)
    # as YOLO pseudo-label text.
    text = (
        "0 0.385742187 0.639648437 0.041207188 0.045113438\n"
        "1 0.386199209 0.221591701 0.053012202 0.052183886\n"
        "2 0.544821658 0.768485772 0.050527255 0.0505272553\n"
        "3 0.340820312 0.728515625 0.025390625 0.023437541\n"
    )
    boxes = parse_yolo_lines(text)
    print(f"[parse]  {len(boxes)} boxes from Code-4 pseudo-labels")
    assert len(boxes) == 4

    coco = build_coco([{"file_name": "1SAE231111M0622_top.png", "boxes": boxes,
                        "width": 512, "height": 512}])
    print(f"[coco]   {len(coco['images'])} image, {len(coco['annotations'])} annotations, "
          f"categories={[c['name'] for c in coco['categories']]}")
    assert len(coco["annotations"]) == 4
    assert len(coco["categories"]) == 4

    # check bbox math against a hand calculation (CableEntry, cls 0)
    bbox = coco["annotations"][0]["bbox"]
    exp_w = 0.041207188 * 512
    exp_x = 0.385742187 * 512 - exp_w / 2.0
    assert abs(bbox[0] - round(exp_x, 3)) < 1e-2, (bbox[0], exp_x)
    assert abs(bbox[2] - round(exp_w, 3)) < 1e-2, (bbox[2], exp_w)
    # all boxes fit inside the image
    for ann in coco["annotations"]:
        x, y, w, h = ann["bbox"]
        assert 0 <= x <= 512 and 0 <= y <= 512 and x + w <= 512 + 1e-6 and y + h <= 512 + 1e-6

    print("[bbox]   YOLO (normalized) -> COCO (pixel) conversion verified")
    print("\n[ok] COCO generation from Label Studio / YOLO pseudo-labels passed.")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
    _demo()
