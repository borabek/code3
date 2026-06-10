# 2D detections -> 3D crops ("the workaround").
#
# Background: the end-to-end network did not reliably find features in 3D meshes.
# The workaround is:
#
#   1. Render the 3D part from 6 directions orthographically -> 6 images (512x512).
#   2. YOLOv6 draws bounding boxes around features in each 2D image.
#      Detections below a confidence threshold are discarded.
#   3. Each 2D box lies flat on one side of the part -- it has no depth.
#      We "pull" the box all the way through the model along the view direction
#      and get an axis-aligned 3D box (a slab).
#   4. The part of the mesh inside this 3D box is cropped and analyzed further
#      (centroid / vectors -> see feature_geometry.py).
#
# Important convention: forward projection (render) and inverse (box -> slab)
# must use the same camera convention. Anyone generating the 6 images for YOLOv6
# should use project_points() / render_view() from this file so that axes and
# mirroring stay consistent.
#
# Reference: Scheffler (2022), §5.3.7, §2.4
#   §5.3.7 / Abb.46 – 2D bbox -> 3D slab crop: stretch bbox through full depth axis
#   §2.4 / Abb.28   – 6 orthographic views: top/bottom/front/rear/left/right =
#                      z+/z-/y+/y-/x+/x-

import os
import json
import logging
from dataclasses import dataclass, field

import numpy as np

from feature_geometry import convex_hull_volume_centroid

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# camera convention for the 6 orthographic views
# ---------------------------------------------------------------------------
# Each view looks along one axis. Two axes lie in the image plane
# (u = image column, v = image row); the third is depth.
# Row 0 is at the top, so v is flipped during rendering.

@dataclass(frozen=True)
class ViewSpec:
    name: str
    depth_axis: int          # axis along which the model is "pulled through"
    u_axis: int              # world axis -> image column
    v_axis: int              # world axis -> image row
    u_sign: int = 1
    v_sign: int = 1


# §2.4 / Abb.28 – 6 orthographic views: z+=top, z-=bottom, y+=front, y-=rear, x+=right, x-=left
# x=0, y=1, z=2
VIEWS = {
    "z+": ViewSpec("z+", depth_axis=2, u_axis=0, v_axis=1, u_sign=+1, v_sign=+1),  # top
    "z-": ViewSpec("z-", depth_axis=2, u_axis=0, v_axis=1, u_sign=+1, v_sign=-1),  # bottom
    "y+": ViewSpec("y+", depth_axis=1, u_axis=0, v_axis=2, u_sign=+1, v_sign=+1),  # front
    "y-": ViewSpec("y-", depth_axis=1, u_axis=0, v_axis=2, u_sign=-1, v_sign=+1),  # rear
    "x+": ViewSpec("x+", depth_axis=0, u_axis=1, v_axis=2, u_sign=-1, v_sign=+1),  # right
    "x-": ViewSpec("x-", depth_axis=0, u_axis=1, v_axis=2, u_sign=+1, v_sign=+1),  # left
}

VIEW_ORDER = ["x+", "x-", "y+", "y-", "z+", "z-"]

# In the final pipeline, the YOLO->3D crop is used ONLY for snap points:
# contacts and cable entries can be occluded in the 6 views, but snap points
# are always visible from at least two views and span the full width of the part.
# The other classes come from DiffusionNet.
SNAP_POINT = 2
YOLO_PIPELINE_CLASSES = (SNAP_POINT,)


# ---------------------------------------------------------------------------
# part bounding box (shared reference frame for render + back-projection)
# ---------------------------------------------------------------------------

def world_aabb(vertices, pad_frac=0.0):
    """Axis-aligned bounding box (lo, hi) of the vertices.

    pad_frac expands the box symmetrically by this fraction of the diagonal,
    so nothing ends up exactly at the image border when rendering.
    """
    V = np.asarray(vertices, dtype=float)
    lo = V.min(axis=0)
    hi = V.max(axis=0)
    if pad_frac:
        pad = pad_frac * float(np.linalg.norm(hi - lo))
        lo = lo - pad
        hi = hi + pad
    return lo, hi


def _extent(lo, hi):
    ext = np.asarray(hi, dtype=float) - np.asarray(lo, dtype=float)
    ext[ext == 0] = 1.0  # avoid division by zero on degenerate axis
    return ext


# ---------------------------------------------------------------------------
# forward: 3D vertices -> pixels (for generating training images / testing)
# ---------------------------------------------------------------------------

def project_points(view, points, bounds, res=512):
    """Project 3D points to pixel coordinates in the given view.

    Returns: array (N, 2) with (px, py), px = column, py = row, both in [0, res-1].
    """
    if isinstance(view, str):
        view = VIEWS[view]
    P = np.asarray(points, dtype=float).reshape(-1, 3)
    lo, hi = bounds
    lo = np.asarray(lo, dtype=float)
    ext = _extent(lo, hi)

    t_u = (P[:, view.u_axis] - lo[view.u_axis]) / ext[view.u_axis]
    t_v = (P[:, view.v_axis] - lo[view.v_axis]) / ext[view.v_axis]

    u_norm = t_u if view.u_sign > 0 else 1.0 - t_u
    v_norm = t_v if view.v_sign > 0 else 1.0 - t_v

    px = u_norm * (res - 1)
    py = (1.0 - v_norm) * (res - 1)  # row 0 is at the top
    return np.stack([px, py], axis=1)


def render_view(view, vertices, bounds, res=512):
    """Lightweight orthographic depth/silhouette renderer (vertex splat).

    NOT a full triangle rasterizer -- it splatts vertices onto the grid and
    keeps the nearest depth. Good enough for a quick preview and for showing
    the 6-view setup. The thesis used a proper rasterizer for YOLOv6 training images.

    Returns: (depth, mask) each (res, res). depth is normalized to [0,1]
    (0 = close to camera); mask is bool where a vertex was hit.
    """
    if isinstance(view, str):
        view = VIEWS[view]
    lo, hi = bounds
    lo = np.asarray(lo, dtype=float)
    ext = _extent(lo, hi)

    px = project_points(view, vertices, bounds, res)
    cols = np.clip(np.round(px[:, 0]).astype(int), 0, res - 1)
    rows = np.clip(np.round(px[:, 1]).astype(int), 0, res - 1)

    P = np.asarray(vertices, dtype=float)
    t_d = (P[:, view.depth_axis] - lo[view.depth_axis]) / ext[view.depth_axis]

    depth = np.full((res, res), np.inf, dtype=float)
    for r, c, d in zip(rows, cols, t_d):
        if d < depth[r, c]:
            depth[r, c] = d
    mask = np.isfinite(depth)
    depth[~mask] = 1.0
    return depth, mask


# ---------------------------------------------------------------------------
# YOLOv6 detections
# ---------------------------------------------------------------------------

@dataclass
class Detection:
    """A 2D bounding box from YOLOv6 in a specific view.

    bbox is in pixels (x0, y0, x1, y1), x = column, y = row, y0 = top.
    cls is the class id in the 3D/connector convention (snap-point=2).
    YOLO/COCO files use the 2D convention (Code 4, snap-point=3); loaders
    translate via `class_map` so YOLO_PIPELINE_CLASSES and the snap-point
    filter work correctly.
    """
    view: str
    bbox: tuple           # (x0, y0, x1, y1) in pixels
    conf: float = 1.0
    cls: int = -1

    def as_xyxy(self):
        return tuple(float(v) for v in self.bbox)


# Robotic-vocabulary alias for a 2D image-space detection.
ImageDetection = Detection
# Alias for the orthographic camera view specification.
CameraViewSpec = ViewSpec


def filter_detections(detections, conf_thresh=0.25):
    """Discard detections below the confidence threshold."""
    kept = [d for d in detections if d.conf >= conf_thresh]
    dropped = len(detections) - len(kept)
    if dropped:
        logger.debug("filter_detections: %d of %d below conf=%.2f discarded",
                     dropped, len(detections), conf_thresh)
    return kept


def load_detections(path, res=512, class_map=None):
    """Load detections from a file.

    Two formats are recognized:
      * .json  -> list of objects {view, bbox:[x0,y0,x1,y1], conf, cls}
                  or {view, bbox_norm:[xc,yc,w,h], conf, cls} (YOLO normalized)
      * .txt   -> YOLO lines "view cls xc yc w h [conf]" (normalized, 0..1)

    class_map : optional dict {file-id -> target-id}. YOLO/COCO files use the
                2D convention (Code 4); pass class_map=coco_labels.YOLO2D_TO_CONNECTOR
                to translate ids into the 3D/connector convention. Unknown ids
                are passed through unchanged.
    """
    def _map(cls):
        return cls if class_map is None else class_map.get(cls, cls)

    ext = os.path.splitext(path)[1].lower()
    out = []
    if ext == ".json":
        with open(path, encoding="utf-8") as fh:
            data = json.load(fh)
        for d in data:
            if "bbox" in d:
                bbox = tuple(d["bbox"])
            else:  # normalized xywh -> pixel xyxy
                bbox = _xywhn_to_xyxy(d["bbox_norm"], res)
            out.append(Detection(d["view"], bbox, float(d.get("conf", 1.0)),
                                 _map(int(d.get("cls", -1)))))
    elif ext == ".txt":
        for ln in open(path, encoding="utf-8"):
            s = ln.strip()
            if not s or s.startswith("#"):
                continue
            p = s.split()
            view, cls = p[0], int(float(p[1]))
            bbox = _xywhn_to_xyxy([float(p[2]), float(p[3]), float(p[4]), float(p[5])], res)
            conf = float(p[6]) if len(p) >= 7 else 1.0
            out.append(Detection(view, bbox, conf, _map(cls)))
    else:
        raise ValueError(f"unknown detection format: {ext} (only .json / .txt)")
    logger.debug("load_detections: %d detections from %s", len(out), os.path.basename(path))
    return out


def _xywhn_to_xyxy(xywh, res):
    xc, yc, w, h = xywh
    x0 = (xc - w / 2.0) * (res - 1)
    y0 = (yc - h / 2.0) * (res - 1)
    x1 = (xc + w / 2.0) * (res - 1)
    y1 = (yc + h / 2.0) * (res - 1)
    return (x0, y0, x1, y1)


# ---------------------------------------------------------------------------
# inverse: 2D box -> 3D slab (the box "pulled through" the model)
# ---------------------------------------------------------------------------

# §5.3.7 / Abb.46 – 2D bbox -> 3D slab: stretch bbox through full depth axis
def box_to_3d_slab(view, bbox, bounds, res=512, pad_frac=0.0, depth_pad_frac=0.0,
                   min_depth_extent=0.0):
    """Convert a 2D pixel box to an axis-aligned 3D slab.

    In the two image-plane axes the range comes from the box; along the depth
    axis the slab runs all the way through the part. `min_depth_extent` enforces
    a minimum slab thickness along the depth axis, preventing degenerate (zero
    thickness) crops on perfectly flat parts.

    Returns: (lo3, hi3) as a 3D AABB.
    """
    if isinstance(view, str):
        view = VIEWS[view]
    lo, hi = np.asarray(bounds[0], dtype=float), np.asarray(bounds[1], dtype=float)
    ext = _extent(lo, hi)
    x0, y0, x1, y1 = bbox

    def col_to_world(px):
        u_norm = px / (res - 1)
        t_u = u_norm if view.u_sign > 0 else 1.0 - u_norm
        return lo[view.u_axis] + t_u * ext[view.u_axis]

    def row_to_world(py):
        v_norm = 1.0 - py / (res - 1)  # undo the row flip from rendering
        t_v = v_norm if view.v_sign > 0 else 1.0 - v_norm
        return lo[view.v_axis] + t_v * ext[view.v_axis]

    u_vals = sorted([col_to_world(x0), col_to_world(x1)])
    v_vals = sorted([row_to_world(y0), row_to_world(y1)])

    lo3 = np.array(lo, dtype=float)
    hi3 = np.array(hi, dtype=float)
    lo3[view.u_axis], hi3[view.u_axis] = u_vals
    lo3[view.v_axis], hi3[view.v_axis] = v_vals
    # depth axis: full extent through the part (stays lo..hi)
    lo3[view.depth_axis] = lo[view.depth_axis]
    hi3[view.depth_axis] = hi[view.depth_axis]

    if pad_frac:
        for a in (view.u_axis, view.v_axis):
            m = pad_frac * (hi3[a] - lo3[a])
            lo3[a] -= m
            hi3[a] += m
    if depth_pad_frac:
        a = view.depth_axis
        m = depth_pad_frac * (hi3[a] - lo3[a])
        lo3[a] -= m
        hi3[a] += m

    if min_depth_extent > 0.0:
        a = view.depth_axis
        deficit = float(min_depth_extent) - (hi3[a] - lo3[a])
        if deficit > 0.0:
            lo3[a] -= deficit / 2.0
            hi3[a] += deficit / 2.0

    return lo3, hi3


# ---------------------------------------------------------------------------
# crop: reduce the mesh to the 3D box
# ---------------------------------------------------------------------------

def crop_mesh_to_box(vertices, faces, aabb3d, face_mode="all", eps_frac=1e-4,
                     depth_axis=None, front_band_frac=None):
    """Cut out the part of the mesh that lies inside the 3D box.

    face_mode = "all"  -> only triangles whose all 3 corners are inside the box.
    face_mode = "any"  -> triangles with at least one corner inside the box.

    The slab is unbounded along depth, so it also grabs geometry on the BACK
    of the part that projects into the same 2D box. If front_band_frac is set
    (e.g. 0.25) and depth_axis is known, only the frontmost depth layer is kept
    (renderer convention: smallest depth coordinate = front). Default None =
    off (full slab, needed for snap-point rails that run the full depth).

    Returns: (sub_V, sub_F, vertex_index)
      sub_V, sub_F : compact sub-mesh (re-indexed)
      vertex_index : original vertex indices of kept corners (for mapping labels
                     back), so that sub_V == vertices[vertex_index].
    """
    V = np.asarray(vertices, dtype=float)
    F = np.asarray(faces, dtype=int)
    lo3, hi3 = aabb3d
    lo3 = np.asarray(lo3, dtype=float)
    hi3 = np.asarray(hi3, dtype=float)

    eps = eps_frac * float(np.linalg.norm(hi3 - lo3))
    inside = np.all((V >= lo3 - eps) & (V <= hi3 + eps), axis=1)

    # optionally keep only the front depth layer (to avoid back-face bleed)
    if front_band_frac is not None and depth_axis is not None and inside.any():
        d = V[:, depth_axis]
        in_d = d[inside]
        extent = float(in_d.max() - in_d.min()) + 1e-9
        front = float(in_d.min())
        inside = inside & (d <= front + float(front_band_frac) * extent)

    tri_inside = inside[F]
    if face_mode == "all":
        keep_face = tri_inside.all(axis=1)
    elif face_mode == "any":
        keep_face = tri_inside.any(axis=1)
    else:
        raise ValueError(f"face_mode must be 'all' or 'any', not {face_mode!r}")

    kept_faces = F[keep_face]
    if len(kept_faces) == 0:
        return np.empty((0, 3), dtype=float), np.empty((0, 3), dtype=int), np.empty(0, dtype=int)

    # with face_mode "any", corners outside the box may be included too
    vertex_index = np.unique(kept_faces)
    remap = np.full(len(V), -1, dtype=int)
    remap[vertex_index] = np.arange(len(vertex_index))
    sub_V = V[vertex_index]
    sub_F = remap[kept_faces]
    return sub_V, sub_F, vertex_index


def crop_labels(labels, vertex_index):
    """Slice labels to match the crop (same order as sub_V)."""
    return np.asarray(labels)[vertex_index]


# ---------------------------------------------------------------------------
# crop analysis: centroid + principal axes
# ---------------------------------------------------------------------------

def crop_center_of_gravity(sub_V, sub_F=None):
    """Compute centroid and principal axes of a crop.

    Returns dict with:
      center        : cluster centroid (area-weighted if faces given, else vertex mean)
      volume_center : volume centroid via convex hull -- more robust to uneven meshes
      axes (3x3)    : principal axes from PCA, axis 0 = largest extent
      extents (3,)  : extents along the principal axes
    """
    V = np.asarray(sub_V, dtype=float)
    if len(V) == 0:
        z = np.zeros(3)
        return {"center": z, "volume_center": z, "axes": np.eye(3), "extents": z}

    if sub_F is not None and len(sub_F) > 0:
        F = np.asarray(sub_F, dtype=int)
        v0, v1, v2 = V[F[:, 0]], V[F[:, 1]], V[F[:, 2]]
        tri_c = (v0 + v1 + v2) / 3.0
        area = 0.5 * np.linalg.norm(np.cross(v1 - v0, v2 - v0), axis=1)
        if area.sum() > 0:
            center = (tri_c * area[:, None]).sum(axis=0) / area.sum()
        else:
            center = V.mean(axis=0)
    else:
        center = V.mean(axis=0)

    volume_center = convex_hull_volume_centroid(V)

    X = V - center
    cov = (X.T @ X) / max(len(V), 1)
    eigval, eigvec = np.linalg.eigh(cov)        # ascending
    order = np.argsort(eigval)[::-1]            # largest extent first
    axes = eigvec[:, order].T
    extents = np.sqrt(np.clip(eigval[order], 0, None)) * 2.0
    return {"center": center, "volume_center": volume_center, "axes": axes, "extents": extents}


# ---------------------------------------------------------------------------
# all together: detections -> crops
# ---------------------------------------------------------------------------

@dataclass
class Crop:
    detection: Detection
    aabb3d: tuple                 # (lo3, hi3)
    vertices: np.ndarray
    faces: np.ndarray
    vertex_index: np.ndarray
    cog: dict = field(default_factory=dict)

    @property
    def confidence_score(self):
        """Detector confidence carried through from the source 2D detection."""
        return float(self.detection.conf)


def detections_to_crops(vertices, faces, detections, bounds=None, res=512,
                        conf_thresh=0.25, pad_frac=0.0, face_mode="all",
                        bbox_pad_frac=0.0, only_classes=None, front_band_frac=None,
                        min_depth_extent=0.0):
    """Complete steps 3+4: convert each (filtered) detection into a 3D crop
    and compute its centroid/axes.

    bounds      : (lo, hi) of the part. If None, computed from vertices --
                  but must match the reference frame used when rendering,
                  otherwise the boxes won't line up.
    only_classes: optional set/list of class ids; only detections of those
                  classes are processed. For the final pipeline set this to
                  YOLO_PIPELINE_CLASSES (snap-point only).
    front_band_frac : keep only the frontmost depth layer of the slab (to avoid
                  back-face bleed). None = off (full slab, correct for snap points).
                  Use e.g. 0.2-0.3 for shallow, one-sided features.
    Returns: list of Crop (empty crops are skipped).
    """
    if bounds is None:
        bounds = world_aabb(vertices, pad_frac=pad_frac)

    kept = filter_detections(detections, conf_thresh)
    if only_classes is not None:
        allow = set(only_classes)
        before = len(kept)
        kept = [d for d in kept if d.cls in allow]
        logger.debug("detections_to_crops: %d of %d detections after class filter %s",
                     len(kept), before, sorted(allow))
    crops = []
    for det in kept:
        if det.view not in VIEWS:
            logger.warning("detections_to_crops: unknown view %r skipped", det.view)
            continue
        aabb = box_to_3d_slab(det.view, det.as_xyxy(), bounds, res, pad_frac=bbox_pad_frac,
                              min_depth_extent=min_depth_extent)
        sub_V, sub_F, vidx = crop_mesh_to_box(
            vertices, faces, aabb, face_mode=face_mode,
            depth_axis=VIEWS[det.view].depth_axis, front_band_frac=front_band_frac)
        if len(sub_V) == 0:
            logger.debug("detections_to_crops: empty crop for %s (cls=%d)", det.view, det.cls)
            continue
        cog = crop_center_of_gravity(sub_V, sub_F)
        crops.append(Crop(det, aabb, sub_V, sub_F, vidx, cog))

    logger.debug("detections_to_crops: %d crops from %d detections", len(crops), len(kept))
    return crops


# ---------------------------------------------------------------------------
# self-test
# ---------------------------------------------------------------------------

def _demo():
    # A plate with a round cavity (like demo_normal). We project the cavity
    # into the top view, take its pixel bbox as a "detection", back-project
    # to a 3D slab and crop -- the crop must contain the cavity and its
    # centroid must be centered.
    res = 60
    xs = np.linspace(0, 1.0, res)
    cx = cy = 0.5
    cavity_r, depth = 0.28, 0.25
    V, lab = [], []
    for y in xs:
        for x in xs:
            r = np.hypot(x - cx, y - cy)
            z = -depth * (1.0 - r / cavity_r) if r < cavity_r else 0.0
            V.append((x, y, z))
            lab.append(1 if r < cavity_r else 0)
    V = np.array(V, dtype=float)
    lab = np.array(lab, dtype=int)

    F = []
    for iy in range(res - 1):
        for ix in range(res - 1):
            a = iy * res + ix
            b, c, d = a + 1, a + res, a + res + 1
            F.append((a, b, d)); F.append((a, d, c))
    F = np.array(F, dtype=int)

    bounds = world_aabb(V, pad_frac=0.02)
    R = 512

    # forward: project cavity vertices and get their pixel bbox
    cavity_pts = V[lab == 1]
    px = project_points("z+", cavity_pts, bounds, res=R)
    x0, y0 = px.min(axis=0)
    x1, y1 = px.max(axis=0)
    det = Detection("z+", (x0, y0, x1, y1), conf=0.9, cls=1)
    print(f"[forward]  cavity -> pixel bbox = ({x0:.0f},{y0:.0f},{x1:.0f},{y1:.0f})")

    # inverse: back to slab + crop
    crops = detections_to_crops(V, F, [det], bounds=bounds, res=R,
                                conf_thresh=0.25, face_mode="any")
    assert crops, "no crop produced -- round-trip failed"
    crop = crops[0]
    sub_lab = crop_labels(lab, crop.vertex_index)
    recall = (sub_lab == 1).sum() / max((lab == 1).sum(), 1)

    print(f"[inverse]  slab lo={np.round(crop.aabb3d[0], 3)}  hi={np.round(crop.aabb3d[1], 3)}")
    print(f"[crop]     {len(crop.vertices)} vertices, {len(crop.faces)} faces")
    print(f"           of which cavity: {(sub_lab == 1).sum()} / {(lab == 1).sum()}  (recall={recall:.2f})")
    print(f"[cog]      cluster centroid = {np.round(crop.cog['center'], 3)}  (expected ~ [0.5, 0.5, ~-0.08])")
    print(f"           volume centroid  = {np.round(crop.cog['volume_center'], 3)}  (convex hull)")
    print(f"           principal axis   = {np.round(crop.cog['axes'][0], 2)}")

    depth_map, mask = render_view("z+", V, bounds, res=64)
    print(f"[render]   z+ silhouette: {int(mask.sum())} pixels hit (of {mask.size})")

    ok = recall > 0.9 and abs(crop.cog["center"][0] - 0.5) < 0.05 and abs(crop.cog["center"][1] - 0.5) < 0.05
    print("\n[ok] round-trip passed." if ok else "\n[!!] round-trip suspicious -- please check.")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
    _demo()
