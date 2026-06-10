"""
3D visualization and rendering: .obj scene export + 6-view orthographic renderer.

Reference: Scheffler (2022), §5.3.4, §2.4
  §5.3.4 / Abb.41,42 – YOLOv6 with RepVGG backbone, 6 grayscale 512x512 views,
                        VariFocal + IoU loss (view rendering for YOLO input)
  §2.4 / Abb.28       – 6 orthographic views: top/bottom/front/rear/left/right =
                         z+/z-/y+/y-/x+/x-
"""

import os
import logging
import numpy as np
from collections import defaultdict

from projection_2d3d import VIEWS, VIEW_ORDER, project_points, world_aabb

logger = logging.getLogger(__name__)


# =============================================================================
# SCENE EXPORT (.OBJ)
# =============================================================================

def export_scene_obj(path, instances, vertices=None, faces=None, arrow=None,
                     insertion_len=None):
    """Export connector scene to .obj file for visualization.

    Args:
        path: Output .obj file path
        instances: List of ConnectionPoint objects
        vertices: Optional original mesh vertices
        faces: Optional original mesh faces
        arrow: Arrow scale for normal vectors (auto-scaled if None)
        insertion_len: Length of the robot insertion-path line drawn from each
            connection point's entry point along its approach vector. Defaults
            to 100 mm (the ~10 cm approach stroke in the thesis) when the part
            appears to be in millimetres, otherwise auto-scaled to the scene.

    Returns:
        path to written file
    """
    V = []  # all vertices in output

    def add_v(p):
        V.append((float(p[0]), float(p[1]), float(p[2])))
        return len(V)  # obj counts from 1

    # Auto-scale arrows relative to scene
    if arrow is None:
        if vertices is not None:
            lo, hi = np.min(vertices, axis=0), np.max(vertices, axis=0)
        else:
            pts = np.array([inst.center for inst in instances])
            lo, hi = pts.min(axis=0), pts.max(axis=0)
        arrow = 0.12 * float(np.linalg.norm(hi - lo))
    else:
        if vertices is not None:
            lo, hi = np.min(vertices, axis=0), np.max(vertices, axis=0)
        else:
            pts = np.array([inst.center for inst in instances])
            lo, hi = pts.min(axis=0), pts.max(axis=0)

    # Robot insertion-path length: ~10 cm stroke if the part is in mm, else
    # auto-scaled so the line stays visible on small/normalized meshes.
    diag = float(np.linalg.norm(hi - lo))
    if insertion_len is None:
        insertion_len = 100.0 if diag > 50.0 else 0.5 * diag

    # Optional: write original mesh as context
    mesh_faces = []
    if vertices is not None and faces is not None:
        base = len(V)
        for p in vertices:
            V.append((float(p[0]), float(p[1]), float(p[2])))
        for f in faces:
            mesh_faces.append((int(f[0]) + base + 1,
                               int(f[1]) + base + 1,
                               int(f[2]) + base + 1))

    # Per feature: center point and normal vector as line
    center_idx = {}
    lines = []
    points = []
    for inst in instances:
        ci = add_v(inst.center)
        center_idx[id(inst)] = ci
        points.append(("centers", ci))
        tip = np.asarray(inst.center) + np.asarray(inst.normal3d) * arrow
        ti = add_v(tip)
        lines.append(("normals", ci, ti))

        # robot insertion path: from the entry point straight along the approach
        # vector. Only drawn for points the robot may actually act on.
        if getattr(inst, "is_valid_for_robot", True):
            entry = getattr(inst, "entry_point", None)
            entry = inst.center if entry is None else np.asarray(entry)
            approach = np.asarray(getattr(inst, "approach_vector", inst.normal3d))
            ei = add_v(entry)
            pi = add_v(entry + approach * insertion_len)
            points.append(("entry_points", ei))
            lines.append(("insertion_paths", ei, pi))

    # Feature links and wiring candidates as lines
    # (wires are geometry candidates, not electrical wiring)
    for inst in instances:
        a = center_idx[id(inst)]
        for tgt in inst.points_to:
            lines.append(("feature_links", a, center_idx[id(tgt)]))
        for w in inst.wires:
            lines.append(("wires", a, center_idx[id(w.dst)]))

    with open(path, "w", encoding="utf-8") as fh:
        fh.write("# connector scene\n")
        for p in V:
            fh.write(f"v {p[0]} {p[1]} {p[2]}\n")

        if mesh_faces:
            fh.write("g part\n")
            for f in mesh_faces:
                fh.write(f"f {f[0]} {f[1]} {f[2]}\n")

        # Write lines sorted by group
        gl = defaultdict(list)
        for g, i, j in lines:
            gl[g].append((i, j))
        for g, segs in gl.items():
            fh.write(f"g {g}\n")
            for i, j in segs:
                fh.write(f"l {i} {j}\n")

        # Write points sorted by group
        gp = defaultdict(list)
        for g, i in points:
            gp[g].append(i)
        for g, ids in gp.items():
            fh.write(f"g {g}\n")
            for i in ids:
                fh.write(f"p {i}\n")

    return path


# =============================================================================
# 6-VIEW RENDERING
# =============================================================================

def _face_normals(V, F):
    """Compute per-face normals."""
    v0, v1, v2 = V[F[:, 0]], V[F[:, 1]], V[F[:, 2]]
    n = np.cross(v1 - v0, v2 - v0)
    ln = np.linalg.norm(n, axis=1, keepdims=True)
    ln[ln == 0] = 1.0
    return n / ln


# §5.3.4 / Abb.41,42 – render one grayscale 512x512 orthographic view (YOLOv6 input)
# §2.4 / Abb.28 – view convention: z+=top, z-=bottom, y+=front, y-=rear, x+=right, x-=left
def render_view(view, vertices, faces, bounds, res=512,
                background=255, ambient=0.25):
    """Render single orthographic view as grayscale image.

    Uses proper triangle rasterizer with z-buffer and Lambert shading
    (light from camera).

    Args:
        view: VIEWS[name] or str like "z+", "x-", etc.
        vertices: (N, 3) mesh vertices
        faces: (M, 3) mesh faces
        bounds: [lo, hi] world AABB
        res: image resolution (res×res)
        background: background pixel value (0-255)
        ambient: ambient shading factor (0-1)

    Returns:
        uint8 image array (res, res)
    """
    if isinstance(view, str):
        view = VIEWS[view]
    V = np.asarray(vertices, dtype=float)
    F = np.asarray(faces, dtype=int)
    lo, hi = np.asarray(bounds[0], dtype=float), np.asarray(bounds[1], dtype=float)
    ext = (hi - lo).copy()
    ext[ext == 0] = 1.0

    px = project_points(view, V, bounds, res)            # (N,2) pixels
    t_d = (V[:, view.depth_axis] - lo[view.depth_axis]) / ext[view.depth_axis]  # 0..1, 0=lo
    # For "+" views the camera sits on the high-coordinate side, so the near
    # surface has the LARGEST coordinate (t_d near 1). Flip so that "smaller
    # z-buffer value = closer to camera" holds for every view.
    if view.name.endswith("+"):
        t_d = 1.0 - t_d

    # Per-face shading: |normal along view axis| + ambient
    fn = _face_normals(V, F)
    shade = ambient + (1.0 - ambient) * np.abs(fn[:, view.depth_axis])
    grey = np.clip(shade * 220.0, 0, 255)                # component grayscale

    img = np.full((res, res), float(background), dtype=float)
    zbuf = np.full((res, res), np.inf, dtype=float)

    p = px
    for fi in range(len(F)):
        a, b, c = F[fi]
        x0, y0 = p[a]; x1, y1 = p[b]; x2, y2 = p[c]
        # Pixel bounding box of triangle
        xmin = max(int(np.floor(min(x0, x1, x2))), 0)
        xmax = min(int(np.ceil(max(x0, x1, x2))), res - 1)
        ymin = max(int(np.floor(min(y0, y1, y2))), 0)
        ymax = min(int(np.ceil(max(y0, y1, y2))), res - 1)
        if xmax < xmin or ymax < ymin:
            continue

        xs = np.arange(xmin, xmax + 1)
        ys = np.arange(ymin, ymax + 1)
        gx, gy = np.meshgrid(xs, ys)                     # pixel centers

        # Barycentric coordinates via edge functions
        denom = (y1 - y2) * (x0 - x2) + (x2 - x1) * (y0 - y2)
        if abs(denom) < 1e-9:
            continue
        w0 = ((y1 - y2) * (gx - x2) + (x2 - x1) * (gy - y2)) / denom
        w1 = ((y2 - y0) * (gx - x2) + (x0 - x2) * (gy - y2)) / denom
        w2 = 1.0 - w0 - w1
        inside = (w0 >= 0) & (w1 >= 0) & (w2 >= 0)
        if not inside.any():
            continue

        depth = w0 * t_d[a] + w1 * t_d[b] + w2 * t_d[c]
        sub_z = zbuf[ymin:ymax + 1, xmin:xmax + 1]
        sub_i = img[ymin:ymax + 1, xmin:xmax + 1]
        closer = inside & (depth < sub_z)
        sub_z[closer] = depth[closer]
        sub_i[closer] = grey[fi]

    return img.astype(np.uint8)


# §2.4 / Abb.28 – render all 6 orthographic views (top/bottom/front/rear/left/right)
def render_all_views(vertices, faces, bounds=None, res=512, pad_frac=0.05):
    """Render all 6 orthographic views.

    Args:
        vertices: (N, 3) mesh vertices
        faces: (M, 3) mesh faces
        bounds: [lo, hi] world AABB (auto-computed if None)
        res: image resolution per view
        pad_frac: padding fraction for bounds

    Returns:
        dict {view_name -> uint8_image}
    """
    if bounds is None:
        bounds = world_aabb(vertices, pad_frac=pad_frac)
    return {name: render_view(name, vertices, faces, bounds, res=res)
            for name in VIEW_ORDER}


# =============================================================================
# IMAGE SAVING
# =============================================================================

def save_image(path, img):
    """Save image as PNG/JPG or portable PGM (no deps).

    Args:
        path: Output file path (.png, .jpg, or .pgm)
        img: uint8 image array

    Returns:
        path to saved file
    """
    img = np.asarray(img, dtype=np.uint8)
    ext = os.path.splitext(path)[1].lower()
    if ext in (".png", ".jpg", ".jpeg"):
        try:
            from PIL import Image
            Image.fromarray(img, mode="L").save(path)
            return path
        except ImportError:
            path = os.path.splitext(path)[0] + ".pgm"   # fallback
    # PGM (P5) format — no library needed
    h, w = img.shape
    with open(path, "wb") as fh:
        fh.write(f"P5\n{w} {h}\n255\n".encode("ascii"))
        fh.write(img.tobytes())
    return path


def save_views(out_dir, views, stem="component", res=512):
    """Save 6 rendered views to disk.

    Args:
        out_dir: Output directory
        views: dict {view_name -> image}
        stem: filename stem
        res: image resolution

    Returns:
        dict {view_name -> saved_path}
    """
    os.makedirs(out_dir, exist_ok=True)
    paths = {}
    for name, img in views.items():
        # Map render names (x+/z-) to thesis names (right/bottom)
        nice = {"z+": "top", "z-": "bottom", "y+": "front", "y-": "rear",
                "x+": "right", "x-": "left"}.get(name, name)
        paths[name] = save_image(os.path.join(out_dir, f"{stem}_{nice}.png"), img)
    return paths


# =============================================================================
# SELF-TEST
# =============================================================================

def _demo():
    """Test rendering on funnel-shaped mesh."""
    # Funnel-shaped plate (like demo_normal): shows occlusion + shading
    res = 70
    xs = np.linspace(0, 1.0, res)
    cx = cy = 0.5
    V = []
    for y in xs:
        for x in xs:
            r = np.hypot(x - cx, y - cy)
            z = -0.25 * (1.0 - r / 0.28) if r < 0.28 else 0.0
            V.append((x, y, z))
    V = np.array(V, dtype=float)
    F = []
    for iy in range(res - 1):
        for ix in range(res - 1):
            a = iy * res + ix
            b, c, d = a + 1, a + res, a + res + 1
            F.append((a, b, d)); F.append((a, d, c))
    F = np.array(F, dtype=int)

    views = render_all_views(V, F, res=128)
    for name, img in views.items():
        covered = int((img != 255).sum())
        print(f"   view {name:3s}: {covered:6d} component pixels of {img.size}  (grey {img.min()}..{img.max()})")
    top = views["z+"]
    assert (top != 255).sum() > 0, "top view is empty"
    print("[ok] 6-view renderer: proper rasterization with z-buffer + shading")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
    _demo()
