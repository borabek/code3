# feature_geometry.py: PCA normal alignment, convex hull centroid, boundary peeling
#
# Reference: Scheffler (2022), §5.3.6
#   §5.3.6 / Abb.44 – convex-hull volume centroid (v_v)
#   §5.3.6 / Abb.44 – surface centroid via boundary peeling (v_s),
#                      opening midpoint (v_o)
#   §5.3.6 / Abb.44 – normal vector = v_o − v_s, aligned to bbox axes

import logging
import numpy as np
from collections import Counter
from connector_constants import MIN_VOLUME_THRESHOLD, VERTEX_NORMAL_EPSILON

logger = logging.getLogger(__name__)


def _principal_axes(points):
    c = points.mean(0)
    _, vecs = np.linalg.eigh((points - c).T @ (points - c) / max(len(points), 1))
    return vecs.T


# §5.3.6 / Abb.44 – normal vector aligned to principal (bbox) axes
def align_to_bbox(points, rough):
    axes = _principal_axes(points)
    k = int(np.argmax(np.abs(axes @ rough)))
    n = axes[k].copy()
    if (axes[k] @ rough) < 0:
        n = -n
    return n


def normal_stability_score(points, rough):
    """Reliability of the bbox-aligned normal, in [0, 1].

    Measures how unambiguously one principal axis dominates the rough
    direction: a value near 1 means a single axis carries almost all of the
    rough vector (stable), near 0 means it is split across axes (unreliable,
    e.g. on a near-spherical or skewed feature).
    """
    rough = np.asarray(rough, dtype=float)
    rn = np.linalg.norm(rough)
    if rn < VERTEX_NORMAL_EPSILON or len(points) < 3:
        return 0.0
    proj = np.abs(_principal_axes(points) @ (rough / rn))
    proj = np.sort(proj)[::-1]
    # dominance of the strongest axis over the runner-up
    return float(np.clip(proj[0] - proj[1], 0.0, 1.0))


# A robot approaching from the wrong side of the normal would collide with the
# housing, so the approach vector must point away from the component body.
def orient_outward(normal, feature_center, body_center):
    """Flip `normal` so it points away from the component body center."""
    n = np.asarray(normal, dtype=float)
    outward = np.asarray(feature_center, dtype=float) - np.asarray(body_center, dtype=float)
    if float(np.dot(n, outward)) < 0.0:
        n = -n
    return n


def insertion_depth(v_o, v_s):
    """Insertion depth = distance between the opening midpoint v_o and the
    surface centroid v_s (Scheffler §5.3.6 / Abb.44)."""
    return float(np.linalg.norm(np.asarray(v_o, dtype=float) - np.asarray(v_s, dtype=float)))


def insertion_depth_along_axis(points, insertion_axis):
    """Physical insertion depth: the extent of the feature's vertices measured
    along the insertion axis (the inward direction the tool travels).

    Complements the thesis proxy `insertion_depth` (|v_o - v_s|): instead of the
    distance between two centroids it returns the actual travel span into the
    feature along the approach/insertion direction, which is robust to asymmetric
    or off-centre openings. Sign-invariant (max - min projection), so passing the
    inward or outward axis gives the same value. Returns 0.0 for degenerate input.
    """
    a = np.asarray(insertion_axis, dtype=float)
    na = np.linalg.norm(a)
    P = np.asarray(points, dtype=float).reshape(-1, 3)
    if na < VERTEX_NORMAL_EPSILON or len(P) < 2:
        return 0.0
    proj = P @ (a / na)
    return float(proj.max() - proj.min())


def validate_feature_geometry(points, max_skew_deg=75.0):
    """Check whether a feature's geometry is suitable for robotic interaction.

    Rejects features whose principal axes are extremely skewed/elongated
    (e.g. a 10 mm × 0.1 mm sliver from a segmentation error). Returns
    (is_valid, info) where info carries the principal extents, aspect ratio
    and the planarity/skew angle.
    """
    P = np.asarray(points, dtype=float).reshape(-1, 3)
    if len(P) < 3:
        return False, {"reason": "too_few_points", "aspect_ratio": float("inf"),
                       "extents": np.zeros(3), "skew_deg": 90.0}
    c = P.mean(0)
    cov = (P - c).T @ (P - c) / len(P)
    eigval = np.sort(np.clip(np.linalg.eigvalsh(cov), 0.0, None))[::-1]
    extents = np.sqrt(eigval) * 2.0
    major, minor = extents[0], extents[1]
    aspect_ratio = float(major / minor) if minor > 1e-9 else float("inf")
    # skew: how far the in-plane shape departs from a clean planar patch.
    thickness = extents[2]
    skew_deg = float(np.degrees(np.arctan2(thickness, major + 1e-12)))
    is_valid = aspect_ratio <= 50.0 and skew_deg <= max_skew_deg
    return is_valid, {"aspect_ratio": aspect_ratio, "extents": extents,
                      "skew_deg": skew_deg}


# §5.3.6 / Abb.44 – convex-hull volume centroid (v_v): centroid weighted by tetrahedron volumes
def convex_hull_volume_centroid(points):
    P = np.asarray(points, dtype=float).reshape(-1, 3)
    if len(P) < 4:
        return P.mean(0) if len(P) else np.zeros(3)
    try:
        from scipy.spatial import ConvexHull
        hull = ConvexHull(P)
    except ImportError:
        return P.mean(0)
    except Exception:
        return P.mean(0)
    apex = P[hull.vertices].mean(0)
    w = 0.0
    centroid = np.zeros(3)
    for ai, bi, ci in hull.simplices:
        a, b, c_ = P[ai], P[bi], P[ci]
        v = abs(np.dot(a - apex, np.cross(b - apex, c_ - apex))) / 6.0
        w += v; centroid += v * (apex + a + b + c_) / 4.0
    if w < MIN_VOLUME_THRESHOLD:
        return P.mean(0)
    return centroid / w


def _boundary_vertices(tris):
    cnt = Counter()
    for a, b, c in tris:
        for e in ((a, b), (b, c), (a, c)):
            cnt[tuple(sorted(e))] += 1
    return {u for (u, v), c in cnt.items() if c == 1} | {v for (u, v), c in cnt.items() if c == 1}


# §5.3.6 / Abb.44 – boundary peeling to compute surface centroid (v_s) and outer ring (v_o)
def _peel_to_centroid(V, sub):
    active = set(np.unique(sub).tolist())
    outer_ring = None
    Nloc = len(V)
    while True:
        mask = np.zeros(Nloc, dtype=bool)
        mask[list(active)] = True
        cur = sub[mask[sub].all(1)]
        if not len(cur):
            break
        rand = _boundary_vertices(cur)
        if outer_ring is None:
            outer_ring = rand
        inner = active - rand
        if not inner:
            break
        active = inner
    return V[list(active)].mean(0), outer_ring


# §5.3.6 / Abb.44 – compute v_o, v_s, and bbox-aligned normal for one feature instance.
# Returns (v_o, v_s, normal, depth_mm, depth_axis_mm, stability):
#   depth_mm      – thesis insertion depth |v_o - v_s| (§5.3.6),
#   depth_axis_mm – physical travel span of the feature along the insertion axis,
#   stability     – in [0, 1], rates the bbox-aligned normal's reliability.
def feature_direction(idx, vertices, faces, fallback_normal, smooth_subdiv=0):
    idx = np.asarray(idx)
    N = len(vertices)
    inside = np.zeros(N, dtype=bool)
    inside[idx] = True
    sub = faces[inside[faces].all(1)]
    if not len(sub):
        c = vertices[idx].mean(0)
        return c, c, fallback_normal, 0.0, 0.0, 0.0
    v_s, outer_ring = _peel_to_centroid(vertices, sub)
    v_o = vertices[list(outer_ring)].mean(0) if outer_ring else vertices[idx].mean(0)
    if smooth_subdiv > 0:
        from remesh import subdivide_loop
        loc = np.unique(sub)
        remap = np.full(N, -1, dtype=int)
        remap[loc] = np.arange(len(loc))
        Vl, Fl = vertices[loc], remap[sub]
        for _ in range(smooth_subdiv):
            Vl, Fl = subdivide_loop(Vl, Fl)
        v_s, _ = _peel_to_centroid(Vl, Fl)
    center = vertices[idx].mean(0)
    scale = float(np.linalg.norm(vertices[idx] - center, axis=1).max()) or 1.0
    d = v_o - v_s
    nl = np.linalg.norm(d)
    rough = (d / nl) if nl >= 0.02 * scale else np.asarray(fallback_normal, dtype=float)
    rn = np.linalg.norm(rough)
    if rn > VERTEX_NORMAL_EPSILON: rough = rough / rn
    normal = align_to_bbox(vertices[idx], rough)
    depth_mm = insertion_depth(v_o, v_s)                              # thesis proxy |v_o - v_s|
    depth_axis_mm = insertion_depth_along_axis(vertices[idx], normal)  # physical travel span
    stability = normal_stability_score(vertices[idx], rough)
    return v_o, v_s, normal, depth_mm, depth_axis_mm, stability
