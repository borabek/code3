# main module for the 3D connector: mesh + labels -> feature instances + connector graph
#
# Reference: Scheffler (2022), §5.3.5, §5.3.6, §2.6.1/2.6.2
#   §5.3.5 / Abb.43   – instance segmentation by graph connectivity (Union-Find),
#                        boundary peeling, min_vertices threshold
#   §5.3.6 / Abb.44   – cluster centroid (v_s = mean of vertices), convex-hull
#                        volume centroid (v_v), surface centroid via boundary peeling,
#                        opening midpoint (v_o)
#   §5.3.6 / Abb.44   – normal vector = v_o − v_s, aligned to bbox axes
#   §5.3.6             – feature size = sum of triangle areas
#   §2.6.1/2.6.2       – triangle mesh as 2-manifold

import logging
from typing import List
from dataclasses import dataclass, field
import numpy as np
from feature_geometry import (
    feature_direction, convex_hull_volume_centroid, orient_outward,
    validate_feature_geometry,
)
# SNAP_POINT and CABLE_ENTRY are re-exported here (not used inside this module)
# so callers can import the full label vocabulary from connector3d.
from connector_constants import (
    HOUSING, CONTACT, SNAP_POINT, CABLE_ENTRY, LABEL_SURFACE,  # noqa: F401
    LABEL_NAMES, ROBOTIC_LABEL_NAMES, TERMINAL_TYPES,
    MIN_NORMAL_STABILITY_SCORE, MIN_FEATURE_AREA_MM2,
    VERTEX_NORMAL_EPSILON,
    MIN_PLAUSIBLE_PART_MM, MAX_PLAUSIBLE_PART_MM,
)

logger = logging.getLogger(__name__)


# §5.3.5 / Abb.43 – Union-Find for graph-connectivity instance segmentation
# standard union-find, used twice: once for fragment detection
# and once to merge fragments into instances
class UnionFind:
    """Efficient disjoint-set with path compression and union by size."""

    def __init__(self, n: int):
        """Initialize n disjoint sets (one per element)."""
        self.parent = list(range(n))
        self.size = [1] * n

    def find(self, i: int) -> int:
        """Find root of set containing i, with path compression."""
        root = i
        while self.parent[root] != root:
            root = self.parent[root]
        # compress path for faster future lookups
        while self.parent[i] != root:
            self.parent[i], i = root, self.parent[i]
        return root

    def union(self, a: int, b: int) -> None:
        """Union sets containing a and b by attaching smaller to larger."""
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return
        if self.size[ra] < self.size[rb]:
            ra, rb = rb, ra
        self.parent[rb] = ra
        self.size[ra] += self.size[rb]


# ---------------------------------------------------------------------------
# Instance segmentation helpers (merged from instances.py)
# ---------------------------------------------------------------------------

@dataclass
class SegmentedRegion:
    """A connected, single-class region of the mesh (one instance of a feature class)."""
    label: int
    vertex_index: np.ndarray
    center: np.ndarray = field(default=None)


Instance = SegmentedRegion  # backward-compatible alias


def _inst_adj(faces, n):
    nb = [set() for _ in range(n)]
    for a, b, c in np.asarray(faces, dtype=int):
        nb[a].update((b, c)); nb[b].update((a, c)); nb[c].update((a, b))
    return nb


def remove_boundary_layer(faces, labels, bg=HOUSING, iterations=1):
    """Reclassify boundary vertices as background (boundary peeling, §5.3.5)."""
    L = np.asarray(labels, dtype=int).copy()
    nb = _inst_adj(faces, len(L))
    for _ in range(max(1, int(iterations))):
        out = L.copy(); changed = False
        for i in range(len(L)):
            if L[i] != bg and any(L[j] != L[i] for j in nb[i]):
                out[i] = bg; changed = True
        L = out
        if not changed:
            break
    return L


def connected_components(faces, labels, bg=HOUSING):
    """Find connected components of each label via union-find (§5.3.5)."""
    L = np.asarray(labels, dtype=int)
    uf = UnionFind(len(L))
    for a, b, c in np.asarray(faces, dtype=int):
        for u, v in ((a, b), (b, c), (a, c)):
            if L[u] == L[v] != bg:
                uf.union(u, v)
    groups = {}
    for i in range(len(L)):
        if L[i] != bg:
            groups.setdefault(uf.find(i), []).append(i)
    return [Instance(int(L[idx[0]]), np.array(idx, dtype=int)) for idx in groups.values()]


def instances_by_connectivity(vertices, faces, labels, bg=HOUSING,
                              min_vertices=20, peel=True, peel_iterations=1):
    """Extract instances via boundary peeling + connectivity + size filtering."""
    V = np.asarray(vertices, dtype=float)
    work = (remove_boundary_layer(faces, labels, bg, iterations=peel_iterations)
            if peel else np.asarray(labels, dtype=int))
    comps = connected_components(faces, work, bg)
    kept = [c for c in comps if len(c.vertex_index) >= min_vertices]
    for inst in kept:
        inst.center = V[inst.vertex_index].mean(axis=0)
    return kept


def _kmeans(P, k, iters=50, seed=0):
    P = np.asarray(P, dtype=float)
    if k <= 1 or len(P) <= k:
        return np.zeros(len(P), dtype=int), P.mean(0, keepdims=True) if len(P) else P
    rng = np.random.default_rng(seed)
    C = [P[rng.integers(len(P))]]
    for _ in range(1, k):
        d2 = np.min([((P - c)**2).sum(1) for c in C], 0)
        C.append(P[rng.choice(len(P), p=d2/d2.sum() if d2.sum() > 0 else None)])
    C = np.array(C); lab = np.zeros(len(P), dtype=int)
    for _ in range(iters):
        new = ((P[:, None] - C[None])**2).sum(2).argmin(1)
        if np.array_equal(new, lab): break
        lab = new
        for j in range(k):
            if (lab == j).any(): C[j] = P[lab == j].mean(0)
    return lab, C


def instances_by_kmeans(vertices, labels, target_class, k):
    """Partition vertices of a class into k instances via k-Means."""
    V, L = np.asarray(vertices, dtype=float), np.asarray(labels, dtype=int)
    idx = np.where(L == target_class)[0]
    if not len(idx) or k < 1:
        return []
    km, _ = _kmeans(V[idx], k)
    return [Instance(int(target_class), idx[km == j], V[idx[km == j]].mean(0))
            for j in range(int(km.max()) + 1) if (km == j).any()]


# ---------------------------------------------------------------------------
# area-weighted vertex normal: each triangle contributes proportional to its area
# so small boundary triangles don't skew the result
def vertex_normals(vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
    """Compute area-weighted vertex normals for triangular mesh."""
    vn = np.zeros_like(vertices, dtype=float)
    v0, v1, v2 = vertices[faces[:, 0]], vertices[faces[:, 1]], vertices[faces[:, 2]]
    fn = np.cross(v1 - v0, v2 - v0)
    for k in range(3):
        np.add.at(vn, faces[:, k], fn)
    norm = np.linalg.norm(vn, axis=1, keepdims=True)
    norm[norm == 0] = 1.0
    return vn / norm


def face_areas(vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
    """Compute triangle areas for each face."""
    v0, v1, v2 = vertices[faces[:, 0]], vertices[faces[:, 1]], vertices[faces[:, 2]]
    return 0.5 * np.linalg.norm(np.cross(v1 - v0, v2 - v0), axis=1)


# a detection patch is a contiguous region of uniform label in the mesh
# (a raw geometric segment). e.g. a rail-mount could be split into two
# patches by a gap.
class DetectionPatch:
    """A connected region of the mesh with uniform label (raw geometric segment)."""

    def __init__(self, idx: np.ndarray, points: np.ndarray, label: int,
                 normal: np.ndarray, area: float):
        self.idx = idx
        self.points = points
        self.label = label
        self.center = points.mean(axis=0)
        self.radius = float(np.linalg.norm(points - self.center, axis=1).max()) if len(points) > 1 else 0.0
        self.normal = normal
        self.area = area


# Backward-compatible alias (previously named Fragment).
Fragment = DetectionPatch


# a connection point merges one or more detection patches belonging to the same
# physical feature. this is the actual robot-actionable result we want.
class ConnectionPoint:
    """A physical connection point for the wiring robot.

    Merges detection patches of the same class that are spatially close, and
    carries the robot-ready geometry: refined approach vector, insertion depth,
    confidence and a robot-validity flag.
    """

    def __init__(self, fragments: List[DetectionPatch]):
        """Initialize a connection point from one or more detection patches."""
        if not fragments:
            raise ValueError("ConnectionPoint requires at least one detection patch")
        if not all(f.label == fragments[0].label for f in fragments):
            raise ValueError("All detection patches must have the same label")

        self.fragments = fragments
        self.label = fragments[0].label

        pts = np.vstack([f.points for f in fragments])
        self.center = pts.mean(axis=0)

        # area-weighted average normal; small splinters should not skew the result
        weights = np.array([f.area for f in fragments])
        if weights.sum() == 0:
            weights = np.ones(len(fragments))
        normals = np.array([f.normal for f in fragments])
        n = (normals * weights[:, None]).sum(axis=0)
        nl = np.linalg.norm(n)
        self.normal = n / nl if nl > VERTEX_NORMAL_EPSILON else np.array([0.0, 0.0, 1.0])

        self.size = float(sum(f.area for f in fragments))  # §5.3.6 – feature size = sum of triangle areas
        self.n_vertices = int(sum(len(f.idx) for f in fragments))

        # §5.3.6 / Abb.44 – cluster centroid = mean of all vertices (v_s),
        # v_o (opening midpoint), v_v (volume centroid via convex hull)
        # are filled by compute_direction -- together these are the three
        # centroids from Figure 44 of the thesis.
        self.v_o = None
        self.v_s = None
        self.v_v = None
        self.normal3d = self.normal  # preliminary value until compute_direction runs

        # ---- robot-ready geometry (filled by compute_direction / validation) ----
        self.insertion_depth_mm = 0.0       # thesis proxy |v_o - v_s| (§5.3.6)
        self.insertion_depth_axis_mm = 0.0  # physical travel span along insertion axis
        self.normal_stability = 0.0
        self.confidence_score = 1.0        # refined when detector confidence is known
        self.is_valid_for_robot = True     # set by robotic_validation()
        self.invalid_reason = None         # why the point was rejected (None if valid)
        self.approach_distance_mm = 0.0    # standoff distance, set by robotic_validation()
        self.terminal_type = TERMINAL_TYPES.get(self.label, None)

        self.points_to = []  # e.g. contact -> label surface
        self.wires = []      # wiring candidates to other contacts

    @property
    def approach_vector(self) -> np.ndarray:
        """Unit vector the robot tool follows to approach this point (outward normal)."""
        return self.normal3d

    @property
    def entry_point(self) -> np.ndarray:
        """World-coordinate point the robot aims for (opening midpoint v_o, else center)."""
        return self.v_o if self.v_o is not None else self.center

    @property
    def insertion_axis(self) -> np.ndarray:
        """Direction the tool travels while inserting -- opposite the outward approach."""
        return -self.normal3d

    @property
    def standoff_point(self) -> np.ndarray:
        """Pre-insertion standoff pose: entry point offset outward along the
        approach vector by `approach_distance_mm` (0 standoff = entry point)."""
        return self.entry_point + self.approach_vector * float(self.approach_distance_mm)

    # §5.3.6 / Abb.44 – normal vector = v_o − v_s, aligned to bbox axes;
    #                    v_v = convex-hull volume centroid
    def compute_direction(self, vertices: np.ndarray, faces: np.ndarray,
                         smooth_subdiv: int = 0, body_center=None) -> np.ndarray:
        """Compute refined approach vector via boundary peeling (feature_geometry module).

        If `body_center` is given, the approach vector is oriented to point away
        from the component body so the robot never approaches through the housing.
        """
        idx = np.concatenate([f.idx for f in self.fragments])
        (self.v_o, self.v_s, self.normal3d, self.insertion_depth_mm,
         self.insertion_depth_axis_mm, self.normal_stability) = \
            feature_direction(idx, vertices, faces, self.normal, smooth_subdiv=smooth_subdiv)
        if body_center is not None:
            self.normal3d = orient_outward(self.normal3d, self.center, body_center)
        # §5.3.6 / Abb.44 – volume centroid via convex hull
        self.v_v = convex_hull_volume_centroid(vertices[idx])
        return self.normal3d

    def robotic_validation(self, cfg=None, max_skew_deg=75.0):
        """Decide whether this point is safe for the robot to act on.

        Combines several robot-readiness checks and records the first failing
        one in `invalid_reason`:
          * geometry sanity (aspect ratio / skew),
          * normal stability (reliable approach vector),
          * minimum surface area (reject sub-noise specks),
          * minimum insertion depth for terminal/cable-entry classes, and
          * detector confidence (when a config is supplied).
        Also stores the configured standoff distance so `standoff_point` is
        meaningful. Sets and returns `is_valid_for_robot`.
        """
        area_min = getattr(cfg, "min_feature_area_mm2", MIN_FEATURE_AREA_MM2)
        depth_min = getattr(cfg, "min_insertion_depth_mm", 0.0) if cfg is not None else 0.0
        conf_min = getattr(cfg, "required_confidence_score", 0.0) if cfg is not None else 0.0
        self.approach_distance_mm = getattr(cfg, "robotic_approach_distance_mm", 0.0) if cfg is not None else 0.0

        pts = np.vstack([f.points for f in self.fragments])
        geom_ok, _info = validate_feature_geometry(pts, max_skew_deg=max_skew_deg)
        # insertion depth only constrains classes the robot actually inserts into.
        # Both depth measures are always reported; cfg.insertion_depth_metric
        # selects which one gates validity ('thesis' = |v_o - v_s| (default),
        # 'axis' = physical travel span along the insertion axis).
        metric = getattr(cfg, "insertion_depth_metric", "thesis") if cfg is not None else "thesis"
        depth_val = self.insertion_depth_axis_mm if metric == "axis" else self.insertion_depth_mm
        depth_ok = (self.label not in TERMINAL_TYPES) or (depth_val >= depth_min)

        reason = None
        if not geom_ok:
            reason = "bad_geometry"
        elif self.normal_stability < MIN_NORMAL_STABILITY_SCORE:
            reason = "unstable_normal"
        elif self.size < area_min:
            reason = "below_min_area"
        elif not depth_ok:
            reason = "below_min_insertion_depth"
        elif self.confidence_score < conf_min:
            reason = "below_confidence_threshold"

        self.invalid_reason = reason
        self.is_valid_for_robot = reason is None
        return self.is_valid_for_robot


# Backward-compatible alias (previously named FeatureInstance).
FeatureInstance = ConnectionPoint


# §5.3.5 / Abb.43 – graph-connectivity segmentation with min_vertices threshold
# STEP 1: split mesh into contiguous regions of uniform label.
# edges between two different labels are ignored so that e.g. a snap point
# bordering housing is not merged with the housing.
def build_fragments(vertices, faces, labels, min_vertices=20, skip_label=HOUSING):
    vertices = np.asarray(vertices, dtype=float)
    faces = np.asarray(faces, dtype=int)
    labels = np.asarray(labels, dtype=int)

    vn = vertex_normals(vertices, faces)
    fa = face_areas(vertices, faces)

    uf = UnionFind(len(vertices))
    for a, b, c in faces:
        for u, v in ((a, b), (b, c), (a, c)):
            if labels[u] == labels[v]:
                uf.union(u, v)

    # sum area per contiguous region
    area_per_root = {}
    for fi, (a, b, c) in enumerate(faces):
        ra, rb, rc = uf.find(a), uf.find(b), uf.find(c)
        if ra == rb == rc:
            area_per_root[ra] = area_per_root.get(ra, 0.0) + fa[fi]

    groups = {}
    for i in range(len(vertices)):
        groups.setdefault(uf.find(i), []).append(i)

    fragments = []
    skipped_noise = 0
    for root, idx in groups.items():
        lab = int(labels[idx[0]])
        if lab == skip_label:
            continue  # housing is not of interest
        if len(idx) < min_vertices:
            skipped_noise += 1
            continue  # too small, probably noise
        pts = vertices[idx]
        nrm = vn[idx].mean(axis=0)
        nl = np.linalg.norm(nrm)
        nrm = nrm / nl if nl > VERTEX_NORMAL_EPSILON else np.array([0.0, 0.0, 1.0])
        area = area_per_root.get(root, 0.0)
        fragments.append(Fragment(idx, pts, lab, nrm, area))

    if skipped_noise:
        logger.debug("build_fragments: %d regions discarded as noise (< %d vertices)",
                     skipped_noise, min_vertices)
    logger.debug("build_fragments: %d fragments found", len(fragments))
    return fragments


# STEP 2: merge fragments of the same label when they are close enough
# and point in similar directions.
# The gap threshold is relative to the bounding-box diagonal so the same
# value works for large and small parts.
def connect_fragments(fragments, vertices, gap_frac=0.05, angle_max_deg=30.0):
    lo = vertices.min(axis=0)
    hi = vertices.max(axis=0)
    diag = float(np.linalg.norm(hi - lo))
    max_gap = gap_frac * diag
    cos_min = np.cos(np.deg2rad(angle_max_deg))

    uf = UnionFind(len(fragments))
    merges = 0
    for i in range(len(fragments)):
        for j in range(i + 1, len(fragments)):
            fi, fj = fragments[i], fragments[j]
            if fi.label != fj.label:
                continue
            d = np.linalg.norm(fi.center - fj.center)
            gap = d - (fi.radius + fj.radius)  # approx surface to surface
            if gap > max_gap:
                continue
            if float(np.dot(fi.normal, fj.normal)) < cos_min:
                continue  # normals too different -> do not merge
            uf.union(i, j)
            merges += 1

    logger.debug("connect_fragments: %d fragment merges", merges)

    buckets = {}
    for i, frag in enumerate(fragments):
        buckets.setdefault(uf.find(i), []).append(frag)

    instances = [FeatureInstance(frags) for frags in buckets.values()]
    logger.debug("connect_fragments: %d fragments -> %d instances", len(fragments), len(instances))
    return instances, uf


# every contact gets a pointer to the nearest label surface.
# used later in the digital twin to know which label belongs to which contact.
def link_features(instances, source_label=CONTACT, target_label=LABEL_SURFACE):
    targets = [x for x in instances if x.label == target_label]
    if not targets:
        logger.debug("link_features: no label surface found, skipping")
        return
    for inst in instances:
        if inst.label != source_label:
            continue
        dists = [np.linalg.norm(inst.center - t.center) for t in targets]
        inst.points_to.append(targets[int(np.argmin(dists))])
    logger.debug("link_features: %d contacts linked",
                 sum(1 for x in instances if x.label == source_label))


# a single wiring candidate from contact A to contact B
class WireLink:
    def __init__(self, src, dst, distance, score):
        self.src = src
        self.dst = dst
        self.distance = distance
        self.score = score  # lower is better


# compute geometric wiring candidates between contacts.
#
# IMPORTANT: this is NOT electrical wiring. we only provide candidates based on
# geometry (close + openings facing each other).
# which contact actually connects to which is decided by the schematic/netlist,
# which is external and must be matched against these candidates.
def build_wiring(instances, k=3, see_tol=-0.1, cfg=None):
    contacts = [x for x in instances if x.label == CONTACT]
    if cfg is not None:
        k = getattr(cfg, "k_wires", k)
        see_tol = getattr(cfg, "see_tol", see_tol)
        # only wire points the robot is actually allowed to act on
        n_before = len(contacts)
        contacts = [c for c in contacts if getattr(c, "is_valid_for_robot", True)]
        n_dropped = n_before - len(contacts)
        if n_dropped:
            logger.info("build_wiring: %d of %d contacts excluded as not robot-valid "
                        "(check required_confidence_score / stability / area / depth)",
                        n_dropped, n_before)
    total_candidates = 0
    for a in contacts:
        cands = []
        for b in contacts:
            if b is a:
                continue
            seg = b.center - a.center
            L = float(np.linalg.norm(seg))
            if L < 1e-9:
                continue
            u = seg / L
            sees_a = float(np.dot(u, a.normal3d))   # does A's opening face B?
            sees_b = float(np.dot(-u, b.normal3d))  # does B's opening face A?
            if sees_a < see_tol or sees_b < see_tol:
                continue  # one opening faces away -> no viable candidate
            score = L * (2.0 - sees_a - sees_b + 0.01)
            cands.append(WireLink(a, b, L, score))
        cands.sort(key=lambda w: w.score)
        a.wires = cands[:k]
        total_candidates += len(a.wires)
    logger.debug("build_wiring: %d contacts, %d candidates total", len(contacts), total_candidates)


# build the final connector graph as a dict, ready to serialize
def build_connector_graph(instances):
    id_of = {id(inst): i for i, inst in enumerate(instances)}

    def vec(v):
        return None if v is None else [round(float(x), 5) for x in v]

    nodes = []
    links = []
    for i, inst in enumerate(instances):
        entry = inst.v_o if inst.v_o is not None else inst.center
        nodes.append({
            "id": i,
            "label": LABEL_NAMES[inst.label],
            "robot_class": ROBOTIC_LABEL_NAMES[inst.label],
            "position": vec(inst.center),    # §5.3.6 / Abb.44 – cluster centroid (mean vertices)
            "volume_center": vec(inst.v_v),  # §5.3.6 / Abb.44 – volume centroid (convex hull)
            "opening": vec(inst.v_o),        # §5.3.6 / Abb.44 – opening midpoint (boundary vertices)
            "surface": vec(inst.v_s),        # §5.3.6 / Abb.44 – surface centroid via boundary peeling
            "normal": vec(inst.normal3d),
            "size": round(float(inst.size), 6),
            "n_vertices": inst.n_vertices,
            # ---- robot-ready fields ----
            "entry_point": vec(entry),                          # world coords the tool aims for
            "approach_vector": vec(inst.normal3d),              # outward: where the tool comes FROM
            "insertion_axis": vec(getattr(inst, "insertion_axis", -inst.normal3d)),  # inward travel direction
            "standoff_point": vec(getattr(inst, "standoff_point", entry)),  # pre-insertion pose
            "approach_distance_mm": round(float(getattr(inst, "approach_distance_mm", 0.0)), 5),
            "insertion_depth_mm": round(float(getattr(inst, "insertion_depth_mm", 0.0)), 5),
            "insertion_depth_axis_mm": round(float(getattr(inst, "insertion_depth_axis_mm", 0.0)), 5),
            "confidence_score": round(float(getattr(inst, "confidence_score", 1.0)), 4),
            "normal_stability": round(float(getattr(inst, "normal_stability", 0.0)), 4),
            "is_valid_for_robot": bool(getattr(inst, "is_valid_for_robot", True)),
            "invalid_reason": getattr(inst, "invalid_reason", None),
            "terminal_type": getattr(inst, "terminal_type", None),
        })
        for tgt in inst.points_to:
            links.append({"from": i, "to": id_of[id(tgt)], "type": "feature_link"})
        for w in inst.wires:
            links.append({
                "from": i,
                "to": id_of[id(w.dst)],
                "type": "wire_candidate",  # candidate, not an electrical connection
                "distance": round(float(w.distance), 5),
            })

    return {"nodes": nodes, "links": links}


def save_connector_graph(instances, path):
    import json
    graph = build_connector_graph(instances)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(graph, fh, indent=2, ensure_ascii=False)
    logger.debug("Graph saved: %s", path)
    return graph


# §5.3.5 / Abb.43 – tool-clearance check: two connection points closer than the
# tool footprint can't both be serviced without collision.
def enforce_tool_clearance(instances, clearance_mm):
    """Flag connection points whose nearest neighbour is closer than the robot
    tool footprint. Already-invalid points are left untouched. No-op when
    clearance_mm <= 0."""
    if clearance_mm <= 0.0 or len(instances) < 2:
        return
    n_flagged = 0
    for a in instances:
        if not a.is_valid_for_robot:
            continue
        nearest = min((float(np.linalg.norm(a.center - b.center))
                       for b in instances if b is not a), default=float("inf"))
        if nearest < clearance_mm:
            a.is_valid_for_robot = False
            a.invalid_reason = "tool_clearance_conflict"
            n_flagged += 1
    if n_flagged:
        logger.info("enforce_tool_clearance: %d points flagged (neighbour < %.2f mm)",
                    n_flagged, clearance_mm)


# The robot-validity thresholds are in millimetres but the mesh is never
# rescaled, so they only mean something if the coordinates are already in mm.
# This catches the common unit mistake (normalized/unit-scale CAD, metres, ...)
# before is_valid_for_robot is computed from values that would be meaningless.
def check_mesh_scale(vertices, cfg):
    """Warn when the mesh extent is inconsistent with the active mm-scale checks.

    Returns the bounding-box diagonal (in mesh units), or None when the mm-scale
    thresholds are all disabled (the unit-scale demos set them to 0) or no cfg is
    given, in which case the check is skipped.
    """
    if cfg is None:
        return None
    mm_active = any(getattr(cfg, k, 0.0) > 0.0 for k in
                    ("min_feature_area_mm2", "robot_tool_clearance_mm",
                     "min_insertion_depth_mm"))
    if not mm_active:
        return None
    vertices = np.asarray(vertices, dtype=float)
    diag = float(np.linalg.norm(vertices.max(axis=0) - vertices.min(axis=0)))
    if diag < MIN_PLAUSIBLE_PART_MM or diag > MAX_PLAUSIBLE_PART_MM:
        logger.warning(
            "mesh bounding-box diagonal is %.3g, outside the plausible "
            "[%.0f, %.0f] mm range for a connector part. The mm-scale robot "
            "thresholds (min_feature_area_mm2=%.3g, robot_tool_clearance_mm=%.3g, "
            "min_insertion_depth_mm=%.3g) assume millimetre coordinates, so "
            "is_valid_for_robot / insertion_depth_mm may be meaningless. Rescale "
            "the mesh to mm, or disable the mm checks (set those three to 0, e.g. "
            "ConnectionPointDetectorConfig.for_demo()).",
            diag, MIN_PLAUSIBLE_PART_MM, MAX_PLAUSIBLE_PART_MM,
            getattr(cfg, "min_feature_area_mm2", 0.0),
            getattr(cfg, "robot_tool_clearance_mm", 0.0),
            getattr(cfg, "min_insertion_depth_mm", 0.0))
    return diag


# all in one pass: optional boundary peeling, build fragments, merge them,
# compute directions, link features, set wiring candidates
def run(vertices, faces, labels, min_vertices=20, gap_frac=0.05, angle_max_deg=30.0,
        surface_subdiv=0, cfg=None, peel_iterations=None):
    vertices = np.asarray(vertices, dtype=float)
    faces = np.asarray(faces, dtype=int)
    labels = np.asarray(labels, dtype=int)
    body_center = vertices.mean(axis=0)
    check_mesh_scale(vertices, cfg)  # warn early if mm thresholds vs. mesh units mismatch

    # §5.3.5 / Abb.43 – optional boundary peeling to separate touching features
    # before fragment detection (off by default to preserve raw segmentation).
    if peel_iterations is None:
        peel_iterations = getattr(cfg, "boundary_peel_iterations", 0) if cfg is not None else 0
    if peel_iterations and peel_iterations > 0:
        labels = remove_boundary_layer(faces, labels, bg=HOUSING, iterations=peel_iterations)
        logger.debug("run: peeled %d boundary layer(s)", peel_iterations)

    frags = build_fragments(vertices, faces, labels, min_vertices=min_vertices)
    instances, uf = connect_fragments(frags, vertices, gap_frac, angle_max_deg)
    max_skew = getattr(cfg, "max_feature_skew_deg", 75.0) if cfg is not None else 75.0
    for inst in instances:
        inst.compute_direction(vertices, faces, smooth_subdiv=surface_subdiv,
                               body_center=body_center)
        inst.robotic_validation(cfg=cfg, max_skew_deg=max_skew)
    clearance = getattr(cfg, "robot_tool_clearance_mm", 0.0) if cfg is not None else 0.0
    enforce_tool_clearance(instances, clearance)
    link_features(instances)
    build_wiring(instances, cfg=cfg)
    return instances, frags


# quick self-test when run directly
def _demo():
    r, s = 40, 1.0
    v = np.array([(x, y, 0.0) for y in np.linspace(0, s, r) for x in np.linspace(0, s, r)], dtype=float)
    a = np.arange(r*r).reshape(r, r)
    f = np.column_stack([a[:-1,:-1].ravel(), a[:-1,1:].ravel(), a[1:,1:].ravel(),
                          a[:-1,:-1].ravel(), a[1:,1:].ravel(), a[1:,:-1].ravel()]).reshape(-1, 3)
    x, y = v[:,0], v[:,1]; L = np.zeros(len(v), dtype=int)
    L[(x<0.25)&(y>0.72)] = 1; L[(x>0.40)&(x<0.60)&(y>0.42)&(y<0.58)] = 4
    L[(y<0.14)&(x>0.30)&(x<0.47)] = 2; L[(y<0.14)&(x>0.53)&(x<0.70)] = 2
    inst, frags = run(v, f, L, 10)
    print(f"[demo] {len(v)}v {len(f)}f -> {len(inst)} features")
    for i in inst: print(f"  {LABEL_NAMES[i.label]:20s} ctr={np.round(i.center,2)}  frags={len(i.fragments)}")


if __name__ == "__main__":
    _demo()
