# instances.py: connected components + k-Means for mesh instance separation (§5.3.5)
#
# Reference: Scheffler (2022), §5.3.5
#   §5.3.5 / Abb.43 – instance segmentation by graph connectivity (Union-Find),
#                      boundary peeling, min_vertices threshold
#   §5.3.5          – k-Means instance segmentation

import logging
from dataclasses import dataclass, field
import numpy as np
from connector_constants import HOUSING

logger = logging.getLogger(__name__)


@dataclass
class SegmentedRegion:
    """A connected, single-class region of the mesh (one instance of a feature class)."""
    label: int                      # Feature class label (0-4)
    vertex_index: np.ndarray        # Vertex indices belonging to this region
    center: np.ndarray = field(default=None)  # Centroid of region vertices


# Backward-compatible alias (previously named Instance).
Instance = SegmentedRegion


def _adj(faces, n):
    """Build vertex adjacency list from triangle faces."""
    nb = [set() for _ in range(n)]
    for a, b, c in np.asarray(faces, dtype=int):
        nb[a].update((b, c)); nb[b].update((a, c)); nb[c].update((a, b))
    return nb


# §5.3.5 / Abb.43 – boundary peeling: reclassify boundary vertices as background
def remove_boundary_layer(faces, labels, bg=HOUSING, iterations=1):
    """Remove the outermost boundary layer(s) of each feature (boundary peeling).

    Vertices not surrounded exclusively by same-label neighbors are reclassified
    as background. This separates touching features. `iterations` > 1 peels
    multiple layers for a stricter filter (thesis §5.3.5), which yields cleaner
    instance cores at the cost of shrinking small features.
    """
    L = np.asarray(labels, dtype=int).copy()
    nb = _adj(faces, len(L))
    for _ in range(max(1, int(iterations))):
        out = L.copy()
        changed = False
        for i in range(len(L)):
            if L[i] != bg and any(L[j] != L[i] for j in nb[i]):
                out[i] = bg
                changed = True
        L = out
        if not changed:
            break
    return L


# §5.3.5 / Abb.43 – graph connectivity (Union-Find) for instance segmentation
def connected_components(faces, labels, bg=HOUSING):
    """Find connected components (instances) of each label via union-find.

    Returns all connected components of non-background labels.
    """
    from connector3d import UnionFind
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
    """Extract instances via boundary peeling + connectivity + size filtering.

    Optionally peel boundary layer(s), find connected components,
    filter out small noise clusters, compute centroids. `peel_iterations`
    controls how many layers are peeled (stricter separation, §5.3.5).
    """
    V = np.asarray(vertices, dtype=float)
    work = (remove_boundary_layer(faces, labels, bg, iterations=peel_iterations)
            if peel else np.asarray(labels, dtype=int))
    comps = connected_components(faces, work, bg)
    kept = [c for c in comps if len(c.vertex_index) >= min_vertices]
    for inst in kept:
        inst.center = V[inst.vertex_index].mean(axis=0)
    logger.debug("instances_by_connectivity: %d comps, %d after filter (>=%d)",
                 len(comps), len(kept), min_vertices)
    return kept


def _kmeans(P, k, iters=50, seed=0):
    """Simple Lloyd's k-Means with k-Means++ initialization."""
    P = np.asarray(P, dtype=float)
    if k <= 1 or len(P) <= k:
        return np.zeros(len(P), dtype=int), P.mean(0, keepdims=True) if len(P) else P
    rng = np.random.default_rng(seed)
    C = [P[rng.integers(len(P))]]
    for _ in range(1, k):
        d2 = np.min([((P - c)**2).sum(1) for c in C], 0)
        C.append(P[rng.choice(len(P), p=d2/d2.sum() if d2.sum() > 0 else None)])
    C = np.array(C)
    lab = np.zeros(len(P), dtype=int)
    for _ in range(iters):
        new = ((P[:, None] - C[None])**2).sum(2).argmin(1)
        if np.array_equal(new, lab): break
        lab = new
        for j in range(k):
            if (lab == j).any():
                C[j] = P[lab == j].mean(0)
    return lab, C


# §5.3.5 – k-Means instance segmentation (partition YOLO-detected class into k instances)
def instances_by_kmeans(vertices, labels, target_class, k):
    """Partition vertices of a class into k instances via k-Means.

    Used to split predictions from external detectors (e.g., YOLO)
    where the detector specifies k instances per class.
    """
    V, L = np.asarray(vertices, dtype=float), np.asarray(labels, dtype=int)
    idx = np.where(L == target_class)[0]
    if not len(idx) or k < 1:
        return []
    km, _ = _kmeans(V[idx], k)
    return [Instance(int(target_class), idx[km == j], V[idx[km == j]].mean(0))
            for j in range(int(km.max()) + 1) if (km == j).any()]


def _demo():
    r = 30; xs = np.linspace(0, 1, r)
    V = np.array([(x, y, 0.) for y in xs for x in xs], dtype=float)
    a = np.arange(r*r).reshape(r, r)
    tl = a[:-1,:-1].ravel(); tr = a[:-1,1:].ravel(); bl = a[1:,:-1].ravel(); br = a[1:,1:].ravel()
    F = np.column_stack([tl,tr,bl,bl,tr,br]).reshape(-1, 3)
    x, y = V[:,0], V[:,1]; L = np.zeros(len(V), dtype=int)
    L[(x<.42)&(y>.3)&(y<.7)] = 1; L[(x>.58)&(y>.3)&(y<.7)] = 1; L[(y>.47)&(y<.53)] = 1
    raw = connected_components(F, L)
    print(f"[raw] {len([c for c in raw if c.label==1])} class-1 components")
    inst = instances_by_connectivity(V, F, L, min_vertices=10, peel=True)
    n1 = [i for i in inst if i.label == 1]
    print(f"[peel] {len(n1)} instances")
    for i in n1:
        print(f"  center={np.round(i.center,2)} ({len(i.vertex_index)} v)")
    km = instances_by_kmeans(V, L, 1, 2)
    print(f"[kmeans] k=2 -> {len(km)} instances")
    print("[OK]")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
    _demo()
