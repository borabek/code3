# mesh remeshing via external tools (Meshlabserver, PyMeshLab)

import shutil
import logging
import subprocess

import numpy as np

logger = logging.getLogger(__name__)

# helper: unique edges + midpoint index per edge

def _edge_midpoints(faces, n_vertices):
    F = np.asarray(faces, dtype=int)
    edges = np.vstack([F[:, [0, 1]], F[:, [1, 2]], F[:, [0, 2]]])
    edges = np.sort(edges, axis=1)
    uniq, inv = np.unique(edges, axis=0, return_inverse=True)
    # midpoint vertex index per edge (starts after the original vertices)
    mid_index = np.arange(len(uniq)) + n_vertices
    # inv is (3*F,) -> per triangle the 3 edges in order ab,bc,ca
    inv = inv.reshape(3, len(F)).T
    return uniq, mid_index, inv

# simple subdivision: each triangle split into 4

def subdivide_simple(vertices, faces, labels=None):
    """Split each triangle into 4 (edge midpoints)."""
    V = np.asarray(vertices, dtype=float)
    F = np.asarray(faces, dtype=int)
    uniq, _mid_index, inv = _edge_midpoints(F, len(V))

    mids = 0.5 * (V[uniq[:, 0]] + V[uniq[:, 1]])
    V2 = np.vstack([V, mids])

    ab, bc, ca = inv[:, 0] + len(V), inv[:, 1] + len(V), inv[:, 2] + len(V)
    a, b, c = F[:, 0], F[:, 1], F[:, 2]
    F2 = np.vstack([
        np.stack([a, ab, ca], axis=1),
        np.stack([ab, b, bc], axis=1),
        np.stack([ca, bc, c], axis=1),
        np.stack([ab, bc, ca], axis=1),
    ]).astype(int)

    if labels is None:
        return V2, F2

    L = np.asarray(labels, dtype=int)
    la, lb = L[uniq[:, 0]], L[uniq[:, 1]]
    mid_lab = np.where(la == lb, la, 0)
    L2 = np.concatenate([L, mid_lab])
    return V2, F2, L2

# smooth (Loop) subdivision

def subdivide_loop(vertices, faces, labels=None):
    """Loop subdivision with smoothing. Labels propagated like subdivide_simple."""
    V = np.asarray(vertices, dtype=float)
    F = np.asarray(faces, dtype=int)
    n = len(V)
    uniq, _mid_index, inv = _edge_midpoints(F, n)

    # neighbor list for smoothing the old ("even") vertices
    neighbors = [set() for _ in range(n)]
    for a, b, c in F:
        neighbors[a].update((b, c))
        neighbors[b].update((a, c))
        neighbors[c].update((a, b))

    # detect boundary edges (appear in only one triangle)
    cnt = {}
    for tri in F:
        for e in ((tri[0], tri[1]), (tri[1], tri[2]), (tri[0], tri[2])):
            k = (min(e), max(e))
            cnt[k] = cnt.get(k, 0) + 1
    is_boundary = np.zeros(n, dtype=bool)
    for (u, v), c in cnt.items():
        if c == 1:
            is_boundary[u] = is_boundary[v] = True

    # new edge points: 3/8 endpoints + 1/8 opposite; on boundary use midpoint
    opp = {i: [] for i in range(len(uniq))}
    edge_lookup = {(min(uniq[i]), max(uniq[i])): i for i in range(len(uniq))}
    for a, b, c in F:
        for (u, v, w) in ((a, b, c), (b, c, a), (a, c, b)):
            ei = edge_lookup[(min(u, v), max(u, v))]
            opp[ei].append(w)

    mids = np.zeros((len(uniq), 3), dtype=float)
    for i, (u, v) in enumerate(uniq):
        if cnt[(min(u, v), max(u, v))] == 1 or len(opp[i]) < 2:
            mids[i] = 0.5 * (V[u] + V[v])                 # boundary edge
        else:
            w0, w1 = opp[i][0], opp[i][1]
            mids[i] = (3.0 / 8.0) * (V[u] + V[v]) + (1.0 / 8.0) * (V[w0] + V[w1])

    # smooth old vertices (Warren weights)
    V_even = V.copy()
    for i in range(n):
        nb = list(neighbors[i])
        k = len(nb)
        if is_boundary[i] or k == 0:
            continue  # boundary points not moved (simplified)
        beta = (5.0 / 8.0 - (3.0 / 8.0 + 0.25 * np.cos(2 * np.pi / k)) ** 2) / k
        V_even[i] = (1 - k * beta) * V[i] + beta * V[nb].sum(axis=0)

    V2 = np.vstack([V_even, mids])

    ab, bc, ca = inv[:, 0] + n, inv[:, 1] + n, inv[:, 2] + n
    a, b, c = F[:, 0], F[:, 1], F[:, 2]
    F2 = np.vstack([
        np.stack([a, ab, ca], axis=1),
        np.stack([ab, b, bc], axis=1),
        np.stack([ca, bc, c], axis=1),
        np.stack([ab, bc, ca], axis=1),
    ]).astype(int)

    if labels is None:
        return V2, F2
    L = np.asarray(labels, dtype=int)
    la, lb = L[uniq[:, 0]], L[uniq[:, 1]]
    L2 = np.concatenate([L, np.where(la == lb, la, 0)])
    return V2, F2, L2

# downsampling via vertex clustering (voxel grid)

def decimate_vertex_cluster(vertices, faces, target_vertices, labels=None):
    """Decimate to ~target_vertices via voxel clustering + majority vote labels."""
    V = np.asarray(vertices, dtype=float)
    F = np.asarray(faces, dtype=int)
    if len(V) <= target_vertices:
        return (V, F, labels) if labels is not None else (V, F)

    lo, hi = V.min(axis=0), V.max(axis=0)
    diag = float(np.linalg.norm(hi - lo)) or 1.0

    def cluster(cell):
        keys = np.floor((V - lo) / cell).astype(np.int64)
        _, inv = np.unique(keys, axis=0, return_inverse=True)
        return inv

    # binary search on cell size: smaller cell = more clusters
    cell_lo, cell_hi = diag * 1e-4, diag
    inv = cluster(diag / 64.0)
    for _ in range(24):
        cell = 0.5 * (cell_lo + cell_hi)
        inv = cluster(cell)
        n_clusters = inv.max() + 1
        if n_clusters > target_vertices:
            cell_lo = cell        # too many -> coarser cells
        else:
            cell_hi = cell        # too few  -> finer cells
        if abs(int(n_clusters) - target_vertices) <= max(50, target_vertices // 100):
            break

    n_clusters = inv.max() + 1
    # cluster representative = mean of vertices in cluster
    V2 = np.zeros((n_clusters, 3), dtype=float)
    counts = np.zeros(n_clusters, dtype=int)
    np.add.at(V2, inv, V)
    np.add.at(counts, inv, 1)
    V2 /= counts[:, None]

    F2 = inv[F]
    # remove degenerate triangles (two corners in the same cluster)
    good = (F2[:, 0] != F2[:, 1]) & (F2[:, 1] != F2[:, 2]) & (F2[:, 0] != F2[:, 2])
    F2 = F2[good]

    if labels is None:
        return V2, F2

    L = np.asarray(labels, dtype=int)
    n_classes = int(L.max(initial=0)) + 1
    votes = np.zeros((n_clusters, n_classes), dtype=int)
    np.add.at(votes, (inv, L), 1)
    L2 = votes.argmax(axis=1)
    return V2, F2, L2

# label upscaling via nearest neighbor

def upscale_labels_nn(vertices_new, vertices_old, labels_old):
    """Assign each new vertex the label of the nearest old vertex.

    Uses scipy cKDTree if available, otherwise falls back to numpy
    (O(n*m), only reasonable for small meshes).
    """
    Vn = np.asarray(vertices_new, dtype=float)
    Vo = np.asarray(vertices_old, dtype=float)
    Lo = np.asarray(labels_old, dtype=int)
    try:
        from scipy.spatial import cKDTree
        _, idx = cKDTree(Vo).query(Vn)
        return Lo[idx]
    except ImportError:
        out = np.empty(len(Vn), dtype=int)
        for i, p in enumerate(Vn):
            out[i] = Lo[int(np.argmin(((Vo - p) ** 2).sum(axis=1)))]
        return out

# scale mesh to a uniform vertex count

def scale_to_target(vertices, faces, labels=None, target_vertices=6000,
                    smooth=False, max_subdiv=4):
    """Bring mesh to approximately target_vertices.

    Subdivides until above the target if needed, then decimates down.
    smooth=True uses Loop subdivision instead of simple 1-to-4 split.
    """
    V = np.asarray(vertices, dtype=float)
    F = np.asarray(faces, dtype=int)
    L = None if labels is None else np.asarray(labels, dtype=int)
    subdiv = subdivide_loop if smooth else subdivide_simple

    it = 0
    while len(V) < target_vertices and it < max_subdiv:
        if L is None:
            V, F = subdiv(V, F)
        else:
            V, F, L = subdiv(V, F, L)
        it += 1

    out = decimate_vertex_cluster(V, F, target_vertices, labels=L)
    logger.debug("scale_to_target: -> %d vertices (%d subdivision iterations)", len(out[0]), it)
    return out

# watertight 2-manifold via Manifold/ManifoldPlus (external binary)

def watertight_manifold(in_obj, out_obj, depth=8, plus=True):
    """Generate a watertight 2-manifold mesh (Huang et al.).

    Calls the external binary `manifold` or `manifoldplus` (from the repos
    hjwdzh/Manifold and hjwdzh/ManifoldPlus). If neither is in PATH, a clear
    error is raised -- this tessellation step is in C++ and not re-implemented
    in Python.
    """
    binary = "manifoldplus" if plus else "manifold"
    exe = shutil.which(binary)
    if exe is None:
        raise RuntimeError(
            f"'{binary}' not found in PATH. Watertight 2-manifolds require "
            "Manifold/ManifoldPlus (hjwdzh/Manifold, hjwdzh/ManifoldPlus) -- "
            "build and add to PATH. The rest of remesh.py works without it."
        )
    if plus:
        cmd = [exe, "--input", in_obj, "--output", out_obj, "--depth", str(depth)]
    else:
        cmd = [exe, in_obj, out_obj, str(depth)]
    logger.info("watertight_manifold: %s", " ".join(cmd))
    subprocess.run(cmd, check=True)
    return out_obj

# self-test

def _demo():
    # flat grid plate
    res = 20
    xs = np.linspace(0, 1, res)
    V = np.array([(x, y, 0.0) for y in xs for x in xs], dtype=float)
    F = []
    for iy in range(res - 1):
        for ix in range(res - 1):
            a = iy * res + ix
            b, c, d = a + 1, a + res, a + res + 1
            F.append((a, b, d)); F.append((a, d, c))
    F = np.array(F, dtype=int)
    L = np.zeros(len(V), dtype=int)
    L[(V[:, 0] > 0.4) & (V[:, 0] < 0.6) & (V[:, 1] > 0.4) & (V[:, 1] < 0.6)] = 1

    print(f"[start]  {len(V)} vertices, {len(F)} faces, {(L == 1).sum()} feature vertices")

    V1, F1, L1 = subdivide_simple(V, F, L)
    print(f"[simple] -> {len(V1)} vertices, {len(F1)} faces (expected 4x faces)")
    assert len(F1) == 4 * len(F)

    V2, F2, L2 = subdivide_loop(V, F, L)
    print(f"[loop]   -> {len(V2)} vertices, {len(F2)} faces")

    Vs, Fs, Ls = scale_to_target(V, F, L, target_vertices=300)
    print(f"[scale]  -> {len(Vs)} vertices (target 300), {(Ls == 1).sum()} feature vertices preserved")

    Lup = upscale_labels_nn(V1, V, L)
    print(f"[nn-up]  {(Lup == L1).mean() * 100:.0f}% of propagated labels match nn assignment")

    print("\n[ok] remesh self-test passed.")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
    _demo()
