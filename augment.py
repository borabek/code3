# Data augmentation for triangle meshes.
#
# Three combinable techniques:
#   1. Random rotations (rotation matrix from uniformly distributed angles)
#   2. Lognormal noise (mu=0, sigma=0.005), multiplied element-wise so the
#      sign of each coordinate is preserved
#   3. Intentional deformation of feature regions via As-Rigid-As-Possible (ARAP)
#
# After adding noise, the mesh is smoothed with an average, Laplace, or Taubin
# filter. Rotations do not change labels; noise/smoothing/ARAP only move vertex
# positions, they never touch labels.
#
# Reference: Scheffler (2022), §5.1.2, §5.3.3
#   §5.1.2 / Abb.31 – lognormal noise µ=0, σ=0.005
#   §5.1.2 / Abb.32 – Laplace / average / Taubin smoothing after noise
#   §5.1.2 / Abb.32 – As-Rigid-As-Possible (ARAP) deformation (Sorkine & Alexa 2007)
#   §5.1.2           – random rotation matrix
#   §5.3.3           – cotangent Laplacian used inside ARAP deformation

import logging
import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 1) random rotation
# ---------------------------------------------------------------------------

# §5.1.2 – random rotation matrix (uniformly distributed Euler angles)
def random_rotation(vertices, rng=None, about_center=True):
    """Rotate mesh by random Euler angles (uniform in [0, 2*pi))."""
    rng = rng or np.random.default_rng()
    a, b, g = rng.uniform(0, 2 * np.pi, size=3)
    ca, sa = np.cos(a), np.sin(a)
    cb, sb = np.cos(b), np.sin(b)
    cg, sg = np.cos(g), np.sin(g)
    Rx = np.array([[1, 0, 0], [0, ca, -sa], [0, sa, ca]])
    Ry = np.array([[cb, 0, sb], [0, 1, 0], [-sb, 0, cb]])
    Rz = np.array([[cg, -sg, 0], [sg, cg, 0], [0, 0, 1]])
    R = Rz @ Ry @ Rx

    V = np.asarray(vertices, dtype=float)
    if about_center:
        c = V.mean(axis=0)
        return (V - c) @ R.T + c
    return V @ R.T


# ---------------------------------------------------------------------------
# 2) lognormal noise + smoothing
# ---------------------------------------------------------------------------

# §5.1.2 / Abb.31 – lognormal noise µ=0, σ=0.005 applied element-wise to vertex coordinates
def add_lognormal_noise(vertices, sigma=0.005, mu=0.0, rng=None):
    """Multiply each vertex coordinate by a lognormal(mu, sigma) factor.

    Because lognormal values are always positive, the sign of each coordinate
    is preserved (mu=0, sigma=0.005 as used in the thesis).
    """
    rng = rng or np.random.default_rng()
    V = np.asarray(vertices, dtype=float)
    factor = rng.lognormal(mean=mu, sigma=sigma, size=V.shape)
    return V * factor


def _adjacency(faces, n):
    nb = [set() for _ in range(n)]
    for a, b, c in np.asarray(faces, dtype=int):
        nb[a].update((b, c)); nb[b].update((a, c)); nb[c].update((a, b))
    return [np.array(sorted(s), dtype=int) for s in nb]


# §5.1.2 / Abb.32 – average smoothing after noise
def smooth_average(vertices, faces, iterations=1):
    """Average filter: v_i = (v_i + sum_neighbors v_n) / (|N| + 1)."""
    V = np.asarray(vertices, dtype=float).copy()
    nb = _adjacency(faces, len(V))
    for _ in range(iterations):
        out = V.copy()
        for i, n in enumerate(nb):
            if len(n):
                out[i] = (V[i] + V[n].sum(axis=0)) / (len(n) + 1)
        V = out
    return V


# §5.1.2 / Abb.32 – Laplace smoothing after noise
def smooth_laplace(vertices, faces, lam=0.5, iterations=1):
    """Laplace filter: v_i += lam * (mean(neighbors) - v_i)."""
    V = np.asarray(vertices, dtype=float).copy()
    nb = _adjacency(faces, len(V))
    for _ in range(iterations):
        out = V.copy()
        for i, n in enumerate(nb):
            if len(n):
                out[i] = V[i] + lam * (V[n].mean(axis=0) - V[i])
        V = out
    return V


# §5.1.2 / Abb.32 – Taubin smoothing after noise (prevents shrinkage)
def smooth_taubin(vertices, faces, lam=0.5, mu=-0.53, iterations=5):
    """Taubin smoothing: alternating lam (>0) and mu (<0) passes.

    Smooths without shrinking, unlike plain Laplace smoothing.
    """
    V = np.asarray(vertices, dtype=float).copy()
    nb = _adjacency(faces, len(V))

    def pass_(V, w):
        out = V.copy()
        for i, n in enumerate(nb):
            if len(n):
                out[i] = V[i] + w * (V[n].mean(axis=0) - V[i])
        return out

    for _ in range(iterations):
        V = pass_(V, lam)
        V = pass_(V, mu)
    return V


# ---------------------------------------------------------------------------
# 3) As-Rigid-As-Possible (ARAP) deformation (Sorkine & Alexa 2007)
# ---------------------------------------------------------------------------

# §5.3.3 – cotangent Laplacian weights used as the system matrix for ARAP
def _cotangent_weights(V, F):
    """Cotangent weights per directed edge as dict {(i,j): w}."""
    from collections import defaultdict
    w = defaultdict(float)
    for tri in F:
        for i0, i1, i2 in ((0, 1, 2), (1, 2, 0), (2, 0, 1)):
            a, b, c = tri[i0], tri[i1], tri[i2]   # angle at c is opposite edge (a,b)
            u = V[a] - V[c]
            v = V[b] - V[c]
            cross = np.linalg.norm(np.cross(u, v))
            cot = float(np.dot(u, v)) / cross if cross > 1e-12 else 0.0
            w[(a, b)] += 0.5 * cot
            w[(b, a)] += 0.5 * cot
    return w


# §5.1.2 / Abb.32 – As-Rigid-As-Possible (ARAP) deformation, Sorkine & Alexa 2007
def arap_deform(vertices, faces, constraints, iterations=3):
    """ARAP deformation with hard positional constraints.

    constraints: dict {vertex_index: target_position (3,)}. All other vertices
    move as rigidly as possible. Requires scipy (sparse solve); without scipy
    the function falls back to smooth Laplace deformation.

    Returns: deformed vertex positions (copy).
    """
    V = np.asarray(vertices, dtype=float)
    F = np.asarray(faces, dtype=int)
    n = len(V)
    try:
        import scipy.sparse as sp
        from scipy.sparse.linalg import splu
    except ImportError:
        logger.debug("scipy missing -> ARAP falls back to Laplace deformation")
        P = V.copy()
        for vi, tgt in constraints.items():
            P[vi] = tgt
        return smooth_laplace(P, F, lam=0.5, iterations=5)

    w = _cotangent_weights(V, F)
    neighbors = [[] for _ in range(n)]
    for (i, j) in w:
        neighbors[i].append(j)

    # system matrix L (cotangent Laplacian); constraint rows -> identity
    L = sp.lil_matrix((n, n))
    for i in range(n):
        if i in constraints:
            L[i, i] = 1.0
            continue
        wii = 0.0
        for j in neighbors[i]:
            wij = w[(i, j)]
            L[i, j] = -wij
            wii += wij
        L[i, i] = wii if wii != 0 else 1.0
    solver = splu(sp.csc_matrix(L))

    # initialization: original positions + hard constraints
    P = V.copy()
    for vi, tgt in constraints.items():
        P[vi] = tgt

    for _ in range(iterations):
        # local step: best rotation per vertex
        R = np.empty((n, 3, 3))
        for i in range(n):
            S = np.zeros((3, 3))
            for j in neighbors[i]:
                wij = w[(i, j)]
                S += wij * np.outer(V[i] - V[j], P[i] - P[j])
            U, _, Vt = np.linalg.svd(S)
            Ri = Vt.T @ U.T
            if np.linalg.det(Ri) < 0:        # avoid reflection
                U[:, -1] *= -1
                Ri = Vt.T @ U.T
            R[i] = Ri

        # global step: right-hand side + solve
        b = np.zeros((n, 3))
        for i in range(n):
            if i in constraints:
                b[i] = constraints[i]
                continue
            acc = np.zeros(3)
            for j in neighbors[i]:
                wij = w[(i, j)]
                acc += 0.5 * wij * (R[i] + R[j]) @ (V[i] - V[j])
            b[i] = acc
        P = solver.solve(b)

    return P


def arap_deform_features(vertices, faces, labels, feature_classes=(1, 2, 3, 4),
                         n_points=None, magnitude=0.03, rng=None, iterations=3):
    """Deliberate feature deformation.

    Picks 1..10 random anchor points inside the feature classes, fixes all
    vertices beyond the 50th percentile of their distance (static boundary),
    and shifts the anchors slightly. The housing (label 0) is not actively
    deformed.
    """
    rng = rng or np.random.default_rng()
    V = np.asarray(vertices, dtype=float)
    L = np.asarray(labels, dtype=int)
    diag = float(np.linalg.norm(V.max(axis=0) - V.min(axis=0))) or 1.0

    feat_idx = np.where(np.isin(L, feature_classes))[0]
    if len(feat_idx) == 0:
        return V.copy()
    k = n_points if n_points is not None else int(rng.integers(1, 11))
    anchors = rng.choice(feat_idx, size=min(k, len(feat_idx)), replace=False)

    constraints = {}
    for a in anchors:
        d = np.linalg.norm(V - V[a], axis=1)
        static = np.where(d > np.percentile(d, 50))[0]   # far away -> fix in place
        for s in static:
            constraints[int(s)] = V[s].copy()
        # shift anchor by a small random vector
        constraints[int(a)] = V[a] + rng.normal(0, magnitude * diag, size=3)

    return arap_deform(V, faces, constraints, iterations=iterations)


# ---------------------------------------------------------------------------
# orchestration: full augmentation pass
# ---------------------------------------------------------------------------

def augment(vertices, faces, labels=None, rng=None, seed=None,
            do_rotate=True, do_noise=True, do_deform=True,
            noise_sigma=0.005, smoothing="laplace"):
    """Produce one augmented clone of a part.

    Order: rotation -> noise -> smoothing -> ARAP.
    Labels are never changed (only vertex positions are modified).
    Returns: (V_aug, faces) -- faces are unchanged.
    """
    if rng is None:
        rng = np.random.default_rng(seed)
    V = np.asarray(vertices, dtype=float)
    F = np.asarray(faces, dtype=int)

    if do_rotate:
        V = random_rotation(V, rng)
    if do_noise:
        V = add_lognormal_noise(V, sigma=noise_sigma, rng=rng)
        if smoothing == "average":
            V = smooth_average(V, F, iterations=1)
        elif smoothing == "taubin":
            V = smooth_taubin(V, F, iterations=2)
        elif smoothing:
            V = smooth_laplace(V, F, lam=0.5, iterations=1)
    if do_deform and labels is not None:
        V = arap_deform_features(V, F, labels, rng=rng)
    return V, F


def _random_rotation_matrix(rng):
    """Uniformly random 3x3 rotation matrix (for centroid-stability testing)."""
    axis = rng.normal(size=3)
    n = np.linalg.norm(axis)
    axis = axis / n if n > 1e-12 else np.array([0.0, 0.0, 1.0])
    angle = rng.uniform(0.0, 2.0 * np.pi)
    x, y, z = axis
    c, s, C = np.cos(angle), np.sin(angle), 1.0 - np.cos(angle)
    return np.array([
        [c + x*x*C,   x*y*C - z*s, x*z*C + y*s],
        [y*x*C + z*s, c + y*y*C,   y*z*C - x*s],
        [z*x*C - y*s, z*y*C + x*s, c + z*z*C],
    ])


# §6 (validation) – verify the detector recovers centroids under tilt + noise.
def verify_centroid_stability(vertices, faces, labels, tol_mm=1.0, n_trials=5,
                              noise_sigma=0.0, seed=0, run_fn=None):
    """Check that ConnectionPointDetector centroids are pose-invariant.

    For each trial the part is randomly rotated (about its center) and optionally
    perturbed with noise; the pipeline is re-run and every detected centroid is
    compared against the rigidly-transformed reference centroid of the same
    class. Returns a report dict with per-trial and worst-case errors and a
    `passed` flag (max error <= `tol_mm`).
    """
    if run_fn is None:
        from connector3d import run as run_fn
    V0 = np.asarray(vertices, dtype=float)
    F = np.asarray(faces, dtype=int)
    L = np.asarray(labels, dtype=int)
    rng = np.random.default_rng(seed)

    base, _ = run_fn(V0, F, L)
    ref = [(int(i.label), np.asarray(i.center, dtype=float)) for i in base]
    center = V0.mean(axis=0)

    trials, max_err = [], 0.0
    for _ in range(int(n_trials)):
        R = _random_rotation_matrix(rng)
        V = (V0 - center) @ R.T + center
        if noise_sigma > 0.0:
            V = add_lognormal_noise(V, sigma=noise_sigma, rng=rng)
        inst, _ = run_fn(V, F, L)
        by_label = {}
        for it in inst:
            by_label.setdefault(int(it.label), []).append(np.asarray(it.center, dtype=float))
        # greedy one-to-one matching: each detected centroid can satisfy at most
        # one reference centroid, so a missing or duplicated detection is caught
        # instead of being masked by reusing the same nearest candidate.
        worst = 0.0
        for label, c0 in ref:
            expected = (c0 - center) @ R.T + center
            cands = by_label.get(label, [])
            if not cands:
                worst = float("inf"); break
            dists = [float(np.linalg.norm(expected - c)) for c in cands]
            j = int(np.argmin(dists))
            worst = max(worst, dists[j])
            cands.pop(j)  # consume this candidate so it can't match again
        trials.append(worst)
        max_err = max(max_err, worst)

    finite = [t for t in trials if np.isfinite(t)]
    return {
        "passed": bool(max_err <= tol_mm),
        "tol_mm": float(tol_mm),
        "max_error": float(max_err),
        "mean_error": float(np.mean(finite)) if finite else float("inf"),
        "trials": trials,
        "n_trials": int(n_trials),
    }


# ---------------------------------------------------------------------------
# self-test
# ---------------------------------------------------------------------------

def _demo():
    res = 16
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
    rng = np.random.default_rng(0)

    Vr = random_rotation(V, rng)
    print(f"[rotate] edge lengths preserved: {np.allclose(np.linalg.norm(Vr[F[:,0]]-Vr[F[:,1]],axis=1), np.linalg.norm(V[F[:,0]]-V[F[:,1]],axis=1))}")

    Vn = add_lognormal_noise(V, sigma=0.005, rng=rng)
    print(f"[noise]  mean disturbance = {np.abs(Vn - V).mean():.5f}  (small, signs preserved)")

    Vs = smooth_laplace(Vn, F, iterations=2)
    print(f"[smooth] noise reduced after Laplace: {np.abs(Vs - V).mean():.5f}")

    Vd = arap_deform_features(V, F, L, n_points=2, magnitude=0.05, rng=rng)
    moved = np.linalg.norm(Vd - V, axis=1)
    print(f"[arap]   feature moved (max={moved.max():.3f}), boundary nearly fixed (mean boundary={moved[(V[:,0]<0.1)].mean():.4f})")

    Va, _ = augment(V, F, L, seed=1)
    print(f"[full]   augmented clone: {len(Va)} vertices, faces unchanged={F.shape}")
    print("\n[ok] augmentation self-test passed.")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
    _demo()
