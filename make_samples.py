# Generate a few VALID synthetic ABB-style part JSONs (mesh + connection points)
# so the DiffusionNet backbone (which needs real faces for its Laplacian
# eigenbasis) can run end-to-end. These are synthetic plates -- good enough to
# prove the pipeline runs and emit real metrics, NOT a substitute for the real
# 429-file corpus.
import json, os, sys
import numpy as np


def grid_plate(nx, ny, w, h):
    """Triangulated open plate: (V, F). z=0 plane, size w x h mm."""
    xs = np.linspace(0, w, nx)
    ys = np.linspace(0, h, ny)
    X, Y = np.meshgrid(xs, ys)
    V = np.column_stack([X.ravel(), Y.ravel(), np.zeros(X.size)])
    F = []
    for j in range(ny - 1):
        for i in range(nx - 1):
            a = j * nx + i
            b = a + 1
            c = a + nx
            d = c + 1
            F.append([a, b, c])
            F.append([b, d, c])
    return V, np.asarray(F, dtype=np.int64)


def make_part(idx, rng):
    nx = int(rng.integers(24, 34))
    ny = int(rng.integers(24, 34))
    w = float(rng.uniform(80, 140))
    h = float(rng.uniform(80, 140))
    V, F = grid_plate(nx, ny, w, h)
    # 2-3 connection points on the plate, snapped to nearby vertices,
    # insert direction outward (+z) from the top face.
    k = int(rng.integers(2, 4))
    cps = []
    for c in range(k):
        px = float(rng.uniform(0.15 * w, 0.85 * w))
        py = float(rng.uniform(0.15 * h, 0.85 * h))
        vi = int(np.argmin(np.linalg.norm(V[:, :2] - [px, py], axis=1)))
        p = V[vi]
        cps.append({
            "Index": c,
            "Name": f"X{idx}-{c}",
            "Point": {"X": float(p[0]), "Y": float(p[1]), "Z": float(p[2])},
            "InsertDirection": {"X": 0.0, "Y": 0.0, "Z": 1.0},
        })
    obj = {
        "PartNr": f"SYN.{idx:03d}",
        "Graphic3d": {
            "Points": [{"X": float(x), "Y": float(y), "Z": float(z)} for x, y, z in V],
            "Indices": F.ravel().tolist(),
        },
        "BoundingBox": {"Dimension": {"X": w, "Y": h, "Z": 0.0},
                        "Location": {"X": 0.0, "Y": 0.0, "Z": 0.0}},
        "ConnectionPoints": cps,
    }
    return obj


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    out = argv[0] if len(argv) > 0 else "abb_samples"
    n = int(argv[1]) if len(argv) > 1 else 6
    os.makedirs(out, exist_ok=True)
    rng = np.random.default_rng(0)
    for i in range(n):
        obj = make_part(i, rng)
        with open(os.path.join(out, f"part_{i:03d}.json"), "w", encoding="utf-8") as fh:
            json.dump(obj, fh)
        print("wrote", obj["PartNr"], len(obj["Graphic3d"]["Points"]), "verts",
              len(obj["Graphic3d"]["Indices"]) // 3, "faces",
              len(obj["ConnectionPoints"]), "cps")
    print("done ->", os.path.abspath(out))


if __name__ == "__main__":
    main()
