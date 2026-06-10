# mesh + label I/O (.off/.obj/.stl/.labels)
#
# Reference: Scheffler (2022), §5.1.2
#   §5.1.2 / Code 1,2,3 – STL binary/ASCII format, OBJ format, OFF format
#   §5.1.2               – STEP->STL->OBJ->OFF conversion pipeline (I/O end)

import os
import struct
import logging
import numpy as np
from connector_constants import COORD_ROUNDING

logger = logging.getLogger(__name__)


# §5.1.2 / Code 3 – OFF format loader
def load_off(path):
    """Load OFF mesh. Raises FileNotFoundError/ValueError."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"OFF file not found: {path}")
    
    try:
        with open(path, encoding="utf-8-sig") as fh:
            rows = []
            for ln in fh:
                s = ln.strip()
                if not s or s.startswith("#"):
                    continue
                rows.append(s)
    except IOError as e:
        raise IOError(f"Cannot read OFF file {path}: {e}")
    
    if not rows:
        raise ValueError(f"OFF file {path} is empty")
    
    tok = " ".join(rows).split()
    if not tok:
        raise ValueError(f"OFF file {path} has no data")
    
    i = 0
    if tok[0].upper().startswith("OFF"):
        i = 1  # skip header token
    
    try:
        nv, nf = int(tok[i]), int(tok[i + 1])
    except (IndexError, ValueError) as e:
        raise ValueError(f"Invalid OFF header in {path}: expected 'OFF nv nf', got '{tok[:i+2]}'") from e
    
    i += 3  # nv nf ne — we don't need ne

    V = np.empty((nv, 3), dtype=float)
    for k in range(nv):
        try:
            V[k] = (float(tok[i]), float(tok[i + 1]), float(tok[i + 2]))
            i += 3
        except (IndexError, ValueError) as e:
            raise ValueError(f"Truncated vertex data in {path} (expected {nv} vertices, got {k})") from e

    F = []
    for fi in range(nf):
        try:
            cnt = int(tok[i])
            i += 1
            poly = [int(tok[i + j]) for j in range(cnt)]
            i += cnt
            # Validate vertex indices
            if any(v < 0 or v >= nv for v in poly):
                logger.warning("Face %d has out-of-bounds vertex indices in %s", fi, path)
                continue
            # quad or higher — triangulate via fan
            for j in range(1, cnt - 1):
                F.append((poly[0], poly[j], poly[j + 1]))
        except (IndexError, ValueError) as e:
            raise ValueError(f"Truncated face data in {path} (expected {nf} faces, got {fi})") from e
    
    return V, np.array(F, dtype=int)


# §5.1.2 / Code 2 – OBJ format loader
def load_obj(path):
    """Load OBJ mesh. Raises FileNotFoundError/ValueError."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"OBJ file not found: {path}")
    
    V, F = [], []
    try:
        with open(path, encoding="utf-8-sig") as fh:
            for li, ln in enumerate(fh, 1):
                try:
                    if ln.startswith("v "):
                        p = ln.split()
                        if len(p) < 4:
                            raise ValueError(f"Invalid vertex line: {ln.strip()}")
                        V.append((float(p[1]), float(p[2]), float(p[3])))
                    elif ln.startswith("f "):
                        # "f 1//1 2//1 3//1" -> take only the first index per corner
                        parts = ln.split()[1:]
                        if len(parts) < 3:
                            raise ValueError(f"Face has < 3 vertices: {ln.strip()}")
                        idx = [int(p.split("/")[0]) - 1 for p in parts]  # OBJ counts from 1
                        for j in range(1, len(idx) - 1):
                            F.append((idx[0], idx[j], idx[j + 1]))
                except ValueError as e:
                    logger.warning("Parse error at line %d of %s: %s", li, path, e)
                    continue
    except IOError as e:
        raise IOError(f"Cannot read OBJ file {path}: {e}")
    
    if not V:
        raise ValueError(f"No vertices found in {path}")
    
    return np.array(V, dtype=float), np.array(F, dtype=int)


# §5.1.2 / Code 1 – STL binary/ASCII format loader (both variants detected by file size)
def load_stl(path):
    # STL has two variants: ASCII ("solid ... facet normal ...") and binary
    # (80-byte header, then uint32 triangle count, then per triangle 12 floats
    # + 2-byte attribute). Some binary files also start with "solid", so we
    # detect the format via the expected file size, not just the header.
    with open(path, "rb") as fh:
        raw = fh.read()

    # Format detection: the binary size formula (84 + n_tri*50) is the most
    # reliable test — it also works for binary files whose header happens to
    # start with "solid". If it doesn't match and the file starts with "solid",
    # it's ASCII.
    binary_size = -1
    if len(raw) >= 84:
        n_tri = struct.unpack_from("<I", raw, 80)[0]
        binary_size = 84 + n_tri * 50

    if len(raw) == binary_size:
        is_binary = True
    elif raw[:5].lower() == b"solid":
        is_binary = False
    else:
        is_binary = True  # non-standard header / padding -> best-effort binary

    if is_binary:
        return _load_stl_binary(raw)
    return _load_stl_ascii(raw.decode("utf-8-sig", errors="replace"))


def _load_stl_binary(raw):
    n_tri = struct.unpack_from("<I", raw, 80)[0]
    # Each 50-byte record is: normal(3f) + 3 corners(3f) + 2-byte attribute.
    # Parse the whole block at once with a structured dtype instead of a Python
    # per-triangle loop, then drop the normals and keep the corners.
    rec = np.dtype([("normal", "<f4", 3), ("corners", "<f4", (3, 3)),
                    ("attr", "<u2")])
    expected = 84 + n_tri * rec.itemsize
    if len(raw) < expected:
        raise ValueError(f"binary STL truncated: need {expected} bytes, got {len(raw)}")
    data = np.frombuffer(raw, dtype=rec, count=n_tri, offset=84)
    tris = data["corners"].astype(float)
    return _weld_triangle_soup(tris)


def _load_stl_ascii(text):
    verts = []
    for ln in text.splitlines():
        s = ln.strip()
        if s.startswith("vertex"):
            p = s.split()
            verts.append((float(p[1]), float(p[2]), float(p[3])))
    if len(verts) % 3 != 0:
        raise ValueError(f"ASCII STL has {len(verts)} vertices, not a multiple of 3")
    tris = np.array(verts, dtype=float).reshape(-1, 3, 3)
    return _weld_triangle_soup(tris)


def _weld_triangle_soup(tris, decimals=COORD_ROUNDING):
    # STL stores a "triangle soup" with no shared vertices — each corner is stored
    # separately. The pipeline needs an indexed mesh, so we merge identical coordinates
    # (rounded to `decimals` decimal places).
    flat = tris.reshape(-1, 3)
    keys = np.round(flat, decimals)
    uniq, first_idx, inverse = np.unique(keys, axis=0, return_index=True, return_inverse=True)
    # keep vertices in order of first occurrence (stable output) rather than
    # np.unique's sorted order
    order = np.argsort(first_idx)
    remap = np.empty(len(uniq), dtype=int)
    remap[order] = np.arange(len(uniq))
    V = flat[first_idx[order]]
    F = remap[inverse].reshape(-1, 3)
    return V.astype(float), F.astype(int)


def load_mesh(path):
    ext = os.path.splitext(path)[1].lower()
    if ext == ".off":
        return load_off(path)
    if ext == ".obj":
        return load_obj(path)
    if ext == ".stl":
        return load_stl(path)
    raise ValueError(f"unsupported format: {ext} (only .off / .obj / .stl)")


def load_labels(path):
    # one label per vertex, whitespace or newline as separator
    vals = []
    with open(path, encoding="utf-8-sig") as fh:
        for ln in fh:
            s = ln.strip()
            if not s or s.startswith("#"):
                continue
            vals.extend(int(float(t)) for t in s.split())
    return np.array(vals, dtype=int)


def save_off(path, V, F):
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("OFF\n")
        fh.write(f"{len(V)} {len(F)} 0\n")
        for v in V:
            fh.write(f"{v[0]} {v[1]} {v[2]}\n")
        for t in F:
            fh.write(f"3 {t[0]} {t[1]} {t[2]}\n")


def save_obj(path, V, F):
    with open(path, "w", encoding="utf-8") as fh:
        for v in V:
            fh.write(f"v {v[0]} {v[1]} {v[2]}\n")
        for t in F:
            fh.write(f"f {int(t[0])+1} {int(t[1])+1} {int(t[2])+1}\n")


def save_labels(path, labels):
    with open(path, "w", encoding="utf-8") as fh:
        for v in labels:
            fh.write(f"{int(v)}\n")