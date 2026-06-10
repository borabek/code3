# Streaming loader for ABB connection-point training JSON (§5.1 data ingestion).
#
# Each training part is a JSON object with:
#   PartNr               : str
#   Graphic3d.Points     : [{X,Y,Z}, ...]   vertices in mm
#   Graphic3d.Indices    : [i0, i1, i2, ...] flat triangle list (len % 3 == 0)
#   BoundingBox          : {Dimension{X,Y,Z}, Location{X,Y,Z}}
#   ConnectionPoints     : [{Index, Name, Point{X,Y,Z}, InsertDirection{X,Y,Z}}]
#
# The full corpus is ~1 GB. It may be delivered either as
#   (a) one big JSON file whose top level is a LIST of part objects, or
#   (b) a directory of per-part JSON files (one object per file, like the samples).
# This module auto-detects both and *streams* parts so the whole corpus is
# never held in RAM at once (ijson for the big-array case).

import os
import json
import glob
import logging
from dataclasses import dataclass

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class Part:
    """One training part: mesh + ground-truth connection points."""
    part_nr: str
    vertices: np.ndarray          # (N, 3) float64, mm
    faces: np.ndarray             # (M, 3) int64
    cp_points: np.ndarray         # (K, 3) float64, mm  (ConnectionPoints[].Point)
    cp_directions: np.ndarray     # (K, 3) float64, unit (ConnectionPoints[].InsertDirection, OUTWARD)
    cp_names: list                # K strings (ConnectionPoints[].Name)

    @property
    def n_vertices(self):
        return len(self.vertices)

    @property
    def n_cps(self):
        return len(self.cp_points)


# ---------------------------------------------------------------------------
# core: build a Part from an already-parsed dict
# ---------------------------------------------------------------------------

def _xyz(d):
    return [float(d["X"]), float(d["Y"]), float(d["Z"])]


def part_from_dict(obj):
    """Build a Part from a parsed JSON dict. Raises ValueError on malformed input."""
    if "Graphic3d" not in obj:
        raise ValueError("missing 'Graphic3d'")
    g = obj["Graphic3d"]
    pts = g.get("Points") or []
    idx = g.get("Indices") or []
    if len(idx) % 3 != 0:
        raise ValueError(f"Indices length {len(idx)} not divisible by 3")

    V = np.array([_xyz(p) for p in pts], dtype=np.float64)
    F = np.asarray(idx, dtype=np.int64).reshape(-1, 3)
    if len(V) and F.size and F.max() >= len(V):
        raise ValueError(f"face index {F.max()} >= n_vertices {len(V)}")

    cps = obj.get("ConnectionPoints") or []
    if cps:
        cp_pts = np.array([_xyz(c["Point"]) for c in cps], dtype=np.float64)
        cp_dir = np.array([_xyz(c["InsertDirection"]) for c in cps], dtype=np.float64)
        # normalise the insert directions (defensive; samples are already unit)
        nrm = np.linalg.norm(cp_dir, axis=1, keepdims=True)
        nrm[nrm == 0] = 1.0
        cp_dir = cp_dir / nrm
        cp_names = [str(c.get("Name", i)) for i, c in enumerate(cps)]
    else:
        cp_pts = np.zeros((0, 3), dtype=np.float64)
        cp_dir = np.zeros((0, 3), dtype=np.float64)
        cp_names = []

    return Part(
        part_nr=str(obj.get("PartNr", "unknown")),
        vertices=V, faces=F,
        cp_points=cp_pts, cp_directions=cp_dir, cp_names=cp_names,
    )


def load_part_file(path):
    """Load a single per-part JSON file into a Part (small/medium files)."""
    with open(path, encoding="utf-8-sig") as fh:
        obj = json.load(fh)
    return part_from_dict(obj)


# ---------------------------------------------------------------------------
# deduplicate coincident connection points -> terminal-block locations
# ---------------------------------------------------------------------------
# In the ABB data a single physical terminal block (e.g. the mains input with
# terminals L1/L2/L3/PE) is stored as several ConnectionPoints that share the
# exact same Point and InsertDirection -- one entry per electrical terminal.
# Geometry can only localise the block, not the individual terminals, so for
# training we collapse coincident points into one block and keep the terminal
# Names as metadata on it.

@dataclass
class TerminalBlock:
    """A distinct connection location, possibly holding several named terminals."""
    point: np.ndarray        # (3,) mm
    direction: np.ndarray    # (3,) unit, OUTWARD
    names: list              # terminal names sharing this location

    @property
    def n_terminals(self):
        return len(self.names)


def dedup_connection_points(part, tol_mm=1e-3):
    """Collapse a Part's coincident CPs into unique TerminalBlocks.

    Points within `tol_mm` of each other (and, defensively, averaged in
    direction) are merged. Returns (blocks, block_points (B,3),
    block_directions (B,3)).
    """
    pts = part.cp_points
    dirs = part.cp_directions
    names = part.cp_names
    blocks = []
    if len(pts) == 0:
        return blocks, np.zeros((0, 3)), np.zeros((0, 3))

    assigned = np.full(len(pts), -1, dtype=int)
    for i in range(len(pts)):
        if assigned[i] != -1:
            continue
        d = np.linalg.norm(pts - pts[i], axis=1)
        members = np.where((d <= tol_mm) & (assigned == -1))[0]
        assigned[members] = len(blocks)
        mdir = dirs[members].mean(0)
        nrm = np.linalg.norm(mdir)
        mdir = mdir / nrm if nrm > 0 else dirs[i]
        blocks.append(TerminalBlock(
            point=pts[members].mean(0),
            direction=mdir,
            names=[names[m] for m in members],
        ))
    bp = np.array([b.point for b in blocks], dtype=np.float64)
    bd = np.array([b.direction for b in blocks], dtype=np.float64)
    return blocks, bp, bd


# ---------------------------------------------------------------------------
# streaming over a corpus (auto-detect single-array-file vs directory of files)
# ---------------------------------------------------------------------------

def _detect_top_level(path):
    """Peek at the first non-whitespace byte: '[' => array of parts, '{' => one object."""
    with open(path, "rb") as fh:
        while True:
            b = fh.read(1)
            if not b:
                return None
            if b.isspace():
                continue
            return b.decode("latin-1")


def iter_parts_from_array_file(path):
    """Stream parts from one big JSON file whose top level is a list of part objects.

    Uses ijson so the 1 GB file is never fully materialised in memory.
    """
    import ijson
    with open(path, "rb") as fh:
        for obj in ijson.items(fh, "item"):
            try:
                yield part_from_dict(obj)
            except ValueError as exc:
                logger.warning("skipping malformed part in %s: %s", path, exc)


def iter_parts(source, pattern="*.json"):
    """Yield Part objects from a corpus, auto-detecting the layout.

    source may be:
      - a directory  -> every matching JSON file is read (one part each), recursively;
      - a single .json that is a TOP-LEVEL ARRAY -> streamed part by part (ijson);
      - a single .json that is one part object   -> yields that one part;
      - a list/tuple of file paths               -> each read as one part.
    """
    if isinstance(source, (list, tuple)):
        for p in source:
            try:
                yield load_part_file(p)
            except ValueError as exc:
                logger.warning("skipping malformed file %s: %s", p, exc)
        return

    if os.path.isdir(source):
        files = sorted(glob.glob(os.path.join(source, "**", pattern), recursive=True))
        logger.info("iter_parts: %d files under %s", len(files), source)
        for p in files:
            try:
                yield load_part_file(p)
            except ValueError as exc:
                logger.warning("skipping malformed file %s: %s", p, exc)
        return

    # single file: array -> stream; object -> one part
    top = _detect_top_level(source)
    if top == "[":
        yield from iter_parts_from_array_file(source)
    else:
        yield load_part_file(source)


def part_ids(source, pattern="*.json"):
    """Cheaply list the PartNr of every part in the corpus (for train/val splits).

    For a directory this reads only the PartNr field per file; for a big array
    file it streams PartNr via ijson without building meshes.
    """
    ids = []
    if os.path.isdir(source):
        for p in sorted(glob.glob(os.path.join(source, "**", pattern), recursive=True)):
            try:
                with open(p, encoding="utf-8-sig") as fh:
                    ids.append(str(json.load(fh).get("PartNr", os.path.basename(p))))
            except (OSError, ValueError):
                continue
        return ids

    top = _detect_top_level(source)
    if top == "[":
        import ijson
        with open(source, "rb") as fh:
            for nr in ijson.items(fh, "item.PartNr"):
                ids.append(str(nr))
    else:
        ids.append(load_part_file(source).part_nr)
    return ids


# ---------------------------------------------------------------------------
# selftest
# ---------------------------------------------------------------------------

def _selftest():
    """Build a tiny synthetic part, round-trip through part_from_dict."""
    obj = {
        "PartNr": "TEST.PART",
        "Graphic3d": {
            "Points": [{"X": 0, "Y": 0, "Z": 0}, {"X": 1, "Y": 0, "Z": 0},
                       {"X": 0, "Y": 1, "Z": 0}, {"X": 0, "Y": 0, "Z": 1}],
            "Indices": [0, 1, 2, 0, 1, 3, 0, 2, 3, 1, 2, 3],
        },
        "BoundingBox": {"Dimension": {"X": 1, "Y": 1, "Z": 1},
                        "Location": {"X": 0, "Y": 0, "Z": 0}},
        "ConnectionPoints": [
            {"Index": 0, "Name": "1", "Point": {"X": 0.3, "Y": 0.3, "Z": 0.0},
             "InsertDirection": {"X": 0.0, "Y": 0.0, "Z": -1.0}},
        ],
    }
    part = part_from_dict(obj)
    assert part.n_vertices == 4, part.n_vertices
    assert part.faces.shape == (4, 3), part.faces.shape
    assert part.n_cps == 1
    assert abs(np.linalg.norm(part.cp_directions[0]) - 1.0) < 1e-9
    print("json_dataset selftest OK:", part.part_nr,
          part.n_vertices, "verts", len(part.faces), "faces", part.n_cps, "cps")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    _selftest()
