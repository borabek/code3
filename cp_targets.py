# Connection-point target encoding/decoding for direct regression (Option B).
#
# The detector is a per-vertex DiffusionNet (outputs_at="vertices"). Instead of
# the 5-class segmentation head, the connection-point regressor predicts, per
# vertex, a 7-channel target:
#
#   [0]     heatmap   h in [0,1]   = exp(-d^2 / (2 sigma^2)), d = dist to nearest CP
#   [1:4]   offset    o (mm)       = (CP_point - vertex)            (vector to that CP)
#   [4:7]   direction u (unit)     = InsertDirection of that CP     (OUTWARD)
#
# Decoding inverts this: high-heatmap vertices vote for a CP at
# (vertex + offset); votes are clustered with a greedy NMS whose radius is the
# robot tool clearance, giving one prediction per real connection point with a
# sub-vertex-accurate location and an averaged outward direction.
#
# Reference: Scheffler (2022) backbone (§5.3.3); heatmap/offset keypoint
# regression is the Option-B head chosen for the ABB connection-point data.

import logging
import numpy as np

logger = logging.getLogger(__name__)

HEATMAP = 0
OFFSET = slice(1, 4)
DIRECTION = slice(4, 7)
N_CHANNELS = 7


# ---------------------------------------------------------------------------
# encode: (vertices, cp_points, cp_directions) -> (N, 7) target
# ---------------------------------------------------------------------------

def _nearest_cp(vertices, cp_points):
    """For each vertex, index of and distance to the nearest CP. Chunked to
    keep the (N x K) distance matrix small on large meshes."""
    N = len(vertices)
    nearest = np.zeros(N, dtype=np.int64)
    dist = np.zeros(N, dtype=np.float64)
    chunk = 4096
    for s in range(0, N, chunk):
        v = vertices[s:s + chunk]                      # (c, 3)
        d = np.linalg.norm(v[:, None, :] - cp_points[None, :, :], axis=2)  # (c, K)
        j = np.argmin(d, axis=1)
        nearest[s:s + chunk] = j
        dist[s:s + chunk] = d[np.arange(len(v)), j]
    return nearest, dist


def encode_targets(vertices, cp_points, cp_directions, sigma_mm=None,
                   offset_radius_mm=None):
    """Build the (N, 7) per-vertex regression target.

    sigma_mm         : Gaussian width of the heatmap. Default = 2% of the mesh
                       bounding-box diagonal (scale-adaptive).
    offset_radius_mm : offset+direction channels are only meaningful near a CP;
                       this radius defines the supervision mask returned to the
                       loss. Default = 3 * sigma_mm.

    Returns (target (N,7) float32, mask (N,) bool, sigma_mm). `mask` marks the
    vertices whose offset/direction channels should be supervised.
    """
    vertices = np.asarray(vertices, dtype=np.float64)
    N = len(vertices)
    target = np.zeros((N, N_CHANNELS), dtype=np.float32)

    if len(cp_points) == 0:
        mask = np.zeros(N, dtype=bool)
        return target, mask, (sigma_mm or 0.0)

    if sigma_mm is None:
        diag = np.linalg.norm(vertices.max(0) - vertices.min(0))
        sigma_mm = max(diag * 0.02, 1e-6)
    if offset_radius_mm is None:
        offset_radius_mm = 3.0 * sigma_mm

    nearest, dist = _nearest_cp(vertices, cp_points)
    target[:, HEATMAP] = np.exp(-(dist ** 2) / (2.0 * sigma_mm ** 2))
    target[:, OFFSET] = (cp_points[nearest] - vertices).astype(np.float32)
    target[:, DIRECTION] = cp_directions[nearest].astype(np.float32)

    # Guarantee one EXACT peak (heat=1) per CP -- the nearest vertex to it. On a
    # coarse mesh the Gaussian skirt may never reach ~1 at any vertex, so this
    # snaps the true peak. Penalty-reduced focal losses (cp_regressor heat_loss
    # 'centernet') key their positives off heat==1 and collapse without it.
    for j in range(len(cp_points)):
        i = int(np.argmin(np.linalg.norm(vertices - cp_points[j], axis=1)))
        target[i, HEATMAP] = 1.0

    mask = dist <= offset_radius_mm
    return target, mask, sigma_mm


# ---------------------------------------------------------------------------
# decode: (N, 7) prediction -> list of connection points
# ---------------------------------------------------------------------------

def decode_predictions(vertices, pred, heatmap_thresh=0.3, nms_radius_mm=5.0,
                       min_votes=1):
    """Turn a (N, 7) prediction into discrete connection points.

    Each vertex with heatmap > thresh votes for a CP at (vertex + offset),
    weighted by its heatmap value. Votes are greedily clustered: the highest
    remaining vote seeds a cluster, all votes within nms_radius_mm are merged
    into it (weighted mean of position and direction), and removed.

    Returns list of dicts: {point (3,), direction (3,) unit OUTWARD,
    insertion_axis (3,) unit INWARD, score, n_votes}.
    """
    vertices = np.asarray(vertices, dtype=np.float64)
    pred = np.asarray(pred, dtype=np.float64)
    h = pred[:, HEATMAP]
    keep = np.where(h > heatmap_thresh)[0]
    if len(keep) == 0:
        return []

    votes = vertices[keep] + pred[keep, OFFSET]   # (n, 3) predicted CP location
    dirs = pred[keep, DIRECTION]
    scores = h[keep]
    order = np.argsort(-scores)                   # high score first

    used = np.zeros(len(keep), dtype=bool)
    out = []
    for oi in order:
        if used[oi]:
            continue
        seed = votes[oi]
        d = np.linalg.norm(votes - seed, axis=1)
        members = (d <= nms_radius_mm) & (~used)
        used[members] = True
        if members.sum() < min_votes:
            continue
        w = scores[members]
        wsum = w.sum()
        pt = (votes[members] * w[:, None]).sum(0) / wsum
        dv = (dirs[members] * w[:, None]).sum(0)
        nrm = np.linalg.norm(dv)
        dv = dv / nrm if nrm > 0 else np.array([0.0, 0.0, 1.0])
        out.append({
            "point": pt,
            "direction": dv,             # OUTWARD (= approach_vector)
            "insertion_axis": -dv,       # INWARD (travel into the part)
            "score": float(w.max()),
            "n_votes": int(members.sum()),
        })
    return out


# ---------------------------------------------------------------------------
# export decoded connection points -> repo robot-ready JSON node schema
# ---------------------------------------------------------------------------
# Mirrors connector3d.build_connector_graph() node fields so the regression
# detector emits the same robot-ready output as the segmentation pipeline.

def decoded_to_robot_nodes(preds, names_per_block=None, approach_distance_mm=0.0,
                           robot_class="TerminalContact"):
    """Convert decode_predictions() output -> list of robot-ready node dicts.

    preds                : list from decode_predictions (point/direction/insertion_axis/score).
    names_per_block      : optional list[list[str]] of terminal names per block,
                           aligned to preds order (carried as 'terminal_names').
    approach_distance_mm : standoff distance along the outward approach vector.
    """
    def vec(v):
        return None if v is None else [round(float(x), 5) for x in np.asarray(v)]

    nodes = []
    for i, p in enumerate(preds):
        entry = np.asarray(p["point"], dtype=float)
        approach = np.asarray(p["direction"], dtype=float)          # OUTWARD
        nodes.append({
            "id": i,
            "robot_class": robot_class,
            "entry_point": vec(entry),
            "approach_vector": vec(approach),
            "insertion_axis": vec(p.get("insertion_axis", -approach)),  # INWARD travel
            "standoff_point": vec(entry + approach * float(approach_distance_mm)),
            "approach_distance_mm": round(float(approach_distance_mm), 5),
            "confidence_score": round(float(p.get("score", 1.0)), 4),
            "n_votes": int(p.get("n_votes", 0)),
            "terminal_names": list(names_per_block[i]) if names_per_block else None,
            "n_terminals": (len(names_per_block[i]) if names_per_block else None),
        })
    return nodes


# ---------------------------------------------------------------------------
# selftest: encode a real-ish part, decode, check we recover the CPs
# ---------------------------------------------------------------------------

def _selftest():
    # a flat-ish plate of vertices in a 100mm square, z=0
    xs, ys = np.meshgrid(np.linspace(0, 100, 60), np.linspace(0, 100, 60))
    V = np.column_stack([xs.ravel(), ys.ravel(), np.zeros(xs.size)])
    # 3 ground-truth CPs, each with an outward (+z) direction
    cp_pts = np.array([[25, 25, 0], [75, 30, 0], [50, 80, 0]], dtype=float)
    cp_dir = np.array([[0, 0, 1.0]] * 3)

    tgt, mask, sigma = encode_targets(V, cp_pts, cp_dir)
    assert tgt.shape == (len(V), 7)
    # decode the *ground-truth* target -> should recover the 3 CPs precisely
    got = decode_predictions(V, tgt, heatmap_thresh=0.3, nms_radius_mm=10.0)
    assert len(got) == 3, f"expected 3 CPs, got {len(got)}"
    # match each prediction to nearest GT and check error
    errs = []
    for g in got:
        dd = np.linalg.norm(cp_pts - g["point"], axis=1)
        errs.append(dd.min())
        assert g["direction"][2] > 0.99       # outward +z
        assert g["insertion_axis"][2] < -0.99  # inward -z
    # exercise the robot-ready exporter
    nodes = decoded_to_robot_nodes(got, names_per_block=[["a"], ["b"], ["c"]],
                                   approach_distance_mm=10.0)
    assert len(nodes) == 3 and nodes[0]["robot_class"] == "TerminalContact"
    assert nodes[0]["insertion_axis"][2] < -0.99   # inward = -approach
    assert nodes[0]["terminal_names"] is not None
    print(f"cp_targets selftest OK: sigma={sigma:.2f}mm, "
          f"recovered {len(got)} CPs, max loc err={max(errs):.3f}mm, "
          f"robot nodes={len(nodes)}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    _selftest()
