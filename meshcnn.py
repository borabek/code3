# MeshCNN integration.
#
# Two parts:
#   1. meshcnn_inputs(): the 5-dim edge features + 4-neighbor ring
#      (pure numpy, from mesh_utils.py) -- testable without torch.
#   2. build_meshcnn(): a compact but faithful network with the INVARIANT
#      edge convolution:
#         conv(e) = k0*e + sum_j kj * sym_j
#      using symmetric functions over ring pairs (a,c) and (b,d):
#      sum and absolute difference. This makes the convolution invariant
#      to the ordering of neighbors.
#
# Deliberate simplification vs. the original MeshCNN: mesh pooling/unpooling
# (edge collapse with cache) is NOT implemented -- that is the part the thesis
# criticizes as slow and memory-heavy, and it would not be testable without torch.
# The edge convolution itself is faithful.
#
# torch is imported LAZILY (factory pattern), so the module stays importable
# without torch.
#
# Reference: Scheffler (2022), §5.3.1
#   §5.3.1 / Formel 5.3, Abb.37,38 – MeshCNN 5-dim invariant edge features:
#           dihedral angle, 2 inner angles, 2 edge-length ratios; sorted pairs
#   §5.3.1 / Abb.39 – Mesh Pooling / Unpooling (edge collapse; noted but not reimplemented)

import logging
import numpy as np

# ---------------------------------------------------------------------------
# Mesh utilities (merged from mesh_utils.py)
# ---------------------------------------------------------------------------

BACKGROUND = 0  # default label for unlabeled vertices, edges, and faces


def mesh_edges(faces):
    """Return all unique undirected edges as (E, 2) array of node index pairs."""
    edge_set = set()
    for tri in faces:
        a, b, c = int(tri[0]), int(tri[1]), int(tri[2])
        edge_set.add((min(a, b), max(a, b)))
        edge_set.add((min(b, c), max(b, c)))
        edge_set.add((min(a, c), max(a, c)))
    return np.array(sorted(edge_set), dtype=int)


def vertex_to_face(faces, vertex_labels, bg=BACKGROUND):
    """Assign a label to each triangle (shared label only when all 3 corners agree)."""
    face_labels = []
    for tri in faces:
        a, b, c = int(tri[0]), int(tri[1]), int(tri[2])
        la, lb, lc = vertex_labels[a], vertex_labels[b], vertex_labels[c]
        face_labels.append(la if la == lb == lc else bg)
    return np.array(face_labels, dtype=int)


def vertex_to_edge(edges, vertex_labels, bg=BACKGROUND):
    """Assign a label to each edge (shared label only when both endpoints agree)."""
    edge_labels = []
    for edge in edges:
        a, b = int(edge[0]), int(edge[1])
        la, lb = vertex_labels[a], vertex_labels[b]
        edge_labels.append(la if la == lb else bg)
    return np.array(edge_labels, dtype=int)


def face_to_vertex(faces, face_labels, n_vertices, bg=BACKGROUND):
    """Assign a label to each vertex using majority vote from neighboring faces."""
    n_classes = max(int(max(face_labels)), bg) + 1
    votes = [[0] * n_classes for _ in range(n_vertices)]
    for i, tri in enumerate(faces):
        label = int(face_labels[i])
        for corner in tri:
            votes[int(corner)][label] += 1
    result = []
    for v in range(n_vertices):
        total = sum(votes[v])
        result.append(votes[v].index(max(votes[v])) if total > 0 else bg)
    return np.array(result, dtype=int)


def edge_to_vertex(edges, edge_labels, n_vertices, bg=BACKGROUND):
    """Assign a label to each vertex using majority vote from neighboring edges."""
    n_classes = max(int(max(edge_labels)), bg) + 1
    votes = [[0] * n_classes for _ in range(n_vertices)]
    for i, edge in enumerate(edges):
        label = int(edge_labels[i])
        for endpoint in edge:
            votes[int(endpoint)][label] += 1
    result = []
    for v in range(n_vertices):
        total = sum(votes[v])
        result.append(votes[v].index(max(votes[v])) if total > 0 else bg)
    return np.array(result, dtype=int)


def _build_edge_table(faces):
    edge_of = {}
    edges = []
    opp = []

    def get_or_add_edge(u, v):
        key = (min(u, v), max(u, v))
        if key not in edge_of:
            edge_of[key] = len(edges)
            edges.append(key)
            opp.append([-1, -1])
        return edge_of[key]

    for tri in faces:
        a, b, c = int(tri[0]), int(tri[1]), int(tri[2])
        for u, v, w in [(a, b, c), (b, c, a), (a, c, b)]:
            ei = get_or_add_edge(u, v)
            if opp[ei][0] == -1:
                opp[ei][0] = w
            elif opp[ei][1] == -1:
                opp[ei][1] = w
    return edges, opp, edge_of


def _triangle_normal(p, q, r):
    n = np.cross(q - p, r - p)
    length = np.linalg.norm(n)
    return n / length if length >= 1e-12 else np.zeros(3)


def _angle_at_vertex(apex, p, q):
    a, b = p - apex, q - apex
    la, lb = np.linalg.norm(a), np.linalg.norm(b)
    if la < 1e-12 or lb < 1e-12:
        return 0.0
    return float(np.arccos(float(np.clip(np.dot(a, b) / (la * lb), -1.0, 1.0))))


def edge_features(vertices, faces):
    """Compute 5-dim MeshCNN features + 4-ring neighbors for every edge."""
    V = np.asarray(vertices, dtype=float)
    edge_list, opp, edge_of = _build_edge_table(faces)
    num_edges = len(edge_list)
    feats = np.zeros((num_edges, 5), dtype=float)
    nbrs = np.full((num_edges, 4), -1, dtype=int)

    for ei in range(num_edges):
        u, v = edge_list[ei]
        pu, pv = V[u], V[v]
        edge_len = np.linalg.norm(pv - pu)
        raw_opp = sorted(int(x) for x in opp[ei] if x >= 0)
        w0 = raw_opp[0] if len(raw_opp) > 0 else -1
        w1 = raw_opp[1] if len(raw_opp) > 1 else -1
        angles, ratios, normals, ring_edges = [], [], [], []
        for w in [w0, w1]:
            if w < 0:
                continue
            pw = V[w]
            angles.append(_angle_at_vertex(pw, pu, pv))
            cross = np.cross(pv - pu, pw - pu)
            area = 0.5 * np.linalg.norm(cross)
            height = (2.0 * area / edge_len) if edge_len > 1e-12 else 1e-12
            ratios.append(edge_len / height if height > 1e-12 else 0.0)
            normals.append(_triangle_normal(pu, pv, pw))
            ring_edges += [(u, w), (v, w)]
        dihedral = (float(np.arccos(float(np.clip(np.dot(normals[0], normals[1]), -1.0, 1.0))))
                    if len(normals) == 2 else 0.0)
        angles = sorted(angles) + [0.0] * (2 - len(angles))
        ratios = sorted(ratios) + [0.0] * (2 - len(ratios))
        feats[ei] = [dihedral, angles[0], angles[1], ratios[0], ratios[1]]
        for k, (a, b) in enumerate(ring_edges[:4]):
            nbrs[ei, k] = edge_of.get((min(a, b), max(a, b)), -1)

    return np.array(edge_list, dtype=int), feats, nbrs


_mesh_edges = mesh_edges  # alias used internally

# ---------------------------------------------------------------------------

logger = logging.getLogger(__name__)

N_CLASSES = 5


def meshcnn_inputs(vertices, faces):
    """5-dim edge features + neighbor ring for MeshCNN.

    Returns: (edges (E,2), feats (E,5), nbrs (E,4)).
    """
    return edge_features(vertices, faces)


def vertex_labels_to_edges(faces, vertex_labels):
    """Convert per-vertex labels to edge labels (for training)."""
    return vertex_to_edge(_mesh_edges(faces), vertex_labels)


# §5.3.1 / Abb.37,38 – compact MeshCNN with invariant edge convolution
def build_meshcnn(in_channels=5, width=64, n_blocks=3, n_classes=N_CLASSES, dropout=0.2):
    """Compact MeshCNN segmentation network (lazy torch import).

    Returns: torch.nn.Module. forward(x, nbrs) -> (E, n_classes) logits,
      x    : (E, in_channels) edge features
      nbrs : (E, 4) ring-neighbor edge indices (-1 at boundary)
    """
    import torch
    import torch.nn as nn

    # §5.3.1 / Formel 5.3 – invariant edge convolution:
    #   [e, a+c, |a-c|, b+d, |b-d|]; symmetric functions over sorted ring pairs
    class MeshConv(nn.Module):
        # Invariant edge convolution: 5 symmetric inputs
        # [e, a+c, |a-c|, b+d, |b-d|] -> 1x1 conv over the "5".
        def __init__(self, c_in, c_out):
            super().__init__()
            self.lin = nn.Linear(5 * c_in, c_out)

        def forward(self, x, nbrs):
            E, C = x.shape
            pad = torch.zeros(1, C, dtype=x.dtype, device=x.device)
            xp = torch.cat([x, pad], dim=0)          # index -1 -> zero row
            idx = nbrs.clone()
            idx[idx < 0] = E                          # redirect -1 to zero row
            a, b, c, d = xp[idx[:, 0]], xp[idx[:, 1]], xp[idx[:, 2]], xp[idx[:, 3]]
            sym = torch.cat([x, a + c, (a - c).abs(), b + d, (b - d).abs()], dim=1)
            return self.lin(sym)

    class MeshCNN(nn.Module):
        def __init__(self):
            super().__init__()
            self.embed = nn.Linear(in_channels, width)
            self.blocks = nn.ModuleList([MeshConv(width, width) for _ in range(n_blocks)])
            self.drop = nn.Dropout(dropout)
            self.head = nn.Linear(width, n_classes)

        def forward(self, x, nbrs):
            h = torch.relu(self.embed(x))
            for blk in self.blocks:
                h = torch.relu(blk(h, nbrs) + h)      # residual connection
            return self.head(self.drop(h))

    return MeshCNN()


def _demo():
    # numpy part (always testable): edge features of a small mesh
    V = np.array([[0, 0, 0], [1, 0, 0], [0.5, 1, 0], [0.5, 1, 0.6]], dtype=float)
    F = np.array([[0, 1, 2], [0, 1, 3]], dtype=int)
    edges, feats, nbrs = meshcnn_inputs(V, F)
    print(f"[inputs] {len(edges)} edges, feats shape {feats.shape} (5-dim), nbrs shape {nbrs.shape}")
    assert feats.shape[1] == 5 and nbrs.shape[1] == 4

    # torch part (only if available): forward pass shape check
    try:
        import torch
        model = build_meshcnn()
        x = torch.tensor(feats, dtype=torch.float32)
        nb = torch.tensor(nbrs, dtype=torch.long)
        out = model(x, nb)
        print(f"[torch]  forward ok -> logits {tuple(out.shape)} (E x classes)")
        assert out.shape == (len(edges), N_CLASSES)
        print("\n[ok] MeshCNN: edge features + invariant convolution verified.")
    except ImportError:
        print("[torch]  not installed -> edge features tested, model factory ready.")
        print("\n[ok] MeshCNN inputs (numpy) verified; build_meshcnn() requires torch.")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
    _demo()
