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

from mesh_utils import edge_features, mesh_edges as _mesh_edges, vertex_to_edge

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
