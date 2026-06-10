"""
Mesh utility functions: label conversions and edge descriptors.

Label conversions:
  - vertex to face: a triangle gets a label only if all 3 corners share the same label
  - vertex to edge: an edge gets a label only if both endpoints share the same label
  - face/edge to vertex: each vertex gets the label that appears most among its neighbors

Edge features (MeshCNN):
  - 5 numbers per edge: dihedral angle, two inner angles, two edge-length ratios
  - these values do not change when the mesh is translated, rotated, or scaled

Reference: Scheffler (2022), §5.3.1, §5.2.1, §2.6.1/2.6.2, §4.2.2
  §5.3.1 / Formel 5.3, Abb.37,38 – MeshCNN 5-dim invariant edge features:
          dihedral angle, 2 inner angles, 2 edge-length ratios; sorted pairs
          for invariance
  §5.2.1          – vertex -> edge label conversion
  §2.6.1/2.6.2    – triangle mesh as 2-manifold
  §4.2.2          – NumPy vectorization as primary performance strategy
"""

import numpy as np


BACKGROUND = 0  # default label for unlabeled vertices, edges, and faces


# =============================================================================
# LABEL CONVERSIONS
# =============================================================================

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
    """Assign a label to each triangle.

    A triangle gets the shared label only when all 3 corners have the same label.
    Otherwise it gets the background label.
    """
    face_labels = []
    for tri in faces:
        a, b, c = int(tri[0]), int(tri[1]), int(tri[2])
        la = vertex_labels[a]
        lb = vertex_labels[b]
        lc = vertex_labels[c]
        if la == lb == lc:
            face_labels.append(la)
        else:
            face_labels.append(bg)
    return np.array(face_labels, dtype=int)


# §5.2.1 – vertex -> edge label conversion (shared label if both endpoints agree)
def vertex_to_edge(edges, vertex_labels, bg=BACKGROUND):
    """Assign a label to each edge.

    An edge gets the shared label only when both endpoints have the same label.
    Otherwise it gets the background label.
    """
    edge_labels = []
    for edge in edges:
        a, b = int(edge[0]), int(edge[1])
        la = vertex_labels[a]
        lb = vertex_labels[b]
        if la == lb:
            edge_labels.append(la)
        else:
            edge_labels.append(bg)
    return np.array(edge_labels, dtype=int)


def face_to_vertex(faces, face_labels, n_vertices, bg=BACKGROUND):
    """Assign a label to each vertex using majority vote from neighboring faces."""
    # Count how many times each label appears around each vertex
    n_classes = max(int(max(face_labels)), bg) + 1
    votes = [[0] * n_classes for _ in range(n_vertices)]

    for i, tri in enumerate(faces):
        label = int(face_labels[i])
        for corner in tri:
            votes[int(corner)][label] += 1

    result = []
    for v in range(n_vertices):
        total = sum(votes[v])
        if total == 0:
            result.append(bg)
        else:
            result.append(votes[v].index(max(votes[v])))
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
        if total == 0:
            result.append(bg)
        else:
            result.append(votes[v].index(max(votes[v])))
    return np.array(result, dtype=int)


# =============================================================================
# EDGE FEATURES (MeshCNN)
# =============================================================================

def _build_edge_table(faces):
    """Find all unique edges and for each edge remember the opposite vertices.

    Each edge can be shared by at most 2 triangles.
    For each triangle sharing an edge, we store the vertex that is NOT on that edge.

    Returns:
        edges   : list of (u, v) pairs (sorted, unique)
        opp     : for each edge, up to 2 opposite vertex indices (-1 if missing)
        edge_of : dict mapping (u, v) -> edge index
    """
    edge_of = {}  # (u, v) -> index in edges list
    edges = []    # list of (u, v) pairs
    opp = []      # for each edge: [opposite_vertex_tri0, opposite_vertex_tri1]

    def get_or_add_edge(u, v):
        key = (min(u, v), max(u, v))
        if key not in edge_of:
            edge_of[key] = len(edges)
            edges.append(key)
            opp.append([-1, -1])
        return edge_of[key]

    for tri in faces:
        a, b, c = int(tri[0]), int(tri[1]), int(tri[2])
        # For each edge of this triangle, record the opposite vertex
        for u, v, w in [(a, b, c), (b, c, a), (a, c, b)]:
            ei = get_or_add_edge(u, v)
            if opp[ei][0] == -1:
                opp[ei][0] = w
            elif opp[ei][1] == -1:
                opp[ei][1] = w
            # else: non-manifold edge shared by 3+ triangles — extra triangles ignored

    return edges, opp, edge_of


def _triangle_normal(p, q, r):
    """Unit normal of the triangle formed by points p, q, r."""
    n = np.cross(q - p, r - p)
    length = np.linalg.norm(n)
    if length < 1e-12:
        return np.zeros(3)
    return n / length


def _angle_at_vertex(apex, p, q):
    """Angle in radians at 'apex' inside the triangle (apex, p, q)."""
    a = p - apex
    b = q - apex
    len_a = np.linalg.norm(a)
    len_b = np.linalg.norm(b)
    if len_a < 1e-12 or len_b < 1e-12:
        return 0.0
    cos_angle = np.dot(a, b) / (len_a * len_b)
    cos_angle = float(np.clip(cos_angle, -1.0, 1.0))
    return float(np.arccos(cos_angle))


# §5.3.1 / Formel 5.3, Abb.37,38 – MeshCNN 5-dim invariant edge features:
#   dihedral angle, 2 inner angles, 2 edge-length ratios; sorted pairs for invariance
def edge_features(vertices, faces):
    """Compute 5-dimensional MeshCNN features for every edge.

    For each edge the 5 features are:
      [dihedral angle, smaller inner angle, larger inner angle,
       smaller edge/height ratio, larger edge/height ratio]

    Also returns the 4 ring-neighbor edge indices for each edge (-1 at boundary).

    Returns:
        edges : (E, 2) array of node index pairs
        feats : (E, 5) float array of features
        nbrs  : (E, 4) int array of neighbor edge indices
    """
    V = np.asarray(vertices, dtype=float)
    edge_list, opp, edge_of = _build_edge_table(faces)

    num_edges = len(edge_list)
    feats = np.zeros((num_edges, 5), dtype=float)
    nbrs = np.full((num_edges, 4), -1, dtype=int)

    for ei in range(num_edges):
        u, v = edge_list[ei]
        pu = V[u]
        pv = V[v]
        edge_len = np.linalg.norm(pv - pu)

        # Sort the opposite vertices so the feature order is deterministic
        raw_opp = [int(x) for x in opp[ei] if x >= 0]
        raw_opp = sorted(raw_opp)
        w0 = raw_opp[0] if len(raw_opp) > 0 else -1
        w1 = raw_opp[1] if len(raw_opp) > 1 else -1

        angles = []
        ratios = []
        normals = []
        ring_edges = []  # pairs (a, b) whose edge index becomes a neighbor

        for w in [w0, w1]:
            if w < 0:
                continue
            pw = V[w]

            # Inner angle at the opposite vertex
            angles.append(_angle_at_vertex(pw, pu, pv))

            # Edge-to-height ratio: edge_len / height of triangle from w
            cross = np.cross(pv - pu, pw - pu)
            area = 0.5 * np.linalg.norm(cross)
            if edge_len > 1e-12:
                height = (2.0 * area) / edge_len
            else:
                height = 1e-12
            if height > 1e-12:
                ratios.append(edge_len / height)
            else:
                ratios.append(0.0)

            normals.append(_triangle_normal(pu, pv, pw))

            # The two edges connecting u and v to the opposite vertex w
            ring_edges.append((u, w))
            ring_edges.append((v, w))

        # Dihedral angle between the two adjacent triangles
        if len(normals) == 2:
            dot = float(np.clip(np.dot(normals[0], normals[1]), -1.0, 1.0))
            dihedral = float(np.arccos(dot))
        else:
            dihedral = 0.0

        # §5.3.1 / Formel 5.3 – sort pairs so feature order is invariant to triangle winding
        # Sort angles and ratios so smaller comes first
        angles = sorted(angles)
        ratios = sorted(ratios)
        # Pad to length 2 in case there is only one adjacent triangle
        while len(angles) < 2:
            angles.append(0.0)
        while len(ratios) < 2:
            ratios.append(0.0)

        feats[ei, 0] = dihedral
        feats[ei, 1] = angles[0]
        feats[ei, 2] = angles[1]
        feats[ei, 3] = ratios[0]
        feats[ei, 4] = ratios[1]

        # Fill in the 4 ring-neighbor edge indices
        for k, (a, b) in enumerate(ring_edges[:4]):
            key = (min(a, b), max(a, b))
            nbrs[ei, k] = edge_of.get(key, -1)

    edges_array = np.array(edge_list, dtype=int)
    return edges_array, feats, nbrs


# =============================================================================
# SELF-TEST
# =============================================================================

def _demo_labels():
    """Test label conversion node to edge to face and back."""
    V = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]], dtype=float)
    F = np.array([[0, 1, 2], [0, 2, 3]], dtype=int)
    vlab = np.array([1, 1, 1, 0], dtype=int)  # nodes 0,1,2 = class 1

    E = mesh_edges(F)
    flab = vertex_to_face(F, vlab)
    elab = vertex_to_edge(E, vlab)
    back = face_to_vertex(F, flab, len(V))

    print("edges       :", E.tolist())
    print("face labels :", flab.tolist(), " (tri 0 all class 1 -> 1, tri 1 mixed -> 0)")
    print("edge labels :", elab.tolist())
    print("back to vert:", back.tolist(), " (node 1 only in tri 0 -> class 1; nodes 0,2 tie -> bg)")
    assert flab.tolist() == [1, 0]
    assert back[1] == 1 and back[3] == 0
    print("[ok] label conversion node<->edge<->face passed")


def _demo_edge_features():
    """Test MeshCNN edge features on a simple two-triangle mesh."""
    V = np.array([[0, 0, 0], [1, 0, 0], [0.5, 1, 0], [0.5, 1, 0.5]], dtype=float)
    F = np.array([[0, 1, 2], [0, 1, 3]], dtype=int)  # shared edge (0,1)
    edges, feats, nbrs = edge_features(V, F)

    # Find the index of edge (0, 1)
    ei = -1
    for i, (a, b) in enumerate(edges):
        if {a, b} == {0, 1}:
            ei = i
            break

    print("\nedges:", edges.tolist())
    print(f"inner edge (0,1) feature = {np.round(feats[ei], 3)}")
    print(f"   dihedral={np.degrees(feats[ei, 0]):.1f} deg  (two triangles bent)")
    print(f"   neighbor edges (a,b,c,d) = {nbrs[ei].tolist()}")
    assert feats.shape[1] == 5
    assert feats[ei, 0] > 0.0, "bent edge should have dihedral > 0"
    assert (nbrs[ei] >= 0).sum() == 4, "inner edge has 4 ring-neighbors"
    print("[ok] MeshCNN edge features passed")


if __name__ == "__main__":
    _demo_labels()
    _demo_edge_features()
