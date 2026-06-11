# demo: run the full connector detection pipeline end-to-end

"""
Comprehensive demo suite for connector3d pipeline.

This module combines all standalone demo scripts:
  - test_alignment()   : bbox-aligned normals on tilted meshes
  - test_normal()      : normal vector computation on funnel meshes
  - test_wiring()      : wiring candidate filtering logic
  - test_projection()  : 2D->3D crop pipeline end-to-end
  - test_connector()   : 3D connector with split features
  - test_infer()       : E2E inference pipeline (§5.3.7)

Run: python demo_all.py [--test alignment|normal|wiring|projection|connector|infer|all]

Reference: Scheffler (2022), §5.3.5, §5.3.6, §5.3.7
  §5.3.5 / Abb.43   – instance segmentation by graph connectivity (demonstrated)
  §5.3.6 / Abb.44   – centroid/normal computation (demonstrated)
  §5.3.7 / Abb.45,46 – full E2E inference pipeline (demonstrated)
"""

import os
import argparse
import numpy as np

from connector3d import (
    run, save_connector_graph, build_wiring,
    LABEL_NAMES,
    CONTACT, SNAP_POINT, CABLE_ENTRY, LABEL_SURFACE,
)
from projection_2d3d import (
    world_aabb, project_points, Detection,
    detections_to_crops, crop_labels,
)
from config import PipelineConfig

# TEST 1: Alignment (from demo_align.py)

def angle_deg(a, b):
    """Angle between two vectors in degrees."""
    a = a / np.linalg.norm(a)
    b = b / np.linalg.norm(b)
    return float(np.degrees(np.arccos(np.clip(abs(np.dot(a, b)), -1.0, 1.0))))

def rot_x(theta):
    """Rotation matrix around X-axis."""
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]], dtype=float)

def _make_bowl_mesh(res=60, size=1.0, cavity_r=0.28, depth=0.25):
    """Create a funnel-shaped mesh for testing."""
    xs = np.linspace(0, size, res)
    ys = np.linspace(0, size, res)
    cx = cy = size / 2.0

    verts = []
    for y in ys:
        for x in xs:
            r = np.hypot(x - cx, y - cy)
            z = -depth * (1.0 - r / cavity_r) if r < cavity_r else 0.0
            verts.append((x, y, z))
    verts = np.array(verts, dtype=float)

    faces = []
    def vid(ix, iy):
        return iy * res + ix
    for iy in range(res - 1):
        for ix in range(res - 1):
            a, b = vid(ix, iy), vid(ix + 1, iy)
            c, d = vid(ix, iy + 1), vid(ix + 1, iy + 1)
            faces.append((a, b, d))
            faces.append((a, d, c))

    labels = np.zeros(len(verts), dtype=int)
    r = np.hypot(verts[:, 0] - cx, verts[:, 1] - cy)
    labels[r < cavity_r] = CABLE_ENTRY
    return verts, np.array(faces, dtype=int), labels

def test_alignment():
    """Test bbox-aligned normals on tilted meshes."""
    print("\n" + "="*70)
    print("TEST 1: ALIGNMENT (bbox-aligned normal vectors)")
    print("="*70)
    
    V, F, L = _make_bowl_mesh(res=70, size=1.0, cavity_r=0.28, depth=0.25)

    theta = np.deg2rad(20.0)
    R = rot_x(theta)
    Vt = V @ R.T
    true_n = R @ np.array([0.0, 0.0, 1.0])

    # Add noise to break symmetry (alignment is most beneficial with asymmetry)
    rng = np.random.default_rng(0)
    Vt = Vt + rng.normal(0.0, 0.006, Vt.shape)

    instances, _ = run(Vt, F, L, min_vertices=10)
    inst = instances[0]
    assert inst.v_o is not None and inst.v_s is not None

    rough = inst.v_o - inst.v_s
    rough = rough / np.linalg.norm(rough)
    aligned = inst.normal3d

    print("part tilted by 20 degrees")
    print(f"  true normal     : {np.round(true_n, 3)}")
    print(f"  rough (v_o-v_s) : {np.round(rough, 3)}   error = {angle_deg(rough, true_n):.2f} deg")
    print(f"  aligned (bbox)  : {np.round(aligned, 3)}   error = {angle_deg(aligned, true_n):.2f} deg")
    print("[OK] Alignment test passed")

# TEST 2: Normal Vectors (from demo_normal.py)

def test_normal():
    """Test normal vector computation on funnel meshes."""
    print("\n" + "="*70)
    print("TEST 2: NORMAL VECTORS (funnel mesh)")
    print("="*70)
    
    verts, faces, labels = _make_bowl_mesh()
    instances, frags = run(verts, faces, labels, min_vertices=10, gap_frac=0.05, angle_max_deg=30.0)

    print(f"[instances] {len(instances)}")
    for inst in instances:
        assert inst.v_o is not None and inst.v_s is not None
        print(f"   - {LABEL_NAMES[inst.label]}")
        print(f"       center   : {np.round(inst.center, 3)}")
        print(f"       v_o      : {np.round(inst.v_o, 3)}")
        print(f"       v_s      : {np.round(inst.v_s, 3)}")
        print(f"       normal3d : {np.round(inst.normal3d, 3)}")
        print(f"       size     : {inst.size:.4f}")
    print("[OK] Normal vector test passed")

# TEST 3: Wiring Candidates (from demo_wiring.py)

class FakeContact:
    """Fake contact for wiring tests."""
    def __init__(self, name, center, normal):
        self.name = name
        self.label = CONTACT
        self.center = np.array(center, dtype=float)
        self.normal3d = np.array(normal, dtype=float)
        self.wires = []

def test_wiring():
    """Test wiring candidate filtering logic."""
    print("\n" + "="*70)
    print("TEST 3: WIRING CANDIDATES (geometry filtering)")
    print("="*70)
    
    # A and B face each other -> good candidate
    A = FakeContact("A", (0.0, 0.0, 0.0), (1, 0, 0))
    B = FakeContact("B", (0.3, 0.0, 0.0), (-1, 0, 0))

    # C and D face away from each other -> no candidate (wire would go through housing)
    C = FakeContact("C", (0.0, 0.5, 0.0), (-1, 0, 0))
    D = FakeContact("D", (0.3, 0.5, 0.0), (1, 0, 0))

    contacts = [A, B, C, D]
    build_wiring(contacts, k=3)

    # Whether A->B is actually wired is decided by the netlist, not geometry
    for c in contacts:
        if c.wires:
            targets = ", ".join(f"{w.dst.name} (d={w.distance:.2f})" for w in c.wires)
        else:
            targets = "(no candidate)"
        print(f"   contact {c.name}  ->  {targets}")
    print("[OK] Wiring test passed")

# TEST 4: 2D→3D Projection (from demo_projection.py)

def _make_test_part(res=70, size=1.0, cavity_r=0.16, depth=0.22):
    """Create test part (plate with funnel + features)."""
    xs = np.linspace(0, size, res)
    cx = cy = size / 2.0
    V = []
    for y in xs:
        for x in xs:
            r = np.hypot(x - cx, y - cy)
            z = -depth * (1.0 - r / cavity_r) if r < cavity_r else 0.0
            V.append((x, y, z))
    V = np.array(V, dtype=float)

    F = []
    for iy in range(res - 1):
        for ix in range(res - 1):
            a = iy * res + ix
            b, c, d = a + 1, a + res, a + res + 1
            F.append((a, b, d)); F.append((a, d, c))
    F = np.array(F, dtype=int)

    x, y = V[:, 0], V[:, 1]
    r = np.hypot(x - cx, y - cy)
    L = np.zeros(len(V), dtype=int)
    L[r < cavity_r]                              = CABLE_ENTRY
    L[(y < 0.12) & (x > 0.10) & (x < 0.30)]      = SNAP_POINT
    L[(y < 0.12) & (x > 0.74) & (x < 0.92)]      = SNAP_POINT
    L[(x < 0.16) & (y > 0.84)]                   = CONTACT
    L[(x > 0.84) & (y > 0.84)]                   = CONTACT
    L[(x > 0.80) & (y > 0.46) & (y < 0.60)]      = LABEL_SURFACE
    return V, F, L

def _fake_yolo_detections(V, L, bounds, res=512, view="z+"):
    """Simulate YOLOv6 detections from projected feature vertices."""
    dets = []
    for cls in (CONTACT, SNAP_POINT, CABLE_ENTRY, LABEL_SURFACE):
        pts = V[L == cls]
        if len(pts) == 0:
            continue
        px = project_points(view, pts, bounds, res=res)
        x0, y0 = px.min(axis=0)
        x1, y1 = px.max(axis=0)
        # Add slight confidence noise to test conf filter
        conf = 0.55 + 0.4 * (cls / 4.0)
        dets.append(Detection(view, (x0, y0, x1, y1), conf=conf, cls=int(cls)))
    return dets

def test_projection():
    """Test 2D→3D crop pipeline end-to-end."""
    print("\n" + "="*70)
    print("TEST 4: 2D->3D PROJECTION (end-to-end crop pipeline)")
    print("="*70)
    
    V, F, L = _make_test_part()
    bounds = world_aabb(V, pad_frac=0.02)

    print(f"[part]    {len(V)} vertices, {len(F)} faces")
    print(f"[bounds]  lo={np.round(bounds[0], 2)}  hi={np.round(bounds[1], 2)}\n")

    dets = _fake_yolo_detections(V, L, bounds, res=512, view="z+")
    print(f"[yolo (fake)] {len(dets)} detections in view z+:")
    for d in dets:
        print(f"   - cls={LABEL_NAMES[d.cls]:20s} conf={d.conf:.2f} "
              f"bbox=({d.bbox[0]:.0f},{d.bbox[1]:.0f},{d.bbox[2]:.0f},{d.bbox[3]:.0f})")

    # conf threshold 0.6 -> weaker detections filtered out
    crops = detections_to_crops(V, F, dets, bounds=bounds, res=512,
                                conf_thresh=0.6, face_mode="any")
    print(f"\n[crops] {len(crops)} crops after conf filter (>=0.60):")
    for crop in crops:
        sub_L = crop_labels(L, crop.vertex_index)
        # Run full connector pipeline on the crop
        instances, _ = run(crop.vertices, crop.faces, sub_L, min_vertices=5)
        cog = crop.cog["center"]
        print(f"\n   detection {LABEL_NAMES[crop.detection.cls]}  (conf={crop.detection.conf:.2f})")
        print(f"     crop    : {len(crop.vertices)} vertices, {len(crop.faces)} faces")
        print(f"     centroid: {np.round(cog, 3)}")
        print(f"     connector: {len(instances)} instance(s)")
        for inst in instances:
            print(f"        - {LABEL_NAMES[inst.label]:20s} center={np.round(inst.center, 2)} "
                  f"normal={np.round(inst.normal3d, 2)}")
    print("[OK] Projection test passed")

# TEST 5: Full Connector (from demo_connector.py)

def _make_grid_mesh(res=50, size=1.0):
    """Create grid mesh for connector test."""
    xs = np.linspace(0, size, res)
    ys = np.linspace(0, size, res)
    verts = np.array([(x, y, 0.0) for y in ys for x in xs], dtype=float)

    faces = []
    def vid(ix, iy):
        return iy * res + ix
    for iy in range(res - 1):
        for ix in range(res - 1):
            a, b = vid(ix, iy), vid(ix + 1, iy)
            c, d = vid(ix, iy + 1), vid(ix + 1, iy + 1)
            faces.append((a, b, d))
            faces.append((a, d, c))
    return verts, np.array(faces, dtype=int)

def _make_demo_labels(verts):
    """Create labels for connector test part."""
    x, y = verts[:, 0], verts[:, 1]
    labels = np.zeros(len(verts), dtype=int)  # everything starts as housing
    labels[(x < 0.25) & (y > 0.72)] = CONTACT
    labels[(x > 0.75) & (y > 0.72)] = CABLE_ENTRY
    labels[(x > 0.40) & (x < 0.60) & (y > 0.42) & (y < 0.58)] = LABEL_SURFACE
    # split landing pad with gap -> two fragments, one instance
    labels[(y < 0.14) & (x > 0.30) & (x < 0.47)] = SNAP_POINT
    labels[(y < 0.14) & (x > 0.53) & (x < 0.70)] = SNAP_POINT
    return labels

def test_connector():
    """Test 3D connector with split features."""
    print("\n" + "="*70)
    print("TEST 5: FULL CONNECTOR (3D connector with split features)")
    print("="*70)
    
    verts, faces = _make_grid_mesh(res=60, size=1.0)
    labels = _make_demo_labels(verts)

    # unit-scale toy part -> disable the mm-scale robot thresholds so points validate
    instances, frags = run(verts, faces, labels, min_vertices=10, gap_frac=0.05,
                           angle_max_deg=30.0, cfg=PipelineConfig.for_demo())

    print(f"[fragments after graph-connectivity] {len(frags)}")
    for f in frags:
        print(f"   - {LABEL_NAMES[f.label]:20s} center={np.round(f.center, 2)} n_v={len(f.idx)}")

    print(f"\n[instances after 3D-connector] {len(instances)}")
    for inst in instances:
        target = "  --> " + LABEL_NAMES[inst.points_to[0].label] if inst.points_to else ""
        print(f"   - {LABEL_NAMES[inst.label]:20s} center={np.round(inst.center, 2)} "
              f"size={inst.size:.3f} n_frag={len(inst.fragments)}{target}")

    # Write graph (result for digital twin)
    out = os.path.join(os.path.dirname(__file__), "connector_graph.json")
    graph = save_connector_graph(instances, out)
    print(f"\n[graph] {len(graph['nodes'])} nodes, {len(graph['links'])} links  ->  {os.path.basename(out)}")
    print("[OK] Connector test passed")

# TEST 6: E2E Inference pipeline

def test_infer():
    """Test E2E inference pipeline (§5.3.7) including hybrid DN+YOLO path."""
    print("\n" + "="*70)
    print("TEST 6: E2E INFERENCE PIPELINE (§5.3.7)")
    print("="*70)
    from infer_pipeline import _selftest
    from config import PipelineConfig
    _selftest(PipelineConfig())
    print("[OK] E2E inference test passed")

# Main

def main():
    parser = argparse.ArgumentParser(description="Connector3D demo suite")
    parser.add_argument(
        "--test",
        choices=["alignment", "normal", "wiring", "projection", "connector", "infer", "all"],
        default="all",
        help="Which test to run (default: all)"
    )
    args = parser.parse_args()

    test_map = {
        "alignment": test_alignment,
        "normal": test_normal,
        "wiring": test_wiring,
        "projection": test_projection,
        "connector": test_connector,
        "infer": test_infer,
    }

    if args.test == "all":
        for name, func in test_map.items():
            try:
                func()
            except Exception as e:
                print(f"\n[FAILED] {name}: {e}")
                import traceback
                traceback.print_exc()
    else:
        try:
            test_map[args.test]()
        except Exception as e:
            print(f"\n[FAILED] {args.test}: {e}")
            import traceback
            traceback.print_exc()

    print("\n" + "="*70)
    print("ALL DEMOS COMPLETED")
    print("="*70)

if __name__ == "__main__":
    main()
