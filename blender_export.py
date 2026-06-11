# Blender script: import mesh, assign labels, export scene

"""
Headless Blender integration for the connector-3D pipeline.

This script runs *inside* Blender's own Python interpreter and uses bpy to
import the mesh, assign labels, and export the scene. No Blender window is opened.

Reference: Scheffler (2022), §5.2.1
  §5.2.1 – vertex/edge/face label convention, 5 classes:
           Housing(0), Contact(1), SnapPoint(2), CableEntry(3), LabelSurface(4)
  §5.2.1 – Blender vertex-group labeling -> 1-D label tensor

Usage from the terminal:
    blender --background --python blender_export.py -- \\
        --mesh    part.off    \\
        --labels  part.labels \\
        --out     part_connector.json \\
        [--scene  part_scene.obj]     \\
        [--config cfg.json]

Why '--' before our own arguments?
    Everything before '--' goes to Blender; everything after goes to this script.
    Without '--' Blender would reject the unknown flags.

Requirements:
    - Blender >= 3.x (tested with 3.6 LTS and 4.x)
    - numpy must be available in Blender's Python.
      Check with: blender --background --python-expr "import numpy; print(numpy.__version__)"
      If missing: blender --background --python-expr "import pip; pip.main(['install','numpy'])"
    - The other modules (meshio, connector3d, ...) must be in the same directory
      as this script OR CONNECTOR3D_PATH must be set.
"""

from __future__ import annotations

import sys
import os
import argparse
import logging
from typing import Any

# bpy is only available inside Blender's Python. The module stays importable
# without bpy (e.g. for tests/docs) -- only main() will abort with a clear
# message when run outside Blender. Typed as Any so static analysis outside
# Blender (where bpy can't be resolved) doesn't flag every bpy.data/ops/context.
bpy: Any
try:
    import bpy  # type: ignore
    _HAVE_BPY = True
except ImportError:
    bpy = None
    _HAVE_BPY = False

# Alias for the Blender Object type in annotations. bpy has no stubs outside
# Blender, so a `bpy.types.Object` annotation isn't a valid type form to the
# static checker; alias it to Any so the function signatures stay readable.
BlenderObject = Any

# Make connector3d modules findable.
# Priority: 1) CONNECTOR3D_PATH env var  2) same directory as this script
_connector_path = os.environ.get("CONNECTOR3D_PATH", os.path.dirname(os.path.abspath(__file__)))
if _connector_path not in sys.path:
    sys.path.insert(0, _connector_path)

try:
    from meshio import load_mesh, load_labels
    from connector3d import run, save_connector_graph, LABEL_NAMES
    from visualization import export_scene_obj
    from config import PipelineConfig
except ImportError as exc:
    raise SystemExit(
        f"Connector3D module not found: {exc}\n"
        f"Please set CONNECTOR3D_PATH or place this script in the project directory.\n"
        f"  export CONNECTOR3D_PATH=/path/to/project"
    )

# ---- logging ----
logging.basicConfig(
    level=logging.INFO,
    format="[blender_export] %(levelname)s  %(message)s",
)
logger = logging.getLogger(__name__)

# 0=Housing, 1=Contact, 2=SnapPoint, 3=CableEntry, 4=LabelSurface
CLASS_ORDER = ["Housing", "Contact", "SnapPoint", "CableEntry", "LabelSurface"]

# ---- argparse: everything after '--' is treated as our arguments ----
def _parse_args() -> argparse.Namespace:
    """Read arguments from sys.argv after the Blender separator '--'."""
    try:
        separator = sys.argv.index("--")
        raw = sys.argv[separator + 1:]
    except ValueError:
        raw = []

    parser = argparse.ArgumentParser(
        prog="blender --background --python blender_export.py --",
        description="Run the Connector-3D pipeline headless via Blender.",
    )
    # In run mode --mesh/--labels are required; in extract mode they are not
    # (labels are the OUTPUT there). Validation happens in main().
    parser.add_argument("--mesh",   default=None,   help="Path to mesh file (.off or .obj)")
    parser.add_argument("--labels", default=None,   help="Path to labels file (one number per vertex)")
    parser.add_argument("--out",    default=None,   help="Output JSON (default: <mesh>_connector.json)")
    parser.add_argument("--scene",  default=None,   help="Optional .obj scene for blender/meshlab (default: <mesh>_scene.obj)")
    parser.add_argument("--config", default=None,   help="Optional JSON config file")
    parser.add_argument("--no-scene", action="store_true", help="Do not write a .obj scene")

    parser.add_argument("--extract-labels", default=None, metavar="OUT",
                        help="Extract a 1D label tensor from Blender vertex groups "
                             "and write it to OUT (.labels). "
                             "Typical use with an open .blend: "
                             "blender file.blend --background --python blender_export.py -- --extract-labels part.labels")
    parser.add_argument("--object", default=None,
                        help="Name of the mesh object to use (extract mode). "
                             "Default: first/active mesh object in the scene.")
    return parser.parse_args(raw)

# ---- vertex groups -> per-vertex labels ----
def extract_vertex_group_labels(obj, class_order=None, default=0):
    """Build a label tensor from the Blender vertex groups of a mesh object.

    Labels vertices by matching vertex group names to class names. When a
    vertex belongs to multiple groups, the one with the highest weight wins.
    Vertices with no matching group get `default` (housing).

    Returns: numpy array (n_vertices,) of class ids.
    """
    import numpy as np
    class_order = class_order or CLASS_ORDER
    mesh = obj.data
    n = len(mesh.vertices)
    labels = np.full(n, default, dtype=int)

    name_to_id = {name.lower(): i for i, name in enumerate(class_order)}
    group_to_class = {}
    unmatched = []
    for vg in obj.vertex_groups:
        cid = name_to_id.get(vg.name.lower())
        if cid is not None:
            group_to_class[vg.index] = cid
        else:
            unmatched.append(vg.name)
    if unmatched:
        logger.warning("  vertex groups with no class match (ignored): %s", unmatched)

    best_weight = np.full(n, -1.0)
    for vi, vert in enumerate(mesh.vertices):
        for g in vert.groups:
            cid = group_to_class.get(g.group)
            if cid is not None and g.weight >= best_weight[vi]:
                best_weight[vi] = g.weight
                labels[vi] = cid
    return labels

def _pick_mesh_object(name=None):
    """Pick a mesh object from the current Blender scene.

    If name is given, returns that specific object; otherwise returns the
    active object, or the first mesh if nothing is active.
    """
    meshes = [o for o in bpy.data.objects if o.type == "MESH"]
    if not meshes:
        raise SystemExit("No mesh objects found in scene "
                         "(in extract mode, open a .blend with a labeled mesh).")
    if name:
        for o in meshes:
            if o.name == name:
                return o
        raise SystemExit(f"Mesh object '{name}' not found. "
                         f"Available: {[o.name for o in meshes]}")
    active = bpy.context.view_layer.objects.active
    if active is not None and active.type == "MESH":
        return active
    return meshes[0]

def _run_extract(args) -> None:
    """Extract mode: vertex groups -> write .labels file."""
    from meshio import save_labels
    if args.mesh:
        # explicit mesh given -> import it
        obj = _import_mesh_to_blender(args.mesh)
    else:
        obj = _pick_mesh_object(args.object)
    logger.info("Extracting labels from object '%s' (%d vertices, %d vertex groups)",
                obj.name, len(obj.data.vertices), len(obj.vertex_groups))
    labels = extract_vertex_group_labels(obj)
    save_labels(args.extract_labels, labels)
    counts = {CLASS_ORDER[c]: int((labels == c).sum())
              for c in range(len(CLASS_ORDER)) if (labels == c).any()}
    logger.info("Labels written: %s  (%d vertices, classes: %s)",
                args.extract_labels, len(labels), counts)

# ---- import mesh into Blender ----
def _import_mesh_to_blender(mesh_path: str) -> BlenderObject:
    """Import a mesh into the Blender scene and return the object.

    Supported formats: .off (via temp .obj), .obj, .fbx, .stl, .ply.
    .off has no native Blender importer so it is converted to a temp .obj first.
    """
    from meshio import load_mesh
    import tempfile

    ext = os.path.splitext(mesh_path)[1].lower()

    if ext == ".off":
        # .off -> temp .obj -> Blender import
        V, F = load_mesh(mesh_path)
        tmp = tempfile.NamedTemporaryFile(suffix=".obj", delete=False)
        tmp.close()
        _write_obj_simple(tmp.name, V, F)
        logger.info("  .off imported via temp .obj: %s", tmp.name)
        bpy.ops.wm.obj_import(filepath=tmp.name)
        os.unlink(tmp.name)

    elif ext == ".obj":
        bpy.ops.wm.obj_import(filepath=mesh_path)

    elif ext == ".stl":
        bpy.ops.import_mesh.stl(filepath=mesh_path)

    elif ext == ".ply":
        bpy.ops.import_mesh.ply(filepath=mesh_path)

    else:
        raise ValueError(f"Unsupported mesh format for Blender import: {ext}")

    obj = bpy.context.selected_objects[-1]
    logger.info("  Blender object: '%s' (%d vertices)", obj.name, len(obj.data.vertices))
    return obj

def _write_obj_simple(path: str, V, F) -> None:
    """Write a minimal .obj file (vertices + faces only, no deps on viz.py)."""
    with open(path, "w", encoding="utf-8") as fh:
        for v in V:
            fh.write(f"v {v[0]} {v[1]} {v[2]}\n")
        for f in F:
            fh.write(f"f {f[0]+1} {f[1]+1} {f[2]+1}\n")

# ---- write labels as vertex colors into the Blender mesh ----
def _apply_labels_as_vertex_colors(obj: BlenderObject, labels) -> None:
    """Write labels as a vertex color layer so the segmentation is visible in Blender.

    Each class gets a distinct color; this can be used later in the Material Editor.
    """
    # one color per label class (RGBA, 0..1)
    CLASS_COLORS = {
        0: (0.6, 0.6, 0.6, 1.0),  # Housing      -> grey
        1: (1.0, 0.2, 0.2, 1.0),  # Contact       -> red
        2: (0.2, 0.8, 0.2, 1.0),  # SnapPoint     -> green
        3: (0.2, 0.4, 1.0, 1.0),  # CableEntry    -> blue
        4: (1.0, 0.8, 0.1, 1.0),  # LabelSurface  -> yellow
    }
    fallback = (0.9, 0.0, 0.9, 1.0)  # magenta for unknown classes

    mesh = obj.data
    if "connector_labels" not in mesh.vertex_colors:
        mesh.vertex_colors.new(name="connector_labels")
    vcol = mesh.vertex_colors["connector_labels"]

    for poly in mesh.polygons:
        for loop_idx in poly.loop_indices:
            vi = mesh.loops[loop_idx].vertex_index
            lab = int(labels[vi]) if vi < len(labels) else 0
            vcol.data[loop_idx].color = CLASS_COLORS.get(lab, fallback)

    logger.info("  vertex colors set (%d loops)", len(vcol.data))

# ---- export Blender scene as .obj ----
def _export_blender_scene(scene_path: str) -> None:
    """Export the entire Blender scene as .obj."""
    bpy.ops.wm.obj_export(
        filepath=scene_path,
        export_selected_objects=False,
        export_materials=False,
    )
    logger.info("  Blender scene exported: %s", scene_path)

# ---- main ----
def main() -> None:
    if not _HAVE_BPY:
        raise SystemExit(
            "ImportError: bpy not found.\n"
            "This script must be run with Blender:\n"
            "  blender --background --python blender_export.py -- --mesh ... --labels ...\n"
            "  blender file.blend --background --python blender_export.py -- --extract-labels part.labels"
        )
    args = _parse_args()

    # ---- mode 1: label extraction -> write .labels, then done ----
    if args.extract_labels:
        _run_extract(args)
        return

    # ---- mode 2: connector pipeline (needs mesh + labels) ----
    if not args.mesh or not args.labels:
        raise SystemExit(
            "Pipeline mode requires --mesh and --labels.\n"
            "For label extraction use --extract-labels <out.labels> instead.")

    cfg = PipelineConfig.from_json(args.config) if args.config else PipelineConfig()
    if args.no_scene:
        cfg.write_scene = False
    logger.info("Config:\n%s", cfg)

    base = os.path.splitext(args.mesh)[0]
    out_json  = args.out   or (base + "_connector.json")
    out_scene = args.scene or (base + "_scene.obj")

    # ---- step 1: load mesh and labels ----
    logger.info("Loading mesh: %s", args.mesh)
    import numpy as np
    V, F = load_mesh(args.mesh)
    L    = load_labels(args.labels)
    logger.info("  %d vertices, %d faces, %d labels", len(V), len(F), len(L))

    if len(L) != len(V):
        logger.warning(
            "Labels (%d) do not match vertices (%d)! "
            "This may produce incorrect results.",
            len(L), len(V),
        )

    # ---- step 2: run connector pipeline ----
    logger.info("Running connector pipeline ...")
    instances, frags = run(
        V, F, L,
        min_vertices  = cfg.min_vertices,
        gap_frac      = cfg.gap_frac,
        angle_max_deg = cfg.angle_max_deg,
    )
    logger.info("  %d fragments  ->  %d instances", len(frags), len(instances))

    for inst in instances:
        target = ("  -> " + LABEL_NAMES[inst.points_to[0].label]) if inst.points_to else ""
        logger.info("    - %-22s center=%s%s",
                    LABEL_NAMES[inst.label], np.round(inst.center, 3), target)

    # ---- step 3: save connector graph as JSON ----
    graph = save_connector_graph(instances, out_json)
    logger.info("Connector graph written: %s  (%d nodes, %d links)",
                out_json, len(graph["nodes"]), len(graph["links"]))

    # ---- step 4: import mesh into Blender + visualize labels ----
    logger.info("Importing mesh into Blender ...")
    # remove default objects (cube, camera, light)
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete()

    obj = _import_mesh_to_blender(args.mesh)
    _apply_labels_as_vertex_colors(obj, L)

    # ---- step 5: export scene ----
    if cfg.write_scene:
        export_scene_obj(out_scene, instances, V, F)
        logger.info("Connector scene written: %s", out_scene)
        blender_scene = base + "_blender_scene.obj"
        _export_blender_scene(blender_scene)
        logger.info("Blender scene exported: %s", blender_scene)

    logger.info("Done.")

if __name__ == "__main__":
    main()
