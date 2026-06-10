# run_on_file.py: connector pipeline on a mesh file
#
# Reference: Scheffler (2022), §5.3.5, §5.3.6, §5.3.7
#   §5.3.5 / Abb.43  – instance segmentation by graph connectivity (invoked here)
#   §5.3.6 / Abb.44  – centroid / normal computation (invoked here)
#   §5.3.7           – CLI entry point for the full inference pipeline

import os, sys, logging, argparse
import numpy as np
from meshio import load_mesh, load_labels, save_off, save_labels
from visualization import export_scene_obj
from config import PipelineConfig
from connector3d import (
    run, save_connector_graph, LABEL_NAMES,
    CONTACT,
)

HERE = os.path.dirname(os.path.abspath(__file__))
logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-8s  %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)


def _parser():
    p = argparse.ArgumentParser(description="Connector-3D: mesh+labels -> connector graph JSON",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="examples:\n  python run_on_file.py\n  python run_on_file.py part.off part.labels\n  python run_on_file.py part.off part.labels --out g.json --no-scene\n  python run_on_file.py --batch list.txt\n")
    p.add_argument("mesh", nargs="?", default=None)
    p.add_argument("labels", nargs="?", default=None)
    p.add_argument("--out", default=None)
    p.add_argument("--batch", default=None)
    p.add_argument("--config", default=None)
    p.add_argument("--min-vertices", type=int, default=None)
    p.add_argument("--gap-frac", type=float, default=None)
    p.add_argument("--angle-max", type=float, default=None)
    p.add_argument("--no-scene", action="store_true")
    p.add_argument("--save-config", default=None)
    p.add_argument("-v", "--verbose", action="store_true")
    return p


def _make_testpart(res=70, size=1.0, cavity_r=0.16, depth=0.22):
    xs = np.linspace(0, size, res); cx = cy = size/2
    V = np.array([(x, y, (-depth*(1-np.hypot(x-cx,y-cy)/cavity_r) if np.hypot(x-cx,y-cy)<cavity_r else 0.)) for y in xs for x in xs], dtype=float)
    a = np.arange(res*res).reshape(res, res)
    tl = a[:-1,:-1].ravel(); tr = a[:-1,1:].ravel(); bl = a[1:,:-1].ravel(); br = a[1:,1:].ravel()
    F = np.column_stack([tl,tr,bl,bl,tr,br]).reshape(-1, 3)
    x, y = V[:,0], V[:,1]; r = np.hypot(x-cx, y-cy)
    L = np.zeros(len(V), dtype=int)
    L[r<cavity_r] = 3; L[(y<.12)&(x>.10)&(x<.30)] = 2; L[(y<.12)&(x>.36)&(x<.56)] = 2
    L[(y<.12)&(x>.74)&(x<.92)] = 2; L[(x<.16)&(y>.84)] = 1; L[(x>.42)&(x<.58)&(y>.84)] = 1
    L[(x>.84)&(y>.84)] = 1; L[(x>.80)&(y>.46)&(y<.60)] = 4
    return V, F, L


def _run_single(mesh_path, label_path, out_json, cfg):
    logger.info("Loading: %s", mesh_path)
    try:
        V, F = load_mesh(mesh_path); L = load_labels(label_path)
    except Exception as exc:
        logger.error("Load failed: %s", exc); return False
    if len(L) != len(V):
        logger.warning("Labels (%d) != vertices (%d)", len(L), len(V))
    logger.info("Running pipeline (min_v=%d, gap=%.3f, angle=%.1f)", cfg.min_vertices, cfg.gap_frac, cfg.angle_max_deg)
    try:
        instances, frags = run(V, F, L, min_vertices=cfg.min_vertices, gap_frac=cfg.gap_frac,
                               angle_max_deg=cfg.angle_max_deg, surface_subdiv=cfg.surface_subdiv,
                               cfg=cfg)
    except Exception as exc:
        logger.error("Pipeline error: %s", exc, exc_info=True); return False
    logger.info("Fragments: %d  ->  Instances: %d", len(frags), len(instances))
    for i in instances:
        tgt = f"  -> {LABEL_NAMES[i.points_to[0].label]}" if i.points_to else ""
        logger.info("  - %-22s pos=%s n=%s size=%.3f frag=%d%s", LABEL_NAMES[i.label],
                    np.round(i.center,3), np.round(i.normal3d,2), i.size, len(i.fragments), tgt)
    contacts = [x for x in instances if x.label == CONTACT]
    if len(contacts) >= 2:
        logger.info("Wiring candidates (geometry only, not netlist):")
        for c in contacts:
            t = ", ".join(f"{np.round(w.dst.center,2)} (d={w.distance:.3f})" for w in c.wires)
            logger.info("  Contact @ %s  ->  %s", np.round(c.center,2), t or "(none)")
    try:
        graph = save_connector_graph(instances, out_json)
        logger.info("Graph: %s (%d nodes, %d links)", os.path.basename(out_json), len(graph["nodes"]), len(graph["links"]))
    except Exception as exc:
        logger.error("Save graph failed: %s", exc); return False
    if cfg.write_scene:
        scene = os.path.splitext(mesh_path)[0] + "_scene.obj"
        try:
            export_scene_obj(scene, instances, V, F)
            logger.info("Scene: %s", os.path.basename(scene))
        except Exception as exc:
            logger.warning("Scene failed: %s", exc)
    return True


def _run_batch(list_path, cfg):
    if not os.path.isfile(list_path):
        logger.error("Batch list not found: %s", list_path); sys.exit(1)
    jobs = []
    with open(list_path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#"): continue
            parts = line.split()
            if len(parts) < 2: logger.warning("Skipping line: %r", line); continue
            jobs.append((parts[0], parts[1], parts[2] if len(parts)>=3 else os.path.splitext(parts[0])[0]+"_connector.json"))
    logger.info("Batch: %d jobs", len(jobs))
    ok = err = 0
    for m, l, o in jobs:
        if _run_single(m, l, o, cfg): ok += 1
        else: err += 1
    logger.info("Batch done: %d ok, %d errors", ok, err)
    if err: sys.exit(1)


def _selftest(cfg):
    # synthetic ~1-unit test part -> disable mm-scale robot thresholds so points
    # validate (see ConnectionPointDetectorConfig.for_demo).
    import dataclasses
    cfg = dataclasses.replace(cfg, min_feature_area_mm2=0.0, robot_tool_clearance_mm=0.0,
                              min_insertion_depth_mm=0.0, required_confidence_score=0.0)
    V, F, L = _make_testpart()
    save_off(os.path.join(HERE, "_selftest.off"), V, F)
    save_labels(os.path.join(HERE, "_selftest.labels"), L)
    logger.info("Selftest: synthetic mesh (%d v, %d f)", len(V), len(F))
    _run_single(os.path.join(HERE, "_selftest.off"), os.path.join(HERE, "_selftest.labels"),
                os.path.join(HERE, "_selftest_connector.json"), cfg)


def main():
    args = _parser().parse_args()
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    cfg = PipelineConfig.from_json(args.config) if args.config else PipelineConfig()
    if args.min_vertices is not None: cfg.min_vertices = args.min_vertices
    if args.gap_frac is not None: cfg.gap_frac = args.gap_frac
    if args.angle_max is not None: cfg.angle_max_deg = args.angle_max
    if args.no_scene: cfg.write_scene = False
    if args.save_config: cfg.save(args.save_config)
    if args.batch: _run_batch(args.batch, cfg)
    elif args.mesh and args.labels:
        out = args.out or os.path.splitext(args.mesh)[0] + "_connector.json"
        if not _run_single(args.mesh, args.labels, out, cfg): sys.exit(1)
    else: _selftest(cfg)


if __name__ == "__main__":
    main()
