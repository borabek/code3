# E2E inference (§5.3.7): mesh -> DN(+YOLO-SnapPoint) -> connector graph
#
# Reference: Scheffler (2022), §5.3.7, §6.2.3
#   §5.3.7 / Abb.45,46 – full inference pipeline: STEP->mesh->DiffusionNet+YOLOv6
#                         ->instance->graph
#   §6.2.3              – hybrid inference: DiffusionNet for all classes except
#                         SnapPoint; YOLOv6 for SnapPoint only

import os, sys, json, logging, argparse, tempfile
import numpy as np
from meshio import load_mesh, save_off
from config import ConnectionPointDetectorConfig, PipelineConfig
from connector3d import (
    run as connector_run, build_connector_graph, LABEL_NAMES,
    DetectionPatch, ConnectionPoint, vertex_normals, face_areas,
    SNAP_POINT, HOUSING, CABLE_ENTRY,
)
from projection_2d3d import project_points, Detection, world_aabb

logger = logging.getLogger(__name__)


def _free_gpu_memory():
    """Release cached GPU memory between heavy stages (no-op on CPU)."""
    try:
        from diffusionnet import free_gpu_memory
        free_gpu_memory()
    except Exception:
        pass


def _instance_confidence(inst, probs):
    """Mean DiffusionNet probability of the instance's own class over its vertices."""
    if probs is None:
        return 1.0
    idx = np.concatenate([f.idx for f in inst.fragments])
    label = int(inst.label)
    if label >= probs.shape[1]:
        return 1.0
    return float(np.clip(probs[idx, label].mean(), 0.0, 1.0))


def _resolve_mesh(mesh_path, workdir=None):
    ext = os.path.splitext(mesh_path)[1].lower()
    if ext in {".step", ".stp"}:
        from dataset_convert import convert_file
        workdir = workdir or os.path.dirname(os.path.abspath(mesh_path))
        obj = convert_file(mesh_path, workdir)
        logger.info("STEP->OBJ: %s", obj)
        return load_mesh(obj)
    return load_mesh(mesh_path)


def diffusionnet_segmenter(checkpoint, device="cpu", with_probs=True):
    from diffusionnet import load_checkpoint, predict
    model, meta, cfg = load_checkpoint(checkpoint, device=device)
    logger.info("DN loaded: %s (%s, k_eig=%s)", checkpoint, meta.get("input_features"), meta.get("k_eig"))
    if with_probs:
        # return (labels, probs) so the pipeline can derive a confidence score
        return lambda V, F: predict(model, meta, V, F, device=device, return_probs=True)
    return lambda V, F: predict(model, meta, V, F, device=device)


# §6.2.3 – YOLOv6 owns the SnapPoint/RailMount class: each detected box becomes
# one RailMount connection point, with its confidence taken from the detector.
def _crops_to_instances(crops, V, F, vn, subdiv=0, body_center=None, cfg=None):
    out = []
    for c in crops:
        if c.detection.cls != SNAP_POINT:
            continue
        idx = np.asarray(c.vertex_index, dtype=int)
        if len(idx) == 0:
            continue
        n = vn[idx].mean(0); nl = np.linalg.norm(n); n = n/nl if nl > 0 else [0,0,1]
        a = float(face_areas(c.vertices, c.faces).sum()) if len(c.faces) else 0.0
        inst = ConnectionPoint([DetectionPatch(idx, V[idx], SNAP_POINT, n, a)])
        inst.compute_direction(V, F, smooth_subdiv=subdiv, body_center=body_center)
        inst.confidence_score = float(c.detection.conf)
        max_skew = getattr(cfg, "max_feature_skew_deg", 75.0) if cfg is not None else 75.0
        inst.robotic_validation(cfg=cfg, max_skew_deg=max_skew)
        out.append(inst)
    return out


# §6.2.3 – hybrid inference: DiffusionNet for all classes except SnapPoint,
#            YOLO for SnapPoint only (each YOLO box -> one SnapPoint instance)
def run_inference_hybrid(mesh_path, segmenter, detections=None, cfg=None,
                         target_vertices=None, out_json=None, write_scene=True,
                         view_res=None, conf_thresh=None, render_dir=None, workdir=None):
    cfg = cfg or ConnectionPointDetectorConfig()
    # fall back to the config values so a --config JSON actually takes effect;
    # an explicit argument still wins.
    view_res = cfg.view_res if view_res is None else view_res
    conf_thresh = cfg.det_conf_thresh if conf_thresh is None else conf_thresh
    V, F = _resolve_mesh(mesh_path, workdir=workdir)
    logger.info("loaded: %d v, %d f", len(V), len(F))

    if target_vertices:
        from remesh import scale_to_target
        V, F = scale_to_target(V, F, target_vertices=target_vertices)
        logger.info("scaled: %d v", len(V))

    if render_dir:
        from yolo6 import render_for_yolo
        render_for_yolo(V, F, out_dir=render_dir, res=view_res)

    # DiffusionNet segmentation. Segmenter may return labels, or (labels, probs)
    # so we can derive a per-instance confidence score.
    seg_out = segmenter(V, F)
    if isinstance(seg_out, tuple):
        labels, probs = seg_out
        probs = np.asarray(probs, dtype=float)
    else:
        labels, probs = seg_out, None
    labels = np.asarray(labels, dtype=int).ravel()
    if len(labels) != len(V):
        raise ValueError(f"segmenter: {len(labels)} labels for {len(V)} vertices")
    _free_gpu_memory()  # release segmentation activations before geometry stage

    crops = []
    if detections is not None:
        from yolo6 import yolo_to_crops, load_yolo_predictions
        dets = load_yolo_predictions(detections, res=view_res) if isinstance(detections, str) else detections
        bnd = world_aabb(V, pad_frac=0.05)
        crops = yolo_to_crops(V, F, dets, bnd, res=view_res, conf_thresh=conf_thresh,
                              snap_point_only=True,
                              min_depth_extent=getattr(cfg, "min_depth_extent_mm", 0.0))
        # §6.2.3 – YOLOv6 is authoritative for SnapPoint/RailMount; drop DN's.
        labels = labels.copy(); labels[labels == SNAP_POINT] = HOUSING
        logger.info("YOLO: %d crops", len(crops))

    logger.info("segmented: %s",
                {LABEL_NAMES[c]: int((labels==c).sum()) for c in range(len(LABEL_NAMES)) if (labels==c).any()})

    inst, frags = connector_run(V, F, labels, min_vertices=cfg.min_vertices,
                                gap_frac=cfg.gap_frac, angle_max_deg=cfg.angle_max_deg,
                                surface_subdiv=cfg.surface_subdiv, cfg=cfg)

    # confidence for DiffusionNet instances = mean softmax prob of their class
    max_skew = getattr(cfg, "max_feature_skew_deg", 75.0)
    for ci in inst:
        ci.confidence_score = _instance_confidence(ci, probs)
        ci.robotic_validation(cfg=cfg, max_skew_deg=max_skew)

    if crops:
        vn = vertex_normals(V, F)
        body_center = V.mean(axis=0)
        yolo_inst = _crops_to_instances(crops, V, F, vn, cfg.surface_subdiv,
                                        body_center=body_center, cfg=cfg)
        inst += yolo_inst
        logger.info("YOLO added: %d instances (from %d crops)", len(yolo_inst), len(crops))

    _free_gpu_memory()
    n_valid = sum(1 for ci in inst if getattr(ci, "is_valid_for_robot", True))
    logger.info("fragments: %d -> instances: %d (%d robot-valid)", len(frags), len(inst), n_valid)
    graph = build_connector_graph(inst)
    if out_json:
        with open(out_json, "w", encoding="utf-8") as fh:
            json.dump(graph, fh, indent=2, ensure_ascii=False)
        logger.info("graph: %s (%d nodes, %d links)", out_json, len(graph["nodes"]), len(graph["links"]))
        if write_scene:
            from visualization import export_scene_obj
            try:
                export_scene_obj(os.path.splitext(out_json)[0] + "_scene.obj", inst, V, F)
            except Exception as exc:
                logger.warning("scene failed: %s", exc)
    return {"graph": graph, "instances": inst, "labels": labels, "vertices": V, "faces": F, "crops": crops}


def run_inference(mesh_path, segmenter, cfg=None, target_vertices=None,
                  out_json=None, write_scene=True, workdir=None):
    return run_inference_hybrid(
        mesh_path, segmenter, detections=None, cfg=cfg,
        target_vertices=target_vertices, out_json=out_json,
        write_scene=write_scene, workdir=workdir)


# ---------------------------------------------------------------------------
# Main entry point for the wiring robot: WiringRobot.ConnectionPointDetector
# ---------------------------------------------------------------------------
class ConnectionPointDetector:
    """High-level detector that turns a connector mesh into robot-ready
    connection points (§5.3.7 / §6.2.3).

    Wraps the hybrid DiffusionNet + YOLOv6 pipeline behind a single object:

        det = ConnectionPointDetector(checkpoint="best.pt", device="cuda")
        result = det.detect("part.step", detections="preds.json", out_json="graph.json")
        points = result["instances"]   # list[ConnectionPoint], robot-ready

    A custom segmenter callable ``(V, F) -> labels`` (or ``(labels, probs)``)
    can be supplied instead of a checkpoint, which is what the selftests use.
    """

    def __init__(self, checkpoint=None, segmenter=None, cfg=None, device="cpu"):
        if segmenter is None and checkpoint is None:
            raise ValueError("ConnectionPointDetector needs a checkpoint or a segmenter")
        self.cfg = cfg or ConnectionPointDetectorConfig()
        self.device = device
        self.checkpoint = checkpoint
        self.segmenter = segmenter or diffusionnet_segmenter(checkpoint, device=device)

    def detect(self, mesh_path, detections=None, target_vertices=None,
               out_json=None, write_scene=True, view_res=None, conf_thresh=None,
               render_dir=None, workdir=None):
        """Run detection on a single part and return the full result dict.
        view_res/conf_thresh default to the config values when left as None."""
        return run_inference_hybrid(
            mesh_path, self.segmenter, detections=detections, cfg=self.cfg,
            target_vertices=target_vertices, out_json=out_json,
            write_scene=write_scene, view_res=view_res, conf_thresh=conf_thresh,
            render_dir=render_dir, workdir=workdir)

    # callable sugar: det(mesh) == det.detect(mesh)
    __call__ = detect


# Short alias matching the TODO ("main Detector class").
Detector = ConnectionPointDetector


def _build_parser():
    p = argparse.ArgumentParser(
        description="E2E inference: mesh -> connector graph",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
examples:
  python infer_pipeline.py                      # selftest
  python infer_pipeline.py part.off --checkpoint best.pt
  python infer_pipeline.py part.step --checkpoint best.pt --scale 6000 --out graph.json
  python infer_pipeline.py part.off --render-views ./views --scale 6000
  python infer_pipeline.py part.off --checkpoint best.pt --yolo preds.json
""",
    )
    p.add_argument("mesh", nargs="?", help="STEP/OBJ/OFF/STL")
    p.add_argument("--checkpoint", help="trained DN")
    p.add_argument("--out", help="output JSON")
    p.add_argument("--scale", type=int, help="resize to N vertices")
    p.add_argument("--device", default="cpu")
    p.add_argument("--no-scene", action="store_true")
    p.add_argument("--config", help="PipelineConfig JSON")
    p.add_argument("--yolo", help="YOLO preds (json/txt)")
    p.add_argument("--render-views", metavar="DIR", help="render 6 views")
    p.add_argument("--view-res", type=int, default=None,
                   help="override config view_res (default: from config)")
    p.add_argument("--conf-thresh", type=float, default=None,
                   help="override config det_conf_thresh (default: from config)")
    p.add_argument("-v", "--verbose", action="store_true")
    return p


def main():
    args = _build_parser().parse_args()
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="%(asctime)s %(levelname)-8s %(message)s", datefmt="%H:%M:%S")
    cfg = PipelineConfig.from_json(args.config) if args.config else PipelineConfig()

    if args.mesh is None:
        _selftest(cfg); return

    if args.checkpoint is None:
        if args.render_views:
            from yolo6 import render_for_yolo
            V, F = _resolve_mesh(args.mesh)
            if args.scale:
                from remesh import scale_to_target
                V, F = scale_to_target(V, F, target_vertices=args.scale)
            render_for_yolo(V, F, out_dir=args.render_views, res=args.view_res or cfg.view_res)
            logger.info("6 views -> %s. Run YOLO, then re-run with --checkpoint --yolo <preds>", args.render_views)
            return
        logger.error("--checkpoint required (or no mesh for selftest).")
        sys.exit(2)

    seg = diffusionnet_segmenter(args.checkpoint, device=args.device)
    out = args.out or os.path.splitext(args.mesh)[0] + "_connector.json"
    run_inference_hybrid(args.mesh, seg, detections=args.yolo, cfg=cfg,
                         target_vertices=args.scale, out_json=out,
                         write_scene=not args.no_scene, view_res=args.view_res,
                         conf_thresh=args.conf_thresh, render_dir=args.render_views)


def _selftest(cfg):
    # The selftest part is a synthetic ~1-unit mesh; disable the mm-scale robot
    # thresholds so its points validate (see ConnectionPointDetectorConfig.for_demo).
    import dataclasses
    cfg = dataclasses.replace(cfg, min_feature_area_mm2=0.0, robot_tool_clearance_mm=0.0,
                              min_insertion_depth_mm=0.0, required_confidence_score=0.0)
    xs = np.linspace(0, 1, 50); cx = cy = 0.5
    V = np.array([(x, y, (-0.2*(1-np.hypot(x-cx,y-cy)/0.16) if np.hypot(x-cx,y-cy)<0.16 else 0.))
                  for y in xs for x in xs], dtype=float)
    a = np.arange(2500).reshape(50, 50)
    f = np.column_stack([a[:-1,:-1].ravel(), a[:-1,1:].ravel(), a[1:,1:].ravel(),
                         a[:-1,:-1].ravel(), a[1:,1:].ravel(), a[1:,:-1].ravel()]).reshape(-1, 3)
    x, y = V[:,0], V[:,1]; r = np.hypot(x-0.5, y-0.5)
    L = np.zeros(len(V), dtype=int)
    L[r<0.16] = 3; L[(y<0.10)&((x>.12)&(x<.34)|(x>.66)&(x<.88))] = 2
    L[((x<.14)|(x>.86))&(y>.86)] = 1
    t = tempfile.mkdtemp(prefix="inf_"); p = os.path.join(t, "t.off"); save_off(p, V, f)
    print("E2E selftest (§5.3.7)")

    def seg(v, _):
        x, y = v[:,0], v[:,1]; r = np.hypot(x-0.5, y-0.5)
        o = np.zeros(len(v), dtype=int)
        o[r<0.16]=3; o[(y<0.10)&((x>.12)&(x<.34)|(x>.66)&(x<.88))]=2
        o[((x<.14)|(x>.86))&(y>.86)]=1; return o

    r1 = run_inference(p, seg, cfg=cfg, out_json=os.path.join(t,"g.json"), write_scene=True)
    c1 = _cl(r1["graph"])
    assert c1.get("CableEntry",0)>=1 and c1.get("SnapPoint",0)==2

    def mrg(v, _):
        o = seg(v, _); x, y = v[:,0], v[:,1]
        o[(y<0.10)&(x>.12)&(x<.88)] = 2; return o

    nd = _cl(run_inference(p, mrg, cfg=cfg, write_scene=False)["graph"]).get("SnapPoint",0)
    b = world_aabb(V, 0.05); d = []
    for x0, x1 in [(.12,.34),(.66,.88)]:
        s = (V[:,1]<0.10)&(V[:,0]>=x0)&(V[:,0]<=x1)
        px = project_points("z+", V[s], b, 256); p0, p1 = px.min(0), px.max(0)
        d.append(Detection("z+", (*p0,*p1), .9, 2))
    nh = _cl(run_inference_hybrid(p, mrg, detections=d, cfg=cfg, view_res=256)["graph"]).get("SnapPoint",0)
    print(f"  DN:{nd}  YOLO:{nh}")
    assert nd == 1 and nh == 2
    print("[OK] E2E verified.")

    # Robustness: a real DiffusionNet never returns perfect labels, so confirm the
    # post-processing degrades gracefully (no crash, strong features survive)
    # rather than only ever seeing ground-truth segmentation.
    print("Robustness under label noise (§6.2.1)")

    def _noisy(frac):
        # flip a fraction of vertex labels to a random class (seeded -> reproducible)
        def _seg(v, _):
            o = seg(v, _).copy()
            rng = np.random.default_rng(0)
            flip = rng.random(len(o)) < frac
            o[flip] = rng.integers(0, len(LABEL_NAMES), size=int(flip.sum()))
            return o
        return _seg

    for frac in (0.05, 0.15, 0.30):
        res = run_inference(p, _noisy(frac), cfg=cfg, write_scene=False)
        insts = res["instances"]
        n_valid = sum(1 for ci in insts if ci.is_valid_for_robot)
        has_entry = any(ci.label == CABLE_ENTRY for ci in insts)
        print(f"  noise {int(frac*100):>3d}%:  {len(insts)} instances, "
              f"{n_valid} robot-valid, CableEntry={'yes' if has_entry else 'no'}")
        # graceful degradation: the pipeline must stay crash-free and still report
        # connection points from the noisy segmentation (we don't pin exact counts).
        assert len(insts) >= 1, f"no instances survived at {frac:.0%} label noise"
    print("[OK] robustness verified (graceful degradation under label noise).")


def _cl(g):
    o = {}
    for n in g["nodes"]: o[n["label"]] = o.get(n["label"], 0) + 1
    return o


if __name__ == "__main__":
    main()
