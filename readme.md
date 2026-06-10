# WiringRobot.ConnectionPointDetector (`wiringrobot-cpd`)

Production-oriented pipeline that turns a connector mesh into **robot-ready
connection points** for an automated wiring robot.

Takes the per-vertex label output of a DiffusionNet segmentation model (plus
optional YOLOv6 rail detections) and produces a structured connector graph —
now enriched with the geometry a robot controller needs: an outward-pointing
approach vector, an entry point, insertion depth, a confidence score and a
robot-validity flag.

> Historically this package was named `connector3d`. The original names
> (`Fragment`, `FeatureInstance`, `PipelineConfig`, `LABEL_NAMES`, …) still work
> as aliases, so existing scripts and saved configs keep running unchanged.

### Main entry point

```python
from infer_pipeline import ConnectionPointDetector

det = ConnectionPointDetector(checkpoint="best.pt", device="cuda")
result = det.detect("part.step", detections="preds.json", out_json="graph.json")
for cp in result["instances"]:        # list[ConnectionPoint]
    if cp.is_valid_for_robot:
        print(cp.label, cp.entry_point, cp.approach_vector, cp.insertion_depth_mm)
```

Each graph node carries robot-ready fields alongside the thesis centroids:

```json
{
  "label": "CableEntry", "robot_class": "CableEntry",
  "position": [x, y, z],          "normal": [nx, ny, nz],
  "entry_point": [x, y, z],       "insertion_axis": [nx, ny, nz],
  "approach_vector": [nx, ny, nz],
  "insertion_depth_mm": 2.4,      "insertion_depth_axis_mm": 2.9,
  "confidence_score": 0.92,       "normal_stability": 0.81,
  "is_valid_for_robot": true,     "terminal_type": "cable_gland"
}
```

---

## Where this fits — the full pipeline

The connector graph is the **last** stage of a longer pipeline. The earlier
stages (data prep, training, 2D→3D cropping) are now scaffolded in this repo too:

```
 CAD dataset            training                 inference                 this repo's core
 ───────────            ────────                 ─────────                 ────────────────
 STEP ─┐                                     6× ortho views (512²)
       │  dataset_convert.py    hpo_ray.py    │                           connector3d.py
 STL ──┼──►  STEP→STL→OBJ  ──►  DiffusionNet  │  YOLOv6 boxes (2D)         feature_geometry.py
       │    (~200 GB, 46 068    HPO via       │       │                    viz.py
 OBJ ──┘     files, 75 mfrs)    Ray Tune      │       ▼  projection_2d3d.py
                                              │   2D box → 3D slab → crop ──►  fragments → instances
                                              ▼       (cut the part out)       → normals → wiring
                                          per-vertex labels  ───────────────►  → connector graph JSON
```

| Stage | Module | What it does |
|---|---|---|
| **1. Dataset** | `dataset_convert.py` | Convert the CAD corpus (STEP → STL → OBJ) and report metrics (≈200 GB, 46 068 files, 75 manufacturers). |
| **2. Training / HPO** | `hpo_ray.py` | Tune DiffusionNet hyperparameters (learning rate, # diffusion blocks, …) with **Ray Tune** — parallel trials across GPUs, early-stop bad ones. MeshCNN/YOLOv6 use literature defaults. |
| **3. 2D→3D crop** | `projection_2d3d.py` | The workaround: YOLOv6 draws 2D boxes on 6 rendered views; each box is stretched through the model into a 3D box; the enclosed part is cut out and its center of gravity / vectors computed. |
| **4. Feature graph** | `connector3d.py` + friends | Turn per-vertex labels into a connector graph (this was the original repo). |

> The 2D→3D crop exists because the end-to-end network did not localize features
> reliably on the raw 3D mesh. See the module header in `projection_2d3d.py`.

---

## Setup

### Option A — Conda (recommended, uses Anaconda / Miniconda)

```bash
# 1. create and activate the environment (installs Python, numpy, setuptools, etc.)
conda env create -f environment.yml
conda activate connector3d

# 2. install the package in editable mode (changes apply immediately)
pip install -e .
```

> **Anaconda Navigator users:** go to *Environments → Import* and select `environment.yml`.

### Option B — Plain pip (no conda)

```bash
# 1. create a virtual environment
python -m venv .venv

# Windows
.venv\Scripts\activate
# Mac / Linux
source .venv/bin/activate

# 2. install dependencies
pip install -r requirements.txt

# 3. install the package in editable mode
pip install -e .
```

> The requirements file is `requirements.txt` (the older `requriements.txt`
> typo has been fixed).

---

## Usage

**Selftest** (no files needed — builds a synthetic mesh):
```bash
python run_on_file.py
# or, after pip install -e .:
run-connector
```

**Single file:**
```bash
python run_on_file.py part.off part.labels
python run_on_file.py part.off part.labels --out graph.json --no-scene
python run_on_file.py part.off part.labels --config my_config.json
```

**Batch (multiple files at once):**
```bash
# file_list.txt: one "mesh.off labels.labels" pair per line
python run_on_file.py --batch file_list.txt
```

**End-to-end inference** (mesh → DiffusionNet → connector graph):
```bash
# selftest: synthetic part + synthetic segmenter + fake YOLO, no torch needed
python infer_pipeline.py

# DiffusionNet-only: needs a trained checkpoint (see Training below)
python infer_pipeline.py part.off  --checkpoint best.pt --out graph.json
python infer_pipeline.py part.step --checkpoint best.pt --scale 6000
```

**Hybrid DiffusionNet + YOLO** (YOLO splits the snap-point rails the net merges).
Two steps, because YOLOv6 runs in its own repo:
```bash
# 1) render the 6 views the detector needs
python infer_pipeline.py part.off --render-views ./views --scale 6000

# 2) run meituan/YOLOv6 on ./views, then feed its predictions back in:
python infer_pipeline.py part.off --checkpoint best.pt --scale 6000 --yolo preds.json
```
DiffusionNet handles every class except SnapPoint; each YOLO box becomes
exactly one SnapPoint instance, so two rails stay two instances even when the
net would merge them.

**Headless Blender** (two modes):
```bash
# (a) extract vertex-group labels → .labels tensor — open the .blend you labelled
blender part.blend --background --python blender_export.py -- \
    --extract-labels part.labels

# (b) run the pipeline + colour the mesh by class for viewing
blender --background --python blender_export.py -- \
    --mesh part.off --labels part.labels --out graph.json
```

---

## Upstream stages

### Dataset conversion & metrics (`dataset_convert.py`)

```bash
# dataset metrics (file count, total size, manufacturers, formats)
python dataset_convert.py stats ./raw_data

# convert one file (STEP needs a CAD kernel; STL → OBJ does not)
python dataset_convert.py file part.step --out ./obj

# convert a whole tree and write a manifest.csv
python dataset_convert.py tree ./raw_data --out ./obj --manifest m.csv
```

`STL → OBJ` is pure numpy (via `meshio`). `STEP → STL` needs a CAD kernel —
install one with `pip install -e ".[cad]"` (gmsh) or use FreeCAD/OpenCASCADE.

### Hyperparameter optimization (`hpo_ray.py`)

```bash
# show the DiffusionNet search space + run the offline demo (no Ray needed)
python hpo_ray.py
```

```python
from hpo_ray import run_hpo, DIFFUSIONNET_SPACE

def train_diffusionnet(config):           # your real training loop
    from ray import tune
    for epoch in range(config.get("epochs", 100)):
        val_iou = ...                      # train one epoch
        tune.report({"val_iou": val_iou})  # Ray decides whether to keep going

best = run_hpo(train_diffusionnet, DIFFUSIONNET_SPACE,
               num_samples=20, gpus_per_trial=1, scheduler="asha")
```

Needs Ray Tune: `pip install -e ".[hpo]"`. Without it, `python hpo_ray.py`
falls back to a small local random search just so the harness runs.

**Training knobs in `diffusionnet.py`** (all read from the `config` dict, single
defaulting point via `DEFAULTS`):

| key | default | what it does |
|---|---|---|
| `dropout` | `0.3` | the **actual** rate — applied via `set_dropout_rate`, not the bool toggle diffusion-net exposes |
| `tversky_weight` | `0.0` | weight of a Tversky overlap term added to NLL/CE; `>0` optimizes IoU/Dice directly and lifts the rare SnapPoint class |
| `tversky_alpha` / `tversky_beta` | `0.3` / `0.7` | FP vs **FN** weighting (`beta>alpha` punishes missed rare-class vertices harder) |
| `accum_steps` | `1` | gradient accumulation (meshes can't be batched into one tensor) |

The training loop keeps all spectral operators on the **CPU** and moves one mesh
to the GPU at a time (then frees it), so a full `n_eig=1024` search no longer
OOMs an 11 GB card; class weights are normalized to mean 1 and the sample order
is shuffled each epoch.

### 2D→3D projection / crop (`projection_2d3d.py`)

```bash
# round-trip self-test (project a feature, invert the box, crop it back out)
python projection_2d3d.py

# full chain on a synthetic part: fake YOLOv6 detections → crop → connector graph
python demo_projection.py
```

```python
from projection_2d3d import load_detections, detections_to_crops, world_aabb
from meshio import load_mesh

V, F = load_mesh("part.obj")
bounds = world_aabb(V, pad_frac=0.02)          # same frame used to render the 6 views
dets = load_detections("part_yolo.json")       # YOLOv6 output (json or yolo-txt)
crops = detections_to_crops(V, F, dets, bounds=bounds, conf_thresh=0.25)
for c in crops:
    print(c.detection.cls, c.cog["center"], c.cog["axes"][0])
```

> **View convention matters.** Render the 6 input views with this module's
> `project_points()` / `render_view()` so the inverse box→slab mapping lines up.

> **Occlusion / back-face bleed.** The slab is depth-unbounded, so a 2D box drags
> through the *whole* part and a feature on the opposite face that projects to the
> same box gets cropped too. For SnapPoint rails (full depth) that's correct
> and the default. For flat one-sided features, pass `front_band_frac=0.25`
> (`detections_to_crops` / `yolo_to_crops`) to keep only the front depth layer.

---

## Tuning parameters

Pass a JSON config file to override any default:

```json
{
  "min_vertices":    15,
  "gap_frac":        0.05,
  "angle_max_deg":   30.0,
  "k_wires":         3,
  "write_scene":     true,

  "view_res":        512,
  "det_conf_thresh": 0.25,
  "crop_pad_frac":   0.0,
  "crop_face_mode":  "all",

  "required_confidence_score":  0.75,
  "robotic_approach_distance_mm": 5.0,
  "min_insertion_depth_mm":     1.0,
  "insertion_depth_metric":     "thesis",
  "max_feature_skew_deg":       75.0,
  "min_depth_extent_mm":        0.5
}
```

The last five are the **robotic** parameters: the minimum confidence/geometry
quality a connection point must reach to be flagged `is_valid_for_robot`, the
safe tool standoff distance, the minimum insertion depth for a secure
connection, the maximum principal-axis skew before a feature's geometry is
rejected, and a floor on slab thickness that prevents degenerate crops on flat
parts.

> **Units matter — the robotic parameters are millimetres.** The pipeline never
> rescales the input, so `is_valid_for_robot` / `insertion_depth_mm` are only
> meaningful if the mesh coordinates are already in mm. `connector3d.run()` now
> sanity-checks the bounding-box diagonal (`check_mesh_scale`) and **warns** when
> it falls outside a plausible `[3, 2000] mm` connector-part range (normalized /
> unit-scale CAD, metres, …). Either rescale the mesh to mm, or disable the
> mm checks by setting `min_feature_area_mm2` / `robot_tool_clearance_mm` /
> `min_insertion_depth_mm` to `0` (what `ConnectionPointDetectorConfig.for_demo()`
> does for the unit-scale selftests).

> **Two insertion-depth measures.** Every node now reports both
> `insertion_depth_mm` — the thesis value `|v_o − v_s|` (Scheffler §5.3.6 /
> Abb.44), kept unchanged — and `insertion_depth_axis_mm`, the *physical* travel
> span of the feature along the insertion axis (the actual distance the tool
> moves inward). The axis measure is more robust to asymmetric or off-centre
> openings, where the centroid-to-centroid proxy under-reads. `min_insertion_depth_mm`
> gates `is_valid_for_robot` on whichever one `insertion_depth_metric` selects
> (`"thesis"` by default for backward compatibility; `"axis"` to use the physical
> span).

The last four control the upstream 2D→3D crop (`projection_2d3d.py`): render
resolution, the YOLOv6 confidence cutoff, how much to widen each box, and whether
a face needs all three / any vertices inside the 3D box to be kept.

```bash
python run_on_file.py part.off part.labels --config my_config.json

# save the current defaults to a file for reference:
python run_on_file.py --save-config defaults.json
```

Or set individual values on the command line:
```bash
python run_on_file.py part.off part.labels --gap-frac 0.08 --angle-max 25
```

---

## Output

| File | Description |
|---|---|
| `*_connector.json` | Connector graph: nodes (robot-ready connection points: entry point, approach vector, insertion depth, confidence, validity) + links (wiring candidates, feature links) |
| `*_scene.obj` | 3D scene with normals, robot **insertion paths** and links as lines — open in Blender or MeshLab |

---

## Module overview

| File | Purpose |
|---|---|
| `connector3d.py` | Core pipeline: fragment detection, merging, wiring |
| `feature_geometry.py` | Normal vector estimation via boundary-peeling |
| `meshio.py` | Load/save `.off`, `.obj`, `.stl` meshes + label files |
| `viz.py` | Export `.obj` scene for visualization |
| `config.py` | `PipelineConfig` dataclass — all tunable parameters |
| `run_on_file.py` | CLI entry point (single file, batch, selftest) |
| `projection_2d3d.py` | 2D→3D workaround: YOLOv6 box → 3D slab → mesh crop → center of gravity/vectors |
| `render6.py` | Real numpy triangle rasterizer → six greyscale 512² views (YOLOv6 input) |
| `remesh.py` | Subdivision (simple/Loop), decimation, scale-to-~6000-verts, label upscaling |
| `augment.py` | Augmentation: rotations, log-normal noise (σ=0.005), Laplace/Taubin smoothing, ARAP |
| `mesh_labels.py` | Node ↔ edge ↔ face label conversion |
| `mesh_edges.py` | MeshCNN 5-dim edge descriptor + edge neighbourhood (invariant conv) |
| `instances.py` | Instance separation: graph connectivity (+ boundary peel) and k-Means |
| `metrics.py` | Segmentation metrics: Dice, Jaccard/IoU (per-class, mean, weighted), accuracy |
| `dataset_convert.py` | CAD dataset prep: STEP→STL→OBJ, metrics, SHA-256 dedup, category filter, **300 s/file timeout** |
| `coco_labels.py` | Label Studio / YOLO pseudo-labels → **COCO JSON** for YOLOv6 training |
| `hpo_ray.py` | DiffusionNet HPO search space + Ray Tune harness |
| `diffusionnet.py` | DiffusionNet model + weighted-NLL training + **checkpoint save/load + `predict()`** |
| `meshcnn.py` | Compact MeshCNN with the invariant edge convolution |
| `yolo6.py` | YOLOv6 pipeline glue: render → detect → 3D crop (snap-point only) |
| `infer_pipeline.py` | **End-to-end inference**: mesh → DiffusionNet (+ optional YOLO snap-point) → connector graph |
| `blender_export.py` | Headless Blender: **vertex-group → label extraction** + label-colour viz |
| `demo_projection.py` | End-to-end demo of the 2D→3D crop feeding the connector pipeline |

### Coverage & honesty

The full pipeline is scaffolded, including the connecting glue that used to be
missing. What runs purely on numpy is fully implemented and self-tested (each
module's `python <mod>.py`):
preprocessing (`remesh`, `augment`, `render6`, `mesh_labels`, `mesh_edges`),
instance separation (`instances`), metrics (`metrics`), dataset hygiene
(`dataset_convert`), 2D-label → COCO conversion (`coco_labels`), the end-to-end
inference glue (`infer_pipeline`, run with a synthetic segmenter), and the whole
post-processing/feature-geometry core.

The inference chain is now runnable end to end: train a DiffusionNet with
`train_diffusionnet(..., checkpoint_path="best.pt")`, then
`python infer_pipeline.py part.off --checkpoint best.pt` (mesh → `predict()` →
per-vertex labels → connector graph). The hybrid DiffusionNet+YOLO path is wired
in too (`--yolo preds.json`): YOLO 2D→3D crops replace the net's SnapPoint
predictions so each rail is its own instance. The Blender side
(`blender_export.py`) **extracts** vertex-group labels into a `.labels` tensor,
not only colours them for viewing.

The real DiffusionNet never emits perfect labels, so `infer_pipeline.py`'s
selftest also runs a **label-noise robustness check**: it perturbs 5/15/30 % of
the synthetic segmentation and confirms the post-processing degrades gracefully
(stays crash-free, drops weak points from `is_valid_for_robot`, keeps the strong
features) instead of only ever seeing ground-truth labels. This characterizes the
post-processing's tolerance — it is **not** a substitute for evaluating real
segmentation accuracy, which still depends on a trained checkpoint and real data.

Three things still legitimately simplify or defer, exactly as in the thesis:
- **MeshCNN mesh-pooling/-unpooling and the sparse MedMeshCNN variant**
  are not rebuilt — only the invariant edge convolution is. This is the slow,
  memory-heavy part the thesis itself recommends against.
- **The networks** — `diffusionnet.py` / `meshcnn.py` integrate PyTorch (and the
  upstream `diffusion-net` package); they import without torch and raise a clear
  message when it's missing. `yolo6.py` is glue around the official
  `meituan/YOLOv6` repo — the rendering→detection→3D-crop bridge is real and
  tested, only the network inference itself is external.
- **Watertight 2-manifold remeshing** — `remesh.watertight_manifold()` shells out
  to Manifold/ManifoldPlus (C++); the subdivision/decimation/scaling around it is
  pure numpy.

---

## Dependencies

| Package | Why |
|---|---|
| `numpy>=1.24` | Mesh math, normals, clustering |
| `setuptools>=65.0` | Required by `setup.py` and `pip install -e .` |

The core pipeline (and the 2D→3D crop and STL→OBJ conversion) need **only numpy**.
The upstream stages have optional, heavier extras:

| Extra | Install | Needed for |
|---|---|---|
| `dev` | `pip install -e ".[dev]"` | `pytest`, `black`, `flake8` |
| `hpo` | `pip install -e ".[hpo]"` | Ray Tune (`hpo_ray.py`) |
| `cad` | `pip install -e ".[cad]"` | gmsh — STEP→STL in `dataset_convert.py` (STL→OBJ works without it) |

---

## Note on wiring

The `wire_candidate` links in the output are **geometry-only candidates** —
they indicate which contacts can physically see each other based on their
positions and normal orientations. They are **not** the electrical netlist.
The actual wiring must come from a circuit diagram and be matched against
these candidates separately.

---

## Thesis References

Scheffler, B. (2022). *Semantische Segmentierung von Dreiecksnetzen durch
Geometrisches Deep Learning*. Master's Thesis, FAU Erlangen-Nürnberg.

Each module implements the following primary thesis sections:

| Module | Primary thesis sections | Topic |
|---|---|---|
| `augment.py` | §5.1.2, §5.3.3 | Lognormal noise (σ=0.005, Abb.31), Laplace/Taubin/average smoothing (Abb.32), ARAP deformation (Abb.32, Sorkine & Alexa 2007), random rotation matrix; cotangent Laplacian |
| `blender_export.py` | §5.2.1 | Blender vertex-group labeling → 1-D label tensor; 5-class label convention |
| `coco_labels.py` | §5.2.2 | YOLO/COCO bounding-box labels, 2D convention (Code 4): CableEntry=0, Contact=1, LabelSurface=2, SnapPoint=3 |
| `config.py` | §5.3.4, §5.3.5, §5.3.7 | Pipeline parameters: min_vertices, gap_frac, view_res=512, det_conf_thresh |
| `connector3d.py` | §5.3.5, §5.3.6, §2.6.1/2.6.2 | Instance segmentation by graph connectivity (Union-Find, Abb.43); cluster centroid, convex-hull volume centroid, opening midpoint, normal = v_o−v_s (Abb.44); feature size = sum of triangle areas |
| `connector_constants.py` | §5.2.1 | 5-class label convention: Housing(0), Contact(1), SnapPoint(2), CableEntry(3), LabelSurface(4) |
| `dataset_convert.py` | §5.1.1, §5.1.2 | Web scraping / data aggregation; STEP→STL→OBJ conversion pipeline; SHA-256 dedup; category filter; 300 s/file timeout |
| `demo_all.py` | §5.3.5, §5.3.6, §5.3.7 | Integration demo for instance segmentation, centroid/normal computation, E2E inference |
| `diffusionnet.py` | §5.3.3, §5.4.1, §6.2.1, §6.2.2, §6.2.3, §2.2.3 | DiffusionNet: MLP + diffusion layer + spatial gradient (Abb.40); HKS vs XYZ input; NLL/CE loss; HPO search space (Tabelle 3); class weights (Tabelle 4); best hyperparameters (Tabelle 5); hybrid inference |
| `feature_geometry.py` | §5.3.6 | Convex-hull volume centroid (Abb.44), boundary peeling → surface centroid (v_s) + opening midpoint (v_o), normal aligned to bbox axes |
| `hpo_ray.py` | §5.4.1, §5.4.2, §6.2.1, §6.2.2 | HPO search space Tabelle 3 (lr, k_eig, n_blocks, dropout, …); Ray Tune + ASHA scheduler; class loss weights Tabelle 4; best config Tabelle 5 |
| `infer_pipeline.py` | §5.3.7, §6.2.3 | Full inference pipeline: STEP→mesh→DiffusionNet+YOLOv6→instance→graph (Abb.45,46); hybrid DN+YOLO (SnapPoint only) |
| `instances.py` | §5.3.5 | Instance segmentation by graph connectivity + boundary peeling (Abb.43); k-Means instance segmentation |
| `mesh_utils.py` | §5.3.1, §5.2.1, §2.6.1/2.6.2, §4.2.2 | MeshCNN 5-dim invariant edge features: dihedral angle, 2 inner angles, 2 edge-length ratios; sorted pairs (Formel 5.3, Abb.37,38); vertex→edge label conversion; NumPy vectorization |
| `meshcnn.py` | §5.3.1 | MeshCNN invariant edge convolution (Abb.37,38); Mesh Pooling/Unpooling (Abb.39, noted) |
| `meshio.py` | §5.1.2 | STL binary/ASCII (Code 1), OBJ (Code 2), OFF (Code 3) format loaders/writers |
| `metrics.py` | §6.2.1 | Class-weighted IoU/Dice evaluation (compensates housing ~70% imbalance) |
| `projection_2d3d.py` | §5.3.7, §2.4 | 2D bbox → 3D slab crop: stretch bbox through full depth axis (Abb.46); 6 orthographic views convention (Abb.28) |
| `remesh.py` | §5.1.2 | Simple + Loop (smooth) subdivision (Abb.29,30); vertex-cluster decimation; scale to ~6000 vertices; Manifold/ManifoldPlus watertight repair; nearest-neighbor label transfer |
| `run_on_file.py` | §5.3.5, §5.3.6, §5.3.7 | CLI entry point invoking the full connector pipeline |
| `visualization.py` | §5.3.4, §2.4 | 6 grayscale 512×512 orthographic views for YOLOv6 (Abb.41,42); view convention z+/z-/y+/y-/x+/x- (Abb.28) |
| `yolo6.py` | §5.3.4, §6.2.3 | YOLOv6 pipeline glue: render → detect → 3D crop; SnapPoint-only hybrid path (Abb.41,42) |
