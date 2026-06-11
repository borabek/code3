# Pipeline configuration in one place.
# All thresholds you might want to tune are here so you don't have to dig
# through algorithm files to find them.
#
# Usage:
#   cfg = PipelineConfig()                     # defaults
#   cfg = PipelineConfig(gap_frac=0.08)        # change one value
#   cfg = PipelineConfig.from_json("my.json")  # load from file
#
# Reference: Scheffler (2022), §5.3.5, §5.3.4, §5.3.7
#   §5.3.5 – min_vertices threshold, gap_frac, angle_max_deg for instance merging
#   §5.3.4 – view_res=512 (6 grayscale 512x512 views for YOLOv6)
#   §5.3.7 – det_conf_thresh, crop parameters for the 2D->3D slab crop

import json
import logging
import dataclasses
from dataclasses import dataclass
from typing import Any
from connector_constants import MIN_FEATURE_AREA_MM2, ROBOT_TOOL_CLEARANCE_MM

logger = logging.getLogger(__name__)


@dataclass
class ConnectionPointDetectorConfig:
    """Configuration for the WiringRobot ConnectionPointDetector engine."""

    # ---- fragment detection ----
    min_vertices: int = 20  # Increased for robotic stability
    """Minimum vertex count for a fragment; smaller ones are discarded as noise."""

    # ---- fragment merging ----
    gap_frac: float = 0.05
    """Max gap between fragments as a fraction of the mesh diagonal, used to merge them into one instance."""

    angle_max_deg: float = 30.0
    """Max angle between fragment normals (degrees) still accepted as 'same orientation'."""

    # ---- robotic wiring logic ----
    k_wires: int = 3
    """Maximum number of wire candidates per contact point."""

    insertion_tolerance_mm: float = 0.5  # Refined for high-precision wiring
    """The allowed deviation for the centroid before a detection is considered unsafe."""

    required_confidence_score: float = 0.75
    """Minimum confidence for a connection point to be considered valid for robotic action."""

    robotic_approach_distance_mm: float = 5.0
    """Safe standoff distance for the robot tool to approach before fine positioning."""

    min_insertion_depth_mm: float = 1.0
    """Minimum insertion depth for a terminal/cable entry to count as a secure connection."""

    insertion_depth_metric: str = "thesis"
    """Which insertion-depth measure gates is_valid_for_robot:
    'thesis' = |v_o - v_s| (Scheffler §5.3.6 / Abb.44; default, backward-compatible),
    'axis'   = physical travel span of the feature along the insertion axis.
    Both values are always reported in the graph (insertion_depth_mm /
    insertion_depth_axis_mm); this only selects which one the depth gate uses.
    Any value other than 'axis' is treated as 'thesis'."""

    max_feature_skew_deg: float = 75.0
    """Max skew (deg) of a feature's principal axis before its geometry is rejected as unreliable."""

    min_depth_extent_mm: float = 0.5
    """Minimum depth a 2D->3D slab is given, preventing degenerate crops on very thin features.
    Threaded into the YOLO crop step (box_to_3d_slab) by the inference pipeline."""

    min_feature_area_mm2: float = MIN_FEATURE_AREA_MM2
    """Connection points with total surface area below this are rejected as noise specks."""

    robot_tool_clearance_mm: float = ROBOT_TOOL_CLEARANCE_MM
    """Min spacing between two connection points; closer pairs are flagged as a
    tool-collision risk (`tool_clearance_conflict`). Set 0.0 to disable the check."""

    boundary_peel_iterations: int = 0
    """If >0, peel this many boundary layers (instances.remove_boundary_layer)
    before fragment detection to separate touching features (§5.3.5). 0 = off."""

    see_tol: float = -0.1
    """Minimum dot product for one contact to 'see' another.
    Negative allows some side-facing contacts; -1 disables the filter entirely."""

    # ---- surface centroid ----
    surface_subdiv: int = 0
    """Number of Loop subdivision steps on the feature sub-mesh before computing v_s.
    >0 refines the surface centroid (slower); 0 = off."""

    # ---- 2D->3D projection ----
    # These values control the upstream YOLOv6 crop step: from 6 render views
    # and 2D detections a 3D bounding box is derived and the mesh is cropped to it.
    view_res: int = 512
    """Resolution of the 6 render views (square), matching YOLOv6 (512x512)."""

    det_conf_thresh: float = 0.25
    """Minimum confidence for a YOLOv6 detection; weaker ones are discarded."""

    crop_pad_frac: float = 0.0
    """Bounding box is expanded along image-plane axes by this fraction (tolerance for loose boxes)."""

    crop_face_mode: str = "all"
    """'all' = only triangles fully inside the box, 'any' = triangles with at least one corner inside."""

    # ---- output ----
    write_scene: bool = True
    """If True, also write a .obj scene for blender/meshlab."""

    @classmethod
    def from_json(cls, path: str) -> "ConnectionPointDetectorConfig":
        """Load configuration from a JSON file.

        Unknown keys are ignored with a warning, so you can pass a partial
        config containing only the values you want to override.
        """
        with open(path, encoding="utf-8") as fh:
            data = json.load(fh)
        known = {f.name for f in dataclasses.fields(cls)}
        unknown = set(data.keys()) - known
        if unknown:
            logger.warning(
                "Unknown config keys in %s (ignored): %s", path, sorted(unknown)
            )
        filtered = {k: v for k, v in data.items() if k in known}
        return cls(**filtered)

    @classmethod
    def for_demo(cls, **overrides) -> "ConnectionPointDetectorConfig":
        """Config for the synthetic *unit-scale* selftest parts.

        The production defaults express real robot limits in millimetres
        (`min_feature_area_mm2`, `robot_tool_clearance_mm`, `min_insertion_depth_mm`,
        `required_confidence_score`). The demo meshes are toy parts ~1 unit across
        with sub-mm features, so those mm thresholds would reject every point. This
        factory disables the absolute-scale checks (sets them to 0) so the demos
        exercise the full valid-point path; real mm-scale parts should use the
        normal defaults. Extra keyword overrides are passed through.
        """
        demo: dict[str, Any] = dict(
            min_feature_area_mm2=0.0, robot_tool_clearance_mm=0.0,
            min_insertion_depth_mm=0.0, required_confidence_score=0.0)
        demo.update(overrides)
        return cls(**demo)

    def save(self, path: str) -> None:
        """Save current configuration as JSON (e.g. for reproducibility)."""
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(dataclasses.asdict(self), fh, indent=2, ensure_ascii=False)

    def __str__(self) -> str:
        lines = ["ConnectionPointDetectorConfig:"]
        for f in dataclasses.fields(self):
            lines.append(f"  {f.name} = {getattr(self, f.name)}")
        return "\n".join(lines)


# Backward-compatible alias: the pipeline was historically configured via
# `PipelineConfig`. Existing callers / saved JSON keep working unchanged.
PipelineConfig = ConnectionPointDetectorConfig
