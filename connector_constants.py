"""Centralized constants and label definitions for WiringRobot.ConnectionPointDetector.

All label IDs, class names, and configuration constants should be imported from here
to avoid duplication across modules.

The project carries two parallel vocabularies for the same five semantic classes:

  * the original thesis names (Housing, Contact, SnapPoint, CableEntry, LabelSurface),
    kept as the canonical JSON / serialization names for backward compatibility, and
  * the robotic names (Housing, TerminalContact, RailMount, CableEntry,
    IdentificationSurface) used by the ConnectionPointDetector when a
    robot-oriented vocabulary is preferred.

Both share the same underlying integer IDs, so `FeatureClass.CONTACT is
FeatureClass.TERMINAL_CONTACT` and detections stay interchangeable.

Reference: Scheffler (2022), §5.2.1
  §5.2.1 – vertex/edge/face label convention, 5 classes:
           Housing(0), Contact(1), SnapPoint(2), CableEntry(3), LabelSurface(4)
"""

from enum import IntEnum
from typing import Dict


# §5.2.1 – 5 semantic classes: Housing(0), Contact(1), SnapPoint(2), CableEntry(3), LabelSurface(4)
class FeatureClass(IntEnum):
    """Connector feature class labels (output from DiffusionNet segmentation).

    The robotic names are aliases of the original members (identical integer
    values), so e.g. ``FeatureClass.RAIL_MOUNT == FeatureClass.SNAP_POINT``.
    """

    # ---- original thesis names (canonical members) ----
    HOUSING = 0           # Housing/enclosure (background, non-actionable)
    CONTACT = 1           # Electrical terminal contact point
    SNAP_POINT = 2        # Snap/latch point for DIN-rail mounting
    CABLE_ENTRY = 3       # Cable entry point
    LABEL_SURFACE = 4     # Surface for labels/markings

    # ---- robotic aliases (same underlying IDs) ----
    TERMINAL_CONTACT = 1        # alias of CONTACT
    RAIL_MOUNT = 2              # alias of SNAP_POINT
    IDENTIFICATION_SURFACE = 4  # alias of LABEL_SURFACE


# Module-level constants (original names)
HOUSING = FeatureClass.HOUSING
CONTACT = FeatureClass.CONTACT
SNAP_POINT = FeatureClass.SNAP_POINT
CABLE_ENTRY = FeatureClass.CABLE_ENTRY
LABEL_SURFACE = FeatureClass.LABEL_SURFACE

# Module-level constants (robotic names; same IDs as above)
TERMINAL_CONTACT = FeatureClass.TERMINAL_CONTACT
RAIL_MOUNT = FeatureClass.RAIL_MOUNT
IDENTIFICATION_SURFACE = FeatureClass.IDENTIFICATION_SURFACE

# Canonical serialization names (kept stable for tooling and the JSON schema).
LABEL_NAMES: Dict[int, str] = {
    HOUSING:       "Housing",
    CONTACT:       "Contact",
    SNAP_POINT:    "SnapPoint",
    CABLE_ENTRY:   "CableEntry",
    LABEL_SURFACE: "LabelSurface",
}

# Robotic display names, keyed by the same IDs.
ROBOTIC_LABEL_NAMES: Dict[int, str] = {
    HOUSING:       "Housing",
    CONTACT:       "TerminalContact",
    SNAP_POINT:    "RailMount",
    CABLE_ENTRY:   "CableEntry",
    LABEL_SURFACE: "IdentificationSurface",
}

CLASS_NAMES = [
    "Housing",
    "Contact",
    "SnapPoint",
    "CableEntry",
    "LabelSurface",
]

CLASS_NAMES_ROBOTIC = [
    "Housing",
    "TerminalContact",
    "RailMount",
    "CableEntry",
    "IdentificationSurface",
]

NUM_CLASSES = 5

# Terminal geometry hints for TERMINAL_CONTACT points, surfaced as `terminal_type`
# in the robot-ready graph. Real geometry classification would refine this; the
# default reflects the most common spring/push-in terminals on rail components.
DEFAULT_TERMINAL_TYPE = "push_in_terminal"
TERMINAL_TYPES = {
    CONTACT: DEFAULT_TERMINAL_TYPE,
    CABLE_ENTRY: "cable_gland",
}

# Thresholds and geometry parameters
MIN_VOLUME_THRESHOLD = 1e-12  # minimum volume before falling back to vertex mean
VERTEX_NORMAL_EPSILON = 1e-12  # minimum norm before normalizing vectors
COORD_ROUNDING = 6             # decimal places for mesh welding (dedupe vertices)

# ---- robotic validation thresholds ----
MIN_FEATURE_AREA_MM2 = 1.0        # features below this surface area are treated as noise
MIN_NORMAL_STABILITY_SCORE = 0.6  # below this the approach vector is considered unreliable
ROBOT_TOOL_CLEARANCE_MM = 3.0     # min clearance the robot tool needs around a connection point

# ---- plausible real-world part scale (mm) ----
# The robot thresholds above are in millimetres, but the pipeline never rescales
# the input mesh, so they are only meaningful if the coordinates are already in mm.
# These bounds bracket the bounding-box diagonal of a believable connector part
# (a few mm for a single push-in terminal, up to large terminal blocks/housings).
# A mesh whose diagonal falls outside this range is almost certainly in the wrong
# units (unit-scale/normalized CAD, metres, ...) and triggers a warning.
MIN_PLAUSIBLE_PART_MM = 3.0
MAX_PLAUSIBLE_PART_MM = 2000.0
