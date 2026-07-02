"""Real annotation parsing and conversion helpers."""

from .conversion import (
    REAL_CLASS_NAMES,
    REAL_LABEL_REMAP,
    build_real_annotation_sample,
    collapse_synthetic_semantic_to_real,
)
from .jfilament import SnakePoint, parse_jfilament_snakes
from .labkit import LabkitLabels, load_real_labels

__all__ = [
    "LabkitLabels",
    "REAL_CLASS_NAMES",
    "REAL_LABEL_REMAP",
    "SnakePoint",
    "build_real_annotation_sample",
    "collapse_synthetic_semantic_to_real",
    "load_real_labels",
    "parse_jfilament_snakes",
]
