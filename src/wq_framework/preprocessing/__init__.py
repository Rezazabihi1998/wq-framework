from .base import OutlierDetector, OutlierReport
from .registry import (
    available_outlier_detectors,
    get_outlier_detector,
    register_outlier_detector,
)
from .runner import run_outlier_detection

# Importing this registers "iqr" with the registry as a side effect.
from .methods import iqr as _iqr  # noqa: F401

__all__ = [
    "OutlierDetector",
    "OutlierReport",
    "available_outlier_detectors",
    "get_outlier_detector",
    "register_outlier_detector",
    "run_outlier_detection",
]
