"""Name -> class registry for outlier-detection strategies.

Usage:
    @register_outlier_detector("iqr")
    class IQROutlierDetector(OutlierDetector):
        ...

    detector = get_outlier_detector("iqr", k=3.0)
"""

from __future__ import annotations

from .base import OutlierDetector

_OUTLIER_DETECTOR_REGISTRY: dict[str, type[OutlierDetector]] = {}


def register_outlier_detector(name: str):
    def decorator(cls: type[OutlierDetector]) -> type[OutlierDetector]:
        if name in _OUTLIER_DETECTOR_REGISTRY:
            raise ValueError(f"outlier detector '{name}' is already registered")
        _OUTLIER_DETECTOR_REGISTRY[name] = cls
        return cls

    return decorator


def get_outlier_detector(name: str, **params) -> OutlierDetector:
    if name not in _OUTLIER_DETECTOR_REGISTRY:
        available = sorted(_OUTLIER_DETECTOR_REGISTRY.keys())
        raise ValueError(
            f"unknown outlier detector '{name}'. Available: {available}. "
            f"Register a new one with @register_outlier_detector('{name}')."
        )
    return _OUTLIER_DETECTOR_REGISTRY[name](**params)


def available_outlier_detectors() -> list[str]:
    return sorted(_OUTLIER_DETECTOR_REGISTRY.keys())
