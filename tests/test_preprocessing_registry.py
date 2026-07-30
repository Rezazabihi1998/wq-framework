"""Tests for the Strategy+Registry plumbing itself — independent of any
concrete outlier-detection method (IQR is tested separately once added)."""

import pandas as pd
import pytest

from wq_framework.preprocessing.base import OutlierDetector, OutlierReport
from wq_framework.preprocessing.registry import (
    _OUTLIER_DETECTOR_REGISTRY,
    available_outlier_detectors,
    get_outlier_detector,
    register_outlier_detector,
)


@pytest.fixture(autouse=True)
def clean_registry():
    """Each test gets a clean registry so tests don't leak into each other."""
    saved = dict(_OUTLIER_DETECTOR_REGISTRY)
    _OUTLIER_DETECTOR_REGISTRY.clear()
    yield
    _OUTLIER_DETECTOR_REGISTRY.clear()
    _OUTLIER_DETECTOR_REGISTRY.update(saved)


class _DummyDetector(OutlierDetector):
    """A trivial detector: flags nothing, for testing the plumbing only."""

    def __init__(self, some_param: int = 1):
        self.some_param = some_param

    def fit_transform(self, series):
        report = OutlierReport(
            parameter=series.name or "unknown",
            method="dummy",
            params={"some_param": self.some_param},
            n_removed=0,
            n_total=len(series),
        )
        return series.copy(), report


def test_register_and_retrieve():
    register_outlier_detector("dummy")(_DummyDetector)
    detector = get_outlier_detector("dummy", some_param=5)
    assert isinstance(detector, _DummyDetector)
    assert detector.some_param == 5


def test_duplicate_registration_raises():
    register_outlier_detector("dummy")(_DummyDetector)
    with pytest.raises(ValueError, match="already registered"):
        register_outlier_detector("dummy")(_DummyDetector)


def test_unknown_name_raises_with_available_list():
    register_outlier_detector("dummy")(_DummyDetector)
    with pytest.raises(ValueError, match="unknown outlier detector"):
        get_outlier_detector("does_not_exist")


def test_available_outlier_detectors_lists_registered_names():
    assert available_outlier_detectors() == []
    register_outlier_detector("dummy")(_DummyDetector)
    assert available_outlier_detectors() == ["dummy"]


def test_dummy_detector_fit_transform_contract():
    register_outlier_detector("dummy")(_DummyDetector)
    detector = get_outlier_detector("dummy")
    series = pd.Series([1, 2, 3], name="TDS")
    cleaned, report = detector.fit_transform(series)
    assert list(cleaned) == [1, 2, 3]
    assert report.parameter == "TDS"
    assert report.n_removed == 0
    assert report.n_total == 3
