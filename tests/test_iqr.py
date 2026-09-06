import numpy as np
import pandas as pd
import pytest

from wq_framework.preprocessing import get_outlier_detector
from wq_framework.preprocessing.methods.iqr import IQROutlierDetector


@pytest.fixture
def clustered_series_with_one_outlier():
    # Tight cluster of "normal" values plus one clear outlier.
    values = [100, 102, 98, 101, 99, 100, 103, 97, 100, 400]
    return pd.Series(values, name="TDS")


def test_iqr_registered_and_retrievable_via_registry():
    detector = get_outlier_detector("iqr", k=3.0)
    assert isinstance(detector, IQROutlierDetector)
    assert detector.k == 3.0


def test_iqr_default_k_is_3():
    detector = IQROutlierDetector()
    assert detector.k == 3.0


def test_flags_the_single_clear_outlier(clustered_series_with_one_outlier):
    detector = IQROutlierDetector(k=3.0)
    cleaned, report = detector.fit_transform(clustered_series_with_one_outlier)

    assert report.n_removed == 1
    assert report.n_total == 10
    assert np.isnan(cleaned.iloc[9])  # the 400 was flagged
    # everything else untouched
    assert list(cleaned.iloc[:9]) == [100, 102, 98, 101, 99, 100, 103, 97, 100]


def test_no_outliers_in_uniform_data():
    series = pd.Series([50, 51, 49, 50, 50, 51, 49], name="pH")
    detector = IQROutlierDetector(k=3.0)
    cleaned, report = detector.fit_transform(series)
    assert report.n_removed == 0
    assert cleaned.equals(series)


def test_larger_k_flags_fewer_points(clustered_series_with_one_outlier):
    strict = IQROutlierDetector(k=1.5)
    lenient = IQROutlierDetector(k=10.0)

    _, strict_report = strict.fit_transform(clustered_series_with_one_outlier)
    _, lenient_report = lenient.fit_transform(clustered_series_with_one_outlier)

    assert lenient_report.n_removed <= strict_report.n_removed


def test_preexisting_nan_is_not_treated_as_outlier_and_not_double_counted():
    series = pd.Series([100, 101, np.nan, 99, 100, 400], name="TDS")
    detector = IQROutlierDetector(k=3.0)
    cleaned, report = detector.fit_transform(series)

    # the pre-existing NaN stays NaN, isn't in n_total or n_removed
    assert report.n_total == 5  # 5 non-null values, not 6
    assert np.isnan(cleaned.iloc[2])  # original NaN preserved
    assert np.isnan(cleaned.iloc[5])  # the 400 flagged as an outlier


def test_report_contains_bounds_and_method_name(clustered_series_with_one_outlier):
    detector = IQROutlierDetector(k=3.0)
    _, report = detector.fit_transform(clustered_series_with_one_outlier)
    assert report.method == "iqr"
    assert report.params == {"k": 3.0}
    assert report.lower_bound is not None
    assert report.upper_bound is not None
    assert report.lower_bound < report.upper_bound
