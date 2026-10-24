import math
from datetime import date

import pandas as pd
import pytest

from wq_framework.satellite.lag_features import compute_monthly_lag_features


def _daily_series(start, n_days, value_fn):
    """Builds a daily series of length n_days starting at `start`, where
    value_fn(i) gives the value for day i (0-indexed)."""
    index = pd.date_range(start, periods=n_days, freq="D")
    values = [value_fn(i) for i in range(n_days)]
    return pd.Series(values, index=index)


def test_sum_aggregation_over_full_window():
    # 6 days of constant precipitation = 1.0/day, window_days=3, n_lags=2.
    series = _daily_series(date(2020, 1, 1), 6, lambda i: 1.0)
    features, report = compute_monthly_lag_features(
        series, reference_date=date(2020, 1, 7), n_lags=2,
        aggregation="sum", window_days=3,
    )
    assert features["lag_1"] == 3.0  # Jan 4-6
    assert features["lag_2"] == 3.0  # Jan 1-3
    assert report.days_available == {"lag_1": 3, "lag_2": 3}


def test_mean_aggregation_over_full_window():
    series = _daily_series(date(2020, 1, 1), 6, lambda i: float(i))  # 0,1,2,3,4,5
    features, _ = compute_monthly_lag_features(
        series, reference_date=date(2020, 1, 7), n_lags=2,
        aggregation="mean", window_days=3,
    )
    assert features["lag_1"] == pytest.approx(4.0)  # days 3,4,5 -> mean 4
    assert features["lag_2"] == pytest.approx(1.0)  # days 0,1,2 -> mean 1


def test_windows_are_non_overlapping():
    series = _daily_series(date(2020, 1, 1), 9, lambda i: float(i))
    _, report = compute_monthly_lag_features(
        series, reference_date=date(2020, 1, 10), n_lags=3,
        aggregation="mean", window_days=3,
    )
    # 3 lags x 3-day windows = 9 days total, each window fully populated
    # and none overlapping -> every lag sees exactly 3 distinct days.
    assert report.days_available == {"lag_1": 3, "lag_2": 3, "lag_3": 3}


def test_partial_missing_data_aggregates_over_available_days_only():
    series = _daily_series(date(2020, 1, 1), 3, lambda i: 10.0)
    # Only 3 days of data exist; ask for a 5-day window -> 2 days missing.
    features, report = compute_monthly_lag_features(
        series, reference_date=date(2020, 1, 6), n_lags=1,
        aggregation="sum", window_days=5,
    )
    assert features["lag_1"] == 30.0  # sum of the 3 available days only
    assert report.days_available == {"lag_1": 3}


def test_empty_window_is_nan_not_zero_for_sum():
    # Series exists but entirely outside the requested window.
    series = _daily_series(date(2019, 1, 1), 3, lambda i: 10.0)
    features, report = compute_monthly_lag_features(
        series, reference_date=date(2020, 1, 1), n_lags=1,
        aggregation="sum", window_days=5,
    )
    assert math.isnan(features["lag_1"])
    assert report.days_available == {"lag_1": 0}


def test_empty_window_is_nan_for_mean_too():
    series = _daily_series(date(2019, 1, 1), 3, lambda i: 10.0)
    features, _ = compute_monthly_lag_features(
        series, reference_date=date(2020, 1, 1), n_lags=1,
        aggregation="mean", window_days=5,
    )
    assert math.isnan(features["lag_1"])


def test_invalid_aggregation_raises():
    series = _daily_series(date(2020, 1, 1), 3, lambda i: 1.0)
    with pytest.raises(ValueError):
        compute_monthly_lag_features(
            series, reference_date=date(2020, 1, 5), n_lags=1,
            aggregation="median", window_days=3,
        )


def test_zero_or_negative_n_lags_raises():
    series = _daily_series(date(2020, 1, 1), 3, lambda i: 1.0)
    with pytest.raises(ValueError):
        compute_monthly_lag_features(
            series, reference_date=date(2020, 1, 5), n_lags=0,
            aggregation="sum", window_days=3,
        )


def test_zero_or_negative_window_days_raises():
    series = _daily_series(date(2020, 1, 1), 3, lambda i: 1.0)
    with pytest.raises(ValueError):
        compute_monthly_lag_features(
            series, reference_date=date(2020, 1, 5), n_lags=1,
            aggregation="sum", window_days=0,
        )