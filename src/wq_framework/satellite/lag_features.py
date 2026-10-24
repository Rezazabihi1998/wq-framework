"""Monthly lag-feature generation from a daily satellite-variable series.

Builds N non-overlapping, backward-looking windows ending immediately
before a reference date (e.g. a water-quality sample's collection date).
Each window is aggregated into one feature value — separate, simultaneous
windows rather than cumulative ones, to avoid collinearity between
adjacent lags.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date

import pandas as pd

_AGGREGATORS = ("sum", "mean")


@dataclass
class LagFeatureReport:
    """Transparency record for one compute_monthly_lag_features() call."""

    variable: str
    reference_date: date
    window_days: int
    aggregation: str
    n_lags: int
    days_available: dict = field(default_factory=dict)

    def __str__(self) -> str:
        parts = ", ".join(
            f"{name}={n}/{self.window_days}d" for name, n in self.days_available.items()
        )
        return (
            f"lag features for '{self.variable}' as of {self.reference_date} "
            f"({self.aggregation}, {self.window_days}-day windows): {parts}"
        )


def compute_monthly_lag_features(
    daily_series: pd.Series,
    reference_date: date,
    n_lags: int,
    aggregation: str,
    variable_name: str = "value",
    window_days: int = 30,
) -> tuple[dict, LagFeatureReport]:
    """Build N non-overlapping lag features from a daily time series.

    Window 1 ("lag_1") covers the `window_days` days immediately before
    `reference_date` (not including it); window 2 ("lag_2") covers the
    `window_days` days before that; and so on. Windows never overlap.

    Missing days *within* a window are skipped when aggregating (the
    framework's general preference is not to drop data outright), but if a
    window has no data at all, the feature is NaN rather than a misleading
    0 — this matters specifically for `aggregation="sum"`, where pandas'
    default skip-NaN behavior would otherwise silently turn "no data" into
    "zero precipitation".

    Parameters
    ----------
    daily_series : date-indexed pandas Series (e.g. one retriever's fetch() output)
    reference_date : the date each window counts backward from (exclusive)
    n_lags : how many windows to build (lag_1 .. lag_N)
    aggregation : "sum" (e.g. precipitation) or "mean" (e.g. temperature)
    variable_name : only used for the report, not the returned dict keys
    window_days : length of each window in days (default 30)
    """
    if aggregation not in _AGGREGATORS:
        raise ValueError(
            f"Unknown aggregation '{aggregation}' — expected one of {_AGGREGATORS}"
        )
    if n_lags < 1:
        raise ValueError("n_lags must be >= 1")
    if window_days < 1:
        raise ValueError("window_days must be >= 1")

    if not isinstance(daily_series.index, pd.DatetimeIndex):
        daily_series = daily_series.copy()
        daily_series.index = pd.to_datetime(daily_series.index)

    reference_ts = pd.Timestamp(reference_date)
    features = {}
    days_available = {}

    for i in range(1, n_lags + 1):
        window_end = reference_ts - pd.Timedelta(days=(i - 1) * window_days + 1)
        window_start = window_end - pd.Timedelta(days=window_days - 1)

        window_values = daily_series.loc[window_start:window_end].dropna()
        lag_name = f"lag_{i}"
        days_available[lag_name] = len(window_values)

        if window_values.empty:
            features[lag_name] = float("nan")
        elif aggregation == "sum":
            features[lag_name] = float(window_values.sum())
        else:
            features[lag_name] = float(window_values.mean())

    report = LagFeatureReport(
        variable=variable_name,
        reference_date=reference_date,
        window_days=window_days,
        aggregation=aggregation,
        n_lags=n_lags,
        days_available=days_available,
    )
    return features, report