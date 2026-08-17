"""
Behavior (confirmed with the user, based on the original thesis code):
- k=3 by default (more conservative than the textbook-standard 1.5), to
  preserve hydrologically meaningful extremes while catching clear
  computational/instrument/human errors.
- Each column's Q1/Q3 (and therefore its outlier bounds) are computed
  **independently**, on that column's own values — not on a
  progressively-shrinking dataset affected by other columns' outlier
  removal (unlike the original code, which dropped rows column-by-column
  in sequence, so later columns' quantiles were computed on already
  reduced data).
- Outliers are **nulled at the cell level only** — the row is never
  dropped. This differs from the original code (which dropped the entire
  row) and was a deliberate choice to align with the framework's
- Pre-existing NaN values (e.g. from schema validation) are left alone —
  they are not counted as outliers and are not part of the quantile
  calculation (pandas' `.quantile()` already ignores NaN by default).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..base import OutlierDetector, OutlierReport
from ..registry import register_outlier_detector


@register_outlier_detector("iqr")
class IQROutlierDetector(OutlierDetector):
    def __init__(self, k: float = 3.0):
        self.k = k

    def fit_transform(self, series: pd.Series) -> tuple[pd.Series, OutlierReport]:
        q1 = series.quantile(0.25)
        q3 = series.quantile(0.75)
        iqr = q3 - q1
        lower_bound = q1 - self.k * iqr
        upper_bound = q3 + self.k * iqr

        is_outlier = series.notna() & ((series < lower_bound) | (series > upper_bound))

        cleaned = series.copy()
        cleaned[is_outlier] = np.nan

        report = OutlierReport(
            parameter=series.name or "unknown",
            method="iqr",
            params={"k": self.k},
            lower_bound=lower_bound,
            upper_bound=upper_bound,
            n_removed=int(is_outlier.sum()),
            n_total=int(series.notna().sum()),
        )
        return cleaned, report
