"""Applies a single OutlierDetector across multiple columns of a DataFrame.

preprocessing runs once per station, on the
full parameter set. Each column is handled independently (see methods/iqr.py
docstring for why), so this is a straightforward per-column loop — no
cross-column state.
"""

from __future__ import annotations

import pandas as pd

from .base import OutlierDetector, OutlierReport


def run_outlier_detection(
    df: pd.DataFrame,
    columns: list[str],
    detector: OutlierDetector,
) -> tuple[pd.DataFrame, list[OutlierReport]]:
    """Run `detector` on each of `columns` in `df`, independently.

    Returns a cleaned copy of `df` (only the listed columns are touched,
    outlier cells set to NaN) plus one OutlierReport per column that was
    actually present in `df`. Columns not present in `df` are silently
    skipped (an optional parameter simply wasn't provided).
    """
    cleaned = df.copy()
    reports: list[OutlierReport] = []

    for column in columns:
        if column not in cleaned.columns:
            continue
        cleaned_series, report = detector.fit_transform(cleaned[column])
        cleaned[column] = cleaned_series
        reports.append(report)

    return cleaned, reports
