"""Base interface for preprocessing (outlier detection) strategies.

See project brief Section 6: preprocessing runs once per station, on the
full parameter set, before any target parameter is chosen. Each concrete
method (e.g. IQR) is a Strategy+Registry plugin registered via
`registry.register_outlier_detector`.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import pandas as pd


@dataclass
class OutlierReport:
    """Transparency record of what an OutlierDetector did to one column."""

    parameter: str
    method: str
    params: dict = field(default_factory=dict)
    lower_bound: float | None = None
    upper_bound: float | None = None
    n_removed: int = 0
    n_total: int = 0

    def __str__(self) -> str:
        return (
            f"{self.parameter}: {self.n_removed}/{self.n_total} value(s) flagged "
            f"as outliers by '{self.method}' "
            f"(bounds: [{self.lower_bound}, {self.upper_bound}])"
        )


class OutlierDetector(ABC):
    """Interface every outlier-detection strategy must implement."""

    @abstractmethod
    def fit_transform(self, series: pd.Series) -> tuple[pd.Series, OutlierReport]:
        """Return a cleaned copy of `series` (outliers set to NaN) plus a
        report describing what was removed and why. Must not mutate the
        input series in place.
        """
        raise NotImplementedError
