"""Local caching wrapper for any SatelliteVariableRetriever — avoids
re-fetching (re-submitting a GEE Export job, or re-reading a manual file)
for data already retrieved once. Works via composition (wraps any
retriever), so it applies uniformly to every source without changing them.
"""

from __future__ import annotations

import pickle
from datetime import date
from pathlib import Path

from .base import SatelliteVariableRetriever


def _safe_filename(value) -> str:
    return "".join(c if c.isalnum() or c in ("-", "_") else "_" for c in str(value))


class CachingRetriever(SatelliteVariableRetriever):
    """Wraps another retriever; checks a local pickle cache before
    calling it. `variable_name` is passed explicitly (not inferred from
    `inner`) so this works identically for manual and GEE-backed sources.
    """

    def __init__(
        self,
        inner: SatelliteVariableRetriever,
        variable_name: str,
        cache_dir: str | Path = "cache/satellite",
    ):
        super().__init__(buffer_km=inner.buffer_km)
        self.inner = inner
        self.variable_name = variable_name
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _cache_path(self, station_id: str, start_date: date, end_date: date) -> Path:
        # buffer_km is part of the key: a cached result fetched with one
        # buffer radius must never be returned for a different radius, since
        # the underlying spatial average would be different.
        filename = (
            f"{_safe_filename(station_id)}_{_safe_filename(self.variable_name)}_"
            f"buf{_safe_filename(self.buffer_km)}_{start_date}_{end_date}.pkl"
        )
        return self.cache_dir / filename

    def fetch(self, station_id, latitude, longitude, start_date, end_date):
        path = self._cache_path(station_id, start_date, end_date)

        if path.exists():
            with open(path, "rb") as f:
                series, report = pickle.load(f)
            report.params["from_cache"] = True
            return series, report

        series, report = self.inner.fetch(station_id, latitude, longitude, start_date, end_date)
        with open(path, "wb") as f:
            pickle.dump((series, report), f)
        return series, report