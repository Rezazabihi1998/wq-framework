"""Local caching wrapper for any SatelliteVariableRetriever — avoids
re-fetching (re-submitting a GEE Export job, or re-reading a manual file)
for data already retrieved once. Works via composition (wraps any
retriever), so it applies uniformly to every source without changing them.

Cache entries are versioned. A cached series is only reusable if it was
written by a version of the retrieval code that produced the same
contract; otherwise a hit would return data in a form the current code no
longer agrees with, and nothing downstream would notice. The case that
motivated this: entries written before retrievers normalised units hold
raw ERA5-Land values (metres, kelvin, joules). Reusing one would hand the
models values on a scale three orders of magnitude away from the
canonical unit, silently. Unknown, older, or unreadable entries are
therefore treated as a miss and overwritten.
"""

from __future__ import annotations

import pickle
from datetime import date
from pathlib import Path

from .base import SatelliteVariableRetriever

_CACHE_SCHEMA_VERSION = 2
"""Bump whenever the shape or the unit contract of a cached value changes.

Version 1 was an unversioned ``(series, report)`` tuple. Version 2 stores
a versioned envelope and requires series in canonical units.
"""


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

    def _read_cache(self, path: Path):
        """Return the cached (series, report), or None if the entry is
        absent, unreadable, or was written under a different schema
        version."""
        if not path.exists():
            return None

        try:
            with open(path, "rb") as f:
                payload = pickle.load(f)
        except Exception:
            # Truncated or corrupt file: a miss is always safer than
            # propagating a deserialisation error.
            return None

        if not isinstance(payload, dict):
            return None  # pre-versioning tuple
        if payload.get("version") != _CACHE_SCHEMA_VERSION:
            return None
        if "series" not in payload or "report" not in payload:
            return None

        return payload["series"], payload["report"]

    def _write_cache(self, path: Path, series, report) -> None:
        payload = {
            "version": _CACHE_SCHEMA_VERSION,
            "variable": self.variable_name,
            "buffer_km": self.buffer_km,
            "series": series,
            "report": report,
        }
        with open(path, "wb") as f:
            pickle.dump(payload, f)

    def fetch(self, station_id, latitude, longitude, start_date, end_date):
        path = self._cache_path(station_id, start_date, end_date)

        cached = self._read_cache(path)
        if cached is not None:
            series, report = cached
            report.params["from_cache"] = True
            return series, report

        series, report = self.inner.fetch(station_id, latitude, longitude, start_date, end_date)
        self._write_cache(path, series, report)
        return series, report