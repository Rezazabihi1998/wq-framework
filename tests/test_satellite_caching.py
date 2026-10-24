from datetime import date

import pandas as pd
import pytest

from wq_framework.satellite.base import RetrievalReport, SatelliteVariableRetriever
from wq_framework.satellite.caching import CachingRetriever


class _CountingRetriever(SatelliteVariableRetriever):
    """Counts how many times fetch() actually ran, to prove the cache
    prevents redundant calls."""

    def __init__(self, buffer_km: float = 0.0):
        super().__init__(buffer_km=buffer_km)
        self.call_count = 0

    def fetch(self, station_id, latitude, longitude, start_date, end_date):
        self.call_count += 1
        series = pd.Series([1.0, 2.0], index=pd.date_range(start_date, periods=2))
        report = RetrievalReport(
            variable="dummy",
            source="counting",
            station_id=station_id,
            buffer_km=self.buffer_km,
            start_date=start_date,
            end_date=end_date,
        )
        return series, report


def test_second_call_uses_cache_not_inner(tmp_path):
    inner = _CountingRetriever()
    cached = CachingRetriever(inner, variable_name="dummy", cache_dir=tmp_path)

    cached.fetch("DAM_UP", 32.1, 51.5, date(2020, 1, 1), date(2020, 1, 2))
    cached.fetch("DAM_UP", 32.1, 51.5, date(2020, 1, 1), date(2020, 1, 2))

    assert inner.call_count == 1  # only called once, second was served from cache


def test_cache_file_is_created(tmp_path):
    inner = _CountingRetriever()
    cached = CachingRetriever(inner, variable_name="precipitation", cache_dir=tmp_path)
    cached.fetch("DAM_UP", 32.1, 51.5, date(2020, 1, 1), date(2020, 1, 2))

    files = list(tmp_path.glob("*.pkl"))
    assert len(files) == 1
    assert "DAM_UP" in files[0].name
    assert "precipitation" in files[0].name


def test_different_stations_get_separate_cache_entries(tmp_path):
    inner = _CountingRetriever()
    cached = CachingRetriever(inner, variable_name="dummy", cache_dir=tmp_path)

    cached.fetch("DAM_UP", 32.1, 51.5, date(2020, 1, 1), date(2020, 1, 2))
    cached.fetch("DAM_DOWN", 32.0, 51.6, date(2020, 1, 1), date(2020, 1, 2))

    assert inner.call_count == 2  # different stations, both genuinely fetched


def test_cached_result_flags_from_cache(tmp_path):
    inner = _CountingRetriever()
    cached = CachingRetriever(inner, variable_name="dummy", cache_dir=tmp_path)

    _, first_report = cached.fetch("DAM_UP", 32.1, 51.5, date(2020, 1, 1), date(2020, 1, 2))
    _, second_report = cached.fetch("DAM_UP", 32.1, 51.5, date(2020, 1, 1), date(2020, 1, 2))

    assert "from_cache" not in first_report.params
    assert second_report.params["from_cache"] is True


def test_cached_series_values_match_original(tmp_path):
    inner = _CountingRetriever()
    cached = CachingRetriever(inner, variable_name="dummy", cache_dir=tmp_path)

    series1, _ = cached.fetch("DAM_UP", 32.1, 51.5, date(2020, 1, 1), date(2020, 1, 2))
    series2, _ = cached.fetch("DAM_UP", 32.1, 51.5, date(2020, 1, 1), date(2020, 1, 2))

    pd.testing.assert_series_equal(series1, series2)


def test_different_buffer_km_gets_separate_cache_entry(tmp_path):
    # Two retrievers for the same station/variable/date-range but different
    # buffer radii must not share a cache entry, since the spatial average
    # differs. Reuses the same underlying inner (call_count) to detect it.
    inner = _CountingRetriever(buffer_km=15.0)
    cached_15 = CachingRetriever(inner, variable_name="dummy", cache_dir=tmp_path)

    inner.buffer_km = 10.0
    cached_10 = CachingRetriever(inner, variable_name="dummy", cache_dir=tmp_path)

    cached_15.fetch("DAM_UP", 32.1, 51.5, date(2020, 1, 1), date(2020, 1, 2))
    cached_10.fetch("DAM_UP", 32.1, 51.5, date(2020, 1, 1), date(2020, 1, 2))

    assert inner.call_count == 2  # different buffer radii must not share a cache entry