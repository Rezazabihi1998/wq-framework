"""Tests for the satellite-variable Strategy+Registry plumbing itself —
independent of any real data source (ERA5-Land/GEE connection is Stage 3.2;
concrete variables are Stage 3.3)."""

from datetime import date

import pandas as pd
import pytest

from wq_framework.satellite.base import RetrievalReport, SatelliteVariableRetriever
from wq_framework.satellite.registry import (
    _SATELLITE_VARIABLE_REGISTRY,
    available_satellite_variables,
    get_satellite_variable,
    register_satellite_variable,
)


@pytest.fixture(autouse=True)
def clean_registry():
    """Each test gets a clean registry so tests don't leak into each other."""
    saved = dict(_SATELLITE_VARIABLE_REGISTRY)
    _SATELLITE_VARIABLE_REGISTRY.clear()
    yield
    _SATELLITE_VARIABLE_REGISTRY.clear()
    _SATELLITE_VARIABLE_REGISTRY.update(saved)


class _DummyRetriever(SatelliteVariableRetriever):
    """A trivial retriever: returns a constant value per day, for testing
    the plumbing only — no network/auth involved."""

    def __init__(self, buffer_km: float = 0.0, constant_value: float = 1.0):
        super().__init__(buffer_km=buffer_km)
        self.constant_value = constant_value

    def fetch(self, station_id, latitude, longitude, start_date, end_date):
        days = pd.date_range(start_date, end_date, freq="D")
        series = pd.Series(self.constant_value, index=days, name="dummy")
        report = RetrievalReport(
            variable="dummy",
            source="dummy_source",
            station_id=station_id,
            buffer_km=self.buffer_km,
            start_date=start_date,
            end_date=end_date,
            n_days_requested=len(days),
            n_days_returned=len(days),
        )
        return series, report


def test_register_and_retrieve():
    register_satellite_variable("dummy")(_DummyRetriever)
    retriever = get_satellite_variable("dummy", buffer_km=15.0, constant_value=5.0)
    assert isinstance(retriever, _DummyRetriever)
    assert retriever.buffer_km == 15.0
    assert retriever.constant_value == 5.0


def test_duplicate_registration_raises():
    register_satellite_variable("dummy")(_DummyRetriever)
    with pytest.raises(ValueError, match="already registered"):
        register_satellite_variable("dummy")(_DummyRetriever)


def test_unknown_name_raises_with_available_list():
    register_satellite_variable("dummy")(_DummyRetriever)
    with pytest.raises(ValueError, match="unknown satellite variable"):
        get_satellite_variable("does_not_exist")


def test_available_satellite_variables_lists_registered_names():
    assert available_satellite_variables() == []
    register_satellite_variable("dummy")(_DummyRetriever)
    assert available_satellite_variables() == ["dummy"]


def test_default_buffer_km_is_zero():
    register_satellite_variable("dummy")(_DummyRetriever)
    retriever = get_satellite_variable("dummy")
    assert retriever.buffer_km == 0.0


def test_dummy_retriever_fetch_contract():
    register_satellite_variable("dummy")(_DummyRetriever)
    retriever = get_satellite_variable("dummy", buffer_km=15.0, constant_value=2.0)

    series, report = retriever.fetch(
        "DAM_UP", 32.1, 51.5, date(2020, 1, 1), date(2020, 1, 5)
    )

    assert len(series) == 5  # inclusive of both endpoints
    assert (series == 2.0).all()
    assert report.station_id == "DAM_UP"
    assert report.n_days_requested == 5
    assert report.n_days_returned == 5
    assert report.buffer_km == 15.0
    assert report.start_date == date(2020, 1, 1)
    assert report.end_date == date(2020, 1, 5)