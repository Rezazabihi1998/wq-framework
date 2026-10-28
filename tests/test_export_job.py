from datetime import date
from unittest.mock import MagicMock

import pytest

import wq_framework.satellite.export_job as export_job
from wq_framework.satellite.export_job import (
    ExportJobFailed,
    ExportJobHandle,
    ExportJobTimedOut,
    submit_export_job,
    wait_for_export_job,
)


class _FakeTask:
    """Simulates ee.batch.Task.status() returning a sequence of states."""

    def __init__(self, states):
        self._states = iter(states)
        self._last = None

    def status(self):
        try:
            self._last = next(self._states)
        except StopIteration:
            pass  # keep returning the last state forever
        return self._last


def _handle_with_states(states):
    return ExportJobHandle(
        task=_FakeTask([{"state": s} for s in states]),
        description="test_job",
        drive_file_name="test_job.csv",
    )


def test_completes_immediately():
    handle = _handle_with_states(["COMPLETED"])
    wait_for_export_job(handle, sleep_fn=lambda s: None, time_fn=_fake_clock())


def test_polls_through_pending_and_running_before_completing():
    handle = _handle_with_states(["PENDING", "RUNNING", "RUNNING", "COMPLETED"])
    sleeps = []
    wait_for_export_job(
        handle, sleep_fn=lambda s: sleeps.append(s), time_fn=_fake_clock()
    )
    assert len(sleeps) == 3  # slept once before each non-terminal poll


def test_raises_on_failed():
    handle = _handle_with_states(
        [{"state": "FAILED", "error_message": "quota exceeded"}]
    )
    # override to inject the error_message directly
    handle.task = _FakeTask([{"state": "FAILED", "error_message": "quota exceeded"}])
    with pytest.raises(ExportJobFailed, match="quota exceeded"):
        wait_for_export_job(handle, sleep_fn=lambda s: None, time_fn=_fake_clock())


def test_raises_on_cancelled():
    handle = _handle_with_states(["CANCELLED"])
    with pytest.raises(ExportJobFailed):
        wait_for_export_job(handle, sleep_fn=lambda s: None, time_fn=_fake_clock())


def test_raises_timeout_if_never_completes():
    handle = _handle_with_states(["RUNNING"])  # never transitions
    clock = _fake_clock(step=100.0)
    with pytest.raises(ExportJobTimedOut):
        wait_for_export_job(
            handle, timeout_seconds=250, sleep_fn=lambda s: None, time_fn=clock
        )


def _fake_clock(step: float = 1.0):
    """Returns a callable that advances by `step` seconds each call —
    lets timeout logic be tested without real sleeping."""
    state = {"t": 0.0}

    def _tick():
        state["t"] += step
        return state["t"]

    return _tick


class _FakeValue:
    def get(self, band):
        return "value_stub"


class _FakeImage:
    """Records the kwargs passed to reduceRegion, then returns a stub."""

    def __init__(self, captured):
        self._captured = captured

    def reduceRegion(self, **kwargs):
        self._captured["reduce_kwargs"] = kwargs
        return _FakeValue()

    def date(self):
        formatted = MagicMock()
        formatted.format.return_value = "2020-01-05"
        return formatted


class _FakeCollection:
    """Records filterDate's bounds and runs the mapped function once, so
    the reduceRegion call inside it is actually exercised."""

    def __init__(self, captured):
        self._captured = captured

    def filterDate(self, start, end):
        self._captured["filterDate"] = (start, end)
        return self

    def select(self, band):
        return self

    def map(self, fn):
        fn(_FakeImage(self._captured))
        return "feature_collection_stub"


def _install_fake_ee(monkeypatch, captured):
    fake_ee = MagicMock()
    fake_ee.ImageCollection.return_value = _FakeCollection(captured)
    fake_ee.Reducer.mean.return_value = "reducer_stub"
    fake_ee.Geometry.Point.return_value = MagicMock()
    fake_ee.FeatureCollection.return_value = "fc_stub"
    fake_ee.Feature.return_value = "feature_stub"

    monkeypatch.setattr(export_job, "ee", fake_ee)
    monkeypatch.setattr(export_job, "ensure_ee_installed", lambda: None)
    return fake_ee


def _submit(captured, monkeypatch, **overrides):
    _install_fake_ee(monkeypatch, captured)
    kwargs = dict(
        image_collection_id="ECMWF/ERA5_LAND/DAILY_AGGR",
        band_name="total_precipitation_sum",
        reducer_name="mean",
        latitude=32.1,
        longitude=51.5,
        buffer_km=15.0,
        start_date=date(2020, 1, 1),
        end_date=date(2020, 1, 5),
        description="test_job",
    )
    kwargs.update(overrides)
    return submit_export_job(**kwargs)


def test_filter_date_end_bound_is_inclusive(monkeypatch):
    # filterDate's end is exclusive, so end_date=2020-01-05 must be sent
    # as 2020-01-06 — otherwise the final day is silently dropped.
    captured = {}
    _submit(captured, monkeypatch)
    assert captured["filterDate"] == ("2020-01-01", "2020-01-06")


def test_reduce_region_uses_era5land_native_scale_by_default(monkeypatch):
    captured = {}
    _submit(captured, monkeypatch)
    assert captured["reduce_kwargs"]["scale"] == 11132


def test_reduce_region_scale_is_configurable(monkeypatch):
    captured = {}
    _submit(captured, monkeypatch, scale_m=5000)
    assert captured["reduce_kwargs"]["scale"] == 5000


def test_submit_returns_handle_with_original_description(monkeypatch):
    captured = {}
    handle = _submit(captured, monkeypatch)
    assert handle.description == "test_job"
    assert handle.drive_file_name == "test_job"