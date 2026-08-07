from unittest.mock import MagicMock

import pytest

from wq_framework.satellite.export_job import (
    ExportJobFailed,
    ExportJobHandle,
    ExportJobTimedOut,
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