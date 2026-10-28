from datetime import date

import pandas as pd
import pytest

from wq_framework.satellite import get_satellite_variable
from wq_framework.satellite.variables import manual  # noqa: F401 - registers "manual"


@pytest.fixture
def sample_manual_file(tmp_path):
    path = tmp_path / "manual_satellite.csv"
    pd.DataFrame(
        {
            "station_id": ["DAM_UP", "DAM_UP", "DAM_UP", "DAM_DOWN"],
            "date": ["2020-01-01", "2020-01-02", "2020-01-03", "2020-01-01"],
            "precipitation": [0.0, 2.5, 1.1, 5.0],
        }
    ).to_csv(path, index=False)
    return path


def test_registered_under_manual():
    retriever = get_satellite_variable(
        "manual", file_path="dummy.csv", value_column="precipitation"
    )
    assert retriever.value_column == "precipitation"


def test_fetch_filters_by_station_and_date_range(sample_manual_file):
    retriever = get_satellite_variable(
        "manual", file_path=sample_manual_file, value_column="precipitation"
    )
    series, report = retriever.fetch(
        "DAM_UP", 32.1, 51.5, date(2020, 1, 1), date(2020, 1, 3)
    )
    assert list(series.values) == [0.0, 2.5, 1.1]
    assert report.station_id == "DAM_UP"
    assert report.source == "manual"


def test_fetch_ignores_other_stations(sample_manual_file):
    retriever = get_satellite_variable(
        "manual", file_path=sample_manual_file, value_column="precipitation"
    )
    series, _ = retriever.fetch(
        "DAM_DOWN", 32.0, 51.6, date(2020, 1, 1), date(2020, 1, 3)
    )
    assert list(series.values) == [5.0]


def test_missing_required_column_raises(tmp_path):
    bad_file = tmp_path / "bad.csv"
    pd.DataFrame({"station_id": ["A"], "date": ["2020-01-01"]}).to_csv(
        bad_file, index=False
    )
    retriever = get_satellite_variable(
        "manual", file_path=bad_file, value_column="precipitation"
    )
    with pytest.raises(ValueError, match="missing required column"):
        retriever.fetch("A", 0, 0, date(2020, 1, 1), date(2020, 1, 1))


def test_file_loaded_lazily_and_cached(sample_manual_file, monkeypatch):
    retriever = get_satellite_variable(
        "manual", file_path=sample_manual_file, value_column="precipitation"
    )
    assert retriever._data is None
    retriever.fetch("DAM_UP", 32.1, 51.5, date(2020, 1, 1), date(2020, 1, 1))
    assert retriever._data is not None
    loaded_once = retriever._data
    retriever.fetch("DAM_UP", 32.1, 51.5, date(2020, 1, 2), date(2020, 1, 2))
    assert retriever._data is loaded_once  # same object, not reloaded


# --- Unit contract --------------------------------------------------------
#
# The manual file is expected to already be in the canonical unit, so its
# values must reach the caller untouched. What matters is that the unit is
# stated rather than left ambiguous, and that a mismatch with the GEE path
# is refused instead of silently producing two different scales.


def test_report_states_canonical_unit_for_known_variable(sample_manual_file):
    retriever = get_satellite_variable(
        "manual", file_path=sample_manual_file, value_column="precipitation"
    )
    _, report = retriever.fetch(
        "DAM_UP", 32.1, 51.5, date(2020, 1, 1), date(2020, 1, 3)
    )
    assert report.canonical_unit == "mm/day"


def test_values_are_not_rescaled(sample_manual_file):
    retriever = get_satellite_variable(
        "manual", file_path=sample_manual_file, value_column="precipitation"
    )
    series, _ = retriever.fetch(
        "DAM_UP", 32.1, 51.5, date(2020, 1, 1), date(2020, 1, 3)
    )
    # Already mm/day in the file — must not be multiplied by 1000.
    assert list(series.values) == [0.0, 2.5, 1.1]


def test_declared_unit_mismatch_raises(sample_manual_file):
    with pytest.raises(ValueError, match="expects 'mm/day'"):
        get_satellite_variable(
            "manual",
            file_path=sample_manual_file,
            value_column="precipitation",
            canonical_unit="m/day",
        )


def test_explicit_matching_unit_is_accepted(sample_manual_file):
    retriever = get_satellite_variable(
        "manual",
        file_path=sample_manual_file,
        value_column="precipitation",
        canonical_unit="mm/day",
    )
    assert retriever.canonical_unit == "mm/day"


def test_unknown_variable_has_no_unit_constraint(tmp_path):
    path = tmp_path / "custom.csv"
    pd.DataFrame(
        {"station_id": ["A"], "date": ["2020-01-01"], "custom_var": [7.0]}
    ).to_csv(path, index=False)
    retriever = get_satellite_variable(
        "manual", file_path=path, value_column="custom_var"
    )
    series, report = retriever.fetch("A", 0, 0, date(2020, 1, 1), date(2020, 1, 1))
    assert list(series.values) == [7.0]
    assert report.canonical_unit is None