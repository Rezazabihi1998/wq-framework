from datetime import date

import pandas as pd
import pytest

from wq_framework.satellite.download_result import parse_export_csv_to_series


@pytest.fixture
def sample_export_csv(tmp_path):
    csv_path = tmp_path / "DAM_UP_precipitation.csv"
    pd.DataFrame(
        {
            "date": ["2020-01-01", "2020-01-02", "2020-01-03"],
            "value": [0.0, 2.5, 1.1],
        }
    ).to_csv(csv_path, index=False)
    return csv_path


def test_parses_series_correctly(sample_export_csv):
    series, report = parse_export_csv_to_series(
        sample_export_csv,
        station_id="DAM_UP",
        variable="precipitation",
        buffer_km=15.0,
        start_date=date(2020, 1, 1),
        end_date=date(2020, 1, 3),
    )
    assert list(series.values) == [0.0, 2.5, 1.1]
    assert series.name == "precipitation"
    assert series.index[0] == pd.Timestamp("2020-01-01")


def test_report_fields_populated(sample_export_csv):
    _, report = parse_export_csv_to_series(
        sample_export_csv,
        station_id="DAM_UP",
        variable="precipitation",
        buffer_km=15.0,
        start_date=date(2020, 1, 1),
        end_date=date(2020, 1, 3),
    )
    assert report.station_id == "DAM_UP"
    assert report.n_days_requested == 3
    assert report.n_days_returned == 3
    assert report.source == "gee_era5land"


def test_missing_value_days_reflected_in_report(tmp_path):
    # GEE sometimes has fewer returned rows than requested days (e.g. a
    # gap in the source collection) — report should reflect that honestly.
    csv_path = tmp_path / "partial.csv"
    pd.DataFrame({"date": ["2020-01-01", "2020-01-03"], "value": [0.0, 1.1]}).to_csv(
        csv_path, index=False
    )
    _, report = parse_export_csv_to_series(
        csv_path,
        station_id="DAM_UP",
        variable="precipitation",
        buffer_km=15.0,
        start_date=date(2020, 1, 1),
        end_date=date(2020, 1, 3),
    )
    assert report.n_days_requested == 3
    assert report.n_days_returned == 2


def test_raises_on_wrong_columns(tmp_path):
    bad_csv = tmp_path / "bad.csv"
    pd.DataFrame({"foo": [1], "bar": [2]}).to_csv(bad_csv, index=False)
    with pytest.raises(ValueError, match="doesn't look like an export result"):
        parse_export_csv_to_series(
            bad_csv,
            station_id="DAM_UP",
            variable="precipitation",
            buffer_km=15.0,
            start_date=date(2020, 1, 1),
            end_date=date(2020, 1, 1),
        )