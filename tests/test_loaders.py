from pathlib import Path

import pandas as pd
import pytest

from wq_framework.io.loaders import (
    InputFileError,
    load_measurements,
    load_stations,
    validate_station_references,
)
from wq_framework.io.schema import Schema

SCHEMA_PATH = Path(__file__).parent.parent / "config" / "schema.yaml"
EXAMPLES = Path(__file__).parent.parent / "examples"


@pytest.fixture
def schema():
    return Schema.load(SCHEMA_PATH)


def test_load_stations_ok():
    df = load_stations(EXAMPLES / "stations.csv")
    assert set(df["station_id"]) == {"DAM_UP", "DAM_DOWN"}


def test_load_stations_missing_column(tmp_path):
    bad = tmp_path / "stations.csv"
    bad.write_text("station_id,latitude\nA,1.0\n")
    with pytest.raises(InputFileError, match="missing required column"):
        load_stations(bad)


def test_load_stations_duplicate_id(tmp_path):
    bad = tmp_path / "stations.csv"
    bad.write_text(
        "station_id,station_name,latitude,longitude\n"
        "A,Station A,1.0,2.0\n"
        "A,Station A dup,1.1,2.1\n"
    )
    with pytest.raises(InputFileError, match="duplicate station_id"):
        load_stations(bad)


def test_load_measurements_ok(schema):
    df = load_measurements(EXAMPLES / "measurements.csv", schema)
    assert len(df) == 5
    assert pd.api.types.is_datetime64_any_dtype(df["date"])


def test_load_measurements_unknown_column_rejected(tmp_path, schema):
    bad = tmp_path / "measurements.csv"
    bad.write_text("station_id,date,TDS,mystery_column\nA,2020-01-01,10,99\n")
    with pytest.raises(InputFileError, match="not defined in schema.yaml"):
        load_measurements(bad, schema)


def test_load_measurements_bad_date_format(tmp_path, schema):
    bad = tmp_path / "measurements.csv"
    # Non-ISO format (e.g. a Shamsi-style or DD/MM/YYYY date) must be rejected —
    # calendar conversion is explicitly the user's responsibility.
    bad.write_text("station_id,date,TDS\nA,21/03/2010,10\n")
    with pytest.raises(InputFileError, match="date"):
        load_measurements(bad, schema)


def test_load_measurements_missing_required_column(tmp_path, schema):
    bad = tmp_path / "measurements.csv"
    bad.write_text("date,TDS\n2020-01-01,10\n")  # no station_id
    with pytest.raises(InputFileError, match="missing required column"):
        load_measurements(bad, schema)


def test_validate_station_references_detects_unknown_station(schema):
    stations = load_stations(EXAMPLES / "stations.csv")
    measurements = load_measurements(EXAMPLES / "measurements.csv", schema)
    # sanity check: examples are consistent with each other
    assert validate_station_references(measurements, stations) == []

    measurements.loc[0, "station_id"] = "UNKNOWN_STATION"
    unknown = validate_station_references(measurements, stations)
    assert unknown == ["UNKNOWN_STATION"]


def test_supports_xlsx(tmp_path, schema):
    df = load_measurements(EXAMPLES / "measurements.csv", schema)
    xlsx_path = tmp_path / "measurements.xlsx"
    df_for_excel = df.copy()
    df_for_excel["date"] = df_for_excel["date"].dt.strftime("%Y-%m-%d")
    df_for_excel.to_excel(xlsx_path, index=False)

    reloaded = load_measurements(xlsx_path, schema)
    assert len(reloaded) == len(df)
