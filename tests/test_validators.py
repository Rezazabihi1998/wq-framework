from pathlib import Path

import pandas as pd
import pytest

from wq_framework.io.loaders import load_measurements
from wq_framework.io.schema import Schema
from wq_framework.io.validators import SchemaValidator, run_validation_gate

SCHEMA_PATH = Path(__file__).parent.parent / "config" / "schema.yaml"
EXAMPLES = Path(__file__).parent.parent / "examples"


@pytest.fixture
def schema():
    return Schema.load(SCHEMA_PATH)


@pytest.fixture
def measurements(schema):
    return load_measurements(EXAMPLES / "measurements.csv", schema)


def test_detects_all_known_invalid_cells(schema, measurements):
    report = SchemaValidator(schema).validate(measurements)
    # example file has: TDS=-410, Cl=-78, K=-2.9 (row 3) and pH=15.2 (row 5)
    flagged = {(i.parameter, i.reason) for i in report.issues}
    assert ("TDS", "below minimum (0)") in flagged
    assert ("Cl", "below minimum (0)") in flagged
    assert ("K", "below minimum (0)") in flagged
    assert ("pH", "above maximum (14)") in flagged
    assert len(report.issues) == 4


def test_only_offending_cell_is_nulled_rest_of_row_preserved(schema, measurements):
    report = SchemaValidator(schema).validate(measurements)
    bad_row = report.cleaned_df[report.cleaned_df["TDS"].isna()]
    assert len(bad_row) == 1
    row = bad_row.iloc[0]
    # TDS was nulled...
    assert pd.isna(row["TDS"])
    # ...but sibling columns in the same row are untouched
    assert row["HCO3"] == 190
    assert row["Ca"] == 55


def test_no_row_or_file_is_ever_dropped(schema, measurements):
    report = SchemaValidator(schema).validate(measurements)
    assert len(report.cleaned_df) == len(measurements)  # same row count


def test_clean_data_produces_no_issues(schema):
    clean_df = pd.DataFrame(
        {
            "station_id": ["A", "A"],
            "date": pd.to_datetime(["2020-01-01", "2020-02-01"]),
            "TDS": [100, 110],
            "pH": [7.1, 7.4],
        }
    )
    report = SchemaValidator(schema).validate(clean_df)
    assert not report.has_issues
    assert report.summary() == "No invalid values found."


def test_gate_proceeds_when_confirmed(schema, measurements):
    result = run_validation_gate(measurements, schema, confirm_callback=lambda report: True)
    assert result["TDS"].isna().sum() == 1


def test_gate_halts_when_declined(schema, measurements):
    with pytest.raises(RuntimeError, match="Execution halted"):
        run_validation_gate(measurements, schema, confirm_callback=lambda report: False)


def test_gate_skips_confirmation_when_no_issues(schema):
    clean_df = pd.DataFrame(
        {
            "station_id": ["A"],
            "date": pd.to_datetime(["2020-01-01"]),
            "TDS": [100],
        }
    )
    calls = []
    result = run_validation_gate(
        clean_df, schema, confirm_callback=lambda report: calls.append(1) or True
    )
    assert calls == []  # confirm_callback never invoked — nothing to confirm
    assert len(result) == 1
