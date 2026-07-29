from pathlib import Path

from wq_framework.io.schema import Schema

SCHEMA_PATH = Path(__file__).parent.parent / "config" / "schema.yaml"


def test_schema_loads_all_parameters():
    schema = Schema.load(SCHEMA_PATH)
    expected = {"TDS", "pH", "HCO3", "Cl", "SO4", "Ca", "Mg", "Na", "K", "EC", "discharge"}
    assert expected.issubset(schema.parameters.keys())


def test_ph_has_correct_bounds():
    schema = Schema.load(SCHEMA_PATH)
    ph = schema.parameters["pH"]
    assert ph.min_value == 0
    assert ph.max_value == 14
    assert ph.is_out_of_range(-1)
    assert ph.is_out_of_range(15)
    assert not ph.is_out_of_range(7.2)


def test_tds_has_no_upper_bound():
    schema = Schema.load(SCHEMA_PATH)
    tds = schema.parameters["TDS"]
    assert tds.min_value == 0
    assert tds.max_value is None
    assert tds.is_out_of_range(-1)
    assert not tds.is_out_of_range(10_000)  # no upper sanity bound by design


def test_required_columns_and_policy_loaded():
    schema = Schema.load(SCHEMA_PATH)
    assert "station_id" in schema.required_columns
    assert "date" in schema.required_columns
    assert schema.policy.on_out_of_range == "null_cell"
    assert schema.policy.on_unknown_column == "reject_file"


def test_known_columns_includes_parameters_and_required():
    schema = Schema.load(SCHEMA_PATH)
    assert "TDS" in schema.known_columns
    assert "station_id" in schema.known_columns
