"""Minimal demo of the input-loading and validation stage.

Run from the repo root:
    python examples/run_example.py
"""

from pathlib import Path

from wq_framework.io import Schema, load_measurements, load_stations, run_validation_gate
from wq_framework.io.loaders import validate_station_references

ROOT = Path(__file__).parent.parent


def main():
    schema = Schema.load(ROOT / "config" / "schema.yaml")

    stations = load_stations(ROOT / "examples" / "stations.csv")
    print(f"Loaded {len(stations)} station(s): {list(stations['station_id'])}")

    measurements = load_measurements(ROOT / "examples" / "measurements.csv", schema)
    print(f"Loaded {len(measurements)} measurement row(s)")

    unknown_stations = validate_station_references(measurements, stations)
    if unknown_stations:
        raise SystemExit(f"measurements reference unknown station_id(s): {unknown_stations}")

    # Auto-confirm for this non-interactive demo. In a real CLI run, omit
    # confirm_callback to get the interactive [y/N] prompt.
    cleaned = run_validation_gate(measurements, schema, confirm_callback=lambda report: True)

    n_flagged = cleaned.isna().sum().sum()
    print(f"Validation complete. {n_flagged} cell(s) were nulled (see report above).")
    print(cleaned)


if __name__ == "__main__":
    main()
