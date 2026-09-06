"""Loading stations.csv and the measurements file (CSV or Excel).

Both CSV and Excel are
supported for the measurements file via this loader abstraction, which
normalizes either into the same internal pandas representation. CSV is the
canonical format for the repo itself (e.g. examples/).
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from .schema import Schema

STATION_REQUIRED_COLUMNS = ("station_id", "station_name", "latitude", "longitude")


class InputFileError(ValueError):
    """Raised for structural problems with an input file (not per-cell issues)."""


def load_stations(path: str | Path) -> pd.DataFrame:
    """Load stations.csv and check it has the required columns.

    Does not validate coordinate ranges — that is left to the caller if
    needed; this function only enforces the file's *structure*.
    """
    path = Path(path)
    df = pd.read_csv(path)

    missing = set(STATION_REQUIRED_COLUMNS) - set(df.columns)
    if missing:
        raise InputFileError(
            f"stations file '{path}' is missing required column(s): {sorted(missing)}"
        )

    if df["station_id"].duplicated().any():
        dupes = df.loc[df["station_id"].duplicated(), "station_id"].tolist()
        raise InputFileError(f"stations file '{path}' has duplicate station_id value(s): {dupes}")

    return df


def load_measurements(path: str | Path, schema: Schema) -> pd.DataFrame:
    """Load the measurements file (CSV or Excel) and check its structure
    against schema.yaml: unknown columns, required columns, and date
    parsing. Per-cell value validation (min/max bounds) is a separate
    step — see validators.py.
    """
    path = Path(path)

    if path.suffix.lower() == ".csv":
        df = pd.read_csv(path)
    elif path.suffix.lower() in (".xlsx", ".xls"):
        df = pd.read_excel(path)
    else:
        raise InputFileError(
            f"Unsupported measurements file type '{path.suffix}' — use .csv or .xlsx"
        )

    # Required non-parameter columns.
    missing_required = set(schema.required_columns) - set(df.columns)
    if missing_required:
        raise InputFileError(
            f"measurements file '{path}' is missing required column(s): "
            f"{sorted(missing_required)}"
        )

    # Unknown-column policy.
    unknown = set(df.columns) - schema.known_columns
    if unknown and schema.policy.on_unknown_column == "reject_file":
        raise InputFileError(
            f"measurements file '{path}' has column(s) not defined in schema.yaml: "
            f"{sorted(unknown)}. Add them to schema.yaml's `parameters` section, "
            f"or remove them from the file."
        )

    # Parse date column strictly (Gregorian, ISO 8601 — converting dates
    # from any other calendar system is the user's responsibility).
    try:
        df["date"] = pd.to_datetime(df["date"], format=schema.date_format, errors="raise")
    except (ValueError, TypeError) as exc:
        raise InputFileError(
            f"measurements file '{path}' has a 'date' value that doesn't match "
            f"the required format '{schema.date_format}' (ISO 8601, Gregorian): {exc}"
        ) from exc

    return df


def validate_station_references(
    measurements: pd.DataFrame, stations: pd.DataFrame
) -> list[str]:
    """Return any station_id values used in measurements but absent from
    stations.csv (a foreign-key check). Empty list = all references valid.
    """
    known_ids = set(stations["station_id"])
    used_ids = set(measurements["station_id"])
    return sorted(used_ids - known_ids)
