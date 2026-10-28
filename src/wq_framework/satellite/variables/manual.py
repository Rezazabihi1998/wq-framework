"""Fully offline/manual satellite data retriever.

Registered under the generic name "manual" (not per-variable) since one
class serves any variable — the specific column to read is passed at
construction time via `value_column`.

The file is expected to already hold values in the variable's canonical
unit (see satellite/units.py): a user preparing this table by hand is
working with the physical quantity they want the models to see, not with
Earth Engine's internal storage units. No conversion is applied, and the
report says so explicitly rather than leaving the unit ambiguous.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path

import pandas as pd

from .. import units
from ..base import RetrievalReport, SatelliteVariableRetriever
from ..registry import register_satellite_variable


@register_satellite_variable("manual")
class ManualFileRetriever(SatelliteVariableRetriever):
    """Expects a CSV/Excel file with columns: station_id, date, and one
    column per variable (e.g. 'precipitation', 'temperature', ...) — the
    same general shape as the main measurements file, just for satellite
    variables instead of water quality parameters.

    `canonical_unit` may be supplied to state which unit the column is in;
    it is recorded in the report and never used to convert. Passing a
    value that disagrees with the variable's expected canonical unit
    raises, since a silent mismatch between the two retrieval modes would
    be invisible downstream.
    """

    def __init__(
        self,
        file_path: str | Path,
        value_column: str,
        buffer_km: float = 0.0,
        canonical_unit: str | None = None,
    ):
        super().__init__(buffer_km=buffer_km)
        self.file_path = Path(file_path)
        self.value_column = value_column
        self.canonical_unit = canonical_unit or units.CANONICAL_UNITS.get(value_column)
        expected = units.CANONICAL_UNITS.get(value_column)
        if expected is not None and self.canonical_unit != expected:
            raise ValueError(
                f"manual satellite column '{value_column}' is declared as "
                f"'{self.canonical_unit}' but the framework expects '{expected}' "
                f"for this variable — convert the column before loading it"
            )
        self._data: pd.DataFrame | None = None  # loaded lazily, cached across calls

    def _load(self) -> pd.DataFrame:
        if self._data is None:
            if self.file_path.suffix.lower() == ".csv":
                df = pd.read_csv(self.file_path)
            else:
                df = pd.read_excel(self.file_path)

            required = {"station_id", "date", self.value_column}
            missing = required - set(df.columns)
            if missing:
                raise ValueError(
                    f"manual satellite file '{self.file_path}' is missing "
                    f"required column(s): {sorted(missing)}"
                )
            df["date"] = pd.to_datetime(df["date"])
            self._data = df
        return self._data

    def fetch(
        self,
        station_id: str,
        latitude: float,
        longitude: float,
        start_date: date,
        end_date: date,
    ) -> tuple[pd.Series, RetrievalReport]:
        df = self._load()
        station_rows = df[
            (df["station_id"] == station_id)
            & (df["date"] >= pd.Timestamp(start_date))
            & (df["date"] <= pd.Timestamp(end_date))
        ]

        series = station_rows.set_index("date")[self.value_column].sort_index()
        series.name = self.value_column

        n_requested = (pd.Timestamp(end_date) - pd.Timestamp(start_date)).days + 1
        report = RetrievalReport(
            variable=self.value_column,
            source="manual",
            station_id=station_id,
            buffer_km=self.buffer_km,
            start_date=start_date,
            end_date=end_date,
            n_days_requested=n_requested,
            n_days_returned=len(series),
            canonical_unit=self.canonical_unit,
        )
        return series, report