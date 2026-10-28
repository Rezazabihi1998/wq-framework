"""Tests for the GEE-backed ERA5-Land retrievers.

Two things are checked here, deliberately kept separate:

  - the wiring: which band, which collection, which buffer radius, and in
    what order the network steps happen;
  - the boundary contract: that values leave the retriever in canonical
    units rather than in ERA5-Land's storage units.

The second matters because the manual/offline mode must be interchangeable
with this one. A retriever that silently returned metres where the manual
path returned millimetres would produce two different model inputs for the
same physical quantity, and nothing downstream would notice.
"""

from datetime import date
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from wq_framework.satellite import get_satellite_variable
from wq_framework.satellite.export_job import ExportJobHandle
from wq_framework.satellite.variables import (  # noqa: F401 - import registers all five
    gee_era5land,
)


# variable name -> (band, default buffer_km)
VARIABLES = {
    "precipitation": ("total_precipitation_sum", 15.0),
    "temperature": ("temperature_2m", 15.0),
    "evapotranspiration": ("potential_evaporation_sum", 15.0),
    "solar_radiation": ("surface_solar_radiation_downwards_sum", 15.0),
    "soil_moisture": ("volumetric_soil_water_layer_1", 0.0),
}


def _fetch_with_raw_values(tmp_path, variable, raw_values):
    """Run one retriever over a CSV holding `raw_values` exactly as
    ERA5-Land stores them, with every network touchpoint mocked.

    Returns (series, report, submit_mock). The mock is returned rather
    than asserted on inside, so callers can inspect how the export job was
    submitted without the helper deciding what matters.
    """
    csv_path = tmp_path / f"{variable}.csv"
    pd.DataFrame(
        {
            "date": pd.date_range("2020-01-01", periods=len(raw_values)).strftime(
                "%Y-%m-%d"
            ),
            "value": raw_values,
        }
    ).to_csv(csv_path, index=False)

    handle = ExportJobHandle(task=MagicMock(), description="job", drive_file_name="job")

    with patch.object(gee_era5land, "connect"), patch.object(
        gee_era5land, "submit_export_job", return_value=handle
    ) as submit, patch.object(gee_era5land, "wait_for_export_job"), patch.object(
        gee_era5land, "download_export_csv", return_value=csv_path
    ):
        retriever = get_satellite_variable(variable)
        series, report = retriever.fetch(
            "DAM_UP",
            32.1,
            51.5,
            date(2020, 1, 1),
            date(2020, 1, 1 + len(raw_values) - 1),
        )

    return series, report, submit


# --- Wiring ---------------------------------------------------------------


@pytest.mark.parametrize("variable", sorted(VARIABLES))
def test_uses_the_documented_band_and_collection(tmp_path, variable):
    expected_band, _ = VARIABLES[variable]
    _, report, submit = _fetch_with_raw_values(tmp_path, variable, [0.0])

    _, kwargs = submit.call_args
    assert kwargs["band_name"] == expected_band
    assert kwargs["image_collection_id"] == "ECMWF/ERA5_LAND/DAILY_AGGR"
    assert report.variable == variable


@pytest.mark.parametrize("variable", sorted(VARIABLES))
def test_default_buffer_matches_the_variable(variable):
    _, expected_buffer = VARIABLES[variable]
    assert get_satellite_variable(variable).buffer_km == expected_buffer


@pytest.mark.parametrize("variable", sorted(VARIABLES))
def test_every_variable_declares_a_unit_transform(variable):
    # A variable added without a transform would reach the models in
    # ERA5-Land's storage unit, so fail loudly here rather than silently
    # there.
    assert get_satellite_variable(variable).unit_transform is not None


@pytest.mark.parametrize("variable", sorted(VARIABLES))
def test_report_states_the_canonical_unit(tmp_path, variable):
    _, report, _ = _fetch_with_raw_values(tmp_path, variable, [0.0])
    assert report.canonical_unit is not None


def test_connect_runs_before_the_export_is_submitted(tmp_path):
    csv_path = tmp_path / "s.csv"
    pd.DataFrame({"date": ["2020-01-01"], "value": [0.001]}).to_csv(
        csv_path, index=False
    )
    handle = ExportJobHandle(task=MagicMock(), description="job", drive_file_name="job")
    order = MagicMock()

    with patch.object(gee_era5land, "connect") as connect_mock, patch.object(
        gee_era5land, "submit_export_job", return_value=handle
    ) as submit_mock, patch.object(
        gee_era5land, "wait_for_export_job"
    ), patch.object(
        gee_era5land, "download_export_csv", return_value=csv_path
    ):
        order.attach_mock(connect_mock, "connect")
        order.attach_mock(submit_mock, "submit")
        get_satellite_variable("precipitation").fetch(
            "DAM_UP", 32.1, 51.5, date(2020, 1, 1), date(2020, 1, 1)
        )

    called = [entry[0] for entry in order.mock_calls]
    assert called.index("connect") < called.index("submit")


# --- Unit contract at the retriever boundary ------------------------------
#
# Each case uses a raw value that is realistic for that band, so the
# expected canonical value follows directly from the conversion table
# instead of having to be taken on trust.


def test_precipitation_returns_mm_per_day(tmp_path):
    # 0.012 m of water over one day is 12 mm.
    series, report, _ = _fetch_with_raw_values(
        tmp_path, "precipitation", [0.012, 0.0]
    )
    assert series.iloc[0] == pytest.approx(12.0)
    assert report.canonical_unit == "mm/day"


def test_precipitation_floors_packing_noise_and_reports_it(tmp_path):
    series, report, _ = _fetch_with_raw_values(
        tmp_path, "precipitation", [-0.0002, 0.001]
    )
    assert series.iloc[0] == pytest.approx(0.0)
    assert report.n_values_clamped == 1
    assert "clamped" in str(report)


def test_temperature_returns_degrees_celsius(tmp_path):
    series, report, _ = _fetch_with_raw_values(
        tmp_path, "temperature", [295.15, 283.15]
    )
    assert series.iloc[0] == pytest.approx(22.0)
    assert series.iloc[1] == pytest.approx(10.0)
    assert report.canonical_unit == "degC"


def test_temperature_keeps_negative_celsius_unclamped(tmp_path):
    # Sub-freezing is physically meaningful and must not be floored.
    series, report, _ = _fetch_with_raw_values(tmp_path, "temperature", [263.15])
    assert series.iloc[0] == pytest.approx(-10.0)
    assert report.n_values_clamped == 0


def test_evapotranspiration_returns_positive_mm_per_day(tmp_path):
    # Stored as a negative (upward) flux in metres.
    series, report, _ = _fetch_with_raw_values(
        tmp_path, "evapotranspiration", [-0.003, -0.001]
    )
    assert series.iloc[0] == pytest.approx(3.0)
    assert series.iloc[1] == pytest.approx(1.0)
    assert report.canonical_unit == "mm/day"
    assert report.n_values_clamped == 0


def test_evapotranspiration_floors_condensation_days(tmp_path):
    # A positive raw value is condensation, not evaporative demand.
    series, report, _ = _fetch_with_raw_values(
        tmp_path, "evapotranspiration", [0.0005, -0.002]
    )
    assert series.iloc[0] == pytest.approx(0.0)
    assert series.iloc[1] == pytest.approx(2.0)
    assert report.n_values_clamped == 1


def test_solar_radiation_returns_megajoules_per_m2_per_day(tmp_path):
    series, report, _ = _fetch_with_raw_values(
        tmp_path, "solar_radiation", [2.0e7, 1.5e7]
    )
    assert series.iloc[0] == pytest.approx(20.0)
    assert series.iloc[1] == pytest.approx(15.0)
    assert report.canonical_unit == "MJ/m2/day"


def test_soil_moisture_is_not_rescaled(tmp_path):
    series, report, _ = _fetch_with_raw_values(tmp_path, "soil_moisture", [0.28])
    assert series.iloc[0] == pytest.approx(0.28)
    assert report.canonical_unit == "m3/m3"


def test_report_records_the_conversion_applied(tmp_path):
    _, report, _ = _fetch_with_raw_values(tmp_path, "evapotranspiration", [-0.003])
    assert "unit_conversion" in report.params
    assert "mm/day" in report.params["unit_conversion"]