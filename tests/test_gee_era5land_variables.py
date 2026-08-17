from datetime import date
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from wq_framework.satellite import get_satellite_variable
from wq_framework.satellite.export_job import ExportJobHandle
from wq_framework.satellite.variables import gee_era5land  # noqa: F401 - registers "precipitation", "temperature"


@pytest.fixture
def mocked_pipeline(tmp_path):
    """Mocks every GEE/network touchpoint; only the pure CSV-parsing step
    runs for real, against a small local sample file."""
    sample_csv = tmp_path / "sample.csv"
    pd.DataFrame(
        {"date": ["2020-01-01", "2020-01-02"], "value": [1.0, 2.0]}
    ).to_csv(sample_csv, index=False)

    fake_handle = ExportJobHandle(
        task=MagicMock(), description="fake_job", drive_file_name="fake_job"
    )

    with patch.object(gee_era5land, "connect") as mock_connect, patch.object(
        gee_era5land, "submit_export_job", return_value=fake_handle
    ) as mock_submit, patch.object(
        gee_era5land, "wait_for_export_job"
    ) as mock_wait, patch.object(
        gee_era5land, "download_export_csv", return_value=sample_csv
    ) as mock_download:
        yield {
            "connect": mock_connect,
            "submit": mock_submit,
            "wait": mock_wait,
            "download": mock_download,
        }


def test_precipitation_uses_correct_band_name(mocked_pipeline):
    retriever = get_satellite_variable("precipitation")
    series, report = retriever.fetch(
        "DAM_UP", 32.1, 51.5, date(2020, 1, 1), date(2020, 1, 2)
    )
    _, kwargs = mocked_pipeline["submit"].call_args
    assert kwargs["band_name"] == "total_precipitation_sum"
    assert report.variable == "precipitation"
    assert list(series.values) == [1.0, 2.0]


def test_temperature_uses_correct_band_name(mocked_pipeline):
    retriever = get_satellite_variable("temperature")
    retriever.fetch("DAM_UP", 32.1, 51.5, date(2020, 1, 1), date(2020, 1, 2))
    _, kwargs = mocked_pipeline["submit"].call_args
    assert kwargs["band_name"] == "temperature_2m"


def test_default_buffer_is_15km():
    retriever = get_satellite_variable("precipitation")
    assert retriever.buffer_km == 15.0


def test_connect_called_before_submit(mocked_pipeline):
    retriever = get_satellite_variable("precipitation")
    retriever.fetch("DAM_UP", 32.1, 51.5, date(2020, 1, 1), date(2020, 1, 2))
    assert mocked_pipeline["connect"].called
    assert mocked_pipeline["submit"].called


def test_uses_era5_land_daily_aggr_collection(mocked_pipeline):
    retriever = get_satellite_variable("precipitation")
    retriever.fetch("DAM_UP", 32.1, 51.5, date(2020, 1, 1), date(2020, 1, 2))
    _, kwargs = mocked_pipeline["submit"].call_args
    assert kwargs["image_collection_id"] == "ECMWF/ERA5_LAND/DAILY_AGGR"


def test_soil_moisture_uses_correct_band_name(mocked_pipeline):
    retriever = get_satellite_variable("soil_moisture")
    retriever.fetch("DAM_UP", 32.1, 51.5, date(2020, 1, 1), date(2020, 1, 2))
    _, kwargs = mocked_pipeline["submit"].call_args
    assert kwargs["band_name"] == "volumetric_soil_water_layer_1"


def test_soil_moisture_default_buffer_is_zero():
    retriever = get_satellite_variable("soil_moisture")
    assert retriever.buffer_km == 0.0


def test_evapotranspiration_uses_correct_band_name(mocked_pipeline):
    retriever = get_satellite_variable("evapotranspiration")
    retriever.fetch("DAM_UP", 32.1, 51.5, date(2020, 1, 1), date(2020, 1, 2))
    _, kwargs = mocked_pipeline["submit"].call_args
    assert kwargs["band_name"] == "potential_evaporation_sum"


def test_evapotranspiration_default_buffer_is_15km():
    retriever = get_satellite_variable("evapotranspiration")
    assert retriever.buffer_km == 15.0


def test_solar_radiation_uses_correct_band_name(mocked_pipeline):
    retriever = get_satellite_variable("solar_radiation")
    retriever.fetch("DAM_UP", 32.1, 51.5, date(2020, 1, 1), date(2020, 1, 2))
    _, kwargs = mocked_pipeline["submit"].call_args
    assert kwargs["band_name"] == "surface_solar_radiation_downwards_sum"


def test_solar_radiation_default_buffer_is_15km():
    retriever = get_satellite_variable("solar_radiation")
    assert retriever.buffer_km == 15.0