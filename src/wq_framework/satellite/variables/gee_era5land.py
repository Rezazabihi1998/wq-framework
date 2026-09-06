"""GEE-backed satellite variable retrievers using ERA5-Land daily
aggregates. Each concrete class only needs to declare its band name —
the shared fetch() logic (connect -> submit export -> wait -> download ->
parse) lives once in the base class here.
"""

from __future__ import annotations

from datetime import date

import pandas as pd

from ..base import RetrievalReport, SatelliteVariableRetriever
from ..download_result import download_export_csv, parse_export_csv_to_series
from ..export_job import submit_export_job, wait_for_export_job
from ..gee_client import GEEConfig, connect
from ..registry import register_satellite_variable

ERA5_LAND_DAILY_COLLECTION = "ECMWF/ERA5_LAND/DAILY_AGGR"


class ERA5LandVariableRetriever(SatelliteVariableRetriever):
    """Base class for any ERA5-Land daily variable. Subclasses just set
    `band_name` and `variable_name` class attributes.
    """

    band_name: str = None
    variable_name: str = None
    reducer_name: str = "mean"

    def __init__(
        self,
        buffer_km: float = 15.0,
        gee_config: GEEConfig | None = None,
        cache_dir: str = "cache/satellite",
        drive_folder: str = "wq_framework_exports",
        poll_interval_seconds: float = 30.0,
        timeout_seconds: float = 3600.0,
    ):
        super().__init__(buffer_km=buffer_km)
        if self.band_name is None or self.variable_name is None:
            raise NotImplementedError(
                f"{type(self).__name__} must set band_name and variable_name"
            )
        self.gee_config = gee_config
        self.cache_dir = cache_dir
        self.drive_folder = drive_folder
        self.poll_interval_seconds = poll_interval_seconds
        self.timeout_seconds = timeout_seconds

    def fetch(
        self,
        station_id: str,
        latitude: float,
        longitude: float,
        start_date: date,
        end_date: date,
    ) -> tuple[pd.Series, RetrievalReport]:
        connect(self.gee_config)  # may raise GEEAuthenticationRequired

        description = f"{station_id}_{self.variable_name}_{start_date}_{end_date}"

        handle = submit_export_job(
            image_collection_id=ERA5_LAND_DAILY_COLLECTION,
            band_name=self.band_name,
            reducer_name=self.reducer_name,
            latitude=latitude,
            longitude=longitude,
            buffer_km=self.buffer_km,
            start_date=start_date,
            end_date=end_date,
            description=description,
            drive_folder=self.drive_folder,
        )
        wait_for_export_job(
            handle,
            poll_interval_seconds=self.poll_interval_seconds,
            timeout_seconds=self.timeout_seconds,
        )
        local_csv = download_export_csv(handle, self.cache_dir, drive_folder=self.drive_folder)
        return parse_export_csv_to_series(
            local_csv,
            station_id=station_id,
            variable=self.variable_name,
            buffer_km=self.buffer_km,
            start_date=start_date,
            end_date=end_date,
        )


@register_satellite_variable("precipitation")
class PrecipitationRetriever(ERA5LandVariableRetriever):
    band_name = "total_precipitation_sum"
    variable_name = "precipitation"


@register_satellite_variable("temperature")
class TemperatureRetriever(ERA5LandVariableRetriever):
    band_name = "temperature_2m"
    variable_name = "temperature"


@register_satellite_variable("soil_moisture")
class SoilMoistureRetriever(ERA5LandVariableRetriever):
    band_name = "volumetric_soil_water_layer_1"
    variable_name = "soil_moisture"

    def __init__(self, buffer_km: float = 0.0, **kwargs):
        # Point-sampled by design: soil moisture is far more spatially
        # heterogeneous than the climate variables, so averaging over a
        # buffer would mix unrelated land-cover/soil types. Default
        # buffer is therefore overridden to 0 (the caller can still
        # override it via config if they want a buffered average).
        super().__init__(buffer_km=buffer_km, **kwargs)


@register_satellite_variable("evapotranspiration")
class EvapotranspirationRetriever(ERA5LandVariableRetriever):
    band_name = "potential_evaporation_sum"
    variable_name = "evapotranspiration"
    # buffer_km default (15.0) inherited from the base class — same as
    # precipitation/temperature, unlike soil moisture's point-sample default


@register_satellite_variable("solar_radiation")
class SolarRadiationRetriever(ERA5LandVariableRetriever):
    band_name = "surface_solar_radiation_downwards_sum"
    variable_name = "solar_radiation"
    # buffer_km default (15.0) inherited from the base class