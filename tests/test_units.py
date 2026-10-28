import numpy as np
import pandas as pd
import pytest

from wq_framework.satellite import units
from wq_framework.satellite.units import UnitTransform


def _series(values, name="x"):
    return pd.Series(values, dtype=float, name=name)


def test_scaling_only():
    transform = UnitTransform(raw_unit="m/day", canonical_unit="mm/day", scale=1000.0)
    out, n_clamped = transform.apply(_series([0.0, 0.001, 0.0025]))
    assert list(out) == [0.0, 1.0, 2.5]
    assert n_clamped == 0


def test_offset_only_leaves_scale_at_one():
    transform = UnitTransform(raw_unit="K", canonical_unit="degC", offset=-273.15)
    out, _ = transform.apply(_series([273.15, 300.15]))
    assert out.iloc[0] == pytest.approx(0.0)
    assert out.iloc[1] == pytest.approx(27.0)


def test_clamp_counts_and_floors_values():
    transform = UnitTransform(
        raw_unit="m/day", canonical_unit="mm/day", scale=1000.0, clamp_min=0.0
    )
    out, n_clamped = transform.apply(_series([-0.000001, 0.0, 0.002]))
    assert list(out) == [0.0, 0.0, 2.0]
    assert n_clamped == 1


def test_clamp_counts_negative_values_only_not_nan():
    transform = UnitTransform(raw_unit="x", canonical_unit="y", clamp_min=0.0)
    out, n_clamped = transform.apply(_series([-1.0, np.nan, 2.0]))
    assert out.isna().sum() == 1
    assert n_clamped == 1
    assert out.iloc[0] == 0.0


def test_no_clamp_keeps_negative_values():
    transform = UnitTransform(raw_unit="K", canonical_unit="degC", offset=-273.15)
    out, n_clamped = transform.apply(_series([263.15]))
    assert out.iloc[0] == pytest.approx(-10.0)
    assert n_clamped == 0


def test_input_series_is_not_mutated():
    original = _series([0.001, -0.002])
    before = original.copy()
    units.PRECIPITATION.apply(original)
    pd.testing.assert_series_equal(original, before)


def test_describe_includes_scale_offset_and_floor():
    assert "x1000" in units.PRECIPITATION.describe()
    assert "mm/day" in units.PRECIPITATION.describe()
    assert "-273.15" in units.TEMPERATURE.describe()
    assert "floored at 0" in units.SOIL_MOISTURE.describe()


def test_describe_says_no_scaling_when_identity():
    transform = UnitTransform(raw_unit="a", canonical_unit="b")
    assert "no scaling" in transform.describe()


# --- Per-variable canonical transforms -------------------------------------


def test_precipitation_converts_metres_to_millimetres():
    out, _ = units.PRECIPITATION.apply(_series([0.012]))
    assert out.iloc[0] == pytest.approx(12.0)


def test_precipitation_floors_grib_packing_noise():
    out, n_clamped = units.PRECIPITATION.apply(_series([-0.0002]))
    assert out.iloc[0] == 0.0
    assert n_clamped == 1


def test_temperature_converts_kelvin_to_celsius():
    out, _ = units.TEMPERATURE.apply(_series([295.15]))
    assert out.iloc[0] == pytest.approx(22.0)


def test_evapotranspiration_negates_upward_flux():
    # ERA5 stores evaporation as a negative (upward) flux; canonical is
    # a positive depth of water lost.
    out, n_clamped = units.EVAPOTRANSPIRATION.apply(_series([-0.003]))
    assert out.iloc[0] == pytest.approx(3.0)
    assert n_clamped == 0


def test_evapotranspiration_floors_condensation_days():
    # A positive raw value means condensation, not evaporative demand.
    out, n_clamped = units.EVAPOTRANSPIRATION.apply(_series([0.0005, -0.002]))
    assert out.iloc[0] == 0.0
    assert out.iloc[1] == pytest.approx(2.0)
    assert n_clamped == 1


def test_solar_radiation_converts_joules_to_megajoules():
    out, _ = units.SOLAR_RADIATION.apply(_series([2.0e7]))
    assert out.iloc[0] == pytest.approx(20.0)


def test_soil_moisture_passes_through_unchanged():
    out, _ = units.SOIL_MOISTURE.apply(_series([0.28]))
    assert out.iloc[0] == pytest.approx(0.28)


def test_canonical_units_covers_every_shipped_variable():
    assert units.CANONICAL_UNITS == {
        "precipitation": "mm/day",
        "temperature": "degC",
        "evapotranspiration": "mm/day",
        "solar_radiation": "MJ/m2/day",
        "soil_moisture": "m3/m3",
    }
