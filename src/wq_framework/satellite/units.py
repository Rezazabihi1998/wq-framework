"""Unit conversion for satellite variables.

Every retriever must return its series in a single canonical unit per
variable, regardless of the source it read from. Raw ERA5-Land bands are
stored in units that are not directly usable as model inputs — depths in
metres, temperatures in kelvin, radiation in joules per square metre — so
a GEE-backed retriever and a user-supplied offline table would otherwise
deliver numbers on wildly different scales for the same physical quantity.

`UnitTransform` holds one variable's conversion as data rather than as a
scattered `* 1000` in the retrieval code, so that:

  - the conversion is declared once, next to the band it belongs to;
  - the same object can be applied by any retriever, keeping the GEE and
    manual paths consistent;
  - what was done (and how many values were clamped) is recorded in the
    retrieval report instead of happening invisibly.

A `clamp_min` is not a cosmetic guard: for accumulation bands, Earth
Engine documents that GRIB packing can produce small negative values that
have no physical meaning, and for the evaporation band a positive raw
value means condensation rather than evaporation. Clamping floors those at
zero and counts them, so the count can be inspected rather than assumed.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class UnitTransform:
    """An affine conversion from a raw stored unit to a canonical unit.

    `value_canonical = value_raw * scale + offset`, then floored at
    `clamp_min` if one is given.
    """

    raw_unit: str
    canonical_unit: str
    scale: float = 1.0
    offset: float = 0.0
    clamp_min: float | None = None

    def apply(self, series: pd.Series) -> tuple[pd.Series, int]:
        """Return the converted series and the number of values clamped.

        NaN values pass through untouched and are never counted as
        clamped. The input series is not mutated.
        """
        converted = series.astype(float) * self.scale + self.offset

        n_clamped = 0
        if self.clamp_min is not None:
            below = converted < self.clamp_min
            n_clamped = int(below.sum())
            if n_clamped:
                converted = converted.mask(below, self.clamp_min)

        return converted, n_clamped

    def describe(self) -> str:
        """Human-readable summary, stored in the retrieval report."""
        parts = []
        if self.scale != 1.0:
            parts.append(f"x{self.scale:g}")
        if self.offset != 0.0:
            parts.append(f"{self.offset:+g}")
        steps = " ".join(parts) if parts else "no scaling"
        text = f"{self.raw_unit} -> {self.canonical_unit} ({steps}"
        if self.clamp_min is not None:
            text += f", floored at {self.clamp_min:g}"
        return text + ")"


# --- Per-variable canonical transforms -------------------------------------
#
# Canonical units, and why each conversion is what it is:
#
# precipitation       m/day              -> mm/day    (x1000)
# temperature         K                  -> degC      (-273.15)
# evapotranspiration  m/day, upward < 0  -> mm/day    (x-1000)
# solar_radiation     J/m2/day           -> MJ/m2/day (x1e-6)
# soil_moisture       m3/m3              -> m3/m3     (already volumetric)

PRECIPITATION = UnitTransform(
    raw_unit="m/day",
    canonical_unit="mm/day",
    scale=1000.0,
    clamp_min=0.0,
)

TEMPERATURE = UnitTransform(
    raw_unit="K",
    canonical_unit="degC",
    offset=-273.15,
    # No clamp: a negative value in degrees Celsius is simply sub-freezing.
)

EVAPOTRANSPIRATION = UnitTransform(
    # ERA5 follows the meteorological sign convention where downward
    # fluxes are positive, so evaporation (an upward flux) is negative.
    # Negating turns it into a positive water depth; flooring at zero
    # drops days whose flux was condensation, which is not evaporative
    # demand and would otherwise subtract from a multi-day total.
    raw_unit="m/day (negative = upward flux)",
    canonical_unit="mm/day",
    scale=-1000.0,
    clamp_min=0.0,
)

SOLAR_RADIATION = UnitTransform(
    raw_unit="J/m2/day",
    canonical_unit="MJ/m2/day",
    scale=1e-6,
    clamp_min=0.0,
)

SOIL_MOISTURE = UnitTransform(
    # Already a volumetric fraction over the 0-7 cm layer; no scaling.
    raw_unit="m3/m3",
    canonical_unit="m3/m3",
    clamp_min=0.0,
)


CANONICAL_UNITS = {
    "precipitation": PRECIPITATION.canonical_unit,
    "temperature": TEMPERATURE.canonical_unit,
    "evapotranspiration": EVAPOTRANSPIRATION.canonical_unit,
    "solar_radiation": SOLAR_RADIATION.canonical_unit,
    "soil_moisture": SOIL_MOISTURE.canonical_unit,
}
"""Canonical unit expected of every retriever, keyed by variable name."""
