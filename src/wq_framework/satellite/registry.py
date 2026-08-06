"""Name -> class registry for satellite variable retrieval strategies.

Usage:
    @register_satellite_variable("precipitation")
    class PrecipitationRetriever(SatelliteVariableRetriever):
        ...

    retriever = get_satellite_variable("precipitation", buffer_km=15)
"""

from __future__ import annotations

from .base import SatelliteVariableRetriever

_SATELLITE_VARIABLE_REGISTRY: dict[str, type[SatelliteVariableRetriever]] = {}


def register_satellite_variable(name: str):
    def decorator(
        cls: type[SatelliteVariableRetriever],
    ) -> type[SatelliteVariableRetriever]:
        if name in _SATELLITE_VARIABLE_REGISTRY:
            raise ValueError(f"satellite variable '{name}' is already registered")
        _SATELLITE_VARIABLE_REGISTRY[name] = cls
        return cls

    return decorator


def get_satellite_variable(name: str, **params) -> SatelliteVariableRetriever:
    if name not in _SATELLITE_VARIABLE_REGISTRY:
        available = sorted(_SATELLITE_VARIABLE_REGISTRY.keys())
        raise ValueError(
            f"unknown satellite variable '{name}'. Available: {available}. "
            f"Register a new one with @register_satellite_variable('{name}')."
        )
    return _SATELLITE_VARIABLE_REGISTRY[name](**params)


def available_satellite_variables() -> list[str]:
    return sorted(_SATELLITE_VARIABLE_REGISTRY.keys())