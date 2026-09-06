"""Base interface for satellite variable retrieval strategies.

Variables can come from an automated GEE connection OR a manually-supplied
offline file — three user-facing modes (automated GEE, automated GEE with a
one-time interactive authentication prompt, and fully offline) all share
this one interface.
This interface is deliberately source-agnostic — a future retriever could
pull from a different product without changing anything else in the
pipeline. Each concrete retriever is a Strategy+Registry plugin registered
via `registry.register_satellite_variable`.

This module only defines the interface and transparency record — no actual
GEE connection lives here, so it can be developed and tested independently
of network/auth concerns.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import date

import pandas as pd


@dataclass
class RetrievalReport:
    """Transparency record of one retrieval call, for logging/debugging."""

    variable: str
    source: str
    station_id: str
    buffer_km: float
    start_date: date
    end_date: date
    n_days_requested: int = 0
    n_days_returned: int = 0
    params: dict = field(default_factory=dict)

    def __str__(self) -> str:
        return (
            f"{self.variable} for station '{self.station_id}' "
            f"({self.source}, buffer={self.buffer_km}km): "
            f"{self.n_days_returned}/{self.n_days_requested} day(s) retrieved "
            f"[{self.start_date} .. {self.end_date}]"
        )


class SatelliteVariableRetriever(ABC):
    """Interface every satellite-variable retrieval strategy must implement.

    One instance is configured for one variable (e.g. precipitation) with
    its own buffer radius; `fetch` is called once per station per date
    range needed.
    """

    def __init__(self, buffer_km: float = 0.0):
        self.buffer_km = buffer_km

    @abstractmethod
    def fetch(
        self,
        station_id: str,
        latitude: float,
        longitude: float,
        start_date: date,
        end_date: date,
    ) -> tuple[pd.Series, RetrievalReport]:
        """Return a daily time series (indexed by date, inclusive of both
        endpoints) for the given station, plus a RetrievalReport.

        `station_id` is passed alongside coordinates so that retrievers
        which don't need coordinates at all — e.g. a manual/offline
        retriever reading from a user-supplied file organized by station
        identity, per the "fully offline" mode —
        can look data up by station identity rather than requiring an
        exact coordinate match. GEE-backed retrievers use `latitude`/
        `longitude` for the actual query and `station_id` only for
        caching/reporting.

        Implementations are responsible for any request-batching logic
        needed by their data source (GEE-backed retrievers, for instance,
        must work around per-request compute-time limits and account-level
        request-rate quotas) — callers only see the final combined daily
        series.
        """
        raise NotImplementedError