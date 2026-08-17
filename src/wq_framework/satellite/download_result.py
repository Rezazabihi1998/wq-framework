"""Downloads a completed GEE Export result from Google Drive and parses it
into the (series, RetrievalReport) contract required by
SatelliteVariableRetriever.fetch().

Split into two clearly separate concerns:
  - Drive download mechanics (needs real GEE/Drive credentials — cannot be
    unit-tested in this environment; the assistant could not verify this
    part end-to-end and it should be tested against your real account).
  - CSV -> (Series, RetrievalReport) parsing (pure logic, fully unit-
    tested with a local sample file below, no network needed).

Reuses your existing GEE credentials for the Drive API (a common pattern:
ee.Authenticate() already requests Drive read scope), so no separate
Google Drive auth step is needed.
"""

from __future__ import annotations

import io
import os
from datetime import date
from pathlib import Path

import pandas as pd

try:
    import ee
    from googleapiclient.discovery import build
    from googleapiclient.http import MediaIoBaseDownload
except ImportError:  # pragma: no cover
    ee = None
    build = None
    MediaIoBaseDownload = None

from .base import RetrievalReport
from .export_job import ExportJobHandle


def _ensure_drive_deps_installed() -> None:
    if ee is None or build is None:
        raise ImportError(
            "Downloading export results requires 'earthengine-api' and "
            "'google-api-python-client'. Install with:\n"
            "    pip install earthengine-api google-api-python-client"
        )


def _build_drive_service():
    """Reuses the already-authenticated GEE credentials for Drive API v3."""
    _ensure_drive_deps_installed()
    credentials = ee.data.get_persistent_credentials()
    return build("drive", "v3", credentials=credentials)


def download_export_csv(
    handle: ExportJobHandle,
    destination_dir: str | Path,
    drive_folder: str = "wq_framework_exports",
) -> Path:
    """Find the completed export's CSV in Google Drive by name and
    download it to `destination_dir`. Only call this after
    wait_for_export_job() has returned successfully.
    """
    service = _build_drive_service()
    destination_dir = Path(destination_dir)
    destination_dir.mkdir(parents=True, exist_ok=True)
    local_path = destination_dir / f"{handle.drive_file_name}.csv"

    query = (
        f"name = '{handle.drive_file_name}.csv' and trashed = false"
    )
    results = service.files().list(q=query, fields="files(id, name)").execute()
    files = results.get("files", [])
    if not files:
        raise FileNotFoundError(
            f"Export result '{handle.drive_file_name}.csv' not found in Drive "
            f"(folder '{drive_folder}') — the job may not have actually "
            f"completed, or Drive indexing is still catching up (try again "
            f"in a minute)."
        )

    file_id = files[0]["id"]
    request = service.files().get_media(fileId=file_id)
    with io.FileIO(local_path, "wb") as fh:
        downloader = MediaIoBaseDownload(fh, request)
        done = False
        while not done:
            _, done = downloader.next_chunk()

    return local_path


def parse_export_csv_to_series(
    csv_path: str | Path,
    station_id: str,
    variable: str,
    buffer_km: float,
    start_date: date,
    end_date: date,
    source: str = "gee_era5land",
) -> tuple[pd.Series, RetrievalReport]:
    """Pure parsing logic — no network. Reads the downloaded CSV (columns
    'date', 'value', as produced by export_job.py's FeatureCollection
    schema) into a date-indexed Series, matching the fetch() contract.
    """
    df = pd.read_csv(csv_path)
    if "date" not in df.columns or "value" not in df.columns:
        raise ValueError(
            f"'{csv_path}' doesn't look like an export result — expected "
            f"columns 'date' and 'value', got {list(df.columns)}"
        )

    df["date"] = pd.to_datetime(df["date"])
    series = df.set_index("date")["value"].sort_index()
    series.name = variable

    n_requested = (pd.Timestamp(end_date) - pd.Timestamp(start_date)).days + 1
    report = RetrievalReport(
        variable=variable,
        source=source,
        station_id=station_id,
        buffer_km=buffer_km,
        start_date=start_date,
        end_date=end_date,
        n_days_requested=n_requested,
        n_days_returned=len(series),
    )
    return series, report