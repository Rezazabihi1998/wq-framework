"""Submitting and polling a single GEE batch Export job (project brief
Section 7): builds the buffered-region reduction over an entire
ImageCollection date range as ONE server-side computation, submits it as
an Export task (not a synchronous call), and polls until it finishes.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from datetime import date

try:
    import ee
except ImportError:  # pragma: no cover
    ee = None

from .gee_client import ensure_ee_installed


@dataclass
class ExportJobHandle:
    """Reference to a submitted (or in-progress) GEE Export task."""

    task: "ee.batch.Task"
    description: str
    drive_file_name: str


class ExportJobFailed(Exception):
    def __init__(self, description: str, error_message: str):
        self.description = description
        self.error_message = error_message
        super().__init__(f"Export job '{description}' failed: {error_message}")


class ExportJobTimedOut(Exception):
    def __init__(self, description: str, elapsed_seconds: float):
        self.description = description
        self.elapsed_seconds = elapsed_seconds
        super().__init__(
            f"Export job '{description}' did not finish within "
            f"{elapsed_seconds:.0f}s — it may still be running in GEE; "
            f"check the Tasks tab at code.earthengine.google.com."
        )


def submit_export_job(
    image_collection_id: str,
    band_name: str,
    reducer_name: str,
    latitude: float,
    longitude: float,
    buffer_km: float,
    start_date: date,
    end_date: date,
    description: str,
    drive_folder: str = "wq_framework_exports",
) -> ExportJobHandle:
    """Build the reduceRegion-over-collection computation for the WHOLE
    date range at once and submit it as a single Export task. Returns
    immediately (does not wait) — see wait_for_export_job() below.
    """
    ensure_ee_installed()

    point = ee.Geometry.Point([longitude, latitude])
    region = point.buffer(buffer_km * 1000) if buffer_km > 0 else point
    reducer = getattr(ee.Reducer, reducer_name)()

    collection = (
        ee.ImageCollection(image_collection_id)
        .filterDate(str(start_date), str(end_date))
        .select(band_name)
    )

    def _reduce_image(image):
        value = image.reduceRegion(reducer=reducer, geometry=region, scale=1000)
        return ee.Feature(
            None,
            {"date": image.date().format("YYYY-MM-dd"), "value": value.get(band_name)},
        )

    feature_collection = ee.FeatureCollection(collection.map(_reduce_image))

    task = ee.batch.Export.table.toDrive(
        collection=feature_collection,
        description=description,
        folder=drive_folder,
        fileNamePrefix=description,
        fileFormat="CSV",
    )
    task.start()

    return ExportJobHandle(task=task, description=description, drive_file_name=description)


def wait_for_export_job(
    handle: ExportJobHandle,
    poll_interval_seconds: float = 30.0,
    timeout_seconds: float = 3600.0,
    sleep_fn=time.sleep,
    time_fn=time.monotonic,
) -> None:
    """Block, polling handle.task.status(), until the job COMPLETES,
    raising ExportJobFailed or ExportJobTimedOut otherwise.

    `sleep_fn`/`time_fn` are injectable only so tests can simulate the
    passage of time without actually waiting — production callers never
    need to pass them.
    """
    start = time_fn()
    while True:
        status = handle.task.status()
        state = status.get("state")

        if state == "COMPLETED":
            return
        if state in ("FAILED", "CANCELLED"):
            raise ExportJobFailed(handle.description, status.get("error_message", state))

        elapsed = time_fn() - start
        if elapsed >= timeout_seconds:
            raise ExportJobTimedOut(handle.description, elapsed)

        sleep_fn(poll_interval_seconds)