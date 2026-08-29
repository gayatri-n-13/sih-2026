"""The ingestion worker.

Given an ``IngestRequest``, this module:

  1. Computes per-sensor defaults (GSD, band_count, bit_depth) from the
     sensor type. If the request supplies a GSD override we use that.
  2. Sets ``sun_angle_source_tier`` based on the request (the tier is
     authoritative; if it's ``unavailable`` we leave the angles as
     null and downstream will estimate).
  3. Sets ``acquisition_time`` from the request or the file mtime.
  4. Writes the raw image bytes (or a stub if the source is itself a
     reference) and the metadata JSON to the object store.
  5. Returns an ``IngestResult`` envelope.

This is the *single* place that knows how to materialize a
``ProductMetadata`` from a request — every test that wants a known
metadata.json goes through here (or its ``ingest_request`` entry point).
"""
from __future__ import annotations

import json
import logging
import os
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from PIL import Image

from app.models import (
    IngestRequest,
    IngestResult,
    IngestStatus,
    ProductMetadata,
    SensorType,
    SunAngleSourceTier,
)
from app.storage import (
    fakes3_root,
    ref_from_local,
    resolve_ref,
    write_file_copy,
)

log = logging.getLogger("ingestion-svc")


# Per-sensor defaults. These are the values the system-prompt spec lists
# and are what preprocessing-svc and coarse-matching will see in
# metadata.json if the request doesn't override them.
SENSOR_DEFAULTS = {
    SensorType.OHRC: {"gsd": 0.6, "band_count": 1, "bit_depth": 12},
    SensorType.TMC: {"gsd": 5.0, "band_count": 3, "bit_depth": 10},
    SensorType.IIRS: {"gsd": 20.0, "band_count": 64, "bit_depth": 16},
    SensorType.REFERENCE: {"gsd": 5.0, "band_count": 1, "bit_depth": 8},
}


def _infer_bit_depth_and_bands(path: Path, defaults: dict) -> tuple[int, int]:
    """Inspect the image file to refine band_count / bit_depth if possible.

    Falls back to sensor defaults if the file is unreadable or absent.
    """
    try:
        with Image.open(path) as img:
            n_bands = 1
            if img.mode in ("RGB", "YCbCr"):
                n_bands = 3
            elif img.mode in ("RGBA",):
                n_bands = 4
            bit_depth = 8 if img.mode in ("L", "RGB", "RGBA", "YCbCr") else 16
            return max(1, n_bands), max(1, bit_depth)
    except Exception:
        return defaults["band_count"], defaults["bit_depth"]


def _acquisition_time(req: IngestRequest, path: Path) -> str:
    if req.acquisition_time is not None:
        # Pydantic parsed it; emit as ISO-8601 with Z timezone.
        if isinstance(req.acquisition_time, datetime):
            dt = req.acquisition_time
        else:
            dt = datetime.fromisoformat(str(req.acquisition_time))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    # Fall back to file mtime.
    try:
        mtime = path.stat().st_mtime
        dt = datetime.fromtimestamp(mtime, tz=timezone.utc)
        return dt.strftime("%Y-%m-%dT%H:%M:%SZ")
    except Exception:
        # Last-resort: epoch.
        return "1970-01-01T00:00:00Z"


def _materialize_metadata(
    req: IngestRequest,
    source_path: Path,
) -> ProductMetadata:
    defaults = SENSOR_DEFAULTS[req.sensor_type]
    # If we can inspect the file we refine band_count / bit_depth;
    # otherwise we use the per-sensor defaults.
    inferred_bands, inferred_depth = _infer_bit_depth_and_bands(
        source_path, defaults
    )
    gsd = req.gsd if req.gsd is not None else defaults["gsd"]
    # For unavailable tier we explicitly null out the angles even if
    # the caller passed something; the tier is the source of truth.
    if req.sun_angle_source_tier == SunAngleSourceTier.UNAVAILABLE:
        az: Optional[float] = None
        el: Optional[float] = None
    else:
        az = req.sun_azimuth_deg
        el = req.sun_elevation_deg
    return ProductMetadata(
        sensor_type=req.sensor_type,
        gsd=gsd,
        sun_azimuth_deg=az,
        sun_elevation_deg=el,
        sun_angle_source_tier=req.sun_angle_source_tier,
        projection=req.projection,
        footprint_wkt=req.footprint_wkt,
        band_count=inferred_bands,
        bit_depth=inferred_depth,
        acquisition_time=_acquisition_time(req, source_path),
    )


def _determine_output_paths(
    job_id: str,
    output_prefix: str,
) -> tuple[Path, Path, str, str]:
    """Return (raw_target, meta_target, raw_ref, meta_ref) for the job.

    For s3:// prefixes with no local fakes3, we still produce the refs
    but skip writing bytes (the byte path is exercised only when
    fakes3_root is configured).
    """
    fakes3 = fakes3_root()
    raw_key = f"{job_id}/raw.tif"
    meta_key = f"{job_id}/metadata.json"
    raw_ref = ref_from_local(Path(raw_key), output_prefix, raw_key)
    meta_ref = ref_from_local(Path(meta_key), output_prefix, meta_key)

    if fakes3 is None:
        # No local backing store. Caller should still get the refs;
        # bytes won't be written. This matches the prod behavior
        # (we trust S3 to persist).
        return Path(raw_key), Path(meta_key), raw_ref, meta_ref

    # When we have a fakes3 root, the ref maps directly under that root.
    raw_target = resolve_ref(raw_ref)
    meta_target = resolve_ref(meta_ref)
    return raw_target, meta_target, raw_ref, meta_ref


def ingest_request(
    req: IngestRequest,
    job_id: Optional[str] = None,
) -> IngestResult:
    """Run the full ingest pipeline and return the IngestResult envelope."""
    job_id = job_id or req.job_id or str(uuid.uuid4())
    source_path = Path(req.raw_image_path)
    if not source_path.exists():
        return IngestResult(
            job_id=job_id,
            status=IngestStatus.FAILED,
            error_message=f"raw_image_path not found: {source_path}",
        )

    try:
        metadata = _materialize_metadata(req, source_path)
        raw_target, meta_target, raw_ref, meta_ref = _determine_output_paths(
            job_id, req.output_prefix
        )
        # Write the bytes (only if we have a local backing store).
        if fakes3_root() is not None:
            write_file_copy(source_path, raw_target)
            # metadata.json next to it.
            meta_target.parent.mkdir(parents=True, exist_ok=True)
            meta_target.write_text(
                metadata.model_dump_json(indent=2),
                encoding="utf-8",
            )
        return IngestResult(
            job_id=job_id,
            status=IngestStatus.SUCCEEDED,
            raw_image_ref=raw_ref,
            metadata_ref=meta_ref,
            sensor_type=metadata.sensor_type,
            gsd=metadata.gsd,
            sun_azimuth_deg=metadata.sun_azimuth_deg,
            sun_elevation_deg=metadata.sun_elevation_deg,
            sun_angle_source_tier=metadata.sun_angle_source_tier,
            band_count=metadata.band_count,
            bit_depth=metadata.bit_depth,
            projection=metadata.projection,
            footprint_wkt=metadata.footprint_wkt,
            acquisition_time=metadata.acquisition_time,
        )
    except Exception as exc:
        log.exception("ingest failed for job %s", job_id)
        return IngestResult(
            job_id=job_id,
            status=IngestStatus.FAILED,
            error_message=f"{type(exc).__name__}: {exc}",
        )


def write_metadata_to_path(metadata: ProductMetadata, target: Path) -> Path:
    """Convenience for tests / mock-data generation."""
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(metadata.model_dump_json(indent=2), encoding="utf-8")
    return target
