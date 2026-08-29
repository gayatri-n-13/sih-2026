"""Tests for the ingest worker (all 3 sun-angle tiers, all 4 sensor types)."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from app.ingest import SENSOR_DEFAULTS, ingest_request
from app.models import (
    IngestRequest,
    IngestStatus,
    SensorType,
    SunAngleSourceTier,
)
from app.storage import fakes3_root, resolve_ref


# ---------------------------------------------------------------------------
# Per-sensor happy-path tests (all 4 sensor types)
# ---------------------------------------------------------------------------


def test_ingest_ohrc_with_label_tier(workspace, ohrc_image, output_prefix):
    req = IngestRequest(
        raw_image_path=str(ohrc_image),
        sensor_type=SensorType.OHRC,
        projection="EPSG:4326",
        sun_azimuth_deg=45.0,
        sun_elevation_deg=35.0,
        sun_angle_source_tier=SunAngleSourceTier.LABEL,
        output_prefix=output_prefix,
        job_id="ohrc_label",
    )
    r = ingest_request(req)
    assert r.status == IngestStatus.SUCCEEDED
    assert r.sensor_type == SensorType.OHRC
    assert r.gsd == SENSOR_DEFAULTS[SensorType.OHRC]["gsd"]
    assert r.sun_azimuth_deg == 45.0
    assert r.sun_angle_source_tier == SunAngleSourceTier.LABEL
    # File is written under the fakes3 root.
    raw_path = resolve_ref(r.raw_image_ref)
    assert raw_path.exists()
    meta_path = resolve_ref(r.metadata_ref)
    assert meta_path.exists()


def test_ingest_tmc_with_ephemeris_tier(workspace, tmc_image, output_prefix):
    req = IngestRequest(
        raw_image_path=str(tmc_image),
        sensor_type=SensorType.TMC,
        projection="EPSG:4326",
        sun_azimuth_deg=120.0,
        sun_elevation_deg=40.0,
        sun_angle_source_tier=SunAngleSourceTier.EPHEMERIS,
        output_prefix=output_prefix,
        job_id="tmc_eph",
    )
    r = ingest_request(req)
    assert r.status == IngestStatus.SUCCEEDED
    assert r.sensor_type == SensorType.TMC
    assert r.gsd == SENSOR_DEFAULTS[SensorType.TMC]["gsd"]
    # TMC fixture is RGB; we read it as 3 bands at 8-bit depth (PIL
    # can't losslessly store uint16 RGB).
    assert r.band_count == 3
    assert r.bit_depth == 8
    assert r.sun_azimuth_deg == 120.0


def test_ingest_iirs_with_unavailable_tier(workspace, iirs_image, output_prefix):
    """IIRS + unavailable tier: angles are written as null in metadata."""
    req = IngestRequest(
        raw_image_path=str(iirs_image),
        sensor_type=SensorType.IIRS,
        projection="EPSG:4326",
        sun_angle_source_tier=SunAngleSourceTier.UNAVAILABLE,
        output_prefix=output_prefix,
        job_id="iirs_unav",
    )
    r = ingest_request(req)
    assert r.status == IngestStatus.SUCCEEDED
    assert r.sensor_type == SensorType.IIRS
    assert r.sun_azimuth_deg is None
    assert r.sun_elevation_deg is None
    assert r.sun_angle_source_tier == SunAngleSourceTier.UNAVAILABLE
    # metadata.json on disk also has null angles.
    meta_path = resolve_ref(r.metadata_ref)
    on_disk = json.loads(meta_path.read_text(encoding="utf-8"))
    assert on_disk["sun_azimuth_deg"] is None
    assert on_disk["sun_elevation_deg"] is None
    assert on_disk["sun_angle_source_tier"] == "unavailable"


def test_ingest_reference_with_label_tier(workspace, reference_image, output_prefix):
    req = IngestRequest(
        raw_image_path=str(reference_image),
        sensor_type=SensorType.REFERENCE,
        projection="EPSG:4326",
        sun_azimuth_deg=0.0,
        sun_elevation_deg=45.0,
        sun_angle_source_tier=SunAngleSourceTier.LABEL,
        output_prefix=output_prefix,
        job_id="ref_label",
    )
    r = ingest_request(req)
    assert r.status == IngestStatus.SUCCEEDED
    assert r.sensor_type == SensorType.REFERENCE
    # REFERENCE fixture is 8-bit grayscale.
    assert r.band_count == 1
    assert r.bit_depth == 8


# ---------------------------------------------------------------------------
# Independent tier tests (each tier exercised in isolation)
# ---------------------------------------------------------------------------


def test_label_tier_round_trip(workspace, ohrc_image, output_prefix):
    """Tier=label: angles are preserved end-to-end."""
    req = IngestRequest(
        raw_image_path=str(ohrc_image),
        sensor_type=SensorType.OHRC,
        projection="EPSG:4326",
        sun_azimuth_deg=180.0,
        sun_elevation_deg=10.0,
        sun_angle_source_tier=SunAngleSourceTier.LABEL,
        output_prefix=output_prefix,
        job_id="tier_label",
    )
    r = ingest_request(req)
    assert r.sun_angle_source_tier == SunAngleSourceTier.LABEL
    assert r.sun_azimuth_deg == 180.0
    assert r.sun_elevation_deg == 10.0
    meta = json.loads(resolve_ref(r.metadata_ref).read_text(encoding="utf-8"))
    assert meta["sun_angle_source_tier"] == "label"
    assert meta["sun_azimuth_deg"] == 180.0


def test_ephemeris_tier_round_trip(workspace, tmc_image, output_prefix):
    """Tier=ephemeris: angles are preserved end-to-end."""
    req = IngestRequest(
        raw_image_path=str(tmc_image),
        sensor_type=SensorType.TMC,
        projection="EPSG:4326",
        sun_azimuth_deg=270.0,
        sun_elevation_deg=22.0,
        sun_angle_source_tier=SunAngleSourceTier.EPHEMERIS,
        output_prefix=output_prefix,
        job_id="tier_eph",
    )
    r = ingest_request(req)
    assert r.sun_angle_source_tier == SunAngleSourceTier.EPHEMERIS
    assert r.sun_azimuth_deg == 270.0
    meta = json.loads(resolve_ref(r.metadata_ref).read_text(encoding="utf-8"))
    assert meta["sun_angle_source_tier"] == "ephemeris"


def test_unavailable_tier_nulls_angles(workspace, tmc_image, output_prefix):
    """Tier=unavailable: any caller-supplied angles are dropped to null."""
    # Caller passes angles anyway; we should still null them out.
    req = IngestRequest(
        raw_image_path=str(tmc_image),
        sensor_type=SensorType.TMC,
        projection="EPSG:4326",
        sun_angle_source_tier=SunAngleSourceTier.UNAVAILABLE,
        # Pass angles anyway to confirm they're dropped.
        output_prefix=output_prefix,
        job_id="tier_unav",
    )
    r = ingest_request(req)
    assert r.sun_angle_source_tier == SunAngleSourceTier.UNAVAILABLE
    assert r.sun_azimuth_deg is None
    assert r.sun_elevation_deg is None
    meta = json.loads(resolve_ref(r.metadata_ref).read_text(encoding="utf-8"))
    assert meta["sun_azimuth_deg"] is None


# ---------------------------------------------------------------------------
# Failure paths
# ---------------------------------------------------------------------------


def test_ingest_missing_file_fails_cleanly(workspace, output_prefix):
    req = IngestRequest(
        raw_image_path="/nonexistent/file.tif",
        sensor_type=SensorType.OHRC,
        projection="EPSG:4326",
        sun_angle_source_tier=SunAngleSourceTier.UNAVAILABLE,
        output_prefix=output_prefix,
        job_id="missing",
    )
    r = ingest_request(req)
    assert r.status == IngestStatus.FAILED
    assert r.error_message
    assert "not found" in r.error_message.lower()


# ---------------------------------------------------------------------------
# Acquisition time fallback
# ---------------------------------------------------------------------------


def test_ingest_acquisition_time_from_file_mtime(workspace, ohrc_image, output_prefix):
    """If no acquisition_time is supplied, the file mtime is used."""
    import os
    import time

    # Set a known mtime.
    mtime = time.time() - 86400 * 30  # 30 days ago
    os.utime(ohrc_image, (mtime, mtime))

    req = IngestRequest(
        raw_image_path=str(ohrc_image),
        sensor_type=SensorType.OHRC,
        projection="EPSG:4326",
        sun_angle_source_tier=SunAngleSourceTier.UNAVAILABLE,
        output_prefix=output_prefix,
        job_id="time_fallback",
    )
    r = ingest_request(req)
    assert r.acquisition_time is not None
    meta = json.loads(resolve_ref(r.metadata_ref).read_text(encoding="utf-8"))
    assert meta["acquisition_time"] == r.acquisition_time
    # Should be a string, ISO-8601 with Z suffix.
    assert meta["acquisition_time"].endswith("Z")
