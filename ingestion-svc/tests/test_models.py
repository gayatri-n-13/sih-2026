"""Tests for the Pydantic models."""
from __future__ import annotations

import pytest
from pydantic import ValidationError

from app.models import (
    IngestRequest,
    IngestResult,
    ProductMetadata,
    SensorType,
    SunAngleSourceTier,
)


def test_sensor_type_enum_values():
    assert SensorType.OHRC.value == "OHRC"
    assert SensorType.TMC.value == "TMC"
    assert SensorType.IIRS.value == "IIRS"
    assert SensorType.REFERENCE.value == "REFERENCE"


def test_sun_angle_tier_enum_values():
    assert SunAngleSourceTier.LABEL.value == "label"
    assert SunAngleSourceTier.EPHEMERIS.value == "ephemeris"
    assert SunAngleSourceTier.UNAVAILABLE.value == "unavailable"


def test_ingest_request_label_requires_angles():
    with pytest.raises(ValidationError):
        IngestRequest(
            raw_image_path="/tmp/x.tif",
            sensor_type=SensorType.OHRC,
            projection="EPSG:4326",
            sun_angle_source_tier=SunAngleSourceTier.LABEL,
            # angles missing
        )


def test_ingest_request_ephemeris_requires_angles():
    with pytest.raises(ValidationError):
        IngestRequest(
            raw_image_path="/tmp/x.tif",
            sensor_type=SensorType.TMC,
            projection="EPSG:4326",
            sun_azimuth_deg=90.0,
            # sun_elevation_deg missing
            sun_angle_source_tier=SunAngleSourceTier.EPHEMERIS,
        )


def test_ingest_request_label_with_angles_ok():
    req = IngestRequest(
        raw_image_path="/tmp/x.tif",
        sensor_type=SensorType.OHRC,
        projection="EPSG:4326",
        sun_azimuth_deg=45.0,
        sun_elevation_deg=30.0,
        sun_angle_source_tier=SunAngleSourceTier.LABEL,
    )
    assert req.sun_azimuth_deg == 45.0


def test_ingest_request_unavailable_with_null_angles_ok():
    req = IngestRequest(
        raw_image_path="/tmp/x.tif",
        sensor_type=SensorType.OHRC,
        projection="EPSG:4326",
        sun_angle_source_tier=SunAngleSourceTier.UNAVAILABLE,
    )
    assert req.sun_azimuth_deg is None
    assert req.sun_elevation_deg is None


def test_ingest_request_string_to_none_coercion():
    """Empty string (e.g. from CLI) is coerced to None for nullable fields."""
    req = IngestRequest(
        raw_image_path="/tmp/x.tif",
        sensor_type=SensorType.OHRC,
        projection="EPSG:4326",
        sun_azimuth_deg="",  # empty string -> None
        sun_elevation_deg="",
        sun_angle_source_tier=SunAngleSourceTier.UNAVAILABLE,
    )
    assert req.sun_azimuth_deg is None


def test_ingest_request_gsd_must_be_positive():
    with pytest.raises(ValidationError):
        IngestRequest(
            raw_image_path="/tmp/x.tif",
            sensor_type=SensorType.OHRC,
            projection="EPSG:4326",
            sun_angle_source_tier=SunAngleSourceTier.UNAVAILABLE,
            gsd=0.0,
        )


def test_ingest_request_azimuth_range():
    """Azimuth must be in [0, 360). 360 itself is rejected (exclusive)."""
    with pytest.raises(ValidationError):
        IngestRequest(
            raw_image_path="/tmp/x.tif",
            sensor_type=SensorType.OHRC,
            projection="EPSG:4326",
            sun_azimuth_deg=360.0,
            sun_elevation_deg=0.0,
            sun_angle_source_tier=SunAngleSourceTier.LABEL,
        )


def test_product_metadata_serialization_roundtrip():
    m = ProductMetadata(
        sensor_type=SensorType.TMC,
        gsd=5.0,
        sun_azimuth_deg=120.0,
        sun_elevation_deg=45.0,
        sun_angle_source_tier=SunAngleSourceTier.EPHEMERIS,
        projection="EPSG:4326",
        footprint_wkt=None,
        band_count=3,
        bit_depth=10,
        acquisition_time="2026-08-28T12:00:00Z",
    )
    j = m.model_dump_json()
    # Round-trip via JSON.
    m2 = ProductMetadata.model_validate_json(j)
    assert m2 == m


def test_ingest_result_defaults():
    r = IngestResult(job_id="abc", status="PENDING")
    assert r.raw_image_ref == ""
    assert r.metadata_ref == ""
    assert r.error_message == ""
