"""Contract test: validate the metadata.json shape we write against the
JSON Schema published in contracts/metadata.schema.json.

This is the same schema Preprocessing will validate against — if this
test passes, our writer and their reader agree on the shape.
"""
from __future__ import annotations

import json

import pytest
from jsonschema import Draft202012Validator

from app.models import (
    IngestResult,
    JobStatus,
    MetadataSidecar,
    SensorType,
    SunAngleTier,
)


def _validator(schema_path):
    return Draft202012Validator(json.loads(schema_path.read_text()))


def test_metadata_sidecar_validates_against_published_schema(contract_schema_path):
    """A representative valid sidecar must validate against the published schema."""
    sidecar = MetadataSidecar(
        sensor_type=SensorType.OHRC,
        gsd=0.32,
        acquisition_time="2021-08-15T04:32:11Z",
        sun_azimuth_deg=123.4,
        sun_elevation_deg=42.1,
        sun_angle_source_tier=SunAngleTier.LABEL,
        projection="EPSG:4326",
        footprint_wkt=None,
        band_count=1,
        bit_depth=16,
    )
    payload = sidecar.model_dump(mode="json")
    errors = list(_validator(contract_schema_path).iter_errors(payload))
    assert errors == [], f"schema violations: {[e.message for e in errors]}"


def test_metadata_sidecar_with_unavailable_sun_angles_validates(contract_schema_path):
    """Tier 3 ('unavailable') path: sun angles null, tier=unavailable. Must still validate."""
    sidecar = MetadataSidecar(
        sensor_type=SensorType.REFERENCE,
        gsd=5.0,  # per-sensor fallback when label has none; Preprocessing requires positive
        acquisition_time=None,
        sun_azimuth_deg=None,
        sun_elevation_deg=None,
        sun_angle_source_tier=SunAngleTier.UNAVAILABLE,
        projection="unknown",
        footprint_wkt=None,
        band_count=3,
        bit_depth=8,
    )
    payload = sidecar.model_dump(mode="json")
    errors = list(_validator(contract_schema_path).iter_errors(payload))
    assert errors == [], f"schema violations: {[e.message for e in errors]}"


def test_ingest_result_requires_error_on_failed():
    """A FAILED result without an error_message is invalid (validation model catches this)."""
    with pytest.raises(Exception):
        IngestResult(job_id="x", status=JobStatus.FAILED, error_message=None)


def test_ingest_result_completed_without_error():
    """A COMPLETED result must not carry an error_message."""
    with pytest.raises(Exception):
        IngestResult(
            job_id="x",
            status=JobStatus.COMPLETED,
            raw_image_ref="s3://b/raw.cog",
            metadata_ref="s3://b/m.json",
            error_message="this shouldn't be here",
        )
