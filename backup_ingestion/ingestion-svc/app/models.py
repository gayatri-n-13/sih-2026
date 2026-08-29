"""Pydantic models — mirror of contracts/ingestion.openapi.yaml.

Keep these in sync with the OpenAPI spec. The contract test
(tests/test_contract.py) round-trips a sample through both to catch drift.
"""
from __future__ import annotations

from enum import Enum
from typing import Literal

from pydantic import BaseModel, Field, field_validator


class SensorType(str, Enum):
    OHRC = "OHRC"
    TMC = "TMC"
    IIRS = "IIRS"
    REFERENCE = "REFERENCE"


class JobStatus(str, Enum):
    PENDING = "PENDING"
    RUNNING = "RUNNING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"


class SunAngleTier(str, Enum):
    LABEL = "label"
    EPHEMERIS = "ephemeris"
    UNAVAILABLE = "unavailable"


class IngestRequest(BaseModel):
    job_id: str = Field(..., min_length=1, max_length=128)
    source_file_uri: str = Field(..., min_length=1)
    sensor_type: SensorType
    config_ref: str | None = Field(
        default=None,
        description="s3:// URI to per-job config; falls back to default if None.",
    )


class JobHandle(BaseModel):
    job_id: str
    status: JobStatus


class IngestResult(BaseModel):
    job_id: str
    status: JobStatus
    raw_image_ref: str | None = None
    metadata_ref: str | None = None
    error_message: str | None = None

    @field_validator("error_message")
    @classmethod
    def _error_only_on_failed(cls, v: str | None, info) -> str | None:
        status = info.data.get("status")
        if status == JobStatus.FAILED and not v:
            raise ValueError("error_message must be populated when status=FAILED")
        if status != JobStatus.FAILED and v:
            raise ValueError("error_message must be null unless status=FAILED")
        return v


class MetadataSidecar(BaseModel):
    """The metadata.json schema — mirrors contracts/metadata.schema.json.

    Field shape reconciled against preprocessing-svc's
    preprocessing_svc/config.py::IngestMetadata. gsd MUST be a positive
    number (Preprocessing's Pydantic model uses `gt=0`); when the source
    label carries no GSD, parser.py substitutes a per-sensor default and
    logs a warning. Sun-angle bounds match Preprocessing's validators.
    """

    model_config = {"extra": "allow"}  # downstream may add fields later

    sensor_type: SensorType
    gsd: float = Field(..., gt=0)
    acquisition_time: str | None = None
    sun_azimuth_deg: float | None = Field(default=None, ge=0, le=360)
    sun_elevation_deg: float | None = Field(default=None, ge=-90, le=90)
    sun_angle_source_tier: SunAngleTier
    projection: str
    footprint_wkt: str | None = None
    band_count: int = Field(..., ge=1)
    bit_depth: int = Field(..., ge=1)


class HealthResponse(BaseModel):
    status: Literal["ok", "degraded"]
    version: str


class ErrorResponse(BaseModel):
    code: str
    message: str
