"""Pydantic models for the ingestion service.

These are the SOURCE OF TRUTH for what the service produces. The JSON
Schema in ``contracts/metadata.schema.json`` and the OpenAPI document
in ``contracts/ingestion.openapi.yaml`` are *derived* from these
models. The ``test_contract_matches_models`` test enforces that they
stay in sync — if you change a model, regenerate the contracts.

Contract surface (consumed by Member 2's preprocessing-svc):

    IngestResult envelope:
        raw_image_ref:  string  (s3://… or file://…)
        metadata_ref:   string  (s3://… or file://…)
        job_id:         string
        status:         "PENDING" | "RUNNING" | "SUCCEEDED" | "FAILED"
        error_message:  string

    metadata.json:
        sensor_type:            OHRC | TMC | IIRS | REFERENCE
        gsd:                    float (m/px), > 0
        sun_azimuth_deg:        float | null,   [0, 360)
        sun_elevation_deg:      float | null,   [-90, 90]
        sun_angle_source_tier:  label | ephemeris | unavailable
        projection:             string (WKT or EPSG/proj)
        footprint_wkt:          string | null
        band_count:             int, ≥ 1
        bit_depth:              int, ≥ 1
        acquisition_time:       ISO-8601 string | null
"""
from __future__ import annotations

import enum
from datetime import datetime
from typing import Optional

from pydantic import BaseModel, Field, field_validator, model_validator


class SensorType(str, enum.Enum):
    OHRC = "OHRC"
    TMC = "TMC"
    IIRS = "IIRS"
    REFERENCE = "REFERENCE"


class SunAngleSourceTier(str, enum.Enum):
    LABEL = "label"
    EPHEMERIS = "ephemeris"
    UNAVAILABLE = "unavailable"


class IngestStatus(str, enum.Enum):
    PENDING = "PENDING"
    RUNNING = "RUNNING"
    SUCCEEDED = "SUCCEEDED"
    FAILED = "FAILED"


class IngestRequest(BaseModel):
    """Request to ingest a single raw image product.

    The service will:
      1. Open the file at ``raw_image_path`` (local path; in production
         this would be a presigned URL or staged upload).
      2. Read sensor metadata (from sidecar, embedded, or filename
         convention).
      3. Compute the Sun-angle tier:
           - ``label``      if a calibrated product label is attached
           - ``ephemeris``  if a SPICE-computed angle is available
           - ``unavailable`` if neither is present (we will pass the
             angle through as null; downstream handles estimation)
      4. Write the image to the object store at ``raw_image_ref`` and
         write ``metadata.json`` to the metadata reference.
      5. Return the IngestResult envelope.
    """

    job_id: Optional[str] = Field(
        None, description="Optional client-supplied ID. If empty, we assign one."
    )
    raw_image_path: str = Field(..., description="Local path to the raw image file")
    sensor_type: SensorType
    projection: str = Field(..., description="WKT or EPSG/proj string for the image frame")
    # The angle-source tier tells us how the sun angles were obtained.
    # We attach the actual angles when tier is label or ephemeris.
    sun_azimuth_deg: Optional[float] = Field(None, ge=0, lt=360)
    sun_elevation_deg: Optional[float] = Field(None, ge=-90, le=90)
    sun_angle_source_tier: SunAngleSourceTier
    # Optional metadata fields:
    footprint_wkt: Optional[str] = None
    gsd: Optional[float] = Field(
        None,
        gt=0,
        description="Override GSD (m/px). If absent, inferred from sensor defaults.",
    )
    acquisition_time: Optional[datetime] = Field(
        None,
        description="ISO-8601 acquisition timestamp. Defaults to file mtime if absent.",
    )
    output_prefix: str = Field(
        "s3://ingestion-bucket/",
        description=(
            "Object-store prefix for the output (raw_image_ref and "
            "metadata_ref will be built under this)."
        ),
    )

    @field_validator("sun_azimuth_deg", "sun_elevation_deg", "gsd", mode="before")
    @classmethod
    def _coerce_empty_str(cls, v):
        return None if v == "" else v

    @model_validator(mode="after")
    def _angles_consistent_with_tier(self):
        if self.sun_angle_source_tier in (SunAngleSourceTier.LABEL, SunAngleSourceTier.EPHEMERIS):
            if self.sun_azimuth_deg is None or self.sun_elevation_deg is None:
                raise ValueError(
                    "sun_azimuth_deg and sun_elevation_deg must both be set when "
                    "sun_angle_source_tier is 'label' or 'ephemeris'"
                )
        return self


class ProductMetadata(BaseModel):
    """metadata.json as written by the ingestion service.

    Every field here is what Member 2's preprocessing-svc reads. Names
    MUST match ``contracts/metadata.schema.json``.
    """

    sensor_type: SensorType
    gsd: float = Field(..., gt=0)
    sun_azimuth_deg: Optional[float] = Field(None, ge=0, lt=360)
    sun_elevation_deg: Optional[float] = Field(None, ge=-90, le=90)
    sun_angle_source_tier: SunAngleSourceTier
    projection: str
    footprint_wkt: Optional[str] = None
    band_count: int = Field(..., gt=0)
    bit_depth: int = Field(..., gt=0)
    acquisition_time: Optional[str] = Field(
        None,
        description="ISO-8601 acquisition timestamp (always string in JSON)",
    )


class IngestResult(BaseModel):
    job_id: str
    status: IngestStatus
    raw_image_ref: str = ""
    metadata_ref: str = ""
    sensor_type: Optional[SensorType] = None
    gsd: Optional[float] = None
    sun_azimuth_deg: Optional[float] = None
    sun_elevation_deg: Optional[float] = None
    sun_angle_source_tier: Optional[SunAngleSourceTier] = None
    band_count: Optional[int] = None
    bit_depth: Optional[int] = None
    projection: Optional[str] = None
    footprint_wkt: Optional[str] = None
    acquisition_time: Optional[str] = None
    error_message: str = ""
