"""Configuration and contract types for the preprocessing service.

These mirror the gRPC contract from the system specification but are exposed
as Pydantic models so the service can run over plain HTTP/JSON in addition
to (or instead of) gRPC. The wire format is identical, so the contract with
Member 3 (Coarse Matching) is preserved.
"""
from __future__ import annotations

import enum
from dataclasses import dataclass, field
from typing import Optional

from pydantic import BaseModel, Field, field_validator


class SensorType(str, enum.Enum):
    """Sensor kind of the input product."""

    OHRC = "OHRC"
    TMC = "TMC"
    IIRS = "IIRS"
    REFERENCE = "REFERENCE"


class SunAngleSourceTier(str, enum.Enum):
    """How confident we are in the reported sun angles.

    - LABEL: taken directly from a calibrated product label (highest confidence)
    - EPHEMERIS: derived from SPICE / on-board ephemeris (high confidence)
    - UNAVAILABLE: not in metadata; service must estimate from image
    """

    LABEL = "label"
    EPHEMERIS = "ephemeris"
    UNAVAILABLE = "unavailable"


class JobStatus(str, enum.Enum):
    PENDING = "PENDING"
    RUNNING = "RUNNING"
    SUCCEEDED = "SUCCEEDED"
    FAILED = "FAILED"


class IngestMetadata(BaseModel):
    """metadata.json as produced by ingestion-svc (Member 1).

    NOTE on reconciliation status (Member 2 -> Member 1):
      preprocessing-svc was built before ingestion-svc published its
      real contract. As of this writing, ingestion-svc's contracts
      directory is not yet present in the repo (see CONTRACT.md for
      the deferral flag). The fields below are the assumptions
      preprocessing-svc made from the system-prompt spec. When the
      real ingestion contract lands, every field here must be
      re-verified — particularly the `acquisition_time` field, which
      preprocessing-svc is forward-compatible with (accepted but
      not used downstream).
    """

    sensor_type: SensorType
    gsd: float = Field(..., gt=0, description="Ground sample distance in meters/pixel")
    sun_azimuth_deg: Optional[float] = Field(
        None,
        ge=0,
        le=360,
        description="Sun azimuth in degrees, 0=N, 90=E (image frame)",
    )
    sun_elevation_deg: Optional[float] = Field(
        None,
        ge=-90,
        le=90,
        description="Sun elevation above horizon in degrees",
    )
    sun_angle_source_tier: SunAngleSourceTier
    projection: str = Field(..., description="WKT or EPSG/proj string")
    footprint_wkt: Optional[str] = None
    band_count: int = Field(..., gt=0)
    bit_depth: int = Field(..., gt=0)
    # Forward-compat with ingestion-svc's expected schema. Optional,
    # currently unused by the pipeline. We accept (and ignore) it so
    # we don't drop metadata when Member 1 lands the real contract.
    acquisition_time: Optional[str] = Field(
        None,
        description=(
            "ISO-8601 acquisition timestamp. Accepted but not "
            "currently used by the preprocessing pipeline."
        ),
    )

    @field_validator("sun_azimuth_deg", "sun_elevation_deg", mode="before")
    @classmethod
    def _coerce_none(cls, v):
        return None if v == "" else v


class PyramidConfig(BaseModel):
    """Multi-scale pyramid parameters."""

    # Margin in octaves added beyond the GSD-ratio requirement so that the
    # coarsest level is comfortably coarser than the reference.
    margin_octaves: int = 2
    # Cap on level count so we don't blow up memory for huge GSD ratios.
    max_levels: int = 10
    # Reference GSD (m/px). If unknown, the service uses the upstream value
    # from the orchestrator. For a standalone run we default to 5.0 m/px
    # (LOLA-derived reference product).
    reference_gsd_m: float = 5.0


class InvariantConfig(BaseModel):
    """Invariant-channel parameters."""

    # Phase congruency: number of orientations in the Log-Gabor bank.
    n_orientations: int = 6
    # Number of scales for the Log-Gabor bank.
    n_scales: int = 4
    # Wavelength of the smallest scale filter, in pixels.
    min_wavelength: float = 3.0
    # Multiplicative factor between successive scales.
    scaling_factor: float = 2.0
    # Gaussian sigma for the quadrature-pair weighting on phase congruency.
    sigma_on_f: float = 0.55
    # Cut-off below which phase congruency is zeroed (noise suppression).
    noise_threshold: float = 1.5
    # Whether to include the dense gradient-orientation field.
    include_gradient_orientation: bool = True


class DenoiseConfig(BaseModel):
    """Denoise + destripe parameters."""

    # Bilateral filter parameters; tuned for the moon (smooth mare, sharp rims).
    bilateral_d: int = 5
    bilateral_sigma_color: float = 0.08
    bilateral_sigma_space: float = 3.0
    # Whether to apply destriping (pushbroom artifacts only). Set to false
    # for frame sensors (e.g. OHRC).
    apply_destripe: bool = True
    # How aggressively to match column moments. 1.0 = full, 0.0 = skip.
    destripe_strength: float = 0.6


class PreprocessConfig(BaseModel):
    """Top-level service configuration.

    Typically loaded from a JSON referenced by config_ref. The defaults are
    sensible for the lunar use case and the regression test fixture.
    """

    pyramid: PyramidConfig = Field(default_factory=PyramidConfig)
    invariant: InvariantConfig = Field(default_factory=InvariantConfig)
    denoise: DenoiseConfig = Field(default_factory=DenoiseConfig)
    # Whether to attempt orthorectification if a DEM reference is supplied.
    enable_orthorectify: bool = True
    # Working dtype for normalized images.
    working_dtype: str = "float32"
    # Percentile bounds for the radiometric stretch.
    stretch_low_pct: float = 2.0
    stretch_high_pct: float = 98.0


class PreprocessRequest(BaseModel):
    """PreprocessRequest message (mirrors the gRPC contract)."""

    job_id: str
    raw_image_ref: str
    metadata_ref: str
    dem_ref: str = ""
    config_ref: str = ""


class JobHandle(BaseModel):
    job_id: str


class PreprocessResult(BaseModel):
    """PreprocessResult message (mirrors the gRPC contract)."""

    job_id: str
    status: JobStatus
    pyramid_ref: str = ""
    invariant_channels_ref: str = ""
    scale_factors: list[float] = Field(default_factory=list)
    error_message: str = ""


# Internal job-record type for the in-memory job store.
@dataclass
class JobRecord:
    job_id: str
    status: JobStatus = JobStatus.PENDING
    result: Optional[PreprocessResult] = None
    error: Optional[str] = None
    extras: dict = field(default_factory=dict)
