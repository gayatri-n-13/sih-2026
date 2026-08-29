"""Per-job config loader.

A "config" is a YAML file containing:
  - field_mapping: logical_field -> label_tag path
  - validation_rules: structural checks
  - spice_kernels: list of kernel filenames (for tier-2 ephemeris)
  - expected_band_count: optional override

The loader is deliberately permissive: if the file at config_ref can't be
reached, we fall back to the baked-in default config and log a warning.
This keeps ingestion robust to transient config-bucket failures.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

log = logging.getLogger(__name__)


@dataclass
class FieldMapping:
    """Maps logical metadata fields to the actual label tag paths to read."""

    gsd: str | None = None
    acquisition_time: str | None = None
    sun_azimuth_deg: str | None = None
    sun_elevation_deg: str | None = None
    projection: str | None = None
    footprint_wkt: str | None = None
    bit_depth: str | None = None
    band_count: str | None = None


@dataclass
class ValidationRules:
    min_band_count: int = 1
    max_band_count: int = 16
    reject_all_nan: bool = True
    max_dim: int = 100_000  # hard cap to prevent OOM


@dataclass
class IngestConfig:
    field_mapping: FieldMapping = field(default_factory=FieldMapping)
    validation: ValidationRules = field(default_factory=ValidationRules)
    spice_kernels: list[str] = field(default_factory=list)
    expected_band_count: int | None = None
    output_bucket: str | None = None  # override per-job
    output_prefix_template: str | None = None

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "IngestConfig":
        fm = FieldMapping(**(d.get("field_mapping") or {}))
        vr_dict = d.get("validation") or {}
        vr = ValidationRules(**vr_dict)
        return cls(
            field_mapping=fm,
            validation=vr,
            spice_kernels=list(d.get("spice_kernels") or []),
            expected_band_count=d.get("expected_band_count"),
            output_bucket=d.get("output_bucket"),
            output_prefix_template=d.get("output_prefix_template"),
        )


def load_config(config_ref: str | None) -> IngestConfig:
    """Load a config from a URI, falling back to the default on failure.

    Accepted URIs:
      - None / ""         -> default config baked into the image
      - "file://..."      -> local file
      - any other string  -> treated as a key in the default config dir, or
                             fetched via boto3 if it looks like s3://
    """
    if config_ref is None or config_ref == "":
        return _load_default()

    if config_ref.startswith("file://"):
        path = Path(config_ref.removeprefix("file://"))
        if not path.is_file():
            log.warning("config %s not found, falling back to default", config_ref)
            return _load_default()
        return IngestConfig.from_dict(yaml.safe_load(path.read_text()))

    if config_ref.startswith("s3://"):
        # Lazy import so unit tests don't need boto3 wired up.
        try:
            import boto3  # type: ignore

            client = boto3.client("s3")
            bucket, _, key = config_ref.removeprefix("s3://").partition("/")
            obj = client.get_object(Bucket=bucket, Key=key)
            data = yaml.safe_load(obj["Body"].read())
            return IngestConfig.from_dict(data)
        except Exception as exc:
            log.warning(
                "could not fetch config %s (%s), falling back to default",
                config_ref,
                exc,
            )
            return _load_default()

    # Treat as a relative filename inside the default config dir.
    candidate = Path(__file__).parent / f"{config_ref}.yaml"
    if candidate.is_file():
        return IngestConfig.from_dict(yaml.safe_load(candidate.read_text()))

    log.warning("unknown config_ref %r, falling back to default", config_ref)
    return _load_default()


def _load_default() -> IngestConfig:
    default_path = Path(__file__).parent / "default.yaml"
    if not default_path.is_file():
        log.warning("no default config at %s, using empty config", default_path)
        return IngestConfig()
    return IngestConfig.from_dict(yaml.safe_load(default_path.read_text()))
