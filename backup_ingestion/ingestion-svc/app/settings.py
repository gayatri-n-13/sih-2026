"""Runtime settings for ingestion-svc.

Loaded from env vars (12-factor). Defaults match docker-compose.yml.
"""
from __future__ import annotations

from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Process-wide settings."""

    model_config = SettingsConfigDict(env_prefix="INGESTION_", case_sensitive=False)

    # --- HTTP server ---
    host: str = "0.0.0.0"
    port: int = 8000
    log_level: str = "info"

    # --- Object storage ---
    s3_endpoint_url: str | None = Field(
        default="http://minio:9000",
        description="Override for local MinIO; None means use real AWS.",
    )
    s3_region: str = "us-east-1"
    s3_access_key: str = "minioadmin"
    s3_secret_key: str = "minioadmin"
    output_bucket: str = "ingestion-output"
    output_prefix_template: str = "{job_id}/ingestion"  # s3://{bucket}/{prefix}/

    # --- Default config (per-job field-mapping + validation rules) ---
    default_config_path: Path = Path(__file__).parent / "config" / "default.yaml"

    # --- Ephemeris (tier 2) ---
    spice_kernels_dir: Path | None = None

    # --- Job table ---
    # First pass: in-process dict. Swap with Redis/Postgres later.
    job_table_backend: str = "memory"


def get_settings() -> Settings:
    """Cached settings accessor for FastAPI dependency injection."""
    return Settings()
