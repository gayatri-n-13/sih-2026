"""Test configuration: shared fixtures and helpers."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from preprocessing_svc.config import (  # noqa: E402  (import after sys.path)
    IngestMetadata,
    PreprocessConfig,
    SensorType,
    SunAngleSourceTier,
)


@pytest.fixture
def small_terrain() -> np.ndarray:
    """A small synthetic terrain heightmap (Y, X) float32."""
    rng = np.random.default_rng(0)
    y, x = 128, 128
    yy, xx = np.meshgrid(np.arange(y), np.arange(x), indexing="ij")
    h = np.zeros((y, x), dtype=np.float32)
    for _ in range(8):
        cy = rng.integers(10, y - 10)
        cx = rng.integers(10, x - 10)
        amp = rng.uniform(-1.0, 1.0)
        sigma = rng.uniform(6, 18)
        h += amp * np.exp(-((yy - cy) ** 2 + (xx - cx) ** 2) / (2 * sigma * sigma))
    h += 0.2 * np.sin(yy / 12.0) * np.cos(xx / 16.0)
    h += 0.05 * rng.standard_normal((y, x)).astype(np.float32)
    return h


def hillshade(height: np.ndarray, sun_az_deg: float, sun_el_deg: float) -> np.ndarray:
    """Simple Lambertian hillshade, matching mock_ingestion.generate_mock."""
    import math

    gy = np.zeros_like(height)
    gx = np.zeros_like(height)
    gy[1:-1, :] = height[2:, :] - height[:-2, :]
    gx[:, 1:-1] = height[:, 2:] - height[:, :-2]
    gy *= 0.5
    gx *= 0.5
    az = math.radians(sun_az_deg)
    el = math.radians(max(sun_el_deg, 0.1))
    slope = np.arctan(np.hypot(gx, gy))
    aspect = np.arctan2(-gx, gy)
    shaded = (
        np.cos(el) * np.cos(slope)
        + np.sin(el) * np.sin(slope) * np.cos(az - aspect)
    )
    return np.clip(shaded, 0.0, 1.0).astype(np.float32)


@pytest.fixture
def ohrc_metadata() -> IngestMetadata:
    return IngestMetadata(
        sensor_type=SensorType.OHRC,
        gsd=0.6,
        sun_azimuth_deg=45.0,
        sun_elevation_deg=35.0,
        sun_angle_source_tier=SunAngleSourceTier.LABEL,
        projection="EPSG:4326",
        footprint_wkt=None,
        band_count=1,
        bit_depth=12,
    )


@pytest.fixture
def iirs_metadata() -> IngestMetadata:
    return IngestMetadata(
        sensor_type=SensorType.IIRS,
        gsd=20.0,
        sun_azimuth_deg=60.0,
        sun_elevation_deg=45.0,
        sun_angle_source_tier=SunAngleSourceTier.EPHEMERIS,
        projection="EPSG:4326",
        footprint_wkt=None,
        band_count=64,
        bit_depth=16,
    )


@pytest.fixture
def default_config() -> PreprocessConfig:
    return PreprocessConfig()


@pytest.fixture
def fast_inv_config() -> PreprocessConfig:
    """A small / fast invariant config for tests."""
    return PreprocessConfig(
        invariant=PreprocessConfig.model_fields["invariant"].default_factory().__class__(
            n_scales=3, n_orientations=4, min_wavelength=4.0
        )
    )
