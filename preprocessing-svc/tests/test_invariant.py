"""Unit tests for the invariant channel module directly."""
from __future__ import annotations

import numpy as np
import pytest

from preprocessing_svc.invariant import (
    compute_invariant_channels,
    gradient_orientation_field,
    phase_congruency,
    sdn_relief,
)


def test_phase_congruency_shape_and_range():
    rng = np.random.default_rng(0)
    img = rng.uniform(0, 1, (64, 64)).astype(np.float32)
    pc = phase_congruency(img, n_scales=2, n_orientations=3, min_wavelength=4.0)
    assert pc.shape == img.shape
    assert pc.dtype == np.float32
    assert pc.min() >= 0.0 - 1e-6
    assert pc.max() <= 1.0 + 1e-6


def test_phase_congruency_rejects_3d():
    with pytest.raises(ValueError):
        phase_congruency(np.zeros((3, 16, 16), dtype=np.float32))


def test_sdn_relief_shape():
    img = np.random.default_rng(0).uniform(0, 1, (32, 32)).astype(np.float32)
    out = sdn_relief(img, sun_azimuth_deg=90.0)
    assert out.shape == img.shape
    assert out.min() >= 0.0 - 1e-6
    assert out.max() <= 1.0 + 1e-6


def test_sdn_relief_changes_with_azimuth():
    """A non-trivial terrain should give different SDN-Relief maps for
    very different sun azimuths."""
    rng = np.random.default_rng(0)
    yy, xx = np.meshgrid(np.arange(64), np.arange(64), indexing="ij")
    img = (0.5 + 0.4 * np.sin(xx / 8.0) * np.cos(yy / 12.0)).astype(np.float32)
    a = sdn_relief(img, sun_azimuth_deg=0.0)
    b = sdn_relief(img, sun_azimuth_deg=90.0)
    # The two maps should differ — they encode different Sun-facing slopes.
    assert not np.allclose(a, b)


def test_sdn_relief_rejects_3d():
    with pytest.raises(ValueError):
        sdn_relief(np.zeros((3, 16, 16), dtype=np.float32), 0.0)


def test_gradient_orientation_field_shape():
    img = np.random.default_rng(0).uniform(0, 1, (32, 32)).astype(np.float32)
    o, c = gradient_orientation_field(img)
    assert o.shape == img.shape
    assert c.shape == img.shape
    assert o.min() >= 0.0
    assert o.max() <= np.pi + 1e-6
    assert c.min() >= 0.0 - 1e-6
    assert c.max() <= 1.0 + 1e-6


def test_compute_invariant_channels_returns_metadata():
    img = np.random.default_rng(0).uniform(0, 1, (32, 32)).astype(np.float32)
    out = compute_invariant_channels(
        img,
        sun_azimuth_deg=90.0,
        sun_angle_source_tier="label",
        n_scales=2,
        n_orientations=3,
        min_wavelength=4.0,
    )
    assert "phase_congruency" in out
    assert "sdn_relief" in out
    assert "sdn_relief_provenance" in out
    assert "gradient_orientation" in out
    assert "gradient_coherence" in out


def test_compute_invariant_channels_3d_input():
    img = np.random.default_rng(0).uniform(0, 1, (3, 32, 32)).astype(np.float32)
    out = compute_invariant_channels(
        img,
        sun_azimuth_deg=45.0,
        sun_angle_source_tier="ephemeris",
        n_scales=2,
        n_orientations=3,
        min_wavelength=4.0,
    )
    assert "phase_congruency" in out
    # Should still produce the same shapes as the spatial dims.
    assert out["phase_congruency"].shape == (32, 32)


def test_compute_invariant_channels_no_gradient_orientation():
    img = np.random.default_rng(0).uniform(0, 1, (32, 32)).astype(np.float32)
    out = compute_invariant_channels(
        img,
        sun_azimuth_deg=0.0,
        sun_angle_source_tier="label",
        n_scales=2,
        n_orientations=3,
        min_wavelength=4.0,
        include_gradient_orientation=False,
    )
    assert "gradient_orientation" not in out
