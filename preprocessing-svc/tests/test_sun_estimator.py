"""Unit tests for the image-based sun estimator fallback."""
from __future__ import annotations

import numpy as np
import pytest

from preprocessing_svc.sun_estimator import estimate_sun_azimuth


def test_estimator_returns_dict_keys():
    img = np.random.default_rng(0).uniform(0, 1, (32, 32)).astype(np.float32)
    est = estimate_sun_azimuth(img)
    assert "sun_azimuth_deg" in est
    assert "confidence" in est
    assert "gradient_orientation_deg" in est


def test_estimator_handles_3d_input():
    img = np.random.default_rng(0).uniform(0, 1, (3, 32, 32)).astype(np.float32)
    est = estimate_sun_azimuth(img)
    assert "sun_azimuth_deg" in est


def test_estimator_rejects_bad_ndim():
    with pytest.raises(ValueError):
        estimate_sun_azimuth(np.zeros((2, 2, 2, 2), dtype=np.float32))


def test_estimator_returns_zero_on_degenerate_input():
    """A uniform image has no usable gradient distribution; we should
    return a zero azimuth with zero confidence rather than crashing."""
    img = np.full((32, 32), 0.5, dtype=np.float32)
    est = estimate_sun_azimuth(img)
    assert est["sun_azimuth_deg"] == 0.0
    assert est["confidence"] == 0.0
