"""Unit tests for the multi-scale pyramid."""
from __future__ import annotations

import math

import numpy as np
import pytest

from preprocessing_svc.pyramid import compute_scale_factors, gaussian_pyramid


def test_scale_factors_pow2_and_count():
    s = compute_scale_factors(source_gsd=5.0, reference_gsd=5.0, margin_octaves=2, max_levels=10)
    # Same GSD: need 0 bridge halvings + 2 margin octaves = 3 levels.
    assert len(s) == 3
    for v in s:
        # Each scale factor should be 1 / 2^k for some integer k.
        k = -math.log2(v)
        assert abs(k - round(k)) < 1e-6


def test_scale_factors_picks_enough_levels():
    s = compute_scale_factors(source_gsd=20.0, reference_gsd=5.0, margin_octaves=2, max_levels=10)
    # 20/5 = 4 -> log2 = 2; ceil = 2; + 1 (finest) + 2 margin = 5.
    assert len(s) == 5


def test_scale_factors_caps_at_max_levels():
    s = compute_scale_factors(source_gsd=200.0, reference_gsd=1.0, margin_octaves=5, max_levels=6)
    assert len(s) == 6


def test_scale_factors_rejects_bad_gsd():
    with pytest.raises(ValueError):
        compute_scale_factors(source_gsd=0.0, reference_gsd=1.0)
    with pytest.raises(ValueError):
        compute_scale_factors(source_gsd=1.0, reference_gsd=-2.0)


def test_gaussian_pyramid_shape_progression():
    img = np.random.default_rng(0).uniform(0, 1, (64, 64)).astype(np.float32)
    factors = [1.0, 0.5, 0.25, 0.125]
    levels = gaussian_pyramid(img, factors)
    assert len(levels) == 4
    assert levels[0].shape == (64, 64)
    assert levels[1].shape == (32, 32)
    assert levels[2].shape == (16, 16)
    assert levels[3].shape == (8, 8)


def test_gaussian_pyramid_3d():
    img = np.random.default_rng(0).uniform(0, 1, (3, 32, 32)).astype(np.float32)
    factors = [1.0, 0.5, 0.25]
    levels = gaussian_pyramid(img, factors)
    assert levels[0].shape == (3, 32, 32)
    assert levels[1].shape == (3, 16, 16)
    assert levels[2].shape == (3, 8, 8)


def test_gaussian_pyramid_rejects_bad_ndim():
    with pytest.raises(ValueError):
        gaussian_pyramid(np.zeros((2, 2, 2, 2), dtype=np.float32), [1.0, 0.5])
