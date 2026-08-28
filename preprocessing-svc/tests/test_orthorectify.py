"""Unit tests for orthorectification.

Both code paths (DEM present and no-DEM no-op) must be unit-tested as
real code paths, per the system spec.
"""
from __future__ import annotations

import numpy as np
import pytest

from preprocessing_svc.orthorectify import (
    no_op_passthrough,
    orthorectify,
    orthorectify_with_dem,
)


def test_no_op_passthrough_returns_new_array():
    img = np.random.default_rng(0).uniform(0, 1, (16, 16)).astype(np.float32)
    out = no_op_passthrough(img)
    assert out is not img
    assert out.shape == img.shape
    assert np.allclose(out, img)


def test_orthorectify_disabled_returns_noop():
    img = np.random.default_rng(0).uniform(0, 1, (16, 16)).astype(np.float32)
    dem = np.random.default_rng(1).uniform(0, 100, (16, 16)).astype(np.float32)
    out, info = orthorectify(img, dem=dem, enable=False)
    assert np.allclose(out, img)
    assert info["ortho_path"] == "noop"
    assert info["ortho_reason"] == "disabled_by_config"


def test_orthorectify_no_dem_returns_noop():
    img = np.random.default_rng(0).uniform(0, 1, (16, 16)).astype(np.float32)
    out, info = orthorectify(img, dem=None, enable=True)
    assert np.allclose(out, img)
    assert info["ortho_path"] == "noop"
    assert info["ortho_reason"] == "no_dem_provided"


def test_orthorectify_with_dem_runs():
    """With a matching DEM and ortho enabled, the warp path runs and
    produces a finite result. We don't assert on a specific
    transformation — the goal is to confirm the code path executes
    and the contract (path label) is honored."""
    img = np.random.default_rng(0).uniform(0, 1, (32, 32)).astype(np.float32)
    yy, xx = np.meshgrid(np.arange(32), np.arange(32), indexing="ij")
    dem = (yy * 0.5 + xx * 0.3).astype(np.float32)
    out, info = orthorectify(img, dem=dem, enable=True)
    assert out.shape == img.shape
    assert np.isfinite(out).all()
    assert info["ortho_path"] == "dem_warp"


def test_orthorectify_with_mismatched_dem_falls_back_to_noop():
    img = np.random.default_rng(0).uniform(0, 1, (16, 16)).astype(np.float32)
    dem = np.zeros((8, 8), dtype=np.float32)  # mismatched
    out, info = orthorectify(img, dem=dem, enable=True)
    # Falls back to noop because the shapes don't match.
    assert np.allclose(out, img)
    assert info["ortho_path"] == "noop"
