"""Unit tests for radiometric normalization."""
from __future__ import annotations

import numpy as np
import pytest

from preprocessing_svc.radiometric import percentile_stretch, to_grayscale_if_multiband


def test_stretch_2d_maps_to_unit_range():
    img = np.random.default_rng(0).uniform(0, 1000, (64, 64)).astype(np.float32)
    out = percentile_stretch(img)
    assert out.shape == img.shape
    assert out.dtype == np.float32
    assert out.min() >= 0.0 - 1e-6
    assert out.max() <= 1.0 + 1e-6
    # The 2 and 98 percentiles should map to roughly 0 and 1.
    assert abs(float(np.percentile(out, 2)) - 0.0) < 1e-3
    assert abs(float(np.percentile(out, 98)) - 1.0) < 1e-3


def test_stretch_3d_preserves_shape():
    img = np.random.default_rng(0).uniform(0, 1000, (3, 64, 64)).astype(np.float32)
    out = percentile_stretch(img)
    assert out.shape == img.shape
    assert out.dtype == np.float32


def test_stretch_clamps_outliers():
    """A small number of saturated pixels well above the 98th percentile
    should be clipped to 1.0 in the output (not preserve their magnitude)."""
    rng = np.random.default_rng(0)
    img = rng.uniform(0, 100, (32, 32)).astype(np.float32)
    img[0, 0] = 1e6
    img[0, 1] = 1e7
    out = percentile_stretch(img)
    # The 1e6 / 1e7 outliers are above the 98th percentile, so they
    # should clip to 1.0.
    assert out[0, 0] == pytest.approx(1.0, abs=1e-4)
    assert out[0, 1] == pytest.approx(1.0, abs=1e-4)


def test_stretch_handles_constant_image():
    img = np.full((16, 16), 7.0, dtype=np.float32)
    out = percentile_stretch(img)
    # Constant input => constant output (mid of out_range).
    assert out.shape == img.shape
    assert np.all(out == 0.5)


def test_to_grayscale_2d_passthrough():
    img = np.random.default_rng(0).uniform(0, 1, (16, 16)).astype(np.float32)
    out = to_grayscale_if_multiband(img)
    assert out.shape == img.shape
    assert np.allclose(out, img)


def test_to_grayscale_3d_averages():
    img = np.stack(
        [np.full((16, 16), 0.2, dtype=np.float32), np.full((16, 16), 0.6, dtype=np.float32)],
        axis=0,
    )
    out = to_grayscale_if_multiband(img)
    assert out.shape == (16, 16)
    assert np.allclose(out, 0.4, atol=1e-6)


def test_stretch_rejects_bad_ndim():
    with pytest.raises(ValueError):
        percentile_stretch(np.zeros((2, 2, 2, 2), dtype=np.float32))
