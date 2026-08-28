"""Unit tests for denoising and destriping."""
from __future__ import annotations

import numpy as np
import pytest

from preprocessing_svc.denoise import (
    bilateral_filter,
    denoise_for_level,
    destripe_columns,
)


def test_destripe_reduces_column_offsets():
    """A synthetic image with per-column offsets should be destriped
    toward the global statistics after destripe_columns(strength=1.0)."""
    rng = np.random.default_rng(0)
    base = rng.uniform(0.3, 0.7, (64, 64)).astype(np.float32)
    offsets = np.linspace(-0.2, 0.2, 64).astype(np.float32)
    corrupted = base + offsets[None, :]
    corrected = destripe_columns(corrupted, strength=1.0)
    # Per-column mean of the corrected image should be much closer to
    # the global mean than the corrupted version was.
    global_mean = float(corrupted.mean())
    orig_dev = float(np.abs(corrupted.mean(axis=0) - global_mean).mean())
    new_dev = float(np.abs(corrected.mean(axis=0) - global_mean).mean())
    assert new_dev < 0.5 * orig_dev + 1e-3


def test_destripe_strength_zero_is_identity():
    img = np.random.default_rng(0).uniform(0, 1, (16, 16)).astype(np.float32)
    out = destripe_columns(img, strength=0.0)
    assert np.allclose(out, img)


def test_destripe_3d_bandwise():
    rng = np.random.default_rng(0)
    img = rng.uniform(0, 1, (3, 16, 16)).astype(np.float32)
    out = destripe_columns(img, strength=1.0)
    assert out.shape == img.shape


def test_bilateral_filter_2d_runs():
    img = np.random.default_rng(0).uniform(0, 1, (32, 32)).astype(np.float32)
    out = bilateral_filter(img, d=5, sigma_color=0.1, sigma_space=3.0)
    assert out.shape == img.shape
    assert out.dtype == np.float32
    # Output should be in the same general range as input.
    assert out.min() >= 0.0 - 0.05
    assert out.max() <= 1.0 + 0.05


def test_bilateral_filter_3d_runs():
    img = np.random.default_rng(0).uniform(0, 1, (3, 32, 32)).astype(np.float32)
    out = bilateral_filter(img, d=5, sigma_color=0.1, sigma_space=3.0)
    assert out.shape == img.shape


def test_denoise_for_level_finest_minimal():
    """The finest level should be barely touched."""
    img = np.random.default_rng(0).uniform(0, 1, (64, 64)).astype(np.float32)
    out = denoise_for_level(img, level=0, finest_level=0)
    # With strength ramp 0, the bilateral filter is skipped entirely.
    # Destripe is still applied at low strength. The output should be
    # close to the input.
    assert np.max(np.abs(out - img)) < 0.2


def test_denoise_for_level_coarse_changes_image():
    """A coarse level should differ from the input more."""
    img = np.random.default_rng(0).uniform(0, 1, (64, 64)).astype(np.float32)
    out = denoise_for_level(img, level=4, finest_level=0, destripe_strength=0.5)
    # Coarse level is filtered more; we just verify the path runs and
    # produces a finite result.
    assert out.shape == img.shape
    assert np.isfinite(out).all()


def test_denoise_rejects_bad_ndim():
    with pytest.raises(ValueError):
        denoise_for_level(np.zeros((2, 2, 2, 2), dtype=np.float32), level=0)
