"""Unit tests for IIRS band reduction."""
from __future__ import annotations

import numpy as np
import pytest

from preprocessing_svc.iirs_reduce import band_ratio_composite, pca_reduce


def test_pca_shape_and_variance():
    """PCA of an (C, Y, X) cube must return (K, Y, X) where K <= n_components,
    and explained variance must sum to < 1 (it's only the top-K components)."""
    rng = np.random.default_rng(0)
    C, Y, X = 16, 32, 32
    # Build a cube whose bands are correlated along a single direction
    # plus noise; PCA should capture most variance in the first component.
    direction = rng.standard_normal((C, 1, 1)).astype(np.float32)
    pattern = rng.standard_normal((1, Y, X)).astype(np.float32)
    cube = direction * pattern + 0.01 * rng.standard_normal((C, Y, X)).astype(np.float32)
    components, mean, explained = pca_reduce(cube, n_components=3, variance_target=0.90)
    assert components.ndim == 3
    assert components.shape[0] >= 1
    assert components.shape[0] <= 3
    assert components.shape[1:] == (Y, X)
    assert mean.shape == (C,)
    assert explained.shape[0] == components.shape[0]
    # Explained variance ratio entries must be in [0, 1].
    assert (explained >= 0).all()
    assert (explained <= 1).all()
    # For a single-dominant-direction cube, the first component should
    # capture a large share of the variance.
    assert explained[0] > 0.5


def test_pca_handles_more_components_than_needed():
    """If variance_target is already met by the first component, we
    should still return at least one component."""
    rng = np.random.default_rng(0)
    C, Y, X = 8, 16, 16
    cube = rng.standard_normal((C, Y, X)).astype(np.float32)
    components, _, _ = pca_reduce(cube, n_components=3, variance_target=0.50)
    assert components.shape[0] >= 1
    assert components.shape[0] <= 3


def test_pca_rejects_bad_shape():
    with pytest.raises(ValueError):
        pca_reduce(np.zeros((16, 16), dtype=np.float32), n_components=2)


def test_band_ratio_composite_shape():
    rng = np.random.default_rng(0)
    cube = rng.uniform(0.1, 1.0, (8, 16, 16)).astype(np.float32)
    out = band_ratio_composite(cube, numerator_bands=[0, 1], denominator_bands=[2, 3])
    assert out.shape == (16, 16)
    assert out.dtype == np.float32


def test_band_ratio_composite_avoids_div_by_zero():
    """The composite must be finite even with zero denominator bands."""
    rng = np.random.default_rng(0)
    cube = rng.uniform(0.0, 0.01, (4, 8, 8)).astype(np.float32)
    out = band_ratio_composite(cube, numerator_bands=[0], denominator_bands=[1])
    assert np.isfinite(out).all()
