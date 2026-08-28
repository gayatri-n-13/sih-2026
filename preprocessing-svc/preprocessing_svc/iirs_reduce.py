"""IIRS hyperspectral band reduction.

IIRS is a 256-band hyperspectral imager in the 0.8-5.0 micron range. Most
of these bands are highly correlated for terrain-imaging purposes. We
reduce the cube to a pan-like composite using PCA (first 2-3 components
capturing >90% variance). The full cube is referenced in metadata for
provenance even though only the composite is used downstream.
"""
from __future__ import annotations

import numpy as np


def pca_reduce(
    cube: np.ndarray,
    n_components: int = 3,
    variance_target: float = 0.90,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """PCA reduction of a hyperspectral cube.

    Parameters
    ----------
    cube : np.ndarray
        Input of shape (C, Y, X) where C is the number of bands.
    n_components : int
        Maximum number of components to keep.
    variance_target : float
        Keep at least this fraction of the variance; the result is
        ``min(n_components, k)`` where ``k`` is the smallest number of
        components achieving ``>= variance_target``.

    Returns
    -------
    components : (K, Y, X) float32
        Top-K principal components as spatial maps.
    mean : (C,) float32
        Per-band mean subtracted before PCA.
    explained_variance_ratio : (K,) float32
        Fraction of variance explained by each component.
    """
    if cube.ndim != 3:
        raise ValueError(f"cube must be 3-D (C, Y, X), got {cube.ndim}-D")
    C, Y, X = cube.shape
    flat = cube.reshape(C, -1).astype(np.float32)
    mean = flat.mean(axis=1, keepdims=True)
    centered = flat - mean
    # SVD on the (C, N) matrix; left singular vectors are the PCA basis.
    # Use economy-size SVD; for C << N (typical for hyperspectral) this is fast.
    U, S, _ = np.linalg.svd(centered, full_matrices=False)
    # Explained variance ratio: S^2 / sum(S^2)
    var_total = float((S ** 2).sum())
    if var_total <= 0:
        # Degenerate input; return zero-variance components.
        explained = np.zeros(n_components, dtype=np.float32)
        components = np.zeros((min(n_components, C), Y, X), dtype=np.float32)
        return components, mean.squeeze(-1).astype(np.float32), explained
    explained = (S ** 2) / var_total
    cum = np.cumsum(explained)
    # Smallest k s.t. cum[k-1] >= variance_target, capped at n_components.
    k_var = int(np.searchsorted(cum, variance_target) + 1)
    k = max(1, min(n_components, k_var, C))
    # Project centered data onto top-k components.
    components_flat = U[:, :k].T @ centered  # (k, N)
    components = components_flat.reshape(k, Y, X).astype(np.float32)
    return components, mean.squeeze(-1).astype(np.float32), explained[:k].astype(np.float32)


def band_ratio_composite(
    cube: np.ndarray,
    numerator_bands: list[int],
    denominator_bands: list[int],
    eps: float = 1e-6,
) -> np.ndarray:
    """Build a band-ratio composite.

    Mean of numerator bands divided by mean of denominator bands, with
    epsilon for numerical safety. This is the configurable alternative to
    PCA. The output is a single 2-D float32 map.
    """
    if cube.ndim != 3:
        raise ValueError(f"cube must be 3-D (C, Y, X), got {cube.ndim}-D")
    num = cube[numerator_bands].mean(axis=0).astype(np.float32)
    den = cube[denominator_bands].mean(axis=0).astype(np.float32) + eps
    return (num / den).astype(np.float32)
