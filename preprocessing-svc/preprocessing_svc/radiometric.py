"""Radiometric normalization.

The Moon's reflectance spans a wide dynamic range and TMC/OHRC bit-depth
choices vary across products, so the first step is to map each image into
a common working range. We use a robust per-percentile stretch to float32
[0, 1] to suppress the influence of saturated pixels and noise floor.
"""
from __future__ import annotations

import numpy as np


def percentile_stretch(
    image: np.ndarray,
    low_pct: float = 2.0,
    high_pct: float = 98.0,
    out_range: tuple[float, float] = (0.0, 1.0),
) -> np.ndarray:
    """Robust per-image percentile stretch.

    Parameters
    ----------
    image : np.ndarray
        Input image. 2-D (Y, X) for pan, 3-D (C, Y, X) for multi-band.
    low_pct, high_pct : float
        Percentile cutoffs. Defaults follow the spec.
    out_range : (float, float)
        Output range mapped to after clipping.

    Returns
    -------
    np.ndarray
        float32 image with the same shape as input, mapped to ``out_range``.
    """
    if image.ndim not in (2, 3):
        raise ValueError(f"image must be 2-D or 3-D, got {image.ndim}-D")
    arr = image.astype(np.float32, copy=False)
    lo, hi = np.percentile(arr, [low_pct, high_pct])
    if not np.isfinite(lo) or not np.isfinite(hi) or hi - lo < 1e-6:
        # Degenerate image; return a constant mid-range.
        out = np.full_like(arr, 0.5 * (out_range[0] + out_range[1]), dtype=np.float32)
        return out
    out_lo, out_hi = out_range
    norm = (arr - lo) / (hi - lo)
    norm = np.clip(norm, 0.0, 1.0)
    norm = norm * (out_hi - out_lo) + out_lo
    return norm.astype(np.float32)


def to_grayscale_if_multiband(image: np.ndarray) -> np.ndarray:
    """Reduce a (C, Y, X) array to a single 2-D grayscale via mean over bands.

    Used as a fallback for non-IIRS multi-band inputs that have no
    documented band weights (e.g. TMC multi-spectral in the absence of
    ingestion-side weighting).
    """
    if image.ndim == 2:
        return image
    if image.ndim == 3:
        return image.mean(axis=0).astype(np.float32)
    raise ValueError(f"image must be 2-D or 3-D, got {image.ndim}-D")
