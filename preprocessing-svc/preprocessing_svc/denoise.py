"""Edge-preserving denoising and pushbroom destriping.

We use OpenCV's bilateral filter because it preserves sharp edges (crater
rims, boulders) while smoothing low-frequency noise, and it is fast enough
to run on multi-megapixel images. The strength is scaled by pyramid level:
the finest level is essentially untouched so the sub-pixel refinement
service downstream sees unmodified detail; coarser levels are filtered more
heavily because the matching step only needs structural cues.

Destriping is a column-wise moment-matching operation that targets the
fixed-pattern striping artefacts common in pushbroom hyperspectral
sensors (IIRS in particular). It does NOT apply to frame sensors (OHRC).
"""
from __future__ import annotations

import numpy as np


def _to_cv(image: np.ndarray) -> np.ndarray:
    """OpenCV bilateral filter requires uint8 or float32 in [0, 1] or [0, 255]."""
    arr = image.astype(np.float32, copy=False)
    return np.ascontiguousarray(arr)


def bilateral_filter(
    image: np.ndarray,
    d: int = 5,
    sigma_color: float = 0.08,
    sigma_space: float = 3.0,
) -> np.ndarray:
    """Apply OpenCV bilateral filter.

    For multi-band images (3-D) we filter band-by-band.
    """
    import cv2  # local import: keep module importable in pure-numpy tests

    if image.ndim == 2:
        return cv2.bilateralFilter(_to_cv(image), d, sigma_color, sigma_space)
    if image.ndim == 3:
        out = np.empty_like(image, dtype=np.float32)
        for c in range(image.shape[0]):
            out[c] = cv2.bilateralFilter(
                _to_cv(image[c]), d, sigma_color, sigma_space
            )
        return out
    raise ValueError(f"image must be 2-D or 3-D, got {image.ndim}-D")


def destripe_columns(
    image: np.ndarray,
    strength: float = 0.6,
) -> np.ndarray:
    """Column-wise moment-matching destriper for pushbroom striping.

    Computes the per-column mean and std, then matches each column to the
    image-wide mean and std with a strength blend. ``strength=0`` is the
    identity; ``strength=1`` fully matches each column to the global
    statistics. We use a robust per-column mean (median) so a few bright
    boulders do not bias the correction.

    The algorithm targets the typical 1-pixel-wide vertical striping from
    detector-element gain/offset variation. The result preserves the
    horizontal signal, which is the dominant terrain information.
    """
    if image.ndim == 2:
        return _destripe_2d(image, strength)
    if image.ndim == 3:
        out = np.empty_like(image, dtype=np.float32)
        for c in range(image.shape[0]):
            out[c] = _destripe_2d(image[c], strength)
        return out
    raise ValueError(f"image must be 2-D or 3-D, got {image.ndim}-D")


def _destripe_2d(image: np.ndarray, strength: float) -> np.ndarray:
    arr = image.astype(np.float32, copy=False)
    # Robust per-column statistic: median is unaffected by single-pixel
    # outliers; for a wider robust estimate we also use the column MAD.
    col_med = np.median(arr, axis=0, keepdims=True)
    col_mad = np.median(np.abs(arr - col_med), axis=0, keepdims=True) * 1.4826  # ~std
    col_mad = np.maximum(col_mad, 1e-3)  # avoid div-by-zero
    global_med = float(np.median(arr))
    global_mad = float(np.median(np.abs(arr - global_med))) * 1.4826
    global_mad = max(global_mad, 1e-3)

    # z-score per column, rescale to global z, convert back.
    z = (arr - col_med) / col_mad
    matched = z * global_mad + global_med
    return (arr * (1.0 - strength) + matched * strength).astype(np.float32)


def denoise_for_level(
    image: np.ndarray,
    level: int,
    finest_level: int = 0,
    bilateral_d: int = 5,
    bilateral_sigma_color: float = 0.08,
    bilateral_sigma_space: float = 3.0,
    apply_destripe: bool = True,
    destripe_strength: float = 0.6,
) -> np.ndarray:
    """Apply denoising scaled to a pyramid level.

    The finest level is essentially untouched (only a very light destripe
    if enabled). Each coarser level is filtered more aggressively, up to
    full strength at ``finest_level + 4`` or below.
    """
    if image.ndim not in (2, 3):
        raise ValueError(f"image must be 2-D or 3-D, got {image.ndim}-D")

    # Strength ramp: 0.0 at finest, 1.0 at finest+4 and beyond.
    steps_in = max(level - finest_level, 0)
    strength = min(steps_in / 4.0, 1.0)

    out = image.astype(np.float32, copy=True)

    if apply_destripe and destripe_strength > 0:
        # At the finest level we use a much weaker destripe to avoid
        # removing genuine narrow features.
        s = destripe_strength * (0.15 + 0.85 * strength)
        out = destripe_columns(out, strength=s)

    if strength > 0.05:
        out = bilateral_filter(
            out,
            d=bilateral_d,
            sigma_color=bilateral_sigma_color * (0.4 + 0.6 * strength),
            sigma_space=bilateral_sigma_space,
        )
    return out
