"""Multi-scale pyramid construction.

The level count is chosen so the coarsest level of the source and the
reference have comparable effective GSD. We also add a margin in octaves
so the matcher has comfortable overlap on both sides.
"""
from __future__ import annotations

import math
from typing import Sequence

import numpy as np


def compute_scale_factors(
    source_gsd: float,
    reference_gsd: float,
    margin_octaves: int = 2,
    max_levels: int = 10,
) -> list[float]:
    """Return per-level scale factors (1.0 = full res, 0.5 = half, ...).

    Number of levels is chosen so that the coarsest level is coarser
    than the reference GSD by ``margin_octaves`` extra halvings.
    """
    if source_gsd <= 0 or reference_gsd <= 0:
        raise ValueError("GSD values must be positive")
    # Number of halvings to bridge source GSD to reference GSD.
    n_bridge = max(0, math.ceil(math.log2(source_gsd / reference_gsd)))
    n_levels = max(1, min(max_levels, n_bridge + margin_octaves + 1))
    return [1.0 / (2 ** i) for i in range(n_levels)]


def gaussian_pyramid(
    image: np.ndarray,
    scale_factors: Sequence[float],
) -> list[np.ndarray]:
    """Build a Gaussian pyramid.

    Each level is computed by Gaussian smoothing and decimation, NOT by
    taking ``image[::2, ::2]`` directly, so the low-frequency content is
    well-sampled.

    Multi-band (C, Y, X) images are handled band-by-band; 2-D (Y, X) is
    handled directly.
    """
    if image.ndim not in (2, 3):
        raise ValueError(f"image must be 2-D or 3-D, got {image.ndim}-D")
    if image.ndim == 2:
        return _pyr_2d(image, scale_factors)
    out = []
    for c in range(image.shape[0]):
        out.append(_pyr_2d(image[c], scale_factors))
    # Stack per-band lists into a per-level list of (C, y, x) arrays.
    n_levels = len(scale_factors)
    return [
        np.stack([out[c][i] for c in range(image.shape[0])], axis=0).astype(np.float32)
        for i in range(n_levels)
    ]


def _pyr_2d(image: np.ndarray, scale_factors: Sequence[float]) -> list[np.ndarray]:
    img = image.astype(np.float32, copy=False)
    levels: list[np.ndarray] = [img]
    for s in scale_factors[1:]:
        prev = levels[-1]
        smoothed = _gaussian_blur(prev)
        # Compute new shape using floor so we never over-sample.
        new_y = max(1, int(np.floor(prev.shape[0] * s / scale_factors[len(levels) - 1])))
        new_x = max(1, int(np.floor(prev.shape[1] * s / scale_factors[len(levels) - 1])))
        # Actually, since scale_factors are powers of two, we can just halve.
        new_y = max(1, prev.shape[0] // 2)
        new_x = max(1, prev.shape[1] // 2)
        decimated = smoothed[::2, ::2][:new_y, :new_x]
        levels.append(decimated.astype(np.float32))
    return levels


def _gaussian_blur(image: np.ndarray, sigma: float = 1.0) -> np.ndarray:
    """Separable 5x5-ish Gaussian blur via reflect padding."""
    img = image.astype(np.float32, copy=False)
    r = int(np.ceil(3 * sigma))
    x = np.arange(-r, r + 1, dtype=np.float32)
    k = np.exp(-(x * x) / (2 * sigma * sigma))
    k = (k / k.sum()).astype(np.float32)
    # Horizontal pass
    ap = np.pad(img, ((0, 0), (r, r)), mode="reflect")
    tmp = np.zeros_like(img, dtype=np.float32)
    for i, w in enumerate(k):
        tmp = tmp + (np.float32(w) * ap[:, i : i + img.shape[1]]).astype(np.float32)
    # Vertical pass
    ap = np.pad(tmp, ((r, r), (0, 0)), mode="reflect")
    out = np.zeros_like(tmp, dtype=np.float32)
    for i, w in enumerate(k):
        out = out + (np.float32(w) * ap[i : i + img.shape[0], :]).astype(np.float32)
    return out
