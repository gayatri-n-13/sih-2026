"""Image-based Sun-direction estimator (fallback).

Used ONLY when ``sun_angle_source_tier == "unavailable"``. The idea:
shadows in a single lunar image all point away from the Sun in image
space, so the gradient-orientation distribution of an image is bimodal
along a direction perpendicular to the Sun azimuth. We detect this
bimodality robustly and return the implied Sun direction.

This is a *proxy* for the true Sun direction — it conditions the
invariant representation but is not claimed to be calibrated geometry.
"""
from __future__ import annotations

import numpy as np

# Image frame convention (matches the rest of the service):
#   azimuth_deg = 0   => shadows point to image-right (East in image frame)
#   azimuth_deg = 90  => shadows point to image-bottom (South in image frame)
# A gradient orientation in [0, 180) is the direction of intensity
# increase; shadow boundaries are oriented perpendicular to the
# Sun-to-ground direction, so a Sun azimuth phi produces a strong
# gradient-orientation cluster at phi + 90 deg (and phi - 90 deg).


def _gradient_orientation(image: np.ndarray) -> np.ndarray:
    """Return per-pixel gradient orientation in degrees, in [0, 180)."""
    # Sobel via numpy (avoids OpenCV dependency for the orientation step).
    kx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float32)
    ky = kx.T
    gx = _convolve2d_same(image, kx)
    gy = _convolve2d_same(image, ky)
    mag = np.hypot(gx, gy)
    # Suppress very low-gradient (flat) regions to avoid noise dominating.
    thr = np.percentile(mag, 50.0)
    mask = mag > thr
    orient = np.degrees(np.arctan2(gy, gx)) % 180.0
    orient[~mask] = np.nan
    return orient


def _convolve2d_same(image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """2-D convolution with same-shape output, using reflect padding."""
    kh, kw = kernel.shape
    ph, pw = kh // 2, kw // 2
    padded = np.pad(image, ((ph, ph), (pw, pw)), mode="reflect")
    out = np.zeros_like(image, dtype=np.float32)
    # Naive implementation: fine for the small kernels we use.
    for i in range(kh):
        for j in range(kw):
            out += kernel[i, j] * padded[i : i + image.shape[0], j : j + image.shape[1]]
    return out.astype(np.float32)


def estimate_sun_azimuth(
    image: np.ndarray,
    hist_bins: int = 36,
) -> dict[str, float]:
    """Estimate Sun azimuth (degrees) from gradient orientations.

    Returns a dict with:
      - ``sun_azimuth_deg``: best estimate, in [0, 360)
      - ``confidence``:     peakiness of the bimodal histogram (0-1)
      - ``gradient_orientation_deg``: the dominant gradient orientation,
        in [0, 180). Sun azimuth is perpendicular to this.
    """
    if image.ndim == 3:
        image = image.mean(axis=0)
    if image.ndim != 2:
        raise ValueError(f"image must be 2-D, got {image.ndim}-D")

    orient = _gradient_orientation(image)
    valid = orient[~np.isnan(orient)]
    if valid.size < 100:
        return {
            "sun_azimuth_deg": 0.0,
            "confidence": 0.0,
            "gradient_orientation_deg": 0.0,
        }

    hist, edges = np.histogram(valid, bins=hist_bins, range=(0.0, 180.0))
    hist = hist.astype(np.float32) / max(hist.sum(), 1)
    # Look for a strong single peak. (Shadows from one Sun direction
    # produce a single dominant gradient orientation modulo 180 deg.)
    peak_idx = int(np.argmax(hist))
    grad_orient_deg = 0.5 * (edges[peak_idx] + edges[peak_idx + 1])
    # Sun azimuth is perpendicular to gradient orientation: the gradient
    # runs along the rim (perpendicular to the shadow direction), so the
    # shadow direction (and hence the Sun direction we want to use for
    # the SDN-Relief computation) is gradient + 90.
    # We return the Sun azimuth in image-frame degrees [0, 360).
    sun_az = (grad_orient_deg + 90.0) % 360.0

    # Confidence: peakiness of the histogram, normalized by uniform baseline.
    uniform = 1.0 / hist_bins
    peak = float(hist[peak_idx])
    confidence = float(min(1.0, (peak - uniform) / max(1.0 - uniform, 1e-3)))

    return {
        "sun_azimuth_deg": float(sun_az),
        "confidence": confidence,
        "gradient_orientation_deg": float(grad_orient_deg),
    }
