import numpy as np
from scipy.optimize import least_squares
from typing import Tuple, Optional, List
import cv2

def refine_point_phase_correlation(template: np.ndarray, reference: np.ndarray) -> Tuple[float, float, float]:
    """
    Refines a point match using OpenCV's phaseCorrelate for sub-pixel accuracy.
    """
    # Ensure float32
    t = template.astype(np.float32)
    r = reference.astype(np.float32)

    # Apply a Hanning window to reduce edge effects
    h, w = t.shape
    win = np.outer(np.hanning(h), np.hanning(w)).astype(np.float32)
    t *= win
    r *= win

    try:
        (dx, dy), response = cv2.phaseCorrelate(t, r)
        confidence = float(response)
    except Exception:
        dx, dy, confidence = 0.0, 0.0, 0.0

    return float(dy), float(dx), confidence

def interpolate_patch(image: np.ndarray, dy: float, dx: float, size: int) -> np.ndarray:
    """
    Extracts a patch from the image centered at (0,0) shifted by (dy, dx)
    using bilinear interpolation.
    """
    h, w = image.shape
    y, x = np.indices((size, size))

    ry = y - size // 2 + dy
    rx = x - size // 2 + dx

    ry_clipped = np.clip(ry, 0, h - 1)
    rx_clipped = np.clip(rx, 0, w - 1)

    y_floor = np.floor(ry_clipped).astype(int)
    x_floor = np.floor(rx_clipped).astype(int)
    y_ceil = np.minimum(y_floor + 1, h - 1)
    x_ceil = np.minimum(x_floor + 1, w - 1)

    wy = ry_clipped - y_floor
    wx = rx_clipped - x_floor

    val_00 = image[y_floor, x_floor]
    val_01 = image[y_floor, x_ceil]
    val_10 = image[y_ceil, x_floor]
    val_11 = image[y_ceil, x_ceil]

    return (1-wy) * ((1-wx)*val_00 + wx*val_01) + \
           wy * ((1-wx)*val_10 + wx*val_11)

def lsm_objective(params, template, reference):
    """
    Objective function for Least-Squares Matching.
    params: [dy, dx, g, b]
    """
    dy, dx, g, b = params
    M = np.float32([[1, 0, dx], [0, 1, dy]])
    ref_patch = cv2.warpAffine(reference, M, (reference.shape[1], reference.shape[0]),
                                flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)

    error = ref_patch - (g * template + b)
    return error.flatten()

def refine_point_lsm(template: np.ndarray, reference: np.ndarray,
                    initial_shift: Tuple[float, float]) -> Tuple[float, float, float, float]:
    """
    Refines a point match using Least-Squares Matching.
    Returns (dy, dx, g, b).
    """
    dy_init, dx_init = initial_shift
    params_init = [dy_init, dx_init, 1.0, 0.0]
    bounds = ([ dy_init - 1.0, dx_init - 1.0, 0.0, -100.0],
              [ dy_init + 1.0, dx_init + 1.0, 10.0, 100.0])

    res = least_squares(lsm_objective, params_init, args=(template, reference),
                        bounds=bounds, ftol=1e-4, xtol=1e-4)

    return res.x # dy, dx, g, b
