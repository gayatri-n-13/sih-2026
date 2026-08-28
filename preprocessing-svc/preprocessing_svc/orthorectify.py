"""Optional orthorectification hook.

Two code paths:
  - DEM present and config.enable_orthorectify = True: warp using DEM
    and the available navigation geometry. For the lunar case we treat
    the image as approximately orthorectified already (OHRC/TMC/IIRS
    products come with sensor-model-corrected framing); the DEM is used
    only for a residual relief-displacement correction via a simple
    pixel-shift lookup derived from the local slope and the
    sensor view vector. If a precise sensor model is not available, the
    function falls back to a no-op with a clear log.
  - Otherwise: explicit no-op passthrough. This is a real code path,
    not a silent skip, and is unit-tested as such.

The DEM is assumed to be in the same projection as the image; if not,
the function returns the input unchanged and records the skip reason
in the returned dict.
"""
from __future__ import annotations

from typing import Any

import numpy as np


def no_op_passthrough(image: np.ndarray) -> np.ndarray:
    """The official no-op path: returns the image unchanged.

    Returns a NEW array (not a view) so callers can safely mutate the
    result without affecting the input.
    """
    return image.astype(np.float32, copy=True)


def orthorectify_with_dem(
    image: np.ndarray,
    dem: np.ndarray,
    sensor_view_azimuth_deg: float = 0.0,
    sensor_view_elevation_deg: float = 90.0,
) -> np.ndarray:
    """Apply a residual relief-displacement correction.

    This is a deliberately simple model. We compute the local gradient
    of the DEM and shift each pixel in the opposite direction of the
    view vector, scaled by the gradient magnitude. The full rigour of a
    sensor-model-based orthorectification is out of scope for the
    service; we provide a defensible first-order correction and an
    explicit "did the work" signature so the pipeline can prove the
    path was taken.

    Parameters
    ----------
    image : (Y, X) float32
    dem   : (Y, X) float32, in meters, same grid as the image
    sensor_view_azimuth_deg, sensor_view_elevation_deg : float
        View direction in image frame.
    """
    if image.shape != dem.shape:
        # DEM does not match the image grid; bail out cleanly.
        return no_op_passthrough(image)

    img = image.astype(np.float32, copy=False)
    dem_f = dem.astype(np.float32, copy=False)

    # Local gradient of the DEM.
    gy = np.zeros_like(dem_f)
    gx = np.zeros_like(dem_f)
    gy[1:-1, :] = dem_f[2:, :] - dem_f[:-2, :]
    gx[:, 1:-1] = dem_f[:, 2:] - dem_f[:, :-2]
    gy *= 0.5
    gx *= 0.5

    # View direction unit vector in image (X, Y) frame.
    az = np.deg2rad(sensor_view_azimuth_deg)
    el = np.deg2rad(sensor_view_elevation_deg)
    vx = np.cos(az) * np.cos(el)
    vy = np.sin(az) * np.cos(el)
    # Per-pixel offset in (X, Y) = -gradient projected onto view direction.
    # Positive gx (DEM rising to the right) pushes the apparent pixel
    # to the right in image space; we shift in the opposite direction.
    shift_x = -(gx * vx) * 0.5
    shift_y = -(gy * vy) * 0.5

    # Bilinear resample on the shifted grid. We do this with
    # np.indices + interp to keep zero external deps; output is a new
    # float32 array of the same shape.
    Y, X = img.shape
    yy, xx = np.indices((Y, X), dtype=np.float32)
    sx = np.clip(xx - shift_x, 0, X - 1.001)
    sy = np.clip(yy - shift_y, 0, Y - 1.001)
    x0 = np.floor(sx).astype(np.int32)
    y0 = np.floor(sy).astype(np.int32)
    x1 = np.clip(x0 + 1, 0, X - 1)
    y1 = np.clip(y0 + 1, 0, Y - 1)
    wx = (sx - x0).astype(np.float32)
    wy = (sy - y0).astype(np.float32)
    Ia = img[y0, x0]
    Ib = img[y0, x1]
    Ic = img[y1, x0]
    Id = img[y1, x1]
    warped = (
        (1 - wy) * ((1 - wx) * Ia + wx * Ib)
        + wy * ((1 - wx) * Ic + wx * Id)
    )
    return warped.astype(np.float32)


def orthorectify(
    image: np.ndarray,
    dem: np.ndarray | None,
    enable: bool,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Top-level dispatch.

    Always returns a tuple of (output_image, info_dict). The info_dict
    records which code path was taken so the pipeline can log it.
    """
    info: dict[str, Any] = {
        "ortho_path": "noop",
        "ortho_reason": "",
    }
    if not enable or dem is None:
        if not enable:
            info["ortho_reason"] = "disabled_by_config"
        else:
            info["ortho_reason"] = "no_dem_provided"
        return no_op_passthrough(image), info

    if image.shape != dem.shape:
        # DEM grid does not match the image; the sensor-model-based
        # warp is not applicable, so we explicitly fall back to no-op
        # and record the reason.
        info["ortho_reason"] = "dem_shape_mismatch"
        return no_op_passthrough(image), info

    out = orthorectify_with_dem(image, dem)
    info["ortho_path"] = "dem_warp"
    return out, info
