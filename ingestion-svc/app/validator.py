"""Structural validation of a RawProduct.

Fail-fast with a SPECIFIC error_message naming the offending value —
downstream and operators diagnose from this string.
"""
from __future__ import annotations

import logging

import numpy as np

from .config import IngestConfig
from .readers import RawProduct, ReaderError

log = logging.getLogger(__name__)


def validate(raw: RawProduct, cfg: IngestConfig) -> None:
    """Raise ReaderError with a specific message if invalid."""
    arr = raw.array
    if arr is None or arr.size == 0:
        raise ReaderError("empty array")

    if arr.ndim not in (2, 3):
        raise ReaderError(
            f"unexpected array ndim: got {arr.ndim}, expected 2 (single-band) or 3 (multi-band)"
        )

    bands = arr.shape[0] if arr.ndim == 3 else 1
    h, w = arr.shape[-2], arr.shape[-1]

    if not (cfg.validation.min_band_count <= bands <= cfg.validation.max_band_count):
        raise ReaderError(
            f"corrupt band count: expected {cfg.validation.min_band_count}"
            f"-{cfg.validation.max_band_count}, got {bands}"
        )

    if h > cfg.validation.max_dim or w > cfg.validation.max_dim:
        raise ReaderError(
            f"dimensions exceed max_dim={cfg.validation.max_dim} (got {h}x{w})"
        )

    if cfg.validation.reject_all_nan and _is_all_nan(arr):
        raise ReaderError("array is entirely NaN — corrupt input")

    if cfg.expected_band_count is not None and bands != cfg.expected_band_count:
        raise ReaderError(
            f"corrupt band count: expected {cfg.expected_band_count}, got {bands}"
        )

    log.debug("validation passed: bands=%d shape=%s", bands, arr.shape)


def _is_all_nan(arr: np.ndarray) -> bool:
    try:
        return bool(np.isnan(arr).all())
    except TypeError:
        return False  # integer dtype: NaN check is meaningless
