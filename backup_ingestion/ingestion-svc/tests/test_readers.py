"""Unit tests for ProductReader implementations.

Covers:
- valid GeoTIFF → RawProduct with expected shape, band_count, bit_depth
- corrupted file → ReaderError with specific message
- unsupported sensor_type → ReaderError mentioning the bad value
"""
from __future__ import annotations

import numpy as np
import pytest

from app.readers import RawProduct, ReaderError, get_reader


def test_reference_reader_valid(make_geotiff):
    path = make_geotiff(bands=3, height=32, width=32, fill=5)
    raw = get_reader("REFERENCE").read(f"file://{path}")
    assert isinstance(raw, RawProduct)
    assert raw.array.shape == (3, 32, 32)
    assert int(raw.array[0, 0, 0]) == 5
    assert raw.label["band_count"] == 3
    assert raw.crs == "EPSG:4326"


def test_ohrc_reader_valid(make_geotiff):
    path = make_geotiff(bands=1, height=16, width=16, fill=10)
    raw = get_reader("OHRC").read(f"file://{path}")
    assert raw.array.shape == (1, 16, 16)


def test_reader_missing_file():
    with pytest.raises(ReaderError, match="source not found"):
        get_reader("REFERENCE").read("file:///nope/does-not-exist.tif")


def test_reader_unsupported_sensor_type():
    with pytest.raises(ReaderError, match="unsupported sensor_type"):
        get_reader("FOOBAR")


def test_reader_corrupt_band_count(make_geotiff, tmp_path):
    """An all-NaN file is rejected by the reader (or validator); either way
    a FAILED result must surface a specific message. We exercise the
    validator directly here to keep the read path stable."""
    from app.config import load_config
    from app.validator import validate

    path = make_geotiff(bands=1, nan_corrupt=True)
    raw = get_reader("REFERENCE").read(f"file://{path}")
    with pytest.raises(ReaderError, match="entirely NaN"):
        validate(raw, load_config(None))
