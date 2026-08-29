"""Shared pytest fixtures."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture
def fixtures_dir() -> Path:
    return FIXTURES


@pytest.fixture
def make_geotiff(tmp_path):
    """Write a small synthetic GeoTIFF and return its path."""
    import rasterio
    from rasterio.transform import from_origin

    def _make(
        *,
        name: str = "sample.tif",
        bands: int = 1,
        height: int = 64,
        width: int = 64,
        dtype: str = "uint16",
        data: np.ndarray | None = None,
        crs: str = "EPSG:4326",
        tags: dict[str, str] | None = None,
        fill: float | int = 1,
        nan_corrupt: bool = False,
    ) -> Path:
        path = tmp_path / name
        if nan_corrupt:
            # NaN can only be stored in float arrays. Force a float dtype so
            # the fixture round-trips even when the caller asked for uint.
            dtype = "float32"
        if data is None:
            arr = np.full((bands, height, width), fill, dtype=dtype)
            if nan_corrupt:
                arr[:] = np.nan
            data = arr
        elif nan_corrupt:
            data = data.copy().astype("float32")
            data[:] = np.nan

        transform = from_origin(0.0, height, 1.0, 1.0)
        profile = {
            "driver": "GTiff",
            "height": height,
            "width": width,
            "count": bands,
            "dtype": str(data.dtype),
            "crs": crs,
            "transform": transform,
        }
        with rasterio.open(path, "w", **profile) as dst:
            dst.write(data)
            if tags:
                dst.update_tags(**tags)
        return path

    return _make


@pytest.fixture
def contract_schema_path() -> Path:
    return Path(__file__).parents[1] / "contracts" / "metadata.schema.json"
