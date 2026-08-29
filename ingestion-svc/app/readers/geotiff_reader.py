"""GeoTIFF-backed readers.

Used by REFERENCE (external lunar basemaps) and as a fallback for any
sensor_type whose product has already been delivered as a GeoTIFF
(e.g. ISRO-released OHRC GeoTIFF exports).

For PDS3/PDS4-labeled products that aren't GeoTIFFs, swap in a
PDSReader that uses `pdr` or `planetaryimage`. The interface
(`ProductReader`) is what the rest of the code depends on — never
hard-code a format assumption outside this module.
"""
from __future__ import annotations

import logging
from typing import Any

import numpy as np

from . import ProductReader, RawProduct, ReaderError, _open_local_or_s3, register

log = logging.getLogger(__name__)


def _read_geotiff(uri: str, *, label_for_error: str) -> RawProduct:
    """Shared GeoTIFF decode path. Raises ReaderError on any failure."""
    try:
        import rasterio  # local import so tests can mock
    except ImportError as exc:  # pragma: no cover
        raise ReaderError("rasterio not installed") from exc

    path = _open_local_or_s3(uri)
    try:
        with rasterio.open(path) as ds:
            arr = ds.read()  # (bands, height, width)
            crs = str(ds.crs) if ds.crs else None
            transform = ds.transform.to_gdal() if ds.transform else None
            label: dict[str, Any] = dict(ds.tags())
            label["band_count"] = ds.count
            label["bit_depth"] = _dtype_bit_depth(ds.dtypes[0])
            label["width"] = ds.width
            label["height"] = ds.height
    except Exception as exc:
        # Surface "wrong format" as a specific message rather than letting
        # rasterio's generic exception leak out.
        msg = str(exc).lower()
        if "not recognized as a supported file format" in msg or "driver" in msg and "open" in msg:
            raise ReaderError(
                f"unsupported file format for {label_for_error} at {uri!r}; "
                "expected GeoTIFF or PDS-labeled input"
            ) from exc
        raise ReaderError(
            f"failed to read {label_for_error} product {uri!r}: {exc}"
        ) from exc

    return RawProduct(array=arr, label=label, crs=crs, transform=transform)


@register("REFERENCE")
class ReferenceReader:
    """Generic GeoTIFF reader for reference basemaps."""

    sensor_type = "REFERENCE"

    def can_handle(self, uri: str) -> bool:  # noqa: D401
        return True

    def read(self, uri: str) -> RawProduct:
        return _read_geotiff(uri, label_for_error="REFERENCE")


class _ChandrayaanGeoTiffReader(ProductReader):
    """Placeholder for CH2 sensor readers.

    ISRO distributes some CH2 products as GeoTIFF and others as PDS-labeled
    raw arrays. Until a real sample is inspected, this reader handles the
    GeoTIFF case (same as REFERENCE but with sensor-specific label-tag
    expectations applied via metadata_parser config). When a PDS-labeled
    sample arrives, subclass this with a PDS-aware decode path and register
    it under the same sensor_type — the selector will pick the most specific
    one.

    `sensor_type` is set as a class attribute by the concrete subclass
    (see the @register-decorated classes below).
    """

    sensor_type: str = ""  # set by subclass below

    def can_handle(self, uri: str) -> bool:  # noqa: D401
        return True

    def read(self, uri: str) -> RawProduct:
        return _read_geotiff(uri, label_for_error=self.sensor_type)


# Register each CH2 sensor under its own class so the registry holds one
# class per sensor_type. Using __init_subclass__ would also work, but a
# tiny per-sensor subclass is clearer than a metaclass.
@register("OHRC")
class _OhrcReader(_ChandrayaanGeoTiffReader):
    sensor_type = "OHRC"


@register("TMC")
class _TmcReader(_ChandrayaanGeoTiffReader):
    sensor_type = "TMC"


@register("IIRS")
class _IirsReader(_ChandrayaanGeoTiffReader):
    sensor_type = "IIRS"


def _dtype_bit_depth(dtype_str: str) -> int:
    """Translate a rasterio dtype string to a bit depth.

    For float types we report the mantissa+exponent width, matching the
    common interpretation in remote-sensing metadata (e.g. a float32 file
    is "32-bit")."""
    return {
        "uint8": 8,
        "uint16": 16,
        "uint32": 32,
        "int8": 8,
        "int16": 16,
        "int32": 32,
        "float32": 32,
        "float64": 64,
    }.get(dtype_str, 0)
