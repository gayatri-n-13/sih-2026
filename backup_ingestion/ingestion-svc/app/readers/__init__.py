"""Pluggable ProductReader interface.

A ProductReader's job is to take a source URI and return a `RawProduct`:
the decoded image array plus whatever label metadata the file format
exposes. Format-specific decoding (PDS3/PDS4/GeoTIFF) is hidden behind
this interface.

Concrete readers are selected by sensor_type via `get_reader()` — the
selector is data-driven (a registry), so adding a new sensor_type means
writing a new Reader subclass and registering it, NOT editing the
selector.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, ClassVar, Protocol

import numpy as np


@dataclass
class RawProduct:
    """Decoded image + whatever metadata the file format exposed.

    `label` is an opaque dict of raw label fields, useful for debugging
    but NOT relied on by downstream services — the canonical metadata
    the pipeline consumes is the MetadataSidecar written by
    metadata_parser.py.
    """

    array: np.ndarray
    label: dict[str, Any] = field(default_factory=dict)
    crs: str | None = None
    transform: tuple[float, ...] | None = None  # GDAL-style geotransform


class ReaderError(Exception):
    """Reader-specific failure. The pipeline converts this to a
    FAILED IngestResult with the message preserved verbatim so callers
    get actionable diagnostics."""


class ProductReader(Protocol):
    """Protocol every concrete reader satisfies."""

    sensor_type: ClassVar[str]

    def can_handle(self, uri: str) -> bool:
        """Return True if this reader recognizes the input at `uri`.

        Default: handle anything (used by REFERENCE / generic GeoTIFF).
        PDS-aware readers should sniff the header.
        """
        ...

    def read(self, uri: str) -> RawProduct:
        """Decode the file at `uri` and return the RawProduct.

        Raises:
            ReaderError: on any decode/structure failure. Message must
                include the offending field/value so callers can
                diagnose.
        """
        ...


_REGISTRY: dict[str, type[ProductReader]] = {}


def _register_default_readers() -> None:
    """Idempotently import built-in readers so they self-register.

    Importing this module should be enough to make OHRC/TMC/IIRS/REFERENCE
    available in the registry — callers (and tests) don't need to remember
    to import geotiff_reader explicitly.
    """
    # Local import keeps the side-effect in this package's import graph and
    # avoids creating an import cycle through app.main.
    from . import geotiff_reader  # noqa: F401


def register(sensor_type: str):
    """Class decorator: register a reader under its sensor_type.

    Usage:
        @register("OHRC")
        class OhrcReader:
            sensor_type = "OHRC"
            ...
    """

    def deco(cls: type[ProductReader]) -> type[ProductReader]:
        if not getattr(cls, "sensor_type", None):
            raise TypeError(f"{cls.__name__} must set class attr 'sensor_type'")
        if cls.sensor_type in _REGISTRY:
            raise ValueError(f"sensor_type {cls.sensor_type!r} already registered")
        _REGISTRY[cls.sensor_type] = cls
        return cls

    return deco


def get_reader(sensor_type: str) -> ProductReader:
    """Look up the registered reader for a sensor_type.

    Raises:
        ReaderError: unsupported sensor_type — message names the bad value.
    """
    try:
        return _REGISTRY[sensor_type]()
    except KeyError as exc:
        raise ReaderError(
            f"unsupported sensor_type {sensor_type!r}; "
            f"known: {sorted(_REGISTRY)}"
        ) from exc


def _open_local_or_s3(uri: str) -> Path:
    """Resolve a source URI to a local Path.

    For first pass, only file:// and plain filesystem paths are supported.
    s3:// support is stubbed so the rest of the pipeline can be exercised
    against local fixtures; real S3 fetching is a follow-up once a MinIO
    fixture is wired into docker-compose.
    """
    if uri.startswith("file://"):
        p = Path(uri.removeprefix("file://"))
    elif uri.startswith("s3://"):
        # TODO: implement S3 download to a tmp file via boto3
        raise ReaderError(f"s3 fetching not yet implemented for {uri!r}")
    else:
        p = Path(uri)
    if not p.is_file():
        raise ReaderError(f"source not found: {uri!r}")
    return p


# Eagerly register built-in readers at module import time. Must come AFTER
# _open_local_or_s3 is defined, since geotiff_reader imports it back from
# this module.
_register_default_readers()
