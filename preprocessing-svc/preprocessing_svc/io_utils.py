"""Object-storage reference resolution and Zarr I/O.

The contract says services exchange REFERENCES (s3://...); they do not
exchange raw pixel bytes. For local development and tests we accept:
  - s3://bucket/key     (requires S3 credentials; we do not pull bytes,
                         but we record the path so the contract is honored)
  - file:///abs/path    (local absolute path)
  - /abs/path or rel    (local path)

The Zarr writer chunks along (Y, X) so downstream readers can stream tiles.
"""
from __future__ import annotations

import json
import os
import re
import tempfile
from pathlib import Path
from typing import Any, Union

import numpy as np
import zarr
from PIL import Image


PathLike = Union[str, os.PathLike]


_S3_PATTERN = re.compile(r"^s3://([^/]+)/(.+)$")


def resolve_ref(ref: str, local_root: PathLike | None = None) -> Path:
    """Resolve a service reference to a local filesystem path.

    For s3:// refs, if the env var ``PREPROC_LOCAL_FAKES3`` is set to a
    directory, we treat the bucket as a subdirectory there. Otherwise we
    fall back to a per-job staging dir so the contract is honored.

    For file:// refs and bare paths, return as-is (absolute).
    """
    if not ref:
        raise ValueError("Empty reference")
    m = _S3_PATTERN.match(ref)
    if m:
        bucket, key = m.group(1), m.group(2)
        fakes3 = os.environ.get("PREPROC_LOCAL_FAKES3")
        if fakes3:
            return Path(fakes3) / bucket / key
        # No S3 backend: stage in a temp dir and emit a warning via metadata.
        stage = Path(os.environ.get("PREPROC_STAGE_DIR", tempfile.gettempdir())) / "preproc" / bucket
        stage.mkdir(parents=True, exist_ok=True)
        return stage / key
    if ref.startswith("file://"):
        return Path(ref[len("file://") :])
    p = Path(ref)
    if not p.is_absolute() and local_root is not None:
        p = Path(local_root) / p
    return p


def read_image_array(path: Path) -> np.ndarray:
    """Read a TIFF/PNG/COG into a numpy array.

    For multi-band inputs we return ``(C, Y, X)`` float32. For single-band
    inputs we return ``(Y, X)`` float32. The array is always float32 to
    keep downstream numerics stable. Multi-page TIFFs are stacked along
    a leading axis (so an IIRS-style 64-page TIFF becomes a (64, Y, X)
    cube).
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Image not found: {path}")
    img = Image.open(path)
    # Multi-page TIFF: stack all pages.
    pages = []
    try:
        while True:
            pages.append(np.asarray(img))
            img.seek(img.tell() + 1)
    except EOFError:
        pass
    if len(pages) > 1:
        # Stack as (C, Y, X) for any 2-D page; if pages are 3-D (rare),
        # we collapse their channel axis first.
        cleaned = []
        for p in pages:
            if p.ndim == 3:
                # Multi-channel page; collapse channels by mean.
                p = p.mean(axis=-1)
            cleaned.append(p.astype(np.float32))
        return np.stack(cleaned, axis=0)

    arr = pages[0] if pages else np.asarray(img)
    if arr.ndim == 2:
        return arr.astype(np.float32)
    if arr.ndim == 3:
        # PIL gives (Y, X, C). Move to (C, Y, X) for consistency.
        return np.moveaxis(arr, -1, 0).astype(np.float32)
    raise ValueError(f"Unsupported image shape: {arr.shape}")


def read_metadata_json(path: Path) -> dict[str, Any]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Metadata not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_metadata_json(path: Path, data: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, sort_keys=True)


def _zarr_version() -> int:
    """Return major zarr version (2 or 3)."""
    v = zarr.__version__
    major = int(v.split(".")[0])
    return major


def write_zarr(
    store_path: Path,
    arrays: dict[str, np.ndarray],
    attrs: dict[str, Any] | None = None,
    chunk_size: int = 256,
) -> Path:
    """Write a Zarr group with named arrays.

    Each array is chunked along (Y, X) (or first axis for 1-D data) with
    a square-ish chunk of ``chunk_size`` pixels. ``attrs`` is attached to
    the group root.
    """
    store_path = Path(store_path)
    if store_path.exists():
        import shutil

        shutil.rmtree(store_path)
    store_path.mkdir(parents=True, exist_ok=True)

    v = _zarr_version()
    if v == 2:
        store = zarr.DirectoryStore(str(store_path))
        root = zarr.group(store=store, overwrite=True)
    else:
        root = zarr.open_group(str(store_path), mode="w")

    for name, arr in arrays.items():
        a = np.asarray(arr)
        if a.ndim >= 2:
            chunks = (chunk_size,) * (a.ndim - 1) + (chunk_size,)
            chunks = tuple(max(1, min(c, s)) for c, s in zip(chunks, a.shape))
        else:
            chunks = (max(1, min(chunk_size, a.shape[0])),)
        if v == 2:
            ds = root.create_dataset(
                name,
                shape=a.shape,
                chunks=chunks,
                dtype=a.dtype,
                overwrite=True,
            )
            ds[:] = a
        else:
            ds = root.create_array(
                name,
                shape=a.shape,
                chunks=chunks,
                dtype=a.dtype,
                overwrite=True,
            )
            ds[:] = a

    if attrs:
        if v == 2:
            root.attrs.put(attrs)
        else:
            root.attrs.update(attrs)
    return store_path


def read_zarr_array(store_path: Path, name: str) -> np.ndarray:
    store_path = Path(store_path)
    root = zarr.open_group(str(store_path), mode="r")
    return np.asarray(root[name])
