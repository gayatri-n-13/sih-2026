"""Object-storage writer.

Writes:
  s3://{bucket}/{job_id}/ingestion/raw.cog     (Cloud-Optimized GeoTIFF)
  s3://{job_id}/ingestion/metadata.json

First pass: local filesystem under ./local_s3/ — the layout mirrors the
S3 key layout 1:1, so swapping in real boto3 calls is a 1-function
change later. The contract (URIs returned to caller) is identical either way.
"""
from __future__ import annotations

import json
import logging
import os
import tempfile
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)


def write_outputs(
    *,
    job_id: str,
    array: Any,                       # np.ndarray
    metadata: dict[str, Any],
    bucket: str,
    prefix_template: str,
    local_root: Path | None = None,
    s3_endpoint_url: str | None = None,
    s3_access_key: str | None = None,
    s3_secret_key: str | None = None,
    s3_region: str = "us-east-1",
) -> tuple[str, str]:
    """Write raw.cog + metadata.json, return (raw_image_ref, metadata_ref).

    Strategy:
      1. Write to a local temp file via rasterio (creates a proper COG).
      2. Upload to s3 if endpoint is configured; else copy into
         local_root/{bucket}/{prefix}/ for offline / docker-compose dev.
      3. Return the canonical s3:// URI.
    """
    prefix = prefix_template.format(job_id=job_id)
    raw_key = f"{prefix}/raw.cog"
    meta_key = f"{prefix}/metadata.json"

    raw_local = _write_cog_local(array)
    meta_local = _write_json_local(metadata)

    try:
        if s3_endpoint_url and not s3_endpoint_url.startswith("http://localhost") \
                and os.environ.get("INGESTION_FORCE_LOCAL", "0") != "1":
            _upload_s3(
                raw_local, meta_local,
                bucket=bucket,
                raw_key=raw_key,
                meta_key=meta_key,
                endpoint_url=s3_endpoint_url,
                access_key=s3_access_key,
                secret_key=s3_secret_key,
                region=s3_region,
            )
        else:
            target_root = (local_root or Path("./local_s3")).resolve()
            (target_root / bucket / raw_key).parent.mkdir(parents=True, exist_ok=True)
            (target_root / bucket / raw_key).write_bytes(raw_local.read_bytes())
            (target_root / bucket / meta_key).parent.mkdir(parents=True, exist_ok=True)
            (target_root / bucket / meta_key).write_bytes(meta_local.read_bytes())
    finally:
        raw_local.unlink(missing_ok=True)
        meta_local.unlink(missing_ok=True)

    return f"s3://{bucket}/{raw_key}", f"s3://{bucket}/{meta_key}"


def _write_cog_local(array) -> Path:
    try:
        import rasterio
        from rasterio.io import MemoryFile
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("rasterio required for COG write") from exc

    if array.ndim == 2:
        array = array[None, :, :]  # rasterio wants (bands, H, W)
    bands, h, w = array.shape
    tmp = Path(tempfile.mkstemp(suffix=".tif")[1])
    profile = {
        "driver": "GTiff",
        "height": h,
        "width": w,
        "count": bands,
        "dtype": str(array.dtype),
        "tiled": True,
        "blockxsize": 256,
        "blockysize": 256,
        "compress": "deflate",
    }
    with rasterio.open(tmp, "w", **profile) as dst:
        dst.write(array)
    # Promote to COG. rasterio >= 1.3 ships `rasterio.shutil.cog_copy`.
    cog_tmp = Path(tempfile.mkstemp(suffix=".cog.tif")[1])
    try:
        rasterio.shutil.cog_copy(
            tmp, cog_tmp, compress="deflate", blocksize=256, overview_resampling="average"
        )
        tmp.unlink(missing_ok=True)
        return cog_tmp
    except Exception:
        # Fall back to plain tiled GeoTIFF if COG promotion fails
        # (e.g. rasterio < 1.3). The downstream contract still holds.
        return tmp


def _write_json_local(metadata: dict[str, Any]) -> Path:
    tmp = Path(tempfile.mkstemp(suffix=".json")[1])
    tmp.write_text(json.dumps(metadata, indent=2, sort_keys=True, default=str))
    return tmp


def _upload_s3(
    raw_local: Path,
    meta_local: Path,
    *,
    bucket: str,
    raw_key: str,
    meta_key: str,
    endpoint_url: str,
    access_key: str | None,
    secret_key: str | None,
    region: str,
) -> None:
    try:
        import boto3  # type: ignore
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("boto3 required for S3 upload") from exc

    client = boto3.client(
        "s3",
        endpoint_url=endpoint_url,
        aws_access_key_id=access_key,
        aws_secret_access_key=secret_key,
        region_name=region,
    )
    # Ensure bucket exists (MinIO friendly).
    try:
        client.head_bucket(Bucket=bucket)
    except Exception:
        client.create_bucket(Bucket=bucket)

    client.upload_file(str(raw_local), bucket, raw_key)
    client.upload_file(str(meta_local), bucket, meta_key, ExtraArgs={"ContentType": "application/json"})
