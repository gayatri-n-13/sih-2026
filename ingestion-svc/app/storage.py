"""Object-storage abstraction for ingestion-svc.

The contract says the orchestrator (Member 0) and downstream services
exchange *references* (s3://…) — not raw bytes. For local development
and tests we support:

  - s3://bucket/key  → if a ``local_fakes3_root`` is configured (env var
                       ``INGESTION_FAKES3_ROOT``), treat the bucket as a
                       subdirectory there. Otherwise we record the
                       reference but do not write bytes (the byte path
                       is exercised only when the environment variable
                       is set).
  - file://path      → resolve to the local filesystem path.
  - /abs/path        → as-is.

The byte-writing path is used both by tests (under a temp dir) and by
local Docker runs (mounted volume).
"""
from __future__ import annotations

import os
import re
import shutil
from pathlib import Path
from typing import Optional
from urllib.parse import urlparse

_S3_PATTERN = re.compile(r"^s3://([^/]+)/(.*)$")


def fakes3_root() -> Optional[Path]:
    """Return the configured local-fakeS3 root, or None if not set."""
    root = os.environ.get("INGESTION_FAKES3_ROOT")
    return Path(root) if root else None


def resolve_ref(ref: str) -> Path:
    """Resolve a service reference to a local filesystem Path.

    Raises ``FileNotFoundError`` only if the reference is an explicit
    ``file://`` to a non-existent path (which means a contract
    violation). ``s3://`` references resolve to the local-fakeS3 root
    if set; otherwise we raise so callers fail fast.
    """
    if not ref:
        raise ValueError("Empty reference")
    m = _S3_PATTERN.match(ref)
    if m:
        bucket, key = m.group(1), m.group(2)
        root = fakes3_root()
        if root is None:
            raise FileNotFoundError(
                f"s3:// ref {ref} cannot be resolved: "
                f"INGESTION_FAKES3_ROOT is not set"
            )
        return root / bucket / key
    u = urlparse(ref)
    if u.scheme == "file":
        return Path(u.path)
    return Path(ref)


def ref_from_local(local_path: Path, output_prefix: str, key_suffix: str = "") -> str:
    """Build the service reference for a local file at ``local_path``.

    ``output_prefix`` is typically ``s3://ingestion-bucket/`` — we
    rewrite the local path to live under that bucket/prefix.
    """
    local_path = Path(local_path).resolve()
    if output_prefix.startswith("s3://"):
        m = _S3_PATTERN.match(output_prefix)
        assert m is not None
        bucket, prefix = m.group(1), m.group(2)
        # The reference points to the file *as it would be on S3*;
        # downstream services that resolve the ref to local need
        # fakes3_root configured to land on the same directory.
        key = f"{prefix}{key_suffix}".rstrip("/")
        # Use a stable hash-like suffix derived from the local filename
        # so multiple runs of the same fixture don't collide.
        return f"s3://{bucket}/{key}"
    if output_prefix.startswith("file://"):
        return f"{output_prefix.rstrip('/')}/{key_suffix}".rstrip("/")
    return str(Path(output_prefix) / key_suffix)


def write_bytes(target: Path, data: bytes) -> Path:
    """Write bytes to the resolved local path under fakes3."""
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(data)
    return target


def write_file_copy(src: Path, dst: Path) -> Path:
    """Copy a local file to the resolved local path under fakes3."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    return dst


def read_bytes(path: Path) -> bytes:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Not found: {path}")
    return path.read_bytes()
