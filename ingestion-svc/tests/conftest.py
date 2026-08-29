"""Shared test fixtures."""
from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from PIL import Image  # noqa: E402
import numpy as np  # noqa: E402


@pytest.fixture
def workspace(tmp_path, monkeypatch):
    """Configure a clean ingestion workspace under tmp_path.

    - Sets INGESTION_SYNC=1 so HTTP handlers run synchronously
    - Sets INGESTION_FAKES3_ROOT so byte-writing paths are exercised
    - Returns the tmp_path for the test's own use
    """
    fakes3 = tmp_path / "fakes3"
    fakes3.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("INGESTION_SYNC", "1")
    monkeypatch.setenv("INGESTION_FAKES3_ROOT", str(fakes3))
    return tmp_path


@pytest.fixture
def ohrc_image(tmp_path) -> Path:
    """A small synthetic OHRC-like TIFF (1 band, uint16)."""
    arr = (np.random.default_rng(0).uniform(1000, 4000, (64, 64))).astype(np.uint16)
    p = tmp_path / "ohrc.tif"
    Image.fromarray(arr).save(p)
    return p


@pytest.fixture
def tmc_image(tmp_path) -> Path:
    """A small synthetic TMC-like TIFF (3 bands).

    For multi-band uint16, PIL's RGB mode expects uint8; we scale and
    store as uint8 RGB so PIL can roundtrip the file. The ingest path
    only checks band_count and bit_depth via mode-name heuristics, so
    this is sufficient for exercising the contract.
    """
    rng = np.random.default_rng(1)
    arr = (rng.uniform(500, 3000, (64, 64, 3))).astype(np.uint16)
    arr8 = (arr / 256).clip(0, 255).astype(np.uint8)
    p = tmp_path / "tmc.tif"
    Image.fromarray(arr8, mode="RGB").save(p)
    return p


@pytest.fixture
def iirs_image(tmp_path) -> Path:
    """A small synthetic IIRS-like TIFF (3 bands, uint8 RGB)."""
    rng = np.random.default_rng(2)
    arr = (rng.uniform(0, 255, (32, 32, 3))).astype(np.uint8)
    p = tmp_path / "iirs.tif"
    Image.fromarray(arr, mode="RGB").save(p)
    return p


@pytest.fixture
def reference_image(tmp_path) -> Path:
    """A small synthetic REFERENCE basemap (8-bit grayscale)."""
    arr = np.random.default_rng(3).integers(0, 255, (64, 64), dtype=np.uint8)
    p = tmp_path / "ref.tif"
    Image.fromarray(arr).save(p)
    return p


@pytest.fixture
def output_prefix() -> str:
    return "s3://ingestion-bucket/"
