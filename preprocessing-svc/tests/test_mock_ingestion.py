"""Unit tests for the mock IngestResult generator."""
from __future__ import annotations

import json
from pathlib import Path

import pytest
from PIL import Image

from mock_ingestion.generate_mock import generate


def test_generate_ohrc(tmp_path):
    refs = generate(out_dir=tmp_path / "ohrc", sensor="OHRC", height=64, width=64)
    assert Path(refs["raw_image_ref"].replace("file://", "")).exists()
    assert Path(refs["metadata_ref"].replace("file://", "")).exists()
    meta = json.loads(Path(refs["metadata_ref"].replace("file://", "")).read_text())
    assert meta["sensor_type"] == "OHRC"
    assert meta["band_count"] == 1
    assert meta["bit_depth"] == 12


def test_generate_tmc(tmp_path):
    refs = generate(out_dir=tmp_path / "tmc", sensor="TMC", height=64, width=64)
    meta = json.loads(Path(refs["metadata_ref"].replace("file://", "")).read_text())
    assert meta["sensor_type"] == "TMC"
    assert meta["band_count"] == 3


def test_generate_iirs(tmp_path):
    refs = generate(out_dir=tmp_path / "iirs", sensor="IIRS", height=32, width=32)
    meta = json.loads(Path(refs["metadata_ref"].replace("file://", "")).read_text())
    assert meta["sensor_type"] == "IIRS"
    assert meta["band_count"] == 64
    # The mock writes the first band to the mock raw.cog; we just check
    # the file exists and is readable.
    assert Path(refs["raw_image_ref"].replace("file://", "")).exists()


def test_generate_reference(tmp_path):
    refs = generate(out_dir=tmp_path / "ref", sensor="REFERENCE", height=64, width=64)
    meta = json.loads(Path(refs["metadata_ref"].replace("file://", "")).read_text())
    assert meta["sensor_type"] == "REFERENCE"
    assert meta["bit_depth"] == 8


def test_generate_unknown_sensor_raises(tmp_path):
    with pytest.raises(ValueError):
        generate(out_dir=tmp_path / "x", sensor="NOPE")
