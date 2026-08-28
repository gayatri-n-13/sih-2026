"""End-to-end pipeline tests using the mock IngestResult generator."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from mock_ingestion.generate_mock import generate as generate_mock
from preprocessing_svc.config import PreprocessConfig
from preprocessing_svc.io_utils import read_metadata_json, read_zarr_array, write_metadata_json
from preprocessing_svc.pipeline import run as run_pipeline


@pytest.fixture
def ohrc_mock(tmp_path) -> dict:
    return generate_mock(
        out_dir=tmp_path / "ohrc",
        sensor="OHRC",
        height=128,
        width=128,
        sun_azimuth_deg=45.0,
        sun_elevation_deg=35.0,
        sun_angle_source_tier="label",
    )


@pytest.fixture
def iirs_mock(tmp_path) -> dict:
    return generate_mock(
        out_dir=tmp_path / "iirs",
        sensor="IIRS",
        height=64,
        width=64,
        sun_azimuth_deg=60.0,
        sun_elevation_deg=45.0,
        sun_angle_source_tier="ephemeris",
    )


def test_pipeline_ohrc_end_to_end(ohrc_mock, tmp_path):
    req = {
        "job_id": "job_ohrc",
        "raw_image_ref": ohrc_mock["raw_image_ref"],
        "metadata_ref": ohrc_mock["metadata_ref"],
        "dem_ref": "",
        "config_ref": "",
    }
    result = run_pipeline(req, output_dir=tmp_path / "out")
    assert result.status == "SUCCEEDED"
    assert Path(result.pyramid_ref).exists()
    assert Path(result.invariant_channels_ref).exists()
    assert len(result.scale_factors) >= 1
    # All scale factors are positive.
    assert all(s > 0 for s in result.scale_factors)
    # Each pyramid level must be readable.
    for i in range(len(result.scale_factors)):
        arr = read_zarr_array(Path(result.pyramid_ref), f"level_{i:02d}")
        assert arr.ndim == 2
        assert np.isfinite(arr).all()
    # Invariant channels must include the primary one.
    pc = read_zarr_array(Path(result.invariant_channels_ref), "phase_congruency")
    assert pc.ndim == 2
    assert pc.min() >= 0.0 - 1e-6
    assert pc.max() <= 1.0 + 1e-6


def test_pipeline_iirs_end_to_end(iirs_mock, tmp_path):
    req = {
        "job_id": "job_iirs",
        "raw_image_ref": iirs_mock["raw_image_ref"],
        "metadata_ref": iirs_mock["metadata_ref"],
        "dem_ref": "",
        "config_ref": "",
    }
    result = run_pipeline(req, output_dir=tmp_path / "out")
    assert result.status == "SUCCEEDED"
    side = read_metadata_json(Path(result.pyramid_ref).parent / "preprocess_result.json")
    assert side["iirs_reduction"]["reduction"] == "pca"
    assert side["iirs_reduction"]["n_components"] >= 1
    # Sum of explained variance should be sane (<= 1.0).
    ev = side["iirs_reduction"]["explained_variance_ratio"]
    assert 0.0 < sum(ev) <= 1.0 + 1e-6


def test_pipeline_uses_image_sun_estimate_when_unavailable(tmp_path):
    refs = generate_mock(
        out_dir=tmp_path / "ohrc_no_sun",
        sensor="OHRC",
        height=128,
        width=128,
        sun_azimuth_deg=None,
        sun_elevation_deg=None,
        sun_angle_source_tier="unavailable",
    )
    req = {
        "job_id": "job_no_sun",
        "raw_image_ref": refs["raw_image_ref"],
        "metadata_ref": refs["metadata_ref"],
        "dem_ref": "",
        "config_ref": "",
    }
    result = run_pipeline(req, output_dir=tmp_path / "out")
    assert result.status == "SUCCEEDED"
    side = read_metadata_json(Path(result.pyramid_ref).parent / "preprocess_result.json")
    assert side["sun_angle_source_tier"] == "unavailable"
    # The pipeline still computed SDN-Relief using the image estimate.
    pc = read_zarr_array(Path(result.invariant_channels_ref), "phase_congruency")
    assert pc.shape == (128, 128)


def test_pipeline_ortho_with_dem(tmp_path):
    refs = generate_mock(
        out_dir=tmp_path / "ohrc_dem",
        sensor="OHRC",
        height=64,
        width=64,
        sun_azimuth_deg=45.0,
        sun_elevation_deg=35.0,
    )
    # Build a synthetic DEM (linear ramp, same shape as the OHRC image).
    yy, xx = np.meshgrid(np.arange(64), np.arange(64), indexing="ij")
    dem = (yy * 0.5 + xx * 0.3).astype(np.float32)
    dem_path = tmp_path / "ohrc_dem" / "dem.tif"
    Image.fromarray(dem).save(dem_path)
    req = {
        "job_id": "job_dem",
        "raw_image_ref": refs["raw_image_ref"],
        "metadata_ref": refs["metadata_ref"],
        "dem_ref": f"file://{dem_path.resolve()}",
        "config_ref": "",
    }
    cfg = PreprocessConfig(enable_orthorectify=True)
    result = run_pipeline(req, output_dir=tmp_path / "out", config=cfg)
    assert result.status == "SUCCEEDED"
    side = read_metadata_json(Path(result.pyramid_ref).parent / "preprocess_result.json")
    assert side["ortho"]["ortho_path"] == "dem_warp"


def test_pipeline_ortho_no_op_when_disabled(tmp_path):
    refs = generate_mock(
        out_dir=tmp_path / "ohrc_no_ortho",
        sensor="OHRC",
        height=64,
        width=64,
        sun_azimuth_deg=45.0,
        sun_elevation_deg=35.0,
    )
    req = {
        "job_id": "job_no_ortho",
        "raw_image_ref": refs["raw_image_ref"],
        "metadata_ref": refs["metadata_ref"],
        "dem_ref": "",
        "config_ref": "",
    }
    cfg = PreprocessConfig(enable_orthorectify=False)
    result = run_pipeline(req, output_dir=tmp_path / "out", config=cfg)
    assert result.status == "SUCCEEDED"
    side = read_metadata_json(Path(result.pyramid_ref).parent / "preprocess_result.json")
    assert side["ortho"]["ortho_path"] == "noop"
    assert side["ortho"]["ortho_reason"] == "disabled_by_config"


def test_pipeline_propagates_failure(tmp_path):
    """If the raw image path is wrong, the pipeline fails cleanly."""
    req = {
        "job_id": "job_bad",
        "raw_image_ref": "file:///nonexistent/raw.tif",
        "metadata_ref": "file:///nonexistent/metadata.json",
        "dem_ref": "",
        "config_ref": "",
    }
    with pytest.raises(FileNotFoundError):
        run_pipeline(req, output_dir=tmp_path / "out")


def test_pipeline_uses_config_ref(tmp_path):
    """A config_ref pointing to a JSON file should be honored."""
    refs = generate_mock(
        out_dir=tmp_path / "ohrc_cfg",
        sensor="OHRC",
        height=64,
        width=64,
        sun_azimuth_deg=45.0,
        sun_elevation_deg=35.0,
    )
    cfg_path = tmp_path / "ohrc_cfg" / "config.json"
    write_metadata_json(
        cfg_path,
        {
            "pyramid": {"reference_gsd_m": 1.0, "margin_octaves": 1, "max_levels": 4},
        },
    )
    req = {
        "job_id": "job_cfg",
        "raw_image_ref": refs["raw_image_ref"],
        "metadata_ref": refs["metadata_ref"],
        "dem_ref": "",
        "config_ref": f"file://{cfg_path.resolve()}",
    }
    result = run_pipeline(req, output_dir=tmp_path / "out")
    assert result.status == "SUCCEEDED"
    # The reference_gsd_m we set was 1.0, source gsd is 0.6. ceil(log2(0.6/1.0)) = 0,
    # so we get margin+1 = 2 levels.
    assert len(result.scale_factors) == 2
