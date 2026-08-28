"""Tests for the API server (FastAPI app)."""
from __future__ import annotations

import os

import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def client(monkeypatch, tmp_path):
    monkeypatch.setenv("PREPROC_SYNC", "1")
    monkeypatch.setenv("PREPROC_OUTPUT_DIR", str(tmp_path / "out"))
    # Re-import the app fresh so env is honored.
    from importlib import reload
    import preprocessing_svc.api as api_mod
    reload(api_mod)
    return TestClient(api_mod.app)


def test_health_endpoint(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_preprocess_unknown_job_returns_404(client):
    r = client.get("/preprocess/does_not_exist")
    assert r.status_code == 404


def test_preprocess_submission_round_trip(client, tmp_path):
    from mock_ingestion.generate_mock import generate

    refs = generate(out_dir=tmp_path / "mock", sensor="OHRC", height=64, width=64)
    body = {
        "job_id": "api_test_job",
        "raw_image_ref": refs["raw_image_ref"],
        "metadata_ref": refs["metadata_ref"],
        "dem_ref": "",
        "config_ref": "",
    }
    r = client.post("/preprocess", json=body)
    assert r.status_code == 200
    payload = r.json()
    assert payload["job_id"] == "api_test_job"
    assert payload["status"] == "SUCCEEDED"
    # We can also fetch the same job.
    r2 = client.get("/preprocess/api_test_job")
    assert r2.status_code == 200
    assert r2.json()["status"] == "SUCCEEDED"


def test_preprocess_rejects_duplicate_job_id(client, tmp_path):
    from mock_ingestion.generate_mock import generate

    refs = generate(out_dir=tmp_path / "mock2", sensor="OHRC", height=64, width=64)
    body = {
        "job_id": "dupe",
        "raw_image_ref": refs["raw_image_ref"],
        "metadata_ref": refs["metadata_ref"],
        "dem_ref": "",
        "config_ref": "",
    }
    r1 = client.post("/preprocess", json=body)
    assert r1.status_code == 200
    r2 = client.post("/preprocess", json=body)
    assert r2.status_code == 409


def test_preprocess_validation_error_on_bad_metadata(client, tmp_path):
    body = {
        "job_id": "bad",
        "raw_image_ref": "file:///nonexistent/raw.tif",
        "metadata_ref": "file:///nonexistent/meta.json",
        "dem_ref": "",
        "config_ref": "",
    }
    r = client.post("/preprocess", json=body)
    # With PREPROC_SYNC=1 the failure is captured and returned in the body.
    assert r.status_code == 200
    payload = r.json()
    assert payload["status"] == "FAILED"
    assert payload["error_message"]
