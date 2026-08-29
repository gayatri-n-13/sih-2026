from fastapi.testclient import TestClient
from main import app
from tests.synthetic.generator import SyntheticLunarDataGenerator
import zarr
import os
import time
import pytest

def test_api_internal():
    client = TestClient(app)

    # 1. Setup data
    gen = SyntheticLunarDataGenerator()
    data = gen.create_dataset('int_test_internal', theta=0.1, scale=1.05, tx=10, ty=-5)

    # 2. Trigger match
    payload = {
        "job_id": "int_test_internal",
        "pyramid_source_ref": data["pyramid_source"],
        "pyramid_reference_ref": data["pyramid_ref"],
        "invariant_channels_source_ref": data["invariant_channels_source"],
        "invariant_channels_reference_ref": data["invariant_channels_ref"],
    }
    resp = client.post("/match", json=payload)
    assert resp.status_code == 200

    # 3. Poll status
    for i in range(10):
        time.sleep(1)
        status_resp = client.get("/status/int_test_internal")
        status = status_resp.json()
        if status["status"] == "COMPLETED":
            break
        if status["status"] == "FAILED":
            pytest.fail(f"Job failed: {status['error']}")
    else:
        pytest.fail("Timed out")

def test_get_status_404():
    client = TestClient(app)
    resp = client.get("/status/non_existent_job")
    assert resp.status_code == 404
    assert resp.json()["detail"] == "Job not found"

def test_match_failure():
    client = TestClient(app)
    payload = {
        "job_id": "fail_job",
        "pyramid_source_ref": "invalid/path",
        "pyramid_reference_ref": "invalid/path",
        "invariant_channels_source_ref": "invalid/path",
        "invariant_channels_reference_ref": "invalid/path",
    }
    client.post("/match", json=payload)

    # Poll for failure
    for _ in range(5):
        time.sleep(1)
        status_resp = client.get("/status/fail_job")
        status = status_resp.json()
        if status["status"] == "FAILED":
            assert "error" in status
            return
    pytest.fail("Job did not fail as expected")

def test_remine_flow():
    client = TestClient(app)
    gen = SyntheticLunarDataGenerator()
    job_id = "remine_test_job"
    data = gen.create_dataset(job_id, theta=0.1, scale=1.05, tx=10, ty=-5)

    # 1. Complete a match first
    payload = {
        "job_id": job_id,
        "pyramid_source_ref": data["pyramid_source"],
        "pyramid_reference_ref": data["pyramid_ref"],
        "invariant_channels_source_ref": data["invariant_channels_source"],
        "invariant_channels_reference_ref": data["invariant_channels_ref"],
    }
    client.post("/match", json=payload)

    for _ in range(10):
        time.sleep(1)
        status_resp = client.get(f"/status/{job_id}")
        if status_resp.json()["status"] == "COMPLETED":
            break
    else:
        pytest.fail("Initial match timed out")

    # 2. Test Remine success
    remine_payload = {
        "job_id": job_id,
        "tile_id": 1,
        "tile_bounds": "10,10,100,100",
        "relaxed_confidence_threshold": 0.3
    }
    resp = client.post("/remine", json=remine_payload)
    assert resp.status_code == 200
    assert "additional_candidates_ref" in resp.json()

def test_remine_failure():
    client = TestClient(app)
    # Remine for non-existent job
    remine_payload = {
        "job_id": "no_job",
        "tile_id": 1,
        "tile_bounds": "10,10,100,100",
        "relaxed_confidence_threshold": 0.3
    }
    resp = client.post("/remine", json=remine_payload)
    assert resp.status_code == 500 # It hits the general exception handler since it can't open Zarr

