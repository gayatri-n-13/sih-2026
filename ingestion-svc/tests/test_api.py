"""HTTP round-trip tests for ingestion-svc.

These cover the pipeline that the user explicitly required:
``IngestProduct -> GetIngestStatus`` for OHRC, TMC, IIRS, and REFERENCE,
exercised end-to-end through the FastAPI TestClient (mirrors real
network round-trip).
"""
from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

import app.main as main_module
from app.main import app


@pytest.fixture
def client(tmp_path, monkeypatch) -> TestClient:
    """Configure workspace env and reset the in-memory job dict."""
    fakes3 = tmp_path / "fakes3"
    fakes3.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("INGESTION_SYNC", "1")
    monkeypatch.setenv("INGESTION_FAKES3_ROOT", str(fakes3))
    # Reset the in-memory jobs dict so each test starts clean.
    with main_module._jobs_lock:
        main_module._jobs.clear()
    return TestClient(app)


def _post_ingest(client, image_path: Path, sensor: str, *, job_id: str, tier: str = "label",
                 azimuth: float = 90.0, elevation: float = 30.0) -> dict:
    body = {
        "job_id": job_id,
        "raw_image_path": str(image_path),
        "sensor_type": sensor,
        "projection": "EPSG:4326",
        "sun_angle_source_tier": tier,
        "output_prefix": "s3://ingestion-bucket/",
    }
    if tier != "unavailable":
        body["sun_azimuth_deg"] = azimuth
        body["sun_elevation_deg"] = elevation
    resp = client.post("/ingest", json=body)
    return resp


# ---------------------------------------------------------------------------
# /health
# ---------------------------------------------------------------------------


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert body["service"] == "ingestion-svc"


# ---------------------------------------------------------------------------
# Pipeline round-trip per sensor (POST /ingest -> GET /ingest/{job_id})
# ---------------------------------------------------------------------------


def test_pipeline_round_trip_ohrc(client, ohrc_image):
    job_id = "http_ohrc"
    r = _post_ingest(client, ohrc_image, "OHRC", job_id=job_id)
    assert r.status_code == 200, r.text
    post_result = r.json()
    assert post_result["status"] == "SUCCEEDED"
    assert post_result["sensor_type"] == "OHRC"
    assert post_result["job_id"] == job_id

    g = client.get(f"/ingest/{job_id}")
    assert g.status_code == 200, g.text
    fetched = g.json()
    assert fetched == post_result


def test_pipeline_round_trip_tmc(client, tmc_image):
    job_id = "http_tmc"
    r = _post_ingest(client, tmc_image, "TMC", job_id=job_id, tier="ephemeris",
                     azimuth=210.0, elevation=15.0)
    assert r.status_code == 200, r.text
    post = r.json()
    assert post["status"] == "SUCCEEDED"
    assert post["sensor_type"] == "TMC"
    assert post["sun_angle_source_tier"] == "ephemeris"
    assert post["sun_azimuth_deg"] == 210.0
    assert post["band_count"] == 3

    g = client.get(f"/ingest/{job_id}")
    assert g.status_code == 200
    assert g.json() == post


def test_pipeline_round_trip_iirs(client, iirs_image):
    job_id = "http_iirs"
    r = _post_ingest(client, iirs_image, "IIRS", job_id=job_id, tier="unavailable")
    assert r.status_code == 200, r.text
    post = r.json()
    assert post["status"] == "SUCCEEDED"
    assert post["sensor_type"] == "IIRS"
    assert post["sun_azimuth_deg"] is None
    assert post["sun_elevation_deg"] is None
    assert post["sun_angle_source_tier"] == "unavailable"

    g = client.get(f"/ingest/{job_id}")
    assert g.status_code == 200
    assert g.json() == post


def test_pipeline_round_trip_reference(client, reference_image):
    job_id = "http_ref"
    r = _post_ingest(client, reference_image, "REFERENCE", job_id=job_id,
                     azimuth=0.0, elevation=45.0)
    assert r.status_code == 200, r.text
    post = r.json()
    assert post["status"] == "SUCCEEDED"
    assert post["sensor_type"] == "REFERENCE"
    assert post["bit_depth"] == 8

    g = client.get(f"/ingest/{job_id}")
    assert g.status_code == 200
    assert g.json() == post


# ---------------------------------------------------------------------------
# Error and edge cases
# ---------------------------------------------------------------------------


def test_get_unknown_job_returns_404(client):
    r = client.get("/ingest/never-submitted")
    assert r.status_code == 404


def test_post_ingest_validation_error(client, ohrc_image):
    """label tier without angles must be rejected by the validator."""
    body = {
        "job_id": "bad_tier",
        "raw_image_path": str(ohrc_image),
        "sensor_type": "OHRC",
        "projection": "EPSG:4326",
        "sun_angle_source_tier": "label",
        # no angles provided
        "output_prefix": "s3://ingestion-bucket/",
    }
    r = client.post("/ingest", json=body)
    assert r.status_code == 422  # FastAPI validation error


def test_post_ingest_missing_file_returns_failed(client, tmp_path, monkeypatch):
    """If the source file doesn't exist, status is FAILED (not 5xx)."""
    body = {
        "job_id": "missing_http",
        "raw_image_path": str(tmp_path / "does_not_exist.tif"),
        "sensor_type": "OHRC",
        "projection": "EPSG:4326",
        "sun_angle_source_tier": "unavailable",
        "output_prefix": "s3://ingestion-bucket/",
    }
    r = client.post("/ingest", json=body)
    assert r.status_code == 200, r.text
    j = r.json()
    assert j["status"] == "FAILED"
    assert j["error_message"]
