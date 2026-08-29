"""End-to-end tests through the FastAPI app.

Exercises the round-trip IngestProduct → GetIngestStatus for each sensor_type,
plus failure paths (corrupt input, unsupported sensor_type).
"""
from __future__ import annotations

import asyncio
import json

import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.settings import Settings


@pytest.fixture(autouse=True)
def _force_local(monkeypatch, tmp_path):
    """Make storage write to a per-test local dir instead of S3."""
    monkeypatch.setenv("INGESTION_FORCE_LOCAL", "1")
    monkeypatch.setenv("INGESTION_OUTPUT_BUCKET", "test-bucket")
    yield


def _wait_for_terminal(client: TestClient, job_id: str, timeout: float = 5.0):
    """Poll the status endpoint until the job leaves PENDING/RUNNING."""
    import time
    deadline = time.time() + timeout
    last = None
    while time.time() < deadline:
        r = client.get(f"/v1/ingest/{job_id}")
        assert r.status_code == 200, r.text
        last = r.json()
        if last["status"] in ("COMPLETED", "FAILED"):
            return last
        time.sleep(0.05)
    raise AssertionError(f"job {job_id} did not finish; last={last}")


def test_round_trip_reference(make_geotiff):
    path = make_geotiff(bands=1, height=16, width=16)
    client = TestClient(app)
    r = client.post("/v1/ingest", json={
        "job_id": "rt-1",
        "source_file_uri": f"file://{path}",
        "sensor_type": "REFERENCE",
    })
    assert r.status_code == 202, r.text
    result = _wait_for_terminal(client, "rt-1")
    assert result["status"] == "COMPLETED", result
    assert result["raw_image_ref"] and result["raw_image_ref"].endswith("raw.cog")
    assert result["metadata_ref"] and result["metadata_ref"].endswith("metadata.json")

    # Verify the written metadata.json conforms to the published schema.
    raw_uri = result["raw_image_ref"].replace("s3://", "./local_s3/")
    meta_uri = result["metadata_ref"].replace("s3://", "./local_s3/")
    import pathlib
    payload = json.loads(pathlib.Path(meta_uri).read_text())
    assert payload["sensor_type"] == "REFERENCE"
    assert payload["band_count"] == 1
    assert payload["sun_angle_source_tier"] in ("label", "ephemeris", "unavailable")


def test_corrupt_input_fails_with_specific_message(make_geotiff):
    path = make_geotiff(bands=1, nan_corrupt=True)
    client = TestClient(app)
    r = client.post("/v1/ingest", json={
        "job_id": "corrupt-1",
        "source_file_uri": f"file://{path}",
        "sensor_type": "REFERENCE",
    })
    assert r.status_code == 202
    result = _wait_for_terminal(client, "corrupt-1")
    assert result["status"] == "FAILED"
    assert "NaN" in (result["error_message"] or "")


def test_unsupported_sensor_type_returns_400(make_geotiff):
    path = make_geotiff()
    client = TestClient(app)
    r = client.post("/v1/ingest", json={
        "job_id": "bad-1",
        "source_file_uri": f"file://{path}",
        "sensor_type": "FOO",
    })
    # Pydantic enum validation rejects before the handler runs.
    assert r.status_code == 422


def test_unknown_job_id_returns_404():
    client = TestClient(app)
    r = client.get("/v1/ingest/does-not-exist")
    assert r.status_code == 404


@pytest.mark.parametrize(
    "sensor_type",
    ["OHRC", "TMC", "IIRS", "REFERENCE"],
)
def test_round_trip_per_sensor_type(make_geotiff, sensor_type):
    """IngestProduct → GetIngestStatus succeeds for every supported sensor."""
    path = make_geotiff(bands=1, height=16, width=16)
    client = TestClient(app)
    job_id = f"rt-{sensor_type.lower()}"
    r = client.post("/v1/ingest", json={
        "job_id": job_id,
        "source_file_uri": f"file://{path}",
        "sensor_type": sensor_type,
    })
    assert r.status_code == 202, r.text
    result = _wait_for_terminal(client, job_id)
    assert result["status"] == "COMPLETED", result
    assert result["raw_image_ref"] and result["raw_image_ref"].endswith("raw.cog")
    assert result["metadata_ref"] and result["metadata_ref"].endswith("metadata.json")

    # The written metadata.json must round-trip against the published schema.
    meta_uri = result["metadata_ref"].replace("s3://", "./local_s3/")
    import pathlib
    payload = json.loads(pathlib.Path(meta_uri).read_text())
    assert payload["sensor_type"] == sensor_type
    assert payload["sun_angle_source_tier"] in ("label", "ephemeris", "unavailable")
    # GSD falls back to per-sensor default when label has none
    expected_gsd = {"OHRC": 0.6, "TMC": 5.0, "IIRS": 20.0, "REFERENCE": 5.0}[sensor_type]
    assert payload["gsd"] == expected_gsd
