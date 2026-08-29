"""Smoke test for the /health endpoint."""
from __future__ import annotations

from fastapi.testclient import TestClient

from app.main import app


def test_health_ok():
    client = TestClient(app)
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] in ("ok", "degraded")
    assert isinstance(body["version"], str) and body["version"]
