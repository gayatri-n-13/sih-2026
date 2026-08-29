"""FastAPI service for ingestion.

Endpoints:
    POST /ingest           -> submit an ingest job
    GET  /ingest/{job_id}  -> poll job status / fetch IngestResult
    GET  /health           -> liveness
"""
from __future__ import annotations

import logging
import os
import threading
import uuid
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException
from pydantic import ValidationError

from app.ingest import ingest_request
from app.models import IngestRequest, IngestResult, IngestStatus
from app.storage import fakes3_root

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("ingestion-svc")

app = FastAPI(title="Ingestion Service", version="0.1.0")

_jobs: dict[str, IngestResult] = {}
_jobs_lock = threading.Lock()


def _run_sync(req: IngestRequest, job_id: str) -> IngestResult:
    return ingest_request(req, job_id=job_id)


@app.get("/health")
def health() -> dict:
    return {
        "status": "ok",
        "service": "ingestion-svc",
        "fakes3_root": str(fakes3_root()) if fakes3_root() else None,
    }


@app.post("/ingest", response_model=IngestResult)
def submit_ingest(req: IngestRequest) -> IngestResult:
    """Submit an ingest job.

    If env var ``INGESTION_SYNC=1`` (default in tests), runs synchronously
    and returns the final IngestResult. Otherwise registers a pending
    handle and runs on a background thread.
    """
    job_id = req.job_id or str(uuid.uuid4())
    with _jobs_lock:
        if job_id in _jobs and _jobs[job_id].status not in (
            IngestStatus.FAILED,
            IngestStatus.SUCCEEDED,
        ):
            raise HTTPException(status_code=409, detail="job_id already in flight")

    if os.environ.get("INGESTION_SYNC", "0") == "1":
        result = _run_sync(req, job_id)
        with _jobs_lock:
            _jobs[job_id] = result
        return result

    # Async path
    placeholder = IngestResult(job_id=job_id, status=IngestStatus.PENDING)
    with _jobs_lock:
        _jobs[job_id] = placeholder
    t = threading.Thread(
        target=lambda: _jobs.update({job_id: _run_sync(req, job_id)}),
        daemon=True,
    )
    t.start()
    return placeholder


@app.get("/ingest/{job_id}", response_model=IngestResult)
def get_ingest(job_id: str) -> IngestResult:
    with _jobs_lock:
        rec = _jobs.get(job_id)
    if rec is None:
        raise HTTPException(status_code=404, detail="unknown job_id")
    return rec


def main() -> None:
    """Run the API server with uvicorn."""
    import uvicorn

    host = os.environ.get("INGESTION_HOST", "0.0.0.0")
    port = int(os.environ.get("INGESTION_PORT", "8080"))
    uvicorn.run(app, host=host, port=port, log_level="info")


if __name__ == "__main__":
    main()
