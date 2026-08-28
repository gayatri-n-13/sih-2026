"""HTTP API server for the preprocessing service.

The wire-level contract mirrors the gRPC spec in the system prompt:

    service PreprocessingService {
      rpc Preprocess (PreprocessRequest) returns (JobHandle);
      rpc GetPreprocessStatus (JobHandle) returns (PreprocessResult);
    }

We use HTTP/JSON rather than raw gRPC for two reasons: (1) the contract
is small (two endpoints) and the cost of a gRPC codegen pipeline is not
worth it, and (2) Pydantic gives us the same request/response shape
with strong validation. The Orchestrator (Member 0) and Coarse Matching
(Member 3) talk to us over HTTP, with the same JSON fields the gRPC
service would expose.

Endpoints
---------
POST /preprocess       -> {"job_id": "..."}
GET  /preprocess/{id}  -> PreprocessResult
GET  /health           -> {"status": "ok"}
"""
from __future__ import annotations

import json
import logging
import os
import threading
import uuid
from pathlib import Path
from typing import Any, Optional

from fastapi import FastAPI, HTTPException
from pydantic import ValidationError

from preprocessing_svc.config import (
    IngestMetadata,
    JobRecord,
    JobStatus,
    PreprocessConfig,
    PreprocessRequest,
    PreprocessResult,
)
from preprocessing_svc.io_utils import read_metadata_json, resolve_ref
from preprocessing_svc.pipeline import run as run_pipeline

log = logging.getLogger("preprocessing-svc")
logging.basicConfig(level=logging.INFO)

app = FastAPI(title="Preprocessing Service", version="0.1.0")

# In-memory job store. Replace with Redis/Postgres for HA. The contract
# only requires the orchestrator to be able to poll by job_id, which we
# support here.
_jobs: dict[str, JobRecord] = {}
_jobs_lock = threading.Lock()

# Where to write outputs. Defaults to ./var/outputs; override via env.
_OUTPUT_DIR = Path(os.environ.get("PREPROC_OUTPUT_DIR", "./var/outputs"))
_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def _load_config_from_ref(config_ref: str) -> PreprocessConfig:
    if not config_ref:
        return PreprocessConfig()
    p = Path(resolve_ref(config_ref))
    if not p.exists():
        log.warning("config_ref %s does not exist; using defaults", p)
        return PreprocessConfig()
    with p.open("r", encoding="utf-8") as f:
        return PreprocessConfig.model_validate(json.load(f))


def _spawn_job(req: PreprocessRequest) -> None:
    """Run the pipeline on a background thread."""
    record = JobRecord(job_id=req.job_id, status=JobStatus.RUNNING)
    with _jobs_lock:
        _jobs[req.job_id] = record
    try:
        cfg = _load_config_from_ref(req.config_ref)
        result = run_pipeline(
            request_dict=req.model_dump(),
            output_dir=_OUTPUT_DIR,
            config=cfg,
        )
        with _jobs_lock:
            _jobs[req.job_id].status = JobStatus.SUCCEEDED
            _jobs[req.job_id].result = result
    except (ValidationError, FileNotFoundError, ValueError) as exc:
        log.exception("job %s failed", req.job_id)
        with _jobs_lock:
            _jobs[req.job_id].status = JobStatus.FAILED
            _jobs[req.job_id].error = str(exc)
            _jobs[req.job_id].result = PreprocessResult(
                job_id=req.job_id,
                status=JobStatus.FAILED,
                error_message=str(exc),
            )
    except Exception as exc:  # last-resort guard
        log.exception("job %s crashed", req.job_id)
        with _jobs_lock:
            _jobs[req.job_id].status = JobStatus.FAILED
            _jobs[req.job_id].error = repr(exc)
            _jobs[req.job_id].result = PreprocessResult(
                job_id=req.job_id,
                status=JobStatus.FAILED,
                error_message=repr(exc),
            )


@app.get("/health")
def health() -> dict[str, Any]:
    return {"status": "ok", "service": "preprocessing-svc"}


@app.post("/preprocess", response_model=PreprocessResult)
def submit_preprocess(req: PreprocessRequest) -> PreprocessResult:
    """Submit a preprocessing job. The result is delivered async via the
    GET endpoint; the response here is an immediate handle with the
    assigned job_id and PENDING status. For test friendliness we run
    synchronously if the env var PREPROC_SYNC=1 is set."""
    if not req.job_id:
        req = req.model_copy(update={"job_id": str(uuid.uuid4())})
    with _jobs_lock:
        if req.job_id in _jobs:
            raise HTTPException(status_code=409, detail="job_id already exists")

    if os.environ.get("PREPROC_SYNC") == "1":
        _spawn_job(req)
        with _jobs_lock:
            rec = _jobs[req.job_id]
            if rec.result is None:
                return PreprocessResult(
                    job_id=req.job_id,
                    status=rec.status,
                    error_message=rec.error or "",
                )
            return rec.result
    # Async: register and return a PENDING handle immediately.
    with _jobs_lock:
        _jobs[req.job_id] = JobRecord(job_id=req.job_id, status=JobStatus.PENDING)
    t = threading.Thread(target=_spawn_job, args=(req,), daemon=True)
    t.start()
    return PreprocessResult(
        job_id=req.job_id, status=JobStatus.PENDING
    )


@app.get("/preprocess/{job_id}", response_model=PreprocessResult)
def get_preprocess(job_id: str) -> PreprocessResult:
    with _jobs_lock:
        rec = _jobs.get(job_id)
    if rec is None:
        raise HTTPException(status_code=404, detail="unknown job_id")
    if rec.result is not None:
        return rec.result
    return PreprocessResult(job_id=job_id, status=rec.status)


def main() -> None:
    """Run the API server with uvicorn."""
    import uvicorn

    host = os.environ.get("PREPROC_HOST", "0.0.0.0")
    port = int(os.environ.get("PREPROC_PORT", "8080"))
    uvicorn.run(app, host=host, port=port, log_level="info")


if __name__ == "__main__":
    main()
