"""FastAPI app entrypoint for ingestion-svc.

Wires health + v1/ingest routes. Actual ingestion logic lives in
`ingest.pipeline.run_ingest` and is dispatched asynchronously.
"""
from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import BackgroundTasks, FastAPI, HTTPException, Path
from fastapi.responses import JSONResponse

from . import __version__
from .config import load_config
from .ingest import run_ingest
from .job_table import get_job_store
from .models import (
    ErrorResponse,
    HealthResponse,
    IngestRequest,
    IngestResult,
    JobHandle,
    JobStatus,
)
from .settings import get_settings

# Reader registration happens in app.readers.__init__ when that package is
# imported (see app/readers/__init__.py:_register_default_readers).

log = logging.getLogger("ingestion")


async def _dispatch_ingest(req, settings, store) -> None:
    """Await run_ingest inside the active event loop.

    BackgroundTasks.add_task calls this coroutine after the response is sent;
    awaiting inside an async helper guarantees asyncio.create_task can find
    the running loop and schedule the work cleanly.
    """
    await run_ingest(req, settings, store)


@asynccontextmanager
async def lifespan(app: FastAPI):
    log.info("ingestion-svc starting (version=%s)", __version__)
    yield
    log.info("ingestion-svc shutting down")


app = FastAPI(
    title="Ingestion Service",
    version=__version__,
    description="First-stage ingest for the lunar image-registration pipeline.",
    lifespan=lifespan,
)


@app.get("/health", response_model=HealthResponse, tags=["meta"])
async def health() -> HealthResponse:
    """Liveness probe. Returns 'degraded' if the default config can't load."""
    try:
        load_config(None)
        status = "ok"
    except Exception as exc:  # pragma: no cover - exercised via test
        log.warning("health degraded: %s", exc)
        status = "degraded"
    return HealthResponse(status=status, version=__version__)


@app.post(
    "/v1/ingest",
    response_model=JobHandle,
    status_code=202,
    tags=["ingest"],
    responses={400: {"model": ErrorResponse}},
)
async def ingest_product(
    req: IngestRequest, background: BackgroundTasks
) -> JobHandle:
    """Submit an ingestion job. Returns immediately with PENDING; the actual
    work runs in the background and is observable via GET /v1/ingest/{job_id}.
    """
    store = get_job_store()

    # Reject duplicate job_id with an explicit error rather than clobbering.
    existing = await store.get(req.job_id)
    if existing is not None and existing.status not in (
        JobStatus.FAILED,
        JobStatus.COMPLETED,
    ):
        raise HTTPException(
            status_code=400,
            detail=f"job_id {req.job_id!r} already in use (status={existing.status})",
        )

    await store.put(
        IngestResult(job_id=req.job_id, status=JobStatus.PENDING)
    )

    settings = get_settings()
    background.add_task(_dispatch_ingest, req, settings, store)

    return JobHandle(job_id=req.job_id, status=JobStatus.PENDING)


@app.get(
    "/v1/ingest/{job_id}",
    response_model=IngestResult,
    tags=["ingest"],
    responses={404: {"model": ErrorResponse}},
)
async def get_ingest_status(job_id: str = Path(..., min_length=1, max_length=128)) -> IngestResult:
    """Poll for the current state of a previously-submitted ingestion job."""
    result = await get_job_store().get(job_id)
    if result is None:
        raise HTTPException(status_code=404, detail=f"unknown job_id {job_id!r}")
    return result


@app.exception_handler(Exception)
async def _unhandled(request, exc):  # pragma: no cover
    log.exception("unhandled error")
    return JSONResponse(
        status_code=500,
        content=ErrorResponse(code="INTERNAL", message=str(exc)).model_dump(),
    )
