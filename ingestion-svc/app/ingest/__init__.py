"""Ingestion pipeline.

`run_ingest` is the async function scheduled by the API endpoint. It:
  1. Marks the job RUNNING.
  2. Loads the per-job config (or default).
  3. Resolves a reader for the requested sensor_type and reads the file.
  4. Validates structure.
  5. Parses metadata with the 3-tier sun-angle fallback.
  6. Writes COG + metadata.json to object storage.
  7. Marks the job COMPLETED or FAILED with a specific error_message.
"""
from __future__ import annotations

import logging
from typing import Protocol

from ..config import IngestConfig, load_config
from ..job_table import JobStore
from ..metadata_parser import parse_metadata
from ..models import (
    IngestRequest,
    IngestResult,
    JobStatus,
    MetadataSidecar,
    SensorType,
)
from ..readers import ReaderError, get_reader
from ..settings import Settings
from ..storage import write_outputs
from ..validator import validate

log = logging.getLogger("ingestion.pipeline")


class _FailingWriter:
    pass  # placeholder for future custom error reporters


async def run_ingest(req: IngestRequest, settings: Settings, store: JobStore) -> None:
    """Execute one ingestion job, updating `store` as it progresses."""
    await store.put(IngestResult(job_id=req.job_id, status=JobStatus.RUNNING))

    try:
        cfg = load_config(req.config_ref)
        reader = get_reader(req.sensor_type.value)

        if not reader.can_handle(req.source_file_uri):
            raise ReaderError(
                f"reader for sensor_type {req.sensor_type.value} "
                f"rejected {req.source_file_uri!r}"
            )

        raw = reader.read(req.source_file_uri)
        validate(raw, cfg)
        sidecar = parse_metadata(raw, SensorType(req.sensor_type.value), cfg)

        bucket = cfg.output_bucket or settings.output_bucket
        prefix = cfg.output_prefix_template or settings.output_prefix_template
        raw_ref, meta_ref = write_outputs(
            job_id=req.job_id,
            array=raw.array,
            metadata=sidecar.model_dump(mode="json"),
            bucket=bucket,
            prefix_template=prefix,
            s3_endpoint_url=settings.s3_endpoint_url,
            s3_access_key=settings.s3_access_key,
            s3_secret_key=settings.s3_secret_key,
            s3_region=settings.s3_region,
        )

        await store.put(
            IngestResult(
                job_id=req.job_id,
                status=JobStatus.COMPLETED,
                raw_image_ref=raw_ref,
                metadata_ref=meta_ref,
            )
        )
        log.info("job %s completed: %s", req.job_id, raw_ref)

    except ReaderError as exc:
        log.warning("job %s failed: %s", req.job_id, exc)
        await store.put(
            IngestResult(
                job_id=req.job_id,
                status=JobStatus.FAILED,
                error_message=str(exc),
            )
        )
    except Exception as exc:  # last-resort safety net
        log.exception("job %s crashed", req.job_id)
        await store.put(
            IngestResult(
                job_id=req.job_id,
                status=JobStatus.FAILED,
                error_message=f"unexpected: {type(exc).__name__}: {exc}",
            )
        )
