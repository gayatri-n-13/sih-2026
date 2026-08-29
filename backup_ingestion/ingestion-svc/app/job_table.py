"""In-process job table.

First pass: a simple thread-safe dict. Persistence across restarts is a known
gap — swap with Redis/Postgres via the `job_table_backend` setting when
productionizing. The interface (JobStore protocol) is stable; only the impl
needs to change.
"""
from __future__ import annotations

import threading
from typing import Protocol

from .models import IngestResult


class JobStore(Protocol):
    async def put(self, result: IngestResult) -> None: ...
    async def get(self, job_id: str) -> IngestResult | None: ...


class InMemoryJobStore:
    """Thread-safe in-process job table. Sufficient for first pass / tests."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._jobs: dict[str, IngestResult] = {}

    async def put(self, result: IngestResult) -> None:
        with self._lock:
            self._jobs[result.job_id] = result

    async def get(self, job_id: str) -> IngestResult | None:
        with self._lock:
            return self._jobs.get(job_id)


_singleton: JobStore | None = None


def get_job_store() -> JobStore:
    global _singleton
    if _singleton is None:
        _singleton = InMemoryJobStore()
    return _singleton
