from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
import uuid
import time
from .state_machine import RegistrationJob, JobStatus

app = FastAPI(title="Lunar Registration Orchestrator")

# In-memory job store (to be replaced by Postgres)
jobs: Dict[str, RegistrationJob] = {}

class SubmitJobRequest(BaseModel):
    source_file_uri: str
    reference_file_uri: str
    sensor_type: str # OHRC, TMC, IIRS
    config_ref: Optional[str] = None

class JobHandle(BaseModel):
    job_id: str
    status: str

class StageRecordSchema(BaseModel):
    stage: str
    status: str
    start_time: float
    end_time: Optional[float]
    metrics: Dict[str, Any]

class JobStatusResponse(BaseModel):
    job_id: str
    current_stage: str
    status: str
    stage_history: List[StageRecordSchema]

class JobResult(BaseModel):
    job_id: str
    status: str
    registered_image_ref: Optional[str] = None
    metrics_ref: Optional[str] = None
    error_message: Optional[str] = None

async def run_pipeline(job_id: str):
    """
    The main state machine driver.
    This is where the actual service calls (M1 -> M2 -> M3 -> M4 -> M5 -> M6) happen.
    """
    job = jobs[job_id]

    try:
        # 1. Ingestion (Source & Reference)
        job.transition_to(JobStatus.INGESTING)
        # TODO: Call ingestion-svc.submit(source)
        # TODO: Call ingestion-svc.submit(reference)
        # Mocking success for now
        job.context["source_ingest"] = {"raw_image_ref": "s3://.../src/raw.cog", "metadata_ref": "s3://.../src/meta.json"}
        job.context["ref_ingest"] = {"raw_image_ref": "s3://.../ref/raw.cog", "metadata_ref": "s3://.../ref/meta.json"}

        # 2. Preprocessing
        job.transition_to(JobStatus.PREPROCESSING)
        # TODO: Call preprocessing-svc.process(source_ingest)
        # TODO: Call preprocessing-svc.process(ref_ingest)
        job.context["source_preproc"] = {"pyramid_ref": "s3://.../src/pyramid.zarr"}
        job.context["ref_preproc"] = {"pyramid_ref": "s3://.../ref/pyramid.zarr"}

        # 3. Coarse Matching (MISSING SERVICE)
        job.transition_to(JobStatus.COARSE_MATCHING)
        # TODO: Call coarse-matching-svc.match(source_preproc, ref_preproc)
        job.context["coarse_matches"] = {"candidate_matches_ref": "s3://.../matches.parquet"}

        # 4. Verification
        job.transition_to(JobStatus.VERIFYING)
        # TODO: Call verification-svc.verify(coarse_matches)
        job.context["verified_matches"] = {"verified_matches_ref": "s3://.../verified.parquet"}

        # 5. Refinement/Registration (ON BRANCH)
        job.transition_to(JobStatus.REGISTERING)
        # TODO: Call refinement-registration-svc.register(verified_matches)
        job.context["registered_image"] = "s3://.../final_registered.tif"

        # 6. Evaluation
        job.transition_to(JobStatus.EVALUATING)
        # TODO: Call evaluation-svc.evaluate(registered_image)
        job.context["evaluation_metrics"] = "s3://.../metrics.json"

        job.transition_to(JobStatus.DONE)

    except Exception as e:
        job.fail(str(e))

@app.post("/v1/submit", response_model=JobHandle)
async def submit_registration_job(request: SubmitJobRequest, background_tasks: BackgroundTasks):
    job_id = str(uuid.uuid4())
    job = RegistrationJob(job_id)
    jobs[job_id] = job

    background_tasks.add_task(run_pipeline, job_id)

    return JobHandle(job_id=job_id, status=job.status.value)

@app.get("/v1/status/{job_id}", response_model=JobStatusResponse)
async def get_job_status(job_id: str):
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")

    job = jobs[job_id]
    return JobStatusResponse(
        job_id=job.job_id,
        current_stage=job.current_stage.value,
        status=job.status.value,
        stage_history=[s.to_dict() for s in job.stage_history]
    )

@app.get("/v1/result/{job_id}", response_model=JobResult)
async def get_job_result(job_id: str):
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")

    job = jobs[job_id]
    return JobResult(
        job_id=job.job_id,
        status=job.status.value,
        registered_image_ref=job.context.get("registered_image"),
        metrics_ref=job.context.get("evaluation_metrics"),
        error_message=job.error_message
    )
