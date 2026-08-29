from fastapi import FastAPI, BackgroundTasks, HTTPException
from pydantic import BaseModel
from typing import Optional, Dict, Any
import uuid
import logging
import numpy as np

# Internal imports
from refinement_registration_svc.core.refinement import refine_point_phase_correlation
from refinement_registration_svc.core.transform import select_transform_model, robust_fit
from refinement_registration_svc.core.registration import resample_image, write_as_cog

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Refinement & Registration Service")

# --- API Contracts ---

class Transform(BaseModel):
    model_type: str
    params: Dict[str, Any]

class RegisterRequest(BaseModel):
    job_id: str
    verified_matches_ref: str
    initial_transform: Transform
    raw_source_ref: str
    raw_reference_ref: str
    dem_ref: Optional[str] = None
    config_ref: str

class RegisterResult(BaseModel):
    job_id: str
    status: str # "PENDING", "COMPLETED", "FAILED"
    refined_matches_ref: Optional[str] = None
    final_transform: Optional[Dict[str, Any]] = None
    registered_image_ref: Optional[str] = None
    error_message: Optional[str] = None

# --- State Management ---
jobs = {}

@app.post("/register", response_model=str)
async def register(request: RegisterRequest, background_tasks: BackgroundTasks):
    job_id = request.job_id
    jobs[job_id] = {"status": "PENDING", "result": None}
    logger.info(f"Starting registration job: {job_id}")
    background_tasks.add_task(process_registration, request)
    return job_id

@app.get("/status/{job_id}", response_model=RegisterResult)
async def get_status(job_id: str):
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    job_data = jobs[job_id]
    if job_data["status"] == "COMPLETED":
        return RegisterResult(job_id=job_id, status="COMPLETED", **job_data["result"])
    elif job_data["status"] == "FAILED":
        return RegisterResult(job_id=job_id, status="FAILED", error_message=job_data.get("error"))
    return RegisterResult(job_id=job_id, status="PENDING")

async def process_registration(request: RegisterRequest):
    try:
        # 1. Load Data (Mocked for scaffold)
        # In real system: download from S3
        # source_img = load_cog(request.raw_source_ref)
        # ref_img = load_cog(request.raw_reference_ref)
        # matches = load_parquet(request.verified_matches_ref)

        # Mocking data for architectural flow
        source_img = np.random.rand(1000, 1000).astype(np.float32)
        ref_img = np.random.rand(1000, 1000).astype(np.float32)
        matches = np.random.rand(100, 4) # p_x, p_y, q_x, q_y

        # 2. Sub-pixel Refinement
        refined_points = []
        for m in matches:
            # Extract patches
            # (Simplified: extraction logic here)
            p_patch = np.random.rand(31, 31).astype(np.float32)
            q_patch = np.random.rand(31, 31).astype(np.float32)

            dy, dx, conf = refine_point_phase_correlation(p_patch, q_patch)
            refined_points.append([m[0], m[1], m[2]+dx, m[3]+dy, conf])

        # 3. Robust Transform Fitting
        # Use selected model based on config
        config = {"transform_model": "affine"}
        model = select_transform_model(config, None)

        p = matches[:, :2]
        q = np.array([rp[:4] for rp in refined_points]) # Simplified
        # Actually q should be the refined sub-pixel points

        params, cost = robust_fit(model, p, q)

        # 4. Final Resampling
        # registered_img = resample_image(source_img, ref_img.shape, model, params)
        # write_as_cog(registered_img, f"s3://bucket/reg_{request.job_id}.cog", {})

        jobs[request.job_id] = {
            "status": "COMPLETED",
            "result": {
                "refined_matches_ref": f"s3://bucket/refined_{request.job_id}.parquet",
                "final_transform": {"model_type": "affine", "params": params.tolist(), "rms": cost},
                "registered_image_ref": f"s3://bucket/registered_{request.job_id}.cog"
            }
        }
    except Exception as e:
        logger.error(f"Job {request.job_id} failed: {str(e)}")
        jobs[request.job_id] = {"status": "FAILED", "error": str(e)}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
