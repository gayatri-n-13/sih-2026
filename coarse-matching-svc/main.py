import uvicorn
from fastapi import FastAPI, BackgroundTasks, HTTPException
from api.schemas.requests import MatchRequestModel, RemineRequestModel
from core.bootstrap.log_polar import LogPolarBootstrap
from core.matching.classical import ClassicalMatcher
from core.matching.deep_wrapper import DeepMatcher
from core.matching.fusion import MatchFusion
from models.backbone.mock_backbone import MockBackbone
import zarr
import pandas as pd
import os
import uuid
import traceback
import logging
import numpy as np
from models.data_models import Transform

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = FastAPI(title="Coarse Matching Service")

# Global state for jobs (in-memory for scaffold)
jobs = {}

# Initialize components
bootstrap = LogPolarBootstrap()
classical_matcher = ClassicalMatcher()
deep_matcher = DeepMatcher(backbone=MockBackbone())
fusion = MatchFusion()

def run_matching_job(job_id: str, req: MatchRequestModel):
    try:
        logger.info(f"Starting job {job_id}...")
        # 1. Load data from Zarr (S3 Mock)
        logger.info("Loading Zarr arrays...")
        ref_pyr = zarr.open(req.pyramid_reference_ref)
        src_pyr = zarr.open(req.pyramid_source_ref)

        logger.info("Accessing level 2...")
        ref_img = ref_pyr["level_2"][:]
        src_img = src_pyr["level_2"][:]

        logger.info("Computing bootstrap...")
        transform = bootstrap.estimate(ref_img, src_img)

        logger.info("Accessing level 0...")
        ref_img_0 = ref_pyr["level_0"][:]
        src_img_0 = src_pyr["level_0"][:]

        logger.info("Running classical matcher...")
        c_matches = classical_matcher.match(ref_img_0, src_img_0, transform)

        logger.info("Running deep matcher...")
        d_matches = deep_matcher.match(ref_img_0, src_img_0, transform)

        logger.info("Fusing matches...")
        final_matches = fusion.fuse(c_matches, d_matches)

        logger.info("Saving results...")
        output_path = f"s3_mock/{job_id}/candidates.parquet"
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        df = pd.DataFrame([
            {"p_x": m.pt_ref[0], "p_y": m.pt_ref[1],
             "q_x": m.pt_src[0], "q_y": m.pt_src[1],
             "source": m.source, "confidence": m.confidence}
            for m in final_matches
        ])
        df.to_parquet(output_path)

        jobs[job_id] = {
            "status": "COMPLETED",
            "transform": transform.to_dict() if hasattr(transform, 'to_dict') else transform.__dict__,
            "candidates_ref": output_path
        }
        logger.info(f"Job {job_id} completed successfully.")
    except Exception as e:
        logger.error(f"Job {job_id} failed: {str(e)}")
        logger.error(traceback.format_exc())
        jobs[job_id] = {"status": "FAILED", "error": str(e)}

@app.post("/match")
async def match(req: MatchRequestModel, background_tasks: BackgroundTasks):
    job_id = req.job_id or str(uuid.uuid4())
    jobs[job_id] = {"status": "PROCESSING"}
    background_tasks.add_task(run_matching_job, job_id, req)
    return {"job_id": job_id}

@app.get("/status/{job_id}")
async def get_status(job_id: str):
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    return jobs[job_id]

@app.post("/remine")
async def remine(req: RemineRequestModel):
    try:
        logger.info(f"Remining tile {req.tile_id} for job {req.job_id}...")

        # 1. Load data
        ref_pyr = zarr.open(f"s3_mock/{req.job_id}/ref_pyramid.zarr")
        src_pyr = zarr.open(f"s3_mock/{req.job_id}/src_pyramid.zarr")

        # 2. Get bounds
        x0, y0, x1, y1 = map(int, req.tile_bounds.split(','))

        # 3. Extract tiles
        ref_tile = ref_pyr["level_0"][y0:y1, x0:x1]
        src_tile = src_pyr["level_0"][y0:y1, x0:x1]

        # 4. Match
        job_status = jobs.get(req.job_id)
        if not job_status or job_status["status"] != "COMPLETED":
            raise HTTPException(status_code=400, detail="Original job must be completed first")

        transform_dict = job_status["transform"]
        transform = Transform(**transform_dict)

        # Create a temporary image with only the tile and padding
        full_ref = np.zeros((1024, 1024))
        full_ref[y0:y1, x0:x1] = ref_tile
        full_src = np.zeros((1024, 1024))
        full_src[y0:y1, x0:x1] = src_tile

        matches = deep_matcher.match(full_ref, full_src, transform)

        # Filter matches to be within the tile bounds
        tile_matches = [
            m for m in matches
            if x0 <= m.pt_ref[0] < x1 and y0 <= m.pt_ref[1] < y1
        ]

        # 5. Save
        remine_path = f"s3_mock/{req.job_id}/remine_{req.tile_id}.parquet"
        os.makedirs(os.path.dirname(remine_path), exist_ok=True)
        df = pd.DataFrame([
            {"p_x": m.pt_ref[0], "p_y": m.pt_ref[1],
             "q_x": m.pt_src[0], "q_y": m.pt_src[1],
             "source": m.source, "confidence": m.confidence}
            for m in tile_matches
        ])
        df.to_parquet(remine_path)

        return {"tile_id": req.tile_id, "additional_candidates_ref": remine_path}

    except Exception as e:
        logger.error(f"Remine failed: {str(e)}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import sys
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8001
    uvicorn.run(app, host="0.0.0.0", port=port)
