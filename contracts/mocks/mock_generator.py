import uuid
import json
import random
from typing import Dict, Any, Optional

class MockGenerator:
    """
    Generates mock payloads based on the verified contracts of the SIH-2026 pipeline.
    These are used for fast, deterministic CI integration tests.
    """

    def __init__(self, bucket_prefix: str = "s3://mock-bucket"):
        self.bucket_prefix = bucket_prefix

    def generate_job_id(self) -> str:
        return str(uuid.uuid4())

    def mock_ingest_result(self, job_id: str, status: str = "COMPLETED", error: Optional[str] = None) -> Dict[str, Any]:
        """Mocks the IngestResult from ingestion-svc."""
        if status == "FAILED":
            return {
                "job_id": job_id,
                "status": "FAILED",
                "error_message": error or "Random ingestion failure"
            }

        return {
            "job_id": job_id,
            "status": "COMPLETED",
            "raw_image_ref": f"{self.bucket_prefix}/{job_id}/ingestion/raw.cog",
            "metadata_ref": f"{self.bucket_prefix}/{job_id}/ingestion/metadata.json",
            "error_message": ""
        }

    def mock_ingest_metadata(self, sensor_type: str = "OHRC") -> Dict[str, Any]:
        """Mocks the metadata.json sidecar written by ingestion-svc."""
        return {
            "sensor_type": sensor_type,
            "gsd": 0.25 if sensor_type == "OHRC" else 5.0,
            "acquisition_time": "2026-08-30T12:00:00Z",
            "sun_azimuth_deg": random.uniform(0, 360),
            "sun_elevation_deg": random.uniform(-90, 90),
            "sun_angle_source_tier": "label",
            "projection": "EPSG:4326",
            "footprint_wkt": "POLYGON((...))",
            "band_count": 1,
            "bit_depth": 16
        }

    def mock_preprocess_result(self, job_id: str, status: str = "SUCCEEDED", error: Optional[str] = None) -> Dict[str, Any]:
        """Mocks the PreprocessResult from preprocessing-svc."""
        if status == "FAILED":
            return {
                "job_id": job_id,
                "status": "FAILED",
                "error_message": error or "Preprocessing algorithm crash"
            }

        return {
            "job_id": job_id,
            "status": "SUCCEEDED",
            "pyramid_ref": f"{self.bucket_prefix}/{job_id}/preprocessing/pyramid.zarr",
            "invariant_channels_ref": f"{self.bucket_prefix}/{job_id}/preprocessing/invariant_channels.zarr",
            "scale_factors": [1.0, 0.5, 0.25, 0.125],
            "sensor_type": "OHRC",
            "gsd": 0.25,
            "reference_gsd_m": 5.0,
            "sun_azimuth_used": 145.2,
            "sun_azimuth_source": "metadata",
            "error_message": ""
        }

    def mock_verify_result(self, job_id: str, status: str = "COMPLETED", error: Optional[str] = None) -> Dict[str, Any]:
        """Mocks the VerifyResult from verification-svc."""
        if status == "FAILED":
            return {
                "job_id": job_id,
                "status": "FAILED",
                "error_message": error or "Geometric verification failed to find inliers"
            }

        return {
            "job_id": job_id,
            "status": "COMPLETED",
            "verified_matches_ref": f"{self.bucket_prefix}/{job_id}/verification/verified_matches.parquet",
            "coverage_report": {
                "tile_grid_rows": 10,
                "tile_grid_cols": 10,
                "per_tile_counts": [random.randint(5, 50) for _ in range(100)],
                "under_covered_tiles": [],
                "coverage_fraction": 0.98,
                "remine_calls_total": 2,
                "remine_iterations_used": 1
            },
            "updated_transform": {
                "theta": 0.01,
                "scale": 1.0,
                "tx": 0.5,
                "ty": -0.2,
                "confidence": 0.95
            },
            "error_message": ""
        }

if __name__ == "__main__":
    gen = MockGenerator()
    jid = gen.generate_job_id()
    print(f"Job ID: {jid}")
    print("Ingest Mock:", json.dumps(gen.mock_ingest_result(jid), indent=2))
    print("Preprocess Mock:", json.dumps(gen.mock_preprocess_result(jid), indent=2))
    print("Verify Mock:", json.dumps(gen.mock_verify_result(jid), indent=2))
