import grpc
from concurrent import futures
import logging
import uuid
import pandas as pd
from . import verification_pb2
from . import verification_pb2_grpc
from .engine import VerificationEngine, Transform, CoverageReport
from .mocks import CoarseMatchingMock

class VerificationService(verification_pb2_grpc.VerificationServiceServicer):
    def __init__(self):
        self.mock_client = CoarseMatchingMock()
        self.engine = VerificationEngine(self.mock_client)
        self.jobs = {}
        self.logger = logging.getLogger("VerificationService")

    def Verify(self, request, context):
        job_id = request.job_id if request.job_id else str(uuid.uuid4())

        # In a real system, this would be an async task.
        # Here we run it synchronously for the skeleton.
        try:
            initial_transform = Transform(
                theta=request.initial_transform.theta,
                scale=request.initial_transform.scale,
                tx=request.initial_transform.tx,
                ty=request.initial_transform.ty,
                confidence=request.initial_transform.confidence
            )

            config = {
                'tile_grid_rows': 8,
                'tile_grid_cols': 8,
                'm_min': 5,
                'm_max': 20,
                'remine_budget': 2,
                'image_width': 1000,
                'image_height': 1000,
                'relaxed_threshold': 0.4
            }

            verified_df, report, updated_transform = self.engine.verify(
                job_id=job_id,
                candidate_matches_ref=request.candidate_matches_ref,
                initial_transform=initial_transform,
                pyramid_source_ref=request.pyramid_source_ref,
                pyramid_reference_ref=request.pyramid_reference_ref,
                config=config
            )

            # Save verified matches to "S3" (mock)
            ref = f"s3://mock-bucket/{job_id}/verified.parquet"
            self.mock_client.generated_files[ref] = verified_df

            self.jobs[job_id] = verification_pb2.VerifyResult(
                job_id=job_id,
                status=verification_pb2.VerifyResult.COMPLETED,
                verified_matches_ref=ref,
                coverage_report=verification_pb2.CoverageReport(
                    tile_grid_rows=report.tile_grid_rows,
                    tile_grid_cols=report.tile_grid_cols,
                    per_tile_counts=report.per_tile_counts,
                    under_covered_tiles=report.under_covered_tiles,
                    coverage_fraction=report.coverage_fraction,
                    remine_calls_total=report.remine_calls_total,
                    remine_iterations_used=report.remine_iterations_used
                ),
                updated_transform=verification_pb2.CoarseTransform(
                    theta=updated_transform.theta,
                    scale=updated_transform.scale,
                    tx=updated_transform.tx,
                    ty=updated_transform.ty,
                    confidence=updated_transform.confidence
                )
            )

            return verification_pb2.JobHandle(job_id=job_id)

        except Exception as e:
            self.logger.error(f"Verification failed: {e}")
            self.jobs[job_id] = verification_pb2.VerifyResult(
                job_id=job_id,
                status=verification_pb2.VerifyResult.FAILED,
                error_message=str(e)
            )
            return verification_pb2.JobHandle(job_id=job_id)

    def GetVerifyStatus(self, request, context):
        job_id = request.job_id
        if job_id not in self.jobs:
            context.set_code(grpc.StatusCode.NOT_FOUND)
            context.set_details("Job not found")
            return verification_pb2.VerifyResult()

        return self.jobs[job_id]

def serve():
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    verification_pb2_grpc.add_VerificationServiceServicer_to_server(VerificationService(), server)
    server.add_insecure_port('[::]:50051')
    logging.info("Verification Service starting on port 50051...")
    server.start()
    server.wait_for_termination()

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    serve()
