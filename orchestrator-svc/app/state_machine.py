from enum import Enum
from typing import List, Optional, Dict, Any
import logging

logger = logging.getLogger(__name__)

class JobStatus(Enum):
    PENDING = "PENDING"
    INGESTING = "INGESTING"
    PREPROCESSING = "PREPROCESSING"
    COARSE_MATCHING = "COARSE_MATCHING"
    VERIFYING = "VERIFYING"
    REGISTERING = "REGISTERING"
    EVALUATING = "EVALUATING"
    DONE = "DONE"
    FAILED = "FAILED"

class StageRecord:
    def __init__(self, stage: JobStatus, status: str, start_time: float, end_time: Optional[float] = None, metrics: Optional[Dict[str, Any]] = None):
        self.stage = stage
        self.status = status
        self.start_time = start_time
        self.end_time = end_time
        self.metrics = metrics or {}

    def to_dict(self):
        return {
            "stage": self.stage.value,
            "status": self.status,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "metrics": self.metrics
        }

class RegistrationJob:
    def __init__(self, job_id: str):
        self.job_id = job_id
        self.status = JobStatus.PENDING
        self.current_stage = JobStatus.PENDING
        self.stage_history: List[StageRecord] = []
        self.context: Dict[str, Any] = {} # Store references to S3 objects, etc.
        self.error_message: Optional[str] = None

    def transition_to(self, stage: JobStatus):
        logger.info(f"Job {self.job_id} transitioning from {self.current_stage} to {stage}")
        self.current_stage = stage
        if stage != JobStatus.DONE and stage != JobStatus.FAILED:
            self.status = JobStatus.PENDING # Mark as pending for the current stage
        elif stage == JobStatus.DONE:
            self.status = JobStatus.DONE
        elif stage == JobStatus.FAILED:
            self.status = JobStatus.FAILED

    def fail(self, error: str):
        logger.error(f"Job {self.job_id} failed at stage {self.current_stage}: {error}")
        self.error_message = error
        self.transition_to(JobStatus.FAILED)
