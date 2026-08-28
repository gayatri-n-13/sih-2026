"""Preprocessing & Illumination-Invariant Representation Microservice.

Consumes RawProduct (raw image + metadata) from ingestion-svc and produces
a multi-scale pyramid plus illumination-invariant channels for downstream
coarse matching.
"""
from preprocessing_svc.config import (
    IngestMetadata,
    PreprocessConfig,
    PreprocessRequest,
    PreprocessResult,
    JobHandle,
    JobStatus,
    SensorType,
)

__all__ = [
    "IngestMetadata",
    "PreprocessConfig",
    "PreprocessRequest",
    "PreprocessResult",
    "JobHandle",
    "JobStatus",
    "SensorType",
]

__version__ = "0.1.0"
