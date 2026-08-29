from pydantic import BaseModel
from typing import Optional

class MatchRequestModel(BaseModel):
    job_id: str
    pyramid_source_ref: str
    pyramid_reference_ref: str
    invariant_channels_source_ref: str
    invariant_channels_reference_ref: str
    config_ref: Optional[str] = None

class RemineRequestModel(BaseModel):
    job_id: str
    tile_id: int
    tile_bounds: str  # "x0,y0,x1,y1"
    relaxed_confidence_threshold: float
