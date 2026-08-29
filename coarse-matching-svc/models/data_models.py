from dataclasses import dataclass
from typing import Tuple

@dataclass
class Transform:
    theta: float  # Radians
    scale: float  # Scaling factor
    tx: float     # Translation X
    ty: float     # Translation Y
    confidence: float = 0.0

@dataclass
class Correspondence:
    pt_ref: Tuple[float, float]
    pt_src: Tuple[float, float]
    confidence: float
    source: str  # "classical" or "deep"
    pyramid_level: int = 0
