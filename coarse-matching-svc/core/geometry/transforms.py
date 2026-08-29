from dataclasses import dataclass

@dataclass
class Transform:
    theta: float  # Radians
    scale: float  # Scale factor
    tx: float     # Translation X
    ty: float     # Translation Y
    confidence: float = 0.0

    def to_dict(self):
        return {
            "theta": self.theta,
            "scale": self.scale,
            "tx": self.tx,
            "ty": self.ty,
            "confidence": self.confidence
        }
