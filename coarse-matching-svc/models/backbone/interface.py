from abc import ABC, abstractmethod
import numpy as np
from typing import List, Tuple

class Backbone(ABC):
    """Abstract interface for Deep Learning feature extractors."""
    @abstractmethod
    def match(self, patch_ref: np.ndarray, patch_src: np.ndarray) -> List[Tuple[Tuple[float, float], Tuple[float, float], float]]:
        """
        Matches a reference patch against a source patch.
        Returns: List of (point_ref, point_src, confidence)
        """
        pass
