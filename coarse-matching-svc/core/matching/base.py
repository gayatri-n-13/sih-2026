from abc import ABC, abstractmethod
from typing import List, Tuple
import numpy as np
from models.data_models import Correspondence, Transform

class Matcher(ABC):
    """Abstract base for all matching strategies."""
    @abstractmethod
    def match(self, img_ref: np.ndarray, img_src: np.ndarray,
              initial_transform: Transform) -> List[Correspondence]:
        """
        Returns a list of candidate correspondences.
        """
        pass
