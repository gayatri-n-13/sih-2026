import numpy as np
from typing import List, Tuple
from models.backbone.interface import Backbone

class MockBackbone(Backbone):
    """
    CI/CD Mock implementation of a Deep Matcher.
    Returns a few synthetic matches to test the tiling and fusion logic.
    """
    def match(self, patch_ref: np.ndarray, patch_src: np.ndarray) -> List[Tuple[Tuple[float, float], Tuple[float, float], float]]:
        # Simulate finding a few matches in the center of the patch
        h, w = patch_ref.shape[-2:]
        matches = [
            ((w // 4, h // 4), (w // 4 + 2, h // 4 + 2), 0.9),
            ((3 * w // 4, 3 * h // 4), (3 * w // 4 - 1, 3 * h // 4 - 1), 0.85),
        ]
        return matches
