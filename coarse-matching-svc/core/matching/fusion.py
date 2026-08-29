from typing import List
import numpy as np
from models.data_models import Correspondence

class MatchFusion:
    """
    Fuses matches from multiple sources (classical and deep)
    and normalizes confidence scores.
    """
    def __init__(self, distance_threshold: float = 5.0):
        self.distance_threshold = distance_threshold

    def fuse(self, classical_matches: List[Correspondence], deep_matches: List[Correspondence]) -> List[Correspondence]:
        """
        Merges matches, deduplicates, and normalizes.
        """
        all_matches = classical_matches + deep_matches
        if not all_matches:
            return []

        # Simple deduplication: if two matches are very close in ref image, keep the most confident one
        fused = []
        # Sort by confidence descending
        sorted_matches = sorted(all_matches, key=lambda x: x.confidence, reverse=True)

        while sorted_matches:
            best = sorted_matches.pop(0)
            fused.append(best)

            # Remove others that are too close to 'best'
            sorted_matches = [
                m for m in sorted_matches
                if np.linalg.norm(np.array(m.pt_ref) - np.array(best.pt_ref)) > self.distance_threshold
            ]

        return fused
