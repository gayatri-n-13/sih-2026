import numpy as np
import cv2
from typing import List, Tuple
from core.matching.base import Matcher
from models.data_models import Correspondence, Transform

class ClassicalMatcher(Matcher):
    """
    Classical modality-robust matcher using structural descriptors
    on invariant channels.
    """
    def __init__(self, confidence_threshold: float = 0.5):
        self.confidence_threshold = confidence_threshold

    def _extract_interest_points(self, image: np.ndarray) -> List[Tuple[int, int]]:
        """Finds peaks in the invariant channel (e.g., Phase Congruency)."""
        # Convert to uint8 for OpenCV features
        img_u8 = (image * 255).astype(np.uint8)
        pts = cv2.goodFeaturesToTrack(img_u8,
                                       maxCorners=1000,
                                       qualityLevel=0.001, # Lower threshold
                                       minDistance=10)
        if pts is not None:
            return [tuple(p.ravel()) for p in pts]
        return []

    def _get_descriptor(self, image: np.ndarray, pt: Tuple[float, float], window_size: int = 15) -> np.ndarray:
        """Extracts a local structural descriptor around a point."""
        x, y = map(int, pt)
        h, w = image.shape

        # Extract window
        x0, x1 = max(0, x - window_size // 2), min(w, x + window_size // 2)
        y0, y1 = max(0, y - window_size // 2), min(h, y + window_size // 2)

        window = image[y0:y1, x0:x1]

        # Normalize window to create a scale/intensity invariant descriptor
        if window.size == 0:
            return np.zeros(window_size * window_size)

        # Pad if necessary
        if window.shape != (window_size, window_size):
            pad_h = window_size - window.shape[0]
            pad_w = window_size - window.shape[1]
            window = np.pad(window, ((0, pad_h), (0, pad_w)), mode='constant')

        return window.flatten()

    def match(self, img_ref: np.ndarray, img_src: np.ndarray,
              initial_transform: Transform) -> List[Correspondence]:
        """
        Matches features between ref and src.
        """
        # 1. Warp src image to ref using initial transform to bound search
        h, w = img_ref.shape
        center = (w // 2, h // 2)
        M_rot_scale = np.float32([
            [initial_transform.scale * np.cos(initial_transform.theta), -initial_transform.scale * np.sin(initial_transform.theta), 0],
            [initial_transform.scale * np.sin(initial_transform.theta),  initial_transform.scale * np.cos(initial_transform.theta), 0],
            [0, 0, 1]
        ])
        M_center = np.float32([[1, 0, -center[0]], [0, 1, -center[1]], [0, 0, 1]])
        M_uncenter = np.float32([[1, 0, center[0]], [0, 1, center[1]], [0, 0, 1]])
        M_final = np.dot(M_uncenter, np.dot(M_rot_scale, M_center))
        warped_src = cv2.warpAffine(img_src, M_final[:2, :], (w, h))

        # 2. Interest points
        pts_ref = self._extract_interest_points(img_ref)
        pts_src_warped = self._extract_interest_points(warped_src)

        if not pts_ref or not pts_src_warped:
            return []

        # 3. Descriptors
        desc_ref = np.array([self._get_descriptor(img_ref, p) for p in pts_ref])
        desc_src = np.array([self._get_descriptor(warped_src, p) for p in pts_src_warped])

        # 4. Nearest Neighbor Matching
        matches = []
        for i, d_ref in enumerate(desc_ref):
            dists = np.linalg.norm(desc_src - d_ref, axis=1)
            idx_min = np.argmin(dists)

            # Ratio test (Lowe's ratio)
            sorted_dists = np.sort(dists)
            if len(sorted_dists) > 1 and sorted_dists[0] < 0.8 * sorted_dists[1]:
                conf = 1.0 - (sorted_dists[0] / (sorted_dists[0] + 1e-6))
                if conf > self.confidence_threshold:
                    matches.append(Correspondence(
                        pt_ref=pts_ref[i],
                        pt_src=pts_src_warped[idx_min],
                        confidence=conf,
                        source="classical"
                    ))

        return matches
