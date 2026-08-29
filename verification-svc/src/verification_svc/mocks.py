import numpy as np
import pandas as pd
import uuid
import random
from typing import Dict, List, Tuple

class CoarseMatchingMock:
    """
    Mocks the CoarseMatchingService API and the data it produces.
    """
    def __init__(self, storage_root: str = "s3://mock-bucket/"):
        self.storage_root = storage_root
        self.generated_files: Dict[str, pd.DataFrame] = {}

    def generate_initial_candidates(self, num_matches: int = 1000, inlier_ratio: float = 0.3):
        """
        Generates a synthetic set of candidate matches.
        """
        job_id = str(uuid.uuid4())

        # Create synthetic ground truth transform (simple affine)
        # For simplicity: scale=1.0, theta=0, tx=10, ty=20
        tx, ty = 10.0, 20.0

        # Generate p points (source image)
        p_x = np.random.uniform(0, 1000, num_matches)
        p_y = np.random.uniform(0, 1000, num_matches)

        q_x = np.zeros(num_matches)
        q_y = np.zeros(num_matches)

        mask = np.random.random(num_matches) < inlier_ratio

        # Inliers: q = p + transform + noise
        q_x[mask] = p_x[mask] + tx + np.random.normal(0, 0.5, np.sum(mask))
        q_y[mask] = p_y[mask] + ty + np.random.normal(0, 0.5, np.sum(mask))

        # Outliers: completely random
        q_x[~mask] = np.random.uniform(0, 1000, np.sum(~mask))
        q_y[~mask] = np.random.uniform(0, 1000, np.sum(~mask))

        df = pd.DataFrame({
            'p_x': p_x,
            'p_y': p_y,
            'q_x': q_x,
            'q_y': q_y,
            'source_method': ['classical' if i % 2 == 0 else 'deep' for i in range(num_matches)],
            'confidence': np.random.uniform(0.5, 1.0, num_matches),
            'pyramid_level': np.random.randint(0, 3, num_matches)
        })

        ref = f"{self.storage_root}{job_id}/candidates.parquet"
        self.generated_files[ref] = df
        return ref, job_id

    def remine_tile(self, job_id: str, tile_id: int, tile_bounds: List[float], relaxed_threshold: float):
        """
        Mocks the RemineTile RPC call.
        Returns a few high-confidence inliers for the given tile.
        """
        # In a real scenario, this would look at the actual images.
        # Here, we just generate 5-10 synthetic inliers for the requested tile area.
        xmin, ymin, xmax, ymax = tile_bounds
        num_new = random.randint(5, 15)

        p_x = np.random.uniform(xmin, xmax, num_new)
        p_y = np.random.uniform(ymin, ymax, num_new)

        # Using the same synthetic GT: tx=10, ty=20
        q_x = p_x + 10.0 + np.random.normal(0, 0.2, num_new)
        q_y = p_y + 20.0 + np.random.normal(0, 0.2, num_new)

        df = pd.DataFrame({
            'p_x': p_x,
            'p_y': p_y,
            'q_x': q_x,
            'q_y': q_y,
            'source_method': ['deep'] * num_new,
            'confidence': np.random.uniform(relaxed_threshold, 1.0, num_new),
            'pyramid_level': [0] * num_new
        })

        ref = f"{self.storage_root}{job_id}/remine_{tile_id}.parquet"
        self.generated_files[ref] = df
        return ref

    def read_parquet(self, ref: str) -> pd.DataFrame:
        """
        Simulates reading from S3.
        """
        if ref not in self.generated_files:
            raise FileNotFoundError(f"Reference {ref} not found in mock storage")
        return self.generated_files[ref]
