import pandas as pd
import numpy as np
import uuid
import os

def generate_mock_verified_matches(num_points=100, image_size=(1000, 1000), noise_level=0.5):
    """
    Generates a mock set of verified correspondences.
    In a real scenario, these would be loaded from a parquet file in S3.
    """
    # Generate points in source image (p)
    p_x = np.random.uniform(0, image_size[0], num_points)
    p_y = np.random.uniform(0, image_size[1], num_points)

    # Define a ground truth transform (e.g., slight rotation and scale)
    # Simple affine: q = A*p + t
    angle = np.radians(2.0)  # 2 degrees
    scale = 1.05
    A = np.array([[scale * np.cos(angle), -scale * np.sin(angle)],
                  [scale * np.sin(angle),  scale * np.cos(angle)]])
    t = np.array([10.0, -5.0])

    points_p = np.stack([p_x, p_y], axis=1)
    points_q = (points_p @ A.T) + t

    q_x = points_q[:, 0] + np.random.normal(0, noise_level, num_points)
    q_y = points_q[:, 1] + np.random.normal(0, noise_level, num_points)

    confidence = np.random.uniform(0.7, 1.0, num_points)
    tile_id = np.random.randint(0, 16, num_points) # 4x4 grid

    df = pd.DataFrame({
        'p_x': p_x,
        'p_y': p_y,
        'q_x': q_x,
        'q_y': q_y,
        'confidence': confidence,
        'tile_id': tile_id
    })

    return df

def get_mock_initial_transform():
    """Returns a mock initial transform produced by verification-svc."""
    return {
        "model_type": "affine",
        "params": {
            "matrix": [[1.0, 0.0], [0.0, 1.0]],
            "offset": [0.0, 0.0]
        },
        "rms": 1.2
    }
