import numpy as np
from scipy.optimize import least_squares
from typing import Tuple, Optional, Dict, Any, List

class TransformModel:
    """Base class for geometric transform models."""
    def apply(self, points: np.ndarray, params: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def get_initial_params(self, p: np.ndarray, q: np.ndarray) -> np.ndarray:
        raise NotImplementedError

class SimilarityTransform(TransformModel):
    """Similarity: rotation, scale, translation (4 DOF)."""
    def apply(self, points: np.ndarray, params: np.ndarray) -> np.ndarray:
        # params: [scale, angle, tx, ty]
        s, a, tx, ty = params
        cos_a = np.cos(a)
        sin_a = np.sin(a)

        x = points[:, 0]
        y = points[:, 1]

        qx = s * (cos_a * x - sin_a * y) + tx
        qy = s * (sin_a * x + cos_a * y) + ty
        return np.stack([qx, qy], axis=1)

    def get_initial_params(self, p: np.ndarray, q: np.ndarray) -> np.ndarray:
        # Simple centroid-based initial guess
        cp = np.mean(p, axis=0)
        cq = np.mean(q, axis=0)
        return np.array([1.0, 0.0, cq[0] - cp[0], cq[1] - cp[1]])

class AffineTransform(TransformModel):
    """Affine: 6 DOF."""
    def apply(self, points: np.ndarray, params: np.ndarray) -> np.ndarray:
        # params: [a, b, c, d, tx, ty]
        a, b, c, d, tx, ty = params
        x = points[:, 0]
        y = points[:, 1]

        qx = a * x + b * y + tx
        qy = c * x + d * y + ty
        return np.stack([qx, qy], axis=1)

    def get_initial_params(self, p: np.ndarray, q: np.ndarray) -> np.ndarray:
        # Initial guess as identity + translation
        cp = np.mean(p, axis=0)
        cq = np.mean(q, axis=0)
        return np.array([1.0, 0.0, 0.0, 1.0, cq[0] - cp[0], cq[1] - cp[1]])

class ProjectiveTransform(TransformModel):
    """Projective (Homography): 8 DOF."""
    def apply(self, points: np.ndarray, params: np.ndarray) -> np.ndarray:
        # params: [h00, h01, h02, h10, h11, h12, h20, h21] (h22 fixed to 1)
        h = np.append(params, 1.0).reshape(3, 3)

        # Homogeneous coordinates
        pts_h = np.hstack([points, np.ones((points.shape[0], 1))])
        res_h = pts_h @ h.T

        return res_h[:, :2] / res_h[:, 2:3]

    def get_initial_params(self, p: np.ndarray, q: np.ndarray) -> np.ndarray:
        return np.array([1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0])

def robust_fit(model: TransformModel, p: np.ndarray, q: np.ndarray, weights: Optional[np.ndarray] = None):
    """
    Fits a geometric transform model using iteratively reweighted least squares (IRLS)
    with a Huber loss function for robustness.
    """
    if weights is None:
        weights = np.ones(len(p))

    params_init = model.get_initial_params(p, q)

    def objective(params):
        pred = model.apply(p, params)
        residuals = q - pred
        # We return the residuals for least_squares to handle the loss
        return residuals.flatten()

    # Use robust loss 'huber' in least_squares
    res = least_squares(objective, params_init, loss='huber', f_scale=0.1)

    return res.x, res.cost

def select_transform_model(config: Dict[str, Any], residuals: np.ndarray) -> TransformModel:
    """
    Logic to choose between Similarity, Affine, and Projective based on
    config override or residual patterns.
    """
    model_type = config.get("transform_model", "auto")

    if model_type == "similarity":
        return SimilarityTransform()
    elif model_type == "affine":
        return AffineTransform()
    elif model_type == "projective":
        return ProjectiveTransform()

    # 'auto' heuristic:
    # If residuals show a systematic spatial trend, use a higher-order model.
    # For now, default to Affine as a safe middle ground.
    return AffineTransform()
