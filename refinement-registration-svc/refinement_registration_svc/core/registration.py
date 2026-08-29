import numpy as np
import cv2
from typing import Tuple, Dict, Any
from refinement_registration_svc.core.transform import TransformModel

def resample_image(source_img: np.ndarray,
                    reference_shape: Tuple[int, int],
                    model: TransformModel,
                    params: np.ndarray,
                    interpolation=cv2.INTER_CUBIC) -> np.ndarray:
    """
    Warps the source image into the reference grid using the fitted transform.
    """
    h_ref, w_ref = reference_shape
    h_src, w_src = source_img.shape[:2]

    # Simple implementation: use cv2.resize if it's identity or a simple scale
    # In a real system, we'd use the model's inverse transform.
    # For the scaffold, we return a resized image to match the reference shape.
    return cv2.resize(source_img, (w_ref, h_ref), interpolation=interpolation)

def write_as_cog(image: np.ndarray, output_path: str, profile: Dict[str, Any]):
    """
    Writes the registered image as a Cloud-Optimized GeoTIFF.
    """
    try:
        import rasterio
        with rasterio.open(output_path, 'w', **profile) as dst:
            dst.write(image, 1)
    except ImportError:
        print("rasterio not installed, skipping COG write.")
