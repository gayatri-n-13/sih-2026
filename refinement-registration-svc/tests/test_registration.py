import numpy as np
import pytest
import cv2
from unittest.mock import MagicMock
import sys

# Mock rasterio since it's hard to install on some environments
mock_rasterio = MagicMock()
sys.modules["rasterio"] = mock_rasterio

from refinement_registration_svc.core.registration import resample_image
from refinement_registration_svc.core.transform import AffineTransform

def test_resampling_correctness():
    """
    Verifies that the resampling produces a correctly shaped output
    and preserves the reference grid.
    """
    source_img = np.random.rand(100, 100).astype(np.float32)
    ref_shape = (120, 120)

    model = AffineTransform()
    params = np.array([1.0, 0.0, 0.0, 1.0, 0.0, 0.0])

    registered = resample_image(source_img, ref_shape, model, params)

    print(f"\nResampling Check:")
    print(f"Source shape: {source_img.shape}")
    print(f"Reference shape: {ref_shape}")
    print(f"Output shape: {registered.shape}")
    print(f"Output dtype: {registered.dtype}")

    assert registered.shape == ref_shape, f"Output shape {registered.shape} does not match reference {ref_shape}"
    assert registered.dtype == source_img.dtype, "Output dtype mismatch"

if __name__ == "__main__":
    pytest.main([__file__])
