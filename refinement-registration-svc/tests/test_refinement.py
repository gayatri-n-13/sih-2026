import numpy as np
import pytest
import cv2
from refinement_registration_svc.core.refinement import refine_point_phase_correlation

def create_synthetic_patch(size=256, seed=42):
    """Creates a patch with high-frequency structural details for robust matching."""
    np.random.seed(seed)
    img = np.random.randn(size, size).astype(np.float64)
    img = cv2.GaussianBlur(img, (5, 5), 1.0)
    cv2.circle(img, (size // 2, size // 2), size // 4, 2.0, -1)
    for _ in range(5):
        r = np.random.randint(2, size // 4)
        cx = np.random.randint(0, size)
        cy = np.random.randint(0, size)
        cv2.circle(img, (cx, cy), r, np.random.uniform(-2, 2), -1)

    img = (img - img.min()) / (img.max() - img.min())
    return img

def shift_image(img, dy, dx):
    """Shifts image by sub-pixel amounts using bicubic interpolation."""
    h, w = img.shape
    M = np.float32([[1, 0, dx], [0, 1, dy]])
    return cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_CUBIC)

@pytest.mark.parametrize("dy, dx", [
    (0.1, 0.1),
    (0.25, -0.1),
    (-0.3, 0.2),
    (0.0, 0.0),
])
def test_subpixel_shift_recovery(dy, dx):
    """
    Hard Gate: Verifies that sub-pixel shifts are recovered within a tight tolerance.
    """
    patch = create_synthetic_patch(256)
    shifted_patch = shift_image(patch, dy, dx)

    rec_dy, rec_dx, conf = refine_point_phase_correlation(patch, shifted_patch)

    print(f"\nInjected Shift: dy={dy:.3f}, dx={dx:.3f}")
    print(f"Recovered Shift: dy={rec_dy:.3f}, dx={rec_dx:.3f}")
    print(f"Error: dy_err={abs(dy-rec_dy):.4f}, dx_err={abs(dx-rec_dx):.4f}")
    print(f"Confidence: {conf:.4f}")

    # Tolerance: 0.05 pixels
    assert rec_dy == pytest.approx(dy, abs=0.05), f"Failed to recover dy {dy}, got {rec_dy}"
    assert rec_dx == pytest.approx(dx, abs=0.05), f"Failed to recover dx {dx}, got {rec_dx}"
    assert conf > 1.0, "Confidence should be significantly positive"

if __name__ == "__main__":
    pytest.main([__file__])
