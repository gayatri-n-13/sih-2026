import numpy as np
import pytest
from refinement_registration_svc.core.transform import AffineTransform, robust_fit

def test_affine_robust_fit():
    """
    Verifies that the robust fit recovers the true affine parameters
    even in the presence of noise and outliers.
    """
    np.random.seed(42)
    p = np.random.uniform(0, 1000, (1000, 2))

    A = np.array([[1.05, 0.02],
                  [0.01, 0.98]])
    t = np.array([10.0, -5.0])

    q_gt = (p @ A.T) + t
    q_noisy = q_gt + np.random.normal(0, 0.5, q_gt.shape)

    outlier_idx = np.random.choice(1000, 100, replace=False)
    q_noisy[outlier_idx] = np.random.uniform(0, 1000, (100, 2))

    model = AffineTransform()
    params_rec, cost = robust_fit(model, p, q_noisy)

    expected = np.array([1.05, 0.02, 0.01, 0.98, 10.0, -5.0])

    print("\nAffine Transform Recovery:")
    print(f"True Params: {expected}")
    print(f"Recov Params: {params_rec}")
    print(f"Absolute Error: {np.abs(expected - params_rec)}")

    assert params_rec == pytest.approx(expected, abs=0.1), f"Failed to recover affine params, got {params_rec}"

if __name__ == "__main__":
    pytest.main([__file__])
