import numpy as np
import pytest
from tests.synthetic.generator import SyntheticLunarDataGenerator
from core.bootstrap.log_polar import LogPolarBootstrap
from core.geometry.transforms import Transform

def test_bootstrap_recovery():
    # 1. Setup
    gen = SyntheticLunarDataGenerator(storage_root="test_s3_bootstrap")
    # Use a small rotation and scale for the bootstrap
    gt_theta = 0.1  # ~5.7 degrees
    gt_scale = 1.05
    gt_tx = 10.0
    gt_ty = -5.0

    # Create synthetic data
    data = gen.create_dataset("bootstrap_test", theta=gt_theta, scale=gt_scale, tx=gt_tx, ty=gt_ty)

    # Load the coarsest pyramid level (level 2)
    import zarr
    ref_pyr = zarr.open(data["pyramid_ref"])
    src_pyr = zarr.open(data["pyramid_source"])

    ref_img = ref_pyr["level_2"][:]
    src_img = src_pyr["level_2"][:]

    # 2. Execute
    bootstrap = LogPolarBootstrap()
    est = bootstrap.estimate(ref_img, src_img)

    # 3. Verify
    # Note: Log-polar bootstrap is a 'coarse' estimate.
    # Tolerances are relaxed but should be in the ballpark.
    # Since we used level_2 (256x256), the recovered tx/ty are 1/4 of the 1024x1024 GT
    scale_factor = 1024 / 256
    gt_tx_scaled = gt_tx / scale_factor
    gt_ty_scaled = gt_ty / scale_factor

    print(f"GT (scaled): theta={gt_theta}, scale={gt_scale}, tx={gt_tx_scaled}, ty={gt_ty_scaled}")
    print(f"Est: theta={est.theta}, scale={est.scale}, tx={est.tx}, ty={est.ty}")

    assert abs(est.theta - gt_theta) < 0.1  # ~5 degrees
    assert abs(est.scale - gt_scale) < 0.05
    assert abs(est.tx - gt_tx_scaled) < 5.0
    assert abs(est.ty - gt_ty_scaled) < 5.0

if __name__ == "__main__":
    test_bootstrap_recovery()
