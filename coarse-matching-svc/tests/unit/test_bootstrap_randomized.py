import numpy as np
import pytest
import zarr
from tests.synthetic.generator import SyntheticLunarDataGenerator
from core.bootstrap.log_polar import LogPolarBootstrap
from core.geometry.transforms import Transform

def run_single_recovery_test(case_id, theta, scale, tx, ty):
    print(f"\n--- Case {case_id}: theta={theta:.4f}, scale={scale:.4f}, tx={tx:.2f}, ty={ty:.2f} ---")

    gen = SyntheticLunarDataGenerator(storage_root=f"test_s3_rand_{case_id}")
    data = gen.create_dataset(f"case_{case_id}", theta=theta, scale=scale, tx=tx, ty=ty)

    ref_pyr = zarr.open(data["pyramid_ref"])
    src_pyr = zarr.open(data["pyramid_source"])

    # Use level 2 (256x256) as in the original test
    ref_img = ref_pyr["level_2"][:]
    src_img = src_pyr["level_2"][:]

    bootstrap = LogPolarBootstrap()
    est = bootstrap.estimate(ref_img, src_img)

    # Scale factor for level 2 (1024 -> 256)
    scale_factor = 1024 / 256
    gt_tx_scaled = tx / scale_factor
    gt_ty_scaled = ty / scale_factor

    print(f"GT (scaled): tx={gt_tx_scaled:.4f}, ty={gt_ty_scaled:.4f}")
    print(f"Est: theta={est.theta:.4f}, scale={est.scale:.4f}, tx={est.tx:.4f}, ty={est.ty:.4f}")

    err_tx = abs(est.tx - gt_tx_scaled)
    err_ty = abs(est.ty - gt_ty_scaled)
    print(f"Errors: tx_err={err_tx:.4f}, ty_err={err_ty:.4f}")

    assert abs(est.theta - theta) < 0.1
    assert abs(est.scale - scale) < 0.05
    assert err_tx < 5.0
    assert err_ty < 5.0
    print("Result: PASSED")

def test_bootstrap_randomized():
    # Original hard case
    run_single_recovery_test("ORIGINAL", 0.1, 1.05, 10.0, -5.0)

    # Randomized cases
    # We keep theta and scale small as bootstrap is for coarse alignment
    cases = [
        (0.05, 0.98, -15.0, 20.0),
        (-0.1, 1.02, 30.0, -30.0),
        (0.15, 1.08, -5.0, -10.0),
        (0.0, 1.0, 40.0, 40.0),
    ]

    for i, (theta, scale, tx, ty) in enumerate(cases):
        run_single_recovery_test(i+1, theta, scale, tx, ty)

if __name__ == "__main__":
    test_bootstrap_randomized()
