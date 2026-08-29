import numpy as np
import zarr
import cv2
from tests.synthetic.generator import SyntheticLunarDataGenerator
from core.bootstrap.log_polar import LogPolarBootstrap
from core.geometry.transforms import Transform

def run_diag():
    gen = SyntheticLunarDataGenerator(storage_root="diag_v2_s3")
    # Case: tx=10, ty=-5
    gt = {"theta": 0.1, "scale": 1.05, "tx": 10.0, "ty": -5.0}
    data = gen.create_dataset("diag_job", **gt)

    ref_pyr = zarr.open(data["pyramid_ref"])
    src_pyr = zarr.open(data["pyramid_source"])
    ref_img = ref_pyr["level_2"][:]
    src_img = src_pyr["level_2"][:]

    bootstrap = LogPolarBootstrap()
    est = bootstrap.estimate(ref_img, src_img)

    print(f"INJECTED: tx={gt['tx']}, ty={gt['ty']}")
    print(f"RECOVERED: tx={est.tx}, ty={est.ty}")
    print(f"ERRORS: dx={est.tx - gt['tx']}, dy={est.ty - gt['ty']}")
    
    # Let's look at the translation if we use a high-feature image
    print("\n--- Testing with high-complexity image ---")
    data_high = gen.create_dataset("diag_high", **gt, complexity=5.0)
    ref_high = zarr.open(data_high["pyramid_ref"])["level_2"][:]
    src_high = zarr.open(data_high["pyramid_source"])["level_2"][:]
    est_high = bootstrap.estimate(ref_high, src_high)
    print(f"RECOVERED (High): tx={est_high.tx}, ty={est_high.ty}")
    print(f"ERRORS (High): dx={est_high.tx - gt['tx']}, dy={est_high.ty - gt['ty']}")

if __name__ == "__main__":
    run_diag()
