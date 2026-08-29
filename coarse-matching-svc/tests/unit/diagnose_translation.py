import numpy as np
import zarr
from tests.synthetic.generator import SyntheticLunarDataGenerator
from core.bootstrap.log_polar import LogPolarBootstrap
from core.geometry.transforms import Transform

def run_diag():
    gen = SyntheticLunarDataGenerator(storage_root="diag_s3")
    # Use a simple transform
    gt = {"theta": 0.0, "scale": 1.0, "tx": 50.0, "ty": -30.0}
    data = gen.create_dataset("diag_job", **gt)

    ref_pyr = zarr.open(data["pyramid_ref"])
    src_pyr = zarr.open(data["pyramid_source"])
    ref_img = ref_pyr["level_2"][:]
    src_img = src_pyr["level_2"][:]

    h, w = ref_img.shape
    print(f"Image size: {w}x{h}, Center: ({w//2}, {h//2})")

    bootstrap = LogPolarBootstrap()
    est = bootstrap.estimate(ref_img, src_img)

    print(f"Injected: tx={gt['tx']}, ty={gt['ty']}")
    print(f"Recovered: tx={est.tx}, ty={est.ty}")
    print(f"Error: dx={est.tx - gt['tx']}, dy={est.ty - gt['ty']}")
    print(f"Distance to center: dx_to_center={abs(est.tx - w//2)}, dy_to_center={abs(est.ty - h//2)}")

if __name__ == "__main__":
    run_diag()
