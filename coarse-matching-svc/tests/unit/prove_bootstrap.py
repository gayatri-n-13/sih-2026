import numpy as np
import zarr
from tests.synthetic.generator import SyntheticLunarDataGenerator
from core.bootstrap.log_polar import LogPolarBootstrap
from core.geometry.transforms import Transform

def run_proof():
    gen = SyntheticLunarDataGenerator(storage_root="proof_s3")
    # Define a non-trivial transform
    gt = {"theta": 0.2618, "scale": 1.15, "tx": 40.0, "ty": -22.0} # theta = 15 deg
    data = gen.create_dataset("proof_job", **gt)

    ref_pyr = zarr.open(data["pyramid_ref"])
    src_pyr = zarr.open(data["pyramid_source"])
    ref_img = ref_pyr["level_2"][:]
    src_img = src_pyr["level_2"][:]

    bootstrap = LogPolarBootstrap()
    est = bootstrap.estimate(ref_img, src_img)

    print(f"INJECTED: theta={gt['theta']}, scale={gt['scale']}, tx={gt['tx']}, ty={gt['ty']}")
    print(f"RECOVERED: theta={est.theta}, scale={est.scale}, tx={est.tx}, ty={est.ty}")
    print(f"CONFIDENCE: {est.confidence}")

if __name__ == "__main__":
    run_proof()
