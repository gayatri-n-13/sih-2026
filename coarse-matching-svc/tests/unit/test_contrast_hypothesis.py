import numpy as np
import zarr
from tests.synthetic.generator import SyntheticLunarDataGenerator
from core.bootstrap.log_polar import LogPolarBootstrap

def run_test(complexity):
    gen = SyntheticLunarDataGenerator(storage_root=f"test_s3_{complexity}")
    gt = {"theta": 0.1, "scale": 1.05, "tx": 10.0, "ty": -5.0}
    data = gen.create_dataset(f"job_{complexity}", **gt, complexity=complexity)

    ref_pyr = zarr.open(data["pyramid_ref"])
    src_pyr = zarr.open(data["pyramid_source"])
    ref_img = ref_pyr["level_2"][:]
    src_img = src_pyr["level_2"][:]

    bootstrap = LogPolarBootstrap()
    est = bootstrap.estimate(ref_img, src_img)
    
    print(f"Complexity: {complexity} | Recov tx: {est.tx:.2f}, ty: {est.ty:.2f} | Error: dx={abs(est.tx-gt['tx']):.2f}, dy={abs(est.ty-gt['ty']):.2f}")

if __name__ == "__main__":
    print("Testing translation recovery across different complexities...")
    for c in [0.1, 1.0, 5.0]:
        run_test(c)
