import numpy as np
import zarr
import cv2
from tests.synthetic.generator import SyntheticLunarDataGenerator
from core.bootstrap.log_polar import LogPolarBootstrap
from core.geometry.transforms import Transform

def run_proof():
    # Use a single dot image to isolate translation
    size = 256
    ref_img = np.zeros((size, size), dtype=np.float32)
    ref_img[128, 128] = 1.0
    
    gt = {"theta": 0.0, "scale": 1.0, "tx": 40.0, "ty": -22.0}
    
    # Create src manually to match the generator's logic
    center = (size // 2, size // 2)
    M_affine = np.float32([[1.0 * np.cos(0), -1.0 * np.sin(0), 40.0],
                           [1.0 * np.sin(0),  1.0 * np.cos(0), -22.0],
                           [0, 0, 1]])
    M_center = np.float32([[1, 0, -center[0]], [0, 1, -center[1]], [0, 0, 1]])
    M_uncenter = np.float32([[1, 0, center[0]], [0, 1, center[1]], [0, 0, 1]])
    M_final = np.dot(M_uncenter, np.dot(M_affine, M_center))
    src_img = cv2.warpAffine(ref_img, M_final[:2, :], (size, size))

    bootstrap = LogPolarBootstrap()
    est = bootstrap.estimate(ref_img, src_img)

    print(f"INJECTED: tx={gt['tx']}, ty={gt['ty']}")
    print(f"RECOVERED: tx={est.tx}, ty={est.ty}")

if __name__ == "__main__":
    run_proof()
