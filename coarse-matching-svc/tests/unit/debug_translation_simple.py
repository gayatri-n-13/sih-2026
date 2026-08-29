import numpy as np
import cv2
from core.bootstrap.log_polar import LogPolarBootstrap
from core.geometry.transforms import Transform

def run_debug():
    # Simple 256x256 images
    ref = np.zeros((256, 256), dtype=np.float32)
    ref[128, 128] = 1.0  # Single pixel at center
    
    # Source is just translated
    tx, ty = 20.0, -10.0
    src = np.zeros((256, 256), dtype=np.float32)
    src[128 + int(ty), 128 + int(tx)] = 1.0 # Remember: src[y, x]
    
    bootstrap = LogPolarBootstrap()
    # we skip rotation/scale for now to isolate translation
    # simulate warped_src as just the shifted src
    dx, dy, conf = bootstrap.phase_correlation(ref, src)
    print(f"Injected: tx={tx}, ty={ty}")
    print(f"Recovered: dx={dx}, dy={dy}")

if __name__ == "__main__":
    run_debug()
