import numpy as np
import cv2

def test_sign():
    ref = np.zeros((100, 100), dtype=np.float32)
    ref[40:60, 40:60] = 1.0
    
    src = np.zeros((100, 100), dtype=np.float32)
    # Shift source by +10, -5
    src[40+10:60+10, 40-5:60-5] = 1.0
    
    # OpenCV phaseCorrelate(img1, img2) -> shift of img1 to match img2
    # ref needs to shift +10, -5 to match src
    shift, conf = cv2.phaseCorrelate(ref, src)
    print(f"Shift: {shift}")

if __name__ == "__main__":
    test_sign()
