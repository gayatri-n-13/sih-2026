import numpy as np
import zarr
import os
import cv2
from typing import Tuple, Dict

class SyntheticLunarDataGenerator:
    """
    Generates synthetic lunar image pairs with a known ground-truth transform.
    Saves them as Zarr arrays to simulate the output of the Preprocessing service.
    """
    def __init__(self, storage_root: str = "s3_mock"):
        self.storage_root = storage_root
        os.makedirs(storage_root, exist_ok=True)

    def generate_base_image(self, size: int = 1024) -> np.ndarray:
        """Creates a synthetic lunar-like image with craters and boulders."""
        image = np.zeros((size, size), dtype=np.float32)
        # Simulate craters (Gaussian blobs with edges)
        for _ in range(20):
            cx, cy = np.random.randint(0, size, 2)
            r = np.random.randint(5, 50)
            y, x = np.ogrid[:size, :size]
            dist = np.sqrt((x - cx)**2 + (y - cy)**2)
            # Crater rim
            image += np.exp(- (dist - r)**2 / 2)
            # Crater floor
            image += 0.5 * np.exp(- dist**2 / (2 * r**2))

        # Add some random noise
        image += np.random.normal(0, 0.05, (size, size))
        return (image - image.min()) / (image.max() - image.min())

    def apply_transform(self, image: np.ndarray, theta: float, scale: float, tx: float, ty: float) -> np.ndarray:
        """Applies rotation, scale, and translation."""
        h, w = image.shape
        center = (w // 2, h // 2)

        M_rot_scale = np.float32([[scale * np.cos(theta), -scale * np.sin(theta), 0],
                                 [scale * np.sin(theta),  scale * np.cos(theta), 0],
                                 [0, 0, 1]])

        M_trans = np.float32([[1, 0, tx],
                             [0, 1, ty],
                             [0, 0, 1]])

        M_affine = np.dot(M_trans, M_rot_scale)

        M_center = np.float32([[1, 0, -center[0]],
                               [0, 1, -center[1]],
                               [0, 0, 1]])
        M_uncenter = np.float32([[1, 0, center[0]],
                                 [0, 1, center[1]],
                                 [0, 0, 1]])

        M_final_3x3 = np.dot(M_uncenter, np.dot(M_affine, M_center))
        M_final = M_final_3x3[:2, :]

        return cv2.warpAffine(image, M_final, (w, h), flags=cv2.INTER_LINEAR)

    def create_dataset(self, job_id: str, theta: float, scale: float, tx: float, ty: float):
        """Creates the Zarr pyramids and invariant channels for source and reference."""
        ref_img = self.generate_base_image()
        src_img = self.apply_transform(ref_img, theta, scale, tx, ty)

        import cv2
        def make_pyramid(img):
            pyramid = []
            curr = img
            for _ in range(3):
                pyramid.append(curr)
                curr = cv2.pyrDown(curr)
            return pyramid

        ref_pyramid = make_pyramid(ref_img)
        src_pyramid = make_pyramid(src_img)

        def make_channels(pyramid):
            channels = {}
            for i in range(3):
                channels[f"level_{i}_phase_congruency"] = pyramid[i] * 0.8
                channels[f"level_{i}_sdn_relief"] = pyramid[i] * 1.2
                channels[f"level_{i}_gradient_orientation"] = np.roll(pyramid[i], 1, axis=0)
            return channels

        ref_channels = make_channels(ref_pyramid)
        src_channels = make_channels(src_pyramid)

        ref_pyramid_zarr = zarr.open(f"{self.storage_root}/{job_id}/ref_pyramid.zarr", mode='w')
        for i, lvl in enumerate(ref_pyramid):
            ref_pyramid_zarr[f"level_{i}"] = lvl

        src_pyramid_zarr = zarr.open(f"{self.storage_root}/{job_id}/src_pyramid.zarr", mode='w')
        for i, lvl in enumerate(src_pyramid):
            src_pyramid_zarr[f"level_{i}"] = lvl

        ref_chan_zarr = zarr.open(f"{self.storage_root}/{job_id}/ref_channels.zarr", mode='w')
        for k, v in ref_channels.items():
            ref_chan_zarr[k] = v

        src_chan_zarr = zarr.open(f"{self.storage_root}/{job_id}/src_channels.zarr", mode='w')
        for k, v in src_channels.items():
            src_chan_zarr[k] = v

        return {
            "pyramid_ref": f"{self.storage_root}/{job_id}/ref_pyramid.zarr",
            "pyramid_source": f"{self.storage_root}/{job_id}/src_pyramid.zarr",
            "invariant_channels_ref": f"{self.storage_root}/{job_id}/ref_channels.zarr",
            "invariant_channels_source": f"{self.storage_root}/{job_id}/src_channels.zarr",
            "gt": {"theta": theta, "scale": scale, "tx": tx, "ty": ty}
        }

if __name__ == "__main__":
    gen = SyntheticLunarDataGenerator()
    data = gen.create_dataset("test_job_1", theta=0.1, scale=1.05, tx=10, ty=-5)
    print(f"Generated dataset at: {data['pyramid_ref']}")
