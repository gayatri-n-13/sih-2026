import numpy as np
import cv2
from typing import Tuple, Optional
from core.geometry.transforms import Transform

class LogPolarBootstrap:
    """
    Implements global coarse alignment using Log-Polar FFT Phase Correlation.
    Estimates rotation, scale, and translation between two images.
    """
    def __init__(self, threshold_psr: float = 3.0):
        self.threshold_psr = threshold_psr

    def compute_magnitude_spectrum(self, image: np.ndarray) -> np.ndarray:
        """Computes the magnitude of the 2D FFT."""
        f = np.fft.fft2(image)
        fshift = np.fft.fftshift(f)
        return np.abs(fshift)

    def _spatial_phase_correlation(self, img1: np.ndarray, img2: np.ndarray) -> Tuple[float, float, float]:
        """
        Computes the translation between two images.
        """
        h, w = img1.shape
        best_tx, best_ty, best_conf = 0.0, 0.0, -1.0

        scales = [4, 2, 1]
        for s in scales:
            if s > 1:
                i1 = cv2.resize(img1, (0, 0), fx=1/s, fy=1/s)
                i2 = cv2.resize(img2, (0, 0), fx=1/s, fy=1/s)
            else:
                i1, i2 = img1, img2

            shift = cv2.phaseCorrelate(i1.astype(np.float32), i2.astype(np.float32))
            tx_s, ty_s = shift[0]
            conf_s = shift[1]

            tx = tx_s * s
            ty = ty_s * s

            if conf_s > best_conf:
                best_tx, best_ty, best_conf = tx, ty, conf_s

        print(f"[DIAG] Spatial Phase Correlation - Raw Winning Shift: tx={best_tx}, ty={best_ty}, conf={best_conf}")
        return best_tx, best_ty, best_conf

    def _log_polar_warp(self, image: np.ndarray, r_min: float = 10, r_max: float = 512) -> np.ndarray:
        """
        Custom Log-Polar warp implementation.
        Maps (rho, theta) -> (x, y).
        """
        h, w = image.shape
        cx, cy = w // 2, h // 2
        rho_axis = np.linspace(np.log(r_min), np.log(r_max), r_max)
        theta_axis = np.linspace(0, 2 * np.pi, 360)
        rho, theta = np.meshgrid(rho_axis, theta_axis)
        x = cx + np.exp(rho) * np.cos(theta)
        y = cy + np.exp(rho) * np.sin(theta)
        map_x = x.astype(np.float32)
        map_y = y.astype(np.float32)
        return cv2.remap(image, map_x, map_y, interpolation=cv2.INTER_LINEAR)

    def estimate_rotation_scale(self, img_ref: np.ndarray, img_src: np.ndarray) -> Tuple[float, float, float]:
        """
        Estimates rotation and scale using log-polar mapping of magnitude spectra.
        Returns (theta, scale, confidence).
        """
        mag_ref = self.compute_magnitude_spectrum(img_ref)
        mag_src = self.compute_magnitude_spectrum(img_src)
        h, w = mag_ref.shape
        cy, cx = h // 2, w // 2
        mask = np.ones((h, w))
        cv2.circle(mask, (cx, cy), 10, 0, -1)
        mag_ref *= mask
        mag_src *= mask
        r_min = 10
        r_max = min(h, w) // 2
        polar_ref = self._log_polar_warp(mag_ref, r_min, r_max)
        polar_src = self._log_polar_warp(mag_src, r_min, r_max)

        # For polar domain, a simple phase correlation is usually enough
        shift = cv2.phaseCorrelate(polar_ref.astype(np.float32), polar_src.astype(np.float32))
        d_rho, d_theta = shift[0]
        confidence = shift[1]

        theta = (d_theta / 360.0) * 2 * np.pi
        theta = (theta + np.pi) % (2 * np.pi) - np.pi
        log_step = (np.log(r_max) - np.log(r_min)) / r_max
        scale = np.exp(-d_rho * log_step)
        return theta, scale, confidence

    def estimate(self, img_ref: np.ndarray, img_src: np.ndarray) -> Transform:
        """Full bootstrap pipeline: Rotation/Scale then Translation."""
        theta, scale, conf_rs = self.estimate_rotation_scale(img_ref, img_src)
        h, w = img_ref.shape
        center = (w // 2, h // 2)

        # To align src to ref, we must apply the INVERSE of the estimated rotation and scale
        theta_inv = -theta
        scale_inv = 1.0 / scale

        M_rot_scale_inv = np.float32([
            [scale_inv * np.cos(theta_inv), -scale_inv * np.sin(theta_inv), 0],
            [scale_inv * np.sin(theta_inv),  scale_inv * np.cos(theta_inv), 0],
            [0, 0, 1]
        ])
        M_center = np.float32([
            [1, 0, -center[0]],
            [0, 1, -center[1]],
            [0, 0, 1]
        ])
        M_uncenter = np.float32([
            [1, 0, center[0]],
            [0, 1, center[1]],
            [0, 0, 1]
        ])
        # M_final removes rotation and scale around the center
        M_final_3x3 = np.dot(M_uncenter, np.dot(M_rot_scale_inv, M_center))
        M_final = M_final_3x3[:2, :]
        warped_src = cv2.warpAffine(img_src, M_final, (w, h))

        # Recover translation in the warped space
        tx_rec, ty_rec, conf_t = self._spatial_phase_correlation(img_ref, warped_src)

        # The recovered shift is in the rotated space: t_rec = R(-theta) * t_orig
        # Rotate it back by theta to get the translation in the original centered frame
        tx = tx_rec * np.cos(theta) - ty_rec * np.sin(theta)
        ty = tx_rec * np.sin(theta) + ty_rec * np.cos(theta)

        final_conf = min(conf_rs, conf_t)
        return Transform(
            theta=theta,
            scale=scale,
            tx=tx,
            ty=ty,
            confidence=final_conf
        )
