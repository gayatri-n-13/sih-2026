"""Illumination-invariant channels.

This is the scientific heart of the service. The Moon has no atmosphere,
vegetation, or seasons — the dominant source of appearance variation
between two images of the same terrain is the SUN ILLUMINATION ANGLE.
Shadow direction and length change dramatically even when the 3D
terrain is identical.

We produce three invariant channels:

1. Phase congruency (Kovesi-style): a contrast- and shading-independent
   structural significance map computed from a multi-orientation
   Log-Gabor filter bank. This is our PRIMARY invariant channel.

2. SDN-Relief: a shadow-direction-normalized relief channel. We rotate
   the gradient field into a Sun-relative frame and compute a bounded
   relief-emphasis map. If the Sun direction is unavailable, we estimate
   it from the image itself (see sun_estimator).

3. Dense gradient-orientation field (structure tensor): a smooth
   orientation map that captures local structure independent of absolute
   intensity.
"""
from __future__ import annotations

import numpy as np

from preprocessing_svc.sun_estimator import estimate_sun_azimuth


# ----------------------------------------------------------------------
# 1. Phase congruency
# ----------------------------------------------------------------------


def _log_gabor_filter(
    image: np.ndarray,
    wavelength: float,
    orientation_deg: float,
    sigma_on_f: float = 0.55,
) -> tuple[np.ndarray, np.ndarray]:
    """Return even and odd Log-Gabor responses at one scale/orientation.

    Uses a small FFT-based implementation. Image is 2-D float32.
    """
    Y, X = image.shape
    # Build a Log-Gabor filter in the frequency domain.
    yv, xv = np.meshgrid(
        np.arange(-Y // 2, Y // 2),
        np.arange(-X // 2, X // 2),
        indexing="ij",
    )
    # Rotate to filter orientation.
    theta = np.deg2rad(orientation_deg)
    # Pixel-domain rotations: in image frame, columns are X, rows are Y.
    # We rotate (xv, yv) by -theta so the filter is oriented along theta.
    xr = xv * np.cos(theta) + yv * np.sin(theta)
    yr = -xv * np.sin(theta) + yv * np.cos(theta)
    radius = np.sqrt(xr * xr + yr * yr) + 1e-6
    # Log-Gabor radial profile: exp(- (log(radius / (1/wavelength)))^2 / (2 sigma^2))
    radial = np.exp(
        -0.5
        * (np.log(radius * wavelength) ** 2)
        / (sigma_on_f ** 2)
    )
    radial[0, 0] = 0.0  # DC zero
    # Angular Gaussian around the filter orientation.
    eps = 1e-6
    angular = np.exp(-0.5 * (yr / (radius * np.sin(0.5 * np.pi / 6.0) + eps)) ** 2)
    # Combined filter.
    filt = radial * angular
    # FFT
    F = np.fft.fftshift(np.fft.fft2(image))
    # Even and odd parts via Hilbert-like construction in Fourier space.
    # We use a quadrature pair: the same filter shifted by pi/2 in the
    # angular dimension gives the odd (90 deg phase-shifted) response.
    # For a Log-Gabor on real images, the analytic signal is built by
    # zeroing negative frequencies.
    # Build even/odd quadrature pair in Fourier domain.
    # Construct analytic signal by suppressing half-plane negative freqs.
    H = np.zeros_like(F)
    # Sign of xr determines half-plane.
    half = (xr >= 0).astype(np.float32)
    H_even = F * filt * half
    H_odd = F * filt * half * 1j  # +90 deg
    # iFFT to spatial domain; take real/imag as even/odd responses.
    even = np.real(np.fft.ifft2(np.fft.ifftshift(H_even)))
    odd = np.real(np.fft.ifft2(np.fft.ifftshift(H_odd)))
    return even.astype(np.float32), odd.astype(np.float32)


def phase_congruency(
    image: np.ndarray,
    n_scales: int = 4,
    n_orientations: int = 6,
    min_wavelength: float = 3.0,
    scaling_factor: float = 2.0,
    sigma_on_f: float = 0.55,
    noise_threshold: float = 1.5,
) -> np.ndarray:
    """Kovesi-style phase congruency.

    Parameters
    ----------
    image : (Y, X) float32
        The input grayscale image, expected to be in [0, 1] (the result is
        largely insensitive to overall scale because PC normalizes by
        local energy).
    n_scales, n_orientations : int
        Size of the Log-Gabor bank.
    min_wavelength : float
        Wavelength of the smallest-scale filter, in pixels.
    scaling_factor : float
        Multiplicative factor between successive scale wavelengths.
    sigma_on_f : float
        Width of the Log-Gabor radial profile on the log-frequency axis.
    noise_threshold : float
        Multiplier on the estimated noise energy, below which PC is zeroed.

    Returns
    -------
    pc : (Y, X) float32
        Phase congruency in [0, 1].
    """
    if image.ndim != 2:
        raise ValueError(f"image must be 2-D, got {image.ndim}-D")
    img = image.astype(np.float32, copy=False)

    # Estimate noise energy from the finest-scale responses.
    noise_finest = _log_gabor_filter(
        img, wavelength=min_wavelength, orientation_deg=0.0, sigma_on_f=sigma_on_f
    )
    noise_e = np.abs(noise_finest[0]).mean()
    threshold = noise_threshold * noise_e + 1e-6

    energy = np.zeros_like(img, dtype=np.float32)
    sum_an = np.zeros_like(img, dtype=np.float32)
    total_amp = np.zeros_like(img, dtype=np.float32)

    for s in range(n_scales):
        wavelength = min_wavelength * (scaling_factor ** s)
        for o in range(n_orientations):
            orient = 180.0 * o / n_orientations
            even, odd = _log_gabor_filter(
                img, wavelength=wavelength, orientation_deg=orient, sigma_on_f=sigma_on_f
            )
            amp = np.sqrt(even * even + odd * odd)
            # Phase deviation measure (Kovesi 1999 eqn 13): use a sin/cos
            # approximation to keep things fast and avoid branchy atan2.
            # We use the signed angle via the even/odd components:
            # sin(angle) = odd / amp, cos(angle) = even / amp
            # but for energy summation we use the absolute phase.
            sign = np.sign(even * even - odd * odd)  # crude sign of cos(2 phi)
            energy += amp
            # Absolute value of the projection of each filter response on
            # the dominant direction: |even| if even dominates else |odd|.
            sum_an += amp * np.where(np.abs(even) > np.abs(odd), np.abs(even) / np.maximum(amp, 1e-6),
                                      np.abs(odd) / np.maximum(amp, 1e-6))
            total_amp += amp

    # Kovesi weighting: subtract noise threshold and rescale by total amplitude.
    # The classic formulation:
    #   PC = (sum(|F_so| * delta_phi) - T) / (sum(A_so) + epsilon)
    pc = (sum_an - threshold) / (total_amp + 1e-6)
    # Sigmoid-shaped soft threshold to keep smooth values, then clip.
    pc = np.clip(pc, 0.0, 1.0)
    # Tanh squash to [0, 1] for a softer, monotonic mapping.
    pc = np.tanh(pc * 1.5) ** 2
    return pc.astype(np.float32)


# ----------------------------------------------------------------------
# 2. SDN-Relief
# ----------------------------------------------------------------------


def _structure_tensor_field(image: np.ndarray, sigma: float = 1.5) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute the structure tensor fields Jxx, Jyy, Jxy from a 2-D image.

    The tensor is smoothed with a Gaussian of the given sigma to suppress
    noise. We do this with separable Gaussian smoothing via numpy.
    """
    gx = np.zeros_like(image, dtype=np.float32)
    gy = np.zeros_like(image, dtype=np.float32)
    # Central differences
    gx[:, 1:-1] = image[:, 2:] - image[:, :-2]
    gy[1:-1, :] = image[2:, :] - image[:-2, :]
    gx *= 0.5
    gy *= 0.5

    def gauss(s):
        # 1-D Gaussian kernel of half-width ~3*sigma, then outer product.
        r = int(np.ceil(3 * s))
        x = np.arange(-r, r + 1, dtype=np.float32)
        k = np.exp(-(x * x) / (2 * s * s))
        k /= k.sum()
        return k

    k = gauss(sigma)
    # Separable smoothing via numpy (reflect mode is fine for terrain).
    def smooth(a):
        a32 = a.astype(np.float32, copy=False)
        # horizontal
        r = len(k) // 2
        ap = np.pad(a32, ((0, 0), (r, r)), mode="reflect")
        out = np.zeros_like(a32, dtype=np.float32)
        for i, w in enumerate(k):
            out = out + (np.float32(w) * ap[:, i : i + a32.shape[1]]).astype(np.float32)
        # vertical
        ap = np.pad(out, ((r, r), (0, 0)), mode="reflect")
        out2 = np.zeros_like(out, dtype=np.float32)
        for i, w in enumerate(k):
            out2 = out2 + (np.float32(w) * ap[i : i + a32.shape[0], :]).astype(np.float32)
        return out2

    Jxx = smooth(gx * gx)
    Jyy = smooth(gy * gy)
    Jxy = smooth(gx * gy)
    return Jxx, Jyy, Jxy


def sdn_relief(
    image: np.ndarray,
    sun_azimuth_deg: float,
    bound: float = 3.0,
) -> np.ndarray:
    """Shadow-direction-normalized relief.

    Rotates the local gradient orientation into a Sun-relative frame and
    emphasizes the "uphill facing the Sun" component. The output is a
    bounded relief map; positive values indicate Sun-facing slopes, zero
    indicates flat / perpendicular-to-Sun terrain.

    Parameters
    ----------
    image : (Y, X) float32
        The input grayscale image in [0, 1].
    sun_azimuth_deg : float
        Sun azimuth in the image frame, in degrees [0, 360). The Sun
        direction is the unit vector
            (cos(az), sin(az))
        in image (X, Y) coordinates.
    bound : float
        Magnitude bound applied via tanh; controls the dynamic range of
        the relief channel.
    """
    if image.ndim != 2:
        raise ValueError(f"image must be 2-D, got {image.ndim}-D")
    img = image.astype(np.float32, copy=False)

    gx = np.zeros_like(img)
    gy = np.zeros_like(img)
    gx[:, 1:-1] = img[:, 2:] - img[:, :-2]
    gy[1:-1, :] = img[2:, :] - img[:-2, :]
    gx *= 0.5
    gy *= 0.5

    az = np.deg2rad(sun_azimuth_deg)
    # Sun direction in image (X, Y) frame.
    sx, sy = np.cos(az), np.sin(az)
    # Project gradient onto the Sun direction: this gives a signed
    # measure of "facing toward the Sun" (positive) vs "shadow side"
    # (negative). Shadows correspond to negative values; bright Sun-facing
    # slopes to positive.
    s_relief = gx * sx + gy * sy
    # Bounded via tanh to keep the dynamic range reasonable.
    relief = np.tanh(s_relief * bound)
    # Normalize to [0, 1] for downstream use as a structural channel.
    relief = (relief - relief.min()) / (relief.max() - relief.min() + 1e-6)
    return relief.astype(np.float32)


def estimate_sun_and_compute_sdn_relief(
    image: np.ndarray,
    metadata_sun_azimuth_deg: float | None,
    sun_angle_source_tier: str,
) -> tuple[np.ndarray, dict[str, float]]:
    """Compute SDN-Relief, estimating the Sun direction if needed.

    Returns the relief map and a provenance dict with the Sun azimuth
    used (and the confidence if estimated).
    """
    if (
        metadata_sun_azimuth_deg is not None
        and sun_angle_source_tier in ("label", "ephemeris")
    ):
        az = float(metadata_sun_azimuth_deg)
        provenance = {
            "sun_azimuth_deg_used": az,
            "sun_azimuth_source": "metadata",
        }
        return sdn_relief(image, az), provenance

    est = estimate_sun_azimuth(image)
    az = est["sun_azimuth_deg"]
    provenance = {
        "sun_azimuth_deg_used": az,
        "sun_azimuth_source": "image_estimate",
        "image_estimate_confidence": est["confidence"],
        "image_estimate_gradient_orientation_deg": est["gradient_orientation_deg"],
    }
    return sdn_relief(image, az), provenance


# ----------------------------------------------------------------------
# 3. Gradient-orientation field
# ----------------------------------------------------------------------


def gradient_orientation_field(
    image: np.ndarray,
    sigma: float = 1.5,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (orientation, coherence) of the local gradient field.

    orientation : (Y, X) float32, in [0, pi)
    coherence   : (Y, X) float32, in [0, 1] (1 = strong anisotropy)
    """
    Jxx, Jyy, Jxy = _structure_tensor_field(image, sigma=sigma)
    # Eigenvalues of the 2x2 tensor:
    #   lambda = 0.5 * (Jxx + Jyy) +/- sqrt(((Jxx-Jyy)/2)^2 + Jxy^2)
    Jxx = Jxx.astype(np.float32, copy=False)
    Jyy = Jyy.astype(np.float32, copy=False)
    Jxy = Jxy.astype(np.float32, copy=False)
    diff = (Jxx - Jyy) * 0.5
    disc = np.sqrt((diff * diff + Jxy * Jxy).astype(np.float32) + 1e-12)
    lambda1 = (0.5 * (Jxx + Jyy) + disc).astype(np.float32)
    lambda2 = (0.5 * (Jxx + Jyy) - disc).astype(np.float32)
    orient = 0.5 * np.arctan2(2 * Jxy, Jxx - Jyy + 1e-12)  # in [-pi/2, pi/2]
    orient = np.mod(orient, np.pi).astype(np.float32)
    # Coherence: (lambda1 - lambda2) / (lambda1 + lambda2), clipped to [0, 1].
    # When the tensor energy is tiny (flat regions), we report coherence=0
    # rather than NaN from a 0/0.
    energy = (lambda1 + lambda2).astype(np.float32)
    coherence = np.where(
        energy > 1e-8,
        (lambda1 - lambda2).astype(np.float32) / np.maximum(energy, 1e-8),
        np.float32(0.0),
    ).astype(np.float32)
    return orient, coherence


# ----------------------------------------------------------------------
# Top-level entrypoint
# ----------------------------------------------------------------------


def compute_invariant_channels(
    image: np.ndarray,
    sun_azimuth_deg: float | None,
    sun_angle_source_tier: str,
    n_scales: int = 4,
    n_orientations: int = 6,
    min_wavelength: float = 3.0,
    scaling_factor: float = 2.0,
    sigma_on_f: float = 0.55,
    noise_threshold: float = 1.5,
    include_gradient_orientation: bool = True,
) -> dict[str, np.ndarray]:
    """Compute all invariant channels.

    Returns a dict with keys:
      - ``phase_congruency``  : primary invariant channel
      - ``sdn_relief``        : shadow-direction-normalized relief
      - ``gradient_orientation`` : dense orientation in [0, pi) (if enabled)
      - ``gradient_coherence``   : local anisotropy in [0, 1] (if enabled)
    Plus ``sdn_relief_provenance`` as a dict of floats (the Sun azimuth
    provenance, for traceability into metadata).
    """
    if image.ndim == 3:
        # Take the first band as the working grayscale.
        gray = image[0].astype(np.float32, copy=False)
    else:
        gray = image.astype(np.float32, copy=False)

    out: dict[str, np.ndarray] = {}

    pc = phase_congruency(
        gray,
        n_scales=n_scales,
        n_orientations=n_orientations,
        min_wavelength=min_wavelength,
        scaling_factor=scaling_factor,
        sigma_on_f=sigma_on_f,
        noise_threshold=noise_threshold,
    )
    out["phase_congruency"] = pc

    relief, prov = estimate_sun_and_compute_sdn_relief(
        gray, sun_azimuth_deg, sun_angle_source_tier
    )
    out["sdn_relief"] = relief
    out["sdn_relief_provenance"] = np.array(
        [prov.get("sun_azimuth_deg_used", 0.0)], dtype=np.float32
    )

    if include_gradient_orientation:
        orient, coherence = gradient_orientation_field(gray)
        out["gradient_orientation"] = orient
        out["gradient_coherence"] = coherence
    return out
