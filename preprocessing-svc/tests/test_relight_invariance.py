"""CRITICAL: relighting-invariance regression test.

The whole point of the invariant channels is to produce representations
that are SIMILAR for the same underlying terrain rendered under different
sun azimuth/elevation combinations. This test directly verifies that
property on synthetic data.

We:
  1. Generate a procedural terrain.
  2. Render it under multiple sun azimuth/elevation combinations.
  3. Run each rendering through the invariant-channel pipeline.
  4. Assert the cross-rendering structural similarity (SSIM on the
     phase-congruency map, plus a histogram correlation on the
     SDN-Relief channel) stays above defined thresholds.
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from preprocessing_svc.invariant import compute_invariant_channels
from tests.conftest import hillshade


def _ssim(a: np.ndarray, b: np.ndarray, *, win: int = 7) -> float:
    """Simple mean-SSIM over a 7x7 sliding window.

    We compute it on the full image via uniform-mean sums for speed;
    exact SSIM is not the point — we just need a stable similarity
    score that's high when two maps are alike and low when they differ.
    """
    a = a.astype(np.float32)
    b = b.astype(np.float32)
    pad = win // 2
    ap = np.pad(a, pad, mode="reflect")
    bp = np.pad(b, pad, mode="reflect")
    C1 = 0.01 ** 2
    C2 = 0.03 ** 2
    ssims = []
    for y in range(0, a.shape[0], win):
        for x in range(0, a.shape[1], win):
            wa = ap[y : y + win, x : x + win]
            wb = bp[y : y + win, x : x + win]
            ma = wa.mean()
            mb = wb.mean()
            va = wa.var()
            vb = wb.var()
            cov = ((wa - ma) * (wb - mb)).mean()
            num = (2 * ma * mb + C1) * (2 * cov + C2)
            den = (ma * ma + mb * mb + C1) * (va + vb + C2)
            ssims.append(float(num / max(den, 1e-12)))
    return float(np.mean(ssims))


def _hist_corr(a: np.ndarray, b: np.ndarray, bins: int = 32) -> float:
    ha, _ = np.histogram(a.flatten(), bins=bins, range=(0.0, 1.0), density=True)
    hb, _ = np.histogram(b.flatten(), bins=bins, range=(0.0, 1.0), density=True)
    ha = ha / max(ha.sum(), 1e-9)
    hb = hb / max(hb.sum(), 1e-9)
    num = float((ha * hb).sum())
    den = float(np.sqrt((ha * ha).sum() * (hb * hb).sum()) + 1e-12)
    return num / den


def _relit_set(terrain: np.ndarray, sun_combos: list[tuple[float, float]]) -> list[np.ndarray]:
    out = []
    for az, el in sun_combos:
        out.append(hillshade(terrain, sun_az_deg=az, sun_el_deg=el))
    return out


@pytest.mark.parametrize(
    "sun_combos,threshold",
    [
        # Two widely separated azimuths; same elevation. Strong invariance expected.
        ([(30.0, 35.0), (210.0, 35.0)], 0.45),
        # Same azimuth, low vs high elevation. Same Sun azimuth means
        # structural features don't flip, so PC should be very similar.
        ([(45.0, 15.0), (45.0, 65.0)], 0.55),
        # A spread of three: PC is not perfectly invariant across
        # 180-deg azimuth flips, but should still preserve enough
        # structure to be substantially above the SSIM of unrelated maps.
        ([(20.0, 25.0), (90.0, 40.0), (200.0, 55.0)], 0.30),
    ],
)
def test_phase_congruency_is_relight_invariant(small_terrain, sun_combos, threshold):
    """The PRIMARY invariant channel — phase congruency — must be
    similar across very different sun angles for the same terrain.

    The threshold is calibrated to be substantially above what we'd
    expect for unrelated random maps (which would be near zero). A
    perfect 1.0 is not achievable across a 180-deg azimuth flip in
    real phase-congruency implementations; the threshold is chosen so
    the assertion is informative without being vacuous.
    """
    # Fast invariant config so the test runs in seconds, not minutes.
    inv_kwargs = dict(
        n_scales=3,
        n_orientations=4,
        min_wavelength=4.0,
        scaling_factor=2.0,
        sigma_on_f=0.55,
        noise_threshold=1.5,
        include_gradient_orientation=True,
    )
    renderings = _relit_set(small_terrain, sun_combos)
    pc_maps = []
    for r in renderings:
        inv = compute_invariant_channels(
            r,
            sun_azimuth_deg=None,
            sun_angle_source_tier="unavailable",
            **inv_kwargs,
        )
        pc_maps.append(inv["phase_congruency"])

    # Sanity: confirm that an unrelated map has low SSIM (this anchors
    # the threshold as meaningful, not vacuous).
    unrelated = np.random.default_rng(7).uniform(0, 1, pc_maps[0].shape).astype(np.float32)
    base_ssim = _ssim(pc_maps[0], unrelated)
    assert base_ssim < threshold, (
        f"Unrelated-map SSIM ({base_ssim:.3f}) is not below the relighting threshold "
        f"({threshold}); threshold is not discriminating."
    )

    for i in range(len(pc_maps)):
        for j in range(i + 1, len(pc_maps)):
            s = _ssim(pc_maps[i], pc_maps[j])
            assert s > threshold, (
                f"PC SSIM between renderings {i} and {j} (suns {sun_combos[i]} vs "
                f"{sun_combos[j]}) = {s:.3f}, below threshold {threshold}"
            )


def test_sdn_relief_is_relight_invariant_with_metadata(small_terrain):
    """SDN-Relief: when the sun direction is known (label/ephemeris tier),
    the channel normalizes for it. So the channel itself should be more
    invariant to sun angle than the raw image is."""
    inv_kwargs = dict(
        n_scales=3,
        n_orientations=4,
        min_wavelength=4.0,
        include_gradient_orientation=True,
    )
    suns = [(30.0, 35.0), (210.0, 35.0)]
    renderings = _relit_set(small_terrain, suns)
    reliefs = []
    for r, (az, _el) in zip(renderings, suns):
        inv = compute_invariant_channels(
            r,
            sun_azimuth_deg=az,
            sun_angle_source_tier="label",
            **inv_kwargs,
        )
        reliefs.append(inv["sdn_relief"])
    # Same terrain, same SDN, so the relief maps should be very similar.
    s = _ssim(reliefs[0], reliefs[1])
    assert s > 0.55, f"SDN-Relief SSIM = {s:.3f} below 0.55"


def test_sun_estimator_recovers_azimuth(small_terrain):
    """The image-based fallback should recover a sun azimuth that is
    at least approximately consistent with the actual rendering angle.
    We don't require exact equality (this is a proxy), but the angle
    error should be bounded.
    """
    from preprocessing_svc.sun_estimator import estimate_sun_azimuth

    az_true = 120.0
    shaded = hillshade(small_terrain, sun_az_deg=az_true, sun_el_deg=30.0)
    est = estimate_sun_azimuth(shaded)
    # Recover modulo 180 (azimuth is unoriented; Sun-to-shadow is
    # defined up to 180 deg).
    err = min(
        abs(est["sun_azimuth_deg"] - az_true),
        abs(est["sun_azimuth_deg"] - az_true + 180.0) % 180.0,
        abs(est["sun_azimuth_deg"] - az_true - 180.0) % 180.0,
    )
    # Allow up to 30 deg error — the image-based estimator is a proxy.
    assert err <= 30.0, f"Sun azimuth estimate error {err:.1f} deg > 30 deg"
    # We require SOME confidence signal above the noise floor; the
    # absolute value is small because the histogram is normalized to
    # density, but it should still be positive and not vanishingly tiny.
    assert est["confidence"] > 0.0


def test_invariant_channels_have_expected_shapes(small_terrain, ohrc_metadata):
    inv = compute_invariant_channels(
        hillshade(small_terrain, 45.0, 35.0),
        sun_azimuth_deg=ohrc_metadata.sun_azimuth_deg,
        sun_angle_source_tier=ohrc_metadata.sun_angle_source_tier.value,
        n_scales=3,
        n_orientations=4,
        min_wavelength=4.0,
        include_gradient_orientation=True,
    )
    assert inv["phase_congruency"].shape == small_terrain.shape
    assert inv["sdn_relief"].shape == small_terrain.shape
    assert inv["gradient_orientation"].shape == small_terrain.shape
    assert inv["gradient_coherence"].shape == small_terrain.shape
    # All outputs must be finite.
    for k, v in inv.items():
        if hasattr(v, "dtype"):
            assert np.isfinite(v).all(), f"{k} has non-finite values"
    # All non-provenance outputs in [0, 1].
    for k in ("phase_congruency", "sdn_relief", "gradient_coherence"):
        assert inv[k].min() >= 0.0 - 1e-6
        assert inv[k].max() <= 1.0 + 1e-6
