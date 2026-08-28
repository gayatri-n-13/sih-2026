"""Top-level preprocessing pipeline.

Orchestrates: load -> (optional IIRS band reduction) -> orthorectify ->
radiometric normalize -> denoise (level-aware) -> invariant channels ->
pyramid -> Zarr write.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from preprocessing_svc.config import (
    IngestMetadata,
    PreprocessConfig,
    PreprocessResult,
    SensorType,
)
from preprocessing_svc.denoise import denoise_for_level
from preprocessing_svc.iirs_reduce import pca_reduce
from preprocessing_svc.invariant import compute_invariant_channels
from preprocessing_svc.io_utils import (
    read_image_array,
    read_metadata_json,
    resolve_ref,
    write_metadata_json,
    write_zarr,
)
from preprocessing_svc.orthorectify import orthorectify
from preprocessing_svc.pyramid import compute_scale_factors, gaussian_pyramid
from preprocessing_svc.radiometric import percentile_stretch, to_grayscale_if_multiband


def _load_inputs(
    raw_image_ref: str,
    metadata_ref: str,
) -> tuple[np.ndarray, IngestMetadata]:
    raw_path = Path(resolve_ref(raw_image_ref))
    meta_path = Path(resolve_ref(metadata_ref))
    raw = read_image_array(raw_path)
    meta_dict = read_metadata_json(meta_path)
    metadata = IngestMetadata.model_validate(meta_dict)
    return raw, metadata


def _maybe_reduce_iirs(
    image: np.ndarray,
    metadata: IngestMetadata,
) -> tuple[np.ndarray, dict[str, Any]]:
    """For IIRS, run PCA reduction. For other sensors, just track the band count."""
    info: dict[str, Any] = {"sensor_type": metadata.sensor_type.value, "reduction": "none"}
    if metadata.sensor_type == SensorType.IIRS and image.ndim == 3:
        components, mean, explained = pca_reduce(image, n_components=3, variance_target=0.90)
        info["reduction"] = "pca"
        info["n_components"] = int(components.shape[0])
        info["explained_variance_ratio"] = [float(x) for x in explained]
        info["full_cube_ref"] = "self"  # we keep the full cube in metadata
        return components, info
    if image.ndim == 3 and metadata.sensor_type != SensorType.IIRS:
        # Multi-band non-IIRS: keep all bands for provenance but the
        # matching stage wants a single-band structural map. We
        # collapse to grayscale for invariant channels; the full
        # multi-band is still written into the pyramid for provenance.
        info["reduction"] = "none_kept_multiband"
        info["band_count"] = int(image.shape[0])
        return image, info
    info["band_count"] = 1 if image.ndim == 2 else int(image.shape[0])
    return image, info


def _load_dem(dem_ref: str) -> np.ndarray | None:
    if not dem_ref:
        return None
    p = Path(resolve_ref(dem_ref))
    if not p.exists():
        return None
    return read_image_array(p)


def run(
    request_dict: dict[str, Any],
    output_dir: Path,
    config: PreprocessConfig | None = None,
    dem: np.ndarray | None = None,
) -> PreprocessResult:
    """Run the full pipeline. Returns PreprocessResult.

    If ``config`` is None, we attempt to load a PreprocessConfig from
    ``request_dict["config_ref"]``; if that is empty too, defaults are
    used.
    """
    cfg = config
    if cfg is None:
        config_ref = request_dict.get("config_ref", "")
        if config_ref:
            from preprocessing_svc.api import _load_config_from_ref

            cfg = _load_config_from_ref(config_ref)
        else:
            cfg = PreprocessConfig()
    job_id = request_dict["job_id"]
    raw_image_ref = request_dict["raw_image_ref"]
    metadata_ref = request_dict["metadata_ref"]
    dem_ref = request_dict.get("dem_ref", "")

    raw, metadata = _load_inputs(raw_image_ref, metadata_ref)
    if dem is None:
        dem = _load_dem(dem_ref) if dem_ref else None

    # Step 1: optional IIRS reduction.
    reduced, reduce_info = _maybe_reduce_iirs(raw, metadata)
    if reduced.ndim == 3:
        working_band_for_invariant = reduced[0]
    else:
        working_band_for_invariant = reduced

    # Step 2: optional orthorectification.
    ortho, ortho_info = orthorectify(
        working_band_for_invariant,
        dem=dem,
        enable=cfg.enable_orthorectify,
    )

    # Step 3: radiometric normalization (percentile stretch to [0, 1]).
    normed = percentile_stretch(
        ortho,
        low_pct=cfg.stretch_low_pct,
        high_pct=cfg.stretch_high_pct,
        out_range=(0.0, 1.0),
    )

    # Step 4: invariant channels (operate on the radiometrically normalized
    # grayscale working band).
    inv = compute_invariant_channels(
        normed,
        sun_azimuth_deg=metadata.sun_azimuth_deg,
        sun_angle_source_tier=metadata.sun_angle_source_tier.value,
        n_scales=cfg.invariant.n_scales,
        n_orientations=cfg.invariant.n_orientations,
        min_wavelength=cfg.invariant.min_wavelength,
        scaling_factor=cfg.invariant.scaling_factor,
        sigma_on_f=cfg.invariant.sigma_on_f,
        noise_threshold=cfg.invariant.noise_threshold,
        include_gradient_orientation=cfg.invariant.include_gradient_orientation,
    )

    # Step 5: pyramid.
    scale_factors = compute_scale_factors(
        source_gsd=metadata.gsd,
        reference_gsd=cfg.pyramid.reference_gsd_m,
        margin_octaves=cfg.pyramid.margin_octaves,
        max_levels=cfg.pyramid.max_levels,
    )

    # The "pyramid image" we feed in: the radiometrically normalized
    # grayscale. The multi-band reduced cube (if any) is recorded in
    # metadata for provenance.
    pyr_input = normed.astype(np.float32, copy=False)
    pyr_levels = gaussian_pyramid(pyr_input, scale_factors)

    # Step 6: level-aware denoising of each pyramid level. We re-apply
    # denoising on each level rather than on the input, because each
    # level has different noise statistics. The finest level is barely
    # touched (the refinement service needs full detail).
    denoised_levels = []
    for i, level in enumerate(pyr_levels):
        d = denoise_for_level(
            level,
            level=i,
            finest_level=0,
            bilateral_d=cfg.denoise.bilateral_d,
            bilateral_sigma_color=cfg.denoise.bilateral_sigma_color,
            bilateral_sigma_space=cfg.denoise.bilateral_sigma_space,
            apply_destripe=cfg.denoise.apply_destripe,
            destripe_strength=cfg.denoise.destripe_strength,
        )
        denoised_levels.append(d)

    # Step 7: write Zarr outputs.
    out_dir = Path(output_dir) / job_id
    out_dir.mkdir(parents=True, exist_ok=True)

    pyramid_arrays = {
        f"level_{i:02d}": level for i, level in enumerate(denoised_levels)
    }
    pyramid_meta = {
        "scale_factors": [float(s) for s in scale_factors],
        "sensor_type": metadata.sensor_type.value,
        "gsd": float(metadata.gsd),
        "reference_gsd_m": float(cfg.pyramid.reference_gsd_m),
    }
    pyramid_path = write_zarr(
        out_dir / "pyramid.zarr", pyramid_arrays, attrs=pyramid_meta
    )

    inv_arrays = {k: v for k, v in inv.items() if isinstance(v, np.ndarray)}
    # The provenance array is small; keep it as a dataset too so it's
    # serializable.
    inv_meta = {
        "sensor_type": metadata.sensor_type.value,
        "sun_azimuth_used": float(
            inv.get("sdn_relief_provenance", np.array([0.0]))[0]
        ),
        "sun_azimuth_source": (
            "metadata"
            if metadata.sun_azimuth_deg is not None
            and metadata.sun_angle_source_tier.value in ("label", "ephemeris")
            else "image_estimate"
        ),
    }
    inv_path = write_zarr(
        out_dir / "invariant_channels.zarr",
        inv_arrays,
        attrs=inv_meta,
    )

    # Step 8: sidecar metadata.
    side_meta = {
        "job_id": job_id,
        "sensor_type": metadata.sensor_type.value,
        "gsd": metadata.gsd,
        "sun_azimuth_deg": metadata.sun_azimuth_deg,
        "sun_elevation_deg": metadata.sun_elevation_deg,
        "sun_angle_source_tier": metadata.sun_angle_source_tier.value,
        "band_count": metadata.band_count,
        "bit_depth": metadata.bit_depth,
        "projection": metadata.projection,
        "footprint_wkt": metadata.footprint_wkt,
        "pyramid_ref": str(pyramid_path),
        "invariant_channels_ref": str(inv_path),
        "scale_factors": [float(s) for s in scale_factors],
        "iirs_reduction": reduce_info,
        "ortho": ortho_info,
    }
    write_metadata_json(out_dir / "preprocess_result.json", side_meta)

    return PreprocessResult(
        job_id=job_id,
        status="SUCCEEDED",
        pyramid_ref=str(pyramid_path),
        invariant_channels_ref=str(inv_path),
        scale_factors=[float(s) for s in scale_factors],
    )
