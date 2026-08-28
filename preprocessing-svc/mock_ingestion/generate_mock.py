"""Mock IngestResult generator.

Produces a directory that matches the upstream contract from Member 1's
ingestion service: ``raw.cog`` (we emit a plain TIFF for portability
that the IO layer can read) and ``metadata.json`` with the documented
schema.

Usage:
    python -m mock_ingestion.generate_mock --sensor OHRC --out ./var/mock/ohrc1
    python -m mock_ingestion.generate_mock --sensor IIRS --out ./var/mock/iirs1
"""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image


def _synthetic_terrain(y: int, x: int, rng: np.random.Generator) -> np.ndarray:
    """Procedural heightmap: a few Gaussians + low-freq noise."""
    yy, xx = np.meshgrid(np.arange(y), np.arange(x), indexing="ij")
    h = np.zeros((y, x), dtype=np.float32)
    # Add a handful of crater-like depressions and hills.
    n_features = max(3, int((y * x) ** 0.5 / 40))
    for _ in range(n_features):
        cy = rng.integers(10, y - 10)
        cx = rng.integers(10, x - 10)
        amp = rng.uniform(-1.0, 1.0)
        sigma = rng.uniform(8, 25)
        h += amp * np.exp(-((yy - cy) ** 2 + (xx - cx) ** 2) / (2 * sigma * sigma))
    # Low-frequency terrain.
    h += 0.3 * np.sin(yy / 30.0) * np.cos(xx / 40.0)
    h += 0.1 * rng.standard_normal((y, x)).astype(np.float32)
    return h


def _hillshade(
    height: np.ndarray,
    sun_azimuth_deg: float,
    sun_elevation_deg: float,
) -> np.ndarray:
    """Simple Lambertian hillshade from a heightmap."""
    gy = np.zeros_like(height)
    gx = np.zeros_like(height)
    gy[1:-1, :] = height[2:, :] - height[:-2, :]
    gx[:, 1:-1] = height[:, 2:] - height[:, :-2]
    gy *= 0.5
    gx *= 0.5
    az = math.radians(sun_azimuth_deg)
    el = math.radians(max(sun_elevation_deg, 0.1))
    slope = np.arctan(np.hypot(gx, gy))
    aspect = np.arctan2(-gx, gy)  # in image-frame convention
    shaded = (
        np.cos(el) * np.cos(slope)
        + np.sin(el) * np.sin(slope) * np.cos(az - aspect)
    )
    shaded = np.clip(shaded, 0.0, 1.0)
    return shaded.astype(np.float32)


def _make_ohrc(height: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """OHRC: 12-bit panchromatic, ~0.6 m GSD. We render at 1 m/px for tests."""
    h = _hillshade(height, sun_azimuth_deg=45.0, sun_elevation_deg=35.0)
    # 12-bit equivalent, scale to [0, 4095].
    img = (h * 4095).astype(np.uint16)
    # Tiny bit of photon noise.
    img = np.clip(img + rng.integers(0, 16, img.shape, dtype=np.int32), 0, 4095).astype(np.uint16)
    return img


def _make_tmc(height: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """TMC: 3-band visible+NIR, ~5 m GSD."""
    h_vis = _hillshade(height, sun_azimuth_deg=120.0, sun_elevation_deg=40.0)
    h_nir = _hillshade(height, sun_azimuth_deg=120.0, sun_elevation_deg=40.0)
    p = (h_vis * 4095).astype(np.uint16)
    q = (h_nir * 4095).astype(np.uint16)
    # Stack as 3 bands: vis, red, nir (just to be plausible).
    return np.stack([p, q, ((p + q) // 2)], axis=0)


def _make_iirs(height: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """IIRS: 64-band hyperspectral cube in [0, 1] float32."""
    bands = 64
    out = np.zeros((bands,) + height.shape, dtype=np.float32)
    for b in range(bands):
        # Each band has a slightly different "spectral" weighting of the
        # same terrain: a sinusoid in the band index modulates the
        # hillshade. Bands are highly correlated so PCA collapses them.
        weight = 0.7 + 0.3 * np.cos(2 * math.pi * b / bands)
        h_b = _hillshade(height, sun_azimuth_deg=60.0, sun_elevation_deg=45.0) * weight
        # Add a small spectral offset.
        h_b += 0.05 * np.sin(2 * math.pi * b / 9.0 + height * 2.0)
        h_b = np.clip(h_b, 0.0, 1.0)
        # Add per-band noise.
        h_b = h_b + 0.005 * rng.standard_normal(h_b.shape).astype(np.float32)
        out[b] = h_b
    return out


def _make_reference(height: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """REFERENCE: a typical LOLA-style basemap, 5 m/px, uint8."""
    h = _hillshade(height, sun_azimuth_deg=0.0, sun_elevation_deg=45.0)
    img = (h * 255).astype(np.uint8)
    img = np.clip(img + rng.integers(0, 4, img.shape, dtype=np.int32), 0, 255).astype(np.uint8)
    return img


def _write_image(path: Path, arr: np.ndarray) -> None:
    """Write the array to ``path``.

    PIL doesn't recognize ``.cog`` as a real format. The contract names
    the file ``raw.cog``; in local development we write a TIFF alongside
    (or under) that name so the IO layer can read it. We force the
    format to TIFF by passing ``format="TIFF"`` when the suffix is
    .cog, which keeps the contract filename while making the file
    readable. For multi-band cubes (e.g. IIRS) we use a multi-page TIFF
    so the full cube is preserved.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    fmt = "TIFF" if path.suffix.lower() in (".cog", ".tif", ".tiff") else None
    if arr.ndim == 2:
        Image.fromarray(arr).save(path, format=fmt)
        return
    # Multi-band: write each band as a page in a multi-page TIFF.
    bands = arr.shape[0]
    pages = []
    for b in range(bands):
        band = arr[b]
        if np.issubdtype(band.dtype, np.floating):
            band = (np.clip(band, 0, 1) * 65535).astype(np.uint16)
        else:
            band = band.astype(np.uint16)
        pages.append(Image.fromarray(band))
    if len(pages) == 1:
        pages[0].save(path, format=fmt)
    else:
        pages[0].save(
            path, format=fmt, save_all=True, append_images=pages[1:]
        )


def _write_metadata(path: Path, meta: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, sort_keys=True)


def _gsd_for(sensor: str) -> float:
    return {
        "OHRC": 0.6,
        "TMC": 5.0,
        "IIRS": 20.0,
        "REFERENCE": 5.0,
    }[sensor]


def _band_count(sensor: str) -> int:
    return {"OHRC": 1, "TMC": 3, "IIRS": 64, "REFERENCE": 1}[sensor]


def _bit_depth(sensor: str) -> int:
    return {"OHRC": 12, "TMC": 10, "IIRS": 16, "REFERENCE": 8}[sensor]


def generate(
    out_dir: str | Path,
    sensor: str = "OHRC",
    height: int = 256,
    width: int = 256,
    sun_azimuth_deg: float | None = 45.0,
    sun_elevation_deg: float | None = 30.0,
    sun_angle_source_tier: str = "label",
    seed: int = 0,
) -> dict[str, str]:
    """Generate a mock IngestResult at ``out_dir``.

    Returns a dict of references (file://... or absolute path strings) so
    the caller can pass them straight into the PreprocessRequest.
    """
    rng = np.random.default_rng(seed)
    terrain = _synthetic_terrain(height, width, rng)
    if sensor == "OHRC":
        arr = _make_ohrc(terrain, rng)
    elif sensor == "TMC":
        arr = _make_tmc(terrain, rng)
    elif sensor == "IIRS":
        arr = _make_iirs(terrain, rng)
    elif sensor == "REFERENCE":
        arr = _make_reference(terrain, rng)
    else:
        raise ValueError(f"Unknown sensor: {sensor}")

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    raw_path = out / "raw.cog"
    meta_path = out / "metadata.json"

    # The mock writes a TIFF in place of a COG; the IO layer reads either.
    _write_image(raw_path, arr)
    meta = {
        "sensor_type": sensor,
        "gsd": _gsd_for(sensor),
        "sun_azimuth_deg": sun_azimuth_deg,
        "sun_elevation_deg": sun_elevation_deg,
        "sun_angle_source_tier": sun_angle_source_tier,
        "projection": "EPSG:4326",
        "footprint_wkt": None,
        "band_count": _band_count(sensor),
        "bit_depth": _bit_depth(sensor),
        "acquisition_time": "2026-08-28T12:00:00Z",  # mock placeholder
    }
    _write_metadata(meta_path, meta)

    return {
        "raw_image_ref": f"file://{raw_path.resolve()}",
        "metadata_ref": f"file://{meta_path.resolve()}",
    }


def main() -> None:
    p = argparse.ArgumentParser(description="Mock IngestResult generator")
    p.add_argument("--out", required=True, help="Output directory")
    p.add_argument(
        "--sensor",
        default="OHRC",
        choices=["OHRC", "TMC", "IIRS", "REFERENCE"],
    )
    p.add_argument("--height", type=int, default=256)
    p.add_argument("--width", type=int, default=256)
    p.add_argument("--sun-azimuth-deg", type=float, default=45.0)
    p.add_argument("--sun-elevation-deg", type=float, default=30.0)
    p.add_argument(
        "--sun-angle-source-tier",
        default="label",
        choices=["label", "ephemeris", "unavailable"],
    )
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    refs = generate(
        out_dir=args.out,
        sensor=args.sensor,
        height=args.height,
        width=args.width,
        sun_azimuth_deg=args.sun_azimuth_deg,
        sun_elevation_deg=args.sun_elevation_deg,
        sun_angle_source_tier=args.sun_angle_source_tier,
        seed=args.seed,
    )
    print(json.dumps(refs, indent=2))


if __name__ == "__main__":
    main()
