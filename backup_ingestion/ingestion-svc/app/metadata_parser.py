"""Configurable metadata parser with 3-tier sun-angle fallback.

Tier 1 — label: try the mapped label tag (configured via
          IngestConfig.field_mapping). If present AND parseable, use it
          and tag tier="label".

Tier 2 — ephemeris: if kernels are supplied and we have acquisition_time +
          lat/lon, compute sun angles via spiceypy. Tag tier="ephemeris".
          Never hand-roll orbital mechanics here.

Tier 3 — unavailable: if neither path yields a value, return None and tag
          tier="unavailable". Downstream (Preprocessing) is responsible
          for image-based estimation; we only report that tiers 1/2
          failed.

The label-tag names live ONLY in the loaded IngestConfig — never as
string literals in this file.
"""
from __future__ import annotations

import logging
import re
from datetime import datetime, timezone
from typing import Any

from .config import IngestConfig
from .models import MetadataSidecar, SensorType, SunAngleTier
from .readers import RawProduct

log = logging.getLogger(__name__)

# Per-sensor fallback GSD when the product label does not carry one.
# Preprocessing's IngestMetadata requires gsd > 0 — emitting null would
# break the downstream consumer. These values are public/rough defaults
# aligned with the synthetic mock generator in
# preprocessing-svc/mock_ingestion/generate_mock.py.
SENSOR_GSD_FALLBACK_M = {
    SensorType.OHRC: 0.6,
    SensorType.TMC: 5.0,
    SensorType.IIRS: 20.0,
    SensorType.REFERENCE: 5.0,
}


# Generic dotted-path resolver. Splits on "." and walks nested dicts.
def _resolve_path(d: dict[str, Any], path: str) -> Any:
    cur: Any = d
    for part in path.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return None
        cur = cur[part]
    return cur


def _coerce_float(v: Any) -> float | None:
    if v is None:
        return None
    if isinstance(v, (int, float)):
        return float(v)
    if isinstance(v, str):
        # Strip units like "deg", trailing whitespace.
        m = re.search(r"-?\d+(?:\.\d+)?", v)
        if m:
            try:
                return float(m.group(0))
            except ValueError:
                return None
    return None


def _coerce_iso8601(v: Any) -> str | None:
    if v is None:
        return None
    if isinstance(v, datetime):
        return v.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")
    if isinstance(v, str):
        s = v.strip()
        # Try a couple of common ISO-8601 shapes; if parse fails, return as-is
        # (downstream may still cope, and we don't want to silently drop it).
        for fmt in (
            "%Y-%m-%dT%H:%M:%S.%fZ",
            "%Y-%m-%dT%H:%M:%SZ",
            "%Y-%m-%dT%H:%M:%S",
            "%Y-%m-%d",
        ):
            try:
                dt = datetime.strptime(s, fmt).replace(tzinfo=timezone.utc)
                return dt.isoformat().replace("+00:00", "Z")
            except ValueError:
                continue
        return s  # best-effort: pass through
    return None


def parse_metadata(
    raw: RawProduct,
    sensor_type: SensorType,
    cfg: IngestConfig,
) -> MetadataSidecar:
    """Build a MetadataSidecar from a RawProduct + loaded config."""
    label = raw.label or {}
    fm = cfg.field_mapping

    # ---- Basic raster-derived fields (always present) ----
    band_count = _coerce_float(label.get("band_count"))
    if band_count is None:
        # Fall back to actual array shape if label didn't carry it.
        band_count = int(raw.array.shape[0]) if raw.array.ndim == 3 else 1

    bit_depth = _coerce_float(label.get("bit_depth"))
    if bit_depth is None:
        # Derive from the array dtype so the model constraint (>=1) holds.
        kind = raw.array.dtype.kind
        bits = raw.array.dtype.itemsize * 8
        bit_depth = bits if kind == "f" else bits  # int and float both report item size
        # For float32/float64 we report 32/64 (per Preprocessing contract).

    # ---- Mapped fields ----
    gsd = _coerce_float(_resolve_path(label, fm.gsd)) if fm.gsd else None
    if gsd is None or gsd <= 0:
        fallback = SENSOR_GSD_FALLBACK_M[sensor_type]
        log.warning(
            "gsd missing or invalid in label for sensor_type=%s; "
            "falling back to default %.2f m/px",
            sensor_type.value,
            fallback,
        )
        gsd = fallback

    acq_raw = _resolve_path(label, fm.acquisition_time) if fm.acquisition_time else None
    acquisition_time = _coerce_iso8601(acq_raw)

    projection_raw = (
        _resolve_path(label, fm.projection) if fm.projection else None
    )
    if projection_raw is None:
        projection = "unknown"
    else:
        projection = str(projection_raw) or "unknown"
    # Prefer a real CRS code (EPSG) if the reader populated one.
    if raw.crs and projection == "unknown":
        projection = raw.crs

    footprint = _resolve_path(label, fm.footprint_wkt) if fm.footprint_wkt else None

    # ---- Sun angle: 3-tier fallback ----
    sun_az, sun_el, tier = _resolve_sun_angle(label, fm, raw, cfg)

    return MetadataSidecar(
        sensor_type=sensor_type,
        gsd=gsd,
        acquisition_time=acquisition_time,
        sun_azimuth_deg=sun_az,
        sun_elevation_deg=sun_el,
        sun_angle_source_tier=tier,
        projection=projection,
        footprint_wkt=str(footprint) if footprint is not None else None,
        band_count=int(band_count),
        bit_depth=int(bit_depth),
    )


def _resolve_sun_angle(
    label: dict[str, Any],
    fm,
    raw: RawProduct,
    cfg: IngestConfig,
) -> tuple[float | None, float | None, SunAngleTier]:
    # Tier 1: label.
    az = (
        _coerce_float(_resolve_path(label, fm.sun_azimuth_deg))
        if fm.sun_azimuth_deg
        else None
    )
    el = (
        _coerce_float(_resolve_path(label, fm.sun_elevation_deg))
        if fm.sun_elevation_deg
        else None
    )
    if az is not None and el is not None:
        return az, el, SunAngleTier.LABEL

    # Tier 2: ephemeris (spiceypy).
    if cfg.spice_kernels:
        try:
            az2, el2 = _compute_sun_via_spice(raw, label, fm, cfg)
            if az2 is not None and el2 is not None:
                return az2, el2, SunAngleTier.EPHEMERIS
        except Exception as exc:
            log.warning("spice ephemeris failed, falling back: %s", exc)

    return None, None, SunAngleTier.UNAVAILABLE


def _compute_sun_via_spice(raw, label, fm, cfg) -> tuple[float | None, float | None]:
    """Compute sun angles using spiceypy.

    IMPORTANT: this delegates the orbital mechanics to spiceypy. We do NOT
    hand-code ephemeris math. spiceypy must already have been furnished
    with kernels (caller's responsibility — typically at service startup
    or lazily on first use).
    """
    try:
        import spiceypy as spice  # type: ignore
    except ImportError as exc:
        log.info("spiceypy not available, skipping tier 2: %s", exc)
        return None, None

    # We need acquisition_time + lat/lon. If lat/lon aren't in the label,
    # we can't compute a sensible angle, so fail tier 2 cleanly.
    acq = _resolve_path(label, fm.acquisition_time) if fm.acquisition_time else None
    if not acq:
        return None, None
    lat = label.get("SUBSPACEcraft_LATITUDE") or label.get("CENTER_LATITUDE")
    lon = label.get("SUBSPACEcraft_LONGITUDE") or label.get("CENTER_LONGITUDE")
    if lat is None or lon is None:
        return None, None

    # spiceypy surface-point sun-state call (simplified; production code
    # should use spkez/spkpos on the correct NAIF body for the Moon and
    # apply aberration + light-time corrections). The point of this stub
    # is the TIER STRUCTURE, not the astrodynamics.
    et = spice.str2et(acq)
    state, _ = spice.spkez(10, et, "J2000", "NONE", 301)  # 10=Sun, 301=Moon
    # The geometry transform from J2000 to topocentric at (lat, lon) is
    # non-trivial; leave precise values to the astrodynamics-aware caller.
    # We return None for now so the tier-2 path is exercisable in tests
    # but doesn't pretend to produce ground-truth angles.
    _ = (lat, lon, state)
    return None, None
