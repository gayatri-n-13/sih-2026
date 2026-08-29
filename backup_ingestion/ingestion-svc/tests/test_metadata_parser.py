"""Unit tests for the 3-tier sun-angle fallback and configurable mapping.

Each tier is tested independently:
  tier 1: label fields present → returned as-is
  tier 2: label missing + kernels configured + ephemeris path returns value → tier=ephemeris
  tier 3: label missing + no kernels → tier=unavailable, values null
"""
from __future__ import annotations

import pytest

from app.config import FieldMapping, IngestConfig, ValidationRules
from app.metadata_parser import _resolve_sun_angle, parse_metadata
from app.models import SensorType, SunAngleTier
from app.readers import RawProduct


def _raw(label: dict | None = None) -> RawProduct:
    import numpy as np
    return RawProduct(array=np.zeros((1, 4, 4), dtype="uint8"), label=label or {})


def test_tier1_label_hit():
    cfg = IngestConfig(
        field_mapping=FieldMapping(
            sun_azimuth_deg="SUN_AZIMUTH", sun_elevation_deg="SUN_ELEVATION"
        )
    )
    az, el, tier = _resolve_sun_angle(
        {"SUN_AZIMUTH": "123.4 deg", "SUN_ELEVATION": "42.5"},
        cfg.field_mapping,
        _raw(),
        cfg,
    )
    assert tier == SunAngleTier.LABEL
    assert az == pytest.approx(123.4)
    assert el == pytest.approx(42.5)


def test_tier1_label_partial_falls_through_to_tier3():
    """If only one of (az, el) is in the label, tier 1 fails and we go to tier 3
    (no kernels configured)."""
    cfg = IngestConfig(
        field_mapping=FieldMapping(
            sun_azimuth_deg="SUN_AZIMUTH", sun_elevation_deg="SUN_ELEVATION"
        )
    )
    az, el, tier = _resolve_sun_angle(
        {"SUN_AZIMUTH": 10.0},  # elevation missing
        cfg.field_mapping,
        _raw(),
        cfg,
    )
    assert tier == SunAngleTier.UNAVAILABLE
    assert az is None and el is None


def test_tier2_ephemeris_used_when_label_missing():
    """When label has no sun-angle fields but kernels are configured and
    _compute_sun_via_spice returns values, tier 2 wins."""
    from app import metadata_parser

    cfg = IngestConfig(
        field_mapping=FieldMapping(
            sun_azimuth_deg="SUN_AZIMUTH",
            sun_elevation_deg="SUN_ELEVATION",
            acquisition_time="START_TIME",
        ),
        spice_kernels=["naif0012.tls", "de440s.bsp"],  # non-empty triggers tier 2
    )

    def fake_spice(raw, label, fm, c):
        return 200.0, 15.0

    monkey = pytest.MonkeyPatch()
    monkey.setattr(metadata_parser, "_compute_sun_via_spice", fake_spice)
    try:
        az, el, tier = _resolve_sun_angle(
            {"START_TIME": "2021-08-15T04:32:11Z"},
            cfg.field_mapping,
            _raw(),
            cfg,
        )
        assert tier == SunAngleTier.EPHEMERIS
        assert az == 200.0 and el == 15.0
    finally:
        monkey.undo()


def test_tier3_unavailable_when_no_label_no_kernels():
    cfg = IngestConfig(
        field_mapping=FieldMapping(
            sun_azimuth_deg="SUN_AZIMUTH", sun_elevation_deg="SUN_ELEVATION"
        )
    )
    az, el, tier = _resolve_sun_angle(
        {},  # empty label
        cfg.field_mapping,
        _raw(),
        cfg,
    )
    assert tier == SunAngleTier.UNAVAILABLE
    assert az is None and el is None


def test_tier2_falls_through_when_spice_fails():
    """spiceypy not installed / kernels missing → tier 3."""
    cfg = IngestConfig(
        field_mapping=FieldMapping(
            sun_azimuth_deg="SUN_AZIMUTH",
            sun_elevation_deg="SUN_ELEVATION",
            acquisition_time="START_TIME",
        ),
        spice_kernels=["x.tls"],
    )
    # No label means tier 1 misses; tier 2 also misses (no kernels actually
    # loaded + lat/lon absent). Should fall through cleanly.
    az, el, tier = _resolve_sun_angle(
        {"START_TIME": "2021-08-15T04:32:11Z"},
        cfg.field_mapping,
        _raw(),
        cfg,
    )
    assert tier == SunAngleTier.UNAVAILABLE
    assert az is None and el is None


def test_parse_metadata_uses_crs_fallback_when_projection_unmapped():
    cfg = IngestConfig(field_mapping=FieldMapping())
    raw = _raw(label={})
    raw.crs = "EPSG:4326"
    sidecar = parse_metadata(raw, SensorType.REFERENCE, cfg)
    assert sidecar.projection == "EPSG:4326"
    assert sidecar.sun_angle_source_tier == SunAngleTier.UNAVAILABLE
    # gsd falls back to per-sensor default since label has none
    assert sidecar.gsd == 5.0  # REFERENCE default


def test_parse_metadata_gsd_falls_back_to_per_sensor_default():
    """When the label has no gsd field, parser uses SENSOR_GSD_FALLBACK_M
    (Preprocessing requires positive gsd)."""
    cfg = IngestConfig(field_mapping=FieldMapping())  # no gsd mapping
    raw = _raw(label={})
    for sensor, expected in [
        (SensorType.OHRC, 0.6),
        (SensorType.TMC, 5.0),
        (SensorType.IIRS, 20.0),
        (SensorType.REFERENCE, 5.0),
    ]:
        sidecar = parse_metadata(raw, sensor, cfg)
        assert sidecar.gsd == expected, f"{sensor.value}: expected {expected}, got {sidecar.gsd}"


def test_parse_metadata_rejects_zero_or_negative_gsd_from_label():
    """A label-sourced gsd <= 0 also falls back (rather than writes null
    and breaking Preprocessing)."""
    cfg = IngestConfig(field_mapping=FieldMapping(gsd="MAP_RESOLUTION"))
    raw = _raw(label={"MAP_RESOLUTION": "0"})
    sidecar = parse_metadata(raw, SensorType.OHRC, cfg)
    assert sidecar.gsd == 0.6  # fell back


def test_parse_metadata_coerces_iso8601():
    cfg = IngestConfig(field_mapping=FieldMapping(acquisition_time="START_TIME"))
    raw = _raw(label={"START_TIME": "2021-08-15T04:32:11Z"})
    sidecar = parse_metadata(raw, SensorType.OHRC, cfg)
    assert sidecar.acquisition_time == "2021-08-15T04:32:11Z"


def test_no_hardcoded_label_tags_in_metadata_module():
    """Regression guard: ensure no Python literal in metadata_parser matches
    an obvious label tag like 'SUN_AZIMUTH' (those live ONLY in config)."""
    import inspect
    from app import metadata_parser

    src = inspect.getsource(metadata_parser)
    for tag in ("SUN_AZIMUTH", "SUN_ELEVATION", "START_TIME", "MAP_RESOLUTION"):
        assert tag not in src, (
            f"hard-coded label tag {tag!r} found in metadata_parser — must "
            "come from config"
        )
