# Contract: Preprocessing → Coarse Matching

This document is the binding interface for the downstream consumer
(Member 3: Coarse Matching). The format below mirrors the upstream
contract in `../config.py`; both are Pydantic models so the JSON
schema is authoritative.

## Inputs to coarse-matching (per job)

| Field                       | Type    | Source                          | Notes                                      |
| --------------------------- | ------- | ------------------------------- | ------------------------------------------ |
| `pyramid_ref`               | string  | PreprocessResult                | `s3://…/pyramid.zarr` (or `file://…`)      |
| `invariant_channels_ref`    | string  | PreprocessResult                | `s3://…/invariant_channels.zarr`           |
| `scale_factors`             | float[] | PreprocessResult                | per-level scale; 1.0 = full, 0.5 = half    |
| `sensor_type`               | string  | Zarr group attribute            | `OHRC` / `TMC` / `IIRS` / `REFERENCE`      |
| `gsd`                       | float   | Zarr group attribute            | source GSD in m/px                          |
| `reference_gsd_m`           | float   | Zarr group attribute            | reference GSD in m/px (pyramid anchor)      |
| `sun_azimuth_used`          | float   | Zarr group attribute            | value used for SDN-Relief (deg)            |
| `sun_azimuth_source`        | string  | Zarr group attribute            | `metadata` or `image_estimate`              |

## Zarr layout

### `pyramid.zarr`

Group with datasets `level_00`, `level_01`, …, one per pyramid level.
Each dataset is `float32`, shape `(Y, X)` (or `(C, Y, X)` for multi-band
preserved inputs). Chunked along `(Y, X)` with a default chunk size of
256 pixels.

### `invariant_channels.zarr`

Group with at least these datasets:

- `phase_congruency`         — `float32 (Y, X)`, in [0, 1] (primary)
- `sdn_relief`               — `float32 (Y, X)`, in [0, 1]
- `gradient_orientation`     — `float32 (Y, X)`, in [0, π) (auxiliary)
- `gradient_coherence`       — `float32 (Y, X)`, in [0, 1] (auxiliary)

If the input was IIRS, an additional `iirs_components` dataset may be
present with the PCA-reduced pan-like composite.

## Reading partial tiles

Both Zarr stores use chunked storage so coarse-matching can read a
window of any size without loading the whole image. Use
`zarr.open_group(..., mode="r")` and read with explicit slice arguments.

## Status polling

A submitted job is a JSON object with the same fields as the gRPC
`PreprocessResult`. Poll until `status == "SUCCEEDED"` or
`status == "FAILED"`. On `FAILED`, `error_message` is populated.

## Error semantics

The service is fail-soft on a few specific issues:

- `sun_angle_source_tier == "unavailable"`: continues using the
  image-based Sun estimate; does NOT fail.
- Missing DEM or `enable_orthorectify == false`: explicit no-op
  passthrough; does NOT fail.
- Malformed raw image: job transitions to FAILED with
  `error_message` populated.

Hard failures (raw image not found, metadata invalid) are FAILED with
an explanatory error string. The orchestrator should retry with
backoff and surface persistent failures.
