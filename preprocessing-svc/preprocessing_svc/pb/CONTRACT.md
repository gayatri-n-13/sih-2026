# Contract: Preprocessing → Coarse Matching

This document is the binding interface between **preprocessing-svc
(Member 2)** and **coarse-matching-svc (Member 3)**. It also
documents what preprocessing-svc **consumes** from **ingestion-svc
(Member 1)** so each side can verify against the same source of truth.

> **Reconciliation status — verified against ingestion-svc contracts on
> 2026-08-28:**
>
> The fields documented under "Upstream (consumed from ingestion-svc)"
> are the **assumed** schema preprocessing-svc was built against from
> the system-prompt spec. As of this writing, the repository at
> `github.com/gayatri-n-13/sih-2026` has **no** `ingestion-svc/`
> directory on any branch or tag — the real ingestion contract is not
> yet published. preprocessing-svc has been made **forward-compatible**
> with the field most likely to be added (`acquisition_time`) and is
> otherwise self-consistent: every field it reads from `metadata.json`
> is documented here and present in its own `IngestMetadata` Pydantic
> model. When Member 1's contract lands, this note must be updated and
> every field re-verified.

## Upstream (consumed from ingestion-svc)

IngestResult envelope (the JSON envelope wrapping the per-product
metadata):

| Field            | Type   | Description                                          |
| ---------------- | ------ | ---------------------------------------------------- |
| `raw_image_ref`  | string | `s3://…` or `file://…` to the raw image              |
| `metadata_ref`   | string | `s3://…` or `file://…` to `metadata.json`            |

`metadata.json` (loaded from `metadata_ref`):

| Field                    | Type                | Required | Notes                                       |
| ------------------------ | ------------------- | -------- | ------------------------------------------- |
| `sensor_type`            | string enum         | yes      | `OHRC` / `TMC` / `IIRS` / `REFERENCE`       |
| `gsd`                    | float (m/px)        | yes      | > 0                                         |
| `sun_azimuth_deg`        | float or null       | no       | 0–360; null if unknown                      |
| `sun_elevation_deg`      | float or null       | no       | -90 to 90; null if unknown                  |
| `sun_angle_source_tier`  | string enum         | no       | `label` / `ephemeris` / `unavailable`       |
| `projection`             | string              | yes      | WKT or EPSG/proj string                     |
| `footprint_wkt`          | string or null      | no       | WKT footprint                               |
| `band_count`             | int                 | yes      | ≥ 1                                         |
| `bit_depth`              | int                 | yes      | ≥ 1                                         |
| `acquisition_time`       | ISO-8601 string     | no       | **Forward-compat**: accepted but not used    |

When `sun_angle_source_tier == "unavailable"`, preprocessing-svc
estimates the Sun direction from the image itself (see
`sun_estimator.py`) and continues without failure.

## Inputs to coarse-matching (per job)

| Field                       | Type    | Source                          | Notes                                      |
| --------------------------- | ------- | ------------------------------- | ------------------------------------------ |
| `pyramid_ref`               | string  | PreprocessResult                | `s3://…/pyramid.zarr` (or `file://…`)      |
| `invariant_channels_ref`    | string  | PreprocessResult                | `s3://…/invariant_channels.zarr`           |
| `scale_factors`             | float[] | PreprocessResult                | per-level scale; 1.0 = full res, 0.5 = half |
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

## Reconciliation checklist (for when Member 1 publishes)

When `ingestion-svc/contracts/metadata.schema.json` lands, run
through this list:

1. Every field in the table above is present in the JSON Schema with
   a compatible type and (for enums) a compatible value set.
2. Confirm or correct `sun_azimuth_deg` / `sun_elevation_deg` units.
3. Confirm `sun_angle_source_tier` allowed values.
4. Confirm `projection` is a string (Pydantic accepts anything
   coercable; if ingestion-svc emits a structured object we may need
   to add a parser).
5. Confirm or correct `band_count` semantics: should it match the
   actual array axis 0 length? (mock currently always emits it.)
6. If `acquisition_time` is required (not optional), remove the
   `Optional` from `IngestMetadata.acquisition_time`.
