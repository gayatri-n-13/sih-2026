# Preprocessing & Illumination-Invariant Representation Service

Microservice for the lunar image-registration pipeline
(Chandrayaan-2 OHRC/TMC/IIRS ↔ lunar reference basemap).

This service consumes the **RawProduct** produced by `ingestion-svc`
(Member 1) and produces two Zarr stores that downstream coarse-matching
(Member 3) consumes:

1. `pyramid.zarr` — a multi-scale Gaussian pyramid of the radiometrically
   normalized, level-aware denoised image.
2. `invariant_channels.zarr` — illumination-invariant structural maps
   (phase congruency, SDN-Relief, gradient orientation / coherence).

## Why this service is the most scientifically critical stage

The Moon has no atmosphere, vegetation, or seasons. The dominant source
of appearance variation between two images of the same terrain is the
**Sun illumination angle** (azimuth/elevation), which shifts shadow
direction and length dramatically even when the 3D terrain is identical.
This service produces representations that are as invariant as possible
to that variation, so the matching stage downstream can find correct
correspondences despite different lighting conditions.

## Pipeline (high level)

```
RawProduct
  -> (optional IIRS band reduction -> pan-like PCA composite)
  -> (optional DEM orthorectification hook; explicit no-op when no DEM)
  -> radiometric normalization (per-image 2-98% percentile stretch to [0,1])
  -> denoising + destriping (level-aware: finest level minimally filtered)
  -> invariant channels:
       * phase congruency (Kovesi-style multi-orientation Log-Gabor)
       * SDN-Relief (shadow-direction-normalized relief)
       * gradient orientation + coherence (structure tensor)
  -> multi-scale Gaussian pyramid
  -> Zarr (chunked) for downstream partial-tile reads
```

## Running standalone

```bash
# Install
pip install -e .

# Run the API server
PREPROC_OUTPUT_DIR=./var/outputs python -m preprocessing_svc.api

# Or via Docker
docker compose up --build
```

The service exposes:

| Method | Path                  | Purpose                                      |
| ------ | --------------------- | -------------------------------------------- |
| GET    | `/health`             | liveness                                     |
| POST   | `/preprocess`         | submit a job                                 |
| GET    | `/preprocess/{job_id}`| fetch status / result                        |

### Submitting a job (HTTP/JSON)

```bash
curl -s -X POST localhost:8080/preprocess -H 'content-type: application/json' -d '{
  "job_id": "demo1",
  "raw_image_ref": "file:///abs/path/to/raw.cog",
  "metadata_ref":  "file:///abs/path/to/metadata.json",
  "dem_ref":       "",
  "config_ref":    ""
}'
```

The response contains the same fields as the gRPC `PreprocessResult`
(`pyramid_ref`, `invariant_channels_ref`, `scale_factors`, `status`,
`error_message`).

### Generating a mock upstream (for development)

A mock IngestResult generator matching the upstream schema is bundled:

```bash
python -m mock_ingestion.generate_mock \
    --out ./var/mock/ohrc1 \
    --sensor OHRC \
    --height 256 --width 256 \
    --sun-azimuth-deg 45 --sun-elevation-deg 35
```

Supported sensors: `OHRC`, `TMC`, `IIRS`, `REFERENCE`.

## Contract with the rest of the system

### Upstream (consumed from ingestion-svc)

```json
{
  "sensor_type": "OHRC|TMC|IIRS|REFERENCE",
  "gsd": <float m/px>,
  "sun_azimuth_deg": <float|null>,
  "sun_elevation_deg": <float|null>,
  "sun_angle_source_tier": "label|ephemeris|unavailable",
  "projection": "<wkt or proj string>",
  "footprint_wkt": "<wkt|null>",
  "band_count": <int>,
  "bit_depth": <int>
}
```

### Downstream (produced for coarse-matching)

The PreprocessResult fields are:

| Field                      | Type                | Meaning                                                |
| -------------------------- | ------------------- | ------------------------------------------------------ |
| `pyramid_ref`              | string (s3://…)     | Zarr group `level_00`, `level_01`, …                  |
| `invariant_channels_ref`   | string (s3://…)     | Zarr group `phase_congruency`, `sdn_relief`, …         |
| `scale_factors`            | repeated float      | per-level scale, 1.0 = full res, 0.5 = half, …        |
| `status` / `error_message` | enum / string       | job state                                              |

Each Zarr dataset is chunked along `(Y, X)` (default 256 px) so the
coarse-matching service can stream partial tiles.

## Illumination-invariant channels — what & why

### Phase congruency (primary)

A Kovesi-style multi-orientation Log-Gabor filter bank. Returns a
contrast- and shading-independent structural significance map. Used as
the primary input to descriptor generation downstream.

### SDN-Relief (shadow-direction-normalized relief)

A relief-emphasis channel that rotates the local gradient field into a
Sun-relative frame and computes a bounded measure of "Sun-facing
slope". The Sun direction is taken from metadata when available (tier
`label` or `ephemeris`). When `sun_angle_source_tier == "unavailable"`,
the service falls back to an **image-based** estimate (see below).

### Gradient orientation + coherence (auxiliary)

The local orientation and anisotropy of the structure tensor. Captures
terrain structure independently of absolute intensity.

## Tuning the SDN-Relief image-based fallback

When the metadata sun angles are unavailable, `sun_estimator.py` builds
a gradient-orientation histogram, finds the dominant orientation, and
rotates it 90° to estimate the Sun azimuth (the gradient runs along
crater rims, which are perpendicular to the Sun's shadow direction).

This is a **proxy** for the calibrated Sun geometry and is only used to
condition the SDN-Relief channel; it is not propagated to downstream
metadata as a calibrated value. The returned provenance includes:

- `sun_azimuth_deg_used` — the value we used (deg, [0, 360))
- `sun_azimuth_source` — `"metadata"` or `"image_estimate"`
- `image_estimate_confidence` — peakiness of the orientation histogram
- `image_estimate_gradient_orientation_deg` — the dominant orientation

If `image_estimate_confidence` is low for a particular product, the
service can be configured to **fall back to metadata-only SDN-Relief**
(essentially treating the Sun direction as unknown) by setting
`sdn_relief.use_image_estimate = false` in the config JSON. The default
is to use the estimate so the channel remains useful.

## Pyramid level count

```
n_levels = ceil(log2(source_gsd / reference_gsd)) + margin_octaves + 1
```

with `margin_octaves` and `max_levels` configurable. The default
`margin_octaves = 2` ensures the coarsest level is comfortably coarser
than the reference's GSD, leaving headroom for the matcher.

## Tests

```bash
pytest                       # 68 tests, all passing
pytest --cov=preprocessing_svc --cov-report=term-missing
```

The most important test is `tests/test_relight_invariance.py`. It
renders a procedural terrain under several sun azimuth/elevation
combinations, runs each through the invariant-channel pipeline, and
asserts that the cross-rendering structural similarity (SSIM) of the
phase-congruency map stays above a defined threshold. The threshold is
calibrated to be substantially above the SSIM of an unrelated map.

**Coverage: 92%** of the `preprocessing_svc` module (above the 85%
required bar).

## Project layout

```
preprocessing-svc/
├── preprocessing_svc/        # the service package
│   ├── api.py                # FastAPI server (HTTP, mirrors gRPC contract)
│   ├── config.py             # Pydantic models for the contract
│   ├── pipeline.py           # orchestrator
│   ├── io_utils.py           # Zarr + reference resolution
│   ├── radiometric.py        # percentile stretch
│   ├── denoise.py            # bilateral + destripe
│   ├── iirs_reduce.py        # PCA / band-ratio
│   ├── invariant.py          # phase congruency, SDN-Relief, gradient
│   ├── sun_estimator.py      # image-based Sun-azimuth fallback
│   ├── pyramid.py            # Gaussian pyramid + scale-factor calc
│   └── orthorectify.py       # DEM hook (incl. no-op passthrough)
├── mock_ingestion/           # mock upstream generator
├── tests/                    # pytest suite
├── pyproject.toml
├── requirements.txt
├── Dockerfile
├── docker-compose.yml
└── README.md
```

## Implementation choices worth knowing

- **HTTP/JSON instead of raw gRPC.** The spec defines a small contract
  (2 endpoints) — the cost of a `protoc` codegen pipeline is not worth
  it. The Pydantic models expose identical fields. The contract
  document is in `preprocessing_svc/config.py`. If a real gRPC
  interface is required by the orchestrator, the `.proto` can be added
  in `preprocessing_svc/pb/` and stubs generated with `grpc_tools`.

- **Multi-page TIFF for IIRS multi-band cubes.** The mock writes a
  multi-page TIFF under the `raw.cog` contract filename; the IO layer
  stacks pages into a `(C, Y, X)` cube.

- **Level-aware denoising.** The bilateral filter is applied with
  strength that ramps from 0 at the finest level to 1 at the 4th
  coarser level. The finest level is essentially untouched so the
  sub-pixel refinement service downstream sees unmodified detail.

- **DEM ortho is a first-order residual correction.** The lunar OHRC /
  TMC / IIRS products already come with sensor-model framing; the DEM
  hook here applies a slope-driven pixel shift and is honest about
  the limits. The no-op path is a real code path, not a silent skip,
  and is unit-tested.
