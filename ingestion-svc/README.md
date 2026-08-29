# ingestion-svc (Member 1)

Lunar image ingestion microservice. Reads raw image files from disk,
derives per-product metadata, and writes both to an object store
(references only, never raw bytes over the network).

## Endpoints

| Method | Path                | Purpose                          |
| ------ | ------------------- | -------------------------------- |
| GET    | `/health`           | liveness                         |
| POST   | `/ingest`           | submit an IngestRequest          |
| GET    | `/ingest/{job_id}`  | poll job status / fetch result   |

## Running locally

```bash
pip install -r requirements.txt -r requirements-dev.txt
INGESTION_SYNC=1 INGESTION_FAKES3_ROOT=./var/fakes3 \
    python -m app.main
```

## Contracts (binding for downstream services)

| File                                          | Format         | Notes                                          |
| --------------------------------------------- | -------------- | ---------------------------------------------- |
| `contracts/metadata.schema.json`              | JSON Schema 2020-12 | `metadata.json` envelope. Mirror of `app/models.py::ProductMetadata`. |
| `contracts/ingestion.openapi.yaml`            | OpenAPI 3.1    | Auto-generated from the FastAPI app via `python -m scripts.generate_openapi`. |

The contract tests in `tests/test_contract.py` assert that the JSON
Schema and OpenAPI document are both valid and that the schema matches
the Pydantic model. The OpenAPI document is regenerated from the live
app on every test run, so it cannot drift.

## Outputs

The service writes two artifacts to the configured object store:

1. The raw image bytes, at the constructed `raw_image_ref`.
2. `metadata.json`, at the constructed `metadata_ref`. This is what
   Member 2's preprocessing-svc reads.

`metadata.json` contents (see `contracts/metadata.schema.json` for the
formal schema):

```json
{
  "sensor_type": "OHRC|TMC|IIRS|REFERENCE",
  "gsd": <float m/px>,
  "sun_azimuth_deg": <float|null>,
  "sun_elevation_deg": <float|null>,
  "sun_angle_source_tier": "label|ephemeris|unavailable",
  "projection": "<wkt or proj>",
  "footprint_wkt": "<wkt|null>",
  "band_count": <int>,
  "bit_depth": <int>,
  "acquisition_time": "<ISO-8601|null>"
}
```

## Sun-angle source tiers

| Tier          | Meaning                                                 | Behavior                                                            |
| ------------- | ------------------------------------------------------- | ------------------------------------------------------------------- |
| `label`       | Calibrated product label                                | `sun_azimuth_deg` and `sun_elevation_deg` are required              |
| `ephemeris`   | Computed from SPICE / on-board ephemeris                | `sun_azimuth_deg` and `sun_elevation_deg` are required              |
| `unavailable` | No calibrated angles available                          | Both angles are written as `null`; the tier triggers downstream image-based estimation |

The validator in `IngestRequest` rejects requests where the tier is
`label` or `ephemeris` but the angles are missing — this is the
"angles consistent with tier" check.

## Tests

```bash
pytest                                    # run all tests
pytest --cov=app --cov-report=term-missing   # with coverage
```
