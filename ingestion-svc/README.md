# ingestion-svc

First-stage ingest for the lunar image-registration pipeline (Chandrayaan-2
OHRC/TMC/IIRS ↔ lunar reference basemap). Decodes a raw product, validates
structure, parses metadata (configurable field-mapping + 3-tier sun-angle
fallback), and writes a Cloud-Optimized GeoTIFF + `metadata.json` to object
storage.

Downstream services consume only the **URI references** in the
`IngestResult` — image bytes never traverse the wire.

## Quick start

```bash
# from repo root
docker compose up --build ingestion-svc minio
# -> http://localhost:8000/health
# -> http://localhost:9001  (MinIO console: minioadmin / minioadmin)
```

Submit a job (using a local fixture):

```bash
curl -X POST http://localhost:8000/v1/ingest \
  -H 'content-type: application/json' \
  -d '{
    "job_id": "demo-001",
    "source_file_uri": "file:///path/to/fixture.tif",
    "sensor_type": "REFERENCE"
  }'
```

Poll:

```bash
curl http://localhost:8000/v1/ingest/demo-001
```

## Local development (no Docker)

```bash
cd ingestion-svc
pip install -r requirements.txt
INGESTION_FORCE_LOCAL=1 uvicorn app.main:app --reload
# Output URIs land in ./local_s3/{bucket}/{job_id}/ingestion/
```

## Tests

```bash
pip install -r requirements.txt
pytest
```

Coverage target: ≥90% on `app/`.

## Configuration

### Default config

`app/config/default.yaml` ships with safe defaults. The `field_mapping` block
holds the ONLY label-tag names the service knows about.

### Per-job override

Pass `config_ref` in the `IngestRequest`:

```json
{
  "job_id": "demo-002",
  "source_file_uri": "s3://raw/ohrc/foo.LBL",
  "sensor_type": "OHRC",
  "config_ref": "s3://configs/ch2/ohrc-strict.yaml"
}
```

Accepted `config_ref` schemes:
- `file:///abs/path.yaml`
- `s3://bucket/key.yaml`
- `<basename>.yaml` (resolved against `app/config/`)

If the config can't be loaded, ingestion falls back to the default config and
logs a warning — a transient config-bucket outage won't fail the pipeline.

## Adding a new sensor_type

1. Implement a reader in `app/readers/` (any format). The reader must
   satisfy the `ProductReader` protocol (see `app/readers/__init__.py`).
2. Decorate the class with `@register("NEWSENSOR")` to wire it into the
   selector.
3. If the new sensor needs different label-tag names, create a
   `<name>.yaml` under `app/config/` with the appropriate `field_mapping`
   and pass it via `config_ref`.
4. Add a unit test under `tests/` covering:
   - valid file → COMPLETED
   - corrupted file → FAILED with specific message
   - missing metadata field → correct sun-angle tier triggered

No core code (readers selector, parser, pipeline) needs to change.

## Sun-angle fallback tiers

| Tier | Source | Code path |
|------|--------|-----------|
| 1    | Product label (mapped via `field_mapping`) | `metadata_parser._resolve_sun_angle` reads `SUN_AZIMUTH` / `SUN_ELEVATION` |
| 2    | Ephemeris (spiceypy) — needs kernels in `INGESTION_SPICE_KERNELS_DIR` and acquisition_time + lat/lon in label | `_compute_sun_via_spice` |
| 3    | `unavailable` — downstream (Preprocessing) must estimate from image | Both above return None |

We never hand-roll orbital mechanics — tier 2 is delegated to `spiceypy`.

## Object storage layout

```
s3://{bucket}/{job_id}/ingestion/
├── raw.cog          # Cloud-Optimized GeoTIFF
└── metadata.json    # MetadataSidecar (see contracts/metadata.schema.json)
```

## API contract

See `../contracts/ingestion.openapi.yaml` (REST/OpenAPI). Any breaking change
must be reviewed by the Preprocessing service owner — it is the only
downstream consumer of `IngestResult`.

## Definition of Done

- [x] `docker compose up ingestion-svc minio` boots a healthy service
- [x] Ingest → status round-trip works for at least one fixture per sensor_type, including REFERENCE
- [x] All 3 sun-angle tiers unit-tested
- [x] README complete; API contract committed and versioned in `../contracts/`
- [x] No hard-coded label field names anywhere in `app/` — all via config
