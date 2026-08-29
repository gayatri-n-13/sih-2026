# API Contracts

This directory holds the wire-level contracts that cross service boundaries in the
lunar image-registration pipeline. **Contracts are shared code** — once published, a
change to any file here requires review from every consumer (see the owning service's
README for the consumer list).

## Files

| File | Purpose | Owner | Consumers |
|------|---------|-------|-----------|
| `ingestion.openapi.yaml` | Ingestion Service REST API (IngestRequest, JobHandle, IngestResult) | ingestion-svc | preprocessing-svc (Orchestrator when it exists) |
| `metadata.schema.json`   | Schema for the `metadata.json` sidecar written to object storage by ingestion-svc | ingestion-svc | preprocessing-svc |

## Versioning

- These files are **versioned via git**, not via semver tags. Bump the `info.version`
  in the OpenAPI spec on breaking changes; consumers track via commit hash.
- Breaking change = removing/renaming a field, changing a field's type, or changing
  the value of an enum.
- Additive changes (new optional field, new enum value) are non-breaking but must
  still be announced in the consumer's PR.

## Status

- `ingestion.openapi.yaml` — **DRAFT v0.1.0**. Stable enough for Preprocessing to
  build a mock client. Comments in the spec call out which fields are still TBD.
- `metadata.schema.json` — **DRAFT v0.1.0**. Schema is frozen for v0.1 of the
  pipeline; pre-1.0 it may evolve but breaking changes will be flagged in PR review.
