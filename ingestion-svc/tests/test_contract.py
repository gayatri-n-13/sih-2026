"""Contract tests.

These tests prove that the published contract artifacts
(``contracts/metadata.schema.json`` and
``contracts/ingestion.openapi.yaml``) accurately describe what the
service actually does. If they fail:

  - A Pydantic model changed but the schema wasn't updated, or
  - The OpenAPI doc is stale relative to the live app, or
  - A round-tripped IngestResult fails JSON-Schema validation.

The schema and the OpenAPI are the binding surface for downstream
services (notably Member 2's preprocessing-svc); they MUST stay
synchronized with the code.
"""
from __future__ import annotations

import json
from pathlib import Path

import jsonschema
import pytest
import yaml
from openapi_spec_validator import validate as validate_openapi

from app.ingest import ingest_request
from app.models import (
    IngestRequest,
    IngestResult,
    ProductMetadata,
    SensorType,
    SunAngleSourceTier,
)
from app.storage import resolve_ref

ROOT = Path(__file__).resolve().parents[1]
SCHEMA_PATH = ROOT / "contracts" / "metadata.schema.json"
OPENAPI_PATH = ROOT / "contracts" / "ingestion.openapi.yaml"


@pytest.fixture(scope="module")
def metadata_schema() -> dict:
    with SCHEMA_PATH.open(encoding="utf-8") as f:
        return json.load(f)


@pytest.fixture(scope="module")
def openapi_doc() -> dict:
    with OPENAPI_PATH.open(encoding="utf-8") as f:
        return yaml.safe_load(f)


# ---------------------------------------------------------------------------
# Schema validity
# ---------------------------------------------------------------------------


def test_metadata_schema_is_valid_json_schema_2020(metadata_schema):
    """Just instantiating the schema and using jsonschema validates the
    meta-schema. We don't need a separate validator call."""
    assert metadata_schema["$schema"].startswith("https://json-schema.org/draft/2020-12")
    # If the schema is structurally broken, jsonschema's Draft202012Validator
    # will reject it on first use.
    jsonschema.Draft202012Validator.check_schema(metadata_schema)


def test_metadata_schema_required_fields_present(metadata_schema):
    required = set(metadata_schema["required"])
    expected = {
        "sensor_type",
        "gsd",
        "sun_angle_source_tier",
        "projection",
        "band_count",
        "bit_depth",
    }
    assert expected.issubset(required), (
        f"metadata.schema.json is missing required fields: {expected - required}"
    )


# ---------------------------------------------------------------------------
# Schema-matches-model: a metadata instance produced by the live code
# must validate against the schema.
# ---------------------------------------------------------------------------


def test_metadata_schema_matches_pydantic_model(metadata_schema, ohrc_image, tmp_path, monkeypatch):
    """Produce a metadata instance via the real ingest pipeline and
    validate it against the schema. This is the no-drift proof."""
    # Configure the workspace.
    fakes3 = tmp_path / "fakes3"
    fakes3.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("INGESTION_FAKES3_ROOT", str(fakes3))

    req = IngestRequest(
        raw_image_path=str(ohrc_image),
        sensor_type=SensorType.OHRC,
        projection="EPSG:4326",
        sun_azimuth_deg=45.0,
        sun_elevation_deg=35.0,
        sun_angle_source_tier=SunAngleSourceTier.LABEL,
        output_prefix="s3://ingestion-bucket/",
        job_id="contract_test",
    )
    r = ingest_request(req)
    assert r.status == "SUCCEEDED"
    # Load the actual on-disk metadata.json.
    meta_path = resolve_ref(r.metadata_ref)
    instance = json.loads(meta_path.read_text(encoding="utf-8"))
    # Validate.
    jsonschema.validate(instance=instance, schema=metadata_schema)


def test_metadata_schema_accepts_minimal_instance(metadata_schema):
    """A bare-minimum instance (only required fields) should validate."""
    instance = {
        "sensor_type": "REFERENCE",
        "gsd": 5.0,
        "sun_angle_source_tier": "unavailable",
        "projection": "EPSG:4326",
        "band_count": 1,
        "bit_depth": 8,
    }
    jsonschema.validate(instance=instance, schema=metadata_schema)


def test_metadata_schema_rejects_unknown_sensor_type(metadata_schema):
    """An unknown sensor_type should fail validation."""
    instance = {
        "sensor_type": "NOPE",
        "gsd": 5.0,
        "sun_angle_source_tier": "unavailable",
        "projection": "EPSG:4326",
        "band_count": 1,
        "bit_depth": 8,
    }
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(instance=instance, schema=metadata_schema)


def test_metadata_schema_rejects_negative_gsd(metadata_schema):
    instance = {
        "sensor_type": "OHRC",
        "gsd": -1.0,
        "sun_angle_source_tier": "unavailable",
        "projection": "EPSG:4326",
        "band_count": 1,
        "bit_depth": 12,
    }
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(instance=instance, schema=metadata_schema)


# ---------------------------------------------------------------------------
# OpenAPI validity
# ---------------------------------------------------------------------------


def test_openapi_doc_is_valid(openapi_doc):
    """The OpenAPI doc must conform to the OpenAPI 3.x spec."""
    validate_openapi(openapi_doc)


def test_openapi_doc_exposes_required_paths(openapi_doc):
    paths = set(openapi_doc["paths"].keys())
    assert "/health" in paths
    assert "/ingest" in paths
    assert "/ingest/{job_id}" in paths


def test_openapi_doc_defines_required_schemas(openapi_doc):
    schemas = openapi_doc["components"]["schemas"]
    for required in (
        "IngestRequest",
        "IngestResult",
        "SensorType",
        "SunAngleSourceTier",
    ):
        assert required in schemas, f"OpenAPI doc missing schema: {required}"


# ---------------------------------------------------------------------------
# product_metadata matches the JSON-Schema field-by-field
# ---------------------------------------------------------------------------


def test_product_metadata_field_names_match_schema(metadata_schema):
    """The set of fields in the Pydantic model and the JSON Schema
    must agree on the *required* names at minimum."""
    schema_props = set(metadata_schema["properties"].keys())
    model_fields = set(ProductMetadata.model_fields.keys())
    # The model should not be missing any field the schema declares.
    missing_in_model = schema_props - model_fields
    assert not missing_in_model, (
        f"metadata.schema.json declares fields not in ProductMetadata: "
        f"{missing_in_model}"
    )


# ---------------------------------------------------------------------------
# OpenAPI doc is regenerated from the live FastAPI app and matches
# the committed copy. If this test fails after a code change, regenerate
# the committed file via:  python -m scripts.generate_openapi
# ---------------------------------------------------------------------------


def test_openapi_doc_does_not_drift_from_app():
    """Regenerate the OpenAPI doc from the live FastAPI app and
    compare to the committed copy. The schemas are byte-identical when
    sorted by key (which our generator does)."""
    import yaml as _yaml

    from app.main import app as fastapi_app

    live = fastapi_app.openapi()
    with OPENAPI_PATH.open(encoding="utf-8") as f:
        committed = _yaml.safe_load(f)

    # Compare schema-by-schema: any schema that's in the live doc must
    # match the committed version exactly. This is more robust than a
    # raw byte compare (which can differ on whitespace from FastAPI's
    # default generator) but still proves no drift in shape.
    live_schemas = live["components"]["schemas"]
    committed_schemas = committed["components"]["schemas"]
    assert set(live_schemas.keys()) == set(committed_schemas.keys()), (
        f"Schema drift: live has {set(live_schemas) - set(committed_schemas)}, "
        f"committed has {set(committed_schemas) - set(live_schemas)}"
    )
    for name in live_schemas:
        assert live_schemas[name] == committed_schemas[name], (
            f"Schema '{name}' has drifted from the live FastAPI app. "
            f"Regenerate via: python -m scripts.generate_openapi"
        )
