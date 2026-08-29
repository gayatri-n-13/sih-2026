import json
import jsonschema
from contracts.mocks.mock_generator import MockGenerator

def test_ingestion_metadata_schema():
    print("Validating Ingestion Metadata Mock against JSON Schema...")
    with open("ingestion-svc/contracts/metadata.schema.json", "r") as f:
        schema = json.load(f)

    gen = MockGenerator()
    mock_meta = gen.mock_ingest_metadata("OHRC")

    try:
        jsonschema.validate(instance=mock_meta, schema=schema)
        print("✅ Ingestion metadata mock is consistent with schema.")
    except jsonschema.ValidationError as e:
        print(f"❌ Ingestion metadata mock failed validation: {e.message}")
        raise

def test_preprocess_mock_structure():
    print("Validating Preprocessing Mock against CONTRACT.md requirements...")
    gen = MockGenerator()
    mock = gen.mock_preprocess_result("test-job")

    required_fields = ["pyramid_ref", "invariant_channels_ref", "scale_factors", "sensor_type", "gsd", "reference_gsd_m", "sun_azimuth_used", "sun_azimuth_source"]
    for field in required_fields:
        if field not in mock:
            print(f"❌ Preprocessing mock missing required field: {field}")
            raise KeyError(field)

    print("✅ Preprocessing mock contains all required contract fields.")

def test_verify_mock_structure():
    print("Validating Verification Mock against CONTRACT.md / Proto requirements...")
    gen = MockGenerator()
    mock = gen.mock_verify_result("test-job")

    required_fields = ["verified_matches_ref", "coverage_report", "updated_transform", "status"]
    for field in required_fields:
        if field not in mock:
            print(f"❌ Verification mock missing required field: {field}")
            raise KeyError(field)

    coverage = mock["coverage_report"]
    coverage_required = ["tile_grid_rows", "tile_grid_cols", "per_tile_counts", "under_covered_tiles", "coverage_fraction"]
    for field in coverage_required:
        if field not in coverage:
            print(f"❌ Verification coverage_report missing required field: {field}")
            raise KeyError(field)

    print("✅ Verification mock contains all required contract fields.")

if __name__ == "__main__":
    try:
        test_ingestion_metadata_schema()
        test_preprocess_mock_structure()
        test_verify_mock_structure()
        print("\nALL MOCKS VERIFIED AGAINST CONTRACTS.")
    except Exception as e:
        print(f"\nCONTRACT VALIDATION FAILED: {e}")
        exit(1)
