"""Generate ``contracts/ingestion.openapi.yaml`` from the live FastAPI app.

Run this whenever the API surface changes. The generator imports
``app.main:app`` and dumps its OpenAPI schema to YAML.

Usage:
    python -m scripts.generate_openapi
"""
from __future__ import annotations

import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.main import app  # noqa: E402

OUT = ROOT / "contracts" / "ingestion.openapi.yaml"


def main() -> None:
    schema = app.openapi()
    # Resolve local $refs to the file (best-effort; this is mainly a
    # human-readable artifact). The schema is otherwise self-contained.
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", encoding="utf-8") as f:
        yaml.safe_dump(schema, f, sort_keys=False, default_flow_style=False)
    print(f"Wrote {OUT} ({OUT.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
