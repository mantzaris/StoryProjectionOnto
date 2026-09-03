"""Emit deterministic JSON Schemas for public and model-interaction contracts."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import tempfile
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from story_projection_onto.contracts import (
    MODEL_VISIBLE_SCHEMA_TYPES,
    PUBLIC_SCHEMA_TYPES,
    SCHEMA_VERSION,
    SCORER_ONLY_SCHEMA_TYPES,
    canonical_json,
    canonical_json_schema,
    canonical_sha256,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIRECTORY = PROJECT_ROOT / "schemas" / "jsonschema"


def _snake_case(name: str) -> str:
    first_pass = re.sub(r"(.)([A-Z][a-z]+)", r"\1_\2", name)
    return re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", first_pass).lower()


def _atomic_write(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, path)
    except BaseException:
        temporary_path.unlink(missing_ok=True)
        raise


def _schema_bytes(schema: dict[str, Any]) -> bytes:
    rendered = json.dumps(
        schema,
        ensure_ascii=False,
        allow_nan=False,
        indent=2,
        sort_keys=True,
    )
    return f"{rendered}\n".encode()


def _assert_gold_firewall(schema: dict[str, Any], *, contract_name: str) -> None:
    serialized = canonical_json(schema)
    forbidden_names = {model_type.__name__ for model_type in SCORER_ONLY_SCHEMA_TYPES}
    forbidden_tokens = forbidden_names | {
        "scorer_namespace",
        "gold_projection_id",
        "expected_effect",
    }
    leaked = sorted(token for token in forbidden_tokens if token in serialized)
    if leaked:
        raise ValueError(
            f"schema {contract_name} crosses the scorer/model firewall: {', '.join(leaked)}"
        )


def _export_models() -> tuple[tuple[type[BaseModel], str], ...]:
    surfaces: dict[type[BaseModel], str] = {}
    for model_type in MODEL_VISIBLE_SCHEMA_TYPES:
        surfaces[model_type] = "model_interaction"
    for model_type in PUBLIC_SCHEMA_TYPES:
        surfaces.setdefault(model_type, "public_contract")
    return tuple(sorted(surfaces.items(), key=lambda item: item[0].__name__))


def emit_schemas(output_directory: Path = DEFAULT_OUTPUT_DIRECTORY) -> dict[str, Any]:
    """Atomically write deterministic schemas and their byte-hash manifest."""

    output_directory = Path(output_directory)
    schema_entries: list[dict[str, Any]] = []
    for model_type, surface in _export_models():
        schema = canonical_json_schema(model_type)
        _assert_gold_firewall(schema, contract_name=model_type.__name__)
        filename = f"{_snake_case(model_type.__name__)}.schema.json"
        content = _schema_bytes(schema)
        _atomic_write(output_directory / filename, content)
        schema_entries.append(
            {
                "bytes": len(content),
                "contract": model_type.__name__,
                "file": filename,
                "sha256": hashlib.sha256(content).hexdigest(),
                "surface": surface,
            }
        )

    manifest_payload: dict[str, Any] = {
        "generator": "scripts/generate_schemas.py",
        "schema_version": SCHEMA_VERSION,
        "schemas": schema_entries,
        "scorer_only_contracts_included": False,
    }
    manifest = {
        **manifest_payload,
        "manifest_hash": canonical_sha256(manifest_payload),
    }
    _atomic_write(output_directory / "schema_manifest.json", _schema_bytes(manifest))
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-directory",
        type=Path,
        default=DEFAULT_OUTPUT_DIRECTORY,
        help="directory for emitted schemas and schema_manifest.json",
    )
    arguments = parser.parse_args()
    manifest = emit_schemas(arguments.output_directory)
    print(f"wrote {len(manifest['schemas'])} schemas; manifest={manifest['manifest_hash']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
