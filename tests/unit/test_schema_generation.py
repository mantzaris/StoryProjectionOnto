from __future__ import annotations

import hashlib
import json
from pathlib import Path

from scripts.generate_schemas import OPERATIONAL_SCHEMA_TYPES, emit_schemas
from story_projection_onto.contracts import (
    MODEL_VISIBLE_SCHEMA_TYPES,
    PUBLIC_SCHEMA_TYPES,
    SCORER_ONLY_SCHEMA_TYPES,
    canonical_sha256,
)


def emitted_files(directory: Path) -> dict[str, bytes]:
    return {path.name: path.read_bytes() for path in sorted(directory.glob("*.json"))}


def test_schema_generation_is_deterministic_and_hash_manifested(tmp_path: Path) -> None:
    first_directory = tmp_path / "first"
    second_directory = tmp_path / "second"
    first_manifest = emit_schemas(first_directory)
    second_manifest = emit_schemas(second_directory)

    assert first_manifest == second_manifest
    assert emitted_files(first_directory) == emitted_files(second_directory)
    expected_count = len(
        set(MODEL_VISIBLE_SCHEMA_TYPES) | set(PUBLIC_SCHEMA_TYPES) | set(OPERATIONAL_SCHEMA_TYPES)
    )
    assert len(first_manifest["schemas"]) == expected_count

    for entry in first_manifest["schemas"]:
        content = (first_directory / entry["file"]).read_bytes()
        assert entry["sha256"] == hashlib.sha256(content).hexdigest()
        assert entry["bytes"] == len(content)

    manifest_payload = {
        key: value for key, value in first_manifest.items() if key != "manifest_hash"
    }
    assert first_manifest["manifest_hash"] == canonical_sha256(manifest_payload)
    assert not list(first_directory.glob("*.tmp"))

    operational_entries = [
        entry
        for entry in first_manifest["schemas"]
        if entry["surface"] == "public_operational_contract"
    ]
    assert [entry["file"] for entry in operational_entries] == [
        "fallback_control_plane_incident.schema.json",
        "fallback_v5_control_plane_incident.schema.json",
        "fallback_v6_control_plane_incident.schema.json",
        "fallback_v7_runtime_incident.schema.json",
        "fallback_v8_lease_repair_receipt.schema.json",
        "fallback_v8_runtime_incident.schema.json",
    ]


def test_emitted_schemas_exclude_all_scorer_only_contracts(tmp_path: Path) -> None:
    manifest = emit_schemas(tmp_path)
    emitted_contracts = {entry["contract"] for entry in manifest["schemas"]}
    scorer_contracts = {model_type.__name__ for model_type in SCORER_ONLY_SCHEMA_TYPES}
    assert emitted_contracts.isdisjoint(scorer_contracts)
    assert manifest["scorer_only_contracts_included"] is False

    schema_text = "\n".join(
        path.read_text(encoding="utf-8") for path in sorted(tmp_path.glob("*.schema.json"))
    )
    assert not any(contract_name in schema_text for contract_name in scorer_contracts)
    assert '"scorer_namespace"' not in schema_text
    assert '"gold_projection_id"' not in schema_text


def test_generation_atomically_replaces_an_existing_schema(tmp_path: Path) -> None:
    manifest = emit_schemas(tmp_path)
    target = tmp_path / manifest["schemas"][0]["file"]
    target.write_text("not-json", encoding="utf-8")

    regenerated = emit_schemas(tmp_path)
    parsed = json.loads(target.read_text(encoding="utf-8"))
    regenerated_entry = next(
        entry for entry in regenerated["schemas"] if entry["file"] == target.name
    )
    assert isinstance(parsed, dict)
    assert regenerated_entry["sha256"] == hashlib.sha256(target.read_bytes()).hexdigest()
    assert not list(tmp_path.glob(f".{target.name}.*.tmp"))
