from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
from decimal import Decimal
from pathlib import Path

import pytest

from story_projection_onto.public_release import load_public_entries, scan_public_entries


def _load_script():
    path = Path("scripts/build_interim_accounting_tables.py")
    spec = importlib.util.spec_from_file_location("build_interim_accounting_tables", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_checked_in_interim_accounting_tables_regenerate_exactly(tmp_path: Path) -> None:
    module = _load_script()
    module.build_tables(
        Path("artifacts/public/results/phase1_gpu_acceptance_v1_failed.json"),
        Path("artifacts/public/results/phase1_gpu_acceptance_v2_failed.json"),
        Path(
            "artifacts/public/results/"
            "fallback_gpu_acceptance_development_v1.json.controller-handoff.json"
        ),
        Path("artifacts/public/results/fallback_gpu_acceptance_development_v3.json"),
        Path(
            "artifacts/public/manifests/"
            "fallback_gpu_acceptance_development_v3_incident.json"
        ),
        Path(
            "artifacts/public/manifests/"
            "fallback_gpu_acceptance_development_v4_control_plane_incident.json"
        ),
        tmp_path,
    )
    for name in ("resource_accounting.csv", "failure_accounting.csv"):
        assert (tmp_path / name).read_bytes() == (Path("reports/tables") / name).read_bytes()


def test_checked_in_interim_rows_reconcile_exactly_without_double_counting() -> None:
    with Path("reports/tables/resource_accounting.csv").open(
        encoding="utf-8", newline=""
    ) as stream:
        resource_rows = list(csv.DictReader(stream))
    seconds = next(
        row
        for row in resource_rows
        if row["metric"] == "actual_allocated_gpu_time" and row["unit"] == "seconds"
    )
    assert seconds["scope"] == "study_through_fallback_v3"
    assert seconds["status"] == "observed"
    assert Decimal(seconds["value"]) == Decimal("815.215409")

    with Path("reports/tables/failure_accounting.csv").open(
        encoding="utf-8", newline=""
    ) as stream:
        failure_rows = list(csv.DictReader(stream))
    assert [row["run_id"] for row in failure_rows] == [
        "phase1-qwen3-14b-awq-v1",
        "phase1-qwen3-14b-awq-v2",
        "fallback-qwen3-8b-awq-development-v1",
        "fallback-qwen3-8b-awq-development-v3",
        "fallback-qwen3-8b-awq-development-v4",
    ]
    assert sum(Decimal(row["allocated_gpu_seconds"]) for row in failure_rows) == Decimal(
        "815.215409"
    )
    assert all(
        Decimal(row["allocated_gpu_seconds"]) > 0 for row in failure_rows[:-1]
    )
    assert all(row["completed_generation_calls"] == "0" for row in failure_rows)
    assert all(row["gate_passed"] == "false" for row in failure_rows)
    assert all(row["vllm_service_stopped"] == "true" for row in failure_rows)
    control_plane_row = failure_rows[-1]
    assert control_plane_row["model_repository"] == "not_applicable"
    assert control_plane_row["immutable_revision"] == "not_applicable"
    assert (
        control_plane_row["failure_type"]
        == "guardian_outer_result_path_identity_mismatch_before_readiness"
    )
    assert control_plane_row["allocated_gpu_seconds"] == "0.000000"
    assert "authorized retry slot" in control_plane_row["interpretation"]


def test_zero_gpu_incident_cannot_rebaseline_positive_gpu_attempts(
    tmp_path: Path,
) -> None:
    module = _load_script()
    source = Path(
        "artifacts/public/manifests/"
        "fallback_gpu_acceptance_development_v4_control_plane_incident.json"
    )
    forged_payload = json.loads(source.read_text(encoding="utf-8"))
    for ledger_position in ("before", "after"):
        summary = forged_payload["accounting"][ledger_position]["summary"]
        summary["total_allocated_microseconds"] += 1
        summary["by_kind_microseconds"]["failure"] += 1
    immutable = {
        key: value
        for key, value in forged_payload.items()
        if key != "manifest_sha256"
    }
    forged_payload["manifest_sha256"] = module._canonical_sha256(immutable)
    forged_incident = tmp_path / source.name
    forged_incident.write_text(
        json.dumps(
            forged_payload,
            ensure_ascii=False,
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    with pytest.raises(
        ValueError,
        match="control-plane incident changes cumulative GPU accounting",
    ):
        module.build_tables(
            Path("artifacts/public/results/phase1_gpu_acceptance_v1_failed.json"),
            Path("artifacts/public/results/phase1_gpu_acceptance_v2_failed.json"),
            Path(
                "artifacts/public/results/"
                "fallback_gpu_acceptance_development_v1.json.controller-handoff.json"
            ),
            Path("artifacts/public/results/fallback_gpu_acceptance_development_v3.json"),
            Path(
                "artifacts/public/manifests/"
                "fallback_gpu_acceptance_development_v3_incident.json"
            ),
            forged_incident,
            tmp_path / "tables",
        )


def test_interim_public_allowlist_hashes_and_includes_recovered_sources() -> None:
    module = _load_script()
    path = Path("reports/public_bundle_inputs.json")
    payload = json.loads(path.read_text(encoding="utf-8"))
    supplied_hash = payload.pop("manifest_sha256")
    assert supplied_hash == module._canonical_sha256(payload)
    entries = {
        row["source_relative_path"]: row for row in payload["entries"]
    }
    assert {
        "artifacts/public/results/"
        "fallback_gpu_acceptance_development_v1.json.controller-handoff.json",
        "artifacts/public/results/fallback_gpu_acceptance_development_v3.json",
        "artifacts/public/manifests/"
        "fallback_gpu_acceptance_development_v3_incident.json",
        "artifacts/public/manifests/"
        "fallback_gpu_acceptance_development_v4_control_plane_incident.json",
        "schemas/jsonschema/fallback_control_plane_incident.schema.json",
        "scripts/build_fallback_control_plane_incident.py",
        "src/story_projection_onto/fallback_control_plane_incident.py",
        "tests/unit/test_fallback_control_plane_incident.py",
    }.issubset(entries)
    assert "artifacts/public/manifests/environment.json" not in entries
    for relative_path, entry in entries.items():
        assert entry["release_class"] == "public"
        assert hashlib.sha256(Path(relative_path).read_bytes()).hexdigest() == entry["sha256"]
    scan_public_entries(Path("."), load_public_entries(path))
