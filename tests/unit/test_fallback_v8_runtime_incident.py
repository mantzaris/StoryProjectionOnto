from __future__ import annotations

import json
import shutil
import sqlite3
from collections.abc import Callable
from datetime import datetime
from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

from scripts.build_fallback_v8_runtime_incident import main as build_incident_main
from story_projection_onto.contracts import canonical_json, canonical_sha256
from story_projection_onto.fallback_v8_runtime_incident import (
    EXPECTED_PROTECTED_CONTENT_SHA256,
    EXPECTED_RUN_ROOT_INVENTORY_SHA256,
    EXPECTED_SERVICE_MICROSECONDS,
    EXPECTED_TERMINAL_MICROSECONDS,
    FALLBACK_V8_ROOT_CAUSE,
    FALLBACK_V8_RUNTIME_INCIDENT_KIND,
    FallbackV8RuntimeIncident,
    build_fallback_v8_runtime_incident,
    load_fallback_v8_runtime_incident,
    validate_fallback_v8_runtime_incident,
    write_fallback_v8_runtime_incident,
)
from story_projection_onto.public_release import scan_public_bytes

ROOT = Path(__file__).resolve().parents[2]
INCIDENT_PATH = (
    ROOT
    / "artifacts/public/manifests/"
    "fallback_gpu_acceptance_development_v8_runtime_incident.json"
)
RECOVERY_ROOT = (
    ROOT
    / "artifacts/restricted/recovery_validation/"
    "v8_launch_20260905T2016Z"
)


def _incident() -> FallbackV8RuntimeIncident:
    return load_fallback_v8_runtime_incident(INCIDENT_PATH)


def _copy_exact_evidence(tmp_path: Path) -> dict[str, Path]:
    if not RECOVERY_ROOT.is_dir():
        pytest.skip("restricted V8 evidence is intentionally absent from this checkout")
    public_manifests = tmp_path / "public-manifests"
    public_results = tmp_path / "public-results"
    recovery = tmp_path / "recovery"
    public_manifests.mkdir()
    public_results.mkdir()
    recovery.mkdir()
    sources = {
        "source_association": ROOT
        / "artifacts/public/manifests/source_tree_fallback_second_recovery_v8.association.json",
        "authorization_overlay": ROOT
        / "artifacts/restricted/fallback-second-recovery-v8.authorized.json",
        "preflight": ROOT
        / "artifacts/public/manifests/fallback_gpu_acceptance_development_v8.preflight.json",
        "controller_handoff": ROOT
        / "artifacts/public/results/"
        "fallback_gpu_acceptance_development_v8.json.controller-handoff.json",
        "run_result": ROOT
        / "artifacts/public/results/fallback_gpu_acceptance_development_v8.json",
        "orphan_cleanup": ROOT
        / "artifacts/public/results/"
        "fallback_gpu_acceptance_development_v8.json.orphan-cleanup.json",
    }
    copied: dict[str, Path] = {}
    for name, source in sources.items():
        parent = public_manifests if name in {
            "source_association",
            "preflight",
        } else public_results
        destination = parent / source.name
        shutil.copyfile(source, destination)
        copied[name] = destination
    run_root = tmp_path / "run-root"
    shutil.copytree(ROOT / "artifacts/restricted/fallback-development-v8", run_root)
    copied["run_root"] = run_root
    for basename in (
        "manual_stop_intent.json",
        "manual_stop_outcome.json",
        "terminal_accounting_receipt.json",
        "manual_stop_v8.py",
        "recover_v8_ledger.py",
        "phase1_acceptance.before_v8.local.sqlite",
        "phase1_acceptance.before_recovery.sqlite",
        "phase1_acceptance.after_recovery.sqlite",
    ):
        shutil.copyfile(RECOVERY_ROOT / basename, recovery / basename)
    copied.update(
        {
            "recovery_root": recovery,
            "before_v8": recovery / "phase1_acceptance.before_v8.local.sqlite",
            "before_recovery": recovery / "phase1_acceptance.before_recovery.sqlite",
            "terminal": recovery / "phase1_acceptance.after_recovery.sqlite",
        }
    )
    return copied


def _build_from_copy(paths: dict[str, Path]) -> FallbackV8RuntimeIncident:
    return build_fallback_v8_runtime_incident(
        run_id="fallback-qwen3-8b-awq-development-v8",
        source_association_path=paths["source_association"],
        authorization_overlay_path=paths["authorization_overlay"],
        preflight_path=paths["preflight"],
        controller_handoff_path=paths["controller_handoff"],
        run_result_path=paths["run_result"],
        orphan_cleanup_path=paths["orphan_cleanup"],
        restricted_run_root=paths["run_root"],
        manual_recovery_root=paths["recovery_root"],
        ledger_before_v8_path=paths["before_v8"],
        ledger_before_manual_recovery_path=paths["before_recovery"],
        terminal_ledger_path=paths["terminal"],
        audited_at=datetime.fromisoformat("2026-09-05T21:07:57.812706+00:00"),
    )


def _rewrite_self_hashed_json(
    path: Path, mutate: Callable[[dict[str, Any]], None]
) -> None:
    value = json.loads(path.read_text(encoding="utf-8"))
    mutate(value)
    value["manifest_sha256"] = canonical_sha256(
        {key: item for key, item in value.items() if key != "manifest_sha256"}
    )
    path.write_text(canonical_json(value) + "\n", encoding="utf-8")


def test_checked_in_v8_incident_records_exact_terminal_outcome() -> None:
    incident = _incident()

    assert incident.kind == FALLBACK_V8_RUNTIME_INCIDENT_KIND
    assert incident.failure_sequence.root_cause == FALLBACK_V8_ROOT_CAUSE
    assert incident.accounting.delta.inference_model_calls == 0
    assert incident.accounting.delta.accepted_outputs == 0
    assert incident.accounting.service.service_microseconds == EXPECTED_SERVICE_MICROSECONDS
    assert (
        incident.accounting.terminal.summary.total_allocated_microseconds
        == EXPECTED_TERMINAL_MICROSECONDS
    )
    assert (
        incident.accounting.terminal.protected_content_sha256
        == EXPECTED_PROTECTED_CONTENT_SHA256
    )
    assert incident.manual_recovery.signals_sent == ("SIGTERM",)
    assert incident.terminal_state.resume_allowed is False
    assert incident.manifest_sha256 == canonical_sha256(
        incident.model_dump(mode="python", exclude={"manifest_sha256"})
    )


def test_public_v8_incident_contains_no_private_payloads() -> None:
    serialized = canonical_json(_incident())

    for forbidden in (
        "/workspace/",
        "127.0.0.1",
        "Traceback",
        "launch_command",
        '"service_instance_token_sha256"',
        "scorer_namespace",
        "gold_projection_id",
    ):
        assert forbidden not in serialized
    scan_public_bytes(
        INCIDENT_PATH.read_bytes(),
        relative_path=(
            "artifacts/public/manifests/"
            "fallback_gpu_acceptance_development_v8_runtime_incident.json"
        ),
    )


def test_v8_incident_rejects_manifest_and_accounting_tampering() -> None:
    incident = _incident()
    value = incident.model_dump(mode="python")
    value["terminal_state"]["accepted_output_count"] = 1
    with pytest.raises(ValidationError):
        FallbackV8RuntimeIncident.model_validate(value)

    value = incident.model_dump(mode="python")
    value["accounting"]["terminal"]["protected_content_sha256"] = "0" * 64
    immutable = {key: item for key, item in value.items() if key != "manifest_sha256"}
    value["manifest_sha256"] = canonical_sha256(immutable)
    with pytest.raises(ValidationError, match="ledger identities"):
        FallbackV8RuntimeIncident.model_validate(value)


@pytest.mark.parametrize(
    "mutate, message",
    (
        (
            lambda value: value["source_and_authorization"].__setitem__(
                "source_git_commit", "0" * 40
            ),
            "source, authorization, or preflight identity",
        ),
        (
            lambda value: value["public_evidence"]["run_result"].__setitem__(
                "size_bytes", 9218
            ),
            "public result identity",
        ),
        (
            lambda value: value["restricted_evidence"].__setitem__(
                "run_root_inventory_sha256", "0" * 64
            ),
            "Input should be",
        ),
        (
            lambda value: value["manual_recovery"].__setitem__(
                "signals_sent", ["SIGTERM", "SIGTERM"]
            ),
            "Tuple should have at most 1 item",
        ),
        (
            lambda value: value["accounting"]["terminal"]["summary"][
                "by_kind_microseconds"
            ].update(
                {
                    "gpu_session_start": 441_721_081,
                    "service_overhead": 1_491_497_669,
                }
            ),
            "ledger binding or summary",
        ),
    ),
)
def test_v8_contract_rejects_rehashed_exact_identity_substitutions(
    mutate: Callable[[dict[str, Any]], None], message: str
) -> None:
    value = _incident().model_dump(mode="python")
    mutate(value)
    value["manifest_sha256"] = canonical_sha256(
        {key: item for key, item in value.items() if key != "manifest_sha256"}
    )
    with pytest.raises(ValidationError, match=message):
        FallbackV8RuntimeIncident.model_validate(value)


def test_v8_append_only_writer_and_expected_identity_validator(tmp_path: Path) -> None:
    incident = _incident()
    destination = tmp_path / "incident.json"
    write_fallback_v8_runtime_incident(destination, incident)
    write_fallback_v8_runtime_incident(destination, incident)
    assert load_fallback_v8_runtime_incident(destination) == incident
    validate_fallback_v8_runtime_incident(
        destination,
        expected_source_association_manifest_sha256=(
            incident.source_and_authorization.source_association.manifest_sha256
        ),
        expected_overlay_manifest_sha256=(
            incident.source_and_authorization.authorization_overlay.manifest_sha256
        ),
        expected_preflight_manifest_sha256=(
            incident.source_and_authorization.execution_preflight.manifest_sha256
        ),
        expected_terminal_ledger_file_sha256=(
            incident.accounting.terminal.file_sha256
        ),
    )
    destination.write_text("{}\n", encoding="utf-8")
    with pytest.raises(FileExistsError, match="already differs"):
        write_fallback_v8_runtime_incident(destination, incident)


def test_v8_schema_is_strict_and_public_safe() -> None:
    schema = FallbackV8RuntimeIncident.model_json_schema(mode="validation")
    assert schema["additionalProperties"] is False
    assert all(
        definition.get("additionalProperties") is False
        for definition in schema.get("$defs", {}).values()
        if definition.get("type") == "object"
    )
    serialized = canonical_json(schema)
    assert '"absolute_path"' not in serialized
    assert '"log_text"' not in serialized
    assert '"command"' not in serialized
    assert '"service_instance_token_sha256"' not in serialized
    assert EXPECTED_RUN_ROOT_INVENTORY_SHA256 in serialized
    assert '"const":885933' in serialized
    assert '"const":"SIGTERM"' in serialized


def test_fixture_copy_rebuild_is_immutable_and_creates_no_sqlite_sidecars(
    tmp_path: Path,
) -> None:
    paths = _copy_exact_evidence(tmp_path)
    ledger_hashes = {
        name: paths[name].read_bytes()
        for name in ("before_v8", "before_recovery", "terminal")
    }
    sidecars = [
        Path(f"{paths[name]}{suffix}")
        for name in ("before_v8", "before_recovery", "terminal")
        for suffix in ("-wal", "-shm", "-journal")
    ]
    assert not any(path.exists() for path in sidecars)

    assert _build_from_copy(paths) == _incident()

    assert all(paths[name].read_bytes() == contents for name, contents in ledger_hashes.items())
    assert not any(path.exists() for path in sidecars)


def test_builder_rejects_extra_run_root_artifact(tmp_path: Path) -> None:
    paths = _copy_exact_evidence(tmp_path)
    (paths["run_root"] / "substituted.json").write_text("{}\n", encoding="utf-8")

    with pytest.raises(ValueError, match="exact preserved artifact set"):
        _build_from_copy(paths)


def test_builder_rejects_symlinked_empty_run_root_lock(tmp_path: Path) -> None:
    paths = _copy_exact_evidence(tmp_path)
    external_lock = tmp_path / "external-empty-lock"
    external_lock.touch()
    lock = paths["run_root"] / "checkpoint.json.guardian.lock"
    lock.unlink()
    lock.symlink_to(external_lock)

    with pytest.raises(ValueError, match="cannot have symlink ancestry"):
        _build_from_copy(paths)


def test_builder_rejects_rehashed_public_execution_substitution(tmp_path: Path) -> None:
    paths = _copy_exact_evidence(tmp_path)

    def mutate(value: dict[str, Any]) -> None:
        value["execution_identity"]["source_tree_sha256"] = "0" * 64

    _rewrite_self_hashed_json(paths["run_result"], mutate)
    with pytest.raises(ValueError, match="public execution identity"):
        _build_from_copy(paths)


@pytest.mark.parametrize(
    ("field", "replacement"),
    (
        ("controller_pid", 889_999),
        ("controller_output", "/substituted/result.json"),
        ("registered_at", "2026-09-05T20:19:00+00:00"),
    ),
)
def test_builder_rejects_rehashed_controller_identity_or_chronology(
    tmp_path: Path, field: str, replacement: object
) -> None:
    paths = _copy_exact_evidence(tmp_path)
    receipt = paths["run_root"] / "checkpoint.json.internal-controller-000002.json"
    _rewrite_self_hashed_json(receipt, lambda value: value.__setitem__(field, replacement))

    with pytest.raises(ValueError, match="controller receipt chain"):
        _build_from_copy(paths)


@pytest.mark.parametrize("target", ("lease", "member", "receipt"))
def test_builder_rejects_rehashed_manual_recovery_substitution(
    tmp_path: Path, target: str
) -> None:
    paths = _copy_exact_evidence(tmp_path)
    if target == "receipt":
        receipt = paths["recovery_root"] / "terminal_accounting_receipt.json"
        _rewrite_self_hashed_json(
            receipt,
            lambda value: value["after_summary"].__setitem__("event_count", 8),
        )
    else:
        intent = paths["recovery_root"] / "manual_stop_intent.json"
        if target == "lease":
            _rewrite_self_hashed_json(
                intent, lambda value: value.__setitem__("lease_file_sha256", "0" * 64)
            )
        else:
            _rewrite_self_hashed_json(
                intent,
                lambda value: value["member_identities"][1].__setitem__(
                    "start_ticks", 557_683_181
                ),
            )

    with pytest.raises(ValueError, match=r"stop intent|accounting receipt"):
        _build_from_copy(paths)


def test_builder_rejects_count_neutral_protected_sqlite_tamper(tmp_path: Path) -> None:
    paths = _copy_exact_evidence(tmp_path)
    terminal = paths["terminal"]
    connection = sqlite3.connect(terminal)
    try:
        count_before = connection.execute("SELECT COUNT(*) FROM attempts").fetchone()[0]
        connection.execute("DROP TRIGGER attempts_reject_update")
        connection.execute("UPDATE attempts SET input_hash = ?", ("0" * 64,))
        connection.commit()
        count_after = connection.execute("SELECT COUNT(*) FROM attempts").fetchone()[0]
    finally:
        connection.close()
    assert count_before == count_after == 1

    with pytest.raises(ValueError, match=r"ledger binding or summary|ledger identities"):
        _build_from_copy(paths)


def test_real_restricted_evidence_rebuilds_identical_incident(tmp_path: Path) -> None:
    if not RECOVERY_ROOT.is_dir():
        pytest.skip("restricted V8 evidence is intentionally absent from this checkout")
    output = tmp_path / "rebuilt.json"
    arguments = [
        "--source-association",
        str(
            ROOT
            / "artifacts/public/manifests/"
            "source_tree_fallback_second_recovery_v8.association.json"
        ),
        "--authorization-overlay",
        str(ROOT / "artifacts/restricted/fallback-second-recovery-v8.authorized.json"),
        "--preflight",
        str(
            ROOT
            / "artifacts/public/manifests/"
            "fallback_gpu_acceptance_development_v8.preflight.json"
        ),
        "--controller-handoff",
        str(
            ROOT
            / "artifacts/public/results/"
            "fallback_gpu_acceptance_development_v8.json.controller-handoff.json"
        ),
        "--run-result",
        str(
            ROOT
            / "artifacts/public/results/"
            "fallback_gpu_acceptance_development_v8.json"
        ),
        "--orphan-cleanup",
        str(
            ROOT
            / "artifacts/public/results/"
            "fallback_gpu_acceptance_development_v8.json.orphan-cleanup.json"
        ),
        "--restricted-run-root",
        str(ROOT / "artifacts/restricted/fallback-development-v8"),
        "--manual-recovery-root",
        str(RECOVERY_ROOT),
        "--ledger-before-v8",
        str(RECOVERY_ROOT / "phase1_acceptance.before_v8.local.sqlite"),
        "--ledger-before-manual-recovery",
        str(RECOVERY_ROOT / "phase1_acceptance.before_recovery.sqlite"),
        "--terminal-ledger",
        str(RECOVERY_ROOT / "phase1_acceptance.after_recovery.sqlite"),
        "--audited-at",
        _incident().audited_at.isoformat(),
        "--output",
        str(output),
    ]
    assert build_incident_main(arguments) == 0
    assert output.read_bytes() == INCIDENT_PATH.read_bytes()
