from __future__ import annotations

import copy
import fcntl
import hashlib
import importlib
import json
import os
import shutil
import signal
import sqlite3
import subprocess
import sys
import time
from collections.abc import Mapping
from contextlib import suppress
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace
from typing import cast

import pytest

import story_projection_onto.fallback_acceptance as fallback_acceptance_module
from story_projection_onto.contracts import (
    RUNTIME_STRUCTURAL_ONLY_DIAGNOSTIC,
    ConditionName,
    OutputBudgets,
    PrequeryPreparationBinding,
    QueryAccessEvent,
    RunOutcome,
    canonical_json,
    canonical_sha256,
)
from story_projection_onto.development_continuation import (
    create_production_development_adopter,
)
from story_projection_onto.development_runtime import (
    DevelopmentCallKind,
    DevelopmentCallManifest,
    DevelopmentExecutionResult,
    DevelopmentForecastResult,
    DevelopmentGateResult,
    DevelopmentITTRecord,
    DevelopmentPhase,
    DevelopmentPrequeryInputs,
    FixedSchemaDerivationPlan,
    FixedSchemaDerivationReceipt,
    ForecastControl,
    InjectedLiveDevelopmentService,
    LiveServiceIdentity,
    RequestStartState,
    UnitPrequeryBinding,
    load_development_call_manifest,
)
from story_projection_onto.experiment import (
    AllocatedGPUMeter,
    ResourceLimits,
    StorageAllocationPlan,
)
from story_projection_onto.fallback_acceptance import (
    AMENDED_FALLBACK_STARTUP_WATCHDOG_SECONDS,
    REPAIR_TRIGGER_RULE,
    SECOND_RECOVERY_EVIDENCE_BRIDGE_IMPLEMENTATION_PATH,
    SECOND_RECOVERY_EVIDENCE_BRIDGE_REGRESSION_TEST_PATH,
    SECOND_RECOVERY_EVIDENCE_BRIDGE_SECTION_NAME,
    SECOND_RECOVERY_OVERLAY_KIND,
    SECOND_RECOVERY_POST_TWO_FIX_C0_IMPLEMENTATION_SHA256,
    SECOND_RECOVERY_POST_TWO_FIX_C0_REGRESSION_TEST_SHA256,
    SECOND_RECOVERY_RETRY_CALL_ID,
    SECOND_RECOVERY_V3_C1_IMPLEMENTATION_SHA256,
    SECOND_RECOVERY_V3_DECODER_SCHEMA_SHA256,
    SECOND_RECOVERY_V3_INCIDENT_FILE_SHA256,
    SECOND_RECOVERY_V3_INCIDENT_MANIFEST_SHA256,
    SECOND_RECOVERY_V3_REQUEST_SHA256,
    SECOND_RECOVERY_V3_RESULT_FILE_SHA256,
    SECOND_RECOVERY_V3_RESULT_MANIFEST_SHA256,
    SECOND_RECOVERY_V3_RETRY_AMENDMENT_SHA256,
    SECOND_RECOVERY_V3_RUN_ID,
    SECOND_RECOVERY_V4_C1_CONDITION_PATHWAY_TEST_SHA256,
    SECOND_RECOVERY_V4_C1_DEVELOPMENT_ASSESSMENT_TEST_SHA256,
    SECOND_RECOVERY_V4_C1_IMPLEMENTATION_SHA256,
    DevelopmentAdopterRegistration,
    DevelopmentContinuationBootstrap,
    DevelopmentContinuationHandoff,
    FallbackAcceptanceRunner,
    FallbackCapabilityBehaviorError,
    PreparedDevelopmentContinuation,
    build_fallback_acceptance_request,
    build_fallback_repair_request,
    build_second_fallback_recovery_overlay,
    fallback_pilot_calls,
    fallback_plan_manifest,
    main,
    parse_arguments,
    pre_fallback_gpu_accounting_baseline,
    validate_fallback_service_retry_amendment,
    validate_pre_fallback_gpu_accounting,
    validate_second_fallback_recovery_overlay,
    validate_source_association,
)
from story_projection_onto.gpu_runtime import (
    FALLBACK_MODEL_REPOSITORY,
    FALLBACK_MODEL_REVISION,
    FALLBACK_SERVED_MODEL_NAME,
    ChatMessage,
    GenerationResult,
    GuidedJSONRequest,
    ResourceWatchdog,
    RuntimeTransportError,
    ServiceState,
    ServiceUptime,
    TokenizerManifest,
    VLLMLaunchConfiguration,
    VLLMService,
)
from story_projection_onto.ledger_verify import verify_ledger
from story_projection_onto.manifest import build_source_association, build_source_manifest
from story_projection_onto.model_gate import FallbackModelPolicy
from story_projection_onto.phase1_acceptance import validate_acceptance_generation
from story_projection_onto.phase1_legacy_provenance import (
    CERTIFICATE_RELATIVE_PATH,
    Phase1LegacyEvidenceProvenanceBridge,
)
from story_projection_onto.public_release import scan_public_bytes
from story_projection_onto.store import (
    ArtifactIntegrityError,
    ArtifactStore,
    AttemptKind,
    BlobStore,
    FailureKind,
    GpuEventKind,
    GpuServiceJournalState,
    GpuSummary,
    JobState,
    Ledger,
    ReadOnlyLedger,
    ReleaseClass,
    SemanticAssessmentScope,
    StorageBudget,
    StorageBudgetExceeded,
    StoragePreflight,
    ValidationStatus,
)

ROOT = Path(__file__).resolve().parents[2]
HASH_A = "a" * 64
HASH_B = "b" * 64


def legacy_provenance_bridge() -> Phase1LegacyEvidenceProvenanceBridge:
    return Phase1LegacyEvidenceProvenanceBridge.load(ROOT)


def with_legacy_source_hashes(value: Mapping[str, object]) -> dict[str, object]:
    """Model the source-only fields now required from non-Fixed fallback output."""

    result = copy.deepcopy(dict(value))
    bridge = legacy_provenance_bridge()
    by_evidence_id = {
        binding.evidence_id: binding.source.source_artifact_hash
        for binding in bridge.certificate.evidence_bindings
    }
    graph = cast(dict[str, object], result["instance_graph"])
    assertions = cast(list[dict[str, object]], graph["assertions"])
    for assertion in assertions:
        for provenance in cast(list[dict[str, object]], assertion["provenance"]):
            provenance["source_artifact_hash"] = by_evidence_id[
                cast(str, provenance["evidence_id"])
            ]
    return result


class FakeTokenizer:
    def encode(self, text: str, **kwargs: object) -> list[int]:
        del kwargs
        return list(range(len(text.split())))

    def apply_chat_template(self, conversation: object, **kwargs: object) -> list[int]:
        assert kwargs["enable_thinking"] is False
        assert isinstance(conversation, (tuple, list))
        words = sum(len(str(row["content"]).split()) for row in conversation)
        # The packing contract explicitly records the positive wrapper delta.
        return list(range(words + 8))


def fallback_tokenizer_manifest() -> TokenizerManifest:
    return TokenizerManifest(
        schema_version="1.0.0",
        repository=FALLBACK_MODEL_REPOSITORY,
        revision=FALLBACK_MODEL_REVISION,
        tokenizer_class="tests.FakeTokenizer",
        tokenizer_revision=FALLBACK_MODEL_REVISION,
        tokenizer_file_sha256=(("tokenizer.json", HASH_A),),
        eos_token_id=7,
        end_of_turn_token_ids=(8,),
        stop_token_ids=(7, 8),
        chat_template_sha256=HASH_A,
        nonthinking_probe_sha256=HASH_B,
        nonthinking_probe_token_count=10,
        local_files_only=True,
        trust_remote_code=False,
        enable_thinking=False,
    )


def v3_fallback_tokenizer_manifest() -> TokenizerManifest:
    predecessor = json.loads(
        (
            ROOT
            / "artifacts/public/results/fallback_gpu_acceptance_development_v3.json"
        ).read_text(encoding="utf-8")
    )
    payload = copy.deepcopy(predecessor["runtime"]["tokenizer"])
    expected_manifest_sha256 = payload.pop("manifest_sha256")
    payload["tokenizer_file_sha256"] = tuple(
        tuple(row) for row in payload["tokenizer_file_sha256"]
    )
    payload["end_of_turn_token_ids"] = tuple(payload["end_of_turn_token_ids"])
    payload["stop_token_ids"] = tuple(payload["stop_token_ids"])
    manifest = TokenizerManifest(**payload)
    assert manifest.manifest_sha256 == expected_manifest_sha256
    return manifest


def test_fallback_plan_binds_exact_calls_reserves_and_no_primary_block() -> None:
    plan = fallback_plan_manifest(ROOT)
    calls = plan["calls"]

    assert plan["base_call_count"] == 4
    assert plan["maximum_repair_call_count"] == 1
    assert plan["maximum_inference_seconds"] == 720
    assert plan["service_start_event_count"] == 1
    assert plan["process_restart_event_count"] == 0
    assert plan["controller_invocation_count"] == 2
    assert plan["supervising_orchestrator_required"] is True
    assert plan["internal_controller_stages_require_live_guard"] is True
    assert plan["append_only_orchestration_invocation_required"] is True
    assert plan["independent_lease_guardian_required_before_service_start"] is True
    assert plan["guardian_binds_hard_stop_and_exact_execution_arguments"] is True
    assert plan["durable_preexec_service_identity_gate_required"] is True
    assert plan["fresh_controller_outputs_must_be_absent"] is True
    assert plan["resume_requires_exact_invocation_guard_and_stage_bindings"] is True
    assert plan["normal_exit_requires_verified_shutdown"] is True
    assert plan["pre_fallback_ledger_must_exactly_reproduce_primary_result"] is True
    assert plan["fresh_gpu_ledger_forbidden"] is True
    assert plan["normal_acceptance_block"]["executed"] is False
    second_recovery = plan["second_recovery_overlay_support"]
    assert second_recovery["overlay_schema_version"] == "1.7.0"
    assert second_recovery["authorized_recovery_run_id"] == (
        fallback_acceptance_module.SECOND_RECOVERY_V9_RUN_ID
    )
    assert second_recovery["authorized_source_revision"] == (
        fallback_acceptance_module.SECOND_RECOVERY_V9_SOURCE_REVISION
    )
    assert second_recovery["additional_service_start_events"] == 3
    assert second_recovery["protected_resource_sample_drain_seconds"] == 120.0
    assert second_recovery["protected_process_shutdown_seconds"] == 60
    assert second_recovery["protected_hard_stop_reserve_seconds"] == 180.0
    assert second_recovery["intervening_v7_runtime_incident_required"] is True
    assert second_recovery["intervening_v8_runtime_incident_required"] is True
    assert second_recovery["intervening_v8_lease_repair_receipt_required"] is True
    assert second_recovery["terminal_v4_through_v8_runs_must_not_resume"] is True
    assert second_recovery["essential_recovery_contingency"] == {
        "service_start_count": 1,
        "inference_attempt_count": 0,
        "development_call_count": 0,
        "scheduled_admission_before_service_start": False,
        "hard_contingency_admission_before_service_start": True,
        "post_start_inference_requires_fresh_scheduled_admission": True,
        "post_start_development_requires_fresh_scheduled_admission": True,
        "hard_limit_remains_strict": True,
    }
    assert second_recovery["intervening_zero_gpu_control_plane_incident_required"] is True
    assert second_recovery["intervening_v4_control_plane_incident_required"] is True
    assert second_recovery["intervening_v5_control_plane_incident_required"] is True
    assert second_recovery["intervening_v6_control_plane_incident_required"] is True
    assert second_recovery["evidence_provenance_bridge_binding_required"] is True
    assert second_recovery["retry_wire_delta_scope"] == (
        "guided_schema_schema_derived_runtime_hashes_and_hash_bound_provenance_only"
    )
    assert second_recovery["historical_c0_two_fix_layer_is_constant_bound"] is True
    assert second_recovery["historical_c1_semantic_status_layer_is_constant_bound"] is True
    assert second_recovery["immutable_v3_fallback_input_comparison_required"] is True
    assert second_recovery["whole_wire_payload_byte_identity_to_failed_v3_claimed"] is False
    assert second_recovery["projection_dependency_correction_required"] is True
    assert second_recovery["concurrent_integrity_disclosure_required"] is True
    assert second_recovery["predecessor_accepted_output_count"] == 0
    assert second_recovery["predecessor_base_output_count"] == 0
    assert second_recovery["predecessor_development_output_count"] == 0
    assert second_recovery["integrity_correction_model_output_count"] == 0
    assert second_recovery["integrity_correction_gpu_call_count"] == 0
    assert second_recovery["c1_prequery_construction_acceptance_unchanged"] is True
    assert second_recovery[
        "final_projection_selection_feasibility_changed_before_outputs"
    ] is True
    assert second_recovery[
        "final_projection_validation_targeting_changed_before_outputs"
    ] is True
    assert second_recovery["future_projection_and_analysis_outputs_may_change"] is True
    continuation = plan["development_continuation"]
    assert continuation["adopter_protocol"] == "fallback-live-development-adopter-v1"
    assert continuation["selected_model_freeze_bootstrap_requires_development_completion"] is False
    assert continuation["adopter_model_load_start_shutdown_counts"] == [0, 0, 0]
    assert continuation["fallback_owner_performs_final_shutdown"] is True
    assert continuation["callback_returns"] == "canonical DevelopmentExecutionResult"
    assert continuation["fallback_owner_derives_continuation_receipt"] is True
    assert continuation["micro_pilot_c1_operator_scope"] == ("representative_grounded_subset")
    assert continuation["complete_c1_operator_inventory_gate"] == (
        "DevelopmentScientificAssessment.c1_all_construction_operators_exercised"
    )
    assert continuation["standalone_cli_gpu_execution_enabled"] is True
    assert continuation["standalone_registered_factory"] == (
        "story_projection_onto.development_continuation:create_production_development_adopter"
    )
    module_name, factory_name = continuation["standalone_registered_factory"].split(":")
    assert callable(getattr(importlib.import_module(module_name), factory_name))
    implementation_paths = {row["path"] for row in plan["implementation_files"]}
    assert {
        "configs/study/development_construction.json",
        "src/story_projection_onto/development_adapter.py",
        "src/story_projection_onto/development_artifacts.py",
        "src/story_projection_onto/development_assessment_bridge.py",
        "src/story_projection_onto/development_continuation.py",
        "src/story_projection_onto/development_execution.py",
        "src/story_projection_onto/development_runtime.py",
        "src/story_projection_onto/experiment.py",
        "src/story_projection_onto/fallback_v7_lease_repair.py",
        "src/story_projection_onto/fallback_v7_runtime_incident.py",
        "src/story_projection_onto/fallback_v8_lease_repair.py",
        "src/story_projection_onto/fallback_v8_runtime_incident.py",
        "src/story_projection_onto/gpu_runtime.py",
        "src/story_projection_onto/phase1_acceptance.py",
        "src/story_projection_onto/conditions/c0.py",
        "src/story_projection_onto/conditions/c1.py",
        "src/story_projection_onto/conditions/c2.py",
        "src/story_projection_onto/conditions/fixed_select.py",
        "tests/unit/test_fallback_v7_lease_repair.py",
        "tests/unit/test_fallback_v7_runtime_incident.py",
        "tests/unit/test_fallback_v8_lease_repair.py",
        "tests/unit/test_fallback_v8_runtime_incident.py",
        "tests/unit/test_experiment.py",
        "tests/unit/test_gpu_runtime.py",
        CERTIFICATE_RELATIVE_PATH,
        SECOND_RECOVERY_EVIDENCE_BRIDGE_IMPLEMENTATION_PATH,
        SECOND_RECOVERY_EVIDENCE_BRIDGE_REGRESSION_TEST_PATH,
    }.issubset(implementation_paths)
    bridge = legacy_provenance_bridge()
    plan_bridge = plan["legacy_evidence_provenance_bridge"]
    assert plan_bridge["certificate_manifest_sha256"] == bridge.manifest_sha256
    assert plan_bridge["certificate_file_sha256"] == bridge.certificate_file_sha256
    assert plan_bridge["model_visible_section_name"] == (
        SECOND_RECOVERY_EVIDENCE_BRIDGE_SECTION_NAME
    )
    assert plan_bridge["required_for_base_resume_and_repair_validation"] is True
    assert plan_bridge["source_critical_validation_fields"] == [
        "evidence_id",
        "locator",
        "source_artifact_hash",
        "confidence_ceiling",
    ]
    assert [row["watchdog_seconds"] for row in calls] == [240, 150, 150, 90]
    assert [row["reserve_call_class"] for row in calls] == [
        "reserve_long",
        "reserve_standard",
        "reserve_standard",
        "reserve_short",
    ]
    all_construction_operators = {
        "abstraction",
        "contextual_type",
        "epistemic_qualification",
        "event_reification",
        "include_exclude",
        "merge",
        "rare_preservation",
        "schema_relation",
        "split",
        "temporal_qualification",
    }
    assert set(calls[0]["required_constructive_operators"]) == {
        "event_reification",
        "merge",
        "rare_preservation",
        "schema_relation",
        "temporal_qualification",
    }
    assert set(calls[0]["required_constructive_operators"]) < all_construction_operators
    assert (
        set(calls[1]["required_constructive_operators"])
        | set(calls[2]["required_constructive_operators"])
    ) == all_construction_operators
    assert plan["repair"]["watchdog_seconds"] == 90
    assert plan["repair"]["trigger_rule"] == REPAIR_TRIGGER_RULE


def test_fallback_request_uses_exact_candidate_and_complete_repair_pack() -> None:
    policy = FallbackModelPolicy.load(ROOT / "configs/study/fallback_model.json")
    call = fallback_pilot_calls(policy)[1]
    tokenizer = FakeTokenizer()
    tokenizer_manifest = fallback_tokenizer_manifest()
    bridge = legacy_provenance_bridge()
    base = build_fallback_acceptance_request(
        root=ROOT,
        call=call,
        tokenizer=tokenizer,
        tokenizer_manifest=tokenizer_manifest,
        legacy_provenance_bridge=bridge,
    )
    invalid = json.loads((ROOT / "tests/fixtures/phase1/c2_query_output.json").read_text())
    repair = build_fallback_repair_request(
        root=ROOT,
        base_call=call.acceptance_call(),
        base_request=base,
        invalid_draft=invalid,
        diagnostics=(
            {
                "code": "invalid_evidence_id",
                "path": "instance_graph.assertions",
                "message": "field group failed a model-visible validator",
                "related_ids": [],
            },
        ),
        tokenizer=tokenizer,
        tokenizer_manifest=tokenizer_manifest,
        legacy_provenance_bridge=bridge,
    )

    assert base.model_name == repair.model_name == FALLBACK_SERVED_MODEL_NAME
    assert "fallback_capability_probe" in base.packing.required_section_names
    assert SECOND_RECOVERY_EVIDENCE_BRIDGE_SECTION_NAME in (
        base.packing.required_section_names
    )
    assert repair.decoding.decoding_pass.value == "repair"
    assert repair.packing.truncation_applied is False
    assert repair.packing.complete_evidence_packet is True
    assert {"invalid_draft", "validation_diagnostics"}.issubset(
        repair.packing.required_section_names
    )
    base_payload = json.loads(base.messages[-1].content)
    repair_payload = json.loads(repair.messages[-1].content)
    assert repair_payload[SECOND_RECOVERY_EVIDENCE_BRIDGE_SECTION_NAME] == (
        base_payload[SECOND_RECOVERY_EVIDENCE_BRIDGE_SECTION_NAME]
    )
    base_bridge_pack = next(
        section
        for section in base.packing.sections
        if section.name == SECOND_RECOVERY_EVIDENCE_BRIDGE_SECTION_NAME
    )
    repair_bridge_pack = next(
        section
        for section in repair.packing.sections
        if section.name == SECOND_RECOVERY_EVIDENCE_BRIDGE_SECTION_NAME
    )
    assert repair_bridge_pack.section_content_hash == base_bridge_pack.section_content_hash


def test_fallback_repair_rejects_rebaselined_provenance_section() -> None:
    policy = FallbackModelPolicy.load(ROOT / "configs/study/fallback_model.json")
    call = fallback_pilot_calls(policy)[1]
    base = build_fallback_acceptance_request(
        root=ROOT,
        call=call,
        tokenizer=FakeTokenizer(),
        tokenizer_manifest=fallback_tokenizer_manifest(),
        legacy_provenance_bridge=legacy_provenance_bridge(),
    )
    payload = json.loads(base.messages[-1].content)
    payload[SECOND_RECOVERY_EVIDENCE_BRIDGE_SECTION_NAME]["records"][0][
        "text_hash"
    ] = "0" * 64
    tampered_section_sha256 = hashlib.sha256(
        canonical_json(payload[SECOND_RECOVERY_EVIDENCE_BRIDGE_SECTION_NAME]).encode("utf-8")
    ).hexdigest()
    tampered_packing = base.packing.model_copy(
        update={
            "sections": tuple(
                section.model_copy(
                    update={"content_hash_value": tampered_section_sha256}
                )
                if section.name == SECOND_RECOVERY_EVIDENCE_BRIDGE_SECTION_NAME
                else section
                for section in base.packing.sections
            )
        }
    )
    tampered = replace(
        base,
        messages=(
            base.messages[0],
            ChatMessage(role="user", content=canonical_json(payload)),
        ),
        packing=tampered_packing,
    )
    invalid = json.loads((ROOT / "tests/fixtures/phase1/c2_query_output.json").read_text())
    with pytest.raises(ValueError, match="exact provenance bridge"):
        build_fallback_repair_request(
            root=ROOT,
            base_call=call.acceptance_call(),
            base_request=tampered,
            invalid_draft=invalid,
            diagnostics=(
                {
                    "code": "invalid_evidence_id",
                    "path": "instance_graph.assertions",
                    "message": "field group failed a model-visible validator",
                    "related_ids": [],
                },
            ),
            tokenizer=FakeTokenizer(),
            tokenizer_manifest=fallback_tokenizer_manifest(),
            legacy_provenance_bridge=legacy_provenance_bridge(),
        )


def test_fixed_select_prompt_preserves_raw_seal_and_null_bridge_source_hashes() -> None:
    policy = FallbackModelPolicy.load(ROOT / "configs/study/fallback_model.json")
    call = fallback_pilot_calls(policy)[-1]
    request = build_fallback_acceptance_request(
        root=ROOT,
        call=call,
        tokenizer=FakeTokenizer(),
        tokenizer_manifest=fallback_tokenizer_manifest(),
        legacy_provenance_bridge=legacy_provenance_bridge(),
    )
    payload = json.loads(request.messages[-1].content)
    fixture = json.loads((ROOT / call.request_fixture).read_text(encoding="utf-8"))
    assert payload["sealed_ontology"] == fixture["fixed_ontology"]
    bridge_section = payload[SECOND_RECOVERY_EVIDENCE_BRIDGE_SECTION_NAME]
    assert all(
        row["provenance"]["source_artifact_hash"] is None
        and isinstance(row["effective_validation_source_artifact_hash"], str)
        for row in bridge_section["records"]
    )


def test_cli_plan_mode_and_execute_requirements_parse_without_side_effects() -> None:
    plan = parse_arguments(["--output", "plan.json"])
    assert plan.execute is False
    assert plan.output == Path("plan.json")

    execute = parse_arguments(
        [
            "--output",
            "result.json",
            "--execute",
            "--controller-stage",
            "run",
            "--activation-certificate",
            "activation.json",
        ]
    )
    assert execute.execute is True
    assert execute.controller_stage == "run"
    assert execute.activation_certificate == Path("activation.json")

    validation = parse_arguments(
        [
            "--output",
            "preflight.json",
            "--validate-only",
            "--controller-stage",
            "orchestrate",
        ]
    )
    assert validation.validate_only is True
    assert validation.execute is False

    builder = parse_arguments(
        [
            "--output",
            "restricted/proposal.json",
            "--build-second-recovery-overlay",
            "--prior-v5-control-plane-incident",
            "v5-incident.json",
            "--prior-v6-control-plane-incident",
            "v6-incident.json",
            "--prior-v7-runtime-incident",
            "v7-incident.json",
            "--prior-v8-runtime-incident",
            "v8-incident.json",
            "--prior-v8-lease-repair-receipt",
            "v8-lease-repair.json",
            "--second-recovery-authorized-at",
            "2026-09-04T23:59:00Z",
        ]
    )
    assert builder.build_second_recovery_overlay is True
    assert builder.prior_v5_control_plane_incident == Path("v5-incident.json")
    assert builder.prior_v6_control_plane_incident == Path("v6-incident.json")
    assert builder.prior_v7_runtime_incident == Path("v7-incident.json")
    assert builder.prior_v8_runtime_incident == Path("v8-incident.json")
    assert builder.prior_v8_lease_repair_receipt == Path("v8-lease-repair.json")
    assert builder.second_recovery_authorized_at == datetime(
        2026,
        9,
        4,
        23,
        59,
        tzinfo=UTC,
    )

    with pytest.raises(SystemExit):
        parse_arguments(["--output", "invalid.json", "--execute", "--validate-only"])
    with pytest.raises(SystemExit):
        parse_arguments(
            [
                "--output",
                "invalid.json",
                "--execute",
                "--build-second-recovery-overlay",
            ]
        )


def test_validate_only_receipt_is_append_only(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output = tmp_path / "preflight.json"
    arguments = [
        "--validate-only",
        "--controller-stage",
        "orchestrate",
        "--project-root",
        str(ROOT),
        "--output",
        str(output),
        "--run-id",
        "fallback-preflight-append-only",
    ]
    for name in (
        "primary-result",
        "activation-certificate",
        "cache-replacement-receipt",
        "snapshot",
        "shared-cache",
        "verified-model-manifest",
        "source-association",
        "ledger",
        "artifact-root",
        "checkpoint",
        "quota-root",
    ):
        arguments.extend((f"--{name}", str(tmp_path / name)))
    payload = {
        "kind": "phase1_fallback_execution_preflight",
        "passed": True,
        "manifest_sha256": HASH_A,
    }
    monkeypatch.setattr(
        fallback_acceptance_module,
        "_validate_execution_preflight",
        lambda _options, *, root: payload,
    )

    assert main(arguments) == 0
    assert json.loads(output.read_text(encoding="utf-8")) == payload
    with pytest.raises(FileExistsError, match="already exists"):
        main(arguments)


def test_fallback_requires_exact_rejected_primary_gpu_ledger_baseline() -> None:
    primary = {
        "manifest_sha256": HASH_A,
        "runtime": {
            "gpu_accounting": {
                "total_allocated_microseconds": 212_281_778,
                "event_count": 2,
                "service_session_count": 2,
                "by_kind_microseconds": {
                    "failure": 96_330,
                    "timeout": 212_185_448,
                },
            }
        },
    }
    observed = GpuSummary(
        total_allocated_microseconds=212_281_778,
        event_count=2,
        service_session_count=2,
        by_kind_microseconds=(
            (GpuEventKind.FAILURE, 96_330),
            (GpuEventKind.TIMEOUT, 212_185_448),
        ),
    )
    baseline = validate_pre_fallback_gpu_accounting(primary, observed)
    assert baseline == pre_fallback_gpu_accounting_baseline(primary)
    assert baseline["primary_result_manifest_sha256"] == HASH_A
    assert baseline["total_allocated_microseconds"] == 212_281_778

    fresh_ledger = GpuSummary(
        total_allocated_microseconds=0,
        event_count=0,
        service_session_count=0,
        by_kind_microseconds=(),
    )
    with pytest.raises(RuntimeError, match="does not reproduce"):
        validate_pre_fallback_gpu_accounting(primary, fresh_ledger)


def test_authorized_fallback_retry_binds_exact_failure_and_recovered_ledger() -> None:
    primary = json.loads(
        (ROOT / "artifacts/public/results/phase1_gpu_acceptance_v2_failed.json").read_text()
    )
    activation = json.loads(
        (ROOT / "artifacts/public/manifests/fallback_activation_v2.json").read_text()
    )
    prior_path = (
        ROOT / "artifacts/public/results/"
        "fallback_gpu_acceptance_development_v1.json.controller-handoff.json"
    )
    prior = json.loads(prior_path.read_text())
    accounting = prior["runtime"]["gpu_accounting"]
    observed = GpuSummary(
        total_allocated_microseconds=accounting["total_allocated_microseconds"],
        event_count=accounting["event_count"],
        service_session_count=accounting["service_session_count"],
        by_kind_microseconds=tuple(
            (GpuEventKind(kind), microseconds)
            for kind, microseconds in accounting["by_kind_microseconds"].items()
        ),
    )

    amendment, predecessor = validate_fallback_service_retry_amendment(
        root=ROOT,
        amendment_path=ROOT / "configs/study/fallback_service_retry_amendment.json",
        prior_failure_path=prior_path,
        run_id="fallback-qwen3-8b-awq-development-v3",
        policy=FallbackModelPolicy.load(ROOT / "configs/study/fallback_model.json"),
        activation_certificate=activation,
        primary_result=primary,
        limits=ResourceLimits.load(ROOT / "configs/study/resource_limits.json"),
        observed=observed,
    )

    assert amendment["manifest_sha256"] == (
        "9ef82782c03d9915c081e89cf554a531ef3d1fba42836c522d6ebf97cdce0f28"
    )
    assert predecessor["manifest_sha256"] == prior["manifest_sha256"]
    assert amendment["amendment"]["recovery_service_start_watchdog_seconds"] == (
        AMENDED_FALLBACK_STARTUP_WATCHDOG_SECONDS
    )


def test_second_recovery_overlay_is_exact_and_proposed_cannot_execute(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    policy = FallbackModelPolicy.load(ROOT / "configs/study/fallback_model.json")
    limits = ResourceLimits.load(ROOT / "configs/study/resource_limits.json")
    primary = json.loads(
        (ROOT / "artifacts/public/results/phase1_gpu_acceptance_v2_failed.json").read_text()
    )
    activation = json.loads(
        (ROOT / "artifacts/public/manifests/fallback_activation_v2.json").read_text()
    )
    prior_amendment_path = ROOT / "configs/study/fallback_service_retry_amendment.json"
    prior_failure_path = (
        ROOT / "artifacts/public/results/"
        "fallback_gpu_acceptance_development_v1.json.controller-handoff.json"
    )
    v3_result_path = ROOT / "artifacts/public/results/fallback_gpu_acceptance_development_v3.json"
    v3_incident_path = (
        ROOT / "artifacts/public/manifests/fallback_gpu_acceptance_development_v3_incident.json"
    )
    v3_result = json.loads(v3_result_path.read_text())
    incident = json.loads(v3_incident_path.read_text())
    current_source_path = tmp_path / "current-source-association.json"
    current_source_path.write_text('{"test":"source-freeze"}\n', encoding="utf-8")
    current_source = {
        "manifest_sha256": "a" * 64,
        "local_tree_sha256": "b" * 64,
    }
    bridge = legacy_provenance_bridge()
    retry_request = build_fallback_acceptance_request(
        root=ROOT,
        call=fallback_pilot_calls(policy)[0],
        tokenizer=FakeTokenizer(),
        tokenizer_manifest=v3_fallback_tokenizer_manifest(),
        legacy_provenance_bridge=bridge,
    )
    accounting = v3_result["runtime"]["gpu_accounting"]
    predecessor_source = incident["provenance"]["source_tree"]
    corrected_forecast = fallback_acceptance_module._second_recovery_corrected_forecast(
        predecessor=v3_result,
        inventory=fallback_acceptance_module.GPUCallInventory.load(
            ROOT / "configs/study/gpu_call_inventory.json"
        ),
        limits=limits,
    )
    assert corrected_forecast["corrected_remaining_mandatory_forecast_seconds"] == (
        pytest.approx(30_516.0027264791)
    )
    assert corrected_forecast["actual_plus_remaining_and_service_start_seconds"] == (
        pytest.approx(31_722.61868077492)
    )
    assert corrected_forecast["scheduled_reserve_seconds"] == pytest.approx(677.38131922508)
    assert corrected_forecast["service_start_watchdog_seconds"] == 300.0
    assert corrected_forecast["additional_service_allocation_forecast_seconds"] == pytest.approx(
        391.40054529582005
    )
    assert "protected_resource_sample_drain_seconds" not in corrected_forecast
    assert "protected_hard_stop_reserve_seconds" not in corrected_forecast

    def file_sha256(path: Path) -> str:
        return hashlib.sha256(path.read_bytes()).hexdigest()

    base_payload: dict[str, object] = {
        "schema_version": "1.2.0",
        "kind": SECOND_RECOVERY_OVERLAY_KIND,
        "authorization": {
            "status": "proposed",
            "basis": "Pending explicit user authorization after final source freeze.",
            "authorized_by": None,
            "recorded_at": None,
        },
        "authorized_recovery_run_id": "fallback-qwen3-8b-awq-development-v4",
        "authoritative_plan_file_sha256": {
            "methodological": file_sha256(
                ROOT / "plan_notes/METHODOLOGICAL_PLAN_QUERY_DEPENDENT_TEMPORAL_ONTOLOGY.md"
            ),
            "implementation": file_sha256(
                ROOT / "plan_notes/IMPLEMENTATION_PLAN_QUERY_DEPENDENT_TEMPORAL_ONTOLOGY.md"
            ),
        },
        "base_gpu_call_inventory_file_sha256": file_sha256(
            ROOT / "configs/study/gpu_call_inventory.json"
        ),
        "fallback_policy_file_sha256": file_sha256(ROOT / "configs/study/fallback_model.json"),
        "fallback_activation_manifest_sha256": activation["manifest_sha256"],
        "primary_rejection_manifest_sha256": primary["manifest_sha256"],
        "model": {
            "repository": policy.repository,
            "revision": policy.revision,
            "served_model_name": policy.served_model_name,
        },
        "predecessor": {
            "run_id": SECOND_RECOVERY_V3_RUN_ID,
            "result_file_sha256": SECOND_RECOVERY_V3_RESULT_FILE_SHA256,
            "result_manifest_sha256": SECOND_RECOVERY_V3_RESULT_MANIFEST_SHA256,
            "incident_file_sha256": SECOND_RECOVERY_V3_INCIDENT_FILE_SHA256,
            "incident_manifest_sha256": SECOND_RECOVERY_V3_INCIDENT_MANIFEST_SHA256,
            "prior_retry_amendment_file_sha256": file_sha256(prior_amendment_path),
            "prior_retry_amendment_manifest_sha256": (SECOND_RECOVERY_V3_RETRY_AMENDMENT_SHA256),
            "failed_call_id": SECOND_RECOVERY_RETRY_CALL_ID,
            "failed_request_sha256": SECOND_RECOVERY_V3_REQUEST_SHA256,
            "failed_decoder_schema_sha256": (SECOND_RECOVERY_V3_DECODER_SCHEMA_SHA256),
            "failure_type": "RuntimeTransportError",
            "inference_call_reached_generation": False,
            "service_shutdown_verified": True,
            "consumed_recovery_service_starts": 1,
            "consumed_reserve_class": "reserve_long",
            "consumed_reserve_slots": 1,
        },
        "cumulative_gpu_accounting": accounting,
        "decoder_compatibility": {
            "protocol": "vllm-0.10.2-xgrammar-ignored-string-keywords-v1",
            "vllm_version": "0.10.2",
            "xgrammar_version": "0.1.23",
            "structured_decoder": "vllm-0.10.2-xgrammar-no-fallback",
            "canonical_validation_schema_file_sha256": file_sha256(
                ROOT / "schemas/jsonschema/ontology_draft.schema.json"
            ),
            "compatibility_implementation_file_sha256": file_sha256(
                ROOT / "src/story_projection_onto/llm.py"
            ),
            "phase1_schema_builder_file_sha256": file_sha256(
                ROOT / "src/story_projection_onto/phase1_acceptance.py"
            ),
            "regression_test_file_sha256": file_sha256(
                ROOT / "tests/integration/test_xgrammar_decoder_compatibility.py"
            ),
            "compatible_decoder_schema_sha256": canonical_sha256(retry_request.output_schema),
            "retry_request_sha256": retry_request.request_hash,
            **fallback_acceptance_module._second_recovery_validate_retry_request_inputs(
                root=ROOT,
                predecessor=v3_result,
                policy=policy,
                retry_request=retry_request,
                compatible_schema=retry_request.output_schema,
                legacy_provenance_bridge=bridge,
            ),
            "stripped_string_keywords": [
                "format",
                "maxLength",
                "minLength",
                "pattern",
            ],
            "ontology_draft_schema_and_pydantic_validation_unchanged": True,
            "cpu_xgrammar_compilation_required_before_gpu": True,
        },
        "evidence_provenance_bridge": (
            fallback_acceptance_module._second_recovery_evidence_bridge_binding(
                root=ROOT,
                bridge=bridge,
                retry_request=retry_request,
            )
        ),
        "source": {
            "predecessor_association_manifest_sha256": predecessor_source[
                "association_manifest_sha256"
            ],
            "predecessor_tree_sha256": predecessor_source["tree_sha256"],
            "current_association_file_sha256": file_sha256(current_source_path),
            "current_association_manifest_sha256": current_source["manifest_sha256"],
            "current_tree_sha256": current_source["local_tree_sha256"],
        },
        "amendment": {
            "additional_fallback_service_loads": 1,
            "recovery_service_start_watchdog_seconds": 300,
            "authorized_retry_inference_attempts": 1,
            "retry_call_id": SECOND_RECOVERY_RETRY_CALL_ID,
            "retry_attempt_kind": "retry",
            "retry_reserve_call_class": "reserve_long",
            "retry_watchdog_seconds": 240,
            "additional_unreserved_inference_attempts": 0,
            "original_accounting_events": 286,
            "prior_effective_accounting_events": 287,
            "amended_effective_accounting_events": 288,
            "original_maximum_inference_attempts": 278,
            "amended_maximum_inference_attempts": 278,
            "prior_consumed_reserve_long_slots": 1,
            "projected_consumed_reserve_long_slots_after_retry": 2,
            "registered_reserve_long_slot_count": 4,
        },
        "corrected_forecast": corrected_forecast,
        "c0_pre_data_correction": {
            "classification": "pre_data_implementation_correction",
            "condition": "C0",
            "implementation_file": "src/story_projection_onto/conditions/c0.py",
            "predecessor_implementation_file_sha256": (
                "bb7ec5f00df86ec24eedeaea6b90a1bda473511d0ab9f334cecda9f1970f27c7"
            ),
            "current_implementation_file_sha256": (
                SECOND_RECOVERY_POST_TWO_FIX_C0_IMPLEMENTATION_SHA256
            ),
            "regression_test_file": "tests/unit/test_c0_condition.py",
            "regression_test_file_sha256": (
                SECOND_RECOVERY_POST_TWO_FIX_C0_REGRESSION_TEST_SHA256
            ),
            "change_ids": [
                "parse_story_step_point_and_validity_interval",
                "admit_supported_served_as_predicate",
            ],
            "predecessor_completed_base_call_count": 0,
            "predecessor_development_call_count": 0,
            "predecessor_development_output_count": 0,
            "retry_request_affected": False,
            "authoritative_plan_rewritten": False,
        },
        "semantic_validation_correction": {
            "classification": "post_v3_integrity_correction",
            "condition": "C1",
            "implementation_file": "src/story_projection_onto/conditions/c1.py",
            "predecessor_implementation_file_sha256": (SECOND_RECOVERY_V3_C1_IMPLEMENTATION_SHA256),
            "current_implementation_file_sha256": (SECOND_RECOVERY_V4_C1_IMPLEMENTATION_SHA256),
            "condition_pathway_regression_file": ("tests/integration/test_condition_pathways.py"),
            "condition_pathway_regression_file_sha256": (
                SECOND_RECOVERY_V4_C1_CONDITION_PATHWAY_TEST_SHA256
            ),
            "development_assessment_regression_file": ("tests/unit/test_development_assessment.py"),
            "development_assessment_regression_file_sha256": (
                SECOND_RECOVERY_V4_C1_DEVELOPMENT_ASSESSMENT_TEST_SHA256
            ),
            "change_ids": ["semantic-validation-correction-c1-structural-status-v1"],
            "predecessor_accepted_output_count": 0,
            "predecessor_completed_base_call_count": 0,
            "predecessor_development_call_count": 0,
            "predecessor_development_output_count": 0,
            "retry_request_affected": False,
            "changed_surface": "post_generation_validation_metadata_only",
            "validation_status": "accepted",
            "evidence_support_status": "not_applicable",
            "temporal_status": "not_applicable",
            "commitment_status": "not_applicable",
            "standardized_diagnostic": RUNTIME_STRUCTURAL_ONLY_DIAGNOSTIC,
            "unchanged_controls": {
                "prompts": True,
                "evidence": True,
                "model_visible_payload_byte_identity_to_v3_claimed": False,
                "model_visible_delta_limited_to_hash_bound_provenance": True,
                "llm_ontology_draft_schema": True,
                "construction_semantics": True,
                "selection_semantics": True,
                "object_and_display_budgets": True,
                "seeds": True,
            },
            "authoritative_plan_rewritten": False,
        },
        "frozen_input_controls": (
            fallback_acceptance_module._second_recovery_frozen_input_controls(ROOT)
        ),
        "projection_dependency_correction": (
            fallback_acceptance_module._second_recovery_projection_dependency_correction(
                root=ROOT,
                predecessor=v3_result,
                incident=incident,
            )
        ),
        "concurrent_integrity_disclosure": (
            fallback_acceptance_module._second_recovery_concurrent_integrity_disclosure(
                root=ROOT,
                predecessor=v3_result,
                incident=incident,
            )
        ),
        "unchanged_scientific_controls": {
            "model_snapshot": True,
            "runtime_stack": True,
            "no_cpu_weight_offload": True,
            "generation_concurrency_one": True,
            "c1_prequery_llm_construction_semantics": True,
            "c2_query_dependent_construction_semantics": True,
            "a_fixed_select_mechanical_selection_semantics": True,
            "a_fixed_select_raw_seal_checked_before_effective_provenance_binding": True,
            "post_generation_validation_required": True,
            "single_bounded_repair_limit": True,
            "repair_model_prompt_fact_free_diagnostics": True,
            "immutable_v3_fallback_input_comparison_required": True,
        },
        "scope": (
            "One reserve-long retry of the exact v3 fallback C1 transport failure "
            "and one additional recovery service start; no other retry is authorized."
        ),
        "authoritative_plans_rewritten": False,
    }

    def write_overlay(payload: Mapping[str, object], *, name: str) -> Path:
        overlay_path = tmp_path / name
        overlay_path.write_text(
            json.dumps(
                {**payload, "manifest_sha256": canonical_sha256(payload)},
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        return overlay_path

    observed = GpuSummary(
        total_allocated_microseconds=accounting["total_allocated_microseconds"],
        event_count=accounting["event_count"],
        service_session_count=accounting["service_session_count"],
        by_kind_microseconds=tuple(
            (GpuEventKind(kind), microseconds)
            for kind, microseconds in accounting["by_kind_microseconds"].items()
        ),
    )
    validation_arguments = {
        "root": ROOT,
        "v3_result_path": v3_result_path,
        "v3_incident_path": v3_incident_path,
        "prior_retry_amendment_path": prior_amendment_path,
        "prior_retry_failure_path": prior_failure_path,
        "run_id": "fallback-qwen3-8b-awq-development-v4",
        "policy": policy,
        "activation_certificate": activation,
        "primary_result": primary,
        "limits": limits,
        "source_association": current_source,
        "source_association_path": current_source_path,
        "retry_request": retry_request,
        "legacy_provenance_bridge": bridge,
        "observed": observed,
        "verify_decoder_compilation": False,
    }
    proposed_path = write_overlay(base_payload, name="second-recovery-proposed.json")
    proposed, predecessor, validated_incident = validate_second_fallback_recovery_overlay(
        overlay_path=proposed_path,
        require_authorized=False,
        **validation_arguments,
    )
    assert proposed["authorization"]["status"] == "proposed"
    assert predecessor["manifest_sha256"] == SECOND_RECOVERY_V3_RESULT_MANIFEST_SHA256
    assert validated_incident["manifest_sha256"] == (SECOND_RECOVERY_V3_INCIDENT_MANIFEST_SHA256)
    with pytest.raises(PermissionError, match="remains proposed"):
        validate_second_fallback_recovery_overlay(
            overlay_path=proposed_path,
            require_authorized=True,
            **validation_arguments,
        )

    changed_user_payload = json.loads(retry_request.messages[1].content)
    changed_user_payload["evidence_snapshot"][0]["text"] += " Altered."
    changed_message_request = replace(
        retry_request,
        messages=(
            retry_request.messages[0],
            ChatMessage(role="user", content=canonical_json(changed_user_payload)),
        ),
    )
    changed_message_overlay = copy.deepcopy(base_payload)
    changed_message_overlay["decoder_compatibility"]["retry_request_sha256"] = (
        changed_message_request.request_hash
    )
    with pytest.raises(ValueError, match="frozen v3 scientific inputs"):
        validate_second_fallback_recovery_overlay(
            overlay_path=write_overlay(
                changed_message_overlay,
                name="second-recovery-rebaselined-message.json",
            ),
            require_authorized=False,
            **{**validation_arguments, "retry_request": changed_message_request},
        )

    changed_bridge_payload = json.loads(retry_request.messages[1].content)
    changed_bridge_payload[SECOND_RECOVERY_EVIDENCE_BRIDGE_SECTION_NAME]["records"][0][
        "text_hash"
    ] = "0" * 64
    changed_bridge_request = replace(
        retry_request,
        messages=(
            retry_request.messages[0],
            ChatMessage(role="user", content=canonical_json(changed_bridge_payload)),
        ),
    )
    changed_bridge_overlay = copy.deepcopy(base_payload)
    changed_bridge_overlay["decoder_compatibility"]["retry_request_sha256"] = (
        changed_bridge_request.request_hash
    )
    with pytest.raises(ValueError, match="frozen v3 scientific inputs"):
        validate_second_fallback_recovery_overlay(
            overlay_path=write_overlay(
                changed_bridge_overlay,
                name="second-recovery-rebaselined-provenance-section.json",
            ),
            require_authorized=False,
            **{**validation_arguments, "retry_request": changed_bridge_request},
        )

    changed_decoding_payload = retry_request.decoding.model_dump(
        mode="json",
        exclude={"content_hash"},
    )
    changed_decoding_payload["seed"] = 1
    changed_seed_request = replace(
        retry_request,
        decoding=type(retry_request.decoding).model_validate(changed_decoding_payload),
    )
    changed_seed_overlay = copy.deepcopy(base_payload)
    changed_seed_overlay["decoder_compatibility"]["retry_request_sha256"] = (
        changed_seed_request.request_hash
    )
    with pytest.raises(ValueError, match="frozen v3 scientific inputs"):
        validate_second_fallback_recovery_overlay(
            overlay_path=write_overlay(
                changed_seed_overlay,
                name="second-recovery-rebaselined-seed.json",
            ),
            require_authorized=False,
            **{**validation_arguments, "retry_request": changed_seed_request},
        )

    authorized_payload = copy.deepcopy(base_payload)
    authorized_payload["authorization"] = {
        "status": "authorized",
        "basis": "The user explicitly authorized this exact hash-bound v4 recovery.",
        "authorized_by": "user",
        "recorded_at": "2026-09-04T23:59:00Z",
    }
    authorized_path = write_overlay(
        authorized_payload,
        name="second-recovery-authorized.json",
    )
    authorized, _, _ = validate_second_fallback_recovery_overlay(
        overlay_path=authorized_path,
        require_authorized=True,
        **validation_arguments,
    )
    assert authorized["authorization"]["status"] == "authorized"
    assert authorized["amendment"]["amended_maximum_inference_attempts"] == 278
    assert authorized["c0_pre_data_correction"] == base_payload["c0_pre_data_correction"]
    assert (
        authorized["semantic_validation_correction"]
        == base_payload["semantic_validation_correction"]
    )
    assert authorized["c0_pre_data_correction"][
        "current_implementation_file_sha256"
    ] == SECOND_RECOVERY_POST_TWO_FIX_C0_IMPLEMENTATION_SHA256
    assert authorized["c0_pre_data_correction"][
        "regression_test_file_sha256"
    ] == SECOND_RECOVERY_POST_TWO_FIX_C0_REGRESSION_TEST_SHA256
    assert fallback_acceptance_module._second_recovery_c0_pre_data_correction(
        root=tmp_path,
        predecessor=v3_result,
        incident=incident,
    ) == base_payload["c0_pre_data_correction"]
    assert file_sha256(
        ROOT / "src/story_projection_onto/conditions/c0.py"
    ) != SECOND_RECOVERY_POST_TWO_FIX_C0_IMPLEMENTATION_SHA256
    assert authorized["semantic_validation_correction"][
        "current_implementation_file_sha256"
    ] == SECOND_RECOVERY_V4_C1_IMPLEMENTATION_SHA256
    assert fallback_acceptance_module._second_recovery_semantic_validation_correction(
        root=tmp_path,
        predecessor=v3_result,
        incident=incident,
    ) == base_payload["semantic_validation_correction"]
    assert file_sha256(
        ROOT / "src/story_projection_onto/conditions/c1.py"
    ) != SECOND_RECOVERY_V4_C1_IMPLEMENTATION_SHA256
    decoder = authorized["decoder_compatibility"]
    assert decoder["failed_v3_decoder_schema_reconstructed_sha256"] == (
        SECOND_RECOVERY_V3_DECODER_SCHEMA_SHA256
    )
    assert decoder["failed_v3_request_reconstructed_sha256"] == (
        SECOND_RECOVERY_V3_REQUEST_SHA256
    )
    assert decoder["failed_v3_request_exactly_reconstructed"] is True
    assert decoder["retry_scientific_inputs_exactly_reconstructed"] is True
    assert decoder["retry_wire_delta_scope"] == (
        "guided_schema_schema_derived_runtime_hashes_and_hash_bound_provenance_only"
    )
    evidence_bridge = authorized["evidence_provenance_bridge"]
    assert evidence_bridge["certificate_manifest_sha256"] == bridge.manifest_sha256
    assert evidence_bridge["certificate_file_sha256"] == bridge.certificate_file_sha256
    assert evidence_bridge["retry_call_id"] == SECOND_RECOVERY_RETRY_CALL_ID
    assert evidence_bridge["retry_legacy_evidence_sha256"] == (
        "a20b09f014e86dbb5ef490beb35134b75a1ad306536de6d0d07e4570de8f1b73"
    )
    assert evidence_bridge["source_critical_validation_fields"] == [
        "evidence_id",
        "locator",
        "source_artifact_hash",
        "confidence_ceiling",
    ]
    assert evidence_bridge["immutable_v3_fixture_bytes_changed"] is False
    assert evidence_bridge["cpu_semantic_inference_performed"] is False
    assert evidence_bridge[
        "fixed_raw_semantic_comparison_precedes_effective_binding"
    ] is True
    assert evidence_bridge["fixed_prompt_preserves_raw_null_source_artifact_hashes"] is True
    assert evidence_bridge["repairs_preserve_exact_model_visible_section"] is True

    frozen_inputs = authorized["frozen_input_controls"]
    assert frozen_inputs["predecessor_source_manifest_file_sha256"] == (
        fallback_acceptance_module.SECOND_RECOVERY_V3_SOURCE_MANIFEST_FILE_SHA256
    )
    assert frozen_inputs["predecessor_source_tree_sha256"] == (
        fallback_acceptance_module.SECOND_RECOVERY_V3_SOURCE_TREE_SHA256
    )
    assert tuple(frozen_inputs["fallback_call_seed_values"]) == (0, 0, 1, 1)
    assert frozen_inputs["whole_wire_payload_byte_identity_to_failed_v3_claimed"] is False
    assert frozen_inputs["wire_payload_difference"] == (
        "registered_decoder_compatibility_and_hash_bound_provenance_bridge"
    )
    assert all(item["byte_identical"] for item in frozen_inputs["byte_identical_files"])
    assert {
        item["path"] for item in frozen_inputs["byte_identical_files"]
    } == set(fallback_acceptance_module.SECOND_RECOVERY_FROZEN_INPUT_SOURCE_PATHS)

    projection_correction = authorized["projection_dependency_correction"]
    assert projection_correction["correction_id"] == (
        "second-recovery-predata-projection-dependency-integrity-v1"
    )
    assert projection_correction["predecessor_c0_implementation_file_sha256"] == (
        SECOND_RECOVERY_POST_TWO_FIX_C0_IMPLEMENTATION_SHA256
    )
    assert projection_correction["predecessor_c1_implementation_file_sha256"] == (
        SECOND_RECOVERY_V4_C1_IMPLEMENTATION_SHA256
    )
    assert projection_correction["current_c0_implementation_file_sha256"] == file_sha256(
        ROOT / "src/story_projection_onto/conditions/c0.py"
    )
    assert projection_correction["current_c1_implementation_file_sha256"] == file_sha256(
        ROOT / "src/story_projection_onto/conditions/c1.py"
    )
    assert projection_correction["query_time_selection_feasibility_changed"] is True
    assert projection_correction["final_projection_validation_targeting_changed"] is True
    assert projection_correction["final_projection_validation_target_domain"] == (
        "ontology-projection-structural-validation-target-v1"
    )
    assert projection_correction["final_projection_validation_target_excludes_cyclic_fields"]
    assert projection_correction["repair_parent_hash_and_attempt_required_together"]
    assert projection_correction["future_projection_outputs_may_change"] is True
    assert projection_correction["predecessor_base_output_count"] == 0
    assert projection_correction["correction_model_output_count"] == 0
    assert projection_correction["correction_gpu_call_count"] == 0
    assert projection_correction["unchanged_controls"][
        "frozen_input_controls_sha256"
    ] == canonical_sha256(frozen_inputs)

    integrity_disclosure = authorized["concurrent_integrity_disclosure"]
    assert "registered-display-qualified-dependency-closure-v2" in (
        integrity_disclosure["change_ids"]
    )
    assert integrity_disclosure["predecessor_accepted_output_count"] == 0
    assert integrity_disclosure["predecessor_base_output_count"] == 0
    assert integrity_disclosure["correction_model_output_count"] == 0
    assert integrity_disclosure["correction_gpu_call_count"] == 0
    assert integrity_disclosure["retry_request_affected"] is False
    assert integrity_disclosure["frozen_input_controls_sha256"] == canonical_sha256(
        authorized["frozen_input_controls"]
    )
    assert integrity_disclosure[
        "whole_wire_payload_byte_identity_to_failed_v3_claimed"
    ] is False
    assert integrity_disclosure["c1_prequery_construction_acceptance_changed"] is False
    assert integrity_disclosure["final_projection_display_feasibility_changed"] is True
    assert "condition_semantics" not in authorized["unchanged_scientific_controls"]
    assert "validation_and_repair_policy" not in authorized["unchanged_scientific_controls"]
    assert {
        "c1_prequery_llm_construction_semantics",
        "c2_query_dependent_construction_semantics",
        "a_fixed_select_mechanical_selection_semantics",
        "a_fixed_select_raw_seal_checked_before_effective_provenance_binding",
        "single_bounded_repair_limit",
        "repair_model_prompt_fact_free_diagnostics",
    }.issubset(authorized["unchanged_scientific_controls"])
    assert "c2_condition_implementation_byte_identical" not in (
        authorized["unchanged_scientific_controls"]
    )
    assert "a_fixed_select_condition_implementation_byte_identical" not in (
        authorized["unchanged_scientific_controls"]
    )

    for required_correction in (
        "evidence_provenance_bridge",
        "frozen_input_controls",
        "projection_dependency_correction",
        "concurrent_integrity_disclosure",
    ):
        missing_layer = copy.deepcopy(authorized_payload)
        missing_layer.pop(required_correction)
        missing_layer_path = write_overlay(
            missing_layer,
            name=f"second-recovery-missing-{required_correction}.json",
        )
        with pytest.raises(ValueError, match="contract is invalid"):
            validate_second_fallback_recovery_overlay(
                overlay_path=missing_layer_path,
                require_authorized=True,
                **validation_arguments,
            )

    misstated_bridge = copy.deepcopy(authorized_payload)
    misstated_bridge["evidence_provenance_bridge"]["certificate_file_sha256"] = "0" * 64
    with pytest.raises(
        ValueError,
        match="evidence provenance bridge binding changed",
    ):
        validate_second_fallback_recovery_overlay(
            overlay_path=write_overlay(
                misstated_bridge,
                name="second-recovery-misstated-provenance-certificate.json",
            ),
            require_authorized=True,
            **validation_arguments,
        )

    legacy_schema = copy.deepcopy(authorized_payload)
    legacy_schema["schema_version"] = "1.0.0"
    legacy_schema_path = write_overlay(
        legacy_schema,
        name="second-recovery-legacy-schema.json",
    )
    with pytest.raises(ValueError, match="contract is invalid"):
        validate_second_fallback_recovery_overlay(
            overlay_path=legacy_schema_path,
            require_authorized=True,
            **validation_arguments,
        )

    for hash_field in (
        "current_c0_implementation_file_sha256",
        "current_c1_implementation_file_sha256",
        "validate_implementation_file_sha256",
        "contracts_implementation_file_sha256",
        "c0_regression_test_file_sha256",
        "c1_regression_test_file_sha256",
        "contracts_regression_test_file_sha256",
    ):
        misstated_projection_hash = copy.deepcopy(authorized_payload)
        misstated_projection_hash["projection_dependency_correction"][hash_field] = "0" * 64
        misstated_projection_hash_path = write_overlay(
            misstated_projection_hash,
            name=f"second-recovery-misstated-projection-{hash_field}.json",
        )
        with pytest.raises(ValueError, match="misstates the projection-dependency correction"):
            validate_second_fallback_recovery_overlay(
                overlay_path=misstated_projection_hash_path,
                require_authorized=True,
                **validation_arguments,
            )

    nonzero_projection_output = copy.deepcopy(authorized_payload)
    nonzero_projection_output["projection_dependency_correction"][
        "correction_model_output_count"
    ] = 1
    nonzero_projection_output_path = write_overlay(
        nonzero_projection_output,
        name="second-recovery-nonzero-projection-output.json",
    )
    with pytest.raises(ValueError, match="contract is invalid"):
        validate_second_fallback_recovery_overlay(
            overlay_path=nonzero_projection_output_path,
            require_authorized=True,
            **validation_arguments,
        )

    equalized_projection_bytes = copy.deepcopy(authorized_payload)
    equalized_projection_bytes["projection_dependency_correction"][
        "current_c0_implementation_file_sha256"
    ] = equalized_projection_bytes["projection_dependency_correction"][
        "predecessor_c0_implementation_file_sha256"
    ]
    equalized_projection_bytes_path = write_overlay(
        equalized_projection_bytes,
        name="second-recovery-equalized-projection-bytes.json",
    )
    with pytest.raises(ValueError, match="contract is invalid"):
        validate_second_fallback_recovery_overlay(
            overlay_path=equalized_projection_bytes_path,
            require_authorized=True,
            **validation_arguments,
        )

    misstated_integrity_hash = copy.deepcopy(authorized_payload)
    misstated_integrity_hash["concurrent_integrity_disclosure"][
        "semantic_assessment_scope"
    ]["sources"][0]["sha256"] = "0" * 64
    misstated_integrity_hash_path = write_overlay(
        misstated_integrity_hash,
        name="second-recovery-misstated-integrity-hash.json",
    )
    with pytest.raises(ValueError, match="misstates the concurrent integrity disclosure"):
        validate_second_fallback_recovery_overlay(
            overlay_path=misstated_integrity_hash_path,
            require_authorized=True,
            **validation_arguments,
        )

    changed_integrity_retry = copy.deepcopy(authorized_payload)
    changed_integrity_retry["concurrent_integrity_disclosure"]["retry_request_affected"] = True
    changed_integrity_retry_path = write_overlay(
        changed_integrity_retry,
        name="second-recovery-changed-integrity-retry.json",
    )
    with pytest.raises(ValueError, match="contract is invalid"):
        validate_second_fallback_recovery_overlay(
            overlay_path=changed_integrity_retry_path,
            require_authorized=True,
            **validation_arguments,
        )

    missing_correction = copy.deepcopy(authorized_payload)
    missing_correction.pop("semantic_validation_correction")
    missing_correction_path = write_overlay(
        missing_correction,
        name="second-recovery-missing-semantic-validation-correction.json",
    )
    with pytest.raises(ValueError, match="contract is invalid"):
        validate_second_fallback_recovery_overlay(
            overlay_path=missing_correction_path,
            require_authorized=True,
            **validation_arguments,
        )

    for hash_field in (
        "current_implementation_file_sha256",
        "condition_pathway_regression_file_sha256",
        "development_assessment_regression_file_sha256",
    ):
        misstated_c1_hash = copy.deepcopy(authorized_payload)
        misstated_c1_hash["semantic_validation_correction"][hash_field] = "0" * 64
        misstated_c1_hash_path = write_overlay(
            misstated_c1_hash,
            name=f"second-recovery-misstated-{hash_field}.json",
        )
        with pytest.raises(
            ValueError,
            match="misstates the C1 semantic-validation correction",
        ):
            validate_second_fallback_recovery_overlay(
                overlay_path=misstated_c1_hash_path,
                require_authorized=True,
                **validation_arguments,
            )

    weakened_c1_flag = copy.deepcopy(authorized_payload)
    weakened_c1_flag["semantic_validation_correction"]["unchanged_controls"]["prompts"] = False
    weakened_c1_flag_path = write_overlay(
        weakened_c1_flag,
        name="second-recovery-weakened-c1-flag.json",
    )
    with pytest.raises(ValueError, match="contract is invalid"):
        validate_second_fallback_recovery_overlay(
            overlay_path=weakened_c1_flag_path,
            require_authorized=True,
            **validation_arguments,
        )

    changed_retry_effect = copy.deepcopy(authorized_payload)
    changed_retry_effect["semantic_validation_correction"]["retry_request_affected"] = True
    changed_retry_effect_path = write_overlay(
        changed_retry_effect,
        name="second-recovery-changed-retry-effect.json",
    )
    with pytest.raises(ValueError, match="contract is invalid"):
        validate_second_fallback_recovery_overlay(
            overlay_path=changed_retry_effect_path,
            require_authorized=True,
            **validation_arguments,
        )

    misstated_c1_status = copy.deepcopy(authorized_payload)
    misstated_c1_status["semantic_validation_correction"]["evidence_support_status"] = "supported"
    misstated_c1_status_path = write_overlay(
        misstated_c1_status,
        name="second-recovery-misstated-c1-status.json",
    )
    with pytest.raises(ValueError, match="contract is invalid"):
        validate_second_fallback_recovery_overlay(
            overlay_path=misstated_c1_status_path,
            require_authorized=True,
            **validation_arguments,
        )

    misstated_c1_diagnostic = copy.deepcopy(authorized_payload)
    misstated_c1_diagnostic["semantic_validation_correction"]["standardized_diagnostic"] = (
        "A different diagnostic."
    )
    misstated_c1_diagnostic_path = write_overlay(
        misstated_c1_diagnostic,
        name="second-recovery-misstated-c1-diagnostic.json",
    )
    with pytest.raises(ValueError, match="contract is invalid"):
        validate_second_fallback_recovery_overlay(
            overlay_path=misstated_c1_diagnostic_path,
            require_authorized=True,
            **validation_arguments,
        )

    invalid_delta = copy.deepcopy(authorized_payload)
    invalid_delta["amendment"]["amended_maximum_inference_attempts"] = 279
    invalid_path = write_overlay(invalid_delta, name="second-recovery-invalid.json")
    with pytest.raises(ValueError, match="contract is invalid"):
        validate_second_fallback_recovery_overlay(
            overlay_path=invalid_path,
            require_authorized=True,
            **validation_arguments,
        )

    for hash_field in (
        "predecessor_implementation_file_sha256",
        "current_implementation_file_sha256",
        "regression_test_file_sha256",
    ):
        misstated_c0 = copy.deepcopy(authorized_payload)
        misstated_c0["c0_pre_data_correction"][hash_field] = "0" * 64
        misstated_c0_path = write_overlay(
            misstated_c0,
            name=f"second-recovery-misstated-c0-{hash_field}.json",
        )
        with pytest.raises(ValueError, match="misstates the pre-data C0 correction"):
            validate_second_fallback_recovery_overlay(
                overlay_path=misstated_c0_path,
                require_authorized=True,
                **validation_arguments,
            )

    changed_observed = GpuSummary(
        total_allocated_microseconds=accounting["total_allocated_microseconds"] + 1,
        event_count=accounting["event_count"],
        service_session_count=accounting["service_session_count"],
        by_kind_microseconds=observed.by_kind_microseconds,
    )
    with pytest.raises(RuntimeError, match="terminal v3 ledger"):
        validate_second_fallback_recovery_overlay(
            overlay_path=authorized_path,
            require_authorized=True,
            **{**validation_arguments, "observed": changed_observed},
        )

    v6_source_path = (
        tmp_path / "source_tree_fallback_second_recovery_v6.association.json"
    )
    v6_source_path.write_text('{"test":"v6-source-freeze"}\n', encoding="utf-8")
    v6_source = {
        **current_source,
        "revision_label": fallback_acceptance_module.SECOND_RECOVERY_V6_SOURCE_REVISION,
    }
    v7_source_path = (
        tmp_path / "source_tree_fallback_second_recovery_v7.association.json"
    )
    v7_source_path.write_text('{"test":"v7-source-freeze"}\n', encoding="utf-8")
    v7_source = {
        **current_source,
        "revision_label": fallback_acceptance_module.SECOND_RECOVERY_V7_SOURCE_REVISION,
    }
    v8_source_path = (
        tmp_path / "source_tree_fallback_second_recovery_v8.association.json"
    )
    v8_source_path.write_text('{"test":"v8-source-freeze"}\n', encoding="utf-8")
    v8_source = {
        **current_source,
        "revision_label": fallback_acceptance_module.SECOND_RECOVERY_V8_SOURCE_REVISION,
    }
    v9_source_path = (
        tmp_path / "source_tree_fallback_second_recovery_v9.association.json"
    )
    v9_source_path.write_text('{"test":"v9-source-freeze"}\n', encoding="utf-8")
    v9_source = {
        **current_source,
        "revision_label": fallback_acceptance_module.SECOND_RECOVERY_V9_SOURCE_REVISION,
    }

    def validate_test_source(path: Path, **_kwargs: object) -> Mapping[str, object]:
        if Path(path).name == v6_source_path.name:
            return v6_source
        if Path(path).name == v7_source_path.name:
            return v7_source
        if Path(path).name == v8_source_path.name:
            return v8_source
        if Path(path).name == v9_source_path.name:
            return v9_source
        return current_source

    monkeypatch.setattr(
        fallback_acceptance_module,
        "validate_source_association",
        validate_test_source,
    )
    restricted_root = tmp_path / "restricted"
    built_proposed_path = restricted_root / "second-recovery-built.proposed.json"
    builder_arguments = {
        "root": ROOT,
        "restricted_output_root": restricted_root,
        "v3_result_path": v3_result_path,
        "v3_incident_path": v3_incident_path,
        "prior_retry_amendment_path": prior_amendment_path,
        "prior_retry_failure_path": prior_failure_path,
        "run_id": "fallback-qwen3-8b-awq-development-v4",
        "policy": policy,
        "activation_certificate": activation,
        "primary_result": primary,
        "limits": limits,
        "source_association": current_source,
        "source_association_path": current_source_path,
        "retry_request": retry_request,
        "legacy_provenance_bridge": bridge,
        "observed": observed,
        "verify_decoder_compilation": False,
    }
    built_proposed = build_second_fallback_recovery_overlay(
        output_path=built_proposed_path,
        **builder_arguments,
    )
    first_bytes = built_proposed_path.read_bytes()
    replayed = build_second_fallback_recovery_overlay(
        output_path=built_proposed_path,
        **builder_arguments,
    )
    assert replayed == built_proposed
    assert built_proposed_path.read_bytes() == first_bytes
    assert built_proposed["authorization"] == {
        "status": "proposed",
        "basis": "Pending explicit user authorization after final source freeze.",
        "authorized_by": None,
        "recorded_at": None,
    }
    assert built_proposed["schema_version"] == "1.2.0"
    assert built_proposed["frozen_input_controls"] == base_payload["frozen_input_controls"]
    assert built_proposed["projection_dependency_correction"] == (
        base_payload["projection_dependency_correction"]
    )
    assert built_proposed["concurrent_integrity_disclosure"] == (
        base_payload["concurrent_integrity_disclosure"]
    )
    with pytest.raises(FileExistsError, match="append-only"):
        build_second_fallback_recovery_overlay(
            output_path=built_proposed_path,
            authorization_basis="A changed proposed basis.",
            **builder_arguments,
        )

    built_authorized = build_second_fallback_recovery_overlay(
        output_path=restricted_root / "second-recovery-built.authorized.json",
        authorization_status="authorized",
        authorization_basis=(
            "The user explicitly authorized the exact bound recovery in this test."
        ),
        authorized_at=datetime(2026, 9, 4, 23, 59, tzinfo=UTC),
        **builder_arguments,
    )
    assert built_authorized["authorization"]["status"] == "authorized"
    v4_control_plane_incident = (
        ROOT
        / "artifacts/public/manifests/"
        "fallback_gpu_acceptance_development_v4_control_plane_incident.json"
    )
    built_v5 = build_second_fallback_recovery_overlay(
        output_path=restricted_root / "second-recovery-v5-built.authorized.json",
        authorization_status="authorized",
        authorization_basis="The user authorized continued GPU execution in this test.",
        authorized_at=datetime(2026, 9, 5, 14, 45, tzinfo=UTC),
        **{
            **builder_arguments,
            "run_id": "fallback-qwen3-8b-awq-development-v5",
            "prior_control_plane_incident_path": v4_control_plane_incident,
        },
    )
    assert built_v5["schema_version"] == "1.3.0"
    assert built_v5["intervening_control_plane_incident"] == (
        fallback_acceptance_module._second_recovery_control_plane_incident_binding(
            v4_control_plane_incident
        )
    )
    v5_control_plane_incident = (
        ROOT
        / "artifacts/public/manifests/"
        "fallback_gpu_acceptance_development_v5_control_plane_incident.json"
    )
    built_v6 = build_second_fallback_recovery_overlay(
        output_path=restricted_root / "second-recovery-v6-built.authorized.json",
        authorization_status="authorized",
        authorization_basis="The user authorized continued GPU execution in this test.",
        authorized_at=datetime(2026, 9, 5, 17, 30, tzinfo=UTC),
        **{
            **builder_arguments,
            "run_id": fallback_acceptance_module.SECOND_RECOVERY_V6_RUN_ID,
            "prior_control_plane_incident_path": v4_control_plane_incident,
            "prior_v5_control_plane_incident_path": v5_control_plane_incident,
            "source_association": v6_source,
            "source_association_path": v6_source_path,
        },
    )
    assert built_v6["schema_version"] == "1.4.0"
    assert built_v6["authorized_recovery_run_id"] == (
        fallback_acceptance_module.SECOND_RECOVERY_V6_RUN_ID
    )
    assert built_v6["intervening_control_plane_incident"] == (
        fallback_acceptance_module._second_recovery_control_plane_incident_binding(
            v4_control_plane_incident
        )
    )
    assert built_v6["intervening_v5_control_plane_incident"] == (
        fallback_acceptance_module._second_recovery_v5_control_plane_incident_binding(
            v5_control_plane_incident
        )
    )
    v6_control_plane_incident = (
        ROOT
        / "artifacts/public/manifests/"
        "fallback_gpu_acceptance_development_v6_control_plane_incident.json"
    )
    built_v7 = build_second_fallback_recovery_overlay(
        output_path=restricted_root / "second-recovery-v7-built.authorized.json",
        authorization_status="authorized",
        authorization_basis="The user authorized bounded continued GPU execution in this test.",
        authorized_at=datetime(2026, 9, 5, 18, 30, tzinfo=UTC),
        **{
            **builder_arguments,
            "run_id": fallback_acceptance_module.SECOND_RECOVERY_V7_RUN_ID,
            "prior_control_plane_incident_path": v4_control_plane_incident,
            "prior_v5_control_plane_incident_path": v5_control_plane_incident,
            "prior_v6_control_plane_incident_path": v6_control_plane_incident,
            "source_association": v7_source,
            "source_association_path": v7_source_path,
        },
    )
    assert built_v7["schema_version"] == "1.5.0"
    assert built_v7["authorized_recovery_run_id"] == (
        fallback_acceptance_module.SECOND_RECOVERY_V7_RUN_ID
    )
    assert built_v7["intervening_v6_control_plane_incident"] == (
        fallback_acceptance_module._second_recovery_v6_control_plane_incident_binding(
            v6_control_plane_incident
        )
    )
    v7_runtime_incident = (
        ROOT
        / "artifacts/public/manifests/"
        "fallback_gpu_acceptance_development_v7_runtime_incident.json"
    )
    terminal_v7_observed = GpuSummary(
        total_allocated_microseconds=1_507_850_965,
        event_count=6,
        service_session_count=5,
        by_kind_microseconds=(
            (GpuEventKind.FAILURE, 225_183_297),
            (GpuEventKind.GPU_SESSION_START, 214_034_494),
            (GpuEventKind.SERVICE_OVERHEAD, 645_767_518),
            (GpuEventKind.TIMEOUT, 422_865_656),
        ),
    )
    built_v8 = build_second_fallback_recovery_overlay(
        output_path=restricted_root / "second-recovery-v8-built.authorized.json",
        authorization_status="authorized",
        authorization_basis="The user authorized bounded continued GPU execution in this test.",
        authorized_at=datetime(2026, 9, 5, 18, 45, tzinfo=UTC),
        **{
            **builder_arguments,
            "run_id": fallback_acceptance_module.SECOND_RECOVERY_V8_RUN_ID,
            "prior_control_plane_incident_path": v4_control_plane_incident,
            "prior_v5_control_plane_incident_path": v5_control_plane_incident,
            "prior_v6_control_plane_incident_path": v6_control_plane_incident,
            "prior_v7_runtime_incident_path": v7_runtime_incident,
            "source_association": v8_source,
            "source_association_path": v8_source_path,
            "observed": terminal_v7_observed,
        },
    )
    assert built_v8["schema_version"] == "1.6.0"
    assert built_v8["authorized_recovery_run_id"] == (
        fallback_acceptance_module.SECOND_RECOVERY_V8_RUN_ID
    )
    assert built_v8["intervening_v7_runtime_incident"] == (
        fallback_acceptance_module._second_recovery_v7_runtime_incident_binding(
            v7_runtime_incident
        )
    )
    assert built_v8["cumulative_gpu_accounting"] == {
        "total_allocated_microseconds": 1_507_850_965,
        "event_count": 6,
        "service_session_count": 5,
        "by_kind_microseconds": {
            "failure": 225_183_297,
            "gpu_session_start": 214_034_494,
            "service_overhead": 645_767_518,
            "timeout": 422_865_656,
        },
    }
    assert built_v8["amendment"] == {
        **built_v7["amendment"],
        "additional_fallback_service_loads": 2,
        "prior_effective_accounting_events": 288,
        "amended_effective_accounting_events": 289,
    }
    assert built_v8["corrected_forecast"][
        "corrected_remaining_mandatory_forecast_seconds"
    ] == 29_459.0
    assert built_v8["corrected_forecast"][
        "actual_plus_remaining_and_service_start_seconds"
    ] == pytest.approx(31_358.251510)
    assert built_v8["corrected_forecast"]["scheduled_reserve_seconds"] == pytest.approx(
        1_041.748490
    )
    assert built_v8["corrected_forecast"][
        "protected_resource_sample_drain_seconds"
    ] == pytest.approx(120.0)
    assert built_v8["corrected_forecast"][
        "protected_shutdown_seconds"
    ] == pytest.approx(60.0)
    assert built_v8["corrected_forecast"][
        "protected_hard_stop_reserve_seconds"
    ] == pytest.approx(180.0)
    assert built_v8["corrected_forecast"][
        "hard_contingency_after_start_and_shutdown_seconds"
    ] == pytest.approx(4_461.748490)
    v8_runtime_incident = (
        ROOT
        / "artifacts/public/manifests/"
        "fallback_gpu_acceptance_development_v8_runtime_incident.json"
    )
    v8_lease_repair = (
        ROOT
        / "artifacts/restricted/recovery_validation/"
        "v8_terminal_lease_repair_20260905T2205Z/lease_repair_receipt.json"
    )
    terminal_v8_observed = GpuSummary(
        total_allocated_microseconds=2_581_267_703,
        event_count=7,
        service_session_count=6,
        by_kind_microseconds=(
            (GpuEventKind.FAILURE, 225_183_297),
            (GpuEventKind.GPU_SESSION_START, 441_721_080),
            (GpuEventKind.SERVICE_OVERHEAD, 1_491_497_670),
            (GpuEventKind.TIMEOUT, 422_865_656),
        ),
    )
    v9_arguments = {
        **builder_arguments,
        "run_id": fallback_acceptance_module.SECOND_RECOVERY_V9_RUN_ID,
        "prior_control_plane_incident_path": v4_control_plane_incident,
        "prior_v5_control_plane_incident_path": v5_control_plane_incident,
        "prior_v6_control_plane_incident_path": v6_control_plane_incident,
        "prior_v7_runtime_incident_path": v7_runtime_incident,
        "prior_v8_runtime_incident_path": v8_runtime_incident,
        "prior_v8_lease_repair_receipt_path": v8_lease_repair,
        "source_association": v9_source,
        "source_association_path": v9_source_path,
        "observed": terminal_v8_observed,
    }
    built_v9_path = restricted_root / "second-recovery-v9-built.proposed.json"
    built_v9 = build_second_fallback_recovery_overlay(
        output_path=built_v9_path,
        **v9_arguments,
    )
    assert built_v9["schema_version"] == "1.7.0"
    assert built_v9["authorization"]["status"] == "proposed"
    assert built_v9["authorized_recovery_run_id"] == (
        fallback_acceptance_module.SECOND_RECOVERY_V9_RUN_ID
    )
    assert built_v9["cumulative_gpu_accounting"] == {
        "total_allocated_microseconds": 2_581_267_703,
        "event_count": 7,
        "service_session_count": 6,
        "by_kind_microseconds": {
            "failure": 225_183_297,
            "gpu_session_start": 441_721_080,
            "service_overhead": 1_491_497_670,
            "timeout": 422_865_656,
        },
    }
    assert built_v9["amendment"] == {
        **built_v8["amendment"],
        "additional_fallback_service_loads": 3,
        "prior_effective_accounting_events": 289,
        "amended_effective_accounting_events": 290,
    }
    v9_forecast = built_v9["corrected_forecast"]
    assert v9_forecast["prior_actual_allocated_seconds"] == pytest.approx(2_581.267703)
    assert v9_forecast["corrected_remaining_mandatory_forecast_seconds"] == 29_459.0
    assert v9_forecast["additional_service_allocation_forecast_seconds"] == pytest.approx(
        391.40054529582005
    )
    assert v9_forecast["actual_plus_remaining_and_service_start_seconds"] == pytest.approx(
        32_431.66824829582
    )
    assert v9_forecast["scheduled_reserve_seconds"] == pytest.approx(-31.66824829582)
    assert v9_forecast["admitted"] is False
    assert v9_forecast[
        "hard_contingency_after_start_and_shutdown_seconds"
    ] == pytest.approx(3_388.33175170418)
    contingency = built_v9["essential_recovery_contingency"]
    assert contingency["authorized_service_start_count"] == 1
    assert contingency["contingency_inference_attempt_count"] == 0
    assert contingency["contingency_development_call_count"] == 0
    assert contingency["scheduled_admission_before_service_start"] is False
    assert contingency["hard_contingency_admission_before_service_start"] is True
    assert contingency[
        "maximum_service_increment_for_post_start_scheduled_admission_seconds"
    ] == pytest.approx(359.732297)
    with pytest.raises(PermissionError, match="remains proposed"):
        validate_second_fallback_recovery_overlay(
            root=ROOT,
            overlay_path=built_v9_path,
            v3_result_path=v3_result_path,
            v3_incident_path=v3_incident_path,
            prior_control_plane_incident_path=v4_control_plane_incident,
            prior_v5_control_plane_incident_path=v5_control_plane_incident,
            prior_v6_control_plane_incident_path=v6_control_plane_incident,
            prior_v7_runtime_incident_path=v7_runtime_incident,
            prior_v8_runtime_incident_path=v8_runtime_incident,
            prior_v8_lease_repair_receipt_path=v8_lease_repair,
            prior_retry_amendment_path=prior_amendment_path,
            prior_retry_failure_path=prior_failure_path,
            run_id=fallback_acceptance_module.SECOND_RECOVERY_V9_RUN_ID,
            policy=policy,
            activation_certificate=activation,
            primary_result=primary,
            limits=limits,
            source_association=v9_source,
            source_association_path=v9_source_path,
            retry_request=retry_request,
            legacy_provenance_bridge=bridge,
            observed=terminal_v8_observed,
            require_authorized=True,
            verify_decoder_compilation=False,
        )
    with pytest.raises(ValueError, match="intervening v4 incident"):
        build_second_fallback_recovery_overlay(
            output_path=restricted_root / "v5-missing-incident.json",
            **{**builder_arguments, "run_id": "fallback-qwen3-8b-awq-development-v5"},
        )
    with pytest.raises(ValueError, match="both terminal v4 and v5 incidents"):
        build_second_fallback_recovery_overlay(
            output_path=restricted_root / "v6-missing-incident.json",
            **{
                **builder_arguments,
                "run_id": fallback_acceptance_module.SECOND_RECOVERY_V6_RUN_ID,
                "source_association": v6_source,
                "source_association_path": v6_source_path,
            },
        )
    with pytest.raises(ValueError, match="terminal v4/v5/v6 incident chain"):
        build_second_fallback_recovery_overlay(
            output_path=restricted_root / "v7-missing-incident.json",
            **{
                **builder_arguments,
                "run_id": fallback_acceptance_module.SECOND_RECOVERY_V7_RUN_ID,
                "prior_control_plane_incident_path": v4_control_plane_incident,
                "prior_v5_control_plane_incident_path": v5_control_plane_incident,
                "source_association": v7_source,
                "source_association_path": v7_source_path,
            },
        )
    with pytest.raises(ValueError, match="terminal v4/v5/v6/v7 incident chain"):
        build_second_fallback_recovery_overlay(
            output_path=restricted_root / "v8-missing-incident.json",
            **{
                **builder_arguments,
                "run_id": fallback_acceptance_module.SECOND_RECOVERY_V8_RUN_ID,
                "prior_control_plane_incident_path": v4_control_plane_incident,
                "prior_v5_control_plane_incident_path": v5_control_plane_incident,
                "prior_v6_control_plane_incident_path": v6_control_plane_incident,
                "source_association": v8_source,
                "source_association_path": v8_source_path,
                "observed": terminal_v7_observed,
            },
        )
    with pytest.raises(ValueError, match="aware timestamp"):
        build_second_fallback_recovery_overlay(
            output_path=restricted_root / "missing-time.json",
            authorization_status="authorized",
            authorization_basis="Explicit test-only authorization.",
            authorized_at=None,
            **builder_arguments,
        )
    with pytest.raises(ValueError, match="explicitly restricted"):
        build_second_fallback_recovery_overlay(
            output_path=tmp_path / "public" / "escape.json",
            restricted_output_root=tmp_path / "public",
            **{
                key: value
                for key, value in builder_arguments.items()
                if key != "restricted_output_root"
            },
        )
    symlink_target = tmp_path / "private-target"
    symlink_target.mkdir()
    intermediate_link = tmp_path / "intermediate-link"
    intermediate_link.symlink_to(symlink_target, target_is_directory=True)
    linked_restricted_root = intermediate_link / "restricted"
    with pytest.raises(ValueError, match="ancestry"):
        build_second_fallback_recovery_overlay(
            output_path=linked_restricted_root / "redirected.json",
            restricted_output_root=linked_restricted_root,
            **{
                key: value
                for key, value in builder_arguments.items()
                if key != "restricted_output_root"
            },
        )
    assert not (symlink_target / "restricted").exists()

    outside_parent = tmp_path / "outside-not-created" / "nested"
    with pytest.raises(ValueError, match="escaped"):
        build_second_fallback_recovery_overlay(
            output_path=outside_parent / "escape.json",
            **builder_arguments,
        )
    assert not outside_parent.exists()


@pytest.mark.parametrize(
    ("version", "source_commit", "expected_manifest_sha256"),
    (
        (
            4,
            "fb3399690fe121e9d314f53e40337cdec415782c",
            "f53d2c75eee8a9e77c398b0d61ad7d3fde17630b790a96b7fc408ae8d391bddc",
        ),
        (
            5,
            "bfe43bca2f5a242a694b4a6998b7e384fe7dfc1b",
            "72a78b79500facc6a04fdf854c3ea73cba2d67d6ca13e13e9caf3c201d77a631",
        ),
        (
            6,
            "96ee3a39f38f8d5f49a58d2babd53fd4c1b5d5a4",
            "c61e0adf7bb6b2c369b9bc2dbd8bb863af68e03b3a9cf9b6b64de4846aa32592",
        ),
        (
            7,
            "0758e487740c608b7eb1060a7a940352032069ef",
            "96d5526d8160e4d84bac6e0b0044a4e4e5fe0bb7eff40698aa6615afcc869c0a",
        ),
        (
            8,
            "4b5b0cc768844f7661fc7ca316bd7e63cc15a2a6",
            "c9a0e50862d7c1301c2bae4fc73911014ab6ab44aebbbac1f0b6710ef159ad4e",
        ),
    ),
)
def test_frozen_v4_through_v8_overlays_replay_against_their_source_revisions(
    tmp_path: Path,
    version: int,
    source_commit: str,
    expected_manifest_sha256: str,
) -> None:
    available = subprocess.run(
        ["git", "cat-file", "-e", f"{source_commit}^{{commit}}"],
        cwd=ROOT,
        check=False,
        capture_output=True,
    )
    if available.returncode != 0:
        pytest.skip("historical recovery regression requires the repository commit history")
    archive = tmp_path / f"fallback-v{version}.tar"
    subprocess.run(
        ["git", "archive", "--format=tar", "--output", str(archive), source_commit],
        cwd=ROOT,
        check=True,
    )
    historical_root = tmp_path / f"fallback-v{version}-root"
    historical_root.mkdir()
    shutil.unpack_archive(archive, historical_root)

    policy = FallbackModelPolicy.load(
        historical_root / "configs/study/fallback_model.json"
    )
    limits = ResourceLimits.load(
        historical_root / "configs/study/resource_limits.json"
    )
    primary = json.loads(
        (
            historical_root
            / "artifacts/public/results/phase1_gpu_acceptance_v2_failed.json"
        ).read_text(encoding="utf-8")
    )
    activation = json.loads(
        (
            historical_root
            / "artifacts/public/manifests/fallback_activation_v2.json"
        ).read_text(encoding="utf-8")
    )
    bridge = Phase1LegacyEvidenceProvenanceBridge.load(historical_root)
    retry_request = build_fallback_acceptance_request(
        root=historical_root,
        call=fallback_pilot_calls(policy)[0],
        tokenizer=FakeTokenizer(),
        tokenizer_manifest=v3_fallback_tokenizer_manifest(),
        legacy_provenance_bridge=bridge,
    )
    source_association_path = (
        ROOT
        / "artifacts/public/manifests"
        / f"source_tree_fallback_second_recovery_v{version}.association.json"
    )
    source_association = json.loads(
        source_association_path.read_text(encoding="utf-8")
    )
    incident_paths = (
        ROOT
        / "artifacts/public/manifests/"
        "fallback_gpu_acceptance_development_v4_control_plane_incident.json",
        ROOT
        / "artifacts/public/manifests/"
        "fallback_gpu_acceptance_development_v5_control_plane_incident.json",
        ROOT
        / "artifacts/public/manifests/"
        "fallback_gpu_acceptance_development_v6_control_plane_incident.json",
        ROOT
        / "artifacts/public/manifests/fallback_gpu_acceptance_development_v7_runtime_incident.json",
    )
    supplied_incidents = incident_paths[: max(0, version - 4)]
    incident_arguments = {
        name: supplied_incidents[index] if index < len(supplied_incidents) else None
        for index, name in enumerate(
            (
                "prior_control_plane_incident_path",
                "prior_v5_control_plane_incident_path",
                "prior_v6_control_plane_incident_path",
                "prior_v7_runtime_incident_path",
            )
        )
    }
    overlay, _, _ = validate_second_fallback_recovery_overlay(
        root=historical_root,
        overlay_path=(
            ROOT / f"artifacts/restricted/fallback-second-recovery-v{version}.authorized.json"
        ),
        v3_result_path=(
            historical_root
            / "artifacts/public/results/fallback_gpu_acceptance_development_v3.json"
        ),
        v3_incident_path=(
            historical_root
            / "artifacts/public/manifests/fallback_gpu_acceptance_development_v3_incident.json"
        ),
        prior_retry_amendment_path=(
            historical_root / "configs/study/fallback_service_retry_amendment.json"
        ),
        prior_retry_failure_path=(
            historical_root
            / "artifacts/public/results/"
            "fallback_gpu_acceptance_development_v1.json.controller-handoff.json"
        ),
        run_id=getattr(fallback_acceptance_module, f"SECOND_RECOVERY_V{version}_RUN_ID"),
        policy=policy,
        activation_certificate=activation,
        primary_result=primary,
        limits=limits,
        source_association=source_association,
        source_association_path=source_association_path,
        retry_request=retry_request,
        legacy_provenance_bridge=bridge,
        require_authorized=True,
        verify_decoder_compilation=False,
        **incident_arguments,
    )
    assert overlay["manifest_sha256"] == expected_manifest_sha256


def test_second_recovery_rejects_self_consistent_substitute_v4_incident(
    tmp_path: Path,
) -> None:
    incident_path = (
        ROOT
        / "artifacts/public/manifests/"
        "fallback_gpu_acceptance_development_v4_control_plane_incident.json"
    )
    incident = json.loads(incident_path.read_text(encoding="utf-8"))
    incident["audited_at"] = "2026-09-05T14:42:25.000000Z"
    incident["manifest_sha256"] = canonical_sha256(
        {key: value for key, value in incident.items() if key != "manifest_sha256"}
    )
    substitute = tmp_path / incident_path.name
    substitute.write_text(canonical_json(incident) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="frozen public bytes"):
        fallback_acceptance_module._second_recovery_control_plane_incident_binding(
            substitute
        )


def test_second_recovery_integrity_file_binding_rejects_symlinks_and_missing_files(
    tmp_path: Path,
) -> None:
    regular = tmp_path / "regular.py"
    regular.write_text("value = 1\n", encoding="utf-8")
    linked = tmp_path / "linked.py"
    linked.symlink_to(regular)

    binding = fallback_acceptance_module._second_recovery_bound_file(
        tmp_path,
        "regular.py",
    )
    assert binding == {
        "path": "regular.py",
        "sha256": hashlib.sha256(regular.read_bytes()).hexdigest(),
    }
    with pytest.raises(ValueError, match="contains a symlink"):
        fallback_acceptance_module._second_recovery_bound_file(tmp_path, "linked.py")
    with pytest.raises(FileNotFoundError):
        fallback_acceptance_module._second_recovery_bound_file(tmp_path, "missing.py")


def test_second_recovery_semantic_scope_inventory_covers_every_explicit_runtime_use() -> None:
    discovered = set()
    for path in sorted((ROOT / "src/story_projection_onto").rglob("*.py")):
        if path.name == "fallback_acceptance.py":
            continue
        source = path.read_text(encoding="utf-8")
        if any(
            marker in source
            for marker in (
                "SemanticAssessmentScope",
                "semantic_assessment_scope",
                "runtime_structural_acceptance_record",
            )
        ):
            discovered.add(path.relative_to(ROOT).as_posix())
    declared = set(
        fallback_acceptance_module.SECOND_RECOVERY_INTEGRITY_SEMANTIC_SCOPE_SOURCE_PATHS
    )
    generated_contracts = {
        "schemas/jsonschema/ontology_projection.schema.json",
        "schemas/jsonschema/schema_manifest.json",
        "schemas/jsonschema/validated_generation.schema.json",
    }
    assert declared == discovered | generated_contracts | {
        "src/story_projection_onto/fallback_acceptance.py"
    }


def test_second_recovery_display_inventory_covers_every_production_callsite() -> None:
    discovered = set()
    for path in sorted((ROOT / "src/story_projection_onto").rglob("*.py")):
        if path.name == "fallback_acceptance.py":
            continue
        source = path.read_text(encoding="utf-8")
        if any(
            marker in source
            for marker in (
                "VisualizationContentScope.REGISTERED_DISPLAY",
                "compile_registered_display_selection",
            )
        ):
            discovered.add(path.relative_to(ROOT).as_posix())
    discovered.add("ui/app.js")
    assert discovered == set(
        fallback_acceptance_module.SECOND_RECOVERY_INTEGRITY_DISPLAY_SOURCE_PATHS
    )


def test_second_recovery_lifecycle_inventory_covers_every_production_writer() -> None:
    discovered = set()
    lifecycle_markers = (
        ".record_validation(",
        ".record_projection(",
        ".transition_job(",
        ".advance_job_lifecycle(",
        ".record_metric(",
        ".record_visualization(",
        "resolve_model_call_job(",
        "resolve_job_identity(",
        "resolve_projection_artifact(",
    )
    for path in sorted((ROOT / "src/story_projection_onto").rglob("*.py")):
        if path.name == "fallback_acceptance.py":
            continue
        source = path.read_text(encoding="utf-8")
        if any(marker in source for marker in lifecycle_markers):
            discovered.add(path.relative_to(ROOT).as_posix())
    declared = set(
        fallback_acceptance_module.SECOND_RECOVERY_INTEGRITY_LIFECYCLE_SOURCE_PATHS
    )
    assert discovered <= declared
    assert {
        "src/story_projection_onto/fallback_acceptance.py",
        "src/story_projection_onto/held_out_binding.py",
        "src/story_projection_onto/ledger_verify.py",
        "src/story_projection_onto/scorer_only/development_assessment.py",
    } <= declared


@pytest.mark.parametrize(
    "mutated_relative",
    [
        "prompts/c1_pre/prompt_v1.md",
        "tests/fixtures/phase1/c1_pre_request.json",
        "configs/study/decoding.json",
        "schemas/jsonschema/ontology_draft.schema.json",
    ],
)
def test_second_recovery_frozen_inputs_cannot_rebaseline_current_mutations(
    tmp_path: Path,
    mutated_relative: str,
) -> None:
    frozen_root = tmp_path / "frozen-input-root"
    required_paths = (
        fallback_acceptance_module.SECOND_RECOVERY_V3_SOURCE_MANIFEST_PATH,
        *fallback_acceptance_module.SECOND_RECOVERY_FROZEN_INPUT_SOURCE_PATHS,
    )
    for relative in required_paths:
        destination = frozen_root / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / relative, destination)

    frozen = fallback_acceptance_module._second_recovery_frozen_input_controls(
        frozen_root
    )
    assert frozen["whole_wire_payload_byte_identity_to_failed_v3_claimed"] is False

    mutated = frozen_root / mutated_relative
    mutated.write_text(mutated.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="frozen fallback input differs from v3"):
        fallback_acceptance_module._second_recovery_frozen_input_controls(frozen_root)


def test_standalone_cli_dispatches_only_to_registered_orchestrator(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    arguments = [
        "--execute",
        "--controller-stage",
        "orchestrate",
        "--project-root",
        str(ROOT),
        "--output",
        str(tmp_path / "result.json"),
        "--run-id",
        "fallback-cli-no-adopter",
    ]
    for name in (
        "primary-result",
        "activation-certificate",
        "cache-replacement-receipt",
        "snapshot",
        "shared-cache",
        "verified-model-manifest",
        "source-association",
        "ledger",
        "artifact-root",
        "checkpoint",
        "quota-root",
    ):
        arguments.extend((f"--{name}", str(tmp_path / name)))

    observed: list[object] = []

    def orchestrate(options: object) -> int:
        observed.append(options)
        return 7

    monkeypatch.setattr(
        "story_projection_onto.fallback_acceptance._orchestrate_controller_processes",
        orchestrate,
    )
    assert main(arguments) == 7
    assert len(observed) == 1

    assert not (tmp_path / "ledger").exists()
    assert not (tmp_path / "checkpoint").exists()
    assert not (tmp_path / "result.json").exists()


def test_public_launcher_establishes_a_dedicated_orchestrator_group(
    tmp_path: Path,
) -> None:
    program = (
        "import json, os; "
        "from story_projection_onto.fallback_acceptance import "
        "establish_fallback_orchestrator_process_group; "
        "arguments=['--execute','--controller-stage','orchestrate','--output',"
        f"{str(tmp_path / 'unused.json')!r}]; "
        "establish_fallback_orchestrator_process_group(arguments); "
        "print(json.dumps({'pid':os.getpid(),'pgrp':os.getpgrp()}))"
    )
    completed = subprocess.run(
        (sys.executable, "-c", program),
        check=True,
        capture_output=True,
        text=True,
    )
    identity = json.loads(completed.stdout)
    assert identity["pid"] == identity["pgrp"]


def test_second_recovery_builder_cli_dispatches_without_gpu_runtime(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observed: list[object] = []

    def build(options: object, *, root: Path) -> dict[str, object]:
        observed.extend((options, root))
        return {"authorization": {"status": "proposed"}}

    monkeypatch.setattr(
        fallback_acceptance_module,
        "_build_second_recovery_overlay_from_cli",
        build,
    )
    output = tmp_path / "restricted" / "proposal.json"
    assert (
        main(
            [
                "--build-second-recovery-overlay",
                "--project-root",
                str(ROOT),
                "--output",
                str(output),
            ]
        )
        == 0
    )
    assert len(observed) == 2
    assert observed[1] == ROOT
    assert not output.exists()


def test_checked_in_source_association_hash_is_verified(tmp_path: Path) -> None:
    path = ROOT / "artifacts/public/manifests/source_tree_phase1_pilot_v2.association.json"
    association = validate_source_association(path)
    assert association["local_tree_sha256"] == association["remote_tree_sha256"]

    tampered = dict(association)
    tampered["branch"] = "main"
    temporary = tmp_path / "source-association.json"
    temporary.write_text(json.dumps(tampered), encoding="utf-8")
    with pytest.raises(ValueError, match="hash"):
        validate_source_association(temporary)


def test_execution_source_association_rehashes_manifest_and_current_tree(
    tmp_path: Path,
) -> None:
    source_root = tmp_path / "source"
    source_root.mkdir()
    (source_root / "README.md").write_text("frozen\n", encoding="utf-8")
    manifest = build_source_manifest(source_root, "fallback-test-revision").to_dict()
    manifests = tmp_path / "manifests"
    manifests.mkdir()
    local_path = manifests / "source.local.json"
    local_path.write_text(
        json.dumps(manifest, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    payload: dict[str, object] = {
        "schema_version": "1.0.0",
        "kind": "local_remote_source_tree_association",
        "branch": "implementation/query-dependent-temporal-ontology",
        "revision_label": "fallback-test-revision",
        "local_manifest": local_path.name,
        "local_manifest_file_sha256": hashlib.sha256(local_path.read_bytes()).hexdigest(),
        "local_tree_sha256": manifest["tree_sha256"],
        "remote_tree_sha256": manifest["tree_sha256"],
    }
    association = {
        **payload,
        "manifest_sha256": canonical_sha256(payload),
    }
    association_path = manifests / "association.json"
    association_path.write_text(json.dumps(association), encoding="utf-8")

    validate_source_association(association_path, source_root=source_root)
    (source_root / "README.md").write_text("changed\n", encoding="utf-8")
    with pytest.raises(ValueError, match="current source tree"):
        validate_source_association(association_path, source_root=source_root)


def test_operator_probe_rejects_labels_without_behavior() -> None:
    from story_projection_onto.fallback_acceptance import (
        _require_call_operator_coverage,
    )

    policy = FallbackModelPolicy.load(ROOT / "configs/study/fallback_model.json")
    call = fallback_pilot_calls(policy)[1]
    output = with_legacy_source_hashes(
        json.loads((ROOT / "tests/fixtures/phase1/c2_query_output.json").read_text())
    )
    audit = validate_acceptance_generation(
        root=ROOT,
        call=call.acceptance_call(),
        parsed_object=output,
        legacy_provenance_bridge=legacy_provenance_bridge(),
    )
    behavior = _require_call_operator_coverage(call, audit, output, root=ROOT)
    assert behavior["behaviorally_valid"] is True

    spoofed = copy.deepcopy(output)
    split = next(decision for decision in spoofed["decisions"] if decision["operator"] == "split")
    split["created_object_ids"] = split["created_object_ids"][:1]
    spoofed_audit = validate_acceptance_generation(
        root=ROOT,
        call=call.acceptance_call(),
        parsed_object=spoofed,
        legacy_provenance_bridge=legacy_provenance_bridge(),
    )
    with pytest.raises(FallbackCapabilityBehaviorError, match="split"):
        _require_call_operator_coverage(call, spoofed_audit, spoofed, root=ROOT)


@dataclass
class FakeResourceSampler:
    storage: StoragePreflight | None = None
    samples: tuple[object, ...] = ()

    def sample(self, **kwargs: object) -> None:
        del kwargs


@dataclass
class FailingAfterFirstCallResourceSampler(FakeResourceSampler):
    def sample(self, **kwargs: object) -> None:
        if str(kwargs.get("sample_id", "")).endswith("-resources"):
            raise RuntimeError("synthetic post-call resource failure")


@dataclass
class FakeFallbackService:
    configuration: VLLMLaunchConfiguration
    ledger: Ledger
    outputs: Mapping[str, Mapping[str, object]]
    live: dict[str, object]
    state: ServiceState = ServiceState.STOPPED
    pid: int = 7301
    start_count: int = 0
    shutdown_count: int = 0
    resume_count: int = 0
    remaining_required_seconds: list[float] = field(default_factory=list)
    start_arguments: list[dict[str, object]] = field(default_factory=list)
    actual_allocated_service_seconds: float = 0.0
    periodic_resource_watchdog: ResourceWatchdog | None = None
    periodic_drain_allowed: bool = True

    def _record_event(
        self,
        *,
        event_id: str,
        event_kind: GpuEventKind,
        seconds: float,
        job_id: str | None = None,
        attempt_id: str | None = None,
        details: Mapping[str, object] | None = None,
    ) -> None:
        start = datetime(2026, 9, 3, tzinfo=UTC)
        self.ledger.record_gpu_event(
            event_id=event_id,
            event_kind=event_kind,
            allocated_seconds=seconds,
            started_at=start,
            ended_at=start + timedelta(seconds=seconds),
            succeeded=True,
            job_id=job_id,
            attempt_id=attempt_id,
            details=details,
        )

    def start(self, *, event_id: str, **kwargs: object) -> None:
        self.start_arguments.append({"event_id": event_id, **kwargs})
        self.remaining_required_seconds.append(cast(float, kwargs["remaining_required_seconds"]))
        self.start_count += 1
        started_at = datetime(2026, 9, 3, tzinfo=UTC)
        session_id = cast(str, kwargs["session_id"])
        allocation_baseline = self.ledger.gpu_summary().total_allocated_seconds
        self.ledger.record_gpu_service_observation(
            service_session_id=event_id,
            state=GpuServiceJournalState.OPENED,
            session_id=session_id,
            configuration_hash=self.configuration.configuration_hash,
            service_started_at=started_at,
            elapsed_seconds=0,
            ledger_allocated_seconds_before_session=allocation_baseline,
            hard_limit_seconds=36_000,
            observed_at=started_at,
            details={"fake_service_contract": "terminal_accounting_fixture"},
        )
        self._record_event(
            event_id=event_id,
            event_kind=GpuEventKind.GPU_SESSION_START,
            seconds=5,
        )
        self.live.update(
            {
                "running": True,
                "pid": self.pid,
                "session_id": session_id,
                "service_event_id": event_id,
                "service_started_at": started_at,
                "allocation_baseline": allocation_baseline,
            }
        )
        self.state = ServiceState.READY

    def write_resume_checkpoint(
        self,
        path: Path,
        *,
        controller_restart_handoff: bool = False,
    ) -> None:
        path.write_text(
            json.dumps(
                {
                    "configuration_hash": self.configuration.configuration_hash,
                    "controller_pid": os.getpid(),
                    "controller_restart_handoff": controller_restart_handoff,
                    "pid": self.pid,
                    "process_start_ticks": 987_654,
                    "session_id": "fallback-unit",
                    "accounting_session_id": "fallback-unit-service-start-001",
                }
            ),
            encoding="utf-8",
        )

    def detach_for_controller_restart(self, path: Path) -> None:
        self.write_resume_checkpoint(path, controller_restart_handoff=True)
        self.state = ServiceState.STOPPED

    def resume_from_checkpoint(
        self,
        path: Path,
        *,
        allow_same_controller_cleanup: bool = False,
    ) -> bool:
        checkpoint = json.loads(path.read_text(encoding="utf-8"))
        if (
            checkpoint["controller_restart_handoff"]
            and checkpoint["controller_pid"] == os.getpid()
            and not allow_same_controller_cleanup
        ):
            raise RuntimeError("same controller")
        self.resume_count += 1
        if (
            checkpoint["configuration_hash"] != self.configuration.configuration_hash
            or checkpoint["pid"] != self.live.get("pid")
            or self.live.get("running") is not True
        ):
            return False
        self.pid = cast(int, checkpoint["pid"])
        self.state = ServiceState.READY
        return True

    def _generation(self, request: GuidedJSONRequest) -> GenerationResult:
        parsed = self.outputs[request.request_id]
        body = json.dumps(
            {
                "id": f"private-{request.request_id}",
                "choices": [
                    {
                        "message": {"content": json.dumps(parsed)},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {
                    "prompt_tokens": request.rendered_input_token_count,
                    "completion_tokens": 50,
                },
            },
            sort_keys=True,
        ).encode()
        return GenerationResult(
            request_id=request.request_id,
            request_hash=request.request_hash,
            response_sha256=hashlib.sha256(body).hexdigest(),
            parsed_object=parsed,
            raw_response=body,
            prompt_tokens=request.rendered_input_token_count,
            completion_tokens=50,
            finish_reason="stop",
            service_request_id=f"private-{request.request_id}",
        )

    def run_fallback_test(
        self,
        request: GuidedJSONRequest,
        *,
        event_id: str,
        reserve_call_class: str,
        reserve_reservation_id: str,
        job_id: str,
        attempt_id: str,
        **kwargs: object,
    ) -> GenerationResult:
        self.remaining_required_seconds.append(cast(float, kwargs["remaining_required_seconds"]))
        self._record_event(
            event_id=event_id,
            event_kind=GpuEventKind.FALLBACK_TEST,
            seconds=1,
            job_id=job_id,
            attempt_id=attempt_id,
            details={
                "request_id": request.request_id,
                "request_hash": request.request_hash,
                "reserve_call_class": reserve_call_class,
                "reserve_reservation_id": reserve_reservation_id,
            },
        )
        return self._generation(request)

    def generate(
        self,
        request: GuidedJSONRequest,
        *,
        event_id: str,
        job_id: str,
        attempt_id: str,
        accounting_details: Mapping[str, object],
        **kwargs: object,
    ) -> GenerationResult:
        self.remaining_required_seconds.append(cast(float, kwargs["remaining_required_seconds"]))
        self._record_event(
            event_id=event_id,
            event_kind=GpuEventKind.REPAIR,
            seconds=1,
            job_id=job_id,
            attempt_id=attempt_id,
            details={
                "request_id": request.request_id,
                "request_hash": request.request_hash,
                **accounting_details,
            },
        )
        return self._generation(request)

    def require_hard_stop_margin(self) -> None:
        return None

    def emergency_stop(self) -> None:
        self.shutdown()

    def request_emergency_stop(self) -> None:
        self.emergency_stop()

    def start_periodic_resource_watchdog(
        self,
        watchdog: ResourceWatchdog,
    ) -> ResourceWatchdog:
        if self.periodic_resource_watchdog is not None:
            raise RuntimeError("test service already owns a resource watchdog")
        self.periodic_resource_watchdog = watchdog
        return watchdog.start()

    def stop_periodic_resource_watchdog(
        self,
        watchdog: ResourceWatchdog,
        *,
        raise_failure: bool = True,
        completion_timeout_seconds: float | None = None,
    ) -> bool:
        owned = self.periodic_resource_watchdog
        if owned is not None and owned is not watchdog:
            raise RuntimeError("test service does not own this resource watchdog")
        if not self.periodic_drain_allowed:
            watchdog._stop.set()
            return False
        drained = watchdog.stop(
            raise_failure=raise_failure,
            completion_timeout_seconds=completion_timeout_seconds,
        )
        if drained:
            self.periodic_resource_watchdog = None
        return drained

    def shutdown(self) -> ServiceUptime | None:
        owned = self.periodic_resource_watchdog
        if owned is not None and not self.stop_periodic_resource_watchdog(
            owned,
            raise_failure=False,
        ):
            raise RuntimeError(
                "cannot terminate fake vLLM while periodic sample remains owned"
            )
        self.shutdown_count += 1
        if self.live.get("running") is not True:
            self.state = ServiceState.STOPPED
            return None
        self.live["running"] = False
        self.state = ServiceState.STOPPED
        session_id = cast(str, self.live["session_id"])
        service_event_id = cast(str, self.live["service_event_id"])
        start = cast(datetime, self.live["service_started_at"])
        baseline = cast(float, self.live["allocation_baseline"])
        classified_seconds = self.ledger.gpu_summary().total_allocated_seconds - baseline
        seconds = classified_seconds
        ended_at = start + timedelta(seconds=seconds)
        self.ledger.record_gpu_service_observation(
            service_session_id=service_event_id,
            state=GpuServiceJournalState.PROCESS_STOPPED,
            session_id=session_id,
            configuration_hash=self.configuration.configuration_hash,
            service_started_at=start,
            elapsed_seconds=seconds,
            ledger_allocated_seconds_before_session=baseline,
            hard_limit_seconds=36_000,
            observed_at=ended_at,
            details={"fake_process_absence_verified": True},
        )
        self.ledger.close_gpu_service_journal(
            service_session_id=service_event_id,
            session_id=session_id,
            service_seconds=seconds,
            classified_event_seconds=classified_seconds,
            started_at=start,
            ended_at=ended_at,
            details={"fake_service_contract": "terminal_accounting_fixture"},
        )
        return ServiceUptime(
            session_id=session_id,
            started_at=start,
            ended_at=ended_at,
            service_seconds=seconds,
            allocated_event_seconds=seconds,
        )


def development_adopter_registration() -> DevelopmentAdopterRegistration:
    manifest = load_development_call_manifest(ROOT)
    return DevelopmentAdopterRegistration(
        adopter_id="development-unit-adopter",
        implementation_sha256="1" * 64,
        source_sha256="6" * 64,
        development_call_manifest_sha256=manifest.content_hash,
        prequery_input_builder_sha256="3" * 64,
    )


def _preexisting_development_conditions(
    unit_id: str,
) -> tuple[ConditionName, ...]:
    return {
        "dev-unit-01": (
            ConditionName.C0_CLASSICAL_PRE,
            ConditionName.C2_LLM_QUERY,
            ConditionName.A_NO_CONTEXT,
        ),
        "dev-unit-02": (
            ConditionName.C0_CLASSICAL_PRE,
            ConditionName.C2_LLM_QUERY,
            ConditionName.A_NO_TEMPORAL_EPISTEMIC,
        ),
        "dev-unit-03": (
            ConditionName.C0_CLASSICAL_PRE,
            ConditionName.C2_LLM_QUERY,
            ConditionName.A_NO_RARE_GUARD,
        ),
        "dev-unit-04": (
            ConditionName.C0_CLASSICAL_PRE,
            ConditionName.C2_LLM_QUERY,
        ),
    }[unit_id]


def _development_prequery_inputs(
    manifest: DevelopmentCallManifest,
    bootstrap: DevelopmentContinuationBootstrap,
) -> DevelopmentPrequeryInputs:
    bindings: list[UnitPrequeryBinding] = []
    completed = datetime(2026, 9, 3, tzinfo=UTC)
    for unit_id in ("dev-unit-01", "dev-unit-02", "dev-unit-03", "dev-unit-04"):
        prequery = next(call.prequery_stage for call in manifest.calls if call.unit_id == unit_id)
        evidence = json.loads(
            (ROOT / prequery.relative_path / "evidence.json").read_text(encoding="utf-8")
        )
        snapshot = evidence["snapshot"]
        preparations = tuple(
            PrequeryPreparationBinding(
                unit_id=snapshot["world_or_window_id"],
                condition=condition,
                seed_block=(
                    None if condition is ConditionName.C0_CLASSICAL_PRE else manifest.seed_block
                ),
                snapshot_hash=snapshot["content_hash"],
                preparation_hash=hashlib.sha256(
                    f"{unit_id}:{condition.value}:preparation".encode()
                ).hexdigest(),
                lineage_artifact_hash=hashlib.sha256(
                    f"{unit_id}:{condition.value}:lineage".encode()
                ).hexdigest(),
                completed_at=completed,
            )
            for condition in _preexisting_development_conditions(unit_id)
        )
        bindings.append(
            UnitPrequeryBinding(
                unit_id=unit_id,
                runtime_unit_id=snapshot["world_or_window_id"],
                snapshot_hash=snapshot["content_hash"],
                staged_model_visible_evidence_hash=prequery.evidence_artifact_hash,
                neutral_full_evidence_artifact_hash=hashlib.sha256(
                    f"{unit_id}:neutral".encode()
                ).hexdigest(),
                evidence_equivalence_certificate_hash=hashlib.sha256(
                    f"{unit_id}:equivalence".encode()
                ).hexdigest(),
                preexisting_preparation_bindings=preparations,
                completed_at=completed,
            )
        )
    config_hashes = tuple(
        None
        if call.kind is DevelopmentCallKind.FIXED_SELECTION
        else hashlib.sha256(f"config:{call.call_id}".encode()).hexdigest()
        for call in manifest.calls
    )
    fixed_plans = tuple(
        FixedSchemaDerivationPlan(
            ordinal=call.ordinal,
            call_id=call.call_id,
            unit_id=call.unit_id,
            source_c1_call_id=call.source_c1_call_id,
            budgets=OutputBudgets(
                node_budget=10,
                assertion_budget=20,
                display_node_budget=10,
                display_assertion_budget=20,
            ),
            model_stack_hash="5" * 64,
            decoding_family_hash="6" * 64,
            seed_manifest_hash="7" * 64,
            prompt_hash="8" * 64,
            base_output_schema_hash="9" * 64,
            scored_schema_hash="a" * 64,
            capability_manifest_hash="b" * 64,
            validator_hash="4" * 64,
            upper_ontology_hash="1" * 64,
            prequery_evidence_artifact_hash=call.prequery_stage.evidence_artifact_hash,
            query_stage_manifest_hash=call.query_stage.staging_manifest_hash,
        )
        for call in manifest.calls
        if call.kind is DevelopmentCallKind.FIXED_SELECTION
    )
    return DevelopmentPrequeryInputs(
        source_tree_hash=bootstrap.source_tree_hash,
        selected_model_freeze_hash=bootstrap.selected_model_freeze_hash,
        upper_ontology_hash="1" * 64,
        prompt_family_hash="2" * 64,
        output_schema_hash="3" * 64,
        validator_hash="4" * 64,
        run_condition_config_hashes=config_hashes,
        fixed_schema_derivation_plans=fixed_plans,
        unit_bindings=tuple(bindings),
    )


@dataclass
class FakeDevelopmentServiceAdapter:
    service: FakeFallbackService
    identity_value: LiveServiceIdentity

    def identity(self) -> LiveServiceIdentity:
        return self.identity_value

    def allocated_gpu_seconds(self) -> float:
        return self.service.ledger.gpu_summary().total_allocated_seconds

    def open_query(self, *args: object, **kwargs: object) -> object:
        del args, kwargs
        raise AssertionError("fake development continuation must not open a query")

    def execute_call(self, *args: object, **kwargs: object) -> object:
        del args, kwargs
        raise AssertionError("fake development continuation has no model calls")

    def recover_call(self, *args: object, **kwargs: object) -> None:
        del args, kwargs
        return None


@dataclass
class FakeDevelopmentAdopter:
    service: FakeFallbackService
    checkpoint_path: Path
    registration_value: DevelopmentAdopterRegistration = field(init=False)
    handoffs: list[DevelopmentContinuationHandoff] = field(init=False)
    adapter: FakeDevelopmentServiceAdapter | None = field(init=False, default=None)
    prepared_value: PreparedDevelopmentContinuation | None = field(
        init=False,
        default=None,
    )

    def __post_init__(self) -> None:
        self.registration_value = development_adopter_registration()
        self.handoffs: list[DevelopmentContinuationHandoff] = []

    def registration(self) -> DevelopmentAdopterRegistration:
        return self.registration_value

    def prepare(
        self,
        bootstrap: DevelopmentContinuationBootstrap,
    ) -> PreparedDevelopmentContinuation:
        identity = LiveServiceIdentity(
            owner_run_id=bootstrap.owner_run_id,
            service_pid=bootstrap.service_pid,
            service_start_ticks=bootstrap.service_start_ticks,
            gpu_session_event_id=bootstrap.gpu_session_event_id,
            launcher_configuration_hash=bootstrap.launcher_configuration_hash,
            model_snapshot_hash=bootstrap.model_snapshot_manifest_hash,
            selected_model_freeze_hash=bootstrap.selected_model_freeze_hash,
            source_execution_hash=bootstrap.source_tree_hash,
        )
        self.adapter = FakeDevelopmentServiceAdapter(self.service, identity)
        manifest = load_development_call_manifest(ROOT)
        inputs = _development_prequery_inputs(manifest, bootstrap)
        forecast = ForecastControl(
            forecast_receipt_hash="a" * 64,
            gpu_call_inventory_file_sha256=(manifest.gpu_call_inventory_file_sha256),
            post_development_mandatory_forecast_seconds=123.0,
        )
        self.prepared_value = PreparedDevelopmentContinuation(
            call_manifest=manifest,
            prequery_inputs=inputs,
            forecast_control=forecast,
            execution_id="development-unit-execution",
            execution_manifest_hash="b" * 64,
            checkpoint_path=self.checkpoint_path,
            service_adapter=cast(InjectedLiveDevelopmentService, self.adapter),
        )
        return self.prepared_value

    def adopt_and_run(
        self,
        handoff: DevelopmentContinuationHandoff,
        service: InjectedLiveDevelopmentService,
    ) -> DevelopmentExecutionResult:
        assert self.service.state is ServiceState.READY
        assert self.service.live["running"] is True
        assert service is self.adapter
        assert self.prepared_value is not None
        self.handoffs.append(handoff)
        self.checkpoint_path.write_text("development complete\n", encoding="utf-8")
        allocated = self.service.ledger.gpu_summary().total_allocated_seconds
        manifest = self.prepared_value.call_manifest
        inputs = self.prepared_value.prequery_inputs
        completed = datetime(2026, 9, 3, 1, tzinfo=UTC)

        def digest(label: str) -> str:
            return hashlib.sha256(label.encode()).hexdigest()

        access_by_stage: dict[str, QueryAccessEvent] = {}
        for call in manifest.calls:
            if call.query_stage is None:
                continue
            stage = call.query_stage
            if stage.staging_manifest_hash in access_by_stage:
                continue
            binding = inputs.binding_for(call.unit_id)
            access_by_stage[stage.staging_manifest_hash] = QueryAccessEvent(
                access_event_id=f"access-{len(access_by_stage) + 1:02d}",
                execution_id=handoff.development_execution_id,
                query_context_hash=digest(f"context:{stage.stage_id}"),
                model_visible_query_hash=digest(f"visible:{stage.stage_id}"),
                snapshot_hash=binding.snapshot_hash,
                stage_manifest_hash=stage.staging_manifest_hash,
                query_artifact_hash=stage.query_artifact_hash,
                prequery_barrier_hash="d" * 64,
                registered_revealed_at=completed,
                accessed_at=completed,
            )
        rows: list[DevelopmentITTRecord] = []
        for call in manifest.calls:
            preparations: tuple[PrequeryPreparationBinding, ...] = ()
            preparation_hash = None
            attempt_hash = None
            if call.call_class == "development_c1":
                binding = inputs.binding_for(call.unit_id)
                preparation_hash = digest(f"{call.call_id}:preparation")
                preparations = tuple(
                    PrequeryPreparationBinding(
                        unit_id=binding.runtime_unit_id,
                        condition=condition,
                        seed_block=call.seed_block,
                        snapshot_hash=binding.snapshot_hash,
                        preparation_hash=(
                            preparation_hash
                            if condition is ConditionName.C1_LLM_PRE
                            else digest(f"{call.call_id}:fixed-preparation")
                        ),
                        lineage_artifact_hash=digest(f"{call.call_id}:{condition.value}:lineage"),
                        completed_at=completed,
                    )
                    for condition in (
                        ConditionName.C1_LLM_PRE,
                        ConditionName.A_FIXED_SELECT,
                    )
                )
            else:
                attempt_hash = digest(f"{call.call_id}:attempt")
            access = (
                None
                if call.query_stage is None
                else access_by_stage[call.query_stage.staging_manifest_hash]
            )
            fixed_derivation = None
            run_config_hash = (
                inputs.run_config_hash_for(call.ordinal)
                if call.kind is not DevelopmentCallKind.FIXED_SELECTION
                else digest(f"{call.call_id}:derived-config")
            )
            if call.kind is DevelopmentCallKind.FIXED_SELECTION:
                plan = inputs.fixed_schema_plan_for(call.ordinal)
                fixed_derivation = FixedSchemaDerivationReceipt(
                    plan_hash=plan.content_hash,
                    call_id=call.call_id,
                    unit_id=call.unit_id,
                    source_c1_construction_seal_hash=digest(f"{call.source_c1_call_id}:seal"),
                    source_c1_preparation_hash=digest(f"{call.source_c1_call_id}:preparation"),
                    fixed_ontology_hash=digest(f"{call.call_id}:fixed-ontology"),
                    evidence_alias_bijection_hash=digest(f"{call.call_id}:aliases"),
                    derived_output_schema_hash=digest(f"{call.call_id}:schema"),
                    derived_decoding_manifest_hash=digest(f"{call.call_id}:decoding"),
                    exact_run_condition_config_hash=run_config_hash,
                    exact_run_condition_config_artifact_hash=digest(
                        f"{call.call_id}:config-artifact"
                    ),
                    derived_at=completed - timedelta(seconds=1),
                )
            rows.append(
                DevelopmentITTRecord(
                    ordinal=call.ordinal,
                    call_id=call.call_id,
                    call_class=call.call_class,
                    condition=call.condition,
                    unit_id=call.unit_id,
                    outcome=RunOutcome.SUCCEEDED,
                    request_start_state=RequestStartState.STARTED,
                    service_result_hash=digest(f"{call.call_id}:result"),
                    response_artifact_hash=digest(f"{call.call_id}:response"),
                    validated_generation_hash=digest(f"{call.call_id}:validated"),
                    validation_record_hash=digest(f"{call.call_id}:validation"),
                    condition_attempt_hash=attempt_hash,
                    condition_preparation_hash=preparation_hash,
                    run_condition_config_hash=run_config_hash,
                    fixed_schema_derivation=fixed_derivation,
                    ledger_receipt_hash=digest(f"{call.call_id}:ledger"),
                    gpu_event_id=f"gpu-{call.call_id}",
                    query_access_event_hash=(None if access is None else access.content_hash),
                    prequery_preparation_bindings=preparations,
                    allocated_gpu_seconds=0,
                    prompt_tokens=1,
                    completion_tokens=1,
                    completed_at=completed,
                )
            )
        forecast = DevelopmentForecastResult(
            forecast_receipt_hash=handoff.forecast_receipt_hash,
            gpu_call_inventory_file_sha256=handoff.gpu_call_inventory_file_sha256,
            timing_by_call_class=(),
            actual_allocated_seconds_before=allocated,
            development_allocated_seconds=0,
            actual_allocated_seconds_after=allocated,
            post_development_mandatory_forecast_seconds=(
                handoff.post_development_mandatory_forecast_seconds
            ),
            actual_plus_remaining_seconds=(
                allocated + handoff.post_development_mandatory_forecast_seconds
            ),
            scheduled_limit_seconds=9 * 3600,
            hard_limit_seconds=10 * 3600,
            scheduled_admitted=True,
            below_hard_stop=True,
        )
        gate = DevelopmentGateResult(
            exact_24_call_manifest=True,
            every_planned_call_has_itt_row=True,
            every_planned_request_started=True,
            every_planned_call_succeeded=True,
            twelve_query_access_events=True,
            same_live_service_identity=True,
            forecast_admitted=True,
            scientific_assessment_hash="e" * 64,
            scientific_thresholds_passed=True,
            passed=True,
        )
        return DevelopmentExecutionResult(
            execution_id=handoff.development_execution_id,
            execution_manifest_hash=handoff.development_execution_manifest_hash,
            call_manifest_hash=handoff.development_call_manifest_hash,
            source_plan_hash=handoff.development_source_plan_hash,
            service_identity_hash=handoff.live_service_identity_hash,
            prequery_inputs_hash=handoff.development_prequery_inputs_hash,
            preconstruction_barrier_hash="c" * 64,
            prequery_barrier_hash="d" * 64,
            phase=DevelopmentPhase.COMPLETED,
            itt_records=tuple(rows),
            query_access_events=tuple(access_by_stage.values()),
            forecast=forecast,
            gate=gate,
        )


def _fallback_launch_configuration(tmp_path: Path) -> VLLMLaunchConfiguration:
    cache = tmp_path / "cache"
    snapshot = cache / "hub" / "models--Qwen--Qwen3-8B-AWQ" / "snapshots" / FALLBACK_MODEL_REVISION
    snapshot.mkdir(parents=True)
    return VLLMLaunchConfiguration.from_model_configuration(
        snapshot_path=snapshot,
        shared_cache=cache,
        model_configuration_path=ROOT / "configs/study/model.json",
        model_candidate="fallback",
        verified_snapshot_manifest_sha256=HASH_A,
    )


def _runner(
    *,
    tmp_path: Path,
    ledger: Ledger,
    service: FakeFallbackService,
    register_development_adopter: bool = True,
) -> FallbackAcceptanceRunner:
    adopter = (
        FakeDevelopmentAdopter(service, tmp_path / "development.checkpoint.json")
        if register_development_adopter
        else None
    )
    return FallbackAcceptanceRunner(
        root=ROOT,
        legacy_provenance_bridge=legacy_provenance_bridge(),
        run_id="fallback-unit",
        service=cast(object, service),
        ledger=ledger,
        artifacts=ArtifactStore(BlobStore(tmp_path / "blobs"), ledger),
        resource_sampler=cast(
            object,
            FakeResourceSampler(storage=StoragePreflight(tmp_path)),
        ),
        tokenizer=FakeTokenizer(),
        tokenizer_manifest=fallback_tokenizer_manifest(),
        checkpoint_path=tmp_path / "fallback.checkpoint.json",
        activation_certificate={"manifest_sha256": HASH_A},
        replacement_receipt={"manifest_sha256": HASH_B},
        snapshot_manifest={
            "repository": FALLBACK_MODEL_REPOSITORY,
            "revision": FALLBACK_MODEL_REVISION,
            "manifest_sha256": "c" * 64,
        },
        source_association={
            "manifest_sha256": "d" * 64,
            "local_tree_sha256": "e" * 64,
        },
        development_adopter=adopter,
    )


@dataclass(frozen=True)
class _ValidatedV9Fixture:
    overlay: Mapping[str, object]
    v3_result: Mapping[str, object]
    v3_incident: Mapping[str, object]
    retry_amendment: Mapping[str, object]
    prior_failure: Mapping[str, object]
    activation_certificate: Mapping[str, object]
    source_association: Mapping[str, object]
    validation_inputs: fallback_acceptance_module._SecondRecoveryValidationInputs
    primary_result: Mapping[str, object]


def _terminal_v8_gpu_summary() -> GpuSummary:
    return GpuSummary(
        total_allocated_microseconds=2_581_267_703,
        event_count=7,
        service_session_count=6,
        by_kind_microseconds=(
            (GpuEventKind.FAILURE, 225_183_297),
            (GpuEventKind.GPU_SESSION_START, 441_721_080),
            (GpuEventKind.SERVICE_OVERHEAD, 1_491_497_670),
            (GpuEventKind.TIMEOUT, 422_865_656),
        ),
    )


def _seed_terminal_v8_gpu_accounting(ledger: Ledger) -> None:
    started = datetime(2026, 9, 5, 22, tzinfo=UTC)
    event_rows = (
        (
            "fallback-qwen3-8b-awq-development-v3-fallback-c1-01-gpu",
            GpuEventKind.FAILURE,
            0.852878,
            False,
            {
                "reserve_call_class": "reserve_long",
                "reserve_reservation_id": (
                    "fallback-qwen3-8b-awq-development-v3:fallback-c1-01"
                ),
            },
        ),
        (
            "base-service-start-failure-001",
            GpuEventKind.FAILURE,
            100.0,
            False,
            {"intended_event_kind": GpuEventKind.GPU_SESSION_START.value},
        ),
        (
            "base-service-start-failure-002",
            GpuEventKind.FAILURE,
            124.330419,
            False,
            {"intended_event_kind": GpuEventKind.GPU_SESSION_START.value},
        ),
        (
            "base-service-start-timeout-003",
            GpuEventKind.TIMEOUT,
            422.865656,
            False,
            {"intended_event_kind": GpuEventKind.GPU_SESSION_START.value},
        ),
        (
            "fallback-qwen3-8b-awq-development-v3-service-start-001",
            GpuEventKind.GPU_SESSION_START,
            441.721078,
            True,
            {},
        ),
        (
            "fallback-qwen3-8b-awq-development-v7-service-start-001",
            GpuEventKind.GPU_SESSION_START,
            0.000001,
            True,
            {},
        ),
        (
            "fallback-qwen3-8b-awq-development-v8-service-start-001",
            GpuEventKind.GPU_SESSION_START,
            0.000001,
            True,
            {},
        ),
    )
    for event_id, kind, seconds, succeeded, details in event_rows:
        ledger.record_gpu_event(
            event_id=event_id,
            event_kind=kind,
            allocated_seconds=seconds,
            started_at=started,
            ended_at=started + timedelta(seconds=seconds),
            succeeded=succeeded,
            details=details,
        )
    overhead_rows = (1_491.497665, 0.000001, 0.000001, 0.000001, 0.000001, 0.000001)
    for index, seconds in enumerate(overhead_rows, start=1):
        ledger.record_gpu_service_session(
            service_session_id=f"terminal-v8-session-{index:03d}",
            session_id=f"terminal-v8-{index:03d}",
            service_seconds=seconds,
            classified_event_seconds=0,
            started_at=started,
            ended_at=started + timedelta(seconds=seconds),
        )
    assert ledger.gpu_summary() == _terminal_v8_gpu_summary()


def _build_validated_v9_fixture(tmp_path: Path) -> _ValidatedV9Fixture:
    policy = FallbackModelPolicy.load(ROOT / "configs/study/fallback_model.json")
    limits = ResourceLimits.load(ROOT / "configs/study/resource_limits.json")
    primary_path = (
        ROOT / "artifacts/public/results/phase1_gpu_acceptance_v2_failed.json"
    )
    activation_path = (
        ROOT / "artifacts/public/manifests/fallback_activation_v2.json"
    )
    primary = json.loads(primary_path.read_text(encoding="utf-8"))
    activation = json.loads(activation_path.read_text(encoding="utf-8"))
    v3_result_path = (
        ROOT / "artifacts/public/results/fallback_gpu_acceptance_development_v3.json"
    )
    v3_incident_path = (
        ROOT
        / "artifacts/public/manifests/"
        "fallback_gpu_acceptance_development_v3_incident.json"
    )
    retry_amendment_path = (
        ROOT / "configs/study/fallback_service_retry_amendment.json"
    )
    prior_failure_path = (
        ROOT
        / "artifacts/public/results/"
        "fallback_gpu_acceptance_development_v1.json.controller-handoff.json"
    )
    incident_paths = (
        ROOT
        / "artifacts/public/manifests/"
        "fallback_gpu_acceptance_development_v4_control_plane_incident.json",
        ROOT
        / "artifacts/public/manifests/"
        "fallback_gpu_acceptance_development_v5_control_plane_incident.json",
        ROOT
        / "artifacts/public/manifests/"
        "fallback_gpu_acceptance_development_v6_control_plane_incident.json",
        ROOT
        / "artifacts/public/manifests/"
        "fallback_gpu_acceptance_development_v7_runtime_incident.json",
        ROOT
        / "artifacts/public/manifests/"
        "fallback_gpu_acceptance_development_v8_runtime_incident.json",
    )
    lease_repair_path = (
        ROOT
        / "artifacts/restricted/recovery_validation/"
        "v8_terminal_lease_repair_20260905T2205Z/lease_repair_receipt.json"
    )

    source_directory = tmp_path / "v9-source"
    source_directory.mkdir()
    revision = fallback_acceptance_module.SECOND_RECOVERY_V9_SOURCE_REVISION
    source_manifest = build_source_manifest(ROOT, revision).to_dict()
    source_manifest_bytes = (
        json.dumps(source_manifest, ensure_ascii=False, indent=2, sort_keys=True)
        + "\n"
    )
    local_manifest_path = source_directory / "source_tree_v9.local.json"
    remote_manifest_path = source_directory / "source_tree_v9.remote.json"
    local_manifest_path.write_text(source_manifest_bytes, encoding="utf-8")
    remote_manifest_path.write_text(source_manifest_bytes, encoding="utf-8")
    source_association = build_source_association(
        local_manifest_path=local_manifest_path,
        remote_manifest_path=remote_manifest_path,
        branch="implementation/query-dependent-temporal-ontology",
        git_commit="0" * 40,
        revision_label=revision,
        recorded_at=datetime(2026, 9, 5, 22, 30, tzinfo=UTC),
    )
    source_association_path = (
        source_directory
        / "source_tree_fallback_second_recovery_v9.association.json"
    )
    source_association_path.write_text(
        json.dumps(source_association, ensure_ascii=False, indent=2, sort_keys=True)
        + "\n",
        encoding="utf-8",
    )
    validated_source = validate_source_association(
        source_association_path,
        source_root=ROOT,
    )
    retry_request = build_fallback_acceptance_request(
        root=ROOT,
        call=fallback_pilot_calls(policy)[0],
        tokenizer=FakeTokenizer(),
        tokenizer_manifest=v3_fallback_tokenizer_manifest(),
        legacy_provenance_bridge=legacy_provenance_bridge(),
    )
    restricted_root = tmp_path / "restricted"
    overlay_path = restricted_root / "fallback-second-recovery-v9.authorized.json"
    overlay = build_second_fallback_recovery_overlay(
        root=ROOT,
        output_path=overlay_path,
        restricted_output_root=restricted_root,
        v3_result_path=v3_result_path,
        v3_incident_path=v3_incident_path,
        prior_control_plane_incident_path=incident_paths[0],
        prior_v5_control_plane_incident_path=incident_paths[1],
        prior_v6_control_plane_incident_path=incident_paths[2],
        prior_v7_runtime_incident_path=incident_paths[3],
        prior_v8_runtime_incident_path=incident_paths[4],
        prior_v8_lease_repair_receipt_path=lease_repair_path,
        prior_retry_amendment_path=retry_amendment_path,
        prior_retry_failure_path=prior_failure_path,
        run_id=fallback_acceptance_module.SECOND_RECOVERY_V9_RUN_ID,
        policy=policy,
        activation_certificate=activation,
        primary_result=primary,
        limits=limits,
        source_association=validated_source,
        source_association_path=source_association_path,
        retry_request=retry_request,
        legacy_provenance_bridge=legacy_provenance_bridge(),
        observed=_terminal_v8_gpu_summary(),
        authorization_status="authorized",
        authorization_basis="Explicit test-only V9 authorization.",
        authorized_at=datetime(2026, 9, 5, 22, 31, tzinfo=UTC),
        verify_decoder_compilation=False,
    )
    validated_overlay, v3_result, v3_incident = (
        validate_second_fallback_recovery_overlay(
            root=ROOT,
            overlay_path=overlay_path,
            v3_result_path=v3_result_path,
            v3_incident_path=v3_incident_path,
            prior_control_plane_incident_path=incident_paths[0],
            prior_v5_control_plane_incident_path=incident_paths[1],
            prior_v6_control_plane_incident_path=incident_paths[2],
            prior_v7_runtime_incident_path=incident_paths[3],
            prior_v8_runtime_incident_path=incident_paths[4],
            prior_v8_lease_repair_receipt_path=lease_repair_path,
            prior_retry_amendment_path=retry_amendment_path,
            prior_retry_failure_path=prior_failure_path,
            run_id=fallback_acceptance_module.SECOND_RECOVERY_V9_RUN_ID,
            policy=policy,
            activation_certificate=activation,
            primary_result=primary,
            limits=limits,
            source_association=validated_source,
            source_association_path=source_association_path,
            retry_request=retry_request,
            legacy_provenance_bridge=legacy_provenance_bridge(),
            observed=_terminal_v8_gpu_summary(),
            require_authorized=True,
            verify_decoder_compilation=False,
        )
    )
    assert validated_overlay == overlay
    return _ValidatedV9Fixture(
        overlay=overlay,
        v3_result=v3_result,
        v3_incident=v3_incident,
        retry_amendment=json.loads(retry_amendment_path.read_text(encoding="utf-8")),
        prior_failure=json.loads(prior_failure_path.read_text(encoding="utf-8")),
        activation_certificate=activation,
        source_association=validated_source,
        validation_inputs=fallback_acceptance_module._SecondRecoveryValidationInputs(
            primary_result_path=primary_path,
            activation_certificate_path=activation_path,
            overlay_path=overlay_path,
            v3_result_path=v3_result_path,
            v3_incident_path=v3_incident_path,
            prior_control_plane_incident_path=incident_paths[0],
            prior_v5_control_plane_incident_path=incident_paths[1],
            prior_v6_control_plane_incident_path=incident_paths[2],
            prior_v7_runtime_incident_path=incident_paths[3],
            prior_v8_runtime_incident_path=incident_paths[4],
            prior_v8_lease_repair_receipt_path=lease_repair_path,
            prior_retry_amendment_path=retry_amendment_path,
            prior_retry_failure_path=prior_failure_path,
            source_association_path=source_association_path,
        ),
        primary_result=primary,
    )


def _authorized_v9_runner(
    *,
    tmp_path: Path,
    ledger: Ledger,
    service: FakeFallbackService,
    preconstruction_mutation: str | None = None,
) -> FallbackAcceptanceRunner:
    fixture = _build_validated_v9_fixture(tmp_path)
    _seed_terminal_v8_gpu_accounting(ledger)
    supplied_overlay = dict(fixture.overlay)
    supplied_source_association = dict(fixture.source_association)
    if preconstruction_mutation == "overlay":
        supplied_overlay["preconstruction_mutation"] = True
    elif preconstruction_mutation == "source_association":
        supplied_source_association["preconstruction_mutation"] = True
    elif preconstruction_mutation is not None:
        raise ValueError("unknown V9 fixture mutation")
    amendment_hash = SECOND_RECOVERY_V3_RETRY_AMENDMENT_SHA256
    overlay_hash = cast(str, fixture.overlay["manifest_sha256"])
    tokenizer_manifest = v3_fallback_tokenizer_manifest()
    adopter = create_production_development_adopter(
        root=ROOT,
        service=cast(object, service),
        artifacts=ArtifactStore(BlobStore(tmp_path / "blobs"), ledger),
        tokenizer=FakeTokenizer(),
        tokenizer_manifest=tokenizer_manifest,
        launcher_configuration_hash=service.configuration.configuration_hash,
        model_snapshot_manifest_hash="c" * 64,
        source_association=supplied_source_association,
        checkpoint_path=tmp_path / "fallback.checkpoint.json",
        assessment_factory=lambda **_kwargs: (_ for _ in ()).throw(
            AssertionError("development assessment cannot run before service start")
        ),
        retry_amendment_sha256=amendment_hash,
        second_recovery_overlay_sha256=overlay_hash,
        recovery_service_start_event_ids=(
            fallback_acceptance_module._second_recovery_service_start_event_ids(
                fallback_acceptance_module.SECOND_RECOVERY_V9_RUN_ID
            )
        ),
    )
    return FallbackAcceptanceRunner(
        root=ROOT,
        legacy_provenance_bridge=legacy_provenance_bridge(),
        run_id=fallback_acceptance_module.SECOND_RECOVERY_V9_RUN_ID,
        service=cast(object, service),
        ledger=ledger,
        artifacts=ArtifactStore(BlobStore(tmp_path / "blobs"), ledger),
        resource_sampler=cast(object, FakeResourceSampler()),
        tokenizer=FakeTokenizer(),
        tokenizer_manifest=tokenizer_manifest,
        checkpoint_path=tmp_path / "fallback.checkpoint.json",
        activation_certificate=fixture.activation_certificate,
        replacement_receipt={"manifest_sha256": HASH_B},
        snapshot_manifest={
            "repository": FALLBACK_MODEL_REPOSITORY,
            "revision": FALLBACK_MODEL_REVISION,
            "manifest_sha256": "c" * 64,
        },
        source_association=supplied_source_association,
        pre_fallback_gpu_accounting=pre_fallback_gpu_accounting_baseline(
            fixture.primary_result
        ),
        retry_amendment=fixture.retry_amendment,
        prior_fallback_failure=fixture.prior_failure,
        second_recovery_overlay=supplied_overlay,
        second_recovery_v3_result=fixture.v3_result,
        second_recovery_v3_incident=fixture.v3_incident,
        second_recovery_validation_inputs=fixture.validation_inputs,
        service_start_watchdog_seconds=AMENDED_FALLBACK_STARTUP_WATCHDOG_SECONDS,
        development_adopter=adopter,
    )


def test_direct_v9_runner_rejects_overlay_without_full_validator_inputs(
    tmp_path: Path,
) -> None:
    configuration = _fallback_launch_configuration(tmp_path)
    with Ledger(tmp_path / "ledger.sqlite3") as ledger:
        service = FakeFallbackService(
            configuration,
            ledger,
            _fallback_outputs(trigger_repair=False),
            {},
        )
        with pytest.raises(ValueError, match="full-validator inputs"):
            FallbackAcceptanceRunner(
                root=ROOT,
                legacy_provenance_bridge=legacy_provenance_bridge(),
                run_id=fallback_acceptance_module.SECOND_RECOVERY_V9_RUN_ID,
                service=cast(object, service),
                ledger=ledger,
                artifacts=ArtifactStore(BlobStore(tmp_path / "blobs"), ledger),
                resource_sampler=cast(object, FakeResourceSampler()),
                tokenizer=FakeTokenizer(),
                tokenizer_manifest=v3_fallback_tokenizer_manifest(),
                checkpoint_path=tmp_path / "fallback.checkpoint.json",
                activation_certificate={"manifest_sha256": HASH_A},
                replacement_receipt={"manifest_sha256": HASH_B},
                snapshot_manifest={"manifest_sha256": "c" * 64},
                source_association={"manifest_sha256": "d" * 64},
                retry_amendment={
                    "manifest_sha256": SECOND_RECOVERY_V3_RETRY_AMENDMENT_SHA256,
                    "authorized_recovery_run_id": SECOND_RECOVERY_V3_RUN_ID,
                },
                prior_fallback_failure={"manifest_sha256": "e" * 64},
                second_recovery_overlay={"manifest_sha256": "f" * 64},
                second_recovery_v3_result={"manifest_sha256": HASH_A},
                second_recovery_v3_incident={"manifest_sha256": HASH_B},
                service_start_watchdog_seconds=(
                    AMENDED_FALLBACK_STARTUP_WATCHDOG_SECONDS
                ),
            )
        assert service.start_count == 0
        assert not (tmp_path / "fallback.checkpoint.json").exists()


@pytest.mark.parametrize("mutated_payload", ("overlay", "source_association"))
def test_v9_runner_rejects_mutated_payload_with_genuine_validation_inputs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mutated_payload: str,
) -> None:
    monkeypatch.setattr(
        fallback_acceptance_module,
        "_verify_second_recovery_decoder_compiles",
        lambda _schema: None,
    )
    configuration = _fallback_launch_configuration(tmp_path)
    with Ledger(tmp_path / "ledger.sqlite3") as ledger:
        service = FakeFallbackService(
            configuration,
            ledger,
            _fallback_outputs(trigger_repair=False),
            {},
        )
        with pytest.raises(ValueError, match="changed after full validation"):
            _authorized_v9_runner(
                tmp_path=tmp_path,
                ledger=ledger,
                service=service,
                preconstruction_mutation=mutated_payload,
            )
        assert service.start_count == 0
        assert not (tmp_path / "fallback.checkpoint.json").exists()


@pytest.mark.parametrize("mutated_payload", ("overlay", "source_association"))
def test_v9_runner_revalidates_mutable_payloads_before_service_start(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mutated_payload: str,
) -> None:
    monkeypatch.setattr(
        fallback_acceptance_module,
        "_verify_second_recovery_decoder_compiles",
        lambda _schema: None,
    )
    configuration = _fallback_launch_configuration(tmp_path)
    with Ledger(tmp_path / "ledger.sqlite3") as ledger:
        service = FakeFallbackService(
            configuration,
            ledger,
            _fallback_outputs(trigger_repair=False),
            {},
        )
        runner = _authorized_v9_runner(
            tmp_path=tmp_path,
            ledger=ledger,
            service=service,
        )
        target = cast(
            dict[str, object],
            (
                runner.second_recovery_overlay
                if mutated_payload == "overlay"
                else runner.source_association
            ),
        )
        target["post_validation_mutation"] = True

        with pytest.raises(ValueError, match="changed after full validation"):
            runner.prepare_controller_restart()

        assert service.start_count == 0
        assert not runner.checkpoint_path.exists()


def test_v8_is_terminal_and_v9_prepare_uses_exact_service_only_contingency(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """V8 cannot resume; explicitly authorized V9 alone reaches the start boundary."""

    monkeypatch.setattr(
        fallback_acceptance_module,
        "_verify_second_recovery_decoder_compiles",
        lambda _schema: None,
    )
    configuration = _fallback_launch_configuration(tmp_path)
    live: dict[str, object] = {}
    with Ledger(tmp_path / "ledger.sqlite3") as ledger:
        service = FakeFallbackService(
            configuration,
            ledger,
            _fallback_outputs(trigger_repair=False),
            live,
        )
        reached_start = RuntimeError("mocked-service-start-boundary")
        captured_start: dict[str, object] = {}

        def forbid_gpu_start(**kwargs: object) -> None:
            captured_start.update(kwargs)
            service.start_count += 1
            raise reached_start

        monkeypatch.setattr(service, "start", forbid_gpu_start)
        with pytest.raises(ValueError, match="v4/v5/v6/v7/v8 are terminal"):
            FallbackAcceptanceRunner(
                root=ROOT,
                legacy_provenance_bridge=legacy_provenance_bridge(),
                run_id=fallback_acceptance_module.SECOND_RECOVERY_V8_RUN_ID,
                service=cast(object, service),
                ledger=ledger,
                artifacts=ArtifactStore(BlobStore(tmp_path / "v8-blobs"), ledger),
                resource_sampler=cast(object, FakeResourceSampler()),
                tokenizer=FakeTokenizer(),
                tokenizer_manifest=fallback_tokenizer_manifest(),
                checkpoint_path=tmp_path / "v8.checkpoint.json",
                activation_certificate={"manifest_sha256": HASH_A},
                replacement_receipt={"manifest_sha256": HASH_B},
                snapshot_manifest={"manifest_sha256": "c" * 64},
                source_association={"manifest_sha256": "d" * 64},
            )
        runner = _authorized_v9_runner(
            tmp_path=tmp_path,
            ledger=ledger,
            service=service,
        )
        monkeypatch.setattr(
            FallbackAcceptanceRunner,
            "_validate_second_recovery_request_binding",
            lambda _self: None,
        )
        monkeypatch.setattr(
            FallbackAcceptanceRunner,
            "_second_recovery_retry_lineage",
            lambda _self: None,
        )
        monkeypatch.setattr(
            fallback_acceptance_module,
            "_resource_gate",
            lambda _ledger, _limits: {"accepted": True},
        )

        effective_inventory = runner._effective_inventory_manifest()
        assert runner._base_inventory_consumed_service_starts() == (3, 3)
        receipts = fallback_acceptance_module._reserve_receipts(ledger, ())
        assert [receipt["reservation_id"] for receipt in receipts] == [
            "fallback-qwen3-8b-awq-development-v3:fallback-c1-01"
        ]
        assert runner._remaining_mandatory_forecast_seconds(
            runner._initial_state("a" * 64)
        ) == 29_459.0
        assert effective_inventory["recovery_service_start_events"] == 4
        assert effective_inventory["effective_accounting_events"] == 290
        assert effective_inventory["effective_inference_attempts"] == 278
        execution_identity = runner._execution_identity("a" * 64)
        assert execution_identity["prior_v8_runtime_incident_sha256"] == (
            fallback_acceptance_module.SECOND_RECOVERY_V8_INCIDENT_MANIFEST_SHA256
        )
        assert execution_identity["prior_v8_lease_repair_receipt_sha256"] == (
            fallback_acceptance_module.SECOND_RECOVERY_V8_LEASE_REPAIR_MANIFEST_SHA256
        )

        with pytest.raises(RuntimeError, match="mocked-service-start-boundary") as exc:
            runner.prepare_controller_restart()

    assert exc.value is reached_start
    assert service.start_count == 1
    assert captured_start == {
        "session_id": fallback_acceptance_module.SECOND_RECOVERY_V9_RUN_ID,
        "event_id": "fallback-qwen3-8b-awq-development-v9-service-start-001",
        "watchdog_seconds": 300,
        "remaining_required_seconds": 29_459.0,
        "admission_forecast_seconds": 391.40054529582005,
        "contingency_unlocked": True,
        "essential_recovery": True,
    }


def test_v9_forecast_drift_fails_before_attempt_checkpoint_or_service_start(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        fallback_acceptance_module,
        "_verify_second_recovery_decoder_compiles",
        lambda _schema: None,
    )
    configuration = _fallback_launch_configuration(tmp_path)
    live: dict[str, object] = {}
    with Ledger(tmp_path / "ledger.sqlite3") as ledger:
        service = FakeFallbackService(
            configuration,
            ledger,
            _fallback_outputs(trigger_repair=False),
            live,
        )
        runner = _authorized_v9_runner(
            tmp_path=tmp_path,
            ledger=ledger,
            service=service,
        )
        monkeypatch.setattr(
            FallbackAcceptanceRunner,
            "_remaining_mandatory_forecast_seconds",
            lambda _self, _state, **_kwargs: 29_458.0,
        )
        monkeypatch.setattr(
            FallbackAcceptanceRunner,
            "_validate_second_recovery_request_binding",
            lambda _self: None,
        )
        monkeypatch.setattr(
            FallbackAcceptanceRunner,
            "_second_recovery_retry_lineage",
            lambda _self: None,
        )
        monkeypatch.setattr(
            fallback_acceptance_module,
            "_resource_gate",
            lambda _ledger, _limits: {"accepted": True},
        )

        with pytest.raises(RuntimeError, match="contingency binding changed"):
            runner.prepare_controller_restart()

        checkpoint = json.loads(runner.checkpoint_path.read_text(encoding="utf-8"))
        assert checkpoint["service_start_attempted"] is False
        assert checkpoint["failed_call_id"] is None
        assert service.start_count == 0
        assert service.shutdown_count == 0


def test_v9_post_adoption_schedule_slip_stops_before_scientific_side_effects(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        fallback_acceptance_module,
        "_verify_second_recovery_decoder_compiles",
        lambda _schema: None,
    )
    configuration = _fallback_launch_configuration(tmp_path)
    live: dict[str, object] = {}
    with Ledger(tmp_path / "ledger.sqlite3") as ledger:
        service = FakeFallbackService(
            configuration,
            ledger,
            _fallback_outputs(trigger_repair=False),
            live,
        )
        runner = _authorized_v9_runner(
            tmp_path=tmp_path,
            ledger=ledger,
            service=service,
        )
        monkeypatch.setattr(
            FallbackAcceptanceRunner,
            "_validate_second_recovery_request_binding",
            lambda _self: None,
        )
        monkeypatch.setattr(
            FallbackAcceptanceRunner,
            "_second_recovery_retry_lineage",
            lambda _self: None,
        )
        monkeypatch.setattr(
            fallback_acceptance_module,
            "_resource_gate",
            lambda _ledger, _limits: {"accepted": True},
        )
        controller_pid = 41_001
        monkeypatch.setattr(os, "getpid", lambda: controller_pid)
        runner.prepare_controller_restart()
        assert runner._base_inventory_consumed_service_starts() == (3, 4)
        state_after_start = json.loads(
            runner.checkpoint_path.read_text(encoding="utf-8")
        )
        assert runner._remaining_mandatory_forecast_seconds(state_after_start) == 29_459.0
        assert runner._effective_inventory_manifest()["effective_accounting_events"] == 290

        generation_reached = False

        def forbidden_generation(*_args: object, **_kwargs: object) -> None:
            nonlocal generation_reached
            generation_reached = True
            raise AssertionError("V9 schedule rejection reached model generation")

        monkeypatch.setattr(service, "run_fallback_test", forbidden_generation)
        service.actual_allocated_service_seconds = 3_000.0
        controller_pid = 41_002
        with pytest.raises(RuntimeError, match="fresh scheduled admission"):
            runner.run()

        checkpoint = json.loads(runner.checkpoint_path.read_text(encoding="utf-8"))
        assert checkpoint["failed_call_id"] == "v9-post-adoption-scheduled-admission"
        assert checkpoint["reserve_consumption"] == []
        assert checkpoint["completed_call_ids"] == []
        assert generation_reached is False
        assert service.shutdown_count == 1
        with sqlite3.connect(ledger.path) as read_only:
            counts = {
                table: read_only.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
                for table in (
                    "jobs",
                    "query_access_events",
                    "attempts",
                    "model_calls",
                    "failures",
                )
            }
        assert counts == {
            "jobs": 0,
            "query_access_events": 0,
            "attempts": 0,
            "model_calls": 0,
            "failures": 0,
        }


def test_v9_recovered_prepare_schedule_slip_stops_before_rehandoff_or_science(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        fallback_acceptance_module,
        "_verify_second_recovery_decoder_compiles",
        lambda _schema: None,
    )
    configuration = _fallback_launch_configuration(tmp_path)
    live: dict[str, object] = {}
    with Ledger(tmp_path / "ledger.sqlite3") as ledger:
        service = FakeFallbackService(
            configuration,
            ledger,
            _fallback_outputs(trigger_repair=False),
            live,
        )
        runner = _authorized_v9_runner(
            tmp_path=tmp_path,
            ledger=ledger,
            service=service,
        )
        plan_hash = cast(str, fallback_plan_manifest(ROOT)["manifest_sha256"])
        execution_hash = canonical_sha256(runner._execution_identity(plan_hash))
        state = runner._initial_state(execution_hash)
        state["service_start_attempted"] = True
        state["stage_one_controller_pid"] = 42_001
        runner._save(state)

        monkeypatch.setattr(os, "getpid", lambda: 42_001)
        service.start(
            session_id=fallback_acceptance_module.SECOND_RECOVERY_V9_RUN_ID,
            event_id="fallback-qwen3-8b-awq-development-v9-service-start-001",
            watchdog_seconds=300,
            remaining_required_seconds=29_459.0,
            admission_forecast_seconds=391.40054529582005,
            contingency_unlocked=True,
            essential_recovery=True,
        )
        service.detach_for_controller_restart(runner._service_checkpoint_path)
        service.actual_allocated_service_seconds = 3_000.0
        reserve_before = fallback_acceptance_module._reserve_receipts(ledger, ())

        def forbidden_rehandoff(_path: Path) -> None:
            raise AssertionError("unadmitted recovered service was re-detached")

        monkeypatch.setattr(
            service,
            "detach_for_controller_restart",
            forbidden_rehandoff,
        )
        monkeypatch.setattr(os, "getpid", lambda: 42_002)

        with pytest.raises(RuntimeError, match="fresh scheduled admission"):
            runner.recover_controller_restart_preparation()

        checkpoint = json.loads(runner.checkpoint_path.read_text(encoding="utf-8"))
        assert checkpoint["failed_call_id"] == (
            "v9-recovered-prepare-scheduled-admission"
        )
        assert checkpoint["controller_handoff_complete"] is False
        assert checkpoint["reserve_consumption"] == []
        assert service.start_count == 1
        assert service.resume_count == 1
        assert service.shutdown_count == 1
        assert service.state is ServiceState.STOPPED
        assert live["running"] is False
        assert fallback_acceptance_module._reserve_receipts(ledger, ()) == (
            reserve_before
        )
        with sqlite3.connect(ledger.path) as read_only:
            counts = {
                table: read_only.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
                for table in (
                    "jobs",
                    "query_access_events",
                    "attempts",
                    "model_calls",
                    "failures",
                )
            }
        assert counts == {
            "jobs": 0,
            "query_access_events": 0,
            "attempts": 0,
            "model_calls": 0,
            "failures": 0,
        }


def test_second_recovery_service_identity_derivation_preserves_terminal_lineage() -> None:
    assert fallback_acceptance_module._second_recovery_service_start_event_ids(
        fallback_acceptance_module.SECOND_RECOVERY_V7_RUN_ID
    ) == (
        "fallback-qwen3-8b-awq-development-v3-service-start-001",
        "fallback-qwen3-8b-awq-development-v7-service-start-001",
    )
    assert fallback_acceptance_module._second_recovery_service_start_event_ids(
        fallback_acceptance_module.SECOND_RECOVERY_V8_RUN_ID
    ) == (
        "fallback-qwen3-8b-awq-development-v3-service-start-001",
        "fallback-qwen3-8b-awq-development-v7-service-start-001",
        "fallback-qwen3-8b-awq-development-v8-service-start-001",
    )
    assert fallback_acceptance_module._second_recovery_service_start_event_ids(
        fallback_acceptance_module.SECOND_RECOVERY_V9_RUN_ID
    ) == (
        "fallback-qwen3-8b-awq-development-v3-service-start-001",
        "fallback-qwen3-8b-awq-development-v7-service-start-001",
        "fallback-qwen3-8b-awq-development-v8-service-start-001",
        "fallback-qwen3-8b-awq-development-v9-service-start-001",
    )
    with pytest.raises(ValueError, match="restricted to v7/v8/v9"):
        fallback_acceptance_module._second_recovery_service_start_event_ids(
            fallback_acceptance_module.SECOND_RECOVERY_V6_RUN_ID
        )


def _fallback_outputs(*, trigger_repair: bool) -> dict[str, Mapping[str, object]]:
    accepted_c1 = with_legacy_source_hashes(
        json.loads((ROOT / "tests/fixtures/phase1/c1_pre_output.json").read_text())
    )
    first_c1 = copy.deepcopy(accepted_c1)
    if trigger_repair:
        first_c1["decisions"] = [
            decision
            for decision in cast(list[dict[str, object]], first_c1["decisions"])
            if decision["operator"] != "rare_preservation"
        ]
    return {
        "fallback-c1-01": first_c1,
        "fallback-c1-01-repair-01": accepted_c1,
        "fallback-c2-01": with_legacy_source_hashes(
            json.loads((ROOT / "tests/fixtures/phase1/c2_query_output.json").read_text())
        ),
        "fallback-c2-02": with_legacy_source_hashes(
            json.loads((ROOT / "tests/fixtures/phase1/c2_query_output_2.json").read_text())
        ),
        "fallback-fixed-01": json.loads(
            (ROOT / "tests/fixtures/phase1/fixed_select_output.json").read_text()
        ),
    }


@pytest.mark.parametrize("trigger_repair", [False, True])
def test_two_controller_fallback_runner_is_bounded_resumable_and_audited(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    trigger_repair: bool,
) -> None:
    configuration = _fallback_launch_configuration(tmp_path)
    live: dict[str, object] = {}
    outputs = _fallback_outputs(trigger_repair=trigger_repair)
    with Ledger(tmp_path / "ledger.sqlite3") as ledger:
        first_service = FakeFallbackService(configuration, ledger, outputs, live)
        first_runner = _runner(
            tmp_path=tmp_path,
            ledger=ledger,
            service=first_service,
        )
        monkeypatch.setattr(os, "getpid", lambda: 41001)
        handoff = first_runner.prepare_controller_restart()
        assert handoff["next_required_stage"] == "run_under_a_different_controller_pid"
        assert live["running"] is True
        assert first_service.start_count == 1
        assert first_service.remaining_required_seconds == [30_059]

        second_service = FakeFallbackService(configuration, ledger, outputs, live)
        second_runner = _runner(
            tmp_path=tmp_path,
            ledger=ledger,
            service=second_service,
        )
        monkeypatch.setattr(os, "getpid", lambda: 41002)
        result = second_runner.run()

        assert result["micro_pilot_passed"] is True
        assert result["phase1_gate_passed"] is True
        assert result["gate_passed"] is True
        assert result["development_handoff_pending"] is False
        assert result["development_continuation_integration_pending"] is False
        assert result["physical_service_live"] is False
        assert result["vllm_service_stopped"] is True
        assert result["completed_base_call_count"] == 4
        assert result["repair_attempt_count"] == int(trigger_repair)
        assert result["controller_resume_gate"]["controller_process_restart"] is True
        assert result["controller_resume_gate"]["model_process_restart"] is False
        assert result["operator_coverage_gate"]["c1_pilot_complete"] is True
        assert result["operator_coverage_gate"]["c1_global_complete_in_micro_pilot"] is False
        assert result["operator_coverage_gate"]["c1_global_completion_gate"] == (
            "integrated_development_scientific_assessment.c1_all_construction_operators_exercised"
        )
        assert result["operator_coverage_gate"]["c2_complete"] is True
        assert result["actual_plus_remaining_forecast"]["admitted"] is True
        assert result["normal_acceptance_block_executed"] is False
        assert result["selected_model_freeze"]["scope"] == ("registered_development_llm_conditions")
        assert result["selected_model_freeze"]["held_out_execution_allowed"] is False
        assert result["development_continuation_gate"]["passed"] is True
        assert result["development_continuation_receipt"]["terminal_call_count"] == 24
        storage_admission = result["development_storage_admission"]
        assert storage_admission["phase"] == "phase_3"
        assert storage_admission["allowed"] is True
        assert storage_admission["declared_growth_bytes"] == 2_500_000_000
        assert storage_admission["largest_atomic_temporary_bytes"] == 268_435_456
        assert storage_admission["quarantine_allowance_bytes"] == 268_435_456
        assert storage_admission["release_staging_bytes"] == 268_435_456
        assert result["development_continuation_gate"]["phase3_storage_admitted"] is True
        assert (
            result["development_continuation_bootstrap"][
                "development_storage_admission_hash"
            ]
            == storage_admission["content_hash"]
            == result["development_handoff"]["development_storage_admission_hash"]
            == result["development_continuation_receipt"][
                "development_storage_admission_hash"
            ]
        )
        storage_rows = ledger.storage_samples_with_phase_prefix(
            "phase3_development:fallback-unit"
        )
        assert len(storage_rows) == 1
        assert storage_rows[0].sample_id == storage_admission["storage_sample_id"]
        assert storage_rows[0].additional_reserved_bytes == 3_305_306_368
        assert (
            result["development_continuation_bootstrap"]["selected_model_freeze_hash"]
            == result["selected_model_freeze"]["manifest_sha256"]
        )
        assert result["development_preparation"]["query_access_event_count"] == 0
        assert (
            result["development_handoff"]["live_service_identity"]["service_pid"]
            == first_service.pid
        )
        assert "checkpoint_path" not in result["development_preparation"]
        assert "development_checkpoint_path" not in result["development_handoff"]
        assert result["development_preparation"]["restricted_fields_withheld"] == [
            "checkpoint_path"
        ]
        scan_public_bytes(
            json.dumps(result, sort_keys=True).encode(),
            relative_path="fallback-result.json",
        )
        adopter = cast(FakeDevelopmentAdopter, second_runner.development_adopter)
        assert adopter.adapter is not None
        assert not hasattr(adopter.adapter, "start")
        assert not hasattr(adopter.adapter, "shutdown")
        assert second_service.resume_count == 1
        assert first_service.pid == second_service.pid
        assert (
            sum(event.event_kind is GpuEventKind.GPU_SESSION_START for event in ledger.gpu_events())
            == 1
        )
        assert (
            sum(event.event_kind is GpuEventKind.FALLBACK_TEST for event in ledger.gpu_events())
            == 4
        )
        assert sum(event.event_kind is GpuEventKind.REPAIR for event in ledger.gpu_events()) == int(
            trigger_repair
        )
        assert len(result["reserve_consumption"]) == 4 + int(trigger_repair)
        assert len(second_service.remaining_required_seconds) == 4 + int(trigger_repair)
        assert min(second_service.remaining_required_seconds) > 29_000
        assert live["running"] is False

        specs = fallback_pilot_calls(
            FallbackModelPolicy.load(ROOT / "configs/study/fallback_model.json")
        )
        expected_roles = {
            ConditionName.C1_LLM_PRE: "prebuild",
            ConditionName.C2_LLM_QUERY: "query_time",
            ConditionName.A_FIXED_SELECT: "fixed_select",
        }
        for spec in specs:
            model_call = ledger.get_model_call(f"fallback-unit-{spec.call_id}")
            transitions = ledger.transitions(model_call.job_id)
            expected_states = [JobState.PLANNED, JobState.PREQUERY_SEALED]
            if spec.condition is not ConditionName.C1_LLM_PRE:
                expected_states.append(JobState.QUERY_REVEALED)
                assert transitions[2].occurred_at > transitions[1].occurred_at
            expected_states.append(JobState.GENERATED)
            if trigger_repair and spec.call_id == "fallback-c1-01":
                expected_states.append(JobState.REPAIRED)
            expected_states.extend((JobState.VALIDATED, JobState.FINALIZED))
            assert [item.to_state for item in transitions] == expected_states
            assert model_call.call_role.value == expected_roles[spec.condition]
            validation = ledger.get_validation(
                f"fallback-unit-{spec.call_id}-attempt-validation"
            )
            assert validation.input_artifact_hash == model_call.response_artifact_hash
            assert validation.semantic_assessment_scope is (
                SemanticAssessmentScope.RUNTIME_STRUCTURAL_ONLY_NOT_ASSESSED
            )
            assert validation.validation_status.value == (
                "rejected"
                if trigger_repair and spec.call_id == "fallback-c1-01"
                else "accepted"
            )
        assert ledger.count_rows("projections") == 0
        assert ledger.count_rows("validations") == 4 + int(trigger_repair)

        state = json.loads((tmp_path / "fallback.checkpoint.json").read_text())
        handoff_count = len(adopter.handoffs)
        replayed_development = second_runner._run_development_continuation(
            provisional_result=cast(Mapping[str, object], state["accepted_micro_pilot_result"]),
            state=state,
            execution_hash=cast(str, state["execution_hash"]),
            registration=adopter.registration(),
        )
        assert replayed_development.content_hash == result["development_continuation_receipt"][
            "content_hash"
        ]
        assert len(adopter.handoffs) == handoff_count
        assert len(
            ledger.storage_samples_with_phase_prefix("phase3_development:fallback-unit")
        ) == 1
        before_replay_counts = {
            table: ledger.count_rows(table)
            for table in (
                "artifacts",
                "attempts",
                "failures",
                "job_transitions",
                "model_calls",
                "storage_samples",
                "validations",
            )
        }
        resumed, _, timings = second_runner._resume_completed(
            state=state,
            call=specs[0],
        )
        replayed, _, _ = second_runner._resume_completed(state=state, call=specs[0])
        assert resumed["status"] == "resumed"
        assert replayed == resumed
        assert len(timings) == 1 + int(trigger_repair)
        assert before_replay_counts == {
            table: ledger.count_rows(table) for table in before_replay_counts
        }
        if trigger_repair:
            assert resumed["base_attempt"]["packing_report"]["complete_evidence_snapshot"] is True

    verification = verify_ledger(tmp_path / "ledger.sqlite3", tmp_path / "blobs")
    assert verification.valid, verification.issues


def test_fallback_runner_shutdown_remains_blocked_after_periodic_drain_refusal(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    configuration = _fallback_launch_configuration(tmp_path)
    live: dict[str, object] = {}
    outputs = _fallback_outputs(trigger_repair=False)
    with Ledger(tmp_path / "ledger.sqlite3") as ledger:
        first_service = FakeFallbackService(configuration, ledger, outputs, live)
        first_runner = _runner(
            tmp_path=tmp_path,
            ledger=ledger,
            service=first_service,
        )
        monkeypatch.setattr(os, "getpid", lambda: 42_001)
        first_runner.prepare_controller_restart()

        second_service = FakeFallbackService(
            configuration,
            ledger,
            outputs,
            live,
            periodic_drain_allowed=False,
        )
        second_runner = _runner(
            tmp_path=tmp_path,
            ledger=ledger,
            service=second_service,
        )
        monkeypatch.setattr(os, "getpid", lambda: 42_002)

        with pytest.raises(RuntimeError, match="periodic sample remains owned"):
            second_runner.run()

        assert live["running"] is True
        assert second_service.shutdown_count == 0
        watchdog = second_service.periodic_resource_watchdog
        assert watchdog is not None
        second_service.periodic_drain_allowed = True
        assert second_service.stop_periodic_resource_watchdog(
            watchdog,
            raise_failure=False,
        )
        second_service.shutdown()
        assert live["running"] is False


def test_development_continuation_records_rejected_phase3_storage_before_adopter(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    configuration = _fallback_launch_configuration(tmp_path)
    live: dict[str, object] = {}
    outputs = _fallback_outputs(trigger_repair=False)
    with Ledger(tmp_path / "ledger.sqlite3") as ledger:
        first_service = FakeFallbackService(configuration, ledger, outputs, live)
        first_runner = _runner(tmp_path=tmp_path, ledger=ledger, service=first_service)
        monkeypatch.setattr(os, "getpid", lambda: 41501)
        first_runner.prepare_controller_restart()

        second_service = FakeFallbackService(configuration, ledger, outputs, live)
        second_runner = _runner(tmp_path=tmp_path, ledger=ledger, service=second_service)
        second_runner.resource_sampler = cast(
            object,
            FakeResourceSampler(
                storage=StoragePreflight(
                    tmp_path,
                    budget=StorageBudget(
                        total_allocation_bytes=30_000_000_000,
                        max_occupied_bytes=50,
                        min_headroom_bytes=5_000_000_000,
                    ),
                )
            ),
        )
        monkeypatch.setattr(os, "getpid", lambda: 41502)

        with pytest.raises(StorageBudgetExceeded):
            second_runner.run()

        adopter = cast(FakeDevelopmentAdopter, second_runner.development_adopter)
        assert adopter.prepared_value is None
        assert adopter.handoffs == []
        assert live["running"] is False
        rows = ledger.storage_samples_with_phase_prefix(
            "phase3_development:fallback-unit"
        )
        assert len(rows) == 1
        assert rows[0].allowed is False
        state = json.loads((tmp_path / "fallback.checkpoint.json").read_text())
        admission = state["development_storage_admissions"][-1]
        assert admission["storage_sample_id"] == rows[0].sample_id
        assert admission["allowed"] is False
        assert admission["sampled_total_allocation_bytes"] == 30_000_000_000
        assert admission["sampled_max_occupied_bytes"] == 50
        assert admission["sampled_min_headroom_bytes"] == 5_000_000_000
        assert admission["projected_allocation_free_bytes"] == (
            admission["sampled_total_allocation_bytes"]
            - admission["projected_occupied_bytes"]
        )
        assert state["development_continuation_ready"] is False


def test_development_storage_admission_recovers_ledger_first_crash_and_rechecks(
    tmp_path: Path,
) -> None:
    configuration = _fallback_launch_configuration(tmp_path)
    live: dict[str, object] = {}
    outputs = _fallback_outputs(trigger_repair=False)
    with Ledger(tmp_path / "ledger.sqlite3") as ledger:
        service = FakeFallbackService(configuration, ledger, outputs, live)
        runner = _runner(tmp_path=tmp_path, ledger=ledger, service=service)
        state = runner._initial_state(HASH_A)
        runner._save(state)

        reservation = StorageAllocationPlan.load(
            ROOT / "configs/study/storage_phase_allocations.json"
        ).reservation_for("phase_3")
        assert runner.resource_sampler.storage is not None
        orphan_report = runner.resource_sampler.storage.check(
            **reservation.preflight_arguments()
        )
        orphan_sample_id = ledger.record_storage_sample(
            orphan_report,
            phase="phase3_development:fallback-unit",
            sampled_at=datetime.now(UTC) - timedelta(seconds=1),
        )

        recovered = runner._reconcile_development_storage_admissions(state)
        assert len(recovered) == 1
        assert recovered[0].storage_sample_id == orphan_sample_id
        assert recovered[0].sampled_total_allocation_bytes == (
            orphan_report.budget.total_allocation_bytes
        )
        assert recovered[0].sampled_max_occupied_bytes == (
            orphan_report.budget.max_occupied_bytes
        )
        assert recovered[0].sampled_min_headroom_bytes == (
            orphan_report.budget.min_headroom_bytes
        )
        assert state["development_storage_admission_hash"] == recovered[0].content_hash
        checkpoint_after_recovery = runner.checkpoint_path.read_bytes()

        replayed = runner._reconcile_development_storage_admissions(state)
        assert replayed == recovered
        assert runner.checkpoint_path.read_bytes() == checkpoint_after_recovery
        assert len(
            ledger.storage_samples_with_phase_prefix("phase3_development:fallback-unit")
        ) == 1

        fresh, fresh_report = runner._record_development_storage_admission(state)
        assert fresh_report.allowed is True
        assert fresh.storage_sample_id != orphan_sample_id
        assert fresh.sampled_at > recovered[0].sampled_at
        assert state["development_storage_admission_hash"] == fresh.content_hash
        assert [
            item.storage_sample_id
            for item in runner._verify_development_storage_admissions(
                state,
                required=True,
            )
        ] == [orphan_sample_id, fresh.storage_sample_id]


def test_development_storage_admission_rejects_a_looser_runtime_budget(
    tmp_path: Path,
) -> None:
    configuration = _fallback_launch_configuration(tmp_path)
    live: dict[str, object] = {}
    with Ledger(tmp_path / "ledger.sqlite3") as ledger:
        service = FakeFallbackService(configuration, ledger, {}, live)
        runner = _runner(tmp_path=tmp_path, ledger=ledger, service=service)
        runner.resource_sampler = cast(
            object,
            FakeResourceSampler(
                storage=StoragePreflight(
                    tmp_path,
                    budget=StorageBudget(
                        total_allocation_bytes=31_000_000_000,
                        max_occupied_bytes=26_000_000_000,
                        min_headroom_bytes=5_000_000_000,
                    ),
                )
            ),
        )
        state = runner._initial_state(HASH_A)

        with pytest.raises(RuntimeError, match="more permissive than registered limits"):
            runner._record_development_storage_admission(state)


def test_fallback_rejects_missing_development_adopter_before_model_start(
    tmp_path: Path,
) -> None:
    configuration = _fallback_launch_configuration(tmp_path)
    live: dict[str, object] = {}
    with Ledger(tmp_path / "ledger.sqlite3") as ledger:
        service = FakeFallbackService(
            configuration,
            ledger,
            _fallback_outputs(trigger_repair=False),
            live,
        )
        runner = _runner(
            tmp_path=tmp_path,
            ledger=ledger,
            service=service,
            register_development_adopter=False,
        )

        with pytest.raises(RuntimeError, match="registered live development adopter"):
            runner.prepare_controller_restart()

        assert service.start_count == 0
        assert live.get("running") is not True
        assert not (tmp_path / "fallback.checkpoint.json").exists()


@pytest.mark.parametrize("failure_mode", ["raise", "changed_pid"])
def test_development_adopter_failure_always_returns_lifecycle_to_owner_shutdown(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failure_mode: str,
) -> None:
    configuration = _fallback_launch_configuration(tmp_path)
    live: dict[str, object] = {}
    outputs = _fallback_outputs(trigger_repair=False)
    with Ledger(tmp_path / "ledger.sqlite3") as ledger:
        first_service = FakeFallbackService(configuration, ledger, outputs, live)
        first_runner = _runner(tmp_path=tmp_path, ledger=ledger, service=first_service)
        monkeypatch.setattr(os, "getpid", lambda: 52001)
        first_runner.prepare_controller_restart()

        second_service = FakeFallbackService(configuration, ledger, outputs, live)
        second_runner = _runner(tmp_path=tmp_path, ledger=ledger, service=second_service)
        adopter = cast(FakeDevelopmentAdopter, second_runner.development_adopter)
        original = adopter.adopt_and_run

        def fail(
            handoff: DevelopmentContinuationHandoff,
            service: InjectedLiveDevelopmentService,
        ) -> DevelopmentExecutionResult:
            if failure_mode == "raise":
                raise RuntimeError("synthetic development failure")
            result = original(handoff, service)
            adapter = cast(FakeDevelopmentServiceAdapter, adopter.adapter)
            identity_payload = adapter.identity_value.model_dump(
                mode="python",
                exclude={"content_hash"},
            )
            identity_payload["service_pid"] = handoff.service_pid + 1
            adapter.identity_value = LiveServiceIdentity.model_validate(identity_payload)
            return result

        monkeypatch.setattr(adopter, "adopt_and_run", fail)
        monkeypatch.setattr(os, "getpid", lambda: 52002)
        expected = (
            "synthetic development failure"
            if failure_mode == "raise"
            else "changed live-service identity"
        )
        with pytest.raises(RuntimeError, match=expected):
            second_runner.run()

        assert live["running"] is False
        assert second_service.state is ServiceState.STOPPED
        state = json.loads((tmp_path / "fallback.checkpoint.json").read_text())
        assert state["development_continuation_ready"] is True
        assert state["development_continuation_completed"] is False
        assert state["selected_model_freeze"]["held_out_execution_allowed"] is False
        failure = second_runner.failure_result(
            RuntimeError(expected),
            uptime=second_runner._last_shutdown_uptime,
        )
        assert failure["micro_pilot_passed"] is True
        assert failure["failure_stage"] == "development_continuation"
        assert failure["phase1_gate_passed"] is False
        if failure_mode == "changed_pid":
            assert failure["partial_development_checkpoint"]["canonical_checkpoint_valid"] is False


@pytest.mark.parametrize(
    ("failure_mode", "message"),
    [
        ("query_access", "opened a query"),
        ("wrong_freeze", "selected freeze or source tree"),
        ("wrong_identity", "another live model service"),
        ("lifecycle", "forbidden lifecycle members"),
    ],
)
def test_development_preparation_is_query_blind_and_identity_bound(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failure_mode: str,
    message: str,
) -> None:
    configuration = _fallback_launch_configuration(tmp_path)
    live: dict[str, object] = {}
    outputs = _fallback_outputs(trigger_repair=False)
    with Ledger(tmp_path / "ledger.sqlite3") as ledger:
        first_service = FakeFallbackService(configuration, ledger, outputs, live)
        first_runner = _runner(tmp_path=tmp_path, ledger=ledger, service=first_service)
        monkeypatch.setattr(os, "getpid", lambda: 53001)
        first_runner.prepare_controller_restart()

        second_service = FakeFallbackService(configuration, ledger, outputs, live)
        second_runner = _runner(tmp_path=tmp_path, ledger=ledger, service=second_service)
        adopter = cast(FakeDevelopmentAdopter, second_runner.development_adopter)
        original = adopter.prepare

        def invalid_prepare(
            bootstrap: DevelopmentContinuationBootstrap,
        ) -> PreparedDevelopmentContinuation:
            prepared = original(bootstrap)
            if failure_mode == "query_access":
                return PreparedDevelopmentContinuation(
                    call_manifest=prepared.call_manifest,
                    prequery_inputs=prepared.prequery_inputs,
                    forecast_control=prepared.forecast_control,
                    execution_id=prepared.execution_id,
                    execution_manifest_hash=prepared.execution_manifest_hash,
                    checkpoint_path=prepared.checkpoint_path,
                    service_adapter=prepared.service_adapter,
                    query_access_event_count=cast(object, 1),
                )
            if failure_mode == "wrong_freeze":
                payload = prepared.prequery_inputs.model_dump(
                    mode="python",
                    exclude={"content_hash"},
                )
                payload["selected_model_freeze_hash"] = "f" * 64
                return PreparedDevelopmentContinuation(
                    call_manifest=prepared.call_manifest,
                    prequery_inputs=DevelopmentPrequeryInputs.model_validate(payload),
                    forecast_control=prepared.forecast_control,
                    execution_id=prepared.execution_id,
                    execution_manifest_hash=prepared.execution_manifest_hash,
                    checkpoint_path=prepared.checkpoint_path,
                    service_adapter=prepared.service_adapter,
                )
            adapter = cast(FakeDevelopmentServiceAdapter, prepared.service_adapter)
            if failure_mode == "lifecycle":
                adapter.shutdown = lambda: None  # type: ignore[attr-defined]
                return prepared
            identity_payload = adapter.identity_value.model_dump(
                mode="python",
                exclude={"content_hash"},
            )
            identity_payload["service_pid"] = bootstrap.service_pid + 1
            adapter.identity_value = LiveServiceIdentity.model_validate(identity_payload)
            return prepared

        monkeypatch.setattr(adopter, "prepare", invalid_prepare)
        monkeypatch.setattr(os, "getpid", lambda: 53002)
        with pytest.raises(RuntimeError, match=message):
            second_runner.run()

        assert live["running"] is False
        state = json.loads((tmp_path / "fallback.checkpoint.json").read_text())
        assert state["accepted_micro_pilot_result"]["micro_pilot_passed"] is True
        assert state["selected_model_freeze"]["held_out_execution_allowed"] is False
        assert state["development_continuation_completed"] is False


def test_failed_development_gate_is_reported_and_owner_still_shuts_down(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    configuration = _fallback_launch_configuration(tmp_path)
    live: dict[str, object] = {}
    outputs = _fallback_outputs(trigger_repair=False)
    with Ledger(tmp_path / "ledger.sqlite3") as ledger:
        first_service = FakeFallbackService(configuration, ledger, outputs, live)
        first_runner = _runner(tmp_path=tmp_path, ledger=ledger, service=first_service)
        monkeypatch.setattr(os, "getpid", lambda: 54001)
        first_runner.prepare_controller_restart()

        second_service = FakeFallbackService(configuration, ledger, outputs, live)
        second_runner = _runner(tmp_path=tmp_path, ledger=ledger, service=second_service)
        adopter = cast(FakeDevelopmentAdopter, second_runner.development_adopter)
        original = adopter.adopt_and_run

        def failed_gate(
            handoff: DevelopmentContinuationHandoff,
            service: InjectedLiveDevelopmentService,
        ) -> DevelopmentExecutionResult:
            result = original(handoff, service)
            first = result.itt_records[0]
            first_payload = first.model_dump(mode="python", exclude={"content_hash"})
            first_payload.update(
                {
                    "outcome": RunOutcome.FAILED,
                    "failure_code": "synthetic_development_failure",
                }
            )
            rows = (
                DevelopmentITTRecord.model_validate(first_payload),
                *result.itt_records[1:],
            )
            gate_payload = result.gate.model_dump(
                mode="python",
                exclude={"content_hash"},
            )
            gate_payload.update(
                {
                    "every_planned_call_succeeded": False,
                    "passed": False,
                }
            )
            result_payload = result.model_dump(
                mode="python",
                exclude={"content_hash"},
            )
            result_payload.update(
                {
                    "phase": DevelopmentPhase.COMPLETED_WITH_FAILURES,
                    "itt_records": rows,
                    "gate": DevelopmentGateResult.model_validate(gate_payload),
                }
            )
            return DevelopmentExecutionResult.model_validate(result_payload)

        monkeypatch.setattr(adopter, "adopt_and_run", failed_gate)
        monkeypatch.setattr(os, "getpid", lambda: 54002)
        result = second_runner.run()

        assert result["micro_pilot_passed"] is True
        assert result["phase1_gate_passed"] is False
        assert result["development_continuation_gate"]["completed"] is True
        assert result["development_continuation_gate"]["development_gate_passed"] is False
        assert result["vllm_service_stopped"] is True
        assert live["running"] is False


def test_same_controller_cannot_consume_calls_and_cleanup_stops_orphan(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    configuration = _fallback_launch_configuration(tmp_path)
    live: dict[str, object] = {}
    outputs = _fallback_outputs(trigger_repair=False)
    with Ledger(tmp_path / "ledger.sqlite3") as ledger:
        service = FakeFallbackService(configuration, ledger, outputs, live)
        runner = _runner(tmp_path=tmp_path, ledger=ledger, service=service)
        monkeypatch.setattr(os, "getpid", lambda: 51001)
        runner.prepare_controller_restart()
        with pytest.raises(RuntimeError, match="different controller"):
            runner.run()
        cleanup = runner.cleanup_orphan()
        assert cleanup["physical_shutdown_verified"] is True
        assert cleanup["inference_executed"] is False
        assert live["running"] is False


def test_terminal_accounting_uses_atomic_lease_when_lock_mirror_is_torn(
    tmp_path: Path,
) -> None:
    configuration = _fallback_launch_configuration(tmp_path)
    started_at = datetime(2026, 9, 3, tzinfo=UTC)
    event_id = "fallback-unit-service-start-001"
    lock_path = configuration.shared_cache / ".story-projection-onto-vllm.lock"
    with Ledger(tmp_path / "ledger.sqlite3") as ledger:
        ledger.record_gpu_event(
            event_id=event_id,
            event_kind=GpuEventKind.FAILURE,
            allocated_seconds=3,
            started_at=started_at,
            ended_at=started_at + timedelta(seconds=3),
            succeeded=False,
            details={
                "recovered_from_open_journal": True,
                "intended_event_kind": GpuEventKind.GPU_SESSION_START.value,
            },
        )
        service = VLLMService(
            configuration=configuration,
            client=cast(object, SimpleNamespace()),
            meter=AllocatedGPUMeter(
                ledger,
                monotonic_clock=lambda: 0.0,
                wall_clock=lambda: started_at,
            ),
            monotonic_clock=lambda: 0.0,
            wall_clock=lambda: started_at,
        )
        service._acquire_service_lock()
        service._session_id = "fallback-unit"
        service._accounting_session_id = event_id
        service._started_at = started_at
        service._started_monotonic = 0.0
        service._allocated_at_start = 0.0
        assert service._prepare_durable_exec_gate()
        service._write_service_lock_metadata(
            lease_state="stopped_verified",
            service_pid=None,
            ended_at=started_at + timedelta(seconds=3),
        )
        service._release_service_lock()
        lock_path.write_bytes(b'{"lease_state":"stopped_verified"')

        runner = _runner(
            tmp_path=tmp_path,
            ledger=ledger,
            service=cast(FakeFallbackService, service),
            register_development_adopter=False,
        )
        assert runner._validate_terminal_service_accounting() is None
        authoritative = service.read_authoritative_service_lease()
        assert authoritative is not None
        assert authoritative["lease_state"] == "stopped_verified"
        assert authoritative["accounting_session_id"] == event_id


def test_failure_result_retains_partial_call_reserve_and_shutdown_accounting(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    configuration = _fallback_launch_configuration(tmp_path)
    live: dict[str, object] = {}
    outputs = _fallback_outputs(trigger_repair=False)
    with Ledger(tmp_path / "ledger.sqlite3") as ledger:
        first_service = FakeFallbackService(configuration, ledger, outputs, live)
        first_runner = _runner(tmp_path=tmp_path, ledger=ledger, service=first_service)
        monkeypatch.setattr(os, "getpid", lambda: 62001)
        first_runner.prepare_controller_restart()

        second_service = FakeFallbackService(configuration, ledger, outputs, live)
        second_runner = _runner(tmp_path=tmp_path, ledger=ledger, service=second_service)
        second_runner.resource_sampler = cast(
            object,
            FailingAfterFirstCallResourceSampler(),
        )
        monkeypatch.setattr(os, "getpid", lambda: 62002)
        with pytest.raises(RuntimeError, match="post-call resource failure") as raised:
            second_runner.run()
        failure = second_runner.failure_result(
            raised.value,
            uptime=second_runner._last_shutdown_uptime,
        )

        assert failure["completed_base_call_count"] == 1
        assert len(failure["reserve_consumption"]) == 1
        assert failure["actual_plus_remaining_forecast"]["actual_allocated_seconds"] > 0
        assert failure["vllm_service_stopped"] is True
        assert failure["physical_service_state_unverified"] is False
        assert live["running"] is False


def test_failed_transport_is_accounted_but_not_used_as_latency_proxy(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    configuration = _fallback_launch_configuration(tmp_path)
    live: dict[str, object] = {}
    outputs = _fallback_outputs(trigger_repair=False)
    response_sha256 = "f" * 64
    private_marker = "/work" + "space/restricted/xgrammar-schema"
    with Ledger(tmp_path / "ledger.sqlite3") as ledger:
        first_service = FakeFallbackService(configuration, ledger, outputs, live)
        first_runner = _runner(tmp_path=tmp_path, ledger=ledger, service=first_service)
        monkeypatch.setattr(os, "getpid", lambda: 62501)
        first_runner.prepare_controller_restart()

        second_service = FakeFallbackService(configuration, ledger, outputs, live)
        original = second_service.run_fallback_test

        def failed_transport(*args: object, **kwargs: object) -> GenerationResult:
            original(*args, **kwargs)
            raise RuntimeTransportError(
                "vLLM returned HTTP 400",
                restricted_diagnostics={
                    "http_status": 400,
                    "response_sha256": response_sha256,
                    "response_size_bytes": 123,
                    "response_json_object": True,
                    "error_type": "BadRequestError",
                    "error_message": private_marker,
                    "error_code": 400,
                },
            )

        monkeypatch.setattr(second_service, "run_fallback_test", failed_transport)
        second_runner = _runner(
            tmp_path=tmp_path,
            ledger=ledger,
            service=second_service,
        )
        monkeypatch.setattr(os, "getpid", lambda: 62502)
        result = second_runner.run()

        assert result["micro_pilot_passed"] is False
        assert result["actual_plus_remaining_forecast"]["actual_allocated_seconds"] > 0
        assert "acceptance_c1" not in {row["call_class"] for row in result["timing_by_call_class"]}
        development_c1 = next(
            row
            for row in result["post_fallback_full_manifest_forecast"]["rows"]
            if row["call_class"] == "development_c1"
        )
        assert development_c1["forecast_p95_seconds"] == 180

        failures = ledger.failures_for_lineage("fallback-unit-fallback-c1-01-attempt")
        assert len(failures) == 1
        failure = failures[0]
        assert failure.artifact_hash is not None
        details = json.loads(failure.details_json)
        assert details == {
            "call_id": "fallback-c1-01",
            "exception_type": "RuntimeTransportError",
            "failure_stage": "transport",
            "restricted_diagnostics_artifact_hash": details[
                "restricted_diagnostics_artifact_hash"
            ],
        }
        public_diagnostic_record = ledger.get_artifact(failure.artifact_hash)
        assert public_diagnostic_record.release_class is ReleaseClass.PUBLIC
        restricted_hash = cast(str, details["restricted_diagnostics_artifact_hash"])
        assert restricted_hash != failure.artifact_hash
        diagnostic_record = ledger.get_artifact(restricted_hash)
        assert diagnostic_record.release_class is ReleaseClass.RESTRICTED
        diagnostic = json.loads(
            second_runner.artifacts.blobs.read_bytes(
                diagnostic_record,
                allow_restricted=True,
            )
        )
        assert diagnostic["response_sha256"] == response_sha256
        assert diagnostic["error_message"] == private_marker
        public_bytes = json.dumps(result, sort_keys=True).encode()
        assert private_marker.encode() not in public_bytes
        scan_public_bytes(public_bytes, relative_path="fallback-result.json")
        assert live["running"] is False


def test_transport_failure_before_gpu_event_is_terminal_and_audit_valid(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    configuration = _fallback_launch_configuration(tmp_path)
    live: dict[str, object] = {}
    outputs = _fallback_outputs(trigger_repair=False)
    with Ledger(tmp_path / "ledger.sqlite3") as ledger:
        first_service = FakeFallbackService(configuration, ledger, outputs, live)
        first_runner = _runner(tmp_path=tmp_path, ledger=ledger, service=first_service)
        monkeypatch.setattr(os, "getpid", lambda: 62601)
        first_runner.prepare_controller_restart()

        second_service = FakeFallbackService(configuration, ledger, outputs, live)

        def fail_before_event(*args: object, **kwargs: object) -> GenerationResult:
            del args, kwargs
            raise RuntimeError("synthetic pre-allocation transport failure")

        monkeypatch.setattr(second_service, "run_fallback_test", fail_before_event)
        second_runner = _runner(
            tmp_path=tmp_path,
            ledger=ledger,
            service=second_service,
        )
        monkeypatch.setattr(os, "getpid", lambda: 62602)
        result = second_runner.run()

        assert result["micro_pilot_passed"] is False
        attempt_id = "fallback-unit-fallback-c1-01-attempt"
        attempt = ledger.attempt_lineage(attempt_id)[-1]
        assert [item.to_state for item in ledger.transitions(attempt.job_id)] == [
            JobState.PLANNED,
            JobState.PREQUERY_SEALED,
            JobState.GENERATED,
            JobState.VALIDATED,
            JobState.FINALIZED,
        ]
        with pytest.raises(KeyError):
            ledger.get_model_call("fallback-unit-fallback-c1-01")
        validation = ledger.get_validation(f"{attempt_id}-validation")
        assert validation.validation_status.value == "rejected"
        assert validation.semantic_assessment_scope is (
            SemanticAssessmentScope.RUNTIME_STRUCTURAL_ONLY_NOT_ASSESSED
        )
        failures = tuple(
            failure
            for failure in ledger.failures_for_lineage(attempt_id)
            if failure.attempt_id == attempt_id
        )
        assert len(failures) == 1
        assert failures[0].artifact_hash == validation.diagnostics_artifact_hash
        assert ledger.get_artifact(failures[0].artifact_hash).release_class is ReleaseClass.PUBLIC
        assert live["running"] is False

    verification = verify_ledger(tmp_path / "ledger.sqlite3", tmp_path / "blobs")
    assert verification.valid, verification.issues


def test_guardian_recovers_pre_attempt_controller_death_once(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The durable attempt intent closes the reserve-to-attempt crash window."""

    configuration = _fallback_launch_configuration(tmp_path)
    live: dict[str, object] = {}
    with Ledger(tmp_path / "ledger.sqlite3") as ledger:
        service = FakeFallbackService(
            configuration,
            ledger,
            _fallback_outputs(trigger_repair=False),
            live,
        )
        monkeypatch.setattr(
            service,
            "read_authoritative_service_lease",
            lambda: None,
            raising=False,
        )
        runner = _runner(tmp_path=tmp_path, ledger=ledger, service=service)
        execution_hash = HASH_A
        call = next(
            call
            for call in fallback_pilot_calls(
                FallbackModelPolicy.load(ROOT / "configs/study/fallback_model.json")
            )
            if call.call_id == "fallback-c2-01"
        )
        sealed_at = datetime.now(UTC)
        job = ledger.create_or_resume_job(
            runner._job_identity(call=call, execution_hash=execution_hash),
            release_class=ReleaseClass.PUBLIC,
            created_at=sealed_at,
        )
        runner._prepare_job_lifecycle(
            job_id=job.job_id,
            call=call,
            prequery_sealed_at=sealed_at,
        )
        state = runner._initial_state(execution_hash)
        runner._save(state)
        attempt_id = f"{runner.run_id}-{call.call_id}-attempt"
        created_at = datetime.now(UTC)
        runner._reserve(
            state,
            call_id=call.call_id,
            reserve_call_class=call.reserve_call_class,
            watchdog_seconds=call.watchdog_seconds,
            job_id=job.job_id,
            attempt_id=attempt_id,
            attempt_kind=AttemptKind.BASE,
            parent_attempt_id=None,
            input_hash=HASH_A,
            config_hash=HASH_B,
            seed=call.seed_block,
            attempt_created_at=created_at,
        )

        # Simulate process death before record_attempt(), then replay once before
        # the terminal guardian consumes the same immutable receipt.
        receipt = runner._terminalize_interrupted_call_lifecycle(state)
        assert receipt is not None
        transition_count = len(ledger.transitions(job.job_id))
        failure_count = len(ledger.failures_for_lineage(attempt_id))
        assert runner._terminalize_interrupted_call_lifecycle(state) == receipt
        assert len(ledger.transitions(job.job_id)) == transition_count
        assert len(ledger.failures_for_lineage(attempt_id)) == failure_count

        result = fallback_acceptance_module._guardian_terminalize(
            runner,
            invocation={
                "manifest_sha256": HASH_A,
                "execution_arguments_sha256": HASH_B,
            },
            ticket={"manifest_sha256": "c" * 64},
            trigger="terminal_request",
            terminal_request_sha256="d" * 64,
            controller_takeover_sha256=None,
            control_group_outcomes=(),
        )
        assert result["physical_shutdown_verified"] is True
        checkpoint = json.loads(runner.checkpoint_path.read_text(encoding="utf-8"))
        assert checkpoint["active_call_id"] is None
        assert checkpoint["active_attempt"] is None
        assert checkpoint["failed_call_id"] == call.call_id
        assert [item.to_state for item in ledger.transitions(job.job_id)] == [
            JobState.PLANNED,
            JobState.PREQUERY_SEALED,
            JobState.QUERY_REVEALED,
            JobState.GENERATED,
            JobState.VALIDATED,
            JobState.FINALIZED,
        ]
        validation = ledger.get_validation(f"{attempt_id}-validation")
        assert validation.validation_status is ValidationStatus.REJECTED
        assert validation.semantic_assessment_scope is (
            SemanticAssessmentScope.RUNTIME_STRUCTURAL_ONLY_NOT_ASSESSED
        )
        failures = tuple(
            failure
            for failure in ledger.failures_for_lineage(attempt_id)
            if failure.attempt_id == attempt_id
        )
        assert len(failures) == 1
        assert failures[0].failure_kind is FailureKind.INTERRUPTED

    verification = verify_ledger(tmp_path / "ledger.sqlite3", tmp_path / "blobs")
    assert verification.valid, verification.issues


def test_guardian_recovers_pre_attempt_repair_controller_death_once(
    tmp_path: Path,
) -> None:
    """A reserved repair is reconstructed with its exact parent and replayed once."""

    configuration = _fallback_launch_configuration(tmp_path)
    live: dict[str, object] = {}
    with Ledger(tmp_path / "ledger.sqlite3") as ledger:
        service = FakeFallbackService(
            configuration,
            ledger,
            _fallback_outputs(trigger_repair=True),
            live,
        )
        runner = _runner(tmp_path=tmp_path, ledger=ledger, service=service)
        execution_hash = HASH_A
        call = next(
            call
            for call in fallback_pilot_calls(
                FallbackModelPolicy.load(ROOT / "configs/study/fallback_model.json")
            )
            if call.call_id == "fallback-c2-01"
        )
        sealed_at = datetime.now(UTC)
        job = ledger.create_or_resume_job(
            runner._job_identity(call=call, execution_hash=execution_hash),
            release_class=ReleaseClass.PUBLIC,
            created_at=sealed_at,
        )
        runner._prepare_job_lifecycle(
            job_id=job.job_id,
            call=call,
            prequery_sealed_at=sealed_at,
        )
        state = runner._initial_state(execution_hash)
        runner._save(state)

        base_attempt_id = f"{runner.run_id}-{call.call_id}-attempt"
        base_created_at = (
            datetime.fromisoformat(ledger.transitions(job.job_id)[-1].occurred_at)
            + timedelta(microseconds=1)
        )
        runner._reserve(
            state,
            call_id=call.call_id,
            reserve_call_class=call.reserve_call_class,
            watchdog_seconds=call.watchdog_seconds,
            job_id=job.job_id,
            attempt_id=base_attempt_id,
            attempt_kind=AttemptKind.BASE,
            parent_attempt_id=None,
            input_hash=HASH_A,
            config_hash=HASH_B,
            seed=call.seed_block,
            attempt_created_at=base_created_at,
        )
        ledger.record_attempt(
            attempt_id=base_attempt_id,
            job_id=job.job_id,
            attempt_kind=AttemptKind.BASE,
            input_hash=HASH_A,
            config_hash=HASH_B,
            seed=call.seed_block,
            created_at=base_created_at,
        )
        base_terminal_at = base_created_at + timedelta(seconds=1)
        base_event_id = f"{runner.run_id}-{call.call_id}-gpu"
        ledger.record_gpu_event(
            event_id=base_event_id,
            event_kind=GpuEventKind.FALLBACK_TEST,
            allocated_seconds=1,
            started_at=base_created_at,
            ended_at=base_terminal_at,
            succeeded=True,
            job_id=job.job_id,
            attempt_id=base_attempt_id,
        )
        invalid_artifact = runner.artifacts.put_bytes(
            b'{"invalid":true}\n',
            media_type="application/json",
            release_class=ReleaseClass.PUBLIC,
            created_at=base_terminal_at,
        )
        ledger.record_model_call(
            model_call_id=base_attempt_id.removesuffix("-attempt"),
            job_id=job.job_id,
            attempt_id=base_attempt_id,
            gpu_event_id=base_event_id,
            backend=fallback_acceptance_module.ModelBackend.VLLM_GPU,
            call_role=runner._call_role(call),
            retry_class=call.retry_class,
            model_manifest_hash=configuration.configuration_hash,
            decoding_manifest_hash=HASH_B,
            request_hash=HASH_A,
            response_artifact_hash=invalid_artifact.content_hash,
            construction_unit_hash=fallback_acceptance_module._base_construction_unit_hash(
                call
            ),
            served_context_count=1,
            prompt_tokens=10,
            completion_tokens=5,
            allocated_gpu_seconds=1,
            successful=True,
            created_at=base_terminal_at,
        )
        runner._advance_one(
            job_id=job.job_id,
            call=call,
            state=JobState.GENERATED,
            occurred_at=base_terminal_at,
        )
        base_validation_id = runner._record_structural_validation(
            job_id=job.job_id,
            attempt_id=base_attempt_id,
            input_artifact_hash=invalid_artifact.content_hash,
            validator_manifest_hash=execution_hash,
            accepted=False,
            diagnostics_artifact_hash=invalid_artifact.content_hash,
            created_at=base_terminal_at,
        )
        ledger.record_failure(
            attempt_id=base_attempt_id,
            failure_kind=FailureKind.INVALID_OUTPUT,
            message="Synthetic base output requires repair",
            details={"call_id": call.call_id},
            artifact_hash=invalid_artifact.content_hash,
            occurred_at=base_terminal_at,
        )

        state["repair_parent_call_id"] = call.call_id
        repair_call_id = f"{call.call_id}-repair-01"
        repair_attempt_id = f"{runner.run_id}-{repair_call_id}-attempt"
        repair_created_at = base_terminal_at + timedelta(microseconds=1)
        runner._reserve(
            state,
            call_id=repair_call_id,
            reserve_call_class="reserve_short",
            watchdog_seconds=90,
            job_id=job.job_id,
            attempt_id=repair_attempt_id,
            attempt_kind=AttemptKind.REPAIR,
            parent_attempt_id=base_attempt_id,
            input_hash="c" * 64,
            config_hash="d" * 64,
            seed=call.seed_block,
            attempt_created_at=repair_created_at,
        )

        receipt = runner._terminalize_interrupted_call_lifecycle(state)
        assert receipt is not None
        transition_count = len(ledger.transitions(job.job_id))
        failure_count = len(ledger.failures_for_lineage(repair_attempt_id))
        assert runner._terminalize_interrupted_call_lifecycle(state) == receipt
        assert len(ledger.transitions(job.job_id)) == transition_count
        assert len(ledger.failures_for_lineage(repair_attempt_id)) == failure_count

        repair_attempt = ledger.attempt_lineage(repair_attempt_id)[-1]
        assert repair_attempt.attempt_kind is AttemptKind.REPAIR
        assert repair_attempt.parent_attempt_id == base_attempt_id
        assert [item.to_state for item in ledger.transitions(job.job_id)] == [
            JobState.PLANNED,
            JobState.PREQUERY_SEALED,
            JobState.QUERY_REVEALED,
            JobState.GENERATED,
            JobState.REPAIRED,
            JobState.VALIDATED,
            JobState.FINALIZED,
        ]
        repair_validation = ledger.get_validation(f"{repair_attempt_id}-validation")
        assert repair_validation.validation_status is ValidationStatus.REJECTED
        assert repair_validation.parent_validation_id == base_validation_id
        assert repair_validation.repair_attempt_id == repair_attempt_id
        repair_failures = tuple(
            failure
            for failure in ledger.failures_for_lineage(repair_attempt_id)
            if failure.attempt_id == repair_attempt_id
        )
        assert len(repair_failures) == 1
        assert repair_failures[0].failure_kind is FailureKind.INTERRUPTED

    verification = verify_ledger(tmp_path / "ledger.sqlite3", tmp_path / "blobs")
    assert verification.valid, verification.issues


def test_interrupted_result_commit_preserves_accepted_validation(
    tmp_path: Path,
) -> None:
    configuration = _fallback_launch_configuration(tmp_path)
    live: dict[str, object] = {}
    with Ledger(tmp_path / "ledger.sqlite3") as ledger:
        service = FakeFallbackService(
            configuration,
            ledger,
            _fallback_outputs(trigger_repair=False),
            live,
        )
        runner = _runner(tmp_path=tmp_path, ledger=ledger, service=service)
        execution_hash = HASH_A
        call = next(
            call
            for call in fallback_pilot_calls(
                FallbackModelPolicy.load(ROOT / "configs/study/fallback_model.json")
            )
            if call.call_id == "fallback-c2-01"
        )
        sealed_at = datetime.now(UTC)
        job = ledger.create_or_resume_job(
            runner._job_identity(call=call, execution_hash=execution_hash),
            release_class=ReleaseClass.PUBLIC,
            created_at=sealed_at,
        )
        runner._prepare_job_lifecycle(
            job_id=job.job_id,
            call=call,
            prequery_sealed_at=sealed_at,
        )
        state = runner._initial_state(execution_hash)
        runner._save(state)
        attempt_id = f"{runner.run_id}-{call.call_id}-attempt"
        created_at = datetime.now(UTC)
        runner._reserve(
            state,
            call_id=call.call_id,
            reserve_call_class=call.reserve_call_class,
            watchdog_seconds=call.watchdog_seconds,
            job_id=job.job_id,
            attempt_id=attempt_id,
            attempt_kind=AttemptKind.BASE,
            parent_attempt_id=None,
            input_hash=HASH_A,
            config_hash=HASH_B,
            seed=call.seed_block,
            attempt_created_at=created_at,
        )
        ledger.record_attempt(
            attempt_id=attempt_id,
            job_id=job.job_id,
            attempt_kind=AttemptKind.BASE,
            input_hash=HASH_A,
            config_hash=HASH_B,
            seed=call.seed_block,
            created_at=created_at,
        )
        event_id = f"{runner.run_id}-{call.call_id}-gpu"
        event_end = created_at + timedelta(seconds=1)
        ledger.record_gpu_event(
            event_id=event_id,
            event_kind=GpuEventKind.FALLBACK_TEST,
            allocated_seconds=1,
            started_at=created_at,
            ended_at=event_end,
            succeeded=True,
            job_id=job.job_id,
            attempt_id=attempt_id,
        )
        response = runner.artifacts.put_bytes(
            b'{"durable":"response"}\n',
            media_type="application/json",
            release_class=ReleaseClass.PUBLIC,
            created_at=event_end,
        )
        ledger.record_model_call(
            model_call_id=attempt_id.removesuffix("-attempt"),
            job_id=job.job_id,
            attempt_id=attempt_id,
            gpu_event_id=event_id,
            backend=fallback_acceptance_module.ModelBackend.VLLM_GPU,
            call_role=runner._call_role(call),
            retry_class=call.retry_class,
            model_manifest_hash=configuration.configuration_hash,
            decoding_manifest_hash=HASH_B,
            request_hash=HASH_A,
            response_artifact_hash=response.content_hash,
            construction_unit_hash=fallback_acceptance_module._base_construction_unit_hash(
                call
            ),
            served_context_count=1,
            prompt_tokens=10,
            completion_tokens=5,
            allocated_gpu_seconds=1,
            successful=True,
            created_at=event_end,
        )
        runner._advance_one(
            job_id=job.job_id,
            call=call,
            state=JobState.GENERATED,
            occurred_at=event_end,
        )
        runner._record_structural_validation(
            job_id=job.job_id,
            attempt_id=attempt_id,
            input_artifact_hash=response.content_hash,
            validator_manifest_hash=execution_hash,
            accepted=True,
            created_at=event_end,
        )
        runner._finish_job_lifecycle(
            job_id=job.job_id,
            call=call,
            occurred_at=event_end,
        )

        runner._terminalize_interrupted_call_lifecycle(state)

        validation = ledger.get_validation(f"{attempt_id}-validation")
        assert validation.validation_status is ValidationStatus.ACCEPTED
        failures = tuple(
            failure
            for failure in ledger.failures_for_lineage(attempt_id)
            if failure.attempt_id == attempt_id
        )
        assert len(failures) == 1
        assert failures[0].failure_kind is FailureKind.INTERRUPTED
        assert ledger.get_job(job.job_id).state is JobState.FINALIZED

    verification = verify_ledger(tmp_path / "ledger.sqlite3", tmp_path / "blobs")
    assert verification.valid, verification.issues


def test_failed_diagnostic_repair_retains_accounting_but_not_latency_proxy(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    configuration = _fallback_launch_configuration(tmp_path)
    live: dict[str, object] = {}
    outputs = _fallback_outputs(trigger_repair=True)
    with Ledger(tmp_path / "ledger.sqlite3") as ledger:
        first_service = FakeFallbackService(configuration, ledger, outputs, live)
        first_runner = _runner(tmp_path=tmp_path, ledger=ledger, service=first_service)
        monkeypatch.setattr(os, "getpid", lambda: 63001)
        first_runner.prepare_controller_restart()

        second_service = FakeFallbackService(configuration, ledger, outputs, live)
        original_generate = second_service.generate

        def failed_generate(*args: object, **kwargs: object) -> GenerationResult:
            original_generate(*args, **kwargs)
            raise RuntimeError("synthetic repair transport failure")

        monkeypatch.setattr(second_service, "generate", failed_generate)
        second_runner = _runner(
            tmp_path=tmp_path,
            ledger=ledger,
            service=second_service,
        )
        monkeypatch.setattr(os, "getpid", lambda: 63002)
        result = second_runner.run()

        assert result["micro_pilot_passed"] is False
        assert result["repair_attempt_count"] == 1
        assert len(result["reserve_consumption"]) == 2
        assert result["timing_gate"]["repair_sample_count"] == 0
        assert result["timing_gate"]["completed_repair_transport_count"] == 0
        assert result["timing_gate"]["repair_sample_count_valid"] is True
        assert sum(event.event_kind is GpuEventKind.REPAIR for event in ledger.gpu_events()) == 1
        assert live["running"] is False


def _orchestrator_options(tmp_path: Path) -> object:
    arguments = [
        "--execute",
        "--controller-stage",
        "orchestrate",
        "--project-root",
        str(ROOT),
        "--output",
        str(tmp_path / "result.json"),
        "--run-id",
        "fallback-orchestrator-test",
    ]
    for name in (
        "primary-result",
        "activation-certificate",
        "cache-replacement-receipt",
        "snapshot",
        "shared-cache",
        "verified-model-manifest",
        "source-association",
        "ledger",
        "artifact-root",
        "checkpoint",
        "quota-root",
    ):
        arguments.extend((f"--{name}", str(tmp_path / name)))
    return parse_arguments(arguments)


def _write_orchestrator_status_identity(options: object) -> tuple[dict[str, object], ...]:
    paths = fallback_acceptance_module._orchestration_paths(options)
    created_at = datetime.now(UTC)
    invocation_payload: dict[str, object] = {
        "schema_version": "1.0.0",
        "kind": "fallback_controller_orchestration_invocation",
        "run_id": options.run_id,
        "execution_arguments_sha256": canonical_sha256(
            fallback_acceptance_module._controller_execution_arguments(options)
        ),
        "result_output": str(paths.result_output),
        "handoff_output": str(paths.handoff_output),
        "cleanup_output": str(paths.cleanup_output),
        "checkpoint": str(paths.checkpoint),
        "service_session_id": options.run_id,
        "service_event_id": f"{options.run_id}-service-start-001",
        "invocation_nonce": HASH_A,
        "gpu_seconds_before_invocation": 0.0,
        "protected_shutdown_seconds": 75.0,
        "hard_stop_at": (created_at + timedelta(hours=1)).isoformat(),
        "resume_grace_seconds": 300.0,
        "created_at": created_at.isoformat(),
    }
    invocation = {
        **invocation_payload,
        "manifest_sha256": canonical_sha256(invocation_payload),
    }
    fallback_acceptance_module._write_append_only_json(paths.invocation, invocation)
    ticket = fallback_acceptance_module._guardian_ticket_for_invocation(
        options,
        paths=paths,
        invocation=invocation,
    )
    fallback_acceptance_module._write_append_only_json(paths.guardian_ticket, ticket)
    guard_payload: dict[str, object] = {
        "schema_version": "1.0.0",
        "kind": "fallback_controller_orchestrator_guard",
        "state": "active",
        "run_id": options.run_id,
        "sequence": 1,
        "previous_guard_sha256": None,
        "orchestration_invocation_sha256": invocation["manifest_sha256"],
        "guardian_ticket_sha256": ticket["manifest_sha256"],
        "orchestrator_pid": 999_999_999,
        "orchestrator_start_ticks": 1,
        "orchestrator_command_sha256": HASH_A,
        "orchestrator_process_group_id": 999_999_999,
        "orchestrator_session_id": 999_999_999,
        "execution_arguments_sha256": invocation["execution_arguments_sha256"],
        "started_at": created_at.isoformat(),
    }
    guard = {
        **guard_payload,
        "manifest_sha256": canonical_sha256(guard_payload),
    }
    fallback_acceptance_module._write_append_only_json(
        fallback_acceptance_module._guard_path(options, 1),
        guard,
    )
    return invocation, ticket


def test_guardian_readiness_wait_covers_measured_safe_initialization_bound(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    options = _orchestrator_options(tmp_path)
    paths = fallback_acceptance_module._orchestration_paths(options)
    elapsed = {"seconds": 0.0}
    ready = {"kind": "exact-live-guardian"}

    monkeypatch.setattr(
        fallback_acceptance_module,
        "_live_guardian_receipt",
        lambda *args, **kwargs: (
            ready if elapsed["seconds"] >= 106.0 else None
        ),
    )
    monkeypatch.setattr(
        fallback_acceptance_module,
        "_launch_guardian_process",
        lambda *args, **kwargs: SimpleNamespace(poll=lambda: None),
    )
    monkeypatch.setattr(
        fallback_acceptance_module.time,
        "monotonic",
        lambda: elapsed["seconds"],
    )
    monkeypatch.setattr(
        fallback_acceptance_module.time,
        "sleep",
        lambda seconds: elapsed.__setitem__("seconds", elapsed["seconds"] + 5.0),
    )

    observed = fallback_acceptance_module._ensure_guardian_running(
        options,
        paths=paths,
        invocation={"manifest_sha256": HASH_A},
        ticket={"manifest_sha256": HASH_B},
    )

    assert observed is ready
    assert (
        fallback_acceptance_module.FALLBACK_GUARDIAN_READY_TIMEOUT_SECONDS
        == AMENDED_FALLBACK_STARTUP_WATCHDOG_SECONDS
    )
    assert 106 <= elapsed["seconds"] < 300


def test_live_guardian_receipt_rejects_a_self_hashed_non_guardian_process(
    tmp_path: Path,
) -> None:
    options = _orchestrator_options(tmp_path)
    invocation, ticket = _write_orchestrator_status_identity(options)
    paths = fallback_acceptance_module._orchestration_paths(options)
    pid = os.getpid()
    start_ticks, process_command_sha256 = fallback_acceptance_module._process_identity(pid)
    process_group_id = os.getpgrp()
    session_id = os.getsid(0)
    expected_command_sha256 = fallback_acceptance_module._raw_command_sha256(
        fallback_acceptance_module._guardian_controller_command(
            options,
            output=paths.guardian_result,
            ticket=paths.guardian_ticket,
        )
    )
    assert (
        process_command_sha256 != expected_command_sha256
        or process_group_id != pid
        or session_id != pid
    )
    ready_payload: dict[str, object] = {
        "schema_version": "1.0.0",
        "kind": "fallback_service_guardian_ready",
        "run_id": options.run_id,
        "orchestration_invocation_sha256": invocation["manifest_sha256"],
        "guardian_ticket_sha256": ticket["manifest_sha256"],
        "guardian_command_sha256": ticket["guardian_command_sha256"],
        "guardian_pid": pid,
        "guardian_start_ticks": start_ticks,
        "guardian_process_command_sha256": process_command_sha256,
        "guardian_process_group_id": process_group_id,
        "guardian_session_id": session_id,
        "ready_at": datetime.now(UTC).isoformat(),
    }
    ready = {
        **ready_payload,
        "manifest_sha256": canonical_sha256(ready_payload),
    }
    fallback_acceptance_module._write_append_only_json(
        fallback_acceptance_module._guardian_ready_path(paths, 1),
        ready,
    )

    with pytest.raises(ValueError, match="changed its invocation identity"):
        fallback_acceptance_module._live_guardian_receipt(
            options,
            paths,
            invocation=invocation,
            ticket=ticket,
        )


def test_status_reads_guardian_held_wal_and_allows_only_clean_exact_resume(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    options = _orchestrator_options(tmp_path)
    invocation, ticket = _write_orchestrator_status_identity(options)
    monkeypatch.setattr(
        fallback_acceptance_module,
        "_live_guardian_receipt",
        lambda *args, **kwargs: {"manifest_sha256": ticket["manifest_sha256"]},
    )

    with Ledger(options.ledger) as writable:
        writable.record_gpu_event(
            event_id="status-live-wal",
            event_kind=GpuEventKind.WARM_UP,
            allocated_seconds=2,
            started_at=datetime.now(UTC),
            ended_at=datetime.now(UTC),
            succeeded=True,
        )
        with pytest.raises(ArtifactIntegrityError, match="closed and checkpointed"):
            ReadOnlyLedger(options.ledger)

        status = fallback_acceptance_module._orchestrator_status(options)

    assert status["invocation_sha256"] == invocation["manifest_sha256"]
    assert status["guardian_identity_verified"] is True
    assert status["guardian_live"] is True
    assert status["ledger_snapshot_verified"] is True
    assert status["ledger_snapshot_state"] == "verified_coordinated_read"
    assert status["actual_allocated_gpu_seconds"] == 2
    assert status["unresolved_gpu_allocation_count"] == 0
    assert status["unresolved_gpu_service_count"] == 0
    assert status["resume_accounting_state_verified"] is True
    assert status["resume_allowed"] is True


def test_status_never_allows_resume_with_unresolved_live_gpu_journal(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    options = _orchestrator_options(tmp_path)
    _invocation, ticket = _write_orchestrator_status_identity(options)
    monkeypatch.setattr(
        fallback_acceptance_module,
        "_live_guardian_receipt",
        lambda *args, **kwargs: {"manifest_sha256": ticket["manifest_sha256"]},
    )

    with Ledger(options.ledger) as writable:
        started_at = datetime.now(UTC)
        writable.record_gpu_service_observation(
            service_session_id="status-unresolved-service",
            state=GpuServiceJournalState.OPENED,
            session_id="status-live-service",
            configuration_hash=HASH_A,
            service_started_at=started_at,
            elapsed_seconds=0,
            ledger_allocated_seconds_before_session=0,
            hard_limit_seconds=36_000,
            observed_at=started_at,
        )
        status = fallback_acceptance_module._orchestrator_status(options)

    assert status["ledger_snapshot_verified"] is True
    assert status["unresolved_gpu_service_count"] == 1
    assert status["resume_accounting_state_verified"] is False
    assert status["resume_allowed"] is False


def test_status_preserves_exact_open_service_resume_semantics(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    options = _orchestrator_options(tmp_path)
    _invocation, ticket = _write_orchestrator_status_identity(options)
    paths = fallback_acceptance_module._orchestration_paths(options)
    paths.checkpoint.write_text(
        json.dumps(
            {
                "run_id": options.run_id,
                "service_start_attempted": True,
                "controller_handoff_complete": False,
                "active_call_id": None,
                "failed_call_id": None,
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        fallback_acceptance_module,
        "_live_guardian_receipt",
        lambda *args, **kwargs: {"manifest_sha256": ticket["manifest_sha256"]},
    )
    expected_service_event_id = f"{options.run_id}-service-start-001"

    with Ledger(options.ledger) as writable:
        started_at = datetime.now(UTC)
        writable.record_gpu_service_observation(
            service_session_id=expected_service_event_id,
            state=GpuServiceJournalState.OPENED,
            session_id=options.run_id,
            configuration_hash=HASH_A,
            service_started_at=started_at,
            elapsed_seconds=0,
            ledger_allocated_seconds_before_session=0,
            hard_limit_seconds=36_000,
            observed_at=started_at,
        )
        status = fallback_acceptance_module._orchestrator_status(options)

    assert status["unresolved_gpu_service_count"] == 1
    assert status["resume_accounting_state_verified"] is True
    assert status["resume_allowed"] is True


def test_status_rejects_attempted_service_start_without_exact_open_journal(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    options = _orchestrator_options(tmp_path)
    _invocation, ticket = _write_orchestrator_status_identity(options)
    paths = fallback_acceptance_module._orchestration_paths(options)
    paths.checkpoint.write_text(
        json.dumps(
            {
                "run_id": options.run_id,
                "service_start_attempted": True,
                "controller_handoff_complete": False,
                "active_call_id": None,
                "failed_call_id": None,
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        fallback_acceptance_module,
        "_live_guardian_receipt",
        lambda *args, **kwargs: {"manifest_sha256": ticket["manifest_sha256"]},
    )

    with Ledger(options.ledger):
        status = fallback_acceptance_module._orchestrator_status(options)

    assert status["ledger_snapshot_verified"] is True
    assert status["unresolved_gpu_service_count"] == 0
    assert status["unresolved_gpu_allocation_count"] == 0
    assert status["resume_accounting_state_verified"] is False
    assert status["resume_allowed"] is False


def test_status_reports_stale_wal_without_live_guardian_fail_closed(
    tmp_path: Path,
) -> None:
    options = _orchestrator_options(tmp_path)
    _write_orchestrator_status_identity(options)
    wal = Path(str(options.ledger) + "-wal")
    with Ledger(options.ledger) as writable:
        writable.record_gpu_event(
            event_id="stale-status-wal",
            event_kind=GpuEventKind.WARM_UP,
            allocated_seconds=2,
            started_at=datetime.now(UTC),
            ended_at=datetime.now(UTC),
            succeeded=True,
        )
        stale_wal = wal.read_bytes()
    wal.write_bytes(stale_wal)

    status = fallback_acceptance_module._orchestrator_status(options)

    assert status["ledger_snapshot_verified"] is False
    assert status["ledger_snapshot_state"] == "unavailable_or_unsafe"
    assert status["guardian_live"] is False
    assert status["actual_allocated_gpu_seconds"] is None
    assert status["unresolved_gpu_allocation_count"] is None
    assert status["unresolved_gpu_service_count"] is None
    assert status["resume_allowed"] is False


def test_guardian_cli_preserves_outer_result_identity_across_fresh_interpreter(
    tmp_path: Path,
) -> None:
    options = _orchestrator_options(tmp_path)
    paths = fallback_acceptance_module._orchestration_paths(options)
    created_at = datetime.now(UTC).isoformat()
    invocation_payload = {
        "schema_version": "1.0.0",
        "kind": "fallback_controller_orchestration_invocation",
        "run_id": options.run_id,
        "execution_arguments_sha256": canonical_sha256(
            fallback_acceptance_module._controller_execution_arguments(options)
        ),
        "result_output": str(paths.result_output),
        "handoff_output": str(paths.handoff_output),
        "cleanup_output": str(paths.cleanup_output),
        "checkpoint": str(paths.checkpoint),
        "service_session_id": options.run_id,
        "service_event_id": f"{options.run_id}-service-start-001",
        "invocation_nonce": "a" * 64,
        "gpu_seconds_before_invocation": 0.0,
        "protected_shutdown_seconds": 70.0,
        "hard_stop_at": (datetime.now(UTC) + timedelta(hours=1)).isoformat(),
        "resume_grace_seconds": 300.0,
        "created_at": created_at,
    }
    invocation = {
        **invocation_payload,
        "manifest_sha256": canonical_sha256(invocation_payload),
    }
    fallback_acceptance_module._write_append_only_json(paths.invocation, invocation)
    ticket = fallback_acceptance_module._guardian_ticket_for_invocation(
        options,
        paths=paths,
        invocation=invocation,
    )
    fallback_acceptance_module._write_append_only_json(paths.guardian_ticket, ticket)
    command = fallback_acceptance_module._guardian_controller_command(
        options,
        output=paths.guardian_result,
        ticket=paths.guardian_ticket,
    )
    program = (
        "import sys; "
        "from story_projection_onto import fallback_acceptance as module; "
        "options=module.parse_arguments(sys.argv[1:]); "
        "module._require_execution_arguments(options); "
        "module._validate_guardian_ticket(options); "
        "print('guardian-cli-identity-valid')"
    )

    completed = subprocess.run(
        (sys.executable, "-c", program, *command[2:]),
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
    assert completed.stdout.strip() == "guardian-cli-identity-valid"

    tampered = list(command[2:])
    result_index = tampered.index("--orchestration-result-output") + 1
    tampered[result_index] = str(tmp_path / "different-final-result.json")
    rejected = subprocess.run(
        (sys.executable, "-c", program, *tampered),
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert rejected.returncode != 0
    assert "fallback orchestration invocation identity changed" in rejected.stderr


def test_guardian_launcher_creates_an_independent_exact_process_session(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    options = _orchestrator_options(tmp_path)
    paths = fallback_acceptance_module._orchestration_paths(options)
    receipt = tmp_path / "guardian-process.json"
    program = (
        "import hashlib,json,os,sys; "
        "raw=open(f'/proc/{os.getpid()}/cmdline','rb').read(); "
        "payload={'pid':os.getpid(),'pgrp':os.getpgrp(),'sid':os.getsid(0),"
        "'command_sha256':hashlib.sha256(raw).hexdigest()}; "
        "open(sys.argv[1],'w',encoding='utf-8').write(json.dumps(payload))"
    )
    command = (sys.executable, "-c", program, str(receipt))
    monkeypatch.setattr(
        fallback_acceptance_module,
        "_guardian_controller_command",
        lambda *args, **kwargs: command,
    )

    process = fallback_acceptance_module._launch_guardian_process(
        options,
        paths=paths,
    )
    assert process.wait(timeout=5) == 0
    identity = json.loads(receipt.read_text(encoding="utf-8"))
    assert identity["pid"] == process.pid
    assert identity["pgrp"] == process.pid
    assert identity["sid"] == process.pid
    assert identity["command_sha256"] == fallback_acceptance_module._raw_command_sha256(
        command
    )


def test_guardian_process_identity_binds_raw_command_group_and_session(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    options = _orchestrator_options(tmp_path)
    paths = fallback_acceptance_module._orchestration_paths(options)
    expected_command = (sys.executable, "guardian-sentinel.py", "--bounded")
    expected_hash = fallback_acceptance_module._raw_command_sha256(expected_command)
    pid = os.getpid()
    monkeypatch.setattr(
        fallback_acceptance_module,
        "_guardian_controller_command",
        lambda *args, **kwargs: expected_command,
    )
    monkeypatch.setattr(
        fallback_acceptance_module,
        "_process_identity",
        lambda observed_pid: (12345, expected_hash),
    )
    monkeypatch.setattr(os, "getpgrp", lambda: pid)
    monkeypatch.setattr(os, "getsid", lambda observed_pid: pid)

    assert fallback_acceptance_module._guardian_process_identity(
        options,
        paths=paths,
    ) == (pid, 12345, expected_hash, pid, pid)

    monkeypatch.setattr(os, "getsid", lambda observed_pid: pid + 1)
    with pytest.raises(RuntimeError, match="not isolated"):
        fallback_acceptance_module._guardian_process_identity(options, paths=paths)


@pytest.mark.parametrize(
    ("stage", "expected_suffix"),
    (
        ("prepare", ".controller-handoff.json"),
        ("recover-prepare", ".controller-handoff.json"),
        ("run", "result.json"),
        ("cleanup", ".orphan-cleanup.json"),
        ("guardian", ".guardian-result.json"),
    ),
)
def test_internal_controller_cli_requires_exact_stage_output(
    tmp_path: Path,
    stage: str,
    expected_suffix: str,
) -> None:
    outer = _orchestrator_options(tmp_path)
    paths = fallback_acceptance_module._orchestration_paths(outer)
    expected_output = {
        "prepare": paths.handoff_output,
        "recover-prepare": paths.handoff_output,
        "run": paths.result_output,
        "cleanup": paths.cleanup_output,
        "guardian": paths.guardian_result,
    }[stage]
    command = fallback_acceptance_module._internal_controller_command(
        outer,
        stage=stage,
        output=expected_output,
        guard=None,
    )
    if stage == "guardian":
        command = (*command, "--guardian-ticket", str(paths.guardian_ticket))
    parsed = parse_arguments(command[2:])
    fallback_acceptance_module._require_execution_arguments(parsed)
    assert str(parsed.output).endswith(expected_suffix)
    assert parsed.orchestration_result_output.resolve() == paths.result_output

    parsed.output = tmp_path / "wrong-stage-output.json"
    with pytest.raises(SystemExit, match="does not match its orchestration stage"):
        fallback_acceptance_module._require_execution_arguments(parsed)


def test_v9_controller_identity_and_internal_command_bind_complete_incident_chain(
    tmp_path: Path,
) -> None:
    options = _orchestrator_options(tmp_path)
    options.run_id = fallback_acceptance_module.SECOND_RECOVERY_V9_RUN_ID
    options.retry_amendment = tmp_path / "v3-amendment.json"
    options.prior_fallback_failure = tmp_path / "v3-failure.json"
    options.second_recovery_overlay = tmp_path / "v9-overlay.json"
    options.second_recovery_v3_result = tmp_path / "v3-result.json"
    options.second_recovery_v3_incident = tmp_path / "v3-incident.json"
    options.prior_control_plane_incident = tmp_path / "v4-incident.json"

    with pytest.raises(SystemExit, match="v4/v5/v6/v7/v8"):
        fallback_acceptance_module._require_execution_arguments(options)

    options.prior_v5_control_plane_incident = tmp_path / "v5-incident.json"
    options.prior_v6_control_plane_incident = tmp_path / "v6-incident.json"
    options.prior_v7_runtime_incident = tmp_path / "v7-incident.json"
    options.prior_v8_runtime_incident = tmp_path / "v8-incident.json"
    with pytest.raises(SystemExit, match="v4/v5/v6/v7/v8"):
        fallback_acceptance_module._require_execution_arguments(options)

    options.prior_v8_lease_repair_receipt = tmp_path / "v8-lease-repair.json"
    fallback_acceptance_module._require_execution_arguments(options)
    identity = fallback_acceptance_module._controller_execution_arguments(options)
    assert identity["prior_control_plane_incident"] == str(
        options.prior_control_plane_incident.resolve()
    )
    assert identity["prior_v5_control_plane_incident"] == str(
        options.prior_v5_control_plane_incident.resolve()
    )
    assert identity["prior_v6_control_plane_incident"] == str(
        options.prior_v6_control_plane_incident.resolve()
    )
    assert identity["prior_v7_runtime_incident"] == str(
        options.prior_v7_runtime_incident.resolve()
    )
    assert identity["prior_v8_runtime_incident"] == str(
        options.prior_v8_runtime_incident.resolve()
    )
    assert identity["prior_v8_lease_repair_receipt"] == str(
        options.prior_v8_lease_repair_receipt.resolve()
    )

    command = fallback_acceptance_module._internal_controller_command(
        options,
        stage="prepare",
        output=fallback_acceptance_module._orchestration_paths(options).handoff_output,
        guard=None,
    )
    parsed = parse_arguments(command[2:])
    assert fallback_acceptance_module._controller_execution_arguments(parsed) == identity


def test_public_orchestrator_rejects_private_result_path_and_wrong_process_group(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    options = _orchestrator_options(tmp_path)
    paths = fallback_acceptance_module._orchestration_paths(options)
    options.orchestration_result_output = tmp_path / "injected-result.json"
    with pytest.raises(SystemExit, match="reserved for internal controllers"):
        fallback_acceptance_module._require_execution_arguments(options)

    options.orchestration_result_output = None
    for terminal_run_id in (
        fallback_acceptance_module.SECOND_RECOVERY_V4_RUN_ID,
        fallback_acceptance_module.SECOND_RECOVERY_V5_RUN_ID,
        fallback_acceptance_module.SECOND_RECOVERY_V6_RUN_ID,
        fallback_acceptance_module.SECOND_RECOVERY_V7_RUN_ID,
        fallback_acceptance_module.SECOND_RECOVERY_V8_RUN_ID,
    ):
        options.run_id = terminal_run_id
        with pytest.raises(SystemExit, match="terminal incident"):
            fallback_acceptance_module._require_execution_arguments(options)

    options.run_id = "fallback-orchestration-test"
    monkeypatch.setattr(os, "getpgrp", lambda: os.getpid() + 1)
    with pytest.raises(RuntimeError, match="dedicated process group"):
        fallback_acceptance_module._orchestrate_controller_processes(options)
    assert not paths.invocation.exists()
    assert not paths.guardian_ticket.exists()


def test_status_requires_and_verifies_complete_durable_invocation_arguments(
    tmp_path: Path,
) -> None:
    complete = _orchestrator_options(tmp_path)
    paths = fallback_acceptance_module._orchestration_paths(complete)
    invocation_payload = {
        "schema_version": "1.0.0",
        "kind": "fallback_controller_orchestration_invocation",
        "run_id": complete.run_id,
        "execution_arguments_sha256": canonical_sha256(
            fallback_acceptance_module._controller_execution_arguments(complete)
        ),
        "result_output": str(paths.result_output),
        "handoff_output": str(paths.handoff_output),
        "cleanup_output": str(paths.cleanup_output),
        "checkpoint": str(paths.checkpoint),
        "gpu_seconds_before_invocation": 0.0,
        "hard_stop_at": (datetime.now(UTC) + timedelta(hours=1)).isoformat(),
    }
    invocation = {
        **invocation_payload,
        "manifest_sha256": canonical_sha256(invocation_payload),
    }
    fallback_acceptance_module._write_append_only_json(paths.invocation, invocation)

    minimal = parse_arguments(
        [
            "--status",
            "--output",
            str(complete.output),
            "--run-id",
            complete.run_id,
            "--checkpoint",
            str(complete.checkpoint),
            "--ledger",
            str(complete.ledger),
        ]
    )
    with pytest.raises(
        SystemExit,
        match="status with a durable orchestration invocation requires",
    ):
        fallback_acceptance_module._orchestrator_status(minimal)

    status = fallback_acceptance_module._orchestrator_status(complete)
    assert status["invocation_present"] is True
    assert status["invocation_sha256"] == invocation["manifest_sha256"]

    changed = copy.deepcopy(complete)
    changed.source_association = tmp_path / "different-source-association.json"
    with pytest.raises(ValueError, match="durable orchestration identity"):
        fallback_acceptance_module._orchestrator_status(changed)


def test_orchestrator_supervises_two_controllers_and_closes_guard(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from story_projection_onto.fallback_acceptance import (
        _orchestrate_controller_processes,
    )

    options = _orchestrator_options(tmp_path)
    stages: list[str] = []
    closed: list[dict[str, object]] = []
    invocation = {
        "run_id": options.run_id,
        "execution_arguments_sha256": HASH_A,
        "manifest_sha256": HASH_B,
    }
    ticket = {"manifest_sha256": "c" * 64}
    guard_path = tmp_path / "checkpoint.orchestrator-guard-000001.json"
    guard = {"run_id": options.run_id, "manifest_sha256": "d" * 64}

    def run(command: tuple[str, ...], *, check: bool) -> subprocess.CompletedProcess[str]:
        assert check is False
        stage = command[command.index("--controller-stage") + 1]
        output = Path(command[command.index("--output") + 1])
        stages.append(stage)
        output.write_text("{}\n", encoding="utf-8")
        return subprocess.CompletedProcess(command, 2 if stage == "run" else 0)

    def validate(path: Path, *, expected_stage: str, **kwargs: object) -> dict[str, object]:
        del path, kwargs
        return {
            "phase1_gate_passed": expected_stage != "run",
            "manifest_sha256": "e" * 64,
        }

    class Prepared:
        preparation_seconds = 0.25

        def __init__(self, command):
            self.command = command
            stages.append("cpu-ready")

        def release_and_wait(self):
            return run(self.command, check=False).returncode

        def close(self):
            pass

    monkeypatch.setattr(fallback_acceptance_module, "PreparedController", Prepared)

    monkeypatch.setattr(subprocess, "run", run)
    monkeypatch.setattr(
        fallback_acceptance_module,
        "_new_orchestration_identity",
        lambda options, paths: (invocation, ticket),
    )
    monkeypatch.setattr(
        fallback_acceptance_module,
        "_create_orchestrator_guard",
        lambda options, invocation, ticket: (guard_path, guard),
    )
    monkeypatch.setattr(
        fallback_acceptance_module,
        "_ensure_guardian_running",
        lambda *args, **kwargs: {"manifest_sha256": "f" * 64},
    )
    monkeypatch.setattr(
        fallback_acceptance_module,
        "_validate_controller_stage_result",
        validate,
    )
    monkeypatch.setattr(
        fallback_acceptance_module,
        "_request_guardian_terminal_verification",
        lambda *args, **kwargs: {"manifest_sha256": "1" * 64},
    )
    monkeypatch.setattr(
        fallback_acceptance_module,
        "_close_orchestrator_guard",
        lambda *args, **kwargs: closed.append(dict(kwargs)),
    )
    monkeypatch.setattr(os, "getpgrp", os.getpid)
    assert _orchestrate_controller_processes(options) == 2
    assert stages == ["cpu-ready", "prepare", "run"]
    receipt = json.loads(options.checkpoint.with_name(
        options.checkpoint.name + ".cpu-preparation.json"
    ).read_text())
    assert receipt["completed_before_service_start"] is True
    assert len(closed) == 1
    assert closed[0]["physical_shutdown_verified"] is True


def test_orchestrator_invokes_cleanup_when_run_does_not_verify_shutdown(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from story_projection_onto.fallback_acceptance import (
        _orchestrate_controller_processes,
    )

    options = _orchestrator_options(tmp_path)
    stages: list[str] = []
    invocation = {
        "run_id": options.run_id,
        "execution_arguments_sha256": HASH_A,
        "manifest_sha256": HASH_B,
    }
    ticket = {"manifest_sha256": "c" * 64}
    guard_path = tmp_path / "checkpoint.orchestrator-guard-000001.json"
    guard = {"run_id": options.run_id, "manifest_sha256": "d" * 64}

    def run(command: tuple[str, ...], *, check: bool) -> subprocess.CompletedProcess[str]:
        assert check is False
        stage = command[command.index("--controller-stage") + 1]
        output = Path(command[command.index("--output") + 1])
        stages.append(stage)
        output.write_text("{}\n", encoding="utf-8")
        return subprocess.CompletedProcess(command, 0)

    def validate(path: Path, *, expected_stage: str, **kwargs: object) -> dict[str, object]:
        del path, kwargs
        if expected_stage == "run":
            raise RuntimeError("run lacks verified service shutdown")
        return {"manifest_sha256": "e" * 64, "physical_shutdown_verified": True}

    class Prepared:
        preparation_seconds = 0.25

        def __init__(self, command):
            self.command = command
            stages.append("cpu-ready")

        def release_and_wait(self):
            return run(self.command, check=False).returncode

        def close(self):
            pass

    monkeypatch.setattr(fallback_acceptance_module, "PreparedController", Prepared)

    monkeypatch.setattr(subprocess, "run", run)
    monkeypatch.setattr(
        fallback_acceptance_module,
        "_new_orchestration_identity",
        lambda options, paths: (invocation, ticket),
    )
    monkeypatch.setattr(
        fallback_acceptance_module,
        "_create_orchestrator_guard",
        lambda options, invocation, ticket: (guard_path, guard),
    )
    monkeypatch.setattr(
        fallback_acceptance_module,
        "_ensure_guardian_running",
        lambda *args, **kwargs: {"manifest_sha256": "f" * 64},
    )
    monkeypatch.setattr(
        fallback_acceptance_module,
        "_validate_controller_stage_result",
        validate,
    )
    monkeypatch.setattr(
        fallback_acceptance_module,
        "_request_guardian_terminal_verification",
        lambda *args, **kwargs: {"manifest_sha256": "1" * 64},
    )
    monkeypatch.setattr(
        fallback_acceptance_module,
        "_close_orchestrator_guard",
        lambda *args, **kwargs: None,
    )
    monkeypatch.setattr(os, "getpgrp", os.getpid)
    with pytest.raises(RuntimeError, match="verified service shutdown"):
        _orchestrate_controller_processes(options)
    assert stages == ["cpu-ready", "prepare", "run", "cleanup"]


def test_controller_result_replay_rejects_coherent_outer_tampering(
    tmp_path: Path,
) -> None:
    options = _orchestrator_options(tmp_path)
    invocation_payload = {
        "kind": "fallback_controller_orchestration_invocation",
        "run_id": options.run_id,
        "execution_arguments_sha256": HASH_A,
    }
    invocation = {
        **invocation_payload,
        "manifest_sha256": canonical_sha256(invocation_payload),
    }
    guard_payload = {
        "schema_version": "1.0.0",
        "kind": "fallback_controller_orchestrator_guard",
        "run_id": options.run_id,
        "sequence": 1,
        "previous_guard_sha256": None,
    }
    guard = {**guard_payload, "manifest_sha256": canonical_sha256(guard_payload)}
    guard_path = tmp_path / "checkpoint.orchestrator-guard-000001.json"
    guard_path.write_text(json.dumps(guard), encoding="utf-8")
    original_payload = {
        "schema_version": "1.0.0",
        "kind": "phase1_fallback_micro_pilot_result",
        "run_id": options.run_id,
        "phase1_gate_passed": False,
        "physical_service_live": False,
        "vllm_service_stopped": True,
        "bounded_value": "original",
    }
    original = {
        **original_payload,
        "manifest_sha256": canonical_sha256(original_payload),
    }
    bound = fallback_acceptance_module._bind_controller_stage_result(
        original,
        stage="run",
        guard=guard,
        invocation=invocation,
    )
    bound["bounded_value"] = "coherently-tampered"
    rebound = {key: value for key, value in bound.items() if key != "manifest_sha256"}
    bound["manifest_sha256"] = canonical_sha256(rebound)
    result_path = tmp_path / "result.json"
    result_path.write_text(json.dumps(bound), encoding="utf-8")

    with pytest.raises(ValueError, match="inner result"):
        fallback_acceptance_module._validate_controller_stage_result(
            result_path,
            options=options,
            invocation=invocation,
            expected_stage="run",
        )


def test_resume_orchestrator_recovers_prepare_without_second_service_start(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    options = _orchestrator_options(tmp_path)
    options.resume_orchestrator = True
    Path(options.checkpoint).write_text(
        json.dumps(
            {
                "run_id": options.run_id,
                "service_start_attempted": True,
                "controller_handoff_complete": False,
                "active_call_id": None,
                "failed_call_id": None,
            }
        ),
        encoding="utf-8",
    )
    invocation = {
        "run_id": options.run_id,
        "execution_arguments_sha256": HASH_A,
        "manifest_sha256": HASH_B,
    }
    ticket = {"manifest_sha256": "c" * 64}
    guard_path = tmp_path / "checkpoint.orchestrator-guard-000001.json"
    guard = {"run_id": options.run_id, "manifest_sha256": "d" * 64}
    stages: list[str] = []

    def run(command: tuple[str, ...], *, check: bool) -> subprocess.CompletedProcess[str]:
        assert check is False
        stage = command[command.index("--controller-stage") + 1]
        stages.append(stage)
        Path(command[command.index("--output") + 1]).write_text("{}\n")
        return subprocess.CompletedProcess(command, 2 if stage == "run" else 0)

    def validate(path: Path, *, expected_stage: str, **kwargs: object) -> dict[str, object]:
        del path, kwargs
        return {
            "phase1_gate_passed": False,
            "manifest_sha256": "e" * 64,
            "orchestration_controller_stage": expected_stage,
        }

    monkeypatch.setattr(subprocess, "run", run)
    monkeypatch.setattr(
        fallback_acceptance_module,
        "_load_orchestration_identity",
        lambda options, paths: (invocation, ticket),
    )
    monkeypatch.setattr(
        fallback_acceptance_module,
        "_create_orchestrator_guard",
        lambda options, invocation, ticket: (guard_path, guard),
    )
    monkeypatch.setattr(
        fallback_acceptance_module,
        "_ensure_guardian_running",
        lambda *args, **kwargs: {"manifest_sha256": "f" * 64},
    )
    monkeypatch.setattr(
        fallback_acceptance_module,
        "_validate_controller_stage_result",
        validate,
    )
    monkeypatch.setattr(
        fallback_acceptance_module,
        "_request_guardian_terminal_verification",
        lambda *args, **kwargs: {"manifest_sha256": "1" * 64},
    )
    monkeypatch.setattr(
        fallback_acceptance_module,
        "_close_orchestrator_guard",
        lambda *args, **kwargs: None,
    )
    monkeypatch.setattr(os, "getpgrp", os.getpid)

    assert fallback_acceptance_module._orchestrate_controller_processes(options) == 2
    assert stages == ["recover-prepare", "run"]
    assert "prepare" not in stages


def test_fresh_orchestration_rejects_stale_output_before_guardian_or_controller(
    tmp_path: Path,
) -> None:
    options = _orchestrator_options(tmp_path)
    paths = fallback_acceptance_module._orchestration_paths(options)
    paths.result_output.write_text("stale\n", encoding="utf-8")

    with pytest.raises(RuntimeError, match="absent durable state and outputs"):
        fallback_acceptance_module._new_orchestration_identity(options, paths)


def test_resume_repairs_invocation_to_ticket_power_loss_before_any_controller(
    tmp_path: Path,
) -> None:
    options = _orchestrator_options(tmp_path)
    options.resume_orchestrator = True
    paths = fallback_acceptance_module._orchestration_paths(options)
    arguments_hash = canonical_sha256(
        fallback_acceptance_module._controller_execution_arguments(options)
    )
    created_at = datetime.now(UTC).isoformat()
    invocation_payload = {
        "schema_version": "1.0.0",
        "kind": "fallback_controller_orchestration_invocation",
        "run_id": options.run_id,
        "execution_arguments_sha256": arguments_hash,
        "result_output": str(paths.result_output),
        "handoff_output": str(paths.handoff_output),
        "cleanup_output": str(paths.cleanup_output),
        "checkpoint": str(paths.checkpoint),
        "service_session_id": options.run_id,
        "service_event_id": f"{options.run_id}-service-start-001",
        "invocation_nonce": "a" * 64,
        "gpu_seconds_before_invocation": 0.0,
        "protected_shutdown_seconds": 70.0,
        "hard_stop_at": (datetime.now(UTC) + timedelta(hours=1)).isoformat(),
        "resume_grace_seconds": 300.0,
        "created_at": created_at,
    }
    invocation = {
        **invocation_payload,
        "manifest_sha256": canonical_sha256(invocation_payload),
    }
    fallback_acceptance_module._write_append_only_json(paths.invocation, invocation)

    loaded_invocation, ticket = fallback_acceptance_module._load_orchestration_identity(
        options, paths
    )

    assert loaded_invocation == invocation
    assert paths.guardian_ticket.is_file()
    assert ticket == fallback_acceptance_module._guardian_ticket_for_invocation(
        options,
        paths=paths,
        invocation=invocation,
    )


def test_guard_inventory_rejects_missing_or_reordered_sequence(tmp_path: Path) -> None:
    options = _orchestrator_options(tmp_path)
    guard_path = tmp_path / "checkpoint.orchestrator-guard-000002.json"
    guard_path.write_text("{}\n", encoding="utf-8")

    with pytest.raises(RuntimeError, match="not contiguous"):
        fallback_acceptance_module._guard_paths(options)


def test_orchestrator_liveness_requires_unchanged_process_command(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    guard = {
        "orchestrator_pid": 41001,
        "orchestrator_start_ticks": 9001,
        "orchestrator_command_sha256": HASH_A,
    }
    monkeypatch.setattr(os, "kill", lambda pid, signal_number: None)
    monkeypatch.setattr(
        fallback_acceptance_module,
        "_process_identity",
        lambda pid: (9001, HASH_A),
    )
    assert fallback_acceptance_module._guard_process_is_live(guard)

    monkeypatch.setattr(
        fallback_acceptance_module,
        "_process_identity",
        lambda pid: (9001, HASH_B),
    )
    assert not fallback_acceptance_module._guard_process_is_live(guard)


def test_closed_receipt_cannot_authorize_two_live_orchestrators(
    tmp_path: Path,
) -> None:
    options = _orchestrator_options(tmp_path)
    invocation = {
        "manifest_sha256": HASH_A,
        "execution_arguments_sha256": HASH_B,
    }
    ticket = {"manifest_sha256": "c" * 64}
    start_ticks, command_sha256 = fallback_acceptance_module._process_identity(os.getpid())
    guard_payload = {
        "schema_version": "1.0.0",
        "kind": "fallback_controller_orchestrator_guard",
        "state": "active",
        "run_id": options.run_id,
        "sequence": 1,
        "previous_guard_sha256": None,
        "orchestration_invocation_sha256": HASH_A,
        "guardian_ticket_sha256": "c" * 64,
        "orchestrator_pid": os.getpid(),
        "orchestrator_start_ticks": start_ticks,
        "orchestrator_command_sha256": command_sha256,
        "execution_arguments_sha256": HASH_B,
        "started_at": datetime.now(UTC).isoformat(),
    }
    guard = {**guard_payload, "manifest_sha256": canonical_sha256(guard_payload)}
    guard_path = tmp_path / "checkpoint.orchestrator-guard-000001.json"
    guard_path.write_text(json.dumps(guard), encoding="utf-8")
    close_payload = {
        "schema_version": "1.0.0",
        "kind": "fallback_controller_orchestrator_closed",
        "run_id": options.run_id,
        "orchestrator_guard_sha256": guard["manifest_sha256"],
        "physical_shutdown_verified": True,
    }
    close = {**close_payload, "manifest_sha256": canonical_sha256(close_payload)}
    fallback_acceptance_module._guard_close_path(guard_path).write_text(
        json.dumps(close),
        encoding="utf-8",
    )

    with pytest.raises(RuntimeError, match="live fallback orchestrator"):
        fallback_acceptance_module._create_orchestrator_guard(
            options,
            invocation=invocation,
            ticket=ticket,
        )


def test_guardian_hard_deadline_terminalizes_without_starting_service(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    options = _orchestrator_options(tmp_path)
    invocation = {
        "run_id": options.run_id,
        "execution_arguments_sha256": HASH_A,
        "manifest_sha256": HASH_B,
    }
    ticket = {
        "manifest_sha256": "c" * 64,
        "guardian_command_sha256": "d" * 64,
        "hard_stop_at": (datetime.now(UTC) - timedelta(seconds=1)).isoformat(),
        "resume_grace_seconds": 300.0,
    }
    Path(options.checkpoint).write_text(
        json.dumps(
            {
                "run_id": options.run_id,
                "service_start_attempted": True,
                "active_call_id": None,
                "failed_call_id": None,
            }
        ),
        encoding="utf-8",
    )
    calls: list[str] = []

    class FakeGuardianRunner:
        run_id = options.run_id
        checkpoint_path = Path(options.checkpoint)
        service = SimpleNamespace(
            state=ServiceState.STOPPED,
            configuration=SimpleNamespace(configuration_hash=HASH_A),
            read_authoritative_service_lease=lambda: None,
        )
        _last_service_adoption_failure_type = None
        ledger = SimpleNamespace(
            latest_gpu_service_journal=lambda event_id: None,
            unresolved_gpu_allocations=lambda: (),
            unresolved_gpu_service_journals=lambda: (),
            gpu_summary=lambda: SimpleNamespace(total_allocated_seconds=0.0),
        )
        _service_event_id = f"{options.run_id}-service-start-001"

        def _stop_or_reconcile_service(self) -> tuple[None, bool]:
            calls.append("terminalize")
            return None, False

        def _validate_terminal_service_accounting(self) -> None:
            return None

        def _save(self, state: Mapping[str, object]) -> None:
            calls.append("save")

    monkeypatch.setattr(
        fallback_acceptance_module,
        "_load_orchestration_identity",
        lambda options, paths: (invocation, ticket),
    )
    monkeypatch.setattr(
        fallback_acceptance_module,
        "_guardian_process_identity",
        lambda options, paths: (os.getpid(), 1, HASH_A, os.getpgrp(), os.getsid(0)),
    )
    result = fallback_acceptance_module._run_guardian(
        options,
        runner=cast(FallbackAcceptanceRunner, FakeGuardianRunner()),
        ticket=ticket,
    )

    assert result["trigger"] == "hard_stop_deadline"
    assert result["physical_shutdown_verified"] is True
    assert result["checkpoint_service_adoption_failed"] is False
    assert result["checkpoint_service_adoption_failure_type"] is None
    assert calls == ["terminalize", "save"]


def test_guardian_terminalizes_exact_lease_when_checkpoint_was_lost(
    tmp_path: Path,
) -> None:
    """An atomic service lease remains actionable without a runner checkpoint."""

    calls: list[str] = []
    service = SimpleNamespace(
        state=ServiceState.STOPPED,
        configuration=SimpleNamespace(configuration_hash=HASH_A),
        read_authoritative_service_lease=lambda: {
            "session_id": "fallback-test",
            "accounting_session_id": "fallback-test-service-start-001",
            "configuration_hash": HASH_A,
        },
    )

    class LeaseOnlyRunner:
        run_id = "fallback-test"
        checkpoint_path = tmp_path / "missing-checkpoint.json"
        _last_service_adoption_failure_type = None
        _service_event_id = "fallback-test-service-start-001"
        ledger = SimpleNamespace(
            latest_gpu_service_journal=lambda event_id: None,
            unresolved_gpu_allocations=lambda: (),
            unresolved_gpu_service_journals=lambda: (),
            gpu_summary=lambda: SimpleNamespace(total_allocated_seconds=11.0),
        )

        def _stop_or_reconcile_service(self) -> tuple[None, bool]:
            calls.append("terminalize")
            return None, True

        def _validate_terminal_service_accounting(self) -> None:
            calls.append("accounting")
            return None

    runner = LeaseOnlyRunner()
    runner.service = service
    result = fallback_acceptance_module._guardian_terminalize(
        cast(FallbackAcceptanceRunner, runner),
        invocation={
            "manifest_sha256": HASH_B,
            "execution_arguments_sha256": HASH_A,
        },
        ticket={"manifest_sha256": "c" * 64},
        trigger="hard_stop_deadline",
        terminal_request_sha256=None,
        controller_takeover_sha256="d" * 64,
        control_group_outcomes=(),
    )

    assert calls == ["terminalize", "accounting"]
    assert result["checkpoint_service_adopted"] is True
    assert result["physical_shutdown_verified"] is True


def test_guardian_rejects_terminal_success_with_any_unresolved_service_journal(
    tmp_path: Path,
) -> None:
    """Terminal proof is ledger-wide, not limited to the fallback event ID."""

    class RunnerWithForeignUnresolvedService:
        run_id = "fallback-test"
        checkpoint_path = tmp_path / "missing-checkpoint.json"
        _last_service_adoption_failure_type = None
        _service_event_id = "fallback-test-service-start-001"
        service = SimpleNamespace(
            state=ServiceState.STOPPED,
            configuration=SimpleNamespace(configuration_hash=HASH_A),
            read_authoritative_service_lease=lambda: None,
        )
        ledger = SimpleNamespace(
            latest_gpu_service_journal=lambda event_id: None,
            unresolved_gpu_allocations=lambda: (),
            unresolved_gpu_service_journals=lambda: (
                SimpleNamespace(service_session_id="prior-unresolved-service"),
            ),
            gpu_summary=lambda: SimpleNamespace(total_allocated_seconds=11.0),
        )

        def _validate_terminal_service_accounting(self) -> None:
            return None

    with pytest.raises(RuntimeError, match="could not verify terminal"):
        fallback_acceptance_module._guardian_terminalize(
            cast(FallbackAcceptanceRunner, RunnerWithForeignUnresolvedService()),
            invocation={
                "manifest_sha256": HASH_B,
                "execution_arguments_sha256": HASH_A,
            },
            ticket={"manifest_sha256": "c" * 64},
            trigger="hard_stop_deadline",
            terminal_request_sha256=None,
            controller_takeover_sha256="d" * 64,
            control_group_outcomes=(),
        )


def test_guardian_hard_deadline_kills_lock_owner_then_exact_service_target(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A wedged controller's flock cannot defeat the independent hard stop."""

    options = _orchestrator_options(tmp_path)
    paths = fallback_acceptance_module._orchestration_paths(options)
    controller_lock = tmp_path / "controller.lock"
    controller_ready = tmp_path / "controller.ready"
    controller_program = (
        "import fcntl, os, pathlib, sys, time; "
        "stream=open(sys.argv[1], 'w'); "
        "fcntl.flock(stream.fileno(), fcntl.LOCK_EX); "
        "pathlib.Path(sys.argv[2]).write_text(str(os.getpid())); "
        "time.sleep(120)"
    )
    controller_command = (
        sys.executable,
        "-c",
        controller_program,
        str(controller_lock),
        str(controller_ready),
    )
    controller = subprocess.Popen(controller_command, start_new_session=True)
    service = subprocess.Popen(
        (sys.executable, "-c", "import time; time.sleep(120)"),
        start_new_session=True,
    )
    try:
        deadline = time.monotonic() + 5
        while not controller_ready.is_file() and time.monotonic() < deadline:
            time.sleep(0.01)
        assert controller_ready.is_file()

        controller_ticks, controller_command_hash = fallback_acceptance_module._process_identity(
            controller.pid
        )
        controller_group = os.getpgid(controller.pid)
        controller_session = os.getsid(controller.pid)
        assert controller_group == controller.pid
        service_ticks, service_command_hash = fallback_acceptance_module._process_identity(
            service.pid
        )
        service_identity = {
            "service_pid": service.pid,
            "service_start_ticks": service_ticks,
            "service_command_sha256": service_command_hash,
            "service_process_group_id": os.getpgid(service.pid),
            "service_session_id": os.getsid(service.pid),
        }

        invocation = {
            "run_id": options.run_id,
            "execution_arguments_sha256": HASH_A,
            "manifest_sha256": HASH_B,
        }
        ticket = {
            "manifest_sha256": "c" * 64,
            "guardian_command_sha256": "d" * 64,
            "hard_stop_at": (datetime.now(UTC) - timedelta(seconds=1)).isoformat(),
            "resume_grace_seconds": 300.0,
        }
        guard_payload = {
            "schema_version": "1.0.0",
            "kind": "fallback_controller_orchestrator_guard",
            "state": "active",
            "run_id": options.run_id,
            "sequence": 1,
            "previous_guard_sha256": None,
            "orchestration_invocation_sha256": HASH_B,
            "guardian_ticket_sha256": ticket["manifest_sha256"],
            "orchestrator_pid": controller.pid,
            "orchestrator_start_ticks": controller_ticks,
            "orchestrator_command_sha256": controller_command_hash,
            "orchestrator_process_group_id": controller_group,
            "orchestrator_session_id": controller_session,
            "execution_arguments_sha256": HASH_A,
            "started_at": datetime.now(UTC).isoformat(),
        }
        guard = {
            **guard_payload,
            "manifest_sha256": canonical_sha256(guard_payload),
        }
        fallback_acceptance_module._write_append_only_json(
            fallback_acceptance_module._guard_path(options, 1),
            guard,
        )
        receipt_payload = {
            "schema_version": "1.0.0",
            "kind": "fallback_internal_controller_receipt",
            "run_id": options.run_id,
            "sequence": 1,
            "previous_controller_receipt_sha256": None,
            "controller_stage": "cleanup",
            "controller_output": str(paths.cleanup_output.resolve()),
            "orchestration_invocation_sha256": HASH_B,
            "orchestrator_guard_sha256": guard["manifest_sha256"],
            "execution_arguments_sha256": HASH_A,
            "controller_pid": controller.pid,
            "controller_start_ticks": controller_ticks,
            "controller_command_sha256": controller_command_hash,
            "controller_process_group_id": controller_group,
            "controller_session_id": controller_session,
            "registered_at": datetime.now(UTC).isoformat(),
        }
        receipt = {
            **receipt_payload,
            "manifest_sha256": canonical_sha256(receipt_payload),
        }
        fallback_acceptance_module._write_append_only_json(
            fallback_acceptance_module._controller_receipt_path(options, 1),
            receipt,
        )
        Path(options.checkpoint).write_text(
            json.dumps(
                {
                    "run_id": options.run_id,
                    "service_start_attempted": True,
                    "active_call_id": None,
                    "failed_call_id": None,
                }
            ),
            encoding="utf-8",
        )

        fake_service = SimpleNamespace(
            state=ServiceState.READY,
            configuration=SimpleNamespace(configuration_hash=HASH_A),
            read_authoritative_service_lease=lambda: {
                "session_id": options.run_id,
                "accounting_session_id": f"{options.run_id}-service-start-001",
                "configuration_hash": HASH_A,
            },
        )
        cleanup_observations: list[str] = []

        class LockHoldingGuardianRunner:
            run_id = options.run_id
            checkpoint_path = Path(options.checkpoint)
            _last_service_adoption_failure_type = None
            _service_event_id = f"{options.run_id}-service-start-001"
            service = fake_service
            ledger = SimpleNamespace(
                latest_gpu_service_journal=lambda event_id: SimpleNamespace(
                    service_session_id=event_id
                ),
                unresolved_gpu_allocations=lambda: (),
                unresolved_gpu_service_journals=lambda: (),
                gpu_summary=lambda: SimpleNamespace(total_allocated_seconds=17.0),
            )

            def _stop_or_reconcile_service(self) -> tuple[None, bool]:
                descriptor = os.open(controller_lock, os.O_RDWR)
                try:
                    fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    cleanup_observations.append("controller-lock-released")
                finally:
                    os.close(descriptor)
                assert fallback_acceptance_module._exact_bound_process_is_live(
                    service_identity,
                    prefix="service",
                )
                os.killpg(service.pid, signal.SIGTERM)
                service.wait(timeout=5)
                fake_service.state = ServiceState.STOPPED
                cleanup_observations.append("service-target-stopped")
                return None, True

            def _validate_terminal_service_accounting(self) -> None:
                assert service.poll() is not None
                cleanup_observations.append("accounting-terminal")
                return None

            def _save(self, state: Mapping[str, object]) -> None:
                assert state["orphan_cleanup_completed"] is True

        monkeypatch.setattr(
            fallback_acceptance_module,
            "_load_orchestration_identity",
            lambda options, paths: (invocation, ticket),
        )
        monkeypatch.setattr(
            fallback_acceptance_module,
            "_internal_controller_command",
            lambda options, *, stage, output, guard: controller_command,
        )
        monkeypatch.setattr(
            fallback_acceptance_module,
            "_guardian_process_identity",
            lambda options, paths: (
                os.getpid(),
                1,
                HASH_A,
                os.getpgrp(),
                os.getsid(0),
            ),
        )

        result = fallback_acceptance_module._run_guardian(
            options,
            runner=cast(FallbackAcceptanceRunner, LockHoldingGuardianRunner()),
            ticket=ticket,
        )
        controller.wait(timeout=5)

        assert result["trigger"] == "hard_stop_deadline"
        assert result["controller_takeover_sha256"] is not None
        assert result["physical_shutdown_verified"] is True
        assert result["checkpoint_service_adopted"] is True
        assert cleanup_observations == [
            "controller-lock-released",
            "service-target-stopped",
            "accounting-terminal",
        ]
        assert controller.poll() is not None
        assert service.poll() is not None
        assert paths.controller_takeover.is_file()
        with pytest.raises(RuntimeError, match="takeover"):
            fallback_acceptance_module._register_internal_controller(
                options,
                guard=guard,
                invocation=invocation,
            )
        status = fallback_acceptance_module._orchestrator_status(options)
        assert status["resume_allowed"] is False
        assert status["controller_launch_authority_revoked"] is True
    finally:
        for process in (controller, service):
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGKILL)
            with suppress(subprocess.TimeoutExpired):
                process.wait(timeout=5)


def test_controller_kill_recovers_open_call_before_service_without_double_count(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A pre-existing guardian meter closes a killed controller's open call once."""

    options = _orchestrator_options(tmp_path)
    paths = fallback_acceptance_module._orchestration_paths(options)
    configuration = _fallback_launch_configuration(tmp_path)
    ledger_path = tmp_path / "guardian-open-call.sqlite3"
    ready_path = tmp_path / "controller-open-call.ready"
    target_program = "import time; time.sleep(120)"
    target_command = (sys.executable, "-c", target_program)
    monkeypatch.setattr(
        VLLMLaunchConfiguration,
        "command",
        lambda self, python_executable=sys.executable: target_command,
    )
    controller_program = """
import sys
import time
from pathlib import Path

from story_projection_onto.experiment import AllocatedGPUMeter
from story_projection_onto.gpu_runtime import VLLMLaunchConfiguration, VLLMService
from story_projection_onto.store import Ledger

cache = Path(sys.argv[1])
snapshot = Path(sys.argv[2])
model_configuration = Path(sys.argv[3])
ledger_path = Path(sys.argv[4])
ready_path = Path(sys.argv[5])
run_id = sys.argv[6]
target_program = sys.argv[7]
target_command = (sys.executable, "-c", target_program)
VLLMLaunchConfiguration.command = (
    lambda self, python_executable=sys.executable: target_command
)

class EndpointAbsentClient:
    def endpoint_live(self, timeout_seconds):
        return False

configuration = VLLMLaunchConfiguration.from_model_configuration(
    snapshot_path=snapshot,
    shared_cache=cache,
    model_configuration_path=model_configuration,
    model_candidate="fallback",
    verified_snapshot_manifest_sha256="a" * 64,
)
with Ledger(ledger_path) as ledger:
    meter = AllocatedGPUMeter(ledger)
    service = VLLMService(
        configuration=configuration,
        client=EndpointAbsentClient(),
        meter=meter,
        readiness_check=lambda: True,
    )
    service.start(
        session_id=run_id,
        event_id=f"{run_id}-service-start-001",
        watchdog_seconds=180,
    )
    with meter.inference(
        event_id=f"{run_id}-open-call",
        maximum_seconds=120,
    ):
        ready_path.write_text(str(service.pid), encoding="utf-8")
        time.sleep(120)
"""
    controller_command = (
        sys.executable,
        "-c",
        controller_program,
        str(configuration.shared_cache),
        str(configuration.snapshot_path),
        str(ROOT / "configs/study/model.json"),
        str(ledger_path),
        str(ready_path),
        options.run_id,
        target_program,
    )
    target_pid: int | None = None
    controller: subprocess.Popen[bytes] | None = None
    with Ledger(ledger_path) as ledger:
        # This is the important production ordering: the guardian and its meter
        # exist before the controller opens either journal.
        guardian_meter = AllocatedGPUMeter(ledger)
        controller = subprocess.Popen(controller_command, start_new_session=True)
        try:
            deadline = time.monotonic() + 10
            while not ready_path.is_file() and time.monotonic() < deadline:
                if controller.poll() is not None:
                    raise AssertionError(
                        f"controller exited before opening its call ({controller.returncode})"
                    )
                time.sleep(0.01)
            assert ready_path.is_file()
            target_pid = int(ready_path.read_text(encoding="utf-8"))
            assert ledger.unresolved_gpu_allocations()

            controller_ticks, controller_command_hash = (
                fallback_acceptance_module._process_identity(controller.pid)
            )
            controller_group = os.getpgid(controller.pid)
            controller_session = os.getsid(controller.pid)
            invocation = {
                "run_id": options.run_id,
                "execution_arguments_sha256": HASH_A,
                "manifest_sha256": HASH_B,
            }
            ticket = {"manifest_sha256": "c" * 64}
            guard_payload = {
                "schema_version": "1.0.0",
                "kind": "fallback_controller_orchestrator_guard",
                "state": "active",
                "run_id": options.run_id,
                "sequence": 1,
                "previous_guard_sha256": None,
                "orchestration_invocation_sha256": HASH_B,
                "guardian_ticket_sha256": ticket["manifest_sha256"],
                "orchestrator_pid": controller.pid,
                "orchestrator_start_ticks": controller_ticks,
                "orchestrator_command_sha256": controller_command_hash,
                "orchestrator_process_group_id": controller_group,
                "orchestrator_session_id": controller_session,
                "execution_arguments_sha256": HASH_A,
                "started_at": datetime.now(UTC).isoformat(),
            }
            guard = {
                **guard_payload,
                "manifest_sha256": canonical_sha256(guard_payload),
            }
            fallback_acceptance_module._write_append_only_json(
                fallback_acceptance_module._guard_path(options, 1),
                guard,
            )
            receipt_payload = {
                "schema_version": "1.0.0",
                "kind": "fallback_internal_controller_receipt",
                "run_id": options.run_id,
                "sequence": 1,
                "previous_controller_receipt_sha256": None,
                "controller_stage": "cleanup",
                "controller_output": str(paths.cleanup_output.resolve()),
                "orchestration_invocation_sha256": HASH_B,
                "orchestrator_guard_sha256": guard["manifest_sha256"],
                "execution_arguments_sha256": HASH_A,
                "controller_pid": controller.pid,
                "controller_start_ticks": controller_ticks,
                "controller_command_sha256": controller_command_hash,
                "controller_process_group_id": controller_group,
                "controller_session_id": controller_session,
                "registered_at": datetime.now(UTC).isoformat(),
            }
            receipt = {
                **receipt_payload,
                "manifest_sha256": canonical_sha256(receipt_payload),
            }
            fallback_acceptance_module._write_append_only_json(
                fallback_acceptance_module._controller_receipt_path(options, 1),
                receipt,
            )
            monkeypatch.setattr(
                fallback_acceptance_module,
                "_internal_controller_command",
                lambda options, *, stage, output, guard: controller_command,
            )

            _takeover, outcomes = fallback_acceptance_module._revoke_controller_authority(
                options,
                paths=paths,
                invocation=invocation,
                ticket=ticket,
                trigger="hard_stop_deadline",
            )
            controller.wait(timeout=5)
            assert outcomes[0]["bound_controller_allocation_absent"] is True
            assert ledger.unresolved_gpu_allocations()

            class EndpointAbsentClient:
                def endpoint_live(self, timeout_seconds: float) -> bool:
                    return False

            recovering = VLLMService(
                configuration=configuration,
                client=cast(object, EndpointAbsentClient()),
                meter=guardian_meter,
            )
            assert recovering.resume_live_service_lease(
                expected_session_id=options.run_id,
                expected_event_id=f"{options.run_id}-service-start-001",
                cleanup_only=True,
            )
            uptime = recovering.shutdown(shutdown_seconds=1)
            assert uptime is not None
            assert ledger.unresolved_gpu_allocations() == ()
            assert ledger.unresolved_gpu_service_journals() == ()

            service_record = ledger.get_gpu_service_session(f"{options.run_id}-service-start-001")
            assert service_record is not None
            recovered_call = next(
                event
                for event in ledger.gpu_events()
                if event.event_id == f"{options.run_id}-open-call"
            )
            assert recovered_call.event_kind is GpuEventKind.FAILURE
            assert recovered_call.ended_at == service_record.ended_at
            assert service_record.classified_event_microseconds == sum(
                event.allocated_microseconds for event in ledger.gpu_events()
            )
            assert (
                ledger.gpu_summary().total_allocated_microseconds
                == service_record.service_microseconds
            )
            service_details = json.loads(service_record.details_json)
            assert service_details["controller_lost_allocation_event_ids"] == [
                f"{options.run_id}-open-call"
            ]
            assert datetime.fromisoformat(
                service_details["shared_terminal_recovery_at"]
            ) == datetime.fromisoformat(service_record.ended_at)

            before_replay = ledger.gpu_summary()
            AllocatedGPUMeter(ledger)
            assert ledger.gpu_summary() == before_replay
        finally:
            if controller is not None and controller.poll() is None:
                with suppress(ProcessLookupError):
                    os.killpg(controller.pid, signal.SIGKILL)
                with suppress(subprocess.TimeoutExpired):
                    controller.wait(timeout=5)
            if target_pid is not None:
                with suppress(ProcessLookupError):
                    os.killpg(target_pid, signal.SIGKILL)


def test_cleanup_orphan_still_shuts_down_when_adoption_raises(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    configuration = _fallback_launch_configuration(tmp_path)
    live: dict[str, object] = {}
    outputs = _fallback_outputs(trigger_repair=False)
    with Ledger(tmp_path / "ledger.sqlite3") as ledger:
        service = FakeFallbackService(configuration, ledger, outputs, live)
        runner = _runner(tmp_path=tmp_path, ledger=ledger, service=service)
        monkeypatch.setattr(os, "getpid", lambda: 61001)
        runner.prepare_controller_restart()

        def fail_adoption(*args: object, **kwargs: object) -> bool:
            del args, kwargs
            raise RuntimeError("adoption failed")

        monkeypatch.setattr(service, "resume_from_checkpoint", fail_adoption)
        cleanup = runner.cleanup_orphan()

        assert cleanup["physical_shutdown_verified"] is True
        assert cleanup["checkpoint_service_adopted"] is False
        assert cleanup["checkpoint_service_adoption_failed"] is True
        assert cleanup["checkpoint_service_adoption_failure_type"] == "RuntimeError"
        assert "adoption failed" not in json.dumps(cleanup)
        assert live["running"] is False
        checkpoint = json.loads(runner.checkpoint_path.read_text(encoding="utf-8"))
        assert checkpoint["service_adoption_failure_type"] == "RuntimeError"
