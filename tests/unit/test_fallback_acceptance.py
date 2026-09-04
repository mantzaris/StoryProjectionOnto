from __future__ import annotations

import copy
import hashlib
import importlib
import json
import os
import subprocess
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import cast

import pytest

from story_projection_onto.contracts import (
    ConditionName,
    OutputBudgets,
    PrequeryPreparationBinding,
    QueryAccessEvent,
    RunOutcome,
    canonical_sha256,
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
from story_projection_onto.experiment import ResourceLimits
from story_projection_onto.fallback_acceptance import (
    AMENDED_FALLBACK_STARTUP_WATCHDOG_SECONDS,
    REPAIR_TRIGGER_RULE,
    DevelopmentAdopterRegistration,
    DevelopmentContinuationBootstrap,
    DevelopmentContinuationHandoff,
    FallbackAcceptanceRunner,
    FallbackCapabilityBehaviorError,
    PreparedDevelopmentContinuation,
    build_fallback_acceptance_request,
    build_fallback_repair_request,
    fallback_pilot_calls,
    fallback_plan_manifest,
    main,
    parse_arguments,
    pre_fallback_gpu_accounting_baseline,
    validate_fallback_service_retry_amendment,
    validate_pre_fallback_gpu_accounting,
    validate_source_association,
)
from story_projection_onto.gpu_runtime import (
    FALLBACK_MODEL_REPOSITORY,
    FALLBACK_MODEL_REVISION,
    FALLBACK_SERVED_MODEL_NAME,
    GenerationResult,
    GuidedJSONRequest,
    ServiceState,
    ServiceUptime,
    TokenizerManifest,
    VLLMLaunchConfiguration,
)
from story_projection_onto.manifest import build_source_manifest
from story_projection_onto.model_gate import FallbackModelPolicy
from story_projection_onto.phase1_acceptance import validate_acceptance_generation
from story_projection_onto.public_release import scan_public_bytes
from story_projection_onto.store import (
    ArtifactStore,
    BlobStore,
    GpuEventKind,
    GpuSummary,
    Ledger,
)

ROOT = Path(__file__).resolve().parents[2]
HASH_A = "a" * 64
HASH_B = "b" * 64


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
    assert plan["normal_exit_requires_verified_shutdown"] is True
    assert plan["pre_fallback_ledger_must_exactly_reproduce_primary_result"] is True
    assert plan["fresh_gpu_ledger_forbidden"] is True
    assert plan["normal_acceptance_block"]["executed"] is False
    continuation = plan["development_continuation"]
    assert continuation["adopter_protocol"] == "fallback-live-development-adopter-v1"
    assert continuation["selected_model_freeze_bootstrap_requires_development_completion"] is False
    assert continuation["adopter_model_load_start_shutdown_counts"] == [0, 0, 0]
    assert continuation["fallback_owner_performs_final_shutdown"] is True
    assert continuation["callback_returns"] == "canonical DevelopmentExecutionResult"
    assert continuation["fallback_owner_derives_continuation_receipt"] is True
    assert continuation["micro_pilot_c1_operator_scope"] == (
        "representative_grounded_subset"
    )
    assert continuation["complete_c1_operator_inventory_gate"] == (
        "DevelopmentScientificAssessment.c1_all_construction_operators_exercised"
    )
    assert continuation["standalone_cli_gpu_execution_enabled"] is True
    assert continuation["standalone_registered_factory"] == (
        "story_projection_onto.development_continuation:"
        "create_production_development_adopter"
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
        "src/story_projection_onto/conditions/c0.py",
        "src/story_projection_onto/conditions/c1.py",
        "src/story_projection_onto/conditions/c2.py",
        "src/story_projection_onto/conditions/fixed_select.py",
    }.issubset(implementation_paths)
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
    base = build_fallback_acceptance_request(
        root=ROOT,
        call=call,
        tokenizer=tokenizer,
        tokenizer_manifest=tokenizer_manifest,
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
    )

    assert base.model_name == repair.model_name == FALLBACK_SERVED_MODEL_NAME
    assert "fallback_capability_probe" in base.packing.required_section_names
    assert repair.decoding.decoding_pass.value == "repair"
    assert repair.packing.truncation_applied is False
    assert repair.packing.complete_evidence_packet is True
    assert {"invalid_draft", "validation_diagnostics"}.issubset(
        repair.packing.required_section_names
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

    with pytest.raises(SystemExit):
        parse_arguments(["--output", "invalid.json", "--execute", "--validate-only"])


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
        ROOT
        / "artifacts/public/results/"
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
    output = json.loads((ROOT / "tests/fixtures/phase1/c2_query_output.json").read_text())
    audit = validate_acceptance_generation(
        root=ROOT,
        call=call.acceptance_call(),
        parsed_object=output,
    )
    behavior = _require_call_operator_coverage(call, audit, output, root=ROOT)
    assert behavior["behaviorally_valid"] is True

    spoofed = copy.deepcopy(output)
    split = next(
        decision
        for decision in spoofed["decisions"]
        if decision["operator"] == "split"
    )
    split["created_object_ids"] = split["created_object_ids"][:1]
    spoofed_audit = validate_acceptance_generation(
        root=ROOT,
        call=call.acceptance_call(),
        parsed_object=spoofed,
    )
    with pytest.raises(FallbackCapabilityBehaviorError, match="split"):
        _require_call_operator_coverage(call, spoofed_audit, spoofed, root=ROOT)


@dataclass
class FakeResourceSampler:
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
        self.remaining_required_seconds.append(
            cast(float, kwargs["remaining_required_seconds"])
        )
        self.start_count += 1
        self._record_event(
            event_id=event_id,
            event_kind=GpuEventKind.GPU_SESSION_START,
            seconds=5,
        )
        self.live.update({"running": True, "pid": self.pid})
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
        self.remaining_required_seconds.append(
            cast(float, kwargs["remaining_required_seconds"])
        )
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
        self.remaining_required_seconds.append(
            cast(float, kwargs["remaining_required_seconds"])
        )
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

    def shutdown(self) -> ServiceUptime | None:
        self.shutdown_count += 1
        if self.live.get("running") is not True:
            self.state = ServiceState.STOPPED
            return None
        self.live["running"] = False
        self.state = ServiceState.STOPPED
        seconds = self.ledger.gpu_summary().total_allocated_seconds
        start = datetime(2026, 9, 3, tzinfo=UTC)
        return ServiceUptime(
            session_id="fallback-test",
            started_at=start,
            ended_at=start + timedelta(seconds=seconds),
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
        prequery = next(
            call.prequery_stage for call in manifest.calls if call.unit_id == unit_id
        )
        evidence = json.loads(
            (ROOT / prequery.relative_path / "evidence.json").read_text(encoding="utf-8")
        )
        snapshot = evidence["snapshot"]
        preparations = tuple(
            PrequeryPreparationBinding(
                unit_id=snapshot["world_or_window_id"],
                condition=condition,
                seed_block=(
                    None
                    if condition is ConditionName.C0_CLASSICAL_PRE
                    else manifest.seed_block
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
            gpu_call_inventory_file_sha256=(
                manifest.gpu_call_inventory_file_sha256
            ),
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
                        lineage_artifact_hash=digest(
                            f"{call.call_id}:{condition.value}:lineage"
                        ),
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
                    source_c1_construction_seal_hash=digest(
                        f"{call.source_c1_call_id}:seal"
                    ),
                    source_c1_preparation_hash=digest(
                        f"{call.source_c1_call_id}:preparation"
                    ),
                    fixed_ontology_hash=digest(f"{call.call_id}:fixed-ontology"),
                    evidence_alias_bijection_hash=digest(f"{call.call_id}:aliases"),
                    derived_output_schema_hash=digest(f"{call.call_id}:schema"),
                    derived_decoding_manifest_hash=digest(
                        f"{call.call_id}:decoding"
                    ),
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
                    query_access_event_hash=(
                        None if access is None else access.content_hash
                    ),
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
    snapshot = (
        cache
        / "hub"
        / "models--Qwen--Qwen3-8B-AWQ"
        / "snapshots"
        / FALLBACK_MODEL_REVISION
    )
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
        run_id="fallback-unit",
        service=cast(object, service),
        ledger=ledger,
        artifacts=ArtifactStore(BlobStore(tmp_path / "blobs"), ledger),
        resource_sampler=cast(object, FakeResourceSampler()),
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


def _fallback_outputs(*, trigger_repair: bool) -> dict[str, Mapping[str, object]]:
    accepted_c1 = json.loads(
        (ROOT / "tests/fixtures/phase1/c1_pre_output.json").read_text()
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
        "fallback-c2-01": json.loads(
            (ROOT / "tests/fixtures/phase1/c2_query_output.json").read_text()
        ),
        "fallback-c2-02": json.loads(
            (ROOT / "tests/fixtures/phase1/c2_query_output_2.json").read_text()
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
            "integrated_development_scientific_assessment."
            "c1_all_construction_operators_exercised"
        )
        assert result["operator_coverage_gate"]["c2_complete"] is True
        assert result["actual_plus_remaining_forecast"]["admitted"] is True
        assert result["normal_acceptance_block_executed"] is False
        assert result["selected_model_freeze"]["scope"] == (
            "registered_development_llm_conditions"
        )
        assert result["selected_model_freeze"]["held_out_execution_allowed"] is False
        assert result["development_continuation_gate"]["passed"] is True
        assert result["development_continuation_receipt"]["terminal_call_count"] == 24
        assert result["development_continuation_bootstrap"][
            "selected_model_freeze_hash"
        ] == result["selected_model_freeze"]["manifest_sha256"]
        assert result["development_preparation"]["query_access_event_count"] == 0
        assert result["development_handoff"]["live_service_identity"][
            "service_pid"
        ] == first_service.pid
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
        assert sum(
            event.event_kind is GpuEventKind.GPU_SESSION_START
            for event in ledger.gpu_events()
        ) == 1
        assert sum(
            event.event_kind is GpuEventKind.FALLBACK_TEST
            for event in ledger.gpu_events()
        ) == 4
        assert sum(
            event.event_kind is GpuEventKind.REPAIR
            for event in ledger.gpu_events()
        ) == int(trigger_repair)
        assert len(result["reserve_consumption"]) == 4 + int(trigger_repair)
        assert len(second_service.remaining_required_seconds) == 4 + int(
            trigger_repair
        )
        assert min(second_service.remaining_required_seconds) > 29_000
        assert live["running"] is False

        state = json.loads((tmp_path / "fallback.checkpoint.json").read_text())
        resumed, _, timings = second_runner._resume_completed(
            state=state,
            call=fallback_pilot_calls(
                FallbackModelPolicy.load(ROOT / "configs/study/fallback_model.json")
            )[0],
        )
        assert resumed["status"] == "resumed"
        assert len(timings) == 1 + int(trigger_repair)
        if trigger_repair:
            assert resumed["base_attempt"]["packing_report"][
                "complete_evidence_snapshot"
            ] is True


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
            assert failure["partial_development_checkpoint"][
                "canonical_checkpoint_valid"
            ] is False


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
        assert failure["actual_plus_remaining_forecast"][
            "actual_allocated_seconds"
        ] > 0
        assert failure["vllm_service_stopped"] is True
        assert failure["physical_service_state_unverified"] is False
        assert live["running"] is False


def test_failed_diagnostic_repair_retains_repair_timing_and_reserve(
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
        assert result["timing_gate"]["repair_sample_count"] == 1
        assert result["timing_gate"]["repair_sample_count_valid"] is True
        assert sum(
            event.event_kind is GpuEventKind.REPAIR for event in ledger.gpu_events()
        ) == 1
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


def test_orchestrator_supervises_two_controllers_and_closes_guard(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from story_projection_onto.fallback_acceptance import (
        _orchestrate_controller_processes,
    )

    options = _orchestrator_options(tmp_path)
    stages: list[str] = []

    def run(command: tuple[str, ...], *, check: bool) -> subprocess.CompletedProcess[str]:
        assert check is False
        stage = command[command.index("--controller-stage") + 1]
        output = Path(command[command.index("--output") + 1])
        stages.append(stage)
        if stage == "prepare":
            output.write_text(
                json.dumps(
                    {
                        "run_id": options.run_id,
                        "model_service_left_live_for_controller_restart": True,
                        "physical_service_live": True,
                        "vllm_service_stopped": False,
                    }
                ),
                encoding="utf-8",
            )
        elif stage == "run":
            output.write_text(
                json.dumps(
                    {
                        "micro_pilot_passed": True,
                        "phase1_gate_passed": False,
                        "physical_service_live": False,
                        "vllm_service_stopped": True,
                    }
                ),
                encoding="utf-8",
            )
        return subprocess.CompletedProcess(command, 2 if stage == "run" else 0)

    monkeypatch.setattr(subprocess, "run", run)
    assert _orchestrate_controller_processes(options) == 2
    assert stages == ["prepare", "run"]
    guard = json.loads(
        (tmp_path / "checkpoint.orchestrator-guard.json").read_text()
    )
    assert guard["state"] == "closed"
    assert guard["physical_shutdown_verified"] is True


def test_orchestrator_invokes_cleanup_when_run_does_not_verify_shutdown(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from story_projection_onto.fallback_acceptance import (
        _orchestrate_controller_processes,
    )

    options = _orchestrator_options(tmp_path)
    stages: list[str] = []

    def run(command: tuple[str, ...], *, check: bool) -> subprocess.CompletedProcess[str]:
        assert check is False
        stage = command[command.index("--controller-stage") + 1]
        output = Path(command[command.index("--output") + 1])
        stages.append(stage)
        if stage == "prepare":
            output.write_text(
                json.dumps(
                    {
                        "run_id": options.run_id,
                        "model_service_left_live_for_controller_restart": True,
                        "physical_service_live": True,
                        "vllm_service_stopped": False,
                    }
                ),
                encoding="utf-8",
            )
            Path(str(options.checkpoint) + ".service").write_text("{}")
        elif stage == "run":
            output.write_text(
                json.dumps(
                    {
                        "physical_service_live": None,
                        "vllm_service_stopped": False,
                    }
                ),
                encoding="utf-8",
            )
        else:
            output.write_text(
                json.dumps({"physical_shutdown_verified": True}),
                encoding="utf-8",
            )
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(subprocess, "run", run)
    with pytest.raises(RuntimeError, match="verified service shutdown"):
        _orchestrate_controller_processes(options)
    assert stages == ["prepare", "run", "cleanup"]


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
        with pytest.raises(RuntimeError, match="adoption failed"):
            runner.cleanup_orphan()
        assert live["running"] is False
