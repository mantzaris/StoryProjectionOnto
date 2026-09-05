from __future__ import annotations

import hashlib
from datetime import datetime, timedelta
from pathlib import Path
from types import SimpleNamespace

import pytest

from story_projection_onto.contracts import (
    BudgetAccounting,
    ConditionName,
    ConstructionOperator,
    ConstructionSeal,
    FeedbackAction,
    InstanceGraph,
    LocalContextSchema,
    OmissionRecord,
    OntologyDecision,
    OntologyProjection,
    OutputBudgets,
    ReleaseClass,
    RevelationPosition,
    SpoilerHorizon,
    UpperOntology,
    canonical_sha256,
    projection_validation_target_hash,
    runtime_structural_acceptance_record,
)
from story_projection_onto.feedback_provenance import ResearcherTraceSubmissionReceipt
from story_projection_onto.feedback_runtime import (
    FeedbackAttemptStatus,
    FeedbackEpisodeKind,
    FeedbackLedgerCallReference,
    FeedbackRegenerationReceipt,
    ScriptedRevisionFreeze,
    load_feedback_protocol,
)
from story_projection_onto.phase5_execution import (
    C2RegenerationRequest,
    C2RegenerationResult,
    CpuReprojectionInput,
    InterruptedFeedbackCallRecoveryRequired,
    Phase5CpuLedgerMaterializationError,
    Phase5ExecutionError,
    Phase5ExecutionInputManifest,
    Phase5PrerequisiteError,
    PrimaryHeldOutResultsGate,
    ResearcherTraceInput,
    ScorerOnlyFeedbackBinding,
    ScriptedFeedbackInput,
    SQLiteFeedbackLedgerVerifier,
    _append_exact,
    execute_phase5,
    feedback_gpu_event_record_hash,
    feedback_model_call_record_hash,
    load_phase5_input_manifest,
    load_phase5_runner_configuration,
    materialize_phase5_cpu_ledger,
    phase5_cpu_receiving_job_identity,
    validate_phase5_inputs,
    validate_phase5_ledger_preflight,
    validate_phase5_prerequisites,
)
from story_projection_onto.store import (
    ArtifactRecord,
    ArtifactStore,
    AttemptKind,
    BlobStore,
    CommitmentCheckStatus,
    Compression,
    EvidenceSupportStatus,
    GpuEventKind,
    InputKind,
    JobState,
    Ledger,
    ModelBackend,
    ModelCallRole,
    RetryClass,
    SemanticAssessmentScope,
    TemporalValidationStatus,
    ValidationStatus,
)
from story_projection_onto.ui import (
    MergeSplitIntent,
    MergeSplitOperation,
    RefineContextIntent,
    RevisionDraftSubmission,
    VisualizationContentScope,
    VisualizationTemporalFilter,
    build_merge_split_instruction,
    build_refine_context_instruction,
    build_visualization_bundle,
    filter_visualization_bundle,
)
from tests.unit.test_ui import NOW, context, digest, feedback_anchor, packet, projection


def _without_hashes(value):
    if isinstance(value, dict):
        return {key: _without_hashes(item) for key, item in value.items() if key != "content_hash"}
    if isinstance(value, (list, tuple)):
        return [_without_hashes(item) for item in value]
    return value


def _context(context_id: str):
    values = _without_hashes(context().model_dump(mode="python"))
    values["context_id"] = context_id
    return context().__class__.model_validate(values)


def _rebind_prequery_validation(
    payload: dict,
    *,
    validated_at: datetime,
) -> None:
    condition = ConditionName(payload["condition"])
    if condition not in {ConditionName.C0_CLASSICAL_PRE, ConditionName.C1_LLM_PRE}:
        raise ValueError("test helper only rebinds pre-query projections")
    upper_ontology = UpperOntology.model_validate(payload["upper_ontology"])
    local_schema = LocalContextSchema.model_validate(payload["local_schema"])
    instance_graph = InstanceGraph.model_validate(payload["instance_graph"])
    decisions = tuple(OntologyDecision.model_validate(item) for item in payload["decisions"])
    omissions = tuple(OmissionRecord.model_validate(item) for item in payload["omissions"])
    budget_accounting = BudgetAccounting.model_validate(payload["budget_accounting"])
    budgets = OutputBudgets.model_validate(payload["budgets"])
    target_hash = projection_validation_target_hash(
        condition=condition,
        snapshot_hash=payload["snapshot_hash"],
        packet_hash=payload["packet_hash"],
        context_hash=payload["context_hash"],
        upper_ontology=upper_ontology,
        local_schema=local_schema,
        instance_graph=instance_graph,
        decisions=decisions,
        omissions=omissions,
        budget_accounting=budget_accounting,
        budgets=budgets,
    )
    records = (
        runtime_structural_acceptance_record(
            validation_id=f"validation-{target_hash[:32]}",
            target_id=target_hash,
            validated_at=validated_at,
        ),
    )
    payload["validation_records"] = records
    payload["validation_bundle_hash"] = (
        canonical_sha256(records) if condition is ConditionName.C1_LLM_PRE else None
    )


def _c1_projection(query_context) -> OntologyProjection:
    base = projection(ConditionName.C0_CLASSICAL_PRE, query_context, packet())
    values = _without_hashes(base.model_dump(mode="python"))
    values.update(
        projection_id=f"projection-c1-{query_context.context_id}",
        condition=ConditionName.C1_LLM_PRE.value,
        run_id=f"run-c1-{query_context.context_id}",
        generation_lineage_hash=digest("c1-generation"),
        raw_output_artifact_hash=digest("c1-raw"),
        normalized_draft_hash=digest("c1-normalized"),
        validation_bundle_hash=canonical_sha256(base.validation_records),
    )
    seal = _without_hashes(base.construction_seal.model_dump(mode="python"))
    graph = base.instance_graph
    seal.update(
        seal_id="seal-c1-shared",
        condition=ConditionName.C1_LLM_PRE.value,
        sealed_object_ids=(
            *(item.entity_id for item in graph.entities),
            *(item.event_id for item in graph.events),
            *(item.proposition_content_id for item in graph.proposition_contents),
            *(item.assertion_id for item in graph.assertions),
        ),
    )
    values["construction_seal"] = ConstructionSeal.model_validate(seal)
    _rebind_prequery_validation(values, validated_at=NOW + timedelta(seconds=3))
    return OntologyProjection.model_validate(values)


def _instruction(episode_id: str, context_id: str, action: FeedbackAction):
    before = _context(context_id)
    created = NOW + timedelta(seconds=60)
    if action is FeedbackAction.REFINE_CONTEXT:
        return build_refine_context_instruction(
            revision_id=f"revision-{episode_id}",
            before_context=before,
            intent=RefineContextIntent(
                after_context_id=f"{context_id}-refined",
                after_revealed_at=created,
                lens="refined evidence-grounded causal lens",
            ),
            anchors=(feedback_anchor("ev-a", "m-a"),),
            rationale="Apply the frozen test-only contextual refinement.",
            sequence=1,
            created_at=created,
        )
    return build_merge_split_instruction(
        revision_id=f"revision-{episode_id}",
        context=before,
        intent=MergeSplitIntent(
            operation=MergeSplitOperation.MERGE,
            grouped_mention_candidate_ids=(("m-a",), ("m-b",)),
        ),
        anchors=(feedback_anchor("ev-a", "m-a"), feedback_anchor("ev-b", "m-b")),
        rationale="Apply the frozen test-only identity merge.",
        sequence=1,
        created_at=created,
    )


def _cpu(condition: ConditionName, instruction):
    before = (
        projection(condition, instruction.before_context, packet())
        if condition is ConditionName.C0_CLASSICAL_PRE
        else _c1_projection(instruction.before_context)
    )
    if condition is ConditionName.C0_CLASSICAL_PRE:
        before = _reseal_c0(before)
    after = None
    if instruction.revision.action is FeedbackAction.REFINE_CONTEXT:
        after = (
            projection(condition, instruction.after_context, packet())
            if condition is ConditionName.C0_CLASSICAL_PRE
            else _c1_projection(instruction.after_context)
        )
        if condition is ConditionName.C0_CLASSICAL_PRE:
            after = _reseal_c0(after)
        after = _selection_only(after, instruction)
    return CpuReprojectionInput(
        condition=condition,
        before_projection=before,
        after_projection=after,
        instruction_hash=instruction.content_hash,
        source_query_stage_hash=digest(
            f"source-query-stage-{instruction.before_context.context_id}"
        ),
        source_preparation_hash=digest(
            f"source-preparation-{condition.value}-{instruction.before_context.context_id}"
        ),
        resolver_hash=digest(f"resolver-{condition.value}"),
        started_at=instruction.revision.created_at + timedelta(seconds=1),
        completed_at=instruction.revision.created_at + timedelta(seconds=2),
        resolved_at=instruction.revision.created_at + timedelta(seconds=3),
        checked_at=instruction.revision.created_at + timedelta(seconds=4),
    )


def _reseal_c0(value: OntologyProjection) -> OntologyProjection:
    payload = _without_hashes(value.model_dump(mode="python"))
    graph = value.instance_graph
    seal = _without_hashes(value.construction_seal.model_dump(mode="python"))
    seal["sealed_object_ids"] = (
        *(item.entity_id for item in graph.entities),
        *(item.event_id for item in graph.events),
        *(item.proposition_content_id for item in graph.proposition_contents),
        *(item.assertion_id for item in graph.assertions),
    )
    payload["construction_seal"] = ConstructionSeal.model_validate(seal)
    return OntologyProjection.model_validate(payload)


def _selection_only(value: OntologyProjection, instruction) -> OntologyProjection:
    payload = _without_hashes(value.model_dump(mode="python"))
    graph = value.instance_graph
    selected_ids = (
        *(item.entity_id for item in graph.entities),
        *(item.event_id for item in graph.events),
        *(item.proposition_content_id for item in graph.proposition_contents),
        *(item.assertion_id for item in graph.assertions),
    )
    payload["decisions"] = (
        OntologyDecision(
            decision_id=(
                f"selection-{value.condition.value.lower()}-"
                f"{instruction.after_context.context_id}"
            ),
            operator=ConstructionOperator.SELECTION,
            evidence_ids=(packet().ordered_evidence_ids[0],),
            rationale="Test-only fixed same-seal selection decision.",
            decided_at=instruction.revision.created_at + timedelta(seconds=1),
            input_object_ids=selected_ids,
        ),
    )
    _rebind_prequery_validation(
        payload,
        validated_at=instruction.revision.created_at + timedelta(seconds=2),
    )
    return OntologyProjection.model_validate(payload)


def _inputs():
    protocol = load_feedback_protocol()
    scripts = []
    for selection in protocol.scripted_episodes:
        instruction = _instruction(selection.episode_id, selection.context_id, selection.action)
        bindings = tuple(
            ScorerOnlyFeedbackBinding(
                episode_id=selection.episode_id,
                condition=condition,
                scorer_binding_key=selection.scorer_binding_key,
                before_gold_projection_hash=digest(f"before-{selection.episode_id}"),
                after_gold_projection_hash=digest(f"after-{selection.episode_id}"),
                required_target_change_hash=digest(f"target-{selection.episode_id}"),
            )
            for condition in (
                ConditionName.C0_CLASSICAL_PRE,
                ConditionName.C1_LLM_PRE,
                ConditionName.C2_LLM_QUERY,
            )
        )
        scripts.append(
            ScriptedFeedbackInput(
                episode_id=selection.episode_id,
                instruction=instruction,
                packet=packet(),
                freeze=ScriptedRevisionFreeze(
                    episode_id=selection.episode_id,
                    context_id=selection.context_id,
                    action=selection.action,
                    instruction_hash=instruction.content_hash,
                    anchor_hashes=tuple(item.content_hash for item in instruction.revision.anchors),
                    frozen_at=instruction.revision.created_at - timedelta(seconds=1),
                ),
                cpu_inputs=tuple(
                    _cpu(condition, instruction)
                    for condition in (
                        ConditionName.C0_CLASSICAL_PRE,
                        ConditionName.C1_LLM_PRE,
                    )
                ),
                c2_before_projection=projection(
                    ConditionName.C2_LLM_QUERY, instruction.before_context, packet()
                ),
                scorer_bindings=bindings,
            )
        )
    traces = []
    for index, slot in enumerate(protocol.researcher_trace_slots):
        action = slot.allowed_actions[index % 2]
        instruction = _instruction(slot.episode_id, slot.context_id, action)
        before = projection(
            ConditionName.C2_LLM_QUERY, instruction.before_context, packet()
        )
        before_bundle = build_visualization_bundle(
            before,
            instruction.before_context,
            packet(),
            content_scope=VisualizationContentScope.REGISTERED_DISPLAY,
        )
        submission_values = dict(
            before_projection_id=before.projection_id,
            action=instruction.revision.action,
            anchors=instruction.revision.anchors,
            rationale=instruction.revision.rationale,
            sequence=instruction.revision.sequence,
            seed=protocol.llm_seed,
        )
        if instruction.refine_context_intent is not None:
            intent = instruction.refine_context_intent
            submission_values.update(
                lens=intent.lens,
                story_scope=intent.story_scope,
                spoiler_horizon=intent.spoiler_horizon,
            )
        else:
            assert instruction.merge_split_intent is not None
            intent = instruction.merge_split_intent
            submission_values.update(
                merge_split_operation=intent.operation,
                grouped_mention_candidate_ids=intent.grouped_mention_candidate_ids,
            )
        submission = RevisionDraftSubmission(**submission_values)
        submission_json = submission.to_canonical_json()
        instruction_bytes = (instruction.to_canonical_json() + "\n").encode("utf-8")
        traces.append(
            ResearcherTraceInput(
                episode_id=slot.episode_id,
                instruction=instruction,
                packet=packet(),
                c2_before_projection=before,
                submission_receipt=ResearcherTraceSubmissionReceipt(
                    receipt_id=f"TEST-ONLY-receipt-{slot.episode_id}",
                    episode_id=slot.episode_id,
                    action=instruction.revision.action,
                    requested_at=instruction.revision.created_at,
                    before_projection_id=before.projection_id,
                    before_projection_hash=before.content_hash,
                    before_bundle_hash=before_bundle.content_hash,
                    submission_hash=submission.content_hash,
                    submission_canonical_json=submission_json,
                    submission_file_sha256=hashlib.sha256(
                        (submission_json + "\n").encode("utf-8")
                    ).hexdigest(),
                    instruction_hash=instruction.content_hash,
                    instruction_file_sha256=hashlib.sha256(instruction_bytes).hexdigest(),
                    recorded_at=instruction.revision.created_at,
                ),
            )
        )
    gate = PrimaryHeldOutResultsGate(
        gate_id="TEST-ONLY-primary-results",
        lifecycle_state="held_out_primary_results_frozen",
        benchmark_draft_seal_hash=digest("draft"),
        final_reviewed_seal_hash=digest("final-review"),
        held_out_execution_manifest_relative_path="held_out_execution_manifest.json",
        held_out_execution_manifest_hash=digest("heldout-execution"),
        held_out_execution_manifest_file_sha256=digest("heldout-execution-file"),
        scorer_bridge_relative_path="scorer_bridge.json",
        scorer_bridge_hash=digest("scorer-bridge"),
        scorer_bridge_file_sha256=digest("scorer-bridge-file"),
        result_artifact_hashes=(digest("heldout-results"),),
        conditions=(
            ConditionName.C0_CLASSICAL_PRE,
            ConditionName.C1_LLM_PRE,
            ConditionName.C2_LLM_QUERY,
            ConditionName.A_FIXED_SELECT,
        ),
        frozen_at=NOW + timedelta(seconds=50),
    )
    inputs = Phase5ExecutionInputManifest(
        run_id="TEST-ONLY-phase5-run",
        protocol_hash=protocol.content_hash,
        script_commitment_manifest_hash=digest("phase5-script-commitment"),
        primary_results_gate_hash=gate.content_hash,
        final_reviewed_seal_hash=gate.final_reviewed_seal_hash,
        source_manifest_hash=digest("phase5-source-manifest"),
        held_out_execution_manifest_hash=gate.held_out_execution_manifest_hash,
        held_out_call_manifest_hash=digest("held-out-call-manifest"),
        scorer_binding_authorization_hash=digest("scorer-binding-authorization"),
        frozen_at=NOW + timedelta(seconds=60),
        scripted_inputs=tuple(scripts),
        researcher_trace_inputs=tuple(traces),
    )
    return protocol, gate, inputs


class _Adapter:
    def __init__(self):
        self.calls = []
        self.results = {}

    @staticmethod
    def _ledger_call(
        request: C2RegenerationRequest,
        *,
        started,
        completed,
        successful: bool,
    ) -> FeedbackLedgerCallReference:
        return FeedbackLedgerCallReference(
            model_call_id=f"model-call-{request.episode_id}",
            model_call_record_hash=digest(f"model-call-row-{request.call_slot_id}"),
            gpu_event_id=f"gpu-event-{request.episode_id}",
            gpu_event_record_hash=digest(f"gpu-event-row-{request.call_slot_id}"),
            attempt_id=f"attempt-{request.episode_id}",
            attempt_kind="base",
            call_role="query_time",
            retry_class="standard",
            request_hash=request.content_hash,
            allocated_gpu_seconds=6.0,
            started_at=started,
            completed_at=completed,
            successful=successful,
        )

    def regenerate(self, request: C2RegenerationRequest) -> C2RegenerationResult:
        self.calls.append(request.call_slot_id)
        instruction = request.instruction
        after = projection(
            ConditionName.C2_LLM_QUERY,
            instruction.after_context,
            request.packet,
            merged=instruction.revision.action is FeedbackAction.REQUEST_MERGE_SPLIT,
            decision_offset_seconds=65,
        )
        started = instruction.revision.created_at + timedelta(seconds=1)
        completed = instruction.revision.created_at + timedelta(seconds=7)
        result = C2RegenerationResult(
            request_hash=request.content_hash,
            after_projection=after,
            receipt=FeedbackRegenerationReceipt(
                ledger_calls=(
                    self._ledger_call(
                        request,
                        started=started,
                        completed=completed,
                        successful=True,
                    ),
                ),
                attempt_status=FeedbackAttemptStatus.SUCCEEDED,
                instruction_hash=instruction.content_hash,
                before_projection_hash=request.before_projection_hash,
                after_projection_hash=after.content_hash,
                seed=request.seed,
                allocated_gpu_seconds=6.0,
                started_at=started,
                completed_at=completed,
            ),
            resolver_hash=digest(f"c2-resolver-{request.episode_id}"),
            resolved_at=instruction.revision.created_at + timedelta(seconds=8),
            checked_at=instruction.revision.created_at + timedelta(seconds=9),
            latency_seconds=7.0,
        )
        self.results[request.content_hash] = result
        return result

    def recover(self, request: C2RegenerationRequest) -> C2RegenerationResult | None:
        return self.results.get(request.content_hash)


class _Verifier:
    def __init__(self):
        self.verified = []

    def verify(self, request, receipt) -> None:
        assert receipt.ledger_calls[0].request_hash == request.content_hash
        self.verified.append(request.content_hash)


class _TimeoutFirstAdapter(_Adapter):
    def regenerate(self, request: C2RegenerationRequest) -> C2RegenerationResult:
        if not self.calls:
            self.calls.append(request.call_slot_id)
            instruction = request.instruction
            started = instruction.revision.created_at + timedelta(seconds=1)
            completed = instruction.revision.created_at + timedelta(seconds=7)
            result = C2RegenerationResult(
                request_hash=request.content_hash,
                receipt=FeedbackRegenerationReceipt(
                    ledger_calls=(
                        self._ledger_call(
                            request,
                            started=started,
                            completed=completed,
                            successful=False,
                        ),
                    ),
                    attempt_status=FeedbackAttemptStatus.TIMED_OUT,
                    instruction_hash=instruction.content_hash,
                    before_projection_hash=request.before_projection_hash,
                    failure_artifact_hash=digest(f"failure-{request.episode_id}"),
                    seed=request.seed,
                    allocated_gpu_seconds=6.0,
                    started_at=started,
                    completed_at=completed,
                ),
                resolver_hash=digest(f"c2-resolver-{request.episode_id}"),
                resolved_at=instruction.revision.created_at + timedelta(seconds=8),
                checked_at=instruction.revision.created_at + timedelta(seconds=9),
                latency_seconds=7.0,
            )
            self.results[request.content_hash] = result
            return result
        return super().regenerate(request)


class _CrashBeforeResultAdapter(_Adapter):
    def regenerate(self, request: C2RegenerationRequest) -> C2RegenerationResult:
        self.calls.append(request.call_slot_id)
        raise RuntimeError("TEST-ONLY simulated controller loss after durable slot")


class _AmbiguousRecoveryAdapter(_Adapter):
    def recover(self, request: C2RegenerationRequest) -> C2RegenerationResult | None:
        raise InterruptedFeedbackCallRecoveryRequired(
            f"durable Phase 5 slot {request.call_slot_id} has ambiguous adapter trace"
        )


def test_runner_configuration_binds_frozen_protocol() -> None:
    configuration = load_phase5_runner_configuration()
    assert configuration.feedback_protocol_hash == load_feedback_protocol().content_hash
    assert configuration.authoritative_materialization_receipt_required is True
    assert configuration.controller_owns_model_service_lifecycle is False


def test_production_prerequisite_gate_refuses_missing_review(tmp_path: Path) -> None:
    _protocol, gate, inputs = _inputs()
    gate_path = tmp_path / "gate.json"
    gate_path.write_text(gate.to_canonical_json())
    with pytest.raises(Phase5PrerequisiteError, match="independent review"):
        validate_phase5_prerequisites(
            inputs=inputs,
            primary_results_gate_path=gate_path,
            benchmark_root=Path("data/synthetic"),
            review_completion_root=tmp_path / "missing-review",
        )


def test_prerequisite_replay_parses_real_held_out_contracts(tmp_path: Path, monkeypatch) -> None:
    _protocol, original_gate, original_inputs = _inputs()
    invalid_bytes = b"{}"
    file_hash = hashlib.sha256(invalid_bytes).hexdigest()
    gate_payload = _without_hashes(original_gate.model_dump(mode="python"))
    gate_payload.update(
        held_out_execution_manifest_file_sha256=file_hash,
        scorer_bridge_file_sha256=file_hash,
    )
    gate = PrimaryHeldOutResultsGate.model_validate(gate_payload)
    inputs_payload = _without_hashes(original_inputs.model_dump(mode="python"))
    inputs_payload["primary_results_gate_hash"] = gate.content_hash
    inputs = Phase5ExecutionInputManifest.model_validate(inputs_payload)
    gate_path = tmp_path / "gate.json"
    gate_path.write_text(gate.to_canonical_json())
    (tmp_path / gate.held_out_execution_manifest_relative_path).write_bytes(invalid_bytes)
    (tmp_path / gate.scorer_bridge_relative_path).write_bytes(invalid_bytes)
    review = SimpleNamespace(
        draft=SimpleNamespace(content_hash=gate.benchmark_draft_seal_hash),
        final_seal=SimpleNamespace(content_hash=gate.final_reviewed_seal_hash),
    )
    monkeypatch.setattr(
        "story_projection_onto.phase5_execution.load_completed_review",
        lambda **_kwargs: review,
    )
    with pytest.raises(Phase5PrerequisiteError, match="HeldOutExecutionManifest"):
        validate_phase5_prerequisites(
            inputs=inputs,
            primary_results_gate_path=gate_path,
            benchmark_root=tmp_path,
            review_completion_root=tmp_path,
        )


def test_exact_inventory_executes_once_resumes_and_detects_tamper(tmp_path: Path) -> None:
    protocol, gate, inputs = _inputs()
    validate_phase5_inputs(inputs, protocol)
    adapter = _Adapter()
    verifier = _Verifier()
    root = tmp_path / "TEST-ONLY-phase5"
    finished = NOW + timedelta(minutes=5)
    first = execute_phase5(
        inputs=inputs,
        protocol=protocol,
        prerequisites=gate,
        adapter=adapter,
        ledger_verifier=verifier,
        output_root=root,
        completed_at=finished,
    )
    assert len(adapter.calls) == 9
    assert first.cpu_call_record_count == 12
    assert first.c2_regeneration_slot_count == 9
    assert first.actual_gpu_request_count == 9
    assert len(first.artifact_bindings) == 47
    assert first.scripted_metric_binding_count == 18
    second = execute_phase5(
        inputs=inputs,
        protocol=protocol,
        prerequisites=gate,
        adapter=adapter,
        ledger_verifier=verifier,
        output_root=root,
        completed_at=finished,
    )
    assert second.content_hash == first.content_hash
    assert len(adapter.calls) == 9
    record = next((root / "records").iterdir())
    record.write_text("{}")
    with pytest.raises(Phase5ExecutionError, match="invalid FeedbackExecutionRecord"):
        execute_phase5(
            inputs=inputs,
            protocol=protocol,
            prerequisites=gate,
            adapter=adapter,
            ledger_verifier=verifier,
            output_root=root,
            completed_at=finished,
        )


def test_artifact_store_requires_exact_ledger_preflight_before_adapter_call(
    tmp_path: Path,
) -> None:
    protocol, gate, inputs = _inputs()
    adapter = _Adapter()
    with Ledger(tmp_path / "ledger.sqlite3") as ledger:
        artifacts = ArtifactStore(
            BlobStore(tmp_path / "cas", compression=Compression.GZIP),
            ledger,
        )
        with pytest.raises(
            Phase5CpuLedgerMaterializationError,
            match="materialization failed closed",
        ):
            execute_phase5(
                inputs=inputs,
                protocol=protocol,
                prerequisites=gate,
                adapter=adapter,
                ledger_verifier=_Verifier(),
                output_root=tmp_path / "must-not-start",
                completed_at=NOW + timedelta(minutes=5),
                artifacts=artifacts,
                ledger_study_id="TEST-ONLY-unregistered-phase5-study",
            )
    assert adapter.calls == []
    assert not (tmp_path / "must-not-start").exists()


def _seed_phase5_source_projection_rows(
    artifacts: ArtifactStore,
    inputs: Phase5ExecutionInputManifest,
) -> None:
    ledger = artifacts.ledger
    source_study_id = "TEST-ONLY-held-out-source-study"
    ledger.register_study(
        study_id=source_study_id,
        protocol_hash=inputs.held_out_call_manifest_hash,
        code_manifest_hash=digest("phase5-source-code"),
        configuration_hash=digest("phase5-source-config"),
        release_class=ReleaseClass.PUBLIC,
        created_at=NOW - timedelta(hours=1),
    )
    sources: dict[str, tuple[OntologyProjection, object]] = {}
    for item in inputs.scripted_inputs:
        for cpu in item.cpu_inputs:
            sources[cpu.before_projection.content_hash] = (
                cpu.before_projection,
                item.packet,
            )
        sources[item.c2_before_projection.content_hash] = (
            item.c2_before_projection,
            item.packet,
        )
    for item in inputs.researcher_trace_inputs:
        sources[item.c2_before_projection.content_hash] = (
            item.c2_before_projection,
            item.packet,
        )
    for ordinal, (projection_hash, (semantic, evidence_packet)) in enumerate(
        sources.items(), 1
    ):
        prefix = f"TEST-ONLY-phase5-source-{ordinal}"
        snapshot_artifact = artifacts.put_bytes(
            f"snapshot:{semantic.snapshot_hash}\n".encode(),
            media_type="application/json",
            release_class=ReleaseClass.PUBLIC,
            created_at=NOW - timedelta(minutes=30),
        )
        packet_artifact = artifacts.put_bytes(
            (evidence_packet.to_canonical_json() + "\n").encode(),
            media_type="application/vnd.story-projection.evidence-packet+json",
            release_class=ReleaseClass.PUBLIC,
            created_at=NOW - timedelta(minutes=29),
        )
        snapshot_input = ledger.register_input(
            input_id=f"{prefix}-snapshot-input",
            study_id=source_study_id,
            input_kind=InputKind.EVIDENCE_SNAPSHOT,
            content_hash=semantic.snapshot_hash,
            artifact_hash=snapshot_artifact.content_hash,
            release_class=ReleaseClass.PUBLIC,
            created_at=NOW - timedelta(minutes=30),
        )
        snapshot = ledger.register_evidence_snapshot(
            snapshot_id=f"{prefix}-snapshot",
            input_id=snapshot_input.input_id,
            horizon_hash=digest(f"{prefix}-horizon"),
            evidence_manifest_hash=digest(f"{prefix}-evidence"),
            index_configuration_hash=digest(f"{prefix}-index"),
            prequery_seal_hash=semantic.snapshot_hash,
            eligible_evidence_count=len(evidence_packet.evidence),
            created_at=NOW - timedelta(minutes=30),
        )
        packet_input = ledger.register_input(
            input_id=f"{prefix}-packet-input",
            study_id=source_study_id,
            input_kind=InputKind.EVIDENCE_PACKET,
            content_hash=semantic.packet_hash,
            artifact_hash=packet_artifact.content_hash,
            release_class=ReleaseClass.PUBLIC,
            created_at=NOW - timedelta(minutes=29),
        )
        job = ledger.create_or_resume_job(
            {
                "TEST_ONLY": "held_out_source_projection",
                "projection_hash": projection_hash,
                "lifecycle_kind": "query_time_generation",
            },
            release_class=ReleaseClass.PUBLIC,
            created_at=NOW - timedelta(minutes=20),
        )
        ledger.link_job_to_study(
            study_id=source_study_id,
            job_id=job.job_id,
            created_at=NOW - timedelta(minutes=20),
        )
        ledger.advance_job_lifecycle(
            job.job_id,
            (
                (JobState.PREQUERY_SEALED, NOW - timedelta(minutes=20)),
                (
                    JobState.QUERY_REVEALED,
                    NOW - timedelta(minutes=20) + timedelta(microseconds=1),
                ),
            ),
        )
        attempt = ledger.record_attempt(
            attempt_id=f"{prefix}-attempt",
            job_id=job.job_id,
            attempt_kind=AttemptKind.BASE,
            input_hash=projection_hash,
            config_hash=digest(f"{prefix}-config"),
            seed=0,
            created_at=NOW - timedelta(minutes=19),
        )
        projection_artifact = artifacts.put_bytes(
            (semantic.to_canonical_json() + "\n").encode(),
            media_type="application/vnd.story-projection.ontology-projection+json",
            release_class=ReleaseClass.PUBLIC,
            created_at=NOW - timedelta(minutes=18),
        )
        ledger.advance_job_lifecycle(
            job.job_id,
            (
                (JobState.PREQUERY_SEALED, NOW - timedelta(minutes=20)),
                (
                    JobState.QUERY_REVEALED,
                    NOW - timedelta(minutes=20) + timedelta(microseconds=1),
                ),
                (JobState.GENERATED, NOW - timedelta(minutes=18)),
            ),
        )
        validation = ledger.record_validation(
            validation_id=f"{prefix}-validation",
            job_id=job.job_id,
            attempt_id=attempt.attempt_id,
            input_artifact_hash=projection_artifact.content_hash,
            validator_manifest_hash=digest(f"{prefix}-validator"),
            validation_status=ValidationStatus.ACCEPTED,
            evidence_support_status=EvidenceSupportStatus.NOT_APPLICABLE,
            temporal_status=TemporalValidationStatus.NOT_APPLICABLE,
            commitment_status=CommitmentCheckStatus.NOT_APPLICABLE,
            semantic_assessment_scope=(
                SemanticAssessmentScope.RUNTIME_STRUCTURAL_ONLY_NOT_ASSESSED
            ),
            created_at=NOW - timedelta(minutes=17),
        )
        certificate = semantic.construction_certificate or semantic.construction_seal
        assert certificate is not None
        ledger.record_projection(
            projection_id=f"{prefix}-projection",
            job_id=job.job_id,
            validation_id=validation.validation_id,
            snapshot_id=snapshot.snapshot_id,
            packet_input_id=packet_input.input_id,
            condition_id=semantic.condition.value,
            context_hash=semantic.context_hash,
            upper_ontology_hash=semantic.upper_ontology.content_hash,
            construction_certificate_hash=certificate.content_hash,
            projection_artifact_hash=projection_artifact.content_hash,
            projection_semantic_hash=semantic.content_hash,
            release_class=ReleaseClass.PUBLIC,
            finalized_at=NOW - timedelta(minutes=17),
        )
        ledger.advance_job_lifecycle(
            job.job_id,
            (
                (JobState.PREQUERY_SEALED, NOW - timedelta(minutes=20)),
                (
                    JobState.QUERY_REVEALED,
                    NOW - timedelta(minutes=20) + timedelta(microseconds=1),
                ),
                (JobState.GENERATED, NOW - timedelta(minutes=18)),
                (JobState.VALIDATED, NOW - timedelta(minutes=17)),
                (JobState.FINALIZED, NOW - timedelta(minutes=17)),
            ),
        )


def test_cpu_ledger_materialization_is_complete_replayable_and_tamper_evident(
    tmp_path: Path,
) -> None:
    protocol, _gate, inputs = _inputs()
    with Ledger(tmp_path / "ledger.sqlite3") as ledger:
        artifacts = ArtifactStore(
            BlobStore(tmp_path / "cas", compression=Compression.GZIP),
            ledger,
        )
        _seed_phase5_source_projection_rows(artifacts, inputs)
        ledger_study_id = "TEST-ONLY-phase5-receiving-study"
        ledger.register_study(
            study_id=ledger_study_id,
            protocol_hash=digest("phase5-receiving-protocol"),
            code_manifest_hash=digest("phase5-receiving-code"),
            configuration_hash=digest("phase5-receiving-config"),
            release_class=ReleaseClass.RESTRICTED,
            created_at=NOW - timedelta(minutes=1),
        )
        baseline = {
            table: ledger.count_rows(table)
            for table in ("jobs", "attempts", "validations", "projections", "failures")
        }
        materialize_phase5_cpu_ledger(
            inputs=inputs,
            protocol=protocol,
            artifacts=artifacts,
            ledger_study_id=ledger_study_id,
        )
        validate_phase5_ledger_preflight(
            inputs=inputs,
            artifacts=artifacts,
            ledger_study_id=ledger_study_id,
        )
        expected_deltas = {
            "jobs": 12,
            "attempts": 12,
            "validations": 12,
            "projections": 6,
            "failures": 6,
        }
        observed = {table: ledger.count_rows(table) for table in baseline}
        assert observed == {
            table: baseline[table] + expected_deltas[table] for table in baseline
        }
        materialize_phase5_cpu_ledger(
            inputs=inputs,
            protocol=protocol,
            artifacts=artifacts,
            ledger_study_id=ledger_study_id,
        )
        assert {table: ledger.count_rows(table) for table in baseline} == observed

        script = inputs.scripted_inputs[0]
        cpu = script.cpu_inputs[0]
        job = ledger.resolve_job_identity(
            phase5_cpu_receiving_job_identity(
                inputs=inputs,
                ledger_study_id=ledger_study_id,
                script=script,
                cpu=cpu,
            ),
            study_id=ledger_study_id,
        )
        ledger._connection.execute("DROP TRIGGER job_transitions_reject_update")
        ledger._connection.execute(
            "UPDATE job_transitions SET occurred_at = ? WHERE job_id = ? AND sequence = 3",
            ((cpu.completed_at + timedelta(seconds=1)).isoformat(), job.job_id),
        )
        ledger._connection.commit()
        with pytest.raises(
            Phase5CpuLedgerMaterializationError,
            match="lifecycle differs",
        ):
            materialize_phase5_cpu_ledger(
                inputs=inputs,
                protocol=protocol,
                artifacts=artifacts,
                ledger_study_id=ledger_study_id,
            )


def test_completion_clock_is_evaluated_once_after_all_nine_terminal_calls(
    tmp_path: Path,
) -> None:
    protocol, gate, inputs = _inputs()
    adapter = _Adapter()
    evaluations = []

    def completion_clock():
        assert len(adapter.calls) == 9
        evaluations.append(tuple(adapter.calls))
        return NOW + timedelta(minutes=5)

    result = execute_phase5(
        inputs=inputs,
        protocol=protocol,
        prerequisites=gate,
        adapter=adapter,
        ledger_verifier=_Verifier(),
        output_root=tmp_path / "TEST-ONLY-clock",
        completed_at=completion_clock,
    )
    assert len(evaluations) == 1
    assert result.completed_at == NOW + timedelta(minutes=5)


def test_completion_clock_must_be_aware_and_cannot_backdate_terminal_calls(
    tmp_path: Path,
) -> None:
    protocol, gate, inputs = _inputs()
    adapter = _Adapter()
    with pytest.raises(Phase5ExecutionError, match="aware datetime"):
        execute_phase5(
            inputs=inputs,
            protocol=protocol,
            prerequisites=gate,
            adapter=adapter,
            ledger_verifier=_Verifier(),
            output_root=tmp_path / "TEST-ONLY-naive-clock",
            completed_at=lambda: datetime(2025, 1, 1),
        )
    assert len(adapter.calls) == 9

    backdated_root = tmp_path / "TEST-ONLY-backdated-clock"
    with pytest.raises(Phase5ExecutionError, match="predates"):
        execute_phase5(
            inputs=inputs,
            protocol=protocol,
            prerequisites=gate,
            adapter=_Adapter(),
            ledger_verifier=_Verifier(),
            output_root=backdated_root,
            completed_at=lambda: NOW,
        )
    assert not (backdated_root / "feedback_manifest.json").exists()
    assert not (backdated_root / "execution_index.json").exists()


def test_completion_clock_is_not_evaluated_when_an_episode_raises(tmp_path: Path) -> None:
    protocol, gate, inputs = _inputs()
    evaluated = False

    def forbidden_clock():
        nonlocal evaluated
        evaluated = True
        return NOW + timedelta(minutes=5)

    with pytest.raises(RuntimeError, match="controller loss"):
        execute_phase5(
            inputs=inputs,
            protocol=protocol,
            prerequisites=gate,
            adapter=_CrashBeforeResultAdapter(),
            ledger_verifier=_Verifier(),
            output_root=tmp_path / "TEST-ONLY-clock-on-failure",
            completed_at=forbidden_clock,
        )
    assert evaluated is False


def test_model_visible_c2_request_excludes_scorer_bindings() -> None:
    protocol, _gate, inputs = _inputs()
    script = inputs.scripted_inputs[0]
    request = C2RegenerationRequest(
        call_slot_id="phase5-test-slot",
        episode_id=script.episode_id,
        kind="scripted_known_answer",
        instruction=script.instruction,
        packet=script.packet,
        before_projection_id=script.c2_before_projection.projection_id,
        before_projection_hash=script.c2_before_projection.content_hash,
        before_context_hash=script.c2_before_projection.context_hash,
        before_packet_hash=script.c2_before_projection.packet_hash,
        protocol_hash=protocol.content_hash,
        seed=protocol.llm_seed,
    )
    serialized = str(request.model_visible_payload()).casefold()
    assert "scorer" not in serialized
    assert "gold" not in serialized
    request_serialized = str(request.model_dump(mode="json")).casefold()
    assert "instance_graph" not in request_serialized
    assert "local_schema" not in request_serialized


def test_sqlite_verifier_replays_model_call_and_gpu_event_rows(tmp_path: Path) -> None:
    protocol, _gate, inputs = _inputs()
    script = inputs.scripted_inputs[0]
    request = C2RegenerationRequest(
        call_slot_id="phase5-ledger-test-slot",
        episode_id=script.episode_id,
        kind="scripted_known_answer",
        instruction=script.instruction,
        packet=script.packet,
        before_projection_id=script.c2_before_projection.projection_id,
        before_projection_hash=script.c2_before_projection.content_hash,
        before_context_hash=script.c2_before_projection.context_hash,
        before_packet_hash=script.c2_before_projection.packet_hash,
        protocol_hash=protocol.content_hash,
        seed=protocol.llm_seed,
    )
    started = script.instruction.revision.created_at + timedelta(seconds=1)
    completed = script.instruction.revision.created_at + timedelta(seconds=7)
    ledger_path = tmp_path / "ledger.sqlite3"
    with Ledger(ledger_path) as ledger:
        job = ledger.create_or_resume_job(
            {"phase": 5, "request_hash": request.content_hash},
            release_class=ReleaseClass.RESTRICTED,
            created_at=started,
        )
        attempt = ledger.record_attempt(
            attempt_id="phase5-ledger-attempt",
            job_id=job.job_id,
            attempt_kind=AttemptKind.BASE,
            input_hash=request.content_hash,
            config_hash=digest("phase5-ledger-config"),
            seed=request.seed,
            created_at=started,
        )
        response_hash = digest("phase5-ledger-response")
        ledger.register_artifact(
            ArtifactRecord(
                content_hash=response_hash,
                compression=Compression.ZSTD,
                media_type="application/json",
                raw_size_bytes=10,
                stored_size_bytes=10,
                relative_path="restricted/phase5-ledger-response.json.zst",
                release_class=ReleaseClass.RESTRICTED,
                created_at=completed.isoformat(),
            )
        )
        event = ledger.record_gpu_event(
            event_id="phase5-ledger-event",
            event_kind=GpuEventKind.INFERENCE,
            allocated_seconds=6.0,
            started_at=started,
            ended_at=completed,
            succeeded=True,
            job_id=job.job_id,
            attempt_id=attempt.attempt_id,
        )
        model_call = ledger.record_model_call(
            model_call_id="phase5-ledger-model-call",
            job_id=job.job_id,
            attempt_id=attempt.attempt_id,
            gpu_event_id=event.event_id,
            backend=ModelBackend.VLLM_GPU,
            call_role=ModelCallRole.QUERY_TIME,
            retry_class=RetryClass.STANDARD,
            model_manifest_hash=digest("phase5-ledger-model"),
            decoding_manifest_hash=digest("phase5-ledger-decoding"),
            request_hash=request.content_hash,
            response_artifact_hash=response_hash,
            construction_unit_hash=digest("phase5-ledger-unit"),
            served_context_count=1,
            prompt_tokens=100,
            completion_tokens=50,
            allocated_gpu_seconds=6.0,
            successful=True,
            created_at=completed,
        )
    reference = FeedbackLedgerCallReference(
        model_call_id=model_call.model_call_id,
        model_call_record_hash=feedback_model_call_record_hash(model_call),
        gpu_event_id=event.event_id,
        gpu_event_record_hash=feedback_gpu_event_record_hash(event),
        attempt_id=attempt.attempt_id,
        attempt_kind="base",
        call_role="query_time",
        retry_class="standard",
        request_hash=request.content_hash,
        allocated_gpu_seconds=6.0,
        started_at=started,
        completed_at=completed,
        successful=True,
    )
    receipt = FeedbackRegenerationReceipt(
        ledger_calls=(reference,),
        attempt_status=FeedbackAttemptStatus.SUCCEEDED,
        instruction_hash=request.instruction.content_hash,
        before_projection_hash=request.before_projection_hash,
        after_projection_hash=digest("phase5-ledger-after"),
        seed=request.seed,
        allocated_gpu_seconds=6.0,
        started_at=started,
        completed_at=completed,
    )
    verifier = SQLiteFeedbackLedgerVerifier(ledger_path)
    verifier.verify(request, receipt)

    bad_reference = reference.model_copy(
        update={"model_call_record_hash": digest("tampered-ledger-row")}
    )
    bad_receipt = receipt.model_copy(update={"ledger_calls": (bad_reference,)})
    with pytest.raises(Phase5ExecutionError, match="row hash changed"):
        verifier.verify(request, bad_receipt)


def test_script_requires_one_condition_independent_known_answer() -> None:
    _protocol, _gate, inputs = _inputs()
    payload = _without_hashes(inputs.scripted_inputs[0].model_dump(mode="python"))
    payload["scorer_bindings"][0]["required_target_change_hash"] = digest("condition-leak")
    with pytest.raises(ValueError, match="one shared known answer"):
        ScriptedFeedbackInput.model_validate(payload)


def test_cpu_feedback_must_start_strictly_after_revision() -> None:
    _protocol, _gate, inputs = _inputs()
    script = inputs.scripted_inputs[0]
    payload = _without_hashes(script.model_dump(mode="python"))
    payload["cpu_inputs"][0]["started_at"] = script.instruction.revision.created_at
    with pytest.raises(ValueError, match="start after the revision"):
        ScriptedFeedbackInput.model_validate(payload)


def test_timeout_remains_in_intention_to_treat_manifest(tmp_path: Path) -> None:
    protocol, gate, inputs = _inputs()
    adapter = _TimeoutFirstAdapter()
    root = tmp_path / "TEST-ONLY-timeout"
    execute_phase5(
        inputs=inputs,
        protocol=protocol,
        prerequisites=gate,
        adapter=adapter,
        ledger_verifier=_Verifier(),
        output_root=root,
        completed_at=NOW + timedelta(minutes=5),
    )
    records = [path.read_text() for path in (root / "records").iterdir()]
    assert sum('"attempt_status":"timed_out"' in item for item in records) == 1
    assert len(adapter.calls) == 9


def test_orphaned_call_slot_requires_recovery_and_is_never_resent(tmp_path: Path) -> None:
    protocol, gate, inputs = _inputs()
    root = tmp_path / "TEST-ONLY-interrupted"
    crashed = _CrashBeforeResultAdapter()
    with pytest.raises(RuntimeError, match="controller loss"):
        execute_phase5(
            inputs=inputs,
            protocol=protocol,
            prerequisites=gate,
            adapter=crashed,
            ledger_verifier=_Verifier(),
            output_root=root,
            completed_at=NOW + timedelta(minutes=5),
        )
    assert len(crashed.calls) == 1
    assert len(tuple((root / "call_slots").iterdir())) == 1

    replacement = _AmbiguousRecoveryAdapter()
    with pytest.raises(InterruptedFeedbackCallRecoveryRequired, match="ambiguous adapter trace"):
        execute_phase5(
            inputs=inputs,
            protocol=protocol,
            prerequisites=gate,
            adapter=replacement,
            ledger_verifier=_Verifier(),
            output_root=root,
            completed_at=NOW + timedelta(minutes=5),
        )
    assert replacement.calls == []


def test_outer_phase5_slot_without_adapter_trace_is_issued_exactly_once(
    tmp_path: Path,
) -> None:
    protocol, gate, inputs = _inputs()
    root = tmp_path / "TEST-ONLY-pre-adapter-crash"
    script = inputs.scripted_inputs[0]
    before = script.c2_before_projection
    request = C2RegenerationRequest(
        call_slot_id=f"phase5-c2-{script.episode_id}",
        episode_id=script.episode_id,
        kind=FeedbackEpisodeKind.SCRIPTED_KNOWN_ANSWER,
        instruction=script.instruction,
        packet=script.packet,
        before_projection_id=before.projection_id,
        before_projection_hash=before.content_hash,
        before_context_hash=before.context_hash,
        before_packet_hash=before.packet_hash,
        protocol_hash=protocol.content_hash,
        seed=protocol.llm_seed,
    )
    _append_exact(root / "call_slots" / f"{script.episode_id}.json", request)
    adapter = _Adapter()
    execute_phase5(
        inputs=inputs,
        protocol=protocol,
        prerequisites=gate,
        adapter=adapter,
        ledger_verifier=_Verifier(),
        output_root=root,
        completed_at=NOW + timedelta(minutes=5),
    )
    assert adapter.calls.count(request.call_slot_id) == 1
    assert len(adapter.calls) == 9


def test_symlinked_journal_root_fails_closed(tmp_path: Path) -> None:
    protocol, gate, inputs = _inputs()
    real = tmp_path / "real"
    real.mkdir()
    linked = tmp_path / "linked"
    linked.symlink_to(real, target_is_directory=True)
    with pytest.raises(Phase5ExecutionError, match="symlinked Phase 5 path"):
        execute_phase5(
            inputs=inputs,
            protocol=protocol,
            prerequisites=gate,
            adapter=_Adapter(),
            ledger_verifier=_Verifier(),
            output_root=linked,
            completed_at=NOW + timedelta(minutes=5),
        )


def test_input_loader_rejects_symlinked_ancestor(tmp_path: Path) -> None:
    _protocol, _gate, inputs = _inputs()
    real = tmp_path / "real-inputs"
    real.mkdir()
    (real / "inputs.json").write_text(inputs.to_canonical_json())
    linked = tmp_path / "linked-inputs"
    linked.symlink_to(real, target_is_directory=True)
    with pytest.raises(Phase5ExecutionError, match="symlinked Phase 5 path"):
        load_phase5_input_manifest(linked / "inputs.json")


def test_node_is_hidden_when_description_assertion_exceeds_revelation_horizon() -> None:
    query_context = context()
    evidence_packet = packet()
    original = projection(ConditionName.C0_CLASSICAL_PRE, query_context, evidence_packet)
    values = _without_hashes(original.model_dump(mode="python"))
    assertion = values["instance_graph"]["assertions"][0]
    assertion["temporal_scope"]["revelation_position"]["revelation_order"] = 9
    _rebind_prequery_validation(values, validated_at=NOW + timedelta(seconds=3))
    changed = OntologyProjection.model_validate(values)
    bundle = build_visualization_bundle(changed, query_context, evidence_packet)
    horizon_values = _without_hashes(query_context.spoiler_horizon.model_dump(mode="python"))
    horizon_values["max_revelation_position"] = RevelationPosition(revelation_order=1)
    horizon = SpoilerHorizon.model_validate(horizon_values)
    filtered = filter_visualization_bundle(
        bundle, VisualizationTemporalFilter(spoiler_horizon=horizon)
    )
    ari = next(item for item in bundle.state.nodes if item.projection_object_id == "entity-a")
    assert ari.visualization_node_id not in filtered.state.visible_node_ids
