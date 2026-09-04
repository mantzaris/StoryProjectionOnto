from __future__ import annotations

import hashlib
from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace

import pytest

from story_projection_onto.contracts import (
    ConditionName,
    ConstructionSeal,
    FeedbackAction,
    OntologyProjection,
    ReleaseClass,
    RevelationPosition,
    SpoilerHorizon,
    canonical_sha256,
)
from story_projection_onto.feedback_runtime import (
    FeedbackAttemptStatus,
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
    Phase5ExecutionError,
    Phase5ExecutionInputManifest,
    Phase5PrerequisiteError,
    PrimaryHeldOutResultsGate,
    ResearcherTraceInput,
    ScorerOnlyFeedbackBinding,
    ScriptedFeedbackInput,
    SQLiteFeedbackLedgerVerifier,
    execute_phase5,
    feedback_gpu_event_record_hash,
    feedback_model_call_record_hash,
    load_phase5_input_manifest,
    load_phase5_runner_configuration,
    validate_phase5_inputs,
    validate_phase5_prerequisites,
)
from story_projection_onto.store import (
    ArtifactRecord,
    AttemptKind,
    Compression,
    GpuEventKind,
    Ledger,
    ModelBackend,
    ModelCallRole,
    RetryClass,
)
from story_projection_onto.ui import (
    MergeSplitIntent,
    MergeSplitOperation,
    RefineContextIntent,
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
    return CpuReprojectionInput(
        condition=condition,
        before_projection=before,
        after_projection=after,
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
        traces.append(
            ResearcherTraceInput(
                episode_id=slot.episode_id,
                instruction=instruction,
                packet=packet(),
                c2_before_projection=projection(
                    ConditionName.C2_LLM_QUERY, instruction.before_context, packet()
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

    replacement = _Adapter()
    with pytest.raises(InterruptedFeedbackCallRecoveryRequired, match="no recoverable result"):
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
    values["validation_bundle_hash"] = None
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
