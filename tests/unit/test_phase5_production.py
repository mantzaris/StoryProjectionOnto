from __future__ import annotations

import hashlib
from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace

import pytest

from story_projection_onto.contracts import (
    ConditionName,
    EvidencePacket,
    EvidenceSnapshot,
    OntologyProjection,
    ReleaseClass,
)
from story_projection_onto.development_runtime import RunOutcome
from story_projection_onto.feedback_runtime import ScriptedRevisionFreeze
from story_projection_onto.phase5_commitment import (
    Phase5ParentReadinessFloor,
    Phase5ReviewGateBinding,
    Phase5ScriptCommitmentEntry,
    Phase5ScriptCommitmentManifest,
)
from story_projection_onto.phase5_execution import (
    C2RegenerationRequest,
    SQLiteFeedbackLedgerVerifier,
)
from story_projection_onto.phase5_production import (
    Phase5AdapterError,
    Phase5CASReference,
    Phase5EpisodeSource,
    Phase5HeldOutProjectionExport,
    Phase5MaterializationError,
    Phase5MaterializationSourceManifest,
    Phase5OwnedServiceIdentity,
    Phase5OwnedServiceResult,
    Phase5ScorerBindingAuthorization,
    Phase5SourceModelCallBinding,
    RestrictedPhase5CAS,
    build_metered_phase5_adapter,
    materialize_phase5_input_manifest,
    materialize_phase5_inputs_to_directory,
    persist_phase5_record,
    validate_phase5_materialization_receipt,
)
from story_projection_onto.store import (
    ArtifactStore,
    AttemptKind,
    BlobStore,
    Compression,
    GpuEventKind,
    Ledger,
    ModelBackend,
    ModelCallRole,
    RetryClass,
)
from story_projection_onto.ui import (
    FeedbackEpisodeKind,
    RevisionDraftSubmission,
    VisualizationContentScope,
    _compile_submission,
    build_visualization_bundle,
)
from tests.unit.test_phase5_execution import (
    _inputs,
    _rebind_prequery_validation,
    _without_hashes,
)
from tests.unit.test_ui import NOW, digest, projection


def _request():
    protocol, _gate, inputs = _inputs()
    script = inputs.scripted_inputs[0]
    return (
        protocol,
        script,
        C2RegenerationRequest(
            call_slot_id=f"phase5-c2-{script.episode_id}",
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
        ),
    )


class _OwnedService:
    def __init__(self, *, artifacts: ArtifactStore, identity: Phase5OwnedServiceIdentity):
        self.artifacts = artifacts
        self._identity = identity
        self.execute_count = 0
        self.recovered: dict[str, Phase5OwnedServiceResult] = {}

    def identity(self) -> Phase5OwnedServiceIdentity:
        return self._identity

    def execute_feedback(self, request, model_visible_payload):
        self.execute_count += 1
        assert "scorer" not in str(model_visible_payload).casefold()
        assert "gold" not in str(model_visible_payload).casefold()
        before = self.artifacts.ledger.gpu_summary().total_allocated_seconds
        started = request.instruction.revision.created_at + timedelta(seconds=1)
        completed = request.instruction.revision.created_at + timedelta(seconds=7)
        job = self.artifacts.ledger.create_or_resume_job(
            {"phase": 5, "request_hash": request.content_hash},
            release_class=ReleaseClass.RESTRICTED,
            created_at=started,
        )
        attempt = self.artifacts.ledger.record_attempt(
            attempt_id=f"attempt-{request.episode_id}",
            job_id=job.job_id,
            attempt_kind=AttemptKind.BASE,
            input_hash=request.content_hash,
            config_hash=digest("phase5-owned-service-config"),
            seed=request.seed,
            created_at=started,
        )
        event = self.artifacts.ledger.record_gpu_event(
            event_id=f"event-{request.episode_id}",
            event_kind=GpuEventKind.INFERENCE,
            allocated_seconds=6,
            started_at=started,
            ended_at=completed,
            succeeded=True,
            job_id=job.job_id,
            attempt_id=attempt.attempt_id,
        )
        raw = self.artifacts.put_bytes(
            b'{"TEST_ONLY":"phase5 model response"}\n',
            media_type="application/json",
            release_class=ReleaseClass.RESTRICTED,
            created_at=completed,
        )
        self.artifacts.ledger.record_model_call(
            model_call_id=f"model-{request.episode_id}",
            job_id=job.job_id,
            attempt_id=attempt.attempt_id,
            gpu_event_id=event.event_id,
            backend=ModelBackend.VLLM_GPU,
            call_role=ModelCallRole.QUERY_TIME,
            retry_class=RetryClass.STANDARD,
            model_manifest_hash=self._identity.model_manifest_hash,
            decoding_manifest_hash=self._identity.decoding_manifest_hash,
            request_hash=request.content_hash,
            response_artifact_hash=raw.content_hash,
            construction_unit_hash=digest(f"{request.episode_id}-feedback-unit"),
            served_context_count=1,
            prompt_tokens=101,
            completion_tokens=51,
            allocated_gpu_seconds=6,
            successful=True,
            created_at=completed,
        )
        after = projection(
            ConditionName.C2_LLM_QUERY,
            request.instruction.after_context,
            request.packet,
            decision_offset_seconds=65,
        )
        after_reference = persist_phase5_record(
            self.artifacts,
            after,
            object_kind="phase5_after_projection",
            created_at=completed,
        )
        result = Phase5OwnedServiceResult(
            request_hash=request.content_hash,
            service_identity_hash=self._identity.content_hash,
            attempt_status="succeeded",
            model_call_ids=(f"model-{request.episode_id}",),
            after_projection=after_reference,
            resolver_hash=digest("phase5-owned-resolver"),
            resolved_at=completed + timedelta(seconds=1),
            checked_at=completed + timedelta(seconds=2),
            cumulative_gpu_seconds_before=before,
            cumulative_gpu_seconds_after=(
                self.artifacts.ledger.gpu_summary().total_allocated_seconds
            ),
        )
        self.recovered[request.content_hash] = result
        return result

    def recover_feedback(self, request):
        return self.recovered.get(request.content_hash)


def _adapter_fixture(tmp_path: Path):
    ledger = Ledger(tmp_path / "ledger.sqlite3")
    artifacts = ArtifactStore(BlobStore(tmp_path / "cas", compression=Compression.GZIP), ledger)
    load_started = NOW - timedelta(minutes=10)
    load_ended = load_started + timedelta(seconds=2)
    load_event = ledger.record_gpu_event(
        event_id="phase5-existing-model-load",
        event_kind=GpuEventKind.MODEL_LOAD,
        allocated_seconds=2,
        started_at=load_started,
        ended_at=load_ended,
        succeeded=True,
    )
    from story_projection_onto.phase5_execution import feedback_gpu_event_record_hash

    identity = Phase5OwnedServiceIdentity(
        service_id="TEST-ONLY-already-owned-phase5-service",
        global_accounting_id="TEST-ONLY-global-ledger",
        model_manifest_hash=digest("phase5-model"),
        decoding_manifest_hash=digest("phase5-decoding"),
        model_load_event_id=load_event.event_id,
        model_load_event_record_hash=feedback_gpu_event_record_hash(load_event),
        cumulative_gpu_seconds_at_handoff=2,
        handed_off_at=load_ended + timedelta(seconds=1),
    )
    service = _OwnedService(artifacts=artifacts, identity=identity)
    return ledger, artifacts, service


def test_owned_service_adapter_is_metered_scorer_blind_and_recoverable(
    tmp_path: Path,
) -> None:
    protocol, _script, request = _request()
    ledger, artifacts, service = _adapter_fixture(tmp_path)
    try:
        adapter = build_metered_phase5_adapter(service=service, artifacts=artifacts)
        result = adapter.regenerate(request)
        assert result.request_hash == request.content_hash
        assert result.receipt.seed == protocol.llm_seed
        assert result.receipt.allocated_gpu_seconds == 6
        assert service.execute_count == 1
        SQLiteFeedbackLedgerVerifier(tmp_path / "ledger.sqlite3").verify(request, result.receipt)
        recovered = adapter.recover(request)
        assert recovered == result
        assert service.execute_count == 1
        repeated = adapter.regenerate(request)
        assert repeated == result
        assert service.execute_count == 1
    finally:
        ledger.close()


def test_owned_service_factory_rejects_lifecycle_capability(tmp_path: Path) -> None:
    ledger, artifacts, service = _adapter_fixture(tmp_path)
    try:
        service.start = lambda: None
        with pytest.raises(Phase5AdapterError, match="lifecycle"):
            build_metered_phase5_adapter(service=service, artifacts=artifacts)
    finally:
        ledger.close()


def test_restricted_cas_detects_reference_tamper(tmp_path: Path) -> None:
    _protocol, script, _request_value = _request()
    ledger, artifacts, _service = _adapter_fixture(tmp_path)
    try:
        reference = persist_phase5_record(
            artifacts,
            script.instruction,
            object_kind="revision_instruction",
            created_at=script.instruction.revision.created_at,
        )
        tampered = Phase5CASReference.model_validate(
            {
                **reference.model_dump(mode="python", exclude={"content_hash"}),
                "logical_content_hash": digest("tampered-logical-object"),
            }
        )
        with pytest.raises(Phase5MaterializationError, match="logical hash changed"):
            RestrictedPhase5CAS(artifacts).load(
                tampered,
                type(script.instruction),
                object_kind="revision_instruction",
            )
    finally:
        ledger.close()


def test_phase5_cas_reuses_canonical_public_evidence_without_release_conflict(
    tmp_path: Path,
) -> None:
    _protocol, script, _request_value = _request()
    ledger, artifacts, _service = _adapter_fixture(tmp_path)
    packet_payload = (script.packet.to_canonical_json() + "\n").encode("utf-8")
    snapshot = EvidenceSnapshot(
        snapshot_id="TEST-ONLY-phase5-public-snapshot",
        corpus_id="TEST-ONLY-phase5-public-corpus",
        world_or_window_id="TEST-ONLY-phase5-public-world",
        horizon=script.instruction.before_context.spoiler_horizon,
        eligible_evidence_ids=script.packet.ordered_evidence_ids,
        index_config_hash=digest("TEST-ONLY-phase5-public-index"),
        created_at=NOW - timedelta(seconds=4),
        sealed_at=NOW - timedelta(seconds=3),
        release_class=ReleaseClass.PUBLIC,
    )
    snapshot_payload = (snapshot.to_canonical_json() + "\n").encode("utf-8")
    try:
        existing = artifacts.put_bytes(
            packet_payload,
            media_type="application/vnd.story-projection.evidence-packet+json",
            release_class=ReleaseClass.PUBLIC,
            created_at=NOW - timedelta(seconds=2),
        )
        reference = persist_phase5_record(
            artifacts,
            script.packet,
            object_kind="evidence_packet",
            release_class=ReleaseClass.PUBLIC,
            created_at=NOW - timedelta(seconds=1),
        )
        assert reference.artifact_hash == existing.content_hash
        assert reference.release_class is ReleaseClass.PUBLIC
        assert RestrictedPhase5CAS(artifacts).load(
            reference,
            EvidencePacket,
            object_kind="evidence_packet",
        ) == script.packet
        existing_snapshot = artifacts.put_bytes(
            snapshot_payload,
            media_type="application/vnd.story-projection.evidence-snapshot+json",
            release_class=ReleaseClass.PUBLIC,
            created_at=NOW - timedelta(seconds=2),
        )
        snapshot_reference = persist_phase5_record(
            artifacts,
            snapshot,
            object_kind="evidence_snapshot",
            release_class=ReleaseClass.PUBLIC,
            created_at=NOW - timedelta(seconds=1),
        )
        assert snapshot_reference.artifact_hash == existing_snapshot.content_hash
        assert snapshot_reference.release_class is ReleaseClass.PUBLIC
        assert RestrictedPhase5CAS(artifacts).load(
            snapshot_reference,
            EvidenceSnapshot,
            object_kind="evidence_snapshot",
        ) == snapshot

        with pytest.raises(Phase5MaterializationError, match="may be persisted as public"):
            persist_phase5_record(
                artifacts,
                script.c2_before_projection,
                object_kind="ontology_projection",
                release_class=ReleaseClass.PUBLIC,
                created_at=NOW,
            )
    finally:
        ledger.close()


def test_materializer_refuses_symlinked_output_root(tmp_path: Path) -> None:
    real_root = tmp_path / "real-output"
    real_root.mkdir()
    linked_root = tmp_path / "linked-output"
    linked_root.symlink_to(real_root, target_is_directory=True)
    with pytest.raises(Phase5MaterializationError, match="symlinked"):
        materialize_phase5_inputs_to_directory(
            source=None,
            protocol=None,
            script_commitment_path=tmp_path / "unused-commitment.json",
            primary_results_gate_path=tmp_path / "unused-gate.json",
            benchmark_root=tmp_path,
            review_completion_root=tmp_path,
            cas=None,
            output_root=linked_root,
        )


class _MappedCAS:
    def __init__(self, artifacts: ArtifactStore):
        self.artifacts = artifacts
        self.values = {}

    def reference(self, value, object_kind):
        artifact_hash = digest(f"TEST-ONLY-cas-{object_kind}-{value.content_hash}")
        reference = Phase5CASReference(
            artifact_hash=artifact_hash,
            logical_content_hash=value.content_hash,
            object_kind=object_kind,
            media_type=f"application/TEST-ONLY-{object_kind}+json",
        )
        self.values[artifact_hash] = value
        return reference

    def load(self, reference, model_type, *, object_kind):
        assert reference.object_kind == object_kind
        value = self.values[reference.artifact_hash]
        assert isinstance(value, model_type) or model_type.__name__ == "HeldOutCallManifest"
        assert value.content_hash == reference.logical_content_hash
        return value


def _rebind_projection(base, *, snapshot, evidence_packet):
    payload = _without_hashes(base.model_dump(mode="python"))
    payload["snapshot_hash"] = snapshot.content_hash
    payload["packet_hash"] = evidence_packet.content_hash
    if base.construction_seal is not None:
        seal = _without_hashes(base.construction_seal.model_dump(mode="python"))
        seal["snapshot_hash"] = snapshot.content_hash
        payload["construction_seal"] = seal
    if base.pre_query_inventory is not None:
        inventory_payload = _without_hashes(base.pre_query_inventory.model_dump(mode="python"))
        inventory_payload["snapshot_hash"] = snapshot.content_hash
        inventory = base.pre_query_inventory.__class__.model_validate(inventory_payload)
        certificate_payload = _without_hashes(
            base.construction_certificate.model_dump(mode="python")
        )
        certificate_payload.update(
            snapshot_hash=snapshot.content_hash,
            packet_hash=evidence_packet.content_hash,
            pre_query_inventory_hash=inventory.content_hash,
        )
        payload["pre_query_inventory"] = inventory
        payload["construction_certificate"] = certificate_payload
    if base.condition in {ConditionName.C0_CLASSICAL_PRE, ConditionName.C1_LLM_PRE}:
        _rebind_prequery_validation(
            payload,
            validated_at=base.validation_records[0].validated_at,
        )
    return OntologyProjection.model_validate(payload)


def _source_model_call(
    *,
    artifacts,
    label,
    request_hash,
    role,
    seed,
    started,
):
    ledger = artifacts.ledger
    job = ledger.create_or_resume_job(
        {"TEST_ONLY": label},
        release_class=ReleaseClass.RESTRICTED,
        created_at=started,
    )
    attempt = ledger.record_attempt(
        attempt_id=f"TEST-ONLY-attempt-{label}",
        job_id=job.job_id,
        attempt_kind=AttemptKind.BASE,
        input_hash=request_hash,
        config_hash=digest(f"TEST-ONLY-config-{label}"),
        seed=seed,
        created_at=started,
    )
    event = ledger.record_gpu_event(
        event_id=f"TEST-ONLY-event-{label}",
        event_kind=GpuEventKind.INFERENCE,
        allocated_seconds=1,
        started_at=started,
        ended_at=started + timedelta(seconds=1),
        succeeded=True,
        job_id=job.job_id,
        attempt_id=attempt.attempt_id,
    )
    raw = artifacts.put_bytes(
        f'{{"TEST_ONLY":"{label}"}}\n'.encode(),
        media_type="application/json",
        release_class=ReleaseClass.RESTRICTED,
        created_at=started + timedelta(seconds=1),
    )
    call = ledger.record_model_call(
        model_call_id=f"TEST-ONLY-model-{label}",
        job_id=job.job_id,
        attempt_id=attempt.attempt_id,
        gpu_event_id=event.event_id,
        backend=ModelBackend.VLLM_GPU,
        call_role=role,
        retry_class=(RetryClass.LONG if role is ModelCallRole.PREBUILD else RetryClass.STANDARD),
        model_manifest_hash=digest("TEST-ONLY-heldout-model"),
        decoding_manifest_hash=digest("TEST-ONLY-heldout-decoding"),
        request_hash=request_hash,
        response_artifact_hash=raw.content_hash,
        construction_unit_hash=digest(f"TEST-ONLY-unit-{label}"),
        served_context_count=1,
        prompt_tokens=10,
        completion_tokens=5,
        allocated_gpu_seconds=1,
        successful=True,
        created_at=started + timedelta(seconds=1),
    )
    from story_projection_onto.phase5_execution import (
        feedback_gpu_event_record_hash,
        feedback_model_call_record_hash,
    )

    return Phase5SourceModelCallBinding(
        model_call_id=call.model_call_id,
        model_call_record_hash=feedback_model_call_record_hash(call),
        gpu_event_id=event.event_id,
        gpu_event_record_hash=feedback_gpu_event_record_hash(event),
    )


def test_materializer_replays_exact_sources_without_scorer_gold(
    tmp_path: Path, monkeypatch
) -> None:
    protocol, gate_template, input_template = _inputs()
    ledger = Ledger(tmp_path / "source-ledger.sqlite3")
    artifacts = ArtifactStore(
        BlobStore(tmp_path / "source-cas", compression=Compression.GZIP), ledger
    )
    cas = _MappedCAS(artifacts)
    calls = []
    units = []
    projection_receipts = []
    itt_records = []
    episode_sources = []
    commitment_entries = []
    export_counter = 0
    try:
        all_inputs = (*input_template.scripted_inputs, *input_template.researcher_trace_inputs)
        for ordinal, template in enumerate(all_inputs, 1):
            context = template.instruction.before_context
            snapshot = EvidenceSnapshot(
                snapshot_id=f"TEST-ONLY-snapshot-{ordinal}",
                corpus_id="TEST-ONLY-corpus",
                world_or_window_id=f"TEST-ONLY-world-{ordinal}",
                horizon=context.spoiler_horizon,
                eligible_evidence_ids=template.packet.ordered_evidence_ids,
                index_config_hash=digest("TEST-ONLY-index"),
                created_at=NOW - timedelta(minutes=2),
                sealed_at=NOW - timedelta(minutes=1),
                release_class=ReleaseClass.PUBLIC,
            )
            packet_payload = _without_hashes(template.packet.model_dump(mode="python"))
            packet_payload.update(
                packet_id=f"TEST-ONLY-packet-{ordinal}",
                snapshot_hash=snapshot.content_hash,
            )
            evidence_packet = EvidencePacket.model_validate(packet_payload)
            query_hash = digest(f"TEST-ONLY-query-stage-{ordinal}")
            query_relative_path = (
                f"data/synthetic/model_visible/query_stages/TEST-ONLY-{template.episode_id}"
            )
            query_manifest_file_sha256 = digest(
                f"TEST-ONLY-query-stage-file-{ordinal}"
            )
            query_stage = SimpleNamespace(
                relative_path=query_relative_path,
                manifest_file_sha256=query_manifest_file_sha256,
                staging_manifest_hash=query_hash,
                query_context_hash=context.content_hash,
                horizon_hash=context.spoiler_horizon.content_hash,
                budget_hash=context.budgets.content_hash,
                evidence_artifact_hash=evidence_packet.content_hash,
                query_artifact_hash=digest(f"TEST-ONLY-query-artifact-{ordinal}"),
            )
            unit_id = f"TEST-ONLY-unit-{ordinal}"
            units.append(SimpleNamespace(unit_id=unit_id, query_stages=(query_stage,)))
            packet_ref = cas.reference(evidence_packet, "evidence_packet")
            snapshot_ref = cas.reference(snapshot, "evidence_snapshot")

            c2_projection = _rebind_projection(
                template.c2_before_projection,
                snapshot=snapshot,
                evidence_packet=evidence_packet,
            )
            c2_projection_ref = cas.reference(c2_projection, "ontology_projection")
            c2_request_hash = digest(f"TEST-ONLY-heldout-c2-request-{ordinal}")
            c2_call_hash = digest(f"TEST-ONLY-heldout-c2-call-{ordinal}")
            c2_call = SimpleNamespace(
                content_hash=c2_call_hash,
                call_id=f"TEST-ONLY-heldout-c2-{ordinal}",
                call_class="test_c2",
                unit_id=unit_id,
                seed_block=1,
                frozen_seed=protocol.llm_seed,
                vllm_seed=17,
                query_stage_hash=query_hash,
            )
            calls.append(c2_call)
            c2_binding = _source_model_call(
                artifacts=artifacts,
                label=f"c2-{ordinal}",
                request_hash=c2_request_hash,
                role=ModelCallRole.QUERY_TIME,
                seed=17,
                started=NOW + timedelta(seconds=10 + ordinal),
            )
            c2_result = SimpleNamespace(
                call_id=c2_call.call_id,
                condition=ConditionName.C2_LLM_QUERY,
                outcome=RunOutcome.SUCCEEDED,
                request_started=True,
                request_hash=c2_request_hash,
                output_artifact_hash=digest(f"TEST-ONLY-c2-output-{ordinal}"),
                validation_artifact_hash=digest(f"TEST-ONLY-c2-validation-{ordinal}"),
                ledger_receipt_hash=digest(f"TEST-ONLY-c2-ledger-{ordinal}"),
                evidence_packet_hash=evidence_packet.content_hash,
                horizon_hash=query_stage.horizon_hash,
                budget_hash=query_stage.budget_hash,
                construction_certificate_hash=(c2_projection.construction_certificate.content_hash),
                empty_prequery_inventory_hash=c2_projection.pre_query_inventory.content_hash,
                allocated_gpu_seconds=1,
                repair_attempts=0,
                completed_at=NOW + timedelta(seconds=30),
            )
            c2_itt = SimpleNamespace(
                content_hash=digest(f"TEST-ONLY-c2-itt-{ordinal}"),
                call_spec_hash=c2_call_hash,
                result=c2_result,
            )
            itt_records.append(c2_itt)
            export_counter += 1
            c2_export = Phase5HeldOutProjectionExport(
                export_id=f"TEST-ONLY-export-{export_counter}",
                condition=ConditionName.C2_LLM_QUERY,
                unit_id=unit_id,
                query_stage_hash=query_hash,
                seed_block=1,
                itt_record_hash=c2_itt.content_hash,
                source_output_artifact_hash=c2_result.output_artifact_hash,
                source_condition_output_artifact_hash=c2_result.output_artifact_hash,
                source_validation_artifact_hash=c2_result.validation_artifact_hash,
                source_ledger_receipt_hash=c2_result.ledger_receipt_hash,
                source_model_calls=(c2_binding,),
                projection=c2_projection_ref,
                packet=packet_ref,
                snapshot=snapshot_ref,
                source_completed_at=NOW + timedelta(seconds=30),
                exported_at=NOW + timedelta(seconds=31),
            )

            effective_instruction = template.instruction
            rebound_receipt = None
            if not hasattr(template, "cpu_inputs"):
                rebound_bundle = build_visualization_bundle(
                    c2_projection,
                    template.instruction.before_context,
                    evidence_packet,
                    content_scope=VisualizationContentScope.REGISTERED_DISPLAY,
                )
                submission = RevisionDraftSubmission.model_validate_json(
                    template.submission_receipt.submission_canonical_json
                )
                effective_instruction = _compile_submission(
                    submission,
                    rebound_bundle,
                    created_at=template.submission_receipt.requested_at,
                )
                instruction_bytes = (
                    effective_instruction.to_canonical_json() + "\n"
                ).encode("utf-8")
                receipt_payload = _without_hashes(
                    template.submission_receipt.model_dump(mode="python")
                )
                receipt_payload.update(
                    before_projection_hash=c2_projection.content_hash,
                    before_bundle_hash=rebound_bundle.content_hash,
                    instruction_hash=effective_instruction.content_hash,
                    instruction_file_sha256=hashlib.sha256(instruction_bytes).hexdigest(),
                )
                rebound_receipt = template.submission_receipt.__class__.model_validate(
                    receipt_payload
                )
            instruction_ref = cas.reference(effective_instruction, "revision_instruction")
            c2_export_ref = cas.reference(c2_export, "held_out_projection_export")
            source_values = dict(
                episode_id=template.episode_id,
                context_id=context.context_id,
                kind=(
                    FeedbackEpisodeKind.SCRIPTED_KNOWN_ANSWER
                    if hasattr(template, "cpu_inputs")
                    else FeedbackEpisodeKind.RESEARCHER_TRACE
                ),
                unit_id=unit_id,
                query_stage_hash=query_hash,
                instruction=instruction_ref,
                c2_export=c2_export_ref,
            )
            if hasattr(template, "cpu_inputs"):
                condition_values = []
                cpu_references = []
                for condition, cpu_template in zip(
                    (ConditionName.C0_CLASSICAL_PRE, ConditionName.C1_LLM_PRE),
                    template.cpu_inputs,
                    strict=True,
                ):
                    before = _rebind_projection(
                        cpu_template.before_projection,
                        snapshot=snapshot,
                        evidence_packet=evidence_packet,
                    )
                    after = (
                        None
                        if cpu_template.after_projection is None
                        else _rebind_projection(
                            cpu_template.after_projection,
                            snapshot=snapshot,
                            evidence_packet=evidence_packet,
                        )
                    )
                    cpu_payload = _without_hashes(cpu_template.model_dump(mode="python"))
                    cpu_payload.update(
                        before_projection=before,
                        after_projection=after,
                        source_query_stage_hash=query_hash,
                    )
                    cpu = cpu_template.__class__.model_validate(cpu_payload)
                    projection_ref = cas.reference(before, "ontology_projection")
                    receipt_hash = digest(
                        f"TEST-ONLY-{condition.value}-projection-receipt-{ordinal}"
                    )
                    projection_output = digest(
                        f"TEST-ONLY-{condition.value}-projection-output-{ordinal}"
                    )
                    projection_receipts.append(
                        SimpleNamespace(
                            content_hash=receipt_hash,
                            condition=condition,
                            unit_id=unit_id,
                            query_stage_hash=query_hash,
                            seed_block=(None if condition is ConditionName.C0_CLASSICAL_PRE else 1),
                            outcome=RunOutcome.SUCCEEDED,
                            projection_artifact_hash=projection_output,
                            evidence_packet_hash=evidence_packet.content_hash,
                            horizon_hash=query_stage.horizon_hash,
                            source_construction_seal_hash=(before.construction_seal.content_hash),
                            completed_at=NOW + timedelta(seconds=25),
                        )
                    )
                    model_values = {}
                    if condition is ConditionName.C1_LLM_PRE:
                        c1_request_hash = digest(f"TEST-ONLY-c1-request-{ordinal}")
                        c1_call_hash = digest(f"TEST-ONLY-c1-call-{ordinal}")
                        c1_call = SimpleNamespace(
                            content_hash=c1_call_hash,
                            call_id=f"TEST-ONLY-c1-{ordinal}",
                            call_class="test_c1",
                            unit_id=unit_id,
                            seed_block=1,
                            frozen_seed=protocol.llm_seed,
                            vllm_seed=17,
                            query_stage_hash=None,
                        )
                        calls.append(c1_call)
                        c1_binding = _source_model_call(
                            artifacts=artifacts,
                            label=f"c1-{ordinal}",
                            request_hash=c1_request_hash,
                            role=ModelCallRole.PREBUILD,
                            seed=17,
                            started=NOW + timedelta(seconds=ordinal),
                        )
                        c1_result = SimpleNamespace(
                            call_id=c1_call.call_id,
                            condition=ConditionName.C1_LLM_PRE,
                            outcome=RunOutcome.SUCCEEDED,
                            request_started=True,
                            request_hash=c1_request_hash,
                            output_artifact_hash=digest(f"TEST-ONLY-c1-condition-output-{ordinal}"),
                            validation_artifact_hash=digest(f"TEST-ONLY-c1-validation-{ordinal}"),
                            ledger_receipt_hash=digest(f"TEST-ONLY-c1-ledger-{ordinal}"),
                            evidence_packet_hash=None,
                            construction_seal_hash=before.construction_seal.content_hash,
                            complete_c1_graph_hash=before.construction_seal.ontology_hash,
                            allocated_gpu_seconds=1,
                            repair_attempts=0,
                            completed_at=NOW + timedelta(seconds=20),
                        )
                        c1_itt = SimpleNamespace(
                            content_hash=digest(f"TEST-ONLY-c1-itt-{ordinal}"),
                            call_spec_hash=c1_call_hash,
                            result=c1_result,
                        )
                        itt_records.append(c1_itt)
                        model_values = dict(
                            itt_record_hash=c1_itt.content_hash,
                            source_condition_output_artifact_hash=(c1_result.output_artifact_hash),
                            source_validation_artifact_hash=(c1_result.validation_artifact_hash),
                            source_ledger_receipt_hash=c1_result.ledger_receipt_hash,
                            source_model_calls=(c1_binding,),
                        )
                    export_counter += 1
                    export = Phase5HeldOutProjectionExport(
                        export_id=f"TEST-ONLY-export-{export_counter}",
                        condition=condition,
                        unit_id=unit_id,
                        query_stage_hash=query_hash,
                        seed_block=(None if condition is ConditionName.C0_CLASSICAL_PRE else 1),
                        projection_receipt_hash=receipt_hash,
                        source_output_artifact_hash=projection_output,
                        projection=projection_ref,
                        packet=packet_ref,
                        snapshot=snapshot_ref,
                        source_completed_at=(NOW + timedelta(seconds=25)),
                        exported_at=NOW + timedelta(seconds=31),
                        **model_values,
                    )
                    condition_values.append(cas.reference(export, "held_out_projection_export"))
                    cpu_references.append(cas.reference(cpu, "cpu_reprojection_input"))
                freeze = ScriptedRevisionFreeze(
                    episode_id=template.episode_id,
                    context_id=context.context_id,
                    action=template.instruction.revision.action,
                    instruction_hash=template.instruction.content_hash,
                    anchor_hashes=tuple(
                        item.content_hash for item in template.instruction.revision.anchors
                    ),
                    frozen_at=NOW - timedelta(seconds=1),
                )
                freeze_reference = cas.reference(freeze, "scripted_revision_freeze")
                source_values.update(
                    scripted_freeze=freeze_reference,
                    c0_export=condition_values[0],
                    c1_export=condition_values[1],
                    c0_cpu_input=cpu_references[0],
                    c1_cpu_input=cpu_references[1],
                )
                commitment_entries.append(
                    Phase5ScriptCommitmentEntry(
                        episode_id=template.episode_id,
                        context_id=context.context_id,
                        action=template.instruction.revision.action,
                        query_stage_relative_path=query_relative_path,
                        query_stage_manifest_hash=query_hash,
                        query_stage_manifest_file_sha256=query_manifest_file_sha256,
                        query_artifact_hash=query_stage.query_artifact_hash,
                        model_visible_evidence_hash=query_stage.evidence_artifact_hash,
                        instruction=instruction_ref,
                        freeze=freeze_reference,
                        scheduled_activation_at=template.instruction.revision.created_at,
                    )
                )
            else:
                assert rebound_receipt is not None
                source_values["trace_submission_receipt"] = cas.reference(
                    rebound_receipt,
                    "researcher_trace_submission_receipt",
                )
            episode_sources.append(Phase5EpisodeSource(**source_values))

        call_manifest = SimpleNamespace(
            content_hash=digest("TEST-ONLY-call-manifest"),
            units=tuple(units),
            calls=tuple(calls),
        )
        call_manifest_ref = cas.reference(call_manifest, "held_out_call_manifest")
        review_completion_hash = digest("TEST-ONLY-independent-review-completion")
        execution = SimpleNamespace(
            content_hash=digest("TEST-ONLY-heldout-execution"),
            call_manifest_hash=call_manifest.content_hash,
            review_completion_manifest_hash=review_completion_hash,
            final_reviewed_seal_hash=gate_template.final_reviewed_seal_hash,
            preconstructed_projections=tuple(projection_receipts),
            itt_records=tuple(itt_records),
        )
        bridge = SimpleNamespace(
            content_hash=digest("TEST-ONLY-scorer-bridge"),
            authorized_at=NOW + timedelta(seconds=90),
            output_artifact_hashes=tuple(
                {
                    export.source_output_artifact_hash
                    for export in cas.values.values()
                    if isinstance(export, Phase5HeldOutProjectionExport)
                }
                | {
                    export.source_condition_output_artifact_hash
                    for export in cas.values.values()
                    if isinstance(export, Phase5HeldOutProjectionExport)
                    and export.source_condition_output_artifact_hash is not None
                }
            ),
            projection_receipt_hashes=tuple(item.content_hash for item in projection_receipts),
            itt_record_hashes=tuple(item.content_hash for item in itt_records),
        )
        gate = gate_template.model_copy(
            update={"held_out_execution_manifest_hash": execution.content_hash}
        )
        scorer = Phase5ScorerBindingAuthorization(
            authorization_id="TEST-ONLY-phase5-scorer-authorization",
            protocol_hash=protocol.content_hash,
            scorer_bridge_hash=bridge.content_hash,
            known_answer_source_hash=digest("TEST-ONLY-known-answer-source"),
            review_completion_manifest_hash=review_completion_hash,
            final_reviewed_seal_hash=gate.final_reviewed_seal_hash,
            bindings=tuple(
                binding
                for script in input_template.scripted_inputs
                for binding in script.scorer_bindings
            ),
            authorized_at=NOW + timedelta(seconds=100),
        )
        scorer_ref = cas.reference(scorer, "scorer_binding_authorization")
        commitment = Phase5ScriptCommitmentManifest(
            commitment_id="TEST-ONLY-phase5-script-commitment",
            protocol_hash=protocol.content_hash,
            benchmark_draft_seal_hash=protocol.benchmark_draft_seal_hash,
            review_gate=Phase5ReviewGateBinding(
                review_completion_manifest_hash=review_completion_hash,
                review_completion_manifest_file_sha256=digest(
                    "TEST-ONLY-independent-review-completion-file"
                ),
                final_reviewed_seal_hash=gate.final_reviewed_seal_hash,
                final_reviewed_seal_file_sha256=digest(
                    "TEST-ONLY-final-reviewed-seal-file"
                ),
                review_draft_seal_hash=protocol.benchmark_draft_seal_hash,
            ),
            parent_readiness_floor=Phase5ParentReadinessFloor(
                held_out_call_manifest_hash=call_manifest.content_hash,
                held_out_control_configuration_hash=digest(
                    "TEST-ONLY-held-out-control"
                ),
                held_out_runtime_binding_hash=digest(
                    "TEST-ONLY-held-out-runtime-binding"
                ),
                selected_c1_parent_call_ordinals=(1, 2, 3, 4, 5, 6),
                selected_c2_parent_call_ordinals=(25, 26, 27, 28, 29, 30),
                base_call_watchdog_seconds=1,
                repair_reserve_watchdog_seconds=1,
                service_start_watchdog_seconds=1,
                registered_watchdog_floor_seconds=3,
            ),
            draft_manifest_hash=digest("TEST-ONLY-phase5-script-draft"),
            entries=tuple(commitment_entries),
            committed_at=NOW - timedelta(seconds=1),
            scheduled_activation_at=NOW + timedelta(seconds=60),
            activation_delay_seconds=61,
        )
        source = Phase5MaterializationSourceManifest(
            source_id="TEST-ONLY-phase5-source",
            run_id="TEST-ONLY-phase5-materialized-run",
            protocol_hash=protocol.content_hash,
            script_commitment_manifest_hash=commitment.content_hash,
            primary_results_gate_hash=gate.content_hash,
            final_reviewed_seal_hash=gate.final_reviewed_seal_hash,
            held_out_execution_manifest_hash=execution.content_hash,
            held_out_call_manifest=call_manifest_ref,
            scorer_binding_authorization=scorer_ref,
            episodes=tuple(episode_sources),
            captured_at=NOW + timedelta(seconds=120),
        )
        monkeypatch.setattr(
            "story_projection_onto.phase5_production.replay_phase5_prerequisites",
            lambda **_kwargs: (gate, execution, bridge),
        )
        monkeypatch.setattr(
            "story_projection_onto.phase5_commitment.load_phase5_script_commitment",
            lambda _path: commitment,
        )
        inputs, receipt = materialize_phase5_input_manifest(
            source=source,
            protocol=protocol,
            script_commitment_path=tmp_path / "TEST-ONLY-commitment.json",
            primary_results_gate_path=tmp_path / "TEST-ONLY-gate.json",
            benchmark_root=tmp_path,
            review_completion_root=tmp_path,
            cas=cas,
        )
        assert len(inputs.scripted_inputs) == 6
        assert len(inputs.researcher_trace_inputs) == 3
        assert inputs.source_manifest_hash == source.content_hash
        assert receipt.input_manifest_hash == inputs.content_hash
        assert receipt.script_commitment_manifest_hash == commitment.content_hash
        assert receipt.scorer_gold_read is False
        assert receipt.model_service_called is False
        assert all(item.held_out_seed_block == 1 for item in receipt.episodes)
        validate_phase5_materialization_receipt(inputs, receipt)
        tampered = receipt.model_copy(update={"episodes": receipt.episodes[:-1]})
        with pytest.raises(Phase5MaterializationError, match="authoritative"):
            validate_phase5_materialization_receipt(inputs, tampered)
        wrong_commitment_source = source.model_copy(
            update={"script_commitment_manifest_hash": digest("wrong-commitment")}
        )
        with pytest.raises(Phase5MaterializationError, match="pre-output script commitment"):
            materialize_phase5_input_manifest(
                source=wrong_commitment_source,
                protocol=protocol,
                script_commitment_path=tmp_path / "TEST-ONLY-commitment.json",
                primary_results_gate_path=tmp_path / "TEST-ONLY-gate.json",
                benchmark_root=tmp_path,
                review_completion_root=tmp_path,
                cas=cas,
            )
        tampered_commitment_payload = _without_hashes(
            commitment.model_dump(mode="python")
        )
        tampered_commitment_payload["entries"][0]["instruction"] = (
            tampered_commitment_payload["entries"][1]["instruction"]
        )
        tampered_commitment = Phase5ScriptCommitmentManifest.model_validate(
            tampered_commitment_payload
        )
        monkeypatch.setattr(
            "story_projection_onto.phase5_commitment.load_phase5_script_commitment",
            lambda _path: tampered_commitment,
        )
        tampered_source = source.model_copy(
            update={"script_commitment_manifest_hash": tampered_commitment.content_hash}
        )
        with pytest.raises(Phase5MaterializationError, match="source lineage changed"):
            materialize_phase5_input_manifest(
                source=tampered_source,
                protocol=protocol,
                script_commitment_path=tmp_path / "TEST-ONLY-commitment.json",
                primary_results_gate_path=tmp_path / "TEST-ONLY-gate.json",
                benchmark_root=tmp_path,
                review_completion_root=tmp_path,
                cas=cas,
            )
    finally:
        ledger.close()
