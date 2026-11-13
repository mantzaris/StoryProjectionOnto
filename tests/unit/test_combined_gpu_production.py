from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path

import pytest

import story_projection_onto.combined_gpu_production as production
from story_projection_onto.combined_gpu_block import (
    CombinedCallClass,
    CombinedCallManifest,
    CombinedCallSpec,
    CombinedQuerySource,
    OneSwitchFingerprint,
    RestrictedArtifactPointer,
)
from story_projection_onto.combined_gpu_production import (
    CombinedCallSlot,
    CombinedJournal,
    CombinedPreparedCall,
    CombinedProductionError,
    CombinedRepairClaim,
    CombinedServiceCallResult,
    CombinedServiceIdentity,
    _RepairAuthority,
    persist_paraphrase_reveal,
    validate_prepared_call,
)
from story_projection_onto.conditions.base import ConditionAttemptRecord, ConditionPreparation
from story_projection_onto.contracts import (
    ConditionName,
    ConstructionCapabilities,
    ConstructionRequest,
    EvidencePacket,
    ModelVisibleEvidencePacket,
    PreQueryInventory,
    ReleaseClass,
    RunOutcome,
    RuntimeIdentifiers,
    canonical_sha256,
    to_model_visible_packet,
)
from story_projection_onto.development_adapter import (
    DEFAULT_DEVELOPMENT_CONSTRUCTION_CONFIG,
    DevelopmentConstructionConfiguration,
)
from story_projection_onto.phase5_production import (
    Phase5OwnedServiceIdentity,
    build_metered_phase5_adapter,
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
    PrequeryBarrierRecord,
    RetryClass,
)
from tests.unit.test_combined_gpu_block import combined_fixture, digest
from tests.unit.test_ui import NOW, packet

ROOT = Path(__file__).resolve().parents[2]


def _rebuild(value, **updates):
    payload = value.model_dump(mode="python", exclude={"content_hash"})
    payload.update(updates)
    return type(value).model_validate(payload)


def _artifacts(tmp_path: Path) -> ArtifactStore:
    ledger = Ledger(tmp_path / "ledger.sqlite3")
    blobs = BlobStore(tmp_path / "cas", compression=Compression.GZIP)
    return ArtifactStore(blobs, ledger)


def _identity(runtime, *, at=NOW) -> CombinedServiceIdentity:
    phase5 = Phase5OwnedServiceIdentity(
        service_id="test-combined-service",
        global_accounting_id="test-global-accounting",
        model_manifest_hash=runtime.model_manifest_hash,
        decoding_manifest_hash=runtime.decoding_manifest_hashes[ConditionName.C2_LLM_QUERY],
        model_load_event_id="test-load",
        model_load_event_record_hash=digest("test-load-row"),
        cumulative_gpu_seconds_at_handoff=1.0,
        handed_off_at=at,
    )
    return CombinedServiceIdentity(
        service_id="test-combined-service",
        global_accounting_id="test-global-accounting",
        activation_slot_hash=digest("activation-slot"),
        runtime_binding_hash=runtime.content_hash,
        model_manifest_hash=runtime.model_manifest_hash,
        decoding_manifest_hashes=runtime.decoding_manifest_hashes,
        model_load_event_id="test-load",
        model_load_event_record_hash=digest("test-load-row"),
        service_pid=1234,
        service_start_ticks=5678,
        gpu_seconds_before_load=0.0,
        cumulative_gpu_seconds_after_load=1.0,
        activated_at=at,
        phase5_identity=phase5,
    )


def _prepared(call, runtime, *, outcome_context=None):
    visible_packet = to_model_visible_packet(packet())
    visible_packet = ModelVisibleEvidencePacket(
        packet_hash=call.source.packet_hash,
        evidence=visible_packet.evidence,
        ordered_evidence_ids=visible_packet.ordered_evidence_ids,
        retrieval_method=visible_packet.retrieval_method,
    )
    request = ConstructionRequest(
        request_id=f"request-{call.call_id}",
        condition=call.condition,
        snapshot_hash=call.source.prequery_stage.snapshot_hash,
        packet=visible_packet,
        context=call.model_visible_context(),
        upper_ontology=DevelopmentConstructionConfiguration.load(
            ROOT / DEFAULT_DEVELOPMENT_CONSTRUCTION_CONFIG
        ).upper_ontology,
        budgets=call.source.context.budgets,
        capabilities=(
            ConstructionCapabilities.active_without_temporal_epistemic()
            if call.condition is ConditionName.A_NO_TEMPORAL_EPISTEMIC
            else ConstructionCapabilities.active_construction()
        ),
        runtime=RuntimeIdentifiers(
            model_id=runtime.served_model_name,
            model_revision=runtime.model_revision,
            tokenizer_hash=runtime.tokenizer_manifest_hash,
            runtime_version=runtime.runtime_version,
            prompt_hash=runtime.prompt_hashes[call.condition],
            output_schema_hash=runtime.output_schema_hashes[call.condition],
            decoding_config_hash=runtime.decoding_manifest_hashes[call.condition],
        ),
        requested_at=call.source.query_accessed_at + timedelta(seconds=2),
    )
    lineage = call.source.preparation_for(call.condition)
    common = dict(
        model_manifest_hash=runtime.model_manifest_hash,
        model_revision=runtime.model_revision,
        tokenizer_hash=runtime.tokenizer_manifest_hash,
        packet_hash=call.source.packet_hash,
        ordered_evidence_hash=canonical_sha256(request.packet.ordered_evidence_ids),
        horizon_hash=call.source.context.spoiler_horizon.content_hash,
        upper_ontology_hash=runtime.upper_ontology_hash,
        budgets_hash=call.source.context.budgets.content_hash,
        seed_manifest_hash=runtime.seed_manifest_hash,
        seed_block=1,
        vllm_seed=call.vllm_seed,
        decoding_family_hash=runtime.decoding_family_hash,
        maximum_input_tokens=runtime.maximum_input_tokens,
        maximum_output_tokens=runtime.maximum_output_tokens,
        repair_attempt_budget=1,
        repair_policy_hash=runtime.repair_policy_hash,
        validator_hash=runtime.validator_hash,
    )
    baseline = OneSwitchFingerprint(**common)
    switch = call.configuration_delta.switch_name
    ablated = OneSwitchFingerprint(**common, **{switch: call.configuration_delta.ablated_value})
    return CombinedPreparedCall(
        call_spec_hash=call.content_hash,
        condition=call.condition,
        request=request,
        preparation_hash=lineage.preparation_hash,
        empty_inventory_hash=lineage.inventory_hash,
        prequery_barrier_hash=call.source.prequery_barrier_hash,
        query_access_event_hash=call.source.query_access_event_hash,
        packet_artifact=call.source.packet_artifact,
        baseline_fingerprint=baseline,
        ablated_fingerprint=ablated,
        prepared_at=request.requested_at,
    )


def test_prepared_no_context_is_gold_free_and_exactly_one_switch() -> None:
    *_, runtime, _gate, manifest = combined_fixture()
    call = next(
        item for item in manifest.calls if item.call_class is CombinedCallClass.ABLATION_NO_CONTEXT
    )
    prepared = _prepared(call, runtime)
    validate_prepared_call(call=call, prepared=prepared, runtime=runtime)
    payload = prepared.model_visible_payload()
    serialized = str(payload).casefold()
    assert "scorer" not in serialized
    assert "gold" not in serialized
    assert "spoiler_horizon" not in payload["context"]
    assert set(payload["context"]) == {
        "schema_version",
        "content_hash",
        "generic_request",
        "budgets",
    }


def test_prepared_ablation_rejects_c2_prequery_substitution() -> None:
    *_, runtime, _gate, manifest = combined_fixture()
    call = next(
        item
        for item in manifest.calls
        if item.call_class is CombinedCallClass.ABLATION_NO_RARE_GUARD
    )
    prepared = _prepared(call, runtime)
    changed = prepared.model_copy(
        update={"preparation_hash": call.source.c2_seed1_preparation_hash}
    )
    with pytest.raises(CombinedProductionError, match="condition-matching"):
        validate_prepared_call(call=call, prepared=changed, runtime=runtime)


def test_repair_authority_is_one_use_and_global_pool_never_replenishes(
    tmp_path: Path,
) -> None:
    *_, manifest = combined_fixture()
    call = manifest.calls[21]
    artifacts = _artifacts(tmp_path)
    journal = CombinedJournal(tmp_path / "journal")
    journal.initialize()
    authority = _RepairAuthority(
        journal=journal,
        call=call,
        request_hash=digest("request"),
        consumed_before=3,
        artifacts=artifacts,
        clock=lambda: NOW,
    )
    first = authority.claim(
        request_hash=digest("request"),
        base_model_call_id="base-call",
        claimed_at=NOW,
    )
    replay = authority.claim(
        request_hash=digest("request"),
        base_model_call_id="base-call",
        claimed_at=NOW + timedelta(seconds=1),
    )
    assert first == replay
    assert first.global_slot_number == 4
    second = _RepairAuthority(
        journal=journal,
        call=manifest.calls[22],
        request_hash=digest("second-request"),
        consumed_before=3,
        artifacts=artifacts,
        clock=lambda: NOW,
    )
    with pytest.raises(CombinedProductionError, match="exhausted"):
        second.claim(
            request_hash=digest("second-request"),
            base_model_call_id="second-base",
            claimed_at=NOW,
        )


def test_append_only_journal_rejects_tamper_and_symlink(tmp_path: Path) -> None:
    journal = CombinedJournal(tmp_path / "journal")
    journal.initialize()
    claim = CombinedRepairClaim(
        claim_id="claim",
        call_id="call",
        call_spec_hash=digest("spec"),
        request_hash=digest("request"),
        base_model_call_id="base",
        global_slot_number=1,
        claimed_at=NOW,
    )
    target = Path("repair_claims") / f"{claim.request_hash}.json"
    journal.append(target, claim)
    journal.append(target, claim)
    (journal.root / target).write_text("{}")
    with pytest.raises(CombinedProductionError, match="append-only"):
        journal.append(target, claim)
    linked = tmp_path / "linked"
    linked.symlink_to(journal.root, target_is_directory=True)
    with pytest.raises(CombinedProductionError, match="symlinked"):
        CombinedJournal(linked)


def test_atomic_journal_recovers_complete_temp_and_rejects_truncation(
    tmp_path: Path,
) -> None:
    journal = CombinedJournal(tmp_path / "journal")
    journal.initialize()
    claim = CombinedRepairClaim(
        claim_id="claim",
        call_id="call",
        call_spec_hash=digest("spec"),
        request_hash=digest("request"),
        base_model_call_id="base",
        global_slot_number=1,
        claimed_at=NOW,
    )
    relative = Path("repair_claims") / f"{claim.request_hash}.json"
    target = journal.root / relative
    target.parent.mkdir(parents=True)
    payload = (claim.to_canonical_json() + "\n").encode()
    temporary = production._atomic_temporary_path(target, payload)
    temporary.write_bytes(payload)
    journal.audit()
    journal.append(relative, claim)
    assert target.read_bytes() == payload
    assert not temporary.exists()

    broken_target = journal.root / "repair_claims" / f"{digest('broken')}.json"
    broken_payload = b"complete-payload"
    broken_temp = production._atomic_temporary_path(broken_target, broken_payload)
    broken_temp.write_bytes(b"truncated")
    with pytest.raises(CombinedProductionError, match="truncated atomic"):
        journal.audit()


def test_capacity_gate_uses_live_allocation_not_only_closed_sqlite_events(
    tmp_path: Path,
) -> None:
    configuration, *_rest = combined_fixture()
    artifacts = _artifacts(tmp_path)

    class LiveService:
        actual_allocated_service_seconds = 32_300.0

    with pytest.raises(CombinedProductionError, match="scheduled limit"):
        production._require_combined_capacity(
            artifacts=artifacts,
            configuration=configuration,
            next_watchdog_seconds=150,
            remaining_mandatory_p95_seconds=0,
            service=LiveService(),
        )


def test_paraphrase_reveal_is_post_freeze_durable_and_reuses_packet_once(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    *_, original_manifest = combined_fixture()
    original_call = original_manifest.calls[0]
    artifacts = _artifacts(tmp_path)
    evidence_packet = packet()
    packet_artifact = artifacts.put_bytes(
        (evidence_packet.to_canonical_json() + "\n").encode(),
        media_type="application/vnd.story-projection.evidence-packet+json",
        release_class=ReleaseClass.PUBLIC,
        created_at=original_manifest.created_at,
    )
    barrier_artifact = artifacts.put_bytes(
        b'{"TEST_ONLY":"combined prequery barrier"}\n',
        media_type="application/vnd.story-projection.prequery-barrier+json",
        release_class=ReleaseClass.PUBLIC,
        created_at=original_call.source.prequery_barrier_sealed_at,
    )
    artifacts.ledger.persist_prequery_barrier(
        PrequeryBarrierRecord(
            barrier_hash=original_call.source.prequery_barrier_hash,
            barrier_id="test-combined-prequery-barrier",
            execution_id=original_call.source.execution_id,
            execution_manifest_hash=original_call.source.held_out_execution_manifest_hash,
            barrier_artifact_hash=barrier_artifact.content_hash,
            preparation_count=120,
            sealed_at=original_call.source.prequery_barrier_sealed_at.isoformat(),
            persisted_at=original_call.source.prequery_barrier_sealed_at.isoformat(),
            release_class=ReleaseClass.PUBLIC,
        ),
        barrier_artifact=barrier_artifact,
    )
    source = _rebuild(
        original_call.source,
        packet_hash=evidence_packet.content_hash,
        packet_artifact=production.PublicEvidencePacketPointer(
            artifact_hash=packet_artifact.content_hash,
            logical_content_hash=evidence_packet.content_hash,
        ),
    )
    assert isinstance(source, CombinedQuerySource)
    call = _rebuild(original_call, source=source)
    assert isinstance(call, CombinedCallSpec)
    manifest = _rebuild(
        original_manifest,
        calls=(call, *original_manifest.calls[1:]),
    )
    assert isinstance(manifest, CombinedCallManifest)
    journal = CombinedJournal(tmp_path / "journal")
    journal.initialize()

    first_reveal = manifest.created_at + timedelta(seconds=1)
    original_put = artifacts.put_bytes
    attempts = 0

    def crash_after_intent(*args, **kwargs):
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise RuntimeError("TEST-ONLY crash after durable reveal intent")
        return original_put(*args, **kwargs)

    monkeypatch.setattr(artifacts, "put_bytes", crash_after_intent)
    with pytest.raises(RuntimeError, match="durable reveal intent"):
        persist_paraphrase_reveal(
            call=call,
            manifest=manifest,
            journal=journal,
            artifacts=artifacts,
            revealed_at=first_reveal,
        )
    assert journal.exists(journal.reveal_intent_path(call))
    assert not journal.exists(journal.reveal_path(call))

    receipt = persist_paraphrase_reveal(
        call=call,
        manifest=manifest,
        journal=journal,
        artifacts=artifacts,
        revealed_at=first_reveal + timedelta(hours=1),
    )
    assert receipt.revealed_at == first_reveal
    assert receipt.packet_hash == evidence_packet.content_hash
    access = artifacts.ledger.get_query_access(receipt.access_event.content_hash)
    materialization = artifacts.ledger.get_packet_materialization(
        receipt.packet_reuse_event.content_hash
    )
    assert datetime.fromisoformat(access.accessed_at.replace("Z", "+00:00")) == first_reveal
    assert materialization.packet_artifact_hash == packet_artifact.content_hash
    assert access.release_class.value == "public"
    assert materialization.release_class.value == "public"
    assert (
        artifacts.ledger.get_artifact(packet_artifact.content_hash).release_class.value
        == "public"
    )
    assert (
        persist_paraphrase_reveal(
            call=call,
            manifest=manifest,
            journal=journal,
            artifacts=artifacts,
            revealed_at=first_reveal + timedelta(days=1),
        )
        == receipt
    )

    too_early = CombinedJournal(tmp_path / "too-early")
    too_early.initialize()
    with pytest.raises(CombinedProductionError, match="predates"):
        persist_paraphrase_reveal(
            call=call,
            manifest=manifest,
            journal=too_early,
            artifacts=artifacts,
            revealed_at=manifest.created_at - timedelta(microseconds=1),
        )


def test_prepared_call_binds_public_packet_and_restricted_prequery_lineage(
    tmp_path: Path,
) -> None:
    _configuration, _selections, _sources, runtime, _gate, manifest = combined_fixture()
    original = next(
        item
        for item in manifest.calls
        if item.call_class is CombinedCallClass.ABLATION_NO_RARE_GUARD
    )
    artifacts = _artifacts(tmp_path)
    base_packet = packet()
    packet_payload = base_packet.model_dump(mode="python", exclude={"content_hash"})
    packet_payload["snapshot_hash"] = original.source.prequery_stage.snapshot_hash
    evidence_packet = EvidencePacket.model_validate(packet_payload)
    packet_record = artifacts.put_bytes(
        (evidence_packet.to_canonical_json() + "\n").encode(),
        media_type="application/vnd.story-projection.evidence-packet+json",
        release_class=ReleaseClass.PUBLIC,
        created_at=original.source.query_accessed_at,
    )
    old_lineage = original.source.preparation_for(original.condition)
    inventory = PreQueryInventory(
        inventory_id="test-no-rare-empty-inventory",
        condition=original.condition,
        snapshot_hash=original.source.prequery_stage.snapshot_hash,
        recorded_at=old_lineage.completed_at,
    )
    preparation = ConditionPreparation(
        preparation_id="test-no-rare-preparation",
        condition=original.condition,
        snapshot_hash=original.source.prequery_stage.snapshot_hash,
        completed_at=old_lineage.completed_at,
        empty_inventory=inventory,
    )
    inventory_record = artifacts.put_bytes(
        (inventory.to_canonical_json() + "\n").encode(),
        media_type="application/vnd.story-projection.pre-query-inventory+json",
        release_class=ReleaseClass.RESTRICTED,
        created_at=old_lineage.completed_at,
    )
    preparation_record = artifacts.put_bytes(
        (preparation.to_canonical_json() + "\n").encode(),
        media_type="application/vnd.story-projection.condition-preparation+json",
        release_class=ReleaseClass.RESTRICTED,
        created_at=old_lineage.completed_at,
    )
    lineages = tuple(
        _rebuild(
            item,
            preparation_hash=preparation.content_hash,
            inventory_hash=inventory.content_hash,
            preparation_artifact_hash=preparation_record.content_hash,
            inventory_artifact_hash=inventory_record.content_hash,
        )
        if item.condition is original.condition
        else item
        for item in original.source.ablation_prequery_lineage
    )
    source = _rebuild(
        original.source,
        packet_hash=evidence_packet.content_hash,
        packet_artifact=production.PublicEvidencePacketPointer(
            artifact_hash=packet_record.content_hash,
            logical_content_hash=evidence_packet.content_hash,
        ),
        ablation_prequery_lineage=lineages,
    )
    call = _rebuild(original, source=source)
    prepared = _prepared(call, runtime)
    baseline, ablated = production.derive_one_switch_fingerprints(
        call=call,
        runtime=runtime,
        packet=evidence_packet,
    )
    prepared = _rebuild(
        prepared,
        baseline_fingerprint=baseline,
        ablated_fingerprint=ablated,
    )

    assert production.bind_prepared_call_from_cas(
        call=call,
        prepared=prepared,
        runtime=runtime,
        artifacts=artifacts,
    ) == prepared


def test_phase5_receives_only_lifecycle_free_view_of_same_loaded_service(
    tmp_path: Path,
) -> None:
    configuration, _selections, _sources, runtime, gate, manifest = combined_fixture()
    artifacts = _artifacts(tmp_path)
    started = NOW - timedelta(seconds=2)
    ended = NOW - timedelta(seconds=1)
    event = artifacts.ledger.record_gpu_event(
        event_id="test-combined-shared-load",
        event_kind=GpuEventKind.MODEL_LOAD,
        allocated_seconds=1,
        started_at=started,
        ended_at=ended,
        succeeded=True,
    )
    event_hash = production._record_hash(event, production._GPU_EVENT_FIELDS)
    phase5_identity = Phase5OwnedServiceIdentity(
        service_id="test-combined-shared-service",
        global_accounting_id="test-global-accounting",
        model_manifest_hash=runtime.model_manifest_hash,
        decoding_manifest_hash=runtime.decoding_manifest_hashes[ConditionName.C2_LLM_QUERY],
        model_load_event_id=event.event_id,
        model_load_event_record_hash=event_hash,
        cumulative_gpu_seconds_at_handoff=1,
        handed_off_at=NOW,
    )
    identity = CombinedServiceIdentity(
        service_id=phase5_identity.service_id,
        global_accounting_id=phase5_identity.global_accounting_id,
        activation_slot_hash=digest("phase5-view-slot"),
        runtime_binding_hash=runtime.content_hash,
        model_manifest_hash=runtime.model_manifest_hash,
        decoding_manifest_hashes=runtime.decoding_manifest_hashes,
        model_load_event_id=event.event_id,
        model_load_event_record_hash=event_hash,
        service_pid=1234,
        service_start_ticks=5678,
        gpu_seconds_before_load=0,
        cumulative_gpu_seconds_after_load=1,
        activated_at=NOW,
        phase5_identity=phase5_identity,
    )

    class Service:
        @property
        def actual_allocated_service_seconds(self):
            return artifacts.ledger.gpu_summary().total_allocated_seconds

        def execute_phase5_owned(self, *_args, **_kwargs):
            raise AssertionError("not exercised")

        def recover_phase5_owned(self, _request, _repair_authority, _remaining):
            return None

        def shutdown(self):
            raise AssertionError("lifecycle capability must remain hidden")

    journal = CombinedJournal(tmp_path / "phase5-view")
    journal.initialize()
    calls = tuple(manifest.calls[12:21])
    view = production._CombinedPhase5ServiceView(
        service=Service(),
        identity=identity,
        configuration=configuration,
        manifest=manifest,
        upstream_gate=gate,
        calls=calls,
        journal=journal,
        artifacts=artifacts,
        short_reserve_consumed_before=0,
        clock=lambda: NOW,
    )
    assert view.identity() == identity.phase5_identity
    assert not any(
        hasattr(view, name)
        for name in ("start", "load", "activate", "shutdown", "terminate", "close_service")
    )
    adapter = build_metered_phase5_adapter(service=view, artifacts=artifacts)
    assert adapter.service is view


def test_interrupted_ordinary_slot_recovers_without_resending(
    tmp_path: Path,
    monkeypatch,
) -> None:
    configuration, _selections, _sources, runtime, gate, manifest = combined_fixture()
    call = manifest.calls[21]
    artifacts = _artifacts(tmp_path)
    clock_value = manifest.created_at + timedelta(seconds=10)

    class Clock:
        def __call__(self):
            nonlocal clock_value
            result = clock_value
            clock_value += timedelta(seconds=1)
            return result

    clock = Clock()
    load_started = manifest.created_at + timedelta(seconds=1)
    load_event = artifacts.ledger.record_gpu_event(
        event_id="test-load",
        event_kind=GpuEventKind.MODEL_LOAD,
        allocated_seconds=1,
        started_at=load_started,
        ended_at=load_started + timedelta(seconds=1),
        succeeded=True,
    )
    identity = _identity(runtime, at=load_started + timedelta(seconds=2))

    class Provider:
        def prepare(self, provided_call, reveal, prepared_at):
            assert provided_call == call
            assert reveal is None
            prepared = _prepared(call, runtime)
            request = _rebuild(prepared.request, requested_at=prepared_at)
            return _rebuild(prepared, request=request, prepared_at=prepared_at)

    class Service:
        execute_count = 0
        recover_count = 0
        result = None

        @property
        def actual_allocated_service_seconds(self):
            return artifacts.ledger.gpu_summary().total_allocated_seconds

        def execute_combined(
            self,
            prepared,
            payload,
            repair_authority,
            remaining_required_seconds,
        ):
            del repair_authority
            self.execute_count += 1
            assert "scorer" not in str(payload).casefold()
            assert remaining_required_seconds == production._remaining_mandatory_p95_after_call(
                call=call,
                configuration=configuration,
                manifest=manifest,
                upstream_gate=gate,
                journal=journal,
            )
            assert remaining_required_seconds > 0
            before = artifacts.ledger.gpu_summary().total_allocated_seconds
            started = clock()
            completed = started + timedelta(seconds=2)
            job = artifacts.ledger.create_or_resume_job(
                {"call": call.call_id, "request": prepared.content_hash},
                release_class=ReleaseClass.RESTRICTED,
                created_at=started,
            )
            attempt = artifacts.ledger.record_attempt(
                attempt_id=f"attempt-{call.call_id}",
                job_id=job.job_id,
                attempt_kind=AttemptKind.BASE,
                input_hash=prepared.content_hash,
                config_hash=digest("interrupted-call-config"),
                seed=call.vllm_seed,
                created_at=started,
            )
            event = artifacts.ledger.record_gpu_event(
                event_id=f"event-{call.call_id}",
                event_kind=GpuEventKind.FAILURE,
                allocated_seconds=2,
                started_at=started,
                ended_at=completed,
                succeeded=False,
                job_id=job.job_id,
                attempt_id=attempt.attempt_id,
            )
            model_call = artifacts.ledger.record_model_call(
                model_call_id=f"model-{call.call_id}",
                job_id=job.job_id,
                attempt_id=attempt.attempt_id,
                gpu_event_id=event.event_id,
                backend=ModelBackend.VLLM_GPU,
                call_role=ModelCallRole.QUERY_TIME,
                retry_class=RetryClass.STANDARD,
                model_manifest_hash=runtime.model_manifest_hash,
                decoding_manifest_hash=runtime.decoding_manifest_hashes[call.condition],
                request_hash=prepared.content_hash,
                response_artifact_hash=None,
                construction_unit_hash=call.source.prequery_stage.snapshot_hash,
                served_context_count=1,
                prompt_tokens=100,
                completion_tokens=0,
                allocated_gpu_seconds=2,
                successful=False,
                created_at=completed,
            )
            attempt_value = ConditionAttemptRecord(
                attempt_id=f"condition-{call.call_id}",
                condition=call.condition,
                unit_id=call.source.unit_id,
                seed_block=1,
                outcome=RunOutcome.FAILED,
                failure_code="test_interrupted_controller",
                release_class=ReleaseClass.RESTRICTED,
            )
            stored = artifacts.put_bytes(
                (attempt_value.to_canonical_json() + "\n").encode(),
                media_type="application/vnd.story-projection.condition-attempt+json",
                release_class=ReleaseClass.RESTRICTED,
                created_at=completed,
            )
            self.result = CombinedServiceCallResult(
                call_id=call.call_id,
                call_spec_hash=call.content_hash,
                request_hash=prepared.content_hash,
                semantic_request_hash=prepared.request.content_hash,
                service_identity_hash=identity.content_hash,
                condition=call.condition,
                outcome=RunOutcome.FAILED,
                model_call_ids=(model_call.model_call_id,),
                model_request_hashes=(prepared.content_hash,),
                condition_attempt_artifact=RestrictedArtifactPointer(
                    artifact_hash=stored.content_hash,
                    logical_content_hash=attempt_value.content_hash,
                    object_kind="condition_attempt",
                    media_type=stored.media_type,
                ),
                cumulative_gpu_seconds_before=before,
                cumulative_gpu_seconds_after=(
                    artifacts.ledger.gpu_summary().total_allocated_seconds
                ),
                started_at=started,
                completed_at=completed,
                checked_at=completed,
            )
            raise RuntimeError("TEST-ONLY controller loss after service completion")

        def recover_combined(self, request_hash, _repair_authority, remaining):
            self.recover_count += 1
            assert remaining > 0
            if self.result is not None and self.result.request_hash == request_hash:
                return self.result
            return None

    assert load_event.allocated_seconds == 1
    journal = CombinedJournal(tmp_path / "interrupted")
    journal.initialize()
    service = Service()
    monkeypatch.setattr(production, "bind_prepared_call_from_cas", lambda **_kwargs: None)
    provider = Provider()
    prepared = provider.prepare(call, None, clock())
    # Fault boundary: the outer slot is durable, but the semantic executor has
    # not written its pre-GPU intent.  Recovery must classify this as unissued
    # and execute it exactly once.
    journal.append(
        journal.call_slot_path(call),
        CombinedCallSlot(
            ordinal=call.ordinal,
            call_id=call.call_id,
            call_spec_hash=call.content_hash,
            manifest_hash=manifest.content_hash,
            service_identity_hash=identity.content_hash,
            prepared_call=prepared,
            cumulative_gpu_seconds_before=(
                artifacts.ledger.gpu_summary().total_allocated_seconds
            ),
            created_at=clock(),
        ),
    )
    arguments = dict(
        call=call,
        manifest=manifest,
        configuration=configuration,
        upstream_gate=gate,
        runtime=runtime,
        service=service,
        identity=identity,
        provider=provider,
        journal=journal,
        artifacts=artifacts,
        short_reserve_consumed_before=0,
        clock=clock,
    )
    with pytest.raises(RuntimeError, match="controller loss"):
        production._execute_or_recover_ordinary(**arguments)
    assert journal.exists(journal.call_slot_path(call))
    assert not journal.exists(journal.call_result_path(call))

    recovered = production._execute_or_recover_ordinary(**arguments)
    repeated = production._execute_or_recover_ordinary(**arguments)
    assert repeated == recovered
    assert recovered.outcome is RunOutcome.FAILED
    assert service.execute_count == 1
    assert service.recover_count == 2


@pytest.mark.parametrize(
    ("outcome", "event_kind"),
    (
        (RunOutcome.FAILED, GpuEventKind.FAILURE),
        (RunOutcome.TIMED_OUT, GpuEventKind.TIMEOUT),
    ),
)
def test_failed_or_timed_out_call_is_verified_against_sqlite_and_cas(
    tmp_path: Path,
    outcome: RunOutcome,
    event_kind: GpuEventKind,
) -> None:
    *_, runtime, _gate, manifest = combined_fixture()
    call = manifest.calls[21]
    prepared = _prepared(call, runtime)
    artifacts = _artifacts(tmp_path)
    identity = _identity(runtime)
    started = prepared.request.requested_at + timedelta(seconds=1)
    completed = started + timedelta(seconds=2)
    artifacts.ledger.record_gpu_event(
        event_id="test-load",
        event_kind=GpuEventKind.MODEL_LOAD,
        allocated_seconds=1.0,
        started_at=started - timedelta(seconds=2),
        ended_at=started - timedelta(seconds=1),
        succeeded=True,
    )
    job = artifacts.ledger.create_or_resume_job(
        {"call": call.call_id},
        release_class=ReleaseClass.RESTRICTED,
        created_at=started,
    )
    attempt = artifacts.ledger.record_attempt(
        attempt_id=f"attempt-{call.call_id}",
        job_id=job.job_id,
        attempt_kind=AttemptKind.BASE,
        input_hash=prepared.content_hash,
        config_hash=digest("call-config"),
        seed=call.vllm_seed,
        created_at=started,
    )
    event = artifacts.ledger.record_gpu_event(
        event_id=f"event-{call.call_id}",
        event_kind=event_kind,
        allocated_seconds=2.0,
        started_at=started,
        ended_at=completed,
        succeeded=False,
        job_id=job.job_id,
        attempt_id=attempt.attempt_id,
    )
    model_call = artifacts.ledger.record_model_call(
        model_call_id=f"model-{call.call_id}",
        job_id=job.job_id,
        attempt_id=attempt.attempt_id,
        gpu_event_id=event.event_id,
        backend=ModelBackend.VLLM_GPU,
        call_role=ModelCallRole.QUERY_TIME,
        retry_class=RetryClass.STANDARD,
        model_manifest_hash=runtime.model_manifest_hash,
        decoding_manifest_hash=runtime.decoding_manifest_hashes[call.condition],
        request_hash=prepared.content_hash,
        response_artifact_hash=None,
        construction_unit_hash=call.source.prequery_stage.snapshot_hash,
        served_context_count=1,
        prompt_tokens=100,
        completion_tokens=0,
        allocated_gpu_seconds=2.0,
        successful=False,
        created_at=completed,
    )
    condition_attempt = ConditionAttemptRecord(
        attempt_id=f"condition-{call.call_id}",
        condition=call.condition,
        unit_id=call.source.unit_id,
        seed_block=1,
        outcome=outcome,
        failure_code="test_failure",
        release_class=ReleaseClass.RESTRICTED,
    )
    artifact = artifacts.put_bytes(
        (condition_attempt.to_canonical_json() + "\n").encode(),
        media_type="application/vnd.story-projection.condition-attempt+json",
        release_class=ReleaseClass.RESTRICTED,
        created_at=completed,
    )
    pointer = production.RestrictedArtifactPointer(
        artifact_hash=artifact.content_hash,
        logical_content_hash=condition_attempt.content_hash,
        object_kind="condition_attempt",
        media_type=artifact.media_type,
    )
    slot = CombinedCallSlot(
        ordinal=call.ordinal,
        call_id=call.call_id,
        call_spec_hash=call.content_hash,
        manifest_hash=manifest.content_hash,
        service_identity_hash=identity.content_hash,
        prepared_call=prepared,
        cumulative_gpu_seconds_before=1.0,
        created_at=started,
    )
    result = CombinedServiceCallResult(
        call_id=call.call_id,
        call_spec_hash=call.content_hash,
        request_hash=prepared.content_hash,
        semantic_request_hash=prepared.request.content_hash,
        service_identity_hash=identity.content_hash,
        condition=call.condition,
        outcome=outcome,
        model_call_ids=(model_call.model_call_id,),
        model_request_hashes=(prepared.content_hash,),
        condition_attempt_artifact=pointer,
        cumulative_gpu_seconds_before=1.0,
        cumulative_gpu_seconds_after=3.0,
        started_at=started,
        completed_at=completed,
        checked_at=completed,
    )
    itt = production._verify_service_call(
        call=call,
        slot=slot,
        result=result,
        identity=identity,
        runtime=runtime,
        artifacts=artifacts,
        repair_claim=None,
    )
    assert itt.outcome is outcome
    assert itt.model_calls[0].retry_class == "standard"

    wrong_outcome = RunOutcome.FAILED if outcome is RunOutcome.TIMED_OUT else RunOutcome.TIMED_OUT
    wrong = result.model_copy(update={"outcome": wrong_outcome})
    with pytest.raises(CombinedProductionError, match="condition attempt differs"):
        production._verify_service_call(
            call=call,
            slot=slot,
            result=wrong,
            identity=identity,
            runtime=runtime,
            artifacts=artifacts,
            repair_claim=None,
        )


def test_invalid_output_may_use_exactly_one_claimed_short_repair(
    tmp_path: Path,
) -> None:
    *_, runtime, _gate, manifest = combined_fixture()
    call = manifest.calls[21]
    prepared = _prepared(call, runtime)
    artifacts = _artifacts(tmp_path)
    identity = _identity(runtime)
    base_started = prepared.request.requested_at + timedelta(seconds=1)
    artifacts.ledger.record_gpu_event(
        event_id="test-load",
        event_kind=GpuEventKind.MODEL_LOAD,
        allocated_seconds=1,
        started_at=base_started - timedelta(seconds=2),
        ended_at=base_started - timedelta(seconds=1),
        succeeded=True,
    )
    job = artifacts.ledger.create_or_resume_job(
        {"call": call.call_id, "case": "repair"},
        release_class=ReleaseClass.RESTRICTED,
        created_at=base_started,
    )
    base_attempt = artifacts.ledger.record_attempt(
        attempt_id=f"base-{call.call_id}",
        job_id=job.job_id,
        attempt_kind=AttemptKind.BASE,
        input_hash=prepared.content_hash,
        config_hash=digest("base-config"),
        seed=call.vllm_seed,
        created_at=base_started,
    )
    base_event = artifacts.ledger.record_gpu_event(
        event_id=f"base-event-{call.call_id}",
        event_kind=GpuEventKind.INFERENCE,
        allocated_seconds=2,
        started_at=base_started,
        ended_at=base_started + timedelta(seconds=2),
        succeeded=False,
        job_id=job.job_id,
        attempt_id=base_attempt.attempt_id,
    )
    base_call = artifacts.ledger.record_model_call(
        model_call_id=f"base-model-{call.call_id}",
        job_id=job.job_id,
        attempt_id=base_attempt.attempt_id,
        gpu_event_id=base_event.event_id,
        backend=ModelBackend.VLLM_GPU,
        call_role=ModelCallRole.QUERY_TIME,
        retry_class=RetryClass.STANDARD,
        model_manifest_hash=runtime.model_manifest_hash,
        decoding_manifest_hash=runtime.decoding_manifest_hashes[call.condition],
        request_hash=prepared.content_hash,
        response_artifact_hash=None,
        construction_unit_hash=call.source.prequery_stage.snapshot_hash,
        served_context_count=1,
        prompt_tokens=100,
        completion_tokens=10,
        allocated_gpu_seconds=2,
        successful=False,
        created_at=base_started + timedelta(seconds=2),
    )
    repair_started = base_started + timedelta(seconds=2)
    repair_attempt = artifacts.ledger.record_attempt(
        attempt_id=f"repair-{call.call_id}",
        job_id=job.job_id,
        attempt_kind=AttemptKind.REPAIR,
        input_hash=digest("repair-request"),
        config_hash=digest("repair-config"),
        seed=call.vllm_seed,
        parent_attempt_id=base_attempt.attempt_id,
        created_at=repair_started,
    )
    repair_event = artifacts.ledger.record_gpu_event(
        event_id=f"repair-event-{call.call_id}",
        event_kind=GpuEventKind.REPAIR,
        allocated_seconds=1,
        started_at=repair_started,
        ended_at=repair_started + timedelta(seconds=1),
        succeeded=True,
        job_id=job.job_id,
        attempt_id=repair_attempt.attempt_id,
    )
    repair_call = artifacts.ledger.record_model_call(
        model_call_id=f"repair-model-{call.call_id}",
        job_id=job.job_id,
        attempt_id=repair_attempt.attempt_id,
        gpu_event_id=repair_event.event_id,
        backend=ModelBackend.VLLM_GPU,
        call_role=ModelCallRole.REPAIR,
        retry_class=RetryClass.SHORT,
        model_manifest_hash=runtime.model_manifest_hash,
        decoding_manifest_hash=runtime.decoding_manifest_hashes[call.condition],
        request_hash=digest("repair-request"),
        response_artifact_hash=None,
        construction_unit_hash=call.source.prequery_stage.snapshot_hash,
        served_context_count=1,
        prompt_tokens=110,
        completion_tokens=20,
        allocated_gpu_seconds=1,
        successful=True,
        created_at=repair_started + timedelta(seconds=1),
    )
    invalid = ConditionAttemptRecord(
        attempt_id=f"condition-{call.call_id}",
        condition=call.condition,
        unit_id=call.source.unit_id,
        seed_block=1,
        outcome=RunOutcome.INVALID,
        failure_code="schema_invalid_after_registered_repair",
        release_class=ReleaseClass.RESTRICTED,
    )
    stored = artifacts.put_bytes(
        (invalid.to_canonical_json() + "\n").encode(),
        media_type="application/vnd.story-projection.condition-attempt+json",
        release_class=ReleaseClass.RESTRICTED,
        created_at=repair_started + timedelta(seconds=1),
    )
    slot = CombinedCallSlot(
        ordinal=call.ordinal,
        call_id=call.call_id,
        call_spec_hash=call.content_hash,
        manifest_hash=manifest.content_hash,
        service_identity_hash=identity.content_hash,
        prepared_call=prepared,
        cumulative_gpu_seconds_before=1,
        created_at=base_started,
    )
    result = CombinedServiceCallResult(
        call_id=call.call_id,
        call_spec_hash=call.content_hash,
        request_hash=prepared.content_hash,
        semantic_request_hash=prepared.request.content_hash,
        service_identity_hash=identity.content_hash,
        condition=call.condition,
        outcome=RunOutcome.INVALID,
        model_call_ids=(base_call.model_call_id, repair_call.model_call_id),
        model_request_hashes=(prepared.content_hash, digest("repair-request")),
        condition_attempt_artifact=RestrictedArtifactPointer(
            artifact_hash=stored.content_hash,
            logical_content_hash=invalid.content_hash,
            object_kind="condition_attempt",
            media_type=stored.media_type,
        ),
        cumulative_gpu_seconds_before=1,
        cumulative_gpu_seconds_after=4,
        started_at=base_started,
        completed_at=repair_started + timedelta(seconds=1),
        checked_at=repair_started + timedelta(seconds=1),
    )
    claim = CombinedRepairClaim(
        claim_id="combined-short-repair-1",
        call_id=call.call_id,
        call_spec_hash=call.content_hash,
        request_hash=prepared.content_hash,
        base_model_call_id=base_call.model_call_id,
        global_slot_number=1,
        claimed_at=repair_started,
    )
    itt = production._verify_service_call(
        call=call,
        slot=slot,
        result=result,
        identity=identity,
        runtime=runtime,
        artifacts=artifacts,
        repair_claim=claim,
    )
    assert itt.outcome is RunOutcome.INVALID
    assert tuple(item.retry_class for item in itt.model_calls) == ("standard", "short")
    assert itt.repair_claim_hash == claim.content_hash
    with pytest.raises(CombinedProductionError, match="short slot"):
        production._verify_service_call(
            call=call,
            slot=slot,
            result=result,
            identity=identity,
            runtime=runtime,
            artifacts=artifacts,
            repair_claim=None,
        )


def test_writable_containment_rejects_escape_and_symlinked_ancestor(
    tmp_path: Path,
) -> None:
    restricted = tmp_path / "artifacts" / "restricted"
    restricted.mkdir(parents=True)
    with pytest.raises(CombinedProductionError, match="outside"):
        production._require_contained_writable_path(
            tmp_path / "artifacts" / "public" / "leak.sqlite3",
            authority_root=restricted,
            label="ledger",
        )

    outside = tmp_path / "outside"
    outside.mkdir()
    (restricted / "redirect").symlink_to(outside, target_is_directory=True)
    with pytest.raises(CombinedProductionError, match="symlinked writable ancestor"):
        production._require_contained_writable_path(
            restricted / "redirect" / "nested" / "record.json",
            authority_root=restricted,
            label="runtime record",
        )


def test_frozen_output_derives_repository_and_separates_public_root(
    tmp_path: Path,
) -> None:
    configuration, _selections, _sources, _runtime, _gate, _manifest = combined_fixture()
    restricted_output = tmp_path / configuration.output_root
    assert (
        production._repository_from_frozen_output(configuration, restricted_output)
        == tmp_path
    )
    with pytest.raises(CombinedProductionError, match="frozen repository path"):
        production._repository_from_frozen_output(
            configuration,
            tmp_path / "arbitrary" / "combined_gpu_block",
        )
    with pytest.raises(CombinedProductionError, match="outside"):
        production._require_contained_writable_path(
            restricted_output,
            authority_root=tmp_path / "artifacts" / "public",
            label="public summary",
        )
