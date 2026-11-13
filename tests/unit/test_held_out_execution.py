from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

import story_projection_onto.held_out_execution as held_out_execution
import story_projection_onto.held_out_primary as held_out
from story_projection_onto.contracts import ConditionName, RunOutcome, canonical_sha256
from story_projection_onto.development_adapter import (
    DEFAULT_DEVELOPMENT_CONSTRUCTION_CONFIG,
    DevelopmentConstructionConfiguration,
)
from story_projection_onto.gpu_runtime import (
    FALLBACK_MODEL_REPOSITORY,
    FALLBACK_MODEL_REVISION,
    GenerationResult,
    TokenizerManifest,
)
from story_projection_onto.held_out_execution import (
    FrozenHeldOutSemanticExecutor,
    HeldOutSemanticExecutionError,
    HeldOutSemanticIntent,
)
from story_projection_onto.held_out_factory import (
    HeldOutFactoryError,
    _canonical_restricted_root,
    _load_held_out_neutral_worlds,
    _registered_phase_three_storage_preflight,
    _require_registered_phase_three_storage_preflight,
    _restricted_descendant,
    create_frozen_production_held_out_bundle,
)
from story_projection_onto.held_out_primary import (
    HeldOutCallEnvelope,
    ReviewedHeldOutPlan,
    load_held_out_control_configuration,
)
from story_projection_onto.store import (
    ArtifactStore,
    BlobStore,
    Compression,
    GpuEventKind,
    JobState,
    Ledger,
    ReleaseClass,
    SemanticAssessmentScope,
    StoragePreflight,
)

ROOT = Path(__file__).resolve().parents[2]
NOW = datetime(2026, 9, 4, tzinfo=UTC)


class _UnusedService:
    @property
    def actual_allocated_service_seconds(self) -> float:
        return 0.0

    def generate(self, *_args, **_kwargs):
        raise AssertionError("an interrupted intent must never be resent")


class _UnusedTokenizer:
    def apply_chat_template(self, _conversation, **_kwargs):
        return []

    def encode(self, _text, **_kwargs):
        return []


class _InvalidRecordingService:
    def __init__(self, ledger: Ledger) -> None:
        self.ledger = ledger
        self.executor: FrozenHeldOutSemanticExecutor | None = None

    @property
    def actual_allocated_service_seconds(self) -> float:
        return 2.0

    def generate(self, request, **kwargs):
        repair = kwargs["repair"]
        if repair:
            assert self.executor is not None
            intent = self.executor._state().active_intent
            assert intent is not None
            assert intent.repair_gpu_event_id == kwargs["event_id"]
            assert intent.repair_semantic_request is not None
            assert intent.repair_packing_report is not None
        offset = 2 if repair else 0
        self.ledger.record_gpu_event(
            event_id=kwargs["event_id"],
            event_kind=GpuEventKind.REPAIR if repair else GpuEventKind.INFERENCE,
            allocated_seconds=1,
            started_at=NOW + timedelta(seconds=offset),
            ended_at=NOW + timedelta(seconds=offset + 1),
            succeeded=True,
            job_id=kwargs["job_id"],
            attempt_id=kwargs["attempt_id"],
        )
        body = json.dumps(
            {
                "id": f"test-{'repair' if repair else 'base'}",
                "choices": [
                    {
                        "message": {"content": "{}"},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {
                    "prompt_tokens": request.rendered_input_token_count,
                    "completion_tokens": 1,
                },
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
        return GenerationResult(
            request_id=request.request_id,
            request_hash=request.request_hash,
            response_sha256=hashlib.sha256(body).hexdigest(),
            parsed_object={},
            raw_response=body,
            prompt_tokens=request.rendered_input_token_count,
            completion_tokens=1,
            finish_reason="stop",
            service_request_id=f"test-{'repair' if repair else 'base'}",
        )


def _reviewed_plan() -> ReviewedHeldOutPlan:
    configuration = load_held_out_control_configuration(ROOT)
    return ReviewedHeldOutPlan(
        call_manifest=held_out._derive_call_manifest(ROOT, configuration),
        review_completion_manifest_hash="a" * 64,
        review_draft_seal_hash="b" * 64,
        final_reviewed_seal_hash="c" * 64,
    )


def _tokenizer_manifest() -> TokenizerManifest:
    return TokenizerManifest(
        schema_version="1.0.0",
        repository=FALLBACK_MODEL_REPOSITORY,
        revision=FALLBACK_MODEL_REVISION,
        tokenizer_class="TEST-ONLY",
        tokenizer_revision=FALLBACK_MODEL_REVISION,
        tokenizer_file_sha256=(("tokenizer.json", "d" * 64),),
        eos_token_id=1,
        end_of_turn_token_ids=(2,),
        stop_token_ids=(1, 2),
        chat_template_sha256="e" * 64,
        nonthinking_probe_sha256="f" * 64,
        nonthinking_probe_token_count=1,
        local_files_only=True,
        trust_remote_code=False,
        enable_thinking=False,
    )


def test_semantic_executor_never_resends_a_persisted_active_intent(tmp_path: Path) -> None:
    reviewed = _reviewed_plan()
    ledger = Ledger(tmp_path / "ledger.sqlite3")
    try:
        executor = FrozenHeldOutSemanticExecutor(
            root=ROOT,
            call_manifest=reviewed.call_manifest,
            construction=DevelopmentConstructionConfiguration.load(
                ROOT / DEFAULT_DEVELOPMENT_CONSTRUCTION_CONFIG
            ),
            tokenizer=_UnusedTokenizer(),
            tokenizer_manifest=_tokenizer_manifest(),
            model_stack_hash="1" * 64,
            seed_manifest_hash="2" * 64,
            validator_hash="3" * 64,
            selected_model_freeze_hash="4" * 64,
            source_tree_hash="5" * 64,
            artifacts=ArtifactStore(
                BlobStore(tmp_path / "cas", compression=Compression.GZIP), ledger
            ),
            neutral_by_unit=_load_held_out_neutral_worlds(ROOT, reviewed),
            state_path=tmp_path / "semantic-state.json",
            clock=lambda: NOW,
        )
        call = reviewed.call_manifest.calls[0]
        unit = next(item for item in reviewed.call_manifest.units if item.unit_id == call.unit_id)
        envelope = HeldOutCallEnvelope(
            call_spec_hash=call.content_hash,
            prequery_stage=unit.prequery_stage,
            query_stage=None,
            source_c1_call_id=None,
            require_empty_prequery_inventory=False,
            construction_operations_permitted=True,
        )
        state = executor._state()
        executor._replace_state(
            state,
            active_intent=HeldOutSemanticIntent(
                call_id=call.call_id,
                call_spec_hash=call.content_hash,
                envelope_hash=envelope.content_hash,
                semantic_request_hash="6" * 64,
                guided_request_hash="7" * 64,
                gpu_event_id="TEST-ONLY-in-flight",
                started_at=NOW,
            ),
        )
        with pytest.raises(HeldOutSemanticExecutionError, match="cannot be resent"):
            executor.execute(
                service=_UnusedService(),
                call=call,
                envelope=envelope,
                remaining_required_seconds=100,
                repair_allowed=True,
            )
        with pytest.raises(
            HeldOutSemanticExecutionError,
            match="reconstructed semantic intent differs",
        ):
            executor.recover(call=call, envelope=envelope)
        assert ledger.gpu_events() == ()
    finally:
        ledger.close()


@pytest.mark.parametrize("terminal_event", [False, True])
def test_semantic_executor_recovers_exact_preallocation_or_terminal_event_boundary(
    tmp_path: Path,
    terminal_event: bool,
) -> None:
    reviewed = _reviewed_plan()
    ledger = Ledger(tmp_path / "ledger.sqlite3")
    try:
        executor = FrozenHeldOutSemanticExecutor(
            root=ROOT,
            call_manifest=reviewed.call_manifest,
            construction=DevelopmentConstructionConfiguration.load(
                ROOT / DEFAULT_DEVELOPMENT_CONSTRUCTION_CONFIG
            ),
            tokenizer=_UnusedTokenizer(),
            tokenizer_manifest=_tokenizer_manifest(),
            model_stack_hash="1" * 64,
            seed_manifest_hash="2" * 64,
            validator_hash="3" * 64,
            selected_model_freeze_hash="4" * 64,
            source_tree_hash="5" * 64,
            artifacts=ArtifactStore(
                BlobStore(tmp_path / "cas", compression=Compression.GZIP), ledger
            ),
            neutral_by_unit=_load_held_out_neutral_worlds(ROOT, reviewed),
            state_path=tmp_path / "semantic-state.json",
            clock=lambda: NOW,
        )
        call = reviewed.call_manifest.calls[0]
        unit = next(item for item in reviewed.call_manifest.units if item.unit_id == call.unit_id)
        envelope = HeldOutCallEnvelope(
            call_spec_hash=call.content_hash,
            prequery_stage=unit.prequery_stage,
            query_stage=None,
            source_c1_call_id=None,
            require_empty_prequery_inventory=False,
            construction_operations_permitted=True,
        )
        semantic, _config, _inputs = executor._semantic_request(call, envelope)
        guided = held_out_execution.build_development_guided_request(
            root=ROOT,
            call_id=call.call_id,
            semantic_request=semantic,
            tokenizer=executor.tokenizer,
            tokenizer_manifest=executor.tokenizer_manifest,
            seed=call.vllm_seed,
        )
        intent = HeldOutSemanticIntent(
            call_id=call.call_id,
            call_spec_hash=call.content_hash,
            envelope_hash=envelope.content_hash,
            semantic_request_hash=semantic.content_hash,
            guided_request_hash=guided.request_hash,
            gpu_event_id=f"{reviewed.call_manifest.manifest_id}-{call.call_id}-base-gpu",
            started_at=NOW,
        )
        executor._replace_state(executor._state(), active_intent=intent)
        if terminal_event:
            ledger.register_study(
                study_id=reviewed.call_manifest.manifest_id,
                protocol_hash=reviewed.call_manifest.content_hash,
                code_manifest_hash="5" * 64,
                configuration_hash="6" * 64,
                release_class=ReleaseClass.RESTRICTED,
                created_at=NOW,
            )
            ledger.record_gpu_event(
                event_id=intent.gpu_event_id,
                event_kind=GpuEventKind.FAILURE,
                allocated_seconds=1,
                started_at=NOW,
                ended_at=NOW + timedelta(seconds=1),
                succeeded=False,
            )
        result = executor.recover(call=call, envelope=envelope)
        if terminal_event:
            assert result is not None
            assert result.request_started
            assert result.outcome is RunOutcome.FAILED
            assert result.allocated_gpu_seconds == 1
            assert (
                executor._state().completed[call.call_id].result_logical_hash
                == result.content_hash
            )
            assert result.artifact_receipt is not None
            assert (
                ledger.get_model_call(result.artifact_receipt.model_call_id).successful
                is False
            )
            job_id = canonical_sha256(executor._job_identity(call))
            assert [item.to_state for item in ledger.transitions(job_id)] == [
                JobState.PLANNED,
                JobState.PREQUERY_SEALED,
                JobState.GENERATED,
                JobState.VALIDATED,
                JobState.FINALIZED,
            ]
            validation_rows = ledger._connection.execute(
                "SELECT * FROM validations WHERE job_id = ?", (job_id,)
            ).fetchall()
            assert len(validation_rows) == 1
            assert validation_rows[0]["semantic_assessment_scope"] == (
                SemanticAssessmentScope.RUNTIME_STRUCTURAL_ONLY_NOT_ASSESSED.value
            )
            row_counts = {
                table: ledger.count_rows(table)
                for table in ("jobs", "job_transitions", "validations", "failures")
            }
            assert executor.recover(call=call, envelope=envelope) == result
            assert {
                table: ledger.count_rows(table) for table in row_counts
            } == row_counts
        else:
            assert result is None
            assert executor._state().active_intent is None
            assert ledger.gpu_events() == ()
    finally:
        ledger.close()


def test_semantic_executor_recovers_terminal_repair_after_result_commit_crash(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    reviewed = _reviewed_plan()
    ledger = Ledger(tmp_path / "ledger.sqlite3")
    try:
        ledger.register_study(
            study_id=reviewed.call_manifest.manifest_id,
            protocol_hash=reviewed.call_manifest.content_hash,
            code_manifest_hash="5" * 64,
            configuration_hash="6" * 64,
            release_class=ReleaseClass.RESTRICTED,
            created_at=NOW,
        )
        executor = FrozenHeldOutSemanticExecutor(
            root=ROOT,
            call_manifest=reviewed.call_manifest,
            construction=DevelopmentConstructionConfiguration.load(
                ROOT / DEFAULT_DEVELOPMENT_CONSTRUCTION_CONFIG
            ),
            tokenizer=_UnusedTokenizer(),
            tokenizer_manifest=_tokenizer_manifest(),
            model_stack_hash="1" * 64,
            seed_manifest_hash="2" * 64,
            validator_hash="3" * 64,
            selected_model_freeze_hash="4" * 64,
            source_tree_hash="5" * 64,
            artifacts=ArtifactStore(
                BlobStore(tmp_path / "cas", compression=Compression.GZIP), ledger
            ),
            neutral_by_unit=_load_held_out_neutral_worlds(ROOT, reviewed),
            state_path=tmp_path / "semantic-state.json",
            clock=lambda: NOW,
        )
        service = _InvalidRecordingService(ledger)
        service.executor = executor
        call = reviewed.call_manifest.calls[0]
        unit = next(item for item in reviewed.call_manifest.units if item.unit_id == call.unit_id)
        envelope = HeldOutCallEnvelope(
            call_spec_hash=call.content_hash,
            prequery_stage=unit.prequery_stage,
            query_stage=None,
            source_c1_call_id=None,
            require_empty_prequery_inventory=False,
            construction_operations_permitted=True,
        )
        replace_state = FrozenHeldOutSemanticExecutor._replace_state

        def crash_before_pointer(self, state, **updates) -> None:
            if "completed" in updates and updates.get("active_intent") is None:
                raise RuntimeError("simulated power loss before result pointer")
            replace_state(self, state, **updates)

        with monkeypatch.context() as fault:
            fault.setattr(
                FrozenHeldOutSemanticExecutor,
                "_replace_state",
                crash_before_pointer,
            )
            with pytest.raises(RuntimeError, match="simulated power loss"):
                executor.execute(
                    service=service,
                    call=call,
                    envelope=envelope,
                    remaining_required_seconds=100,
                    repair_allowed=True,
                )
        interrupted = executor._state().active_intent
        assert interrupted is not None
        assert interrupted.repair_gpu_event_id is not None

        job_id = canonical_sha256(executor._job_identity(call))
        assert [item.to_state for item in ledger.transitions(job_id)] == [
            JobState.PLANNED,
            JobState.PREQUERY_SEALED,
            JobState.GENERATED,
            JobState.REPAIRED,
            JobState.VALIDATED,
            JobState.FINALIZED,
        ]
        assert ledger.count_rows("validations") == 2
        assert ledger.count_rows("failures") == 2
        durable_counts = {
            table: ledger.count_rows(table)
            for table in ("jobs", "job_transitions", "validations", "failures")
        }

        result = executor.recover(call=call, envelope=envelope)

        assert result is not None
        assert result.outcome is RunOutcome.FAILED
        assert result.repair_attempts == 1
        assert result.allocated_gpu_seconds == 2
        assert result.artifact_receipt is not None
        assert result.artifact_receipt.repair_semantic_request is not None
        assert result.artifact_receipt.repair_model_call_id is not None
        assert executor._state().active_intent is None
        assert len(ledger.gpu_events()) == 2
        assert {
            table: ledger.count_rows(table) for table in durable_counts
        } == durable_counts
    finally:
        ledger.close()


def test_concrete_factory_fails_before_paths_when_development_binding_is_pending(
    tmp_path: Path,
) -> None:
    configuration = load_held_out_control_configuration(ROOT)
    with pytest.raises(HeldOutFactoryError, match="accepted development predecessor"):
        create_frozen_production_held_out_bundle(
            repository=ROOT,
            reviewed_plan=_reviewed_plan(),
            configuration=configuration,
            snapshot_path=tmp_path / "missing-snapshot",
            shared_cache=tmp_path / "missing-cache",
            verified_model_manifest_path=tmp_path / "missing-manifest",
            selected_model_freeze_path=tmp_path / "missing-freeze",
            source_association_path=tmp_path / "missing-association",
            development_prequery_inputs_artifact_hash="8" * 64,
            ledger_path=tmp_path / "ledger.sqlite3",
            artifact_root=tmp_path / "cas",
            runtime_root=tmp_path / "runtime",
            restricted_root=ROOT / "artifacts" / "restricted",
            quota_root=tmp_path,
        )


def test_held_out_factory_uses_exact_registered_phase_three_storage_reservation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    storage = StoragePreflight(tmp_path)
    original_check = storage.check
    observed: list[dict[str, int]] = []

    def capture_check(**arguments: int):
        observed.append(dict(arguments))
        return original_check(
            **arguments,
            current_occupied_bytes=0,
            filesystem_free_bytes=30_000_000_000,
        )

    monkeypatch.setattr(storage, "check", capture_check)

    report = _registered_phase_three_storage_preflight(ROOT, storage)

    assert observed == [
        {
            "declared_growth_bytes": 2_500_000_000,
            "largest_atomic_temporary_bytes": 268_435_456,
            "quarantine_allowance_bytes": 268_435_456,
            "release_staging_bytes": 268_435_456,
        }
    ]
    assert report.additional_reserved_bytes == sum(observed[0].values())


def test_held_out_factory_persists_rejected_registered_storage_gate(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    storage = StoragePreflight(tmp_path)
    original_check = storage.check
    rejected = original_check(
        current_occupied_bytes=25_000_000_000,
        filesystem_free_bytes=30_000_000_000,
        declared_growth_bytes=2_500_000_000,
        largest_atomic_temporary_bytes=268_435_456,
        quarantine_allowance_bytes=268_435_456,
        release_staging_bytes=268_435_456,
    )
    monkeypatch.setattr(storage, "check", lambda **_arguments: rejected)

    with Ledger(tmp_path / "study.sqlite3") as ledger:
        with pytest.raises(HeldOutFactoryError, match="storage preflight failed"):
            _require_registered_phase_three_storage_preflight(ROOT, storage, ledger)
        samples = ledger.storage_samples_with_phase_prefix("phase_3:")

    assert len(samples) == 1
    assert samples[0].phase == "phase_3:held_out_primary:factory"
    assert samples[0].allowed is False


def test_primary_executor_conditions_remain_exact_three() -> None:
    assert {call.condition for call in _reviewed_plan().call_manifest.calls} == {
        ConditionName.C1_LLM_PRE,
        ConditionName.C2_LLM_QUERY,
        ConditionName.A_FIXED_SELECT,
    }


def test_neutral_world_loader_rejects_repository_ancestor_symlink(
    tmp_path: Path,
) -> None:
    (tmp_path / "data").symlink_to(ROOT / "data", target_is_directory=True)
    with pytest.raises(HeldOutFactoryError, match="containment or integrity"):
        _load_held_out_neutral_worlds(tmp_path, _reviewed_plan())


def test_restricted_runtime_paths_reject_escape_and_ancestor_symlinks(
    tmp_path: Path,
) -> None:
    restricted = tmp_path / "restricted"
    restricted.mkdir()
    safe = _restricted_descendant(
        restricted,
        restricted / "runtime" / "state.json",
        label="runtime state",
    )
    assert safe == restricted / "runtime" / "state.json"
    with pytest.raises(HeldOutFactoryError, match="outside the restricted root"):
        _restricted_descendant(
            restricted,
            tmp_path / "public" / "ledger.sqlite3",
            label="GPU ledger",
        )
    real = restricted / "real"
    real.mkdir()
    linked = restricted / "linked"
    linked.symlink_to(real, target_is_directory=True)
    with pytest.raises(HeldOutFactoryError, match="symlinked ancestor"):
        _restricted_descendant(
            restricted,
            linked / "cas",
            label="CAS root",
        )


def test_factory_restricted_root_is_exactly_the_repository_private_tree(
    tmp_path: Path,
) -> None:
    repository = tmp_path / "repository"
    restricted = repository / "artifacts" / "restricted"
    public = repository / "artifacts" / "public"
    restricted.mkdir(parents=True)
    public.mkdir()

    assert (
        _canonical_restricted_root(repository, Path("artifacts/restricted"))
        == restricted
    )
    with pytest.raises(HeldOutFactoryError, match="canonical repository root"):
        _canonical_restricted_root(repository, public)
