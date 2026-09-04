from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import cast

import pytest

import story_projection_onto.development_continuation as continuation_module
from story_projection_onto.conditions.c0 import ClassicalPreBuilder
from story_projection_onto.contracts import canonical_sha256
from story_projection_onto.development_adapter import (
    AdapterDurableState,
    persist_opaque_json,
)
from story_projection_onto.development_artifacts import (
    DevelopmentAssessmentBundle,
    DevelopmentPackingPreflight,
    DevelopmentPreparationIndex,
    LogicalCASReference,
)
from story_projection_onto.development_assessment_bridge import (
    build_post_run_development_assessment_provider,
)
from story_projection_onto.development_continuation import (
    DevelopmentContinuationError,
    ProductionDevelopmentContinuationAdopter,
    build_development_forecast_receipt,
    create_production_development_adopter,
)
from story_projection_onto.development_runtime import (
    DevelopmentPrequeryInputs,
    load_development_call_manifest,
)
from story_projection_onto.fallback_acceptance import DevelopmentContinuationBootstrap
from story_projection_onto.gpu_runtime import (
    FALLBACK_MODEL_REPOSITORY,
    FALLBACK_MODEL_REVISION,
    TokenizerManifest,
)
from story_projection_onto.manifest import build_source_manifest
from story_projection_onto.store import (
    ArtifactStore,
    BlobStore,
    Compression,
    GpuEventKind,
    Ledger,
    ReleaseClass,
)

ROOT = Path(__file__).resolve().parents[2]
BASELINE_SECONDS = 212.281778


def _digest(value: object) -> str:
    return hashlib.sha256(repr(value).encode("utf-8")).hexdigest()


class _CompactFakeTokenizer:
    def encode(self, text: str, **kwargs: object) -> list[int]:
        del kwargs
        return list(range(max(1, len(text.split()))))

    def apply_chat_template(self, conversation: object, **kwargs: object) -> list[int]:
        assert kwargs["enable_thinking"] is False
        assert isinstance(conversation, (list, tuple))
        words = sum(len(str(item["content"]).split()) for item in conversation)
        return list(range(words + 8))


class _NoInferenceService:
    def __init__(self, allocated: float = BASELINE_SECONDS) -> None:
        self.allocated = allocated

    @property
    def actual_allocated_service_seconds(self) -> float:
        return self.allocated

    def generate(self, *args: object, **kwargs: object) -> object:
        del args, kwargs
        raise AssertionError("query-blind preparation must not execute inference")


class _TickingClock:
    def __init__(self) -> None:
        self.value = datetime(2035, 1, 1, tzinfo=UTC)

    def __call__(self) -> datetime:
        self.value += timedelta(milliseconds=1)
        return self.value


def test_production_factory_propagates_gpu_recovery_overlay(tmp_path: Path) -> None:
    ledger = Ledger(tmp_path / "ledger.sqlite3")
    amendment_hash = _digest("retry-amendment")
    event_id = "fallback-recovery-service-start-001"
    try:
        adopter = create_production_development_adopter(
            root=ROOT,
            service=_NoInferenceService(),
            artifacts=ArtifactStore(BlobStore(tmp_path / "blobs"), ledger),
            tokenizer=_CompactFakeTokenizer(),
            tokenizer_manifest=_tokenizer_manifest(),
            launcher_configuration_hash=_digest("launcher"),
            model_snapshot_manifest_hash=_digest("snapshot"),
            source_association={
                "revision_label": "recovery-test",
                "local_tree_sha256": _digest("source-tree"),
            },
            checkpoint_path=tmp_path / "run" / "fallback.checkpoint.json",
            assessment_factory=_unreachable_assessment_factory,
            retry_amendment_sha256=amendment_hash,
            recovery_service_start_event_ids=(event_id,),
        )
    finally:
        ledger.close()

    assert adopter.retry_amendment_sha256 == amendment_hash
    assert adopter.recovery_service_start_event_ids == (event_id,)


def _tokenizer_manifest() -> TokenizerManifest:
    return TokenizerManifest(
        schema_version="1.0.0",
        repository=FALLBACK_MODEL_REPOSITORY,
        revision=FALLBACK_MODEL_REVISION,
        tokenizer_class="tests.CompactFakeTokenizer",
        tokenizer_revision=FALLBACK_MODEL_REVISION,
        tokenizer_file_sha256=(("tokenizer.json", _digest("tokenizer")),),
        eos_token_id=7,
        end_of_turn_token_ids=(8,),
        stop_token_ids=(7, 8),
        chat_template_sha256=_digest("chat-template"),
        nonthinking_probe_sha256=_digest("nonthinking-probe"),
        nonthinking_probe_token_count=10,
        local_files_only=True,
        trust_remote_code=False,
        enable_thinking=False,
    )


def _bootstrap(
    adopter: ProductionDevelopmentContinuationAdopter,
    *,
    source_tree_hash: str,
) -> DevelopmentContinuationBootstrap:
    execution_hash = _digest("fallback-execution")
    result_hash = _digest("fallback-result")
    receipt_payload: dict[str, object] = {
        "kind": "micro-pilot-acceptance-receipt",
        "accepted_result_manifest_sha256": result_hash,
        "execution_hash": execution_hash,
        "source_tree_sha256": source_tree_hash,
    }
    receipt_hash = canonical_sha256(receipt_payload)
    receipt = {**receipt_payload, "manifest_sha256": receipt_hash}
    freeze_payload: dict[str, object] = {
        "kind": "selected-model-freeze",
        "micro_pilot_acceptance_receipt_sha256": receipt_hash,
    }
    freeze_hash = canonical_sha256(freeze_payload)
    freeze = {**freeze_payload, "manifest_sha256": freeze_hash}
    return DevelopmentContinuationBootstrap(
        bootstrap_id="fallback-test-development-bootstrap",
        owner_run_id="fallback-test",
        fallback_execution_hash=execution_hash,
        accepted_fallback_result_hash=result_hash,
        micro_pilot_acceptance_receipt=receipt,
        micro_pilot_acceptance_receipt_sha256=receipt_hash,
        adopter_registration_hash=adopter.registration().content_hash,
        selected_model_freeze=freeze,
        selected_model_freeze_hash=freeze_hash,
        source_tree_hash=source_tree_hash,
        service_pid=4242,
        service_start_ticks=917,
        gpu_session_event_id="fallback-test-service-start-001",
        launcher_configuration_hash=adopter.launcher_configuration_hash,
        model_snapshot_manifest_hash=adopter.model_snapshot_manifest_hash,
        service_checkpoint_sha256=_digest("service-checkpoint"),
        allocated_gpu_seconds_before_preparation=BASELINE_SECONDS,
    )


def _unreachable_assessment_factory(**kwargs: object) -> object:
    del kwargs
    raise AssertionError("preparation must not instantiate the scorer")


def test_cumulative_forecast_starts_at_exact_ledger_total_and_consumes_reserve(
    tmp_path: Path,
) -> None:
    manifest = load_development_call_manifest(ROOT)
    started = datetime(2026, 9, 4, tzinfo=UTC)
    with Ledger(tmp_path / "ledger.sqlite3") as ledger:
        ledger.record_gpu_event(
            event_id="primary-and-fallback-cumulative",
            event_kind=GpuEventKind.FAILURE,
            allocated_seconds=BASELINE_SECONDS,
            started_at=started,
            ended_at=started + timedelta(seconds=BASELINE_SECONDS),
            succeeded=False,
            details={
                "reserve_call_class": "reserve_short",
                "reserve_reservation_id": "fallback-reserve-short-01",
            },
        )
        receipt = build_development_forecast_receipt(
            root=ROOT,
            ledger=ledger,
            service=_NoInferenceService(),
            manifest=manifest,
            clock=lambda: started,
        )

    assert receipt.actual_allocated_seconds_before_development == BASELINE_SECONDS
    reserve = next(
        row for row in receipt.inventory_rows if row.call_class == "reserve_short"
    )
    assert reserve.consumed_before_development == 1
    assert reserve.remaining_after_development == 3
    normal = tuple(
        row
        for row in receipt.inventory_rows
        if row.call_class.startswith("acceptance_")
    )
    assert normal and all(row.remaining_after_development == 0 for row in normal)
    assert receipt.consumed_reserves_not_replenished is True
    assert receipt.normal_acceptance_superseded is True


def test_production_prepare_is_query_blind_and_persists_exact_c1_pack(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    revision = "development-continuation-test-tree"
    source_manifest = build_source_manifest(ROOT, revision)
    source_tree_hash = source_manifest.tree_sha256
    monkeypatch.setattr(
        continuation_module,
        "build_source_manifest",
        lambda _root, _revision: source_manifest,
    )
    ledger = Ledger(tmp_path / "ledger.sqlite3")
    blobs = BlobStore(tmp_path / "blobs", compression=Compression.GZIP)
    artifacts = ArtifactStore(blobs, ledger)
    adopter = ProductionDevelopmentContinuationAdopter(
        root=ROOT,
        service=_NoInferenceService(),
        artifacts=artifacts,
        tokenizer=_CompactFakeTokenizer(),
        tokenizer_manifest=_tokenizer_manifest(),
        launcher_configuration_hash=_digest("launcher"),
        model_snapshot_manifest_hash=_digest("snapshot-manifest"),
        source_revision=revision,
        checkpoint_path=tmp_path / "development.checkpoint.json",
        adapter_state_path=tmp_path / "development.adapter-state.json",
        preparation_pointer_path=tmp_path / "development.preparation.json",
        assessment_manifest_path=tmp_path / "development.assessment.json",
        assessment_factory=_unreachable_assessment_factory,
        classical_builder_loader=lambda _path: (ClassicalPreBuilder(), object()),
        clock=_TickingClock(),
    )
    try:
        assert adopter.registration().implementation_sha256 == source_tree_hash
        bootstrap = _bootstrap(adopter, source_tree_hash=source_tree_hash)
        prepared = adopter.prepare(bootstrap)
        assert prepared.query_access_event_count == 0
        assert ledger.count_rows("query_access_events") == 0
        assert ledger.count_rows("packet_materialization_events") == 0
        assert tuple(
            prepared.prequery_inputs.run_condition_config_hashes[index]
            for index in (16, 17, 18, 19)
        ) == (None, None, None, None)

        pointer = json.loads(adopter.preparation_pointer_path.read_text(encoding="utf-8"))
        index_record = ledger.get_artifact(pointer["preparation_index_artifact_hash"])
        assert index_record.release_class is ReleaseClass.RESTRICTED
        index = DevelopmentPreparationIndex.model_validate_json(
            blobs.read_bytes(index_record, allow_restricted=True)
        )
        packing_record = ledger.get_artifact(index.packing_preflight_artifact_hash)
        preflight = DevelopmentPackingPreflight.model_validate_json(
            blobs.read_bytes(packing_record)
        )
        assert len(preflight.receipts) == 4
        assert preflight.worst_rendered_input_tokens < 10_240
        assert all(item.semantic_evidence_truncation is False for item in preflight.receipts)
        assert len(index.prequery_preparation_artifacts) == 11
        assert len(index.run_condition_config_artifacts) == 20

        pointer_bytes = adopter.preparation_pointer_path.read_bytes()
        recovered = adopter.prepare(bootstrap)
        assert recovered.prequery_inputs == prepared.prequery_inputs
        assert recovered.execution_manifest_hash == prepared.execution_manifest_hash
        assert adopter.preparation_pointer_path.read_bytes() == pointer_bytes
        assert ledger.count_rows("query_access_events") == 0

        adapter_state_bytes = adopter.adapter_state_path.read_bytes()
        adapter_state = AdapterDurableState.model_validate_json(adapter_state_bytes)
        state_payload = adapter_state.model_dump(mode="python", exclude={"content_hash"})
        state_payload["query_access_events"] = {
            _digest("stage"): _digest("query-access")
        }
        adopter.adapter_state_path.write_text(
            AdapterDurableState.model_validate(state_payload).to_canonical_json() + "\n",
            encoding="utf-8",
        )
        resumed_after_access = adopter.prepare(bootstrap)
        assert resumed_after_access.prequery_inputs == prepared.prequery_inputs
        assert resumed_after_access.execution_manifest_hash == prepared.execution_manifest_hash

        adopter.preparation_pointer_path.unlink()
        with pytest.raises(DevelopmentContinuationError, match="after query access"):
            adopter.prepare(bootstrap)
        adopter.adapter_state_path.write_bytes(adapter_state_bytes)
        adopter.preparation_pointer_path.write_bytes(pointer_bytes)

        tampered = dict(pointer)
        tampered["bootstrap_hash"] = _digest("another-bootstrap")
        adopter.preparation_pointer_path.write_text(
            json.dumps(tampered), encoding="utf-8"
        )
        with pytest.raises(DevelopmentContinuationError, match="pointer lineage"):
            adopter.prepare(bootstrap)

        adopter.preparation_pointer_path.unlink()
        with pytest.raises(DevelopmentContinuationError, match="root is missing"):
            adopter.prepare(bootstrap)
    finally:
        ledger.close()


def test_post_run_assessment_bridge_verifies_source_and_persists_routing_manifest(
    tmp_path: Path,
) -> None:
    manifest = load_development_call_manifest(ROOT)
    source_manifest = build_source_manifest(ROOT, "development-assessment-bridge-test")
    created_at = datetime(2035, 1, 1, tzinfo=UTC)
    ledger = Ledger(tmp_path / "ledger.sqlite3")
    blobs = BlobStore(tmp_path / "blobs", compression=Compression.GZIP)
    artifacts = ArtifactStore(blobs, ledger)
    try:
        source_ref = persist_opaque_json(
            artifacts,
            source_manifest.to_dict(),
            object_kind="runtime_source_manifest",
            created_at=created_at,
        )

        def logical(label: str) -> LogicalCASReference:
            return LogicalCASReference(
                logical_content_hash=_digest(("logical", label)),
                artifact_hash=_digest(("artifact", label)),
                object_kind="test_record",
            )

        bundle = DevelopmentAssessmentBundle(
            bundle_id="development-assessment-bridge-bundle",
            call_manifest_hash=manifest.content_hash,
            prequery_inputs_hash=_digest("prequery-inputs"),
            source_tree_hash=source_manifest.tree_sha256,
            runtime_source_manifest=source_ref,
            benchmark_manifest_file_sha256=manifest.benchmark_manifest_file_sha256,
            prequery_preparation_artifacts=tuple(
                logical(f"preparation-{index}") for index in range(11)
            ),
            call_receipt_artifact_hashes=tuple(
                _digest(("receipt", index)) for index in range(24)
            ),
            service_result_artifact_hashes=tuple(
                _digest(("result", index)) for index in range(24)
            ),
            cpu_projection_receipt_artifact_hashes=tuple(
                _digest(("cpu", index)) for index in range(24)
            ),
            packing_preflight_artifact_hash=_digest("packing-preflight"),
            created_at=created_at,
        )
        bundle_artifact = artifacts.put_bytes(
            (bundle.to_canonical_json() + "\n").encode("utf-8"),
            media_type=(
                "application/vnd.story-projection.development-assessment-bundle+json"
            ),
            release_class=ReleaseClass.PUBLIC,
            created_at=created_at,
        )
        destination = tmp_path / "assessment-input.json"
        provider = build_post_run_development_assessment_provider(
            root=ROOT,
            ledger=ledger,
            blobs=blobs,
            prequery_inputs=cast(DevelopmentPrequeryInputs, object()),
            assessment_bundle_artifact_hash=bundle_artifact.content_hash,
            assessment_manifest_path=destination,
        )
        payload = json.loads(destination.read_text(encoding="utf-8"))
        assert callable(provider)
        assert payload["assessment_bundle_artifact_hash"] == bundle_artifact.content_hash
        assert payload["source_revision"] == source_manifest.revision
        assert payload["source_tree_hash"] == source_manifest.tree_sha256
    finally:
        ledger.close()
