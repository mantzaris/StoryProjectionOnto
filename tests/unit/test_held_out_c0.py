from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

import story_projection_onto.held_out_primary as held_out
from story_projection_onto.benchmark_runtime import RuntimeStagingManifest
from story_projection_onto.conditions.base import ConditionPreparation, sealed_semantic_ids
from story_projection_onto.conditions.c0 import (
    ClassicalBackendManifest,
    ClassicalEvidenceAnalysis,
    ClassicalPreBuilder,
    ClassicalRuleConfig,
    RuleCandidateBackend,
    load_classical_rule_config,
)
from story_projection_onto.contracts import (
    ConditionName,
    ConstructionOperator,
    EvidencePacket,
    EvidenceRecord,
    OntologyProjection,
    PrequeryBarrier,
    PrequeryPreparationBinding,
    RetrievalMethod,
    canonical_sha256,
)
from story_projection_onto.held_out_c0 import (
    HeldOutC0Error,
    HeldOutC0PreparationState,
    ProductionHeldOutC0Adapter,
)
from story_projection_onto.held_out_primary import (
    HeldOutCASReference,
    HeldOutPrequeryUnit,
    HeldOutQueryOpening,
    HeldOutUnitPlan,
    PublicStageReference,
    load_held_out_control_configuration,
)
from story_projection_onto.query_runtime import AuditedBenchmarkRuntime
from story_projection_onto.store import (
    ArtifactStore,
    BlobStore,
    Compression,
    Ledger,
    ReleaseClass,
)

ROOT = Path(__file__).resolve().parents[2]


class _TickClock:
    def __init__(self) -> None:
        self.value = datetime(2026, 9, 4, tzinfo=UTC)

    def __call__(self) -> datetime:
        self.value += timedelta(microseconds=1)
        return self.value


class _PinnedRuleBackend(RuleCandidateBackend):
    backend_name = "spacy-ner-dependency-plus-deterministic-rules-v2"

    def __init__(self) -> None:
        self.evidence_ids: list[str] = []

    def analyze(
        self,
        evidence: EvidenceRecord,
        config: ClassicalRuleConfig,
    ) -> ClassicalEvidenceAnalysis:
        self.evidence_ids.append(evidence.evidence_id)
        return super().analyze(evidence, config)


@dataclass(frozen=True, slots=True)
class _Harness:
    adapter: ProductionHeldOutC0Adapter
    unit: HeldOutUnitPlan
    backend: _PinnedRuleBackend
    artifacts: ArtifactStore
    clock: _TickClock


@pytest.fixture
def harness(tmp_path: Path) -> Iterator[_Harness]:
    configuration = load_held_out_control_configuration(ROOT)
    manifest = held_out._derive_call_manifest(ROOT, configuration)
    unit = min(
        manifest.units,
        key=lambda item: (
            (ROOT / item.prequery_stage.relative_path / "evidence.json").stat().st_size
        ),
    )
    backends: list[_PinnedRuleBackend] = []

    def load_builder(path: Path) -> tuple[ClassicalPreBuilder, ClassicalBackendManifest]:
        config = load_classical_rule_config(path)
        backend = _PinnedRuleBackend()
        backends.append(backend)
        return (
            ClassicalPreBuilder(config=config, candidate_backend=backend),
            ClassicalBackendManifest(
                config_hash=config.content_hash,
                backend_name=backend.backend_name,
                spacy_library_version="TEST-ONLY-3.8",
                model_package="en_core_web_sm",
                model_version="3.8.0",
                pipeline_components=("tagger", "parser", "ner"),
            ),
        )

    ledger = Ledger(tmp_path / "ledger.sqlite3")
    artifacts = ArtifactStore(
        BlobStore(tmp_path / "cas", compression=Compression.GZIP),
        ledger,
    )
    clock = _TickClock()
    adapter = ProductionHeldOutC0Adapter(
        repository=ROOT,
        call_manifest=manifest,
        configuration=configuration,
        artifacts=artifacts,
        state_directory=tmp_path / "private-c0-state",
        builder_loader=load_builder,
        clock=clock,
    )
    ledger.register_study(
        study_id=manifest.manifest_id,
        protocol_hash=manifest.content_hash,
        code_manifest_hash=canonical_sha256({"test": "held-out-c0"}),
        configuration_hash=configuration.content_hash,
        release_class=ReleaseClass.RESTRICTED,
        created_at=clock(),
    )
    try:
        yield _Harness(adapter, unit, backends[0], artifacts, clock)
    finally:
        ledger.close()


def _prequery_view(unit: HeldOutUnitPlan) -> HeldOutPrequeryUnit:
    return HeldOutPrequeryUnit(unit_id=unit.unit_id, prequery_stage=unit.prequery_stage)


def _reference(
    artifacts: ArtifactStore,
    artifact_hash: str,
    logical_hash: str,
    object_kind: str,
) -> HeldOutCASReference:
    artifact = artifacts.ledger.get_artifact(artifact_hash)
    return HeldOutCASReference(
        artifact_hash=artifact.content_hash,
        logical_content_hash=logical_hash,
        object_kind=object_kind,  # type: ignore[arg-type]
        media_type=artifact.media_type,
        release_class=artifact.release_class.value,
    )


def test_c0_prebuild_reads_no_query_and_seals_the_complete_staged_graph(
    harness: _Harness,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original_read_bytes = Path.read_bytes
    forbidden_reads: list[Path] = []

    def audited_read_bytes(path: Path) -> bytes:
        if "query_stages" in path.parts or "scorer_only" in path.parts or "gold" in path.parts:
            forbidden_reads.append(path)
            raise AssertionError(f"query-blind C0 opened forbidden input: {path}")
        return original_read_bytes(path)

    monkeypatch.setattr(Path, "read_bytes", audited_read_bytes)
    receipt = harness.adapter.build_c0(_prequery_view(harness.unit))
    analysis_count = len(harness.backend.evidence_ids)
    resumed = harness.adapter.build_c0(_prequery_view(harness.unit))

    state_path = harness.adapter.state_directory / f"{harness.unit.unit_id}.json"
    state = HeldOutC0PreparationState.model_validate_json(state_path.read_bytes())
    artifact = harness.artifacts.ledger.get_artifact(state.preparation.artifact_hash)
    preparation = ConditionPreparation.model_validate_json(
        harness.artifacts.blobs.read_bytes(artifact, allow_restricted=True)
    )
    preontology = preparation.sealed_preontology
    assert preontology is not None
    assert forbidden_reads == []

    # The backend sees the full snapshot exactly once; recovery reuses the durable seal.
    assert tuple(harness.backend.evidence_ids) == tuple(
        record.evidence_id
        for record in harness.adapter._load_query_blind_evidence(harness.unit.prequery_stage)[
            1
        ].evidence
    )
    assert len(harness.backend.evidence_ids) == analysis_count
    assert resumed == receipt
    assert state.preparation.logical_content_hash == preparation.content_hash
    assert receipt.construction_seal_hash == preontology.construction_seal.content_hash
    assert receipt.complete_graph_hash == preontology.construction_seal.ontology_hash
    assert preontology.construction_seal.sealed_object_ids == sealed_semantic_ids(preontology.draft)
    state_bytes = state_path.read_bytes()
    assert b"query_context" not in state_bytes
    assert b"query_artifact" not in state_bytes
    assert b"scorer" not in state_bytes
    assert receipt.allocated_gpu_seconds == 0.0


def test_c0_projection_uses_audited_query_cas_and_only_selects_sealed_ids(
    harness: _Harness,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    receipt = harness.adapter.build_c0(_prequery_view(harness.unit))
    visible, neutral, _, _ = harness.adapter._load_query_blind_evidence(harness.unit.prequery_stage)
    barrier = PrequeryBarrier(
        barrier_id="TEST-ONLY-held-out-c0-barrier",
        execution_id="TEST-ONLY-held-out-c0-execution",
        execution_manifest_hash=harness.adapter.call_manifest.content_hash,
        neutral_evidence_artifact_hashes=(visible.content_hash,),
        preparation_bindings=(
            PrequeryPreparationBinding(
                unit_id=harness.unit.unit_id,
                condition=ConditionName.C0_CLASSICAL_PRE,
                snapshot_hash=harness.unit.prequery_stage.snapshot_hash,
                preparation_hash=receipt.content_hash,
                lineage_artifact_hash=receipt.complete_graph_hash,
                completed_at=receipt.completed_at,
            ),
        ),
        sealed_at=harness.clock(),
    )
    runtime = AuditedBenchmarkRuntime(
        ledger=harness.artifacts.ledger,
        blobs=harness.artifacts.blobs,
        clock=harness.clock,
    )
    runtime.persist_prequery_barrier(
        barrier,
        release_class=ReleaseClass(visible.snapshot.release_class.value),
    )

    stage = harness.unit.query_stages[0]
    stage_root = (ROOT / stage.relative_path).resolve(strict=True)
    stage_manifest = RuntimeStagingManifest.model_validate_json(
        (stage_root / "manifest.json").read_bytes()
    )
    audited = runtime.open_query(
        staging_root=stage_root,
        manifest=stage_manifest,
        barrier=barrier,
        execution_id=barrier.execution_id,
        execution_manifest_hash=barrier.execution_manifest_hash,
        access_event_id=f"TEST-ONLY-access-{stage.stage_id}",
    )

    def all_admissible(_evidence: object, _query: object) -> EvidencePacket:
        return EvidencePacket(
            packet_id=f"TEST-ONLY-packet-{stage.stage_id}",
            snapshot_hash=neutral.snapshot.content_hash,
            evidence=neutral.evidence,
            ordered_evidence_ids=neutral.snapshot.eligible_evidence_ids,
            retrieval_method=RetrievalMethod.ALL_ADMISSIBLE,
            token_count=sum(len(record.text.split()) for record in neutral.evidence),
            created_at=harness.clock(),
            release_class=neutral.snapshot.release_class,
        )

    materialized = runtime.materialize_packet(
        audited,
        materialization_event_id=f"TEST-ONLY-materialization-{stage.stage_id}",
        retrieval_config_hash=canonical_sha256({"policy": "all-admissible", "test": True}),
        materializer=all_admissible,
    )
    opened_stage = PublicStageReference(
        **stage.model_dump(
            mode="python",
            exclude={"content_hash", "query_context_hash", "horizon_hash", "budget_hash"},
        ),
        query_context_hash=audited.context.content_hash,
        horizon_hash=audited.context.spoiler_horizon.content_hash,
        budget_hash=audited.context.budgets.content_hash,
    )
    opening = HeldOutQueryOpening(
        sealed_stage_hash=stage.staging_manifest_hash,
        opened_stage=opened_stage,
        query_access_event=audited.access_event,
        query_access_artifact=_reference(
            harness.artifacts,
            audited.persistence.access_event_artifact_hash,
            audited.access_event.content_hash,
            "query_access_event",
        ),
        evidence_packet_hash=materialized.packet.content_hash,
        evidence_packet_artifact=_reference(
            harness.artifacts,
            materialized.persistence.packet_artifact_hash,
            materialized.packet.content_hash,
            "evidence_packet",
        ),
        query_context=audited.context,
        packet_materialization=materialized.event,
        packet_materialization_artifact=_reference(
            harness.artifacts,
            materialized.persistence.materialization_event_artifact_hash,
            materialized.event.content_hash,
            "packet_materialization_event",
        ),
        opened_at=audited.access_event.accessed_at,
    )

    original_read_bytes = Path.read_bytes
    direct_query_reads: list[Path] = []

    def cas_only_read_bytes(path: Path) -> bytes:
        if stage_root in path.parents:
            direct_query_reads.append(path)
            raise AssertionError(f"C0 reread query-stage bytes after audited opening: {path}")
        return original_read_bytes(path)

    monkeypatch.setattr(Path, "read_bytes", cas_only_read_bytes)
    projection_receipt = harness.adapter.project_preconstructed(
        unit=harness.unit,
        query_stage_hash=stage.staging_manifest_hash,
        query_opening=opening,
        condition=ConditionName.C0_CLASSICAL_PRE,
        seed_block=None,
        construction_seal_hash=receipt.construction_seal_hash,
        complete_graph_hash=receipt.complete_graph_hash,
    )
    projection_artifact = harness.artifacts.ledger.get_artifact(
        projection_receipt.projection_artifact_hash
    )
    projection = OntologyProjection.model_validate_json(
        harness.artifacts.blobs.read_bytes(projection_artifact, allow_restricted=True)
    )
    projected_ids = {
        *(item.entity_id for item in projection.instance_graph.entities),
        *(item.event_id for item in projection.instance_graph.events),
        *(item.proposition_content_id for item in projection.instance_graph.proposition_contents),
        *(item.assertion_id for item in projection.instance_graph.assertions),
    }
    assert direct_query_reads == []
    assert projection.condition is ConditionName.C0_CLASSICAL_PRE
    assert projection.packet_hash == materialized.packet.content_hash
    assert projection.context_hash == audited.context.content_hash
    assert projection.construction_seal is not None
    assert projection.construction_seal.content_hash == receipt.construction_seal_hash
    assert projected_ids.issubset(projection.construction_seal.sealed_object_ids)
    assert {decision.operator for decision in projection.decisions} == {
        ConstructionOperator.SELECTION
    }
    assert projection_receipt.source_complete_graph_hash == receipt.complete_graph_hash
    assert projection_receipt.allocated_gpu_seconds == 0.0


def test_c0_adapter_rejects_rewritten_rule_bytes(
    harness: _Harness,
    tmp_path: Path,
) -> None:
    repository = tmp_path / "changed-repository"
    rules = repository / "configs/study/c0_rules.json"
    rules.parent.mkdir(parents=True)
    rules.write_bytes((ROOT / "configs/study/c0_rules.json").read_bytes() + b"\n")

    def forbidden_builder(_path: Path) -> tuple[ClassicalPreBuilder, ClassicalBackendManifest]:
        raise AssertionError("builder must not load after the byte pin fails")

    with pytest.raises(HeldOutC0Error, match="frozen C0 dependency bytes changed"):
        ProductionHeldOutC0Adapter(
            repository=repository,
            call_manifest=harness.adapter.call_manifest,
            configuration=harness.adapter.configuration,
            artifacts=harness.artifacts,
            state_directory=tmp_path / "changed-state",
            builder_loader=forbidden_builder,
            clock=harness.clock,
        )
