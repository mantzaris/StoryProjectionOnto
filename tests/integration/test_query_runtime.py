from __future__ import annotations

import hashlib
import shutil
from collections.abc import Iterator
from dataclasses import replace
from datetime import datetime, timedelta
from pathlib import Path

import pytest

import story_projection_onto.query_runtime as query_runtime_module
from story_projection_onto.benchmark import compile_benchmark
from story_projection_onto.benchmark_runtime import (
    DescriptorBoundRuntimeStage,
    ModelEligibleWorldArtifact,
    QueryRevealArtifact,
    RuntimeStagingManifest,
)
from story_projection_onto.contracts import (
    ConditionName,
    EvidencePacket,
    PrequeryBarrier,
    PrequeryPreparationBinding,
    QueryContext,
    RetrievalMethod,
)
from story_projection_onto.query_runtime import AuditedBenchmarkRuntime, QueryOpeningError
from story_projection_onto.store import (
    BlobStore,
    Compression,
    DuplicateConflictError,
    Ledger,
    ReleaseClass,
)

_QUERY_STAGE = Path(
    "data/synthetic/model_visible/query_stages/reveal_2dbe5463a34a957a8bee"
)
_HASH_A = "a" * 64
_HASH_B = "b" * 64
_HASH_C = "c" * 64


class _Clock:
    def __init__(self, values: Iterator[datetime]) -> None:
        self._values = values
        self.observed: list[datetime] = []

    def __call__(self) -> datetime:
        value = next(self._values)
        self.observed.append(value)
        return value


def _copy_stage(tmp_path: Path) -> tuple[Path, RuntimeStagingManifest]:
    stage = tmp_path / "query-stage"
    shutil.copytree(_QUERY_STAGE, stage)
    manifest = RuntimeStagingManifest.model_validate_json(
        (stage / "manifest.json").read_text(encoding="utf-8")
    )
    return stage, manifest


def _expected_context_and_evidence(
    stage: Path,
) -> tuple[QueryContext, ModelEligibleWorldArtifact, QueryRevealArtifact, tuple]:
    evidence = ModelEligibleWorldArtifact.model_validate_json(
        (stage / "evidence.json").read_text(encoding="utf-8")
    )
    reveal = QueryRevealArtifact.model_validate_json(
        (stage / "query.json").read_text(encoding="utf-8")
    )
    benchmark = compile_benchmark()
    for world_id, reveals in benchmark.query_reveals_by_world.items():
        for index, candidate in enumerate(reveals):
            if candidate.content_hash == reveal.content_hash:
                return (
                    benchmark.contexts_by_world[world_id][index],
                    evidence,
                    reveal,
                    benchmark.narratives[world_id].evidence,
                )
    raise AssertionError("frozen query stage is absent from the compiled benchmark")


def _barrier(
    evidence: ModelEligibleWorldArtifact,
    context: QueryContext,
    *,
    execution_id: str = "query-runtime-execution",
) -> PrequeryBarrier:
    return PrequeryBarrier(
        barrier_id=f"{execution_id}-barrier",
        execution_id=execution_id,
        execution_manifest_hash=_HASH_A,
        neutral_evidence_artifact_hashes=(evidence.content_hash,),
        preparation_bindings=(
            PrequeryPreparationBinding(
                unit_id="development-unit-01",
                condition=ConditionName.C1_LLM_PRE,
                seed_block=1,
                snapshot_hash=evidence.snapshot.content_hash,
                preparation_hash=_HASH_B,
                lineage_artifact_hash=_HASH_C,
                completed_at=context.revealed_at - timedelta(seconds=3),
            ),
        ),
        sealed_at=context.revealed_at - timedelta(seconds=2),
    )


def _runtime(
    tmp_path: Path,
    clock: _Clock,
) -> tuple[Ledger, BlobStore, AuditedBenchmarkRuntime]:
    ledger = Ledger(tmp_path / "ledger.sqlite3")
    blobs = BlobStore(tmp_path / "blobs", compression=Compression.GZIP)
    return ledger, blobs, AuditedBenchmarkRuntime(ledger=ledger, blobs=blobs, clock=clock)


@pytest.mark.integration
def test_query_is_clocked_before_parse_and_persisted_before_return(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stage, manifest = _copy_stage(tmp_path)
    expected_context, evidence, _, _ = _expected_context_and_evidence(stage)
    clock = _Clock(
        iter(
            (
                expected_context.revealed_at - timedelta(seconds=1),
                expected_context.revealed_at + timedelta(seconds=1),
            )
        )
    )
    ledger, blobs, runtime = _runtime(tmp_path, clock)
    barrier = _barrier(evidence, expected_context)
    runtime.persist_prequery_barrier(barrier, release_class=ReleaseClass.PUBLIC)

    order: list[str] = []
    original_read_bytes = Path.read_bytes
    original_parse = query_runtime_module.QueryRevealArtifact.model_validate_json

    def observed_read(path: Path) -> bytes:
        if path.name == "query.json":
            order.append("read")
        return original_read_bytes(path)

    def observed_parse(payload: bytes):
        order.append("parse")
        return original_parse(payload)

    def observed_clock() -> datetime:
        order.append("clock")
        return clock()

    monkeypatch.setattr(Path, "read_bytes", observed_read)
    monkeypatch.setattr(
        query_runtime_module.QueryRevealArtifact,
        "model_validate_json",
        staticmethod(observed_parse),
    )
    runtime.clock = observed_clock
    try:
        opening = runtime.open_query(
            staging_root=stage,
            manifest=manifest,
            barrier=barrier,
            execution_id=barrier.execution_id,
            execution_manifest_hash=barrier.execution_manifest_hash,
            access_event_id="development-query-access-01",
        )
    finally:
        ledger.close()

    assert order[:3] == ["read", "clock", "parse"]
    assert opening.context == expected_context
    assert opening.access_event.query_context_hash == expected_context.content_hash
    assert opening.persistence.access_event_hash == opening.access_event.content_hash

    event_artifact = opening.persistence.access_event_artifact_hash
    with Ledger(tmp_path / "ledger.sqlite3") as reopened:
        persisted = reopened.get_query_access(opening.access_event.content_hash)
        stored = blobs.read_bytes(reopened.get_artifact(event_artifact))
    assert persisted == opening.persistence
    assert stored == (opening.access_event.to_canonical_json() + "\n").encode()


@pytest.mark.integration
def test_descriptor_bound_query_open_survives_stage_path_rename_without_following_replacement(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stage, manifest = _copy_stage(tmp_path)
    expected_context, evidence, _, _ = _expected_context_and_evidence(stage)
    manifest_sha256 = hashlib.sha256((stage / "manifest.json").read_bytes()).hexdigest()
    clock = _Clock(
        iter(
            (
                expected_context.revealed_at - timedelta(seconds=1),
                expected_context.revealed_at + timedelta(seconds=1),
            )
        )
    )
    ledger, _, runtime = _runtime(tmp_path, clock)
    barrier = _barrier(evidence, expected_context)
    runtime.persist_prequery_barrier(barrier, release_class=ReleaseClass.PUBLIC)

    attacker = tmp_path / "attacker-stage"
    shutil.copytree(stage, attacker)
    (attacker / "query.json").write_text("{}", encoding="utf-8")
    detached = tmp_path / "detached-stage"
    original_read = DescriptorBoundRuntimeStage.read_bytes
    swapped = False

    def swap_after_evidence(
        descriptor_stage: DescriptorBoundRuntimeStage,
        name: str,
    ) -> bytes:
        nonlocal swapped
        payload = original_read(descriptor_stage, name)
        if name == "evidence.json" and not swapped:
            stage.rename(detached)
            stage.symlink_to(attacker, target_is_directory=True)
            swapped = True
        return payload

    monkeypatch.setattr(DescriptorBoundRuntimeStage, "read_bytes", swap_after_evidence)
    try:
        opening = runtime.open_query(
            staging_root=None,
            manifest=manifest,
            barrier=barrier,
            execution_id=barrier.execution_id,
            execution_manifest_hash=barrier.execution_manifest_hash,
            access_event_id="descriptor-bound-query-access",
            repository=tmp_path,
            stage_relative_path="query-stage",
            manifest_file_sha256=manifest_sha256,
        )
    finally:
        ledger.close()

    assert swapped
    assert stage.is_symlink()
    assert opening.context == expected_context
    assert opening.reveal.content_hash == manifest.artifact_hashes[1]


@pytest.mark.integration
def test_descriptor_bound_query_open_rejects_byte_identical_leaf_inode_swap(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stage, manifest = _copy_stage(tmp_path)
    expected_context, evidence, _, _ = _expected_context_and_evidence(stage)
    replacement = tmp_path / "replacement-evidence.json"
    shutil.copy2(stage / "evidence.json", replacement)
    clock = _Clock(
        iter(
            (
                expected_context.revealed_at - timedelta(seconds=1),
                expected_context.revealed_at + timedelta(seconds=1),
            )
        )
    )
    ledger, _, runtime = _runtime(tmp_path, clock)
    barrier = _barrier(evidence, expected_context)
    runtime.persist_prequery_barrier(barrier, release_class=ReleaseClass.PUBLIC)
    original_read = DescriptorBoundRuntimeStage.read_bytes

    def replace_after_evidence(
        descriptor_stage: DescriptorBoundRuntimeStage,
        name: str,
    ) -> bytes:
        payload = original_read(descriptor_stage, name)
        if name == "evidence.json":
            replacement.replace(stage / "evidence.json")
        return payload

    monkeypatch.setattr(DescriptorBoundRuntimeStage, "read_bytes", replace_after_evidence)
    try:
        with pytest.raises(PermissionError, match="identity changed"):
            runtime.open_query(
                staging_root=None,
                manifest=manifest,
                barrier=barrier,
                execution_id=barrier.execution_id,
                execution_manifest_hash=barrier.execution_manifest_hash,
                access_event_id="descriptor-leaf-swap",
                repository=tmp_path,
                stage_relative_path="query-stage",
                manifest_file_sha256=hashlib.sha256(
                    (stage / "manifest.json").read_bytes()
                ).hexdigest(),
            )
    finally:
        ledger.close()


@pytest.mark.integration
def test_unpersisted_or_late_barrier_fails_before_query_read(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stage, manifest = _copy_stage(tmp_path)
    context, evidence, _, _ = _expected_context_and_evidence(stage)
    clock = _Clock(iter((context.revealed_at - timedelta(seconds=1),)))
    ledger, _, runtime = _runtime(tmp_path, clock)
    barrier = _barrier(evidence, context)
    query_reads = 0
    original_read_bytes = Path.read_bytes

    def observed_read(path: Path) -> bytes:
        nonlocal query_reads
        if path.name == "query.json":
            query_reads += 1
        return original_read_bytes(path)

    monkeypatch.setattr(Path, "read_bytes", observed_read)
    try:
        with pytest.raises(QueryOpeningError, match="persisted prequery barrier"):
            runtime.open_query(
                staging_root=stage,
                manifest=manifest,
                barrier=barrier,
                execution_id=barrier.execution_id,
                execution_manifest_hash=barrier.execution_manifest_hash,
                access_event_id="unpersisted-query-access",
            )
        late_binding = barrier.preparation_bindings[0].model_copy(
            update={"completed_at": barrier.sealed_at}
        )
        invalid = barrier.model_copy(update={"preparation_bindings": (late_binding,)})
        with pytest.raises(QueryOpeningError, match="preparation"):
            runtime.persist_prequery_barrier(invalid, release_class=ReleaseClass.PUBLIC)
    finally:
        ledger.close()
    assert query_reads == 0


@pytest.mark.integration
def test_query_access_registration_is_atomic_and_duplicate_safe(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stage, manifest = _copy_stage(tmp_path)
    context, evidence, _, _ = _expected_context_and_evidence(stage)
    clock = _Clock(
        iter(
            (
                context.revealed_at - timedelta(seconds=1),
                context.revealed_at + timedelta(seconds=1),
            )
        )
    )
    ledger, _, runtime = _runtime(tmp_path, clock)
    barrier = _barrier(evidence, context)
    runtime.persist_prequery_barrier(barrier, release_class=ReleaseClass.PUBLIC)
    artifact_count = ledger.count_rows("artifacts")
    original_insert = ledger._insert_immutable_row

    def fail_query_insert(cursor, *, table, key_column, fields):
        if table == "query_access_events":
            raise RuntimeError("injected query-access commit failure")
        return original_insert(
            cursor,
            table=table,
            key_column=key_column,
            fields=fields,
        )

    monkeypatch.setattr(ledger, "_insert_immutable_row", fail_query_insert)
    with pytest.raises(RuntimeError, match="injected query-access"):
        runtime.open_query(
            staging_root=stage,
            manifest=manifest,
            barrier=barrier,
            execution_id=barrier.execution_id,
            execution_manifest_hash=barrier.execution_manifest_hash,
            access_event_id="atomic-query-access",
        )
    assert ledger.count_rows("query_access_events") == 0
    assert ledger.count_rows("artifacts") == artifact_count
    ledger.close()


@pytest.mark.integration
def test_exact_access_and_materialization_replay_but_lineage_reuse_conflicts(
    tmp_path: Path,
) -> None:
    stage, manifest = _copy_stage(tmp_path)
    context, evidence, _, full_evidence = _expected_context_and_evidence(stage)
    access_at = context.revealed_at + timedelta(seconds=1)
    materialization_start = access_at + timedelta(seconds=1)
    materialization_end = materialization_start + timedelta(seconds=1)
    clock = _Clock(
        iter(
            (
                context.revealed_at - timedelta(seconds=1),
                access_at,
                materialization_start,
                materialization_end,
            )
        )
    )
    ledger, blobs, runtime = _runtime(tmp_path, clock)
    barrier = _barrier(evidence, context)
    runtime.persist_prequery_barrier(barrier, release_class=ReleaseClass.PUBLIC)
    opening = runtime.open_query(
        staging_root=stage,
        manifest=manifest,
        barrier=barrier,
        execution_id=barrier.execution_id,
        execution_manifest_hash=barrier.execution_manifest_hash,
        access_event_id="replay-query-access",
    )

    query_artifact = ledger.get_artifact(opening.persistence.query_payload_artifact_hash)
    access_artifact = ledger.get_artifact(opening.persistence.access_event_artifact_hash)
    assert (
        ledger.persist_query_access(
            opening.persistence,
            query_payload_artifact=query_artifact,
            access_event_artifact=access_artifact,
        )
        == opening.persistence
    )

    second_event_artifact = blobs.put_bytes(
        b'{"different":"access event"}\n',
        media_type="application/vnd.story-projection.query-access+json",
        release_class=ReleaseClass.PUBLIC,
        created_at=access_at + timedelta(seconds=1),
    )
    conflicting = replace(
        opening.persistence,
        access_event_hash="d" * 64,
        access_event_id="different-query-access",
        query_artifact_hash="e" * 64,
        access_event_artifact_hash=second_event_artifact.content_hash,
        accessed_at=(access_at + timedelta(seconds=1)).isoformat(),
    )
    with pytest.raises(DuplicateConflictError):
        ledger.persist_query_access(
            conflicting,
            query_payload_artifact=query_artifact,
            access_event_artifact=second_event_artifact,
        )

    packet = EvidencePacket(
        packet_id="development-query-packet",
        snapshot_hash=evidence.snapshot.content_hash,
        evidence=full_evidence,
        ordered_evidence_ids=tuple(item.evidence_id for item in full_evidence),
        retrieval_method=RetrievalMethod.ALL_ADMISSIBLE,
        token_count=1,
        created_at=materialization_start + timedelta(microseconds=1),
        release_class=ReleaseClass.PUBLIC,
    )
    materialized = runtime.materialize_packet(
        opening,
        materialization_event_id="development-packet-materialization",
        retrieval_config_hash=_HASH_B,
        materializer=lambda _evidence, _reveal: packet,
    )
    packet_artifact = ledger.get_artifact(materialized.persistence.packet_artifact_hash)
    materialization_artifact = ledger.get_artifact(
        materialized.persistence.materialization_event_artifact_hash
    )
    assert (
        ledger.persist_packet_materialization(
            materialized.persistence,
            packet_artifact=packet_artifact,
            materialization_event_artifact=materialization_artifact,
        )
        == materialized.persistence
    )
    assert ledger.count_rows("packet_materialization_events") == 1
    ledger.close()


def test_query_runtime_never_imports_scorer_or_compiler_modules() -> None:
    source = Path("src/story_projection_onto/query_runtime.py").read_text(encoding="utf-8")
    assert "story_projection_onto.synthetic_benchmark" not in source
    assert "story_projection_onto.benchmark import" not in source
    assert "scorer_only" not in source
    assert "GoldContextualProjection" not in source
