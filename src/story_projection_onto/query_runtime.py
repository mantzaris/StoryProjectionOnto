"""Audited query opening and post-query packet materialization.

The compiler-facing loaders in :mod:`story_projection_onto.benchmark_runtime`
remain read-only verification utilities.  This module is the execution boundary:
it refuses to read ``query.json`` until a persisted pre-query barrier is proven,
then durably records the exact bytes and physical access event before returning
the revealed query to its caller.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path

from story_projection_onto.benchmark_runtime import (
    GoldFirewallError,
    ModelEligibleWorldArtifact,
    QueryRevealArtifact,
    RuntimeStageKind,
    RuntimeStagingManifest,
    _assert_exact_stage_directory,
    _resolve_flat_runtime_file,
    _verified_bytes,
    model_request_payload,
    preconstruction_request_payload,
    scan_model_payload,
)
from story_projection_onto.contracts import (
    EvidencePacket,
    PacketMaterializationEvent,
    PrequeryBarrier,
    QueryAccessEvent,
    QueryContext,
    to_model_visible_query,
)
from story_projection_onto.store import (
    ArtifactIntegrityError,
    BlobStore,
    Ledger,
    PacketMaterializationRecord,
    PrequeryBarrierRecord,
    QueryAccessRecord,
    ReleaseClass,
)


class QueryOpeningError(RuntimeError):
    """The physical query-access protocol could not be proven and persisted."""


@dataclass(frozen=True, slots=True)
class AuditedQueryOpening:
    evidence: ModelEligibleWorldArtifact
    reveal: QueryRevealArtifact
    context: QueryContext
    access_event: QueryAccessEvent
    persistence: QueryAccessRecord


@dataclass(frozen=True, slots=True)
class AuditedPacketMaterialization:
    packet: EvidencePacket
    event: PacketMaterializationEvent
    persistence: PacketMaterializationRecord


def _system_utc_now() -> datetime:
    return datetime.now(UTC)


def _canonical_record_bytes(record: object) -> bytes:
    serializer = getattr(record, "to_canonical_json", None)
    if not callable(serializer):
        raise TypeError("audited CAS records require canonical immutable serialization")
    return (str(serializer()) + "\n").encode("utf-8")


class AuditedBenchmarkRuntime:
    """Persistence-first execution facade around the sealed benchmark stages."""

    def __init__(
        self,
        *,
        ledger: Ledger,
        blobs: BlobStore,
        clock: Callable[[], datetime] = _system_utc_now,
    ) -> None:
        self.ledger = ledger
        self.blobs = blobs
        self.clock = clock

    def _utc_now(self) -> datetime:
        observed = self.clock()
        if (
            not isinstance(observed, datetime)
            or observed.tzinfo is None
            or observed.utcoffset() != timedelta(0)
        ):
            raise QueryOpeningError("trusted query runtime clock must return an aware UTC time")
        return observed

    def persist_prequery_barrier(
        self,
        barrier: PrequeryBarrier,
        *,
        release_class: ReleaseClass,
    ) -> PrequeryBarrierRecord:
        """Persist a barrier before any corresponding query stage is opened."""

        self._validate_barrier_timing(barrier)

        try:
            existing = self.ledger.get_prequery_barrier(barrier.content_hash)
        except KeyError:
            existing = None
        if existing is not None:
            if existing.release_class is not release_class:
                raise QueryOpeningError(
                    "prequery barrier release class differs from persisted lineage"
                )
            self._verify_persisted_barrier(barrier, existing)
            return existing
        persisted_at = self._utc_now()
        if persisted_at < barrier.sealed_at:
            raise QueryOpeningError("prequery barrier cannot be persisted before its seal time")
        artifact = self.blobs.put_bytes(
            _canonical_record_bytes(barrier),
            media_type="application/vnd.story-projection.prequery-barrier+json",
            release_class=release_class,
            created_at=persisted_at,
        )
        record = PrequeryBarrierRecord(
            barrier_hash=barrier.content_hash,
            barrier_id=barrier.barrier_id,
            execution_id=barrier.execution_id,
            execution_manifest_hash=barrier.execution_manifest_hash,
            barrier_artifact_hash=artifact.content_hash,
            preparation_count=len(barrier.preparation_bindings),
            sealed_at=barrier.sealed_at.isoformat(),
            persisted_at=persisted_at.isoformat(),
            release_class=release_class,
        )
        return self.ledger.persist_prequery_barrier(
            record,
            barrier_artifact=artifact,
        )

    def _verify_persisted_barrier(
        self,
        barrier: PrequeryBarrier,
        record: PrequeryBarrierRecord,
    ) -> None:
        self._validate_barrier_timing(barrier)
        if (
            record.barrier_hash != barrier.content_hash
            or record.barrier_id != barrier.barrier_id
            or record.execution_id != barrier.execution_id
            or record.execution_manifest_hash != barrier.execution_manifest_hash
            or record.preparation_count != len(barrier.preparation_bindings)
            or datetime.fromisoformat(record.sealed_at.replace("Z", "+00:00"))
            != barrier.sealed_at
        ):
            raise QueryOpeningError("supplied prequery barrier differs from persisted lineage")
        try:
            artifact = self.ledger.get_artifact(record.barrier_artifact_hash)
            persisted_bytes = self.blobs.read_bytes(
                artifact,
                allow_restricted=record.release_class is ReleaseClass.RESTRICTED,
            )
        except (ArtifactIntegrityError, KeyError) as exc:
            raise QueryOpeningError("persisted prequery barrier CAS verification failed") from exc
        if persisted_bytes != _canonical_record_bytes(barrier):
            raise QueryOpeningError("persisted prequery barrier CAS bytes changed")

    @staticmethod
    def _validate_barrier_timing(barrier: PrequeryBarrier) -> None:
        if any(
            binding.completed_at >= barrier.sealed_at
            for binding in barrier.preparation_bindings
        ):
            raise QueryOpeningError(
                "every prequery preparation must complete before the barrier seal"
            )

    def open_query(
        self,
        *,
        staging_root: Path,
        manifest: RuntimeStagingManifest,
        barrier: PrequeryBarrier,
        execution_id: str,
        execution_manifest_hash: str,
        access_event_id: str,
    ) -> AuditedQueryOpening:
        """Open and persist exactly one query, exposing it only after commit."""

        if manifest.stage_kind is not RuntimeStageKind.QUERY_REVEALED:
            raise GoldFirewallError("audited query opening requires a query-revealed stage")
        if (
            barrier.execution_id != execution_id
            or barrier.execution_manifest_hash != execution_manifest_hash
        ):
            raise QueryOpeningError("prequery barrier belongs to another execution manifest")
        try:
            barrier_record = self.ledger.get_prequery_barrier(barrier.content_hash)
        except KeyError as exc:
            raise QueryOpeningError("query opening requires a persisted prequery barrier") from exc
        self._verify_persisted_barrier(barrier, barrier_record)

        # Directory and manifest verification is deliberately completed before
        # the first read of query.json.
        _assert_exact_stage_directory(staging_root, manifest)
        if manifest.artifact_hashes[0] not in barrier.neutral_evidence_artifact_hashes:
            raise QueryOpeningError("query stage evidence is absent from the prequery barrier")
        evidence_bytes = _verified_bytes(staging_root, manifest, "evidence.json")
        evidence = ModelEligibleWorldArtifact.model_validate_json(evidence_bytes)
        scan_model_payload(evidence.model_dump(mode="json"))
        preconstruction_request_payload(evidence)
        if evidence.content_hash not in barrier.neutral_evidence_artifact_hashes:
            raise QueryOpeningError("verified evidence artifact is absent from the barrier")

        query_path = _resolve_flat_runtime_file(staging_root / "query.json", staging_root)
        query_bytes = query_path.read_bytes()
        accessed_at = self._utc_now()
        # No query parsing, validation, callback, or model-visible return occurs
        # between the physical read above and this trusted timestamp. Verification
        # deliberately occurs below: even parsing an embedded checksum before the
        # timestamp would make the claimed access boundary too late.
        reveal = QueryRevealArtifact.model_validate_json(query_bytes)
        if reveal.content_hash != manifest.artifact_hashes[1]:
            raise GoldFirewallError("query artifact hash disagrees with staging manifest")
        scan_model_payload(reveal.model_dump(mode="json"))
        model_request_payload(evidence, reveal)
        if not reveal.reveal_id.startswith("reveal_"):
            raise QueryOpeningError("synthetic query reveal lacks its frozen opaque identifier")
        context = QueryContext(
            context_id=f"ctx_{reveal.reveal_id.removeprefix('reveal_')}",
            revealed_at=reveal.revealed_at,
            **reveal.query.model_dump(
                mode="python",
                exclude={"schema_version", "content_hash"},
            ),
        )
        if to_model_visible_query(context) != reveal.query:
            raise QueryOpeningError("reconstructed query context differs from its reveal")
        if barrier.sealed_at >= accessed_at:
            raise QueryOpeningError("prequery barrier was not sealed before physical query access")
        persisted_at = datetime.fromisoformat(
            barrier_record.persisted_at.replace("Z", "+00:00")
        )
        if persisted_at >= accessed_at:
            raise QueryOpeningError("prequery barrier was not persisted before query access")
        if barrier_record.release_class.value != evidence.snapshot.release_class.value:
            raise QueryOpeningError("barrier release class differs from staged evidence")

        access = QueryAccessEvent(
            access_event_id=access_event_id,
            execution_id=execution_id,
            query_context_hash=context.content_hash,
            model_visible_query_hash=reveal.query.content_hash,
            snapshot_hash=evidence.snapshot.content_hash,
            stage_manifest_hash=manifest.content_hash,
            query_artifact_hash=reveal.content_hash,
            prequery_barrier_hash=barrier.content_hash,
            packet_hash=None,
            registered_revealed_at=reveal.revealed_at,
            accessed_at=accessed_at,
        )
        release = barrier_record.release_class
        query_artifact = self.blobs.put_bytes(
            query_bytes,
            media_type="application/vnd.story-projection.query-reveal+json",
            release_class=release,
            created_at=accessed_at,
        )
        access_artifact = self.blobs.put_bytes(
            _canonical_record_bytes(access),
            media_type="application/vnd.story-projection.query-access+json",
            release_class=release,
            created_at=accessed_at,
        )
        persistence = self.ledger.persist_query_access(
            QueryAccessRecord(
                access_event_hash=access.content_hash,
                access_event_id=access.access_event_id,
                execution_id=access.execution_id,
                query_context_hash=access.query_context_hash,
                model_visible_query_hash=access.model_visible_query_hash,
                snapshot_hash=access.snapshot_hash,
                stage_manifest_hash=access.stage_manifest_hash,
                query_artifact_hash=access.query_artifact_hash,
                prequery_barrier_hash=access.prequery_barrier_hash,
                packet_hash=None,
                query_payload_artifact_hash=query_artifact.content_hash,
                access_event_artifact_hash=access_artifact.content_hash,
                registered_revealed_at=access.registered_revealed_at.isoformat(),
                accessed_at=access.accessed_at.isoformat(),
                release_class=release,
            ),
            query_payload_artifact=query_artifact,
            access_event_artifact=access_artifact,
        )
        return AuditedQueryOpening(
            evidence=evidence,
            reveal=reveal,
            context=context,
            access_event=access,
            persistence=persistence,
        )

    def materialize_packet(
        self,
        opening: AuditedQueryOpening,
        *,
        materialization_event_id: str,
        retrieval_config_hash: str,
        materializer: Callable[
            [ModelEligibleWorldArtifact, QueryRevealArtifact],
            EvidencePacket,
        ],
    ) -> AuditedPacketMaterialization:
        """Time, execute, and persist one query-dependent packet operation."""

        persisted_access = self.ledger.get_query_access(opening.access_event.content_hash)
        if (
            persisted_access.access_event_artifact_hash
            != opening.persistence.access_event_artifact_hash
            or persisted_access.packet_hash is not None
        ):
            raise QueryOpeningError("query access persistence changed before materialization")
        started_at = self._utc_now()
        if started_at < opening.access_event.accessed_at:
            raise QueryOpeningError("packet materialization cannot predate query access")
        packet = materializer(opening.evidence, opening.reveal)
        completed_at = self._utc_now()
        if completed_at < started_at:
            raise QueryOpeningError("trusted clock moved backwards during materialization")
        if (
            packet.snapshot_hash != opening.access_event.snapshot_hash
            or packet.created_at < started_at
            or packet.created_at > completed_at
        ):
            raise QueryOpeningError("materialized packet has invalid snapshot or timing")
        event = PacketMaterializationEvent(
            materialization_event_id=materialization_event_id,
            execution_id=opening.access_event.execution_id,
            query_access_event_hash=opening.access_event.content_hash,
            snapshot_hash=opening.access_event.snapshot_hash,
            packet_hash=packet.content_hash,
            retrieval_method=packet.retrieval_method,
            retrieval_config_hash=retrieval_config_hash,
            started_at=started_at,
            completed_at=completed_at,
        )
        release = opening.persistence.release_class
        packet_artifact = self.blobs.put_bytes(
            _canonical_record_bytes(packet),
            media_type="application/vnd.story-projection.evidence-packet+json",
            release_class=release,
            created_at=completed_at,
        )
        event_artifact = self.blobs.put_bytes(
            _canonical_record_bytes(event),
            media_type="application/vnd.story-projection.packet-materialization+json",
            release_class=release,
            created_at=completed_at,
        )
        persistence = self.ledger.persist_packet_materialization(
            PacketMaterializationRecord(
                materialization_event_hash=event.content_hash,
                materialization_event_id=event.materialization_event_id,
                execution_id=event.execution_id,
                query_access_event_hash=event.query_access_event_hash,
                snapshot_hash=event.snapshot_hash,
                packet_hash=event.packet_hash,
                retrieval_method=event.retrieval_method.value,
                retrieval_config_hash=event.retrieval_config_hash,
                packet_artifact_hash=packet_artifact.content_hash,
                materialization_event_artifact_hash=event_artifact.content_hash,
                started_at=event.started_at.isoformat(),
                completed_at=event.completed_at.isoformat(),
                release_class=release,
            ),
            packet_artifact=packet_artifact,
            materialization_event_artifact=event_artifact,
        )
        return AuditedPacketMaterialization(
            packet=packet,
            event=event,
            persistence=persistence,
        )


__all__ = [
    "AuditedBenchmarkRuntime",
    "AuditedPacketMaterialization",
    "AuditedQueryOpening",
    "QueryOpeningError",
]
