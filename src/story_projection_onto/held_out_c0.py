"""Production CPU-only C0 adapter for the reviewed held-out controller.

The adapter deliberately has two different input boundaries.  Preconstruction
accepts only :class:`HeldOutPrequeryUnit`, opens the exact evidence-only stage and
its projection-equivalent neutral stage, and durably stores one complete sealed
preontology.  Projection accepts an already-audited ``HeldOutQueryOpening`` and
loads the revealed query only from the query-access CAS record.  It delegates the
actual selection to :func:`project_sealed_c0`; no semantic object is created after
query access.

The production loader is the pinned local spaCy loader from ``conditions.c0``.  It
does not download a model and this module has no GPU or remote-service dependency.
"""

from __future__ import annotations

import hashlib
import os
import re
import tempfile
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, Literal, Protocol, TypeVar, cast

from story_projection_onto.benchmark_runtime import (
    EvidenceProjectionEquivalenceCertificate,
    ModelEligibleWorldArtifact,
    NeutralEvidenceArtifact,
    QueryRevealArtifact,
    RuntimeStageKind,
    RuntimeStagingManifest,
    load_staged_neutral_evidence,
    load_staged_world,
)
from story_projection_onto.conditions.base import (
    SCORED_PROJECTION_SCHEMA_HASH,
    ConditionPreparation,
    RunConditionConfig,
    sealed_semantic_ids,
)
from story_projection_onto.conditions.c0 import (
    ClassicalBackendManifest,
    ClassicalPreBuilder,
    load_classical_rule_config,
    load_production_classical_builder,
    project_sealed_c0,
)
from story_projection_onto.contracts import (
    ConditionName,
    EvidencePacket,
    ImmutableRecord,
    PacketMaterializationEvent,
    PrequeryBarrier,
    QueryAccessEvent,
    QueryContext,
    RetrievalMethod,
    Sha256Digest,
    canonical_sha256,
    to_model_visible_query,
)
from story_projection_onto.development_adapter import DevelopmentConstructionConfiguration
from story_projection_onto.development_continuation import development_validator_hash
from story_projection_onto.development_runtime import RunOutcome
from story_projection_onto.held_out_controller import (
    C0ConstructionReceipt,
    PreconstructedProjectionReceipt,
)
from story_projection_onto.held_out_primary import (
    HeldOutCallManifest,
    HeldOutCASReference,
    HeldOutControlConfiguration,
    HeldOutControlError,
    HeldOutPrequeryUnit,
    HeldOutQueryOpening,
    HeldOutUnitPlan,
    PublicStageReference,
)
from story_projection_onto.store import ArtifactStore
from story_projection_onto.store import ReleaseClass as StoreReleaseClass

# These byte hashes are the frozen values in both accepted fallback-development
# source-tree manifests.  Pinning bytes in addition to the typed configuration
# prevents a post-reveal, self-consistent rewrite of tunable rule vocabulary.
PINNED_C0_RULES_FILE_SHA256 = "c6613daf5e0ea4a2340654042a23fb6f4b4b8b0cedafac01f697a34210062180"
PINNED_DEVELOPMENT_CONSTRUCTION_FILE_SHA256 = (
    "80afe9a6db2b1e6cfcacd00da9a1f33f3285f56122ad77d2e089d07e6c5def9a"
)

_NEUTRAL_ROOT = Path("data/synthetic/condition_inputs/neutral_evidence")
_OPAQUE_STAGE = re.compile(r"^artifact_([0-9a-f]{20})$")
_SAFE_UNIT = re.compile(r"^[a-z0-9][a-z0-9-]*$")


class HeldOutC0Error(HeldOutControlError):
    """A held-out C0 dependency or immutable lineage record changed."""


class _C1ProjectionDelegate(Protocol):
    def project_preconstructed(
        self,
        *,
        unit: HeldOutUnitPlan,
        query_stage_hash: str,
        query_opening: HeldOutQueryOpening,
        condition: ConditionName,
        seed_block: int | None,
        construction_seal_hash: str,
        complete_graph_hash: str,
    ) -> PreconstructedProjectionReceipt: ...


BuilderLoader = Callable[[Path], tuple[ClassicalPreBuilder, ClassicalBackendManifest]]
RecordT = TypeVar("RecordT", bound=ImmutableRecord)


class HeldOutC0PreparationState(ImmutableRecord):
    """Private durable pointer to one exact query-blind C0 preparation."""

    state_version: Literal["held-out-c0-preparation-v1"] = "held-out-c0-preparation-v1"
    unit_id: str
    prequery_stage_hash: Sha256Digest
    model_visible_evidence_hash: Sha256Digest
    neutral_stage_manifest_hash: Sha256Digest
    neutral_evidence_hash: Sha256Digest
    equivalence_certificate_hash: Sha256Digest
    rule_config_hash: Sha256Digest
    backend_manifest_hash: Sha256Digest
    construction_configuration_hash: Sha256Digest
    preparation: HeldOutCASReference
    construction_seal_hash: Sha256Digest
    complete_graph_hash: Sha256Digest

    def validate_pointer(self) -> None:
        if self.preparation.object_kind != "condition_preparation":
            raise HeldOutC0Error("C0 state points to another CAS object kind")


@dataclass(frozen=True, slots=True)
class _HeldOutC0Inputs:
    """Held-out projection envelope after controller/CAS checks.

    ``ProduceInputs`` cannot represent the controller's registered C0 barrier:
    that barrier intentionally uses ``C0ConstructionReceipt.content_hash`` as its
    preparation handle.  The pure C0 projector consumes the same typed members and
    performs no construction, so this envelope preserves that controller contract
    without weakening ``ProduceInputs`` for other pathways.
    """

    preparation: ConditionPreparation
    snapshot: Any
    packet: EvidencePacket
    context: QueryContext
    query_access: QueryAccessEvent
    prequery_barrier: PrequeryBarrier
    query_processing_started_at: datetime
    packet_materialization: PacketMaterializationEvent
    upper_ontology: Any
    run_config: RunConditionConfig
    revisions: tuple[Any, ...] = ()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _aware_utc(value: datetime) -> datetime:
    if value.tzinfo is None or value.utcoffset() is None:
        raise HeldOutC0Error("C0 clock must return a timezone-aware timestamp")
    return value.astimezone(UTC)


def _ledger_timestamp(value: str) -> datetime:
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise HeldOutC0Error("C0 ledger contains an invalid timestamp") from exc
    return _aware_utc(parsed)


@dataclass(slots=True)
class ProductionHeldOutC0Adapter:
    """Concrete query-blind C0 builder and fixed held-out projector.

    A C1 CPU projection delegate may be supplied because the controller's historic
    ``InjectedHeldOutCpu`` protocol multiplexes C0 and C1 projection.  With no
    delegate, this object fails closed for successful C1 projection and remains a
    complete adapter for every C0 method invocation.
    """

    repository: Path
    call_manifest: HeldOutCallManifest
    configuration: HeldOutControlConfiguration
    artifacts: ArtifactStore
    state_directory: Path
    c1_projection_delegate: _C1ProjectionDelegate | None = field(default=None, repr=False)
    builder_loader: BuilderLoader = field(
        default=load_production_classical_builder,
        repr=False,
    )
    clock: Callable[[], datetime] = lambda: datetime.now(UTC)
    backend: Literal["spacy-ner-dependency-plus-deterministic-rules-v2"] = (
        "spacy-ner-dependency-plus-deterministic-rules-v2"
    )
    builder: ClassicalPreBuilder = field(init=False, repr=False)
    backend_manifest: ClassicalBackendManifest = field(init=False)
    construction: DevelopmentConstructionConfiguration = field(init=False, repr=False)
    validator_hash: str = field(init=False)

    def __post_init__(self) -> None:
        self.repository = self.repository.resolve(strict=True)
        self._validate_plan_binding()

        rules_path = self._pinned_file(
            Path("configs/study/c0_rules.json"), PINNED_C0_RULES_FILE_SHA256
        )
        construction_path = self._pinned_file(
            Path("configs/study/development_construction.json"),
            PINNED_DEVELOPMENT_CONSTRUCTION_FILE_SHA256,
        )
        frozen_rules = load_classical_rule_config(rules_path)
        self.builder, self.backend_manifest = self.builder_loader(rules_path)
        self.construction = DevelopmentConstructionConfiguration.load(
            construction_path,
            expected_sha256=PINNED_DEVELOPMENT_CONSTRUCTION_FILE_SHA256,
        )
        if (
            self.builder.config != frozen_rules
            or self.builder.config.content_hash != self.backend_manifest.config_hash
            or self.backend_manifest.backend_name != self.backend
            or self.builder.candidate_backend.backend_name != self.backend
        ):
            raise HeldOutC0Error("C0 builder differs from the frozen spaCy rule manifest")
        self.validator_hash = development_validator_hash(self.repository)

        if self.state_directory.is_symlink():
            raise HeldOutC0Error("C0 state directory cannot be a symlink")
        self.state_directory.mkdir(mode=0o700, parents=True, exist_ok=True)
        self.state_directory = self.state_directory.resolve(strict=True)
        if any(
            marker in {part.casefold() for part in self.state_directory.parts}
            for marker in ("scorer_only", "query_stages", "gold")
        ):
            raise HeldOutC0Error("C0 preparation state is inside a forbidden namespace")
        os.chmod(self.state_directory, 0o700)

    def _validate_plan_binding(self) -> None:
        payload = self.configuration.model_dump(
            mode="python", exclude={"content_hash", "expected_plan_hash"}
        )
        if (
            canonical_sha256(payload) != self.call_manifest.control_configuration_semantic_hash
            or self.call_manifest.benchmark_manifest_file_sha256
            != self.configuration.benchmark_manifest_file_sha256
            or (
                self.configuration.expected_plan_hash != "PENDING"
                and self.configuration.expected_plan_hash != self.call_manifest.content_hash
            )
        ):
            raise HeldOutC0Error("C0 adapter received a different held-out control plan")
        prequery_root = Path(self.configuration.prequery_stage_root)
        query_root = Path(self.configuration.query_stage_root)
        if (
            prequery_root.is_absolute()
            or query_root.is_absolute()
            or ".." in prequery_root.parts
            or ".." in query_root.parts
        ):
            raise HeldOutC0Error("held-out stage roots must be safe repository-relative paths")
        for unit in self.call_manifest.units:
            if Path(unit.prequery_stage.relative_path).parent != prequery_root:
                raise HeldOutC0Error("C0 plan contains a prequery stage outside its frozen root")
            if any(Path(stage.relative_path).parent != query_root for stage in unit.query_stages):
                raise HeldOutC0Error("held-out query stage escaped its frozen root")

    def _pinned_file(self, relative: Path, expected_sha256: str) -> Path:
        if relative.is_absolute() or ".." in relative.parts:
            raise HeldOutC0Error("C0 dependency path is unsafe")
        path = self.repository / relative
        if path.is_symlink() or not path.is_file():
            raise HeldOutC0Error(f"missing or symlinked C0 dependency: {relative}")
        resolved = path.resolve(strict=True)
        if (
            not resolved.is_relative_to(self.repository)
            or _file_sha256(resolved) != expected_sha256
        ):
            raise HeldOutC0Error(f"frozen C0 dependency bytes changed: {relative}")
        return resolved

    def _now_after(self, threshold: datetime) -> datetime:
        observed = _aware_utc(self.clock())
        return max(observed, _aware_utc(threshold) + timedelta(microseconds=1))

    def _registered_prequery(self, unit: HeldOutPrequeryUnit) -> HeldOutUnitPlan:
        matches = tuple(item for item in self.call_manifest.units if item.unit_id == unit.unit_id)
        if len(matches) != 1 or matches[0].prequery_stage != unit.prequery_stage:
            raise HeldOutC0Error("C0 construction unit differs from the reviewed held-out plan")
        return matches[0]

    def _registered_unit(self, unit: HeldOutUnitPlan) -> HeldOutUnitPlan:
        matches = tuple(item for item in self.call_manifest.units if item.unit_id == unit.unit_id)
        if len(matches) != 1 or matches[0] != unit:
            raise HeldOutC0Error("C0 projection unit differs from the reviewed held-out plan")
        return matches[0]

    def _stage_root(self, reference: PublicStageReference, expected_parent: Path) -> Path:
        logical = Path(reference.relative_path)
        if logical.is_absolute() or ".." in logical.parts or logical.parent != expected_parent:
            raise HeldOutC0Error("held-out stage path escaped its exact allowlisted root")
        root = self.repository / logical
        manifest_path = root / "manifest.json"
        if root.is_symlink() or manifest_path.is_symlink():
            raise HeldOutC0Error("held-out stage paths cannot be symlinks")
        resolved = root.resolve(strict=True)
        if not resolved.is_relative_to(self.repository):
            raise HeldOutC0Error("held-out stage escaped the repository")
        return resolved

    def _manifest(
        self,
        root: Path,
        reference: PublicStageReference,
        required_kind: RuntimeStageKind,
    ) -> RuntimeStagingManifest:
        raw = (root / "manifest.json").read_bytes()
        if hashlib.sha256(raw).hexdigest() != reference.manifest_file_sha256:
            raise HeldOutC0Error("held-out stage manifest bytes changed")
        manifest = RuntimeStagingManifest.model_validate_json(raw)
        if (
            manifest.content_hash != reference.staging_manifest_hash
            or manifest.stage_id != reference.stage_id
            or manifest.stage_kind is not required_kind
            or manifest.stage_kind.value != reference.stage_kind
        ):
            raise HeldOutC0Error("held-out stage manifest differs from its reviewed reference")
        return manifest

    def _load_query_blind_evidence(
        self,
        reference: PublicStageReference,
    ) -> tuple[
        ModelEligibleWorldArtifact,
        NeutralEvidenceArtifact,
        EvidenceProjectionEquivalenceCertificate,
        RuntimeStagingManifest,
    ]:
        prequery_parent = Path(self.configuration.prequery_stage_root)
        prequery_root = self._stage_root(reference, prequery_parent)
        prequery_manifest = self._manifest(
            prequery_root, reference, RuntimeStageKind.PREQUERY_EVIDENCE
        )
        visible = load_staged_world(
            prequery_root / "evidence.json",
            prequery_root,
            prequery_manifest,
        )
        if (
            visible.content_hash != reference.evidence_artifact_hash
            or visible.snapshot.content_hash != reference.snapshot_hash
        ):
            raise HeldOutC0Error("prequery evidence differs from its reviewed stage reference")

        match = _OPAQUE_STAGE.fullmatch(prequery_root.name)
        if match is None:
            raise HeldOutC0Error("prequery evidence lacks its opaque stage identity")
        neutral_parent_path = self.repository / _NEUTRAL_ROOT
        if neutral_parent_path.is_symlink():
            raise HeldOutC0Error("neutral evidence root cannot be a symlink")
        neutral_parent = neutral_parent_path.resolve(strict=True)
        if (
            not neutral_parent.is_relative_to(self.repository)
            or neutral_parent.name != "neutral_evidence"
            or neutral_parent.parent.name != "condition_inputs"
        ):
            raise HeldOutC0Error("neutral evidence root differs from its query-blind namespace")
        neutral_root = neutral_parent / f"neutral_{match.group(1)}"
        if neutral_root.is_symlink():
            raise HeldOutC0Error("neutral evidence stage cannot be a symlink")
        neutral_root = neutral_root.resolve(strict=True)
        neutral_manifest = RuntimeStagingManifest.model_validate_json(
            (neutral_root / "manifest.json").read_bytes()
        )
        neutral, certificate = load_staged_neutral_evidence(
            neutral_root,
            neutral_parent,
            neutral_manifest,
            visible,
        )
        if (
            neutral.snapshot != visible.snapshot
            or certificate.model_visible_evidence_artifact_hash != visible.content_hash
            or certificate.neutral_evidence_artifact_hash != neutral.content_hash
        ):
            raise HeldOutC0Error("neutral evidence equivalence lineage changed")
        return visible, neutral, certificate, neutral_manifest

    def _state_path(self, unit_id: str) -> Path:
        if _SAFE_UNIT.fullmatch(unit_id) is None:
            raise HeldOutC0Error("unsafe C0 held-out unit identifier")
        return self.state_directory / f"{unit_id}.json"

    def _persist_record(
        self,
        value: ImmutableRecord,
        *,
        object_kind: str,
        created_at: datetime,
    ) -> HeldOutCASReference:
        media_type = f"application/vnd.story-projection.{object_kind.replace('_', '-')}+json"
        artifact = self.artifacts.put_bytes(
            (value.to_canonical_json() + "\n").encode("utf-8"),
            media_type=media_type,
            release_class=StoreReleaseClass.RESTRICTED,
            created_at=created_at,
        )
        return HeldOutCASReference(
            artifact_hash=artifact.content_hash,
            logical_content_hash=value.content_hash,
            object_kind=cast(Any, object_kind),
            media_type=media_type,
            release_class="restricted",
        )

    def _resolve_reference(
        self,
        reference: HeldOutCASReference,
        model: type[RecordT],
    ) -> RecordT:
        artifact = self.artifacts.ledger.get_artifact(reference.artifact_hash)
        if (
            artifact.media_type != reference.media_type
            or artifact.release_class.value != reference.release_class
        ):
            raise HeldOutC0Error("C0 CAS reference metadata changed")
        raw = self.artifacts.blobs.read_bytes(
            artifact,
            allow_restricted=artifact.release_class is StoreReleaseClass.RESTRICTED,
        )
        value = model.model_validate_json(raw)
        if value.content_hash != reference.logical_content_hash:
            raise HeldOutC0Error("C0 CAS logical object hash changed")
        return value

    def _write_state(self, state: HeldOutC0PreparationState) -> None:
        destination = self._state_path(state.unit_id)
        payload = (state.to_canonical_json() + "\n").encode("utf-8")
        if destination.exists():
            if destination.is_symlink() or destination.read_bytes() != payload:
                raise HeldOutC0Error("durable C0 preparation pointer changed")
            return
        descriptor, name = tempfile.mkstemp(
            prefix=f".{destination.name}.",
            suffix=".tmp",
            dir=destination.parent,
        )
        temporary = Path(name)
        try:
            os.fchmod(descriptor, 0o600)
            with os.fdopen(descriptor, "wb") as stream:
                stream.write(payload)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, destination)
        finally:
            temporary.unlink(missing_ok=True)

    def _load_state(
        self,
        unit: HeldOutUnitPlan,
    ) -> tuple[HeldOutC0PreparationState, ConditionPreparation, NeutralEvidenceArtifact]:
        path = self._state_path(unit.unit_id)
        if path.is_symlink() or not path.is_file():
            raise HeldOutC0Error("sealed held-out C0 preparation is unavailable")
        state = HeldOutC0PreparationState.model_validate_json(path.read_bytes())
        state.validate_pointer()
        visible, neutral, certificate, neutral_manifest = self._load_query_blind_evidence(
            unit.prequery_stage
        )
        if (
            state.unit_id != unit.unit_id
            or state.prequery_stage_hash != unit.prequery_stage.staging_manifest_hash
            or state.model_visible_evidence_hash != visible.content_hash
            or state.neutral_stage_manifest_hash != neutral_manifest.content_hash
            or state.neutral_evidence_hash != neutral.content_hash
            or state.equivalence_certificate_hash != certificate.content_hash
            or state.rule_config_hash != self.builder.config.content_hash
            or state.backend_manifest_hash != self.backend_manifest.content_hash
            or state.construction_configuration_hash != self.construction.content_hash
        ):
            raise HeldOutC0Error("sealed C0 state differs from its frozen prequery inputs")
        preparation = self._resolve_reference(state.preparation, ConditionPreparation)
        preontology = preparation.sealed_preontology
        if (
            preontology is None
            or preparation.condition is not ConditionName.C0_CLASSICAL_PRE
            or preparation.snapshot_hash != neutral.snapshot.content_hash
            or state.preparation.logical_content_hash != preparation.content_hash
            or state.construction_seal_hash != preontology.construction_seal.content_hash
            or state.complete_graph_hash != preontology.construction_seal.ontology_hash
            or set(preontology.construction_seal.sealed_object_ids)
            != set(sealed_semantic_ids(preontology.draft))
        ):
            raise HeldOutC0Error("stored C0 preparation is not its complete sealed graph")
        return state, preparation, neutral

    @staticmethod
    def _construction_receipt(
        unit: HeldOutUnitPlan,
        preparation: ConditionPreparation,
    ) -> C0ConstructionReceipt:
        preontology = preparation.sealed_preontology
        if preontology is None:
            raise HeldOutC0Error("C0 preparation lost its sealed preontology")
        return C0ConstructionReceipt(
            unit_id=unit.unit_id,
            prequery_stage_hash=unit.prequery_stage.staging_manifest_hash,
            outcome=RunOutcome.SUCCEEDED,
            construction_seal_hash=preontology.construction_seal.content_hash,
            complete_graph_hash=preontology.construction_seal.ontology_hash,
            completed_at=preparation.completed_at,
        )

    def build_c0(self, unit: HeldOutPrequeryUnit) -> C0ConstructionReceipt:
        """Build exactly one complete C0 graph without opening any query stage."""

        planned = self._registered_prequery(unit)
        path = self._state_path(unit.unit_id)
        if path.exists():
            _, preparation, _ = self._load_state(planned)
            return self._construction_receipt(planned, preparation)

        visible, neutral, certificate, neutral_manifest = self._load_query_blind_evidence(
            unit.prequery_stage
        )
        constructed_at = self._now_after(neutral.snapshot.sealed_at)
        sealed_at = self._now_after(constructed_at)
        preparation = self.builder.prepare(
            snapshot=neutral.snapshot,
            evidence=neutral.evidence,
            upper_ontology=self.construction.upper_ontology,
            preconstruction_budgets=self.construction.preconstruction_budgets,
            constructed_at=constructed_at,
            sealed_at=sealed_at,
        )
        preontology = preparation.sealed_preontology
        if preontology is None:
            raise HeldOutC0Error("production C0 did not return a sealed preontology")
        preontology.draft.budget_accounting.validate_against(
            self.construction.preconstruction_budgets
        )
        reference = self._persist_record(
            preparation,
            object_kind="condition_preparation",
            created_at=sealed_at,
        )
        state = HeldOutC0PreparationState(
            unit_id=unit.unit_id,
            prequery_stage_hash=unit.prequery_stage.staging_manifest_hash,
            model_visible_evidence_hash=visible.content_hash,
            neutral_stage_manifest_hash=neutral_manifest.content_hash,
            neutral_evidence_hash=neutral.content_hash,
            equivalence_certificate_hash=certificate.content_hash,
            rule_config_hash=self.builder.config.content_hash,
            backend_manifest_hash=self.backend_manifest.content_hash,
            construction_configuration_hash=self.construction.content_hash,
            preparation=reference,
            construction_seal_hash=preontology.construction_seal.content_hash,
            complete_graph_hash=preontology.construction_seal.ontology_hash,
        )
        self._write_state(state)
        return self._construction_receipt(planned, preparation)

    def _registered_query(
        self,
        unit: HeldOutUnitPlan,
        query_stage_hash: str,
        opening: HeldOutQueryOpening,
    ) -> PublicStageReference:
        matches = tuple(
            stage for stage in unit.query_stages if stage.staging_manifest_hash == query_stage_hash
        )
        if len(matches) != 1:
            raise HeldOutC0Error("C0 projection names an unregistered held-out query stage")
        planned = matches[0]
        opened = opening.opened_stage
        static_fields = (
            "relative_path",
            "manifest_file_sha256",
            "staging_manifest_hash",
            "stage_id",
            "stage_kind",
            "evidence_artifact_hash",
            "snapshot_hash",
            "query_artifact_hash",
        )
        if opening.sealed_stage_hash != query_stage_hash or any(
            getattr(opened, name) != getattr(planned, name) for name in static_fields
        ):
            raise HeldOutC0Error("audited query opening differs from its sealed stage")
        return planned

    def _load_query_context(
        self,
        planned: PublicStageReference,
        opening: HeldOutQueryOpening,
    ) -> tuple[QueryContext, QueryAccessEvent]:
        access = self._resolve_reference(opening.query_access_artifact, QueryAccessEvent)
        if access != opening.query_access_event:
            raise HeldOutC0Error("query-access CAS object differs from the controller opening")
        record = self.artifacts.ledger.get_query_access(access.content_hash)
        if (
            record.access_event_artifact_hash != opening.query_access_artifact.artifact_hash
            or record.query_context_hash != access.query_context_hash
            or record.model_visible_query_hash != access.model_visible_query_hash
            or record.snapshot_hash != access.snapshot_hash
            or record.stage_manifest_hash != access.stage_manifest_hash
            or record.query_artifact_hash != access.query_artifact_hash
            or record.prequery_barrier_hash != access.prequery_barrier_hash
        ):
            raise HeldOutC0Error("query-access ledger record differs from its typed event")
        artifact = self.artifacts.ledger.get_artifact(record.query_payload_artifact_hash)
        if artifact.release_class is not record.release_class:
            raise HeldOutC0Error("query reveal CAS release class changed")
        raw = self.artifacts.blobs.read_bytes(
            artifact,
            allow_restricted=artifact.release_class is StoreReleaseClass.RESTRICTED,
        )
        reveal = QueryRevealArtifact.model_validate_json(raw)
        if (
            reveal.content_hash != access.query_artifact_hash
            or reveal.content_hash != planned.query_artifact_hash
            or reveal.evidence_artifact_hash != planned.evidence_artifact_hash
            or reveal.query.content_hash != access.model_visible_query_hash
        ):
            raise HeldOutC0Error("persisted query reveal differs from its sealed query stage")
        if not reveal.reveal_id.startswith("reveal_"):
            raise HeldOutC0Error("held-out query reveal lacks its opaque identity")
        context = QueryContext(
            context_id=f"ctx_{reveal.reveal_id.removeprefix('reveal_')}",
            revealed_at=reveal.revealed_at,
            **reveal.query.model_dump(
                mode="python",
                exclude={"schema_version", "content_hash"},
            ),
        )
        opened = opening.opened_stage
        if (
            to_model_visible_query(context) != reveal.query
            or context != opening.query_context
            or context.content_hash != opened.query_context_hash
            or context.spoiler_horizon.content_hash != opened.horizon_hash
            or context.budgets.content_hash != opened.budget_hash
        ):
            raise HeldOutC0Error("reconstructed C0 query semantics changed")
        return context, access

    def _load_barrier(
        self,
        unit: HeldOutUnitPlan,
        access: QueryAccessEvent,
        receipt: C0ConstructionReceipt,
    ) -> PrequeryBarrier:
        record = self.artifacts.ledger.get_prequery_barrier(access.prequery_barrier_hash)
        artifact = self.artifacts.ledger.get_artifact(record.barrier_artifact_hash)
        raw = self.artifacts.blobs.read_bytes(
            artifact,
            allow_restricted=artifact.release_class is StoreReleaseClass.RESTRICTED,
        )
        barrier = PrequeryBarrier.model_validate_json(raw)
        bindings = tuple(
            item
            for item in barrier.preparation_bindings
            if item.unit_id == unit.unit_id
            and item.condition is ConditionName.C0_CLASSICAL_PRE
            and item.seed_block is None
        )
        if (
            barrier.content_hash != access.prequery_barrier_hash
            or barrier.content_hash != record.barrier_hash
            or barrier.execution_id != access.execution_id
            or barrier.execution_manifest_hash != self.call_manifest.content_hash
            or barrier.sealed_at >= access.accessed_at
            or unit.prequery_stage.evidence_artifact_hash
            not in barrier.neutral_evidence_artifact_hashes
            or len(bindings) != 1
            or bindings[0].snapshot_hash != unit.prequery_stage.snapshot_hash
            or bindings[0].preparation_hash != receipt.content_hash
            or bindings[0].lineage_artifact_hash != receipt.complete_graph_hash
            or bindings[0].completed_at != receipt.completed_at
        ):
            raise HeldOutC0Error("prequery barrier does not bind the exact sealed C0 receipt")
        return barrier

    def _load_packet(
        self,
        opening: HeldOutQueryOpening,
        neutral: NeutralEvidenceArtifact,
        access: QueryAccessEvent,
    ) -> tuple[EvidencePacket, PacketMaterializationEvent]:
        if (
            opening.packet_materialization is None
            or opening.packet_materialization_artifact is None
        ):
            raise HeldOutC0Error("C0 projection lacks persisted packet materialization lineage")
        packet = self._resolve_reference(opening.evidence_packet_artifact, EvidencePacket)
        materialization = self._resolve_reference(
            opening.packet_materialization_artifact,
            PacketMaterializationEvent,
        )
        ledger_record = self.artifacts.ledger.get_packet_materialization(
            materialization.content_hash
        )
        if (
            packet.content_hash != opening.evidence_packet_hash
            or packet.snapshot_hash != neutral.snapshot.content_hash
            or packet.evidence != neutral.evidence
            or packet.ordered_evidence_ids != neutral.snapshot.eligible_evidence_ids
            or packet.retrieval_method is not RetrievalMethod.ALL_ADMISSIBLE
            or packet.created_at < access.accessed_at
            or materialization != opening.packet_materialization
            or materialization.query_access_event_hash != access.content_hash
            or materialization.snapshot_hash != neutral.snapshot.content_hash
            or materialization.packet_hash != packet.content_hash
            or materialization.retrieval_method is not packet.retrieval_method
            or materialization.started_at < access.accessed_at
            or packet.created_at < materialization.started_at
            or packet.created_at > materialization.completed_at
            or ledger_record.materialization_event_hash != materialization.content_hash
            or ledger_record.materialization_event_id
            != materialization.materialization_event_id
            or ledger_record.execution_id != materialization.execution_id
            or ledger_record.query_access_event_hash != access.content_hash
            or ledger_record.snapshot_hash != materialization.snapshot_hash
            or ledger_record.packet_hash != packet.content_hash
            or ledger_record.retrieval_method != materialization.retrieval_method.value
            or ledger_record.retrieval_config_hash != materialization.retrieval_config_hash
            or ledger_record.packet_artifact_hash
            != opening.evidence_packet_artifact.artifact_hash
            or ledger_record.materialization_event_artifact_hash
            != opening.packet_materialization_artifact.artifact_hash
            or _ledger_timestamp(ledger_record.started_at)
            != _aware_utc(materialization.started_at)
            or _ledger_timestamp(ledger_record.completed_at)
            != _aware_utc(materialization.completed_at)
            or ledger_record.release_class.value
            != opening.packet_materialization_artifact.release_class
        ):
            raise HeldOutC0Error("C0 projection packet differs from exact staged evidence")
        return packet, materialization

    def project_preconstructed(
        self,
        *,
        unit: HeldOutUnitPlan,
        query_stage_hash: str,
        query_opening: HeldOutQueryOpening,
        condition: ConditionName,
        seed_block: int | None,
        construction_seal_hash: str,
        complete_graph_hash: str,
    ) -> PreconstructedProjectionReceipt:
        """Project a registered query from the immutable C0 graph only."""

        if condition is ConditionName.C1_LLM_PRE:
            if self.c1_projection_delegate is None:
                raise HeldOutC0Error("successful C1 projection requires its production delegate")
            return self.c1_projection_delegate.project_preconstructed(
                unit=unit,
                query_stage_hash=query_stage_hash,
                query_opening=query_opening,
                condition=condition,
                seed_block=seed_block,
                construction_seal_hash=construction_seal_hash,
                complete_graph_hash=complete_graph_hash,
            )
        if condition is not ConditionName.C0_CLASSICAL_PRE or seed_block is not None:
            raise HeldOutC0Error("C0 projection received another condition or an LLM seed")

        planned_unit = self._registered_unit(unit)
        planned_query = self._registered_query(
            planned_unit,
            query_stage_hash,
            query_opening,
        )
        _, preparation, neutral = self._load_state(planned_unit)
        preontology = preparation.sealed_preontology
        assert preontology is not None
        construction_receipt = self._construction_receipt(planned_unit, preparation)
        if (
            construction_seal_hash != construction_receipt.construction_seal_hash
            or complete_graph_hash != construction_receipt.complete_graph_hash
        ):
            raise HeldOutC0Error("C0 projection source differs from its sealed complete graph")

        context, access = self._load_query_context(planned_query, query_opening)
        barrier = self._load_barrier(planned_unit, access, construction_receipt)
        packet, materialization = self._load_packet(query_opening, neutral, access)
        processing_started_at = self._now_after(
            max(access.accessed_at, materialization.completed_at)
        )
        run_config = RunConditionConfig(
            config_id=f"held-out-c0-{planned_unit.unit_id}-{planned_query.stage_id}",
            condition=ConditionName.C0_CLASSICAL_PRE,
            budgets=context.budgets,
            maximum_input_tokens=self.construction.first_pass_input_tokens,
            maximum_output_tokens=self.construction.first_pass_output_tokens,
            repair_attempt_budget=context.budgets.repair_attempt_budget,
            scored_schema_hash=SCORED_PROJECTION_SCHEMA_HASH,
            validator_hash=self.validator_hash,
            upper_ontology_hash=self.construction.upper_ontology.content_hash,
        )
        inputs = _HeldOutC0Inputs(
            preparation=preparation,
            snapshot=neutral.snapshot,
            packet=packet,
            context=context,
            query_access=access,
            prequery_barrier=barrier,
            query_processing_started_at=processing_started_at,
            packet_materialization=materialization,
            upper_ontology=self.construction.upper_ontology,
            run_config=run_config,
        )
        projection = project_sealed_c0(
            preontology,
            inputs,  # type: ignore[arg-type]
            rule_config=self.builder.config,
        )
        if projection.construction_seal != preontology.construction_seal or any(
            decision.operator.value != "selection" for decision in projection.decisions
        ):
            raise HeldOutC0Error("fixed C0 projection changed its sealed ontology")
        completed_at = self._now_after(processing_started_at)
        artifact = self.artifacts.put_bytes(
            (projection.to_canonical_json() + "\n").encode("utf-8"),
            media_type="application/vnd.story-projection.ontology-projection+json",
            release_class=StoreReleaseClass.RESTRICTED,
            created_at=completed_at,
        )
        return PreconstructedProjectionReceipt(
            unit_id=planned_unit.unit_id,
            query_stage_hash=query_stage_hash,
            condition=ConditionName.C0_CLASSICAL_PRE,
            seed_block=None,
            source_construction_seal_hash=construction_seal_hash,
            source_complete_graph_hash=complete_graph_hash,
            outcome=RunOutcome.SUCCEEDED,
            projection_artifact_hash=artifact.content_hash,
            evidence_packet_hash=packet.content_hash,
            horizon_hash=context.spoiler_horizon.content_hash,
            budget_hash=context.budgets.content_hash,
            completed_at=completed_at,
        )

    def project_unavailable(
        self,
        *,
        unit: HeldOutUnitPlan,
        query_stage_hash: str,
        query_opening: HeldOutQueryOpening,
        condition: ConditionName,
        seed_block: int | None,
        source_failure_hash: str,
    ) -> PreconstructedProjectionReceipt:
        """Carry a failed preconstruction into the registered ITT projection slot."""

        planned = self._registered_unit(unit)
        self._registered_query(planned, query_stage_hash, query_opening)
        if (condition is ConditionName.C0_CLASSICAL_PRE) != (seed_block is None):
            raise HeldOutC0Error("unavailable projection condition/seed lineage changed")
        opened = query_opening.opened_stage
        assert opened.horizon_hash is not None and opened.budget_hash is not None
        return PreconstructedProjectionReceipt(
            unit_id=unit.unit_id,
            query_stage_hash=query_stage_hash,
            condition=cast(Any, condition),
            seed_block=cast(Any, seed_block),
            source_construction_seal_hash=None,
            source_complete_graph_hash=None,
            outcome=RunOutcome.FAILED,
            evidence_packet_hash=query_opening.evidence_packet_hash,
            horizon_hash=opened.horizon_hash,
            budget_hash=opened.budget_hash,
            failure_artifact_hash=source_failure_hash,
            completed_at=self._now_after(query_opening.opened_at),
        )


def build_production_held_out_c0_adapter(
    *,
    repository: Path,
    call_manifest: HeldOutCallManifest,
    configuration: HeldOutControlConfiguration,
    artifacts: ArtifactStore,
    state_directory: Path,
    c1_projection_delegate: _C1ProjectionDelegate | None = None,
    clock: Callable[[], datetime] = lambda: datetime.now(UTC),
) -> ProductionHeldOutC0Adapter:
    """Create the fail-closed local C0 adapter with the real pinned spaCy loader."""

    return ProductionHeldOutC0Adapter(
        repository=repository,
        call_manifest=call_manifest,
        configuration=configuration,
        artifacts=artifacts,
        state_directory=state_directory,
        c1_projection_delegate=c1_projection_delegate,
        clock=clock,
    )


__all__ = [
    "PINNED_C0_RULES_FILE_SHA256",
    "PINNED_DEVELOPMENT_CONSTRUCTION_FILE_SHA256",
    "HeldOutC0Error",
    "HeldOutC0PreparationState",
    "ProductionHeldOutC0Adapter",
    "build_production_held_out_c0_adapter",
]
