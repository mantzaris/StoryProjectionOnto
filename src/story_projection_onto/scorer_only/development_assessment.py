"""Evidence-backed scorer for the frozen 24-call development gate.

This module is deliberately below ``scorer_only``.  It is imported only by the
controller after every registered development call has a terminal ITT row.  The
model service, request packer, and condition implementations must never import it.

The runtime's :class:`~story_projection_onto.development_runtime.ServiceCallResult`
contains logical hashes, but logical record hashes are not CAS addresses (the CAS
also hashes the serialized ``content_hash`` field).  ``DevelopmentCallAuditReceipt``
therefore binds both identities.  The assessor reparses every object from the CAS,
checks the append-only ledger, and recomputes each scientific gate.  No boolean in
the receipt can declare scientific success.
"""

from __future__ import annotations

import ast
import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path, PurePosixPath
from typing import Any, Literal, TypeVar, cast

from pydantic import BaseModel, Field, model_validator

from story_projection_onto.benchmark_runtime import (
    ModelEligibleWorldArtifact,
    QueryRevealArtifact,
    RuntimeStageKind,
    RuntimeStagingManifest,
)
from story_projection_onto.conditions.base import (
    SCORED_PROJECTION_SCHEMA_HASH,
    ComparisonInputManifest,
    ConditionAttemptRecord,
    ConditionPreparation,
    RunConditionConfig,
)
from story_projection_onto.contracts import (
    CONSTRUCTIVE_OPERATORS,
    FIXED_SELECT_ALLOWED_OPERATORS,
    ConditionName,
    ConstructionOperator,
    ConstructionRequest,
    EvidencePacket,
    ImmutableRecord,
    InstanceGraph,
    LocalContextSchema,
    ModelVisibleEvidencePacket,
    OntologyProjection,
    PacketMaterializationEvent,
    PreconstructionRequest,
    QueryAccessEvent,
    QueryContext,
    RunOutcome,
    Sha256Digest,
    TemporalKind,
    ValidationStatus,
    canonical_sha256,
    to_model_visible_packet,
    to_model_visible_query,
)
from story_projection_onto.development_artifacts import (
    DevelopmentAssessmentBundle,
    DevelopmentCallAuditReceipt,
    DevelopmentCPUProjectionReceipt,
    DevelopmentPackingPreflight,
    DevelopmentRepairProbeInput,
    DevelopmentTreatmentSwitches,
    LogicalCASReference,
    OpaqueJSONReference,
)
from story_projection_onto.development_runtime import (
    DEVELOPMENT_CALL_COUNT,
    DEVELOPMENT_UNIT_IDS,
    CallExecutionEnvelope,
    DevelopmentCallKind,
    DevelopmentCallManifest,
    DevelopmentCallSpec,
    DevelopmentIntegrityError,
    DevelopmentITTRecord,
    DevelopmentPrequeryInputs,
    DevelopmentScientificAssessment,
    FixedSchemaDerivationReceipt,
    LiveServiceIdentity,
    RequestStartState,
    ServiceCallResult,
    StageReference,
)
from story_projection_onto.llm import (
    CapabilityManifest,
    FixedSelectCapabilityError,
    FixedSelectOutputAudit,
    PackingReport,
    SealedOntologyInventory,
    base_condition_output_schema,
    enforce_fixed_select_draft,
    enforce_fixed_select_output,
    render_condition_system_prompt,
    sealed_inventory_from_fixed_ontology,
)
from story_projection_onto.manifest import build_source_manifest
from story_projection_onto.metrics.alignment import (
    AlignmentPlan,
    GroundingStatus,
    PredictedAssertion,
    PredictedNode,
    audit_qualified_assertion_grounding,
    prediction_records_from_components,
    score_alignment,
)
from story_projection_onto.store import (
    ArtifactIntegrityError,
    BlobStore,
    Ledger,
    ModelBackend,
)
from story_projection_onto.store import (
    CommitmentCheckStatus as LedgerCommitmentCheckStatus,
)
from story_projection_onto.store import (
    EvidenceSupportStatus as LedgerEvidenceSupportStatus,
)
from story_projection_onto.store import ReleaseClass as LedgerReleaseClass
from story_projection_onto.store import SemanticAssessmentScope as LedgerSemanticAssessmentScope
from story_projection_onto.store import (
    TemporalValidationStatus as LedgerTemporalValidationStatus,
)
from story_projection_onto.store import ValidationStatus as LedgerValidationStatus
from story_projection_onto.validate import (
    BoundaryValidationReport,
    validate_draft_structure,
    validate_repair_preservation,
)


class DevelopmentAssessmentIntegrityError(DevelopmentIntegrityError):
    """A scorer input, CAS object, ledger row, or source binding was inconsistent."""


class RuntimeSourceBinding(ImmutableRecord):
    relative_path: str = Field(min_length=1)
    sha256: Sha256Digest
    role: Literal["core_runtime", "development_service_adapter"]

    @model_validator(mode="after")
    def safe_non_scorer_source(self) -> RuntimeSourceBinding:
        path = PurePosixPath(self.relative_path)
        if path.is_absolute() or ".." in path.parts:
            raise ValueError("runtime source paths must be bounded relative paths")
        if path.suffix != ".py" or path.parts[:2] != ("src", "story_projection_onto"):
            raise ValueError("runtime source bindings must name package Python files")
        lowered = self.relative_path.casefold()
        if "scorer_only" in lowered or "synthetic_benchmark" in lowered:
            raise ValueError("model/runtime source cannot be scorer-bearing")
        return self


_CORE_RUNTIME_SOURCE_PATHS = frozenset(
    {
        "src/story_projection_onto/development_artifacts.py",
        "src/story_projection_onto/development_continuation.py",
        "src/story_projection_onto/development_execution.py",
        "src/story_projection_onto/development_runtime.py",
        "src/story_projection_onto/benchmark_runtime.py",
        "src/story_projection_onto/contracts.py",
        "src/story_projection_onto/evidence.py",
        "src/story_projection_onto/experiment.py",
        "src/story_projection_onto/gpu_runtime.py",
        "src/story_projection_onto/query_runtime.py",
        "src/story_projection_onto/llm.py",
        "src/story_projection_onto/store.py",
        "src/story_projection_onto/validate.py",
        "src/story_projection_onto/conditions/base.py",
        "src/story_projection_onto/conditions/c0.py",
        "src/story_projection_onto/conditions/c1.py",
        "src/story_projection_onto/conditions/c2.py",
        "src/story_projection_onto/conditions/fixed_select.py",
    }
)


class DevelopmentAssessmentInputManifest(ImmutableRecord):
    """Frozen, gold-free routing required by the post-generation scorer."""

    manifest_id: Literal["development-scientific-assessment-input-v1"] = (
        "development-scientific-assessment-input-v1"
    )
    call_manifest_hash: Sha256Digest
    prequery_inputs_hash: Sha256Digest
    source_tree_hash: Sha256Digest
    source_revision: str = Field(min_length=1)
    benchmark_manifest_file_sha256: Sha256Digest
    assessment_bundle_artifact_hash: Sha256Digest
    runtime_source_manifest: OpaqueJSONReference
    prequery_preparation_artifacts: tuple[LogicalCASReference, ...]
    call_receipt_artifact_hashes: tuple[Sha256Digest, ...]
    service_result_artifact_hashes: tuple[Sha256Digest, ...]
    cpu_projection_receipt_artifact_hashes: tuple[Sha256Digest, ...]
    packing_preflight_artifact_hash: Sha256Digest
    runtime_sources: tuple[RuntimeSourceBinding, ...]

    @model_validator(mode="after")
    def complete_gold_free_routing(self) -> DevelopmentAssessmentInputManifest:
        if len(self.prequery_preparation_artifacts) != 11:
            raise ValueError("assessment input requires the exact 11 preexisting preparations")
        logical = tuple(item.logical_content_hash for item in self.prequery_preparation_artifacts)
        if len(logical) != len(set(logical)):
            raise ValueError("prequery preparation references must be unique")
        if len(self.cpu_projection_receipt_artifact_hashes) != 24:
            raise ValueError("assessment input requires 12 C0 and 12 C1 projection receipts")
        if len(set(self.cpu_projection_receipt_artifact_hashes)) != 24:
            raise ValueError("CPU projection receipt artifacts must be unique")
        if len(self.call_receipt_artifact_hashes) != DEVELOPMENT_CALL_COUNT:
            raise ValueError("assessment input requires all 24 call-receipt CAS artifacts")
        if len(set(self.call_receipt_artifact_hashes)) != DEVELOPMENT_CALL_COUNT:
            raise ValueError("call-receipt artifacts must be unique")
        if len(self.service_result_artifact_hashes) != DEVELOPMENT_CALL_COUNT:
            raise ValueError("assessment input requires all 24 service-result CAS artifacts")
        if len(set(self.service_result_artifact_hashes)) != DEVELOPMENT_CALL_COUNT:
            raise ValueError("service-result artifacts must be unique")
        source_paths = tuple(item.relative_path for item in self.runtime_sources)
        if len(source_paths) != len(set(source_paths)):
            raise ValueError("runtime source paths must be unique")
        core = {
            item.relative_path
            for item in self.runtime_sources
            if item.role == "core_runtime"
        }
        if core != _CORE_RUNTIME_SOURCE_PATHS:
            raise ValueError("assessment input must bind the exact core runtime source set")
        adapter_paths = {
            item.relative_path
            for item in self.runtime_sources
            if item.role == "development_service_adapter"
        }
        if adapter_paths != {"src/story_projection_onto/development_adapter.py"}:
            raise ValueError("assessment input requires the production development adapter")
        return self


@dataclass(frozen=True)
class _ProcessedCall:
    spec: DevelopmentCallSpec
    row: DevelopmentITTRecord
    receipt: DevelopmentCallAuditReceipt | None
    semantic_request: PreconstructionRequest | ConstructionRequest | None
    run_config: RunConditionConfig | None
    packing: PackingReport | None
    capabilities: CapabilityManifest | None
    treatment: DevelopmentTreatmentSwitches | None
    generation: Any | None
    condition_result: ConditionPreparation | ConditionAttemptRecord | None
    comparison: ComparisonInputManifest | None
    packet: EvidencePacket | None
    context: QueryContext | None
    fixed_inventory: SealedOntologyInventory | None
    fixed_probe: FixedSelectOutputAudit | None


@dataclass(frozen=True)
class _ProcessedCPUProjection:
    receipt: DevelopmentCPUProjectionReceipt
    attempt: ConditionAttemptRecord
    projection: OntologyProjection
    run_config: RunConditionConfig
    comparison: ComparisonInputManifest
    packet: EvidencePacket
    context: QueryContext


ModelT = TypeVar("ModelT", bound=BaseModel)


class _CASReader:
    def __init__(self, ledger: Ledger, blobs: BlobStore) -> None:
        self.ledger = ledger
        self.blobs = blobs

    def bytes(self, artifact_hash: str) -> bytes:
        try:
            record = self.ledger.get_artifact(artifact_hash)
            if record.release_class is not LedgerReleaseClass.PUBLIC:
                raise DevelopmentAssessmentIntegrityError(
                    "development assessment cannot consume restricted artifacts"
                )
            return self.blobs.read_bytes(record)
        except DevelopmentAssessmentIntegrityError:
            raise
        except (ArtifactIntegrityError, KeyError, OSError, ValueError) as exc:
            raise DevelopmentAssessmentIntegrityError(
                f"CAS artifact {artifact_hash} is missing or corrupt"
            ) from exc

    def logical(self, reference: LogicalCASReference, model: type[ModelT]) -> ModelT:
        payload = self.bytes(reference.artifact_hash)
        try:
            value = model.model_validate_json(payload)
        except Exception as exc:
            raise DevelopmentAssessmentIntegrityError(
                f"{reference.object_kind} CAS bytes do not validate as {model.__name__}"
            ) from exc
        observed = getattr(value, "content_hash", None)
        if observed != reference.logical_content_hash:
            raise DevelopmentAssessmentIntegrityError(
                f"{reference.object_kind} logical hash differs from its CAS reference"
            )
        return value

    def opaque_json(self, reference: OpaqueJSONReference) -> Mapping[str, Any]:
        payload = self.bytes(reference.artifact_hash)
        try:
            parsed = json.loads(payload)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise DevelopmentAssessmentIntegrityError(
                f"{reference.object_kind} is not valid JSON"
            ) from exc
        if not isinstance(parsed, Mapping):
            raise DevelopmentAssessmentIntegrityError(
                f"{reference.object_kind} must be one JSON object"
            )
        if canonical_sha256(parsed) != reference.logical_content_hash:
            raise DevelopmentAssessmentIntegrityError(
                f"{reference.object_kind} logical JSON hash changed"
            )
        return cast(Mapping[str, Any], parsed)

    def receipt(self, artifact_hash: str, model: type[ModelT]) -> ModelT:
        payload = self.bytes(artifact_hash)
        try:
            return model.model_validate_json(payload)
        except Exception as exc:
            raise DevelopmentAssessmentIntegrityError(
                f"receipt artifact {artifact_hash} has the wrong schema"
            ) from exc


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def resolve_development_world_id(root: Path, model_visible_hash: str, neutral_hash: str) -> str:
    """Resolve anonymized runtime evidence only through the frozen scorer routing.

    Runtime unit IDs deliberately are not gold world IDs. This resolver stays
    scorer-only and may run only after generation, never in model preparation.
    """
    from story_projection_onto.synthetic_benchmark import ModelRoutingEntry

    relative = "scorer_only/routing/model_artifacts.json"
    path = root / "data/synthetic" / relative
    benchmark = json.loads((root / "data/synthetic/manifests/benchmark_manifest.json").read_bytes())
    records = [row for row in benchmark["generated_files"] if row["relative_path"] == relative]
    if len(records) != 1 or path.is_symlink() or _sha256_file(path) != records[0]["sha256"]:
        raise DevelopmentAssessmentIntegrityError("development scorer routing hash changed")
    matches = [
        ModelRoutingEntry.model_validate(row)
        for row in json.loads(path.read_bytes())
        if row.get("split") == "development"
        and row.get("artifact_hash") == model_visible_hash
        and row.get("neutral_evidence_artifact_hash") == neutral_hash
    ]
    if len(matches) != 1:
        raise DevelopmentAssessmentIntegrityError("no unique development-only scorer route")
    return matches[0].world_id


def _read_manifest(path: Path, expected_sha256: str) -> DevelopmentAssessmentInputManifest:
    if path.is_symlink() or not path.is_file():
        raise DevelopmentAssessmentIntegrityError(
            "development assessment manifest must be one regular non-symlink file"
        )
    if _sha256_file(path) != expected_sha256:
        raise DevelopmentAssessmentIntegrityError("development assessment manifest bytes changed")
    try:
        return DevelopmentAssessmentInputManifest.model_validate_json(
            path.read_text(encoding="utf-8")
        )
    except Exception as exc:
        raise DevelopmentAssessmentIntegrityError(
            "development assessment manifest failed schema validation"
        ) from exc


def _verify_assessment_bundle(
    input_manifest: DevelopmentAssessmentInputManifest,
    bundle: DevelopmentAssessmentBundle,
) -> None:
    expected = (
        input_manifest.call_manifest_hash,
        input_manifest.prequery_inputs_hash,
        input_manifest.source_tree_hash,
        input_manifest.benchmark_manifest_file_sha256,
        input_manifest.runtime_source_manifest,
        input_manifest.prequery_preparation_artifacts,
        input_manifest.call_receipt_artifact_hashes,
        input_manifest.service_result_artifact_hashes,
        input_manifest.cpu_projection_receipt_artifact_hashes,
        input_manifest.packing_preflight_artifact_hash,
    )
    observed = (
        bundle.call_manifest_hash,
        bundle.prequery_inputs_hash,
        bundle.source_tree_hash,
        bundle.benchmark_manifest_file_sha256,
        bundle.runtime_source_manifest,
        bundle.prequery_preparation_artifacts,
        bundle.call_receipt_artifact_hashes,
        bundle.service_result_artifact_hashes,
        bundle.cpu_projection_receipt_artifact_hashes,
        bundle.packing_preflight_artifact_hash,
    )
    if observed != expected:
        raise DevelopmentAssessmentIntegrityError(
            "assessment manifest differs from its immutable post-run bundle"
        )


def _verify_runtime_source_manifest(
    reader: _CASReader,
    reference: OpaqueJSONReference,
    *,
    root: Path,
    expected_tree_hash: str,
) -> str:
    source_manifest = reader.opaque_json(reference)
    revision = source_manifest.get("revision")
    if (
        not isinstance(revision, str)
        or not revision
        or source_manifest.get("tree_sha256") != expected_tree_hash
    ):
        raise DevelopmentAssessmentIntegrityError(
            "pre-call runtime source manifest has invalid revision/tree lineage"
        )
    try:
        observed = build_source_manifest(root, revision).to_dict()
    except (OSError, ValueError) as exc:
        raise DevelopmentAssessmentIntegrityError(
            "cannot rebuild the frozen development source tree"
        ) from exc
    if source_manifest != observed:
        raise DevelopmentAssessmentIntegrityError(
            "current source inventory differs from the pre-call runtime source manifest"
        )
    return revision


def _parse_imports(path: Path) -> frozenset[str]:
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=path.as_posix())
    except (OSError, SyntaxError, UnicodeDecodeError) as exc:
        raise DevelopmentAssessmentIntegrityError(
            f"cannot audit runtime imports for {path.as_posix()}"
        ) from exc
    modules: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            modules.add(node.module)
    return frozenset(modules)


def _verify_runtime_sources(root: Path, bindings: Sequence[RuntimeSourceBinding]) -> bool:
    for binding in bindings:
        path = (root / binding.relative_path).resolve()
        try:
            path.relative_to(root)
        except ValueError as exc:
            raise DevelopmentAssessmentIntegrityError(
                "runtime source escapes repository root"
            ) from exc
        if path.is_symlink() or not path.is_file() or _sha256_file(path) != binding.sha256:
            raise DevelopmentAssessmentIntegrityError(
                f"runtime source binding changed: {binding.relative_path}"
            )
        forbidden = sorted(
            module
            for module in _parse_imports(path)
            if "scorer_only" in module.casefold()
            or module.endswith("synthetic_benchmark")
        )
        if forbidden:
            raise DevelopmentAssessmentIntegrityError(
                f"runtime source imports scorer/gold modules: {binding.relative_path}: {forbidden}"
            )
    return True


_FORBIDDEN_MODEL_KEYS = frozenset(
    {
        "scorer_namespace",
        "gold_projection",
        "gold_projections",
        "expected_effect",
        "signed_contrast_decisions",
        "answer_signatures_by_query",
        "semantic_atoms_by_query",
        "pair_id",
        "split",
        "world_id",
        "query_id",
    }
)
_FORBIDDEN_MODEL_TEXT = ("scorer_only", "scorer-only", "syn-test-", "held_out")


def _assert_gold_free_payload(value: Any, *, path: str = "request") -> None:
    if isinstance(value, BaseModel):
        value = value.model_dump(mode="python")
    if isinstance(value, Mapping):
        for key, child in value.items():
            normalized = str(key).casefold()
            child_path = f"{path}.{key}"
            if normalized in _FORBIDDEN_MODEL_KEYS:
                raise DevelopmentAssessmentIntegrityError(
                    f"model-visible request contains scorer key at {child_path}"
                )
            _assert_gold_free_payload(child, path=child_path)
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        for index, child in enumerate(value):
            _assert_gold_free_payload(child, path=f"{path}[{index}]")
    elif isinstance(value, str):
        lowered = value.casefold()
        marker = next((item for item in _FORBIDDEN_MODEL_TEXT if item in lowered), None)
        if marker is not None:
            raise DevelopmentAssessmentIntegrityError(
                f"model-visible request contains scorer marker {marker!r} at {path}"
            )


def _object_reference_hashes(value: Any) -> frozenset[str]:
    """Collect hashes from an audit receipt without treating them as trusted evidence."""

    hashes: set[str] = set()
    if isinstance(value, BaseModel):
        value = value.model_dump(mode="python")
    if isinstance(value, Mapping):
        for key, child in value.items():
            if str(key).endswith("hash") and isinstance(child, str) and len(child) == 64:
                hashes.add(child)
            hashes.update(_object_reference_hashes(child))
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        for child in value:
            hashes.update(_object_reference_hashes(child))
    return frozenset(hashes)


def _cited_evidence_ids(value: Any) -> tuple[str, ...]:
    cited: list[str] = []
    if isinstance(value, BaseModel):
        value = value.model_dump(mode="python")
    if isinstance(value, Mapping):
        for key, child in value.items():
            normalized = str(key)
            if normalized == "evidence_id" and isinstance(child, str):
                cited.append(child)
            elif normalized.endswith("evidence_ids") and isinstance(child, Sequence):
                cited.extend(item for item in child if isinstance(item, str))
            cited.extend(_cited_evidence_ids(child))
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        for child in value:
            cited.extend(_cited_evidence_ids(child))
    return tuple(cited)


def _query_ordinal_by_stage(manifest: DevelopmentCallManifest) -> dict[tuple[str, str], int]:
    result: dict[tuple[str, str], int] = {}
    for unit_id in DEVELOPMENT_UNIT_IDS:
        c2_calls = tuple(
            item
            for item in manifest.calls
            if item.unit_id == unit_id and item.kind is DevelopmentCallKind.C2_CONSTRUCTION
        )
        if len(c2_calls) != 3:
            raise DevelopmentAssessmentIntegrityError(
                "development manifest does not expose three primary C2 queries per unit"
            )
        for ordinal, call in enumerate(c2_calls, start=1):
            assert call.query_stage is not None
            result[(unit_id, call.query_stage.staging_manifest_hash)] = ordinal
    return result


def _query_ordinal(
    call: DevelopmentCallSpec,
    by_stage: Mapping[tuple[str, str], int],
) -> int | None:
    if call.query_stage is None:
        return None
    try:
        return by_stage[(call.unit_id, call.query_stage.staging_manifest_hash)]
    except KeyError as exc:
        raise DevelopmentAssessmentIntegrityError(
            f"call {call.call_id} uses an unregistered query stage"
        ) from exc


def _stage_objects(
    root: Path,
    call: DevelopmentCallSpec,
    *,
    query_revealed: bool,
) -> tuple[ModelEligibleWorldArtifact, QueryRevealArtifact | None]:
    """Reparse a complete frozen stage and check semantic, not file-byte, hashes.

    Stage manifests intentionally bind each immutable record's embedded
    ``content_hash``.  That differs from the SHA-256 of the JSON file containing
    the embedded hash, so treating an artifact hash as a file checksum would make
    every genuine stage fail closed for the wrong reason.
    """

    reference = call.query_stage if query_revealed else call.prequery_stage
    if reference is None:
        raise DevelopmentAssessmentIntegrityError("call has no query reveal stage")
    candidate = root / reference.relative_path
    if candidate.is_symlink():
        raise DevelopmentAssessmentIntegrityError("runtime stage cannot be a symlink")
    directory = candidate.resolve()
    try:
        directory.relative_to(root)
    except ValueError as exc:
        raise DevelopmentAssessmentIntegrityError("runtime stage escapes repository root") from exc
    expected_kind = (
        RuntimeStageKind.QUERY_REVEALED
        if query_revealed
        else RuntimeStageKind.PREQUERY_EVIDENCE
    )
    expected_names = (
        {"manifest.json", "evidence.json", "query.json"}
        if query_revealed
        else {"manifest.json", "evidence.json"}
    )
    if (
        not directory.is_dir()
        or {item.name for item in directory.iterdir()} != expected_names
        or any(item.is_symlink() or not item.is_file() for item in directory.iterdir())
    ):
        raise DevelopmentAssessmentIntegrityError("runtime stage is not the exact flat sandbox")
    manifest_path = directory / "manifest.json"
    if _sha256_file(manifest_path) != reference.manifest_file_sha256:
        raise DevelopmentAssessmentIntegrityError("runtime stage manifest bytes changed")
    try:
        stage_manifest = RuntimeStagingManifest.model_validate_json(
            manifest_path.read_text(encoding="utf-8")
        )
        evidence = ModelEligibleWorldArtifact.model_validate_json(
            (directory / "evidence.json").read_text(encoding="utf-8")
        )
        reveal = (
            QueryRevealArtifact.model_validate_json(
                (directory / "query.json").read_text(encoding="utf-8")
            )
            if query_revealed
            else None
        )
    except Exception as exc:
        raise DevelopmentAssessmentIntegrityError("runtime stage artifact is invalid") from exc
    expected_artifact_hashes = (
        (evidence.content_hash, cast(QueryRevealArtifact, reveal).content_hash)
        if query_revealed
        else (evidence.content_hash,)
    )
    if (
        reference.stage_kind is not expected_kind
        or stage_manifest.stage_kind is not expected_kind
        or stage_manifest.stage_id != reference.stage_id
        or stage_manifest.content_hash != reference.staging_manifest_hash
        or stage_manifest.artifact_hashes != expected_artifact_hashes
        or stage_manifest.artifact_hashes[0] != reference.evidence_artifact_hash
        or (
            query_revealed
            and stage_manifest.artifact_hashes[1] != reference.query_artifact_hash
        )
    ):
        raise DevelopmentAssessmentIntegrityError("runtime stage differs from its frozen binding")
    if reveal is not None and reveal.evidence_artifact_hash != evidence.content_hash:
        raise DevelopmentAssessmentIntegrityError("query reveal belongs to another evidence stage")
    return evidence, reveal


def _ledger_call_matches(
    ledger: Ledger,
    receipt: DevelopmentCallAuditReceipt,
    row: DevelopmentITTRecord,
) -> None:
    assert receipt.model_call_id is not None
    try:
        model_call = ledger.get_model_call(receipt.model_call_id)
        ledger.get_job(cast(str, receipt.job_id))
    except KeyError as exc:
        raise DevelopmentAssessmentIntegrityError(
            "call receipt cites a missing ledger row"
        ) from exc
    expected = (
        receipt.job_id,
        receipt.attempt_id,
        row.gpu_event_id,
        row.response_artifact_hash,
        row.prompt_tokens,
        row.completion_tokens,
    )
    # DevelopmentITTRecord intentionally does not retain request_hash.  The model
    # call is instead bound to the receipt's rendered request below.
    rendered = receipt.rendered_model_request
    assert rendered is not None
    observed = (
        model_call.job_id,
        model_call.attempt_id,
        model_call.gpu_event_id,
        model_call.response_artifact_hash,
        model_call.prompt_tokens,
        model_call.completion_tokens,
    )
    if observed != expected:
        raise DevelopmentAssessmentIntegrityError("service receipt and model-call ledger disagree")
    if model_call.backend is not ModelBackend.VLLM_GPU:
        raise DevelopmentAssessmentIntegrityError("development model call was not GPU-backed")
    if model_call.request_hash != rendered.logical_content_hash:
        raise DevelopmentAssessmentIntegrityError(
            "ledger request hash differs from stored wire JSON"
        )
    if model_call.successful != (row.outcome is RunOutcome.SUCCEEDED):
        raise DevelopmentAssessmentIntegrityError("ledger success flag differs from ITT outcome")
    seconds = model_call.allocated_gpu_microseconds / 1_000_000
    if not math.isclose(seconds, row.allocated_gpu_seconds, abs_tol=1e-6):
        raise DevelopmentAssessmentIntegrityError("ledger and ITT GPU durations differ")


def _reconstruct_service_result(
    receipt: DevelopmentCallAuditReceipt,
    row: DevelopmentITTRecord,
) -> ServiceCallResult:
    """Rebuild the returned result without introducing a cyclic CAS reference."""

    rendered = receipt.rendered_model_request
    assert rendered is not None
    return ServiceCallResult(
        call_id=row.call_id,
        outcome=row.outcome,
        request_started=True,
        request_hash=rendered.logical_content_hash,
        response_artifact_hash=row.response_artifact_hash,
        validated_generation_hash=row.validated_generation_hash,
        validation_record_hash=row.validation_record_hash,
        condition_attempt_hash=row.condition_attempt_hash,
        condition_preparation_hash=row.condition_preparation_hash,
        ledger_receipt_hash=row.ledger_receipt_hash,
        gpu_event_id=row.gpu_event_id,
        service_identity_hash=receipt.service_identity.logical_content_hash,
        run_condition_config_hash=cast(str, row.run_condition_config_hash),
        fixed_schema_derivation=row.fixed_schema_derivation,
        query_access_event_hash=row.query_access_event_hash,
        construction_seal_hash=row.construction_seal_hash,
        prequery_preparation_bindings=row.prequery_preparation_bindings,
        repair_parent_raw_output_hash=row.repair_parent_raw_output_hash,
        allocated_gpu_seconds=row.allocated_gpu_seconds,
        prompt_tokens=row.prompt_tokens,
        completion_tokens=row.completion_tokens,
        failure_code=row.failure_code,
        completed_at=row.completed_at,
    )


def _service_result_matches_itt(
    result: ServiceCallResult,
    row: DevelopmentITTRecord,
) -> bool:
    return (
        result.call_id == row.call_id
        and result.outcome is row.outcome
        and result.request_started
        == (row.request_start_state is not RequestStartState.NOT_STARTED)
        and result.response_artifact_hash == row.response_artifact_hash
        and result.validated_generation_hash == row.validated_generation_hash
        and result.validation_record_hash == row.validation_record_hash
        and result.condition_attempt_hash == row.condition_attempt_hash
        and result.condition_preparation_hash == row.condition_preparation_hash
        and result.ledger_receipt_hash == row.ledger_receipt_hash
        and result.gpu_event_id == row.gpu_event_id
        and result.run_condition_config_hash == row.run_condition_config_hash
        and result.fixed_schema_derivation == row.fixed_schema_derivation
        and result.query_access_event_hash == row.query_access_event_hash
        and result.construction_seal_hash == row.construction_seal_hash
        and result.prequery_preparation_bindings == row.prequery_preparation_bindings
        and result.repair_parent_raw_output_hash == row.repair_parent_raw_output_hash
        and math.isclose(
            result.allocated_gpu_seconds,
            row.allocated_gpu_seconds,
            abs_tol=1e-6,
        )
        and result.prompt_tokens == row.prompt_tokens
        and result.completion_tokens == row.completion_tokens
        and result.failure_code == row.failure_code
        and result.completed_at == row.completed_at
    )


def _validate_ledger_validations(
    ledger: Ledger,
    receipt: DevelopmentCallAuditReceipt,
    *,
    raw_response_hash: str | None,
    validator_hash: str,
    successful: bool,
) -> None:
    for validation_id in receipt.ledger_validation_ids:
        try:
            record = ledger.get_validation(validation_id)
        except KeyError as exc:
            raise DevelopmentAssessmentIntegrityError(
                "call receipt cites a missing validation ledger row"
            ) from exc
        if record.job_id != receipt.job_id or record.attempt_id != receipt.attempt_id:
            raise DevelopmentAssessmentIntegrityError(
                "validation ledger row crosses development job/attempt lineage"
            )
        if record.validator_manifest_hash != validator_hash:
            raise DevelopmentAssessmentIntegrityError(
                "validation ledger row used another validator manifest"
            )
        if raw_response_hash is not None and record.input_artifact_hash != raw_response_hash:
            raise DevelopmentAssessmentIntegrityError(
                "validation ledger input differs from the raw model response"
            )
        expected_validation_status = (
            LedgerValidationStatus.ACCEPTED
            if successful
            else LedgerValidationStatus.REJECTED
        )
        if (
            record.validation_status is not expected_validation_status
            or record.evidence_support_status
            is not LedgerEvidenceSupportStatus.NOT_APPLICABLE
            or record.temporal_status is not LedgerTemporalValidationStatus.NOT_APPLICABLE
            or record.commitment_status
            is not LedgerCommitmentCheckStatus.NOT_APPLICABLE
            or record.semantic_assessment_scope
            is not LedgerSemanticAssessmentScope.RUNTIME_STRUCTURAL_ONLY_NOT_ASSESSED
        ):
            raise DevelopmentAssessmentIntegrityError(
                "result disagrees with its structural-only append-only validation verdict"
            )


def _verified_query_access(
    ledger: Ledger,
    reader: _CASReader,
    *,
    access_event_hash: str,
    context: QueryContext,
    packet: EvidencePacket,
    call: DevelopmentCallSpec,
    reveal: QueryRevealArtifact,
    prequery_barrier_hash: str,
) -> QueryAccessEvent:
    try:
        record = ledger.get_query_access(access_event_hash)
    except KeyError as exc:
        raise DevelopmentAssessmentIntegrityError(
            "query-time output lacks persisted query-access lineage"
        ) from exc
    try:
        access = QueryAccessEvent.model_validate_json(
            reader.bytes(record.access_event_artifact_hash)
        )
        persisted_reveal = QueryRevealArtifact.model_validate_json(
            reader.bytes(record.query_payload_artifact_hash)
        )
    except Exception as exc:
        if isinstance(exc, DevelopmentAssessmentIntegrityError):
            raise
        raise DevelopmentAssessmentIntegrityError(
            "query-access CAS artifacts have the wrong typed schema"
        ) from exc
    query_stage = call.query_stage
    if query_stage is None:
        raise DevelopmentAssessmentIntegrityError("query-access audit received a C1 call")
    expected = (
        access_event_hash,
        context.content_hash,
        to_model_visible_query(context).content_hash,
        packet.snapshot_hash,
        query_stage.staging_manifest_hash,
        query_stage.query_artifact_hash,
        prequery_barrier_hash,
        reveal.revealed_at,
        access.accessed_at,
    )
    observed = (
        access.content_hash,
        record.query_context_hash,
        record.model_visible_query_hash,
        record.snapshot_hash,
        record.stage_manifest_hash,
        record.query_artifact_hash,
        record.prequery_barrier_hash,
        datetime.fromisoformat(record.registered_revealed_at.replace("Z", "+00:00")),
        datetime.fromisoformat(record.accessed_at.replace("Z", "+00:00")),
    )
    if (
        observed != expected
        or access.query_context_hash != context.content_hash
        or access.model_visible_query_hash != to_model_visible_query(context).content_hash
        or access.snapshot_hash != packet.snapshot_hash
        or access.stage_manifest_hash != query_stage.staging_manifest_hash
        or access.query_artifact_hash != query_stage.query_artifact_hash
        or access.prequery_barrier_hash != prequery_barrier_hash
        or access.registered_revealed_at != reveal.revealed_at
        or access.packet_hash is not None
        or record.packet_hash is not None
        or record.release_class is not LedgerReleaseClass.PUBLIC
        or persisted_reveal != reveal
    ):
        raise DevelopmentAssessmentIntegrityError(
            "query-access ledger/CAS record differs from the frozen request"
        )
    return access


def _verified_packet_materialization(
    ledger: Ledger,
    reader: _CASReader,
    *,
    materialization_event_hash: str,
    access: QueryAccessEvent,
    packet: EvidencePacket,
    packet_reference: LogicalCASReference,
) -> PacketMaterializationEvent:
    try:
        record = ledger.get_packet_materialization(materialization_event_hash)
    except KeyError as exc:
        raise DevelopmentAssessmentIntegrityError(
            "packet lacks persisted materialization lineage"
        ) from exc
    try:
        event = PacketMaterializationEvent.model_validate_json(
            reader.bytes(record.materialization_event_artifact_hash)
        )
        persisted_packet = EvidencePacket.model_validate_json(
            reader.bytes(record.packet_artifact_hash)
        )
    except Exception as exc:
        if isinstance(exc, DevelopmentAssessmentIntegrityError):
            raise
        raise DevelopmentAssessmentIntegrityError(
            "packet-materialization CAS artifacts have the wrong typed schema"
        ) from exc
    if (
        event.content_hash != materialization_event_hash
        or event.query_access_event_hash != access.content_hash
        or event.snapshot_hash != packet.snapshot_hash
        or event.packet_hash != packet.content_hash
        or event.retrieval_method is not packet.retrieval_method
        or event.started_at < access.accessed_at
        or not (event.started_at <= packet.created_at <= event.completed_at)
        or record.query_access_event_hash != access.content_hash
        or record.materialization_event_hash != event.content_hash
        or record.execution_id != event.execution_id
        or record.snapshot_hash != packet.snapshot_hash
        or record.packet_hash != packet.content_hash
        or record.retrieval_method != packet.retrieval_method.value
        or record.retrieval_config_hash != event.retrieval_config_hash
        or record.packet_artifact_hash != packet_reference.artifact_hash
        or record.release_class is not LedgerReleaseClass.PUBLIC
        or datetime.fromisoformat(record.started_at.replace("Z", "+00:00"))
        != event.started_at
        or datetime.fromisoformat(record.completed_at.replace("Z", "+00:00"))
        != event.completed_at
        or persisted_packet != packet
    ):
        raise DevelopmentAssessmentIntegrityError(
            "packet materialization ledger/CAS differs from the stored packet"
        )
    return event


def _context_and_packet_match_request(
    request: ConstructionRequest,
    context: QueryContext,
    packet: EvidencePacket,
) -> None:
    if request.context != to_model_visible_query(context):
        raise DevelopmentAssessmentIntegrityError(
            "semantic request differs from the stored full query context"
        )
    if request.packet != to_model_visible_packet(packet):
        raise DevelopmentAssessmentIntegrityError(
            "semantic request differs from the stored full evidence packet"
        )
    if request.snapshot_hash != packet.snapshot_hash:
        raise DevelopmentAssessmentIntegrityError("request packet belongs to another snapshot")


def _request_matches_staged_evidence(
    request: PreconstructionRequest | ConstructionRequest,
    staged: ModelEligibleWorldArtifact,
) -> None:
    if request.snapshot_hash != staged.snapshot.content_hash:
        raise DevelopmentAssessmentIntegrityError(
            "semantic request snapshot differs from its frozen evidence stage"
        )
    if isinstance(request, PreconstructionRequest):
        if request.evidence != staged.evidence:
            raise DevelopmentAssessmentIntegrityError(
                "C1 semantic request differs from the complete staged evidence"
            )
        return
    _model_visible_packet_matches_stage(request.packet, request.snapshot_hash, staged)


def _model_visible_packet_matches_stage(
    packet: ModelVisibleEvidencePacket,
    snapshot_hash: str,
    staged: ModelEligibleWorldArtifact,
) -> None:
    if snapshot_hash != staged.snapshot.content_hash:
        raise DevelopmentAssessmentIntegrityError(
            "evidence packet snapshot differs from its frozen stage"
        )
    staged_by_id = {item.evidence_id: item for item in staged.evidence}
    if len(staged_by_id) != len(staged.evidence):
        raise DevelopmentAssessmentIntegrityError("staged evidence contains duplicate IDs")
    try:
        expected = tuple(staged_by_id[item] for item in packet.ordered_evidence_ids)
    except KeyError as exc:
        raise DevelopmentAssessmentIntegrityError(
            "query packet contains evidence outside its frozen stage"
        ) from exc
    if packet.evidence != expected:
        raise DevelopmentAssessmentIntegrityError(
            "query packet evidence differs from the frozen staged records"
        )


def _packing_contains_all_inputs(
    wire: Mapping[str, Any],
    request: PreconstructionRequest | ConstructionRequest,
    packing: PackingReport,
    repair_input: DevelopmentRepairProbeInput | None = None,
) -> None:
    try:
        from story_projection_onto.development_adapter import (
            encode_development_semantic_request,
        )

        messages = wire["messages"]
        if not isinstance(messages, Sequence) or isinstance(messages, (str, bytes)):
            raise TypeError("messages is not a sequence")
        user_messages = tuple(
            item
            for item in messages
            if isinstance(item, Mapping) and item.get("role") == "user"
        )
        if len(user_messages) != 1 or not isinstance(user_messages[0].get("content"), str):
            raise ValueError("wire request does not contain exactly one user payload")
        actual_sections = json.loads(cast(str, user_messages[0]["content"]))
        expected_sections = dict(encode_development_semantic_request(request).sections)
        if repair_input is not None:
            expected_sections["invalid_draft"] = dict(repair_input.invalid_draft)
            expected_sections["validation_diagnostics"] = [
                item.model_dump(
                    mode="json", exclude={"schema_version", "content_hash"}
                )
                for item in repair_input.diagnostics
            ]
    except Exception as exc:
        raise DevelopmentAssessmentIntegrityError(
            "rendered request cannot be decoded as the registered lossless wire format"
        ) from exc
    if actual_sections != expected_sections:
        raise DevelopmentAssessmentIntegrityError(
            "rendered request is not a lossless encoding of its semantic request"
        )
    if isinstance(request, PreconstructionRequest):
        if packing.complete_evidence_snapshot is not True:
            raise DevelopmentAssessmentIntegrityError("C1 packing is not a complete snapshot")
    else:
        if packing.complete_evidence_packet is not True:
            raise DevelopmentAssessmentIntegrityError("query packing is not a complete packet")
        if request.condition is ConditionName.A_FIXED_SELECT:
            fixed = request.fixed_ontology
            if fixed is None or packing.complete_sealed_ontology is not True:
                raise DevelopmentAssessmentIntegrityError(
                    "FixedSelect did not pack its complete sealed ontology"
                )


def _verify_packing_preflight(
    reader: _CASReader,
    preflight: DevelopmentPackingPreflight,
    processed: Sequence[_ProcessedCall],
    manifest: DevelopmentCallManifest,
) -> None:
    from story_projection_onto.development_adapter import (
        ModelWireAliasManifest,
        development_packing_equivalence_hash,
        encode_development_semantic_request,
    )

    c1_calls = tuple(
        item
        for item in processed
        if item.spec.kind is DevelopmentCallKind.C1_PRECONSTRUCTION
    )
    if (
        preflight.call_manifest_hash != manifest.content_hash
        or len(c1_calls) != len(preflight.receipts)
    ):
        raise DevelopmentAssessmentIntegrityError(
            "C1 packing preflight differs from the frozen development manifest"
        )
    for packed, call in zip(preflight.receipts, c1_calls, strict=True):
        receipt = call.receipt
        config = call.run_config
        if receipt is None or config is None or call.packing is None:
            raise DevelopmentAssessmentIntegrityError(
                "C1 packing preflight lacks a complete executed-call binding"
            )
        preflight_request = reader.logical(
            packed.semantic_request, PreconstructionRequest
        )
        preflight_aliases = reader.logical(
            packed.alias_manifest, ModelWireAliasManifest
        )
        preflight_wire = reader.opaque_json(packed.rendered_model_request)
        preflight_packing = reader.logical(packed.packing_report, PackingReport)
        live_request = call.semantic_request
        _packing_contains_all_inputs(
            preflight_wire,
            preflight_request,
            preflight_packing,
        )
        if (
            packed.call_id != call.spec.call_id
            or packed.condition is not ConditionName.C1_LLM_PRE
            or not isinstance(live_request, PreconstructionRequest)
            or packed.packing_equivalence_hash
            != development_packing_equivalence_hash(preflight_request)
            or packed.packing_equivalence_hash
            != development_packing_equivalence_hash(live_request)
            or preflight_aliases
            != encode_development_semantic_request(preflight_request).alias_manifest
            or preflight_packing.input_token_count
            != packed.rendered_input_token_count
            or preflight_wire.get("guided_json") is None
            or call.packing.input_token_count != call.row.prompt_tokens
            or packed.maximum_input_tokens != config.maximum_input_tokens
            or packed.maximum_model_tokens
            != config.maximum_input_tokens + config.maximum_output_tokens
            or packed.semantic_evidence_truncation is not False
        ):
            raise DevelopmentAssessmentIntegrityError(
                "C1 packing preflight does not bind the exact executed request"
            )


def _invariant_run_config_values(config: RunConditionConfig) -> tuple[Any, ...]:
    return (
        config.budgets,
        config.maximum_input_tokens,
        config.maximum_output_tokens,
        config.repair_attempt_budget,
        config.seed_block,
        config.model_stack_hash,
        config.decoding_family_hash,
        config.seed_manifest_hash,
        config.resolved_seed,
        config.scored_schema_hash,
        config.validator_hash,
        config.upper_ontology_hash,
    )


def _comparison_matches_run_config(
    comparison: ComparisonInputManifest,
    config: RunConditionConfig,
) -> bool:
    return (
        comparison.condition is config.condition
        and comparison.budgets == config.budgets
        and comparison.maximum_input_tokens == config.maximum_input_tokens
        and comparison.maximum_output_tokens == config.maximum_output_tokens
        and comparison.repair_attempt_budget == config.repair_attempt_budget
        and comparison.seed_block == config.seed_block
        and comparison.source_c1_seed_block == config.source_c1_seed_block
        and comparison.model_stack_hash == config.model_stack_hash
        and comparison.decoding_manifest_hash == config.decoding_manifest_hash
        and comparison.decoding_family_hash == config.decoding_family_hash
        and comparison.seed_manifest_hash == config.seed_manifest_hash
        and comparison.resolved_seed == config.resolved_seed
        and comparison.prompt_hash == config.prompt_hash
        and comparison.output_schema_hash == config.output_schema_hash
        and comparison.scored_schema_hash == config.scored_schema_hash
        and comparison.capability_manifest_hash == config.capability_manifest_hash
        and comparison.validator_hash == config.validator_hash
    )


def _expected_treatment_switches(
    condition: ConditionName,
) -> DevelopmentTreatmentSwitches:
    if condition is ConditionName.A_NO_CONTEXT:
        return DevelopmentTreatmentSwitches(context_payload_mode="generic")
    if condition is ConditionName.A_NO_TEMPORAL_EPISTEMIC:
        return DevelopmentTreatmentSwitches(temporal_epistemic_fields="disabled")
    if condition is ConditionName.A_NO_RARE_GUARD:
        return DevelopmentTreatmentSwitches(rare_guard="disabled")
    return DevelopmentTreatmentSwitches()


def _context_semantics_hash(context: QueryContext) -> str:
    return canonical_sha256(
        context.model_dump(
            mode="python", exclude={"content_hash", "context_id", "revealed_at"}
        )
    )


def _verify_sealed_projection(
    projection: OntologyProjection,
    preparation: ConditionPreparation,
) -> None:
    sealed = preparation.sealed_preontology
    if sealed is None:
        raise DevelopmentAssessmentIntegrityError("fixed CPU projection lacks a preontology")
    if (
        projection.construction_seal != sealed.construction_seal
        or projection.upper_ontology != sealed.upper_ontology
        or projection.local_schema != sealed.draft.local_schema
    ):
        raise DevelopmentAssessmentIntegrityError(
            "CPU projection changed its sealed schema or construction seal"
        )
    source_records: dict[str, Any] = {
        **{item.entity_id: item for item in sealed.draft.instance_graph.entities},
        **{item.event_id: item for item in sealed.draft.instance_graph.events},
        **{
            item.proposition_content_id: item
            for item in sealed.draft.instance_graph.proposition_contents
        },
        **{item.assertion_id: item for item in sealed.draft.instance_graph.assertions},
    }
    selected_records = (
        *projection.instance_graph.entities,
        *projection.instance_graph.events,
        *projection.instance_graph.proposition_contents,
        *projection.instance_graph.assertions,
    )
    for record in selected_records:
        semantic_id = next(
            getattr(record, field_name)
            for field_name in (
                "entity_id",
                "event_id",
                "proposition_content_id",
                "assertion_id",
            )
            if hasattr(record, field_name)
        )
        if source_records.get(semantic_id) != record:
            raise DevelopmentAssessmentIntegrityError(
                "CPU projection created or altered a post-query semantic object"
            )
    if any(
        item.operator not in FIXED_SELECT_ALLOWED_OPERATORS
        for item in projection.decisions
    ):
        raise DevelopmentAssessmentIntegrityError(
            "CPU fixed projection contains a constructive query-time operation"
        )
    if projection.condition is ConditionName.C1_LLM_PRE:
        generation = sealed.validated_generation
        assert generation is not None
        expected = (
            generation.content_hash,
            generation.raw_output_artifact_hash,
            generation.normalized_draft_hash,
        )
        observed = (
            projection.generation_lineage_hash,
            projection.raw_output_artifact_hash,
            projection.normalized_draft_hash,
        )
        if observed != expected:
            raise DevelopmentAssessmentIntegrityError(
                "C1 CPU projection lost its sealed generation lineage"
            )


def _graph_horizon_leaks(instance_graph: InstanceGraph, horizon: Any) -> int:
    maximum_discourse = horizon.max_discourse_position.ordering_key
    revelation = horizon.max_revelation_position
    maximum_revelation = None if revelation is None else revelation.revelation_order
    leaks = 0
    for assertion in instance_graph.assertions:
        if assertion.temporal_scope.discourse_position.ordering_key > maximum_discourse:
            leaks += 1
        if (
            maximum_revelation is not None
            and assertion.temporal_scope.revelation_position.revelation_order
            > maximum_revelation
        ):
            leaks += 1
    return leaks


def _packet_horizon_leaks(packet: EvidencePacket, context: QueryContext) -> int:
    maximum_discourse = context.spoiler_horizon.max_discourse_position.ordering_key
    return sum(
        item.discourse_position.ordering_key > maximum_discourse
        for item in packet.evidence
    )


def _logical_reference(
    references: Sequence[LogicalCASReference], logical_hash: str
) -> LogicalCASReference:
    matches = tuple(item for item in references if item.logical_content_hash == logical_hash)
    if len(matches) != 1:
        raise DevelopmentAssessmentIntegrityError(
            "assessment manifest does not uniquely resolve a preparation logical hash"
        )
    return matches[0]


def _direct_assertion_ids(scorer: Any, gold: Any) -> frozenset[str]:
    fact_evidence = {
        evidence_id
        for evidence_ids in scorer.fact_evidence_ids.values()
        for evidence_id in evidence_ids
    }
    return frozenset(
        assertion.assertion_id
        for assertion in gold.qualified_assertions
        if set(assertion.evidence_ids).intersection(fact_evidence)
    )


def _context_independent_assertion_slot(assertion_id: str) -> str:
    marker = ".assertion."
    if marker not in assertion_id:
        raise DevelopmentAssessmentIntegrityError(
            "development gold assertion lacks its stable semantic slot"
        )
    return assertion_id.split(marker, maxsplit=1)[1]


def _filtered_alignment_plan(plan: AlignmentPlan, assertion_ids: frozenset[str]) -> AlignmentPlan:
    return AlignmentPlan(
        matcher_revision=plan.matcher_revision,
        source_gold_hash=plan.source_gold_hash,
        source_alternative_set_hash=plan.source_alternative_set_hash,
        executed_alternative_ids=plan.executed_alternative_ids,
        node_targets=plan.node_targets,
        assertion_targets=tuple(
            item for item in plan.assertion_targets if item.target_id in assertion_ids
        ),
    )


def _prediction_bundle(
    *,
    local_schema: LocalContextSchema,
    instance_graph: InstanceGraph,
    plan: AlignmentPlan,
    valid_evidence_ids: frozenset[str],
) -> tuple[
    tuple[PredictedNode, ...],
    tuple[PredictedAssertion, ...],
    frozenset[str],
]:
    provisional_status = {
        item.assertion_id: GroundingStatus.SUPPORTED for item in instance_graph.assertions
    }
    nodes, assertions, _ = prediction_records_from_components(
        local_schema=local_schema,
        instance_graph=instance_graph,
        predicate_aliases=None,
        valid_evidence_ids=valid_evidence_ids,
        grounding_by_assertion_id=provisional_status,
        plan=plan,
    )
    audited_statuses = audit_qualified_assertion_grounding(
        plan=plan,
        predicted_assertions=assertions,
    )
    statuses = dict(audited_statuses)
    supported_ids = frozenset(
        assertion_id
        for assertion_id, status in audited_statuses
        if status is GroundingStatus.SUPPORTED
    )
    nodes, assertions, _ = prediction_records_from_components(
        local_schema=local_schema,
        instance_graph=instance_graph,
        predicate_aliases=None,
        valid_evidence_ids=valid_evidence_ids,
        grounding_by_assertion_id=statuses,
        plan=plan,
    )
    return nodes, assertions, supported_ids


class DevelopmentScientificAssessmentProvider:
    """Callable injected into ``DevelopmentRunner`` after all terminal rows exist."""

    def __init__(
        self,
        *,
        root: Path,
        ledger: Ledger,
        blobs: BlobStore,
        prequery_inputs: DevelopmentPrequeryInputs,
        assessment_manifest_path: Path,
        assessment_manifest_file_sha256: Sha256Digest,
    ) -> None:
        # Do not read the assessment manifest or scorer gold here.  Construction of
        # the closure is safe before the query barrier; all reads occur in __call__.
        self.root = Path(root).resolve()
        self.ledger = ledger
        self.blobs = blobs
        self.prequery_inputs = prequery_inputs
        self.assessment_manifest_path = Path(assessment_manifest_path)
        self.assessment_manifest_file_sha256 = assessment_manifest_file_sha256

    def __call__(
        self,
        manifest: DevelopmentCallManifest,
        rows: tuple[DevelopmentITTRecord, ...],
    ) -> DevelopmentScientificAssessment:
        expected_ids = tuple(item.call_id for item in manifest.calls)
        if (
            len(rows) != DEVELOPMENT_CALL_COUNT
            or tuple(item.call_id for item in rows) != expected_ids
        ):
            raise DevelopmentAssessmentIntegrityError(
                "scientific assessment may run only after the exact 24 terminal ITT rows"
            )

        input_manifest = _read_manifest(
            self.assessment_manifest_path,
            self.assessment_manifest_file_sha256,
        )
        if (
            input_manifest.call_manifest_hash != manifest.content_hash
            or input_manifest.prequery_inputs_hash != self.prequery_inputs.content_hash
            or input_manifest.source_tree_hash != self.prequery_inputs.source_tree_hash
            or input_manifest.benchmark_manifest_file_sha256
            != manifest.benchmark_manifest_file_sha256
        ):
            raise DevelopmentAssessmentIntegrityError(
                "assessment input manifest differs from the frozen execution"
            )
        reader = _CASReader(self.ledger, self.blobs)
        bundle = reader.receipt(
            input_manifest.assessment_bundle_artifact_hash,
            DevelopmentAssessmentBundle,
        )
        _verify_assessment_bundle(input_manifest, bundle)
        source_revision = _verify_runtime_source_manifest(
            reader,
            input_manifest.runtime_source_manifest,
            root=self.root,
            expected_tree_hash=input_manifest.source_tree_hash,
        )
        if source_revision != input_manifest.source_revision:
            raise DevelopmentAssessmentIntegrityError(
                "assessment source revision differs from its pre-call source manifest"
            )
        gold_firewall_verified = _verify_runtime_sources(
            self.root, input_manifest.runtime_sources
        )
        if tuple(item.ledger_receipt_hash for item in rows) != (
            input_manifest.call_receipt_artifact_hashes
        ):
            raise DevelopmentAssessmentIntegrityError(
                "ITT rows differ from the ordered assessment call-receipt index"
            )
        packing_preflight = reader.receipt(
            input_manifest.packing_preflight_artifact_hash,
            DevelopmentPackingPreflight,
        )
        service_results = self._load_service_results(
            reader, input_manifest, manifest, rows
        )
        preparations = self._load_prequery_preparations(reader, input_manifest)
        processed = self._load_gpu_calls(
            reader, manifest, rows, preparations, service_results
        )
        _verify_packing_preflight(reader, packing_preflight, processed, manifest)
        preparations = dict(preparations)
        for call in processed:
            if (
                call.spec.kind is DevelopmentCallKind.C1_PRECONSTRUCTION
                and isinstance(call.condition_result, ConditionPreparation)
            ):
                preparations[(call.spec.unit_id, ConditionName.C1_LLM_PRE)] = (
                    call.condition_result
                )
        cpu = self._load_cpu_projections(reader, input_manifest, manifest, preparations)

        # The first scorer-gold read in the entire provider occurs here, after all
        # 24 rows and every runtime/CAS/ledger input have been verified.
        scorer_by_unit, scorer_file_hashes = self._load_gold(manifest)
        runtime_hashes = {
            hash_value
            for call in processed
            if call.receipt is not None
            for hash_value in _object_reference_hashes(call.receipt)
        }
        if runtime_hashes.intersection(scorer_file_hashes):
            raise DevelopmentAssessmentIntegrityError(
                "a model/runtime receipt references a scorer-gold file hash"
            )
        for call in processed:
            if call.receipt is not None:
                wire = reader.opaque_json(
                    cast(OpaqueJSONReference, call.receipt.rendered_model_request)
                )
                _assert_gold_free_payload(wire)
                _assert_gold_free_payload(cast(BaseModel, call.semantic_request))

        c0 = self._assess_c0(cpu, scorer_by_unit)
        c1 = self._assess_c1(processed, scorer_by_unit)
        c2_present = self._assess_c2_construction(processed, preparations)
        horizon_leaks = self._count_horizon_leaks(processed, cpu)
        packets_equal = self._assess_evidence_equality(processed, cpu, manifest)
        query_blind = self._assess_c1_query_blindness(processed)
        empty_inventories = self._assess_empty_inventories(processed, preparations)
        fixed_packing, fixed_rejection = self._assess_fixed(processed, preparations)
        one_switch = self._assess_one_switch(processed, manifest)
        return DevelopmentScientificAssessment(
            c0_explicit_family_coverage=c0[0],
            c0_direct_assertion_precision=c0[1],
            c0_direct_assertion_recall=c0[2],
            c0_valid_evidence_reference_rate=c0[3],
            c1_schema_valid=c1[0],
            c1_all_construction_operators_exercised=c1[1],
            c1_valid_evidence_id_rate=c1[2],
            c1_grounding_precision=c1[3],
            c1_union_gold_recall_after_seal=c1[4],
            c2_construction_operator_present=c2_present,
            programmed_horizon_leak_count=horizon_leaks,
            evidence_packets_equal=packets_equal,
            c1_query_blindness_verified=query_blind,
            c2_empty_prequery_inventories_verified=empty_inventories,
            fixed_complete_graph_packing_verified=fixed_packing,
            fixed_constructive_operations_rejected=fixed_rejection,
            ablation_one_switch_verified=one_switch,
            gold_firewall_verified=gold_firewall_verified,
        )

    @staticmethod
    def _load_service_results(
        reader: _CASReader,
        input_manifest: DevelopmentAssessmentInputManifest,
        manifest: DevelopmentCallManifest,
        rows: tuple[DevelopmentITTRecord, ...],
    ) -> dict[str, ServiceCallResult]:
        results: dict[str, ServiceCallResult] = {}
        for call, row, artifact_hash in zip(
            manifest.calls,
            rows,
            input_manifest.service_result_artifact_hashes,
            strict=True,
        ):
            result = reader.receipt(artifact_hash, ServiceCallResult)
            if (
                result.content_hash != row.service_result_hash
                or result.call_id != call.call_id
                or not _service_result_matches_itt(result, row)
            ):
                raise DevelopmentAssessmentIntegrityError(
                    "persisted service result differs from its exact ITT row"
                )
            results[call.call_id] = result
        if len(results) != DEVELOPMENT_CALL_COUNT:
            raise DevelopmentAssessmentIntegrityError(
                "service-result CAS index does not cover all development calls"
            )
        return results

    def _load_prequery_preparations(
        self,
        reader: _CASReader,
        input_manifest: DevelopmentAssessmentInputManifest,
    ) -> dict[tuple[str, ConditionName], ConditionPreparation]:
        result: dict[tuple[str, ConditionName], ConditionPreparation] = {}
        for unit in self.prequery_inputs.unit_bindings:
            for binding in unit.preexisting_preparation_bindings:
                reference = _logical_reference(
                    input_manifest.prequery_preparation_artifacts,
                    binding.preparation_hash,
                )
                preparation = reader.logical(reference, ConditionPreparation)
                if (
                    preparation.content_hash != binding.preparation_hash
                    or preparation.condition is not binding.condition
                    or preparation.snapshot_hash != unit.snapshot_hash
                    or preparation.completed_at != binding.completed_at
                ):
                    raise DevelopmentAssessmentIntegrityError(
                        "prequery preparation CAS object differs from its barrier binding"
                    )
                lineage = (
                    preparation.sealed_preontology.construction_seal
                    if preparation.sealed_preontology is not None
                    else preparation.empty_inventory
                )
                if lineage is None or lineage.content_hash != binding.lineage_artifact_hash:
                    raise DevelopmentAssessmentIntegrityError(
                        "prequery preparation lineage differs from its barrier binding"
                    )
                result[(unit.unit_id, binding.condition)] = preparation
        if len(result) != 11:
            raise DevelopmentAssessmentIntegrityError(
                "prequery preparation map does not contain the exact 11 condition/unit pairs"
            )
        return result

    def _load_gpu_calls(
        self,
        reader: _CASReader,
        manifest: DevelopmentCallManifest,
        rows: tuple[DevelopmentITTRecord, ...],
        preparations: Mapping[tuple[str, ConditionName], ConditionPreparation],
        service_results: Mapping[str, ServiceCallResult],
    ) -> tuple[_ProcessedCall, ...]:
        from story_projection_onto.contracts import ValidatedGeneration
        from story_projection_onto.development_adapter import (
            ModelWireAliasManifest,
            build_development_repair_probe_input,
            development_output_schema_for_request,
            encode_development_semantic_request,
            model_wire_source_alias_bijection,
            restore_model_output_source_aliases,
        )

        processed: list[_ProcessedCall] = []
        rows_by_call_id = {item.call_id: item for item in rows}
        service_identity_hashes: set[str] = set()
        execution_ids: set[str] = set()
        execution_manifest_hashes: set[str] = set()
        preconstruction_barrier_hashes: set[str] = set()
        for spec, row in zip(manifest.calls, rows, strict=True):
            service_result = service_results[spec.call_id]
            if (
                row.ordinal != spec.ordinal
                or row.call_id != spec.call_id
                or row.call_class != spec.call_class
                or row.condition is not spec.condition
                or row.unit_id != spec.unit_id
            ):
                raise DevelopmentAssessmentIntegrityError("ITT row differs from call manifest")
            if row.request_start_state is RequestStartState.NOT_STARTED:
                if row.ledger_receipt_hash is not None:
                    raise DevelopmentAssessmentIntegrityError(
                        "unstarted ITT row unexpectedly has a ledger receipt"
                    )
                processed.append(
                    _ProcessedCall(
                        spec, row, None, None, None, None, None, None, None,
                        None, None, None, None, None, None,
                    )
                )
                continue
            if row.ledger_receipt_hash is None or row.service_result_hash is None:
                raise DevelopmentAssessmentIntegrityError(
                    "started ITT row lacks its service/CAS receipt hashes"
                )
            receipt = reader.receipt(
                row.ledger_receipt_hash, DevelopmentCallAuditReceipt
            )
            if (
                receipt.ordinal != spec.ordinal
                or receipt.call_id != spec.call_id
                or receipt.condition is not spec.condition
                or receipt.outcome is not row.outcome
                or receipt.request_started is not True
            ):
                raise DevelopmentAssessmentIntegrityError(
                    "development call receipt differs from its ITT row"
                )
            identity = reader.logical(receipt.service_identity, LiveServiceIdentity)
            envelope = reader.logical(
                cast(LogicalCASReference, receipt.call_execution_envelope),
                CallExecutionEnvelope,
            )
            source_row = rows_by_call_id.get(spec.source_c1_call_id or "")
            parent_row = rows_by_call_id.get(spec.parent_call_id or "")
            expected_source = (
                None if source_row is None else source_row.construction_seal_hash,
                None if source_row is None else source_row.response_artifact_hash,
            )
            expected_parent = (
                None if parent_row is None else parent_row.content_hash,
                None if parent_row is None else parent_row.response_artifact_hash,
            )
            fixed_plan = (
                self.prequery_inputs.fixed_schema_plan_for(spec.ordinal)
                if spec.kind is DevelopmentCallKind.FIXED_SELECTION
                else None
            )
            expected_exact_config = (
                None
                if fixed_plan is not None
                else self.prequery_inputs.run_config_hash_for(spec.ordinal)
            )
            if (
                identity.content_hash != receipt.service_identity.logical_content_hash
                or identity.source_execution_hash != self.prequery_inputs.source_tree_hash
                or identity.selected_model_freeze_hash
                != self.prequery_inputs.selected_model_freeze_hash
                or envelope.call_manifest_hash != manifest.content_hash
                or envelope.prequery_inputs_hash != self.prequery_inputs.content_hash
                or envelope.expected_run_condition_config_hash
                != expected_exact_config
                or envelope.fixed_schema_derivation_plan_hash
                != (None if fixed_plan is None else fixed_plan.content_hash)
                or envelope.unit_prequery_binding_hash
                != self.prequery_inputs.binding_for(spec.unit_id).content_hash
                or envelope.service_identity_hash != identity.content_hash
                or (
                    envelope.source_c1_seal_hash,
                    envelope.source_c1_output_artifact_hash,
                )
                != expected_source
                or (
                    envelope.parent_itt_record_hash,
                    envelope.parent_output_artifact_hash,
                )
                != expected_parent
                or (envelope.prequery_barrier_hash is None)
                != (spec.query_stage is None)
                or (envelope.query_access_event is None)
                != (spec.query_stage is None)
            ):
                raise DevelopmentAssessmentIntegrityError(
                    "call receipt identity/envelope differs from frozen execution inputs"
                )
            service_identity_hashes.add(identity.content_hash)
            execution_ids.add(envelope.execution_id)
            execution_manifest_hashes.add(envelope.execution_manifest_hash)
            preconstruction_barrier_hashes.add(envelope.preconstruction_barrier_hash)
            if _reconstruct_service_result(receipt, row).content_hash != row.service_result_hash:
                raise DevelopmentAssessmentIntegrityError(
                    "ITT row differs from its reconstructed immutable service result"
                )
            if _reconstruct_service_result(receipt, row) != service_result:
                raise DevelopmentAssessmentIntegrityError(
                    "receipt does not reconstruct the persisted service result"
                )
            if receipt.created_at > row.completed_at:
                raise DevelopmentAssessmentIntegrityError(
                    "development receipt was recorded after its returned service result"
                )
            _ledger_call_matches(self.ledger, receipt, row)
            rendered = reader.opaque_json(cast(OpaqueJSONReference, receipt.rendered_model_request))
            run_config = reader.logical(
                cast(LogicalCASReference, receipt.run_condition_config), RunConditionConfig
            )
            packing = reader.logical(
                cast(LogicalCASReference, receipt.packing_report), PackingReport
            )
            capabilities = reader.logical(
                cast(LogicalCASReference, receipt.capability_manifest), CapabilityManifest
            )
            treatment = reader.logical(
                cast(LogicalCASReference, receipt.treatment_switches),
                DevelopmentTreatmentSwitches,
            )
            fixed_derivation = None
            if fixed_plan is not None:
                fixed_reference = receipt.fixed_schema_derivation
                if fixed_reference is None:
                    raise DevelopmentAssessmentIntegrityError(
                        "FixedSelect receipt lacks its schema-derivation CAS reference"
                    )
                fixed_derivation = reader.logical(
                    fixed_reference,
                    FixedSchemaDerivationReceipt,
                )
                if (
                    service_result.fixed_schema_derivation != fixed_derivation
                    or row.fixed_schema_derivation != fixed_derivation
                    or fixed_derivation.plan_hash != fixed_plan.content_hash
                    or fixed_derivation.call_id != spec.call_id
                    or fixed_derivation.unit_id != spec.unit_id
                    or fixed_derivation.derived_at
                    >= cast(QueryAccessEvent, envelope.query_access_event).accessed_at
                ):
                    raise DevelopmentAssessmentIntegrityError(
                        "FixedSelect schema derivation differs from plan/result/timing"
                    )
                expected_config_hash = (
                    fixed_derivation.exact_run_condition_config_hash
                )
                if (
                    fixed_derivation.exact_run_condition_config_artifact_hash
                    != cast(LogicalCASReference, receipt.run_condition_config).artifact_hash
                ):
                    raise DevelopmentAssessmentIntegrityError(
                        "FixedSelect derivation does not identify its exact config CAS"
                    )
            else:
                if receipt.fixed_schema_derivation is not None:
                    raise DevelopmentAssessmentIntegrityError(
                        "non-Fixed call contains a schema derivation"
                    )
                expected_config_hash = self.prequery_inputs.run_config_hash_for(
                    spec.ordinal
                )
            if (
                run_config.content_hash != expected_config_hash
                or row.run_condition_config_hash != expected_config_hash
                or run_config.condition is not spec.condition
                or packing.condition is not spec.condition
                or capabilities.condition is not spec.condition
                or capabilities != CapabilityManifest.for_condition(spec.condition)
                or run_config.capability_manifest_hash != capabilities.content_hash
                or treatment != _expected_treatment_switches(spec.condition)
                or packing.maximum_input_tokens != run_config.maximum_input_tokens
                or packing.reserved_output_tokens != run_config.maximum_output_tokens
            ):
                raise DevelopmentAssessmentIntegrityError(
                    "call runtime/configuration bindings differ from the frozen manifest"
                )
            request_model = (
                PreconstructionRequest
                if spec.kind is DevelopmentCallKind.C1_PRECONSTRUCTION
                else ConstructionRequest
            )
            semantic_request = reader.logical(
                cast(LogicalCASReference, receipt.semantic_request), request_model
            )
            if semantic_request.condition is not spec.condition:
                raise DevelopmentAssessmentIntegrityError("semantic request condition changed")
            alias_manifest = reader.logical(
                cast(LogicalCASReference, receipt.wire_alias_manifest),
                ModelWireAliasManifest,
            )
            if alias_manifest != encode_development_semantic_request(
                semantic_request
            ).alias_manifest:
                raise DevelopmentAssessmentIntegrityError(
                    "stored wire aliases differ from the semantic request"
                )
            if fixed_plan is not None:
                assert isinstance(semantic_request, ConstructionRequest)
                assert fixed_derivation is not None
                fixed_ontology = semantic_request.fixed_ontology
                if fixed_ontology is None or spec.source_c1_call_id is None:
                    raise DevelopmentAssessmentIntegrityError(
                        "FixedSelect semantic request lacks its sealed C1 ontology"
                    )
                source_processed = next(
                    (
                        item
                        for item in processed
                        if item.spec.call_id == spec.source_c1_call_id
                    ),
                    None,
                )
                source_preparation = (
                    None
                    if source_processed is None
                    else source_processed.condition_result
                )
                source_sealed = (
                    source_preparation.sealed_preontology
                    if isinstance(source_preparation, ConditionPreparation)
                    else None
                )
                evidence_aliases = model_wire_source_alias_bijection(
                    semantic_request.packet.evidence
                )
                exact_schema_hash = canonical_sha256(
                    development_output_schema_for_request(semantic_request)
                )
                if (
                    source_sealed is None
                    or fixed_ontology != source_sealed.as_fixed_ontology()
                    or fixed_plan.ordinal != spec.ordinal
                    or fixed_plan.call_id != spec.call_id
                    or fixed_plan.unit_id != spec.unit_id
                    or fixed_plan.source_c1_call_id != spec.source_c1_call_id
                    or fixed_plan.seed_block != spec.seed_block
                    or fixed_plan.resolved_seed != spec.vllm_seed
                    or fixed_plan.budgets != run_config.budgets
                    or fixed_plan.maximum_input_tokens
                    != run_config.maximum_input_tokens
                    or fixed_plan.maximum_output_tokens
                    != run_config.maximum_output_tokens
                    or fixed_plan.repair_attempt_budget
                    != run_config.repair_attempt_budget
                    or fixed_plan.model_stack_hash != run_config.model_stack_hash
                    or fixed_plan.decoding_family_hash
                    != run_config.decoding_family_hash
                    or fixed_plan.seed_manifest_hash != run_config.seed_manifest_hash
                    or fixed_plan.prompt_hash != run_config.prompt_hash
                    or fixed_plan.base_output_schema_hash
                    != canonical_sha256(
                        base_condition_output_schema(ConditionName.A_FIXED_SELECT)
                    )
                    or fixed_plan.scored_schema_hash
                    != SCORED_PROJECTION_SCHEMA_HASH
                    or fixed_plan.capability_manifest_hash
                    != run_config.capability_manifest_hash
                    or fixed_plan.validator_hash != run_config.validator_hash
                    or fixed_plan.upper_ontology_hash
                    != run_config.upper_ontology_hash
                    or fixed_plan.prequery_evidence_artifact_hash
                    != spec.prequery_stage.evidence_artifact_hash
                    or fixed_plan.query_stage_manifest_hash
                    != cast(StageReference, spec.query_stage).staging_manifest_hash
                    or fixed_derivation.source_c1_construction_seal_hash
                    != source_sealed.construction_seal.content_hash
                    or fixed_derivation.source_c1_preparation_hash
                    != cast(ConditionPreparation, source_preparation).content_hash
                    or fixed_derivation.fixed_ontology_hash
                    != fixed_ontology.content_hash
                    or fixed_derivation.evidence_alias_bijection_hash
                    != canonical_sha256(evidence_aliases)
                    or fixed_derivation.derived_output_schema_hash
                    != exact_schema_hash
                    or fixed_derivation.derived_decoding_manifest_hash
                    != run_config.decoding_manifest_hash
                    or fixed_derivation.exact_run_condition_config_hash
                    != run_config.content_hash
                    or fixed_derivation.derived_at
                    <= cast(ConditionPreparation, source_preparation).completed_at
                ):
                    raise DevelopmentAssessmentIntegrityError(
                        "FixedSelect schema/config derivation is not reproducible from C1"
                    )
            repair_input = None
            if spec.kind is DevelopmentCallKind.REPAIR_PROBE:
                if receipt.repair_probe_input is None or spec.parent_call_id is None:
                    raise DevelopmentAssessmentIntegrityError(
                        "repair probe lacks its deterministic input CAS lineage"
                    )
                repair_input = reader.logical(
                    receipt.repair_probe_input,
                    DevelopmentRepairProbeInput,
                )
                parent_processed = next(
                    (
                        item
                        for item in processed
                        if item.spec.call_id == spec.parent_call_id
                    ),
                    None,
                )
                if (
                    parent_processed is None
                    or not isinstance(
                        parent_processed.semantic_request, ConstructionRequest
                    )
                    or parent_processed.receipt is None
                    or parent_processed.receipt.raw_response_artifact_hash is None
                ):
                    raise DevelopmentAssessmentIntegrityError(
                        "repair probe parent lacks verified request/raw lineage"
                    )
                parent_raw = json.loads(
                    reader.bytes(
                        parent_processed.receipt.raw_response_artifact_hash
                    )
                )
                if not isinstance(parent_raw, dict):
                    raise DevelopmentAssessmentIntegrityError(
                        "repair parent raw response is not an object"
                    )
                expected_repair_input = build_development_repair_probe_input(
                    parent_call_id=spec.parent_call_id,
                    parent_semantic_request=parent_processed.semantic_request,
                    repair_semantic_request=cast(
                        ConstructionRequest, semantic_request
                    ),
                    parent_raw_output_hash=(
                        parent_processed.receipt.raw_response_artifact_hash
                    ),
                    parent_raw_output=parent_raw,
                    created_at=repair_input.created_at,
                )
                attempt_lineage = self.ledger.attempt_lineage(
                    cast(str, receipt.attempt_id)
                )
                if (
                    repair_input != expected_repair_input
                    or receipt.job_id != parent_processed.receipt.job_id
                    or len(attempt_lineage) != 2
                    or attempt_lineage[0].attempt_id
                    != parent_processed.receipt.attempt_id
                    or attempt_lineage[1].attempt_id != receipt.attempt_id
                    or attempt_lineage[1].parent_attempt_id
                    != parent_processed.receipt.attempt_id
                ):
                    raise DevelopmentAssessmentIntegrityError(
                        "repair fault/diagnostics or parent attempt lineage changed"
                    )
            elif (
                receipt.repair_probe_input is not None
                or receipt.repair_preservation_report is not None
            ):
                raise DevelopmentAssessmentIntegrityError(
                    "nonrepair call contains repair-only CAS lineage"
                )
            wire_messages = rendered.get("messages")
            system_content = (
                wire_messages[0].get("content")
                if isinstance(wire_messages, Sequence)
                and not isinstance(wire_messages, (str, bytes))
                and wire_messages
                and isinstance(wire_messages[0], Mapping)
                and wire_messages[0].get("role") == "system"
                else None
            )
            expected_system_content = (
                (self.root / "prompts/repair/prompt_v1.md").read_text(
                    encoding="utf-8"
                )
                if spec.kind is DevelopmentCallKind.REPAIR_PROBE
                else render_condition_system_prompt(self.root, spec.condition)
            )
            if (
                semantic_request.upper_ontology.content_hash
                != run_config.upper_ontology_hash
                or semantic_request.budgets != run_config.budgets
                or not isinstance(system_content, str)
                or system_content
                != expected_system_content
                or hashlib.sha256(system_content.encode("utf-8")).hexdigest()
                != run_config.prompt_hash
                or rendered.get("guided_json")
                != development_output_schema_for_request(semantic_request)
                or canonical_sha256(rendered.get("guided_json"))
                != run_config.output_schema_hash
            ):
                raise DevelopmentAssessmentIntegrityError(
                    "wire request differs from its frozen prompt/schema/configuration"
                )
            if semantic_request.requested_at < envelope.preconstruction_barrier_recorded_at:
                raise DevelopmentAssessmentIntegrityError(
                    "semantic request predates the sealed preconstruction barrier"
                )
            prequery_evidence, _ = _stage_objects(
                self.root, spec, query_revealed=False
            )
            _request_matches_staged_evidence(semantic_request, prequery_evidence)
            _assert_gold_free_payload(rendered)
            _assert_gold_free_payload(semantic_request)
            _packing_contains_all_inputs(
                rendered,
                semantic_request,
                packing,
                repair_input,
            )

            comparison = None
            packet = None
            context = None
            if isinstance(semantic_request, ConstructionRequest):
                comparison = reader.logical(
                    cast(LogicalCASReference, receipt.comparison_input_manifest),
                    ComparisonInputManifest,
                )
                packet = reader.logical(
                    cast(LogicalCASReference, receipt.evidence_packet), EvidencePacket
                )
                context = reader.logical(
                    cast(LogicalCASReference, receipt.query_context), QueryContext
                )
                _context_and_packet_match_request(semantic_request, context, packet)
                query_evidence, reveal = _stage_objects(
                    self.root, spec, query_revealed=True
                )
                assert reveal is not None
                if query_evidence != prequery_evidence:
                    raise DevelopmentAssessmentIntegrityError(
                        "query reveal stage changed the sealed evidence snapshot"
                    )
                _request_matches_staged_evidence(semantic_request, query_evidence)
                if (
                    reveal.query != to_model_visible_query(context)
                    or context.spoiler_horizon != query_evidence.snapshot.horizon
                ):
                    raise DevelopmentAssessmentIntegrityError(
                        "stored query context differs from the frozen reveal stage"
                    )
                if comparison.condition is not spec.condition:
                    raise DevelopmentAssessmentIntegrityError(
                        "comparison manifest condition differs from its call"
                    )
                if (
                    not _comparison_matches_run_config(comparison, run_config)
                    or comparison.upper_ontology_hash
                    != semantic_request.upper_ontology.content_hash
                    or comparison.snapshot_hash != semantic_request.snapshot_hash
                    or comparison.packet_hash != packet.content_hash
                    or comparison.ordered_evidence_ids != packet.ordered_evidence_ids
                    or comparison.horizon_hash != context.spoiler_horizon.content_hash
                    or comparison.context_semantics_hash
                    != _context_semantics_hash(context)
                    or comparison.budgets != context.budgets
                    or comparison.query_access_event_hash != row.query_access_event_hash
                ):
                    raise DevelopmentAssessmentIntegrityError(
                        "comparison manifest does not bind the actual request inputs"
                    )
                access = _verified_query_access(
                    self.ledger,
                    reader,
                    access_event_hash=cast(str, row.query_access_event_hash),
                    context=context,
                    packet=packet,
                    call=spec,
                    reveal=reveal,
                    prequery_barrier_hash=cast(str, envelope.prequery_barrier_hash),
                )
                if envelope.query_access_event != access:
                    raise DevelopmentAssessmentIntegrityError(
                        "execution envelope differs from persisted query access"
                    )
                _verified_packet_materialization(
                    self.ledger,
                    reader,
                    materialization_event_hash=cast(
                        str, receipt.packet_materialization_event_hash
                    ),
                    access=access,
                    packet=packet,
                    packet_reference=cast(
                        LogicalCASReference, receipt.evidence_packet
                    ),
                )
                if semantic_request.requested_at < access.accessed_at:
                    raise DevelopmentAssessmentIntegrityError(
                        "query-time semantic request predates physical query access"
                    )

            generation = None
            condition_result = None
            fixed_inventory = None
            fixed_probe = None
            if row.outcome is RunOutcome.SUCCEEDED:
                assert receipt.raw_response_artifact_hash is not None
                reader.bytes(receipt.raw_response_artifact_hash)
                generation = reader.logical(
                    cast(LogicalCASReference, receipt.validated_generation),
                    ValidatedGeneration,
                )
                expected_stage_hash = (
                    spec.prequery_stage.staging_manifest_hash
                    if spec.query_stage is None
                    else spec.query_stage.staging_manifest_hash
                )
                if (
                    generation.condition is not spec.condition
                    or generation.request_hash != semantic_request.content_hash
                    or generation.raw_output_artifact_hash
                    != receipt.raw_response_artifact_hash
                    or row.response_artifact_hash != receipt.raw_response_artifact_hash
                    or row.validated_generation_hash != generation.content_hash
                    or row.validation_record_hash
                    != canonical_sha256(generation.validation_records)
                    or generation.packing_report_hash != packing.content_hash
                    or generation.capability_manifest_hash != capabilities.content_hash
                    or generation.validator_hash != run_config.validator_hash
                    or generation.stage_manifest_hash != expected_stage_hash
                    or generation.model_stack_hash != run_config.model_stack_hash
                    or generation.decoding_manifest_hash
                    != run_config.decoding_manifest_hash
                    or generation.seed_manifest_hash != run_config.seed_manifest_hash
                    or generation.seed != run_config.resolved_seed
                    or generation.prompt_hash != run_config.prompt_hash
                    or generation.output_schema_hash != run_config.output_schema_hash
                    or generation.input_tokens != row.prompt_tokens
                    or generation.output_tokens != row.completion_tokens
                    or generation.generation_started_at < semantic_request.requested_at
                    or generation.validated_at > row.completed_at
                    or generation.query_access_event_hash != row.query_access_event_hash
                    or generation.prequery_barrier_hash
                    != envelope.prequery_barrier_hash
                ):
                    raise DevelopmentAssessmentIntegrityError(
                        "validated generation differs from request/runtime/ITT lineage"
                    )
                if spec.kind is DevelopmentCallKind.REPAIR_PROBE:
                    if (
                        repair_input is None
                        or receipt.repair_preservation_report is None
                        or generation.repair_attempt != 1
                        or generation.repair_parent_raw_output_hash
                        != repair_input.parent_raw_output_hash
                    ):
                        raise DevelopmentAssessmentIntegrityError(
                            "successful repair lacks exact repair-generation lineage"
                        )
                    preservation = reader.logical(
                        receipt.repair_preservation_report,
                        BoundaryValidationReport,
                    )
                    repaired_raw = json.loads(
                        reader.bytes(receipt.raw_response_artifact_hash)
                    )
                    if not isinstance(repaired_raw, dict):
                        raise DevelopmentAssessmentIntegrityError(
                            "repair raw response is not a JSON object"
                        )
                    expected_preservation = validate_repair_preservation(
                        base_draft=restore_model_output_source_aliases(
                            repair_input.invalid_draft,
                            alias_manifest,
                        ),
                        repaired_draft=restore_model_output_source_aliases(
                            repaired_raw,
                            alias_manifest,
                        ),
                        diagnosed_paths=tuple(
                            item.path for item in repair_input.diagnostics
                        ),
                    )
                    if (
                        preservation != expected_preservation
                        or not preservation.accepted
                        or preservation.content_hash
                        not in generation.validator_report_hashes
                    ):
                        raise DevelopmentAssessmentIntegrityError(
                            "repair preservation report is not mechanically reproducible"
                        )
                elif (
                    generation.repair_attempt != 0
                    or generation.repair_parent_raw_output_hash is not None
                ):
                    raise DevelopmentAssessmentIntegrityError(
                        "nonrepair generation claims repair lineage"
                    )
                result_model = (
                    ConditionPreparation
                    if spec.kind is DevelopmentCallKind.C1_PRECONSTRUCTION
                    else ConditionAttemptRecord
                )
                condition_result = reader.logical(
                    cast(LogicalCASReference, receipt.condition_result), result_model
                )
                expected_result_hash = (
                    row.condition_preparation_hash
                    if spec.kind is DevelopmentCallKind.C1_PRECONSTRUCTION
                    else row.condition_attempt_hash
                )
                if condition_result.content_hash != expected_result_hash:
                    raise DevelopmentAssessmentIntegrityError(
                        "condition result differs from the ITT logical hash"
                    )
                if isinstance(condition_result, ConditionPreparation):
                    sealed = condition_result.sealed_preontology
                    if (
                        sealed is None
                        or sealed.validated_generation != generation
                        or row.construction_seal_hash
                        != sealed.construction_seal.content_hash
                    ):
                        raise DevelopmentAssessmentIntegrityError(
                            "C1 condition preparation differs from its generation/seal"
                        )
                else:
                    projection = condition_result.projection
                    if (
                        condition_result.outcome is not RunOutcome.SUCCEEDED
                        or projection is None
                        or projection.generation_lineage_hash != generation.content_hash
                        or projection.packet_hash
                        != cast(EvidencePacket, packet).content_hash
                        or projection.context_hash != cast(QueryContext, context).content_hash
                    ):
                        raise DevelopmentAssessmentIntegrityError(
                            "query-time condition result differs from its request/generation"
                        )
                    certificate = projection.construction_certificate
                    if (
                        certificate is None
                        or certificate.stage_manifest_hash != expected_stage_hash
                        or certificate.prequery_barrier_hash
                        != envelope.prequery_barrier_hash
                        or certificate.query_revealed_at
                        != cast(QueryAccessEvent, envelope.query_access_event).accessed_at
                        or certificate.completed_at > row.completed_at
                    ):
                        raise DevelopmentAssessmentIntegrityError(
                            "query-time construction certificate has invalid stage/timing lineage"
                        )
                    expected_preparation = preparations.get((spec.unit_id, spec.condition))
                    if expected_preparation is not None and projection.pre_query_inventory != (
                        expected_preparation.empty_inventory
                    ):
                        raise DevelopmentAssessmentIntegrityError(
                            "active projection does not bind its pre-query empty inventory"
                        )
                if spec.condition is ConditionName.A_FIXED_SELECT:
                    fixed_inventory = reader.logical(
                        cast(LogicalCASReference, receipt.fixed_sealed_inventory),
                        SealedOntologyInventory,
                    )
                    fixed_probe = reader.logical(
                        cast(LogicalCASReference, receipt.fixed_forbidden_probe),
                        FixedSelectOutputAudit,
                    )
            _validate_ledger_validations(
                self.ledger,
                receipt,
                raw_response_hash=receipt.raw_response_artifact_hash,
                validator_hash=run_config.validator_hash,
                successful=row.outcome is RunOutcome.SUCCEEDED,
            )
            processed.append(
                _ProcessedCall(
                    spec=spec,
                    row=row,
                    receipt=receipt,
                    semantic_request=semantic_request,
                    run_config=run_config,
                    packing=packing,
                    capabilities=capabilities,
                    treatment=treatment,
                    generation=generation,
                    condition_result=condition_result,
                    comparison=comparison,
                    packet=packet,
                    context=context,
                    fixed_inventory=fixed_inventory,
                    fixed_probe=fixed_probe,
                )
            )
        if any(
            len(values) != 1
            for values in (
                service_identity_hashes,
                execution_ids,
                execution_manifest_hashes,
                preconstruction_barrier_hashes,
            )
        ):
            raise DevelopmentAssessmentIntegrityError(
                "development calls do not share one execution/service/barrier lineage"
            )
        return tuple(processed)

    def _load_cpu_projections(
        self,
        reader: _CASReader,
        input_manifest: DevelopmentAssessmentInputManifest,
        manifest: DevelopmentCallManifest,
        preparations: Mapping[tuple[str, ConditionName], ConditionPreparation],
    ) -> tuple[_ProcessedCPUProjection, ...]:
        processed: list[_ProcessedCPUProjection] = []
        keys: list[tuple[ConditionName, str, int]] = []
        query_specs: dict[tuple[str, int], DevelopmentCallSpec] = {}
        for unit_id in DEVELOPMENT_UNIT_IDS:
            calls = tuple(
                item
                for item in manifest.calls
                if item.unit_id == unit_id
                and item.kind is DevelopmentCallKind.C2_CONSTRUCTION
            )
            if len(calls) != 3:
                raise DevelopmentAssessmentIntegrityError(
                    "CPU projection audit cannot resolve the three frozen query stages"
                )
            query_specs.update(
                ((unit_id, ordinal), call)
                for ordinal, call in enumerate(calls, start=1)
            )
        for artifact_hash in input_manifest.cpu_projection_receipt_artifact_hashes:
            receipt = reader.receipt(artifact_hash, DevelopmentCPUProjectionReceipt)
            attempt = reader.logical(receipt.condition_attempt, ConditionAttemptRecord)
            projection = reader.logical(receipt.projection, OntologyProjection)
            run_config = reader.logical(receipt.run_condition_config, RunConditionConfig)
            comparison = reader.logical(
                receipt.comparison_input_manifest, ComparisonInputManifest
            )
            packet = reader.logical(receipt.evidence_packet, EvidencePacket)
            context = reader.logical(receipt.query_context, QueryContext)
            query_spec = query_specs[(receipt.unit_id, receipt.query_ordinal)]
            staged, reveal = _stage_objects(
                self.root, query_spec, query_revealed=True
            )
            assert reveal is not None
            if (
                reveal.query != to_model_visible_query(context)
                or context.spoiler_horizon != staged.snapshot.horizon
            ):
                raise DevelopmentAssessmentIntegrityError(
                    "CPU projection context differs from its frozen query stage"
                )
            _model_visible_packet_matches_stage(
                to_model_visible_packet(packet), packet.snapshot_hash, staged
            )
            if (
                attempt.outcome is not RunOutcome.SUCCEEDED
                or attempt.projection != projection
                or attempt.condition is not receipt.condition
                or projection.condition is not receipt.condition
                or run_config.condition is not receipt.condition
                or comparison.condition is not receipt.condition
            ):
                raise DevelopmentAssessmentIntegrityError(
                    "CPU projection receipt contains inconsistent condition output"
                )
            preparation = preparations[(receipt.unit_id, receipt.condition)]
            sealed = preparation.sealed_preontology
            if (
                sealed is None
                or not _comparison_matches_run_config(comparison, run_config)
                or projection.construction_seal != sealed.construction_seal
                or projection.snapshot_hash != preparation.snapshot_hash
                or packet.snapshot_hash != preparation.snapshot_hash
                or projection.packet_hash != packet.content_hash
                or projection.context_hash != context.content_hash
                or projection.budgets != context.budgets
                or run_config.budgets != context.budgets
                or comparison.packet_hash != packet.content_hash
                or comparison.snapshot_hash != packet.snapshot_hash
                or comparison.ordered_evidence_ids != packet.ordered_evidence_ids
                or comparison.horizon_hash != context.spoiler_horizon.content_hash
                or comparison.context_semantics_hash
                != _context_semantics_hash(context)
                or comparison.upper_ontology_hash
                != projection.upper_ontology.content_hash
                or comparison.query_access_event_hash != projection.query_access_event_hash
                or run_config.content_hash != receipt.run_condition_config.logical_content_hash
            ):
                raise DevelopmentAssessmentIntegrityError(
                    "CPU projection differs from its seal, packet, context, or config"
                )
            _verify_sealed_projection(projection, preparation)
            access = _verified_query_access(
                self.ledger,
                reader,
                access_event_hash=projection.query_access_event_hash,
                context=context,
                packet=packet,
                call=query_spec,
                reveal=reveal,
                prequery_barrier_hash=comparison.prequery_barrier_hash,
            )
            _verified_packet_materialization(
                self.ledger,
                reader,
                materialization_event_hash=receipt.packet_materialization_event_hash,
                access=access,
                packet=packet,
                packet_reference=receipt.evidence_packet,
            )
            try:
                ledger_projection = self.ledger.get_projection(receipt.ledger_projection_id)
                self.ledger.get_job(receipt.job_id)
            except KeyError as exc:
                raise DevelopmentAssessmentIntegrityError(
                    "CPU projection receipt cites missing ledger lineage"
                ) from exc
            if (
                ledger_projection.job_id != receipt.job_id
                or ledger_projection.validation_id not in receipt.ledger_validation_ids
                or ledger_projection.snapshot_id != staged.snapshot.snapshot_id
                or ledger_projection.packet_input_id != packet.packet_id
                or ledger_projection.projection_artifact_hash != receipt.projection.artifact_hash
                or ledger_projection.context_hash != context.content_hash
                or ledger_projection.condition_id != receipt.condition.value
                or ledger_projection.upper_ontology_hash
                != projection.upper_ontology.content_hash
                or ledger_projection.construction_certificate_hash
                != cast(Any, projection.construction_seal).content_hash
                or ledger_projection.release_class is not LedgerReleaseClass.PUBLIC
            ):
                raise DevelopmentAssessmentIntegrityError(
                    "CPU projection CAS object differs from the projection ledger row"
                )
            for validation_id in receipt.ledger_validation_ids:
                try:
                    validation = self.ledger.get_validation(validation_id)
                except KeyError as exc:
                    raise DevelopmentAssessmentIntegrityError(
                        "CPU projection cites a missing validation row"
                    ) from exc
                if (
                    validation.job_id != receipt.job_id
                    or validation.attempt_id != receipt.attempt_id
                    or validation.input_artifact_hash != receipt.projection.artifact_hash
                    or validation.validator_manifest_hash != run_config.validator_hash
                    or validation.validation_status
                    is not LedgerValidationStatus.ACCEPTED
                    or validation.evidence_support_status
                    is not LedgerEvidenceSupportStatus.NOT_APPLICABLE
                    or validation.temporal_status
                    is not LedgerTemporalValidationStatus.NOT_APPLICABLE
                    or validation.commitment_status
                    is not LedgerCommitmentCheckStatus.NOT_APPLICABLE
                    or validation.semantic_assessment_scope
                    is not LedgerSemanticAssessmentScope.RUNTIME_STRUCTURAL_ONLY_NOT_ASSESSED
                ):
                    raise DevelopmentAssessmentIntegrityError(
                        "CPU validation ledger differs from the accepted projection"
                    )
            keys.append((receipt.condition, receipt.unit_id, receipt.query_ordinal))
            processed.append(
                _ProcessedCPUProjection(
                    receipt=receipt,
                    attempt=attempt,
                    projection=projection,
                    run_config=run_config,
                    comparison=comparison,
                    packet=packet,
                    context=context,
                )
            )
        expected_keys = {
            (condition, unit_id, ordinal)
            for condition in (ConditionName.C0_CLASSICAL_PRE, ConditionName.C1_LLM_PRE)
            for unit_id in DEVELOPMENT_UNIT_IDS
            for ordinal in (1, 2, 3)
        }
        if len(keys) != len(set(keys)) or set(keys) != expected_keys:
            raise DevelopmentAssessmentIntegrityError(
                "assessment requires exactly 12 C0 and 12 C1 CPU projection outputs"
            )
        return tuple(processed)

    def _load_gold(
        self,
        manifest: DevelopmentCallManifest,
    ) -> tuple[dict[str, Any], frozenset[str]]:
        from story_projection_onto.synthetic_benchmark import (
            BenchmarkSplit,
            ScorerWorldArtifact,
            SyntheticBenchmarkManifest,
        )

        benchmark_path = self.root / "data/synthetic/manifests/benchmark_manifest.json"
        if benchmark_path.is_symlink() or not benchmark_path.is_file():
            raise DevelopmentAssessmentIntegrityError("benchmark manifest is missing")
        if _sha256_file(benchmark_path) != manifest.benchmark_manifest_file_sha256:
            raise DevelopmentAssessmentIntegrityError("benchmark manifest bytes changed")
        try:
            benchmark = SyntheticBenchmarkManifest.model_validate_json(
                benchmark_path.read_text(encoding="utf-8")
            )
        except Exception as exc:
            raise DevelopmentAssessmentIntegrityError("benchmark manifest is invalid") from exc
        file_records = {
            item.relative_path: item
            for item in benchmark.generated_files
            if item.namespace == "scorer_only"
        }
        scorer_by_unit: dict[str, Any] = {}
        file_hashes: set[str] = set()
        for unit in self.prequery_inputs.unit_bindings:
            world_id = resolve_development_world_id(
                self.root,
                unit.staged_model_visible_evidence_hash,
                unit.neutral_full_evidence_artifact_hash,
            )
            manifest_relative = f"scorer_only/development/{world_id}.json"
            record = file_records.get(manifest_relative)
            path = self.root / "data/synthetic" / manifest_relative
            if record is None or path.is_symlink() or not path.is_file():
                raise DevelopmentAssessmentIntegrityError(
                    "scorer artifact is absent from benchmark manifest: "
                    f"{manifest_relative}"
                )
            observed = _sha256_file(path)
            if observed != record.sha256:
                raise DevelopmentAssessmentIntegrityError(
                    f"development scorer artifact bytes changed: {manifest_relative}"
                )
            try:
                scorer = ScorerWorldArtifact.model_validate_json(
                    path.read_text(encoding="utf-8")
                )
            except Exception as exc:
                raise DevelopmentAssessmentIntegrityError(
                    f"development scorer artifact is invalid: {manifest_relative}"
                ) from exc
            if (
                scorer.world_spec.world_id != world_id
                or scorer.world_spec.split is not BenchmarkSplit.DEVELOPMENT
                or len(scorer.gold_projections) != 3
            ):
                raise DevelopmentAssessmentIntegrityError(
                    "development scorer artifact belongs to another unit or split"
                )
            scorer_by_unit[unit.unit_id] = scorer
            file_hashes.add(observed)
        return scorer_by_unit, frozenset(file_hashes)

    @staticmethod
    def _alignment_plan(scorer: Any, ordinal: int) -> AlignmentPlan:
        from story_projection_onto.metrics.alignment import build_alignment_plan
        from story_projection_onto.synthetic_benchmark import compile_alignment_alternatives

        gold = scorer.gold_projections[ordinal - 1]
        alternatives = scorer.alternatives[ordinal - 1]
        return build_alignment_plan(
            gold,
            alternatives,
            compiled_alternatives=compile_alignment_alternatives(gold, alternatives),
        )

    def _assess_c0(
        self,
        cpu: Sequence[_ProcessedCPUProjection],
        scorer_by_unit: Mapping[str, Any],
    ) -> tuple[float, float, float, float]:
        true_positives = 0
        predicted_count = 0
        gold_count = 0
        valid_citations = 0
        citation_count = 0
        families = {
            "explicit_entity": False,
            "binary_relation": False,
            "event_role": False,
            "story_time": False,
            "validity_time": False,
        }
        for item in cpu:
            if item.receipt.condition is not ConditionName.C0_CLASSICAL_PRE:
                continue
            scorer = scorer_by_unit[item.receipt.unit_id]
            gold = scorer.gold_projections[item.receipt.query_ordinal - 1]
            direct_ids = _direct_assertion_ids(scorer, gold)
            plan = _filtered_alignment_plan(
                self._alignment_plan(scorer, item.receipt.query_ordinal), direct_ids
            )
            valid_ids = frozenset(item.packet.ordered_evidence_ids)
            nodes, assertions, _ = _prediction_bundle(
                local_schema=item.projection.local_schema,
                instance_graph=item.projection.instance_graph,
                plan=plan,
                valid_evidence_ids=valid_ids,
            )
            result = score_alignment(
                plan=plan,
                predicted_nodes=nodes,
                predicted_assertions=assertions,
            )
            true_positives += len(result.strict_assertion_matches)
            predicted_count += len(assertions)
            gold_count += len(plan.assertion_targets)
            families["explicit_entity"] |= bool(result.node_matches)
            matched_target_ids = {match.target_id for match in result.strict_assertion_matches}
            matched_gold = tuple(
                assertion
                for assertion in gold.qualified_assertions
                if assertion.assertion_id in matched_target_ids
            )
            families["binary_relation"] |= any(
                assertion.subject_id is not None and assertion.object_id is not None
                for assertion in matched_gold
            )
            families["event_role"] |= any(assertion.roles for assertion in matched_gold)
            families["story_time"] |= any(
                assertion.temporal_scope.story_time.kind
                not in {
                    TemporalKind.UNKNOWN,
                    TemporalKind.NOT_APPLICABLE,
                    TemporalKind.HORIZON_WITHHELD,
                }
                for assertion in matched_gold
            )
            families["validity_time"] |= any(
                assertion.temporal_scope.validity_time.kind
                not in {
                    TemporalKind.UNKNOWN,
                    TemporalKind.NOT_APPLICABLE,
                    TemporalKind.HORIZON_WITHHELD,
                }
                for assertion in matched_gold
            )
            citations = _cited_evidence_ids(item.projection)
            citation_count += len(citations)
            valid_citations += sum(value in valid_ids for value in citations)
        precision = true_positives / predicted_count if predicted_count else 0.0
        recall = true_positives / gold_count if gold_count else 0.0
        evidence_rate = valid_citations / citation_count if citation_count else 0.0
        return sum(families.values()) / len(families), precision, recall, evidence_rate

    def _assess_c1(
        self,
        processed: Sequence[_ProcessedCall],
        scorer_by_unit: Mapping[str, Any],
    ) -> tuple[bool, bool, float, float, float]:
        calls = tuple(
            item
            for item in processed
            if item.spec.kind is DevelopmentCallKind.C1_PRECONSTRUCTION
        )
        if len(calls) != 4 or any(item.generation is None for item in calls):
            return False, False, 0.0, 0.0, 0.0
        schema_valid = True
        operators: set[ConstructionOperator] = set()
        citations: list[str] = []
        valid_citation_count = 0
        grounding_supported: set[tuple[str, str]] = set()
        emitted_assertions: set[tuple[str, str]] = set()
        matched_union_targets: set[tuple[str, str]] = set()
        union_gold_targets: set[tuple[str, str]] = set()
        for call in calls:
            assert isinstance(call.semantic_request, PreconstructionRequest)
            assert isinstance(call.condition_result, ConditionPreparation)
            sealed = call.condition_result.sealed_preontology
            if sealed is None:
                schema_valid = False
                continue
            sealed_horizon = call.semantic_request.sealed_horizon
            if sealed_horizon is None:
                raise DevelopmentIntegrityError(
                    "development C1 assessment lacks its trusted sealed horizon"
                )
            draft = sealed.draft
            report = validate_draft_structure(
                draft=draft,
                upper_ontology=call.semantic_request.upper_ontology,
                evidence=call.semantic_request.evidence,
                horizon=sealed_horizon,
                budgets=call.semantic_request.budgets,
                capabilities=call.semantic_request.capabilities,
            )
            schema_valid &= report.accepted and bool(call.generation.validation_records)
            schema_valid &= all(
                item.validation_status is ValidationStatus.ACCEPTED
                for item in call.generation.validation_records
            )
            operators.update(item.operator for item in draft.decisions)
            valid_ids = frozenset(item.evidence_id for item in call.semantic_request.evidence)
            draft_citations = _cited_evidence_ids(draft)
            citations.extend(draft_citations)
            valid_citation_count += sum(item in valid_ids for item in draft_citations)
            scorer = scorer_by_unit[call.spec.unit_id]
            for ordinal in (1, 2, 3):
                plan = self._alignment_plan(scorer, ordinal)
                nodes, assertions, supported = _prediction_bundle(
                    local_schema=draft.local_schema,
                    instance_graph=draft.instance_graph,
                    plan=plan,
                    valid_evidence_ids=valid_ids,
                )
                result = score_alignment(
                    plan=plan,
                    predicted_nodes=nodes,
                    predicted_assertions=assertions,
                )
                union_gold_targets.update(
                    (
                        call.spec.unit_id,
                        _context_independent_assertion_slot(target.target_id),
                    )
                    for target in plan.assertion_targets
                )
                matched_union_targets.update(
                    (
                        call.spec.unit_id,
                        _context_independent_assertion_slot(match.target_id),
                    )
                    for match in result.strict_assertion_matches
                )
                grounding_supported.update((call.spec.unit_id, item) for item in supported)
            emitted_assertions.update(
                (call.spec.unit_id, item.assertion_id)
                for item in draft.instance_graph.assertions
            )
        valid_rate = valid_citation_count / len(citations) if citations else 0.0
        grounding = (
            len(grounding_supported) / len(emitted_assertions) if emitted_assertions else 0.0
        )
        recall = (
            len(matched_union_targets) / len(union_gold_targets)
            if union_gold_targets
            else 0.0
        )
        return (
            schema_valid,
            CONSTRUCTIVE_OPERATORS.issubset(operators),
            valid_rate,
            grounding,
            recall,
        )

    @staticmethod
    def _assess_c2_construction(
        processed: Sequence[_ProcessedCall],
        preparations: Mapping[tuple[str, ConditionName], ConditionPreparation],
    ) -> bool:
        calls = tuple(
            item for item in processed if item.spec.kind is DevelopmentCallKind.C2_CONSTRUCTION
        )
        if len(calls) != 12 or any(item.generation is None for item in calls):
            return False
        demonstrated: set[ConstructionOperator] = set()
        required = {
            ConstructionOperator.MERGE,
            ConstructionOperator.SPLIT,
            ConstructionOperator.CONTEXTUAL_TYPE,
            ConstructionOperator.SCHEMA_RELATION,
            ConstructionOperator.EVENT_REIFICATION,
            ConstructionOperator.ABSTRACTION,
            ConstructionOperator.TEMPORAL_QUALIFICATION,
            ConstructionOperator.EPISTEMIC_QUALIFICATION,
            ConstructionOperator.RARE_PRESERVATION,
            ConstructionOperator.SUPPORTED_DESCRIPTION,
        }
        for call in calls:
            result = call.condition_result
            if not isinstance(result, ConditionAttemptRecord) or result.projection is None:
                return False
            projection = result.projection
            certificate = projection.construction_certificate
            preparation = preparations[(call.spec.unit_id, ConditionName.C2_LLM_QUERY)]
            if (
                certificate is None
                or projection.pre_query_inventory != preparation.empty_inventory
                or not any(item.operator in CONSTRUCTIVE_OPERATORS for item in projection.decisions)
            ):
                return False
            demonstrated.update(item.operator for item in projection.decisions)
        return required.issubset(demonstrated)

    def _count_horizon_leaks(
        self,
        processed: Sequence[_ProcessedCall],
        cpu: Sequence[_ProcessedCPUProjection],
    ) -> int:
        leaks = 0
        for item in cpu:
            valid = set(item.packet.ordered_evidence_ids)
            leaks += sum(
                evidence_id not in valid for evidence_id in _cited_evidence_ids(item.projection)
            )
            leaks += _packet_horizon_leaks(item.packet, item.context)
            leaks += _graph_horizon_leaks(
                item.projection.instance_graph, item.context.spoiler_horizon
            )
        for call in processed:
            result = call.condition_result
            if call.packet is not None and call.context is not None:
                leaks += _packet_horizon_leaks(call.packet, call.context)
            if isinstance(result, ConditionAttemptRecord) and result.projection is not None:
                assert call.packet is not None and call.context is not None
                valid = set(call.packet.ordered_evidence_ids)
                leaks += sum(
                    evidence_id not in valid
                    for evidence_id in _cited_evidence_ids(result.projection)
                )
                leaks += _graph_horizon_leaks(
                    result.projection.instance_graph, call.context.spoiler_horizon
                )
            elif isinstance(result, ConditionPreparation):
                sealed = result.sealed_preontology
                request = call.semantic_request
                if sealed is not None and isinstance(request, PreconstructionRequest):
                    stage, _ = _stage_objects(
                        self.root, call.spec, query_revealed=False
                    )
                    valid = {item.evidence_id for item in request.evidence}
                    leaks += sum(
                        evidence_id not in valid
                        for evidence_id in _cited_evidence_ids(sealed.draft)
                    )
                    maximum_discourse = (
                        stage.snapshot.horizon.max_discourse_position.ordering_key
                    )
                    leaks += sum(
                        item.discourse_position.ordering_key > maximum_discourse
                        for item in request.evidence
                    )
                    leaks += _graph_horizon_leaks(
                        sealed.draft.instance_graph, stage.snapshot.horizon
                    )
        return leaks

    @staticmethod
    def _assess_evidence_equality(
        processed: Sequence[_ProcessedCall],
        cpu: Sequence[_ProcessedCPUProjection],
        manifest: DevelopmentCallManifest,
    ) -> bool:
        ordinal_by_stage = _query_ordinal_by_stage(manifest)
        grouped: dict[tuple[str, int], list[tuple[ConditionName, ComparisonInputManifest]]] = {}
        for item in cpu:
            grouped.setdefault((item.receipt.unit_id, item.receipt.query_ordinal), []).append(
                (item.receipt.condition, item.comparison)
            )
        for call in processed:
            if call.comparison is None or call.spec.kind not in {
                DevelopmentCallKind.C2_CONSTRUCTION,
                DevelopmentCallKind.FIXED_SELECTION,
            }:
                continue
            ordinal = _query_ordinal(call.spec, ordinal_by_stage)
            assert ordinal is not None
            grouped.setdefault((call.spec.unit_id, ordinal), []).append(
                (call.spec.condition, call.comparison)
            )
        if set(grouped) != {
            (unit_id, ordinal) for unit_id in DEVELOPMENT_UNIT_IDS for ordinal in (1, 2, 3)
        }:
            return False
        invariant_fields = (
            "snapshot_hash",
            "packet_hash",
            "ordered_evidence_ids",
            "query_access_event_hash",
            "prequery_barrier_hash",
            "horizon_hash",
            "context_semantics_hash",
            "upper_ontology_hash",
            "budgets",
            "maximum_input_tokens",
            "maximum_output_tokens",
            "repair_attempt_budget",
            "scored_schema_hash",
            "validator_hash",
        )
        for values in grouped.values():
            main = {
                condition: comparison
                for condition, comparison in values
                if condition
                in {
                    ConditionName.C0_CLASSICAL_PRE,
                    ConditionName.C1_LLM_PRE,
                    ConditionName.C2_LLM_QUERY,
                }
            }
            if set(main) != {
                ConditionName.C0_CLASSICAL_PRE,
                ConditionName.C1_LLM_PRE,
                ConditionName.C2_LLM_QUERY,
            }:
                return False
            for field_name in invariant_fields:
                if len(
                    {
                        canonical_sha256(getattr(item, field_name))
                        for item in main.values()
                    }
                ) != 1:
                    return False
            baseline = main[ConditionName.C2_LLM_QUERY]
            for condition, comparison in values:
                if condition in {ConditionName.A_NO_CONTEXT}:
                    fields = tuple(
                        name for name in invariant_fields if name != "context_semantics_hash"
                    )
                else:
                    fields = invariant_fields
                if any(
                    canonical_sha256(getattr(comparison, name))
                    != canonical_sha256(getattr(baseline, name))
                    for name in fields
                ):
                    return False
        return True

    def _assess_c1_query_blindness(self, processed: Sequence[_ProcessedCall]) -> bool:
        c1 = tuple(
            item
            for item in processed
            if item.spec.kind is DevelopmentCallKind.C1_PRECONSTRUCTION
        )
        query_calls = tuple(item for item in processed if item.spec.query_stage is not None)
        if len(c1) != 4 or not query_calls or any(item.receipt is None for item in c1):
            return False
        access_times = []
        for item in query_calls:
            if item.row.query_access_event_hash is None:
                continue
            try:
                access = self.ledger.get_query_access(item.row.query_access_event_hash)
            except KeyError:
                return False
            access_times.append(datetime.fromisoformat(access.accessed_at.replace("Z", "+00:00")))
        if not access_times:
            return False
        earliest_access = min(access_times)
        for item in c1:
            if not isinstance(item.semantic_request, PreconstructionRequest):
                return False
            if (
                item.row.query_access_event_hash is not None
                or item.row.completed_at >= earliest_access
            ):
                return False
            if item.packing is None or any(
                section.name == "query_context" for section in item.packing.sections
            ):
                return False
        return True

    @staticmethod
    def _assess_empty_inventories(
        processed: Sequence[_ProcessedCall],
        preparations: Mapping[tuple[str, ConditionName], ConditionPreparation],
    ) -> bool:
        active_conditions = {
            ConditionName.C2_LLM_QUERY,
            ConditionName.A_NO_CONTEXT,
            ConditionName.A_NO_TEMPORAL_EPISTEMIC,
            ConditionName.A_NO_RARE_GUARD,
        }
        active = tuple(
            item for item in processed if item.spec.condition in active_conditions
        )
        if not active:
            return False
        for item in active:
            preparation = preparations.get((item.spec.unit_id, item.spec.condition))
            if preparation is None or preparation.empty_inventory is None:
                return False
            inventory = preparation.empty_inventory
            if any(
                (
                    inventory.ontology_refs,
                    inventory.local_schema_ids,
                    inventory.entity_ids,
                    inventory.event_ids,
                    inventory.assertion_ids,
                    inventory.finalized_predicate_ids,
                )
            ):
                return False
            result = item.condition_result
            if (
                isinstance(result, ConditionAttemptRecord)
                and result.projection is not None
                and result.projection.pre_query_inventory != inventory
            ):
                return False
        return True

    def _assess_fixed(
        self,
        processed: Sequence[_ProcessedCall],
        preparations: Mapping[tuple[str, ConditionName], ConditionPreparation],
    ) -> tuple[bool, bool]:
        fixed = tuple(
            item for item in processed if item.spec.kind is DevelopmentCallKind.FIXED_SELECTION
        )
        if len(fixed) != 4:
            return False, False
        packing_ok = True
        rejection_ok = True
        tested: set[ConstructionOperator] = set()
        for item in fixed:
            if (
                item.generation is None
                or item.packing is None
                or item.fixed_inventory is None
                or item.fixed_probe is None
                or not isinstance(item.semantic_request, ConstructionRequest)
            ):
                packing_ok = False
                rejection_ok = False
                continue
            request = item.semantic_request
            fixed_ontology = request.fixed_ontology
            source_preparation = preparations.get(
                (item.spec.unit_id, ConditionName.C1_LLM_PRE)
            )
            source = (
                None
                if source_preparation is None
                else source_preparation.sealed_preontology
            )
            expected_inventory = (
                None
                if fixed_ontology is None or source is None
                else sealed_inventory_from_fixed_ontology(
                    fixed_ontology,
                    seed_block=item.spec.seed_block,
                    source_draft=source.draft,
                )
            )
            if (
                fixed_ontology is None
                or source is None
                or fixed_ontology != source.as_fixed_ontology()
                or item.packing.complete_sealed_ontology is not True
                or item.fixed_inventory.seal_hash
                != fixed_ontology.construction_seal.content_hash
                or item.fixed_inventory.seed_block != item.spec.seed_block
                or item.fixed_inventory != expected_inventory
            ):
                packing_ok = False
            try:
                enforce_fixed_select_draft(
                    item.generation.draft,
                    sealed=item.fixed_inventory,
                    seed_block=item.spec.seed_block,
                )
            except Exception:
                rejection_ok = False
            probe_payload = item.fixed_probe.model_dump(
                mode="python",
                exclude={"schema_version", "content_hash", "operations"},
            )
            for operation in item.fixed_probe.operations:
                tested.add(operation.capability)
                individual_probe = FixedSelectOutputAudit(
                    **probe_payload,
                    operations=(operation,),
                )
                try:
                    enforce_fixed_select_output(individual_probe, item.fixed_inventory)
                except FixedSelectCapabilityError as exc:
                    expected_violation = (
                        "constructive operation is forbidden in A-FixedSelect: "
                        f"{operation.capability.value}"
                    )
                    if expected_violation not in exc.violations:
                        rejection_ok = False
                else:
                    rejection_ok = False
            if any(
                decision.operator not in FIXED_SELECT_ALLOWED_OPERATORS
                for decision in item.generation.draft.decisions
            ):
                rejection_ok = False
        expected_forbidden = set(
            CapabilityManifest.for_condition(ConditionName.A_FIXED_SELECT).forbidden
        )
        rejection_ok &= tested == expected_forbidden
        return packing_ok, rejection_ok

    @staticmethod
    def _assess_one_switch(
        processed: Sequence[_ProcessedCall],
        manifest: DevelopmentCallManifest,
    ) -> bool:
        by_id = {item.spec.call_id: item for item in processed}
        probes = tuple(
            item for item in processed if item.spec.kind is DevelopmentCallKind.ABLATION_PROBE
        )
        if len(probes) != 3:
            return False
        expected_switch = {
            ConditionName.A_NO_CONTEXT: "context_payload_mode",
            ConditionName.A_NO_TEMPORAL_EPISTEMIC: "temporal_epistemic_fields",
            ConditionName.A_NO_RARE_GUARD: "rare_guard",
        }
        expected_config_differences = {
            ConditionName.A_NO_CONTEXT: {
                "capability_manifest_hash",
                "condition",
                "config_id",
            },
            ConditionName.A_NO_TEMPORAL_EPISTEMIC: {
                "capability_manifest_hash",
                "condition",
                "config_id",
                "decoding_manifest_hash",
                "output_schema_hash",
                "prompt_hash",
            },
            ConditionName.A_NO_RARE_GUARD: {
                "capability_manifest_hash",
                "condition",
                "config_id",
                "prompt_hash",
            },
        }
        for probe in probes:
            parent = by_id.get(probe.spec.parent_call_id or "")
            delta = probe.spec.configuration_delta
            if (
                parent is None
                or parent.run_config is None
                or probe.run_config is None
                or parent.treatment is None
                or probe.treatment is None
                or delta is None
                or delta.switch_name != expected_switch[probe.spec.condition]
            ):
                return False
            parent_payload = parent.treatment.model_dump(
                mode="python", exclude={"schema_version", "content_hash"}
            )
            probe_payload = probe.treatment.model_dump(
                mode="python", exclude={"schema_version", "content_hash"}
            )
            changed = tuple(
                key for key in parent_payload if parent_payload[key] != probe_payload[key]
            )
            if changed != (delta.switch_name,):
                return False
            parent_config = parent.run_config.model_dump(
                mode="python", exclude={"schema_version", "content_hash"}
            )
            probe_config = probe.run_config.model_dump(
                mode="python", exclude={"schema_version", "content_hash"}
            )
            config_differences = {
                key
                for key in parent_config
                if parent_config[key] != probe_config[key]
            }
            if config_differences != expected_config_differences[probe.spec.condition]:
                return False
            if (
                parent_payload[delta.switch_name] != delta.baseline_value
                or probe_payload[delta.switch_name] != delta.probe_value
                or _invariant_run_config_values(parent.run_config)
                != _invariant_run_config_values(probe.run_config)
            ):
                return False
            if not isinstance(parent.semantic_request, ConstructionRequest) or not isinstance(
                probe.semantic_request, ConstructionRequest
            ):
                return False
            if (
                parent.semantic_request.packet != probe.semantic_request.packet
                or parent.semantic_request.snapshot_hash != probe.semantic_request.snapshot_hash
                or parent.semantic_request.upper_ontology != probe.semantic_request.upper_ontology
                or parent.semantic_request.budgets != probe.semantic_request.budgets
            ):
                return False
            if probe.spec.condition is not ConditionName.A_NO_CONTEXT and (
                parent.semantic_request.context != probe.semantic_request.context
            ):
                return False
        return tuple(item.call_id for item in manifest.calls) == tuple(
            item.spec.call_id for item in processed
        )


def build_development_assessment_input_manifest(
    *,
    root: Path,
    ledger: Ledger,
    blobs: BlobStore,
    assessment_bundle_artifact_hash: Sha256Digest,
) -> DevelopmentAssessmentInputManifest:
    """Build the scorer routing manifest from one verified gold-free CAS bundle.

    This post-run bridge reads no scorer artifact.  The caller persists the
    returned canonical JSON and passes both its path and byte SHA-256 to the
    provider factory.
    """

    repository = Path(root).resolve(strict=True)
    reader = _CASReader(ledger, blobs)
    bundle = reader.receipt(
        assessment_bundle_artifact_hash,
        DevelopmentAssessmentBundle,
    )
    source_revision = _verify_runtime_source_manifest(
        reader,
        bundle.runtime_source_manifest,
        root=repository,
        expected_tree_hash=bundle.source_tree_hash,
    )
    source_paths = tuple(sorted(_CORE_RUNTIME_SOURCE_PATHS))
    bindings = (
        *(
            RuntimeSourceBinding(
                relative_path=relative_path,
                sha256=_sha256_file(repository / relative_path),
                role="core_runtime",
            )
            for relative_path in source_paths
        ),
        RuntimeSourceBinding(
            relative_path="src/story_projection_onto/development_adapter.py",
            sha256=_sha256_file(
                repository / "src/story_projection_onto/development_adapter.py"
            ),
            role="development_service_adapter",
        ),
    )
    _verify_runtime_sources(repository, bindings)
    return DevelopmentAssessmentInputManifest(
        call_manifest_hash=bundle.call_manifest_hash,
        prequery_inputs_hash=bundle.prequery_inputs_hash,
        source_tree_hash=bundle.source_tree_hash,
        source_revision=source_revision,
        benchmark_manifest_file_sha256=bundle.benchmark_manifest_file_sha256,
        assessment_bundle_artifact_hash=assessment_bundle_artifact_hash,
        runtime_source_manifest=bundle.runtime_source_manifest,
        prequery_preparation_artifacts=bundle.prequery_preparation_artifacts,
        call_receipt_artifact_hashes=bundle.call_receipt_artifact_hashes,
        service_result_artifact_hashes=bundle.service_result_artifact_hashes,
        cpu_projection_receipt_artifact_hashes=(
            bundle.cpu_projection_receipt_artifact_hashes
        ),
        packing_preflight_artifact_hash=bundle.packing_preflight_artifact_hash,
        runtime_sources=bindings,
    )


def build_development_scientific_assessment_provider(
    *,
    root: Path,
    ledger: Ledger,
    blobs: BlobStore,
    prequery_inputs: DevelopmentPrequeryInputs,
    assessment_manifest_path: Path,
    assessment_manifest_file_sha256: Sha256Digest,
) -> DevelopmentScientificAssessmentProvider:
    """Construct the injected closure without reading any scorer-gold artifact."""

    return DevelopmentScientificAssessmentProvider(
        root=root,
        ledger=ledger,
        blobs=blobs,
        prequery_inputs=prequery_inputs,
        assessment_manifest_path=assessment_manifest_path,
        assessment_manifest_file_sha256=assessment_manifest_file_sha256,
    )


__all__ = [
    "DevelopmentAssessmentInputManifest",
    "DevelopmentAssessmentIntegrityError",
    "DevelopmentCPUProjectionReceipt",
    "DevelopmentCallAuditReceipt",
    "DevelopmentScientificAssessmentProvider",
    "DevelopmentTreatmentSwitches",
    "LogicalCASReference",
    "OpaqueJSONReference",
    "RuntimeSourceBinding",
    "build_development_assessment_input_manifest",
    "build_development_scientific_assessment_provider",
]
