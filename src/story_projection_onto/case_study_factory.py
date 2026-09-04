"""Concrete, fail-closed production factory for the bounded novel transfer study.

The execution plan is deliberately path-free.  This module is the only outer
lifecycle boundary that turns explicitly supplied restricted paths into the
single owned vLLM service used by Phase 6.  It does not discover a corpus, copy
novel prose, or create any public case-study result.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal, cast

from pydantic import AwareDatetime

from story_projection_onto.case_study_execution import (
    CASE_SERVICE_START_WATCHDOG_SECONDS,
    CaseAdmissionEvidenceBundle,
    CaseAdmissionEvidenceReference,
    CaseArtifactReference,
    CaseExecutionAdmissionReceipt,
    CaseExecutionRepository,
    CaseStudyAdmissionError,
    CaseStudyExecutionResult,
    CaseStudyProductionController,
    ProductionCaseClassicalAdapter,
    _gpu_inventory_hash,
    _logical_json_hash,
    _parse_record,
    _persist_mapping,
    _persist_record,
    _publish_private_no_replace,
    _read_reference,
    _total_allocated_seconds,
    validate_case_admission_evidence,
)
from story_projection_onto.case_study_gpu import (
    ProductionCaseStudyGpuAdapter,
    build_production_case_study_gpu_adapter,
)
from story_projection_onto.case_study_runtime import (
    AttestedRestrictedCaseStudy,
    AttestedSelectedModelFreeze,
    CaseStudyAdmissionAttestation,
    CaseStudyExecutionPlan,
    CaseStudyRuntimePolicy,
    load_attested_restricted_case_study,
    load_attested_selected_model_freeze,
    load_case_study_admission_attestation,
    load_case_study_execution_plan,
)
from story_projection_onto.contracts import (
    ImmutableRecord,
    ReleaseClass,
    Sha256Digest,
    canonical_json,
)
from story_projection_onto.development_adapter import (
    DEFAULT_DEVELOPMENT_CONSTRUCTION_CONFIG,
    DevelopmentConstructionConfiguration,
    PackingTokenizer,
)
from story_projection_onto.experiment import AllocatedGPUMeter, ResourceLimits
from story_projection_onto.fallback_acceptance import validate_source_association
from story_projection_onto.gpu_runtime import (
    FALLBACK_MODEL_REPOSITORY,
    FALLBACK_MODEL_REVISION,
    FALLBACK_SERVED_MODEL_NAME,
    ResourceSampler,
    ServiceState,
    VLLMGuidedJSONClient,
    VLLMLaunchConfiguration,
    VLLMService,
    capture_tokenizer_manifest,
)
from story_projection_onto.manifest import build_source_manifest
from story_projection_onto.model_gate import (
    FallbackModelPolicy,
    validate_fallback_snapshot_manifest,
)
from story_projection_onto.store import (
    ArtifactStore,
    BlobStore,
    Ledger,
    StorageBudget,
    StoragePreflight,
    StorageReport,
)
from story_projection_onto.store import ReleaseClass as LedgerReleaseClass

CASE_PRODUCTION_FACTORY = (
    "story_projection_onto.case_study_factory:"
    "create_frozen_production_case_study_bundle"
)
CASE_FACTORY_REVISION = "case-study-production-factory-v1"


class CaseStudyFactoryError(CaseStudyAdmissionError):
    """An explicitly supplied production dependency differs from the freeze."""


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _safe_file(path: Path, *, label: str, maximum_bytes: int = 64 * 1024 * 1024) -> Path:
    if path.is_symlink() or not path.is_file():
        raise CaseStudyFactoryError(f"{label} must be one regular non-symlink file")
    if path.stat().st_size > maximum_bytes:
        raise CaseStudyFactoryError(f"{label} exceeds its bounded input size")
    return path.resolve(strict=True)


def _ensure_private_directory(path: Path, *, restricted_root: Path, label: str) -> Path:
    root = restricted_root.resolve(strict=True)
    lexical = Path(os.path.abspath(path))
    try:
        lexical.relative_to(root)
    except ValueError as error:
        raise CaseStudyFactoryError(f"{label} must remain inside the restricted root") from error
    cursor = root
    relative = lexical.relative_to(root)
    for part in relative.parts:
        cursor /= part
        if cursor.exists() and cursor.is_symlink():
            raise CaseStudyFactoryError(f"{label} cannot traverse a symbolic link")
    lexical.mkdir(mode=0o700, parents=True, exist_ok=True)
    os.chmod(lexical, 0o700)
    resolved = lexical.resolve(strict=True)
    if resolved != lexical:
        raise CaseStudyFactoryError(f"{label} requires a stable real path")
    return resolved


def _inside_quota(path: Path, quota_root: Path, *, label: str) -> None:
    try:
        path.resolve(strict=False).relative_to(quota_root.resolve(strict=True))
    except ValueError as error:
        raise CaseStudyFactoryError(f"{label} lies outside the controlled quota root") from error


def _atomic_private_record(path: Path, value: ImmutableRecord) -> None:
    payload = (value.to_canonical_json() + "\n").encode("utf-8")
    try:
        _publish_private_no_replace(path, payload)
    except Exception as error:
        raise CaseStudyFactoryError("immutable case bootstrap record changed") from error


def _restricted_location(
    path: Path,
    *,
    restricted_root: Path,
    label: str,
    directory: bool,
) -> Path:
    """Require a stable real path with no ancestor symlink under restricted root."""

    root = restricted_root.resolve(strict=True)
    lexical = Path(os.path.abspath(path))
    try:
        relative = lexical.relative_to(root)
    except ValueError as error:
        raise CaseStudyFactoryError(f"{label} must remain inside the restricted root") from error
    cursor = root
    for part in relative.parts:
        cursor /= part
        if cursor.exists() and cursor.is_symlink():
            raise CaseStudyFactoryError(f"{label} cannot traverse a symbolic link")
    target = lexical if directory else lexical.parent
    target.mkdir(mode=0o700, parents=True, exist_ok=True)
    if target.resolve(strict=True) != target:
        raise CaseStudyFactoryError(f"{label} requires a stable real path")
    if directory:
        os.chmod(lexical, 0o700)
    elif lexical.exists() and (lexical.is_symlink() or not lexical.is_file()):
        raise CaseStudyFactoryError(f"{label} must be one regular non-symlink file")
    return lexical


def _load_record(path: Path, model: type[ImmutableRecord], *, label: str) -> ImmutableRecord:
    resolved = _safe_file(path, label=label)
    try:
        return model.model_validate_json(resolved.read_bytes())
    except Exception as error:
        raise CaseStudyFactoryError(f"{label} is not a valid immutable record") from error


class CaseAdmissionBootstrapIntent(ImmutableRecord):
    """Path-free intent written before admission mutates the cumulative ledger."""

    factory_revision: str = CASE_FACTORY_REVISION
    execution_plan_hash: Sha256Digest
    admission_attestation_hash: Sha256Digest
    evidence_bundle_hash: Sha256Digest
    evidence_bundle_reference_hash: Sha256Digest
    construction_configuration_hash: Sha256Digest
    construction_configuration_file_sha256: Sha256Digest
    source_revision: str
    source_tree_sha256: Sha256Digest
    predecessor_ledger_sha256: Sha256Digest
    prior_gpu_event_inventory_hash: Sha256Digest
    allocated_gpu_seconds_before_case: float
    admitted_at: AwareDatetime
    release_class: Literal[ReleaseClass.RESTRICTED] = ReleaseClass.RESTRICTED


@dataclass(slots=True)
class CaseStudyProductionBundle:
    """Owned Phase-6 runtime; the controller remains the sole service owner."""

    controller: CaseStudyProductionController
    admission_reference: CaseArtifactReference
    artifacts: ArtifactStore
    ledger: Ledger
    service: VLLMService = field(repr=False)
    gpu: ProductionCaseStudyGpuAdapter = field(repr=False)
    runtime_root: Path = field(repr=False)
    storage_report: StorageReport
    _closed: bool = field(default=False, init=False, repr=False)

    def persist_result(self, result: CaseStudyExecutionResult) -> CaseArtifactReference:
        """Persist one path-free terminal result in restricted CAS and immutable JSON."""

        reference = _persist_record(
            self.artifacts,
            result,
            object_kind="case_study_execution_result",
            created_at=result.completed_at,
        )
        result_path = self.runtime_root / "results" / f"{result.content_hash}.json"
        _atomic_private_record(result_path, result)
        _atomic_private_record(
            self.runtime_root / "terminal-result-reference.json",
            reference,
        )
        return reference

    def close(self) -> None:
        """Reclaim any live process and close the ledger without hiding failures."""

        if self._closed:
            return
        try:
            if self.service.state is not ServiceState.STOPPED:
                try:
                    self.gpu.shutdown()
                except BaseException as adapter_error:
                    # A startup failure can stop the physical process before the
                    # adapter has a persisted service identity.  The raw owner is
                    # then the only component capable of closing its journal.
                    try:
                        self.service.shutdown()
                    except BaseException as service_error:
                        raise service_error from adapter_error
                    raise
        finally:
            self.ledger.close()
            self._closed = True


def stage_case_admission_evidence(
    *,
    restricted_root: Path,
    admission_attestation_path: Path,
    gate_paths: Mapping[str, Path],
    ledger_path: Path,
    artifact_root: Path,
    bundle_output_path: Path,
    reference_output_path: Path,
    staged_at: datetime,
) -> tuple[CaseAdmissionEvidenceBundle, CaseArtifactReference]:
    """Copy exactly eight hash-valid gate records into the restricted CAS.

    The source gate files may be public reports, but their CAS copies are always
    marked restricted so the later admission validator has one uniform access
    boundary.  JSON is reserialized canonically; no source path is retained.
    """

    expected_names = (
        "synthetic_run_closure_hash",
        "timing_lineage_audit_hash",
        "gold_firewall_audit_hash",
        "registered_metric_regeneration_hash",
        "blinded_error_review_hash",
        "storage_preflight_hash",
        "gpu_schedule_admission_hash",
        "public_release_scan_hash",
    )
    if set(gate_paths) != set(expected_names):
        raise CaseStudyFactoryError("case admission staging requires exactly eight named gates")
    if staged_at.tzinfo is None or staged_at.utcoffset() is None:
        raise CaseStudyFactoryError("case admission staging time must be timezone-aware")
    if not bundle_output_path.is_absolute() or not reference_output_path.is_absolute():
        raise CaseStudyFactoryError("case gate outputs require explicit absolute paths")
    root = restricted_root.resolve(strict=True)
    _ensure_private_directory(
        bundle_output_path.parent,
        restricted_root=root,
        label="case admission bundle output",
    )
    _ensure_private_directory(
        reference_output_path.parent,
        restricted_root=root,
        label="case admission reference output",
    )
    ledger_path = _restricted_location(
        ledger_path,
        restricted_root=root,
        label="case cumulative ledger",
        directory=False,
    )
    artifact_root = _restricted_location(
        artifact_root,
        restricted_root=root,
        label="case restricted CAS",
        directory=True,
    )
    admission = load_case_study_admission_attestation(
        admission_attestation_path,
        restricted_root=restricted_root,
    )
    if ledger_path.is_symlink() or not ledger_path.is_file():
        raise CaseStudyFactoryError("case gate staging requires the existing study ledger")
    ledger = Ledger(ledger_path)
    try:
        if _total_allocated_seconds(ledger) <= 0:
            raise CaseStudyFactoryError("case gate staging requires the cumulative study ledger")
        artifacts = ArtifactStore(BlobStore(artifact_root), ledger)
        references: list[CaseAdmissionEvidenceReference] = []
        for name in expected_names:
            source = _safe_file(gate_paths[name], label=f"case gate {name}")
            try:
                value = json.loads(source.read_bytes())
            except (UnicodeDecodeError, json.JSONDecodeError) as error:
                raise CaseStudyFactoryError(f"case gate {name} is not valid JSON") from error
            if not isinstance(value, Mapping):
                raise CaseStudyFactoryError(f"case gate {name} must contain one JSON object")
            expected_hash = cast(str, getattr(admission, name))
            if _logical_json_hash(value) != expected_hash:
                raise CaseStudyFactoryError(f"case gate {name} differs from its attestation")
            # The terminal newline gives an existing public canonical object a
            # distinct raw CAS identity while preserving its logical JSON hash.
            artifact = artifacts.put_bytes(
                canonical_json(value).encode("utf-8") + b"\n",
                media_type="application/vnd.story-projection.case-admission-gate+json",
                release_class=LedgerReleaseClass.RESTRICTED,
                created_at=staged_at,
            )
            references.append(
                CaseAdmissionEvidenceReference(
                    name=cast(Any, name),
                    logical_content_hash=expected_hash,
                    artifact_hash=artifact.content_hash,
                )
            )
        bundle = CaseAdmissionEvidenceBundle(
            bundle_id=f"case-admission-evidence-{admission.content_hash[:16]}",
            admission_attestation_hash=admission.content_hash,
            evidence=tuple(references),
            frozen_at=staged_at.astimezone(UTC),
        )
        reference = _persist_record(
            artifacts,
            bundle,
            object_kind="case_admission_evidence_bundle",
            created_at=staged_at,
        )
        _atomic_private_record(bundle_output_path, bundle)
        _atomic_private_record(reference_output_path, reference)
        return bundle, reference
    finally:
        ledger.close()


def _load_bootstrap_reference(
    *,
    path: Path,
    artifacts: ArtifactStore,
    plan: CaseStudyExecutionPlan,
    admission_attestation: CaseStudyAdmissionAttestation,
    construction: DevelopmentConstructionConfiguration,
    evidence_bundle_reference: CaseArtifactReference,
    source_tree_sha256: str,
    source_revision: str,
    expected_predecessor_ledger_sha256: str,
    intent_path: Path,
    ledger: Ledger,
) -> tuple[CaseExecutionAdmissionReceipt, CaseArtifactReference] | None:
    if not path.exists():
        return None
    reference = cast(
        CaseArtifactReference,
        _load_record(path, CaseArtifactReference, label="case admission reference"),
    )
    if reference.object_kind != "case_execution_admission":
        raise CaseStudyFactoryError("case admission reference has another object kind")
    admission = cast(
        CaseExecutionAdmissionReceipt,
        _parse_record(artifacts, reference, CaseExecutionAdmissionReceipt),
    )
    if not intent_path.is_file() or intent_path.is_symlink():
        raise CaseStudyFactoryError("persisted case admission lacks its bootstrap intent")
    intent = cast(
        CaseAdmissionBootstrapIntent,
        _load_record(
            intent_path,
            CaseAdmissionBootstrapIntent,
            label="case admission bootstrap intent",
        ),
    )
    source_payload = json.loads(_read_reference(artifacts, admission.source_manifest))
    if (
        intent.factory_revision != CASE_FACTORY_REVISION
        or intent.execution_plan_hash != plan.content_hash
        or intent.admission_attestation_hash != admission_attestation.content_hash
        or intent.evidence_bundle_hash
        != evidence_bundle_reference.logical_content_hash
        or intent.evidence_bundle_reference_hash
        != evidence_bundle_reference.content_hash
        or intent.construction_configuration_hash != construction.content_hash
        or intent.construction_configuration_file_sha256
        != construction.source_file_sha256
        or intent.source_revision != source_revision
        or intent.source_tree_sha256 != source_tree_sha256
        or intent.predecessor_ledger_sha256 != expected_predecessor_ledger_sha256
        or admission.execution_plan_hash != plan.content_hash
        or admission.admission_attestation_hash != admission_attestation.content_hash
        or admission.construction_configuration_hash != construction.content_hash
        or admission.construction_configuration_file_sha256 != construction.source_file_sha256
        or admission.admission_evidence_bundle != evidence_bundle_reference
        or admission.source_revision != source_revision
        or admission.predecessor_ledger_sha256 != intent.predecessor_ledger_sha256
        or admission.prior_gpu_event_inventory_hash
        != intent.prior_gpu_event_inventory_hash
        or abs(
            admission.allocated_gpu_seconds_before_case
            - intent.allocated_gpu_seconds_before_case
        )
        > 1e-6
        or admission.admitted_at != intent.admitted_at
        or not isinstance(source_payload, Mapping)
        or source_payload.get("tree_sha256") != source_tree_sha256
    ):
        raise CaseStudyFactoryError("persisted case admission differs from current frozen inputs")
    current_seconds = _total_allocated_seconds(ledger)
    if (
        not math.isfinite(current_seconds)
        or current_seconds + 1e-6 < intent.allocated_gpu_seconds_before_case
    ):
        raise CaseStudyFactoryError("cumulative GPU allocation regressed after case admission")
    if (
        abs(current_seconds - intent.allocated_gpu_seconds_before_case) <= 1e-6
        and _gpu_inventory_hash(ledger) != intent.prior_gpu_event_inventory_hash
    ):
        raise CaseStudyFactoryError("pre-case GPU event inventory changed")
    return admission, reference


def _build_or_resume_admission(
    *,
    repository: Path,
    runtime_root: Path,
    source_revision: str,
    source_tree_sha256: str,
    plan: CaseStudyExecutionPlan,
    admission_attestation: CaseStudyAdmissionAttestation,
    evidence_bundle: CaseAdmissionEvidenceBundle,
    evidence_bundle_reference: CaseArtifactReference,
    construction: DevelopmentConstructionConfiguration,
    construction_path: Path,
    ledger: Ledger,
    artifacts: ArtifactStore,
    ledger_path: Path,
    expected_predecessor_ledger_sha256: str,
    clock: Callable[[], datetime],
) -> tuple[CaseExecutionAdmissionReceipt, CaseArtifactReference]:
    reference_path = runtime_root / "case-execution-admission-reference.json"
    intent_path = runtime_root / "case-admission-bootstrap-intent.json"
    validate_case_admission_evidence(
        admission=admission_attestation,
        bundle=evidence_bundle,
        artifacts=artifacts,
    )
    if evidence_bundle_reference.logical_content_hash != evidence_bundle.content_hash:
        raise CaseStudyFactoryError("case admission bundle reference changed")
    if _read_reference(artifacts, evidence_bundle_reference) != canonical_json(
        evidence_bundle
    ).encode("utf-8"):
        raise CaseStudyFactoryError("case admission bundle CAS bytes changed")
    if construction.source_file_sha256 != _file_sha256(construction_path):
        raise CaseStudyFactoryError("case construction configuration bytes changed")
    if construction.upper_ontology.content_hash != admission_attestation.upper_ontology_hash:
        raise CaseStudyFactoryError("case upper ontology differs from admission")
    recovered = _load_bootstrap_reference(
        path=reference_path,
        artifacts=artifacts,
        plan=plan,
        admission_attestation=admission_attestation,
        construction=construction,
        evidence_bundle_reference=evidence_bundle_reference,
        source_tree_sha256=source_tree_sha256,
        source_revision=source_revision,
        expected_predecessor_ledger_sha256=expected_predecessor_ledger_sha256,
        intent_path=intent_path,
        ledger=ledger,
    )
    if recovered is not None:
        return recovered

    source = build_source_manifest(repository, source_revision)
    if source.tree_sha256 != source_tree_sha256:
        raise CaseStudyFactoryError("case source association and rebuilt source tree differ")
    prior_seconds = _total_allocated_seconds(ledger)
    if not math.isfinite(prior_seconds) or prior_seconds <= 0:
        raise CaseStudyFactoryError("case execution cannot reset cumulative GPU accounting")
    prior_inventory_hash = _gpu_inventory_hash(ledger)
    observed_ledger_hash = _file_sha256(ledger_path)
    now = clock()
    if now.tzinfo is None or now.utcoffset() is None:
        raise CaseStudyFactoryError("case factory clock must be timezone-aware")
    intent = CaseAdmissionBootstrapIntent(
        execution_plan_hash=plan.content_hash,
        admission_attestation_hash=admission_attestation.content_hash,
        evidence_bundle_hash=evidence_bundle.content_hash,
        evidence_bundle_reference_hash=evidence_bundle_reference.content_hash,
        construction_configuration_hash=construction.content_hash,
        construction_configuration_file_sha256=construction.source_file_sha256,
        source_revision=source_revision,
        source_tree_sha256=source.tree_sha256,
        predecessor_ledger_sha256=expected_predecessor_ledger_sha256,
        prior_gpu_event_inventory_hash=prior_inventory_hash,
        allocated_gpu_seconds_before_case=prior_seconds,
        admitted_at=now.astimezone(UTC),
    )
    if intent_path.exists():
        retained = cast(
            CaseAdmissionBootstrapIntent,
            _load_record(
                intent_path,
                CaseAdmissionBootstrapIntent,
                label="case admission bootstrap intent",
            ),
        )
        static_fields = intent.model_dump(mode="python", exclude={"content_hash", "admitted_at"})
        retained_fields = retained.model_dump(
            mode="python", exclude={"content_hash", "admitted_at"}
        )
        if static_fields != retained_fields:
            raise CaseStudyFactoryError("case admission bootstrap intent changed")
        intent = retained
        if (
            prior_inventory_hash != intent.prior_gpu_event_inventory_hash
            or abs(prior_seconds - intent.allocated_gpu_seconds_before_case) > 1e-6
        ):
            raise CaseStudyFactoryError(
                "case admission bootstrap was interrupted after GPU accounting changed"
            )
    else:
        if observed_ledger_hash != expected_predecessor_ledger_sha256:
            raise CaseStudyFactoryError("cumulative predecessor ledger hash changed")
        _atomic_private_record(intent_path, intent)

    source_reference = _persist_mapping(
        artifacts,
        source.to_dict(),
        object_kind="case_source_manifest",
        created_at=intent.admitted_at,
    )
    admission = CaseExecutionAdmissionReceipt(
        receipt_id=f"admission-{plan.execution_id}",
        execution_plan_hash=plan.content_hash,
        admission_attestation_hash=admission_attestation.content_hash,
        admission_evidence_bundle=evidence_bundle_reference,
        source_manifest=source_reference,
        source_revision=source_revision,
        construction_configuration_file_sha256=construction.source_file_sha256,
        construction_configuration_hash=construction.content_hash,
        predecessor_ledger_sha256=intent.predecessor_ledger_sha256,
        prior_gpu_event_inventory_hash=intent.prior_gpu_event_inventory_hash,
        allocated_gpu_seconds_before_case=intent.allocated_gpu_seconds_before_case,
        admitted_at=intent.admitted_at,
    )
    reference = _persist_record(
        artifacts,
        admission,
        object_kind="case_execution_admission",
        created_at=intent.admitted_at,
    )
    _atomic_private_record(reference_path, reference)
    return admission, reference


def _verify_runtime_model(
    *,
    repository: Path,
    plan: CaseStudyExecutionPlan,
    selected_freeze: AttestedSelectedModelFreeze,
    snapshot_path: Path,
    shared_cache: Path,
    verified_model_manifest_path: Path,
    port: int,
) -> tuple[VLLMLaunchConfiguration, Any, Any, str]:
    policy_path = repository / "configs/study/fallback_model.json"
    policy = FallbackModelPolicy.load(policy_path)
    snapshot_manifest = validate_fallback_snapshot_manifest(
        verified_model_manifest_path,
        policy=policy,
        policy_path=policy_path,
        snapshot_path=snapshot_path,
        shared_cache=shared_cache,
    )
    snapshot_hash = cast(str, snapshot_manifest["manifest_sha256"])
    if (
        selected_freeze.repository != FALLBACK_MODEL_REPOSITORY
        or selected_freeze.revision != FALLBACK_MODEL_REVISION
        or selected_freeze.payload.get("served_model_name") != FALLBACK_SERVED_MODEL_NAME
        or snapshot_hash != plan.model_runtime.selected_snapshot_manifest_hash
    ):
        raise CaseStudyFactoryError("case model snapshot differs from selected fallback freeze")
    launcher = VLLMLaunchConfiguration.from_model_configuration(
        snapshot_path=snapshot_path,
        shared_cache=shared_cache,
        model_configuration_path=repository / "configs/study/model.json",
        model_candidate="fallback",
        verified_snapshot_manifest_sha256=snapshot_hash,
        port=port,
    )
    if launcher.configuration_hash != plan.model_runtime.selected_launcher_configuration_hash:
        raise CaseStudyFactoryError("case vLLM launcher differs from execution plan")
    tokenizer_manifest = capture_tokenizer_manifest(
        snapshot_path,
        repository=FALLBACK_MODEL_REPOSITORY,
        revision=FALLBACK_MODEL_REVISION,
    )
    if tokenizer_manifest.manifest_sha256 != plan.model_runtime.selected_tokenizer_manifest_hash:
        raise CaseStudyFactoryError("case tokenizer differs from execution plan")
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        str(snapshot_path),
        local_files_only=True,
        trust_remote_code=False,
        revision=FALLBACK_MODEL_REVISION,
    )
    return launcher, tokenizer, tokenizer_manifest, snapshot_hash


def create_frozen_production_case_study_bundle(
    *,
    repository: Path,
    restricted_root: Path,
    plan_path: Path,
    index_path: Path,
    index_manifest_path: Path,
    preregistration_path: Path,
    input_attestation_path: Path,
    admission_attestation_path: Path,
    selected_model_freeze_path: Path,
    admission_evidence_bundle_path: Path,
    admission_evidence_bundle_reference_path: Path,
    construction_path: Path,
    ledger_path: Path,
    artifact_root: Path,
    runtime_root: Path,
    quota_root: Path,
    snapshot_path: Path,
    shared_cache: Path,
    verified_model_manifest_path: Path,
    source_association_path: Path,
    expected_predecessor_ledger_sha256: str,
    source_revision: str,
    port: int = 8000,
    clock: Callable[[], datetime] = lambda: datetime.now(UTC),
) -> CaseStudyProductionBundle:
    """Build the exact 4/8/1 controller without starting the model service.

    All path-bearing inputs are explicit.  The returned controller starts vLLM
    only after bounded-packet materialization and lossless C1 packing succeed.
    """

    if repository.is_symlink():
        raise CaseStudyFactoryError("repository root cannot be a symbolic link")
    repository = repository.resolve(strict=True)
    if restricted_root.is_symlink():
        raise CaseStudyFactoryError("restricted root cannot be a symbolic link")
    restricted_root = restricted_root.resolve(strict=True)
    ledger_path = _restricted_location(
        ledger_path,
        restricted_root=restricted_root,
        label="case cumulative ledger",
        directory=False,
    )
    artifact_root = _restricted_location(
        artifact_root,
        restricted_root=restricted_root,
        label="case restricted CAS",
        directory=True,
    )
    runtime_root = _ensure_private_directory(
        runtime_root,
        restricted_root=restricted_root,
        label="case runtime root",
    )
    if quota_root.is_symlink():
        raise CaseStudyFactoryError("quota root cannot be a symbolic link")
    quota_root = quota_root.resolve(strict=True)
    for label, path in (
        ("repository", repository),
        ("restricted root", restricted_root),
        ("ledger", ledger_path),
        ("artifact root", artifact_root),
        ("runtime root", runtime_root),
        ("shared cache", shared_cache),
        ("snapshot", snapshot_path),
    ):
        _inside_quota(path, quota_root, label=label)
    if (
        len(expected_predecessor_ledger_sha256) != 64
        or any(
            character not in "0123456789abcdef"
            for character in expected_predecessor_ledger_sha256
        )
    ):
        raise CaseStudyFactoryError("predecessor ledger hash must be lowercase SHA-256")
    if not source_revision or source_revision.strip() != source_revision:
        raise CaseStudyFactoryError("source revision must be explicit and stripped")
    if ledger_path.is_symlink() or not ledger_path.is_file():
        raise CaseStudyFactoryError("case execution requires the existing cumulative ledger")
    for label, path in (
        ("artifact root", artifact_root),
        ("shared cache", shared_cache),
        ("snapshot", snapshot_path),
    ):
        if path.is_symlink():
            raise CaseStudyFactoryError(f"{label} cannot be a symbolic link")

    plan = load_case_study_execution_plan(plan_path, restricted_root=restricted_root)
    loaded: AttestedRestrictedCaseStudy = load_attested_restricted_case_study(
        restricted_root=restricted_root,
        index_path=index_path,
        manifest_path=index_manifest_path,
        preregistration_path=preregistration_path,
        attestation_path=input_attestation_path,
    )
    admission_attestation = load_case_study_admission_attestation(
        admission_attestation_path,
        restricted_root=restricted_root,
    )
    selected_freeze = load_attested_selected_model_freeze(
        restricted_root=restricted_root,
        selected_model_freeze_path=selected_model_freeze_path,
        admission=admission_attestation,
    )
    if (
        plan.input_attestation_hash != loaded.attestation.content_hash
        or plan.admission_attestation_hash != admission_attestation.content_hash
        or plan.model_runtime.selected_model_freeze_hash != selected_freeze.manifest_sha256
    ):
        raise CaseStudyFactoryError("case plan differs from its exact restricted attestations")
    policy = CaseStudyRuntimePolicy.load(repository / "configs/case_study/runtime.json")
    if (
        plan.runtime_policy_hash != policy.content_hash
        or policy.production_adapter_factory != CASE_PRODUCTION_FACTORY
        or policy.service_start_watchdog_seconds != CASE_SERVICE_START_WATCHDOG_SECONDS
    ):
        raise CaseStudyFactoryError("case execution plan does not bind the production factory")
    association = validate_source_association(
        _safe_file(source_association_path, label="source association"),
        source_root=repository,
    )
    if (
        association.get("revision_label") != source_revision
        or not isinstance(association.get("local_tree_sha256"), str)
    ):
        raise CaseStudyFactoryError("source revision differs from its current association")
    source_tree_sha256 = cast(str, association["local_tree_sha256"])

    expected_construction_path = (
        repository / DEFAULT_DEVELOPMENT_CONSTRUCTION_CONFIG
    ).resolve(strict=True)
    if construction_path.resolve(strict=True) != expected_construction_path:
        raise CaseStudyFactoryError("case construction path differs from the frozen configuration")
    construction = DevelopmentConstructionConfiguration.load(expected_construction_path)
    evidence_bundle = cast(
        CaseAdmissionEvidenceBundle,
        _load_record(
            admission_evidence_bundle_path,
            CaseAdmissionEvidenceBundle,
            label="case admission evidence bundle",
        ),
    )
    evidence_bundle_reference = cast(
        CaseArtifactReference,
        _load_record(
            admission_evidence_bundle_reference_path,
            CaseArtifactReference,
            label="case admission evidence bundle reference",
        ),
    )

    limits = ResourceLimits.load(repository / "configs/study/resource_limits.json")
    storage = StoragePreflight(
        quota_root,
        controlled_paths=(
            repository,
            restricted_root,
            ledger_path.parent,
            artifact_root,
            shared_cache,
        ),
        budget=StorageBudget(
            total_allocation_bytes=limits.maximum_project_allocation_bytes,
            max_occupied_bytes=limits.maximum_project_occupied_bytes,
            min_headroom_bytes=limits.minimum_storage_headroom_bytes,
        ),
    )
    storage_report = storage.check()
    if not storage_report.allowed:
        raise CaseStudyFactoryError("case storage preflight failed")

    ledger = Ledger(ledger_path)
    try:
        artifacts = ArtifactStore(BlobStore(artifact_root), ledger)
        meter = AllocatedGPUMeter.from_limits(ledger, limits)
        admission, admission_reference = _build_or_resume_admission(
            repository=repository,
            runtime_root=runtime_root,
            source_revision=source_revision,
            source_tree_sha256=source_tree_sha256,
            plan=plan,
            admission_attestation=admission_attestation,
            evidence_bundle=evidence_bundle,
            evidence_bundle_reference=evidence_bundle_reference,
            construction=construction,
            construction_path=expected_construction_path,
            ledger=ledger,
            artifacts=artifacts,
            ledger_path=ledger_path,
            expected_predecessor_ledger_sha256=expected_predecessor_ledger_sha256,
            clock=clock,
        )
        storage_report = storage.check()
        if not storage_report.allowed:
            raise CaseStudyFactoryError("case storage preflight failed after admission bootstrap")
        ledger.record_storage_sample(storage_report, phase="case_study:factory")

        launcher, tokenizer, tokenizer_manifest, snapshot_hash = _verify_runtime_model(
            repository=repository,
            plan=plan,
            selected_freeze=selected_freeze,
            snapshot_path=snapshot_path,
            shared_cache=shared_cache,
            verified_model_manifest_path=verified_model_manifest_path,
            port=port,
        )
        sampler = ResourceSampler(limits=limits, storage=storage, ledger=ledger)
        client = VLLMGuidedJSONClient(launcher.base_url)
        log_path = runtime_root / "case-study.vllm.log"
        if log_path.is_symlink():
            raise CaseStudyFactoryError("case vLLM log cannot be a symbolic link")
        service = VLLMService(
            configuration=launcher,
            client=client,
            meter=meter,
            log_path=log_path,
            startup_resource_sampler=sampler,
            preflight_endpoint_check=lambda: client.health(0.25),
            readiness_check=lambda: client.ready(
                2.0,
                model_name=FALLBACK_SERVED_MODEL_NAME,
            ),
        )
        repository_state = CaseExecutionRepository(
            plan=plan,
            artifacts=artifacts,
            restricted_root=restricted_root,
            resume_directory=runtime_root / "resume",
            state_pointer_path=runtime_root / "controller" / "current.json",
            clock=clock,
        )
        classical = ProductionCaseClassicalAdapter(
            root=repository,
            plan=plan,
            construction=construction,
            artifacts=artifacts,
            restricted_state_directory=runtime_root / "c0-state",
            clock=clock,
        )
        gpu = build_production_case_study_gpu_adapter(
            root=repository,
            plan=plan,
            loaded=loaded,
            admission=admission,
            construction=construction,
            tokenizer=cast(PackingTokenizer, tokenizer),
            tokenizer_manifest=tokenizer_manifest,
            service=service,
            artifacts=artifacts,
            repository=repository_state,
            state_pointer_path=runtime_root / "gpu" / "current.json",
            model_manifest_hash=snapshot_hash,
            service_start_watchdog_seconds=policy.service_start_watchdog_seconds,
            clock=clock,
        )
        def token_counter(value: str) -> int:
            return len(tokenizer.encode(value, add_special_tokens=False))
        controller = CaseStudyProductionController(
            root=repository,
            plan=plan,
            loaded=loaded,
            admission=admission,
            repository=repository_state,
            ledger=ledger,
            artifacts=artifacts,
            classical=classical,
            gpu=gpu,
            token_counter=token_counter,
            service_checkpoint_path=runtime_root / "service-checkpoint.json",
            clock=clock,
        )
        return CaseStudyProductionBundle(
            controller=controller,
            admission_reference=admission_reference,
            artifacts=artifacts,
            ledger=ledger,
            service=service,
            gpu=gpu,
            runtime_root=runtime_root,
            storage_report=storage_report,
        )
    except BaseException:
        ledger.close()
        raise


__all__ = [
    "CASE_FACTORY_REVISION",
    "CASE_PRODUCTION_FACTORY",
    "CaseAdmissionBootstrapIntent",
    "CaseStudyFactoryError",
    "CaseStudyProductionBundle",
    "create_frozen_production_case_study_bundle",
    "stage_case_admission_evidence",
]
