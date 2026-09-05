"""Fail-closed orchestration contracts for the bounded first-novel study.

This module compiles immutable, path-free execution manifests.  It deliberately
does not discover a corpus, invoke a model, construct C2 semantics on CPU, or
create reviewer judgments.  The eventual execution adapter must implement the
strict protocols below and return hash-only receipts for validation and resume.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime
from enum import StrEnum
from pathlib import Path, PurePosixPath
from typing import Literal, Protocol, Self, runtime_checkable

from pydantic import AwareDatetime, Field, model_validator

from story_projection_onto.contracts import (
    ConditionName,
    ConstructionCertificate,
    ConstructionSeal,
    Identifier,
    ImmutableRecord,
    PreQueryInventory,
    ReleaseClass,
    RetrievalMethod,
    RunOutcome,
    Sha256Digest,
    canonical_json,
    canonical_sha256,
)
from story_projection_onto.llm import base_condition_output_schema
from story_projection_onto.novel_case import (
    CaseStudyPreregistration,
    NovelSegmentationConfig,
    OperationalRetrievalReceipt,
    RestrictedNovelIndexManifest,
    validate_case_preregistration,
)


class CaseStudyRuntimeError(RuntimeError):
    """A case-study input, plan, chronology, or resume invariant failed."""


class CaseStudyInputError(CaseStudyRuntimeError):
    """Exact restricted input attestation could not be verified."""


class CaseStudyPlanError(CaseStudyRuntimeError):
    """The registered runtime policy or compiled plan is inconsistent."""


class CaseStudyResumeError(CaseStudyRuntimeError):
    """A resume manifest is not an append-only continuation of its plan."""


TERMINAL_OUTCOMES = frozenset(
    {
        RunOutcome.SUCCEEDED,
        RunOutcome.INVALID,
        RunOutcome.FAILED,
        RunOutcome.TIMED_OUT,
        RunOutcome.INTERRUPTED,
    }
)
RUNTIME_POLICY_RELATIVE_PATH = "configs/case_study/runtime.json"


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _safe_relative_path(value: str, *, label: str) -> None:
    path = PurePosixPath(value)
    if (
        not value
        or value.strip() != value
        or path.is_absolute()
        or ".." in path.parts
        or "." in path.parts
        or "\\" in value
    ):
        raise ValueError(f"{label} must be a normalized bounded relative path")


def _reject_symlink_ancestry(path: Path, *, label: str) -> Path:
    """Return the absolute lexical path only when no existing ancestor is a link."""

    lexical = Path(os.path.abspath(path))
    probe = Path(lexical.anchor)
    for component in lexical.parts[1:]:
        probe /= component
        if probe.is_symlink():
            raise CaseStudyInputError(f"{label} cannot traverse a symbolic-link ancestor")
    return lexical


def _resolve_exact_restricted_file(path: Path, root: Path, *, label: str) -> Path:
    if not path.is_absolute() or not root.is_absolute():
        raise CaseStudyInputError(f"{label} and restricted root must be explicit")
    if ".." in path.parts:
        raise CaseStudyInputError(f"{label} cannot contain parent traversal")
    lexical_root = _reject_symlink_ancestry(root, label="restricted root")
    lexical_path = _reject_symlink_ancestry(path, label=label)
    try:
        resolved_root = root.resolve(strict=True)
        resolved = path.resolve(strict=True)
    except OSError as error:
        raise CaseStudyInputError(f"{label} is unavailable") from error
    if not resolved.is_file() or not resolved.is_relative_to(resolved_root):
        raise CaseStudyInputError(f"{label} must be a file inside the restricted root")
    try:
        relative = lexical_path.relative_to(lexical_root)
    except ValueError as error:
        raise CaseStudyInputError(f"{label} escapes the restricted root") from error
    cursor = lexical_root
    for part in relative.parts[:-1]:
        cursor /= part
        if cursor.is_symlink():
            raise CaseStudyInputError(f"{label} cannot traverse a symbolic link")
    return resolved


class CaseStudyRuntimePolicy(ImmutableRecord):
    """Tracked policy; exact dependency hashes are frozen into the compiled plan."""

    configuration_id: Literal["first-novel-runtime-policy-v1"]
    model_config_path: str
    selected_model_policy_path: str
    decoding_config_path: str
    gpu_call_inventory_path: str
    seed_manifest_path: str
    indexing_config_path: str
    c0_rules_path: str
    c1_prompt_path: str
    c2_prompt_path: str
    production_adapter_factory: Literal[
        "story_projection_onto.case_study_factory:create_frozen_production_case_study_bundle"
    ]
    primary_seed_purpose: Literal["llm_block_1"] = "llm_block_1"
    seed_block: Literal[1] = 1
    seed_mapping: Literal["low-31-bits-of-frozen-llm-block-1-v1"]
    case_c1_call_count: Literal[4] = 4
    case_c2_call_count: Literal[8] = 8
    case_full_index_c2_call_count: Literal[1] = 1
    case_c1_call_class: Literal["case_c1"] = "case_c1"
    case_c2_call_class: Literal["case_c2"] = "case_c2"
    case_full_index_c2_call_class: Literal["case_full_index_c2"] = "case_full_index_c2"
    c1_watchdog_seconds: Literal[240] = 240
    c2_watchdog_seconds: Literal[150] = 150
    operational_watchdog_seconds: Literal[150] = 150
    service_start_watchdog_seconds: Literal[300] = 300
    c1_repair_reserve_class: Literal["reserve_long"] = "reserve_long"
    c2_repair_reserve_class: Literal["reserve_standard"] = "reserve_standard"
    packet_persistence: Literal["in_memory_with_hash_receipt_only"] = (
        "in_memory_with_hash_receipt_only"
    )
    restricted_artifact_prefix: str = "case-study"

    @model_validator(mode="after")
    def paths_and_call_inventory_are_fixed(self) -> Self:
        for name in (
            "model_config_path",
            "selected_model_policy_path",
            "decoding_config_path",
            "gpu_call_inventory_path",
            "seed_manifest_path",
            "indexing_config_path",
            "c0_rules_path",
            "c1_prompt_path",
            "c2_prompt_path",
            "restricted_artifact_prefix",
        ):
            _safe_relative_path(getattr(self, name), label=name)
        return self

    @classmethod
    def load(cls, path: Path) -> CaseStudyRuntimePolicy:
        return cls.model_validate_json(path.read_text(encoding="utf-8"))


class CaseStudyInputAttestation(ImmutableRecord):
    """Researcher attestation binding the three exact restricted input files."""

    attestation_id: Identifier
    restricted_index_sha256: Sha256Digest
    restricted_index_manifest_file_sha256: Sha256Digest
    restricted_index_manifest_hash: Sha256Digest
    preregistration_file_sha256: Sha256Digest
    preregistration_hash: Sha256Digest
    lawful_copy_attested: Literal[True] = True
    exact_index_attested: Literal[True] = True
    windows_selected_before_condition_outputs: Literal[True] = True
    condition_outputs_inspected: Literal[False] = False
    attested_by: str = Field(min_length=1)
    attested_at: AwareDatetime
    release_class: Literal[ReleaseClass.RESTRICTED] = ReleaseClass.RESTRICTED


class CaseStudyAdmissionAttestation(ImmutableRecord):
    """Researcher signature over the typed, cross-linked pre-case gate bundle."""

    admission_id: Identifier
    synthetic_run_closure_hash: Sha256Digest
    timing_lineage_audit_hash: Sha256Digest
    gold_firewall_audit_hash: Sha256Digest
    registered_metric_regeneration_hash: Sha256Digest
    blinded_error_review_hash: Sha256Digest
    storage_preflight_hash: Sha256Digest
    gpu_schedule_admission_hash: Sha256Digest
    public_release_scan_hash: Sha256Digest
    semantic_gate_bundle_hash: Sha256Digest
    held_out_results_closure_hash: Sha256Digest
    cumulative_ledger_sha256: Sha256Digest
    gpu_event_inventory_hash: Sha256Digest
    selected_model_freeze_file_sha256: Sha256Digest
    selected_model_freeze_hash: Sha256Digest
    upper_ontology_hash: Sha256Digest
    validator_hash: Sha256Digest
    c1_capability_manifest_hash: Sha256Digest
    c2_capability_manifest_hash: Sha256Digest
    synthetic_run_closed: Literal[True] = True
    timing_lineage_audit_passed: Literal[True] = True
    gold_firewall_audit_passed: Literal[True] = True
    registered_metrics_regenerated: Literal[True] = True
    blinded_error_review_complete: Literal[True] = True
    storage_headroom_passed: Literal[True] = True
    gpu_schedule_admitted: Literal[True] = True
    public_release_scan_passed: Literal[True] = True
    repair_reserve_admitted: Literal[True] = True
    attested_by: str = Field(min_length=1)
    attested_at: AwareDatetime
    release_class: Literal[ReleaseClass.RESTRICTED] = ReleaseClass.RESTRICTED


class CaseSyntheticClosureGate(ImmutableRecord):
    """Semantic proof that the registered synthetic execution is terminal."""

    gate_kind: Literal["synthetic_run_closure"] = "synthetic_run_closure"
    study_id: Identifier
    source_tree_sha256: Sha256Digest
    selected_model_freeze_hash: Sha256Digest
    held_out_execution_plan_hash: Sha256Digest
    held_out_results_closure_hash: Sha256Digest
    synthetic_benchmark_hash: Sha256Digest
    output_receipt_inventory_hash: Sha256Digest
    held_out_world_count: Literal[12] = 12
    held_out_context_count: Literal[36] = 36
    primary_conditions: tuple[
        Literal[ConditionName.C0_CLASSICAL_PRE],
        Literal[ConditionName.C1_LLM_PRE],
        Literal[ConditionName.C2_LLM_QUERY],
        Literal[ConditionName.A_FIXED_SELECT],
    ] = (
        ConditionName.C0_CLASSICAL_PRE,
        ConditionName.C1_LLM_PRE,
        ConditionName.C2_LLM_QUERY,
        ConditionName.A_FIXED_SELECT,
    )
    llm_seed_count: Literal[2] = 2
    all_registered_outputs_terminal: Literal[True] = True
    intention_to_treat_complete: Literal[True] = True
    completed_at: AwareDatetime
    release_class: Literal[ReleaseClass.RESTRICTED] = ReleaseClass.RESTRICTED


class CaseTimingLineageGate(ImmutableRecord):
    """Semantic proof that recovered GPU allocation has one closed lineage."""

    gate_kind: Literal["timing_lineage_audit"] = "timing_lineage_audit"
    study_id: Identifier
    source_tree_sha256: Sha256Digest
    selected_model_freeze_hash: Sha256Digest
    held_out_results_closure_hash: Sha256Digest
    cumulative_ledger_sha256: Sha256Digest
    gpu_event_inventory_hash: Sha256Digest
    actual_allocated_gpu_seconds: float = Field(ge=0.0, lt=36_000.0)
    unresolved_allocation_count: Literal[0] = 0
    service_intervals_reconciled: Literal[True] = True
    failures_and_repairs_included: Literal[True] = True
    completed_at: AwareDatetime
    release_class: Literal[ReleaseClass.RESTRICTED] = ReleaseClass.RESTRICTED


class CaseGoldFirewallGate(ImmutableRecord):
    """Semantic proof that scoring truth stayed outside every model-visible input."""

    gate_kind: Literal["gold_firewall_audit"] = "gold_firewall_audit"
    study_id: Identifier
    source_tree_sha256: Sha256Digest
    selected_model_freeze_hash: Sha256Digest
    held_out_execution_plan_hash: Sha256Digest
    held_out_results_closure_hash: Sha256Digest
    synthetic_benchmark_hash: Sha256Digest
    model_visible_payload_inventory_hash: Sha256Digest
    scorer_only_inventory_hash: Sha256Digest
    gold_firewall_passed: Literal[True] = True
    equal_evidence_passed: Literal[True] = True
    construction_timing_passed: Literal[True] = True
    horizon_equality_passed: Literal[True] = True
    completed_at: AwareDatetime
    release_class: Literal[ReleaseClass.RESTRICTED] = ReleaseClass.RESTRICTED


class CaseMetricRegenerationGate(ImmutableRecord):
    """Semantic proof that all registered synthetic metrics were regenerated."""

    gate_kind: Literal["registered_metric_regeneration"] = (
        "registered_metric_regeneration"
    )
    study_id: Identifier
    source_tree_sha256: Sha256Digest
    selected_model_freeze_hash: Sha256Digest
    held_out_results_closure_hash: Sha256Digest
    synthetic_benchmark_hash: Sha256Digest
    registered_analysis_manifest_hash: Sha256Digest
    canonical_table_manifest_hash: Sha256Digest
    independent_world_count: Literal[12] = 12
    endpoint_families: tuple[
        Literal["fidelity"],
        Literal["rare_pivotal"],
        Literal["entropy_clutter"],
        Literal["community"],
        Literal["paraphrase_contrastive"],
        Literal["ablations"],
        Literal["feedback"],
    ] = (
        "fidelity",
        "rare_pivotal",
        "entropy_clutter",
        "community",
        "paraphrase_contrastive",
        "ablations",
        "feedback",
    )
    all_registered_metrics_regenerated: Literal[True] = True
    contexts_or_seeds_used_as_independent_units: Literal[False] = False
    completed_at: AwareDatetime
    release_class: Literal[ReleaseClass.RESTRICTED] = ReleaseClass.RESTRICTED


class CaseBlindedErrorReviewGate(ImmutableRecord):
    """Semantic proof that the post-run condition-blind error review closed."""

    gate_kind: Literal["blinded_error_review"] = "blinded_error_review"
    study_id: Identifier
    source_tree_sha256: Sha256Digest
    selected_model_freeze_hash: Sha256Digest
    held_out_results_closure_hash: Sha256Digest
    output_receipt_inventory_hash: Sha256Digest
    frozen_selection_manifest_hash: Sha256Digest
    condition_alias_manifest_hash: Sha256Digest
    completed_response_hash: Sha256Digest
    adjudication_hash: Sha256Digest
    reviewed_unit_count: int = Field(gt=0)
    condition_labels_concealed: Literal[True] = True
    review_complete: Literal[True] = True
    completed_at: AwareDatetime
    release_class: Literal[ReleaseClass.RESTRICTED] = ReleaseClass.RESTRICTED


class CaseStoragePreflightGate(ImmutableRecord):
    """Semantic proof that the registered writable-storage envelope remains safe."""

    gate_kind: Literal["storage_preflight"] = "storage_preflight"
    study_id: Identifier
    source_tree_sha256: Sha256Digest
    selected_model_freeze_hash: Sha256Digest
    cumulative_ledger_sha256: Sha256Digest
    gpu_event_inventory_hash: Sha256Digest
    occupied_bytes: int = Field(ge=0, le=25_000_000_000)
    projected_occupied_bytes_after_case: int = Field(ge=0, le=25_000_000_000)
    filesystem_free_bytes: int = Field(ge=5_000_000_000)
    controlled_allocation_bytes: Literal[30_000_000_000] = 30_000_000_000
    headroom_passed: Literal[True] = True
    completed_at: AwareDatetime
    release_class: Literal[ReleaseClass.RESTRICTED] = ReleaseClass.RESTRICTED


class CaseGpuScheduleGate(ImmutableRecord):
    """Semantic proof that the still-mandatory case block fits both GPU caps."""

    gate_kind: Literal["gpu_schedule_admission"] = "gpu_schedule_admission"
    study_id: Identifier
    source_tree_sha256: Sha256Digest
    selected_model_freeze_hash: Sha256Digest
    cumulative_ledger_sha256: Sha256Digest
    gpu_event_inventory_hash: Sha256Digest
    actual_allocated_gpu_seconds: float = Field(ge=0.0, lt=36_000.0)
    case_service_start_watchdog_seconds: Literal[300] = 300
    case_base_watchdog_seconds: Literal[2310] = 2310
    required_next_repair_seconds: Literal[240] = 240
    projected_scheduled_gpu_seconds: float = Field(ge=0.0, le=32_400.0)
    projected_hard_gpu_seconds: float = Field(ge=0.0, lt=36_000.0)
    scheduled_limit_seconds: Literal[32400] = 32_400
    hard_limit_seconds: Literal[36000] = 36_000
    admitted: Literal[True] = True
    completed_at: AwareDatetime
    release_class: Literal[ReleaseClass.RESTRICTED] = ReleaseClass.RESTRICTED

    @model_validator(mode="after")
    def forecast_contains_the_exact_case_envelope(self) -> Self:
        minimum = (
            self.actual_allocated_gpu_seconds
            + self.case_service_start_watchdog_seconds
            + self.case_base_watchdog_seconds
            + self.required_next_repair_seconds
        )
        if self.projected_scheduled_gpu_seconds + 1e-6 < minimum:
            raise ValueError("GPU admission forecast omits part of the case envelope")
        if abs(self.projected_hard_gpu_seconds - self.projected_scheduled_gpu_seconds) > 1e-6:
            raise ValueError("scheduled and hard case forecasts must use the same inventory")
        return self


class CasePublicReleaseScanGate(ImmutableRecord):
    """Semantic proof that the pre-case public candidate is non-reconstructive."""

    gate_kind: Literal["public_release_scan"] = "public_release_scan"
    study_id: Identifier
    source_tree_sha256: Sha256Digest
    selected_model_freeze_hash: Sha256Digest
    canonical_table_manifest_hash: Sha256Digest
    public_candidate_manifest_hash: Sha256Digest
    release_scan_receipt_hash: Sha256Digest
    protected_prose_hit_count: Literal[0] = 0
    reconstructive_offset_hit_count: Literal[0] = 0
    private_path_hit_count: Literal[0] = 0
    restricted_artifact_hit_count: Literal[0] = 0
    scan_passed: Literal[True] = True
    completed_at: AwareDatetime
    release_class: Literal[ReleaseClass.RESTRICTED] = ReleaseClass.RESTRICTED


CASE_SEMANTIC_NATIVE_ARTIFACT_ROLES = frozenset(
    {
        "source_association",
        "selected_model_freeze",
        "synthetic_benchmark_manifest",
        "synthetic_benchmark_config",
        "held_out_results_gate",
        "held_out_call_manifest",
        "held_out_execution_manifest",
        "held_out_scorer_bridge",
        "phase4_analysis_index",
        "phase4_table_manifest",
        "combined_execution_index",
        "combined_call_manifest",
        "independent_review_completion",
        "phase5_journal_index",
        "phase5_source_manifest",
        "phase5_known_answer_source",
        "phase5_feedback_scoring_session",
        "phase5_feedback_metrics",
        "phase5_feedback_scoring_receipt",
        "blinded_error_taxonomy",
        "blinded_error_source_manifest",
        "blinded_error_package",
        "blinded_error_rejoin",
        "blinded_error_completion",
        "blinded_error_adjudication",
        "blinded_error_finalization",
        "blinded_error_table",
        "blinded_community_rubric_template",
        "blinded_community_source_manifest",
        "blinded_community_package",
        "blinded_community_rejoin",
        "blinded_community_completion",
        "blinded_community_finalization",
        "blinded_community_table",
        "gold_original_labels",
        "gold_mutated_labels",
        "gold_original_artifacts",
        "gold_mutated_artifacts",
        "gold_firewall_audit",
        "public_release_allowlist",
        "gpu_call_inventory",
        "resource_limits",
        "storage_allocation_plan",
        "cumulative_ledger",
    }
)


class CaseNativeGateArtifactReference(ImmutableRecord):
    """Resolvable, byte-bound input consumed by semantic gate replay."""

    role: Identifier
    relative_path: str = Field(min_length=1, max_length=500)
    file_sha256: Sha256Digest
    logical_content_hash: Sha256Digest
    release_class: Literal[ReleaseClass.RESTRICTED] = ReleaseClass.RESTRICTED

    @model_validator(mode="after")
    def path_is_bounded_and_portable(self) -> Self:
        _safe_relative_path(self.relative_path, label="native gate artifact path")
        return self


class CaseStudySemanticAdmissionBundle(ImmutableRecord):
    """Eight typed gates with exact shared scientific and accounting lineage."""

    bundle_id: Identifier
    synthetic_run_closure: CaseSyntheticClosureGate
    timing_lineage_audit: CaseTimingLineageGate
    gold_firewall_audit: CaseGoldFirewallGate
    registered_metric_regeneration: CaseMetricRegenerationGate
    blinded_error_review: CaseBlindedErrorReviewGate
    storage_preflight: CaseStoragePreflightGate
    gpu_schedule_admission: CaseGpuScheduleGate
    public_release_scan: CasePublicReleaseScanGate
    native_artifacts: tuple[CaseNativeGateArtifactReference, ...]
    frozen_at: AwareDatetime
    release_class: Literal[ReleaseClass.RESTRICTED] = ReleaseClass.RESTRICTED

    @model_validator(mode="after")
    def semantic_lineage_is_closed(self) -> Self:
        gates = (
            self.synthetic_run_closure,
            self.timing_lineage_audit,
            self.gold_firewall_audit,
            self.registered_metric_regeneration,
            self.blinded_error_review,
            self.storage_preflight,
            self.gpu_schedule_admission,
            self.public_release_scan,
        )
        roles = tuple(item.role for item in self.native_artifacts)
        if (
            set(roles) != CASE_SEMANTIC_NATIVE_ARTIFACT_ROLES
            or len(roles) != len(CASE_SEMANTIC_NATIVE_ARTIFACT_ROLES)
        ):
            raise ValueError("semantic admission lacks the exact native replay inventory")
        if len({item.study_id for item in gates}) != 1:
            raise ValueError("case admission gates name different studies")
        if len({item.selected_model_freeze_hash for item in gates}) != 1:
            raise ValueError("case admission gates name different selected models")
        closure = self.synthetic_run_closure
        if {
            self.timing_lineage_audit.held_out_results_closure_hash,
            self.gold_firewall_audit.held_out_results_closure_hash,
            self.registered_metric_regeneration.held_out_results_closure_hash,
            self.blinded_error_review.held_out_results_closure_hash,
        } != {closure.held_out_results_closure_hash}:
            raise ValueError("case admission gates do not share the held-out closure")
        if self.blinded_error_review.output_receipt_inventory_hash != (
            closure.output_receipt_inventory_hash
        ):
            raise ValueError("blinded review covers another output inventory")
        if {
            self.gold_firewall_audit.held_out_execution_plan_hash,
            closure.held_out_execution_plan_hash,
        } != {closure.held_out_execution_plan_hash}:
            raise ValueError("gold audit covers another held-out execution plan")
        if {
            self.gold_firewall_audit.synthetic_benchmark_hash,
            self.registered_metric_regeneration.synthetic_benchmark_hash,
            closure.synthetic_benchmark_hash,
        } != {closure.synthetic_benchmark_hash}:
            raise ValueError("case admission gates do not share the benchmark")
        timing = self.timing_lineage_audit
        accounting = (self.storage_preflight, self.gpu_schedule_admission)
        if any(
            item.cumulative_ledger_sha256 != timing.cumulative_ledger_sha256
            or item.gpu_event_inventory_hash != timing.gpu_event_inventory_hash
            for item in accounting
        ):
            raise ValueError("case admission resource gates use different ledger snapshots")
        if abs(
            self.gpu_schedule_admission.actual_allocated_gpu_seconds
            - timing.actual_allocated_gpu_seconds
        ) > 1e-6:
            raise ValueError("case GPU admission uses a different allocated-time total")
        if self.public_release_scan.canonical_table_manifest_hash != (
            self.registered_metric_regeneration.canonical_table_manifest_hash
        ):
            raise ValueError("release scan does not cover the regenerated table lineage")
        if any(item.completed_at > self.frozen_at for item in gates):
            raise ValueError("semantic admission bundle predates a constituent gate")
        return self


def validate_case_study_semantic_admission(
    admission: CaseStudyAdmissionAttestation,
    bundle: CaseStudySemanticAdmissionBundle,
) -> None:
    """Bind the signed attestation to each typed gate and shared lineage."""

    expected_hashes = {
        "synthetic_run_closure_hash": bundle.synthetic_run_closure.content_hash,
        "timing_lineage_audit_hash": bundle.timing_lineage_audit.content_hash,
        "gold_firewall_audit_hash": bundle.gold_firewall_audit.content_hash,
        "registered_metric_regeneration_hash": (
            bundle.registered_metric_regeneration.content_hash
        ),
        "blinded_error_review_hash": bundle.blinded_error_review.content_hash,
        "storage_preflight_hash": bundle.storage_preflight.content_hash,
        "gpu_schedule_admission_hash": bundle.gpu_schedule_admission.content_hash,
        "public_release_scan_hash": bundle.public_release_scan.content_hash,
    }
    mismatches = tuple(
        name for name, expected in expected_hashes.items() if getattr(admission, name) != expected
    )
    if mismatches:
        raise CaseStudyInputError(
            "semantic gate hashes differ from admission: " + ", ".join(mismatches)
        )
    closure = bundle.synthetic_run_closure
    timing = bundle.timing_lineage_audit
    if (
        admission.semantic_gate_bundle_hash != bundle.content_hash
        or admission.held_out_results_closure_hash != closure.held_out_results_closure_hash
        or admission.cumulative_ledger_sha256 != timing.cumulative_ledger_sha256
        or admission.gpu_event_inventory_hash != timing.gpu_event_inventory_hash
        or admission.selected_model_freeze_hash != closure.selected_model_freeze_hash
    ):
        raise CaseStudyInputError("semantic admission cross-lineage differs from attestation")
    if admission.attested_at < bundle.frozen_at:
        raise CaseStudyInputError("case admission attestation predates its semantic gates")


@dataclass(frozen=True, slots=True)
class AttestedRestrictedCaseStudy:
    """Loaded restricted records; paths are ephemeral and hidden from repr."""

    index_path: Path = field(repr=False)
    manifest_path: Path = field(repr=False)
    preregistration_path: Path = field(repr=False)
    attestation_path: Path = field(repr=False)
    manifest: RestrictedNovelIndexManifest
    preregistration: CaseStudyPreregistration
    attestation: CaseStudyInputAttestation
    index_manifest_file_sha256: str
    preregistration_file_sha256: str
    attestation_file_sha256: str


@dataclass(frozen=True, slots=True)
class AttestedSelectedModelFreeze:
    """Verified selected-model payload loaded from one explicitly named file."""

    payload: Mapping[str, object] = field(repr=False)
    file_sha256: str
    manifest_sha256: str
    model_candidate: str
    repository: str
    revision: str


def load_attested_selected_model_freeze(
    *,
    restricted_root: Path,
    selected_model_freeze_path: Path,
    admission: CaseStudyAdmissionAttestation,
) -> AttestedSelectedModelFreeze:
    """Verify a materialized accepted fallback freeze, not a bare claimed hash."""

    resolved = _resolve_exact_restricted_file(
        selected_model_freeze_path,
        restricted_root,
        label="selected-model freeze",
    )
    raw = resolved.read_bytes()
    try:
        outer = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise CaseStudyInputError("selected-model freeze is not valid JSON") from error
    if not isinstance(outer, Mapping):
        raise CaseStudyInputError("selected-model freeze file must contain an object")
    nested = outer.get("selected_model_freeze")
    payload = nested if isinstance(nested, Mapping) else outer
    immutable = {key: value for key, value in payload.items() if key != "manifest_sha256"}
    observed_manifest_hash = canonical_sha256(immutable)
    observed_file_hash = _sha256_bytes(raw)
    if (
        payload.get("manifest_sha256") != observed_manifest_hash
        or admission.selected_model_freeze_hash != observed_manifest_hash
        or admission.selected_model_freeze_file_sha256 != observed_file_hash
    ):
        raise CaseStudyInputError("selected-model freeze content/file hash differs from admission")
    required_hash_fields = (
        "activation_certificate_sha256",
        "cache_replacement_receipt_sha256",
        "snapshot_manifest_sha256",
        "launcher_configuration_sha256",
        "tokenizer_manifest_sha256",
        "micro_pilot_acceptance_receipt_sha256",
        "accepted_micro_pilot_result_sha256",
        "accepted_request_family_hash",
        "accepted_operator_gate_sha256",
        "accepted_grounding_horizon_gate_sha256",
        "runtime_stack_manifest_sha256",
        "gpu_hardware_manifest_sha256",
        "source_association_manifest_sha256",
        "source_tree_sha256",
    )
    if any(
        not isinstance(payload.get(name), str)
        or len(str(payload[name])) != 64
        or any(character not in "0123456789abcdef" for character in str(payload[name]))
        for name in required_hash_fields
    ):
        raise CaseStudyInputError("selected-model freeze lacks accepted runtime/gate lineage")
    runtime_setting = payload.get("runtime_feasibility_setting")
    if (
        payload.get("kind") != "selected_llm_model_freeze"
        or payload.get("model_candidate") != "fallback"
        or payload.get("mixed_model_candidates_forbidden") is not True
        or payload.get("held_out_requires_separate_development_and_review_gates") is not True
        or payload.get("applies_symmetrically_to") != ["C1", "C2", "A-FixedSelect", "LLM_ablations"]
        or not isinstance(runtime_setting, Mapping)
        or runtime_setting.get("enforce_eager") is not True
    ):
        raise CaseStudyInputError(
            "selected-model freeze is not the accepted symmetric fallback freeze"
        )
    repository = payload.get("repository")
    revision = payload.get("revision")
    if (
        not isinstance(repository, str)
        or not isinstance(revision, str)
        or len(revision) != 40
        or any(character not in "0123456789abcdef" for character in revision)
    ):
        raise CaseStudyInputError("selected-model freeze lacks immutable model identity")
    return AttestedSelectedModelFreeze(
        payload=dict(payload),
        file_sha256=observed_file_hash,
        manifest_sha256=observed_manifest_hash,
        model_candidate="fallback",
        repository=repository,
        revision=revision,
    )


def load_attested_restricted_case_study(
    *,
    restricted_root: Path,
    index_path: Path,
    manifest_path: Path,
    preregistration_path: Path,
    attestation_path: Path,
) -> AttestedRestrictedCaseStudy:
    """Load only explicitly named, hash-attested files inside one restricted root."""

    index = _resolve_exact_restricted_file(index_path, restricted_root, label="index")
    manifest_file = _resolve_exact_restricted_file(
        manifest_path, restricted_root, label="index manifest"
    )
    preregistration_file = _resolve_exact_restricted_file(
        preregistration_path, restricted_root, label="preregistration"
    )
    attestation_file = _resolve_exact_restricted_file(
        attestation_path, restricted_root, label="input attestation"
    )
    manifest_bytes = manifest_file.read_bytes()
    preregistration_bytes = preregistration_file.read_bytes()
    attestation_bytes = attestation_file.read_bytes()
    manifest = RestrictedNovelIndexManifest.model_validate_json(manifest_bytes)
    preregistration = CaseStudyPreregistration.model_validate_json(preregistration_bytes)
    attestation = CaseStudyInputAttestation.model_validate_json(attestation_bytes)
    observed = {
        "restricted_index_sha256": _sha256_file(index),
        "restricted_index_manifest_file_sha256": _sha256_bytes(manifest_bytes),
        "restricted_index_manifest_hash": manifest.content_hash,
        "preregistration_file_sha256": _sha256_bytes(preregistration_bytes),
        "preregistration_hash": preregistration.content_hash,
    }
    expected = {name: getattr(attestation, name) for name in observed}
    if observed != expected:
        mismatches = ", ".join(name for name in observed if observed[name] != expected[name])
        raise CaseStudyInputError(f"restricted case-study attestation mismatch: {mismatches}")
    if attestation.attested_at < preregistration.frozen_at:
        raise CaseStudyInputError("input attestation predates the frozen preregistration")
    validate_case_preregistration(preregistration, manifest, index_path=index)
    return AttestedRestrictedCaseStudy(
        index_path=index,
        manifest_path=manifest_file,
        preregistration_path=preregistration_file,
        attestation_path=attestation_file,
        manifest=manifest,
        preregistration=preregistration,
        attestation=attestation,
        index_manifest_file_sha256=observed["restricted_index_manifest_file_sha256"],
        preregistration_file_sha256=observed["preregistration_file_sha256"],
        attestation_file_sha256=_sha256_bytes(attestation_bytes),
    )


class CaseGpuCallRole(StrEnum):
    C1_WINDOW_PRECONSTRUCTION = "c1_window_preconstruction"
    C2_BOUNDED_CONSTRUCTION = "c2_bounded_construction"
    C2_FULL_INDEX_OPERATIONAL = "c2_full_index_operational"


class CaseModelRuntimeBinding(ImmutableRecord):
    """Exact model/runtime/configuration identity shared by all case LLM calls."""

    backend: Literal["vllm_gpu"] = "vllm_gpu"
    model_repository: Identifier
    model_revision: str = Field(pattern=r"^[0-9a-f]{40}$")
    served_model_name: Identifier
    model_config_file_sha256: Sha256Digest
    model_config_hash: Sha256Digest
    selected_model_policy_file_sha256: Sha256Digest
    selected_model_policy_hash: Sha256Digest
    selected_model_candidate: Literal["fallback"] = "fallback"
    selected_model_freeze_file_sha256: Sha256Digest
    selected_model_freeze_hash: Sha256Digest
    selected_snapshot_manifest_hash: Sha256Digest
    selected_launcher_configuration_hash: Sha256Digest
    selected_tokenizer_manifest_hash: Sha256Digest
    decoding_config_file_sha256: Sha256Digest
    decoding_config_hash: Sha256Digest
    gpu_call_inventory_file_sha256: Sha256Digest
    gpu_call_inventory_hash: Sha256Digest
    seed_manifest_file_sha256: Sha256Digest
    seed_manifest_hash: Sha256Digest
    seed_entry_hash: Sha256Digest
    frozen_llm_seed: int = Field(ge=0, lt=2**63)
    resolved_vllm_seed: int = Field(ge=0, le=2_147_483_647)
    seed_block: Literal[1] = 1
    seed_mapping: Literal["low-31-bits-of-frozen-llm-block-1-v1"]
    c1_prompt_file_sha256: Sha256Digest
    c2_prompt_file_sha256: Sha256Digest
    output_schema_hash: Sha256Digest
    c1_capability_manifest_hash: Sha256Digest
    c2_capability_manifest_hash: Sha256Digest
    validator_hash: Sha256Digest
    upper_ontology_hash: Sha256Digest
    maximum_input_tokens: Literal[10240] = 10240
    maximum_output_tokens: Literal[2048] = 2048
    repair_maximum_input_tokens: Literal[10752] = 10752
    repair_maximum_output_tokens: Literal[1536] = 1536
    repair_attempt_budget: Literal[1] = 1

    @model_validator(mode="after")
    def seed_is_exact_mapping(self) -> Self:
        if self.resolved_vllm_seed != (self.frozen_llm_seed & (2**31 - 1)):
            raise ValueError("case vLLM seed differs from the frozen low-31-bit mapping")
        return self


class CaseClassicalRuntimeBinding(ImmutableRecord):
    """Credible query-blind C0 backend/configuration identity."""

    backend: Literal["spacy-ner-dependency-plus-deterministic-rules-v2"]
    config_file_sha256: Sha256Digest
    config_hash: Sha256Digest
    query_blind_preconstruction: Literal[True] = True
    deterministic_fixed_projection: Literal[True] = True
    seed: Literal[None] = None


class CaseGpuCallSlot(ImmutableRecord):
    call_id: Identifier
    ordinal: int = Field(ge=1, le=13)
    call_class: Literal["case_c1", "case_c2", "case_full_index_c2"]
    role: CaseGpuCallRole
    condition: ConditionName
    window_id: Identifier | None = None
    context_id: Identifier | None = None
    runtime_binding_hash: Sha256Digest
    seed_block: Literal[1] = 1
    resolved_vllm_seed: int = Field(ge=0, le=2_147_483_647)
    admission_p95_seconds: Literal[150, 240]
    watchdog_seconds: Literal[150, 240]
    repair_reserve_class: Literal["reserve_long", "reserve_standard"]
    operational_only: bool
    causal_comparison_eligible: bool
    query_blind: bool
    backend: Literal["vllm_gpu"] = "vllm_gpu"

    @model_validator(mode="after")
    def registered_role_shape(self) -> Self:
        if self.role is CaseGpuCallRole.C1_WINDOW_PRECONSTRUCTION:
            expected = (
                "case_c1",
                ConditionName.C1_LLM_PRE,
                None,
                240,
                240,
                "reserve_long",
                False,
                True,
            )
        elif self.role is CaseGpuCallRole.C2_BOUNDED_CONSTRUCTION:
            expected = (
                "case_c2",
                ConditionName.C2_LLM_QUERY,
                self.context_id,
                150,
                150,
                "reserve_standard",
                False,
                False,
            )
        else:
            expected = (
                "case_full_index_c2",
                ConditionName.C2_LLM_QUERY,
                self.context_id,
                150,
                150,
                "reserve_standard",
                True,
                False,
            )
        actual = (
            self.call_class,
            self.condition,
            self.context_id,
            self.admission_p95_seconds,
            self.watchdog_seconds,
            self.repair_reserve_class,
            self.operational_only,
            self.query_blind,
        )
        if actual != expected:
            raise ValueError("case GPU call differs from its registered role")
        if self.window_id is None:
            raise ValueError("every case GPU call must identify its bounded window or index")
        if self.role is CaseGpuCallRole.C1_WINDOW_PRECONSTRUCTION:
            if self.causal_comparison_eligible is not True:
                raise ValueError("bounded C1 preconstruction is part of the clean comparison")
        elif self.context_id is None:
            raise ValueError("C2 calls require exactly one context")
        if self.operational_only == self.causal_comparison_eligible:
            raise ValueError("operational and causal-comparison flags must be opposites")
        return self


class CaseWindowExecutionPlan(ImmutableRecord):
    window_id: Identifier
    window_registration_hash: Sha256Digest
    horizon_hash: Sha256Digest
    context_ids: tuple[Identifier, Identifier]
    context_hashes: tuple[Sha256Digest, Sha256Digest]
    registered_revealed_at: tuple[AwareDatetime, AwareDatetime]
    evidence_binding_hash: Sha256Digest
    packet_equality_group_id: Identifier
    c0_preparation_job_id: Identifier
    c1_preconstruction_call_id: Identifier
    c2_empty_inventory_job_id: Identifier
    c0_projection_job_ids: tuple[Identifier, Identifier]
    c1_projection_job_ids: tuple[Identifier, Identifier]
    c2_construction_call_ids: tuple[Identifier, Identifier]

    @model_validator(mode="after")
    def paired_contexts_and_jobs_are_distinct(self) -> Self:
        if len(set(self.context_ids)) != 2 or len(set(self.context_hashes)) != 2:
            raise ValueError("each case window requires two distinct contrastive contexts")
        identifiers = (
            self.c0_preparation_job_id,
            self.c1_preconstruction_call_id,
            self.c2_empty_inventory_job_id,
            *self.c0_projection_job_ids,
            *self.c1_projection_job_ids,
            *self.c2_construction_call_ids,
        )
        if len(identifiers) != len(set(identifiers)):
            raise ValueError("case window execution job IDs must be unique")
        return self


class OperationalCaseExecutionPlan(ImmutableRecord):
    operational_query_id: Identifier
    registration_hash: Sha256Digest
    context_id: Identifier
    context_hash: Sha256Digest
    registered_revealed_at: AwareDatetime
    horizon_hash: Sha256Digest
    full_index_binding_hash: Sha256Digest
    c2_empty_inventory_job_id: Identifier
    retrieval_job_id: Identifier
    c2_call_id: Identifier
    retrieval_method: Literal[RetrievalMethod.SQLITE_FTS5_BM25] = RetrievalMethod.SQLITE_FTS5_BM25
    top_k: int = Field(gt=0)
    max_evidence_tokens: int = Field(gt=0)
    full_index_scope: Literal[True] = True
    operational_only: Literal[True] = True
    causal_comparison_eligible: Literal[False] = False


class RestrictedArtifactStoragePlan(ImmutableRecord):
    restricted_relative_prefix: str
    raw_source_copy_count: Literal[0] = 0
    normalized_source_copy_count: Literal[0] = 0
    bounded_packet_storage: Literal["in_memory_with_hash_receipt_only"]
    raw_attempt_storage: Literal["content_addressed_compressed_restricted"] = (
        "content_addressed_compressed_restricted"
    )
    projection_storage: Literal["content_addressed_compressed_restricted"] = (
        "content_addressed_compressed_restricted"
    )
    resume_manifest_storage: Literal["append_only_hash_named_restricted"] = (
        "append_only_hash_named_restricted"
    )
    public_output_policy: Literal["opaque_ids_hashes_counts_high_level_paraphrase_only"] = (
        "opaque_ids_hashes_counts_high_level_paraphrase_only"
    )
    public_prohibited_payloads: tuple[
        Literal[
            "novel_text",
            "passage_text",
            "raw_packet",
            "detailed_offsets",
            "fts_index",
            "private_paths",
        ],
        ...,
    ] = (
        "novel_text",
        "passage_text",
        "raw_packet",
        "detailed_offsets",
        "fts_index",
        "private_paths",
    )
    release_class: Literal[ReleaseClass.RESTRICTED] = ReleaseClass.RESTRICTED

    @model_validator(mode="after")
    def restricted_prefix_is_bounded(self) -> Self:
        _safe_relative_path(
            self.restricted_relative_prefix,
            label="restricted_relative_prefix",
        )
        if len(self.public_prohibited_payloads) != len(set(self.public_prohibited_payloads)):
            raise ValueError("public prohibited payload inventory must be unique")
        return self


class CaseStudyExecutionPlan(ImmutableRecord):
    """Path-free, exact 4/8/1 production plan compiled before any case output."""

    execution_id: Identifier
    runtime_policy_file_sha256: Sha256Digest
    runtime_policy_hash: Sha256Digest
    input_attestation_hash: Sha256Digest
    input_attestation_file_sha256: Sha256Digest
    admission_attestation_hash: Sha256Digest
    semantic_gate_bundle_hash: Sha256Digest
    restricted_index_manifest_hash: Sha256Digest
    restricted_index_manifest_file_sha256: Sha256Digest
    preregistration_hash: Sha256Digest
    preregistration_file_sha256: Sha256Digest
    indexing_config_file_sha256: Sha256Digest
    indexing_config_hash: Sha256Digest
    c0_runtime: CaseClassicalRuntimeBinding
    model_runtime: CaseModelRuntimeBinding
    windows: tuple[
        CaseWindowExecutionPlan,
        CaseWindowExecutionPlan,
        CaseWindowExecutionPlan,
        CaseWindowExecutionPlan,
    ]
    operational: OperationalCaseExecutionPlan
    gpu_call_slots: tuple[CaseGpuCallSlot, ...]
    detailed_review_context_ids: tuple[Identifier, Identifier, Identifier, Identifier]
    artifact_storage: RestrictedArtifactStoragePlan
    compiled_at: AwareDatetime
    condition_outputs_inspected: Literal[False] = False
    release_class: Literal[ReleaseClass.RESTRICTED] = ReleaseClass.RESTRICTED

    @model_validator(mode="after")
    def registered_design_is_exact(self) -> Self:
        window_ids = tuple(item.window_id for item in self.windows)
        if len(set(window_ids)) != 4:
            raise ValueError("case execution plan requires four distinct windows")
        context_ids = tuple(
            context_id for window in self.windows for context_id in window.context_ids
        )
        if len(context_ids) != 8 or len(set(context_ids)) != 8:
            raise ValueError("case execution plan requires eight distinct bounded contexts")
        if self.operational.context_id in context_ids:
            raise ValueError("full-index operational context must remain separate")
        if (
            set(self.detailed_review_context_ids) - set(context_ids)
            or len(set(self.detailed_review_context_ids)) != 4
        ):
            raise ValueError("detailed matching must select four bounded contexts")
        slots = self.gpu_call_slots
        if len(slots) != 13 or tuple(item.ordinal for item in slots) != tuple(range(1, 14)):
            raise ValueError("case study requires exactly thirteen ordered GPU calls")
        if len({item.call_id for item in slots}) != 13:
            raise ValueError("case GPU call IDs must be unique")
        if any(item.runtime_binding_hash != self.model_runtime.content_hash for item in slots):
            raise ValueError("case GPU call does not bind the exact common model runtime")
        if any(item.resolved_vllm_seed != self.model_runtime.resolved_vllm_seed for item in slots):
            raise ValueError("case GPU calls do not share the registered one seed")
        by_role = {
            role: tuple(item for item in slots if item.role is role) for role in CaseGpuCallRole
        }
        if tuple(len(by_role[role]) for role in CaseGpuCallRole) != (4, 8, 1):
            raise ValueError("case GPU role inventory must be exactly 4 C1, 8 C2, 1 FTS C2")
        expected_call_ids = {
            *(window.c1_preconstruction_call_id for window in self.windows),
            *(call_id for window in self.windows for call_id in window.c2_construction_call_ids),
            self.operational.c2_call_id,
        }
        if {item.call_id for item in slots} != expected_call_ids:
            raise ValueError("case GPU call inventory differs from window/operational jobs")
        equality_groups = tuple(item.packet_equality_group_id for item in self.windows)
        if len(set(equality_groups)) != 4:
            raise ValueError("each bounded window requires one distinct equality group")
        all_job_ids = (
            *(
                identifier
                for window in self.windows
                for identifier in (
                    window.c0_preparation_job_id,
                    window.c1_preconstruction_call_id,
                    window.c2_empty_inventory_job_id,
                    *window.c0_projection_job_ids,
                    *window.c1_projection_job_ids,
                    *window.c2_construction_call_ids,
                )
            ),
            self.operational.c2_empty_inventory_job_id,
            self.operational.retrieval_job_id,
            self.operational.c2_call_id,
        )
        if len(all_job_ids) != len(set(all_job_ids)):
            raise ValueError("case execution plan job IDs must be globally unique")
        return self

    @property
    def bounded_context_ids(self) -> tuple[str, ...]:
        return tuple(context_id for window in self.windows for context_id in window.context_ids)

    @property
    def bounded_projection_job_count(self) -> int:
        return sum(
            len(window.c0_projection_job_ids)
            + len(window.c1_projection_job_ids)
            + len(window.c2_construction_call_ids)
            for window in self.windows
        )


@dataclass(frozen=True, slots=True)
class _TrackedDependency:
    payload: bytes = field(repr=False)
    file_sha256: str
    parsed_json: Mapping[str, object] | None = field(default=None, repr=False)


def _tracked_dependency(repository_root: Path, relative_path: str) -> _TrackedDependency:
    _safe_relative_path(relative_path, label="tracked dependency")
    if not repository_root.is_absolute() or repository_root.is_symlink():
        raise CaseStudyPlanError("repository root must be an explicit non-symlink directory")
    try:
        root = repository_root.resolve(strict=True)
        lexical_candidate = root / relative_path
        if lexical_candidate.is_symlink():
            raise CaseStudyPlanError(
                f"tracked dependency cannot be a symbolic link: {relative_path}"
            )
        cursor = root
        for part in PurePosixPath(relative_path).parts[:-1]:
            cursor /= part
            if cursor.is_symlink():
                raise CaseStudyPlanError(
                    f"tracked dependency cannot traverse a symbolic link: {relative_path}"
                )
        candidate = lexical_candidate.resolve(strict=True)
    except OSError as error:
        raise CaseStudyPlanError(f"tracked dependency is unavailable: {relative_path}") from error
    if not candidate.is_file() or not candidate.is_relative_to(root):
        raise CaseStudyPlanError(f"tracked dependency escapes the repository: {relative_path}")
    payload = candidate.read_bytes()
    parsed: Mapping[str, object] | None = None
    if candidate.suffix == ".json":
        value = json.loads(payload)
        if not isinstance(value, Mapping):
            raise CaseStudyPlanError(f"tracked JSON dependency is not an object: {relative_path}")
        parsed = value
    return _TrackedDependency(
        payload=payload,
        file_sha256=_sha256_bytes(payload),
        parsed_json=parsed,
    )


def _json_dependency(
    repository_root: Path,
    relative_path: str,
) -> tuple[_TrackedDependency, Mapping[str, object]]:
    dependency = _tracked_dependency(repository_root, relative_path)
    if dependency.parsed_json is None:
        raise CaseStudyPlanError(f"expected JSON dependency: {relative_path}")
    return dependency, dependency.parsed_json


def _inventory_class(
    inventory: Mapping[str, object],
    name: str,
) -> Mapping[str, object]:
    classes = inventory.get("classes")
    if not isinstance(classes, Sequence):
        raise CaseStudyPlanError("GPU inventory lacks call classes")
    matches = tuple(
        item for item in classes if isinstance(item, Mapping) and item.get("name") == name
    )
    if len(matches) != 1:
        raise CaseStudyPlanError(f"GPU inventory requires exactly one {name!r} class")
    return matches[0]


def _compile_runtime_bindings(
    *,
    repository_root: Path,
    policy: CaseStudyRuntimePolicy,
    loaded: AttestedRestrictedCaseStudy,
    admission: CaseStudyAdmissionAttestation,
    selected_model_freeze: AttestedSelectedModelFreeze,
) -> tuple[
    CaseModelRuntimeBinding,
    CaseClassicalRuntimeBinding,
    _TrackedDependency,
    NovelSegmentationConfig,
]:
    model_file, model = _json_dependency(repository_root, policy.model_config_path)
    selected_policy_file, selected_policy = _json_dependency(
        repository_root,
        policy.selected_model_policy_path,
    )
    decoding_file, decoding = _json_dependency(repository_root, policy.decoding_config_path)
    inventory_file, inventory = _json_dependency(repository_root, policy.gpu_call_inventory_path)
    seed_file, seed_manifest = _json_dependency(repository_root, policy.seed_manifest_path)
    indexing_file, indexing = _json_dependency(repository_root, policy.indexing_config_path)
    c0_file, c0 = _json_dependency(repository_root, policy.c0_rules_path)
    c1_prompt = _tracked_dependency(repository_root, policy.c1_prompt_path)
    c2_prompt = _tracked_dependency(repository_root, policy.c2_prompt_path)

    if (
        model.get("runtime") != "vllm"
        or model.get("runtime_version") != "0.10.2"
        or model.get("quantization") != "awq"
        or model.get("max_model_len") != 12288
        or model.get("gpu_memory_utilization_pilot") != 0.88
        or model.get("cpu_offload_gb") != 0
        or model.get("thinking_mode") is not False
        or model.get("tensor_parallel_size") != 1
        or model.get("prefix_decoding") is not False
        or model.get("speculative_decoding") is not False
        or model.get("enforce_eager") is not True
        or model.get("request_concurrency") != 1
    ):
        raise CaseStudyPlanError("case model must use the frozen single-GPU vLLM AWQ stack")
    if (
        selected_policy.get("repository") != selected_model_freeze.repository
        or selected_policy.get("revision") != selected_model_freeze.revision
        or selected_policy.get("quantization") != "awq"
        or selected_policy.get("activation") != "only_after_primary_phase1_rejection"
        or selected_policy.get("model_search_allowed") is not False
        or selected_policy.get("maximum_simultaneous_model_snapshots") != 1
        or selected_policy.get("delete_verified_rejected_primary_before_download") is not True
        or selected_policy.get("served_model_name")
        != selected_model_freeze.payload.get("served_model_name")
    ):
        raise CaseStudyPlanError(
            "materialized selected-model freeze differs from the one permitted fallback"
        )
    if selected_model_freeze.manifest_sha256 != admission.selected_model_freeze_hash:
        raise CaseStudyPlanError("selected-model freeze differs from case admission")
    expected_decoding = {
        "first_pass_maximum_input_tokens": 10240,
        "first_pass_maximum_output_tokens": 2048,
        "repair_maximum_input_tokens": 10752,
        "repair_maximum_output_tokens": 1536,
        "n": 1,
        "best_of": 1,
        "beam_search": False,
    }
    if any(decoding.get(name) != value for name, value in expected_decoding.items()):
        raise CaseStudyPlanError("case decoding configuration differs from registered limits")
    entries = seed_manifest.get("entries")
    if not isinstance(entries, Sequence):
        raise CaseStudyPlanError("seed manifest lacks entries")
    seed_entries = tuple(
        item
        for item in entries
        if isinstance(item, Mapping) and item.get("purpose") == policy.primary_seed_purpose
    )
    if len(seed_entries) != 1:
        raise CaseStudyPlanError("seed manifest must contain one primary LLM seed entry")
    seed_entry = seed_entries[0]
    seed_manifest_hash = canonical_sha256(seed_manifest)
    seed_entry_hash = canonical_sha256(seed_entry)
    if (
        seed_manifest.get("content_hash") != seed_manifest_hash
        or seed_entry.get("content_hash") != seed_entry_hash
    ):
        raise CaseStudyPlanError("seed manifest or primary seed entry hash is invalid")
    frozen_seed = seed_entry.get("seed")
    if not isinstance(frozen_seed, int) or isinstance(frozen_seed, bool):
        raise CaseStudyPlanError("primary LLM seed entry is not an integer")
    resolved_seed = frozen_seed & (2**31 - 1)
    if loaded.preregistration.primary_seed != resolved_seed:
        raise CaseStudyPlanError("preregistration primary seed differs from frozen LLM block 1")
    index_config = NovelSegmentationConfig.model_validate(indexing)
    if loaded.manifest.index_config_hash != index_config.content_hash:
        raise CaseStudyPlanError("restricted index used another segmentation configuration")
    if (
        loaded.preregistration.operational_retrieval.top_k != index_config.operational_top_k
        or loaded.preregistration.operational_retrieval.max_evidence_tokens
        != index_config.operational_max_evidence_tokens
    ):
        raise CaseStudyPlanError(
            "operational registration differs from frozen indexing top-k/token limits"
        )
    if c0.get("backend") != "spacy-ner-dependency-plus-deterministic-rules-v2" or (
        c0.get("query_blind") is not True
    ):
        raise CaseStudyPlanError("case C0 must use the credible frozen query-blind backend")
    expected_inventory = {
        policy.case_c1_call_class: (policy.case_c1_call_count, 240),
        policy.case_c2_call_class: (policy.case_c2_call_count, 150),
        policy.case_full_index_c2_call_class: (
            policy.case_full_index_c2_call_count,
            150,
        ),
    }
    for name, (count, p95) in expected_inventory.items():
        row = _inventory_class(inventory, name)
        if row.get("count") != count or row.get("provisional_p95_seconds") != p95:
            raise CaseStudyPlanError(f"GPU inventory class {name!r} differs from the plan")

    model_runtime = CaseModelRuntimeBinding(
        model_repository=selected_model_freeze.repository,
        model_revision=selected_model_freeze.revision,
        served_model_name=str(selected_model_freeze.payload["served_model_name"]),
        model_config_file_sha256=model_file.file_sha256,
        model_config_hash=canonical_sha256(model),
        selected_model_policy_file_sha256=selected_policy_file.file_sha256,
        selected_model_policy_hash=canonical_sha256(selected_policy),
        selected_model_freeze_file_sha256=selected_model_freeze.file_sha256,
        selected_model_freeze_hash=admission.selected_model_freeze_hash,
        selected_snapshot_manifest_hash=str(
            selected_model_freeze.payload["snapshot_manifest_sha256"]
        ),
        selected_launcher_configuration_hash=str(
            selected_model_freeze.payload["launcher_configuration_sha256"]
        ),
        selected_tokenizer_manifest_hash=str(
            selected_model_freeze.payload["tokenizer_manifest_sha256"]
        ),
        decoding_config_file_sha256=decoding_file.file_sha256,
        decoding_config_hash=canonical_sha256(decoding),
        gpu_call_inventory_file_sha256=inventory_file.file_sha256,
        gpu_call_inventory_hash=canonical_sha256(inventory),
        seed_manifest_file_sha256=seed_file.file_sha256,
        seed_manifest_hash=seed_manifest_hash,
        seed_entry_hash=seed_entry_hash,
        frozen_llm_seed=frozen_seed,
        resolved_vllm_seed=resolved_seed,
        seed_mapping=policy.seed_mapping,
        c1_prompt_file_sha256=c1_prompt.file_sha256,
        c2_prompt_file_sha256=c2_prompt.file_sha256,
        output_schema_hash=canonical_sha256(base_condition_output_schema(ConditionName.C1_LLM_PRE)),
        c1_capability_manifest_hash=admission.c1_capability_manifest_hash,
        c2_capability_manifest_hash=admission.c2_capability_manifest_hash,
        validator_hash=admission.validator_hash,
        upper_ontology_hash=admission.upper_ontology_hash,
    )
    c0_runtime = CaseClassicalRuntimeBinding(
        backend="spacy-ner-dependency-plus-deterministic-rules-v2",
        config_file_sha256=c0_file.file_sha256,
        config_hash=canonical_sha256(c0),
    )
    return model_runtime, c0_runtime, indexing_file, index_config


def compile_case_study_execution_plan(
    *,
    repository_root: Path,
    policy: CaseStudyRuntimePolicy,
    loaded: AttestedRestrictedCaseStudy,
    admission: CaseStudyAdmissionAttestation,
    selected_model_freeze: AttestedSelectedModelFreeze,
    compiled_at: datetime,
) -> CaseStudyExecutionPlan:
    """Compile the exact path-free 4/8/1 plan without executing any condition."""

    if compiled_at.tzinfo is None or compiled_at.utcoffset() is None:
        raise CaseStudyPlanError("compiled_at must be timezone-aware")
    if compiled_at < max(
        loaded.attestation.attested_at,
        admission.attested_at,
        loaded.preregistration.frozen_at,
    ):
        raise CaseStudyPlanError("execution plan cannot predate its attestations")
    registered_reveal_times = (
        *(
            context.revealed_at
            for window in loaded.preregistration.windows
            for context in window.contexts
        ),
        loaded.preregistration.operational_retrieval.context.revealed_at,
    )
    if any(revealed_at <= compiled_at for revealed_at in registered_reveal_times):
        raise CaseStudyPlanError(
            "execution plan and attestations must exist before every case query reveal"
        )
    policy_file, policy_payload = _json_dependency(
        repository_root,
        RUNTIME_POLICY_RELATIVE_PATH,
    )
    tracked_policy = CaseStudyRuntimePolicy.model_validate(policy_payload)
    if tracked_policy.content_hash != policy.content_hash:
        raise CaseStudyPlanError("runtime policy differs from the tracked frozen policy")
    model_runtime, c0_runtime, indexing_file, index_config = _compile_runtime_bindings(
        repository_root=repository_root,
        policy=policy,
        loaded=loaded,
        admission=admission,
        selected_model_freeze=selected_model_freeze,
    )
    runtime_hash = model_runtime.content_hash
    windows: list[CaseWindowExecutionPlan] = []
    slots: list[CaseGpuCallSlot] = []
    ordinal = 1
    for window_ordinal, window in enumerate(loaded.preregistration.windows, start=1):
        context_ids = tuple(context.context_id for context in window.contexts)
        context_hashes = tuple(context.content_hash for context in window.contexts)
        prefix = f"case-window-{window_ordinal:02d}"
        c1_call_id = f"{prefix}-c1-pre"
        c2_call_ids = tuple(f"{prefix}-c2-{context_ordinal}" for context_ordinal in ("a", "b"))
        windows.append(
            CaseWindowExecutionPlan(
                window_id=window.window_id,
                window_registration_hash=window.content_hash,
                horizon_hash=window.fixed_horizon.content_hash,
                context_ids=context_ids,  # type: ignore[arg-type]
                context_hashes=context_hashes,  # type: ignore[arg-type]
                registered_revealed_at=tuple(  # type: ignore[arg-type]
                    context.revealed_at for context in window.contexts
                ),
                evidence_binding_hash=canonical_sha256(
                    {
                        "restricted_index_manifest_hash": loaded.manifest.content_hash,
                        "window_registration_hash": window.content_hash,
                        "passage_ids": window.passage_ids,
                        "horizon_hash": window.fixed_horizon.content_hash,
                        "retrieval_method": RetrievalMethod.ALL_ADMISSIBLE,
                    }
                ),
                packet_equality_group_id=f"{prefix}-same-evidence",
                c0_preparation_job_id=f"{prefix}-c0-pre",
                c1_preconstruction_call_id=c1_call_id,
                c2_empty_inventory_job_id=f"{prefix}-c2-empty-inventory",
                c0_projection_job_ids=(
                    f"{prefix}-c0-a",
                    f"{prefix}-c0-b",
                ),
                c1_projection_job_ids=(
                    f"{prefix}-c1-a",
                    f"{prefix}-c1-b",
                ),
                c2_construction_call_ids=c2_call_ids,  # type: ignore[arg-type]
            )
        )
        slots.append(
            CaseGpuCallSlot(
                call_id=c1_call_id,
                ordinal=ordinal,
                call_class=policy.case_c1_call_class,
                role=CaseGpuCallRole.C1_WINDOW_PRECONSTRUCTION,
                condition=ConditionName.C1_LLM_PRE,
                window_id=window.window_id,
                context_id=None,
                runtime_binding_hash=runtime_hash,
                resolved_vllm_seed=model_runtime.resolved_vllm_seed,
                admission_p95_seconds=240,
                watchdog_seconds=policy.c1_watchdog_seconds,
                repair_reserve_class=policy.c1_repair_reserve_class,
                operational_only=False,
                causal_comparison_eligible=True,
                query_blind=True,
            )
        )
        ordinal += 1
    for window, window_plan in zip(
        loaded.preregistration.windows,
        windows,
        strict=True,
    ):
        for context, call_id in zip(
            window.contexts,
            window_plan.c2_construction_call_ids,
            strict=True,
        ):
            slots.append(
                CaseGpuCallSlot(
                    call_id=call_id,
                    ordinal=ordinal,
                    call_class=policy.case_c2_call_class,
                    role=CaseGpuCallRole.C2_BOUNDED_CONSTRUCTION,
                    condition=ConditionName.C2_LLM_QUERY,
                    window_id=window.window_id,
                    context_id=context.context_id,
                    runtime_binding_hash=runtime_hash,
                    resolved_vllm_seed=model_runtime.resolved_vllm_seed,
                    admission_p95_seconds=150,
                    watchdog_seconds=policy.c2_watchdog_seconds,
                    repair_reserve_class=policy.c2_repair_reserve_class,
                    operational_only=False,
                    causal_comparison_eligible=True,
                    query_blind=False,
                )
            )
            ordinal += 1
    operational_registration = loaded.preregistration.operational_retrieval
    operational_call_id = "case-full-index-c2-operational"
    operational = OperationalCaseExecutionPlan(
        operational_query_id=operational_registration.operational_query_id,
        registration_hash=operational_registration.content_hash,
        context_id=operational_registration.context.context_id,
        context_hash=operational_registration.context.content_hash,
        registered_revealed_at=operational_registration.context.revealed_at,
        horizon_hash=operational_registration.context.spoiler_horizon.content_hash,
        full_index_binding_hash=canonical_sha256(
            {
                "restricted_index_manifest_hash": loaded.manifest.content_hash,
                "operational_registration_hash": operational_registration.content_hash,
                "retrieval_method": RetrievalMethod.SQLITE_FTS5_BM25,
            }
        ),
        c2_empty_inventory_job_id="case-full-index-c2-empty-inventory",
        retrieval_job_id="case-full-index-fts-bm25",
        c2_call_id=operational_call_id,
        top_k=index_config.operational_top_k,
        max_evidence_tokens=index_config.operational_max_evidence_tokens,
    )
    slots.append(
        CaseGpuCallSlot(
            call_id=operational_call_id,
            ordinal=ordinal,
            call_class=policy.case_full_index_c2_call_class,
            role=CaseGpuCallRole.C2_FULL_INDEX_OPERATIONAL,
            condition=ConditionName.C2_LLM_QUERY,
            window_id="complete-query-blind-index",
            context_id=operational.context_id,
            runtime_binding_hash=runtime_hash,
            resolved_vllm_seed=model_runtime.resolved_vllm_seed,
            admission_p95_seconds=150,
            watchdog_seconds=policy.operational_watchdog_seconds,
            repair_reserve_class=policy.c2_repair_reserve_class,
            operational_only=True,
            causal_comparison_eligible=False,
            query_blind=False,
        )
    )
    execution_id = f"case-study-{loaded.preregistration.content_hash[:16]}"
    return CaseStudyExecutionPlan(
        execution_id=execution_id,
        runtime_policy_file_sha256=policy_file.file_sha256,
        runtime_policy_hash=policy.content_hash,
        input_attestation_hash=loaded.attestation.content_hash,
        input_attestation_file_sha256=loaded.attestation_file_sha256,
        admission_attestation_hash=admission.content_hash,
        semantic_gate_bundle_hash=admission.semantic_gate_bundle_hash,
        restricted_index_manifest_hash=loaded.manifest.content_hash,
        restricted_index_manifest_file_sha256=loaded.index_manifest_file_sha256,
        preregistration_hash=loaded.preregistration.content_hash,
        preregistration_file_sha256=loaded.preregistration_file_sha256,
        indexing_config_file_sha256=indexing_file.file_sha256,
        indexing_config_hash=index_config.content_hash,
        c0_runtime=c0_runtime,
        model_runtime=model_runtime,
        windows=tuple(windows),  # type: ignore[arg-type]
        operational=operational,
        gpu_call_slots=tuple(slots),
        detailed_review_context_ids=loaded.preregistration.detailed_review_context_ids,
        artifact_storage=RestrictedArtifactStoragePlan(
            restricted_relative_prefix=f"{policy.restricted_artifact_prefix}/{execution_id}",
            bounded_packet_storage=policy.packet_persistence,
        ),
        compiled_at=compiled_at,
    )


class CasePrequeryKind(StrEnum):
    """The four preparation forms that must exist before query access."""

    C0_WINDOW_PRECONSTRUCTION = "c0_window_preconstruction"
    C1_WINDOW_PRECONSTRUCTION = "c1_window_preconstruction"
    C2_BOUNDED_EMPTY_INVENTORY = "c2_bounded_empty_inventory"
    C2_OPERATIONAL_EMPTY_INVENTORY = "c2_operational_empty_inventory"


class CasePrequeryReceipt(ImmutableRecord):
    """Restricted result of one pre-query job, including terminal failures."""

    receipt_id: Identifier
    execution_plan_hash: Sha256Digest
    preparation_job_id: Identifier
    kind: CasePrequeryKind
    window_id: Identifier
    condition: ConditionName
    evidence_binding_hash: Sha256Digest
    snapshot_hash: Sha256Digest
    backend: Literal[
        "spacy-ner-dependency-plus-deterministic-rules-v2",
        "vllm_gpu",
        "controller",
    ]
    terminal_outcome: RunOutcome
    started_at: AwareDatetime
    completed_at: AwareDatetime
    base_attempt_artifact_hash: Sha256Digest | None = None
    repair_attempt_count: Literal[0, 1] = 0
    repair_attempt_artifact_hash: Sha256Digest | None = None
    repair_parent_artifact_hash: Sha256Digest | None = None
    repair_reserve_class: Literal["reserve_long"] | None = None
    construction_seal: ConstructionSeal | None = None
    empty_inventory: PreQueryInventory | None = None
    failure_lineage_hash: Sha256Digest | None = None
    intention_to_treat_included: Literal[True] = True
    release_class: Literal[ReleaseClass.RESTRICTED] = ReleaseClass.RESTRICTED

    @model_validator(mode="after")
    def shape_and_lineage_match_kind(self) -> Self:
        if self.terminal_outcome not in TERMINAL_OUTCOMES:
            raise ValueError("case pre-query receipt requires a terminal outcome")
        if self.completed_at < self.started_at:
            raise ValueError("case pre-query completion cannot predate its start")
        repaired = self.repair_attempt_count == 1
        repair_fields_present = (
            self.repair_attempt_artifact_hash is not None
            and self.repair_parent_artifact_hash is not None
            and self.repair_reserve_class == "reserve_long"
        )
        if repaired != repair_fields_present:
            raise ValueError("C1 repair count, artifacts, parent, and reserve must agree")
        if repaired and self.repair_parent_artifact_hash != self.base_attempt_artifact_hash:
            raise ValueError("C1 repair must cite the rejected base attempt")
        if not repaired and any(
            value is not None
            for value in (
                self.repair_attempt_artifact_hash,
                self.repair_parent_artifact_hash,
                self.repair_reserve_class,
            )
        ):
            raise ValueError("unused C1 repair fields must remain empty")

        succeeded = self.terminal_outcome is RunOutcome.SUCCEEDED
        expected: tuple[ConditionName, str]
        if self.kind is CasePrequeryKind.C0_WINDOW_PRECONSTRUCTION:
            expected = (
                ConditionName.C0_CLASSICAL_PRE,
                "spacy-ner-dependency-plus-deterministic-rules-v2",
            )
            if self.base_attempt_artifact_hash is not None or repaired:
                raise ValueError("deterministic C0 preparation cannot claim an LLM attempt")
            if self.empty_inventory is not None:
                raise ValueError("C0 preparation cannot carry a C2 empty inventory")
        elif self.kind is CasePrequeryKind.C1_WINDOW_PRECONSTRUCTION:
            expected = (ConditionName.C1_LLM_PRE, "vllm_gpu")
            if self.base_attempt_artifact_hash is None:
                raise ValueError("C1 preparation must retain its raw base-attempt hash")
            if self.empty_inventory is not None:
                raise ValueError("C1 preparation cannot carry a C2 empty inventory")
        else:
            expected = (ConditionName.C2_LLM_QUERY, "controller")
            if self.terminal_outcome is not RunOutcome.SUCCEEDED:
                raise ValueError("recording an empty C2 inventory is a deterministic gate")
            if any(
                value is not None
                for value in (
                    self.base_attempt_artifact_hash,
                    self.repair_attempt_artifact_hash,
                    self.repair_parent_artifact_hash,
                    self.repair_reserve_class,
                    self.construction_seal,
                    self.failure_lineage_hash,
                )
            ):
                raise ValueError("C2 empty-inventory receipt cannot claim semantic work")
            if self.empty_inventory is None:
                raise ValueError("C2 requires its typed empty pre-query inventory")

        if (self.condition, self.backend) != expected:
            raise ValueError("pre-query condition/backend differs from its registered kind")
        if self.construction_seal is not None:
            if not succeeded or self.construction_seal.condition is not self.condition:
                raise ValueError("only successful C0/C1 may carry their construction seal")
            if self.construction_seal.snapshot_hash != self.snapshot_hash:
                raise ValueError("construction seal references another snapshot")
            if self.construction_seal.sealed_at > self.completed_at:
                raise ValueError("pre-query receipt cannot complete before its seal")
        elif (
            self.kind
            in {
                CasePrequeryKind.C0_WINDOW_PRECONSTRUCTION,
                CasePrequeryKind.C1_WINDOW_PRECONSTRUCTION,
            }
            and succeeded
        ):
            raise ValueError("successful C0/C1 preparation requires a construction seal")
        if self.empty_inventory is not None:
            if self.empty_inventory.condition is not ConditionName.C2_LLM_QUERY:
                raise ValueError("case empty inventory must belong to C2")
            if self.empty_inventory.snapshot_hash != self.snapshot_hash:
                raise ValueError("C2 empty inventory references another snapshot")
            if self.empty_inventory.recorded_at > self.completed_at:
                raise ValueError("pre-query receipt cannot complete before inventory recording")
        if succeeded != (self.failure_lineage_hash is None):
            raise ValueError("terminal pre-query failures require one failure-lineage hash")
        return self


class CasePrequeryBarrierReceipt(ImmutableRecord):
    """Seal over all 4 C0, 4 C1, 4 bounded C2, and one operational C2 preparations."""

    barrier_id: Identifier
    execution_plan_hash: Sha256Digest
    prequery_receipt_hashes: tuple[Sha256Digest, ...]
    sealed_at: AwareDatetime
    query_access_count_at_seal: Literal[0] = 0
    release_class: Literal[ReleaseClass.RESTRICTED] = ReleaseClass.RESTRICTED

    @model_validator(mode="after")
    def exact_receipt_inventory(self) -> Self:
        if len(self.prequery_receipt_hashes) != 13 or len(set(self.prequery_receipt_hashes)) != 13:
            raise ValueError("case pre-query barrier requires exactly thirteen unique receipts")
        return self


class CaseQueryAccessReceipt(ImmutableRecord):
    """Prose-free query and evidence receipt persisted before condition execution."""

    access_id: Identifier
    execution_plan_hash: Sha256Digest
    context_id: Identifier
    context_hash: Sha256Digest
    window_id: Identifier
    prequery_barrier_hash: Sha256Digest
    evidence_binding_hash: Sha256Digest
    packet_equality_group_id: Identifier
    snapshot_hash: Sha256Digest
    packet_hash: Sha256Digest
    ordered_evidence_ids_hash: Sha256Digest
    evidence_count: int = Field(gt=0)
    retrieval_method: RetrievalMethod
    registered_revealed_at: AwareDatetime
    accessed_at: AwareDatetime
    packet_materialized_at: AwareDatetime
    operational_retrieval_receipt: OperationalRetrievalReceipt | None = None
    operational_only: bool
    causal_comparison_eligible: bool
    release_class: Literal[ReleaseClass.RESTRICTED] = ReleaseClass.RESTRICTED

    @model_validator(mode="after")
    def query_and_retrieval_chronology(self) -> Self:
        if self.accessed_at < self.registered_revealed_at:
            raise ValueError("physical case query access cannot predate registered reveal")
        if self.operational_only == self.causal_comparison_eligible:
            raise ValueError("operational and causal-comparison flags must be opposites")
        if self.operational_only:
            if self.retrieval_method is not RetrievalMethod.SQLITE_FTS5_BM25:
                raise ValueError("operational case query must use SQLite FTS5/BM25")
            if self.packet_materialized_at < self.accessed_at:
                raise ValueError("query-dependent operational retrieval cannot predate access")
            receipt = self.operational_retrieval_receipt
            if receipt is None:
                raise ValueError("operational query requires its rank/omission receipt")
            if (
                receipt.packet_hash != self.packet_hash
                or receipt.query_accessed_at != self.accessed_at
                or receipt.packet_created_at != self.packet_materialized_at
                or len(receipt.ordered_evidence_ids) != self.evidence_count
                or canonical_sha256(receipt.ordered_evidence_ids) != self.ordered_evidence_ids_hash
            ):
                raise ValueError("operational retrieval receipt does not match query access")
        elif self.retrieval_method is not RetrievalMethod.ALL_ADMISSIBLE:
            raise ValueError("bounded comparison packets must contain all admissible evidence")
        elif self.operational_retrieval_receipt is not None:
            raise ValueError("bounded all-admissible query cannot carry an FTS receipt")
        return self


class CaseOutputReceipt(ImmutableRecord):
    """One ITT case output, including an invalid/timeout/failure as an outcome."""

    receipt_id: Identifier
    execution_plan_hash: Sha256Digest
    projection_job_id: Identifier
    condition: ConditionName
    context_id: Identifier
    window_id: Identifier
    backend: Literal[
        "spacy-ner-dependency-plus-deterministic-rules-v2",
        "fixed_projection_cpu",
        "vllm_gpu",
    ]
    query_access_receipt_hash: Sha256Digest
    preparation_receipt_hash: Sha256Digest
    evidence_binding_hash: Sha256Digest
    packet_equality_group_id: Identifier
    snapshot_hash: Sha256Digest
    packet_hash: Sha256Digest
    source_construction_seal_hash: Sha256Digest | None = None
    pre_query_inventory_hash: Sha256Digest | None = None
    base_attempt_artifact_hash: Sha256Digest | None = None
    repair_attempt_count: Literal[0, 1] = 0
    repair_attempt_artifact_hash: Sha256Digest | None = None
    repair_parent_artifact_hash: Sha256Digest | None = None
    repair_reserve_class: Literal["reserve_standard"] | None = None
    terminal_outcome: RunOutcome
    projection_artifact_hash: Sha256Digest | None = None
    failure_lineage_hash: Sha256Digest | None = None
    construction_certificate: ConstructionCertificate | None = None
    query_accessed_at: AwareDatetime
    completed_at: AwareDatetime
    operational_only: bool
    causal_comparison_eligible: bool
    intention_to_treat_included: Literal[True] = True
    release_class: Literal[ReleaseClass.RESTRICTED] = ReleaseClass.RESTRICTED

    @model_validator(mode="after")
    def terminal_lineage_and_capability(self) -> Self:
        if self.terminal_outcome not in TERMINAL_OUTCOMES:
            raise ValueError("case output receipt requires a terminal outcome")
        if self.completed_at <= self.query_accessed_at:
            raise ValueError("case output must complete strictly after query access")
        if self.operational_only == self.causal_comparison_eligible:
            raise ValueError("operational and causal-comparison flags must be opposites")
        succeeded = self.terminal_outcome is RunOutcome.SUCCEEDED
        if succeeded != (self.projection_artifact_hash is not None):
            raise ValueError("only successful ITT outputs may cite a projection artifact")
        if succeeded != (self.failure_lineage_hash is None):
            raise ValueError("non-success ITT outputs require failure lineage")

        is_c2 = self.condition is ConditionName.C2_LLM_QUERY
        if is_c2:
            if self.backend != "vllm_gpu":
                raise ValueError("C2 semantic construction must execute through the GPU LLM")
            if self.pre_query_inventory_hash is None:
                raise ValueError("C2 output requires the empty pre-query inventory")
            if self.source_construction_seal_hash is not None:
                raise ValueError("C2 cannot inherit a hidden preconstructed ontology")
            if self.base_attempt_artifact_hash is None:
                raise ValueError("C2 must retain its raw base-attempt artifact hash")
            if succeeded:
                if self.construction_certificate is None:
                    raise ValueError("successful C2 requires a construction certificate")
                certificate = self.construction_certificate
                if (
                    certificate.condition is not ConditionName.C2_LLM_QUERY
                    or certificate.snapshot_hash != self.snapshot_hash
                    or certificate.packet_hash != self.packet_hash
                    or certificate.pre_query_inventory_hash != self.pre_query_inventory_hash
                    or certificate.query_revealed_at != self.query_accessed_at
                    or certificate.completed_at > self.completed_at
                ):
                    raise ValueError("C2 construction certificate lineage does not match receipt")
                terminal_attempt_hash = (
                    self.repair_attempt_artifact_hash
                    if self.repair_attempt_count == 1
                    else self.base_attempt_artifact_hash
                )
                if certificate.raw_output_artifact_hash != terminal_attempt_hash:
                    raise ValueError("C2 certificate does not cite the terminal generation attempt")
            elif self.construction_certificate is not None:
                raise ValueError("failed C2 cannot claim a successful construction certificate")
        elif self.condition is ConditionName.C0_CLASSICAL_PRE:
            if self.backend != "spacy-ner-dependency-plus-deterministic-rules-v2":
                raise ValueError("C0 output must use the frozen credible classical path")
            if succeeded and self.source_construction_seal_hash is None:
                raise ValueError("C0 projection must bind its query-blind preparation seal")
            if any(
                value is not None
                for value in (
                    self.pre_query_inventory_hash,
                    self.base_attempt_artifact_hash,
                    self.construction_certificate,
                )
            ):
                raise ValueError("C0 projection cannot claim LLM construction lineage")
        elif self.condition is ConditionName.C1_LLM_PRE:
            if self.backend != "fixed_projection_cpu":
                raise ValueError("case C1 uses fixed projection over its sealed preontology")
            if succeeded and self.source_construction_seal_hash is None:
                raise ValueError("C1 projection must bind its query-blind construction seal")
            if any(
                value is not None
                for value in (
                    self.pre_query_inventory_hash,
                    self.base_attempt_artifact_hash,
                    self.construction_certificate,
                )
            ):
                raise ValueError("case C1 projection cannot perform another LLM construction")
        else:
            raise ValueError("bounded novel study includes only C0, C1, and C2")

        repaired = self.repair_attempt_count == 1
        repair_fields_present = (
            self.repair_attempt_artifact_hash is not None
            and self.repair_parent_artifact_hash is not None
            and self.repair_reserve_class == "reserve_standard"
        )
        if repaired != repair_fields_present:
            raise ValueError("C2 repair count, artifacts, parent, and reserve must agree")
        if repaired and (
            not is_c2 or self.repair_parent_artifact_hash != self.base_attempt_artifact_hash
        ):
            raise ValueError("only C2 may repair, citing its rejected base attempt")
        if not repaired and any(
            value is not None
            for value in (
                self.repair_attempt_artifact_hash,
                self.repair_parent_artifact_hash,
                self.repair_reserve_class,
            )
        ):
            raise ValueError("unused repair fields must remain empty")
        return self


class CaseStudyResumeManifest(ImmutableRecord):
    """Append-only restricted controller state; payload artifacts remain content-addressed."""

    resume_id: Identifier
    sequence_number: int = Field(ge=0)
    parent_resume_hash: Sha256Digest | None = None
    execution_plan_hash: Sha256Digest
    prequery_receipts: tuple[CasePrequeryReceipt, ...] = ()
    prequery_barrier: CasePrequeryBarrierReceipt | None = None
    query_access_receipts: tuple[CaseQueryAccessReceipt, ...] = ()
    output_receipts: tuple[CaseOutputReceipt, ...] = ()
    updated_at: AwareDatetime
    release_class: Literal[ReleaseClass.RESTRICTED] = ReleaseClass.RESTRICTED

    @model_validator(mode="after")
    def locally_unique_and_parented(self) -> Self:
        collections = (
            tuple(item.preparation_job_id for item in self.prequery_receipts),
            tuple(item.context_id for item in self.query_access_receipts),
            tuple(item.projection_job_id for item in self.output_receipts),
        )
        if any(len(values) != len(set(values)) for values in collections):
            raise ValueError("resume manifest contains a duplicate scientific job")
        if self.sequence_number == 0 and self.parent_resume_hash is not None:
            raise ValueError("initial resume manifest cannot have a parent")
        if self.sequence_number == 0 and (
            self.prequery_receipts
            or self.prequery_barrier is not None
            or self.query_access_receipts
            or self.output_receipts
        ):
            raise ValueError("initial resume manifest must be an empty controller state")
        if self.sequence_number > 0 and self.parent_resume_hash is None:
            raise ValueError("noninitial resume manifest must cite its parent")
        if any(
            record.execution_plan_hash != self.execution_plan_hash
            for record in (
                *self.prequery_receipts,
                *self.query_access_receipts,
                *self.output_receipts,
            )
        ):
            raise ValueError("resume record references another execution plan")
        if self.prequery_barrier is not None and (
            self.prequery_barrier.execution_plan_hash != self.execution_plan_hash
        ):
            raise ValueError("resume barrier references another execution plan")
        return self


class CaseStudyResumeStatus(ImmutableRecord):
    execution_plan_hash: Sha256Digest
    resume_manifest_hash: Sha256Digest
    completed_prequery_job_count: int = Field(ge=0, le=13)
    query_access_count: int = Field(ge=0, le=9)
    completed_output_count: int = Field(ge=0, le=25)
    non_succeeded_output_count: int = Field(ge=0, le=25)
    invalid_output_count: int = Field(ge=0, le=25)
    failed_output_count: int = Field(ge=0, le=25)
    timed_out_output_count: int = Field(ge=0, le=25)
    interrupted_output_count: int = Field(ge=0, le=25)
    repaired_attempt_count: int = Field(ge=0, le=13)
    pending_prequery_job_ids: tuple[Identifier, ...]
    pending_query_context_ids: tuple[Identifier, ...]
    pending_output_job_ids: tuple[Identifier, ...]
    next_stage: Literal["prequery", "seal_barrier", "query_outputs", "complete"]
    complete: bool
    operational_output_kept_separate: Literal[True] = True
    release_class: Literal[ReleaseClass.RESTRICTED] = ReleaseClass.RESTRICTED

    @model_validator(mode="after")
    def counts_and_stage_reconcile(self) -> Self:
        if self.completed_prequery_job_count + len(self.pending_prequery_job_ids) != 13:
            raise ValueError("resume pre-query counts do not reconcile to thirteen jobs")
        if self.query_access_count + len(self.pending_query_context_ids) != 9:
            raise ValueError("resume query counts do not reconcile to nine contexts")
        if self.completed_output_count + len(self.pending_output_job_ids) != 25:
            raise ValueError("resume output counts do not reconcile to twenty-five outputs")
        classified_failures = (
            self.invalid_output_count
            + self.failed_output_count
            + self.timed_out_output_count
            + self.interrupted_output_count
        )
        if classified_failures != self.non_succeeded_output_count:
            raise ValueError("resume non-success outcome counts do not reconcile")
        if self.non_succeeded_output_count > self.completed_output_count:
            raise ValueError("resume has more failures than completed ITT outputs")
        if self.complete != (self.next_stage == "complete"):
            raise ValueError("resume completion flag differs from its next stage")
        return self


@dataclass(frozen=True, slots=True)
class _ExpectedOutput:
    job_id: str
    condition: ConditionName
    context_id: str
    window_id: str
    preparation_job_id: str
    evidence_binding_hash: str
    packet_equality_group_id: str
    operational_only: bool


def _expected_prequery_jobs(
    plan: CaseStudyExecutionPlan,
) -> tuple[tuple[str, CasePrequeryKind, str, str], ...]:
    jobs: list[tuple[str, CasePrequeryKind, str, str]] = []
    for window in plan.windows:
        jobs.extend(
            (
                (
                    window.c0_preparation_job_id,
                    CasePrequeryKind.C0_WINDOW_PRECONSTRUCTION,
                    window.window_id,
                    window.evidence_binding_hash,
                ),
                (
                    window.c1_preconstruction_call_id,
                    CasePrequeryKind.C1_WINDOW_PRECONSTRUCTION,
                    window.window_id,
                    window.evidence_binding_hash,
                ),
                (
                    window.c2_empty_inventory_job_id,
                    CasePrequeryKind.C2_BOUNDED_EMPTY_INVENTORY,
                    window.window_id,
                    window.evidence_binding_hash,
                ),
            )
        )
    jobs.append(
        (
            plan.operational.c2_empty_inventory_job_id,
            CasePrequeryKind.C2_OPERATIONAL_EMPTY_INVENTORY,
            "complete-query-blind-index",
            plan.operational.full_index_binding_hash,
        )
    )
    return tuple(jobs)


def _expected_outputs(plan: CaseStudyExecutionPlan) -> tuple[_ExpectedOutput, ...]:
    expected: list[_ExpectedOutput] = []
    for window in plan.windows:
        for index, context_id in enumerate(window.context_ids):
            expected.extend(
                (
                    _ExpectedOutput(
                        window.c0_projection_job_ids[index],
                        ConditionName.C0_CLASSICAL_PRE,
                        context_id,
                        window.window_id,
                        window.c0_preparation_job_id,
                        window.evidence_binding_hash,
                        window.packet_equality_group_id,
                        False,
                    ),
                    _ExpectedOutput(
                        window.c1_projection_job_ids[index],
                        ConditionName.C1_LLM_PRE,
                        context_id,
                        window.window_id,
                        window.c1_preconstruction_call_id,
                        window.evidence_binding_hash,
                        window.packet_equality_group_id,
                        False,
                    ),
                    _ExpectedOutput(
                        window.c2_construction_call_ids[index],
                        ConditionName.C2_LLM_QUERY,
                        context_id,
                        window.window_id,
                        window.c2_empty_inventory_job_id,
                        window.evidence_binding_hash,
                        window.packet_equality_group_id,
                        False,
                    ),
                )
            )
    expected.append(
        _ExpectedOutput(
            plan.operational.c2_call_id,
            ConditionName.C2_LLM_QUERY,
            plan.operational.context_id,
            "complete-query-blind-index",
            plan.operational.c2_empty_inventory_job_id,
            plan.operational.full_index_binding_hash,
            "case-full-index-operational-packet",
            True,
        )
    )
    return tuple(expected)


def _registered_contexts(
    plan: CaseStudyExecutionPlan,
) -> dict[str, tuple[str, str, str, str, bool, datetime]]:
    contexts: dict[str, tuple[str, str, str, str, bool, datetime]] = {}
    for window in plan.windows:
        for context_id, context_hash, revealed_at in zip(
            window.context_ids,
            window.context_hashes,
            window.registered_revealed_at,
            strict=True,
        ):
            contexts[context_id] = (
                context_hash,
                window.window_id,
                window.evidence_binding_hash,
                window.packet_equality_group_id,
                False,
                revealed_at,
            )
    contexts[plan.operational.context_id] = (
        plan.operational.context_hash,
        "complete-query-blind-index",
        plan.operational.full_index_binding_hash,
        "case-full-index-operational-packet",
        True,
        plan.operational.registered_revealed_at,
    )
    return contexts


def audit_case_study_resume(
    plan: CaseStudyExecutionPlan,
    resume: CaseStudyResumeManifest,
) -> CaseStudyResumeStatus:
    """Fail closed on timing, reuse, evidence equality, ITT, and operational labels."""

    if resume.execution_plan_hash != plan.content_hash:
        raise CaseStudyResumeError("resume manifest references another execution plan")
    if any(item.started_at < plan.compiled_at for item in resume.prequery_receipts):
        raise CaseStudyResumeError("pre-query work cannot predate execution-plan compilation")
    recorded_times = [item.completed_at for item in resume.prequery_receipts]
    recorded_times.extend(item.accessed_at for item in resume.query_access_receipts)
    recorded_times.extend(item.completed_at for item in resume.output_receipts)
    if resume.prequery_barrier is not None:
        recorded_times.append(resume.prequery_barrier.sealed_at)
    if recorded_times and resume.updated_at < max(recorded_times):
        raise CaseStudyResumeError("resume update time predates a persisted receipt")
    expected_prequery = {
        job_id: (kind, window_id, binding_hash)
        for job_id, kind, window_id, binding_hash in _expected_prequery_jobs(plan)
    }
    prequery_by_job = {receipt.preparation_job_id: receipt for receipt in resume.prequery_receipts}
    unknown_prequery = set(prequery_by_job) - set(expected_prequery)
    if unknown_prequery:
        raise CaseStudyResumeError("resume contains unregistered pre-query jobs")
    for job_id, receipt in prequery_by_job.items():
        kind, window_id, binding_hash = expected_prequery[job_id]
        if (
            receipt.kind is not kind
            or receipt.window_id != window_id
            or receipt.evidence_binding_hash != binding_hash
        ):
            raise CaseStudyResumeError("pre-query receipt differs from the frozen job")
    for window in plan.windows:
        receipts = tuple(
            prequery_by_job[job_id]
            for job_id in (
                window.c0_preparation_job_id,
                window.c1_preconstruction_call_id,
                window.c2_empty_inventory_job_id,
            )
            if job_id in prequery_by_job
        )
        if len({item.snapshot_hash for item in receipts}) > 1:
            raise CaseStudyResumeError("C0/C1/C2 preparations for a window use unequal snapshots")

    barrier = resume.prequery_barrier
    if barrier is not None:
        if set(prequery_by_job) != set(expected_prequery):
            raise CaseStudyResumeError("query barrier cannot seal an incomplete preparation set")
        if barrier.prequery_receipt_hashes != tuple(
            item.content_hash for item in resume.prequery_receipts
        ):
            raise CaseStudyResumeError("query barrier does not seal the exact resume receipts")
        if any(item.completed_at >= barrier.sealed_at for item in resume.prequery_receipts):
            raise CaseStudyResumeError("every preparation must strictly precede the query barrier")
    elif resume.query_access_receipts or resume.output_receipts:
        raise CaseStudyResumeError("query access/output exists without a sealed pre-query barrier")

    contexts = _registered_contexts(plan)
    access_by_context = {receipt.context_id: receipt for receipt in resume.query_access_receipts}
    if set(access_by_context) - set(contexts):
        raise CaseStudyResumeError("resume contains an unregistered case query")
    for context_id, access in access_by_context.items():
        (
            context_hash,
            window_id,
            binding_hash,
            equality_group,
            operational,
            registered_revealed_at,
        ) = contexts[context_id]
        if barrier is None or access.prequery_barrier_hash != barrier.content_hash:
            raise CaseStudyResumeError("query access does not cite the sealed case barrier")
        if access.accessed_at <= barrier.sealed_at:
            raise CaseStudyResumeError("physical query access must strictly follow the barrier")
        if (
            access.registered_revealed_at != registered_revealed_at
            or registered_revealed_at <= barrier.sealed_at
        ):
            raise CaseStudyResumeError(
                "registered case query reveal must strictly follow the sealed barrier"
            )
        if access.packet_materialized_at < plan.compiled_at:
            raise CaseStudyResumeError("case packet cannot predate its exact execution plan")
        if (
            access.context_hash != context_hash
            or access.window_id != window_id
            or access.evidence_binding_hash != binding_hash
            or access.packet_equality_group_id != equality_group
            or access.operational_only is not operational
            or access.causal_comparison_eligible is operational
        ):
            raise CaseStudyResumeError("query/evidence receipt differs from registration")
        if not operational and access.packet_materialized_at >= barrier.sealed_at:
            raise CaseStudyResumeError(
                "all-admissible bounded packets must be materialized before query reveal"
            )
        if operational:
            operational_receipt = access.operational_retrieval_receipt
            if operational_receipt is None or (
                operational_receipt.operational_query_id != plan.operational.operational_query_id
                or operational_receipt.restricted_index_manifest_hash
                != plan.restricted_index_manifest_hash
            ):
                raise CaseStudyResumeError(
                    "operational ranks/omissions reference another query or index"
                )
        preparation_snapshot_hashes = {
            item.snapshot_hash for item in prequery_by_job.values() if item.window_id == window_id
        }
        if preparation_snapshot_hashes != {access.snapshot_hash}:
            raise CaseStudyResumeError(
                "query packet snapshot differs from its pre-query preparations"
            )
    for window in plan.windows:
        pair = tuple(
            access_by_context[context_id]
            for context_id in window.context_ids
            if context_id in access_by_context
        )
        if (
            len(pair) == 2
            and len(
                {
                    (
                        item.snapshot_hash,
                        item.packet_hash,
                        item.ordered_evidence_ids_hash,
                        item.evidence_count,
                    )
                    for item in pair
                }
            )
            != 1
        ):
            raise CaseStudyResumeError(
                "contrastive contexts in one bounded window received unequal evidence"
            )

    expected_outputs = {item.job_id: item for item in _expected_outputs(plan)}
    output_by_job = {receipt.projection_job_id: receipt for receipt in resume.output_receipts}
    if set(output_by_job) - set(expected_outputs):
        raise CaseStudyResumeError("resume contains an unregistered case output")
    for job_id, output in output_by_job.items():
        expected = expected_outputs[job_id]
        access = access_by_context.get(expected.context_id)
        preparation = prequery_by_job.get(expected.preparation_job_id)
        if access is None or preparation is None:
            raise CaseStudyResumeError("case output lacks query/preparation lineage")
        if (
            output.condition is not expected.condition
            or output.context_id != expected.context_id
            or output.window_id != expected.window_id
            or output.preparation_receipt_hash != preparation.content_hash
            or output.query_access_receipt_hash != access.content_hash
            or output.evidence_binding_hash != expected.evidence_binding_hash
            or output.packet_equality_group_id != expected.packet_equality_group_id
            or output.snapshot_hash != access.snapshot_hash
            or output.packet_hash != access.packet_hash
            or output.query_accessed_at != access.accessed_at
            or output.operational_only is not expected.operational_only
            or output.causal_comparison_eligible is expected.operational_only
        ):
            raise CaseStudyResumeError("case output differs from frozen/evidence lineage")
        if expected.condition in {
            ConditionName.C0_CLASSICAL_PRE,
            ConditionName.C1_LLM_PRE,
        }:
            seal = preparation.construction_seal
            if preparation.terminal_outcome is RunOutcome.SUCCEEDED:
                if seal is None or output.source_construction_seal_hash != seal.content_hash:
                    raise CaseStudyResumeError("C0/C1 output does not reuse its sealed ontology")
            else:
                if output.source_construction_seal_hash is not None:
                    raise CaseStudyResumeError(
                        "failed preconstruction cannot contribute a stale source seal"
                    )
                if output.terminal_outcome is RunOutcome.SUCCEEDED:
                    raise CaseStudyResumeError(
                        "failed preconstruction cannot yield a successful output"
                    )
        else:
            inventory = preparation.empty_inventory
            if inventory is None or output.pre_query_inventory_hash != inventory.content_hash:
                raise CaseStudyResumeError("C2 output does not cite its empty inventory")
            if output.construction_certificate is not None and (
                output.construction_certificate.query_access_event_hash != access.content_hash
                or output.construction_certificate.prequery_barrier_hash != barrier.content_hash
                or output.construction_certificate.query_context_hash != access.context_hash
            ):
                raise CaseStudyResumeError("C2 certificate does not bind query access/barrier")

    expected_prequery_ids = tuple(item[0] for item in _expected_prequery_jobs(plan))
    expected_context_ids = tuple(_registered_contexts(plan))
    expected_output_ids = tuple(item.job_id for item in _expected_outputs(plan))
    pending_prequery = tuple(
        job_id for job_id in expected_prequery_ids if job_id not in prequery_by_job
    )
    pending_contexts = tuple(
        context_id for context_id in expected_context_ids if context_id not in access_by_context
    )
    pending_outputs = tuple(job_id for job_id in expected_output_ids if job_id not in output_by_job)
    if pending_prequery:
        next_stage: Literal["prequery", "seal_barrier", "query_outputs", "complete"] = "prequery"
    elif barrier is None:
        next_stage = "seal_barrier"
    elif pending_contexts or pending_outputs:
        next_stage = "query_outputs"
    else:
        next_stage = "complete"
    repair_count = sum(item.repair_attempt_count for item in resume.prequery_receipts)
    repair_count += sum(item.repair_attempt_count for item in resume.output_receipts)
    return CaseStudyResumeStatus(
        execution_plan_hash=plan.content_hash,
        resume_manifest_hash=resume.content_hash,
        completed_prequery_job_count=len(prequery_by_job),
        query_access_count=len(access_by_context),
        completed_output_count=len(output_by_job),
        non_succeeded_output_count=sum(
            item.terminal_outcome is not RunOutcome.SUCCEEDED for item in resume.output_receipts
        ),
        invalid_output_count=sum(
            item.terminal_outcome is RunOutcome.INVALID for item in resume.output_receipts
        ),
        failed_output_count=sum(
            item.terminal_outcome is RunOutcome.FAILED for item in resume.output_receipts
        ),
        timed_out_output_count=sum(
            item.terminal_outcome is RunOutcome.TIMED_OUT for item in resume.output_receipts
        ),
        interrupted_output_count=sum(
            item.terminal_outcome is RunOutcome.INTERRUPTED for item in resume.output_receipts
        ),
        repaired_attempt_count=repair_count,
        pending_prequery_job_ids=pending_prequery,
        pending_query_context_ids=pending_contexts,
        pending_output_job_ids=pending_outputs,
        next_stage=next_stage,
        complete=next_stage == "complete",
    )


def validate_resume_successor(
    previous: CaseStudyResumeManifest,
    successor: CaseStudyResumeManifest,
) -> None:
    """Require a monotonic, append-only resume record; prior receipts cannot mutate."""

    if successor.execution_plan_hash != previous.execution_plan_hash:
        raise CaseStudyResumeError("resume successor references another execution plan")
    if successor.sequence_number != previous.sequence_number + 1:
        raise CaseStudyResumeError("resume successor sequence is not monotonic")
    if successor.parent_resume_hash != previous.content_hash:
        raise CaseStudyResumeError("resume successor does not cite its exact parent")
    if successor.updated_at <= previous.updated_at:
        raise CaseStudyResumeError("resume successor time must advance")
    for old, new, label in (
        (previous.prequery_receipts, successor.prequery_receipts, "pre-query"),
        (previous.query_access_receipts, successor.query_access_receipts, "query-access"),
        (previous.output_receipts, successor.output_receipts, "output"),
    ):
        if tuple(item.content_hash for item in new[: len(old)]) != tuple(
            item.content_hash for item in old
        ):
            raise CaseStudyResumeError(f"resume successor rewrote a prior {label} receipt")
    if previous.prequery_barrier is not None and (
        successor.prequery_barrier is None
        or successor.prequery_barrier.content_hash != previous.prequery_barrier.content_hash
    ):
        raise CaseStudyResumeError("resume successor removed or rewrote the query barrier")


def initialize_case_study_resume(
    plan: CaseStudyExecutionPlan,
    *,
    initialized_at: datetime,
) -> CaseStudyResumeManifest:
    """Create an empty restricted state without opening any query or model call."""

    if initialized_at.tzinfo is None or initialized_at.utcoffset() is None:
        raise CaseStudyResumeError("resume initialization time must be timezone-aware")
    if initialized_at < plan.compiled_at:
        raise CaseStudyResumeError("resume cannot be initialized before plan compilation")
    return CaseStudyResumeManifest(
        resume_id=f"resume-{plan.execution_id}-0000",
        sequence_number=0,
        execution_plan_hash=plan.content_hash,
        updated_at=initialized_at,
    )


class CaseReviewDimension(StrEnum):
    EVIDENCE_SUPPORT = "evidence_support"
    CONTEXTUAL_RELEVANCE = "contextual_relevance"
    TEMPORAL_INTEGRITY = "temporal_integrity"
    RARE_PIVOTAL_PRESERVATION = "rare_pivotal_preservation"
    HIGH_LEVEL_ORGANIZATION = "high_level_organization"


class CaseDetailedMatchingInputTemplate(ImmutableRecord):
    state: Literal["unpopulated"] = "unpopulated"
    assertion_match_rows: tuple[()] = ()
    event_match_rows: tuple[()] = ()
    permissible_alternative_rows: tuple[()] = ()


class CaseUnitReviewInputTemplate(ImmutableRecord):
    review_unit_id: Identifier
    context_id: Identifier
    window_id: Identifier
    evidence_binding_hash: Sha256Digest
    packet_equality_group_id: Identifier
    required_output_job_ids: tuple[Identifier, Identifier, Identifier]
    dimensions: tuple[
        CaseReviewDimension,
        CaseReviewDimension,
        CaseReviewDimension,
        CaseReviewDimension,
        CaseReviewDimension,
    ] = tuple(CaseReviewDimension)  # type: ignore[assignment]
    judgment_state: Literal["unpopulated"] = "unpopulated"
    reviewer_id: Literal[None] = None
    judgments: tuple[()] = ()
    reviewed_at: Literal[None] = None
    detailed_matching: CaseDetailedMatchingInputTemplate | None = None
    release_class: Literal[ReleaseClass.RESTRICTED] = ReleaseClass.RESTRICTED


class CaseStudyReviewInputTemplate(ImmutableRecord):
    template_id: Identifier
    execution_plan_hash: Sha256Digest
    preregistration_hash: Sha256Digest
    units: tuple[
        CaseUnitReviewInputTemplate,
        CaseUnitReviewInputTemplate,
        CaseUnitReviewInputTemplate,
        CaseUnitReviewInputTemplate,
        CaseUnitReviewInputTemplate,
        CaseUnitReviewInputTemplate,
        CaseUnitReviewInputTemplate,
        CaseUnitReviewInputTemplate,
    ]
    detailed_matching_context_ids: tuple[Identifier, Identifier, Identifier, Identifier]
    reviewer_judgment_count: Literal[0] = 0
    release_class: Literal[ReleaseClass.RESTRICTED] = ReleaseClass.RESTRICTED

    @model_validator(mode="after")
    def exact_blank_coverage(self) -> Self:
        context_ids = tuple(unit.context_id for unit in self.units)
        if len(set(context_ids)) != 8:
            raise ValueError("review template must cover eight distinct contexts")
        detailed_ids = {
            unit.context_id for unit in self.units if unit.detailed_matching is not None
        }
        if detailed_ids != set(self.detailed_matching_context_ids):
            raise ValueError("detailed matching template must cover the frozen four contexts")
        if any(unit.judgment_state != "unpopulated" or unit.judgments for unit in self.units):
            raise ValueError("review input template cannot contain reviewer judgments")
        return self


class CaseReviewVerdict(StrEnum):
    MEETS = "meets"
    PARTIALLY_MEETS = "partially_meets"
    DOES_NOT_MEET = "does_not_meet"
    NOT_ASSESSABLE = "not_assessable"


class CaseReviewRole(StrEnum):
    PRIMARY_RESEARCHER = "primary_researcher"
    SECOND_KNOWLEDGEABLE_READER = "second_knowledgeable_reader"


class CaseDimensionJudgment(ImmutableRecord):
    projection_job_id: Identifier
    output_receipt_hash: Sha256Digest
    condition: Literal[
        ConditionName.C0_CLASSICAL_PRE,
        ConditionName.C1_LLM_PRE,
        ConditionName.C2_LLM_QUERY,
    ]
    dimension: CaseReviewDimension
    verdict: CaseReviewVerdict


class CaseDetailedMatchSummary(ImmutableRecord):
    projection_job_id: Identifier
    output_receipt_hash: Sha256Digest
    condition: Literal[
        ConditionName.C0_CLASSICAL_PRE,
        ConditionName.C1_LLM_PRE,
        ConditionName.C2_LLM_QUERY,
    ]
    reference_assertion_count: int = Field(ge=0)
    output_assertion_count: int = Field(ge=0)
    matched_assertion_count: int = Field(ge=0)
    permissible_alternative_assertion_matches: int = Field(ge=0)
    reference_event_count: int = Field(ge=0)
    output_event_count: int = Field(ge=0)
    matched_event_count: int = Field(ge=0)
    permissible_alternative_event_matches: int = Field(ge=0)

    @model_validator(mode="after")
    def matching_counts_are_bounded(self) -> Self:
        if self.matched_assertion_count > min(
            self.reference_assertion_count, self.output_assertion_count
        ):
            raise ValueError("assertion matches exceed a review-side denominator")
        if self.matched_event_count > min(
            self.reference_event_count, self.output_event_count
        ):
            raise ValueError("event matches exceed a review-side denominator")
        if self.permissible_alternative_assertion_matches > self.matched_assertion_count:
            raise ValueError("permissible assertion alternatives exceed all matches")
        if self.permissible_alternative_event_matches > self.matched_event_count:
            raise ValueError("permissible event alternatives exceed all matches")
        return self


class CaseReviewerAssessment(ImmutableRecord):
    reviewer_id: Identifier
    role: CaseReviewRole
    reviewed_at: AwareDatetime
    judgments: tuple[CaseDimensionJudgment, ...]
    detailed_matching: tuple[CaseDetailedMatchSummary, ...] = ()
    judgments_supplied_by_named_human: Literal[True] = True
    automatically_generated_judgments: Literal[False] = False
    release_class: Literal[ReleaseClass.RESTRICTED] = ReleaseClass.RESTRICTED

    @model_validator(mode="after")
    def assessment_inventory_is_unique(self) -> Self:
        judgment_keys = tuple(
            (item.projection_job_id, item.dimension) for item in self.judgments
        )
        if len(self.judgments) != 15 or len(set(judgment_keys)) != 15:
            raise ValueError("one case assessment requires 3 outputs by 5 dimensions")
        detailed_jobs = tuple(item.projection_job_id for item in self.detailed_matching)
        if len(detailed_jobs) not in {0, 3} or len(set(detailed_jobs)) != len(detailed_jobs):
            raise ValueError("detailed matching must cover either zero or three outputs")
        return self


class CaseCompletedUnitReview(ImmutableRecord):
    review_unit_id: Identifier
    context_id: Identifier
    window_id: Identifier
    evidence_binding_hash: Sha256Digest
    packet_equality_group_id: Identifier
    required_output_job_ids: tuple[Identifier, Identifier, Identifier]
    selected_for_paper_example: bool
    primary: CaseReviewerAssessment
    secondary: CaseReviewerAssessment | None = None
    release_class: Literal[ReleaseClass.RESTRICTED] = ReleaseClass.RESTRICTED

    @model_validator(mode="after")
    def reviewer_roles_and_example_selection_agree(self) -> Self:
        if self.primary.role is not CaseReviewRole.PRIMARY_RESEARCHER:
            raise ValueError("every unit requires a primary researcher assessment")
        if self.secondary is not None:
            if self.secondary.role is not CaseReviewRole.SECOND_KNOWLEDGEABLE_READER:
                raise ValueError("secondary assessment has the wrong reviewer role")
            if self.secondary.reviewer_id == self.primary.reviewer_id:
                raise ValueError("a second reader must be distinct from the primary reviewer")
            if not self.selected_for_paper_example:
                raise ValueError("second-reader review is limited to selected paper examples")
        return self


class CaseStudyCompletedReview(ImmutableRecord):
    review_id: Identifier
    execution_plan_hash: Sha256Digest
    review_template_hash: Sha256Digest
    terminal_resume_manifest_hash: Sha256Digest
    bounded_output_receipt_hashes: tuple[Sha256Digest, ...]
    units: tuple[
        CaseCompletedUnitReview,
        CaseCompletedUnitReview,
        CaseCompletedUnitReview,
        CaseCompletedUnitReview,
        CaseCompletedUnitReview,
        CaseCompletedUnitReview,
        CaseCompletedUnitReview,
        CaseCompletedUnitReview,
    ]
    primary_judgment_count: Literal[120] = 120
    operational_output_excluded_from_bounded_review: Literal[True] = True
    completed_at: AwareDatetime
    release_class: Literal[ReleaseClass.RESTRICTED] = ReleaseClass.RESTRICTED

    @model_validator(mode="after")
    def completed_inventory_is_exact(self) -> Self:
        if len(self.bounded_output_receipt_hashes) != 24 or len(
            set(self.bounded_output_receipt_hashes)
        ) != 24:
            raise ValueError("completed case review requires 24 bounded ITT outputs")
        if len({item.context_id for item in self.units}) != 8:
            raise ValueError("completed case review requires eight distinct contexts")
        assessments = tuple(
            assessment
            for unit in self.units
            for assessment in (unit.primary, unit.secondary)
            if assessment is not None
        )
        if any(item.reviewed_at > self.completed_at for item in assessments):
            raise ValueError("completed review predates a reviewer assessment")
        return self


def validate_completed_case_study_review(
    *,
    plan: CaseStudyExecutionPlan,
    resume: CaseStudyResumeManifest,
    template: CaseStudyReviewInputTemplate,
    review: CaseStudyCompletedReview,
) -> None:
    """Validate human review coverage against terminal, lineage-audited ITT outputs."""

    status = audit_case_study_resume(plan, resume)
    if not status.complete:
        raise CaseStudyResumeError("case review requires all 25 terminal ITT outputs")
    expected_template = compile_case_review_input_template(plan)
    if template != expected_template:
        raise CaseStudyResumeError("case review template differs from the frozen plan")
    bounded = tuple(item for item in resume.output_receipts if not item.operational_only)
    receipts = {item.projection_job_id: item for item in bounded}
    if (
        review.execution_plan_hash != plan.content_hash
        or review.review_template_hash != template.content_hash
        or review.terminal_resume_manifest_hash != resume.content_hash
        or review.bounded_output_receipt_hashes
        != tuple(item.content_hash for item in bounded)
    ):
        raise CaseStudyResumeError("completed case review has different plan/output lineage")
    template_units = {item.context_id: item for item in template.units}
    review_units = {item.context_id: item for item in review.units}
    if set(review_units) != set(template_units):
        raise CaseStudyResumeError("completed case review does not cover the frozen contexts")
    latest_output = max(item.completed_at for item in bounded)
    for context_id, unit in review_units.items():
        expected = template_units[context_id]
        if (
            unit.review_unit_id != expected.review_unit_id
            or unit.window_id != expected.window_id
            or unit.evidence_binding_hash != expected.evidence_binding_hash
            or unit.packet_equality_group_id != expected.packet_equality_group_id
            or unit.required_output_job_ids != expected.required_output_job_ids
        ):
            raise CaseStudyResumeError("completed case review rewrites its blank unit")
        for assessment in (unit.primary, unit.secondary):
            if assessment is None:
                continue
            if assessment.reviewed_at <= latest_output:
                raise CaseStudyResumeError("case reviewer assessment predates terminal outputs")
            expected_pairs = {
                (job_id, dimension)
                for job_id in expected.required_output_job_ids
                for dimension in CaseReviewDimension
            }
            observed_pairs = {
                (item.projection_job_id, item.dimension) for item in assessment.judgments
            }
            if observed_pairs != expected_pairs:
                raise CaseStudyResumeError("case assessment has incomplete dimension coverage")
            for judgment in assessment.judgments:
                receipt = receipts.get(judgment.projection_job_id)
                if receipt is None or (
                    judgment.output_receipt_hash != receipt.content_hash
                    or judgment.condition is not receipt.condition
                ):
                    raise CaseStudyResumeError("case judgment cites another output")
                if receipt.terminal_outcome is not RunOutcome.SUCCEEDED and (
                    judgment.verdict is not CaseReviewVerdict.NOT_ASSESSABLE
                ):
                    raise CaseStudyResumeError("non-success ITT output must be not assessable")
            detailed_expected = expected.detailed_matching is not None
            if detailed_expected != bool(assessment.detailed_matching):
                raise CaseStudyResumeError("case detailed matching differs from frozen coverage")
            if detailed_expected:
                detailed = {item.projection_job_id: item for item in assessment.detailed_matching}
                if set(detailed) != set(expected.required_output_job_ids):
                    raise CaseStudyResumeError("case detailed matching omits a condition")
                for job_id, matching in detailed.items():
                    receipt = receipts[job_id]
                    if (
                        matching.output_receipt_hash != receipt.content_hash
                        or matching.condition is not receipt.condition
                    ):
                        raise CaseStudyResumeError("detailed matching cites another output")
                    if receipt.terminal_outcome is not RunOutcome.SUCCEEDED and any(
                        (
                            matching.output_assertion_count,
                            matching.matched_assertion_count,
                            matching.output_event_count,
                            matching.matched_event_count,
                        )
                    ):
                        raise CaseStudyResumeError(
                            "non-success detailed review cannot claim output-side objects"
                        )


def compile_case_review_input_template(
    plan: CaseStudyExecutionPlan,
) -> CaseStudyReviewInputTemplate:
    """Compile blank restricted review rows; never substitute researcher judgments."""

    units: list[CaseUnitReviewInputTemplate] = []
    for window in plan.windows:
        for index, context_id in enumerate(window.context_ids):
            units.append(
                CaseUnitReviewInputTemplate(
                    review_unit_id=f"review-{context_id}",
                    context_id=context_id,
                    window_id=window.window_id,
                    evidence_binding_hash=window.evidence_binding_hash,
                    packet_equality_group_id=window.packet_equality_group_id,
                    required_output_job_ids=(
                        window.c0_projection_job_ids[index],
                        window.c1_projection_job_ids[index],
                        window.c2_construction_call_ids[index],
                    ),
                    detailed_matching=(
                        CaseDetailedMatchingInputTemplate()
                        if context_id in plan.detailed_review_context_ids
                        else None
                    ),
                )
            )
    return CaseStudyReviewInputTemplate(
        template_id=f"review-template-{plan.execution_id}",
        execution_plan_hash=plan.content_hash,
        preregistration_hash=plan.preregistration_hash,
        units=tuple(units),  # type: ignore[arg-type]
        detailed_matching_context_ids=plan.detailed_review_context_ids,
    )


class PublicCaseStudyProgress(ImmutableRecord):
    """Safe progress-only view; contains no paths, packets, prose, or offsets."""

    execution_plan_hash: Sha256Digest
    registered_window_count: Literal[4] = 4
    registered_bounded_context_count: Literal[8] = 8
    registered_c0_output_count: Literal[8] = 8
    registered_c1_preconstruction_count: Literal[4] = 4
    registered_c1_output_count: Literal[8] = 8
    registered_c2_bounded_output_count: Literal[8] = 8
    registered_full_index_operational_count: Literal[1] = 1
    completed_output_count: int = Field(ge=0, le=25)
    non_succeeded_output_count: int = Field(ge=0, le=25)
    complete: bool
    full_index_result_interpretation: Literal[
        "operational_noncausal_not_in_same_evidence_comparison"
    ] = "operational_noncausal_not_in_same_evidence_comparison"
    public_payload_policy: Literal["hashes_counts_and_high_level_paraphrase_only"] = (
        "hashes_counts_and_high_level_paraphrase_only"
    )
    release_class: Literal[ReleaseClass.PUBLIC] = ReleaseClass.PUBLIC


def build_public_case_study_progress(
    status: CaseStudyResumeStatus,
) -> PublicCaseStudyProgress:
    return PublicCaseStudyProgress(
        execution_plan_hash=status.execution_plan_hash,
        completed_output_count=status.completed_output_count,
        non_succeeded_output_count=status.non_succeeded_output_count,
        complete=status.complete,
    )


def _resolve_restricted_destination(
    destination: Path,
    restricted_root: Path,
) -> tuple[Path, Path]:
    if not destination.is_absolute() or not restricted_root.is_absolute():
        raise CaseStudyResumeError("restricted destination/root must be explicit absolute paths")
    if destination.is_symlink() or restricted_root.is_symlink():
        raise CaseStudyResumeError("restricted records cannot use symbolic-link endpoints")
    try:
        root = restricted_root.resolve(strict=True)
        parent = destination.parent.resolve(strict=True)
    except OSError as error:
        raise CaseStudyResumeError("restricted record parent is unavailable") from error
    if not parent.is_relative_to(root):
        raise CaseStudyResumeError("restricted record must remain inside its explicit root")
    lexical_root = Path(os.path.abspath(restricted_root))
    lexical_parent = Path(os.path.abspath(destination.parent))
    try:
        relative_parent = lexical_parent.relative_to(lexical_root)
    except ValueError as error:
        raise CaseStudyResumeError("restricted record escapes its explicit root") from error
    cursor = lexical_root
    for part in relative_parent.parts:
        cursor /= part
        if cursor.is_symlink():
            raise CaseStudyResumeError("restricted record cannot traverse a symbolic link")
    return destination, root


def write_restricted_case_record(
    record: ImmutableRecord,
    destination: Path,
    *,
    restricted_root: Path,
) -> None:
    """Atomically create one restricted record without overwrite or path disclosure."""

    destination, _ = _resolve_restricted_destination(destination, restricted_root)
    payload = canonical_json(record).encode("utf-8") + b"\n"
    if destination.exists() or destination.is_symlink():
        if (
            destination.is_file()
            and not destination.is_symlink()
            and destination.read_bytes() == payload
        ):
            return
        raise CaseStudyResumeError("refusing to overwrite a restricted case record")
    descriptor, temporary_name = tempfile.mkstemp(
        dir=destination.parent,
        prefix=f".{destination.name}.",
        suffix=".tmp",
    )
    temporary = Path(temporary_name)
    try:
        os.fchmod(descriptor, 0o600)
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        try:
            os.link(temporary, destination)
        except FileExistsError:
            if (
                destination.is_symlink()
                or not destination.is_file()
                or destination.read_bytes() != payload
            ):
                raise CaseStudyResumeError(
                    "concurrent restricted record differs from the requested replay"
                ) from None
        directory_descriptor = os.open(
            destination.parent,
            os.O_RDONLY | getattr(os, "O_DIRECTORY", 0),
        )
        try:
            os.fsync(directory_descriptor)
        finally:
            os.close(directory_descriptor)
    except BaseException:
        raise
    finally:
        temporary.unlink(missing_ok=True)


def write_case_resume_manifest(
    resume: CaseStudyResumeManifest,
    destination_directory: Path,
    *,
    restricted_root: Path,
) -> Path:
    """Write one append-only, hash-named resume state and return its restricted path."""

    destination = destination_directory / f"{resume.content_hash}.json"
    write_restricted_case_record(resume, destination, restricted_root=restricted_root)
    return destination


def load_case_study_execution_plan(
    path: Path,
    *,
    restricted_root: Path,
) -> CaseStudyExecutionPlan:
    resolved = _resolve_exact_restricted_file(path, restricted_root, label="execution plan")
    return CaseStudyExecutionPlan.model_validate_json(resolved.read_bytes())


def load_case_study_admission_attestation(
    path: Path,
    *,
    restricted_root: Path,
) -> CaseStudyAdmissionAttestation:
    resolved = _resolve_exact_restricted_file(
        path,
        restricted_root,
        label="admission attestation",
    )
    return CaseStudyAdmissionAttestation.model_validate_json(resolved.read_bytes())


def load_case_study_semantic_admission_bundle(
    path: Path,
    *,
    restricted_root: Path,
) -> CaseStudySemanticAdmissionBundle:
    """Load one exact restricted typed-gate bundle."""

    resolved = _resolve_exact_restricted_file(
        path,
        restricted_root,
        label="semantic admission bundle",
    )
    return CaseStudySemanticAdmissionBundle.model_validate_json(resolved.read_bytes())


def load_completed_case_study_review(
    path: Path,
    *,
    restricted_root: Path,
) -> CaseStudyCompletedReview:
    """Load one exact restricted human-completed case review."""

    resolved = _resolve_exact_restricted_file(
        path,
        restricted_root,
        label="completed case review",
    )
    return CaseStudyCompletedReview.model_validate_json(resolved.read_bytes())


def load_case_study_resume_manifest(
    path: Path,
    *,
    restricted_root: Path,
) -> CaseStudyResumeManifest:
    resolved = _resolve_exact_restricted_file(path, restricted_root, label="resume manifest")
    return CaseStudyResumeManifest.model_validate_json(resolved.read_bytes())


class CaseC0PreparationEnvelope(ImmutableRecord):
    """Hash-only C0 input binding; protected evidence remains in memory."""

    execution_plan_hash: Sha256Digest
    window_plan_hash: Sha256Digest
    evidence_binding_hash: Sha256Digest
    snapshot_hash: Sha256Digest
    packet_hash: Sha256Digest
    query_context_supplied: Literal[False] = False
    backend: Literal["spacy-ner-dependency-plus-deterministic-rules-v2"] = (
        "spacy-ner-dependency-plus-deterministic-rules-v2"
    )
    release_class: Literal[ReleaseClass.RESTRICTED] = ReleaseClass.RESTRICTED


class CaseC0ProjectionEnvelope(ImmutableRecord):
    """Fixed post-query C0 projection binding with construction disabled."""

    execution_plan_hash: Sha256Digest
    window_plan_hash: Sha256Digest
    query_access_receipt_hash: Sha256Digest
    construction_seal_hash: Sha256Digest
    packet_hash: Sha256Digest
    ontology_construction_allowed: Literal[False] = False
    backend: Literal["spacy-ner-dependency-plus-deterministic-rules-v2"] = (
        "spacy-ner-dependency-plus-deterministic-rules-v2"
    )
    release_class: Literal[ReleaseClass.RESTRICTED] = ReleaseClass.RESTRICTED


class CaseC1RequestEnvelope(ImmutableRecord):
    """Hash-only envelope; protected packet is supplied separately in memory."""

    execution_plan_hash: Sha256Digest
    call_slot_hash: Sha256Digest
    evidence_binding_hash: Sha256Digest
    snapshot_hash: Sha256Digest
    packet_hash: Sha256Digest
    query_context_supplied: Literal[False] = False
    backend: Literal["vllm_gpu"] = "vllm_gpu"
    release_class: Literal[ReleaseClass.RESTRICTED] = ReleaseClass.RESTRICTED


class CaseC2RequestEnvelope(ImmutableRecord):
    """Hash-only envelope proving post-query, empty-inventory GPU construction."""

    execution_plan_hash: Sha256Digest
    call_slot_hash: Sha256Digest
    query_access_receipt_hash: Sha256Digest
    evidence_binding_hash: Sha256Digest
    packet_hash: Sha256Digest
    empty_inventory_hash: Sha256Digest
    inherited_ontology_hash: Literal[None] = None
    backend: Literal["vllm_gpu"] = "vllm_gpu"
    release_class: Literal[ReleaseClass.RESTRICTED] = ReleaseClass.RESTRICTED


@runtime_checkable
class StrictCaseStudyGpuAdapter(Protocol):
    """Production adapter protocol; CPU validation may reject, never add C2 semantics."""

    backend: Literal["vllm_gpu"]

    def preconstruct_c1(
        self,
        call: CaseGpuCallSlot,
        envelope: CaseC1RequestEnvelope,
        protected_packet: object,
    ) -> CasePrequeryReceipt:
        """Run one query-blind, GPU-backed C1 preconstruction."""

    def construct_c2(
        self,
        call: CaseGpuCallSlot,
        envelope: CaseC2RequestEnvelope,
        protected_packet: object,
        protected_query: object,
    ) -> CaseOutputReceipt:
        """Run one post-query GPU construction and return validated lineage."""


@runtime_checkable
class StrictCaseStudyClassicalAdapter(Protocol):
    """Production C0 protocol using the frozen credible rules and fixed projection."""

    backend: Literal["spacy-ner-dependency-plus-deterministic-rules-v2"]

    def prepare_c0(
        self,
        window: CaseWindowExecutionPlan,
        envelope: CaseC0PreparationEnvelope,
        protected_packet: object,
    ) -> CasePrequeryReceipt:
        """Construct the classical ontology query-blindly."""

    def project_c0(
        self,
        window: CaseWindowExecutionPlan,
        envelope: CaseC0ProjectionEnvelope,
        protected_query: object,
    ) -> CaseOutputReceipt:
        """Project from the exact C0 seal without construction."""


def case_study_runtime_contract_hashes() -> dict[str, str]:
    """Return public schema hashes without touching a corpus or result artifact."""

    contracts = (
        CaseStudyRuntimePolicy,
        CaseStudyInputAttestation,
        CaseStudyAdmissionAttestation,
        CaseStudySemanticAdmissionBundle,
        CaseStudyExecutionPlan,
        CasePrequeryReceipt,
        CasePrequeryBarrierReceipt,
        CaseQueryAccessReceipt,
        CaseOutputReceipt,
        CaseStudyResumeManifest,
        CaseStudyReviewInputTemplate,
        CaseStudyCompletedReview,
        PublicCaseStudyProgress,
    )
    return {
        contract.__name__: canonical_sha256(contract.model_json_schema(mode="validation"))
        for contract in contracts
    }


__all__ = [
    "CASE_SEMANTIC_NATIVE_ARTIFACT_ROLES",
    "AttestedRestrictedCaseStudy",
    "AttestedSelectedModelFreeze",
    "CaseBlindedErrorReviewGate",
    "CaseC0PreparationEnvelope",
    "CaseC0ProjectionEnvelope",
    "CaseC1RequestEnvelope",
    "CaseC2RequestEnvelope",
    "CaseCompletedUnitReview",
    "CaseDetailedMatchSummary",
    "CaseDetailedMatchingInputTemplate",
    "CaseDimensionJudgment",
    "CaseGoldFirewallGate",
    "CaseGpuCallRole",
    "CaseGpuCallSlot",
    "CaseGpuScheduleGate",
    "CaseMetricRegenerationGate",
    "CaseModelRuntimeBinding",
    "CaseNativeGateArtifactReference",
    "CaseOutputReceipt",
    "CasePrequeryBarrierReceipt",
    "CasePrequeryKind",
    "CasePrequeryReceipt",
    "CasePublicReleaseScanGate",
    "CaseQueryAccessReceipt",
    "CaseReviewDimension",
    "CaseReviewRole",
    "CaseReviewVerdict",
    "CaseReviewerAssessment",
    "CaseStoragePreflightGate",
    "CaseStudyAdmissionAttestation",
    "CaseStudyCompletedReview",
    "CaseStudyExecutionPlan",
    "CaseStudyInputAttestation",
    "CaseStudyInputError",
    "CaseStudyPlanError",
    "CaseStudyResumeError",
    "CaseStudyResumeManifest",
    "CaseStudyResumeStatus",
    "CaseStudyReviewInputTemplate",
    "CaseStudyRuntimePolicy",
    "CaseStudySemanticAdmissionBundle",
    "CaseSyntheticClosureGate",
    "CaseTimingLineageGate",
    "CaseUnitReviewInputTemplate",
    "CaseWindowExecutionPlan",
    "OperationalCaseExecutionPlan",
    "PublicCaseStudyProgress",
    "RestrictedArtifactStoragePlan",
    "StrictCaseStudyClassicalAdapter",
    "StrictCaseStudyGpuAdapter",
    "audit_case_study_resume",
    "build_public_case_study_progress",
    "case_study_runtime_contract_hashes",
    "compile_case_review_input_template",
    "compile_case_study_execution_plan",
    "initialize_case_study_resume",
    "load_attested_restricted_case_study",
    "load_attested_selected_model_freeze",
    "load_case_study_admission_attestation",
    "load_case_study_execution_plan",
    "load_case_study_resume_manifest",
    "load_case_study_semantic_admission_bundle",
    "load_completed_case_study_review",
    "validate_case_study_semantic_admission",
    "validate_completed_case_study_review",
    "validate_resume_successor",
    "write_case_resume_manifest",
    "write_restricted_case_record",
]
