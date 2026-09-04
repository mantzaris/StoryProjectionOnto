"""Production execution controller for the bounded first-novel transfer study.

The controller is intentionally restricted-data first: packets, requests, raw model
responses, and projections are written only to the restricted CAS.  Bounded window
packets are reconstructed in memory from the attested SQLite index and are never
serialized as prose-bearing run artifacts.  The operational FTS packet is created
only after its query-access event and retains a separate rank/omission receipt.

This module owns one case-study model-service lifecycle.  It may start the selected
service once, can adopt the exact still-live process from its private checkpoint,
and always performs the terminal shutdown.  It cannot silently start a second model
load after a stopped or failed case session.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import tempfile
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from enum import StrEnum
from pathlib import Path
from typing import Any, Literal, Protocol, Self, cast

from pydantic import AwareDatetime, Field, model_validator

from story_projection_onto.case_study_runtime import (
    AttestedRestrictedCaseStudy,
    CaseC0PreparationEnvelope,
    CaseC0ProjectionEnvelope,
    CaseC1RequestEnvelope,
    CaseC2RequestEnvelope,
    CaseGpuCallSlot,
    CaseOutputReceipt,
    CasePrequeryBarrierReceipt,
    CasePrequeryKind,
    CasePrequeryReceipt,
    CaseQueryAccessReceipt,
    CaseStudyAdmissionAttestation,
    CaseStudyExecutionPlan,
    CaseStudyResumeManifest,
    CaseStudyResumeStatus,
    CaseWindowExecutionPlan,
    audit_case_study_resume,
    compile_case_review_input_template,
    initialize_case_study_resume,
    validate_resume_successor,
)
from story_projection_onto.conditions.base import (
    SCORED_PROJECTION_SCHEMA_HASH,
    ConditionPreparation,
    RunConditionConfig,
)
from story_projection_onto.conditions.c0 import (
    ClassicalPreBuilder,
    load_production_classical_builder,
    project_sealed_c0,
)
from story_projection_onto.conditions.c1 import build_c1_preconstruction_request
from story_projection_onto.conditions.c2 import (
    prepare_empty_c2_inventory,
)
from story_projection_onto.contracts import (
    ConditionName,
    EvidencePacket,
    EvidenceSnapshot,
    ImmutableRecord,
    PacketMaterializationEvent,
    PreQueryInventory,
    QueryAccessEvent,
    QueryContext,
    ReleaseClass,
    RetrievalMethod,
    RunOutcome,
    Sha256Digest,
    UpperOntology,
    canonical_json,
    canonical_sha256,
    to_model_visible_query,
)
from story_projection_onto.development_adapter import (
    DevelopmentConstructionConfiguration,
    PackingTokenizer,
    build_development_guided_request,
    development_runtime_identifiers,
    encode_development_semantic_request,
)
from story_projection_onto.gpu_runtime import (
    GenerationResult,
    GuidedJSONRequest,
    ServiceState,
    TokenizerManifest,
)
from story_projection_onto.llm import CapabilityManifest
from story_projection_onto.manifest import build_source_manifest
from story_projection_onto.novel_case import (
    OperationalRetrievalResult,
    WindowEvidenceBundle,
    materialize_window_evidence,
    retrieve_operational_packet,
)
from story_projection_onto.store import (
    ArtifactStore,
    Ledger,
    PacketMaterializationRecord,
    PrequeryBarrierRecord,
    QueryAccessRecord,
)
from story_projection_onto.store import (
    ReleaseClass as LedgerReleaseClass,
)

CASE_BASE_WATCHDOG_SECONDS = 4 * 240 + 9 * 150
CASE_REQUIRED_NEXT_REPAIR_SECONDS = 240
SCHEDULED_GPU_LIMIT_SECONDS = 9 * 60 * 60
HARD_GPU_LIMIT_SECONDS = 10 * 60 * 60
CASE_SOURCE_REVISION = "case-study-production-controller-v1"


class CaseStudyExecutionError(RuntimeError):
    """A production case execution or immutable lineage invariant failed."""


class CaseStudyAdmissionError(CaseStudyExecutionError):
    """Required pre-case evidence or cumulative-resource admission failed."""


class CaseStudyExecutionPhase(StrEnum):
    INITIALIZED = "initialized"
    PREQUERY_CPU = "prequery_cpu"
    SERVICE_LIVE = "service_live"
    PREQUERY_COMPLETE = "prequery_complete"
    QUERY_OUTPUTS = "query_outputs"
    COMPLETED = "completed"
    FAILED = "failed"


class CaseArtifactReference(ImmutableRecord):
    """Bind a logical immutable object to its restricted CAS serialization."""

    logical_content_hash: Sha256Digest
    artifact_hash: Sha256Digest
    object_kind: str = Field(pattern=r"^[a-z][a-z0-9_]*$")


class CaseAdmissionEvidenceReference(ImmutableRecord):
    name: Literal[
        "synthetic_run_closure_hash",
        "timing_lineage_audit_hash",
        "gold_firewall_audit_hash",
        "registered_metric_regeneration_hash",
        "blinded_error_review_hash",
        "storage_preflight_hash",
        "gpu_schedule_admission_hash",
        "public_release_scan_hash",
    ]
    logical_content_hash: Sha256Digest
    artifact_hash: Sha256Digest


class CaseAdmissionEvidenceBundle(ImmutableRecord):
    """Concrete CAS routing for every boolean/hash pre-case gate."""

    bundle_id: str = Field(min_length=1)
    admission_attestation_hash: Sha256Digest
    evidence: tuple[CaseAdmissionEvidenceReference, ...]
    frozen_at: AwareDatetime
    release_class: Literal[ReleaseClass.RESTRICTED] = ReleaseClass.RESTRICTED

    @model_validator(mode="after")
    def exact_gate_inventory(self) -> Self:
        expected = {
            "synthetic_run_closure_hash",
            "timing_lineage_audit_hash",
            "gold_firewall_audit_hash",
            "registered_metric_regeneration_hash",
            "blinded_error_review_hash",
            "storage_preflight_hash",
            "gpu_schedule_admission_hash",
            "public_release_scan_hash",
        }
        names = tuple(item.name for item in self.evidence)
        if set(names) != expected or len(names) != len(expected):
            raise ValueError("case admission bundle requires every gate exactly once")
        return self


class CaseExecutionAdmissionReceipt(ImmutableRecord):
    """Fail-closed admission bound to the existing cumulative GPU ledger."""

    receipt_id: str = Field(min_length=1)
    execution_plan_hash: Sha256Digest
    admission_attestation_hash: Sha256Digest
    admission_evidence_bundle: CaseArtifactReference
    source_manifest: CaseArtifactReference
    source_revision: str = Field(min_length=1)
    construction_configuration_file_sha256: Sha256Digest
    construction_configuration_hash: Sha256Digest
    predecessor_ledger_sha256: Sha256Digest
    prior_gpu_event_inventory_hash: Sha256Digest
    allocated_gpu_seconds_before_case: float = Field(ge=0.0)
    case_base_watchdog_seconds: Literal[2310] = CASE_BASE_WATCHDOG_SECONDS
    required_next_repair_seconds: Literal[240] = CASE_REQUIRED_NEXT_REPAIR_SECONDS
    scheduled_gpu_limit_seconds: Literal[32400] = SCHEDULED_GPU_LIMIT_SECONDS
    hard_gpu_limit_seconds: Literal[36000] = HARD_GPU_LIMIT_SECONDS
    admitted_at: AwareDatetime
    existing_cumulative_ledger_required: Literal[True] = True
    second_case_model_load_forbidden: Literal[True] = True
    release_class: Literal[ReleaseClass.RESTRICTED] = ReleaseClass.RESTRICTED

    @model_validator(mode="after")
    def schedule_has_required_margin(self) -> Self:
        forecast = (
            self.allocated_gpu_seconds_before_case
            + self.case_base_watchdog_seconds
            + self.required_next_repair_seconds
        )
        if forecast > self.scheduled_gpu_limit_seconds:
            raise ValueError("case base schedule plus required repair exceeds nine hours")
        if forecast >= self.hard_gpu_limit_seconds:
            raise ValueError("case admission reaches the ten-hour hard stop")
        return self


class CaseBoundedPacketReceipt(ImmutableRecord):
    """Prose-free receipt for a bounded packet retained only in memory."""

    receipt_id: str = Field(min_length=1)
    execution_plan_hash: Sha256Digest
    window_id: str = Field(min_length=1)
    evidence_binding_hash: Sha256Digest
    snapshot_hash: Sha256Digest
    packet_hash: Sha256Digest
    ordered_evidence_ids_hash: Sha256Digest
    evidence_count: int = Field(gt=0)
    snapshot_created_at: AwareDatetime
    snapshot_sealed_at: AwareDatetime
    packet_created_at: AwareDatetime
    prose_persisted: Literal[False] = False
    retrieval_method: Literal[RetrievalMethod.ALL_ADMISSIBLE] = RetrievalMethod.ALL_ADMISSIBLE
    release_class: Literal[ReleaseClass.RESTRICTED] = ReleaseClass.RESTRICTED

    @model_validator(mode="after")
    def chronology_is_query_blind(self) -> Self:
        if not (self.snapshot_created_at <= self.snapshot_sealed_at <= self.packet_created_at):
            raise ValueError("bounded packet timestamps are not ordered")
        return self


class CasePackedRequestReceipt(ImmutableRecord):
    call_id: str = Field(min_length=1)
    call_slot_hash: Sha256Digest
    semantic_request_hash: Sha256Digest
    wire_alias_manifest: CaseArtifactReference
    rendered_request_hash: Sha256Digest
    packing_report: CaseArtifactReference
    rendered_input_tokens: int = Field(gt=0, le=10240)
    maximum_input_tokens: Literal[10240] = 10240
    reserved_output_tokens: Literal[2048] = 2048


class CaseGpuPackingPreflightReceipt(ImmutableRecord):
    receipt_id: str = Field(min_length=1)
    execution_plan_hash: Sha256Digest
    model_runtime_hash: Sha256Digest
    tokenizer_manifest_hash: Sha256Digest
    c1_requests: tuple[
        CasePackedRequestReceipt,
        CasePackedRequestReceipt,
        CasePackedRequestReceipt,
        CasePackedRequestReceipt,
    ]
    c2_requests_must_be_packed_after_audited_query_access: Literal[True] = True
    truncation_allowed: Literal[False] = False
    passed: Literal[True] = True
    completed_at: AwareDatetime
    release_class: Literal[ReleaseClass.RESTRICTED] = ReleaseClass.RESTRICTED

    @model_validator(mode="after")
    def exact_c1_inventory(self) -> Self:
        if len({item.call_id for item in self.c1_requests}) != 4:
            raise ValueError("case packing preflight requires four unique C1 calls")
        return self


@dataclass(frozen=True, slots=True)
class PreparedCaseC1Requests:
    receipt: CaseGpuPackingPreflightReceipt
    receipt_reference: CaseArtifactReference
    semantic_requests: Mapping[str, object] = field(repr=False)
    guided_requests: Mapping[str, GuidedJSONRequest] = field(repr=False)


def preflight_case_c1_requests(
    *,
    root: Path,
    plan: CaseStudyExecutionPlan,
    construction: DevelopmentConstructionConfiguration,
    tokenizer: PackingTokenizer,
    tokenizer_manifest: TokenizerManifest,
    bounded_packets: Mapping[str, WindowEvidenceBundle],
    artifacts: ArtifactStore,
    clock: Callable[[], datetime],
) -> PreparedCaseC1Requests:
    """Pack all four complete C1 requests before the sole model-service start."""

    if tokenizer_manifest.manifest_sha256 != plan.model_runtime.selected_tokenizer_manifest_hash:
        raise CaseStudyAdmissionError("case tokenizer differs from selected-model freeze")
    c1_slots = tuple(
        item for item in plan.gpu_call_slots if item.condition is ConditionName.C1_LLM_PRE
    )
    if len(c1_slots) != 4:
        raise CaseStudyExecutionError("case plan changed its four C1 calls")
    rows: list[CasePackedRequestReceipt] = []
    semantics: dict[str, object] = {}
    guided_requests: dict[str, GuidedJSONRequest] = {}
    request_times: list[datetime] = []
    for call in c1_slots:
        if call.window_id is None or call.window_id not in bounded_packets:
            raise CaseStudyExecutionError("C1 preflight lacks its bounded window packet")
        bundle = bounded_packets[call.window_id]
        snapshot = bundle.snapshot_assembly.snapshot
        runtime = development_runtime_identifiers(
            root=root,
            condition=ConditionName.C1_LLM_PRE,
            tokenizer_manifest=tokenizer_manifest,
            seed=call.resolved_vllm_seed,
        )
        # A deterministic post-seal time makes the complete query-blind request
        # byte-identical when a controller is restarted before query reveal.
        requested = snapshot.sealed_at + timedelta(microseconds=1)
        request_times.append(requested)
        semantic = build_c1_preconstruction_request(
            snapshot_hash=snapshot.content_hash,
            snapshot_sealed_at=snapshot.sealed_at,
            ordered_snapshot_evidence_ids=snapshot.eligible_evidence_ids,
            evidence=bundle.snapshot_assembly.admissible_evidence,
            upper_ontology=construction.upper_ontology,
            preconstruction_budgets=construction.preconstruction_budgets,
            runtime=runtime,
            requested_at=requested,
        )
        guided = build_development_guided_request(
            root=root,
            call_id=call.call_id,
            semantic_request=semantic,
            tokenizer=tokenizer,
            tokenizer_manifest=tokenizer_manifest,
            seed=call.resolved_vllm_seed,
        )
        if (
            guided.rendered_input_token_count > plan.model_runtime.maximum_input_tokens
            or guided.decoding.maximum_output_tokens != plan.model_runtime.maximum_output_tokens
            or runtime.output_schema_hash != plan.model_runtime.output_schema_hash
            or CapabilityManifest.for_condition(ConditionName.C1_LLM_PRE).content_hash
            != plan.model_runtime.c1_capability_manifest_hash
        ):
            raise CaseStudyAdmissionError("packed C1 request differs from case runtime")
        encoded = encode_development_semantic_request(semantic)
        alias_ref = _persist_record(
            artifacts,
            encoded.alias_manifest,
            object_kind="case_model_wire_alias_manifest",
            created_at=requested,
        )
        packing_ref = _persist_record(
            artifacts,
            guided.packing,
            object_kind="case_packing_report",
            created_at=requested,
        )
        rows.append(
            CasePackedRequestReceipt(
                call_id=call.call_id,
                call_slot_hash=call.content_hash,
                semantic_request_hash=semantic.content_hash,
                wire_alias_manifest=alias_ref,
                rendered_request_hash=guided.request_hash,
                packing_report=packing_ref,
                rendered_input_tokens=guided.rendered_input_token_count,
            )
        )
        semantics[call.call_id] = semantic
        guided_requests[call.call_id] = guided
    latest_request = max(request_times)
    if _now(clock) < latest_request:
        raise CaseStudyExecutionError("case packing clock predates sealed evidence")
    receipt = CaseGpuPackingPreflightReceipt(
        receipt_id=f"packing-preflight-{plan.execution_id}",
        execution_plan_hash=plan.content_hash,
        model_runtime_hash=plan.model_runtime.content_hash,
        tokenizer_manifest_hash=tokenizer_manifest.manifest_sha256,
        c1_requests=tuple(rows),  # type: ignore[arg-type]
        completed_at=latest_request + timedelta(microseconds=1),
    )
    receipt_ref = _persist_record(
        artifacts,
        receipt,
        object_kind="case_gpu_packing_preflight",
        created_at=receipt.completed_at,
    )
    return PreparedCaseC1Requests(
        receipt=receipt,
        receipt_reference=receipt_ref,
        semantic_requests=semantics,
        guided_requests=guided_requests,
    )


class CaseRepairDiagnostic(ImmutableRecord):
    code: str = Field(min_length=1)
    path: str = Field(min_length=1)
    message: str = Field(min_length=1)
    related_ids: tuple[str, ...] = ()


class CaseRepairInput(ImmutableRecord):
    parent_attempt_id: str = Field(min_length=1)
    semantic_request_hash: Sha256Digest
    parent_raw_output_hash: Sha256Digest
    invalid_draft: Mapping[str, object]
    diagnostics: tuple[CaseRepairDiagnostic, ...]
    created_at: AwareDatetime

    @model_validator(mode="after")
    def has_fact_free_diagnostics(self) -> Self:
        if not self.diagnostics:
            raise ValueError("repair requires at least one mechanical diagnostic")
        if len({item.code for item in self.diagnostics}) != len(self.diagnostics):
            raise ValueError("repair diagnostic codes must be unique")
        return self


class CaseGpuCallAuditReceipt(ImmutableRecord):
    receipt_id: str = Field(min_length=1)
    execution_plan_hash: Sha256Digest
    call_slot_hash: Sha256Digest
    call_id: str = Field(min_length=1)
    condition: Literal[ConditionName.C1_LLM_PRE, ConditionName.C2_LLM_QUERY]
    job_id: str = Field(min_length=1)
    base_attempt_id: str = Field(min_length=1)
    base_model_call_id: str = Field(min_length=1)
    base_gpu_event_id: str = Field(min_length=1)
    service_identity: CaseArtifactReference
    request_envelope_hash: Sha256Digest
    stage_manifest_hash: Sha256Digest
    semantic_request_hash: Sha256Digest
    wire_alias_manifest: CaseArtifactReference
    rendered_request_hash: Sha256Digest
    packing_report: CaseArtifactReference
    decoding_manifest: CaseArtifactReference
    capability_manifest: CaseArtifactReference
    run_configuration: CaseArtifactReference
    query_access_receipt: CaseArtifactReference | None = None
    prequery_barrier: CaseArtifactReference | None = None
    raw_base_artifact_hash: Sha256Digest
    validation_ids: tuple[str, ...]
    terminal_outcome: RunOutcome
    terminal_receipt_hash: Sha256Digest
    validated_generation: CaseArtifactReference | None = None
    condition_result: CaseArtifactReference | None = None
    repair_attempt_id: str | None = None
    repair_model_call_id: str | None = None
    repair_gpu_event_id: str | None = None
    repair_input: CaseArtifactReference | None = None
    repair_semantic_request_hash: Sha256Digest | None = None
    repair_wire_alias_manifest: CaseArtifactReference | None = None
    repair_rendered_request_hash: Sha256Digest | None = None
    repair_packing_report: CaseArtifactReference | None = None
    repair_decoding_manifest: CaseArtifactReference | None = None
    repair_run_configuration: CaseArtifactReference | None = None
    raw_repair_artifact_hash: Sha256Digest | None = None
    repair_preservation_report: CaseArtifactReference | None = None
    failure_lineage_hash: Sha256Digest | None = None
    created_at: AwareDatetime
    release_class: Literal[ReleaseClass.RESTRICTED] = ReleaseClass.RESTRICTED

    @model_validator(mode="after")
    def terminal_and_repair_lineage(self) -> Self:
        if not self.validation_ids or len(set(self.validation_ids)) != len(self.validation_ids):
            raise ValueError("GPU call requires unique validation ledger rows")
        succeeded = self.terminal_outcome is RunOutcome.SUCCEEDED
        if succeeded != (self.validated_generation is not None):
            raise ValueError("validated generation must exactly match successful outcome")
        if succeeded != (self.condition_result is not None):
            raise ValueError("condition result must exactly match successful outcome")
        if succeeded != (self.failure_lineage_hash is None):
            raise ValueError("failed GPU call requires failure lineage")
        query_fields = (self.query_access_receipt, self.prequery_barrier)
        if self.condition is ConditionName.C1_LLM_PRE and any(
            item is not None for item in query_fields
        ):
            raise ValueError("query-blind C1 audit cannot contain query-access lineage")
        if self.condition is ConditionName.C2_LLM_QUERY and any(
            item is None for item in query_fields
        ):
            raise ValueError("query-time C2 audit requires query-access and barrier lineage")
        repair_fields = (
            self.repair_attempt_id,
            self.repair_model_call_id,
            self.repair_gpu_event_id,
            self.repair_input,
            self.repair_semantic_request_hash,
            self.repair_wire_alias_manifest,
            self.repair_rendered_request_hash,
            self.repair_packing_report,
            self.repair_decoding_manifest,
            self.repair_run_configuration,
            self.raw_repair_artifact_hash,
        )
        used_repair = any(item is not None for item in repair_fields)
        if used_repair and any(item is None for item in repair_fields):
            raise ValueError("repair attempt, request, GPU event, input, and raw output must agree")
        if self.repair_preservation_report is not None and not used_repair:
            raise ValueError("repair preservation report lacks repair lineage")
        return self


class CaseCpuOutputAuditReceipt(ImmutableRecord):
    receipt_id: str = Field(min_length=1)
    execution_plan_hash: Sha256Digest
    projection_job_id: str = Field(min_length=1)
    condition: Literal[ConditionName.C0_CLASSICAL_PRE, ConditionName.C1_LLM_PRE]
    job_id: str = Field(min_length=1)
    attempt_id: str = Field(min_length=1)
    query_access_receipt: CaseArtifactReference
    packet_hash: Sha256Digest
    preparation: CaseArtifactReference
    run_configuration: CaseArtifactReference
    condition_attempt: CaseArtifactReference | None = None
    projection: CaseArtifactReference | None = None
    validation_ids: tuple[str, ...]
    terminal_outcome: RunOutcome
    failure_lineage_hash: Sha256Digest | None = None
    created_at: AwareDatetime
    release_class: Literal[ReleaseClass.RESTRICTED] = ReleaseClass.RESTRICTED

    @model_validator(mode="after")
    def terminal_shape(self) -> Self:
        succeeded = self.terminal_outcome is RunOutcome.SUCCEEDED
        if succeeded != (self.condition_attempt is not None and self.projection is not None):
            raise ValueError("CPU success requires its attempt and projection")
        if succeeded != (self.failure_lineage_hash is None):
            raise ValueError("failed CPU projection requires failure lineage")
        if not self.validation_ids:
            raise ValueError("CPU output requires validation ledger lineage")
        return self


class CaseOperationalPacketAuditReceipt(ImmutableRecord):
    receipt_id: str = Field(min_length=1)
    execution_plan_hash: Sha256Digest
    query_access_event: CaseArtifactReference
    query_access_receipt: CaseArtifactReference
    operational_retrieval_receipt: CaseArtifactReference
    packet_materialization_event: CaseArtifactReference
    packet_hash: Sha256Digest
    retrieved_after_query_access: Literal[True] = True
    causal_comparison_eligible: Literal[False] = False
    created_at: AwareDatetime
    release_class: Literal[ReleaseClass.RESTRICTED] = ReleaseClass.RESTRICTED


class CaseBlankReviewHandoff(ImmutableRecord):
    handoff_id: str = Field(min_length=1)
    execution_plan_hash: Sha256Digest
    terminal_resume_manifest_hash: Sha256Digest
    output_receipt_hashes: tuple[Sha256Digest, ...]
    blank_review_template: CaseArtifactReference
    reviewer_judgment_count: Literal[0] = 0
    output_count: Literal[25] = 25
    operational_output_excluded_from_bounded_review: Literal[True] = True
    created_at: AwareDatetime
    release_class: Literal[ReleaseClass.RESTRICTED] = ReleaseClass.RESTRICTED

    @model_validator(mode="after")
    def exact_outputs(self) -> Self:
        if len(self.output_receipt_hashes) != 25 or len(set(self.output_receipt_hashes)) != 25:
            raise ValueError("review handoff requires exactly 25 unique ITT outputs")
        return self


class CaseControllerState(ImmutableRecord):
    """Append-only CAS controller state; a private pointer selects the latest hash."""

    state_id: str = Field(min_length=1)
    sequence_number: int = Field(ge=0)
    parent_state_hash: Sha256Digest | None = None
    execution_plan_hash: Sha256Digest
    admission_receipt: CaseArtifactReference
    resume_manifest: CaseArtifactReference
    phase: CaseStudyExecutionPhase
    bounded_packet_receipts: Mapping[str, CaseArtifactReference] = Field(default_factory=dict)
    preparation_artifacts: Mapping[str, CaseArtifactReference] = Field(default_factory=dict)
    query_access_event_artifacts: Mapping[str, CaseArtifactReference] = Field(default_factory=dict)
    query_access_receipt_artifacts: Mapping[str, CaseArtifactReference] = Field(
        default_factory=dict
    )
    packet_materialization_artifacts: Mapping[str, CaseArtifactReference] = Field(
        default_factory=dict
    )
    gpu_call_audits: Mapping[str, CaseArtifactReference] = Field(default_factory=dict)
    cpu_output_audits: Mapping[str, CaseArtifactReference] = Field(default_factory=dict)
    model_service_start_count: Literal[0, 1] = 0
    model_load_count: Literal[0, 1] = 0
    model_service_shutdown_count: Literal[0, 1] = 0
    model_service_live: bool = False
    service_configuration_hash: Sha256Digest | None = None
    failure_artifact_hash: Sha256Digest | None = None
    review_handoff: CaseArtifactReference | None = None
    updated_at: AwareDatetime
    release_class: Literal[ReleaseClass.RESTRICTED] = ReleaseClass.RESTRICTED

    @model_validator(mode="after")
    def monotonic_local_shape(self) -> Self:
        if self.sequence_number == 0 and self.parent_state_hash is not None:
            raise ValueError("initial case controller state cannot have a parent")
        if self.sequence_number > 0 and self.parent_state_hash is None:
            raise ValueError("case controller successor requires its parent")
        if self.model_load_count != self.model_service_start_count:
            raise ValueError("one case service start must equal one model load")
        if self.model_service_shutdown_count > self.model_service_start_count:
            raise ValueError("case service cannot shut down before it starts")
        if self.model_service_live and (
            self.model_service_start_count != 1 or self.model_service_shutdown_count != 0
        ):
            raise ValueError("live case service state has inconsistent lifecycle counts")
        if self.phase is CaseStudyExecutionPhase.COMPLETED and (
            self.model_service_start_count != 1
            or self.model_service_shutdown_count != 1
            or self.model_service_live
            or self.review_handoff is None
        ):
            raise ValueError("completed case state requires one stopped load and review handoff")
        return self


class CaseStudyExecutionResult(ImmutableRecord):
    execution_plan_hash: Sha256Digest
    admission_receipt_hash: Sha256Digest
    terminal_state_hash: Sha256Digest
    terminal_resume_manifest_hash: Sha256Digest
    status: CaseStudyResumeStatus
    review_handoff_hash: Sha256Digest | None = None
    model_service_start_count: Literal[0, 1]
    model_load_count: Literal[0, 1]
    model_service_shutdown_count: Literal[0, 1]
    model_service_stopped: bool
    allocated_gpu_seconds_before_case: float = Field(ge=0.0)
    allocated_gpu_seconds_after_case: float = Field(ge=0.0)
    public_scientific_outputs_produced: Literal[False] = False
    restricted_only: Literal[True] = True
    completed_at: AwareDatetime
    release_class: Literal[ReleaseClass.RESTRICTED] = ReleaseClass.RESTRICTED


class OwnedCaseModelService(Protocol):
    """Lifecycle-capable service held only by the case-study controller."""

    state: ServiceState

    @property
    def pid(self) -> int: ...

    @property
    def actual_allocated_service_seconds(self) -> float: ...

    @property
    def configuration(self) -> object: ...

    def start(
        self,
        *,
        session_id: str,
        event_id: str,
        watchdog_seconds: float,
        remaining_required_seconds: float = 0,
    ) -> None: ...

    def resume_from_checkpoint(self, path: Path) -> bool: ...

    def write_resume_checkpoint(self, path: Path) -> None: ...

    def generate(
        self,
        request: GuidedJSONRequest,
        *,
        event_id: str,
        watchdog_seconds: float,
        repair: bool = False,
        job_id: str | None = None,
        attempt_id: str | None = None,
        remaining_required_seconds: float = 0,
        accounting_details: Mapping[str, object] | None = None,
    ) -> GenerationResult: ...

    def shutdown(self) -> object | None: ...


class OperationalPacketProvider(Protocol):
    def __call__(
        self,
        *,
        query_accessed_at: datetime,
        packet_created_at: datetime,
    ) -> OperationalRetrievalResult: ...


def _now(clock: Callable[[], datetime]) -> datetime:
    value = clock()
    if value.tzinfo is None or value.utcoffset() is None:
        raise CaseStudyExecutionError("case controller clock must be timezone-aware")
    return value


def _after(clock: Callable[[], datetime], threshold: datetime) -> datetime:
    value = _now(clock)
    if value <= threshold:
        return threshold + timedelta(microseconds=1)
    return value


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _persist_record(
    artifacts: ArtifactStore,
    record: ImmutableRecord,
    *,
    object_kind: str,
    created_at: datetime,
) -> CaseArtifactReference:
    artifact = artifacts.put_bytes(
        canonical_json(record).encode("utf-8"),
        media_type="application/json",
        release_class=LedgerReleaseClass.RESTRICTED,
        created_at=created_at,
    )
    return CaseArtifactReference(
        logical_content_hash=record.content_hash,
        artifact_hash=artifact.content_hash,
        object_kind=object_kind,
    )


def _persist_mapping(
    artifacts: ArtifactStore,
    value: Mapping[str, object],
    *,
    object_kind: str,
    created_at: datetime,
) -> CaseArtifactReference:
    payload = canonical_json(value).encode("utf-8")
    artifact = artifacts.put_bytes(
        payload,
        media_type="application/json",
        release_class=LedgerReleaseClass.RESTRICTED,
        created_at=created_at,
    )
    return CaseArtifactReference(
        logical_content_hash=canonical_sha256(value),
        artifact_hash=artifact.content_hash,
        object_kind=object_kind,
    )


def _read_reference(
    artifacts: ArtifactStore,
    reference: CaseArtifactReference,
) -> bytes:
    record = artifacts.ledger.get_artifact(reference.artifact_hash)
    return artifacts.blobs.read_bytes(record, allow_restricted=True)


def _parse_record(
    artifacts: ArtifactStore,
    reference: CaseArtifactReference,
    cls: type[ImmutableRecord],
) -> ImmutableRecord:
    value = cls.model_validate_json(_read_reference(artifacts, reference))
    if value.content_hash != reference.logical_content_hash:
        raise CaseStudyExecutionError("CAS logical record hash changed")
    return value


def _logical_json_hash(value: Mapping[str, object]) -> str:
    for hash_field in ("content_hash", "manifest_sha256"):
        advertised = value.get(hash_field)
        if isinstance(advertised, str):
            payload = {key: item for key, item in value.items() if key != hash_field}
            observed = canonical_sha256(payload)
            if advertised != observed:
                raise CaseStudyAdmissionError(f"admission artifact has invalid {hash_field}")
            return observed
    return canonical_sha256(value)


def _gpu_inventory_hash(ledger: Ledger) -> str:
    return canonical_sha256(
        tuple(
            {
                "event_id": event.event_id,
                "event_kind": event.event_kind.value,
                "allocated_microseconds": event.allocated_microseconds,
                "started_at": event.started_at,
                "ended_at": event.ended_at,
                "succeeded": event.succeeded,
                "job_id": event.job_id,
                "attempt_id": event.attempt_id,
                "details_json": event.details_json,
            }
            for event in ledger.gpu_events()
        )
    )


def _total_allocated_seconds(ledger: Ledger) -> float:
    return math.fsum(event.allocated_seconds for event in ledger.gpu_events())


def validate_case_admission_evidence(
    *,
    admission: CaseStudyAdmissionAttestation,
    bundle: CaseAdmissionEvidenceBundle,
    artifacts: ArtifactStore,
) -> None:
    """Re-read every admitted gate from CAS; booleans alone are insufficient."""

    if bundle.admission_attestation_hash != admission.content_hash:
        raise CaseStudyAdmissionError("admission evidence belongs to another attestation")
    for reference in bundle.evidence:
        expected = cast(str, getattr(admission, reference.name))
        if reference.logical_content_hash != expected:
            raise CaseStudyAdmissionError(f"admission hash mismatch: {reference.name}")
        artifact_record = artifacts.ledger.get_artifact(reference.artifact_hash)
        if artifact_record.release_class is not LedgerReleaseClass.RESTRICTED:
            raise CaseStudyAdmissionError("case admission evidence must remain restricted")
        raw = artifacts.blobs.read_bytes(artifact_record, allow_restricted=True)
        try:
            value = json.loads(raw)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise CaseStudyAdmissionError("admission evidence is not canonical JSON") from exc
        if not isinstance(value, Mapping) or _logical_json_hash(value) != expected:
            raise CaseStudyAdmissionError(f"admission artifact content mismatch: {reference.name}")


def build_case_execution_admission(
    *,
    root: Path,
    source_revision: str,
    plan: CaseStudyExecutionPlan,
    admission: CaseStudyAdmissionAttestation,
    evidence_bundle: CaseAdmissionEvidenceBundle,
    evidence_bundle_reference: CaseArtifactReference,
    construction: DevelopmentConstructionConfiguration,
    construction_path: Path,
    ledger: Ledger,
    artifacts: ArtifactStore,
    ledger_path: Path,
    expected_predecessor_ledger_sha256: str,
    clock: Callable[[], datetime],
) -> CaseExecutionAdmissionReceipt:
    """Validate source, semantic config, CAS gates, and cumulative schedule."""

    if not ledger_path.is_file() or ledger_path.is_symlink():
        raise CaseStudyAdmissionError("case execution requires an existing ledger file")
    observed_ledger_sha = _file_sha256(ledger_path)
    if observed_ledger_sha != expected_predecessor_ledger_sha256:
        raise CaseStudyAdmissionError("cumulative predecessor ledger hash changed")
    validate_case_admission_evidence(
        admission=admission,
        bundle=evidence_bundle,
        artifacts=artifacts,
    )
    if evidence_bundle_reference.logical_content_hash != evidence_bundle.content_hash:
        raise CaseStudyAdmissionError("admission bundle CAS reference changed")
    if _read_reference(artifacts, evidence_bundle_reference) != canonical_json(
        evidence_bundle
    ).encode("utf-8"):
        raise CaseStudyAdmissionError("admission bundle CAS bytes changed")
    source = build_source_manifest(root, source_revision)
    source_ref = _persist_mapping(
        artifacts,
        source.to_dict(),
        object_kind="case_source_manifest",
        created_at=_now(clock),
    )
    if source_ref.logical_content_hash != canonical_sha256(source.to_dict()):
        raise CaseStudyAdmissionError("case source manifest hash is inconsistent")
    if construction.upper_ontology.content_hash != admission.upper_ontology_hash:
        raise CaseStudyAdmissionError("case upper ontology differs from admission")
    if construction.source_file_sha256 != _file_sha256(construction_path):
        raise CaseStudyAdmissionError("construction configuration bytes changed")
    prior_seconds = _total_allocated_seconds(ledger)
    if not math.isfinite(prior_seconds) or prior_seconds <= 0:
        raise CaseStudyAdmissionError(
            "case controller cannot reset cumulative GPU accounting to an empty ledger"
        )
    receipt = CaseExecutionAdmissionReceipt(
        receipt_id=f"admission-{plan.execution_id}",
        execution_plan_hash=plan.content_hash,
        admission_attestation_hash=admission.content_hash,
        admission_evidence_bundle=evidence_bundle_reference,
        source_manifest=source_ref,
        source_revision=source_revision,
        construction_configuration_file_sha256=construction.source_file_sha256,
        construction_configuration_hash=construction.content_hash,
        predecessor_ledger_sha256=observed_ledger_sha,
        prior_gpu_event_inventory_hash=_gpu_inventory_hash(ledger),
        allocated_gpu_seconds_before_case=prior_seconds,
        admitted_at=_now(clock),
    )
    if plan.model_runtime.upper_ontology_hash != construction.upper_ontology.content_hash:
        raise CaseStudyAdmissionError("execution plan binds another upper ontology")
    return receipt


@dataclass(slots=True)
class CaseExecutionRepository:
    """Restricted CAS plus append-only resume/controller state persistence."""

    plan: CaseStudyExecutionPlan
    artifacts: ArtifactStore
    restricted_root: Path
    resume_directory: Path
    state_pointer_path: Path
    clock: Callable[[], datetime] = lambda: datetime.now(UTC)

    def __post_init__(self) -> None:
        root = self.restricted_root.resolve(strict=True)
        for path in (self.resume_directory, self.state_pointer_path.parent):
            if path.exists() and path.is_symlink():
                raise CaseStudyExecutionError(
                    "case runtime state cannot use symbolic-link directories"
                )
            path.mkdir(mode=0o700, parents=True, exist_ok=True)
            resolved = path.resolve(strict=True)
            if not resolved.is_relative_to(root):
                raise CaseStudyExecutionError(
                    "case runtime state must remain under the explicit restricted root"
                )

    def _atomic_pointer(self, reference: CaseArtifactReference) -> None:
        payload = (
            canonical_json(
                {
                    "schema_version": "1.0.0",
                    "execution_plan_hash": self.plan.content_hash,
                    "state_artifact_hash": reference.artifact_hash,
                    "state_logical_hash": reference.logical_content_hash,
                }
            ).encode("utf-8")
            + b"\n"
        )
        descriptor, temporary_name = tempfile.mkstemp(
            dir=self.state_pointer_path.parent,
            prefix=f".{self.state_pointer_path.name}.",
            suffix=".tmp",
        )
        temporary = Path(temporary_name)
        try:
            os.fchmod(descriptor, 0o600)
            with os.fdopen(descriptor, "wb") as stream:
                stream.write(payload)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, self.state_pointer_path)
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise

    def persist_resume(self, resume: CaseStudyResumeManifest) -> CaseArtifactReference:
        status = audit_case_study_resume(self.plan, resume)
        if status.resume_manifest_hash != resume.content_hash:
            raise CaseStudyExecutionError("case resume audit hash changed")
        reference = _persist_record(
            self.artifacts,
            resume,
            object_kind="case_resume_manifest",
            created_at=resume.updated_at,
        )
        destination = self.resume_directory / f"{resume.content_hash}.json"
        payload = canonical_json(resume).encode("utf-8") + b"\n"
        if destination.exists():
            if destination.read_bytes() != payload:
                raise CaseStudyExecutionError("append-only resume file changed")
        else:
            descriptor, temporary_name = tempfile.mkstemp(
                dir=self.resume_directory,
                prefix=f".{resume.content_hash}.",
                suffix=".tmp",
            )
            temporary = Path(temporary_name)
            try:
                os.fchmod(descriptor, 0o600)
                with os.fdopen(descriptor, "wb") as stream:
                    stream.write(payload)
                    stream.flush()
                    os.fsync(stream.fileno())
                os.replace(temporary, destination)
            except BaseException:
                temporary.unlink(missing_ok=True)
                raise
        return reference

    def persist_state(self, state: CaseControllerState) -> CaseArtifactReference:
        reference = _persist_record(
            self.artifacts,
            state,
            object_kind="case_controller_state",
            created_at=state.updated_at,
        )
        # The CAS/ledger commit precedes the replaceable convenience pointer.
        self._atomic_pointer(reference)
        return reference

    def load_state(self) -> CaseControllerState | None:
        if not self.state_pointer_path.exists():
            return None
        try:
            pointer = json.loads(self.state_pointer_path.read_bytes())
        except (OSError, json.JSONDecodeError, UnicodeDecodeError) as exc:
            raise CaseStudyExecutionError("case state pointer is invalid") from exc
        if not isinstance(pointer, Mapping):
            raise CaseStudyExecutionError("case state pointer must contain an object")
        if pointer.get("execution_plan_hash") != self.plan.content_hash:
            raise CaseStudyExecutionError("case state pointer references another plan")
        reference = CaseArtifactReference(
            logical_content_hash=cast(str, pointer.get("state_logical_hash")),
            artifact_hash=cast(str, pointer.get("state_artifact_hash")),
            object_kind="case_controller_state",
        )
        state = cast(
            CaseControllerState,
            _parse_record(self.artifacts, reference, CaseControllerState),
        )
        if state.execution_plan_hash != self.plan.content_hash:
            raise CaseStudyExecutionError("case controller state references another plan")
        return state

    def load_resume(self, reference: CaseArtifactReference) -> CaseStudyResumeManifest:
        resume = cast(
            CaseStudyResumeManifest,
            _parse_record(self.artifacts, reference, CaseStudyResumeManifest),
        )
        audit_case_study_resume(self.plan, resume)
        return resume

    def initialize(
        self,
        admission_reference: CaseArtifactReference,
    ) -> CaseControllerState:
        existing = self.load_state()
        if existing is not None:
            if existing.admission_receipt != admission_reference:
                raise CaseStudyExecutionError("case admission changed on resume")
            return existing
        initialized = _after(self.clock, self.plan.compiled_at)
        resume = initialize_case_study_resume(self.plan, initialized_at=initialized)
        resume_reference = self.persist_resume(resume)
        state = CaseControllerState(
            state_id=f"state-{self.plan.execution_id}-0000",
            sequence_number=0,
            execution_plan_hash=self.plan.content_hash,
            admission_receipt=admission_reference,
            resume_manifest=resume_reference,
            phase=CaseStudyExecutionPhase.INITIALIZED,
            updated_at=_after(self.clock, initialized),
        )
        self.persist_state(state)
        return state

    def successor(
        self,
        previous: CaseControllerState,
        *,
        resume: CaseStudyResumeManifest | None = None,
        **updates: object,
    ) -> CaseControllerState:
        values = previous.model_dump(mode="python", exclude={"content_hash"})
        values.update(updates)
        values.update(
            {
                "state_id": (f"state-{self.plan.execution_id}-{previous.sequence_number + 1:04d}"),
                "sequence_number": previous.sequence_number + 1,
                "parent_state_hash": previous.content_hash,
                "updated_at": _after(self.clock, previous.updated_at),
            }
        )
        if resume is not None:
            current_resume = self.load_resume(previous.resume_manifest)
            validate_resume_successor(current_resume, resume)
            values["resume_manifest"] = self.persist_resume(resume)
        state = CaseControllerState.model_validate(values)
        self.persist_state(state)
        return state


def _resume_successor(
    resume: CaseStudyResumeManifest,
    *,
    clock: Callable[[], datetime],
    prequery_receipt: CasePrequeryReceipt | None = None,
    barrier: CasePrequeryBarrierReceipt | None = None,
    query_access_receipt: CaseQueryAccessReceipt | None = None,
    output_receipt: CaseOutputReceipt | None = None,
) -> CaseStudyResumeManifest:
    supplied = sum(
        item is not None
        for item in (prequery_receipt, barrier, query_access_receipt, output_receipt)
    )
    if supplied != 1:
        raise CaseStudyExecutionError("resume successor requires exactly one append")
    appended_time = (
        prequery_receipt.completed_at
        if prequery_receipt is not None
        else barrier.sealed_at
        if barrier is not None
        else query_access_receipt.accessed_at
        if query_access_receipt is not None
        else output_receipt.completed_at
        if output_receipt is not None
        else resume.updated_at
    )
    values = resume.model_dump(mode="python", exclude={"content_hash"})
    values.update(
        {
            "resume_id": (
                f"resume-{resume.execution_plan_hash[:16]}-{resume.sequence_number + 1:04d}"
            ),
            "sequence_number": resume.sequence_number + 1,
            "parent_resume_hash": resume.content_hash,
            "updated_at": _after(clock, max(resume.updated_at, appended_time)),
        }
    )
    if prequery_receipt is not None:
        values["prequery_receipts"] = (*resume.prequery_receipts, prequery_receipt)
    elif barrier is not None:
        if resume.prequery_barrier is not None:
            raise CaseStudyExecutionError("case prequery barrier is already sealed")
        values["prequery_barrier"] = barrier
    elif query_access_receipt is not None:
        values["query_access_receipts"] = (
            *resume.query_access_receipts,
            query_access_receipt,
        )
    else:
        assert output_receipt is not None
        values["output_receipts"] = (*resume.output_receipts, output_receipt)
    successor = CaseStudyResumeManifest.model_validate(values)
    validate_resume_successor(resume, successor)
    return successor


def _configuration_hash(service: OwnedCaseModelService) -> str:
    value = getattr(service.configuration, "configuration_hash", None)
    if not isinstance(value, str) or len(value) != 64:
        raise CaseStudyExecutionError("case service lacks a frozen configuration hash")
    return value


def _gpu_event(ledger: Ledger, event_id: str) -> Any:
    events = tuple(
        item for item in ledger.gpu_events_with_prefix(event_id) if item.event_id == event_id
    )
    if len(events) != 1:
        raise CaseStudyExecutionError("case GPU call lacks one exact metering event")
    return events[0]


def _parse_timestamp(value: str) -> datetime:
    parsed = datetime.fromisoformat(value)
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise CaseStudyExecutionError("ledger timestamp is not timezone-aware")
    return parsed


@dataclass(frozen=True, slots=True)
class _CaseQueryAccessProxy:
    """Case receipt identity with the fields consumed by pure finalizers."""

    content_hash: str
    execution_id: str
    query_context_hash: str
    model_visible_query_hash: str
    snapshot_hash: str
    stage_manifest_hash: str
    prequery_barrier_hash: str
    packet_hash: str
    registered_revealed_at: datetime
    accessed_at: datetime


@dataclass(frozen=True, slots=True)
class _CaseProduceInputs:
    """Typed case-only counterpart to ProduceInputs for pre-reveal bounded packets.

    The common benchmark contract requires query-dependent packet timestamps.  Case
    windows deliberately materialize all admissible evidence before reveal.  Every
    other field and pure condition finalizer is shared unchanged.
    """

    preparation: ConditionPreparation
    snapshot: EvidenceSnapshot
    packet: EvidencePacket
    context: QueryContext
    query_access: _CaseQueryAccessProxy
    prequery_barrier: CasePrequeryBarrierReceipt
    query_processing_started_at: datetime
    packet_materialization: None
    upper_ontology: UpperOntology
    run_config: RunConditionConfig
    revisions: tuple[()] = ()


class CaseStudyLifecycleGpuAdapter(Protocol):
    """Lifecycle and semantic boundary required by the production controller.

    A conforming implementation must do real request packing, GPU generation,
    deterministic validation, optional repair, and CAS/ledger recording.  The
    controller never substitutes model outputs or constructs C2 semantics.
    """

    backend: Literal["vllm_gpu"]

    @property
    def service_configuration_hash(self) -> Sha256Digest: ...

    @property
    def actual_allocated_service_seconds(self) -> float: ...

    def preflight(
        self,
        *,
        plan: CaseStudyExecutionPlan,
        bounded_packets: Mapping[str, WindowEvidenceBundle],
    ) -> CaseArtifactReference: ...

    def start_once(
        self,
        *,
        execution_id: str,
        remaining_required_seconds: float,
    ) -> None: ...

    def resume_live(self, checkpoint_path: Path) -> bool: ...

    def checkpoint(self, checkpoint_path: Path) -> None: ...

    def preconstruct_c1(
        self,
        *,
        call: CaseGpuCallSlot,
        envelope: CaseC1RequestEnvelope,
        protected_packet: WindowEvidenceBundle,
    ) -> CasePrequeryReceipt: ...

    def project_c1(
        self,
        *,
        window: CaseWindowExecutionPlan,
        query_access: CaseQueryAccessReceipt,
        prequery_barrier: CasePrequeryBarrierReceipt,
        protected_packet: WindowEvidenceBundle,
        protected_query: QueryContext,
    ) -> CaseOutputReceipt: ...

    def construct_c2(
        self,
        *,
        call: CaseGpuCallSlot,
        envelope: CaseC2RequestEnvelope,
        query_access: CaseQueryAccessReceipt,
        prequery_barrier: CasePrequeryBarrierReceipt,
        protected_packet: EvidencePacket,
        protected_query: QueryContext,
    ) -> CaseOutputReceipt: ...

    def shutdown(self) -> object | None: ...


class CaseStudyLifecycleClassicalAdapter(Protocol):
    """Frozen C0 path; the interface exposes no query-time construction."""

    backend: Literal["spacy-ner-dependency-plus-deterministic-rules-v2"]

    def prepare_c0(
        self,
        *,
        window: CaseWindowExecutionPlan,
        envelope: CaseC0PreparationEnvelope,
        protected_packet: WindowEvidenceBundle,
    ) -> CasePrequeryReceipt: ...

    def project_c0(
        self,
        *,
        window: CaseWindowExecutionPlan,
        envelope: CaseC0ProjectionEnvelope,
        query_access: CaseQueryAccessReceipt,
        prequery_barrier: CasePrequeryBarrierReceipt,
        protected_packet: WindowEvidenceBundle,
        protected_query: QueryContext,
    ) -> CaseOutputReceipt: ...


@dataclass(frozen=True, slots=True)
class CaseQueryProvider:
    """Open registered restricted queries only after a sealed barrier."""

    loaded: AttestedRestrictedCaseStudy = field(repr=False)

    def open(self, context_id: str, *, barrier_sealed: bool) -> QueryContext:
        if not barrier_sealed:
            raise CaseStudyExecutionError("case query cannot be opened before the barrier")
        contexts = (
            *(
                context
                for window in self.loaded.preregistration.windows
                for context in window.contexts
            ),
            self.loaded.preregistration.operational_retrieval.context,
        )
        matches = tuple(item for item in contexts if item.context_id == context_id)
        if len(matches) != 1:
            raise CaseStudyExecutionError("registered case query identity is unavailable")
        return matches[0]


@dataclass(slots=True)
class ProductionCaseClassicalAdapter:
    """Concrete query-blind C0 builder and fixed query-time projector."""

    root: Path
    plan: CaseStudyExecutionPlan
    construction: DevelopmentConstructionConfiguration
    artifacts: ArtifactStore
    restricted_state_directory: Path
    builder: ClassicalPreBuilder | None = None
    clock: Callable[[], datetime] = lambda: datetime.now(UTC)
    backend: Literal["spacy-ner-dependency-plus-deterministic-rules-v2"] = (
        "spacy-ner-dependency-plus-deterministic-rules-v2"
    )

    def __post_init__(self) -> None:
        if self.builder is None:
            self.builder, manifest = load_production_classical_builder(
                self.root / "configs/study/c0_rules.json"
            )
            if manifest.config_hash != self.plan.c0_runtime.config_hash:
                raise CaseStudyAdmissionError("production C0 manifest changed")
        if self.builder.config.content_hash != self.plan.c0_runtime.config_hash:
            raise CaseStudyAdmissionError("C0 rules differ from the frozen case plan")
        if self.construction.upper_ontology.content_hash != (
            self.plan.model_runtime.upper_ontology_hash
        ):
            raise CaseStudyAdmissionError("C0 upper ontology differs from case plan")
        self.restricted_state_directory.mkdir(mode=0o700, parents=True, exist_ok=True)
        if self.restricted_state_directory.is_symlink():
            raise CaseStudyAdmissionError("C0 state directory cannot be a symlink")

    def _pointer_path(self, window_id: str) -> Path:
        allowed = "abcdefghijklmnopqrstuvwxyz0123456789-_"
        if not window_id or any(character not in allowed for character in window_id):
            raise CaseStudyExecutionError("unsafe case window identifier")
        return self.restricted_state_directory / f"{window_id}.json"

    def _store_preparation(
        self,
        window_id: str,
        preparation: ConditionPreparation,
    ) -> CaseArtifactReference:
        reference = _persist_record(
            self.artifacts,
            preparation,
            object_kind="case_c0_condition_preparation",
            created_at=preparation.completed_at,
        )
        destination = self._pointer_path(window_id)
        payload = canonical_json(reference).encode("utf-8") + b"\n"
        if destination.exists():
            if destination.read_bytes() != payload:
                raise CaseStudyExecutionError("C0 preparation pointer changed")
            return reference
        descriptor, name = tempfile.mkstemp(
            dir=destination.parent,
            prefix=f".{destination.name}.",
            suffix=".tmp",
        )
        temporary = Path(name)
        try:
            os.fchmod(descriptor, 0o600)
            with os.fdopen(descriptor, "wb") as stream:
                stream.write(payload)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, destination)
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise
        return reference

    def _load_preparation(self, window_id: str) -> ConditionPreparation:
        path = self._pointer_path(window_id)
        if not path.is_file() or path.is_symlink():
            raise CaseStudyExecutionError("C0 preparation is unavailable")
        reference = CaseArtifactReference.model_validate_json(path.read_bytes())
        preparation = cast(
            ConditionPreparation,
            _parse_record(
                self.artifacts,
                reference,
                ConditionPreparation,
            ),
        )
        if preparation.condition is not ConditionName.C0_CLASSICAL_PRE:
            raise CaseStudyExecutionError("C0 pointer resolved another condition")
        return preparation

    def prepare_c0(
        self,
        *,
        window: CaseWindowExecutionPlan,
        envelope: CaseC0PreparationEnvelope,
        protected_packet: WindowEvidenceBundle,
    ) -> CasePrequeryReceipt:
        snapshot = protected_packet.snapshot_assembly.snapshot
        packet = protected_packet.packet
        if (
            envelope.execution_plan_hash != self.plan.content_hash
            or envelope.window_plan_hash != window.content_hash
            or envelope.evidence_binding_hash != window.evidence_binding_hash
            or envelope.snapshot_hash != snapshot.content_hash
            or envelope.packet_hash != packet.content_hash
            or packet.snapshot_hash != snapshot.content_hash
        ):
            raise CaseStudyExecutionError("C0 preparation envelope changed")
        existing_path = self._pointer_path(window.window_id)
        if existing_path.exists():
            preparation = self._load_preparation(window.window_id)
        else:
            constructed = _after(self.clock, snapshot.sealed_at)
            sealed = _after(self.clock, constructed)
            preparation = self.builder.prepare(
                snapshot=snapshot,
                evidence=protected_packet.snapshot_assembly.admissible_evidence,
                upper_ontology=self.construction.upper_ontology,
                preconstruction_budgets=self.construction.preconstruction_budgets,
                constructed_at=constructed,
                sealed_at=sealed,
            )
            self._store_preparation(window.window_id, preparation)
        assert preparation.sealed_preontology is not None
        seal = preparation.sealed_preontology.construction_seal
        return CasePrequeryReceipt(
            receipt_id=f"receipt-{window.c0_preparation_job_id}",
            execution_plan_hash=self.plan.content_hash,
            preparation_job_id=window.c0_preparation_job_id,
            kind=CasePrequeryKind.C0_WINDOW_PRECONSTRUCTION,
            window_id=window.window_id,
            condition=ConditionName.C0_CLASSICAL_PRE,
            evidence_binding_hash=window.evidence_binding_hash,
            snapshot_hash=snapshot.content_hash,
            backend=self.backend,
            terminal_outcome=RunOutcome.SUCCEEDED,
            started_at=seal.constructed_at,
            completed_at=preparation.completed_at,
            construction_seal=seal,
        )

    def project_c0(
        self,
        *,
        window: CaseWindowExecutionPlan,
        envelope: CaseC0ProjectionEnvelope,
        query_access: CaseQueryAccessReceipt,
        prequery_barrier: CasePrequeryBarrierReceipt,
        protected_packet: WindowEvidenceBundle,
        protected_query: QueryContext,
    ) -> CaseOutputReceipt:
        preparation = self._load_preparation(window.window_id)
        assert preparation.sealed_preontology is not None
        seal = preparation.sealed_preontology.construction_seal
        if (
            envelope.execution_plan_hash != self.plan.content_hash
            or envelope.window_plan_hash != window.content_hash
            or envelope.query_access_receipt_hash != query_access.content_hash
            or envelope.construction_seal_hash != seal.content_hash
            or envelope.packet_hash != protected_packet.packet.content_hash
        ):
            raise CaseStudyExecutionError("C0 projection envelope changed")
        index = window.context_ids.index(query_access.context_id)
        job_id = window.c0_projection_job_ids[index]
        config = RunConditionConfig(
            config_id=f"config-{job_id}",
            condition=ConditionName.C0_CLASSICAL_PRE,
            budgets=protected_query.budgets,
            maximum_input_tokens=self.plan.model_runtime.maximum_input_tokens,
            maximum_output_tokens=self.plan.model_runtime.maximum_output_tokens,
            repair_attempt_budget=protected_query.budgets.repair_attempt_budget,
            scored_schema_hash=SCORED_PROJECTION_SCHEMA_HASH,
            validator_hash=self.plan.model_runtime.validator_hash,
            upper_ontology_hash=self.construction.upper_ontology.content_hash,
        )
        stage_hash = canonical_sha256(
            {
                "execution_plan_hash": self.plan.content_hash,
                "context_id": protected_query.context_id,
                "context_hash": protected_query.content_hash,
                "registered_revealed_at": protected_query.revealed_at,
            }
        )
        proxy = _CaseQueryAccessProxy(
            content_hash=query_access.content_hash,
            execution_id=self.plan.execution_id,
            query_context_hash=protected_query.content_hash,
            model_visible_query_hash=to_model_visible_query(protected_query).content_hash,
            snapshot_hash=protected_packet.snapshot_assembly.snapshot.content_hash,
            stage_manifest_hash=stage_hash,
            prequery_barrier_hash=prequery_barrier.content_hash,
            packet_hash=protected_packet.packet.content_hash,
            registered_revealed_at=protected_query.revealed_at,
            accessed_at=query_access.accessed_at,
        )
        inputs = _CaseProduceInputs(
            preparation=preparation,
            snapshot=protected_packet.snapshot_assembly.snapshot,
            packet=protected_packet.packet,
            context=protected_query,
            query_access=proxy,
            prequery_barrier=prequery_barrier,
            query_processing_started_at=_after(self.clock, query_access.accessed_at),
            packet_materialization=None,
            upper_ontology=self.construction.upper_ontology,
            run_config=config,
        )
        projection = project_sealed_c0(
            preparation.sealed_preontology,
            inputs,  # type: ignore[arg-type]
            rule_config=self.builder.config,
        )
        artifact = self.artifacts.put_bytes(
            canonical_json(projection).encode("utf-8"),
            media_type="application/vnd.story-projection.ontology-projection+json",
            release_class=LedgerReleaseClass.RESTRICTED,
            created_at=_after(self.clock, query_access.accessed_at),
        )
        completed = _after(self.clock, query_access.accessed_at)
        preparation_receipt = self.prepare_c0(
            window=window,
            envelope=CaseC0PreparationEnvelope(
                execution_plan_hash=self.plan.content_hash,
                window_plan_hash=window.content_hash,
                evidence_binding_hash=window.evidence_binding_hash,
                snapshot_hash=protected_packet.snapshot_assembly.snapshot.content_hash,
                packet_hash=protected_packet.packet.content_hash,
            ),
            protected_packet=protected_packet,
        )
        return CaseOutputReceipt(
            receipt_id=f"receipt-{job_id}",
            execution_plan_hash=self.plan.content_hash,
            projection_job_id=job_id,
            condition=ConditionName.C0_CLASSICAL_PRE,
            context_id=protected_query.context_id,
            window_id=window.window_id,
            backend=self.backend,
            query_access_receipt_hash=query_access.content_hash,
            preparation_receipt_hash=preparation_receipt.content_hash,
            evidence_binding_hash=window.evidence_binding_hash,
            packet_equality_group_id=window.packet_equality_group_id,
            snapshot_hash=protected_packet.snapshot_assembly.snapshot.content_hash,
            packet_hash=protected_packet.packet.content_hash,
            source_construction_seal_hash=seal.content_hash,
            terminal_outcome=RunOutcome.SUCCEEDED,
            projection_artifact_hash=artifact.content_hash,
            query_accessed_at=query_access.accessed_at,
            completed_at=completed,
            operational_only=False,
            causal_comparison_eligible=True,
        )


@dataclass(frozen=True, slots=True)
class OpenedCaseQuery:
    context: QueryContext = field(repr=False)
    packet: EvidencePacket = field(repr=False)
    access_event: QueryAccessEvent
    packet_materialization: PacketMaterializationEvent
    receipt: CaseQueryAccessReceipt
    receipt_reference: CaseArtifactReference


@dataclass(slots=True)
class CaseStudyProductionController:
    """Execute the frozen 4/8/1 case plan with one owned model load."""

    root: Path
    plan: CaseStudyExecutionPlan
    loaded: AttestedRestrictedCaseStudy = field(repr=False)
    admission: CaseExecutionAdmissionReceipt
    repository: CaseExecutionRepository
    ledger: Ledger
    artifacts: ArtifactStore
    classical: CaseStudyLifecycleClassicalAdapter
    gpu: CaseStudyLifecycleGpuAdapter
    token_counter: Callable[[str], int]
    service_checkpoint_path: Path
    query_provider: CaseQueryProvider | None = None
    operational_packet_provider: OperationalPacketProvider | None = None
    reveal_waiter: Callable[[datetime], None] | None = None
    clock: Callable[[], datetime] = lambda: datetime.now(UTC)
    _bounded: dict[str, WindowEvidenceBundle] = field(default_factory=dict, init=False)
    _bounded_receipts: dict[str, CaseBoundedPacketReceipt] = field(default_factory=dict, init=False)

    def __post_init__(self) -> None:
        if self.plan.content_hash != self.admission.execution_plan_hash:
            raise CaseStudyAdmissionError("case admission belongs to another execution plan")
        if self.loaded.manifest.content_hash != self.plan.restricted_index_manifest_hash:
            raise CaseStudyAdmissionError("restricted case index differs from execution plan")
        if self.classical.backend != self.plan.c0_runtime.backend:
            raise CaseStudyAdmissionError("C0 adapter backend differs from frozen plan")
        if self.gpu.backend != "vllm_gpu":
            raise CaseStudyAdmissionError("case C1/C2 adapter must be GPU-backed")
        if (
            self.gpu.service_configuration_hash
            != self.plan.model_runtime.selected_launcher_configuration_hash
        ):
            raise CaseStudyAdmissionError("case model service configuration changed")
        checkpoint_parent = self.service_checkpoint_path.parent.resolve(strict=True)
        restricted = self.repository.restricted_root.resolve(strict=True)
        if not checkpoint_parent.is_relative_to(restricted):
            raise CaseStudyAdmissionError("service checkpoint must remain restricted")
        if self.query_provider is None:
            self.query_provider = CaseQueryProvider(self.loaded)
        if self.operational_packet_provider is None:
            registration = self.loaded.preregistration.operational_retrieval

            def retrieve(
                *,
                query_accessed_at: datetime,
                packet_created_at: datetime,
            ) -> OperationalRetrievalResult:
                return retrieve_operational_packet(
                    index_path=self.loaded.index_path,
                    manifest=self.loaded.manifest,
                    registration=registration,
                    query_accessed_at=query_accessed_at,
                    packet_created_at=packet_created_at,
                    token_counter=self.token_counter,
                )

            self.operational_packet_provider = retrieve

    def _advance(
        self,
        state: CaseControllerState,
        resume: CaseStudyResumeManifest,
        *,
        phase: CaseStudyExecutionPhase,
        **updates: object,
    ) -> CaseControllerState:
        return self.repository.successor(
            state,
            resume=resume,
            phase=phase,
            **updates,
        )

    def _materialize_bounded_packets(
        self,
        state: CaseControllerState,
    ) -> CaseControllerState:
        registrations = {item.window_id: item for item in self.loaded.preregistration.windows}
        references = dict(state.bounded_packet_receipts)
        for window in self.plan.windows:
            registration = registrations.get(window.window_id)
            if registration is None or registration.content_hash != window.window_registration_hash:
                raise CaseStudyExecutionError("case window registration changed")
            prior_ref = references.get(window.window_id)
            if prior_ref is None:
                created = _after(self.clock, self.plan.compiled_at)
                sealed = _after(self.clock, created)
                packet_time = _after(self.clock, sealed)
            else:
                prior = cast(
                    CaseBoundedPacketReceipt,
                    _parse_record(
                        self.artifacts,
                        prior_ref,
                        CaseBoundedPacketReceipt,
                    ),
                )
                created = prior.snapshot_created_at
                sealed = prior.snapshot_sealed_at
                packet_time = prior.packet_created_at
            bundle = materialize_window_evidence(
                index_path=self.loaded.index_path,
                manifest=self.loaded.manifest,
                window=registration,
                created_at=created,
                sealed_at=sealed,
                packet_created_at=packet_time,
                token_counter=self.token_counter,
            )
            self._bounded[window.window_id] = bundle
            receipt = CaseBoundedPacketReceipt(
                receipt_id=f"packet-receipt-{window.window_id}",
                execution_plan_hash=self.plan.content_hash,
                window_id=window.window_id,
                evidence_binding_hash=window.evidence_binding_hash,
                snapshot_hash=bundle.snapshot_assembly.snapshot.content_hash,
                packet_hash=bundle.packet.content_hash,
                ordered_evidence_ids_hash=canonical_sha256(bundle.packet.ordered_evidence_ids),
                evidence_count=len(bundle.packet.evidence),
                snapshot_created_at=bundle.snapshot_assembly.snapshot.created_at,
                snapshot_sealed_at=bundle.snapshot_assembly.snapshot.sealed_at,
                packet_created_at=bundle.packet.created_at,
            )
            reference = _persist_record(
                self.artifacts,
                receipt,
                object_kind="case_bounded_packet_receipt",
                created_at=packet_time,
            )
            if prior_ref is not None and prior_ref != reference:
                raise CaseStudyExecutionError("bounded packet changed during resume")
            references[window.window_id] = reference
            self._bounded_receipts[window.window_id] = receipt
        return self.repository.successor(
            state,
            phase=CaseStudyExecutionPhase.PREQUERY_CPU,
            bounded_packet_receipts=references,
        )

    def _append_prequery(
        self,
        state: CaseControllerState,
        receipt: CasePrequeryReceipt,
    ) -> CaseControllerState:
        resume = self.repository.load_resume(state.resume_manifest)
        matches = tuple(
            item
            for item in resume.prequery_receipts
            if item.preparation_job_id == receipt.preparation_job_id
        )
        if matches:
            if matches != (receipt,):
                raise CaseStudyExecutionError("prequery replay returned different receipt")
            return state
        successor = _resume_successor(
            resume,
            clock=self.clock,
            prequery_receipt=receipt,
        )
        return self._advance(
            state,
            successor,
            phase=CaseStudyExecutionPhase.SERVICE_LIVE,
        )

    def _empty_inventory_receipt(
        self,
        *,
        preparation_job_id: str,
        kind: CasePrequeryKind,
        window_id: str,
        evidence_binding_hash: str,
        snapshot: EvidenceSnapshot,
    ) -> CasePrequeryReceipt:
        started = _after(self.clock, snapshot.sealed_at)
        preparation = prepare_empty_c2_inventory(snapshot, recorded_at=started)
        assert preparation.empty_inventory is not None
        return CasePrequeryReceipt(
            receipt_id=f"receipt-{preparation_job_id}",
            execution_plan_hash=self.plan.content_hash,
            preparation_job_id=preparation_job_id,
            kind=kind,
            window_id=window_id,
            condition=ConditionName.C2_LLM_QUERY,
            evidence_binding_hash=evidence_binding_hash,
            snapshot_hash=snapshot.content_hash,
            backend="controller",
            terminal_outcome=RunOutcome.SUCCEEDED,
            started_at=started,
            completed_at=_after(self.clock, started),
            empty_inventory=preparation.empty_inventory,
        )

    def _operational_empty_inventory(self) -> CasePrequeryReceipt:
        started = _after(self.clock, self.plan.compiled_at)
        inventory = PreQueryInventory(
            inventory_id=f"inventory-{self.plan.operational.c2_empty_inventory_job_id}",
            snapshot_hash=self.plan.restricted_index_manifest_hash,
            recorded_at=started,
        )
        return CasePrequeryReceipt(
            receipt_id=f"receipt-{self.plan.operational.c2_empty_inventory_job_id}",
            execution_plan_hash=self.plan.content_hash,
            preparation_job_id=self.plan.operational.c2_empty_inventory_job_id,
            kind=CasePrequeryKind.C2_OPERATIONAL_EMPTY_INVENTORY,
            window_id="complete-query-blind-index",
            condition=ConditionName.C2_LLM_QUERY,
            evidence_binding_hash=self.plan.operational.full_index_binding_hash,
            snapshot_hash=self.plan.restricted_index_manifest_hash,
            backend="controller",
            terminal_outcome=RunOutcome.SUCCEEDED,
            started_at=started,
            completed_at=_after(self.clock, started),
            empty_inventory=inventory,
        )

    def _seal_barrier(self, state: CaseControllerState) -> CaseControllerState:
        resume = self.repository.load_resume(state.resume_manifest)
        if resume.prequery_barrier is not None:
            return state
        status = audit_case_study_resume(self.plan, resume)
        if status.next_stage != "seal_barrier":
            raise CaseStudyExecutionError("cannot seal an incomplete case prequery set")
        last = max(item.completed_at for item in resume.prequery_receipts)
        barrier = CasePrequeryBarrierReceipt(
            barrier_id=f"barrier-{self.plan.execution_id}",
            execution_plan_hash=self.plan.content_hash,
            prequery_receipt_hashes=tuple(item.content_hash for item in resume.prequery_receipts),
            sealed_at=_after(self.clock, last),
        )
        reference = _persist_record(
            self.artifacts,
            barrier,
            object_kind="case_prequery_barrier",
            created_at=barrier.sealed_at,
        )
        self.ledger.persist_prequery_barrier(
            PrequeryBarrierRecord(
                barrier_hash=barrier.content_hash,
                barrier_id=barrier.barrier_id,
                execution_id=self.plan.execution_id,
                execution_manifest_hash=self.plan.content_hash,
                barrier_artifact_hash=reference.artifact_hash,
                preparation_count=13,
                sealed_at=barrier.sealed_at.isoformat(),
                persisted_at=_after(self.clock, barrier.sealed_at).isoformat(),
                release_class=LedgerReleaseClass.RESTRICTED,
            ),
            barrier_artifact=self.ledger.get_artifact(reference.artifact_hash),
        )
        successor = _resume_successor(resume, clock=self.clock, barrier=barrier)
        return self._advance(
            state,
            successor,
            phase=CaseStudyExecutionPhase.PREQUERY_COMPLETE,
        )

    def _persist_query_opening(
        self,
        *,
        state: CaseControllerState,
        context: QueryContext,
        packet: EvidencePacket | None,
        window_id: str,
        evidence_binding_hash: str,
        equality_group_id: str,
        operational_provider: OperationalPacketProvider | None = None,
    ) -> tuple[CaseControllerState, OpenedCaseQuery]:
        resume = self.repository.load_resume(state.resume_manifest)
        barrier = resume.prequery_barrier
        if barrier is None:
            raise CaseStudyExecutionError("query opening requires the sealed case barrier")
        if context.revealed_at <= barrier.sealed_at:
            raise CaseStudyExecutionError("registered query reveal does not follow barrier")
        if (packet is None) != (operational_provider is not None):
            raise CaseStudyExecutionError(
                "exactly one bounded packet or operational provider is required"
            )
        threshold = max(barrier.sealed_at, context.revealed_at)
        if self.reveal_waiter is not None:
            self.reveal_waiter(threshold)
        accessed = _now(self.clock)
        if accessed <= barrier.sealed_at or accessed < context.revealed_at:
            raise CaseStudyExecutionError(
                "trusted clock has not reached the registered query reveal"
            )
        stage_hash = canonical_sha256(
            {
                "execution_plan_hash": self.plan.content_hash,
                "context_id": context.context_id,
                "context_hash": context.content_hash,
                "registered_revealed_at": context.revealed_at,
            }
        )
        query_artifact = self.artifacts.put_bytes(
            canonical_json(context).encode("utf-8"),
            media_type="application/vnd.story-projection.restricted-query+json",
            release_class=LedgerReleaseClass.RESTRICTED,
            created_at=accessed,
        )
        event = QueryAccessEvent(
            access_event_id=f"query-access-{self.plan.execution_id}-{context.context_id}",
            execution_id=self.plan.execution_id,
            query_context_hash=context.content_hash,
            model_visible_query_hash=to_model_visible_query(context).content_hash,
            snapshot_hash=(
                packet.snapshot_hash
                if packet is not None
                else self.plan.restricted_index_manifest_hash
            ),
            stage_manifest_hash=stage_hash,
            query_artifact_hash=query_artifact.content_hash,
            prequery_barrier_hash=barrier.content_hash,
            packet_hash=None,
            registered_revealed_at=context.revealed_at,
            accessed_at=accessed,
        )
        event_artifact = self.artifacts.blobs.put_bytes(
            canonical_json(event).encode("utf-8"),
            media_type="application/vnd.story-projection.query-access+json",
            release_class=LedgerReleaseClass.RESTRICTED,
            created_at=accessed,
        )
        self.ledger.persist_query_access(
            QueryAccessRecord(
                access_event_hash=event.content_hash,
                access_event_id=event.access_event_id,
                execution_id=event.execution_id,
                query_context_hash=event.query_context_hash,
                model_visible_query_hash=event.model_visible_query_hash,
                snapshot_hash=event.snapshot_hash,
                stage_manifest_hash=event.stage_manifest_hash,
                query_artifact_hash=event.query_artifact_hash,
                prequery_barrier_hash=event.prequery_barrier_hash,
                packet_hash=None,
                query_payload_artifact_hash=query_artifact.content_hash,
                access_event_artifact_hash=event_artifact.content_hash,
                registered_revealed_at=event.registered_revealed_at.isoformat(),
                accessed_at=event.accessed_at.isoformat(),
                release_class=LedgerReleaseClass.RESTRICTED,
            ),
            query_payload_artifact=query_artifact,
            access_event_artifact=event_artifact,
        )
        operational_receipt = None
        if operational_provider is not None:
            packet_time = _after(self.clock, accessed)
            operational = operational_provider(
                query_accessed_at=accessed,
                packet_created_at=packet_time,
            )
            packet = operational.packet
            operational_receipt = operational.receipt
            if packet.snapshot_hash != event.snapshot_hash:
                raise CaseStudyExecutionError(
                    "operational retrieval returned another index snapshot"
                )
        assert packet is not None
        materialized_at = (
            packet.created_at if operational_receipt is not None else _after(self.clock, accessed)
        )
        materialization = PacketMaterializationEvent(
            materialization_event_id=(
                f"packet-handoff-{self.plan.execution_id}-{context.context_id}"
            ),
            execution_id=self.plan.execution_id,
            query_access_event_hash=event.content_hash,
            snapshot_hash=packet.snapshot_hash,
            packet_hash=packet.content_hash,
            retrieval_method=packet.retrieval_method,
            retrieval_config_hash=evidence_binding_hash,
            started_at=materialized_at,
            completed_at=materialized_at,
        )
        packet_receipt_payload = {
            "packet_hash": packet.content_hash,
            "ordered_evidence_ids_hash": canonical_sha256(packet.ordered_evidence_ids),
            "evidence_count": len(packet.evidence),
            "prose_persisted": False,
        }
        packet_receipt_artifact = self.artifacts.blobs.put_bytes(
            canonical_json(packet_receipt_payload).encode("utf-8"),
            media_type="application/vnd.story-projection.packet-hash-receipt+json",
            release_class=LedgerReleaseClass.RESTRICTED,
            created_at=materialized_at,
        )
        materialization_artifact = self.artifacts.blobs.put_bytes(
            canonical_json(materialization).encode("utf-8"),
            media_type="application/vnd.story-projection.packet-materialization+json",
            release_class=LedgerReleaseClass.RESTRICTED,
            created_at=materialized_at,
        )
        self.ledger.persist_packet_materialization(
            PacketMaterializationRecord(
                materialization_event_hash=materialization.content_hash,
                materialization_event_id=materialization.materialization_event_id,
                execution_id=self.plan.execution_id,
                query_access_event_hash=event.content_hash,
                snapshot_hash=packet.snapshot_hash,
                packet_hash=packet.content_hash,
                retrieval_method=packet.retrieval_method.value,
                retrieval_config_hash=evidence_binding_hash,
                packet_artifact_hash=packet_receipt_artifact.content_hash,
                materialization_event_artifact_hash=materialization_artifact.content_hash,
                started_at=materialized_at.isoformat(),
                completed_at=materialized_at.isoformat(),
                release_class=LedgerReleaseClass.RESTRICTED,
            ),
            packet_artifact=packet_receipt_artifact,
            materialization_event_artifact=materialization_artifact,
        )
        is_operational = operational_receipt is not None
        access_receipt = CaseQueryAccessReceipt(
            access_id=f"case-access-{context.context_id}",
            execution_plan_hash=self.plan.content_hash,
            context_id=context.context_id,
            context_hash=context.content_hash,
            window_id=window_id,
            prequery_barrier_hash=barrier.content_hash,
            evidence_binding_hash=evidence_binding_hash,
            packet_equality_group_id=equality_group_id,
            snapshot_hash=packet.snapshot_hash,
            packet_hash=packet.content_hash,
            ordered_evidence_ids_hash=canonical_sha256(packet.ordered_evidence_ids),
            evidence_count=len(packet.evidence),
            retrieval_method=packet.retrieval_method,
            registered_revealed_at=context.revealed_at,
            accessed_at=accessed,
            packet_materialized_at=(materialized_at if is_operational else packet.created_at),
            operational_retrieval_receipt=operational_receipt,
            operational_only=is_operational,
            causal_comparison_eligible=not is_operational,
        )
        access_ref = _persist_record(
            self.artifacts,
            access_receipt,
            object_kind="case_query_access_receipt",
            created_at=materialized_at,
        )
        successor = _resume_successor(
            resume,
            clock=self.clock,
            query_access_receipt=access_receipt,
        )
        event_refs = dict(state.query_access_event_artifacts)
        receipt_refs = dict(state.query_access_receipt_artifacts)
        materialization_refs = dict(state.packet_materialization_artifacts)
        event_refs[context.context_id] = CaseArtifactReference(
            logical_content_hash=event.content_hash,
            artifact_hash=event_artifact.content_hash,
            object_kind="query_access_event",
        )
        receipt_refs[context.context_id] = access_ref
        materialization_refs[context.context_id] = CaseArtifactReference(
            logical_content_hash=materialization.content_hash,
            artifact_hash=materialization_artifact.content_hash,
            object_kind="packet_materialization_event",
        )
        state = self._advance(
            state,
            successor,
            phase=CaseStudyExecutionPhase.QUERY_OUTPUTS,
            query_access_event_artifacts=event_refs,
            query_access_receipt_artifacts=receipt_refs,
            packet_materialization_artifacts=materialization_refs,
        )
        return state, OpenedCaseQuery(
            context=context,
            packet=packet,
            access_event=event,
            packet_materialization=materialization,
            receipt=access_receipt,
            receipt_reference=access_ref,
        )

    def _append_output(
        self,
        state: CaseControllerState,
        receipt: CaseOutputReceipt,
    ) -> CaseControllerState:
        resume = self.repository.load_resume(state.resume_manifest)
        matches = tuple(
            item
            for item in resume.output_receipts
            if item.projection_job_id == receipt.projection_job_id
        )
        if matches:
            if matches != (receipt,):
                raise CaseStudyExecutionError("output replay returned different receipt")
            return state
        successor = _resume_successor(
            resume,
            clock=self.clock,
            output_receipt=receipt,
        )
        return self._advance(
            state,
            successor,
            phase=CaseStudyExecutionPhase.QUERY_OUTPUTS,
        )

    def _start_or_resume_service(
        self,
        state: CaseControllerState,
    ) -> CaseControllerState:
        if state.model_service_start_count == 0:
            # A crash can occur after the adapter persisted the one-load identity
            # and checkpoint but before the controller advanced its own state.
            # Adopt that exact process first; never interpret controller lag as
            # authority for a second model load.
            if self.gpu.resume_live(self.service_checkpoint_path):
                if self.gpu.service_configuration_hash != (
                    self.plan.model_runtime.selected_launcher_configuration_hash
                ):
                    raise CaseStudyExecutionError("adopted case service configuration changed")
                return self.repository.successor(
                    state,
                    phase=CaseStudyExecutionPhase.SERVICE_LIVE,
                    model_service_start_count=1,
                    model_load_count=1,
                    model_service_live=True,
                    service_configuration_hash=self.gpu.service_configuration_hash,
                )
            started = False
            try:
                self.gpu.start_once(
                    execution_id=self.plan.execution_id,
                    remaining_required_seconds=(
                        CASE_BASE_WATCHDOG_SECONDS + CASE_REQUIRED_NEXT_REPAIR_SECONDS
                    ),
                )
                started = True
                self.gpu.checkpoint(self.service_checkpoint_path)
                return self.repository.successor(
                    state,
                    phase=CaseStudyExecutionPhase.SERVICE_LIVE,
                    model_service_start_count=1,
                    model_load_count=1,
                    model_service_live=True,
                    service_configuration_hash=self.gpu.service_configuration_hash,
                )
            except BaseException:
                if started:
                    self.gpu.shutdown()
                raise
        if state.model_service_shutdown_count == 1:
            raise CaseStudyExecutionError(
                "stopped incomplete case run cannot start a second model load"
            )
        if not self.gpu.resume_live(self.service_checkpoint_path):
            raise CaseStudyExecutionError(
                "case controller could not adopt the exact still-live service"
            )
        if state.service_configuration_hash != self.gpu.service_configuration_hash:
            raise CaseStudyExecutionError("resumed case service configuration changed")
        return state

    def _prepare_all(self, state: CaseControllerState) -> CaseControllerState:
        resume = self.repository.load_resume(state.resume_manifest)
        completed = {item.preparation_job_id for item in resume.prequery_receipts}
        slot_by_id = {item.call_id: item for item in self.plan.gpu_call_slots}
        for window in self.plan.windows:
            bundle = self._bounded[window.window_id]
            packet_receipt = self._bounded_receipts[window.window_id]
            if window.c0_preparation_job_id not in completed:
                c0 = self.classical.prepare_c0(
                    window=window,
                    envelope=CaseC0PreparationEnvelope(
                        execution_plan_hash=self.plan.content_hash,
                        window_plan_hash=window.content_hash,
                        evidence_binding_hash=window.evidence_binding_hash,
                        snapshot_hash=bundle.snapshot_assembly.snapshot.content_hash,
                        packet_hash=bundle.packet.content_hash,
                    ),
                    protected_packet=bundle,
                )
                state = self._append_prequery(state, c0)
            if window.c1_preconstruction_call_id not in completed:
                if not state.model_service_live:
                    raise CaseStudyExecutionError(
                        "C1 preconstruction requires the owned live service"
                    )
                call = slot_by_id[window.c1_preconstruction_call_id]
                c1 = self.gpu.preconstruct_c1(
                    call=call,
                    envelope=CaseC1RequestEnvelope(
                        execution_plan_hash=self.plan.content_hash,
                        call_slot_hash=call.content_hash,
                        evidence_binding_hash=window.evidence_binding_hash,
                        snapshot_hash=bundle.snapshot_assembly.snapshot.content_hash,
                        packet_hash=bundle.packet.content_hash,
                    ),
                    protected_packet=bundle,
                )
                state = self._append_prequery(state, c1)
                self.gpu.checkpoint(self.service_checkpoint_path)
            if window.c2_empty_inventory_job_id not in completed:
                c2_empty = self._empty_inventory_receipt(
                    preparation_job_id=window.c2_empty_inventory_job_id,
                    kind=CasePrequeryKind.C2_BOUNDED_EMPTY_INVENTORY,
                    window_id=window.window_id,
                    evidence_binding_hash=window.evidence_binding_hash,
                    snapshot=bundle.snapshot_assembly.snapshot,
                )
                state = self._append_prequery(state, c2_empty)
            # The prose-free receipt is the only durable packet representation.
            if packet_receipt.prose_persisted is not False:  # pragma: no cover
                raise CaseStudyExecutionError("bounded packet persistence policy changed")
            completed = {
                item.preparation_job_id
                for item in self.repository.load_resume(state.resume_manifest).prequery_receipts
            }
        operational_job = self.plan.operational.c2_empty_inventory_job_id
        if operational_job not in completed:
            state = self._append_prequery(state, self._operational_empty_inventory())
        return state

    @staticmethod
    def _prequery_by_id(
        resume: CaseStudyResumeManifest,
    ) -> dict[str, CasePrequeryReceipt]:
        return {item.preparation_job_id: item for item in resume.prequery_receipts}

    def _bounded_queries(self, state: CaseControllerState) -> CaseControllerState:
        assert self.query_provider is not None
        slot_by_id = {item.call_id: item for item in self.plan.gpu_call_slots}
        for window in self.plan.windows:
            bundle = self._bounded[window.window_id]
            for index, context_id in enumerate(window.context_ids):
                resume = self.repository.load_resume(state.resume_manifest)
                accesses = {item.context_id: item for item in resume.query_access_receipts}
                if context_id in accesses:
                    access = accesses[context_id]
                    context = self.query_provider.open(context_id, barrier_sealed=True)
                else:
                    context = self.query_provider.open(context_id, barrier_sealed=True)
                    state, opened = self._persist_query_opening(
                        state=state,
                        context=context,
                        packet=bundle.packet,
                        window_id=window.window_id,
                        evidence_binding_hash=window.evidence_binding_hash,
                        equality_group_id=window.packet_equality_group_id,
                    )
                    access = opened.receipt
                outputs = {
                    item.projection_job_id
                    for item in self.repository.load_resume(state.resume_manifest).output_receipts
                }
                prequery = self._prequery_by_id(self.repository.load_resume(state.resume_manifest))
                active_resume = self.repository.load_resume(state.resume_manifest)
                assert active_resume.prequery_barrier is not None
                c0_job = window.c0_projection_job_ids[index]
                if c0_job not in outputs:
                    c0_preparation = prequery[window.c0_preparation_job_id]
                    assert c0_preparation.construction_seal is not None
                    receipt = self.classical.project_c0(
                        window=window,
                        envelope=CaseC0ProjectionEnvelope(
                            execution_plan_hash=self.plan.content_hash,
                            window_plan_hash=window.content_hash,
                            query_access_receipt_hash=access.content_hash,
                            construction_seal_hash=(c0_preparation.construction_seal.content_hash),
                            packet_hash=bundle.packet.content_hash,
                        ),
                        query_access=access,
                        prequery_barrier=active_resume.prequery_barrier,
                        protected_packet=bundle,
                        protected_query=context,
                    )
                    state = self._append_output(state, receipt)
                c1_job = window.c1_projection_job_ids[index]
                if c1_job not in outputs:
                    receipt = self.gpu.project_c1(
                        window=window,
                        query_access=access,
                        prequery_barrier=active_resume.prequery_barrier,
                        protected_packet=bundle,
                        protected_query=context,
                    )
                    state = self._append_output(state, receipt)
                c2_job = window.c2_construction_call_ids[index]
                if c2_job not in outputs:
                    call = slot_by_id[c2_job]
                    empty = prequery[window.c2_empty_inventory_job_id]
                    assert empty.empty_inventory is not None
                    receipt = self.gpu.construct_c2(
                        call=call,
                        envelope=CaseC2RequestEnvelope(
                            execution_plan_hash=self.plan.content_hash,
                            call_slot_hash=call.content_hash,
                            query_access_receipt_hash=access.content_hash,
                            evidence_binding_hash=window.evidence_binding_hash,
                            packet_hash=bundle.packet.content_hash,
                            empty_inventory_hash=empty.empty_inventory.content_hash,
                        ),
                        query_access=access,
                        prequery_barrier=active_resume.prequery_barrier,
                        protected_packet=bundle.packet,
                        protected_query=context,
                    )
                    state = self._append_output(state, receipt)
                    self.gpu.checkpoint(self.service_checkpoint_path)
        return state

    def _operational_query(self, state: CaseControllerState) -> CaseControllerState:
        assert self.query_provider is not None
        assert self.operational_packet_provider is not None
        context_id = self.plan.operational.context_id
        resume = self.repository.load_resume(state.resume_manifest)
        outputs = {item.projection_job_id for item in resume.output_receipts}
        if self.plan.operational.c2_call_id in outputs:
            return state
        context = self.query_provider.open(context_id, barrier_sealed=True)
        existing_accesses = {item.context_id: item for item in resume.query_access_receipts}
        if context_id in existing_accesses:
            access = existing_accesses[context_id]
            recovered = self.operational_packet_provider(
                query_accessed_at=access.accessed_at,
                packet_created_at=access.packet_materialized_at,
            )
            if (
                recovered.packet.content_hash != access.packet_hash
                or recovered.receipt != access.operational_retrieval_receipt
            ):
                raise CaseStudyExecutionError(
                    "operational packet changed during idempotent recovery"
                )
            packet = recovered.packet
        else:
            state, opened = self._persist_query_opening(
                state=state,
                context=context,
                packet=None,
                window_id="complete-query-blind-index",
                evidence_binding_hash=self.plan.operational.full_index_binding_hash,
                equality_group_id="case-full-index-operational-packet",
                operational_provider=self.operational_packet_provider,
            )
            access = opened.receipt
            packet = opened.packet
        resume = self.repository.load_resume(state.resume_manifest)
        empty = self._prequery_by_id(resume)[self.plan.operational.c2_empty_inventory_job_id]
        assert empty.empty_inventory is not None
        assert resume.prequery_barrier is not None
        call = next(
            item
            for item in self.plan.gpu_call_slots
            if item.call_id == self.plan.operational.c2_call_id
        )
        receipt = self.gpu.construct_c2(
            call=call,
            envelope=CaseC2RequestEnvelope(
                execution_plan_hash=self.plan.content_hash,
                call_slot_hash=call.content_hash,
                query_access_receipt_hash=access.content_hash,
                evidence_binding_hash=self.plan.operational.full_index_binding_hash,
                packet_hash=packet.content_hash,
                empty_inventory_hash=empty.empty_inventory.content_hash,
            ),
            query_access=access,
            prequery_barrier=resume.prequery_barrier,
            protected_packet=packet,
            protected_query=context,
        )
        state = self._append_output(state, receipt)
        self.gpu.checkpoint(self.service_checkpoint_path)
        return state

    def _blank_review_handoff(
        self,
        state: CaseControllerState,
    ) -> tuple[CaseControllerState, CaseBlankReviewHandoff]:
        resume = self.repository.load_resume(state.resume_manifest)
        status = audit_case_study_resume(self.plan, resume)
        if not status.complete:
            raise CaseStudyExecutionError("review handoff requires all 25 ITT outputs")
        template = compile_case_review_input_template(self.plan)
        template_ref = _persist_record(
            self.artifacts,
            template,
            object_kind="case_blank_review_template",
            created_at=_after(self.clock, resume.updated_at),
        )
        handoff = CaseBlankReviewHandoff(
            handoff_id=f"review-handoff-{self.plan.execution_id}",
            execution_plan_hash=self.plan.content_hash,
            terminal_resume_manifest_hash=resume.content_hash,
            output_receipt_hashes=tuple(item.content_hash for item in resume.output_receipts),
            blank_review_template=template_ref,
            created_at=_after(self.clock, resume.updated_at),
        )
        handoff_ref = _persist_record(
            self.artifacts,
            handoff,
            object_kind="case_blank_review_handoff",
            created_at=handoff.created_at,
        )
        return (
            self.repository.successor(state, review_handoff=handoff_ref),
            handoff,
        )

    def run(self, admission_reference: CaseArtifactReference) -> CaseStudyExecutionResult:
        """Run or resume; any owned live service is stopped on every return path."""

        self.ledger.register_study(
            study_id=self.plan.execution_id,
            protocol_hash=self.plan.content_hash,
            code_manifest_hash=self.admission.source_manifest.logical_content_hash,
            configuration_hash=self.admission.construction_configuration_hash,
            release_class=LedgerReleaseClass.RESTRICTED,
            created_at=_now(self.clock),
        )
        state = self.repository.initialize(admission_reference)
        prior_seconds = self.admission.allocated_gpu_seconds_before_case
        if state.phase is CaseStudyExecutionPhase.COMPLETED:
            resume = self.repository.load_resume(state.resume_manifest)
            status = audit_case_study_resume(self.plan, resume)
            assert state.review_handoff is not None
            return CaseStudyExecutionResult(
                execution_plan_hash=self.plan.content_hash,
                admission_receipt_hash=self.admission.content_hash,
                terminal_state_hash=state.content_hash,
                terminal_resume_manifest_hash=resume.content_hash,
                status=status,
                review_handoff_hash=state.review_handoff.logical_content_hash,
                model_service_start_count=state.model_service_start_count,
                model_load_count=state.model_load_count,
                model_service_shutdown_count=state.model_service_shutdown_count,
                model_service_stopped=True,
                allocated_gpu_seconds_before_case=prior_seconds,
                allocated_gpu_seconds_after_case=_total_allocated_seconds(self.ledger),
                completed_at=_after(self.clock, state.updated_at),
            )
        if state.model_service_shutdown_count == 1:
            raise CaseStudyExecutionError(
                "incomplete case state already consumed and stopped its sole model load"
            )
        started_here = False
        try:
            state = self._materialize_bounded_packets(state)
            preflight = self.gpu.preflight(plan=self.plan, bounded_packets=self._bounded)
            preparations = dict(state.preparation_artifacts)
            existing = preparations.get("gpu-packing-preflight")
            if existing is not None and existing != preflight:
                raise CaseStudyExecutionError("case GPU packing preflight changed")
            preparations["gpu-packing-preflight"] = preflight
            state = self.repository.successor(
                state,
                phase=CaseStudyExecutionPhase.PREQUERY_CPU,
                preparation_artifacts=preparations,
            )
            before_start = state.model_service_start_count
            state = self._start_or_resume_service(state)
            started_here = before_start == 0
            state = self._prepare_all(state)
            state = self._seal_barrier(state)
            state = self._bounded_queries(state)
            state = self._operational_query(state)
            state, handoff = self._blank_review_handoff(state)
            self.gpu.shutdown()
            state = self.repository.successor(
                state,
                phase=CaseStudyExecutionPhase.COMPLETED,
                model_service_live=False,
                model_service_shutdown_count=1,
            )
            resume = self.repository.load_resume(state.resume_manifest)
            status = audit_case_study_resume(self.plan, resume)
            return CaseStudyExecutionResult(
                execution_plan_hash=self.plan.content_hash,
                admission_receipt_hash=self.admission.content_hash,
                terminal_state_hash=state.content_hash,
                terminal_resume_manifest_hash=resume.content_hash,
                status=status,
                review_handoff_hash=handoff.content_hash,
                model_service_start_count=state.model_service_start_count,
                model_load_count=state.model_load_count,
                model_service_shutdown_count=state.model_service_shutdown_count,
                model_service_stopped=True,
                allocated_gpu_seconds_before_case=prior_seconds,
                allocated_gpu_seconds_after_case=_total_allocated_seconds(self.ledger),
                completed_at=_after(self.clock, state.updated_at),
            )
        except BaseException as exc:
            # A service adopted from a checkpoint is owned just as strictly as one
            # started in this invocation.  Never leave it orphaned on failure.
            failure = self.artifacts.put_bytes(
                canonical_json(
                    {
                        "execution_plan_hash": self.plan.content_hash,
                        "exception_type": type(exc).__name__,
                        "phase": state.phase,
                    }
                ).encode("utf-8"),
                media_type="application/vnd.story-projection.case-failure+json",
                release_class=LedgerReleaseClass.RESTRICTED,
                created_at=_now(self.clock),
            )
            if state.model_service_live or started_here:
                try:
                    self.gpu.shutdown()
                finally:
                    self.repository.successor(
                        state,
                        phase=CaseStudyExecutionPhase.FAILED,
                        model_service_live=False,
                        model_service_shutdown_count=1,
                        failure_artifact_hash=failure.content_hash,
                    )
            else:
                self.repository.successor(
                    state,
                    phase=CaseStudyExecutionPhase.FAILED,
                    failure_artifact_hash=failure.content_hash,
                )
            raise
