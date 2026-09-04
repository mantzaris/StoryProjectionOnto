"""Gold-free control plane contracts for the registered held-out primary run.

The only entry point that opens held-out runtime stages first reproduces the
independent-review completion.  Plan derivation then uses public/model-visible
stage manifests and the frozen development exclusion list; it never imports or
reads scorer held-out artifacts.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections import Counter
from pathlib import Path
from typing import Literal, Protocol, Self, runtime_checkable

from pydantic import AwareDatetime, Field, model_validator

from story_projection_onto.benchmark_runtime import (
    GoldFirewallError,
    RuntimeStageKind,
    RuntimeStagingManifest,
    load_staged_world,
)
from story_projection_onto.conditions.base import ConditionPreparation
from story_projection_onto.contracts import (
    ConditionName,
    ConstructionCertificate,
    ConstructionSeal,
    Identifier,
    ImmutableRecord,
    PrequeryBarrier,
    PreQueryInventory,
    QueryAccessEvent,
    Sha256Digest,
    canonical_sha256,
)
from story_projection_onto.development_runtime import (
    DevelopmentExecutionResult,
    DevelopmentPhase,
    RunOutcome,
)
from story_projection_onto.independent_review_runtime import load_completed_review

DEFAULT_HELD_OUT_CONTROL_PATH = Path("configs/study/held_out_primary.json")
DEFAULT_HELD_OUT_OUTPUT_ROOT = Path("artifacts/restricted/held_out_primary")
PRIMARY_CALL_COUNT = 168
PRIMARY_SERVICE_START_P95_SECONDS = 180


class HeldOutControlError(RuntimeError):
    """Held-out plan or execution lineage is unsafe or inconsistent."""


class HeldOutReviewGateError(HeldOutControlError):
    """No held-out model input may open before independent review reproduces."""


class CallClassPolicy(ImmutableRecord):
    call_class: Literal["test_c1", "test_c2", "test_fixed_select"]
    p95_seconds: Literal[72, 95, 180]
    watchdog_seconds: Literal[90, 150, 240]
    repair_reserve_class: Literal["reserve_long", "reserve_standard", "reserve_short"]
    repair_attempt_budget: Literal[1] = 1

    @model_validator(mode="after")
    def registered_values(self) -> Self:
        expected = {
            "test_c1": (180, 240, "reserve_long"),
            "test_c2": (95, 150, "reserve_standard"),
            "test_fixed_select": (72, 90, "reserve_short"),
        }[self.call_class]
        if (self.p95_seconds, self.watchdog_seconds, self.repair_reserve_class) != expected:
            raise ValueError("held-out class policy differs from the registered schedule")
        return self


class HeldOutControlConfiguration(ImmutableRecord):
    configuration_id: Identifier
    benchmark_manifest_path: str
    benchmark_manifest_file_sha256: Sha256Digest
    seed_manifest_path: str
    seed_manifest_file_sha256: Sha256Digest
    gpu_call_inventory_path: str
    gpu_call_inventory_file_sha256: Sha256Digest
    development_exclusion_plan_path: str
    development_exclusion_plan_file_sha256: Sha256Digest
    development_execution_result_path: str
    development_execution_result_file_sha256: str = Field(pattern=r"^(?:PENDING|[0-9a-f]{64})$")
    production_adapter_factory: str = Field(
        min_length=1,
        pattern=r"^(?:PENDING|[A-Za-z_][A-Za-z0-9_.]*:[A-Za-z_][A-Za-z0-9_]*)$",
    )
    prequery_stage_root: str
    query_stage_root: str
    expected_plan_hash: str = Field(pattern=r"^(?:PENDING|[0-9a-f]{64})$")
    world_count: Literal[12] = 12
    context_count: Literal[36] = 36
    c0_preconstruction_count: Literal[12] = 12
    c1_call_count: Literal[24] = 24
    c2_call_count: Literal[72] = 72
    fixed_select_call_count: Literal[72] = 72
    seed_blocks: tuple[Literal[1], Literal[2]] = (1, 2)
    scheduled_gpu_seconds_limit: Literal[32400] = 32400
    hard_gpu_seconds_limit: Literal[36000] = 36000
    complete_inventory_forecast_seconds: Literal[31229] = 31229
    primary_forecast_seconds: Literal[16344] = 16344
    call_policies: tuple[CallClassPolicy, CallClassPolicy, CallClassPolicy]

    @model_validator(mode="after")
    def exact_policy_inventory(self) -> Self:
        if {item.call_class for item in self.call_policies} != {
            "test_c1",
            "test_c2",
            "test_fixed_select",
        }:
            raise ValueError("held-out control requires the three primary call policies")
        return self


class PublicStageReference(ImmutableRecord):
    relative_path: str
    manifest_file_sha256: Sha256Digest
    staging_manifest_hash: Sha256Digest
    stage_id: Identifier
    stage_kind: Literal["prequery_evidence", "query_revealed"]
    evidence_artifact_hash: Sha256Digest
    snapshot_hash: Sha256Digest
    query_artifact_hash: Sha256Digest | None = None
    query_context_hash: Sha256Digest | None = None
    horizon_hash: Sha256Digest | None = None
    budget_hash: Sha256Digest | None = None

    @model_validator(mode="after")
    def safe_stage(self) -> Self:
        path = Path(self.relative_path)
        if path.is_absolute() or ".." in path.parts or "\\" in self.relative_path:
            raise ValueError("runtime stage reference must be safe and relative")
        query_values = (
            self.query_artifact_hash,
            self.query_context_hash,
            self.horizon_hash,
            self.budget_hash,
        )
        if self.stage_kind == "query_revealed":
            if self.query_artifact_hash is None:
                raise ValueError("query stages require their sealed query artifact hash")
            semantic_values = query_values[1:]
            if any(item is None for item in semantic_values) and any(
                item is not None for item in semantic_values
            ):
                raise ValueError(
                    "query-stage semantic hashes must be wholly sealed or wholly opened"
                )
        elif any(item is not None for item in query_values):
            raise ValueError("pre-query stages cannot bind query-derived values")
        return self

    @property
    def query_semantics_opened(self) -> bool:
        return self.stage_kind == "query_revealed" and self.query_context_hash is not None


class HeldOutPrequeryUnit(ImmutableRecord):
    """The only unit view supplied to query-blind C0 construction."""

    unit_id: Identifier
    prequery_stage: PublicStageReference

    @model_validator(mode="after")
    def evidence_only(self) -> Self:
        if self.prequery_stage.stage_kind != "prequery_evidence":
            raise ValueError("pre-query C0 view cannot contain a revealed query stage")
        return self


class HeldOutUnitPlan(ImmutableRecord):
    unit_id: Identifier
    prequery_stage: PublicStageReference
    query_stages: tuple[PublicStageReference, PublicStageReference, PublicStageReference]

    @model_validator(mode="after")
    def one_evidence_three_contexts(self) -> Self:
        if self.prequery_stage.stage_kind != "prequery_evidence":
            raise ValueError("unit prequery stage has the wrong kind")
        if any(item.stage_kind != "query_revealed" for item in self.query_stages):
            raise ValueError("unit query stage has the wrong kind")
        if any(
            item.evidence_artifact_hash != self.prequery_stage.evidence_artifact_hash
            for item in self.query_stages
        ):
            raise ValueError("unit query stages do not share the complete evidence artifact")
        if any(
            item.snapshot_hash != self.prequery_stage.snapshot_hash for item in self.query_stages
        ):
            raise ValueError("unit query stages do not share the sealed evidence snapshot")
        if len({item.query_artifact_hash for item in self.query_stages}) != 3:
            raise ValueError("unit requires three distinct query artifacts")
        return self


class HeldOutCallSpec(ImmutableRecord):
    ordinal: int = Field(ge=1, le=PRIMARY_CALL_COUNT)
    call_id: Identifier
    call_class: Literal["test_c1", "test_c2", "test_fixed_select"]
    condition: ConditionName
    unit_id: Identifier
    seed_block: Literal[1, 2]
    frozen_seed: int = Field(ge=0)
    vllm_seed: int = Field(ge=0, le=2**31 - 1)
    p95_seconds: Literal[72, 95, 180]
    watchdog_seconds: Literal[90, 150, 240]
    repair_reserve_class: Literal["reserve_long", "reserve_standard", "reserve_short"]
    repair_attempt_budget: Literal[1] = 1
    prequery_stage_hash: Sha256Digest
    query_stage_hash: Sha256Digest | None = None
    source_c1_call_id: Identifier | None = None
    construction_operations_permitted: bool

    @model_validator(mode="after")
    def condition_contract(self) -> Self:
        expected = {
            "test_c1": (ConditionName.C1_LLM_PRE, 180, 240, True),
            "test_c2": (ConditionName.C2_LLM_QUERY, 95, 150, True),
            "test_fixed_select": (ConditionName.A_FIXED_SELECT, 72, 90, False),
        }[self.call_class]
        if (
            self.condition,
            self.p95_seconds,
            self.watchdog_seconds,
            self.construction_operations_permitted,
        ) != expected:
            raise ValueError("held-out call differs from its registered class contract")
        if self.call_class == "test_c1":
            if self.query_stage_hash is not None or self.source_c1_call_id is not None:
                raise ValueError("C1 prebuild is query-blind and has no source C1")
        elif self.query_stage_hash is None:
            raise ValueError("query-time held-out call requires a query stage")
        if (self.call_class == "test_fixed_select") != (self.source_c1_call_id is not None):
            raise ValueError("exactly FixedSelect calls bind a same-seed C1 source")
        return self


class HeldOutCallManifest(ImmutableRecord):
    manifest_id: Identifier
    control_configuration_semantic_hash: Sha256Digest
    benchmark_manifest_file_sha256: Sha256Digest
    seed_manifest_hash: Sha256Digest
    gpu_call_inventory_file_sha256: Sha256Digest
    development_execution_result_hash: str = Field(pattern=r"^(?:PENDING|[0-9a-f]{64})$")
    allocated_gpu_seconds_before_heldout: float | None = Field(default=None, ge=0.0)
    post_development_mandatory_forecast_seconds: float | None = Field(default=None, ge=0.0)
    units: tuple[HeldOutUnitPlan, ...]
    calls: tuple[HeldOutCallSpec, ...]
    gold_free_runtime_namespace: Literal[True] = True
    scorer_bridge_permitted_only_after_freeze: Literal[True] = True

    @model_validator(mode="after")
    def exact_registered_design(self) -> Self:
        pending_development = self.development_execution_result_hash == "PENDING"
        if pending_development != (
            self.allocated_gpu_seconds_before_heldout is None
            and self.post_development_mandatory_forecast_seconds is None
        ):
            raise ValueError("held-out manifest development-gate fields are inconsistent")
        if len(self.units) != 12 or len({item.unit_id for item in self.units}) != 12:
            raise ValueError("held-out manifest requires exactly twelve opaque units")
        if sum(len(item.query_stages) for item in self.units) != 36:
            raise ValueError("held-out manifest requires exactly 36 contexts")
        if len(self.calls) != PRIMARY_CALL_COUNT:
            raise ValueError("held-out manifest requires exactly 168 model-call slots")
        if tuple(item.ordinal for item in self.calls) != tuple(range(1, 169)):
            raise ValueError("held-out calls must form one ordered contiguous schedule")
        counts = Counter(item.call_class for item in self.calls)
        if counts != Counter({"test_c1": 24, "test_c2": 72, "test_fixed_select": 72}):
            raise ValueError("held-out call class counts changed")
        c1_ids = {
            (item.unit_id, item.seed_block): item.call_id
            for item in self.calls
            if item.call_class == "test_c1"
        }
        for item in self.calls:
            if item.call_class == "test_fixed_select" and item.source_c1_call_id != c1_ids.get(
                (item.unit_id, item.seed_block)
            ):
                raise ValueError("FixedSelect does not bind the same-world same-seed C1 graph")
        units = {item.unit_id: item for item in self.units}
        expected_keys: Counter[tuple[str, str, int, str | None]] = Counter()
        for unit in self.units:
            for seed in (1, 2):
                expected_keys[("test_c1", unit.unit_id, seed, None)] += 1
                for query in unit.query_stages:
                    expected_keys[
                        (
                            "test_c2",
                            unit.unit_id,
                            seed,
                            query.staging_manifest_hash,
                        )
                    ] += 1
                    expected_keys[
                        (
                            "test_fixed_select",
                            unit.unit_id,
                            seed,
                            query.staging_manifest_hash,
                        )
                    ] += 1
        observed_keys = Counter(
            (item.call_class, item.unit_id, item.seed_block, item.query_stage_hash)
            for item in self.calls
        )
        if observed_keys != expected_keys:
            raise ValueError("held-out calls do not cover each unit/context/seed exactly once")
        for item in self.calls:
            unit = units.get(item.unit_id)
            if (
                unit is None
                or item.prequery_stage_hash != unit.prequery_stage.staging_manifest_hash
            ):
                raise ValueError("held-out call does not bind its unit pre-query stage")
        expected_forecast = sum(item.p95_seconds for item in self.calls)
        if expected_forecast != 16344:
            raise ValueError("held-out primary p95 forecast changed")
        return self


class ReviewedHeldOutPlan(ImmutableRecord):
    """Gold-free plan coupled to a fully reproduced independent-review gate."""

    call_manifest: HeldOutCallManifest
    review_completion_manifest_hash: Sha256Digest
    review_draft_seal_hash: Sha256Digest
    final_reviewed_seal_hash: Sha256Digest
    held_out_launch_authorized: Literal[True] = True

    @model_validator(mode="after")
    def review_lineage_is_distinct(self) -> Self:
        if (
            len(
                {
                    self.review_completion_manifest_hash,
                    self.review_draft_seal_hash,
                    self.final_reviewed_seal_hash,
                }
            )
            != 3
        ):
            raise ValueError("review gate hashes must bind three distinct immutable records")
        return self


class RepairReservePoolSnapshot(ImmutableRecord):
    reserve_class: Literal["reserve_long", "reserve_standard", "reserve_short"]
    total_slots: Literal[4, 8]
    consumed_slots: int = Field(ge=0)
    watchdog_seconds: Literal[90, 150, 240]

    @model_validator(mode="after")
    def registered_pool(self) -> Self:
        expected = {
            "reserve_long": (4, 240),
            "reserve_standard": (8, 150),
            "reserve_short": (4, 90),
        }[self.reserve_class]
        if (self.total_slots, self.watchdog_seconds) != expected:
            raise ValueError("repair reserve pool differs from the registered inventory")
        if self.consumed_slots > self.total_slots:
            raise ValueError("repair reserve pool is oversubscribed")
        return self


class GlobalGpuScheduleSnapshot(ImmutableRecord):
    global_accounting_id: Identifier
    gpu_call_inventory_file_sha256: Sha256Digest
    development_execution_result_hash: str = Field(pattern=r"^(?:PENDING|[0-9a-f]{64})$")
    development_predecessor_allocated_gpu_seconds: float = Field(ge=0.0)
    ledger_chain_hash: Sha256Digest
    actual_allocated_gpu_seconds: float = Field(ge=0.0)
    remaining_registered_p95_seconds: float = Field(ge=0.0)
    repair_reserves: tuple[
        RepairReservePoolSnapshot,
        RepairReservePoolSnapshot,
        RepairReservePoolSnapshot,
    ]
    captured_at: AwareDatetime

    @model_validator(mode="after")
    def finite(self) -> Self:
        if not all(
            math.isfinite(value)
            for value in (
                self.actual_allocated_gpu_seconds,
                self.development_predecessor_allocated_gpu_seconds,
                self.remaining_registered_p95_seconds,
            )
        ):
            raise ValueError("GPU schedule counters must be finite")
        if {item.reserve_class for item in self.repair_reserves} != {
            "reserve_long",
            "reserve_standard",
            "reserve_short",
        }:
            raise ValueError("GPU snapshot requires all three protected repair pools")
        return self


class HeldOutSessionIdentity(ImmutableRecord):
    session_id: Identifier
    condition: ConditionName
    service_identity_hash: Sha256Digest
    service_pid: int = Field(gt=0)
    service_start_ticks: int = Field(gt=0)
    global_accounting_id: Identifier
    global_ledger_chain_hash: Sha256Digest
    allocation_event_id: Identifier
    model_load_event_id: Identifier
    model_load_event_hash: Sha256Digest
    model_load_count: Literal[1] = 1
    already_running: Literal[True] = True
    controller_can_start_service: Literal[False] = False

    @model_validator(mode="after")
    def only_gpu_primary_conditions(self) -> Self:
        if self.condition not in {
            ConditionName.C1_LLM_PRE,
            ConditionName.C2_LLM_QUERY,
            ConditionName.A_FIXED_SELECT,
        }:
            raise ValueError("held-out live session has an unsupported condition")
        return self


class HeldOutServiceShutdownReceipt(ImmutableRecord):
    """Proof that one condition-owned model allocation was stopped exactly once."""

    condition: ConditionName
    session_identity_hash: Sha256Digest
    allocation_event_id: Identifier
    model_load_event_id: Identifier
    model_load_event_hash: Sha256Digest
    global_accounting_id: Identifier
    final_ledger_chain_hash: Sha256Digest
    schedule_after_shutdown: GlobalGpuScheduleSnapshot
    stopped_at: AwareDatetime
    shutdown_count: Literal[1] = 1
    service_process_stopped: Literal[True] = True
    pod_terminated: Literal[False] = False

    @model_validator(mode="after")
    def binds_exact_shutdown_snapshot(self) -> Self:
        snapshot = self.schedule_after_shutdown
        if (
            snapshot.global_accounting_id != self.global_accounting_id
            or snapshot.ledger_chain_hash != self.final_ledger_chain_hash
            or snapshot.captured_at != self.stopped_at
        ):
            raise ValueError("shutdown receipt differs from its cumulative GPU snapshot")
        return self


class FixedSelectCapabilityAudit(ImmutableRecord):
    complete_c1_graph_hash: Sha256Digest
    source_c1_seal_hash: Sha256Digest
    constructive_operator_attempt_count: int = Field(ge=0)
    mechanically_rejected_operator_count: int = Field(ge=0)
    accepted_constructive_operator_count: Literal[0] = 0
    complete_graph_packed: Literal[True] = True

    @model_validator(mode="after")
    def every_attempt_rejected(self) -> Self:
        if self.constructive_operator_attempt_count != self.mechanically_rejected_operator_count:
            raise ValueError("FixedSelect did not mechanically reject every construction attempt")
        return self


class HeldOutCASReference(ImmutableRecord):
    """One content-addressed object with both physical and logical hashes."""

    artifact_hash: Sha256Digest
    logical_content_hash: Sha256Digest
    object_kind: Literal[
        "preconstruction_request",
        "construction_request",
        "packing_report",
        "condition_preparation",
        "condition_attempt",
        "pre_query_inventory",
        "query_access_event",
        "evidence_packet",
        "validated_generation",
        "raw_model_response",
        "held_out_call_audit_receipt",
    ]
    media_type: str = Field(
        min_length=1, pattern=r"^application/vnd\.story-projection\.[a-z0-9._+-]+$"
    )
    release_class: Literal["public", "restricted"]


class HeldOutCallArtifactReceipt(ImmutableRecord):
    """Typed CAS lineage for one started request, including its ledger binding."""

    call_id: Identifier
    call_spec_hash: Sha256Digest
    condition: ConditionName
    semantic_request: HeldOutCASReference
    packing_report: HeldOutCASReference
    output: HeldOutCASReference | None = None
    validation: HeldOutCASReference | None = None
    raw_response: HeldOutCASReference | None = None
    gpu_event_id: Identifier
    gpu_event_hash: Sha256Digest
    model_call_id: Identifier
    model_call_record_hash: Sha256Digest
    ledger_chain_hash: Sha256Digest
    completed_at: AwareDatetime

    @model_validator(mode="after")
    def exact_object_kinds(self) -> Self:
        expected_request = (
            "preconstruction_request"
            if self.condition is ConditionName.C1_LLM_PRE
            else "construction_request"
        )
        if self.semantic_request.object_kind != expected_request:
            raise ValueError("held-out request CAS object has the wrong typed kind")
        if self.packing_report.object_kind != "packing_report":
            raise ValueError("held-out packing CAS object has the wrong typed kind")
        if self.validation is not None and self.validation.object_kind != "validated_generation":
            raise ValueError("held-out validation CAS object has the wrong typed kind")
        if self.raw_response is not None and self.raw_response.object_kind != "raw_model_response":
            raise ValueError("held-out raw response CAS object has the wrong typed kind")
        expected_output = (
            "condition_preparation"
            if self.condition is ConditionName.C1_LLM_PRE
            else "condition_attempt"
        )
        if self.output is not None and self.output.object_kind != expected_output:
            raise ValueError("held-out output CAS object has the wrong typed kind")
        return self


class HeldOutC2PrequeryReceipt(ImmutableRecord):
    """Persisted proof that C2 owned no constructed ontology before query reveal."""

    unit_id: Identifier
    seed_block: Literal[1, 2]
    prequery_stage_hash: Sha256Digest
    inventory: PreQueryInventory
    inventory_artifact: HeldOutCASReference
    preparation: ConditionPreparation
    preparation_artifact: HeldOutCASReference
    completed_at: AwareDatetime

    @model_validator(mode="after")
    def exact_empty_inventory(self) -> Self:
        if self.inventory.content_hash != self.inventory_artifact.logical_content_hash:
            raise ValueError("C2 inventory CAS reference differs from its typed inventory")
        if self.inventory_artifact.object_kind != "pre_query_inventory":
            raise ValueError("C2 inventory must use a pre-query-inventory CAS object")
        if (
            self.preparation.condition is not ConditionName.C2_LLM_QUERY
            or self.preparation.empty_inventory != self.inventory
            or self.preparation.content_hash != self.preparation_artifact.logical_content_hash
            or self.preparation_artifact.object_kind != "condition_preparation"
            or self.completed_at < self.preparation.completed_at
        ):
            raise ValueError("C2 receipt differs from its typed condition preparation")
        return self


class HeldOutAblationPrequeryReceipt(ImmutableRecord):
    """Query-free empty preparation reusable by a frozen one-seed ablation subset."""

    unit_id: Identifier
    condition: ConditionName
    seed_block: Literal[1] = 1
    prequery_stage_hash: Sha256Digest
    preparation: ConditionPreparation
    inventory_artifact: HeldOutCASReference
    preparation_artifact: HeldOutCASReference
    completed_at: AwareDatetime

    @model_validator(mode="after")
    def exact_empty_preparation(self) -> Self:
        allowed = {
            ConditionName.A_NO_CONTEXT,
            ConditionName.A_NO_TEMPORAL_EPISTEMIC,
            ConditionName.A_NO_RARE_GUARD,
        }
        inventory = self.preparation.empty_inventory
        if self.condition not in allowed or self.preparation.condition is not self.condition:
            raise ValueError("ablation pre-query receipt has an unsupported condition")
        if (
            inventory is None
            or inventory.condition is not self.condition
            or inventory.content_hash != self.inventory_artifact.logical_content_hash
            or self.inventory_artifact.object_kind != "pre_query_inventory"
            or self.preparation.content_hash != self.preparation_artifact.logical_content_hash
            or self.preparation_artifact.object_kind != "condition_preparation"
            or self.completed_at < self.preparation.completed_at
        ):
            raise ValueError("ablation receipt differs from its typed empty preparation")
        return self


class HeldOutQueryOpening(ImmutableRecord):
    """Audited post-barrier query opening and shared evidence-packet binding."""

    sealed_stage_hash: Sha256Digest
    opened_stage: PublicStageReference
    query_access_event: QueryAccessEvent
    query_access_artifact: HeldOutCASReference
    evidence_packet_hash: Sha256Digest
    evidence_packet_artifact: HeldOutCASReference
    opened_at: AwareDatetime

    @model_validator(mode="after")
    def exact_opening(self) -> Self:
        stage = self.opened_stage
        event = self.query_access_event
        if not stage.query_semantics_opened:
            raise ValueError("query opening requires fully hydrated semantic hashes")
        if (
            self.sealed_stage_hash != stage.staging_manifest_hash
            or event.stage_manifest_hash != stage.staging_manifest_hash
            or event.query_artifact_hash != stage.query_artifact_hash
            or event.query_context_hash != stage.query_context_hash
            or event.snapshot_hash != stage.snapshot_hash
            or event.packet_hash not in {None, self.evidence_packet_hash}
            or event.content_hash != self.query_access_artifact.logical_content_hash
            or self.evidence_packet_artifact.logical_content_hash != self.evidence_packet_hash
            or self.opened_at != event.accessed_at
        ):
            raise ValueError("held-out query opening lineage is inconsistent")
        if self.query_access_artifact.object_kind != "query_access_event":
            raise ValueError("query-access CAS object has the wrong typed kind")
        if self.evidence_packet_artifact.object_kind != "evidence_packet":
            raise ValueError("evidence-packet CAS object has the wrong typed kind")
        return self


class HeldOutServiceResult(ImmutableRecord):
    call_id: Identifier
    condition: ConditionName
    outcome: RunOutcome
    request_started: bool
    request_hash: Sha256Digest | None = None
    output_artifact_hash: Sha256Digest | None = None
    validation_artifact_hash: Sha256Digest | None = None
    ledger_receipt_hash: Sha256Digest | None = None
    ledger_receipt_artifact_hash: Sha256Digest | None = None
    packing_report_hash: Sha256Digest | None = None
    artifact_receipt: HeldOutCallArtifactReceipt | None = None
    global_ledger_chain_hash: Sha256Digest
    evidence_packet_hash: Sha256Digest | None = None
    horizon_hash: Sha256Digest | None = None
    budget_hash: Sha256Digest | None = None
    construction_seal_hash: Sha256Digest | None = None
    complete_c1_graph_hash: Sha256Digest | None = None
    empty_prequery_inventory_hash: Sha256Digest | None = None
    construction_certificate_hash: Sha256Digest | None = None
    construction_seal: ConstructionSeal | None = None
    pre_query_inventory: PreQueryInventory | None = None
    construction_certificate: ConstructionCertificate | None = None
    fixed_select_capability_audit: FixedSelectCapabilityAudit | None = None
    query_revealed_at: AwareDatetime | None = None
    allocated_gpu_seconds: float = Field(ge=0.0)
    repair_attempts: int = Field(ge=0, le=1)
    failure_code: str | None = None
    completed_at: AwareDatetime

    @model_validator(mode="after")
    def terminal_itt_and_condition_lineage(self) -> Self:
        if self.outcome not in {
            RunOutcome.SUCCEEDED,
            RunOutcome.INVALID,
            RunOutcome.FAILED,
            RunOutcome.TIMED_OUT,
        }:
            raise ValueError("held-out result must be terminal")
        if not math.isfinite(self.allocated_gpu_seconds):
            raise ValueError("allocated GPU seconds must be finite")
        if self.request_started and (
            self.request_hash is None
            or self.ledger_receipt_hash is None
            or self.ledger_receipt_artifact_hash is None
            or self.packing_report_hash is None
            or self.artifact_receipt is None
        ):
            raise ValueError("started held-out call requires complete typed CAS receipts")
        if not self.request_started and (
            self.request_hash is not None
            or self.ledger_receipt_hash is not None
            or self.ledger_receipt_artifact_hash is not None
            or self.packing_report_hash is not None
            or self.artifact_receipt is not None
            or self.allocated_gpu_seconds != 0
        ):
            raise ValueError("unstarted call cannot claim GPU work")
        query_time = self.condition in {
            ConditionName.C2_LLM_QUERY,
            ConditionName.A_FIXED_SELECT,
        }
        if self.request_started and query_time and self.query_revealed_at is None:
            raise ValueError("started query-time call requires a physical query-reveal time")
        if (not query_time or not self.request_started) and self.query_revealed_at is not None:
            raise ValueError("only a started query-time call may claim query reveal")
        if self.outcome is RunOutcome.SUCCEEDED and (
            self.output_artifact_hash is None or self.validation_artifact_hash is None
        ):
            raise ValueError("successful held-out call requires output and validation artifacts")
        if self.artifact_receipt is not None:
            receipt = self.artifact_receipt
            if (
                receipt.call_id != self.call_id
                or receipt.condition is not self.condition
                or receipt.semantic_request.logical_content_hash != self.request_hash
                or receipt.packing_report.logical_content_hash != self.packing_report_hash
                or receipt.content_hash != self.ledger_receipt_hash
                or receipt.ledger_chain_hash != self.global_ledger_chain_hash
                or receipt.completed_at != self.completed_at
            ):
                raise ValueError("held-out typed CAS receipt differs from its result")
            if self.outcome is RunOutcome.SUCCEEDED and (
                receipt.output is None
                or receipt.validation is None
                or receipt.output.logical_content_hash != self.output_artifact_hash
                or receipt.validation.logical_content_hash != self.validation_artifact_hash
            ):
                raise ValueError("successful result differs from its typed CAS outputs")
        if self.outcome is not RunOutcome.SUCCEEDED and not self.failure_code:
            raise ValueError("nonsuccessful held-out call requires a failure code")
        if (
            self.condition is ConditionName.C1_LLM_PRE
            and self.outcome is RunOutcome.SUCCEEDED
            and (
                self.construction_seal_hash is None
                or self.complete_c1_graph_hash is None
                or self.construction_seal is None
            )
        ):
            raise ValueError("successful C1 requires its seal and complete ontology graph")
        if self.construction_seal is not None:
            if (
                self.condition is not ConditionName.C1_LLM_PRE
                or self.construction_seal.condition is not ConditionName.C1_LLM_PRE
                or self.construction_seal.content_hash != self.construction_seal_hash
                or self.construction_seal.ontology_hash != self.complete_c1_graph_hash
            ):
                raise ValueError("C1 construction seal object and hash lineage disagree")
        elif self.construction_seal_hash is not None:
            raise ValueError("a C1 seal hash requires its typed immutable seal")
        if (
            self.condition is ConditionName.C2_LLM_QUERY
            and self.request_started
            and (self.empty_prequery_inventory_hash is None or self.pre_query_inventory is None)
        ):
            raise ValueError("started C2 requires its typed empty pre-query inventory")
        if self.pre_query_inventory is not None and (
            self.condition is not ConditionName.C2_LLM_QUERY
            or self.pre_query_inventory.content_hash != self.empty_prequery_inventory_hash
            or self.query_revealed_at is None
            or self.pre_query_inventory.recorded_at >= self.query_revealed_at
        ):
            raise ValueError("C2 empty-inventory object and hash lineage disagree")
        if (
            self.condition is ConditionName.C2_LLM_QUERY
            and self.outcome is RunOutcome.SUCCEEDED
            and (
                self.construction_certificate_hash is None or self.construction_certificate is None
            )
        ):
            raise ValueError("successful C2 requires empty inventory and certificate")
        if self.construction_certificate is not None:
            if (
                self.construction_certificate.content_hash != self.construction_certificate_hash
                or self.construction_certificate.condition is not self.condition
                or self.construction_certificate.query_revealed_at != self.query_revealed_at
            ):
                raise ValueError("construction certificate object and hash lineage disagree")
        elif self.construction_certificate_hash is not None:
            raise ValueError("a certificate hash requires its typed immutable certificate")
        if self.condition is ConditionName.A_FIXED_SELECT and self.request_started:
            if self.fixed_select_capability_audit is None:
                raise ValueError("started FixedSelect requires a capability audit")
            if self.outcome is RunOutcome.SUCCEEDED and self.construction_certificate is None:
                raise ValueError("successful FixedSelect requires its typed certificate")
        elif self.fixed_select_capability_audit is not None:
            raise ValueError("only FixedSelect may carry its capability audit")
        return self


class HeldOutCallEnvelope(ImmutableRecord):
    call_spec_hash: Sha256Digest
    prequery_stage: PublicStageReference
    query_stage: PublicStageReference | None
    source_c1_call_id: Identifier | None
    source_c1_seal_hash: Sha256Digest | None = None
    complete_c1_graph_hash: Sha256Digest | None = None
    source_c1_output: HeldOutCASReference | None = None
    dependency_failure_hash: Sha256Digest | None = None
    prequery_barrier: PrequeryBarrier | None = None
    query_opening: HeldOutQueryOpening | None = None
    c2_prequery_receipt: HeldOutC2PrequeryReceipt | None = None
    require_empty_prequery_inventory: bool
    construction_operations_permitted: bool

    @model_validator(mode="after")
    def fixed_source_is_complete(self) -> Self:
        if self.prequery_stage.stage_kind != "prequery_evidence":
            raise ValueError("held-out call envelope requires an evidence-only pre-query stage")
        if self.query_stage is not None and self.query_stage.stage_kind != "query_revealed":
            raise ValueError("held-out query envelope has an invalid stage kind")
        if self.query_stage is not None and (
            self.query_stage.evidence_artifact_hash != self.prequery_stage.evidence_artifact_hash
            or self.query_stage.snapshot_hash != self.prequery_stage.snapshot_hash
        ):
            raise ValueError("query envelope does not preserve its pre-query evidence snapshot")
        source_values = (
            self.source_c1_call_id,
            self.source_c1_seal_hash,
            self.complete_c1_graph_hash,
            self.source_c1_output,
        )
        if self.construction_operations_permitted:
            if any(item is not None for item in source_values):
                raise ValueError("constructive call cannot inherit a C1 ontology")
            if self.dependency_failure_hash is not None:
                raise ValueError("constructive call cannot claim a C1 dependency failure")
        else:
            has_complete_source = all(item is not None for item in source_values)
            has_dependency_failure = self.dependency_failure_hash is not None
            if has_complete_source == has_dependency_failure:
                raise ValueError(
                    "FixedSelect requires exactly a complete C1 source or dependency failure"
                )
        if self.require_empty_prequery_inventory and self.query_stage is None:
            raise ValueError("query-time C2 requires a query stage")
        query_time = self.query_stage is not None
        if query_time != (self.prequery_barrier is not None and self.query_opening is not None):
            raise ValueError("query-time envelope requires its persisted barrier and opening")
        if query_time:
            assert self.query_stage is not None
            assert self.prequery_barrier is not None
            assert self.query_opening is not None
            if (
                self.query_opening.opened_stage != self.query_stage
                or self.query_opening.query_access_event.prequery_barrier_hash
                != self.prequery_barrier.content_hash
            ):
                raise ValueError("query envelope differs from its audited opening")
        if self.require_empty_prequery_inventory != (self.c2_prequery_receipt is not None):
            raise ValueError("exactly C2 query calls require an empty-inventory receipt")
        if self.c2_prequery_receipt is not None and (
            self.c2_prequery_receipt.prequery_stage_hash
            != self.prequery_stage.staging_manifest_hash
            or self.c2_prequery_receipt.inventory.snapshot_hash != self.prequery_stage.snapshot_hash
        ):
            raise ValueError("C2 inventory receipt differs from its pre-query stage")
        return self


@runtime_checkable
class InjectedHeldOutSession(Protocol):
    """One externally managed, condition-specific live-service session."""

    def identity(self) -> HeldOutSessionIdentity: ...

    def schedule_snapshot(self) -> GlobalGpuScheduleSnapshot: ...

    def execute_call(
        self, call: HeldOutCallSpec, envelope: HeldOutCallEnvelope
    ) -> HeldOutServiceResult: ...

    def recover_call(
        self, call: HeldOutCallSpec, envelope: HeldOutCallEnvelope
    ) -> HeldOutServiceResult | None: ...


@runtime_checkable
class InjectedHeldOutRuntime(Protocol):
    """Audited chronology and typed-CAS boundary shared by all conditions."""

    def prepare_c2_empty_inventory(
        self, unit: HeldOutPrequeryUnit, *, seed_block: int
    ) -> HeldOutC2PrequeryReceipt: ...

    def validate_c2_prequery_receipt(self, receipt: HeldOutC2PrequeryReceipt) -> None: ...

    def prepare_ablation_empty_inventory(
        self,
        unit: HeldOutPrequeryUnit,
        *,
        condition: ConditionName,
    ) -> HeldOutAblationPrequeryReceipt: ...

    def validate_ablation_prequery_receipt(
        self, receipt: HeldOutAblationPrequeryReceipt
    ) -> None: ...

    def persist_prequery_barrier(self, barrier: PrequeryBarrier) -> None: ...

    def open_or_recover_query(
        self,
        stage: PublicStageReference,
        *,
        barrier: PrequeryBarrier,
        persisted: HeldOutQueryOpening | None,
    ) -> HeldOutQueryOpening: ...

    def validate_result_artifacts(
        self,
        call: HeldOutCallSpec,
        envelope: HeldOutCallEnvelope,
        result: HeldOutServiceResult,
    ) -> None: ...

    def shutdown_condition(
        self,
        identity: HeldOutSessionIdentity,
    ) -> tuple[HeldOutServiceShutdownReceipt, GlobalGpuScheduleSnapshot]: ...


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _safe_file(repository: Path, relative: str, expected_hash: str) -> Path:
    path = Path(relative)
    if path.is_absolute() or ".." in path.parts or "\\" in relative:
        raise HeldOutControlError(f"unsafe held-out input path: {relative}")
    candidate = repository / path
    if candidate.is_symlink():
        raise HeldOutControlError(f"symlinked held-out input: {relative}")
    resolved = candidate.resolve(strict=True)
    if not resolved.is_relative_to(repository.resolve(strict=True)):
        raise HeldOutControlError(f"held-out input escapes repository: {relative}")
    if _file_sha256(resolved) != expected_hash:
        raise HeldOutControlError(f"held-out input bytes changed: {relative}")
    return resolved


def load_held_out_control_configuration(
    repository: Path, path: Path = DEFAULT_HELD_OUT_CONTROL_PATH
) -> HeldOutControlConfiguration:
    source = repository / path
    if source.is_symlink():
        raise HeldOutControlError("held-out control configuration cannot be a symlink")
    return HeldOutControlConfiguration.model_validate_json(source.read_bytes())


def _stage_reference(repository: Path, relative: str) -> PublicStageReference:
    """Open one query-blind evidence stage; query stages use the sealed helper below."""

    logical = Path(relative)
    if logical.is_absolute() or ".." in logical.parts or "\\" in relative:
        raise HeldOutControlError("unsafe runtime stage path")
    stage_root = repository / logical
    manifest_path = stage_root / "manifest.json"
    if stage_root.is_symlink() or manifest_path.is_symlink():
        raise HeldOutControlError("runtime stage cannot be a symlink")
    try:
        resolved_root = stage_root.resolve(strict=True)
        repository_root = repository.resolve(strict=True)
        if not resolved_root.is_relative_to(repository_root):
            raise HeldOutControlError("runtime stage escapes the repository")
        raw = manifest_path.read_bytes()
        manifest = RuntimeStagingManifest.model_validate_json(raw)
        if manifest.stage_kind is RuntimeStageKind.PREQUERY_EVIDENCE:
            evidence = load_staged_world(resolved_root / "evidence.json", resolved_root, manifest)
        else:
            raise HeldOutControlError(
                "query-revealed stages cannot be semantically opened during plan derivation"
            )
    except (OSError, ValueError, KeyError, json.JSONDecodeError, GoldFirewallError) as error:
        raise HeldOutControlError(f"invalid runtime stage {relative}: {error}") from error
    return PublicStageReference(
        relative_path=relative,
        manifest_file_sha256=hashlib.sha256(raw).hexdigest(),
        staging_manifest_hash=manifest.content_hash,
        stage_id=manifest.stage_id,
        stage_kind=manifest.stage_kind.value,
        evidence_artifact_hash=evidence.content_hash,
        snapshot_hash=evidence.snapshot.content_hash,
        query_artifact_hash=None,
        query_context_hash=None,
        horizon_hash=None,
        budget_hash=None,
    )


def _sealed_query_stage_reference(
    repository: Path,
    relative: str,
    *,
    snapshots_by_evidence: dict[str, str],
) -> PublicStageReference:
    """Freeze only a query stage's allowlist and hashes, never ``query.json`` bytes."""

    logical = Path(relative)
    if logical.is_absolute() or ".." in logical.parts or "\\" in relative:
        raise HeldOutControlError("unsafe runtime stage path")
    stage_root = repository / logical
    manifest_path = stage_root / "manifest.json"
    if stage_root.is_symlink() or manifest_path.is_symlink():
        raise HeldOutControlError("runtime stage cannot be a symlink")
    try:
        resolved_root = stage_root.resolve(strict=True)
        repository_root = repository.resolve(strict=True)
        if not resolved_root.is_relative_to(repository_root):
            raise HeldOutControlError("runtime stage escapes the repository")
        raw = manifest_path.read_bytes()
        manifest = RuntimeStagingManifest.model_validate_json(raw)
        if manifest.stage_kind is not RuntimeStageKind.QUERY_REVEALED:
            raise HeldOutControlError("sealed query reference requires a query stage")
        expected_names = {*manifest.file_names, manifest.manifest_file_name}
        actual_names = {item.name for item in resolved_root.iterdir()}
        if actual_names != expected_names or any(
            item.is_dir() or item.is_symlink() for item in resolved_root.iterdir()
        ):
            raise HeldOutControlError("query stage directory differs from its exact allowlist")
        evidence_hash, query_hash = manifest.artifact_hashes
        snapshot_hash = snapshots_by_evidence[evidence_hash]
    except KeyError as error:
        raise HeldOutControlError(
            "query stage does not name one registered pre-query evidence artifact"
        ) from error
    except (OSError, ValueError, json.JSONDecodeError, GoldFirewallError) as error:
        raise HeldOutControlError(f"invalid runtime stage {relative}: {error}") from error
    return PublicStageReference(
        relative_path=relative,
        manifest_file_sha256=hashlib.sha256(raw).hexdigest(),
        staging_manifest_hash=manifest.content_hash,
        stage_id=manifest.stage_id,
        stage_kind=manifest.stage_kind.value,
        evidence_artifact_hash=evidence_hash,
        snapshot_hash=snapshot_hash,
        query_artifact_hash=query_hash,
    )


def _derive_call_manifest(
    repository: Path, configuration: HeldOutControlConfiguration
) -> HeldOutCallManifest:
    benchmark_path = _safe_file(
        repository,
        configuration.benchmark_manifest_path,
        configuration.benchmark_manifest_file_sha256,
    )
    seed_path = _safe_file(
        repository,
        configuration.seed_manifest_path,
        configuration.seed_manifest_file_sha256,
    )
    inventory_path = _safe_file(
        repository,
        configuration.gpu_call_inventory_path,
        configuration.gpu_call_inventory_file_sha256,
    )
    development_path = _safe_file(
        repository,
        configuration.development_exclusion_plan_path,
        configuration.development_exclusion_plan_file_sha256,
    )
    development_result: DevelopmentExecutionResult | None = None
    if configuration.development_execution_result_file_sha256 != "PENDING":
        development_result_path = _safe_file(
            repository,
            configuration.development_execution_result_path,
            configuration.development_execution_result_file_sha256,
        )
        try:
            development_result = DevelopmentExecutionResult.model_validate_json(
                development_result_path.read_bytes()
            )
        except Exception as error:
            raise HeldOutControlError(
                f"invalid frozen development execution result: {error}"
            ) from error
        if (
            development_result.phase is not DevelopmentPhase.COMPLETED
            or not development_result.gate.passed
            or not development_result.forecast.scheduled_admitted
            or not development_result.forecast.below_hard_stop
        ):
            raise HeldOutControlError(
                "held-out launch requires a completed, passing development execution"
            )
    benchmark = json.loads(benchmark_path.read_text())
    if benchmark.get("held_out_world_count") != 12 or benchmark.get("held_out_context_count") != 36:
        raise HeldOutControlError("benchmark manifest held-out counts changed")
    inventory = json.loads(inventory_path.read_text())
    classes = {item["name"]: item for item in inventory["classes"]}
    for policy in configuration.call_policies:
        row = classes.get(policy.call_class)
        expected_count = {"test_c1": 24, "test_c2": 72, "test_fixed_select": 72}[policy.call_class]
        if row != {
            "name": policy.call_class,
            "count": expected_count,
            "provisional_p95_seconds": policy.p95_seconds,
        }:
            raise HeldOutControlError("GPU inventory differs from held-out call policy")
    if sum(row["count"] * row["provisional_p95_seconds"] for row in inventory["classes"]) != 31229:
        raise HeldOutControlError("complete registered GPU schedule changed")
    development = json.loads(development_path.read_text())
    excluded_prequery = {item["prequery_stage"]["relative_path"] for item in development["units"]}
    excluded_queries = {
        stage["relative_path"] for item in development["units"] for stage in item["query_stages"]
    }
    prequery_root = repository / configuration.prequery_stage_root
    query_root = repository / configuration.query_stage_root
    prequeries = [
        _stage_reference(repository, path.parent.relative_to(repository).as_posix())
        for path in sorted(prequery_root.glob("*/manifest.json"))
        if path.parent.relative_to(repository).as_posix() not in excluded_prequery
    ]
    snapshots_by_evidence = {item.evidence_artifact_hash: item.snapshot_hash for item in prequeries}
    queries = [
        _sealed_query_stage_reference(
            repository,
            path.parent.relative_to(repository).as_posix(),
            snapshots_by_evidence=snapshots_by_evidence,
        )
        for path in sorted(query_root.glob("*/manifest.json"))
        if path.parent.relative_to(repository).as_posix() not in excluded_queries
    ]
    if len(prequeries) != 12 or len(queries) != 36:
        raise HeldOutControlError("public stage inventory is not exactly 12 worlds/36 contexts")
    queries_by_evidence: dict[str, list[PublicStageReference]] = {}
    for query in queries:
        queries_by_evidence.setdefault(query.evidence_artifact_hash, []).append(query)
    units = tuple(
        HeldOutUnitPlan(
            unit_id=f"held-unit-{prequery.staging_manifest_hash[:16]}",
            prequery_stage=prequery,
            query_stages=tuple(
                sorted(
                    queries_by_evidence.get(prequery.evidence_artifact_hash, []),
                    key=lambda item: item.query_artifact_hash or "",
                )
            ),
        )
        for prequery in sorted(prequeries, key=lambda item: item.staging_manifest_hash)
    )
    seed_manifest = json.loads(seed_path.read_text())
    seed_hash = seed_manifest["content_hash"]
    seeds = {
        int(item["purpose"].removeprefix("llm_block_")): item["seed"]
        for item in seed_manifest["entries"]
        if item["purpose"] in {"llm_block_1", "llm_block_2"}
    }
    policy = {item.call_class: item for item in configuration.call_policies}
    calls: list[HeldOutCallSpec] = []
    for unit in units:
        for seed_block in (1, 2):
            row = policy["test_c1"]
            calls.append(
                HeldOutCallSpec(
                    ordinal=len(calls) + 1,
                    call_id=f"test-c1-{unit.unit_id}-s{seed_block}",
                    call_class="test_c1",
                    condition=ConditionName.C1_LLM_PRE,
                    unit_id=unit.unit_id,
                    seed_block=seed_block,
                    frozen_seed=seeds[seed_block],
                    vllm_seed=seeds[seed_block] & 0x7FFFFFFF,
                    p95_seconds=row.p95_seconds,
                    watchdog_seconds=row.watchdog_seconds,
                    repair_reserve_class=row.repair_reserve_class,
                    prequery_stage_hash=unit.prequery_stage.staging_manifest_hash,
                    construction_operations_permitted=True,
                )
            )
    for call_class, condition in (
        ("test_c2", ConditionName.C2_LLM_QUERY),
        ("test_fixed_select", ConditionName.A_FIXED_SELECT),
    ):
        row = policy[call_class]
        for unit in units:
            for seed_block in (1, 2):
                source_c1 = f"test-c1-{unit.unit_id}-s{seed_block}"
                for query in unit.query_stages:
                    calls.append(
                        HeldOutCallSpec(
                            ordinal=len(calls) + 1,
                            call_id=(
                                f"{call_class.replace('_', '-')}-{unit.unit_id}-"
                                f"s{seed_block}-{query.stage_id}"
                            ),
                            call_class=call_class,
                            condition=condition,
                            unit_id=unit.unit_id,
                            seed_block=seed_block,
                            frozen_seed=seeds[seed_block],
                            vllm_seed=seeds[seed_block] & 0x7FFFFFFF,
                            p95_seconds=row.p95_seconds,
                            watchdog_seconds=row.watchdog_seconds,
                            repair_reserve_class=row.repair_reserve_class,
                            prequery_stage_hash=unit.prequery_stage.staging_manifest_hash,
                            query_stage_hash=query.staging_manifest_hash,
                            source_c1_call_id=(
                                source_c1 if call_class == "test_fixed_select" else None
                            ),
                            construction_operations_permitted=call_class != "test_fixed_select",
                        )
                    )
    configuration_payload = configuration.model_dump(
        mode="python", exclude={"content_hash", "expected_plan_hash"}
    )
    manifest = HeldOutCallManifest(
        manifest_id=(f"held-out-primary-{canonical_sha256(configuration_payload)[:20]}"),
        control_configuration_semantic_hash=canonical_sha256(configuration_payload),
        benchmark_manifest_file_sha256=configuration.benchmark_manifest_file_sha256,
        seed_manifest_hash=seed_hash,
        gpu_call_inventory_file_sha256=configuration.gpu_call_inventory_file_sha256,
        development_execution_result_hash=(
            "PENDING" if development_result is None else development_result.content_hash
        ),
        allocated_gpu_seconds_before_heldout=(
            None
            if development_result is None
            else development_result.forecast.actual_allocated_seconds_after
        ),
        post_development_mandatory_forecast_seconds=(
            None
            if development_result is None
            else development_result.forecast.post_development_mandatory_forecast_seconds
        ),
        units=units,
        calls=tuple(calls),
    )
    if (
        configuration.expected_plan_hash != "PENDING"
        and manifest.content_hash != configuration.expected_plan_hash
    ):
        raise HeldOutControlError("derived held-out call manifest differs from its frozen hash")
    return manifest


def open_reviewed_held_out_plan(
    *,
    repository: Path,
    review_completion_root: Path,
    benchmark_root: Path | None = None,
    configuration_path: Path = DEFAULT_HELD_OUT_CONTROL_PATH,
) -> ReviewedHeldOutPlan:
    """Reproduce review first, then and only then inspect held-out runtime stages."""

    try:
        review = load_completed_review(
            benchmark_root=benchmark_root or repository / "data/synthetic",
            output_root=review_completion_root,
        )
    except Exception as error:
        raise HeldOutReviewGateError(
            f"held-out input opening blocked: independent review is incomplete: {error}"
        ) from error
    configuration = load_held_out_control_configuration(repository, configuration_path)
    if configuration.development_execution_result_file_sha256 == "PENDING":
        raise HeldOutReviewGateError(
            "held-out launch blocked: frozen passing development result is not configured"
        )
    if configuration.production_adapter_factory == "PENDING":
        raise HeldOutReviewGateError(
            "held-out launch blocked: production adapter factory is not configured"
        )
    manifest = _derive_call_manifest(repository, configuration)
    if (
        review.draft.content_hash
        != json.loads((repository / configuration.benchmark_manifest_path).read_text())[
            "draft_seal_hash"
        ]
    ):
        raise HeldOutReviewGateError("reviewed seal and public benchmark manifest disagree")
    if (
        not review.manifest.held_out_launch_authorized
        or review.manifest.draft_seal_hash != review.draft.content_hash
        or review.manifest.final_seal_hash != review.final_seal.content_hash
    ):
        raise HeldOutReviewGateError("independent-review completion lineage is inconsistent")
    return ReviewedHeldOutPlan(
        call_manifest=manifest,
        review_completion_manifest_hash=review.manifest.content_hash,
        review_draft_seal_hash=review.draft.content_hash,
        final_reviewed_seal_hash=review.final_seal.content_hash,
    )


def admit_call(
    *,
    call: HeldOutCallSpec,
    snapshot: GlobalGpuScheduleSnapshot,
    configuration: HeldOutControlConfiguration,
) -> None:
    """Apply scheduled and hard limits before every request or recovery."""

    if snapshot.gpu_call_inventory_file_sha256 != configuration.gpu_call_inventory_file_sha256:
        raise HeldOutControlError("GPU accounting snapshot uses another call inventory")
    if snapshot.actual_allocated_gpu_seconds >= configuration.hard_gpu_seconds_limit:
        raise HeldOutControlError("hard 10-hour GPU allocation stop reached")
    if (
        snapshot.actual_allocated_gpu_seconds + snapshot.remaining_registered_p95_seconds
        > configuration.scheduled_gpu_seconds_limit
    ):
        raise HeldOutControlError("registered remaining work no longer fits the 9-hour schedule")
    if call.watchdog_seconds < call.p95_seconds:
        raise HeldOutControlError("call watchdog is shorter than registered p95")


__all__ = [
    "DEFAULT_HELD_OUT_CONTROL_PATH",
    "DEFAULT_HELD_OUT_OUTPUT_ROOT",
    "PRIMARY_CALL_COUNT",
    "PRIMARY_SERVICE_START_P95_SECONDS",
    "CallClassPolicy",
    "FixedSelectCapabilityAudit",
    "GlobalGpuScheduleSnapshot",
    "HeldOutAblationPrequeryReceipt",
    "HeldOutC2PrequeryReceipt",
    "HeldOutCASReference",
    "HeldOutCallArtifactReceipt",
    "HeldOutCallEnvelope",
    "HeldOutCallManifest",
    "HeldOutCallSpec",
    "HeldOutControlConfiguration",
    "HeldOutControlError",
    "HeldOutPrequeryUnit",
    "HeldOutQueryOpening",
    "HeldOutReviewGateError",
    "HeldOutServiceResult",
    "HeldOutServiceShutdownReceipt",
    "HeldOutSessionIdentity",
    "HeldOutUnitPlan",
    "InjectedHeldOutRuntime",
    "InjectedHeldOutSession",
    "PublicStageReference",
    "RepairReservePoolSnapshot",
    "ReviewedHeldOutPlan",
    "admit_call",
    "load_held_out_control_configuration",
    "open_reviewed_held_out_plan",
]
