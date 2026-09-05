"""Gold-free CAS/ledger transport records for the development execution.

The live model adapter emits these records, while the isolated scientific
assessor consumes them.  Keeping the shared wire contract outside
``scorer_only`` prevents model/runtime code from importing scorer logic.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Literal, Self

from pydantic import AwareDatetime, Field, model_validator

from story_projection_onto.contracts import (
    ConditionName,
    ImmutableRecord,
    RunOutcome,
    Sha256Digest,
)
from story_projection_onto.development_runtime import DEVELOPMENT_CALL_COUNT

HISTORICAL_SECOND_FALLBACK_RECOVERY_SERVICE_START_EVENT_IDS = (
    "fallback-qwen3-8b-awq-development-v3-service-start-001",
    "fallback-qwen3-8b-awq-development-v7-service-start-001",
)
SECOND_FALLBACK_RECOVERY_SERVICE_START_EVENT_IDS = (
    *HISTORICAL_SECOND_FALLBACK_RECOVERY_SERVICE_START_EVENT_IDS,
    "fallback-qwen3-8b-awq-development-v8-service-start-001",
)


class LogicalCASReference(ImmutableRecord):
    """Bind an immutable record's semantic hash to serialized CAS bytes."""

    logical_content_hash: Sha256Digest
    artifact_hash: Sha256Digest
    object_kind: str = Field(min_length=1, pattern=r"^[a-z][a-z0-9_]*$")


class OpaqueJSONReference(ImmutableRecord):
    """Bind JSON without an embedded ``content_hash`` to its CAS address."""

    logical_content_hash: Sha256Digest
    artifact_hash: Sha256Digest
    object_kind: str = Field(min_length=1, pattern=r"^[a-z][a-z0-9_]*$")


class DevelopmentTreatmentSwitches(ImmutableRecord):
    """Gold-free declaration of the three registered conceptual switches."""

    context_payload_mode: Literal["structured", "generic"] = "structured"
    temporal_epistemic_fields: Literal["enabled", "disabled"] = "enabled"
    rare_guard: Literal["enabled", "disabled"] = "enabled"


class DevelopmentRepairDiagnostic(ImmutableRecord):
    """One fact-free, mechanically derived repair instruction."""

    code: Literal["unknown_contextual_type_reference"]
    path: str = Field(min_length=1)
    message: Literal[
        "Replace only the unknown contextual_type_id with an identifier declared in "
        "local_schema.contextual_types."
    ]


class DevelopmentRepairProbeInput(ImmutableRecord):
    """Complete deterministic input to the sole registered development repair."""

    parent_call_id: str = Field(min_length=1)
    parent_semantic_request_hash: Sha256Digest
    repair_semantic_request_hash: Sha256Digest
    parent_raw_output_hash: Sha256Digest
    diagnostic_fixture_id: Literal["development-schema-reference-repair-probe-v1"]
    fault_injection: Literal[
        "deterministic_unknown_reference_after_preserving_raw_parent"
    ]
    invalid_draft: Mapping[str, object]
    diagnostics: tuple[DevelopmentRepairDiagnostic, ...]
    created_at: AwareDatetime

    @model_validator(mode="after")
    def exact_single_diagnostic(self) -> Self:
        if len(self.diagnostics) != 1:
            raise ValueError("development repair probe requires exactly one diagnostic")
        path = self.diagnostics[0].path
        if not (
            path.startswith("instance_graph.entities.")
            or path.startswith("instance_graph.events.")
        ) or not path.endswith(".contextual_type_id"):
            raise ValueError("development repair diagnostic path is not registered")
        return self


class DevelopmentCallAuditReceipt(ImmutableRecord):
    """CAS/ledger bridge emitted for one registered GPU call."""

    receipt_id: str = Field(min_length=1)
    ordinal: int = Field(ge=1, le=DEVELOPMENT_CALL_COUNT)
    call_id: str = Field(min_length=1)
    condition: ConditionName
    outcome: RunOutcome
    request_started: bool
    service_identity: LogicalCASReference
    call_execution_envelope: LogicalCASReference | None = None
    job_id: str | None = None
    attempt_id: str | None = None
    model_call_id: str | None = None
    rendered_model_request: OpaqueJSONReference | None = None
    semantic_request: LogicalCASReference | None = None
    wire_alias_manifest: LogicalCASReference | None = None
    raw_response_artifact_hash: Sha256Digest | None = None
    validated_generation: LogicalCASReference | None = None
    condition_result: LogicalCASReference | None = None
    run_condition_config: LogicalCASReference | None = None
    packing_report: LogicalCASReference | None = None
    capability_manifest: LogicalCASReference | None = None
    comparison_input_manifest: LogicalCASReference | None = None
    evidence_packet: LogicalCASReference | None = None
    packet_materialization_event_hash: Sha256Digest | None = None
    query_context: LogicalCASReference | None = None
    treatment_switches: LogicalCASReference | None = None
    repair_probe_input: LogicalCASReference | None = None
    repair_preservation_report: LogicalCASReference | None = None
    fixed_schema_derivation: LogicalCASReference | None = None
    ledger_validation_ids: tuple[str, ...] = ()
    fixed_sealed_inventory: LogicalCASReference | None = None
    fixed_forbidden_probe: LogicalCASReference | None = None
    created_at: AwareDatetime

    @model_validator(mode="after")
    def exact_started_and_success_shapes(self) -> Self:
        started_fields = (
            self.job_id,
            self.attempt_id,
            self.model_call_id,
            self.call_execution_envelope,
            self.rendered_model_request,
            self.semantic_request,
            self.wire_alias_manifest,
            self.run_condition_config,
            self.packing_report,
            self.capability_manifest,
            self.treatment_switches,
        )
        if self.request_started:
            if any(item is None for item in started_fields):
                raise ValueError("started development calls require complete request lineage")
        elif any(item is not None for item in started_fields):
            raise ValueError("unstarted calls cannot claim request or ledger lineage")

        successful = self.outcome is RunOutcome.SUCCEEDED
        success_fields = (self.raw_response_artifact_hash, self.validated_generation)
        if successful and (
            not self.request_started or any(item is None for item in success_fields)
        ):
            raise ValueError("successful development calls require raw and validated outputs")
        if successful and self.condition_result is None:
            raise ValueError("successful development calls require their condition result")
        if not successful and self.condition_result is not None:
            raise ValueError("nonsuccessful calls cannot claim a valid condition result")
        if not successful and self.validated_generation is not None:
            raise ValueError("nonsuccessful calls cannot claim a validated generation")

        query_time = self.condition is not ConditionName.C1_LLM_PRE
        query_fields = (
            self.comparison_input_manifest,
            self.evidence_packet,
            self.packet_materialization_event_hash,
            self.query_context,
        )
        if self.request_started and query_time and any(item is None for item in query_fields):
            raise ValueError("started query-time calls require packet/context/fairness artifacts")
        if self.condition is ConditionName.C1_LLM_PRE and any(
            item is not None for item in query_fields
        ):
            raise ValueError("C1 preconstruction receipts cannot contain query artifacts")

        fixed_fields = (
            self.fixed_sealed_inventory,
            self.fixed_forbidden_probe,
            self.fixed_schema_derivation,
        )
        if self.condition is ConditionName.A_FIXED_SELECT:
            if self.request_started and any(item is None for item in fixed_fields):
                raise ValueError("FixedSelect requires its sealed inventory and rejection probe")
        elif any(item is not None for item in fixed_fields):
            raise ValueError("only FixedSelect may carry capability-rejection probe artifacts")
        if (
            self.condition is ConditionName.C2_LLM_QUERY
            and self.call_id == "dev-repair-probe-u04-q02"
        ):
            if self.request_started and self.repair_probe_input is None:
                raise ValueError("repair probe requires its deterministic repair input")
            if successful and self.repair_preservation_report is None:
                raise ValueError("successful repair requires its preservation report")
            if not successful and self.repair_preservation_report is not None:
                raise ValueError("failed repair cannot claim a preservation report")
        elif self.repair_probe_input is not None:
            raise ValueError("only the registered repair probe may carry repair input")
        elif self.repair_preservation_report is not None:
            raise ValueError("only the registered repair probe may carry repair validation")
        if self.request_started and not self.ledger_validation_ids:
            raise ValueError("started calls require append-only validation ledger identifiers")
        if len(self.ledger_validation_ids) != len(set(self.ledger_validation_ids)):
            raise ValueError("ledger validation identifiers must be unique")
        return self


class DevelopmentCPUProjectionReceipt(ImmutableRecord):
    """One deterministic C0/C1 query projection outside the 24 GPU calls."""

    receipt_id: str = Field(min_length=1)
    condition: Literal[ConditionName.C0_CLASSICAL_PRE, ConditionName.C1_LLM_PRE]
    unit_id: str = Field(pattern=r"^dev-unit-0[1-4]$")
    query_ordinal: Literal[1, 2, 3]
    job_id: str = Field(min_length=1)
    attempt_id: str = Field(min_length=1)
    ledger_projection_id: str = Field(min_length=1)
    ledger_validation_ids: tuple[str, ...]
    condition_attempt: LogicalCASReference
    projection: LogicalCASReference
    run_condition_config: LogicalCASReference
    comparison_input_manifest: LogicalCASReference
    evidence_packet: LogicalCASReference
    packet_materialization_event_hash: Sha256Digest
    query_context: LogicalCASReference
    created_at: AwareDatetime

    @model_validator(mode="after")
    def validation_ids_are_present_and_unique(self) -> Self:
        if not self.ledger_validation_ids:
            raise ValueError("CPU projection receipts require ledger validation rows")
        if len(self.ledger_validation_ids) != len(set(self.ledger_validation_ids)):
            raise ValueError("CPU projection validation identifiers must be unique")
        return self


class DevelopmentPackedRequestReceipt(ImmutableRecord):
    """Exact CPU tokenizer preflight for one complete model request."""

    call_id: str = Field(min_length=1)
    condition: ConditionName
    semantic_request: LogicalCASReference
    packing_equivalence_hash: Sha256Digest
    alias_manifest: LogicalCASReference
    rendered_model_request: OpaqueJSONReference
    packing_report: LogicalCASReference
    rendered_input_token_count: int = Field(ge=0, le=10_752)
    maximum_input_tokens: int = Field(gt=0, le=10_752)
    maximum_model_tokens: Literal[12_288] = 12_288
    semantic_evidence_truncation: Literal[False] = False

    @model_validator(mode="after")
    def exact_count_is_within_bound(self) -> Self:
        if self.rendered_input_token_count > self.maximum_input_tokens:
            raise ValueError("complete rendered request exceeds its frozen input cap")
        return self


class DevelopmentPackingPreflight(ImmutableRecord):
    """Query-blind exact packing gate for the four complete C1 snapshots."""

    preflight_id: str = Field(min_length=1)
    tokenizer_manifest_hash: Sha256Digest
    call_manifest_hash: Sha256Digest
    receipts: tuple[
        DevelopmentPackedRequestReceipt,
        DevelopmentPackedRequestReceipt,
        DevelopmentPackedRequestReceipt,
        DevelopmentPackedRequestReceipt,
    ]
    worst_rendered_input_tokens: int = Field(ge=0, le=10_240)
    all_complete_and_within_cap: Literal[True] = True
    completed_at: AwareDatetime

    @model_validator(mode="after")
    def exact_c1_preflight(self) -> Self:
        if tuple(item.call_id for item in self.receipts) != (
            "dev-c1-u01",
            "dev-c1-u02",
            "dev-c1-u03",
            "dev-c1-u04",
        ):
            raise ValueError("packing preflight must bind the exact ordered C1 calls")
        if any(item.condition is not ConditionName.C1_LLM_PRE for item in self.receipts):
            raise ValueError("query-blind packing preflight may contain only C1 requests")
        if self.worst_rendered_input_tokens != max(
            item.rendered_input_token_count for item in self.receipts
        ):
            raise ValueError("packing preflight worst count does not match its receipts")
        return self


class DevelopmentForecastInventoryRow(ImmutableRecord):
    """One frozen GPU-inventory row after fallback/development supersession."""

    call_class: str = Field(min_length=1)
    registered_count: int = Field(ge=0)
    consumed_before_development: int = Field(ge=0)
    development_call_count: int = Field(ge=0)
    remaining_after_development: int = Field(ge=0)
    provisional_p95_seconds: float = Field(ge=0.0)
    remaining_forecast_seconds: float = Field(ge=0.0)

    @model_validator(mode="after")
    def arithmetic_reconciles(self) -> Self:
        if (
            self.consumed_before_development
            + self.development_call_count
            + self.remaining_after_development
            != self.registered_count
        ):
            raise ValueError("development forecast inventory counts do not reconcile")
        if self.remaining_forecast_seconds != (
            self.remaining_after_development * self.provisional_p95_seconds
        ):
            raise ValueError("development forecast row seconds do not reconcile")
        return self


class DevelopmentForecastReceipt(ImmutableRecord):
    """Cumulative admission receipt that never replenishes consumed reserves."""

    receipt_id: str = Field(min_length=1)
    gpu_call_inventory_file_sha256: Sha256Digest
    actual_allocated_seconds_before_development: float = Field(ge=0.0)
    development_forecast_seconds: float = Field(ge=0.0)
    post_development_mandatory_forecast_seconds: float = Field(ge=0.0)
    total_forecast_seconds: float = Field(ge=0.0)
    scheduled_limit_seconds: float = Field(gt=0.0)
    hard_limit_seconds: float = Field(gt=0.0)
    inventory_rows: tuple[DevelopmentForecastInventoryRow, ...]
    retry_amendment_sha256: Sha256Digest | None = None
    second_recovery_overlay_sha256: Sha256Digest | None = None
    recovery_service_start_event_ids: tuple[str, ...] = ()
    authorized_additional_service_start_events: int = Field(default=0, ge=0, le=3)
    effective_accounting_events: int = Field(gt=0)
    effective_inference_attempts: int = Field(gt=0)
    normal_acceptance_superseded: Literal[True] = True
    consumed_reserves_not_replenished: Literal[True] = True
    admitted: bool
    created_at: AwareDatetime

    @model_validator(mode="after")
    def exact_cumulative_arithmetic(self) -> Self:
        remaining = sum(item.remaining_forecast_seconds for item in self.inventory_rows)
        expected = (
            self.actual_allocated_seconds_before_development
            + self.development_forecast_seconds
            + self.post_development_mandatory_forecast_seconds
        )
        if self.post_development_mandatory_forecast_seconds != remaining:
            raise ValueError("post-development forecast differs from its inventory rows")
        if self.total_forecast_seconds != expected:
            raise ValueError("development cumulative forecast arithmetic is inconsistent")
        if self.admitted != (expected <= self.scheduled_limit_seconds):
            raise ValueError("development forecast admission flag is inconsistent")
        if self.scheduled_limit_seconds >= self.hard_limit_seconds:
            raise ValueError("scheduled limit must remain below the hard stop")
        if self.authorized_additional_service_start_events != len(
            self.recovery_service_start_event_ids
        ):
            raise ValueError("recovery service-start overlay count is inconsistent")
        if self.second_recovery_overlay_sha256 is not None:
            if (
                self.retry_amendment_sha256 is None
                or self.recovery_service_start_event_ids
                not in {
                    HISTORICAL_SECOND_FALLBACK_RECOVERY_SERVICE_START_EVENT_IDS,
                    SECOND_FALLBACK_RECOVERY_SERVICE_START_EVENT_IDS,
                }
            ):
                raise ValueError(
                    "second recovery must bind an exact ordered v3+v7[/v8] lineage"
                )
        elif (
            len(self.recovery_service_start_event_ids) > 1
            or bool(self.retry_amendment_sha256)
            != bool(self.authorized_additional_service_start_events)
        ):
            raise ValueError("ordinary recovery must bind at most one service start")
        if len(set(self.recovery_service_start_event_ids)) != len(
            self.recovery_service_start_event_ids
        ) or any(
            not identifier
            or len(identifier) > 160
            or any(
                character not in "abcdefghijklmnopqrstuvwxyz0123456789._-"
                for character in identifier
            )
            for identifier in self.recovery_service_start_event_ids
        ):
            raise ValueError("recovery service-start event identifiers are invalid")
        registered_events = sum(item.registered_count for item in self.inventory_rows)
        registered_inference = sum(
            item.registered_count
            for item in self.inventory_rows
            if item.call_class != "gpu_session_start"
        )
        if self.effective_accounting_events != (
            registered_events + self.authorized_additional_service_start_events
        ):
            raise ValueError("effective accounting-event count omits the recovery overlay")
        if self.effective_inference_attempts != registered_inference:
            raise ValueError("recovery overlay changed the inference-attempt inventory")
        return self


class DevelopmentPreparationIndex(ImmutableRecord):
    """Crash-safe CAS root for the complete query-blind development preparation."""

    index_id: str = Field(min_length=1)
    bootstrap_hash: Sha256Digest
    execution_id: str = Field(min_length=1)
    execution_manifest_hash: Sha256Digest
    call_manifest_hash: Sha256Digest
    prequery_inputs: LogicalCASReference
    forecast_control: LogicalCASReference
    forecast_receipt: LogicalCASReference
    runtime_source_manifest: OpaqueJSONReference
    construction_configuration: LogicalCASReference
    prequery_preparation_artifacts: tuple[LogicalCASReference, ...]
    run_condition_config_artifacts: tuple[LogicalCASReference, ...]
    packing_preflight_artifact_hash: Sha256Digest
    checkpoint_path: str = Field(min_length=1)
    adapter_state_path: str = Field(min_length=1)
    created_at: AwareDatetime

    @model_validator(mode="after")
    def exact_query_blind_inventory(self) -> Self:
        if len(self.prequery_preparation_artifacts) != 11:
            raise ValueError("preparation index requires exactly 11 prequery preparations")
        if len(self.run_condition_config_artifacts) != 20:
            raise ValueError("preparation index requires 20 non-Fixed run configurations")
        if self.runtime_source_manifest.object_kind != "runtime_source_manifest":
            raise ValueError("preparation index requires a runtime source manifest")
        if self.prequery_inputs.object_kind != "development_prequery_inputs":
            raise ValueError("preparation index requires typed prequery inputs")
        return self


class DevelopmentAssessmentBundle(ImmutableRecord):
    """Gold-free post-run index handed to the isolated development assessor."""

    bundle_id: str = Field(min_length=1)
    call_manifest_hash: Sha256Digest
    prequery_inputs_hash: Sha256Digest
    source_tree_hash: Sha256Digest
    runtime_source_manifest: OpaqueJSONReference
    benchmark_manifest_file_sha256: Sha256Digest
    prequery_preparation_artifacts: tuple[LogicalCASReference, ...]
    call_receipt_artifact_hashes: tuple[Sha256Digest, ...]
    service_result_artifact_hashes: tuple[Sha256Digest, ...]
    cpu_projection_receipt_artifact_hashes: tuple[Sha256Digest, ...]
    packing_preflight_artifact_hash: Sha256Digest
    created_at: AwareDatetime

    @model_validator(mode="after")
    def exact_development_inventory(self) -> Self:
        if self.runtime_source_manifest.object_kind != "runtime_source_manifest":
            raise ValueError("assessment bundle requires the pre-call runtime source manifest")
        inventories = (
            (self.prequery_preparation_artifacts, 11, "prequery preparations"),
            (self.call_receipt_artifact_hashes, DEVELOPMENT_CALL_COUNT, "call receipts"),
            (self.service_result_artifact_hashes, DEVELOPMENT_CALL_COUNT, "service results"),
            (
                self.cpu_projection_receipt_artifact_hashes,
                DEVELOPMENT_CALL_COUNT,
                "CPU projections",
            ),
        )
        for values, expected, name in inventories:
            if len(values) != expected or len(values) != len(set(values)):
                raise ValueError(f"assessment bundle requires {expected} unique {name}")
        return self


__all__ = [
    "HISTORICAL_SECOND_FALLBACK_RECOVERY_SERVICE_START_EVENT_IDS",
    "SECOND_FALLBACK_RECOVERY_SERVICE_START_EVENT_IDS",
    "DevelopmentAssessmentBundle",
    "DevelopmentCPUProjectionReceipt",
    "DevelopmentCallAuditReceipt",
    "DevelopmentForecastInventoryRow",
    "DevelopmentForecastReceipt",
    "DevelopmentPackedRequestReceipt",
    "DevelopmentPackingPreflight",
    "DevelopmentPreparationIndex",
    "DevelopmentRepairDiagnostic",
    "DevelopmentRepairProbeInput",
    "DevelopmentTreatmentSwitches",
    "LogicalCASReference",
    "OpaqueJSONReference",
]
