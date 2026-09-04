"""Gold-free execution engine for the frozen development service adapter.

The engine receives a lifecycle-free adapter around the already-loaded fallback
model.  It constructs only registered semantic requests, persists complete
CAS/ledger lineage, and returns terminal service receipts.  It never imports a
scorer namespace and cannot start, load, restart, or stop the model service.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

from pydantic import ValidationError

from story_projection_onto.benchmark_runtime import (
    ModelEligibleWorldArtifact,
    NeutralEvidenceArtifact,
)
from story_projection_onto.conditions.base import (
    SCORED_PROJECTION_SCHEMA_HASH,
    ComparisonInputManifest,
    ConditionAttemptRecord,
    ConditionPreparation,
    ProduceInputs,
    RunConditionConfig,
)
from story_projection_onto.conditions.c1 import (
    build_c1_preconstruction_request,
    seal_c1_preconstruction,
)
from story_projection_onto.conditions.c2 import (
    build_c2_construction_request,
    finalize_c2_draft,
)
from story_projection_onto.conditions.fixed_select import (
    build_fixed_select_request,
    finalize_fixed_select_draft,
    prepare_fixed_selection,
)
from story_projection_onto.contracts import (
    CommitmentCheckStatus,
    ConditionName,
    ConstructionOperator,
    ConstructionRequest,
    EvidencePacket,
    EvidenceSupportStatus,
    NoTemporalEpistemicOntologyDraft,
    OntologyDraft,
    PacketMaterializationEvent,
    PreconstructionRequest,
    PrequeryPreparationBinding,
    RetrievalMethod,
    RunOutcome,
    TemporalDeterminationStatus,
    ValidatedGeneration,
    ValidationRecord,
    ValidationStatus,
    canonical_json,
    canonical_sha256,
    normalize_generation_metadata,
    to_model_visible_packet,
)
from story_projection_onto.development_adapter import (
    DevelopmentConstructionConfiguration,
    ModelWireAliasManifest,
    ProductionDevelopmentServiceAdapter,
    build_development_guided_request,
    build_development_repair_probe_input,
    development_request_runtime,
    development_runtime_identifiers,
    encode_development_semantic_request,
    model_wire_source_alias_bijection,
    persist_logical_record,
    persist_opaque_json,
    restore_model_output_source_aliases,
    treatment_switches_for,
)
from story_projection_onto.development_artifacts import (
    DevelopmentCallAuditReceipt,
    DevelopmentRepairProbeInput,
    LogicalCASReference,
)
from story_projection_onto.development_runtime import (
    CallExecutionEnvelope,
    DevelopmentCallKind,
    DevelopmentCallManifest,
    DevelopmentCallSpec,
    FixedSchemaDerivationPlan,
    FixedSchemaDerivationReceipt,
    ServiceCallResult,
)
from story_projection_onto.gpu_runtime import GenerationResult, GuidedJSONRequest, TokenizerManifest
from story_projection_onto.llm import (
    CapabilityManifest,
    FixedSelectCapabilityError,
    FixedSelectOutputAudit,
    ProposedOperation,
    SealedOntologyInventory,
    base_condition_output_schema,
    enforce_fixed_select_output,
    fixed_select_audit_from_draft,
    normalize_no_temporal_epistemic_draft,
    sealed_inventory_from_fixed_ontology,
)
from story_projection_onto.query_runtime import AuditedPacketMaterialization
from story_projection_onto.store import (
    AttemptKind,
    FailureKind,
    InputKind,
    ModelBackend,
    ModelCallRole,
    ReleaseClass,
    RetryClass,
    TemporalValidationStatus,
)
from story_projection_onto.store import (
    CommitmentCheckStatus as LedgerCommitmentCheckStatus,
)
from story_projection_onto.store import (
    EvidenceSupportStatus as LedgerEvidenceSupportStatus,
)
from story_projection_onto.store import (
    ValidationStatus as LedgerValidationStatus,
)
from story_projection_onto.validate import (
    validate_draft_structure,
    validate_repair_preservation,
)


class DevelopmentExecutionError(RuntimeError):
    """A production development call failed a frozen execution invariant."""


@dataclass(slots=True)
class DevelopmentExecutionRepository:
    """All query-blind inputs plus post-query materialization caches."""

    root: Path
    execution_id: str
    manifest: DevelopmentCallManifest
    construction: DevelopmentConstructionConfiguration
    tokenizer: object
    tokenizer_manifest: TokenizerManifest
    source_tree_hash: str
    selected_model_freeze_hash: str
    model_manifest_hash: str
    seed_manifest_hash: str
    validator_hash: str
    neutral_by_unit: Mapping[str, NeutralEvidenceArtifact]
    model_visible_by_unit: Mapping[str, ModelEligibleWorldArtifact]
    preparations: dict[tuple[str, ConditionName], ConditionPreparation]
    preparation_references: tuple[LogicalCASReference, ...]
    run_configs: dict[int, RunConditionConfig]
    fixed_schema_plans: Mapping[int, FixedSchemaDerivationPlan]
    packing_preflight_artifact_hash: str
    post_development_forecast_seconds: float
    c1_requests: dict[str, PreconstructionRequest] = field(default_factory=dict)
    semantic_requests: dict[str, PreconstructionRequest | ConstructionRequest] = field(
        default_factory=dict
    )
    alias_manifests: dict[str, ModelWireAliasManifest] = field(default_factory=dict)
    repair_inputs: dict[str, DevelopmentRepairProbeInput] = field(default_factory=dict)
    materializations: dict[str, AuditedPacketMaterialization] = field(default_factory=dict)
    c1_preparation_references: dict[str, LogicalCASReference] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.root = self.root.resolve(strict=True)
        expected_exact = set(range(1, 25)) - {17, 18, 19, 20}
        if set(self.run_configs) != expected_exact:
            raise DevelopmentExecutionError(
                "executor requires exact non-Fixed configs and four deferred plans"
            )
        if set(self.fixed_schema_plans) != {17, 18, 19, 20}:
            raise DevelopmentExecutionError("executor lacks four FixedSelect derivation plans")
        for call in self.manifest.calls:
            config = self.run_configs.get(call.ordinal)
            if config is not None and call.condition is not config.condition:
                raise DevelopmentExecutionError(
                    "call manifest and run configuration conditions differ"
                )
            plan = self.fixed_schema_plans.get(call.ordinal)
            if plan is not None and (
                call.kind is not DevelopmentCallKind.FIXED_SELECTION
                or plan.call_id != call.call_id
                or plan.unit_id != call.unit_id
            ):
                raise DevelopmentExecutionError(
                    "FixedSelect derivation plan differs from call manifest"
                )

    def run_config(self, call: DevelopmentCallSpec) -> RunConditionConfig:
        try:
            return self.run_configs[call.ordinal]
        except KeyError as exc:
            raise DevelopmentExecutionError(
                "exact FixedSelect config has not yet been derived from its C1 seal"
            ) from exc


def _event_for(adapter: ProductionDevelopmentServiceAdapter, event_id: str) -> Any:
    matches = tuple(
        item
        for item in adapter.artifacts.ledger.gpu_events_with_prefix(event_id)
        if item.event_id == event_id
    )
    if len(matches) != 1:
        raise DevelopmentExecutionError("development call requires exactly one metered GPU event")
    return matches[0]


def _parse_utc(value: str) -> datetime:
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        raise DevelopmentExecutionError("GPU event timestamp is not timezone-aware")
    return parsed.astimezone(UTC)


def _call_role(call: DevelopmentCallSpec) -> ModelCallRole:
    if call.kind is DevelopmentCallKind.C1_PRECONSTRUCTION:
        return ModelCallRole.PREBUILD
    if call.kind is DevelopmentCallKind.FIXED_SELECTION:
        return ModelCallRole.FIXED_SELECT
    if call.kind is DevelopmentCallKind.REPAIR_PROBE:
        return ModelCallRole.REPAIR
    return ModelCallRole.QUERY_TIME


def _retry_class(call: DevelopmentCallSpec) -> RetryClass:
    if call.kind is DevelopmentCallKind.C1_PRECONSTRUCTION:
        return RetryClass.LONG
    if call.kind in {DevelopmentCallKind.FIXED_SELECTION, DevelopmentCallKind.REPAIR_PROBE}:
        return RetryClass.SHORT
    return RetryClass.STANDARD


def _failure_kind(exc: BaseException) -> FailureKind:
    if isinstance(exc, TimeoutError):
        return FailureKind.TIMEOUT
    if isinstance(exc, (ValidationError, ValueError)):
        return FailureKind.INVALID_OUTPUT
    if isinstance(exc, MemoryError) or "out of memory" in str(exc).casefold():
        return FailureKind.OUT_OF_MEMORY
    return FailureKind.SERVICE


@dataclass(slots=True)
class ProductionDevelopmentCallExecutor:
    """Execute all 24 registered calls on a single injected metered service."""

    repository: DevelopmentExecutionRepository
    clock: Callable[[], datetime] = lambda: datetime.now(UTC)

    def _now(self) -> datetime:
        value = self.clock()
        if value.tzinfo is None or value.utcoffset() is None:
            raise DevelopmentExecutionError("executor clock must be timezone-aware")
        return value.astimezone(UTC)

    def _strictly_after(self, threshold: datetime) -> datetime:
        for _ in range(100):
            value = self._now()
            if value > threshold:
                return value
        raise DevelopmentExecutionError("executor clock did not advance")

    def _fixed_call_for_unit(self, unit_id: str) -> DevelopmentCallSpec:
        matches = tuple(
            call
            for call in self.repository.manifest.calls
            if call.unit_id == unit_id
            and call.kind is DevelopmentCallKind.FIXED_SELECTION
        )
        if len(matches) != 1:
            raise DevelopmentExecutionError(
                "development unit does not have exactly one FixedSelect call"
            )
        return matches[0]

    def _validate_fixed_derivation(
        self,
        *,
        call: DevelopmentCallSpec,
        preparation: ConditionPreparation,
        receipt: FixedSchemaDerivationReceipt,
        config: RunConditionConfig,
    ) -> None:
        """Recompute every deterministic post-C1 field from sealed inputs."""

        plan = self.repository.fixed_schema_plans.get(call.ordinal)
        source = preparation.sealed_preontology
        if plan is None or source is None:
            raise DevelopmentExecutionError("FixedSelect derivation lacks its plan or C1 seal")
        fixed = source.as_fixed_ontology()
        evidence = self.repository.model_visible_by_unit[call.unit_id].evidence
        aliases = model_wire_source_alias_bijection(evidence)
        runtime = development_request_runtime(
            root=self.repository.root,
            condition=ConditionName.A_FIXED_SELECT,
            tokenizer_manifest=self.repository.tokenizer_manifest,
            seed=call.vllm_seed,
            fixed_ontology=fixed,
            fixed_evidence=evidence,
        )
        expected_plan = (
            plan.ordinal,
            plan.call_id,
            plan.unit_id,
            plan.source_c1_call_id,
            plan.seed_block,
            plan.resolved_seed,
            plan.budgets,
            plan.maximum_input_tokens,
            plan.maximum_output_tokens,
            plan.repair_attempt_budget,
            plan.model_stack_hash,
            plan.decoding_family_hash,
            plan.seed_manifest_hash,
            plan.prompt_hash,
            plan.base_output_schema_hash,
            plan.scored_schema_hash,
            plan.capability_manifest_hash,
            plan.validator_hash,
            plan.upper_ontology_hash,
            plan.prequery_evidence_artifact_hash,
            plan.query_stage_manifest_hash,
        )
        observed_plan = (
            call.ordinal,
            call.call_id,
            call.unit_id,
            call.source_c1_call_id,
            call.seed_block,
            call.vllm_seed,
            self.repository.construction.projection_budgets_by_unit[call.unit_id],
            self.repository.construction.first_pass_input_tokens,
            self.repository.construction.first_pass_output_tokens,
            self.repository.construction.projection_budgets_by_unit[
                call.unit_id
            ].repair_attempt_budget,
            self.repository.model_manifest_hash,
            runtime.decoding_manifest.comparison_family_hash,
            self.repository.seed_manifest_hash,
            runtime.prompt_hash,
            canonical_sha256(
                base_condition_output_schema(ConditionName.A_FIXED_SELECT)
            ),
            SCORED_PROJECTION_SCHEMA_HASH,
            runtime.capability_manifest.content_hash,
            self.repository.validator_hash,
            self.repository.construction.upper_ontology.content_hash,
            call.prequery_stage.evidence_artifact_hash,
            cast(Any, call.query_stage).staging_manifest_hash,
        )
        if expected_plan != observed_plan:
            raise DevelopmentExecutionError(
                "FixedSelect derivation plan differs from frozen prequery inputs"
            )
        expected_config = RunConditionConfig(
            config_id=f"run-config-{call.call_id}",
            condition=ConditionName.A_FIXED_SELECT,
            budgets=plan.budgets,
            maximum_input_tokens=plan.maximum_input_tokens,
            maximum_output_tokens=plan.maximum_output_tokens,
            repair_attempt_budget=plan.repair_attempt_budget,
            seed_block=plan.seed_block,
            source_c1_seed_block=plan.seed_block,
            model_stack_hash=plan.model_stack_hash,
            decoding_manifest_hash=runtime.decoding_manifest.content_hash,
            decoding_family_hash=runtime.decoding_manifest.comparison_family_hash,
            seed_manifest_hash=plan.seed_manifest_hash,
            resolved_seed=plan.resolved_seed,
            prompt_hash=runtime.prompt_hash,
            output_schema_hash=runtime.output_schema_hash,
            scored_schema_hash=plan.scored_schema_hash,
            capability_manifest_hash=runtime.capability_manifest.content_hash,
            validator_hash=plan.validator_hash,
            upper_ontology_hash=plan.upper_ontology_hash,
        )
        expected_receipt = (
            receipt.plan_hash,
            receipt.call_id,
            receipt.unit_id,
            receipt.source_c1_construction_seal_hash,
            receipt.source_c1_preparation_hash,
            receipt.fixed_ontology_hash,
            receipt.evidence_alias_bijection_hash,
            receipt.derived_output_schema_hash,
            receipt.derived_decoding_manifest_hash,
            receipt.exact_run_condition_config_hash,
        )
        observed_receipt = (
            plan.content_hash,
            call.call_id,
            call.unit_id,
            source.construction_seal.content_hash,
            preparation.content_hash,
            fixed.content_hash,
            canonical_sha256(aliases),
            runtime.output_schema_hash,
            runtime.decoding_manifest.content_hash,
            expected_config.content_hash,
        )
        expected_config_artifact_hash = hashlib.sha256(
            (expected_config.to_canonical_json() + "\n").encode("utf-8")
        ).hexdigest()
        if (
            expected_receipt != observed_receipt
            or config != expected_config
            or receipt.exact_run_condition_config_artifact_hash
            != expected_config_artifact_hash
        ):
            raise DevelopmentExecutionError(
                "FixedSelect derivation receipt is not reproducible from its C1 seal"
            )

    def _derive_fixed_schema_after_c1(
        self,
        *,
        adapter: ProductionDevelopmentServiceAdapter,
        c1_call: DevelopmentCallSpec,
        preparation: ConditionPreparation,
    ) -> FixedSchemaDerivationReceipt:
        """Persist the exact Fixed grammar/config before any query is opened."""

        call = self._fixed_call_for_unit(c1_call.unit_id)
        plan = self.repository.fixed_schema_plans[call.ordinal]
        source = preparation.sealed_preontology
        if source is None:
            raise DevelopmentExecutionError("successful C1 lacks its sealed preontology")
        fixed = source.as_fixed_ontology()
        evidence = self.repository.model_visible_by_unit[call.unit_id].evidence
        aliases = model_wire_source_alias_bijection(evidence)
        runtime = development_request_runtime(
            root=self.repository.root,
            condition=ConditionName.A_FIXED_SELECT,
            tokenizer_manifest=self.repository.tokenizer_manifest,
            seed=call.vllm_seed,
            fixed_ontology=fixed,
            fixed_evidence=evidence,
        )
        config = RunConditionConfig(
            config_id=f"run-config-{call.call_id}",
            condition=ConditionName.A_FIXED_SELECT,
            budgets=plan.budgets,
            maximum_input_tokens=plan.maximum_input_tokens,
            maximum_output_tokens=plan.maximum_output_tokens,
            repair_attempt_budget=plan.repair_attempt_budget,
            seed_block=plan.seed_block,
            source_c1_seed_block=plan.seed_block,
            model_stack_hash=plan.model_stack_hash,
            decoding_manifest_hash=runtime.decoding_manifest.content_hash,
            decoding_family_hash=runtime.decoding_manifest.comparison_family_hash,
            seed_manifest_hash=plan.seed_manifest_hash,
            resolved_seed=plan.resolved_seed,
            prompt_hash=runtime.prompt_hash,
            output_schema_hash=runtime.output_schema_hash,
            scored_schema_hash=plan.scored_schema_hash,
            capability_manifest_hash=runtime.capability_manifest.content_hash,
            validator_hash=plan.validator_hash,
            upper_ontology_hash=plan.upper_ontology_hash,
        )
        derived_at = self._strictly_after(preparation.completed_at)
        config_ref = persist_logical_record(
            adapter.artifacts,
            config,
            object_kind="run_condition_config",
            created_at=derived_at,
        )
        receipt = FixedSchemaDerivationReceipt(
            plan_hash=plan.content_hash,
            call_id=call.call_id,
            unit_id=call.unit_id,
            source_c1_construction_seal_hash=source.construction_seal.content_hash,
            source_c1_preparation_hash=preparation.content_hash,
            fixed_ontology_hash=fixed.content_hash,
            evidence_alias_bijection_hash=canonical_sha256(aliases),
            derived_output_schema_hash=runtime.output_schema_hash,
            derived_decoding_manifest_hash=runtime.decoding_manifest.content_hash,
            exact_run_condition_config_hash=config.content_hash,
            exact_run_condition_config_artifact_hash=config_ref.artifact_hash,
            derived_at=derived_at,
        )
        self._validate_fixed_derivation(
            call=call,
            preparation=preparation,
            receipt=receipt,
            config=config,
        )
        receipt_ref = persist_logical_record(
            adapter.artifacts,
            receipt,
            object_kind="fixed_schema_derivation_receipt",
            created_at=derived_at,
        )
        state = adapter._state()
        derivations = dict(state.fixed_schema_derivations)
        existing = derivations.get(call.call_id)
        if existing is not None and existing != receipt_ref.artifact_hash:
            raise DevelopmentExecutionError(
                "FixedSelect call already has another schema derivation receipt"
            )
        derivations[call.call_id] = receipt_ref.artifact_hash
        adapter._write_state(state, fixed_schema_derivations=derivations)
        self.repository.run_configs[call.ordinal] = config
        return receipt

    def _load_fixed_derivation(
        self,
        *,
        adapter: ProductionDevelopmentServiceAdapter,
        call: DevelopmentCallSpec,
    ) -> FixedSchemaDerivationReceipt:
        """Rehydrate an exact derived Fixed config without reopening a query."""

        state = adapter._state()
        artifact_hash = state.fixed_schema_derivations.get(call.call_id)
        if artifact_hash is None:
            raise DevelopmentExecutionError("FixedSelect schema derivation is unavailable")
        receipt_record = adapter.artifacts.ledger.get_artifact(artifact_hash)
        receipt = FixedSchemaDerivationReceipt.model_validate_json(
            adapter.artifacts.blobs.read_bytes(receipt_record)
        )
        config_record = adapter.artifacts.ledger.get_artifact(
            receipt.exact_run_condition_config_artifact_hash
        )
        config = RunConditionConfig.model_validate_json(
            adapter.artifacts.blobs.read_bytes(config_record)
        )
        preparation = self.repository.preparations.get(
            (call.unit_id, ConditionName.C1_LLM_PRE)
        )
        if preparation is None:
            raise DevelopmentExecutionError(
                "FixedSelect schema cannot be recovered before its C1 preparation"
            )
        self._validate_fixed_derivation(
            call=call,
            preparation=preparation,
            receipt=receipt,
            config=config,
        )
        self.repository.run_configs[call.ordinal] = config
        return receipt

    def rehydrate_completed_call(
        self,
        adapter: ProductionDevelopmentServiceAdapter,
        call: DevelopmentCallSpec,
        receipt: DevelopmentCallAuditReceipt,
        result: ServiceCallResult,
    ) -> None:
        """Restore deterministic in-memory dependencies from durable CAS only."""

        if result.outcome is not RunOutcome.SUCCEEDED:
            return
        if call.query_stage is not None:
            materialized = self._query_materialization(adapter, call)
            if (
                receipt.packet_materialization_event_hash
                != materialized.event.content_hash
            ):
                raise DevelopmentExecutionError(
                    "recovered query packet differs from its call receipt"
                )
        if call.kind is DevelopmentCallKind.C1_PRECONSTRUCTION:
            reference = receipt.condition_result
            if reference is None or reference.object_kind != "condition_preparation":
                raise DevelopmentExecutionError(
                    "recovered C1 receipt lacks its preparation artifact"
                )
            record = adapter.artifacts.ledger.get_artifact(reference.artifact_hash)
            preparation = ConditionPreparation.model_validate_json(
                adapter.artifacts.blobs.read_bytes(record)
            )
            bindings = result.prequery_preparation_bindings
            if (
                preparation.content_hash != reference.logical_content_hash
                or preparation.content_hash != result.condition_preparation_hash
                or len(bindings) != 2
                or bindings[0].preparation_hash != preparation.content_hash
            ):
                raise DevelopmentExecutionError(
                    "recovered C1 preparation differs from its result lineage"
                )
            fixed = prepare_fixed_selection(
                preparation,
                seed_block=call.seed_block,
                prepared_at=bindings[1].completed_at,
            )
            if (
                fixed.content_hash != bindings[1].preparation_hash
                or cast(Any, fixed.fixed_selection).content_hash
                != bindings[1].lineage_artifact_hash
            ):
                raise DevelopmentExecutionError(
                    "recovered same-seed Fixed preparation differs from its C1 binding"
                )
            self.repository.preparations[
                (call.unit_id, ConditionName.C1_LLM_PRE)
            ] = preparation
            self.repository.preparations[
                (call.unit_id, ConditionName.A_FIXED_SELECT)
            ] = fixed
            self.repository.c1_preparation_references[call.unit_id] = reference
            self._load_fixed_derivation(
                adapter=adapter,
                call=self._fixed_call_for_unit(call.unit_id),
            )
        elif call.kind is DevelopmentCallKind.FIXED_SELECTION:
            derivation = self._load_fixed_derivation(adapter=adapter, call=call)
            reference = receipt.fixed_schema_derivation
            if (
                reference is None
                or reference.logical_content_hash != derivation.content_hash
                or result.fixed_schema_derivation != derivation
            ):
                raise DevelopmentExecutionError(
                    "recovered FixedSelect result changed its schema derivation"
                )

    def _query_materialization(
        self,
        adapter: ProductionDevelopmentServiceAdapter,
        call: DevelopmentCallSpec,
    ) -> AuditedPacketMaterialization:
        stage = call.query_stage
        if stage is None or adapter.query_runtime is None:
            raise DevelopmentExecutionError("query-time call lacks audited runtime")
        cached = self.repository.materializations.get(stage.staging_manifest_hash)
        if cached is not None:
            return cached
        opening = adapter.audited_opening(stage)
        neutral = self.repository.neutral_by_unit[call.unit_id]
        if neutral.snapshot != opening.evidence.snapshot:
            raise DevelopmentExecutionError("neutral and model-visible evidence snapshots differ")
        retrieval_hash = canonical_sha256(
            {
                "policy": self.repository.construction.retrieval_policy,
                "unit_id": call.unit_id,
                "stage_manifest_hash": stage.staging_manifest_hash,
            }
        )
        persisted_hash = adapter._state().packet_materialization_events.get(
            stage.staging_manifest_hash
        )
        if persisted_hash is not None:
            record = adapter.artifacts.ledger.get_packet_materialization(
                persisted_hash
            )
            packet_artifact = adapter.artifacts.ledger.get_artifact(
                record.packet_artifact_hash
            )
            event_artifact = adapter.artifacts.ledger.get_artifact(
                record.materialization_event_artifact_hash
            )
            packet = EvidencePacket.model_validate_json(
                adapter.artifacts.blobs.read_bytes(packet_artifact)
            )
            event = PacketMaterializationEvent.model_validate_json(
                adapter.artifacts.blobs.read_bytes(event_artifact)
            )
            if (
                event.content_hash != persisted_hash
                or event.packet_hash != packet.content_hash
                or event.query_access_event_hash != opening.access_event.content_hash
                or event.snapshot_hash != neutral.snapshot.content_hash
                or event.retrieval_config_hash != retrieval_hash
                or record.packet_hash != packet.content_hash
                or record.execution_id != self.repository.execution_id
            ):
                raise DevelopmentExecutionError(
                    "persisted packet materialization changed on controller resume"
                )
            recovered = AuditedPacketMaterialization(
                packet=packet,
                event=event,
                persistence=record,
            )
            self._register_packet_input(adapter, recovered)
            self.repository.materializations[stage.staging_manifest_hash] = recovered
            return recovered

        def materializer(
            evidence: ModelEligibleWorldArtifact,
            _reveal: object,
        ) -> EvidencePacket:
            if evidence != opening.evidence:
                raise DevelopmentExecutionError("materializer received changed staged evidence")
            created_at = self._now()
            token_count = sum(
                len(
                    cast(Any, self.repository.tokenizer).encode(
                        item.text,
                        add_special_tokens=False,
                    )
                )
                for item in neutral.evidence
            )
            return EvidencePacket(
                packet_id=(
                    "packet-"
                    + canonical_sha256(
                        {
                            "execution": self.repository.execution_id,
                            "stage": stage.staging_manifest_hash,
                        }
                    )[:20]
                ),
                snapshot_hash=neutral.snapshot.content_hash,
                evidence=neutral.evidence,
                ordered_evidence_ids=neutral.snapshot.eligible_evidence_ids,
                retrieval_method=RetrievalMethod.ALL_ADMISSIBLE,
                token_count=token_count,
                created_at=created_at,
                release_class=neutral.snapshot.release_class,
            )

        materialized = adapter.query_runtime.materialize_packet(
            opening,
            materialization_event_id=(
                f"{self.repository.execution_id}-packet-{stage.stage_id.removeprefix('stage_')}"
            ),
            retrieval_config_hash=retrieval_hash,
            materializer=materializer,
        )
        self._register_packet_input(adapter, materialized)
        self.repository.materializations[stage.staging_manifest_hash] = materialized
        state = adapter._state()
        events = dict(state.packet_materialization_events)
        existing = events.get(stage.staging_manifest_hash)
        if existing is not None and existing != materialized.event.content_hash:
            raise DevelopmentExecutionError(
                "query stage already maps to another packet materialization"
            )
        events[stage.staging_manifest_hash] = materialized.event.content_hash
        adapter._write_state(state, packet_materialization_events=events)
        return materialized

    def _register_packet_input(
        self,
        adapter: ProductionDevelopmentServiceAdapter,
        materialized: AuditedPacketMaterialization,
    ) -> None:
        ledger = adapter.artifacts.ledger
        packet = materialized.packet
        ledger.register_input(
            input_id=packet.packet_id,
            study_id=self.repository.execution_id,
            input_kind=InputKind.EVIDENCE_PACKET,
            content_hash=packet.content_hash,
            artifact_hash=materialized.persistence.packet_artifact_hash,
            release_class=ReleaseClass.PUBLIC,
            created_at=packet.created_at,
        )

    def _produce_inputs(
        self,
        adapter: ProductionDevelopmentServiceAdapter,
        call: DevelopmentCallSpec,
        envelope: CallExecutionEnvelope,
        config: RunConditionConfig,
    ) -> tuple[ProduceInputs, AuditedPacketMaterialization]:
        stage = call.query_stage
        access = envelope.query_access_event
        if stage is None or access is None or envelope.prequery_barrier_hash is None:
            raise DevelopmentExecutionError("query-time envelope is incomplete")
        opening = adapter.audited_opening(stage)
        if opening.access_event != access:
            raise DevelopmentExecutionError("query opening differs from execution envelope")
        materialized = self._query_materialization(adapter, call)
        preparation = self.repository.preparations.get((call.unit_id, call.condition))
        if call.kind is DevelopmentCallKind.FIXED_SELECTION:
            preparation = self.repository.preparations.get(
                (call.unit_id, ConditionName.A_FIXED_SELECT)
            )
        if preparation is None:
            raise DevelopmentExecutionError("condition preparation is unavailable")
        barrier = adapter.persisted_prequery_barrier(envelope.prequery_barrier_hash)
        return (
            ProduceInputs(
                preparation=preparation,
                snapshot=self.repository.neutral_by_unit[call.unit_id].snapshot,
                packet=materialized.packet,
                context=opening.context,
                query_access=opening.access_event,
                prequery_barrier=barrier,
                query_processing_started_at=opening.access_event.accessed_at,
                packet_materialization=materialized.event,
                upper_ontology=self.repository.construction.upper_ontology,
                run_config=config,
            ),
            materialized,
        )

    def _semantic_request(
        self,
        adapter: ProductionDevelopmentServiceAdapter,
        call: DevelopmentCallSpec,
        envelope: CallExecutionEnvelope,
        config: RunConditionConfig,
    ) -> tuple[
        PreconstructionRequest | ConstructionRequest,
        ProduceInputs | None,
        AuditedPacketMaterialization | None,
    ]:
        requested_at = self._strictly_after(envelope.preconstruction_barrier_recorded_at)
        if call.kind is DevelopmentCallKind.C1_PRECONSTRUCTION:
            runtime = development_runtime_identifiers(
                root=self.repository.root,
                condition=call.condition,
                tokenizer_manifest=self.repository.tokenizer_manifest,
                seed=call.vllm_seed,
            )
            neutral = self.repository.neutral_by_unit[call.unit_id]
            request = build_c1_preconstruction_request(
                snapshot_hash=neutral.snapshot.content_hash,
                snapshot_sealed_at=neutral.snapshot.sealed_at,
                ordered_snapshot_evidence_ids=neutral.snapshot.eligible_evidence_ids,
                evidence=neutral.evidence,
                upper_ontology=self.repository.construction.upper_ontology,
                preconstruction_budgets=config.budgets,
                runtime=runtime,
                requested_at=requested_at,
            )
            return request, None, None
        inputs, materialized = self._produce_inputs(adapter, call, envelope, config)
        requested_at = self._strictly_after(inputs.query_access.accessed_at)
        fixed_ontology = None
        fixed_evidence = None
        if call.kind is DevelopmentCallKind.FIXED_SELECTION:
            fixed_preparation = inputs.preparation.fixed_selection
            if fixed_preparation is None:
                raise DevelopmentExecutionError(
                    "FixedSelect request lacks its same-unit C1 preparation"
                )
            fixed_ontology = fixed_preparation.source_c1_preontology.as_fixed_ontology()
            fixed_evidence = to_model_visible_packet(inputs.packet).evidence
        runtime = development_runtime_identifiers(
            root=self.repository.root,
            condition=call.condition,
            tokenizer_manifest=self.repository.tokenizer_manifest,
            seed=call.vllm_seed,
            repair=call.kind is DevelopmentCallKind.REPAIR_PROBE,
            fixed_ontology=fixed_ontology,
            fixed_evidence=fixed_evidence,
        )
        if call.kind is DevelopmentCallKind.FIXED_SELECTION:
            request = build_fixed_select_request(
                inputs,
                runtime=runtime,
                requested_at=requested_at,
            )
        else:
            request = build_c2_construction_request(
                inputs,
                runtime=runtime,
                requested_at=requested_at,
            )
        return request, inputs, materialized

    def _repair_probe_input(
        self,
        *,
        adapter: ProductionDevelopmentServiceAdapter,
        call: DevelopmentCallSpec,
        envelope: CallExecutionEnvelope,
        semantic: ConstructionRequest,
    ) -> DevelopmentRepairProbeInput:
        """Derive the registered one-field fault from the preserved parent bytes."""

        if (
            call.call_id != "dev-repair-probe-u04-q02"
            or call.parent_call_id is None
            or call.diagnostic_fixture_id
            != "development-schema-reference-repair-probe-v1"
            or call.fault_injection
            != "deterministic_unknown_reference_after_preserving_raw_parent"
            or envelope.parent_output_artifact_hash is None
        ):
            raise DevelopmentExecutionError("repair call metadata differs from its frozen probe")
        state = adapter._state()
        parent_receipt_hash = state.completed_receipts.get(call.parent_call_id)
        parent_result_hash = state.completed_results.get(call.parent_call_id)
        if parent_receipt_hash is None or parent_result_hash is None:
            raise DevelopmentExecutionError("repair parent has no durable receipt/result pair")
        receipt_record = adapter.artifacts.ledger.get_artifact(parent_receipt_hash)
        parent_receipt = DevelopmentCallAuditReceipt.model_validate_json(
            adapter.artifacts.blobs.read_bytes(receipt_record)
        )
        result_record = adapter.artifacts.ledger.get_artifact(parent_result_hash)
        parent_result = ServiceCallResult.model_validate_json(
            adapter.artifacts.blobs.read_bytes(result_record)
        )
        if (
            parent_receipt.call_id != call.parent_call_id
            or parent_receipt.outcome is not RunOutcome.SUCCEEDED
            or parent_result.outcome is not RunOutcome.SUCCEEDED
            or parent_result.ledger_receipt_hash != parent_receipt_hash
            or parent_result.response_artifact_hash
            != envelope.parent_output_artifact_hash
            or parent_receipt.raw_response_artifact_hash
            != envelope.parent_output_artifact_hash
            or parent_receipt.semantic_request is None
        ):
            raise DevelopmentExecutionError("repair parent durable lineage changed")
        parent_semantic_record = adapter.artifacts.ledger.get_artifact(
            parent_receipt.semantic_request.artifact_hash
        )
        parent_semantic = ConstructionRequest.model_validate_json(
            adapter.artifacts.blobs.read_bytes(parent_semantic_record)
        )
        if (
            parent_semantic.content_hash
            != parent_receipt.semantic_request.logical_content_hash
            or parent_semantic.condition is not ConditionName.C2_LLM_QUERY
            or parent_semantic.snapshot_hash != semantic.snapshot_hash
            or parent_semantic.packet != semantic.packet
            or parent_semantic.context != semantic.context
            or parent_semantic.upper_ontology != semantic.upper_ontology
            or parent_semantic.budgets != semantic.budgets
        ):
            raise DevelopmentExecutionError(
                "repair semantic request differs from its registered C2 parent"
            )
        raw_record = adapter.artifacts.ledger.get_artifact(
            envelope.parent_output_artifact_hash
        )
        raw_bytes = adapter.artifacts.blobs.read_bytes(raw_record)
        try:
            parent_raw = json.loads(raw_bytes)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise DevelopmentExecutionError("repair parent raw output is not JSON") from exc
        if not isinstance(parent_raw, dict):
            raise DevelopmentExecutionError("repair parent raw output is not an object")
        value = build_development_repair_probe_input(
            parent_call_id=call.parent_call_id,
            parent_semantic_request=parent_semantic,
            repair_semantic_request=semantic,
            parent_raw_output_hash=envelope.parent_output_artifact_hash,
            parent_raw_output=parent_raw,
            created_at=self._strictly_after(semantic.requested_at),
        )
        # The injected object must be rejected for exactly the registered unknown
        # contextual type before it is ever shown to the model as a repair target.
        aliases = encode_development_semantic_request(semantic).alias_manifest
        restored_invalid = restore_model_output_source_aliases(
            value.invalid_draft, aliases
        )
        invalid_draft = OntologyDraft.model_validate(restored_invalid)
        boundary = validate_draft_structure(
            draft=invalid_draft,
            upper_ontology=semantic.upper_ontology,
            evidence=semantic.packet.evidence,
            budgets=semantic.budgets,
            capabilities=semantic.capabilities,
        )
        observed = tuple((item.code.value, item.path) for item in boundary.diagnostics)
        diagnostic = value.diagnostics[0]
        if observed != ((diagnostic.code, diagnostic.path),):
            raise DevelopmentExecutionError(
                "deterministic repair fault did not yield its exact fact-free diagnostic"
            )
        return value

    def execute(
        self,
        adapter: ProductionDevelopmentServiceAdapter,
        call: DevelopmentCallSpec,
        envelope: CallExecutionEnvelope,
    ) -> ServiceCallResult:
        """Persist a complete terminal result for exactly one manifest row."""

        if envelope.execution_id != self.repository.execution_id:
            raise DevelopmentExecutionError("execution envelope belongs to another run")
        if envelope.call_manifest_hash != self.repository.manifest.content_hash:
            raise DevelopmentExecutionError("execution envelope changed call manifest")
        fixed_derivation = None
        if call.kind is DevelopmentCallKind.FIXED_SELECTION:
            fixed_derivation = self._load_fixed_derivation(adapter=adapter, call=call)
            plan = self.repository.fixed_schema_plans[call.ordinal]
            if (
                envelope.expected_run_condition_config_hash is not None
                or envelope.fixed_schema_derivation_plan_hash != plan.content_hash
                or fixed_derivation.plan_hash != plan.content_hash
            ):
                raise DevelopmentExecutionError(
                    "execution envelope changed the FixedSelect derivation plan"
                )
        elif envelope.fixed_schema_derivation_plan_hash is not None:
            raise DevelopmentExecutionError(
                "non-Fixed call cannot use a deferred schema derivation"
            )
        config = self.repository.run_config(call)
        if (
            call.kind is not DevelopmentCallKind.FIXED_SELECTION
            and config.content_hash != envelope.expected_run_condition_config_hash
        ):
            raise DevelopmentExecutionError("execution envelope changed run configuration")
        semantic, inputs, materialized = self._semantic_request(
            adapter,
            call,
            envelope,
            config,
        )
        repair_input = None
        if call.kind is DevelopmentCallKind.REPAIR_PROBE:
            if not isinstance(semantic, ConstructionRequest):
                raise DevelopmentExecutionError("repair probe lacks a query request")
            repair_input = self._repair_probe_input(
                adapter=adapter,
                call=call,
                envelope=envelope,
                semantic=semantic,
            )
            self.repository.repair_inputs[call.call_id] = repair_input
        guided = build_development_guided_request(
            root=self.repository.root,
            call_id=call.call_id,
            semantic_request=semantic,
            tokenizer=cast(Any, self.repository.tokenizer),
            tokenizer_manifest=self.repository.tokenizer_manifest,
            seed=call.vllm_seed,
            repair=call.kind is DevelopmentCallKind.REPAIR_PROBE,
            repair_probe_input=repair_input,
        )
        runtime = development_request_runtime(
            root=self.repository.root,
            condition=call.condition,
            tokenizer_manifest=self.repository.tokenizer_manifest,
            seed=call.vllm_seed,
            repair=call.kind is DevelopmentCallKind.REPAIR_PROBE,
            fixed_ontology=(
                semantic.fixed_ontology
                if call.kind is DevelopmentCallKind.FIXED_SELECTION
                and isinstance(semantic, ConstructionRequest)
                else None
            ),
            fixed_evidence=(
                semantic.packet.evidence
                if call.kind is DevelopmentCallKind.FIXED_SELECTION
                and isinstance(semantic, ConstructionRequest)
                else None
            ),
        )
        if (
            config.decoding_manifest_hash != guided.decoding.content_hash
            or config.decoding_family_hash != guided.decoding.comparison_family_hash
            or config.prompt_hash != runtime.prompt_hash
            or config.output_schema_hash != runtime.output_schema_hash
            or config.capability_manifest_hash != runtime.capability_manifest.content_hash
        ):
            raise DevelopmentExecutionError("packed request differs from frozen run config")
        self.repository.semantic_requests[call.call_id] = semantic
        alias_manifest = encode_development_semantic_request(semantic).alias_manifest
        self.repository.alias_manifests[call.call_id] = alias_manifest
        fixed_inventory = None
        fixed_probe = None
        if call.kind is DevelopmentCallKind.FIXED_SELECTION:
            if not isinstance(semantic, ConstructionRequest):
                raise DevelopmentExecutionError("FixedSelect call lacks a query request")
            fixed_inventory, fixed_probe = self._fixed_audits(
                call,
                semantic,
                decided_at=self._strictly_after(semantic.requested_at),
            )
        return self._run_and_persist(
            adapter=adapter,
            call=call,
            envelope=envelope,
            config=config,
            semantic=semantic,
            guided=guided,
            alias_manifest=alias_manifest,
            repair_input=repair_input,
            inputs=inputs,
            materialized=materialized,
            fixed_derivation=fixed_derivation,
            fixed_inventory=fixed_inventory,
            fixed_probe=fixed_probe,
        )

    def _run_and_persist(
        self,
        *,
        adapter: ProductionDevelopmentServiceAdapter,
        call: DevelopmentCallSpec,
        envelope: CallExecutionEnvelope,
        config: RunConditionConfig,
        semantic: PreconstructionRequest | ConstructionRequest,
        guided: GuidedJSONRequest,
        alias_manifest: ModelWireAliasManifest,
        repair_input: DevelopmentRepairProbeInput | None,
        inputs: ProduceInputs | None,
        materialized: AuditedPacketMaterialization | None,
        fixed_derivation: FixedSchemaDerivationReceipt | None,
        fixed_inventory: SealedOntologyInventory | None,
        fixed_probe: FixedSelectOutputAudit | None,
    ) -> ServiceCallResult:
        ledger = adapter.artifacts.ledger
        created_at = self._now()
        parent_attempt_id = None
        if call.kind is DevelopmentCallKind.REPAIR_PROBE:
            if call.parent_call_id is None:
                raise DevelopmentExecutionError("repair call lost its registered parent")
            state = adapter._state()
            parent_receipt_hash = state.completed_receipts.get(call.parent_call_id)
            if parent_receipt_hash is None:
                raise DevelopmentExecutionError("repair parent receipt is unavailable")
            parent_receipt_record = ledger.get_artifact(parent_receipt_hash)
            parent_receipt = DevelopmentCallAuditReceipt.model_validate_json(
                adapter.artifacts.blobs.read_bytes(parent_receipt_record)
            )
            if (
                parent_receipt.call_id != call.parent_call_id
                or parent_receipt.job_id is None
                or parent_receipt.attempt_id is None
            ):
                raise DevelopmentExecutionError("repair parent attempt lineage changed")
            job = ledger.get_job(parent_receipt.job_id)
            parent_attempt_id = parent_receipt.attempt_id
        else:
            job = ledger.create_or_resume_job(
                {
                    "execution_id": self.repository.execution_id,
                    "call_id": call.call_id,
                    "call_manifest_hash": self.repository.manifest.content_hash,
                },
                release_class=ReleaseClass.PUBLIC,
                created_at=created_at,
            )
        ledger.link_job_to_study(
            study_id=self.repository.execution_id,
            job_id=job.job_id,
            created_at=created_at,
        )
        attempt_id = f"{self.repository.execution_id}-{call.call_id}-attempt"
        ledger.record_attempt(
            attempt_id=attempt_id,
            job_id=job.job_id,
            attempt_kind=(
                AttemptKind.REPAIR
                if call.kind is DevelopmentCallKind.REPAIR_PROBE
                else AttemptKind.BASE
            ),
            input_hash=guided.request_hash,
            config_hash=config.content_hash,
            seed=call.vllm_seed,
            parent_attempt_id=parent_attempt_id,
            created_at=created_at,
        )
        adapter_state = adapter._state()
        if adapter_state.active_call_id not in {None, call.call_id}:
            raise DevelopmentExecutionError(
                "another development call remains active in durable state"
            )
        adapter._write_state(adapter_state, active_call_id=call.call_id)
        event_id = f"{self.repository.execution_id}-{call.call_id}-gpu"
        remaining = (
            sum(item.watchdog_seconds for item in self.repository.manifest.calls[call.ordinal :])
            + self.repository.post_development_forecast_seconds
        )
        generated: GenerationResult | None = None
        transport_error: BaseException | None = None
        try:
            generated = cast(Any, adapter._service).generate(
                guided,
                event_id=event_id,
                watchdog_seconds=call.watchdog_seconds,
                repair=call.kind is DevelopmentCallKind.REPAIR_PROBE,
                job_id=job.job_id,
                attempt_id=attempt_id,
                remaining_required_seconds=remaining,
                accounting_details={
                    "phase": "development",
                    "ordinal": call.ordinal,
                    "call_class": call.call_class,
                },
            )
        except BaseException as exc:  # metering/ledger receipt still must be retained
            transport_error = exc
        try:
            event = _event_for(adapter, event_id)
        except Exception:
            # A Python-level failure returned control and therefore cannot later be
            # mistaken for an interrupted in-flight request.  A process-killing
            # BaseException leaves the active marker for no-reissue recovery.
            state = adapter._state()
            if state.active_call_id == call.call_id:
                adapter._write_state(state, active_call_id=None)
            raise
        allocated = event.allocated_seconds
        if not math.isfinite(allocated) or allocated < 0:
            raise DevelopmentExecutionError("metered call duration is invalid")
        if generated is None:
            assert transport_error is not None
            return self._persist_failed_call(
                adapter=adapter,
                call=call,
                envelope=envelope,
                config=config,
                guided=guided,
                semantic=semantic,
                alias_manifest=alias_manifest,
                repair_input=repair_input,
                materialized=materialized,
                fixed_derivation=fixed_derivation,
                fixed_inventory=fixed_inventory,
                fixed_probe=fixed_probe,
                job_id=job.job_id,
                attempt_id=attempt_id,
                event_id=event_id,
                allocated=allocated,
                failure=transport_error,
                raw_response_artifact_hash=None,
                prompt_tokens=0,
                completion_tokens=0,
            )
        raw_artifact = adapter.artifacts.put_bytes(
            generated.raw_response,
            media_type="application/json",
            release_class=ReleaseClass.PUBLIC,
            created_at=_parse_utc(event.ended_at),
        )
        try:
            generation, boundary, repair_preservation = self._validated_generation(
                call=call,
                envelope=envelope,
                config=config,
                semantic=semantic,
                guided=guided,
                alias_manifest=alias_manifest,
                repair_input=repair_input,
                generated=generated,
                raw_artifact_hash=raw_artifact.content_hash,
                event=event,
            )
            if call.kind is DevelopmentCallKind.C1_PRECONSTRUCTION:
                result_object, preparation_bindings, fixed_inventory, fixed_probe = (
                    self._finalize_c1(call, semantic, generation)
                )
                self._derive_fixed_schema_after_c1(
                    adapter=adapter,
                    c1_call=call,
                    preparation=result_object,
                )
            else:
                assert inputs is not None
                result_object = self._finalize_query(call, inputs, semantic, generation)
                preparation_bindings = ()
        except BaseException as exc:
            return self._persist_failed_call(
                adapter=adapter,
                call=call,
                envelope=envelope,
                config=config,
                guided=guided,
                semantic=semantic,
                alias_manifest=alias_manifest,
                repair_input=repair_input,
                materialized=materialized,
                fixed_derivation=fixed_derivation,
                fixed_inventory=fixed_inventory,
                fixed_probe=fixed_probe,
                job_id=job.job_id,
                attempt_id=attempt_id,
                event_id=event_id,
                allocated=allocated,
                failure=exc,
                raw_response_artifact_hash=raw_artifact.content_hash,
                prompt_tokens=generated.prompt_tokens,
                completion_tokens=generated.completion_tokens,
            )
        return self._persist_successful_call(
            adapter=adapter,
            call=call,
            envelope=envelope,
            config=config,
            guided=guided,
            semantic=semantic,
            alias_manifest=alias_manifest,
            repair_input=repair_input,
            materialized=materialized,
            job_id=job.job_id,
            attempt_id=attempt_id,
            event_id=event_id,
            allocated=allocated,
            generated=generated,
            generation=generation,
            boundary=boundary,
            repair_preservation=repair_preservation,
            condition_result=result_object,
            preparation_bindings=preparation_bindings,
            fixed_inventory=fixed_inventory,
            fixed_probe=fixed_probe,
            fixed_derivation=fixed_derivation,
        )

    def _validated_generation(
        self,
        *,
        call: DevelopmentCallSpec,
        envelope: CallExecutionEnvelope,
        config: RunConditionConfig,
        semantic: PreconstructionRequest | ConstructionRequest,
        guided: GuidedJSONRequest,
        alias_manifest: ModelWireAliasManifest,
        repair_input: DevelopmentRepairProbeInput | None,
        generated: GenerationResult,
        raw_artifact_hash: str,
        event: Any,
    ) -> tuple[ValidatedGeneration, Any, Any | None]:
        if (
            generated.request_id != guided.request_id
            or generated.request_hash != guided.request_hash
            or generated.response_sha256 != raw_artifact_hash
            or generated.finish_reason != "stop"
            or generated.prompt_tokens != guided.rendered_input_token_count
            or generated.completion_tokens > guided.decoding.maximum_output_tokens
        ):
            raise DevelopmentExecutionError("vLLM response envelope changed")
        parsed = restore_model_output_source_aliases(
            generated.parsed_object,
            alias_manifest,
        )
        if call.condition is ConditionName.A_NO_TEMPORAL_EPISTEMIC:
            ablated = NoTemporalEpistemicOntologyDraft.model_validate(parsed)
            if (
                ablated.budget_accounting.input_tokens != 0
                or ablated.budget_accounting.output_tokens != 0
            ):
                raise DevelopmentExecutionError("model token sentinels must remain zero")
            raw_draft = normalize_no_temporal_epistemic_draft(ablated)
        else:
            raw_draft = OntologyDraft.model_validate(parsed)
            if (
                raw_draft.budget_accounting.input_tokens != 0
                or raw_draft.budget_accounting.output_tokens != 0
            ):
                raise DevelopmentExecutionError("model token sentinels must remain zero")
        repair_preservation = None
        if call.kind is DevelopmentCallKind.REPAIR_PROBE:
            if repair_input is None:
                raise DevelopmentExecutionError("repair generation lost its repair input")
            restored_invalid = restore_model_output_source_aliases(
                repair_input.invalid_draft,
                alias_manifest,
            )
            repair_preservation = validate_repair_preservation(
                base_draft=cast(Mapping[str, object], restored_invalid),
                repaired_draft=cast(Mapping[str, object], parsed),
                diagnosed_paths=tuple(item.path for item in repair_input.diagnostics),
            )
            repair_preservation.raise_for_errors()
        elif repair_input is not None:
            raise DevelopmentExecutionError("nonrepair generation received repair input")
        completed_at = _parse_utc(event.ended_at)
        started_at = _parse_utc(event.started_at)
        normalized = normalize_generation_metadata(
            raw_draft,
            decision_recorded_at=completed_at,
            input_tokens=generated.prompt_tokens,
            output_tokens=generated.completion_tokens,
        )
        evidence = (
            semantic.evidence
            if isinstance(semantic, PreconstructionRequest)
            else semantic.packet.evidence
        )
        if call.kind is DevelopmentCallKind.FIXED_SELECTION:
            fixed = cast(ConstructionRequest, semantic).fixed_ontology
            if fixed is None:
                raise DevelopmentExecutionError("FixedSelect request lost its sealed graph")
            source = self.repository.preparations[
                (call.unit_id, ConditionName.C1_LLM_PRE)
            ].sealed_preontology
            assert source is not None
            sealed = sealed_inventory_from_fixed_ontology(
                fixed,
                seed_block=call.seed_block,
                source_draft=source.draft,
            )
            enforce_fixed_select_output(
                fixed_select_audit_from_draft(
                    normalized,
                    source_seal_hash=sealed.seal_hash,
                    seed_block=call.seed_block,
                ),
                sealed,
            )
        boundary = validate_draft_structure(
            draft=normalized,
            upper_ontology=semantic.upper_ontology,
            evidence=evidence,
            budgets=semantic.budgets,
            capabilities=semantic.capabilities,
        )
        boundary.raise_for_errors()
        validated_at = self._strictly_after(completed_at)
        validation = ValidationRecord(
            validation_id=(
                "validation-"
                + canonical_sha256({"call": call.call_id, "draft": normalized.content_hash})[:20]
            ),
            target_id=normalized.content_hash,
            validation_status=ValidationStatus.ACCEPTED,
            evidence_support_status=EvidenceSupportStatus.SUPPORTED,
            temporal_status=TemporalDeterminationStatus.VALID,
            commitment_status=CommitmentCheckStatus.VALID,
            diagnostics=(
                "deterministic boundary, citation, temporal, and commitment checks accepted",
            ),
            validated_at=validated_at,
        )
        generation = ValidatedGeneration(
            generation_id=(
                "generation-"
                + canonical_sha256({"call": call.call_id, "raw": raw_artifact_hash})[:20]
            ),
            condition=call.condition,
            request_hash=semantic.content_hash,
            raw_output_artifact_hash=raw_artifact_hash,
            raw_parsed_draft=raw_draft,
            draft=normalized,
            normalized_draft_hash=normalized.content_hash,
            stage_manifest_hash=(
                call.prequery_stage.staging_manifest_hash
                if call.query_stage is None
                else call.query_stage.staging_manifest_hash
            ),
            query_access_event_hash=(
                None
                if envelope.query_access_event is None
                else envelope.query_access_event.content_hash
            ),
            prequery_barrier_hash=envelope.prequery_barrier_hash,
            packing_report_hash=guided.packing.content_hash,
            capability_manifest_hash=CapabilityManifest.for_condition(call.condition).content_hash,
            seed_manifest_hash=cast(str, config.seed_manifest_hash),
            prompt_hash=cast(str, config.prompt_hash),
            output_schema_hash=cast(str, config.output_schema_hash),
            decoding_manifest_hash=cast(str, config.decoding_manifest_hash),
            validator_hash=config.validator_hash,
            model_stack_hash=cast(str, config.model_stack_hash),
            seed=call.vllm_seed,
            input_tokens=generated.prompt_tokens,
            output_tokens=generated.completion_tokens,
            generation_started_at=started_at,
            generation_completed_at=completed_at,
            decision_recorded_at=completed_at,
            validation_records=(validation,),
            validator_report_hashes=(
                (boundary.content_hash,)
                if repair_preservation is None
                else (boundary.content_hash, repair_preservation.content_hash)
            ),
            validated_at=validated_at,
            repair_attempt=(1 if call.kind is DevelopmentCallKind.REPAIR_PROBE else 0),
            repair_parent_raw_output_hash=(
                envelope.parent_output_artifact_hash
                if call.kind is DevelopmentCallKind.REPAIR_PROBE
                else None
            ),
        )
        return generation, boundary, repair_preservation

    def _finalize_c1(
        self,
        call: DevelopmentCallSpec,
        semantic: PreconstructionRequest | ConstructionRequest,
        generation: ValidatedGeneration,
    ) -> tuple[
        ConditionPreparation,
        tuple[PrequeryPreparationBinding, ...],
        None,
        None,
    ]:
        if not isinstance(semantic, PreconstructionRequest):
            raise DevelopmentExecutionError("C1 finalizer received query request")
        sealed_at = self._strictly_after(generation.validated_at)
        preparation = seal_c1_preconstruction(
            request=semantic,
            generation=generation,
            seed_block=call.seed_block,
            sealed_at=sealed_at,
        )
        fixed = prepare_fixed_selection(
            preparation,
            seed_block=call.seed_block,
            prepared_at=self._strictly_after(sealed_at),
        )
        self.repository.preparations[(call.unit_id, ConditionName.C1_LLM_PRE)] = preparation
        self.repository.preparations[(call.unit_id, ConditionName.A_FIXED_SELECT)] = fixed
        source = preparation.sealed_preontology
        assert source is not None
        runtime_unit = self.repository.neutral_by_unit[
            call.unit_id
        ].snapshot.world_or_window_id
        bindings = (
            PrequeryPreparationBinding(
                unit_id=runtime_unit,
                condition=ConditionName.C1_LLM_PRE,
                seed_block=call.seed_block,
                snapshot_hash=source.snapshot_hash,
                preparation_hash=preparation.content_hash,
                lineage_artifact_hash=source.construction_seal.content_hash,
                completed_at=preparation.completed_at,
            ),
            PrequeryPreparationBinding(
                unit_id=runtime_unit,
                condition=ConditionName.A_FIXED_SELECT,
                seed_block=call.seed_block,
                snapshot_hash=source.snapshot_hash,
                preparation_hash=fixed.content_hash,
                lineage_artifact_hash=cast(Any, fixed.fixed_selection).content_hash,
                completed_at=fixed.completed_at,
            ),
        )
        return preparation, bindings, None, None

    def _finalize_query(
        self,
        call: DevelopmentCallSpec,
        inputs: ProduceInputs,
        semantic: PreconstructionRequest | ConstructionRequest,
        generation: ValidatedGeneration,
    ) -> ConditionAttemptRecord:
        if not isinstance(semantic, ConstructionRequest):
            raise DevelopmentExecutionError("query finalizer received C1 request")
        if call.kind is DevelopmentCallKind.FIXED_SELECTION:
            return finalize_fixed_select_draft(
                inputs,
                request=semantic,
                generation=generation,
            )
        return finalize_c2_draft(
            inputs,
            request=semantic,
            generation=generation,
        )

    def _fixed_audits(
        self,
        call: DevelopmentCallSpec,
        semantic: ConstructionRequest,
        *,
        decided_at: datetime,
    ) -> tuple[SealedOntologyInventory, FixedSelectOutputAudit]:
        fixed = semantic.fixed_ontology
        source = self.repository.preparations[
            (call.unit_id, ConditionName.C1_LLM_PRE)
        ].sealed_preontology
        if fixed is None or source is None:
            raise DevelopmentExecutionError("FixedSelect source preparation is unavailable")
        inventory = sealed_inventory_from_fixed_ontology(
            fixed,
            seed_block=call.seed_block,
            source_draft=source.draft,
        )
        reference_ids = (
            () if not inventory.objects else (inventory.objects[0].semantic_id,)
        )
        forbidden = FixedSelectOutputAudit(
            source_seal_hash=inventory.seal_hash,
            seed_block=inventory.seed_block,
            selected_objects=(),
            operations=(
                ProposedOperation(
                    operation_id=f"{call.call_id}-forbidden-construction-probe",
                    capability=ConstructionOperator.MERGE,
                    referenced_ids=reference_ids,
                    created_ids=(f"{call.call_id}-forbidden-created-id",),
                    decided_at=decided_at,
                ),
            ),
        )
        try:
            enforce_fixed_select_output(forbidden, inventory)
        except FixedSelectCapabilityError:
            pass
        else:
            raise DevelopmentExecutionError(
                "FixedSelect constructive-operation probe was not rejected"
            )
        return inventory, forbidden

    def _common_references(
        self,
        *,
        adapter: ProductionDevelopmentServiceAdapter,
        call: DevelopmentCallSpec,
        envelope: CallExecutionEnvelope,
        config: RunConditionConfig,
        guided: GuidedJSONRequest,
        semantic: PreconstructionRequest | ConstructionRequest,
        alias_manifest: ModelWireAliasManifest,
        repair_input: DevelopmentRepairProbeInput | None,
        fixed_derivation: FixedSchemaDerivationReceipt | None,
        materialized: AuditedPacketMaterialization | None,
        created_at: datetime,
    ) -> dict[str, object]:
        artifacts = adapter.artifacts
        identity_ref = persist_logical_record(
            artifacts,
            adapter.live_identity,
            object_kind="service_identity",
            created_at=created_at,
        )
        envelope_ref = persist_logical_record(
            artifacts,
            envelope,
            object_kind="call_execution_envelope",
            created_at=created_at,
        )
        semantic_ref = persist_logical_record(
            artifacts,
            semantic,
            object_kind=(
                "preconstruction_request"
                if isinstance(semantic, PreconstructionRequest)
                else "construction_request"
            ),
            created_at=created_at,
        )
        alias_ref = persist_logical_record(
            artifacts,
            alias_manifest,
            object_kind="model_wire_alias_manifest",
            created_at=created_at,
        )
        rendered_ref = persist_opaque_json(
            artifacts,
            guided.wire_payload(),
            object_kind="rendered_model_request",
            created_at=created_at,
        )
        config_ref = persist_logical_record(
            artifacts,
            config,
            object_kind="run_condition_config",
            created_at=created_at,
        )
        packing_ref = persist_logical_record(
            artifacts,
            guided.packing,
            object_kind="packing_report",
            created_at=created_at,
        )
        capability_ref = persist_logical_record(
            artifacts,
            CapabilityManifest.for_condition(call.condition),
            object_kind="capability_manifest",
            created_at=created_at,
        )
        switches_ref = persist_logical_record(
            artifacts,
            treatment_switches_for(call.condition),
            object_kind="treatment_switches",
            created_at=created_at,
        )
        repair_ref = (
            None
            if repair_input is None
            else persist_logical_record(
                artifacts,
                repair_input,
                object_kind="development_repair_probe_input",
                created_at=created_at,
            )
        )
        fixed_derivation_ref = (
            None
            if fixed_derivation is None
            else persist_logical_record(
                artifacts,
                fixed_derivation,
                object_kind="fixed_schema_derivation_receipt",
                created_at=created_at,
            )
        )
        query: dict[str, object] = {
            "comparison_input_manifest": None,
            "evidence_packet": None,
            "packet_materialization_event_hash": None,
            "query_context": None,
        }
        if materialized is not None:
            stage = call.query_stage
            assert stage is not None
            opening = adapter.audited_opening(stage)
            inputs = self._produce_inputs(adapter, call, envelope, config)[0]
            comparison = ComparisonInputManifest.from_inputs(inputs)
            query = {
                "comparison_input_manifest": persist_logical_record(
                    artifacts,
                    comparison,
                    object_kind="comparison_input_manifest",
                    created_at=created_at,
                ),
                "evidence_packet": persist_logical_record(
                    artifacts,
                    materialized.packet,
                    object_kind="evidence_packet",
                    created_at=created_at,
                ),
                "packet_materialization_event_hash": materialized.event.content_hash,
                "query_context": persist_logical_record(
                    artifacts,
                    opening.context,
                    object_kind="query_context",
                    created_at=created_at,
                ),
            }
        return {
            "service_identity": identity_ref,
            "call_execution_envelope": envelope_ref,
            "rendered_model_request": rendered_ref,
            "semantic_request": semantic_ref,
            "wire_alias_manifest": alias_ref,
            "run_condition_config": config_ref,
            "packing_report": packing_ref,
            "capability_manifest": capability_ref,
            "treatment_switches": switches_ref,
            "repair_probe_input": repair_ref,
            "fixed_schema_derivation": fixed_derivation_ref,
            **query,
        }

    def _record_model_call(
        self,
        *,
        adapter: ProductionDevelopmentServiceAdapter,
        call: DevelopmentCallSpec,
        config: RunConditionConfig,
        guided: GuidedJSONRequest,
        job_id: str,
        attempt_id: str,
        event_id: str,
        response_artifact_hash: str | None,
        allocated: float,
        prompt_tokens: int,
        completion_tokens: int,
        successful: bool,
        created_at: datetime,
    ) -> str:
        model_call_id = f"{self.repository.execution_id}-{call.call_id}-model"
        adapter.artifacts.ledger.record_model_call(
            model_call_id=model_call_id,
            job_id=job_id,
            attempt_id=attempt_id,
            gpu_event_id=event_id,
            backend=ModelBackend.VLLM_GPU,
            call_role=_call_role(call),
            retry_class=_retry_class(call),
            model_manifest_hash=self.repository.model_manifest_hash,
            decoding_manifest_hash=guided.decoding.content_hash,
            request_hash=guided.request_hash,
            response_artifact_hash=response_artifact_hash,
            construction_unit_hash=canonical_sha256(
                {
                    "manifest": self.repository.manifest.content_hash,
                    "ordinal": call.ordinal,
                    "unit": call.unit_id,
                    "condition": call.condition,
                }
            ),
            served_context_count=1,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            allocated_gpu_seconds=allocated,
            successful=successful,
            created_at=created_at,
        )
        return model_call_id

    def _persist_successful_call(
        self,
        *,
        adapter: ProductionDevelopmentServiceAdapter,
        call: DevelopmentCallSpec,
        envelope: CallExecutionEnvelope,
        config: RunConditionConfig,
        guided: GuidedJSONRequest,
        semantic: PreconstructionRequest | ConstructionRequest,
        alias_manifest: ModelWireAliasManifest,
        repair_input: DevelopmentRepairProbeInput | None,
        materialized: AuditedPacketMaterialization | None,
        fixed_derivation: FixedSchemaDerivationReceipt | None,
        job_id: str,
        attempt_id: str,
        event_id: str,
        allocated: float,
        generated: GenerationResult,
        generation: ValidatedGeneration,
        boundary: Any,
        repair_preservation: Any | None,
        condition_result: ConditionPreparation | ConditionAttemptRecord,
        preparation_bindings: tuple[PrequeryPreparationBinding, ...],
        fixed_inventory: SealedOntologyInventory | None,
        fixed_probe: FixedSelectOutputAudit | None,
    ) -> ServiceCallResult:
        created_at = self._strictly_after(generation.validated_at)
        common = self._common_references(
            adapter=adapter,
            call=call,
            envelope=envelope,
            config=config,
            guided=guided,
            semantic=semantic,
            alias_manifest=alias_manifest,
            repair_input=repair_input,
            fixed_derivation=fixed_derivation,
            materialized=materialized,
            created_at=created_at,
        )
        generation_ref = persist_logical_record(
            adapter.artifacts,
            generation,
            object_kind="validated_generation",
            created_at=created_at,
        )
        result_ref = persist_logical_record(
            adapter.artifacts,
            condition_result,
            object_kind=(
                "condition_preparation"
                if isinstance(condition_result, ConditionPreparation)
                else "condition_attempt"
            ),
            created_at=created_at,
        )
        inventory_ref = (
            None
            if fixed_inventory is None
            else persist_logical_record(
                adapter.artifacts,
                fixed_inventory,
                object_kind="fixed_sealed_inventory",
                created_at=created_at,
            )
        )
        probe_ref = (
            None
            if fixed_probe is None
            else persist_logical_record(
                adapter.artifacts,
                fixed_probe,
                object_kind="fixed_forbidden_probe",
                created_at=created_at,
            )
        )
        repair_preservation_ref = (
            None
            if repair_preservation is None
            else persist_logical_record(
                adapter.artifacts,
                repair_preservation,
                object_kind="repair_preservation_report",
                created_at=created_at,
            )
        )
        model_call_id = self._record_model_call(
            adapter=adapter,
            call=call,
            config=config,
            guided=guided,
            job_id=job_id,
            attempt_id=attempt_id,
            event_id=event_id,
            response_artifact_hash=generation.raw_output_artifact_hash,
            allocated=allocated,
            prompt_tokens=generated.prompt_tokens,
            completion_tokens=generated.completion_tokens,
            successful=True,
            created_at=created_at,
        )
        validation_id = f"{self.repository.execution_id}-{call.call_id}-validation"
        adapter.artifacts.ledger.record_validation(
            validation_id=validation_id,
            job_id=job_id,
            attempt_id=attempt_id,
            input_artifact_hash=generation.raw_output_artifact_hash,
            validator_manifest_hash=config.validator_hash,
            validation_status=LedgerValidationStatus.ACCEPTED,
            evidence_support_status=LedgerEvidenceSupportStatus.SUPPORTED,
            temporal_status=TemporalValidationStatus.VALID,
            commitment_status=LedgerCommitmentCheckStatus.VALID,
            diagnostics_artifact_hash=persist_logical_record(
                adapter.artifacts,
                boundary,
                object_kind="boundary_validation_report",
                created_at=created_at,
            ).artifact_hash,
            created_at=created_at,
        )
        receipt = DevelopmentCallAuditReceipt(
            receipt_id=f"{self.repository.execution_id}-{call.call_id}-receipt",
            ordinal=call.ordinal,
            call_id=call.call_id,
            condition=call.condition,
            outcome=RunOutcome.SUCCEEDED,
            request_started=True,
            **common,
            job_id=job_id,
            attempt_id=attempt_id,
            model_call_id=model_call_id,
            raw_response_artifact_hash=generation.raw_output_artifact_hash,
            validated_generation=generation_ref,
            condition_result=result_ref,
            ledger_validation_ids=(validation_id,),
            fixed_sealed_inventory=inventory_ref,
            fixed_forbidden_probe=probe_ref,
            repair_preservation_report=repair_preservation_ref,
            created_at=created_at,
        )
        receipt_artifact = adapter.artifacts.put_bytes(
            (receipt.to_canonical_json() + "\n").encode("utf-8"),
            media_type="application/vnd.story-projection.development-call-receipt+json",
            release_class=ReleaseClass.PUBLIC,
            created_at=created_at,
        )
        completed_at = self._strictly_after(created_at)
        source = (
            condition_result.sealed_preontology
            if isinstance(condition_result, ConditionPreparation)
            else None
        )
        service_result = ServiceCallResult(
            call_id=call.call_id,
            outcome=RunOutcome.SUCCEEDED,
            request_started=True,
            request_hash=guided.request_hash,
            response_artifact_hash=generation.raw_output_artifact_hash,
            validated_generation_hash=generation.content_hash,
            validation_record_hash=canonical_sha256(generation.validation_records),
            condition_preparation_hash=(
                condition_result.content_hash
                if isinstance(condition_result, ConditionPreparation)
                else None
            ),
            condition_attempt_hash=(
                condition_result.content_hash
                if isinstance(condition_result, ConditionAttemptRecord)
                else None
            ),
            ledger_receipt_hash=receipt_artifact.content_hash,
            gpu_event_id=event_id,
            service_identity_hash=adapter.live_identity.content_hash,
            run_condition_config_hash=config.content_hash,
            fixed_schema_derivation=fixed_derivation,
            query_access_event_hash=(
                None
                if envelope.query_access_event is None
                else envelope.query_access_event.content_hash
            ),
            construction_seal_hash=(
                None if source is None else source.construction_seal.content_hash
            ),
            prequery_preparation_bindings=preparation_bindings,
            repair_parent_raw_output_hash=(
                envelope.parent_output_artifact_hash
                if call.kind is DevelopmentCallKind.REPAIR_PROBE
                else None
            ),
            allocated_gpu_seconds=allocated,
            prompt_tokens=generated.prompt_tokens,
            completion_tokens=generated.completion_tokens,
            completed_at=completed_at,
        )
        self._commit_result(adapter, call, receipt_artifact.content_hash, service_result)
        if isinstance(condition_result, ConditionPreparation):
            self.repository.c1_preparation_references[call.unit_id] = result_ref
        return service_result

    def _persist_failed_call(
        self,
        *,
        adapter: ProductionDevelopmentServiceAdapter,
        call: DevelopmentCallSpec,
        envelope: CallExecutionEnvelope,
        config: RunConditionConfig,
        guided: GuidedJSONRequest,
        semantic: PreconstructionRequest | ConstructionRequest,
        alias_manifest: ModelWireAliasManifest,
        repair_input: DevelopmentRepairProbeInput | None,
        materialized: AuditedPacketMaterialization | None,
        fixed_derivation: FixedSchemaDerivationReceipt | None,
        fixed_inventory: SealedOntologyInventory | None,
        fixed_probe: FixedSelectOutputAudit | None,
        job_id: str,
        attempt_id: str,
        event_id: str,
        allocated: float,
        failure: BaseException,
        raw_response_artifact_hash: str | None,
        prompt_tokens: int,
        completion_tokens: int,
    ) -> ServiceCallResult:
        created_at = self._now()
        failure_code = f"development_{_failure_kind(failure).value}"
        diagnostics = adapter.artifacts.put_bytes(
            (
                canonical_json(
                    {
                        "call_id": call.call_id,
                        "failure_code": failure_code,
                        "exception_type": type(failure).__name__,
                    }
                )
                + "\n"
            ).encode("utf-8"),
            media_type="application/vnd.story-projection.failure-diagnostics+json",
            release_class=ReleaseClass.PUBLIC,
            created_at=created_at,
        )
        model_call_id = self._record_model_call(
            adapter=adapter,
            call=call,
            config=config,
            guided=guided,
            job_id=job_id,
            attempt_id=attempt_id,
            event_id=event_id,
            response_artifact_hash=raw_response_artifact_hash,
            allocated=allocated,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            successful=False,
            created_at=created_at,
        )
        adapter.artifacts.ledger.record_failure(
            attempt_id=attempt_id,
            failure_kind=_failure_kind(failure),
            message="Development model call failed; inspect public diagnostics artifact",
            details={"call_id": call.call_id, "exception_type": type(failure).__name__},
            artifact_hash=diagnostics.content_hash,
            created_at=created_at,
        )
        validation_id = f"{self.repository.execution_id}-{call.call_id}-validation"
        validation_input = raw_response_artifact_hash or diagnostics.content_hash
        adapter.artifacts.ledger.record_validation(
            validation_id=validation_id,
            job_id=job_id,
            attempt_id=attempt_id,
            input_artifact_hash=validation_input,
            validator_manifest_hash=config.validator_hash,
            validation_status=LedgerValidationStatus.REJECTED,
            evidence_support_status=LedgerEvidenceSupportStatus.UNSUPPORTED,
            temporal_status=TemporalValidationStatus.NOT_APPLICABLE,
            commitment_status=LedgerCommitmentCheckStatus.NOT_APPLICABLE,
            diagnostics_artifact_hash=diagnostics.content_hash,
            created_at=created_at,
        )
        common = self._common_references(
            adapter=adapter,
            call=call,
            envelope=envelope,
            config=config,
            guided=guided,
            semantic=semantic,
            alias_manifest=alias_manifest,
            repair_input=repair_input,
            fixed_derivation=fixed_derivation,
            materialized=materialized,
            created_at=created_at,
        )
        inventory_ref = (
            None
            if fixed_inventory is None
            else persist_logical_record(
                adapter.artifacts,
                fixed_inventory,
                object_kind="fixed_sealed_inventory",
                created_at=created_at,
            )
        )
        probe_ref = (
            None
            if fixed_probe is None
            else persist_logical_record(
                adapter.artifacts,
                fixed_probe,
                object_kind="fixed_forbidden_probe",
                created_at=created_at,
            )
        )
        outcome = (
            RunOutcome.TIMED_OUT
            if isinstance(failure, TimeoutError)
            else RunOutcome.INVALID
            if isinstance(failure, (ValidationError, ValueError))
            else RunOutcome.FAILED
        )
        receipt = DevelopmentCallAuditReceipt(
            receipt_id=f"{self.repository.execution_id}-{call.call_id}-receipt",
            ordinal=call.ordinal,
            call_id=call.call_id,
            condition=call.condition,
            outcome=outcome,
            request_started=True,
            **common,
            job_id=job_id,
            attempt_id=attempt_id,
            model_call_id=model_call_id,
            raw_response_artifact_hash=raw_response_artifact_hash,
            ledger_validation_ids=(validation_id,),
            fixed_sealed_inventory=inventory_ref,
            fixed_forbidden_probe=probe_ref,
            created_at=created_at,
        )
        receipt_artifact = adapter.artifacts.put_bytes(
            (receipt.to_canonical_json() + "\n").encode("utf-8"),
            media_type="application/vnd.story-projection.development-call-receipt+json",
            release_class=ReleaseClass.PUBLIC,
            created_at=created_at,
        )
        result = ServiceCallResult(
            call_id=call.call_id,
            outcome=outcome,
            request_started=True,
            request_hash=guided.request_hash,
            response_artifact_hash=raw_response_artifact_hash,
            ledger_receipt_hash=receipt_artifact.content_hash,
            gpu_event_id=event_id,
            service_identity_hash=adapter.live_identity.content_hash,
            run_condition_config_hash=config.content_hash,
            fixed_schema_derivation=fixed_derivation,
            query_access_event_hash=(
                None
                if envelope.query_access_event is None
                else envelope.query_access_event.content_hash
            ),
            repair_parent_raw_output_hash=(
                envelope.parent_output_artifact_hash
                if call.kind is DevelopmentCallKind.REPAIR_PROBE
                else None
            ),
            allocated_gpu_seconds=allocated,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            failure_code=failure_code,
            completed_at=self._strictly_after(created_at),
        )
        self._commit_result(adapter, call, receipt_artifact.content_hash, result)
        return result

    def _commit_result(
        self,
        adapter: ProductionDevelopmentServiceAdapter,
        call: DevelopmentCallSpec,
        receipt_hash: str,
        result: ServiceCallResult,
    ) -> None:
        result_artifact = adapter.artifacts.put_bytes(
            (result.to_canonical_json() + "\n").encode("utf-8"),
            media_type="application/vnd.story-projection.service-call-result+json",
            release_class=ReleaseClass.PUBLIC,
            created_at=result.completed_at,
        )
        state = adapter._state()
        receipts = dict(state.completed_receipts)
        results = dict(state.completed_results)
        if call.call_id in receipts or call.call_id in results:
            raise DevelopmentExecutionError("development call already has a durable result")
        if state.active_call_id != call.call_id:
            raise DevelopmentExecutionError(
                "durable result commit lost its active-call marker"
            )
        receipts[call.call_id] = receipt_hash
        results[call.call_id] = result_artifact.content_hash
        adapter._write_state(
            state,
            completed_receipts=receipts,
            completed_results=results,
            active_call_id=None,
        )


__all__ = [
    "DevelopmentExecutionError",
    "DevelopmentExecutionRepository",
    "ProductionDevelopmentCallExecutor",
]
