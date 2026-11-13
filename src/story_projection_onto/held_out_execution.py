"""Gold-free semantic executor for the reviewed held-out primary block.

The executor receives an already-loaded, lifecycle-free generation service.  It
persists an intent before every GPU request, never retries an interrupted intent,
and constructs only the registered C1, C2, and FixedSelect semantic requests.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import tempfile
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, Self, cast

from pydantic import Field, ValidationError, model_validator

from story_projection_onto.benchmark_runtime import NeutralEvidenceArtifact
from story_projection_onto.conditions.base import (
    SCORED_PROJECTION_SCHEMA_HASH,
    ConditionAttemptRecord,
    ConditionPreparation,
    RunConditionConfig,
)
from story_projection_onto.conditions.c1 import (
    build_c1_preconstruction_request,
    project_sealed_c1,
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
    ConditionName,
    ConstructionCapabilities,
    ConstructionOperator,
    ConstructionRequest,
    EvidencePacket,
    EvidenceSnapshot,
    ImmutableRecord,
    ModelVisibleRevision,
    OntologyDraft,
    PacketMaterializationEvent,
    PreconstructionRequest,
    PrequeryBarrier,
    PrequeryPreparationBinding,
    QueryAccessEvent,
    QueryContext,
    RunOutcome,
    UpperOntology,
    ValidatedGeneration,
    canonical_json,
    canonical_sha256,
    normalize_generation_metadata,
    runtime_structural_acceptance_record,
    to_model_visible_packet,
    to_model_visible_query,
)
from story_projection_onto.development_adapter import (
    DevelopmentConstructionConfiguration,
    MeteredGenerationService,
    PackingTokenizer,
    build_development_guided_request,
    development_output_schema_for_request,
    development_request_runtime,
    development_runtime_identifiers,
    encode_development_semantic_request,
    restore_model_output_source_aliases,
)
from story_projection_onto.gpu_runtime import (
    ChatMessage,
    GenerationResult,
    GuidedJSONRequest,
    TokenizerManifest,
)
from story_projection_onto.held_out_controller import PreconstructedProjectionReceipt
from story_projection_onto.held_out_primary import (
    FixedSelectCapabilityAudit,
    HeldOutCallArtifactReceipt,
    HeldOutCallEnvelope,
    HeldOutCallManifest,
    HeldOutCallSpec,
    HeldOutCASReference,
    HeldOutServiceResult,
    HeldOutUnitPlan,
)
from story_projection_onto.held_out_production import (
    HeldOutArtifactResolver,
    HeldOutProductionError,
    gpu_ledger_chain_hash,
)
from story_projection_onto.llm import (
    CapabilityManifest,
    PackingReport,
    PackingSection,
    enforce_fixed_select_draft,
    sealed_inventory_from_fixed_ontology,
)
from story_projection_onto.store import (
    ArtifactStore,
    AttemptKind,
    FailureKind,
    InputKind,
    JobState,
    ModelBackend,
    ModelCallRole,
    ReleaseClass,
    RetryClass,
)
from story_projection_onto.store import (
    CommitmentCheckStatus as LedgerCommitmentCheckStatus,
)
from story_projection_onto.store import (
    EvidenceSupportStatus as LedgerEvidenceSupportStatus,
)
from story_projection_onto.store import (
    SemanticAssessmentScope as LedgerSemanticAssessmentScope,
)
from story_projection_onto.store import (
    TemporalValidationStatus as LedgerTemporalValidationStatus,
)
from story_projection_onto.store import (
    ValidationStatus as LedgerValidationStatus,
)
from story_projection_onto.validate import validate_draft_structure, validate_repair_preservation


class HeldOutSemanticExecutionError(HeldOutProductionError):
    """A frozen semantic request or its durable lineage failed validation."""


class HeldOutSemanticIntent(ImmutableRecord):
    call_id: str
    call_spec_hash: str
    envelope_hash: str
    semantic_request_hash: str
    guided_request_hash: str
    gpu_event_id: str
    started_at: datetime
    repair_semantic_request: HeldOutCASReference | None = None
    repair_packing_report: HeldOutCASReference | None = None
    repair_semantic_request_hash: str | None = None
    repair_guided_request_hash: str | None = None
    repair_gpu_event_id: str | None = None
    repair_started_at: datetime | None = None

    @model_validator(mode="after")
    def complete_repair_phase(self) -> Self:
        repair_values = (
            self.repair_semantic_request,
            self.repair_packing_report,
            self.repair_semantic_request_hash,
            self.repair_guided_request_hash,
            self.repair_gpu_event_id,
            self.repair_started_at,
        )
        if any(value is not None for value in repair_values) != all(
            value is not None for value in repair_values
        ):
            raise ValueError("repair semantic intent must be wholly present or absent")
        if self.repair_started_at is not None and self.repair_started_at < self.started_at:
            raise ValueError("repair semantic intent cannot precede its base intent")
        if self.repair_semantic_request is not None and (
            self.repair_semantic_request.object_kind
            not in {"preconstruction_request", "construction_request"}
            or self.repair_packing_report is None
            or self.repair_packing_report.object_kind != "packing_report"
            or self.repair_semantic_request.logical_content_hash
            != self.repair_semantic_request_hash
        ):
            raise ValueError("repair semantic intent has invalid typed CAS lineage")
        return self


class HeldOutCompletedResultPointer(ImmutableRecord):
    call_id: str
    envelope_hash: str
    result_artifact_hash: str
    result_logical_hash: str


class HeldOutSemanticState(ImmutableRecord):
    call_manifest_hash: str
    frozen_inputs_hash: str
    completed: Mapping[str, HeldOutCompletedResultPointer] = Field(default_factory=dict)
    active_intent: HeldOutSemanticIntent | None = None
    updated_at: datetime


class HeldOutProduceInputs(ImmutableRecord):
    """Common query inputs bound to the controller's pseudonymous held-out unit."""

    controller_unit_id: str
    preparation: ConditionPreparation
    snapshot: EvidenceSnapshot
    packet: EvidencePacket
    context: QueryContext
    query_access: QueryAccessEvent
    prequery_barrier: PrequeryBarrier
    query_processing_started_at: datetime
    packet_materialization: PacketMaterializationEvent
    upper_ontology: UpperOntology
    revisions: tuple[ModelVisibleRevision, ...] = ()
    run_config: RunConditionConfig

    @model_validator(mode="after")
    def exact_held_out_boundary(self) -> Self:
        if (
            self.preparation.condition is not self.run_config.condition
            or self.preparation.snapshot_hash != self.snapshot.content_hash
            or self.packet.snapshot_hash != self.snapshot.content_hash
            or self.context.spoiler_horizon != self.snapshot.horizon
            or self.query_access.query_context_hash != self.context.content_hash
            or self.query_access.model_visible_query_hash
            != to_model_visible_query(self.context).content_hash
            or self.query_access.snapshot_hash != self.snapshot.content_hash
            or self.query_access.prequery_barrier_hash != self.prequery_barrier.content_hash
            or self.query_access.execution_id != self.prequery_barrier.execution_id
            or self.prequery_barrier.sealed_at >= self.query_access.accessed_at
            or self.preparation.completed_at >= self.query_access.accessed_at
            or self.query_access.registered_revealed_at != self.context.revealed_at
            or self.packet.created_at < self.context.revealed_at
            or self.packet.created_at < self.query_access.accessed_at
            or (
                self.query_access.packet_hash is not None
                and self.query_access.packet_hash != self.packet.content_hash
            )
            or self.packet_materialization.query_access_event_hash != self.query_access.content_hash
            or self.packet_materialization.packet_hash != self.packet.content_hash
            or self.packet_materialization.snapshot_hash != self.snapshot.content_hash
            or self.packet_materialization.execution_id != self.query_access.execution_id
            or self.packet_materialization.retrieval_method is not self.packet.retrieval_method
            or self.packet_materialization.started_at < self.query_access.accessed_at
            or self.query_processing_started_at
            <= max(
                self.query_access.accessed_at,
                self.packet_materialization.completed_at,
            )
            or self.upper_ontology.content_hash != self.run_config.upper_ontology_hash
            or self.context.budgets != self.run_config.budgets
            or self.packet.ordered_evidence_ids != self.snapshot.eligible_evidence_ids
            or tuple(item.evidence_id for item in self.packet.evidence)
            != self.packet.ordered_evidence_ids
        ):
            raise ValueError("held-out ProduceInputs chronology or content changed")
        matches = tuple(
            item
            for item in self.prequery_barrier.preparation_bindings
            if item.unit_id == self.controller_unit_id
            and item.condition is self.run_config.condition
            and item.seed_block == self.run_config.seed_block
        )
        lineage = (
            self.preparation.sealed_preontology.construction_seal
            if self.preparation.sealed_preontology is not None
            else self.preparation.empty_inventory
            if self.preparation.empty_inventory is not None
            else self.preparation.fixed_selection
        )
        if (
            len(matches) != 1
            or matches[0].snapshot_hash != self.snapshot.content_hash
            or matches[0].preparation_hash != self.preparation.content_hash
            or lineage is None
            or matches[0].lineage_artifact_hash != lineage.content_hash
        ):
            raise ValueError("held-out barrier lacks the exact pseudonymous-unit preparation")
        if self.run_config.condition is ConditionName.A_FIXED_SELECT:
            fixed = self.preparation.fixed_selection
            if fixed is None or fixed.seed_block != self.run_config.seed_block:
                raise ValueError("held-out FixedSelect preparation is not same-seed")
        if self.run_config.condition is ConditionName.C1_LLM_PRE:
            sealed = self.preparation.sealed_preontology
            if sealed is None or sealed.seed_block != self.run_config.seed_block:
                raise ValueError("held-out C1 projection seed differs from its preontology")
        revision_sequences = tuple(item.revision.sequence for item in self.revisions)
        if revision_sequences != tuple(sorted(set(revision_sequences))):
            raise ValueError("held-out model-visible revisions are not strictly ordered")
        return self


@dataclass(frozen=True, slots=True)
class _AttemptArtifacts:
    job_id: str
    attempt_id: str
    semantic: PreconstructionRequest | ConstructionRequest
    guided: GuidedJSONRequest
    semantic_reference: HeldOutCASReference
    packing_reference: HeldOutCASReference
    raw_reference: HeldOutCASReference | None
    event_id: str
    event_hash: str
    model_call_id: str
    model_call_hash: str
    generated: GenerationResult | None
    allocated_seconds: float
    failure: BaseException | None


def _media_type(kind: str) -> str:
    return f"application/vnd.story-projection.{kind.replace('_', '-')}+json"


def _parse_utc(value: str) -> datetime:
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise HeldOutSemanticExecutionError("GPU event timestamp lacks a timezone")
    return parsed.astimezone(UTC)


def _event_hash(event: object) -> str:
    return canonical_sha256(asdict(cast(Any, event)))


def _failure_kind(error: BaseException) -> FailureKind:
    if isinstance(error, TimeoutError):
        return FailureKind.TIMEOUT
    if isinstance(error, (ValidationError, ValueError)):
        return FailureKind.INVALID_OUTPUT
    if isinstance(error, MemoryError) or "out of memory" in str(error).casefold():
        return FailureKind.OUT_OF_MEMORY
    return FailureKind.SERVICE


def _run_outcome(error: BaseException) -> RunOutcome:
    if isinstance(error, TimeoutError):
        return RunOutcome.TIMED_OUT
    if isinstance(error, (ValidationError, ValueError)):
        return RunOutcome.INVALID
    return RunOutcome.FAILED


def _call_role(condition: ConditionName, *, repair: bool) -> ModelCallRole:
    if repair:
        return ModelCallRole.REPAIR
    if condition is ConditionName.C1_LLM_PRE:
        return ModelCallRole.PREBUILD
    if condition is ConditionName.A_FIXED_SELECT:
        return ModelCallRole.FIXED_SELECT
    return ModelCallRole.QUERY_TIME


def _retry_class(call: HeldOutCallSpec) -> RetryClass:
    return {
        "reserve_long": RetryClass.LONG,
        "reserve_standard": RetryClass.STANDARD,
        "reserve_short": RetryClass.SHORT,
    }[call.repair_reserve_class]


@dataclass(slots=True)
class FrozenHeldOutSemanticExecutor:
    """Construct, validate, receipt, and recover the exact held-out LLM calls."""

    root: Path
    call_manifest: HeldOutCallManifest
    construction: DevelopmentConstructionConfiguration
    tokenizer: PackingTokenizer
    tokenizer_manifest: TokenizerManifest
    model_stack_hash: str
    seed_manifest_hash: str
    validator_hash: str
    selected_model_freeze_hash: str
    source_tree_hash: str
    artifacts: ArtifactStore
    neutral_by_unit: Mapping[str, NeutralEvidenceArtifact]
    state_path: Path
    clock: Callable[[], datetime] = lambda: datetime.now(UTC)
    resolver: HeldOutArtifactResolver = field(init=False)

    def __post_init__(self) -> None:
        self.root = self.root.resolve(strict=True)
        self.state_path = self.state_path.resolve(strict=False)
        if set(self.neutral_by_unit) != {item.unit_id for item in self.call_manifest.units}:
            raise HeldOutSemanticExecutionError("held-out executor lacks exact neutral worlds")
        self.resolver = HeldOutArtifactResolver(self.artifacts)
        frozen = canonical_sha256(
            {
                "call_manifest": self.call_manifest.content_hash,
                "construction": self.construction.content_hash,
                "tokenizer": self.tokenizer_manifest.manifest_sha256,
                "model_stack": self.model_stack_hash,
                "seed_manifest": self.seed_manifest_hash,
                "validator": self.validator_hash,
                "selected_model_freeze": self.selected_model_freeze_hash,
                "source_tree": self.source_tree_hash,
            }
        )
        if self.state_path.exists():
            state = self._state()
            if (
                state.call_manifest_hash != self.call_manifest.content_hash
                or state.frozen_inputs_hash != frozen
            ):
                raise HeldOutSemanticExecutionError("semantic state belongs to another freeze")
        else:
            self._write_state(
                HeldOutSemanticState(
                    call_manifest_hash=self.call_manifest.content_hash,
                    frozen_inputs_hash=frozen,
                    updated_at=self._now(),
                )
            )

    def _now(self) -> datetime:
        value = self.clock()
        if value.tzinfo is None or value.utcoffset() is None:
            raise HeldOutSemanticExecutionError("semantic clock must be timezone-aware")
        return value.astimezone(UTC)

    def _after(self, threshold: datetime) -> datetime:
        for _ in range(100):
            value = self._now()
            if value > threshold:
                return value
        return threshold + timedelta(microseconds=1)

    def _state(self) -> HeldOutSemanticState:
        if self.state_path.is_symlink() or not self.state_path.is_file():
            raise HeldOutSemanticExecutionError("semantic state is missing or unsafe")
        return HeldOutSemanticState.model_validate_json(self.state_path.read_bytes())

    def _write_state(self, state: HeldOutSemanticState) -> None:
        parent_existed = self.state_path.parent.exists()
        self.state_path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        if not parent_existed:
            parent_descriptor = os.open(
                self.state_path.parent.parent,
                os.O_RDONLY | os.O_DIRECTORY,
            )
            try:
                os.fsync(parent_descriptor)
            finally:
                os.close(parent_descriptor)
        descriptor, name = tempfile.mkstemp(
            prefix=f".{self.state_path.name}.", suffix=".tmp", dir=self.state_path.parent
        )
        temporary = Path(name)
        try:
            os.fchmod(descriptor, 0o600)
            with os.fdopen(descriptor, "wb") as stream:
                stream.write((state.to_canonical_json() + "\n").encode("utf-8"))
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, self.state_path)
            directory_descriptor = os.open(
                self.state_path.parent,
                os.O_RDONLY | os.O_DIRECTORY,
            )
            try:
                os.fsync(directory_descriptor)
            finally:
                os.close(directory_descriptor)
        finally:
            temporary.unlink(missing_ok=True)

    def _replace_state(self, state: HeldOutSemanticState, **updates: object) -> None:
        payload = state.model_dump(mode="python", exclude={"content_hash"})
        payload.update(updates)
        payload["updated_at"] = self._now()
        self._write_state(HeldOutSemanticState.model_validate(payload))

    def _runtime(
        self,
        call: HeldOutCallSpec,
        *,
        repair: bool = False,
        fixed: ConstructionRequest | None = None,
    ) -> Any:
        return development_request_runtime(
            root=self.root,
            condition=call.condition,
            tokenizer_manifest=self.tokenizer_manifest,
            seed=call.vllm_seed,
            repair=repair,
            fixed_ontology=None if fixed is None else fixed.fixed_ontology,
            fixed_evidence=None if fixed is None else fixed.packet.evidence,
        )

    def _run_config(
        self,
        call: HeldOutCallSpec,
        *,
        budgets: Any,
        repair: bool = False,
        fixed: ConstructionRequest | None = None,
    ) -> RunConditionConfig:
        runtime = self._runtime(call, repair=repair, fixed=fixed)
        maximum_input = (
            self.construction.repair_input_tokens
            if repair
            else self.construction.first_pass_input_tokens
        )
        maximum_output = (
            self.construction.repair_output_tokens
            if repair
            else self.construction.first_pass_output_tokens
        )
        return RunConditionConfig(
            config_id=f"held-out-run-config-{call.call_id}-{'repair' if repair else 'base'}",
            condition=call.condition,
            budgets=budgets,
            maximum_input_tokens=maximum_input,
            maximum_output_tokens=maximum_output,
            repair_attempt_budget=budgets.repair_attempt_budget,
            seed_block=call.seed_block,
            source_c1_seed_block=(
                call.seed_block if call.condition is ConditionName.A_FIXED_SELECT else None
            ),
            model_stack_hash=self.model_stack_hash,
            decoding_manifest_hash=runtime.decoding_manifest.content_hash,
            decoding_family_hash=runtime.decoding_manifest.comparison_family_hash,
            seed_manifest_hash=self.seed_manifest_hash,
            resolved_seed=call.vllm_seed,
            prompt_hash=runtime.prompt_hash,
            output_schema_hash=runtime.output_schema_hash,
            scored_schema_hash=SCORED_PROJECTION_SCHEMA_HASH,
            capability_manifest_hash=runtime.capability_manifest.content_hash,
            validator_hash=self.validator_hash,
            upper_ontology_hash=self.construction.upper_ontology.content_hash,
        )

    def _produce_inputs(
        self,
        call: HeldOutCallSpec,
        envelope: HeldOutCallEnvelope,
        config: RunConditionConfig,
    ) -> HeldOutProduceInputs:
        opening = envelope.query_opening
        barrier = envelope.prequery_barrier
        if (
            opening is None
            or barrier is None
            or opening.query_context is None
            or opening.packet_materialization is None
        ):
            raise HeldOutSemanticExecutionError(
                "query-time call lacks restart-safe query/materialization lineage"
            )
        neutral = self.neutral_by_unit[call.unit_id]
        packet = self.resolver.resolve_record(
            opening.evidence_packet_artifact,
            EvidencePacket,
            required_release=ReleaseClass.PUBLIC,
        )
        if call.condition is ConditionName.C2_LLM_QUERY:
            receipt = envelope.c2_prequery_receipt
            if receipt is None:
                raise HeldOutSemanticExecutionError("C2 lost its empty pre-query receipt")
            preparation = receipt.preparation
        else:
            reference = envelope.source_c1_output
            if reference is None:
                raise HeldOutSemanticExecutionError("FixedSelect lacks its complete C1 output")
            c1_preparation = self.resolver.resolve_record(
                reference,
                ConditionPreparation,
                required_release=ReleaseClass.RESTRICTED,
            )
            matching = tuple(
                item
                for item in barrier.preparation_bindings
                if item.unit_id == call.unit_id
                and item.condition is ConditionName.A_FIXED_SELECT
                and item.seed_block == call.seed_block
            )
            if len(matching) != 1:
                raise HeldOutSemanticExecutionError("FixedSelect pre-query binding is unavailable")
            preparation = prepare_fixed_selection(
                c1_preparation,
                seed_block=call.seed_block,
                prepared_at=matching[0].completed_at,
            )
        threshold = max(
            opening.query_access_event.accessed_at,
            opening.packet_materialization.completed_at,
        )
        return HeldOutProduceInputs(
            controller_unit_id=call.unit_id,
            preparation=preparation,
            snapshot=neutral.snapshot,
            packet=packet,
            context=opening.query_context,
            query_access=opening.query_access_event,
            prequery_barrier=barrier,
            query_processing_started_at=self._after(threshold),
            packet_materialization=opening.packet_materialization,
            upper_ontology=self.construction.upper_ontology,
            run_config=config,
        )

    def _semantic_request(
        self, call: HeldOutCallSpec, envelope: HeldOutCallEnvelope
    ) -> tuple[
        PreconstructionRequest | ConstructionRequest,
        RunConditionConfig,
        HeldOutProduceInputs | None,
    ]:
        neutral = self.neutral_by_unit[call.unit_id]
        if call.condition is ConditionName.C1_LLM_PRE:
            config = self._run_config(call, budgets=self.construction.preconstruction_budgets)
            runtime = development_runtime_identifiers(
                root=self.root,
                condition=call.condition,
                tokenizer_manifest=self.tokenizer_manifest,
                seed=call.vllm_seed,
            )
            requested_at = self._after(neutral.snapshot.sealed_at)
            return (
                build_c1_preconstruction_request(
                    snapshot_hash=neutral.snapshot.content_hash,
                    snapshot_sealed_at=neutral.snapshot.sealed_at,
                    sealed_horizon=neutral.snapshot.horizon,
                    ordered_snapshot_evidence_ids=neutral.snapshot.eligible_evidence_ids,
                    evidence=neutral.evidence,
                    upper_ontology=self.construction.upper_ontology,
                    preconstruction_budgets=config.budgets,
                    runtime=runtime,
                    requested_at=requested_at,
                ),
                config,
                None,
            )
        opening = envelope.query_opening
        if opening is None or opening.query_context is None:
            raise HeldOutSemanticExecutionError("query call lacks its context")
        provisional_fixed = None
        if call.condition is ConditionName.A_FIXED_SELECT:
            reference = envelope.source_c1_output
            if reference is None:
                raise HeldOutSemanticExecutionError("FixedSelect lacks C1")
            source = self.resolver.resolve_record(
                reference,
                ConditionPreparation,
                required_release=ReleaseClass.RESTRICTED,
            )
            sealed = source.sealed_preontology
            if sealed is None:
                raise HeldOutSemanticExecutionError("FixedSelect source is not C1")
            # Only runtime schema derivation uses this exact sealed graph here.
            provisional_runtime = development_runtime_identifiers(
                root=self.root,
                condition=call.condition,
                tokenizer_manifest=self.tokenizer_manifest,
                seed=call.vllm_seed,
                fixed_ontology=sealed.as_fixed_ontology(),
                fixed_evidence=to_model_visible_packet(
                    self.resolver.resolve_record(
                        opening.evidence_packet_artifact,
                        EvidencePacket,
                        required_release=ReleaseClass.PUBLIC,
                    )
                ).evidence,
            )
            provisional_fixed = ConstructionRequest(
                request_id=f"fixed-runtime-probe-{call.call_id}",
                condition=ConditionName.A_FIXED_SELECT,
                snapshot_hash=sealed.snapshot_hash,
                packet=to_model_visible_packet(
                    self.resolver.resolve_record(
                        opening.evidence_packet_artifact,
                        EvidencePacket,
                        required_release=ReleaseClass.PUBLIC,
                    )
                ),
                context=to_model_visible_query(opening.query_context),
                upper_ontology=self.construction.upper_ontology,
                budgets=opening.query_context.budgets,
                capabilities=ConstructionCapabilities.fixed_selection(),
                runtime=provisional_runtime,
                requested_at=self._after(opening.query_access_event.accessed_at),
                fixed_ontology=sealed.as_fixed_ontology(),
            )
        config = self._run_config(
            call,
            budgets=opening.query_context.budgets,
            fixed=provisional_fixed,
        )
        inputs = self._produce_inputs(call, envelope, config)
        fixed_ontology = None
        fixed_evidence = None
        if call.condition is ConditionName.A_FIXED_SELECT:
            fixed_selection = inputs.preparation.fixed_selection
            assert fixed_selection is not None
            fixed_ontology = fixed_selection.source_c1_preontology.as_fixed_ontology()
            fixed_evidence = to_model_visible_packet(inputs.packet).evidence
        runtime = development_runtime_identifiers(
            root=self.root,
            condition=call.condition,
            tokenizer_manifest=self.tokenizer_manifest,
            seed=call.vllm_seed,
            fixed_ontology=fixed_ontology,
            fixed_evidence=fixed_evidence,
        )
        requested_at = self._after(inputs.query_processing_started_at)
        request = (
            build_fixed_select_request(inputs, runtime=runtime, requested_at=requested_at)
            if call.condition is ConditionName.A_FIXED_SELECT
            else build_c2_construction_request(inputs, runtime=runtime, requested_at=requested_at)
        )
        if config.decoding_manifest_hash != request.runtime.decoding_config_hash:
            raise HeldOutSemanticExecutionError("run config differs from semantic request")
        return request, config, inputs

    def _repair_request(
        self,
        *,
        call: HeldOutCallSpec,
        base: PreconstructionRequest | ConstructionRequest,
    ) -> tuple[PreconstructionRequest | ConstructionRequest, RunConditionConfig]:
        fixed = base if isinstance(base, ConstructionRequest) and base.fixed_ontology else None
        runtime = development_runtime_identifiers(
            root=self.root,
            condition=call.condition,
            tokenizer_manifest=self.tokenizer_manifest,
            seed=call.vllm_seed,
            repair=True,
            fixed_ontology=None if fixed is None else fixed.fixed_ontology,
            fixed_evidence=None if fixed is None else fixed.packet.evidence,
        )
        payload = base.model_dump(
            mode="python", exclude={"content_hash", "runtime", "requested_at"}
        )
        payload["runtime"] = runtime
        payload["requested_at"] = self._after(base.requested_at)
        repaired_semantic = type(base).model_validate(payload)
        return repaired_semantic, self._run_config(
            call,
            budgets=base.budgets,
            repair=True,
            fixed=cast(ConstructionRequest | None, fixed),
        )

    def _repair_guided_request(
        self,
        *,
        call: HeldOutCallSpec,
        semantic: PreconstructionRequest | ConstructionRequest,
        invalid: Mapping[str, object],
        diagnostics: Sequence[Mapping[str, object]],
    ) -> GuidedJSONRequest:
        fixed = (
            semantic
            if isinstance(semantic, ConstructionRequest) and semantic.fixed_ontology
            else None
        )
        runtime = self._runtime(call, repair=True, fixed=cast(ConstructionRequest | None, fixed))
        prompt = (self.root / "prompts/repair/prompt_v1.md").read_text(encoding="utf-8")
        sections = dict(encode_development_semantic_request(semantic).sections)
        sections["invalid_draft"] = dict(invalid)
        sections["validation_diagnostics"] = [dict(item) for item in diagnostics]
        messages = (
            ChatMessage(role="system", content=prompt),
            ChatMessage(role="user", content=canonical_json(sections)),
        )
        rendered = self.tokenizer.apply_chat_template(
            [asdict(item) for item in messages],
            tokenize=True,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        if not isinstance(rendered, Sequence) or isinstance(rendered, (str, bytes, bytearray)):
            raise HeldOutSemanticExecutionError("repair tokenizer did not return token IDs")
        schema = development_output_schema_for_request(semantic)
        values: dict[str, object] = {"system_prompt": prompt, **sections}
        packing_sections = [
            PackingSection(
                name="output_schema",
                section_content_hash=canonical_sha256(schema),
                token_count=0,
            )
        ]
        for name, value in values.items():
            text = value if isinstance(value, str) else canonical_json(value)
            packing_sections.append(
                PackingSection(
                    name=name,
                    section_content_hash=hashlib.sha256(text.encode()).hexdigest(),
                    token_count=len(self.tokenizer.encode(text, add_special_tokens=False)),
                )
            )
        encoded_count = sum(item.token_count for item in packing_sections)
        if len(rendered) < encoded_count:
            raise HeldOutSemanticExecutionError("repair section counts exceed rendered prompt")
        if len(rendered) > encoded_count:
            packing_sections.append(
                PackingSection(
                    name="chat_protocol",
                    section_content_hash=self.tokenizer_manifest.nonthinking_probe_sha256,
                    token_count=len(rendered) - encoded_count,
                )
            )
        packing = PackingReport.build(
            condition=call.condition,
            tokenizer_revision=self.tokenizer_manifest.tokenizer_revision,
            maximum_model_tokens=runtime.decoding_manifest.maximum_model_tokens,
            maximum_input_tokens=runtime.decoding_manifest.maximum_input_tokens,
            reserved_output_tokens=runtime.decoding_manifest.maximum_output_tokens,
            sections=packing_sections,
            required_section_names=("output_schema", *values),
            complete_evidence_snapshot=isinstance(semantic, PreconstructionRequest) or None,
            complete_evidence_packet=(
                None if isinstance(semantic, PreconstructionRequest) else True
            ),
            complete_sealed_ontology=(
                True if call.condition is ConditionName.A_FIXED_SELECT else None
            ),
        )
        return GuidedJSONRequest(
            request_id=f"{call.call_id}-repair",
            model_name=semantic.runtime.model_id,
            condition=call.condition,
            messages=messages,
            output_schema=schema,
            decoding=runtime.decoding_manifest,
            packing=packing,
            rendered_input_token_count=len(rendered),
        )

    def _event(self, event_id: str) -> Any:
        matches = tuple(
            item for item in self.artifacts.ledger.gpu_events() if item.event_id == event_id
        )
        if len(matches) != 1:
            raise HeldOutSemanticExecutionError("GPU request lacks exactly one ledger event")
        return matches[0]

    def _job_identity(self, call: HeldOutCallSpec) -> dict[str, object]:
        query_blind = call.condition is ConditionName.C1_LLM_PRE
        return {
            "execution_id": self.call_manifest.manifest_id,
            "call_id": call.call_id,
            "manifest_hash": self.call_manifest.content_hash,
            "condition": call.condition.value,
            "lifecycle_kind": (
                "query_blind_prebuild" if query_blind else "query_time_generation"
            ),
        }

    def _advance_state(self, job_id: str, state: JobState, occurred_at: datetime) -> None:
        """Append one lifecycle state while making terminal replay idempotent."""

        observed = tuple(item.to_state for item in self.artifacts.ledger.transitions(job_id))
        if state in observed:
            return
        self.artifacts.ledger.transition_job(job_id, state, occurred_at=occurred_at)

    def _register_projection_inputs(
        self,
        *,
        inputs: HeldOutProduceInputs,
    ) -> tuple[str, str]:
        """Register study-local aliases for an exact snapshot and evidence packet."""

        study_id = self.call_manifest.manifest_id
        snapshot = inputs.snapshot
        snapshot_release = ReleaseClass(snapshot.release_class.value)
        snapshot_reference = self.resolver.persist_record(
            snapshot,
            object_kind="evidence_snapshot",
            release_class=snapshot_release,
            created_at=snapshot.created_at,
        )
        snapshot_key = canonical_sha256((study_id, snapshot.content_hash))[:32]
        snapshot_input_id = f"projection-snapshot-input-{snapshot_key}"
        snapshot_record_id = f"projection-snapshot-{snapshot_key}"
        self.artifacts.ledger.register_input(
            input_id=snapshot_input_id,
            study_id=study_id,
            input_kind=InputKind.EVIDENCE_SNAPSHOT,
            content_hash=snapshot.content_hash,
            artifact_hash=snapshot_reference.artifact_hash,
            release_class=snapshot_release,
            created_at=snapshot.created_at,
        )
        self.artifacts.ledger.register_evidence_snapshot(
            snapshot_id=snapshot_record_id,
            input_id=snapshot_input_id,
            horizon_hash=snapshot.horizon.content_hash,
            evidence_manifest_hash=canonical_sha256(
                tuple(item.content_hash for item in inputs.packet.evidence)
            ),
            index_configuration_hash=snapshot.index_config_hash,
            prequery_seal_hash=snapshot.content_hash,
            eligible_evidence_count=len(snapshot.eligible_evidence_ids),
            created_at=snapshot.sealed_at,
        )

        materialization = self.artifacts.ledger.get_packet_materialization(
            inputs.packet_materialization.content_hash
        )
        packet_artifact = self.artifacts.ledger.get_artifact(
            materialization.packet_artifact_hash
        )
        packet_release = ReleaseClass(inputs.packet.release_class.value)
        if packet_artifact.release_class is not packet_release:
            raise HeldOutSemanticExecutionError(
                "projection packet artifact release class changed"
            )
        packet_key = canonical_sha256((study_id, inputs.packet.content_hash))[:32]
        packet_input_id = f"projection-packet-input-{packet_key}"
        self.artifacts.ledger.register_input(
            input_id=packet_input_id,
            study_id=study_id,
            input_kind=InputKind.EVIDENCE_PACKET,
            content_hash=inputs.packet.content_hash,
            artifact_hash=packet_artifact.content_hash,
            release_class=packet_release,
            created_at=inputs.packet.created_at,
        )
        return snapshot_record_id, packet_input_id

    def _record_attempt_validation(
        self,
        *,
        call: HeldOutCallSpec,
        attempt: _AttemptArtifacts,
        accepted: bool,
        checked_at: datetime,
        error: BaseException | None = None,
        generation: ValidatedGeneration | None = None,
        parent_validation_id: str | None = None,
        repair: bool,
        terminal: bool,
    ) -> str:
        """Persist one structural verdict and every unsuccessful attempt."""

        if accepted != (generation is not None) or accepted == (error is not None):
            raise HeldOutSemanticExecutionError(
                "ledger validation requires exactly one accepted generation or error"
            )
        diagnostic_payload: dict[str, object] = {
            "accepted": accepted,
            "assessment_scope": "runtime_structural_only",
            "call_id": call.call_id,
            "attempt_id": attempt.attempt_id,
            "raw_output_artifact_hash": (
                None if attempt.raw_reference is None else attempt.raw_reference.artifact_hash
            ),
        }
        if generation is not None:
            diagnostic_payload.update(
                validated_generation_hash=generation.content_hash,
                validator_report_hashes=list(generation.validator_report_hashes),
            )
        else:
            assert error is not None
            diagnostic_payload.update(
                exception_type=type(error).__name__,
                failure_kind=_failure_kind(error).value,
            )
        diagnostic = self.artifacts.put_bytes(
            (canonical_json(diagnostic_payload) + "\n").encode("utf-8"),
            media_type=_media_type("structural_validation_diagnostic"),
            release_class=ReleaseClass.RESTRICTED,
            created_at=checked_at,
        )
        input_hash = (
            diagnostic.content_hash
            if attempt.raw_reference is None
            else attempt.raw_reference.artifact_hash
        )
        validation_id = f"{attempt.attempt_id}-validation"
        self.artifacts.ledger.record_validation(
            validation_id=validation_id,
            job_id=attempt.job_id,
            attempt_id=attempt.attempt_id,
            input_artifact_hash=input_hash,
            validator_manifest_hash=self.validator_hash,
            validation_status=(
                LedgerValidationStatus.ACCEPTED
                if accepted
                else LedgerValidationStatus.REJECTED
            ),
            evidence_support_status=LedgerEvidenceSupportStatus.NOT_APPLICABLE,
            temporal_status=LedgerTemporalValidationStatus.NOT_APPLICABLE,
            commitment_status=LedgerCommitmentCheckStatus.NOT_APPLICABLE,
            semantic_assessment_scope=(
                LedgerSemanticAssessmentScope.RUNTIME_STRUCTURAL_ONLY_NOT_ASSESSED
            ),
            diagnostics_artifact_hash=diagnostic.content_hash,
            parent_validation_id=parent_validation_id,
            repair_attempt_id=attempt.attempt_id if repair else None,
            created_at=checked_at,
        )
        if error is not None:
            self.artifacts.ledger.record_failure(
                attempt_id=attempt.attempt_id,
                failure_kind=_failure_kind(error),
                message=(
                    "Model attempt failed structural acceptance; "
                    "inspect restricted diagnostics"
                ),
                details={
                    "call_id": call.call_id,
                    "exception_type": type(error).__name__,
                    "repair": repair,
                },
                artifact_hash=diagnostic.content_hash,
                occurred_at=checked_at,
            )
        if terminal:
            self._advance_state(attempt.job_id, JobState.VALIDATED, checked_at)
        return validation_id

    def _record_projection(
        self,
        *,
        call: HeldOutCallSpec,
        attempt: _AttemptArtifacts,
        inputs: HeldOutProduceInputs,
        projection: Any,
        validation_id: str,
        finalized_at: datetime,
    ) -> HeldOutCASReference:
        reference = self.resolver.persist_record(
            projection,
            object_kind="ontology_projection",
            release_class=ReleaseClass.RESTRICTED,
            created_at=finalized_at,
        )
        snapshot_id, packet_input_id = self._register_projection_inputs(inputs=inputs)
        certificate = projection.construction_certificate or projection.construction_seal
        if certificate is None:
            raise HeldOutSemanticExecutionError(
                "successful projection lacks a construction certificate or seal"
            )
        self.artifacts.ledger.record_projection(
            projection_id=f"{attempt.attempt_id}-projection",
            job_id=attempt.job_id,
            validation_id=validation_id,
            snapshot_id=snapshot_id,
            packet_input_id=packet_input_id,
            condition_id=call.condition.value,
            context_hash=inputs.context.content_hash,
            upper_ontology_hash=inputs.upper_ontology.content_hash,
            construction_certificate_hash=certificate.content_hash,
            projection_artifact_hash=reference.artifact_hash,
            projection_semantic_hash=projection.content_hash,
            release_class=ReleaseClass.RESTRICTED,
            finalized_at=finalized_at,
        )
        return reference

    def _persist_attempt(
        self,
        *,
        service: MeteredGenerationService,
        call: HeldOutCallSpec,
        semantic: PreconstructionRequest | ConstructionRequest,
        guided: GuidedJSONRequest,
        config: RunConditionConfig,
        repair: bool,
        remaining_required_seconds: float,
        parent_attempt_id: str | None,
        created_at: datetime | None = None,
        prequery_sealed_at: datetime | None = None,
        query_revealed_at: datetime | None = None,
    ) -> _AttemptArtifacts:
        created_at = self._now() if created_at is None else created_at
        semantic_reference = self.resolver.persist_record(
            semantic,
            object_kind=(
                "preconstruction_request"
                if isinstance(semantic, PreconstructionRequest)
                else "construction_request"
            ),
            release_class=ReleaseClass.RESTRICTED,
            created_at=created_at,
        )
        packing_reference = self.resolver.persist_record(
            guided.packing,
            object_kind="packing_report",
            release_class=ReleaseClass.RESTRICTED,
            created_at=created_at,
        )
        sealed_at = (
            self.neutral_by_unit[call.unit_id].snapshot.sealed_at
            if isinstance(semantic, PreconstructionRequest)
            else prequery_sealed_at
        )
        if sealed_at is None:
            raise HeldOutSemanticExecutionError("generation job lacks its prequery seal time")
        if isinstance(semantic, PreconstructionRequest):
            if query_revealed_at is not None:
                raise HeldOutSemanticExecutionError(
                    "query-blind preconstruction cannot receive a query reveal time"
                )
        elif query_revealed_at is None:
            raise HeldOutSemanticExecutionError(
                "query-time generation lacks its query-access timestamp"
            )
        job = self.artifacts.ledger.create_or_resume_job(
            self._job_identity(call),
            release_class=ReleaseClass.RESTRICTED,
            created_at=sealed_at,
        )
        self.artifacts.ledger.link_job_to_study(
            study_id=self.call_manifest.manifest_id,
            job_id=job.job_id,
            created_at=created_at,
        )
        milestones: list[tuple[JobState, datetime]] = [
            (JobState.PREQUERY_SEALED, sealed_at)
        ]
        if query_revealed_at is not None:
            milestones.append((JobState.QUERY_REVEALED, query_revealed_at))
        self.artifacts.ledger.advance_job_lifecycle(job.job_id, milestones)
        attempt_suffix = "repair" if repair else "base"
        attempt_id = f"{self.call_manifest.manifest_id}-{call.call_id}-{attempt_suffix}"
        self.artifacts.ledger.record_attempt(
            attempt_id=attempt_id,
            job_id=job.job_id,
            attempt_kind=AttemptKind.REPAIR if repair else AttemptKind.BASE,
            input_hash=guided.request_hash,
            config_hash=config.content_hash,
            seed=call.vllm_seed,
            parent_attempt_id=parent_attempt_id,
            created_at=created_at,
        )
        event_id = f"{self.call_manifest.manifest_id}-{call.call_id}-{attempt_suffix}-gpu"
        generated = None
        failure = None
        try:
            generated = service.generate(
                guided,
                event_id=event_id,
                watchdog_seconds=call.watchdog_seconds,
                repair=repair,
                job_id=job.job_id,
                attempt_id=attempt_id,
                remaining_required_seconds=remaining_required_seconds,
                accounting_details={
                    "phase": "held_out_primary",
                    "ordinal": call.ordinal,
                    "call_class": call.call_class,
                    "attempt": attempt_suffix,
                },
            )
        except BaseException as error:
            failure = error
        event = self._event(event_id)
        allocated = event.allocated_seconds
        if not math.isfinite(allocated) or allocated < 0:
            raise HeldOutSemanticExecutionError("GPU event has invalid allocation")
        raw_reference = None
        if generated is not None:
            raw_record = self.artifacts.put_bytes(
                generated.raw_response,
                media_type=_media_type("raw_model_response"),
                release_class=ReleaseClass.RESTRICTED,
                created_at=_parse_utc(event.ended_at),
            )
            raw_reference = self.resolver.reference(
                raw_record,
                logical_content_hash=hashlib.sha256(generated.raw_response).hexdigest(),
                object_kind="raw_model_response",
            )
        model_call_id = f"{self.call_manifest.manifest_id}-{call.call_id}-{attempt_suffix}-model"
        self.artifacts.ledger.record_model_call(
            model_call_id=model_call_id,
            job_id=job.job_id,
            attempt_id=attempt_id,
            gpu_event_id=event_id,
            backend=ModelBackend.VLLM_GPU,
            call_role=_call_role(call.condition, repair=repair),
            retry_class=_retry_class(call),
            model_manifest_hash=self.model_stack_hash,
            decoding_manifest_hash=guided.decoding.content_hash,
            request_hash=guided.request_hash,
            response_artifact_hash=(None if raw_reference is None else raw_reference.artifact_hash),
            construction_unit_hash=canonical_sha256(
                {
                    "manifest": self.call_manifest.content_hash,
                    "ordinal": call.ordinal,
                    "condition": call.condition,
                }
            ),
            served_context_count=1,
            prompt_tokens=0 if generated is None else generated.prompt_tokens,
            completion_tokens=0 if generated is None else generated.completion_tokens,
            allocated_gpu_seconds=allocated,
            successful=generated is not None and failure is None,
            created_at=self._now(),
        )
        self._advance_state(
            job.job_id,
            JobState.REPAIRED if repair else JobState.GENERATED,
            _parse_utc(event.ended_at),
        )
        model_call = self.artifacts.ledger.get_model_call(model_call_id)
        return _AttemptArtifacts(
            job_id=job.job_id,
            attempt_id=attempt_id,
            semantic=semantic,
            guided=guided,
            semantic_reference=semantic_reference,
            packing_reference=packing_reference,
            raw_reference=raw_reference,
            event_id=event_id,
            event_hash=_event_hash(event),
            model_call_id=model_call_id,
            model_call_hash=canonical_sha256(asdict(model_call)),
            generated=generated,
            allocated_seconds=allocated,
            failure=failure,
        )

    def _validate_generation(
        self,
        *,
        call: HeldOutCallSpec,
        envelope: HeldOutCallEnvelope,
        attempt: _AttemptArtifacts,
        config: RunConditionConfig,
        repair_attempt: int,
        repair_parent_raw_hash: str | None,
        base_invalid: Mapping[str, object] | None = None,
        diagnosed_paths: tuple[str, ...] = (),
        validated_at_override: datetime | None = None,
    ) -> ValidatedGeneration:
        generated = attempt.generated
        raw = attempt.raw_reference
        if generated is None or raw is None:
            raise attempt.failure or HeldOutSemanticExecutionError("model returned no generation")
        guided = attempt.guided
        if (
            generated.request_id != guided.request_id
            or generated.request_hash != guided.request_hash
            or generated.response_sha256 != raw.logical_content_hash
            or generated.finish_reason != "stop"
            or generated.prompt_tokens != guided.rendered_input_token_count
            or generated.completion_tokens > guided.decoding.maximum_output_tokens
        ):
            raise HeldOutSemanticExecutionError("vLLM response envelope changed")
        alias = encode_development_semantic_request(attempt.semantic).alias_manifest
        restored = restore_model_output_source_aliases(generated.parsed_object, alias)
        raw_draft = OntologyDraft.model_validate(restored)
        if (
            raw_draft.budget_accounting.input_tokens != 0
            or raw_draft.budget_accounting.output_tokens != 0
        ):
            raise HeldOutSemanticExecutionError("model token sentinels must remain zero")
        event = self._event(attempt.event_id)
        started_at = _parse_utc(event.started_at)
        completed_at = _parse_utc(event.ended_at)
        normalized = normalize_generation_metadata(
            raw_draft,
            decision_recorded_at=completed_at,
            input_tokens=generated.prompt_tokens,
            output_tokens=generated.completion_tokens,
        )
        semantic = attempt.semantic
        if call.condition is ConditionName.A_FIXED_SELECT:
            request = cast(ConstructionRequest, semantic)
            fixed = request.fixed_ontology
            source_reference = envelope.source_c1_output
            if fixed is None or source_reference is None:
                raise HeldOutSemanticExecutionError("FixedSelect validation lost C1")
            source_preparation = self.resolver.resolve_record(
                source_reference,
                ConditionPreparation,
                required_release=ReleaseClass.RESTRICTED,
            )
            source = source_preparation.sealed_preontology
            if source is None:
                raise HeldOutSemanticExecutionError("FixedSelect source is not sealed C1")
            inventory = sealed_inventory_from_fixed_ontology(
                fixed, seed_block=call.seed_block, source_draft=source.draft
            )
            enforce_fixed_select_draft(normalized, sealed=inventory, seed_block=call.seed_block)
        evidence = (
            semantic.evidence
            if isinstance(semantic, PreconstructionRequest)
            else semantic.packet.evidence
        )
        boundary = validate_draft_structure(
            draft=normalized,
            upper_ontology=semantic.upper_ontology,
            evidence=evidence,
            horizon=self.neutral_by_unit[call.unit_id].snapshot.horizon,
            budgets=semantic.budgets,
            capabilities=semantic.capabilities,
        )
        boundary.raise_for_errors()
        report_hashes = [boundary.content_hash]
        if repair_attempt:
            if base_invalid is None:
                raise HeldOutSemanticExecutionError("repair lacks its preserved parent")
            preservation = validate_repair_preservation(
                base_draft=base_invalid,
                repaired_draft=cast(Mapping[str, object], restored),
                diagnosed_paths=diagnosed_paths,
            )
            preservation.raise_for_errors()
            report_hashes.append(preservation.content_hash)
        validated_at = (
            self._after(completed_at)
            if validated_at_override is None
            else validated_at_override.astimezone(UTC)
        )
        if validated_at <= completed_at:
            raise HeldOutSemanticExecutionError(
                "validation timestamp must follow generation completion"
            )
        validation = runtime_structural_acceptance_record(
            validation_id=f"validation-{canonical_sha256((call.call_id, raw.artifact_hash))[:20]}",
            target_id=normalized.content_hash,
            repair_parent_hash=repair_parent_raw_hash,
            repair_attempt=repair_attempt,
            validated_at=validated_at,
        )
        return ValidatedGeneration(
            generation_id=(
                f"held-out-generation-{canonical_sha256((call.call_id, raw.artifact_hash))[:20]}"
            ),
            condition=call.condition,
            request_hash=semantic.content_hash,
            raw_output_artifact_hash=raw.artifact_hash,
            raw_parsed_draft=raw_draft,
            draft=normalized,
            normalized_draft_hash=normalized.content_hash,
            stage_manifest_hash=(
                envelope.prequery_stage.staging_manifest_hash
                if envelope.query_stage is None
                else envelope.query_stage.staging_manifest_hash
            ),
            query_access_event_hash=(
                None
                if envelope.query_opening is None
                else envelope.query_opening.query_access_event.content_hash
            ),
            prequery_barrier_hash=(
                None
                if envelope.prequery_barrier is None
                else envelope.prequery_barrier.content_hash
            ),
            packing_report_hash=guided.packing.content_hash,
            capability_manifest_hash=CapabilityManifest.for_condition(call.condition).content_hash,
            seed_manifest_hash=self.seed_manifest_hash,
            prompt_hash=cast(str, config.prompt_hash),
            output_schema_hash=cast(str, config.output_schema_hash),
            decoding_manifest_hash=cast(str, config.decoding_manifest_hash),
            validator_hash=self.validator_hash,
            model_stack_hash=self.model_stack_hash,
            seed=call.vllm_seed,
            input_tokens=generated.prompt_tokens,
            output_tokens=generated.completion_tokens,
            generation_started_at=started_at,
            generation_completed_at=completed_at,
            decision_recorded_at=completed_at,
            validation_records=(validation,),
            validator_report_hashes=tuple(report_hashes),
            validated_at=validated_at,
            repair_attempt=repair_attempt,
            repair_parent_raw_output_hash=repair_parent_raw_hash,
        )

    def _fixed_audit(
        self,
        call: HeldOutCallSpec,
        envelope: HeldOutCallEnvelope,
        parsed: Mapping[str, object] | None,
    ) -> FixedSelectCapabilityAudit | None:
        if call.condition is not ConditionName.A_FIXED_SELECT:
            return None
        constructive = {
            item.value
            for item in ConstructionOperator
            if item
            not in {
                ConstructionOperator.SELECTION,
                ConstructionOperator.COMPRESSION,
                ConstructionOperator.SUPPORTED_DESCRIPTION,
            }
        }
        decisions = () if parsed is None else parsed.get("decisions", ())
        count = 0
        if isinstance(decisions, Sequence) and not isinstance(decisions, (str, bytes)):
            count = sum(
                1
                for item in decisions
                if isinstance(item, Mapping) and item.get("operator") in constructive
            )
        if envelope.complete_c1_graph_hash is None or envelope.source_c1_seal_hash is None:
            raise HeldOutSemanticExecutionError("FixedSelect audit lacks complete C1 lineage")
        return FixedSelectCapabilityAudit(
            complete_c1_graph_hash=envelope.complete_c1_graph_hash,
            source_c1_seal_hash=envelope.source_c1_seal_hash,
            constructive_operator_attempt_count=count,
            mechanically_rejected_operator_count=count,
        )

    def _finalize(
        self,
        *,
        call: HeldOutCallSpec,
        envelope: HeldOutCallEnvelope,
        semantic: PreconstructionRequest | ConstructionRequest,
        inputs: HeldOutProduceInputs | None,
        generation: ValidatedGeneration,
    ) -> tuple[
        ConditionPreparation | ConditionAttemptRecord,
        tuple[PrequeryPreparationBinding, ...],
    ]:
        if call.condition is ConditionName.C1_LLM_PRE:
            request = cast(PreconstructionRequest, semantic)
            preparation = seal_c1_preconstruction(
                request=request,
                generation=generation,
                seed_block=call.seed_block,
                sealed_at=self._after(generation.validated_at),
            )
            fixed = prepare_fixed_selection(
                preparation,
                seed_block=call.seed_block,
                prepared_at=self._after(preparation.completed_at),
            )
            # Persist the deterministic Fixed preparation before the query barrier.
            self.resolver.persist_record(
                fixed,
                object_kind="condition_preparation",
                release_class=ReleaseClass.RESTRICTED,
                created_at=fixed.completed_at,
            )
            source = cast(Any, preparation.sealed_preontology)
            fixed_source = cast(Any, fixed.fixed_selection)
            return preparation, (
                PrequeryPreparationBinding(
                    unit_id=call.unit_id,
                    condition=ConditionName.C1_LLM_PRE,
                    seed_block=call.seed_block,
                    snapshot_hash=source.snapshot_hash,
                    preparation_hash=preparation.content_hash,
                    lineage_artifact_hash=source.construction_seal.content_hash,
                    completed_at=preparation.completed_at,
                ),
                PrequeryPreparationBinding(
                    unit_id=call.unit_id,
                    condition=ConditionName.A_FIXED_SELECT,
                    seed_block=call.seed_block,
                    snapshot_hash=source.snapshot_hash,
                    preparation_hash=fixed.content_hash,
                    lineage_artifact_hash=fixed_source.content_hash,
                    completed_at=fixed.completed_at,
                ),
            )
        if inputs is None:
            raise HeldOutSemanticExecutionError("query finalizer lacks ProduceInputs")
        request = cast(ConstructionRequest, semantic)
        result = (
            finalize_fixed_select_draft(cast(Any, inputs), request=request, generation=generation)
            if call.condition is ConditionName.A_FIXED_SELECT
            else finalize_c2_draft(cast(Any, inputs), request=request, generation=generation)
        )
        return result, ()

    def execute(
        self,
        *,
        service: MeteredGenerationService,
        call: HeldOutCallSpec,
        envelope: HeldOutCallEnvelope,
        remaining_required_seconds: float,
        repair_allowed: bool,
    ) -> HeldOutServiceResult:
        if call.content_hash != envelope.call_spec_hash:
            raise HeldOutSemanticExecutionError("semantic envelope differs from its call")
        state = self._state()
        if call.call_id in state.completed:
            raise HeldOutSemanticExecutionError("completed held-out call cannot be resent")
        if state.active_intent is not None:
            raise HeldOutSemanticExecutionError(
                "an interrupted GPU intent requires recovery and cannot be resent"
            )
        semantic, config, inputs = self._semantic_request(call, envelope)
        guided = build_development_guided_request(
            root=self.root,
            call_id=call.call_id,
            semantic_request=semantic,
            tokenizer=self.tokenizer,
            tokenizer_manifest=self.tokenizer_manifest,
            seed=call.vllm_seed,
        )
        if (
            config.decoding_manifest_hash != guided.decoding.content_hash
            or config.output_schema_hash != canonical_sha256(guided.output_schema)
        ):
            raise HeldOutSemanticExecutionError("packed request differs from run configuration")
        intent = HeldOutSemanticIntent(
            call_id=call.call_id,
            call_spec_hash=call.content_hash,
            envelope_hash=envelope.content_hash,
            semantic_request_hash=semantic.content_hash,
            guided_request_hash=guided.request_hash,
            gpu_event_id=f"{self.call_manifest.manifest_id}-{call.call_id}-base-gpu",
            started_at=self._now(),
        )
        self._replace_state(state, active_intent=intent)
        try:
            base = self._persist_attempt(
                service=service,
                call=call,
                semantic=semantic,
                guided=guided,
                config=config,
                repair=False,
                remaining_required_seconds=remaining_required_seconds,
                parent_attempt_id=None,
                created_at=intent.started_at,
                prequery_sealed_at=(
                    self.neutral_by_unit[call.unit_id].snapshot.sealed_at
                    if isinstance(semantic, PreconstructionRequest)
                    else cast(HeldOutProduceInputs, inputs).prequery_barrier.sealed_at
                ),
                query_revealed_at=(
                    None
                    if isinstance(semantic, PreconstructionRequest)
                    else cast(HeldOutProduceInputs, inputs).query_access.accessed_at
                ),
            )
        except BaseException:
            # No terminal GPU ledger event means the intent remains non-reissuable.
            raise
        generation = None
        final_attempt = base
        validation_failure: BaseException | None = base.failure
        diagnosed_paths: tuple[str, ...] = ()
        base_validation_id: str | None = None
        invalid = cast(Mapping[str, object], base.generated.parsed_object) if base.generated else {}
        if validation_failure is None:
            try:
                generation = self._validate_generation(
                    call=call,
                    envelope=envelope,
                    attempt=base,
                    config=config,
                    repair_attempt=0,
                    repair_parent_raw_hash=None,
                )
            except BaseException as error:
                validation_failure = error
                if isinstance(error, ValidationError):
                    diagnosed_paths = tuple(
                        "/" + "/".join(str(part) for part in item["loc"])
                        for item in error.errors(include_input=False)
                    )
                if not diagnosed_paths:
                    diagnosed_paths = ("/draft",)
        base_checked_at = self._after(_parse_utc(self._event(base.event_id).ended_at))
        if validation_failure is not None:
            base_validation_id = self._record_attempt_validation(
                call=call,
                attempt=base,
                accepted=False,
                checked_at=base_checked_at,
                error=validation_failure,
                repair=False,
                terminal=not (repair_allowed and base.generated is not None),
            )
        elif generation is not None:
            base_validation_id = self._record_attempt_validation(
                call=call,
                attempt=base,
                accepted=True,
                checked_at=generation.validated_at,
                generation=generation,
                repair=False,
                terminal=True,
            )
        repair = None
        if validation_failure is not None and repair_allowed and base.generated is not None:
            repair_semantic, repair_config = self._repair_request(call=call, base=semantic)
            diagnostics = tuple(
                {"code": "boundary_validation", "path": path} for path in diagnosed_paths
            )
            repair_guided = self._repair_guided_request(
                call=call,
                semantic=repair_semantic,
                invalid=invalid,
                diagnostics=diagnostics,
            )
            repair_started_at = self._after(_parse_utc(self._event(base.event_id).ended_at))
            repair_semantic_reference = self.resolver.persist_record(
                repair_semantic,
                object_kind=(
                    "preconstruction_request"
                    if isinstance(repair_semantic, PreconstructionRequest)
                    else "construction_request"
                ),
                release_class=ReleaseClass.RESTRICTED,
                created_at=repair_started_at,
            )
            repair_packing_reference = self.resolver.persist_record(
                repair_guided.packing,
                object_kind="packing_report",
                release_class=ReleaseClass.RESTRICTED,
                created_at=repair_started_at,
            )
            intent_payload = intent.model_dump(mode="python", exclude={"content_hash"})
            intent_payload.update(
                repair_semantic_request=repair_semantic_reference,
                repair_packing_report=repair_packing_reference,
                repair_semantic_request_hash=repair_semantic.content_hash,
                repair_guided_request_hash=repair_guided.request_hash,
                repair_gpu_event_id=(
                    f"{self.call_manifest.manifest_id}-{call.call_id}-repair-gpu"
                ),
                repair_started_at=repair_started_at,
            )
            intent = HeldOutSemanticIntent.model_validate(intent_payload)
            self._replace_state(self._state(), active_intent=intent)
            repair = self._persist_attempt(
                service=service,
                call=call,
                semantic=repair_semantic,
                guided=repair_guided,
                config=repair_config,
                repair=True,
                remaining_required_seconds=max(
                    0.0, remaining_required_seconds - call.watchdog_seconds
                ),
                parent_attempt_id=(f"{self.call_manifest.manifest_id}-{call.call_id}-base"),
                created_at=repair_started_at,
                prequery_sealed_at=(
                    self.neutral_by_unit[call.unit_id].snapshot.sealed_at
                    if isinstance(repair_semantic, PreconstructionRequest)
                    else cast(HeldOutProduceInputs, inputs).prequery_barrier.sealed_at
                ),
                query_revealed_at=(
                    None
                    if isinstance(repair_semantic, PreconstructionRequest)
                    else cast(HeldOutProduceInputs, inputs).query_access.accessed_at
                ),
            )
            final_attempt = repair
            validation_failure = repair.failure
            if validation_failure is None:
                try:
                    generation = self._validate_generation(
                        call=call,
                        envelope=envelope,
                        attempt=repair,
                        config=repair_config,
                        repair_attempt=1,
                        repair_parent_raw_hash=(
                            None if base.raw_reference is None else base.raw_reference.artifact_hash
                        ),
                        base_invalid=invalid,
                        diagnosed_paths=diagnosed_paths,
                    )
                except BaseException as error:
                    validation_failure = error
            repair_checked_at = self._after(
                _parse_utc(self._event(repair.event_id).ended_at)
            )
            if validation_failure is not None:
                self._record_attempt_validation(
                    call=call,
                    attempt=repair,
                    accepted=False,
                    checked_at=repair_checked_at,
                    error=validation_failure,
                    parent_validation_id=base_validation_id,
                    repair=True,
                    terminal=True,
                )
            elif generation is not None:
                base_validation_id = self._record_attempt_validation(
                    call=call,
                    attempt=repair,
                    accepted=True,
                    checked_at=generation.validated_at,
                    generation=generation,
                    parent_validation_id=base_validation_id,
                    repair=True,
                    terminal=True,
                )
        result_object = None
        bindings: tuple[PrequeryPreparationBinding, ...] = ()
        if generation is not None and validation_failure is None:
            try:
                result_object, bindings = self._finalize(
                    call=call,
                    envelope=envelope,
                    semantic=final_attempt.semantic,
                    inputs=inputs,
                    generation=generation,
                )
            except BaseException as error:
                validation_failure = error
        completed_at = self._after(_parse_utc(self._event(final_attempt.event_id).ended_at))
        validation_reference = None
        output_reference = None
        if generation is not None and validation_failure is None and result_object is not None:
            validation_reference = self.resolver.persist_record(
                generation,
                object_kind="validated_generation",
                release_class=ReleaseClass.RESTRICTED,
                created_at=completed_at,
            )
            output_reference = self.resolver.persist_record(
                result_object,
                object_kind=(
                    "condition_preparation"
                    if isinstance(result_object, ConditionPreparation)
                    else "condition_attempt"
                ),
                release_class=ReleaseClass.RESTRICTED,
                created_at=completed_at,
            )
            if isinstance(result_object, ConditionAttemptRecord):
                projection = result_object.projection
                if projection is None or inputs is None or base_validation_id is None:
                    raise HeldOutSemanticExecutionError(
                        "successful condition attempt lacks ledger projection lineage"
                    )
                self._record_projection(
                    call=call,
                    attempt=final_attempt,
                    inputs=inputs,
                    projection=projection,
                    validation_id=base_validation_id,
                    finalized_at=completed_at,
                )
        if validation_failure is not None:
            failure_kind = _failure_kind(validation_failure)
            try:
                self.artifacts.ledger.failures_for_lineage(final_attempt.attempt_id)
                has_terminal_failure = any(
                    item.attempt_id == final_attempt.attempt_id
                    for item in self.artifacts.ledger.failures_for_lineage(
                        final_attempt.attempt_id
                    )
                )
            except KeyError:
                has_terminal_failure = False
            if not has_terminal_failure:
                self.artifacts.ledger.record_failure(
                    attempt_id=final_attempt.attempt_id,
                    failure_kind=failure_kind,
                    message="Held-out output finalization failed; inspect restricted call lineage",
                    details={
                        "call_id": call.call_id,
                        "exception_type": type(validation_failure).__name__,
                    },
                    artifact_hash=(
                        None
                        if final_attempt.raw_reference is None
                        else final_attempt.raw_reference.artifact_hash
                    ),
                    occurred_at=completed_at,
                )
        self._advance_state(final_attempt.job_id, JobState.FINALIZED, completed_at)
        chain_hash = gpu_ledger_chain_hash(self.artifacts)
        receipt = HeldOutCallArtifactReceipt(
            call_id=call.call_id,
            call_spec_hash=call.content_hash,
            condition=call.condition,
            semantic_request=base.semantic_reference,
            packing_report=base.packing_reference,
            output=output_reference,
            validation=validation_reference,
            raw_response=base.raw_reference,
            gpu_event_id=base.event_id,
            gpu_event_hash=base.event_hash,
            model_call_id=base.model_call_id,
            model_call_record_hash=base.model_call_hash,
            repair_semantic_request=None if repair is None else repair.semantic_reference,
            repair_packing_report=None if repair is None else repair.packing_reference,
            repair_raw_response=None if repair is None else repair.raw_reference,
            repair_gpu_event_id=None if repair is None else repair.event_id,
            repair_gpu_event_hash=None if repair is None else repair.event_hash,
            repair_model_call_id=None if repair is None else repair.model_call_id,
            repair_model_call_record_hash=None if repair is None else repair.model_call_hash,
            ledger_chain_hash=chain_hash,
            completed_at=completed_at,
        )
        receipt_reference = self.resolver.persist_record(
            receipt,
            object_kind="held_out_call_audit_receipt",
            release_class=ReleaseClass.RESTRICTED,
            created_at=completed_at,
        )
        success = validation_failure is None and result_object is not None
        terminal_error = validation_failure or HeldOutSemanticExecutionError(
            "missing terminal output"
        )
        fixed_audit = self._fixed_audit(
            call,
            envelope,
            None if final_attempt.generated is None else final_attempt.generated.parsed_object,
        )
        attempt_projection = (
            result_object.projection if isinstance(result_object, ConditionAttemptRecord) else None
        )
        c1_source = (
            result_object.sealed_preontology
            if isinstance(result_object, ConditionPreparation)
            else None
        )
        opening = envelope.query_opening
        c2_inventory = (
            None if envelope.c2_prequery_receipt is None else envelope.c2_prequery_receipt.inventory
        )
        allocated = base.allocated_seconds + (0.0 if repair is None else repair.allocated_seconds)
        result = HeldOutServiceResult(
            call_id=call.call_id,
            condition=call.condition,
            outcome=(
                RunOutcome.SUCCEEDED
                if success
                else _run_outcome(terminal_error)
            ),
            request_started=True,
            request_hash=semantic.content_hash,
            output_artifact_hash=None
            if output_reference is None
            else output_reference.logical_content_hash,
            validation_artifact_hash=(
                None if validation_reference is None else validation_reference.logical_content_hash
            ),
            ledger_receipt_hash=receipt.content_hash,
            ledger_receipt_artifact_hash=receipt_reference.artifact_hash,
            packing_report_hash=guided.packing.content_hash,
            artifact_receipt=receipt,
            global_ledger_chain_hash=chain_hash,
            evidence_packet_hash=(None if opening is None else opening.evidence_packet_hash),
            horizon_hash=(
                None if opening is None else cast(Any, opening.opened_stage.horizon_hash)
            ),
            budget_hash=(None if opening is None else cast(Any, opening.opened_stage.budget_hash)),
            construction_seal_hash=(
                None if c1_source is None else c1_source.construction_seal.content_hash
            ),
            complete_c1_graph_hash=(
                None if c1_source is None else c1_source.construction_seal.ontology_hash
            ),
            construction_seal=(None if c1_source is None else c1_source.construction_seal),
            prequery_preparation_bindings=bindings,
            empty_prequery_inventory_hash=(
                None if c2_inventory is None else c2_inventory.content_hash
            ),
            construction_certificate_hash=(
                None
                if attempt_projection is None or attempt_projection.construction_certificate is None
                else attempt_projection.construction_certificate.content_hash
            ),
            pre_query_inventory=(
                c2_inventory if call.condition is ConditionName.C2_LLM_QUERY else None
            ),
            construction_certificate=(
                None if attempt_projection is None else attempt_projection.construction_certificate
            ),
            fixed_select_capability_audit=fixed_audit,
            query_revealed_at=(None if opening is None else opening.query_access_event.accessed_at),
            allocated_gpu_seconds=allocated,
            repair_attempts=int(repair is not None),
            failure_code=(
                None
                if success
                else f"held_out_{_failure_kind(terminal_error).value}"
            ),
            completed_at=completed_at,
        )
        result_record = self.artifacts.put_bytes(
            (result.to_canonical_json() + "\n").encode(),
            media_type=_media_type("held_out_service_result"),
            release_class=ReleaseClass.RESTRICTED,
            created_at=completed_at,
        )
        state = self._state()
        if state.active_intent != intent:
            raise HeldOutSemanticExecutionError("semantic intent changed before result commit")
        completed = dict(state.completed)
        completed[call.call_id] = HeldOutCompletedResultPointer(
            call_id=call.call_id,
            envelope_hash=envelope.content_hash,
            result_artifact_hash=result_record.content_hash,
            result_logical_hash=result.content_hash,
        )
        self._replace_state(state, completed=completed, active_intent=None)
        return result

    def recover(
        self, *, call: HeldOutCallSpec, envelope: HeldOutCallEnvelope
    ) -> HeldOutServiceResult | None:
        state = self._state()
        pointer = state.completed.get(call.call_id)
        if pointer is not None:
            if pointer.envelope_hash != envelope.content_hash:
                raise HeldOutSemanticExecutionError("recovered call envelope changed")
            artifact = self.artifacts.ledger.get_artifact(pointer.result_artifact_hash)
            result = HeldOutServiceResult.model_validate_json(
                self.artifacts.blobs.read_bytes(artifact, allow_restricted=True)
            )
            if result.content_hash != pointer.result_logical_hash:
                raise HeldOutSemanticExecutionError("recovered result CAS changed")
            return result
        intent = state.active_intent
        if intent is not None:
            if (
                intent.call_id != call.call_id
                or intent.call_spec_hash != call.content_hash
                or intent.envelope_hash != envelope.content_hash
            ):
                raise HeldOutSemanticExecutionError("active semantic intent changed")
            semantic, config, _inputs = self._semantic_request(call, envelope)
            guided = build_development_guided_request(
                root=self.root,
                call_id=call.call_id,
                semantic_request=semantic,
                tokenizer=self.tokenizer,
                tokenizer_manifest=self.tokenizer_manifest,
                seed=call.vllm_seed,
            )
            if (
                semantic.content_hash != intent.semantic_request_hash
                or guided.request_hash != intent.guided_request_hash
                or config.decoding_manifest_hash != guided.decoding.content_hash
                or config.output_schema_hash != canonical_sha256(guided.output_schema)
            ):
                raise HeldOutSemanticExecutionError(
                    "reconstructed semantic intent differs from the frozen call"
                )
            events = tuple(
                item
                for item in self.artifacts.ledger.gpu_events()
                if item.event_id == intent.gpu_event_id
            )
            if len(events) > 1:
                raise HeldOutSemanticExecutionError(
                    "interrupted semantic intent has duplicate GPU events"
                )
            if not events:
                unresolved = {
                    item.allocation_id
                    for item in self.artifacts.ledger.unresolved_gpu_allocations()
                }
                if intent.gpu_event_id in unresolved:
                    return None
                # The durable intent preceded the allocation journal. Clearing
                # only this exact reconstructed intent allows the slotted call
                # to issue once; no model request or GPU interval can exist.
                self._replace_state(state, active_intent=None)
                return None
            return self._recover_terminal_event(
                state=state,
                intent=intent,
                call=call,
                envelope=envelope,
                semantic=semantic,
                guided=guided,
                config=config,
                event=events[0],
            )
        return None

    def _recover_recorded_attempt(
        self,
        *,
        call: HeldOutCallSpec,
        semantic: PreconstructionRequest | ConstructionRequest,
        guided: GuidedJSONRequest,
        config: RunConditionConfig,
        event: Any,
        repair: bool,
        started_at: datetime,
        job: Any,
        prequery_sealed_at: datetime,
        query_revealed_at: datetime | None,
    ) -> _AttemptArtifacts:
        """Rebuild exact terminal attempt lineage without invoking the model."""

        suffix = "repair" if repair else "base"
        expected_event_id = (
            f"{self.call_manifest.manifest_id}-{call.call_id}-{suffix}-gpu"
        )
        if event.event_id != expected_event_id:
            raise HeldOutSemanticExecutionError(
                "interrupted GPU event differs from its semantic intent"
            )
        semantic_reference = self.resolver.persist_record(
            semantic,
            object_kind=(
                "preconstruction_request"
                if isinstance(semantic, PreconstructionRequest)
                else "construction_request"
            ),
            release_class=ReleaseClass.RESTRICTED,
            created_at=started_at,
        )
        packing_reference = self.resolver.persist_record(
            guided.packing,
            object_kind="packing_report",
            release_class=ReleaseClass.RESTRICTED,
            created_at=started_at,
        )
        attempt_id = f"{self.call_manifest.manifest_id}-{call.call_id}-{suffix}"
        self.artifacts.ledger.record_attempt(
            attempt_id=attempt_id,
            job_id=job.job_id,
            attempt_kind=AttemptKind.REPAIR if repair else AttemptKind.BASE,
            input_hash=guided.request_hash,
            config_hash=config.content_hash,
            seed=call.vllm_seed,
            parent_attempt_id=(
                f"{self.call_manifest.manifest_id}-{call.call_id}-base"
                if repair
                else None
            ),
            created_at=started_at,
        )
        milestones: list[tuple[JobState, datetime]] = [
            (JobState.PREQUERY_SEALED, prequery_sealed_at)
        ]
        if query_revealed_at is not None:
            milestones.append((JobState.QUERY_REVEALED, query_revealed_at))
        self.artifacts.ledger.advance_job_lifecycle(job.job_id, milestones)
        construction_unit_hash = canonical_sha256(
            {
                "manifest": self.call_manifest.content_hash,
                "ordinal": call.ordinal,
                "condition": call.condition,
            }
        )
        model_call_id = (
            f"{self.call_manifest.manifest_id}-{call.call_id}-{suffix}-model"
        )
        try:
            model_call = self.artifacts.ledger.get_model_call(model_call_id)
        except KeyError:
            self.artifacts.ledger.record_model_call(
                model_call_id=model_call_id,
                job_id=job.job_id,
                attempt_id=attempt_id,
                gpu_event_id=event.event_id,
                backend=ModelBackend.VLLM_GPU,
                call_role=_call_role(call.condition, repair=repair),
                retry_class=_retry_class(call),
                model_manifest_hash=self.model_stack_hash,
                decoding_manifest_hash=guided.decoding.content_hash,
                request_hash=guided.request_hash,
                response_artifact_hash=None,
                construction_unit_hash=construction_unit_hash,
                served_context_count=1,
                prompt_tokens=0,
                completion_tokens=0,
                allocated_gpu_seconds=event.allocated_seconds,
                successful=False,
                created_at=_parse_utc(event.ended_at),
            )
            model_call = self.artifacts.ledger.get_model_call(model_call_id)
        if (
            model_call.job_id != job.job_id
            or model_call.attempt_id != attempt_id
            or model_call.gpu_event_id != event.event_id
            or model_call.backend is not ModelBackend.VLLM_GPU
            or model_call.call_role is not _call_role(call.condition, repair=repair)
            or model_call.retry_class is not _retry_class(call)
            or model_call.model_manifest_hash != self.model_stack_hash
            or model_call.decoding_manifest_hash != guided.decoding.content_hash
            or model_call.request_hash != guided.request_hash
            or model_call.construction_unit_hash != construction_unit_hash
            or model_call.served_context_count != 1
            or model_call.allocated_gpu_microseconds != event.allocated_microseconds
            or (model_call.successful != (model_call.response_artifact_hash is not None))
        ):
            raise HeldOutSemanticExecutionError(
                "interrupted model-call ledger lineage changed"
            )
        self._advance_state(
            job.job_id,
            JobState.REPAIRED if repair else JobState.GENERATED,
            _parse_utc(event.ended_at),
        )
        raw_reference = None
        generated = None
        if model_call.response_artifact_hash is not None:
            raw_record = self.artifacts.ledger.get_artifact(
                model_call.response_artifact_hash
            )
            raw_bytes = self.artifacts.blobs.read_bytes(
                raw_record,
                allow_restricted=True,
            )
            if (
                raw_record.release_class is not ReleaseClass.RESTRICTED
                or raw_record.media_type != _media_type("raw_model_response")
            ):
                raise HeldOutSemanticExecutionError(
                    "interrupted raw response escaped restricted storage"
                )
            raw_reference = self.resolver.reference(
                raw_record,
                logical_content_hash=hashlib.sha256(raw_bytes).hexdigest(),
                object_kind="raw_model_response",
            )
            try:
                response = json.loads(raw_bytes)
                choice = response["choices"][0]
                parsed_object = json.loads(choice["message"]["content"])
                usage = response["usage"]
                prompt_tokens = usage["prompt_tokens"]
                completion_tokens = usage["completion_tokens"]
                finish_reason = choice.get("finish_reason")
                service_request_id = response.get("id")
            except (IndexError, KeyError, TypeError, json.JSONDecodeError) as error:
                raise HeldOutSemanticExecutionError(
                    "interrupted raw response cannot reconstruct one guided result"
                ) from error
            if (
                not isinstance(parsed_object, Mapping)
                or isinstance(prompt_tokens, bool)
                or not isinstance(prompt_tokens, int)
                or prompt_tokens < 0
                or isinstance(completion_tokens, bool)
                or not isinstance(completion_tokens, int)
                or completion_tokens < 0
                or (
                    finish_reason is not None
                    and not isinstance(finish_reason, str)
                )
                or (
                    service_request_id is not None
                    and not isinstance(service_request_id, str)
                )
                or prompt_tokens != model_call.prompt_tokens
                or completion_tokens != model_call.completion_tokens
            ):
                raise HeldOutSemanticExecutionError(
                    "interrupted raw response metadata changed"
                )
            generated = GenerationResult(
                request_id=guided.request_id,
                request_hash=guided.request_hash,
                response_sha256=hashlib.sha256(raw_bytes).hexdigest(),
                parsed_object=dict(parsed_object),
                raw_response=raw_bytes,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                finish_reason=finish_reason,
                service_request_id=service_request_id,
            )
        return _AttemptArtifacts(
            job_id=job.job_id,
            attempt_id=attempt_id,
            semantic=semantic,
            guided=guided,
            semantic_reference=semantic_reference,
            packing_reference=packing_reference,
            raw_reference=raw_reference,
            event_id=event.event_id,
            event_hash=_event_hash(event),
            model_call_id=model_call_id,
            model_call_hash=canonical_sha256(asdict(model_call)),
            generated=generated,
            allocated_seconds=event.allocated_seconds,
            failure=(
                None
                if generated is not None
                else HeldOutSemanticExecutionError(
                    "terminal GPU event has no recoverable response"
                )
            ),
        )

    def _recover_terminal_event(
        self,
        *,
        state: HeldOutSemanticState,
        intent: HeldOutSemanticIntent,
        call: HeldOutCallSpec,
        envelope: HeldOutCallEnvelope,
        semantic: PreconstructionRequest | ConstructionRequest,
        guided: GuidedJSONRequest,
        config: RunConditionConfig,
        event: Any,
    ) -> HeldOutServiceResult | None:
        """Commit an ITT failure from a terminal event without issuing inference."""

        expected_repair_event_id = (
            f"{self.call_manifest.manifest_id}-{call.call_id}-repair-gpu"
        )
        repair_events = tuple(
            item
            for item in self.artifacts.ledger.gpu_events()
            if item.event_id == expected_repair_event_id
        )
        if len(repair_events) > 1:
            raise HeldOutSemanticExecutionError(
                "interrupted semantic intent has duplicate repair GPU events"
            )
        if intent.repair_gpu_event_id is None and repair_events:
            raise HeldOutSemanticExecutionError(
                "repair GPU event lacks its prior durable semantic intent"
            )
        job = self.artifacts.ledger.create_or_resume_job(
            self._job_identity(call),
            release_class=ReleaseClass.RESTRICTED,
            created_at=(
                self.neutral_by_unit[call.unit_id].snapshot.sealed_at
                if isinstance(semantic, PreconstructionRequest)
                else cast(Any, envelope.prequery_barrier).sealed_at
            ),
        )
        self.artifacts.ledger.link_job_to_study(
            study_id=self.call_manifest.manifest_id,
            job_id=job.job_id,
            created_at=intent.started_at,
        )
        base = self._recover_recorded_attempt(
            call=call,
            semantic=semantic,
            guided=guided,
            config=config,
            event=event,
            repair=False,
            started_at=intent.started_at,
            job=job,
            prequery_sealed_at=(
                self.neutral_by_unit[call.unit_id].snapshot.sealed_at
                if isinstance(semantic, PreconstructionRequest)
                else cast(Any, envelope.prequery_barrier).sealed_at
            ),
            query_revealed_at=(
                None
                if isinstance(semantic, PreconstructionRequest)
                else cast(Any, envelope.query_opening).query_access_event.accessed_at
            ),
        )
        repair = None
        base_validation_error: BaseException | None = base.failure
        base_generation: ValidatedGeneration | None = None
        diagnosed_paths: tuple[str, ...] = ()
        if intent.repair_gpu_event_id is not None:
            if (
                intent.repair_gpu_event_id != expected_repair_event_id
                or intent.repair_semantic_request is None
                or intent.repair_packing_report is None
                or intent.repair_semantic_request.object_kind
                != base.semantic_reference.object_kind
                or intent.repair_started_at is None
                or intent.repair_started_at < _parse_utc(event.ended_at)
            ):
                raise HeldOutSemanticExecutionError(
                    "interrupted repair semantic identity changed"
                )
            if base.generated is None or base.raw_reference is None:
                raise HeldOutSemanticExecutionError(
                    "repair intent lacks its exact persisted base response"
                )
            repaired_type = type(semantic)
            persisted_repair_semantic = self.resolver.resolve_record(
                intent.repair_semantic_request,
                repaired_type,
                required_release=ReleaseClass.RESTRICTED,
            )
            provisional_repair_semantic, repair_config = self._repair_request(
                call=call,
                base=semantic,
            )
            expected_payload = provisional_repair_semantic.model_dump(
                mode="python",
                exclude={"content_hash", "requested_at"},
            )
            expected_payload["requested_at"] = persisted_repair_semantic.requested_at
            expected_repair_semantic = repaired_type.model_validate(expected_payload)
            try:
                base_generation = self._validate_generation(
                    call=call,
                    envelope=envelope,
                    attempt=base,
                    config=config,
                    repair_attempt=0,
                    repair_parent_raw_hash=None,
                    validated_at_override=(
                        _parse_utc(event.ended_at) + timedelta(microseconds=1)
                    ),
                )
            except BaseException as error:
                base_validation_error = error
                if isinstance(error, ValidationError):
                    diagnosed_paths = tuple(
                        "/" + "/".join(str(part) for part in item["loc"])
                        for item in error.errors(include_input=False)
                    )
                if not diagnosed_paths:
                    diagnosed_paths = ("/draft",)
            else:
                raise HeldOutSemanticExecutionError(
                    "repair intent follows a base output that now validates"
                )
            repair_guided = self._repair_guided_request(
                call=call,
                semantic=persisted_repair_semantic,
                invalid=cast(Mapping[str, object], base.generated.parsed_object),
                diagnostics=tuple(
                    {"code": "boundary_validation", "path": path}
                    for path in diagnosed_paths
                ),
            )
            persisted_repair_packing = self.resolver.resolve_record(
                intent.repair_packing_report,
                PackingReport,
                required_release=ReleaseClass.RESTRICTED,
            )
            if (
                persisted_repair_semantic.content_hash
                != intent.repair_semantic_request_hash
                or persisted_repair_semantic != expected_repair_semantic
                or repair_guided.request_hash != intent.repair_guided_request_hash
                or repair_guided.packing != persisted_repair_packing
                or repair_config.decoding_manifest_hash
                != repair_guided.decoding.content_hash
                or repair_config.output_schema_hash
                != canonical_sha256(repair_guided.output_schema)
            ):
                raise HeldOutSemanticExecutionError(
                    "reconstructed repair intent differs from its frozen call"
                )
            if repair_events:
                repair = self._recover_recorded_attempt(
                    call=call,
                    semantic=persisted_repair_semantic,
                    guided=repair_guided,
                    config=repair_config,
                    event=repair_events[0],
                    repair=True,
                    started_at=intent.repair_started_at,
                    job=job,
                    prequery_sealed_at=(
                        self.neutral_by_unit[call.unit_id].snapshot.sealed_at
                        if isinstance(persisted_repair_semantic, PreconstructionRequest)
                        else cast(Any, envelope.prequery_barrier).sealed_at
                    ),
                    query_revealed_at=(
                        None
                        if isinstance(persisted_repair_semantic, PreconstructionRequest)
                        else cast(Any, envelope.query_opening).query_access_event.accessed_at
                    ),
                )
            else:
                unresolved = {
                    item.allocation_id
                    for item in self.artifacts.ledger.unresolved_gpu_allocations()
                }
                if intent.repair_gpu_event_id in unresolved:
                    return None
        final_event = event if repair is None else repair_events[0]
        base_event_end = _parse_utc(event.ended_at)
        final_event_end = _parse_utc(final_event.ended_at)
        completed_at = final_event_end + timedelta(microseconds=2)
        if intent.repair_gpu_event_id is None and base.failure is None:
            try:
                base_generation = self._validate_generation(
                    call=call,
                    envelope=envelope,
                    attempt=base,
                    config=config,
                    repair_attempt=0,
                    repair_parent_raw_hash=None,
                    validated_at_override=(
                        _parse_utc(event.ended_at) + timedelta(microseconds=1)
                    ),
                )
                base_validation_error = None
            except BaseException as error:
                base_validation_error = error
        base_validation_id = self._record_attempt_validation(
            call=call,
            attempt=base,
            accepted=base_generation is not None and base_validation_error is None,
            checked_at=base_event_end + timedelta(microseconds=1),
            error=base_validation_error,
            generation=base_generation,
            repair=False,
            terminal=repair is None,
        )
        final_attempt = base
        final_validation_error = base_validation_error
        if repair is not None:
            repair_generation: ValidatedGeneration | None = None
            repair_validation_error = repair.failure
            if repair_validation_error is None:
                try:
                    repair_generation = self._validate_generation(
                        call=call,
                        envelope=envelope,
                        attempt=repair,
                        config=repair_config,
                        repair_attempt=1,
                        repair_parent_raw_hash=(
                            None
                            if base.raw_reference is None
                            else base.raw_reference.artifact_hash
                        ),
                        base_invalid=cast(
                            Mapping[str, object], base.generated.parsed_object
                        ),
                        diagnosed_paths=diagnosed_paths,
                        validated_at_override=(
                            _parse_utc(final_event.ended_at)
                            + timedelta(microseconds=1)
                        ),
                    )
                except BaseException as error:
                    repair_validation_error = error
            self._record_attempt_validation(
                call=call,
                attempt=repair,
                accepted=(
                    repair_generation is not None and repair_validation_error is None
                ),
                checked_at=final_event_end + timedelta(microseconds=1),
                error=repair_validation_error,
                generation=repair_generation,
                parent_validation_id=base_validation_id,
                repair=True,
                terminal=True,
            )
            final_attempt = repair
            final_validation_error = repair_validation_error
        if final_validation_error is None:
            # A structurally accepted response can still be an ITT failure when
            # power was lost before the durable scientific result pointer.
            self.artifacts.ledger.record_failure(
                attempt_id=final_attempt.attempt_id,
                failure_kind=FailureKind.INTERRUPTED,
                message="Terminal model response was interrupted before result commit",
                details={"call_id": call.call_id, "repair": repair is not None},
                artifact_hash=(
                    None
                    if final_attempt.raw_reference is None
                    else final_attempt.raw_reference.artifact_hash
                ),
                occurred_at=completed_at,
            )
        self._advance_state(final_attempt.job_id, JobState.FINALIZED, completed_at)
        chain_hash = gpu_ledger_chain_hash(self.artifacts)
        receipt = HeldOutCallArtifactReceipt(
            call_id=call.call_id,
            call_spec_hash=call.content_hash,
            condition=call.condition,
            semantic_request=base.semantic_reference,
            packing_report=base.packing_reference,
            raw_response=base.raw_reference,
            gpu_event_id=base.event_id,
            gpu_event_hash=base.event_hash,
            model_call_id=base.model_call_id,
            model_call_record_hash=base.model_call_hash,
            repair_semantic_request=(
                None if repair is None else repair.semantic_reference
            ),
            repair_packing_report=(
                None if repair is None else repair.packing_reference
            ),
            repair_raw_response=None if repair is None else repair.raw_reference,
            repair_gpu_event_id=None if repair is None else repair.event_id,
            repair_gpu_event_hash=None if repair is None else repair.event_hash,
            repair_model_call_id=None if repair is None else repair.model_call_id,
            repair_model_call_record_hash=(
                None if repair is None else repair.model_call_hash
            ),
            ledger_chain_hash=chain_hash,
            completed_at=completed_at,
        )
        receipt_reference = self.resolver.persist_record(
            receipt,
            object_kind="held_out_call_audit_receipt",
            release_class=ReleaseClass.RESTRICTED,
            created_at=completed_at,
        )
        opening = envelope.query_opening
        c2_inventory = (
            None
            if envelope.c2_prequery_receipt is None
            else envelope.c2_prequery_receipt.inventory
        )
        result = HeldOutServiceResult(
            call_id=call.call_id,
            condition=call.condition,
            outcome=RunOutcome.FAILED,
            request_started=True,
            request_hash=semantic.content_hash,
            ledger_receipt_hash=receipt.content_hash,
            ledger_receipt_artifact_hash=receipt_reference.artifact_hash,
            packing_report_hash=guided.packing.content_hash,
            artifact_receipt=receipt,
            global_ledger_chain_hash=chain_hash,
            evidence_packet_hash=(
                None if opening is None else opening.evidence_packet_hash
            ),
            horizon_hash=(
                None if opening is None else cast(Any, opening.opened_stage.horizon_hash)
            ),
            budget_hash=(
                None if opening is None else cast(Any, opening.opened_stage.budget_hash)
            ),
            empty_prequery_inventory_hash=(
                None if c2_inventory is None else c2_inventory.content_hash
            ),
            pre_query_inventory=(
                c2_inventory if call.condition is ConditionName.C2_LLM_QUERY else None
            ),
            fixed_select_capability_audit=self._fixed_audit(call, envelope, None),
            query_revealed_at=(
                None if opening is None else opening.query_access_event.accessed_at
            ),
            allocated_gpu_seconds=(
                base.allocated_seconds
                + (0.0 if repair is None else repair.allocated_seconds)
            ),
            repair_attempts=int(repair is not None),
            failure_code=(
                "held_out_interrupted_before_repair_allocation"
                if intent.repair_gpu_event_id is not None and repair is None
                else "held_out_interrupted_after_terminal_gpu_event"
            ),
            completed_at=completed_at,
        )
        result_record = self.artifacts.put_bytes(
            (result.to_canonical_json() + "\n").encode(),
            media_type=_media_type("held_out_service_result"),
            release_class=ReleaseClass.RESTRICTED,
            created_at=completed_at,
        )
        if self._state().active_intent != intent:
            raise HeldOutSemanticExecutionError(
                "semantic intent changed during terminal reconstruction"
            )
        completed = dict(state.completed)
        completed[call.call_id] = HeldOutCompletedResultPointer(
            call_id=call.call_id,
            envelope_hash=envelope.content_hash,
            result_artifact_hash=result_record.content_hash,
            result_logical_hash=result.content_hash,
        )
        self._replace_state(self._state(), completed=completed, active_intent=None)
        return result

    def recovery_pending(
        self, *, call: HeldOutCallSpec, envelope: HeldOutCallEnvelope
    ) -> bool:
        """Report only a result/intent that is durably bound to this exact slot."""

        state = self._state()
        pointer = state.completed.get(call.call_id)
        if pointer is not None:
            if pointer.envelope_hash != envelope.content_hash:
                raise HeldOutSemanticExecutionError("recovered call envelope changed")
            return True
        intent = state.active_intent
        if intent is None:
            return False
        if (
            intent.call_id != call.call_id
            or intent.call_spec_hash != call.content_hash
            or intent.envelope_hash != envelope.content_hash
        ):
            raise HeldOutSemanticExecutionError("active semantic intent changed")
        return True

    def project_preconstructed(
        self,
        *,
        unit: HeldOutUnitPlan,
        query_stage_hash: str,
        query_opening: Any,
        condition: ConditionName,
        seed_block: int | None,
        construction_seal_hash: str,
        complete_graph_hash: str,
    ) -> PreconstructedProjectionReceipt:
        """Apply the deterministic C1 selector to the exact durable same-seed graph."""

        if condition is not ConditionName.C1_LLM_PRE or seed_block not in {1, 2}:
            raise HeldOutSemanticExecutionError("C1 projector received another condition")
        call = next(
            (
                item
                for item in self.call_manifest.calls
                if item.call_class == "test_c1"
                and item.unit_id == unit.unit_id
                and item.seed_block == seed_block
            ),
            None,
        )
        if call is None:
            raise HeldOutSemanticExecutionError("C1 projector lacks its registered source call")
        pointer = self._state().completed.get(call.call_id)
        if pointer is None:
            raise HeldOutSemanticExecutionError("C1 projector source is not durable")
        result_record = self.artifacts.ledger.get_artifact(pointer.result_artifact_hash)
        result = HeldOutServiceResult.model_validate_json(
            self.artifacts.blobs.read_bytes(result_record, allow_restricted=True)
        )
        output_reference = (
            None if result.artifact_receipt is None else result.artifact_receipt.output
        )
        if (
            result.outcome is not RunOutcome.SUCCEEDED
            or result.construction_seal_hash != construction_seal_hash
            or result.complete_c1_graph_hash != complete_graph_hash
            or output_reference is None
        ):
            raise HeldOutSemanticExecutionError("C1 projection source lineage changed")
        preparation = self.resolver.resolve_record(
            output_reference,
            ConditionPreparation,
            required_release=ReleaseClass.RESTRICTED,
        )
        preontology = preparation.sealed_preontology
        opening = query_opening
        if (
            preontology is None
            or opening.query_context is None
            or opening.packet_materialization is None
            or opening.opened_stage.staging_manifest_hash != query_stage_hash
        ):
            raise HeldOutSemanticExecutionError("C1 projection query lineage is incomplete")
        packet = self.resolver.resolve_record(
            opening.evidence_packet_artifact,
            EvidencePacket,
            required_release=ReleaseClass.PUBLIC,
        )
        barrier_record = self.artifacts.ledger.get_prequery_barrier(
            opening.query_access_event.prequery_barrier_hash
        )
        barrier_artifact = self.artifacts.ledger.get_artifact(barrier_record.barrier_artifact_hash)
        barrier = PrequeryBarrier.model_validate_json(
            self.artifacts.blobs.read_bytes(
                barrier_artifact,
                allow_restricted=barrier_artifact.release_class is ReleaseClass.RESTRICTED,
            )
        )
        config = self._run_config(call, budgets=opening.query_context.budgets)
        started_at = self._after(
            max(
                opening.query_access_event.accessed_at,
                opening.packet_materialization.completed_at,
            )
        )
        inputs = HeldOutProduceInputs(
            controller_unit_id=unit.unit_id,
            preparation=preparation,
            snapshot=self.neutral_by_unit[unit.unit_id].snapshot,
            packet=packet,
            context=opening.query_context,
            query_access=opening.query_access_event,
            prequery_barrier=barrier,
            query_processing_started_at=started_at,
            packet_materialization=opening.packet_materialization,
            upper_ontology=self.construction.upper_ontology,
            run_config=config,
        )
        projection = project_sealed_c1(preontology, cast(Any, inputs))
        completed_at = self._after(started_at)
        artifact = self.artifacts.put_bytes(
            (projection.to_canonical_json() + "\n").encode(),
            media_type=_media_type("ontology_projection"),
            release_class=ReleaseClass.RESTRICTED,
            created_at=completed_at,
        )
        projection_job = self.artifacts.ledger.create_or_resume_job(
            {
                "execution_id": self.call_manifest.manifest_id,
                "call_id": f"{call.call_id}-projection-{query_stage_hash[:16]}",
                "manifest_hash": self.call_manifest.content_hash,
                "condition": ConditionName.C1_LLM_PRE.value,
                "lifecycle_kind": "query_time_projection",
                "source_prebuild_call_id": call.call_id,
                "query_stage_hash": query_stage_hash,
            },
            release_class=ReleaseClass.RESTRICTED,
            created_at=barrier.sealed_at,
        )
        self.artifacts.ledger.link_job_to_study(
            study_id=self.call_manifest.manifest_id,
            job_id=projection_job.job_id,
            created_at=barrier.sealed_at,
        )
        self.artifacts.ledger.advance_job_lifecycle(
            projection_job.job_id,
            (
                (JobState.PREQUERY_SEALED, barrier.sealed_at),
                (JobState.QUERY_REVEALED, opening.query_access_event.accessed_at),
                (JobState.GENERATED, completed_at),
            ),
        )
        cpu_attempt_id = (
            f"{self.call_manifest.manifest_id}-{call.call_id}-"
            f"projection-{query_stage_hash[:16]}"
        )
        self.artifacts.ledger.record_attempt(
            attempt_id=cpu_attempt_id,
            job_id=projection_job.job_id,
            attempt_kind=AttemptKind.BASE,
            input_hash=canonical_sha256(
                {
                    "sealed_preontology": preontology.content_hash,
                    "packet": packet.content_hash,
                    "context": opening.query_context.content_hash,
                }
            ),
            config_hash=config.content_hash,
            seed=call.vllm_seed,
            created_at=started_at,
        )
        diagnostic = self.artifacts.put_bytes(
            (
                canonical_json(
                    {
                        "accepted": True,
                        "assessment_scope": "runtime_structural_only",
                        "projection_hash": projection.content_hash,
                        "source_prebuild_call_id": call.call_id,
                    }
                )
                + "\n"
            ).encode("utf-8"),
            media_type=_media_type("structural_validation_diagnostic"),
            release_class=ReleaseClass.RESTRICTED,
            created_at=completed_at,
        )
        validation_id = f"{cpu_attempt_id}-validation"
        self.artifacts.ledger.record_validation(
            validation_id=validation_id,
            job_id=projection_job.job_id,
            attempt_id=cpu_attempt_id,
            input_artifact_hash=artifact.content_hash,
            validator_manifest_hash=self.validator_hash,
            validation_status=LedgerValidationStatus.ACCEPTED,
            evidence_support_status=LedgerEvidenceSupportStatus.NOT_APPLICABLE,
            temporal_status=LedgerTemporalValidationStatus.NOT_APPLICABLE,
            commitment_status=LedgerCommitmentCheckStatus.NOT_APPLICABLE,
            semantic_assessment_scope=(
                LedgerSemanticAssessmentScope.RUNTIME_STRUCTURAL_ONLY_NOT_ASSESSED
            ),
            diagnostics_artifact_hash=diagnostic.content_hash,
            created_at=completed_at,
        )
        snapshot_id, packet_input_id = self._register_projection_inputs(inputs=inputs)
        self.artifacts.ledger.record_projection(
            projection_id=f"{cpu_attempt_id}-projection",
            job_id=projection_job.job_id,
            validation_id=validation_id,
            snapshot_id=snapshot_id,
            packet_input_id=packet_input_id,
            condition_id=ConditionName.C1_LLM_PRE.value,
            context_hash=inputs.context.content_hash,
            upper_ontology_hash=inputs.upper_ontology.content_hash,
            construction_certificate_hash=construction_seal_hash,
            projection_artifact_hash=artifact.content_hash,
            projection_semantic_hash=projection.content_hash,
            release_class=ReleaseClass.RESTRICTED,
            finalized_at=completed_at,
        )
        self._advance_state(projection_job.job_id, JobState.VALIDATED, completed_at)
        self._advance_state(projection_job.job_id, JobState.FINALIZED, completed_at)
        return PreconstructedProjectionReceipt(
            unit_id=unit.unit_id,
            query_stage_hash=query_stage_hash,
            condition=ConditionName.C1_LLM_PRE,
            seed_block=cast(Any, seed_block),
            source_construction_seal_hash=construction_seal_hash,
            source_complete_graph_hash=complete_graph_hash,
            outcome=RunOutcome.SUCCEEDED,
            projection_artifact_hash=artifact.content_hash,
            evidence_packet_hash=packet.content_hash,
            horizon_hash=opening.query_context.spoiler_horizon.content_hash,
            budget_hash=opening.query_context.budgets.content_hash,
            completed_at=completed_at,
        )


__all__ = [
    "FrozenHeldOutSemanticExecutor",
    "HeldOutCompletedResultPointer",
    "HeldOutSemanticExecutionError",
    "HeldOutSemanticIntent",
    "HeldOutSemanticState",
]
