"""Common condition interface and blocking experimental-integrity invariants.

This module does not invoke a model and does not perform ontology construction.
It defines the envelopes used by all four conditions so timing, evidence, budgets,
seed pairing, and intention-to-treat handling cannot drift in condition-specific
code.
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import datetime
from enum import StrEnum
from typing import Literal, Protocol, runtime_checkable

from pydantic import AwareDatetime, Field, model_validator

from story_projection_onto.contracts import (
    ConditionName,
    ConstructionSeal,
    EvidencePacket,
    EvidenceSnapshot,
    FixedOntologyInput,
    ImmutableRecord,
    ModelVisibleRevision,
    OntologyDraft,
    OntologyProjection,
    OutputBudgets,
    PreQueryInventory,
    QueryContext,
    ReleaseClass,
    RunOutcome,
    Sha256Digest,
    UpperOntology,
    canonical_sha256,
)
from story_projection_onto.evidence import assert_evidence_packet_equality
from story_projection_onto.llm import semantic_fingerprints_from_draft


class ConditionIntegrityError(ValueError):
    """A blocking timing, lineage, evidence, budget, or pairing violation."""


class ExecutionStage(StrEnum):
    """The registered condition-job state machine, including failure terminals."""

    PLANNED = "planned"
    PREQUERY_SEALED = "prequery_sealed"
    QUERY_REVEALED = "query_revealed"
    GENERATED = "generated"
    VALIDATED = "validated"
    REPAIRED = "repaired"
    FINALIZED = "finalized"
    SCORED = "scored"
    RENDERED = "rendered"
    INVALID = "invalid"
    FAILED = "failed"
    TIMED_OUT = "timed_out"
    INTERRUPTED = "interrupted"


_SUCCESS_TRANSITIONS: dict[ExecutionStage, frozenset[ExecutionStage]] = {
    ExecutionStage.PLANNED: frozenset({ExecutionStage.PREQUERY_SEALED}),
    ExecutionStage.PREQUERY_SEALED: frozenset({ExecutionStage.QUERY_REVEALED}),
    ExecutionStage.QUERY_REVEALED: frozenset({ExecutionStage.GENERATED}),
    ExecutionStage.GENERATED: frozenset({ExecutionStage.VALIDATED, ExecutionStage.REPAIRED}),
    ExecutionStage.REPAIRED: frozenset({ExecutionStage.VALIDATED}),
    ExecutionStage.VALIDATED: frozenset({ExecutionStage.FINALIZED}),
    ExecutionStage.FINALIZED: frozenset({ExecutionStage.SCORED}),
    ExecutionStage.SCORED: frozenset({ExecutionStage.RENDERED}),
}
_FAILURE_STAGES = frozenset(
    {
        ExecutionStage.INVALID,
        ExecutionStage.FAILED,
        ExecutionStage.TIMED_OUT,
        ExecutionStage.INTERRUPTED,
    }
)
_FAILURE_ORIGINS = frozenset(
    {
        ExecutionStage.PLANNED,
        ExecutionStage.PREQUERY_SEALED,
        ExecutionStage.QUERY_REVEALED,
        ExecutionStage.GENERATED,
        ExecutionStage.REPAIRED,
        ExecutionStage.VALIDATED,
    }
)


class ExecutionTransition(ImmutableRecord):
    from_stage: ExecutionStage
    to_stage: ExecutionStage
    occurred_at: AwareDatetime
    artifact_hash: Sha256Digest | None = None
    diagnostic_code: str | None = None

    @model_validator(mode="after")
    def transition_is_registered(self) -> ExecutionTransition:
        normal = self.to_stage in _SUCCESS_TRANSITIONS.get(self.from_stage, frozenset())
        failure = self.from_stage in _FAILURE_ORIGINS and self.to_stage in _FAILURE_STAGES
        if not (normal or failure):
            raise ValueError(
                f"unregistered condition transition: {self.from_stage.value} -> "
                f"{self.to_stage.value}"
            )
        if self.to_stage in _FAILURE_STAGES and not self.diagnostic_code:
            raise ValueError("failure transitions require a diagnostic code")
        if self.to_stage not in _FAILURE_STAGES and self.diagnostic_code is not None:
            raise ValueError("successful transitions cannot carry a failure diagnostic")
        return self


class ConditionExecutionTrace(ImmutableRecord):
    """Append-only logical trace; skipped or reordered states are rejected."""

    trace_id: str = Field(min_length=1)
    condition: ConditionName
    transitions: tuple[ExecutionTransition, ...] = ()

    @model_validator(mode="after")
    def transitions_are_contiguous_and_monotonic(self) -> ConditionExecutionTrace:
        expected = ExecutionStage.PLANNED
        previous_time: datetime | None = None
        for transition in self.transitions:
            if transition.from_stage is not expected:
                raise ValueError("condition execution trace is not contiguous")
            if previous_time is not None and transition.occurred_at < previous_time:
                raise ValueError("condition execution timestamps must be monotonic")
            expected = transition.to_stage
            previous_time = transition.occurred_at
            if expected in _FAILURE_STAGES and transition is not self.transitions[-1]:
                raise ValueError("a failure state is terminal")
        return self

    @property
    def current_stage(self) -> ExecutionStage:
        if not self.transitions:
            return ExecutionStage.PLANNED
        return self.transitions[-1].to_stage

    def advance(
        self,
        to_stage: ExecutionStage,
        *,
        occurred_at: datetime,
        artifact_hash: str | None = None,
        diagnostic_code: str | None = None,
    ) -> ConditionExecutionTrace:
        transition = ExecutionTransition(
            from_stage=self.current_stage,
            to_stage=to_stage,
            occurred_at=occurred_at,
            artifact_hash=artifact_hash,
            diagnostic_code=diagnostic_code,
        )
        return ConditionExecutionTrace(
            trace_id=self.trace_id,
            condition=self.condition,
            transitions=(*self.transitions, transition),
        )


def sealed_semantic_ids(draft: OntologyDraft) -> tuple[str, ...]:
    """Return the complete stable semantic inventory for a preontology."""

    return tuple(item.semantic_id for item in semantic_fingerprints_from_draft(draft))


def preontology_semantic_hash(
    upper_ontology: UpperOntology,
    draft: OntologyDraft,
) -> str:
    """Hash semantic preontology material, excluding query-time projection state."""

    return canonical_sha256(
        {
            "upper_ontology": upper_ontology,
            "local_schema": draft.local_schema,
            "instance_graph": draft.instance_graph,
        }
    )


class SealedPreontology(ImmutableRecord):
    """Complete C0/C1 ontology constructed and sealed before any query reveal."""

    artifact_id: str = Field(min_length=1)
    condition: Literal[ConditionName.C0_CLASSICAL_PRE, ConditionName.C1_LLM_PRE]
    snapshot_hash: Sha256Digest
    upper_ontology: UpperOntology
    draft: OntologyDraft
    construction_seal: ConstructionSeal
    seed_block: int | None = Field(default=None, ge=0)

    @model_validator(mode="after")
    def seal_covers_exact_complete_ontology(self) -> SealedPreontology:
        seal = self.construction_seal
        if seal.condition is not self.condition:
            raise ValueError("preontology and construction-seal conditions differ")
        if seal.snapshot_hash != self.snapshot_hash:
            raise ValueError("preontology and construction-seal snapshots differ")
        if seal.ontology_hash != preontology_semantic_hash(self.upper_ontology, self.draft):
            raise ValueError("construction seal does not hash the complete preontology")
        if set(seal.sealed_object_ids) != set(sealed_semantic_ids(self.draft)):
            raise ValueError("construction seal does not enumerate the complete ontology")
        if any(decision.decided_at > seal.sealed_at for decision in self.draft.decisions):
            raise ValueError("a preontology decision was made after its seal")
        if self.condition is ConditionName.C1_LLM_PRE and self.seed_block is None:
            raise ValueError("C1 sealed preontology requires its LLM seed block")
        if self.condition is ConditionName.C0_CLASSICAL_PRE and self.seed_block is not None:
            raise ValueError("deterministic C0 cannot be duplicated under LLM seed blocks")
        return self

    def as_fixed_ontology(self) -> FixedOntologyInput:
        return FixedOntologyInput(
            construction_seal=self.construction_seal,
            upper_ontology=self.upper_ontology,
            local_schema=self.draft.local_schema,
            instance_graph=self.draft.instance_graph,
        )


class FixedSelectionPreparation(ImmutableRecord):
    """Complete, same-seed C1 input inherited by ``A-FixedSelect``."""

    preparation_id: str = Field(min_length=1)
    source_c1_preontology: SealedPreontology
    seed_block: int = Field(ge=0)
    prepared_at: AwareDatetime

    @model_validator(mode="after")
    def source_is_same_seed_c1(self) -> FixedSelectionPreparation:
        source = self.source_c1_preontology
        if source.condition is not ConditionName.C1_LLM_PRE:
            raise ValueError("A-FixedSelect must inherit a C1 preontology")
        if source.seed_block != self.seed_block:
            raise ValueError("A-FixedSelect and C1 must use the same seed block")
        if self.prepared_at < source.construction_seal.sealed_at:
            raise ValueError("fixed selection cannot prepare before the C1 seal exists")
        return self


class ConditionPreparation(ImmutableRecord):
    """One-of preparation artifact returned by the common ``prepare`` operation."""

    preparation_id: str = Field(min_length=1)
    condition: ConditionName
    snapshot_hash: Sha256Digest
    completed_at: AwareDatetime
    sealed_preontology: SealedPreontology | None = None
    empty_inventory: PreQueryInventory | None = None
    fixed_selection: FixedSelectionPreparation | None = None

    @model_validator(mode="after")
    def enforce_condition_specific_preparation(self) -> ConditionPreparation:
        populated = sum(
            item is not None
            for item in (
                self.sealed_preontology,
                self.empty_inventory,
                self.fixed_selection,
            )
        )
        if populated != 1:
            raise ValueError("condition preparation requires exactly one artifact")
        if self.condition in {ConditionName.C0_CLASSICAL_PRE, ConditionName.C1_LLM_PRE}:
            preontology = self.sealed_preontology
            if preontology is None or preontology.condition is not self.condition:
                raise ValueError("C0/C1 preparation requires its own sealed preontology")
            if preontology.snapshot_hash != self.snapshot_hash:
                raise ValueError("preparation and preontology snapshots differ")
            if self.completed_at != preontology.construction_seal.sealed_at:
                raise ValueError("preparation completion must equal seal time")
        elif self.condition in {
            ConditionName.C2_LLM_QUERY,
            ConditionName.A_NO_CONTEXT,
            ConditionName.A_NO_TEMPORAL_EPISTEMIC,
            ConditionName.A_NO_RARE_GUARD,
        }:
            inventory = self.empty_inventory
            if inventory is None or inventory.condition is not self.condition:
                raise ValueError("active construction requires its empty pre-query inventory")
            if inventory.snapshot_hash != self.snapshot_hash:
                raise ValueError("preparation and empty-inventory snapshots differ")
            if self.completed_at != inventory.recorded_at:
                raise ValueError("preparation completion must equal inventory time")
        elif self.condition is ConditionName.A_FIXED_SELECT:
            fixed = self.fixed_selection
            if fixed is None:
                raise ValueError("A-FixedSelect requires complete inherited C1 preparation")
            if fixed.source_c1_preontology.snapshot_hash != self.snapshot_hash:
                raise ValueError("fixed preparation and inherited C1 snapshots differ")
            if self.completed_at != fixed.prepared_at:
                raise ValueError("fixed preparation completion timestamp differs")
        else:
            raise ValueError(f"unsupported condition preparation: {self.condition.value}")
        return self


class RunConditionConfig(ImmutableRecord):
    """Condition-neutral per-projection limits and paired stochastic identifiers."""

    config_id: str = Field(min_length=1)
    condition: ConditionName
    budgets: OutputBudgets
    maximum_input_tokens: int = Field(gt=0)
    maximum_output_tokens: int = Field(gt=0)
    repair_attempt_budget: Literal[0, 1] = 1
    seed_block: int | None = Field(default=None, ge=0)
    source_c1_seed_block: int | None = Field(default=None, ge=0)
    model_stack_hash: Sha256Digest | None = None
    decoding_family_hash: Sha256Digest | None = None
    validator_hash: Sha256Digest
    upper_ontology_hash: Sha256Digest

    @model_validator(mode="after")
    def seed_and_model_fields_match_condition(self) -> RunConditionConfig:
        llm_conditions = {
            ConditionName.C1_LLM_PRE,
            ConditionName.C2_LLM_QUERY,
            ConditionName.A_FIXED_SELECT,
            ConditionName.A_NO_CONTEXT,
            ConditionName.A_NO_TEMPORAL_EPISTEMIC,
            ConditionName.A_NO_RARE_GUARD,
        }
        if self.condition in llm_conditions:
            if self.seed_block is None:
                raise ValueError("LLM conditions require a paired seed block")
            if self.model_stack_hash is None or self.decoding_family_hash is None:
                raise ValueError("LLM conditions require model and decoding-family hashes")
        elif self.condition is ConditionName.C0_CLASSICAL_PRE:
            if self.seed_block is not None:
                raise ValueError("C0 is deterministic and must not be seed-replicated")
            if self.model_stack_hash is not None or self.decoding_family_hash is not None:
                raise ValueError("C0 must not claim an LLM model or decoding family")
        if self.condition is ConditionName.A_FIXED_SELECT:
            if self.source_c1_seed_block != self.seed_block:
                raise ValueError("A-FixedSelect must consume the same-seed C1 ontology")
        elif self.source_c1_seed_block is not None:
            raise ValueError("only A-FixedSelect can declare a source C1 seed block")
        if self.repair_attempt_budget != self.budgets.repair_attempt_budget:
            raise ValueError("run and object-budget repair allowances differ")
        return self


class ProduceInputs(ImmutableRecord):
    """Validated arguments to the common query-time ``produce`` operation."""

    preparation: ConditionPreparation
    snapshot: EvidenceSnapshot
    packet: EvidencePacket
    context: QueryContext
    upper_ontology: UpperOntology
    revisions: tuple[ModelVisibleRevision, ...] = ()
    run_config: RunConditionConfig

    @model_validator(mode="after")
    def enforce_common_query_boundary(self) -> ProduceInputs:
        if self.preparation.condition is not self.run_config.condition:
            raise ValueError("preparation and run-config conditions differ")
        if self.preparation.snapshot_hash != self.snapshot.content_hash:
            raise ValueError("preparation does not belong to the supplied snapshot")
        if self.packet.snapshot_hash != self.snapshot.content_hash:
            raise ValueError("packet does not belong to the supplied snapshot")
        if self.context.spoiler_horizon != self.snapshot.horizon:
            raise ValueError("context and snapshot horizons differ")
        if self.packet.created_at < self.context.revealed_at:
            raise ValueError("query-dependent packet must not predate query reveal")
        if self.preparation.completed_at >= self.context.revealed_at:
            raise ValueError("pre-query preparation must complete strictly before query reveal")
        if self.context.budgets != self.run_config.budgets:
            raise ValueError("context and run-config budgets differ")
        if self.upper_ontology.content_hash != self.run_config.upper_ontology_hash:
            raise ValueError("run config does not bind the supplied upper ontology")
        if self.packet.ordered_evidence_ids != self.snapshot.eligible_evidence_ids:
            raise ValueError("primary packet must contain every admissible evidence item")
        if tuple(item.evidence_id for item in self.packet.evidence) != (
            self.packet.ordered_evidence_ids
        ):
            raise ValueError("packet record order differs from its evidence-ID order")
        if self.run_config.condition is ConditionName.A_FIXED_SELECT:
            fixed = self.preparation.fixed_selection
            if fixed is None or fixed.seed_block != self.run_config.seed_block:
                raise ValueError("fixed selection preparation is not same-seed")
        if self.run_config.condition is ConditionName.C1_LLM_PRE:
            sealed = self.preparation.sealed_preontology
            if sealed is None or sealed.seed_block != self.run_config.seed_block:
                raise ValueError("C1 projection seed differs from its preontology")
        revision_sequences = tuple(item.revision.sequence for item in self.revisions)
        if revision_sequences != tuple(sorted(set(revision_sequences))):
            raise ValueError("model-visible revisions must have unique increasing sequences")
        return self


class ComparisonInputManifest(ImmutableRecord):
    """Hash-bound fairness row emitted for every condition/unit/seed."""

    condition: ConditionName
    snapshot_hash: Sha256Digest
    packet_hash: Sha256Digest
    ordered_evidence_ids: tuple[str, ...]
    horizon_hash: Sha256Digest
    context_semantics_hash: Sha256Digest
    upper_ontology_hash: Sha256Digest
    budgets: OutputBudgets
    maximum_input_tokens: int = Field(gt=0)
    maximum_output_tokens: int = Field(gt=0)
    repair_attempt_budget: Literal[0, 1]
    seed_block: int | None = Field(default=None, ge=0)
    source_c1_seed_block: int | None = Field(default=None, ge=0)
    model_stack_hash: Sha256Digest | None = None
    decoding_family_hash: Sha256Digest | None = None
    validator_hash: Sha256Digest

    @classmethod
    def from_inputs(cls, inputs: ProduceInputs) -> ComparisonInputManifest:
        config = inputs.run_config
        context_payload = inputs.context.model_dump(
            mode="python", exclude={"content_hash", "context_id", "revealed_at"}
        )
        return cls(
            condition=config.condition,
            snapshot_hash=inputs.snapshot.content_hash,
            packet_hash=inputs.packet.content_hash,
            ordered_evidence_ids=inputs.packet.ordered_evidence_ids,
            horizon_hash=inputs.context.spoiler_horizon.content_hash,
            context_semantics_hash=canonical_sha256(context_payload),
            upper_ontology_hash=inputs.upper_ontology.content_hash,
            budgets=config.budgets,
            maximum_input_tokens=config.maximum_input_tokens,
            maximum_output_tokens=config.maximum_output_tokens,
            repair_attempt_budget=config.repair_attempt_budget,
            seed_block=config.seed_block,
            source_c1_seed_block=config.source_c1_seed_block,
            model_stack_hash=config.model_stack_hash,
            decoding_family_hash=config.decoding_family_hash,
            validator_hash=config.validator_hash,
        )


def assert_comparison_fairness(manifests: Sequence[ComparisonInputManifest]) -> None:
    """Block asymmetric evidence, horizon, budgets, stack, or LLM seed pairing."""

    by_condition = {manifest.condition: manifest for manifest in manifests}
    required = {
        ConditionName.C0_CLASSICAL_PRE,
        ConditionName.C1_LLM_PRE,
        ConditionName.C2_LLM_QUERY,
        ConditionName.A_FIXED_SELECT,
    }
    if set(by_condition) != required or len(manifests) != len(required):
        raise ConditionIntegrityError("fairness comparison requires exactly C0, C1, C2, Fixed")

    ordered = tuple(by_condition[condition] for condition in sorted(required, key=str))
    packet_records = []
    for item in ordered:
        # The evidence helper compares exact packet hashes when given real packets;
        # manifest rows preserve the same exact hash and order for offline audits.
        packet_records.append((item.packet_hash, item.ordered_evidence_ids))
    if len(set(packet_records)) != 1:
        raise ConditionIntegrityError("conditions received nonidentical evidence packets")

    equal_fields = (
        "snapshot_hash",
        "horizon_hash",
        "context_semantics_hash",
        "upper_ontology_hash",
        "budgets",
        "maximum_input_tokens",
        "maximum_output_tokens",
        "repair_attempt_budget",
        "validator_hash",
    )
    for field_name in equal_fields:
        values = {canonical_sha256(getattr(item, field_name)) for item in ordered}
        if len(values) != 1:
            raise ConditionIntegrityError(f"condition fairness mismatch: {field_name}")

    c1 = by_condition[ConditionName.C1_LLM_PRE]
    c2 = by_condition[ConditionName.C2_LLM_QUERY]
    fixed = by_condition[ConditionName.A_FIXED_SELECT]
    if not (c1.seed_block == c2.seed_block == fixed.seed_block):
        raise ConditionIntegrityError("C1, C2, and Fixed must use one paired seed block")
    if fixed.source_c1_seed_block != c1.seed_block:
        raise ConditionIntegrityError("Fixed does not cite the same-seed C1 preontology")
    if len({c1.model_stack_hash, c2.model_stack_hash, fixed.model_stack_hash}) != 1:
        raise ConditionIntegrityError("LLM conditions use different model stacks")
    if len({c1.decoding_family_hash, c2.decoding_family_hash, fixed.decoding_family_hash}) != 1:
        raise ConditionIntegrityError("LLM conditions use different decoding families")


class ConditionAttemptRecord(ImmutableRecord):
    """One attempted output retained under the intention-to-treat policy."""

    attempt_id: str = Field(min_length=1)
    condition: ConditionName
    unit_id: str = Field(min_length=1)
    seed_block: int | None = Field(default=None, ge=0)
    outcome: RunOutcome
    projection: OntologyProjection | None = None
    raw_output_hash: Sha256Digest | None = None
    failure_code: str | None = None
    included_in_intention_to_treat: Literal[True] = True
    release_class: ReleaseClass

    @model_validator(mode="after")
    def outcome_preserves_failures(self) -> ConditionAttemptRecord:
        completed_failures = {
            RunOutcome.INVALID,
            RunOutcome.FAILED,
            RunOutcome.TIMED_OUT,
            RunOutcome.INTERRUPTED,
        }
        if self.outcome is RunOutcome.SUCCEEDED:
            if self.projection is None:
                raise ValueError("successful condition attempt requires a projection")
            if self.projection.condition is not self.condition:
                raise ValueError("attempt and projection conditions differ")
            if self.failure_code is not None:
                raise ValueError("successful condition attempt cannot carry failure_code")
        elif self.outcome in completed_failures:
            if not self.failure_code:
                raise ValueError("failed/invalid attempts require a recorded failure code")
            if self.projection is not None:
                raise ValueError("invalid attempt must not masquerade as a valid projection")
        elif self.projection is not None or self.failure_code is not None:
            raise ValueError("planned/running attempts cannot contain terminal results")
        return self

    def semantic_score_for_nonempty_gold(self, successful_score: float | None) -> float | None:
        """Apply the registered ITT rule without filtering invalid outputs."""

        if self.outcome in {
            RunOutcome.INVALID,
            RunOutcome.FAILED,
            RunOutcome.TIMED_OUT,
            RunOutcome.INTERRUPTED,
        }:
            return 0.0
        if self.outcome is RunOutcome.SUCCEEDED:
            if successful_score is None:
                raise ValueError("successful projection requires its computed semantic score")
            return successful_score
        return None


@runtime_checkable
class ConditionInterface(Protocol):
    """Shared two-operation interface mandated by the implementation plan."""

    condition: ConditionName

    def prepare(self, *args: object, **kwargs: object) -> ConditionPreparation:
        """Create a sealed preontology or audited empty inventory before reveal."""

    def produce(self, inputs: ProduceInputs) -> ConditionAttemptRecord:
        """Produce or faithfully retain one condition attempt after query reveal."""


def assert_packet_objects_equal(packets: Sequence[EvidencePacket]) -> None:
    """Public re-export of byte-identical evidence checking for condition runners."""

    try:
        assert_evidence_packet_equality(
            {f"condition-{index}": packet for index, packet in enumerate(packets)}
        )
    except Exception as exc:
        raise ConditionIntegrityError(str(exc)) from exc
