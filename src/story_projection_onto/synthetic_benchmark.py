"""Offline compiler for the registered synthetic narrative benchmark.

World semantics, gold, split labels, factor annotations, and review material
live in this scorer-side module.  The model worker imports only
``benchmark_runtime``.  Gold projections are compiled from typed world and
query semantics; contrast labels are a deterministic diff of the resulting
final-state decision atoms.
"""

from __future__ import annotations

import hashlib
import itertools
import json
import random
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from enum import StrEnum
from functools import partial
from pathlib import Path
from typing import Annotated, Any, Literal, Self

from pydantic import AwareDatetime, Field, model_validator

from story_projection_onto.benchmark_runtime import (
    EvidenceProjectionEquivalenceCertificate,
    GoldFirewallError,
    ModelEligibleWorldArtifact,
    NeutralEvidenceArtifact,
    QueryRevealArtifact,
    RuntimeStageKind,
    RuntimeStagingManifest,
    build_evidence_projection_equivalence_certificate,
    load_staged_neutral_evidence,
    load_staged_query,
    load_staged_world,
    model_request_payload,
    preconstruction_request_payload,
    scan_model_payload,
    verify_neutral_evidence_projection,
)
from story_projection_onto.contracts import (
    AbstractionLevel,
    BenchmarkSplit,
    ConstructionOperator,
    DiscoursePosition,
    EpistemicAttitude,
    EpistemicScope,
    EpistemicViewpoint,
    Event,
    EventCandidate,
    EvidenceRecord,
    EvidenceSnapshot,
    ExplicitValueState,
    GoldAdjudicationStatus,
    GoldAlternativeSet,
    GoldAssertionAnnotation,
    GoldCommunityAssignment,
    GoldConstraintAlternative,
    GoldConstraintOperator,
    GoldContextualProjection,
    GoldContrastDecision,
    GoldContrastDirection,
    GoldContrastInvariant,
    GoldEntityCluster,
    GoldMatchingConstraint,
    GoldRelevanceAnnotation,
    GoldReviewStatus,
    GoldTargetKind,
    HolderRelativeTime,
    Identifier,
    ImmutableRecord,
    LocalContextSchema,
    LocalPredicateDefinition,
    LocalTypeDefinition,
    MentionCandidate,
    NarrativeCommitment,
    OutputBudgets,
    ProvenanceReference,
    QualifiedAssertion,
    QueryContext,
    RelationPhraseCandidate,
    ReleaseClass,
    RevelationPosition,
    RoleBinding,
    Sha256Digest,
    SpoilerHorizon,
    StoryTime,
    TemporalClue,
    TemporalKind,
    TemporalScope,
    ValidityTime,
    canonical_json,
    canonical_sha256,
    to_model_visible_evidence,
    to_model_visible_query,
)

DEFAULT_CONFIG_PATH = Path("configs/study/synthetic_benchmark.json")
DEFAULT_OUTPUT_ROOT = Path("data/synthetic")
MODEL_VISIBLE_DIRECTORY = "model_visible"
CONDITION_INPUT_DIRECTORY = "condition_inputs"
NEUTRAL_EVIDENCE_DIRECTORY = "neutral_evidence"
SCORER_ONLY_DIRECTORY = "scorer_only"
MANIFEST_DIRECTORY = "manifests"
LINEAGE_REFRESH_PATHS = frozenset(
    {
        f"{SCORER_ONLY_DIRECTORY}/held_out/draft_seal.json",
        f"{MANIFEST_DIRECTORY}/benchmark_manifest.json",
    }
)


def _source_file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def benchmark_source_hashes() -> dict[str, str]:
    """Hash every source module that directly defines benchmark semantics."""

    package_root = Path(__file__).parent
    return {
        "synthetic_benchmark.py": _source_file_hash(Path(__file__)),
        "benchmark_runtime.py": _source_file_hash(package_root / "benchmark_runtime.py"),
        "contracts.py": _source_file_hash(package_root / "contracts.py"),
        "metrics/alignment.py": _source_file_hash(package_root / "metrics" / "alignment.py"),
    }


class Difficulty(StrEnum):
    EASY = "easy"
    MEDIUM = "medium"
    HARD = "hard"


class LensFamily(StrEnum):
    ALLEGIANCE_STATE_CHANGE = "allegiance_state_change"
    IDENTITY_KINSHIP = "identity_kinship"
    CAUSAL_CONSEQUENCE = "causal_consequence"
    EVENT_PARTICIPATION_CONFLICT = "event_participation_conflict"
    MOVEMENT_TIME = "movement_time"
    KNOWLEDGE_BELIEF = "knowledge_belief"


class StoryScopeBlock(StrEnum):
    POINT = "point"
    INTERVAL = "interval"
    THROUGH_BOUND = "through_bound"


class HorizonBlock(StrEnum):
    INTERMEDIATE = "intermediate"
    FINAL = "final"


class BenchmarkFactor(StrEnum):
    TEMPORAL_CHANGE = "temporal_change"
    IDENTITY_MERGE_SPLIT = "identity_merge_split"
    EVENT_REIFICATION = "event_reification"
    CONTEXTUAL_SCHEMA_RELATION = "contextual_schema_relation"
    ABSTRACTION_CHANGE = "abstraction_change"
    FREQUENT_IRRELEVANT = "frequent_irrelevant"
    RARE_PIVOTAL = "rare_pivotal"
    HOLDER_BELIEF_REPORT = "holder_belief_report"
    INCOMPLETE_CONFLICTING_EVIDENCE = "incomplete_conflicting_evidence"
    MEANINGFUL_COMMUNITIES = "meaningful_communities"
    FIXED_ONTOLOGY_FRIENDLY = "fixed_ontology_friendly"
    NULL_NONSELECTION_CASE = "null_nonselection_case"


class NeutralCommitment(StrEnum):
    WORLD = "world"
    REPORTED = "reported"
    DENIED = "denied"
    CONTESTED = "contested"


class NeutralFactRole(StrEnum):
    CORE = "core"
    SUPPORT = "support"
    DISTRACTOR = "distractor"
    RARE_PIVOTAL = "rare_pivotal"
    REPORT = "report"


class ContrastFamily(StrEnum):
    MERGE_SPLIT = "merge_split"
    EVENT_REIFICATION = "event_reification"
    SCHEMA_RELATION = "schema_relation"
    ABSTRACTION = "abstraction"
    TEMPORAL_QUALIFICATION = "temporal_qualification"
    EPISTEMIC_QUALIFICATION = "epistemic_qualification"


class RareImpactKind(StrEnum):
    CAUSAL = "causal"
    TEMPORAL = "temporal"
    STATE = "state"
    COMMUNITY = "community"


class CompilerPolicy(StrEnum):
    QUERY_DEPENDENT = "query_dependent"
    FIXED_REFERENCE = "fixed_reference"


class SeedPurpose(StrEnum):
    WORLD = "world"
    NARRATIVE = "narrative"
    QUERY = "query"
    PARAPHRASE = "paraphrase"
    LLM_BLOCK_1 = "llm_block_1"
    LLM_BLOCK_2 = "llm_block_2"
    LAYOUT = "layout"
    LEIDEN = "leiden"
    BOOTSTRAP = "bootstrap"
    REVIEW_SELECTION = "review_selection"
    MUTATION = "mutation"
    A_NO_CONTEXT = "a_no_context"


class SeedEntry(ImmutableRecord):
    purpose: SeedPurpose
    namespace: str
    seed: Annotated[int, Field(ge=0, lt=2**63)]
    derivation_material_hash: Sha256Digest


class SeedDerivationManifest(ImmutableRecord):
    root_seed: Annotated[int, Field(ge=0)]
    algorithm: Literal["sha256-first-63-bits-v3"] = "sha256-first-63-bits-v3"
    entries: tuple[SeedEntry, ...]

    @model_validator(mode="after")
    def all_purposes_once(self) -> Self:
        purposes = tuple(item.purpose for item in self.entries)
        if len(set(purposes)) != len(purposes) or set(purposes) != set(SeedPurpose):
            raise ValueError("seed manifest must derive each registered purpose exactly once")
        return self


def derive_seed(root_seed: int, namespace: str) -> int:
    material = f"story-projection-onto\0{root_seed}\0{namespace}".encode()
    return int.from_bytes(hashlib.sha256(material).digest()[:8], "big") & ((1 << 63) - 1)


def build_seed_manifest(root_seed: int) -> SeedDerivationManifest:
    entries = []
    for purpose in SeedPurpose:
        namespace = f"synthetic-v3/{purpose.value}"
        material = f"story-projection-onto\0{root_seed}\0{namespace}".encode()
        entries.append(
            SeedEntry(
                purpose=purpose,
                namespace=namespace,
                seed=derive_seed(root_seed, namespace),
                derivation_material_hash=hashlib.sha256(material).hexdigest(),
            )
        )
    return SeedDerivationManifest(root_seed=root_seed, entries=tuple(entries))


def seed_for(manifest: SeedDerivationManifest, purpose: SeedPurpose) -> int:
    return next(item.seed for item in manifest.entries if item.purpose is purpose)


class BenchmarkConfiguration(ImmutableRecord):
    root_seed: Annotated[int, Field(ge=0)]
    frozen_at: AwareDatetime
    development_world_count: Literal[4]
    held_out_world_count: Literal[12]
    contexts_per_world: Literal[3]
    paraphrase_context_count: Literal[12]
    independent_review_world_count: Literal[3]
    matcher_revision: str
    generator_revision: str
    index_revision: str
    development_surface_renderers: tuple[str, ...]
    held_out_surface_renderers: tuple[str, ...]
    query_renderers: tuple[str, str]
    gold_node_targets: dict[Difficulty, int]
    minimum_factor_world_count: Literal[3]
    minimum_temporal_change_world_count: Literal[6]
    minimum_rare_ablation_context_count: Literal[8]
    minimum_temporal_epistemic_ablation_context_count: Literal[8]
    minimum_community_world_count: Literal[8]

    @model_validator(mode="after")
    def fixed_registered_shape(self) -> Self:
        if set(self.development_surface_renderers) & set(self.held_out_surface_renderers):
            raise ValueError("held-out renderers must be absent from development")
        if self.gold_node_targets != {
            Difficulty.EASY: 10,
            Difficulty.MEDIUM: 15,
            Difficulty.HARD: 20,
        }:
            raise ValueError("registered output budgets must be 10/15/20")
        return self


def load_benchmark_configuration(path: Path = DEFAULT_CONFIG_PATH) -> BenchmarkConfiguration:
    return BenchmarkConfiguration.model_validate_json(path.read_text(encoding="utf-8"))


def _timestamp(config: BenchmarkConfiguration, seconds: int = 0) -> datetime:
    return config.frozen_at.astimezone(UTC) + timedelta(seconds=seconds)


def _opaque(prefix: str, *parts: object) -> str:
    material = "\0".join(str(part) for part in parts).encode()
    return f"{prefix}_{hashlib.sha256(material).hexdigest()[:20]}"


def _without_hashes(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {
            str(key): _without_hashes(child)
            for key, child in value.items()
            if key != "content_hash"
        }
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [_without_hashes(child) for child in value]
    return value


class WorldPersona(ImmutableRecord):
    persona_id: Identifier
    name: str
    aliases: tuple[str, ...]
    home_collective_id: Identifier
    office_title: str | None = None
    predecessor_persona_id: Identifier | None = None


class WorldCollective(ImmutableRecord):
    collective_id: Identifier
    name: str


class WorldPlace(ImmutableRecord):
    place_id: Identifier
    name: str


class WorldEventSpec(ImmutableRecord):
    event_ref: Identifier
    label: str
    participant_refs: tuple[Identifier, ...]
    place_ref: Identifier
    story_position: Annotated[int, Field(ge=0)]
    duration: Annotated[int, Field(gt=0)]
    causal_predecessor_ref: Identifier | None = None
    focal: bool = False


class WorldFactSpec(ImmutableRecord):
    fact_id: Identifier
    subject_ref: Identifier
    relation: Identifier
    object_ref: Identifier
    story_position: Annotated[int, Field(ge=0)]
    validity_end: Annotated[int, Field(ge=0)] | None = None
    disclosure_order: Annotated[int, Field(ge=0)]
    role: NeutralFactRole
    commitment: NeutralCommitment
    holder_ref: Identifier | None = None
    event_ref: Identifier | None = None
    relevant_lenses: tuple[LensFamily, ...] = ()
    repetition_count: Annotated[int, Field(ge=1, le=4)] = 1
    conflict_group: Identifier | None = None

    @model_validator(mode="after")
    def coherent_fact(self) -> Self:
        if (
            self.commitment in {NeutralCommitment.REPORTED, NeutralCommitment.DENIED}
            and self.holder_ref is None
        ):
            raise ValueError("reported and denied facts require a holder")
        if self.role is NeutralFactRole.RARE_PIVOTAL and self.repetition_count != 1:
            raise ValueError("rare-pivotal facts have exactly one witness")
        if self.validity_end is not None and self.validity_end < self.story_position:
            raise ValueError("fact validity cannot end before its story position")
        return self


class WorldSpec(ImmutableRecord):
    """Representation-neutral scorer semantics, never staged to a model worker."""

    world_id: Identifier
    structure_template_id: Identifier
    split: BenchmarkSplit
    difficulty: Difficulty
    theme: str
    personas: tuple[WorldPersona, ...]
    collectives: tuple[WorldCollective, ...]
    places: tuple[WorldPlace, ...]
    events: tuple[WorldEventSpec, ...]
    facts: tuple[WorldFactSpec, ...]
    temporal_precedence: tuple[tuple[Identifier, Identifier], ...]
    causal_dependencies: tuple[tuple[Identifier, Identifier], ...]
    rare_impact_kind: RareImpactKind
    factors: frozenset[BenchmarkFactor]
    contrast_family: ContrastFamily
    node_budget: Literal[10, 15, 20]
    horizon_block: HorizonBlock
    surface_renderer: str
    rare_deletion_mutation: bool = False

    @model_validator(mode="after")
    def validate_semantic_references(self) -> Self:
        entity_refs = {
            *(item.persona_id for item in self.personas),
            *(item.collective_id for item in self.collectives),
            *(item.place_id for item in self.places),
        }
        event_refs = {item.event_ref for item in self.events}
        fact_ids = {item.fact_id for item in self.facts}
        if len(fact_ids) != len(self.facts) or len(event_refs) != len(self.events):
            raise ValueError("formal fact and event IDs must be unique")
        for persona in self.personas:
            if persona.home_collective_id not in {item.collective_id for item in self.collectives}:
                raise ValueError("persona home collective is absent")
            if persona.predecessor_persona_id not in entity_refs | {None}:
                raise ValueError("office predecessor is absent")
        for event in self.events:
            if not set(event.participant_refs).issubset(entity_refs):
                raise ValueError("event participant is absent")
            if event.place_ref not in entity_refs:
                raise ValueError("event place is absent")
            if event.causal_predecessor_ref not in event_refs | {None}:
                raise ValueError("event predecessor is absent")
        for fact in self.facts:
            if fact.subject_ref not in entity_refs:
                raise ValueError("fact subject is absent")
            if fact.object_ref not in entity_refs | event_refs:
                raise ValueError("fact object is absent")
            if fact.event_ref not in event_refs | {None}:
                raise ValueError("fact event is absent")
            if fact.holder_ref not in entity_refs | {None}:
                raise ValueError("fact holder is absent")
        if any(
            left not in event_refs or right not in event_refs
            for left, right in self.temporal_precedence
        ):
            raise ValueError("temporal precedence must name world events")
        missing_dependency_refs = {
            ref for edge in self.causal_dependencies for ref in edge if ref not in fact_ids
        }
        if missing_dependency_refs and not self.rare_deletion_mutation:
            raise ValueError("causal dependencies must name world facts")
        if sum(event.focal for event in self.events) != 1:
            raise ValueError("each world requires exactly one focal event")
        rare = [item for item in self.facts if item.role is NeutralFactRole.RARE_PIVOTAL]
        if len(rare) != (0 if self.rare_deletion_mutation else 1):
            raise ValueError("each world requires exactly one rare-pivotal fact")
        return self


class QueryAssignment(ImmutableRecord):
    world_id: Identifier
    lenses: tuple[LensFamily, LensFamily, LensFamily]
    story_scopes: tuple[StoryScopeBlock, StoryScopeBlock, StoryScopeBlock]
    abstractions: tuple[AbstractionLevel, AbstractionLevel, AbstractionLevel]
    viewpoint_context_index: Annotated[int, Field(ge=0, le=2)]
    viewpoint_holder_persona_index: Annotated[int, Field(ge=0)] = 1
    node_budget: Literal[10, 15, 20]
    horizon_block: HorizonBlock
    contrast_family: ContrastFamily
    compiler_policies: tuple[CompilerPolicy, CompilerPolicy, CompilerPolicy]
    null_nonselection_context_index: Annotated[int, Field(ge=0, le=2)] | None = None

    @model_validator(mode="after")
    def blocked_values(self) -> Self:
        if len(set(self.lenses)) != 3:
            raise ValueError("each world must receive three distinct lenses")
        if set(self.story_scopes) != set(StoryScopeBlock):
            raise ValueError("each world must receive one of every story-scope block")
        if set(self.abstractions) != set(AbstractionLevel):
            raise ValueError("each world must receive one of every abstraction block")
        return self


class SamplingRejection(ImmutableRecord):
    attempt: Annotated[int, Field(ge=1)]
    reason: str
    proposed_lenses: tuple[LensFamily, ...]


class ConditionalAllocationRecord(ImmutableRecord):
    """Finite conditional choice set and the seeded draw for one world."""

    world_id: Identifier
    selection_seed: Annotated[int, Field(ge=0)]
    eligible_lens_orders: tuple[tuple[LensFamily, LensFamily, LensFamily], ...]
    eligible_scope_orders: tuple[tuple[StoryScopeBlock, StoryScopeBlock, StoryScopeBlock], ...]
    eligible_abstraction_orders: tuple[
        tuple[AbstractionLevel, AbstractionLevel, AbstractionLevel], ...
    ]
    eligible_viewpoint_indices: tuple[Annotated[int, Field(ge=0, le=2)], ...]
    selected_lens_order: tuple[LensFamily, LensFamily, LensFamily]
    selected_scope_order: tuple[StoryScopeBlock, StoryScopeBlock, StoryScopeBlock]
    selected_abstraction_order: tuple[AbstractionLevel, AbstractionLevel, AbstractionLevel]
    selected_viewpoint_index: Annotated[int, Field(ge=0, le=2)]
    conditional_probability: str

    @model_validator(mode="after")
    def selected_values_are_eligible(self) -> Self:
        if self.selected_lens_order not in self.eligible_lens_orders:
            raise ValueError("selected lens order is outside its finite conditional set")
        if self.selected_scope_order not in self.eligible_scope_orders:
            raise ValueError("selected scope order is outside its finite conditional set")
        if self.selected_abstraction_order not in self.eligible_abstraction_orders:
            raise ValueError("selected abstraction order is outside its finite conditional set")
        if self.selected_viewpoint_index not in self.eligible_viewpoint_indices:
            raise ValueError("selected viewpoint position is outside its finite conditional set")
        return self


class QuerySamplingAudit(ImmutableRecord):
    root_query_seed: Annotated[int, Field(ge=0)]
    lens_candidate_multiset: tuple[LensFamily, ...]
    eligible_story_scopes: tuple[StoryScopeBlock, ...]
    eligible_abstractions: tuple[AbstractionLevel, ...]
    eligible_viewpoint_indices: tuple[int, ...]
    eligible_horizons: tuple[HorizonBlock, ...]
    conditional_lens_probability: str
    difficulty_lens_block_permutations: dict[Difficulty, tuple[int, int, int, int]]
    conditional_allocations: tuple[ConditionalAllocationRecord, ...]
    assignments: tuple[QueryAssignment, ...]
    rejections: tuple[SamplingRejection, ...] = ()


class EvidenceBinding(ImmutableRecord):
    candidate_id: Identifier
    evidence_id: Identifier
    semantic_ref: Identifier
    role: Literal["subject", "object", "holder", "office_title", "event", "time"]


@dataclass(frozen=True)
class NarrativeProducts:
    snapshot: EvidenceSnapshot
    evidence: tuple[EvidenceRecord, ...]
    fact_evidence_ids: Mapping[str, tuple[str, ...]]
    causal_evidence_ids: Mapping[tuple[str, str], tuple[str, ...]]
    temporal_evidence_ids: Mapping[tuple[str, str], tuple[str, ...]]
    revelation_order_by_evidence_id: Mapping[str, int]
    ref_mention_ids: Mapping[str, tuple[str, ...]]
    office_title_mentions_by_holder: Mapping[str, tuple[str, ...]]
    bindings: tuple[EvidenceBinding, ...]


class SemanticAtom(ImmutableRecord):
    """Canonical final-state ontology decision on a stable semantic slot."""

    slot_key: Identifier
    operator: ConstructionOperator
    signature: str
    anchor_ids: tuple[Identifier, ...]
    evidence_ids: tuple[Identifier, ...]
    object_ids: tuple[Identifier, ...] = ()

    @model_validator(mode="after")
    def grounded_nonselection_atom(self) -> Self:
        if self.operator in {
            ConstructionOperator.SELECTION,
            ConstructionOperator.COMPRESSION,
            ConstructionOperator.SUPPORTED_DESCRIPTION,
            ConstructionOperator.INCLUDE_EXCLUDE,
        }:
            raise ValueError("semantic atom must encode a nonselection decision")
        if not self.anchor_ids or not self.evidence_ids:
            raise ValueError("semantic atoms require evidence-resolving anchors")
        if "=>" in self.signature:
            raise ValueError("semantic atom signatures reserve '=>' for substitutions")
        return self


class AnswerSignature(ImmutableRecord):
    answer_facts: tuple[str, ...]
    causal_paths: tuple[str, ...]
    temporal_order: tuple[str, ...]
    consequential_states: tuple[str, ...]
    community_partition: tuple[str, ...]


class CommunityRationale(ImmutableRecord):
    anchor_id: Identifier
    community_signature: str
    semantic_basis: Literal[
        "home_collective", "defection", "event_participation", "place_use", "collective_self"
    ]
    evidence_ids: tuple[Identifier, ...]


class AlternativeField(StrEnum):
    JOINT_REPRESENTATION = "joint_representation_signature"


class AlternativeCandidateView(ImmutableRecord):
    entity_partition_signature: str
    assertion_direction_signature: str
    temporal_signature: str
    epistemic_signature: str
    assertion_target_id: Identifier
    assertion_signature: str
    assertion_evidence_signature: str
    joint_representation_signature: str

    @model_validator(mode="after")
    def joint_signature_matches_components(self) -> Self:
        expected = {
            "entity_partition_signature": self.entity_partition_signature,
            "assertion_direction_signature": self.assertion_direction_signature,
            "temporal_signature": self.temporal_signature,
            "epistemic_signature": self.epistemic_signature,
            "assertion_target_id": self.assertion_target_id,
            "assertion_signature": self.assertion_signature,
            "assertion_evidence_signature": self.assertion_evidence_signature,
        }
        if self.joint_representation_signature != canonical_json(expected):
            raise ValueError("joint alternative signature disagrees with candidate components")
        return self


class ExecutableAlternativeValues(ImmutableRecord):
    """Typed primary/alternate targets encoded by one gold alternative."""

    assertion_target_id: Identifier
    primary_assertion_signature: str
    alternate_assertion_signature: str
    assertion_evidence_signature: str
    primary_joint_representation_signature: str
    alternate_joint_representation_signature: str


class AlternativeEvaluation(ImmutableRecord):
    matched: bool
    matched_alternative_id: Identifier | None = None
    failed_constraints: tuple[str, ...] = ()


class NullCaseProof(ImmutableRecord):
    world_id: Identifier
    base_query_id: Identifier
    perturbation: Literal["wording-only-under-fixed-reference"]
    base_nonselection_signature: Sha256Digest
    perturbed_nonselection_signature: Sha256Digest

    @model_validator(mode="after")
    def signatures_equal(self) -> Self:
        if self.base_nonselection_signature != self.perturbed_nonselection_signature:
            raise ValueError("registered null case changed its fixed nonselection semantics")
        return self


class MutationKind(StrEnum):
    SOURCE_FACT = "source_fact"
    QUERY_CONTEXT = "query_context"
    SPOILER_HORIZON = "spoiler_horizon"
    RARE_DELETION = "rare_deletion"


class MutationExpectation(ImmutableRecord):
    mutation_id: Identifier
    kind: MutationKind
    world_id: Identifier
    target_id: Identifier
    removed_signatures: tuple[str, ...]
    added_signatures: tuple[str, ...]
    unaffected_signature_count: Annotated[int, Field(ge=0)]


class RareDeletionProof(ImmutableRecord):
    world_id: Identifier
    query_id: Identifier
    impact_kind: RareImpactKind
    before_answer_hash: Sha256Digest
    after_answer_hash: Sha256Digest
    before_regenerated_bundle_hash: Sha256Digest
    after_regenerated_bundle_hash: Sha256Digest
    changed_components: tuple[str, ...]

    @model_validator(mode="after")
    def answer_and_registered_component_change(self) -> Self:
        if self.before_answer_hash == self.after_answer_hash:
            raise ValueError("rare deletion must change the answer signature")
        if self.before_regenerated_bundle_hash == self.after_regenerated_bundle_hash:
            raise ValueError("rare deletion must change the independently regenerated bundle")
        if (
            "answer_facts" not in self.changed_components
            or self.impact_kind.value not in self.changed_components
        ):
            raise ValueError("rare deletion must change answer plus its registered structural role")
        return self


class RareSupportPathStep(ImmutableRecord):
    source_fact_id: Identifier
    target_fact_id: Identifier
    source_assertion_id: Identifier
    causal_assertion_id: Identifier
    target_assertion_id: Identifier
    evidence_ids: tuple[Identifier, ...]


class RareSupportPath(ImmutableRecord):
    path_id: Identifier
    query_id: Identifier
    rare_assertion_id: Identifier
    outcome_assertion_id: Identifier
    impact_kind: RareImpactKind
    steps: tuple[RareSupportPathStep, RareSupportPathStep]

    @model_validator(mode="after")
    def directed_chain_is_contiguous(self) -> Self:
        first, second = self.steps
        if first.target_fact_id != second.source_fact_id:
            raise ValueError("rare support path dependency steps must be directed and contiguous")
        if first.source_assertion_id != self.rare_assertion_id:
            raise ValueError("rare support path must begin at its rare assertion")
        if second.target_assertion_id != self.outcome_assertion_id:
            raise ValueError("rare support path must end at its outcome assertion")
        return self


class MutationManifest(ImmutableRecord):
    mutation_seed: Annotated[int, Field(ge=0)]
    expectations: tuple[MutationExpectation, ...]
    rare_deletion_proofs: tuple[RareDeletionProof, ...]


class ParaphrasePair(ImmutableRecord):
    world_id: Identifier
    base_context_id: Identifier
    base_context_hash: Sha256Digest
    paraphrase_context: QueryContext
    base_renderer_id: Identifier
    paraphrase_renderer_id: Identifier


class ParaphraseSelectionManifest(ImmutableRecord):
    selection_seed: Annotated[int, Field(ge=0)]
    selection_rule: Literal["one-per-world-exactly-two-per-lens-four-per-difficulty"] = (
        "one-per-world-exactly-two-per-lens-four-per-difficulty"
    )
    pairs: tuple[ParaphrasePair, ...]


class ANoContextSelectionEntry(ImmutableRecord):
    world_id: Identifier
    query_id: Identifier
    difficulty: Difficulty
    lens: LensFamily


class ANoContextSelectionManifest(ImmutableRecord):
    selection_seed: Annotated[int, Field(ge=0)]
    frozen_before_outputs: Literal[True] = True
    rule: Literal["one-per-world-two-per-lens-four-per-difficulty"] = (
        "one-per-world-two-per-lens-four-per-difficulty"
    )
    entries: tuple[ANoContextSelectionEntry, ...]

    @model_validator(mode="after")
    def exact_balance(self) -> Self:
        if len(self.entries) != 12 or len({item.world_id for item in self.entries}) != 12:
            raise ValueError("A-NoContext must select one context from each of twelve worlds")
        if Counter(item.lens for item in self.entries) != Counter({item: 2 for item in LensFamily}):
            raise ValueError("A-NoContext must select exactly two contexts per lens")
        if Counter(item.difficulty for item in self.entries) != Counter(
            {item: 4 for item in Difficulty}
        ):
            raise ValueError("A-NoContext must select exactly four contexts per difficulty")
        return self


_THEMES = (
    "Cedar Harbor Accord",
    "Glass Orchard Succession",
    "Ember Bridge Evacuation",
    "Silver Fen Compact",
    "Northwind Archive Dispute",
    "Lantern Coast Blockade",
    "Hollow Bell Expedition",
    "Red Meadow Arbitration",
    "Quartz River Crossing",
    "Ash Observatory Schism",
    "Copper Vale Relief",
    "Blue Quarry Inquiry",
    "Winter Market Election",
    "Sunken Road Recovery",
    "White Rook Negotiation",
    "Moss Tower Inquiry",
)
_NAMES = ("Ari", "Bela", "Cyra", "Doran", "Evin", "Fara", "Galen", "Hesta", "Ilan", "Jora", "Kelan")


@dataclass(frozen=True)
class WorldStructureTemplate:
    """A representation-neutral relational skeleton selected before rendering."""

    template_id: str
    membership_offset: int
    participant_offset: int
    participant_stride: int
    participant_gap: int
    event_positions: tuple[int, int, int, int]
    bridge_subjects: tuple[int, int]
    outcome_collective_index: int
    rare_subject_index: int
    report_subject_index: int
    distractor_subject_index: int


_DEVELOPMENT_STRUCTURES = (
    WorldStructureTemplate("dev-structure-a", 0, 0, 1, 2, (6, 3, 8, 5), (0, 2), 0, 4, 2, 4),
    WorldStructureTemplate("dev-structure-b", 1, 1, 1, 3, (5, 2, 9, 7), (1, 3), 1, 3, 2, 4),
    WorldStructureTemplate("dev-structure-c", 0, 2, 2, 1, (7, 4, 9, 2), (2, 4), 0, 3, 1, 2),
    WorldStructureTemplate("dev-structure-d", 1, 3, 2, 2, (4, 8, 2, 7), (3, 0), 1, 4, 2, 1),
)

# These twelve structures are scorer-only and have no development counterpart.
# Each row changes several graph invariants, not merely names or prose style.
_HELD_OUT_STRUCTURES = (
    WorldStructureTemplate("held-structure-01", 1, 0, 2, 1, (4, 7, 2, 9), (2, 0), 1, 3, 4, 1),
    WorldStructureTemplate("held-structure-02", 0, 3, 1, 2, (8, 3, 6, 1), (4, 1), 0, 2, 3, 0),
    WorldStructureTemplate("held-structure-03", 1, 1, 3, 1, (5, 9, 2, 7), (3, 2), 1, 4, 0, 2),
    WorldStructureTemplate("held-structure-04", 0, 4, 2, 3, (9, 4, 7, 2), (1, 4), 0, 3, 2, 4),
    WorldStructureTemplate("held-structure-05", 1, 2, 1, 1, (3, 8, 5, 9), (0, 3), 1, 2, 4, 1),
    WorldStructureTemplate("held-structure-06", 0, 0, 3, 2, (7, 2, 9, 4), (4, 2), 0, 3, 1, 0),
    WorldStructureTemplate("held-structure-07", 1, 3, 2, 2, (2, 6, 9, 4), (2, 4), 1, 3, 0, 2),
    WorldStructureTemplate("held-structure-08", 0, 1, 1, 3, (6, 9, 3, 7), (3, 1), 0, 4, 2, 1),
    WorldStructureTemplate("held-structure-09", 1, 4, 3, 1, (8, 2, 5, 9), (1, 3), 1, 2, 0, 4),
    WorldStructureTemplate("held-structure-10", 0, 2, 2, 1, (4, 9, 6, 1), (4, 0), 0, 3, 1, 2),
    WorldStructureTemplate("held-structure-11", 1, 0, 1, 2, (9, 5, 2, 7), (3, 4), 1, 2, 1, 0),
    WorldStructureTemplate("held-structure-12", 0, 3, 3, 2, (5, 1, 8, 6), (2, 1), 0, 4, 3, 2),
)

_CONTRAST_SCHEDULE = (
    ContrastFamily.MERGE_SPLIT,
    ContrastFamily.EVENT_REIFICATION,
    ContrastFamily.SCHEMA_RELATION,
    ContrastFamily.TEMPORAL_QUALIFICATION,
    ContrastFamily.EPISTEMIC_QUALIFICATION,
    ContrastFamily.ABSTRACTION,
    ContrastFamily.MERGE_SPLIT,
    ContrastFamily.EVENT_REIFICATION,
    ContrastFamily.SCHEMA_RELATION,
    ContrastFamily.TEMPORAL_QUALIFICATION,
    ContrastFamily.EPISTEMIC_QUALIFICATION,
    ContrastFamily.ABSTRACTION,
)


def _difficulty(split: BenchmarkSplit, ordinal: int) -> Difficulty:
    if split is BenchmarkSplit.HELD_OUT:
        return (Difficulty.EASY, Difficulty.MEDIUM, Difficulty.HARD)[(ordinal - 1) // 4]
    return (Difficulty.EASY, Difficulty.EASY, Difficulty.MEDIUM, Difficulty.HARD)[ordinal - 1]


def _entity_count(difficulty: Difficulty) -> int:
    return {Difficulty.EASY: 9, Difficulty.MEDIUM: 12, Difficulty.HARD: 15}[difficulty]


def _event_count(difficulty: Difficulty) -> int:
    return {Difficulty.EASY: 2, Difficulty.MEDIUM: 3, Difficulty.HARD: 4}[difficulty]


def _factors(ordinal: int) -> frozenset[BenchmarkFactor]:
    values = {
        BenchmarkFactor.FREQUENT_IRRELEVANT,
        BenchmarkFactor.RARE_PIVOTAL,
    }
    if ordinal % 2 == 0:
        values.add(BenchmarkFactor.TEMPORAL_CHANGE)
        values.add(BenchmarkFactor.EVENT_REIFICATION)
    if ordinal % 2 == 1:
        values.add(BenchmarkFactor.IDENTITY_MERGE_SPLIT)
    if ordinal in {1, 4, 7, 10}:
        values.add(BenchmarkFactor.CONTEXTUAL_SCHEMA_RELATION)
    if ordinal in {2, 5, 8, 11}:
        values.add(BenchmarkFactor.ABSTRACTION_CHANGE)
    if ordinal in {3, 5, 6, 9, 11, 12}:
        values.add(BenchmarkFactor.HOLDER_BELIEF_REPORT)
    if ordinal <= 8:
        values.add(BenchmarkFactor.MEANINGFUL_COMMUNITIES)
    if ordinal in {3, 5, 6, 9, 11, 12}:
        values.add(BenchmarkFactor.INCOMPLETE_CONFLICTING_EVIDENCE)
    if ordinal in {1, 4, 7, 10}:
        values.add(BenchmarkFactor.FIXED_ONTOLOGY_FRIENDLY)
    if ordinal in {4, 8, 12}:
        values.add(BenchmarkFactor.NULL_NONSELECTION_CASE)
    return frozenset(values)


def _build_world_spec(
    config: BenchmarkConfiguration,
    split: BenchmarkSplit,
    ordinal: int,
    horizon: HorizonBlock,
    contrast: ContrastFamily,
    renderer: str,
    structure: WorldStructureTemplate,
) -> WorldSpec:
    world_id = f"syn-{'dev' if split is BenchmarkSplit.DEVELOPMENT else 'test'}-{ordinal:02d}"
    difficulty = _difficulty(split, ordinal)
    theme = _THEMES[(ordinal - 1) + (0 if split is BenchmarkSplit.DEVELOPMENT else 4)]
    stem = theme.split()[0]
    collective_ids = (f"{world_id}.collective.0", f"{world_id}.collective.1")
    collectives = (
        WorldCollective(collective_id=collective_ids[0], name=f"{stem} Guild"),
        WorldCollective(collective_id=collective_ids[1], name=f"{stem} Circle"),
    )
    places = (
        WorldPlace(place_id=f"{world_id}.place.0", name=f"{stem} Gate"),
        WorldPlace(place_id=f"{world_id}.place.1", name=f"{stem} Quay"),
    )
    persona_count = _entity_count(difficulty) - 4
    personas = []
    for index in range(persona_count):
        name = f"{_NAMES[((ordinal * 2) + index) % len(_NAMES)]} {stem}"
        personas.append(
            WorldPersona(
                persona_id=f"{world_id}.persona.{index}",
                name=name,
                aliases=(name, name.split()[0]),
                home_collective_id=collective_ids[(index + structure.membership_offset) % 2],
                office_title=(f"{stem} Warden" if index in {0, 1} else None),
                predecessor_persona_id=(f"{world_id}.persona.0" if index == 1 else None),
            )
        )
    event_count = _event_count(difficulty)
    positions = structure.event_positions
    chronological_indices = sorted(range(event_count), key=lambda index: positions[index])
    predecessor_by_index = {
        current: previous for previous, current in itertools.pairwise(chronological_indices)
    }
    events = []
    for index in range(event_count):
        events.append(
            WorldEventSpec(
                event_ref=f"{world_id}.event.{index}",
                label=f"{stem} {'Confrontation' if index == 0 else f'Turn {index}'}",
                participant_refs=(
                    personas[
                        (structure.participant_offset + index * structure.participant_stride)
                        % len(personas)
                    ].persona_id,
                    personas[
                        (
                            structure.participant_offset
                            + index * structure.participant_stride
                            + structure.participant_gap
                        )
                        % len(personas)
                    ].persona_id,
                ),
                place_ref=places[index % 2].place_id,
                story_position=positions[index],
                duration=1 if index != 2 else 2,
                causal_predecessor_ref=(
                    f"{world_id}.event.{predecessor_by_index[index]}"
                    if index in predecessor_by_index
                    else None
                ),
                focal=index == 0,
            )
        )
    facts: list[WorldFactSpec] = []
    for index, persona in enumerate(personas):
        facts.append(
            WorldFactSpec(
                fact_id=f"{world_id}.fact.member.{index}",
                subject_ref=persona.persona_id,
                relation="member_of",
                object_ref=persona.home_collective_id,
                story_position=1,
                validity_end=None,
                disclosure_order=1 + index,
                role=NeutralFactRole.SUPPORT,
                commitment=NeutralCommitment.WORLD,
                relevant_lenses=(
                    LensFamily.ALLEGIANCE_STATE_CHANGE,
                    LensFamily.IDENTITY_KINSHIP,
                ),
            )
        )
    office_facts = (
        WorldFactSpec(
            fact_id=f"{world_id}.fact.office.predecessor",
            subject_ref=personas[0].persona_id,
            relation="holds_office",
            object_ref=collective_ids[0],
            story_position=1,
            validity_end=4,
            disclosure_order=20,
            role=NeutralFactRole.CORE,
            commitment=NeutralCommitment.WORLD,
            relevant_lenses=(LensFamily.IDENTITY_KINSHIP, LensFamily.MOVEMENT_TIME),
        ),
        WorldFactSpec(
            fact_id=f"{world_id}.fact.office.successor",
            subject_ref=personas[1].persona_id,
            relation="holds_office",
            object_ref=collective_ids[0],
            story_position=5,
            validity_end=None,
            disclosure_order=21,
            role=NeutralFactRole.CORE,
            commitment=NeutralCommitment.WORLD,
            relevant_lenses=(LensFamily.IDENTITY_KINSHIP, LensFamily.MOVEMENT_TIME),
        ),
        WorldFactSpec(
            fact_id=f"{world_id}.fact.office.succession",
            subject_ref=personas[1].persona_id,
            relation="succeeds",
            object_ref=personas[0].persona_id,
            story_position=5,
            validity_end=None,
            disclosure_order=22,
            role=NeutralFactRole.CORE,
            commitment=NeutralCommitment.WORLD,
            relevant_lenses=(LensFamily.IDENTITY_KINSHIP, LensFamily.CAUSAL_CONSEQUENCE),
        ),
    )
    facts.extend(office_facts)
    for event_index, event in enumerate(events):
        for participant_index, participant in enumerate(event.participant_refs):
            facts.append(
                WorldFactSpec(
                    fact_id=f"{world_id}.fact.event.{event_index}.participant.{participant_index}",
                    subject_ref=participant,
                    relation="participates_in",
                    object_ref=event.event_ref,
                    story_position=event.story_position,
                    validity_end=event.story_position + event.duration,
                    disclosure_order=30 + event_index * 3 + participant_index,
                    role=NeutralFactRole.CORE,
                    commitment=NeutralCommitment.WORLD,
                    event_ref=event.event_ref,
                    relevant_lenses=(
                        LensFamily.EVENT_PARTICIPATION_CONFLICT,
                        LensFamily.CAUSAL_CONSEQUENCE,
                        LensFamily.MOVEMENT_TIME,
                    ),
                )
            )
        facts.append(
            WorldFactSpec(
                fact_id=f"{world_id}.fact.event.{event_index}.place",
                subject_ref=event.participant_refs[0],
                relation="acts_at",
                object_ref=event.place_ref,
                story_position=event.story_position,
                validity_end=event.story_position + event.duration,
                disclosure_order=32 + event_index * 3,
                role=NeutralFactRole.SUPPORT,
                commitment=NeutralCommitment.WORLD,
                event_ref=event.event_ref,
                relevant_lenses=(
                    LensFamily.EVENT_PARTICIPATION_CONFLICT,
                    LensFamily.MOVEMENT_TIME,
                ),
            )
        )
    support_fact = WorldFactSpec(
        fact_id=f"{world_id}.fact.causal.bridge",
        subject_ref=personas[structure.bridge_subjects[0]].persona_id,
        relation="coordinates_with",
        object_ref=personas[structure.bridge_subjects[1]].persona_id,
        story_position=5,
        validity_end=8,
        disclosure_order=50,
        role=NeutralFactRole.CORE,
        commitment=NeutralCommitment.WORLD,
        relevant_lenses=(LensFamily.CAUSAL_CONSEQUENCE, LensFamily.ALLEGIANCE_STATE_CHANGE),
    )
    outcome_fact = WorldFactSpec(
        fact_id=f"{world_id}.fact.causal.outcome",
        subject_ref=personas[structure.bridge_subjects[1]].persona_id,
        relation="secures_outcome_for",
        object_ref=collective_ids[structure.outcome_collective_index],
        story_position=7,
        validity_end=None,
        disclosure_order=51,
        role=NeutralFactRole.CORE,
        commitment=NeutralCommitment.WORLD,
        relevant_lenses=(LensFamily.CAUSAL_CONSEQUENCE, LensFamily.ALLEGIANCE_STATE_CHANGE),
    )
    facts.extend((support_fact, outcome_fact))
    impact = tuple(RareImpactKind)[(ordinal - 1) % len(RareImpactKind)]
    rare_subject = personas[structure.rare_subject_index]
    if impact in {RareImpactKind.STATE, RareImpactKind.COMMUNITY}:
        rare_relation = "defects_to"
        rare_object = (
            collective_ids[0]
            if rare_subject.home_collective_id == collective_ids[1]
            else collective_ids[1]
        )
    elif impact is RareImpactKind.TEMPORAL:
        rare_relation, rare_object = "signals_before", events[0].event_ref
    else:
        rare_relation, rare_object = "reveals_route_to", events[0].event_ref
    rare_fact = WorldFactSpec(
        fact_id=f"{world_id}.fact.rare",
        subject_ref=rare_subject.persona_id,
        relation=rare_relation,
        object_ref=rare_object,
        story_position=4,
        validity_end=10,
        disclosure_order=45,
        role=NeutralFactRole.RARE_PIVOTAL,
        commitment=NeutralCommitment.WORLD,
        event_ref=(events[0].event_ref if rare_object == events[0].event_ref else None),
        relevant_lenses=(
            (
                LensFamily.CAUSAL_CONSEQUENCE,
                LensFamily.EVENT_PARTICIPATION_CONFLICT,
                LensFamily.ALLEGIANCE_STATE_CHANGE,
                LensFamily.KNOWLEDGE_BELIEF,
            )
            if impact is RareImpactKind.CAUSAL
            else (
                LensFamily.MOVEMENT_TIME,
                LensFamily.EVENT_PARTICIPATION_CONFLICT,
                LensFamily.CAUSAL_CONSEQUENCE,
                LensFamily.KNOWLEDGE_BELIEF,
            )
            if impact is RareImpactKind.TEMPORAL
            else (
                LensFamily.ALLEGIANCE_STATE_CHANGE,
                LensFamily.IDENTITY_KINSHIP,
                LensFamily.CAUSAL_CONSEQUENCE,
                LensFamily.KNOWLEDGE_BELIEF,
            )
        ),
    )
    facts.append(rare_fact)
    conflict_group = f"{world_id}.conflict.0" if ordinal in {3, 5, 6, 9, 11, 12} else None
    facts.append(
        WorldFactSpec(
            fact_id=f"{world_id}.fact.report",
            subject_ref=personas[structure.report_subject_index].persona_id,
            relation="plans_to_leave",
            object_ref=collective_ids[0],
            story_position=5,
            disclosure_order=60,
            role=NeutralFactRole.REPORT,
            commitment=NeutralCommitment.REPORTED,
            holder_ref=personas[1].persona_id,
            relevant_lenses=(
                LensFamily.KNOWLEDGE_BELIEF,
                LensFamily.ALLEGIANCE_STATE_CHANGE,
            ),
            conflict_group=conflict_group,
        )
    )
    if conflict_group is not None:
        facts.append(
            WorldFactSpec(
                fact_id=f"{world_id}.fact.denial",
                subject_ref=personas[structure.report_subject_index].persona_id,
                relation="plans_to_leave",
                object_ref=collective_ids[0],
                story_position=5,
                disclosure_order=61,
                role=NeutralFactRole.REPORT,
                commitment=NeutralCommitment.DENIED,
                holder_ref=personas[0].persona_id,
                relevant_lenses=(
                    LensFamily.KNOWLEDGE_BELIEF,
                    LensFamily.ALLEGIANCE_STATE_CHANGE,
                ),
                conflict_group=conflict_group,
            )
        )
    facts.extend(
        (
            WorldFactSpec(
                fact_id=f"{world_id}.fact.distractor",
                subject_ref=personas[structure.distractor_subject_index].persona_id,
                relation="greets_at",
                object_ref=places[1].place_id,
                story_position=6,
                validity_end=6,
                disclosure_order=70,
                role=NeutralFactRole.DISTRACTOR,
                commitment=NeutralCommitment.WORLD,
                repetition_count=3,
            ),
            WorldFactSpec(
                fact_id=f"{world_id}.fact.late",
                subject_ref=personas[0].persona_id,
                relation="later_visits",
                object_ref=places[0].place_id,
                story_position=12,
                validity_end=12,
                disclosure_order=100,
                role=NeutralFactRole.SUPPORT,
                commitment=NeutralCommitment.WORLD,
                relevant_lenses=(LensFamily.MOVEMENT_TIME,),
            ),
        )
    )
    chronological_events = sorted(events, key=lambda item: item.story_position)
    temporal = tuple(
        (left.event_ref, right.event_ref)
        for left, right in itertools.pairwise(chronological_events)
    )
    causal = (
        (rare_fact.fact_id, support_fact.fact_id),
        (support_fact.fact_id, outcome_fact.fact_id),
    )
    return WorldSpec(
        world_id=world_id,
        structure_template_id=structure.template_id,
        split=split,
        difficulty=difficulty,
        theme=theme,
        personas=tuple(personas),
        collectives=collectives,
        places=places,
        events=tuple(events),
        facts=tuple(facts),
        temporal_precedence=temporal,
        causal_dependencies=causal,
        rare_impact_kind=impact,
        factors=_factors(ordinal),
        contrast_family=contrast,
        node_budget=config.gold_node_targets[difficulty],
        horizon_block=horizon,
        surface_renderer=renderer,
    )


def build_world_specs(
    config: BenchmarkConfiguration, seeds: SeedDerivationManifest
) -> tuple[WorldSpec, ...]:
    development_contrasts = (
        ContrastFamily.MERGE_SPLIT,
        ContrastFamily.EVENT_REIFICATION,
        ContrastFamily.SCHEMA_RELATION,
        ContrastFamily.TEMPORAL_QUALIFICATION,
    )
    development = tuple(
        _build_world_spec(
            config,
            BenchmarkSplit.DEVELOPMENT,
            ordinal,
            HorizonBlock.FINAL,
            development_contrasts[ordinal - 1],
            config.development_surface_renderers[0],
            _DEVELOPMENT_STRUCTURES[ordinal - 1],
        )
        for ordinal in range(1, 5)
    )
    horizon_rng_seed = seed_for(seeds, SeedPurpose.WORLD)
    horizons: list[HorizonBlock] = []
    for difficulty in Difficulty:
        block = [
            HorizonBlock.INTERMEDIATE,
            HorizonBlock.INTERMEDIATE,
            HorizonBlock.FINAL,
            HorizonBlock.FINAL,
        ]
        random.Random(derive_seed(horizon_rng_seed, f"horizon/{difficulty.value}")).shuffle(block)
        horizons.extend(block)
    renderer_rng = random.Random(derive_seed(horizon_rng_seed, "held-out-renderers"))
    held_out = tuple(
        _build_world_spec(
            config,
            BenchmarkSplit.HELD_OUT,
            ordinal,
            horizons[ordinal - 1],
            _CONTRAST_SCHEDULE[ordinal - 1],
            renderer_rng.choice(config.held_out_surface_renderers),
            _HELD_OUT_STRUCTURES[ordinal - 1],
        )
        for ordinal in range(1, 13)
    )
    return (*development, *held_out)


_LENS_GROUPS = (
    (
        LensFamily.ALLEGIANCE_STATE_CHANGE,
        LensFamily.IDENTITY_KINSHIP,
        LensFamily.CAUSAL_CONSEQUENCE,
    ),
    (
        LensFamily.EVENT_PARTICIPATION_CONFLICT,
        LensFamily.MOVEMENT_TIME,
        LensFamily.KNOWLEDGE_BELIEF,
    ),
    (
        LensFamily.ALLEGIANCE_STATE_CHANGE,
        LensFamily.EVENT_PARTICIPATION_CONFLICT,
        LensFamily.MOVEMENT_TIME,
    ),
    (
        LensFamily.IDENTITY_KINSHIP,
        LensFamily.CAUSAL_CONSEQUENCE,
        LensFamily.KNOWLEDGE_BELIEF,
    ),
)


def _all_orders(values: Sequence[Any]) -> tuple[tuple[Any, Any, Any], ...]:
    return tuple(
        sorted(itertools.permutations(values), key=lambda item: tuple(str(v) for v in item))
    )


def _eligible_abstraction_orders(
    family: ContrastFamily,
    *,
    fixed_friendly_context_index: int | None = None,
) -> tuple[tuple[AbstractionLevel, AbstractionLevel, AbstractionLevel], ...]:
    orders = _all_orders(tuple(AbstractionLevel))
    required = (
        AbstractionLevel.COLLECTIVE_CAUSAL_CHAIN
        if family is ContrastFamily.MERGE_SPLIT
        else AbstractionLevel.EVENT_ROLE
        if family is ContrastFamily.EVENT_REIFICATION
        else None
    )
    if required is not None:
        orders = tuple(item for item in orders if required in item[:2])
    if fixed_friendly_context_index is not None:
        orders = tuple(
            item for item in orders if item[fixed_friendly_context_index] is AbstractionLevel.ACTOR
        )
    return orders


def _eligible_scope_orders(
    family: ContrastFamily,
) -> tuple[tuple[StoryScopeBlock, StoryScopeBlock, StoryScopeBlock], ...]:
    orders = _all_orders(tuple(StoryScopeBlock))
    if family is not ContrastFamily.TEMPORAL_QUALIFICATION:
        return orders
    return tuple(
        item for item in orders if StoryScopeBlock.POINT in item[:2] and item[0] is not item[1]
    )


def _eligible_lens_orders(
    group: tuple[LensFamily, LensFamily, LensFamily],
    family: ContrastFamily,
) -> tuple[tuple[LensFamily, LensFamily, LensFamily], ...]:
    orders = _all_orders(group)
    holder_lenses = {LensFamily.KNOWLEDGE_BELIEF, LensFamily.ALLEGIANCE_STATE_CHANGE}
    if family is ContrastFamily.EPISTEMIC_QUALIFICATION:
        orders = tuple(item for item in orders if holder_lenses.intersection(item[:2]))
    return orders


def sample_held_out_queries(
    held_out_specs: Sequence[WorldSpec],
    seeds: SeedDerivationManifest,
) -> QuerySamplingAudit:
    if len(held_out_specs) != 12:
        raise ValueError("held-out query allocation requires exactly twelve worlds")
    assignments: list[QueryAssignment] = []
    allocation_records: list[ConditionalAllocationRecord] = []
    block_permutations: dict[Difficulty, tuple[int, int, int, int]] = {}
    query_seed = seed_for(seeds, SeedPurpose.QUERY)
    for difficulty in Difficulty:
        difficulty_specs = sorted(
            (item for item in held_out_specs if item.difficulty is difficulty),
            key=lambda item: item.world_id,
        )
        if len(difficulty_specs) != 4:
            raise ValueError("query allocation requires four worlds per difficulty block")
        group_indices = list(range(len(_LENS_GROUPS)))
        random.Random(derive_seed(query_seed, f"lens-permutation/{difficulty.value}")).shuffle(
            group_indices
        )
        block_permutations[difficulty] = tuple(group_indices)  # type: ignore[assignment]
        for spec, group_index in zip(difficulty_specs, group_indices, strict=True):
            group = _LENS_GROUPS[group_index]
            world_seed = derive_seed(query_seed, f"conditional-allocation/{spec.world_id}")
            lens_orders = _eligible_lens_orders(group, spec.contrast_family)
            scope_orders = _eligible_scope_orders(spec.contrast_family)
            fixed_friendly = BenchmarkFactor.FIXED_ONTOLOGY_FRIENDLY in spec.factors
            abstraction_orders = _eligible_abstraction_orders(
                spec.contrast_family,
                fixed_friendly_context_index=(2 if fixed_friendly else None),
            )
            lenses = random.Random(derive_seed(world_seed, "lens-order")).choice(lens_orders)
            scopes = random.Random(derive_seed(world_seed, "scope-order")).choice(scope_orders)
            abstractions = random.Random(derive_seed(world_seed, "abstraction-order")).choice(
                abstraction_orders
            )
            holder_lenses = {LensFamily.KNOWLEDGE_BELIEF, LensFamily.ALLEGIANCE_STATE_CHANGE}
            viewpoint_candidates = tuple(
                index
                for index, lens in enumerate(lenses)
                if lens in holder_lenses
                and (
                    spec.contrast_family is not ContrastFamily.EPISTEMIC_QUALIFICATION or index < 2
                )
            )
            viewpoint_index = random.Random(derive_seed(world_seed, "viewpoint-index")).choice(
                viewpoint_candidates
            )
            viewpoint_holder = (
                0 if BenchmarkFactor.INCOMPLETE_CONFLICTING_EVIDENCE in spec.factors else 1
            )
            fixed_c = bool(
                spec.factors
                & {
                    BenchmarkFactor.FIXED_ONTOLOGY_FRIENDLY,
                    BenchmarkFactor.NULL_NONSELECTION_CASE,
                }
            )
            assignments.append(
                QueryAssignment(
                    world_id=spec.world_id,
                    lenses=lenses,
                    story_scopes=scopes,
                    abstractions=abstractions,
                    viewpoint_context_index=viewpoint_index,
                    viewpoint_holder_persona_index=viewpoint_holder,
                    node_budget=spec.node_budget,
                    horizon_block=spec.horizon_block,
                    contrast_family=spec.contrast_family,
                    compiler_policies=(
                        CompilerPolicy.QUERY_DEPENDENT,
                        CompilerPolicy.QUERY_DEPENDENT,
                        (
                            CompilerPolicy.FIXED_REFERENCE
                            if fixed_c
                            else CompilerPolicy.QUERY_DEPENDENT
                        ),
                    ),
                    null_nonselection_context_index=(
                        2 if BenchmarkFactor.NULL_NONSELECTION_CASE in spec.factors else None
                    ),
                )
            )
            allocation_records.append(
                ConditionalAllocationRecord(
                    world_id=spec.world_id,
                    selection_seed=world_seed,
                    eligible_lens_orders=lens_orders,
                    eligible_scope_orders=scope_orders,
                    eligible_abstraction_orders=abstraction_orders,
                    eligible_viewpoint_indices=viewpoint_candidates,
                    selected_lens_order=lenses,
                    selected_scope_order=scopes,
                    selected_abstraction_order=abstractions,
                    selected_viewpoint_index=viewpoint_index,
                    conditional_probability=(
                        f"1/{len(lens_orders)} * 1/{len(scope_orders)} * "
                        f"1/{len(abstraction_orders)} * 1/{len(viewpoint_candidates)}"
                    ),
                )
            )
    audit = QuerySamplingAudit(
        root_query_seed=query_seed,
        lens_candidate_multiset=tuple(lens for lens in LensFamily for _ in range(6)),
        eligible_story_scopes=tuple(StoryScopeBlock),
        eligible_abstractions=tuple(AbstractionLevel),
        eligible_viewpoint_indices=(0, 1, 2),
        eligible_horizons=tuple(HorizonBlock),
        conditional_lens_probability=(
            "uniform 1/24 permutation of four registered lens blocks within each difficulty"
        ),
        difficulty_lens_block_permutations=block_permutations,
        conditional_allocations=tuple(allocation_records),
        assignments=tuple(assignments),
    )
    validate_query_allocation(audit)
    return audit


def _development_assignments(specs: Sequence[WorldSpec]) -> tuple[QueryAssignment, ...]:
    result = []
    for index, spec in enumerate(sorted(specs, key=lambda item: item.world_id)):
        lenses = _LENS_GROUPS[index]
        holder_candidates = [
            position
            for position, lens in enumerate(lenses)
            if lens in {LensFamily.KNOWLEDGE_BELIEF, LensFamily.ALLEGIANCE_STATE_CHANGE}
        ]
        result.append(
            QueryAssignment(
                world_id=spec.world_id,
                lenses=lenses,
                story_scopes=_eligible_scope_orders(spec.contrast_family)[0],
                abstractions=_eligible_abstraction_orders(spec.contrast_family)[0],
                viewpoint_context_index=holder_candidates[0],
                viewpoint_holder_persona_index=(
                    0 if BenchmarkFactor.INCOMPLETE_CONFLICTING_EVIDENCE in spec.factors else 1
                ),
                node_budget=spec.node_budget,
                horizon_block=spec.horizon_block,
                contrast_family=spec.contrast_family,
                compiler_policies=(
                    CompilerPolicy.QUERY_DEPENDENT,
                    CompilerPolicy.QUERY_DEPENDENT,
                    CompilerPolicy.FIXED_REFERENCE,
                ),
            )
        )
    return tuple(result)


def validate_query_allocation(audit: QuerySamplingAudit) -> None:
    assignments = audit.assignments
    if Counter(lens for item in assignments for lens in item.lenses) != Counter(
        {lens: 6 for lens in LensFamily}
    ):
        raise ValueError("lens allocation must contain exactly six of every family")
    if Counter(scope for item in assignments for scope in item.story_scopes) != Counter(
        {scope: 12 for scope in StoryScopeBlock}
    ):
        raise ValueError("story scopes must be blocked 12/12/12")
    if Counter(level for item in assignments for level in item.abstractions) != Counter(
        {level: 12 for level in AbstractionLevel}
    ):
        raise ValueError("abstractions must be blocked 12/12/12")
    if len(assignments) != 12 or sum(1 for _ in assignments) != 12:
        raise ValueError("query allocation requires twelve world blocks")


_LENS_TEXT = {
    LensFamily.ALLEGIANCE_STATE_CHANGE: (
        "How do consequential affiliations and states change within the stated time?",
        "affiliation and state change",
    ),
    LensFamily.IDENTITY_KINSHIP: (
        "Which people and continuing offices must be distinguished or grouped?",
        "identity and office continuity",
    ),
    LensFamily.CAUSAL_CONSEQUENCE: (
        "What evidence-supported causal chain explains the focal outcome?",
        "causal consequence",
    ),
    LensFamily.EVENT_PARTICIPATION_CONFLICT: (
        "Who participates in the focal conflict, in what role, and where?",
        "event participation and conflict",
    ),
    LensFamily.MOVEMENT_TIME: (
        "Which movements and temporal transitions matter in the requested scope?",
        "movement and temporal order",
    ),
    LensFamily.KNOWLEDGE_BELIEF: (
        "What is reported or denied, by which holder, without treating it as world fact?",
        "knowledge and attributed belief",
    ),
}


def _scope(block: StoryScopeBlock) -> StoryTime:
    if block is StoryScopeBlock.POINT:
        return StoryTime(kind=TemporalKind.POINT, point=6, label="at story step 6")
    if block is StoryScopeBlock.INTERVAL:
        return StoryTime(
            kind=TemporalKind.INTERVAL, start=3, end=9, label="story steps 3 through 9"
        )
    return StoryTime(kind=TemporalKind.INTERVAL, end=10, label="through story step 10")


def compile_query_contexts(
    spec: WorldSpec,
    assignment: QueryAssignment,
    narrative: NarrativeProducts,
    config: BenchmarkConfiguration,
) -> tuple[QueryContext, QueryContext, QueryContext]:
    if assignment.viewpoint_holder_persona_index >= len(spec.personas):
        raise ValueError("viewpoint holder index is outside the world's personas")
    holder_ref = spec.personas[assignment.viewpoint_holder_persona_index].persona_id
    holder_evidence_ids = {
        evidence_id
        for fact in spec.facts
        if fact.holder_ref == holder_ref
        for evidence_id in narrative.fact_evidence_ids.get(fact.fact_id, ())
    }
    holder_mentions = tuple(
        item.candidate_id
        for item in narrative.bindings
        if item.semantic_ref == holder_ref
        and item.role == "holder"
        and item.evidence_id in holder_evidence_ids
    )
    if not holder_mentions:
        raise ValueError("viewpoint holder has no evidence mention")
    contexts = []
    for index, lens in enumerate(assignment.lenses):
        wording, description = _LENS_TEXT[lens]
        contexts.append(
            QueryContext(
                context_id=_opaque("ctx", spec.world_id, index),
                wording=wording,
                lens=description,
                target=f"the actors, offices, events, and consequences in {spec.theme}",
                story_scope=_scope(assignment.story_scopes[index]),
                spoiler_horizon=narrative.snapshot.horizon,
                viewpoint=(
                    EpistemicViewpoint(holder_id=holder_mentions[0])
                    if assignment.viewpoint_context_index == index
                    else None
                ),
                abstraction=assignment.abstractions[index],
                budgets=OutputBudgets(
                    node_budget=assignment.node_budget,
                    assertion_budget=assignment.node_budget * 3,
                    display_node_budget=assignment.node_budget,
                    display_assertion_budget=assignment.node_budget * 3,
                    repair_attempt_budget=1,
                ),
                revealed_at=_timestamp(config, 60 + index),
            )
        )
    return tuple(contexts)


def _sha_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _surface_maps(spec: WorldSpec) -> tuple[dict[str, str], dict[str, str]]:
    labels = {
        **{item.persona_id: item.name for item in spec.personas},
        **{item.collective_id: item.name for item in spec.collectives},
        **{item.place_id: item.name for item in spec.places},
        **{item.event_ref: item.label for item in spec.events},
    }
    types = {
        **{item.persona_id: "person_candidate" for item in spec.personas},
        **{item.collective_id: "collective_candidate" for item in spec.collectives},
        **{item.place_id: "place_candidate" for item in spec.places},
        **{item.event_ref: "event_candidate" for item in spec.events},
    }
    return labels, types


def _render_fact(spec: WorldSpec, fact: WorldFactSpec) -> tuple[str, list[tuple[str, str, str]]]:
    labels, _types = _surface_maps(spec)
    subject = labels[fact.subject_ref]
    object_label = labels[fact.object_ref]
    relation = fact.relation.replace("_", " ")
    mentions: list[tuple[str, str, str]] = [(subject, fact.subject_ref, "subject")]
    if fact.relation == "holds_office":
        title = next(
            item.office_title for item in spec.personas if item.persona_id == fact.subject_ref
        )
        if title is None:
            raise ValueError("office fact holder has no registered title")
        end = (
            f" through step {fact.validity_end}" if fact.validity_end is not None else " thereafter"
        )
        core = f"{subject} served as {title} for {object_label}{end}"
        mentions.extend(
            ((title, fact.subject_ref, "office_title"), (object_label, fact.object_ref, "object"))
        )
    elif fact.commitment is NeutralCommitment.REPORTED:
        holder = labels[fact.holder_ref]
        core = f"{holder} reported that {subject} {relation} {object_label}"
        mentions = [
            (holder, fact.holder_ref, "holder"),
            *mentions,
            (object_label, fact.object_ref, "object"),
        ]
    elif fact.commitment is NeutralCommitment.DENIED:
        holder = labels[fact.holder_ref]
        core = f"{holder} denied that {subject} {relation} {object_label}"
        mentions = [
            (holder, fact.holder_ref, "holder"),
            *mentions,
            (object_label, fact.object_ref, "object"),
        ]
    else:
        core = f"{subject} {relation} {object_label}"
        mentions.append((object_label, fact.object_ref, "object"))
    if spec.surface_renderer == "development_direct_v1":
        text = f"At story step {fact.story_position}, {core}."
    elif spec.surface_renderer == "held_out_chronicle_v1":
        text = (
            f"Entry {fact.disclosure_order} records that {core}, "
            f"at story step {fact.story_position}."
        )
    elif spec.surface_renderer == "held_out_dispatch_v1":
        text = f"Dispatch {fact.disclosure_order}: at story step {fact.story_position}, {core}."
    else:
        raise ValueError(f"unregistered narrative renderer {spec.surface_renderer}")
    return text, mentions


def _make_evidence(
    *,
    spec: WorldSpec,
    source_key: str,
    text: str,
    mention_specs: Sequence[tuple[str, str, str]],
    relation_phrase: str,
    story_position: int,
    revelation_order: int,
    passage_order: int,
    bindings: list[EvidenceBinding],
) -> EvidenceRecord:
    evidence_id = _opaque("ev", spec.world_id, source_key)
    mentions: list[MentionCandidate] = []
    search_from = 0
    for occurrence, (surface, semantic_ref, role) in enumerate(mention_specs):
        try:
            start = text.index(surface, search_from)
        except ValueError:
            start = text.index(surface)
        search_from = start + len(surface)
        candidate_id = _opaque("mn", evidence_id, occurrence, semantic_ref, role)
        mentions.append(
            MentionCandidate(
                candidate_id=candidate_id,
                evidence_id=evidence_id,
                start_char=start,
                end_char=start + len(surface),
                surface=surface,
                surface_hash=_sha_text(surface),
                provisional_type=(
                    "office_role_candidate"
                    if role == "office_title"
                    else _surface_maps(spec)[1].get(semantic_ref, "entity_candidate")
                ),
            )
        )
        bindings.append(
            EvidenceBinding(
                candidate_id=candidate_id,
                evidence_id=evidence_id,
                semantic_ref=semantic_ref,
                role=role,
            )
        )
    relation_start = text.casefold().index(relation_phrase.casefold())
    relation_id = _opaque("rel", evidence_id, relation_phrase)
    subject_index = next(
        (index for index, (_, _, role) in enumerate(mention_specs) if role == "subject"),
        0,
    )
    object_index = next(
        (
            index
            for index in range(len(mention_specs) - 1, -1, -1)
            if mention_specs[index][2] == "object"
        ),
        len(mentions) - 1,
    )
    relation = RelationPhraseCandidate(
        candidate_id=relation_id,
        evidence_id=evidence_id,
        subject_mention_candidate_id=mentions[subject_index].candidate_id,
        object_mention_candidate_id=(
            mentions[object_index].candidate_id if len(mentions) > 1 else None
        ),
        surface_phrase=relation_phrase,
        confidence=0.96,
    )
    event_candidates: tuple[EventCandidate, ...] = ()
    if any(
        _surface_maps(spec)[1].get(binding[1]) == "event_candidate" for binding in mention_specs
    ):
        event_candidates = (
            EventCandidate(
                candidate_id=_opaque("evt", evidence_id, relation_phrase),
                evidence_id=evidence_id,
                trigger_start_char=relation_start,
                trigger_end_char=relation_start + len(relation_phrase),
                trigger_surface=text[relation_start : relation_start + len(relation_phrase)],
                participant_mention_candidate_ids=tuple(
                    item.candidate_id
                    for item, (_, ref, _) in zip(mentions, mention_specs, strict=True)
                    if _surface_maps(spec)[1].get(ref) == "person_candidate"
                ),
                confidence=0.9,
            ),
        )
    clue_target = event_candidates[0].candidate_id if event_candidates else relation_id
    clues = (
        TemporalClue(
            clue_id=_opaque("tmp", evidence_id, "story", story_position),
            evidence_id=evidence_id,
            normalized_expression=f"story-step-{story_position}",
            target_candidate_ids=(clue_target,),
            confidence=0.99,
        ),
        TemporalClue(
            clue_id=_opaque("tmp", evidence_id, "revelation", revelation_order),
            evidence_id=evidence_id,
            normalized_expression=f"revelation-order-{revelation_order}",
            target_candidate_ids=(clue_target,),
            confidence=0.99,
        ),
    )
    provenance = ProvenanceReference(
        provenance_id=_opaque("prv", evidence_id),
        evidence_id=evidence_id,
        extraction_method=f"synthetic-renderer/{_opaque('style', spec.surface_renderer)}",
        locator=f"synthetic:{_opaque('loc', evidence_id)}",
        # A synthetic passage is its own immutable source artifact.  Binding the
        # exact rendered UTF-8 bytes supplies a real provenance ceiling without
        # importing scorer-only facts or query-derived semantics.
        source_artifact_hash=_sha_text(text),
        confidence=1.0,
    )
    return EvidenceRecord(
        evidence_id=evidence_id,
        passage_id=_opaque("psg", evidence_id),
        text=text,
        text_hash=_sha_text(text),
        discourse_position=DiscoursePosition(passage_order=passage_order),
        mention_candidates=tuple(mentions),
        event_candidates=event_candidates,
        relation_phrase_candidates=(relation,),
        temporal_clues=clues,
        provenance=provenance,
        confidence=1.0,
        release_class=ReleaseClass.PUBLIC,
    )


def realize_narrative(
    spec: WorldSpec,
    config: BenchmarkConfiguration,
    seeds: SeedDerivationManifest,
) -> NarrativeProducts:
    """Create query-blind evidence and scorer-only candidate bindings."""

    narrative_seed = seed_for(seeds, SeedPurpose.NARRATIVE)
    labels, _ = _surface_maps(spec)
    fact_by_id = {item.fact_id: item for item in spec.facts}
    event_by_id = {item.event_ref: item for item in spec.events}
    work: list[tuple[int, str, Any]] = []
    for fact in spec.facts:
        for occurrence in range(fact.repetition_count):
            work.append((fact.disclosure_order, f"fact:{fact.fact_id}:{occurrence}", fact))
    for index, dependency in enumerate(spec.causal_dependencies):
        work.append((55 + index, f"cause:{dependency[0]}:{dependency[1]}", dependency))
    for index, precedence in enumerate(spec.temporal_precedence):
        work.append((40 + index, f"time:{precedence[0]}:{precedence[1]}", precedence))

    def discourse_story_position(item: tuple[int, str, Any]) -> int:
        _, source_key, payload = item
        if source_key.startswith("fact:"):
            return payload.story_position
        if source_key.startswith("cause:"):
            return fact_by_id[payload[1]].story_position
        return event_by_id[payload[1]].story_position

    work.sort(
        key=lambda item: (
            discourse_story_position(item),
            derive_seed(narrative_seed, f"narrative-order/{spec.world_id}/{item[1]}"),
        )
    )
    all_evidence: list[tuple[int, EvidenceRecord]] = []
    bindings: list[EvidenceBinding] = []
    fact_evidence: dict[str, list[str]] = defaultdict(list)
    causal_evidence: dict[tuple[str, str], list[str]] = defaultdict(list)
    temporal_evidence: dict[tuple[str, str], list[str]] = defaultdict(list)
    for passage_order, (disclosure, source_key, payload) in enumerate(work):
        if source_key.startswith("fact:"):
            fact = payload
            text, mention_specs = _render_fact(spec, fact)
            relation_phrase = (
                "served as" if fact.relation == "holds_office" else fact.relation.replace("_", " ")
            )
            evidence = _make_evidence(
                spec=spec,
                source_key=source_key,
                text=text,
                mention_specs=mention_specs,
                relation_phrase=relation_phrase,
                story_position=fact.story_position,
                revelation_order=disclosure,
                passage_order=passage_order,
                bindings=bindings,
            )
            fact_evidence[fact.fact_id].append(evidence.evidence_id)
        elif source_key.startswith("cause:"):
            left_id, right_id = payload
            left, right = fact_by_id[left_id], fact_by_id[right_id]
            left_name, right_name = labels[left.subject_ref], labels[right.subject_ref]
            text = (
                f"The record explicitly says that {left_name}'s earlier action enabled "
                f"{right_name}'s later action at story step {right.story_position}."
            )
            evidence = _make_evidence(
                spec=spec,
                source_key=source_key,
                text=text,
                mention_specs=(
                    (left_name, left.subject_ref, "subject"),
                    (right_name, right.subject_ref, "object"),
                ),
                relation_phrase="enabled",
                story_position=right.story_position,
                revelation_order=disclosure,
                passage_order=passage_order,
                bindings=bindings,
            )
            causal_evidence[(left_id, right_id)].append(evidence.evidence_id)
        else:
            left_id, right_id = payload
            left, right = event_by_id[left_id], event_by_id[right_id]
            text = (
                f"The {left.label} occurred before the {right.label}; the latter began "
                f"at story step {right.story_position}."
            )
            evidence = _make_evidence(
                spec=spec,
                source_key=source_key,
                text=text,
                mention_specs=(
                    (left.label, left.event_ref, "event"),
                    (right.label, right.event_ref, "event"),
                ),
                relation_phrase="occurred before",
                story_position=right.story_position,
                revelation_order=disclosure,
                passage_order=passage_order,
                bindings=bindings,
            )
            temporal_evidence[(left_id, right_id)].append(evidence.evidence_id)
        all_evidence.append((disclosure, evidence))
    maximum_disclosure = 90 if spec.horizon_block is HorizonBlock.INTERMEDIATE else 10_000
    admissible_pairs = tuple(
        (disclosure, item) for disclosure, item in all_evidence if disclosure < maximum_disclosure
    )
    admissible = tuple(item for _, item in admissible_pairs)
    admissible_ids = {item.evidence_id for item in admissible}
    admissible_bindings = tuple(item for item in bindings if item.evidence_id in admissible_ids)
    ref_mentions: dict[str, list[str]] = defaultdict(list)
    title_mentions: dict[str, list[str]] = defaultdict(list)
    for binding in admissible_bindings:
        ref_mentions[binding.semantic_ref].append(binding.candidate_id)
        if binding.role == "office_title":
            title_mentions[binding.semantic_ref].append(binding.candidate_id)
    horizon = SpoilerHorizon(
        horizon_id=_opaque("hor", spec.world_id, spec.horizon_block.value),
        max_discourse_position=max(
            (item.discourse_position for item in admissible),
            key=lambda value: value.ordering_key,
        ),
        max_revelation_position=RevelationPosition(
            revelation_order=max(disclosure for disclosure, _ in admissible_pairs)
        ),
    )
    snapshot = EvidenceSnapshot(
        snapshot_id=_opaque("snp", spec.world_id),
        corpus_id=_opaque("corpus", config.generator_revision),
        world_or_window_id=_opaque("unit", spec.world_id),
        horizon=horizon,
        eligible_evidence_ids=tuple(item.evidence_id for item in admissible),
        index_config_hash=canonical_sha256(
            {
                "revision": config.index_revision,
                "renderer": spec.surface_renderer,
                "query_blind": True,
            }
        ),
        created_at=_timestamp(config, -120),
        sealed_at=_timestamp(config, -60),
        release_class=ReleaseClass.PUBLIC,
    )

    def visible(mapping: Mapping[Any, Sequence[str]]) -> dict[Any, tuple[str, ...]]:
        return {
            key: tuple(item for item in values if item in admissible_ids)
            for key, values in mapping.items()
        }

    return NarrativeProducts(
        snapshot=snapshot,
        evidence=admissible,
        fact_evidence_ids=visible(fact_evidence),
        causal_evidence_ids=visible(causal_evidence),
        temporal_evidence_ids=visible(temporal_evidence),
        revelation_order_by_evidence_id={
            item.evidence_id: disclosure for disclosure, item in admissible_pairs
        },
        ref_mention_ids={key: tuple(values) for key, values in ref_mentions.items()},
        office_title_mentions_by_holder={
            key: tuple(values) for key, values in title_mentions.items()
        },
        bindings=admissible_bindings,
    )


def _lens_for_context(context: QueryContext) -> LensFamily:
    return next(lens for lens, (_, text) in _LENS_TEXT.items() if text == context.lens)


def _time_bounds(value: StoryTime) -> tuple[int, int]:
    if value.kind is TemporalKind.POINT:
        assert value.point is not None
        return value.point, value.point
    if value.kind is TemporalKind.INTERVAL:
        return (
            value.start if value.start is not None else -(10**9),
            value.end if value.end is not None else 10**9,
        )
    raise ValueError("synthetic contexts use only point and bounded/through intervals")


def _fact_overlaps_scope(fact: WorldFactSpec, scope: StoryTime) -> bool:
    query_start, query_end = _time_bounds(scope)
    fact_end = fact.validity_end if fact.validity_end is not None else 10**9
    return fact.story_position <= query_end and fact_end >= query_start


def _event_overlaps_scope(event: WorldEventSpec, scope: StoryTime) -> bool:
    query_start, query_end = _time_bounds(scope)
    event_end = event.story_position + event.duration
    return event.story_position <= query_end and event_end >= query_start


def _candidate_ids(evidence: EvidenceRecord) -> set[str]:
    return {
        *(item.candidate_id for item in evidence.mention_candidates),
        *(item.candidate_id for item in evidence.event_candidates),
        *(item.candidate_id for item in evidence.relation_phrase_candidates),
        *(item.clue_id for item in evidence.temporal_clues),
    }


def _grounded_anchors_for_evidence(
    narrative: NarrativeProducts,
    evidence_ids: Sequence[str],
    *,
    limit: int = 3,
) -> tuple[str, ...]:
    allowed = set(evidence_ids)
    result = [
        binding.candidate_id for binding in narrative.bindings if binding.evidence_id in allowed
    ]
    if not result:
        evidence_by_id = {item.evidence_id: item for item in narrative.evidence}
        result = [
            candidate
            for evidence_id in evidence_ids
            for candidate in sorted(_candidate_ids(evidence_by_id[evidence_id]))
        ]
    return tuple(dict.fromkeys(result))[:limit]


def _partition_signature(partition: Sequence[GoldEntityCluster]) -> str:
    groups = sorted(tuple(sorted(item.mention_candidate_ids)) for item in partition)
    return canonical_json(groups)


def _build_entity_partition(
    spec: WorldSpec,
    context: QueryContext,
    policy: CompilerPolicy,
    narrative: NarrativeProducts,
) -> tuple[tuple[GoldEntityCluster, ...], dict[str, str], tuple[SemanticAtom, ...]]:
    abstraction = (
        AbstractionLevel.ACTOR if policy is CompilerPolicy.FIXED_REFERENCE else context.abstraction
    )
    entity_refs = [
        *(item.persona_id for item in spec.personas),
        *(item.collective_id for item in spec.collectives),
        *(item.place_id for item in spec.places),
    ]
    title_by_holder = {
        holder: set(values) for holder, values in narrative.office_title_mentions_by_holder.items()
    }
    title_all = set().union(*title_by_holder.values()) if title_by_holder else set()
    clusters: list[GoldEntityCluster] = []
    ref_to_cluster: dict[str, str] = {}
    for index, ref in enumerate(entity_refs):
        anchors = list(narrative.ref_mention_ids.get(ref, ()))
        if ref in title_by_holder:
            anchors = [item for item in anchors if item not in title_all]
            if abstraction is not AbstractionLevel.COLLECTIVE_CAUSAL_CHAIN:
                anchors.extend(sorted(title_by_holder[ref]))
        if (
            abstraction is AbstractionLevel.COLLECTIVE_CAUSAL_CHAIN
            and ref == spec.collectives[0].collective_id
        ):
            anchors.extend(sorted(title_all))
        if not anchors:
            raise ValueError(f"semantic entity has no admissible evidence anchor: {ref}")
        cluster_id = f"score.{context.context_id}.entity.{index:02d}"
        clusters.append(
            GoldEntityCluster(
                cluster_id=cluster_id, mention_candidate_ids=tuple(dict.fromkeys(anchors))
            )
        )
        ref_to_cluster[ref] = cluster_id
    if abstraction is AbstractionLevel.COLLECTIVE_CAUSAL_CHAIN and len(title_all) < 2:
        raise ValueError("collective office view requires both evidenced title mentions")
    membership: dict[str, str] = {}
    for cluster in clusters:
        for anchor in cluster.mention_candidate_ids:
            membership[anchor] = cluster.cluster_id
    atoms: list[SemanticAtom] = []
    holders = [item for item in spec.personas if item.office_title is not None]
    title_mentions = [
        narrative.office_title_mentions_by_holder[item.persona_id][0] for item in holders
    ]
    person_mentions = []
    for holder in holders:
        title_set = set(narrative.office_title_mentions_by_holder[holder.persona_id])
        person_mentions.append(
            next(
                item
                for item in narrative.ref_mention_ids[holder.persona_id]
                if item not in title_set
            )
        )
    identity_pairs = (
        ("title-continuity", title_mentions[0], title_mentions[1]),
        ("holder-0-title", person_mentions[0], title_mentions[0]),
        ("holder-1-title", person_mentions[1], title_mentions[1]),
    )
    binding_by_id = {item.candidate_id: item for item in narrative.bindings}
    for label, left, right in identity_pairs:
        together = membership[left] == membership[right]
        evidence_ids = tuple(
            sorted({binding_by_id[left].evidence_id, binding_by_id[right].evidence_id})
        )
        atoms.append(
            SemanticAtom(
                slot_key=f"identity/{label}",
                operator=(ConstructionOperator.MERGE if together else ConstructionOperator.SPLIT),
                signature=("partition:co-clustered" if together else "partition:separate"),
                anchor_ids=(left, right),
                evidence_ids=evidence_ids,
            )
        )
    return tuple(clusters), ref_to_cluster, tuple(atoms)


def _reified_event_specs(
    spec: WorldSpec, context: QueryContext, policy: CompilerPolicy
) -> tuple[WorldEventSpec, ...]:
    """Keep graph size fixed while changing which event boundary is reified."""

    count = {Difficulty.EASY: 1, Difficulty.MEDIUM: 2, Difficulty.HARD: 2}[spec.difficulty]
    focal = next(item for item in spec.events if item.focal)
    nonfocal = sorted(
        (item for item in spec.events if not item.focal), key=lambda item: item.event_ref
    )
    fixed_friendly_profile = (
        BenchmarkFactor.FIXED_ONTOLOGY_FRIENDLY in spec.factors
        and context.abstraction is AbstractionLevel.ACTOR
    )
    if (
        policy is CompilerPolicy.FIXED_REFERENCE
        or fixed_friendly_profile
        or context.abstraction is AbstractionLevel.EVENT_ROLE
    ):
        candidates = [focal, *nonfocal]
    else:
        candidates = [*nonfocal, focal]
    return tuple(candidates[:count])


def _focal_reified(spec: WorldSpec, context: QueryContext, policy: CompilerPolicy) -> bool:
    focal = next(item for item in spec.events if item.focal)
    return focal in _reified_event_specs(spec, context, policy)


def _temporal_scope(
    fact: WorldFactSpec,
    evidence: EvidenceRecord,
    revelation_order: int,
    contextual_scope: StoryTime | None = None,
) -> TemporalScope:
    validity_start = fact.story_position
    validity_end = fact.validity_end
    if contextual_scope is not None:
        query_start, query_end = _time_bounds(contextual_scope)
        validity_start = max(validity_start, query_start)
        intrinsic_end = validity_end if validity_end is not None else 10**9
        clipped_end = min(intrinsic_end, query_end)
        validity_end = None if clipped_end == 10**9 else clipped_end
    return TemporalScope(
        story_time=StoryTime(
            kind=TemporalKind.POINT,
            point=fact.story_position,
            label=f"story step {fact.story_position}",
        ),
        validity_time=ValidityTime(
            kind=TemporalKind.INTERVAL,
            start=validity_start,
            end=validity_end,
            label=(
                f"contextually valid from step {validity_start}"
                if validity_end is None
                else f"contextually valid from step {validity_start} through {validity_end}"
            ),
        ),
        discourse_position=evidence.discourse_position,
        revelation_position=RevelationPosition(
            revelation_order=revelation_order,
            label=f"revealed at narrative disclosure {revelation_order}",
        ),
    )


def _contextual_validity_time(
    start: int,
    end: int | None,
    contextual_scope: StoryTime | None,
) -> ValidityTime:
    if contextual_scope is not None:
        query_start, query_end = _time_bounds(contextual_scope)
        clipped_start = max(start, query_start)
        intrinsic_end = end if end is not None else 10**9
        clipped_end = min(intrinsic_end, query_end)
        if clipped_start <= clipped_end:
            start = clipped_start
            end = None if clipped_end == 10**9 else clipped_end
    return ValidityTime(kind=TemporalKind.INTERVAL, start=start, end=end)


def _predicate_id(context: QueryContext, relation: str) -> str:
    return f"score.{context.context_id}.predicate.{relation.replace('_', '-')}"


def _assertion_endpoints(assertion: QualifiedAssertion) -> set[str]:
    return {
        *(item for item in (assertion.subject_id, assertion.object_id) if item is not None),
        *(item.object_id for item in assertion.roles),
    }


def _assertion_directed_subject(assertion: QualifiedAssertion) -> str | None:
    if assertion.subject_id is not None:
        return assertion.subject_id
    return next(
        (
            item.object_id
            for item in assertion.roles
            if item.role in {"initiator", "participant", "earlier_participant"}
        ),
        None,
    )


def _schema_label(
    lens: LensFamily,
    policy: CompilerPolicy,
    *,
    fixed_friendly_profile: bool = False,
) -> tuple[str, str]:
    if policy is CompilerPolicy.FIXED_REFERENCE or fixed_friendly_profile:
        return "fixed-general", "member_of"
    labels = {
        LensFamily.ALLEGIANCE_STATE_CHANGE: ("affiliation-state", "member_of"),
        LensFamily.IDENTITY_KINSHIP: ("office-identity", "succeeds"),
        LensFamily.CAUSAL_CONSEQUENCE: ("causal-chain", "causally_enables"),
        LensFamily.EVENT_PARTICIPATION_CONFLICT: ("conflict-role", "participates_in"),
        LensFamily.MOVEMENT_TIME: ("temporal-movement", "precedes_event"),
        LensFamily.KNOWLEDGE_BELIEF: ("holder-perspective", "plans_to_leave"),
    }
    return labels[lens]


@dataclass(frozen=True)
class CompiledGold:
    projection: GoldContextualProjection
    alternatives: GoldAlternativeSet
    semantic_atoms: tuple[SemanticAtom, ...]
    answer_signature: AnswerSignature
    community_rationales: tuple[CommunityRationale, ...]
    alternative_candidate: AlternativeCandidateView
    rare_support_path: RareSupportPath | None


def _community_semantics(
    spec: WorldSpec,
    context: QueryContext,
    policy: CompilerPolicy,
    narrative: NarrativeProducts,
    clusters: Sequence[GoldEntityCluster],
    ref_to_cluster: Mapping[str, str],
    event_id_by_ref: Mapping[str, str],
    included_facts: Sequence[WorldFactSpec],
) -> tuple[tuple[GoldCommunityAssignment, ...], tuple[CommunityRationale, ...], tuple[str, ...]]:
    community_by_ref: dict[str, str] = {}
    rationale_basis: dict[str, tuple[str, tuple[str, ...]]] = {}
    for collective in spec.collectives:
        signature = f"collective:{collective.collective_id}"
        community_by_ref[collective.collective_id] = signature
        evidence = tuple(
            evidence_id
            for fact in included_facts
            if fact.object_ref == collective.collective_id and fact.relation == "member_of"
            for evidence_id in narrative.fact_evidence_ids.get(fact.fact_id, ())
        )
        rationale_basis[collective.collective_id] = ("collective_self", evidence[:1])
    for persona in spec.personas:
        destination = persona.home_collective_id
        basis = "home_collective"
        supporting_fact = next(
            fact
            for fact in included_facts
            if fact.subject_ref == persona.persona_id and fact.relation == "member_of"
        )
        defection = next(
            (
                item
                for item in included_facts
                if item.subject_ref == persona.persona_id and item.relation == "defects_to"
            ),
            None,
        )
        if defection is not None:
            destination = defection.object_ref
            basis = "defection"
            supporting_fact = defection
        community_by_ref[persona.persona_id] = f"collective:{destination}"
        rationale_basis[persona.persona_id] = (
            basis,
            narrative.fact_evidence_ids[supporting_fact.fact_id],
        )
    # Places inherit the community of the collective whose members use them in events.
    for place in spec.places:
        event = next(item for item in spec.events if item.place_ref == place.place_id)
        participant = event.participant_refs[0]
        community_by_ref[place.place_id] = community_by_ref[participant]
        place_fact = next(
            fact
            for fact in included_facts
            if fact.event_ref == event.event_ref and fact.object_ref == place.place_id
        )
        rationale_basis[place.place_id] = (
            "place_use",
            narrative.fact_evidence_ids[place_fact.fact_id],
        )
    cluster_ref: dict[str, str] = {cluster_id: ref for ref, cluster_id in ref_to_cluster.items()}
    assignments: list[GoldCommunityAssignment] = []
    rationales: list[CommunityRationale] = []
    for cluster in clusters:
        if cluster.cluster_id in cluster_ref:
            ref = cluster_ref[cluster.cluster_id]
            signature = community_by_ref[ref]
            basis, evidence = rationale_basis[ref]
        else:
            # The continuing office role is explicitly attached to the first collective.
            signature = f"collective:{spec.collectives[0].collective_id}"
            office_evidence = tuple(
                evidence_id
                for fact in included_facts
                if fact.relation == "holds_office"
                for evidence_id in narrative.fact_evidence_ids[fact.fact_id]
            )
            basis, evidence = "home_collective", office_evidence
        signature_digest = hashlib.sha256(signature.encode()).hexdigest()[:12]
        community_id = f"score.{context.context_id}.community.{signature_digest}"
        assignments.append(
            GoldCommunityAssignment(anchor_id=cluster.cluster_id, community_id=community_id)
        )
        rationales.append(
            CommunityRationale(
                anchor_id=cluster.cluster_id,
                community_signature=signature,
                semantic_basis=basis,
                evidence_ids=tuple(evidence),
            )
        )
    event_by_id = {item.event_ref: item for item in spec.events}
    for event_ref, event_id in event_id_by_ref.items():
        event = event_by_id[event_ref]
        participant_groups = [community_by_ref[item] for item in event.participant_refs]
        signature = Counter(participant_groups).most_common(1)[0][0]
        evidence = tuple(
            evidence_id
            for fact in included_facts
            if fact.event_ref == event_ref
            for evidence_id in narrative.fact_evidence_ids.get(fact.fact_id, ())
        )
        signature_digest = hashlib.sha256(signature.encode()).hexdigest()[:12]
        community_id = f"score.{context.context_id}.community.{signature_digest}"
        assignments.append(GoldCommunityAssignment(anchor_id=event_id, community_id=community_id))
        rationales.append(
            CommunityRationale(
                anchor_id=event_id,
                community_signature=signature,
                semantic_basis="event_participation",
                evidence_ids=evidence,
            )
        )
    semantic_partition = tuple(
        sorted(
            f"{rationale.anchor_id.split('.')[-1]}:{rationale.community_signature}"
            for rationale in rationales
        )
    )
    return tuple(assignments), tuple(rationales), semantic_partition


def _executable_alternative_values(
    projection: GoldContextualProjection,
) -> ExecutableAlternativeValues:
    """Derive one semantics-preserving inverse assertion representation."""

    from story_projection_onto.metrics.alignment import (
        QualifiedAssertionSignature,
        _assertion_signature,
        _normalize_predicate,
    )

    assertion = next(
        item
        for item in projection.qualified_assertions
        if item.subject_id is not None and item.object_id is not None
    )
    predicate_definition = next(
        item
        for item in projection.local_schema.predicates
        if item.predicate_id == assertion.predicate_id
    )
    primary_signature = _assertion_signature(
        assertion,
        predicate=_normalize_predicate(predicate_definition.label, {}),
    )
    signature_values = _without_hashes(primary_signature.model_dump(mode="python"))
    signature_values["direction"] = "inverse"
    signature_values["subject_target_id"] = primary_signature.object_target_id
    signature_values["object_target_id"] = primary_signature.subject_target_id
    inverse_signature = QualifiedAssertionSignature.model_validate(signature_values)
    alternate_directions = [item.direction for item in projection.qualified_assertions]
    assertion_index = projection.qualified_assertions.index(assertion)
    alternate_directions[assertion_index] = (
        "inverse" if assertion.direction == "forward" else "forward"
    )
    components = _alternative_component_values(projection)
    primary_values = {
        **components,
        "assertion_target_id": assertion.assertion_id,
        "assertion_signature": canonical_json(primary_signature),
        "assertion_evidence_signature": canonical_json(tuple(sorted(assertion.evidence_ids))),
    }
    alternate_values = {
        **primary_values,
        "assertion_direction_signature": canonical_json(sorted(alternate_directions)),
        "assertion_signature": canonical_json(inverse_signature),
    }
    return ExecutableAlternativeValues(
        assertion_target_id=assertion.assertion_id,
        primary_assertion_signature=canonical_json(primary_signature),
        alternate_assertion_signature=canonical_json(inverse_signature),
        assertion_evidence_signature=canonical_json(tuple(sorted(assertion.evidence_ids))),
        primary_joint_representation_signature=canonical_json(primary_values),
        alternate_joint_representation_signature=canonical_json(alternate_values),
    )


def _alternative_component_values(projection: GoldContextualProjection) -> dict[str, str]:
    """Canonical full-projection fields that must move atomically as one alternative."""

    cluster_by_id = {item.cluster_id: item for item in projection.entity_partition}
    temporal = []
    epistemic = []
    for assertion in projection.qualified_assertions:
        scope = assertion.temporal_scope
        temporal.append(
            (
                tuple(sorted(assertion.evidence_ids)),
                scope.story_time.kind.value,
                scope.story_time.point,
                scope.story_time.start,
                scope.story_time.end,
                scope.validity_time.kind.value,
                scope.validity_time.start,
                scope.validity_time.end,
            )
        )
        if assertion.epistemic_scope is not None:
            holder = cluster_by_id[assertion.epistemic_scope.holder_id]
            epistemic.append(
                (
                    tuple(sorted(assertion.evidence_ids)),
                    tuple(sorted(holder.mention_candidate_ids)),
                    assertion.epistemic_scope.attitude.value,
                    assertion.narrative_commitment.value,
                )
            )
    return {
        "entity_partition_signature": _partition_signature(projection.entity_partition),
        "assertion_direction_signature": canonical_json(
            sorted(item.direction for item in projection.qualified_assertions)
        ),
        "temporal_signature": canonical_json(sorted(temporal)),
        "epistemic_signature": canonical_json(sorted(epistemic)),
    }


def _alternative_view(projection: GoldContextualProjection) -> AlternativeCandidateView:
    executable = _executable_alternative_values(projection)
    values = _alternative_component_values(projection)
    return AlternativeCandidateView(
        entity_partition_signature=values["entity_partition_signature"],
        assertion_direction_signature=values["assertion_direction_signature"],
        temporal_signature=values["temporal_signature"],
        epistemic_signature=values["epistemic_signature"],
        assertion_target_id=executable.assertion_target_id,
        assertion_signature=executable.primary_assertion_signature,
        assertion_evidence_signature=executable.assertion_evidence_signature,
        joint_representation_signature=executable.primary_joint_representation_signature,
    )


def evaluate_alternative_set(
    alternatives: GoldAlternativeSet,
    candidate: AlternativeCandidateView,
) -> AlternativeEvaluation:
    """Execute the closed matching language carried by ``GoldAlternativeSet``."""

    component_values = {
        "entity_partition_signature": candidate.entity_partition_signature,
        "assertion_direction_signature": candidate.assertion_direction_signature,
        "temporal_signature": candidate.temporal_signature,
        "epistemic_signature": candidate.epistemic_signature,
        "assertion_target_id": candidate.assertion_target_id,
        "assertion_signature": candidate.assertion_signature,
        "assertion_evidence_signature": candidate.assertion_evidence_signature,
    }
    if candidate.joint_representation_signature != canonical_json(component_values):
        return AlternativeEvaluation(
            matched=False, failed_constraints=("candidate:inconsistent-joint-signature",)
        )
    values = candidate.model_dump(mode="python", exclude={"schema_version", "content_hash"})
    known_fields = {item.value for item in AlternativeField}
    failures: list[str] = []
    for alternative in alternatives.constraint_alternatives:
        alternative_failures = []
        for constraint in alternative.constraints:
            if constraint.field_path not in known_fields:
                raise ValueError(
                    f"unregistered executable alternative field: {constraint.field_path}"
                )
            actual = values[constraint.field_path]
            if (
                constraint.operator
                in {
                    GoldConstraintOperator.EQUALS,
                    GoldConstraintOperator.ENTITY_PARTITION_EQUIVALENT,
                    GoldConstraintOperator.TEMPORALLY_EQUIVALENT,
                    GoldConstraintOperator.EPISTEMICALLY_EQUIVALENT,
                }
                or constraint.operator is GoldConstraintOperator.ONE_OF
            ):
                accepted = actual in constraint.accepted_values
            else:  # pragma: no cover - enum closure protects this branch
                raise ValueError(f"unsupported alternative operator: {constraint.operator}")
            if not accepted:
                alternative_failures.append(f"{constraint.field_path}:{constraint.operator.value}")
        if not alternative_failures:
            return AlternativeEvaluation(
                matched=True,
                matched_alternative_id=alternative.alternative_id,
            )
        failures.extend(alternative_failures)
    return AlternativeEvaluation(matched=False, failed_constraints=tuple(failures))


def normalized_decision_dict(atom: SemanticAtom) -> dict[str, Any]:
    """Return the exact structure consumed by the contrast scorer."""

    return {
        "slot_key": atom.slot_key,
        "operator": atom.operator,
        "signature": atom.signature,
        "anchor_ids": tuple(sorted(set(atom.anchor_ids))),
        "evidence_ids": tuple(sorted(set(atom.evidence_ids))),
    }


def derive_signed_changes(
    before_atoms: Sequence[SemanticAtom],
    after_atoms: Sequence[SemanticAtom],
) -> tuple[GoldContrastDecision, ...]:
    """Derive the registered A-to-B delta from normalized final-state atoms."""

    before = {item.slot_key: item for item in before_atoms}
    after = {item.slot_key: item for item in after_atoms}
    if len(before) != len(before_atoms) or len(after) != len(after_atoms):
        raise ValueError("semantic atom slot keys must be unique before contrast diffing")
    changes: list[GoldContrastDecision] = []
    for slot_key in sorted(set(before) | set(after)):
        old = before.get(slot_key)
        new = after.get(slot_key)
        if (
            old is not None
            and new is not None
            and (old.signature == new.signature and old.operator is new.operator)
        ):
            continue
        if old is None:
            assert new is not None
            direction = GoldContrastDirection.ADD
            operator = new.operator
            signature = new.signature
            anchors = new.anchor_ids
            evidence = new.evidence_ids
        elif new is None:
            direction = GoldContrastDirection.REMOVE
            operator = old.operator
            signature = old.signature
            anchors = old.anchor_ids
            evidence = old.evidence_ids
        else:
            direction = GoldContrastDirection.SUBSTITUTE
            operator = new.operator
            signature = f"{old.signature}=>{new.signature}"
            anchors = tuple(sorted(set(old.anchor_ids) | set(new.anchor_ids)))
            evidence = tuple(sorted(set(old.evidence_ids) | set(new.evidence_ids)))
        normalized = {
            "slot_key": slot_key,
            "operator": operator.value,
            "direction": direction.value,
            "signature": signature,
            "anchors": tuple(sorted(set(anchors))),
            "evidence": tuple(sorted(set(evidence))),
        }
        changes.append(
            GoldContrastDecision(
                contrast_decision_id=f"delta_{canonical_sha256(normalized)[:20]}",
                operator=operator,
                direction=direction,
                anchor_ids=normalized["anchors"],
                expected_signature=signature,
                evidence_ids=normalized["evidence"],
            )
        )
    return tuple(changes)


def normalized_gold_change(decision: GoldContrastDecision) -> dict[str, Any]:
    """Normalize an annotated change for exact compiler-delta regression checks."""

    return {
        "operator": decision.operator,
        "direction": decision.direction,
        "signature": decision.expected_signature,
        "anchor_ids": tuple(sorted(set(decision.anchor_ids))),
        "evidence_ids": tuple(sorted(set(decision.evidence_ids))),
    }


def semantic_atom_to_normalized_decision(atom: SemanticAtom):
    """Convert a compiler atom to the metrics-side normalized decision type."""

    from story_projection_onto.metrics.alignment import DecisionFamily, NormalizedDecision

    family = {
        ConstructionOperator.MERGE: DecisionFamily.MERGE_SPLIT,
        ConstructionOperator.SPLIT: DecisionFamily.MERGE_SPLIT,
        ConstructionOperator.CONTEXTUAL_TYPE: DecisionFamily.CONTEXTUAL_TYPE,
        ConstructionOperator.SCHEMA_RELATION: DecisionFamily.SCHEMA_RELATION,
        ConstructionOperator.EVENT_REIFICATION: DecisionFamily.EVENT_REIFICATION,
        ConstructionOperator.ABSTRACTION: DecisionFamily.ABSTRACTION,
        ConstructionOperator.TEMPORAL_QUALIFICATION: (
            DecisionFamily.TEMPORAL_EPISTEMIC_QUALIFICATION
        ),
        ConstructionOperator.EPISTEMIC_QUALIFICATION: (
            DecisionFamily.TEMPORAL_EPISTEMIC_QUALIFICATION
        ),
    }[atom.operator]
    return NormalizedDecision(
        slot_key=atom.slot_key,
        family=family,
        operator=atom.operator,
        anchor_ids=tuple(sorted(set(atom.anchor_ids) | set(atom.evidence_ids))),
        semantic_signature=atom.signature,
    )


def compile_alignment_alternatives(
    gold: GoldContextualProjection,
    alternatives: GoldAlternativeSet,
):
    """Compile every declared alternative into executable metric targets.

    The import is intentionally local so the scorer compiler remains usable for
    data generation without making the model-side runtime import metric code.
    """

    from story_projection_onto.metrics.alignment import (
        CompiledGoldAlternatives,
        ExplicitAssertionAlternative,
        PermissibleAssertionAlternative,
        QualifiedAssertionSignature,
    )

    if alternatives.gold_projection_id != gold.gold_projection_id:
        raise ValueError("cannot compile alternatives for a different gold projection")
    declared = tuple(item.alternative_id for item in alternatives.constraint_alternatives)
    if not declared or alternatives.permissible_projection_ids:
        raise ValueError("synthetic benchmark expects explicit constraint alternatives only")
    expected = _executable_alternative_values(gold)
    assertion_alternatives = []
    for alternative in alternatives.constraint_alternatives:
        constraints = {item.field_path: item for item in alternative.constraints}
        if len(constraints) != len(alternative.constraints):
            raise ValueError("executable alternative fields must be unique")
        required = {item.value for item in AlternativeField}
        if set(constraints) != required:
            raise ValueError("gold alternative must contain the exact typed executable fields")
        joint_values = constraints[AlternativeField.JOINT_REPRESENTATION.value].accepted_values
        if joint_values != (
            expected.primary_joint_representation_signature,
            expected.alternate_joint_representation_signature,
        ):
            raise ValueError("encoded alternative disagrees with its typed gold targets")
        alternate_payload = json.loads(joint_values[1])
        evidence_ids = tuple(json.loads(alternate_payload["assertion_evidence_signature"]))
        if evidence_ids != tuple(sorted(set(evidence_ids))):
            raise ValueError("encoded executable alternative IDs must be canonical")
        inverse_signature = QualifiedAssertionSignature.model_validate_json(
            alternate_payload["assertion_signature"]
        )
        assertion_alternatives.append(
            ExplicitAssertionAlternative(
                source_alternative_id=alternative.alternative_id,
                target_id=alternate_payload["assertion_target_id"],
                alternative=PermissibleAssertionAlternative(
                    alternative_id=f"{alternative.alternative_id}:inverse",
                    signature=inverse_signature,
                    supporting_evidence_ids=evidence_ids,
                ),
            )
        )
    compiled = CompiledGoldAlternatives(
        source_alternative_set_hash=alternatives.content_hash,
        consumed_alternative_ids=tuple(sorted(declared)),
        assertion_alternatives=tuple(assertion_alternatives),
    )
    return compiled


def _semantic_atom(
    *,
    slot_key: str,
    operator: ConstructionOperator,
    signature: str,
    evidence_ids: Sequence[str],
    narrative: NarrativeProducts,
    anchor_ids: Sequence[str] | None = None,
    object_ids: Sequence[str] = (),
) -> SemanticAtom:
    evidence = tuple(sorted(set(evidence_ids)))
    anchors = tuple(sorted(set(anchor_ids or _grounded_anchors_for_evidence(narrative, evidence))))
    return SemanticAtom(
        slot_key=slot_key,
        operator=operator,
        signature=signature,
        anchor_ids=anchors,
        evidence_ids=evidence,
        object_ids=tuple(sorted(set(object_ids))),
    )


def _fact_signature(fact: WorldFactSpec) -> str:
    end = "open" if fact.validity_end is None else str(fact.validity_end)
    holder = fact.holder_ref or "world"
    return (
        f"fact:{fact.subject_ref}|{fact.relation}|{fact.object_ref}|"
        f"{fact.story_position}:{end}|{fact.commitment.value}:{holder}"
    )


def _fact_assertion(
    *,
    spec: WorldSpec,
    context: QueryContext,
    fact: WorldFactSpec,
    assertion_id: str,
    evidence_ids: tuple[str, ...],
    evidence_by_id: Mapping[str, EvidenceRecord],
    revelation_order_by_evidence_id: Mapping[str, int],
    ref_to_cluster: Mapping[str, str],
    event_id_by_ref: Mapping[str, str],
    event_by_ref: Mapping[str, WorldEventSpec],
    active: bool,
    lens_relevant: bool,
    context_scope: StoryTime,
    viewpoint_holder_ref: str | None,
) -> tuple[QualifiedAssertion, str]:
    relation = fact.relation
    subject_id = ref_to_cluster[fact.subject_ref]
    object_id = ref_to_cluster.get(fact.object_ref) or event_id_by_ref.get(fact.object_ref)
    roles: tuple[RoleBinding, ...] = ()
    if fact.object_ref in event_by_ref and fact.object_ref not in event_id_by_ref:
        framed_event = event_by_ref[fact.object_ref]
        relation = f"unreified_{fact.relation}_frame"
        counterparty = next(
            (item for item in framed_event.participant_refs if item != fact.subject_ref),
            framed_event.participant_refs[0],
        )
        role_values = [
            ("initiator", subject_id),
            ("counterparty", ref_to_cluster[counterparty]),
            ("location", ref_to_cluster[framed_event.place_ref]),
        ]
        roles = tuple(
            RoleBinding(role=role, object_id=object_ref, evidence_ids=evidence_ids)
            for role, object_ref in role_values
        )
        subject_out: str | None = None
        object_out: str | None = None
    elif object_id is not None and fact.relation == "participates_in":
        roles = (
            RoleBinding(role="participant", object_id=subject_id, evidence_ids=evidence_ids),
            RoleBinding(role="event", object_id=object_id, evidence_ids=evidence_ids),
        )
        subject_out = None
        object_out = None
    else:
        if object_id is None:
            raise ValueError(f"fact endpoint has no contextual object: {fact.object_ref}")
        subject_out = subject_id
        object_out = object_id
    proposition_id: str | None = None
    epistemic: EpistemicScope | None = None
    commitment = NarrativeCommitment.WORLD_COMMITTED
    if fact.commitment in {NeutralCommitment.REPORTED, NeutralCommitment.DENIED}:
        if fact.holder_ref is None:
            raise ValueError("attributed fact lost its holder")
        proposition_id = f"{assertion_id}.proposition"
        epistemic = EpistemicScope(
            holder_id=ref_to_cluster[fact.holder_ref],
            attitude=(
                EpistemicAttitude.REPORTED
                if fact.commitment is NeutralCommitment.REPORTED
                else EpistemicAttitude.DENIED
            ),
            proposition_content_id=proposition_id,
            holder_relative_time=HolderRelativeTime(
                kind=TemporalKind.POINT,
                point=fact.story_position,
                label=f"holder state at story step {fact.story_position}",
            ),
            evidence_ids=evidence_ids,
        )
        commitment = (
            NarrativeCommitment.CONTESTED
            if fact.conflict_group is not None and viewpoint_holder_ref is None
            else NarrativeCommitment.HOLDER_ATTRIBUTED
        )
    elif fact.commitment is NeutralCommitment.CONTESTED:
        commitment = NarrativeCommitment.CONTESTED
    evidence = evidence_by_id[evidence_ids[0]]
    assertion = QualifiedAssertion(
        assertion_id=assertion_id,
        proposition_content_id=proposition_id,
        predicate_id=_predicate_id(context, relation),
        subject_id=subject_out,
        object_id=object_out,
        roles=roles,
        direction="forward",
        temporal_scope=_temporal_scope(
            fact,
            evidence,
            revelation_order_by_evidence_id[evidence.evidence_id],
            context_scope if active else None,
        ),
        epistemic_scope=epistemic,
        narrative_commitment=commitment,
        confidence=(0.72 if fact.conflict_group else 0.97),
        evidence_ids=evidence_ids,
        provenance=_frozen_scorer_provenance(evidence_ids, evidence_by_id),
        contextual_relevance=(0.95 if active and lens_relevant else 0.1),
        why_matters=(
            "This supported assertion bears on the requested answer."
            if active and lens_relevant
            else "This supported assertion is retained as controlled contextual background."
        ),
        why_matters_evidence_ids=evidence_ids,
    )
    return assertion, relation


def _frozen_scorer_provenance(
    evidence_ids: Sequence[str],
    evidence_by_id: Mapping[str, EvidenceRecord],
) -> tuple[ProvenanceReference, ...]:
    """Keep scorer-only gold bytes stable while evidence lineage is hardened.

    The query-blind index now carries the exact hash of each rendered synthetic
    passage.  That administrative source binding belongs in neutral and
    model-visible evidence artifacts; copying it into preregistered gold
    assertions would silently rewrite the scorer after its freeze.  Gold remains
    bound to the same exact evidence IDs, while runtime grounding uses the
    strengthened evidence record.
    """

    frozen: list[ProvenanceReference] = []
    for evidence_id in evidence_ids:
        payload = evidence_by_id[evidence_id].provenance.model_dump(
            mode="python",
            exclude={"content_hash", "source_artifact_hash"},
        )
        frozen.append(ProvenanceReference(**payload, source_artifact_hash=None))
    return tuple(frozen)


def _rebuild_gold_with_delta(
    projection: GoldContextualProjection,
    decisions: Sequence[GoldContrastDecision],
) -> GoldContextualProjection:
    data = _without_hashes(projection.model_dump(mode="python"))
    data["signed_contrast_decisions"] = [
        _without_hashes(item.model_dump(mode="python")) for item in decisions
    ]
    return GoldContextualProjection.model_validate(data)


def compile_gold_projection(
    spec: WorldSpec,
    context: QueryContext,
    context_index: int,
    narrative: NarrativeProducts,
    config: BenchmarkConfiguration,
    *,
    policy: CompilerPolicy,
    selected_for_review: bool,
    require_rare: bool = True,
) -> CompiledGold:
    """Compile one projection exclusively from world/query semantics."""

    # A fixed ontology constrains construction, never the meaning of the query.
    effective_lens = _lens_for_context(context)
    effective_scope = context.story_scope
    viewpoint_holder_ref = None
    if context.viewpoint is not None:
        viewpoint_holder_ref = next(
            item.semantic_ref
            for item in narrative.bindings
            if item.candidate_id == context.viewpoint.holder_id
        )
    clusters, ref_to_cluster, identity_atoms = _build_entity_partition(
        spec, context, policy, narrative
    )
    focal_event = next(item for item in spec.events if item.focal)
    focal_reified = _focal_reified(spec, context, policy)
    reified_event_specs = _reified_event_specs(spec, context, policy)
    event_id_by_ref = {
        item.event_ref: f"score.{context.context_id}.event.{index:02d}"
        for index, item in enumerate(reified_event_specs)
    }
    evidence_by_id = {item.evidence_id: item for item in narrative.evidence}
    event_specs_by_ref = {item.event_ref: item for item in spec.events}
    included_facts = tuple(
        item for item in spec.facts if narrative.fact_evidence_ids.get(item.fact_id)
    )
    fact_by_id = {item.fact_id: item for item in included_facts}
    assertions: list[QualifiedAssertion] = []
    assertion_fact: dict[str, WorldFactSpec] = {}
    relation_shapes: dict[str, tuple[int, tuple[str, ...]]] = {}
    event_support: dict[str, list[str]] = defaultdict(list)
    for fact in included_facts:
        active = _fact_overlaps_scope(fact, effective_scope)
        lens_relevant = effective_lens in fact.relevant_lenses
        if viewpoint_holder_ref is not None and fact.commitment in {
            NeutralCommitment.REPORTED,
            NeutralCommitment.DENIED,
        }:
            lens_relevant = lens_relevant and fact.holder_ref == viewpoint_holder_ref
        assertion_id = f"score.{context.context_id}.assertion.fact.{len(assertions):03d}"
        assertion, actual_relation = _fact_assertion(
            spec=spec,
            context=context,
            fact=fact,
            assertion_id=assertion_id,
            evidence_ids=narrative.fact_evidence_ids[fact.fact_id],
            evidence_by_id=evidence_by_id,
            revelation_order_by_evidence_id=narrative.revelation_order_by_evidence_id,
            ref_to_cluster=ref_to_cluster,
            event_id_by_ref=event_id_by_ref,
            event_by_ref=event_specs_by_ref,
            active=active,
            lens_relevant=lens_relevant,
            context_scope=effective_scope,
            viewpoint_holder_ref=viewpoint_holder_ref,
        )
        assertions.append(assertion)
        assertion_fact[assertion_id] = fact
        if assertion.roles:
            relation_shapes[actual_relation] = (
                len(assertion.roles),
                tuple(item.role for item in assertion.roles),
            )
        else:
            relation_shapes[actual_relation] = (2, ())
        if fact.event_ref in event_id_by_ref:
            event_support[fact.event_ref].append(assertion_id)

    # Causal dependencies are explicit world semantics and have their own evidence witnesses.
    causal_assertion_ids: dict[tuple[str, str], str] = {}
    for left_id, right_id in spec.causal_dependencies:
        if left_id not in fact_by_id or right_id not in fact_by_id:
            continue
        evidence_ids = narrative.causal_evidence_ids[(left_id, right_id)]
        if not evidence_ids:
            continue
        left, right = fact_by_id[left_id], fact_by_id[right_id]
        assertion_id = (
            f"score.{context.context_id}.assertion.causal.{len(causal_assertion_ids):02d}"
        )
        evidence = evidence_by_id[evidence_ids[0]]
        active = _fact_overlaps_scope(left, effective_scope) and _fact_overlaps_scope(
            right, effective_scope
        )
        assertions.append(
            QualifiedAssertion(
                assertion_id=assertion_id,
                predicate_id=_predicate_id(context, "causally_enables"),
                subject_id=ref_to_cluster[left.subject_ref],
                object_id=ref_to_cluster[right.subject_ref],
                direction="forward",
                temporal_scope=TemporalScope(
                    story_time=StoryTime(kind=TemporalKind.POINT, point=right.story_position),
                    validity_time=_contextual_validity_time(
                        left.story_position,
                        right.story_position,
                        effective_scope if active else None,
                    ),
                    discourse_position=evidence.discourse_position,
                    revelation_position=RevelationPosition(
                        revelation_order=narrative.revelation_order_by_evidence_id[
                            evidence.evidence_id
                        ]
                    ),
                ),
                narrative_commitment=NarrativeCommitment.WORLD_COMMITTED,
                confidence=0.98,
                evidence_ids=evidence_ids,
                provenance=_frozen_scorer_provenance(evidence_ids, evidence_by_id),
                contextual_relevance=(
                    0.97
                    if active and effective_lens is LensFamily.CAUSAL_CONSEQUENCE
                    else 0.55
                    if active
                    else 0.1
                ),
                why_matters="The evidence explicitly connects two steps in the causal chain.",
                why_matters_evidence_ids=evidence_ids,
            )
        )
        causal_assertion_ids[(left_id, right_id)] = assertion_id
        relation_shapes["causally_enables"] = (2, ())

    # Temporal precedence is represented whether or not the focal occurrence is reified.
    temporal_assertion_ids: dict[tuple[str, str], str] = {}
    event_by_ref = {item.event_ref: item for item in spec.events}
    for left_ref, right_ref in spec.temporal_precedence:
        evidence_ids = narrative.temporal_evidence_ids[(left_ref, right_ref)]
        if not evidence_ids:
            continue
        left, right = event_by_ref[left_ref], event_by_ref[right_ref]
        assertion_id = (
            f"score.{context.context_id}.assertion.temporal.{len(temporal_assertion_ids):02d}"
        )
        evidence = evidence_by_id[evidence_ids[0]]
        if left_ref in event_id_by_ref and right_ref in event_id_by_ref:
            subject_id, object_id, roles = event_id_by_ref[left_ref], event_id_by_ref[right_ref], ()
            relation = "precedes_event"
            relation_shapes[relation] = (2, ())
        else:
            subject_id = object_id = None
            roles = (
                RoleBinding(
                    role="earlier_participant",
                    object_id=ref_to_cluster[left.participant_refs[0]],
                    evidence_ids=evidence_ids,
                ),
                RoleBinding(
                    role="later_participant",
                    object_id=ref_to_cluster[right.participant_refs[0]],
                    evidence_ids=evidence_ids,
                ),
                RoleBinding(
                    role="shared_timeline",
                    object_id=ref_to_cluster[right.place_ref],
                    evidence_ids=evidence_ids,
                ),
            )
            relation = "event_precedence_frame"
            relation_shapes[relation] = (3, tuple(item.role for item in roles))
        active = _event_overlaps_scope(left, effective_scope) or _event_overlaps_scope(
            right, effective_scope
        )
        assertions.append(
            QualifiedAssertion(
                assertion_id=assertion_id,
                predicate_id=_predicate_id(context, relation),
                subject_id=subject_id,
                object_id=object_id,
                roles=roles,
                direction="forward",
                temporal_scope=TemporalScope(
                    story_time=StoryTime(kind=TemporalKind.POINT, point=right.story_position),
                    validity_time=_contextual_validity_time(
                        left.story_position,
                        right.story_position,
                        effective_scope if active else None,
                    ),
                    discourse_position=evidence.discourse_position,
                    revelation_position=RevelationPosition(
                        revelation_order=narrative.revelation_order_by_evidence_id[
                            evidence.evidence_id
                        ]
                    ),
                ),
                narrative_commitment=NarrativeCommitment.WORLD_COMMITTED,
                confidence=0.99,
                evidence_ids=evidence_ids,
                provenance=_frozen_scorer_provenance(evidence_ids, evidence_by_id),
                contextual_relevance=(
                    0.96
                    if active and effective_lens is LensFamily.MOVEMENT_TIME
                    else 0.5
                    if active
                    else 0.1
                ),
                why_matters=(
                    "The evidence explicitly fixes event order without conflating it "
                    "with discourse order."
                ),
                why_matters_evidence_ids=evidence_ids,
            )
        )
        temporal_assertion_ids[(left_ref, right_ref)] = assertion_id

    abstraction = (
        AbstractionLevel.ACTOR if policy is CompilerPolicy.FIXED_REFERENCE else context.abstraction
    )
    schema_variant, focus_relation = _schema_label(
        effective_lens,
        policy,
        fixed_friendly_profile=(
            BenchmarkFactor.FIXED_ONTOLOGY_FRIENDLY in spec.factors
            and context.abstraction is AbstractionLevel.ACTOR
        ),
    )
    if focus_relation not in relation_shapes:
        if effective_lens is LensFamily.MOVEMENT_TIME:
            focus_relation = (
                "event_precedence_frame"
                if "event_precedence_frame" in relation_shapes
                else "precedes_event"
            )
        else:
            focus_relation = sorted(relation_shapes)[0]
    type_ids = {
        "person": f"score.{context.context_id}.type.{schema_variant}.person",
        "collective": f"score.{context.context_id}.type.{schema_variant}.collective",
        "place": f"score.{context.context_id}.type.{schema_variant}.place",
        "event": f"score.{context.context_id}.type.{schema_variant}.event",
    }

    def type_evidence(kind: str) -> str:
        for evidence in narrative.evidence:
            mention_match = any(
                item.provisional_type == f"{kind}_candidate" for item in evidence.mention_candidates
            )
            if mention_match or (kind == "event" and evidence.event_candidates):
                return evidence.evidence_id
        raise ValueError(f"local schema type {kind!r} has no matching evidence candidate")

    contextual_types = tuple(
        LocalTypeDefinition(
            type_id=type_id,
            label=f"{schema_variant} {kind}",
            definition=f"Evidence-grounded {kind} interpreted for the {schema_variant} schema.",
            parent_upper_type=f"upper.{kind}",
            abstraction=abstraction,
            evidence_ids=(type_evidence(kind),),
        )
        for kind, type_id in type_ids.items()
    )
    predicates = tuple(
        LocalPredicateDefinition(
            predicate_id=_predicate_id(context, relation),
            label=relation.replace("_", " "),
            definition=(
                f"Evidence-grounded {relation.replace('_', ' ')} under the {schema_variant} schema."
            ),
            arity=shape[0],
            role_names=shape[1],
            parent_upper_relation=("upper.causal" if "causal" in relation else "upper.association"),
            evidence_ids=tuple(
                sorted(
                    {
                        evidence_id
                        for assertion in assertions
                        if assertion.predicate_id == _predicate_id(context, relation)
                        for evidence_id in assertion.evidence_ids
                    }
                )
            ),
        )
        for relation, shape in sorted(relation_shapes.items())
    )
    schema = LocalContextSchema(
        schema_id=f"score.{context.context_id}.schema.{schema_variant}",
        contextual_types=contextual_types,
        predicates=predicates,
        abstraction=abstraction,
    )

    events: list[Event] = []
    for event_spec in reified_event_specs:
        supporting = tuple(event_support[event_spec.event_ref])
        if not supporting:
            supporting = tuple(
                assertion_id
                for edge, assertion_id in temporal_assertion_ids.items()
                if event_spec.event_ref in edge
            )
        if not supporting:
            raise ValueError("reified event has no supported assertion")
        evidence_ids = tuple(
            sorted(
                {
                    evidence_id
                    for assertion in assertions
                    if assertion.assertion_id in supporting
                    for evidence_id in assertion.evidence_ids
                }
            )
        )
        events.append(
            Event(
                event_id=event_id_by_ref[event_spec.event_ref],
                label=event_spec.label,
                contextual_type_id=type_ids["event"],
                occurrence_time=StoryTime(
                    kind=TemporalKind.INTERVAL,
                    start=event_spec.story_position,
                    end=event_spec.story_position + event_spec.duration,
                    label=(
                        f"story steps {event_spec.story_position} through "
                        f"{event_spec.story_position + event_spec.duration}"
                    ),
                ),
                reification_reason=(
                    "Independent duration and participant roles require an event object "
                    "in this context."
                ),
                uncertainty=ExplicitValueState.KNOWN,
                confidence=0.96,
                evidence_ids=evidence_ids,
                description=(
                    "An evidence-supported occurrence with independently qualified roles and time."
                ),
                description_assertion_ids=supporting,
            )
        )

    if not 10 <= len(clusters) + len(events) <= 20:
        raise ValueError("gold projection left the registered 10--20 node range")
    if len(assertions) > context.budgets.assertion_budget:
        raise ValueError("gold assertions exceed the shared assertion budget")

    fact_relevance: dict[str, bool] = {}
    for assertion_id, fact in assertion_fact.items():
        holder_compatible = (
            viewpoint_holder_ref is None
            or fact.commitment not in {NeutralCommitment.REPORTED, NeutralCommitment.DENIED}
            or fact.holder_ref == viewpoint_holder_ref
        )
        fact_relevance[assertion_id] = (
            _fact_overlaps_scope(fact, effective_scope)
            and holder_compatible
            and effective_lens in fact.relevant_lenses
        )
    assertion_relevance = {
        item.assertion_id: (fact_relevance.get(item.assertion_id, item.contextual_relevance >= 0.9))
        for item in assertions
    }
    relevant_endpoints = (
        set().union(
            *(
                _assertion_endpoints(item)
                for item in assertions
                if assertion_relevance[item.assertion_id]
            )
        )
        if assertions
        else set()
    )
    relevance = [
        GoldRelevanceAnnotation(
            target_id=cluster.cluster_id,
            target_kind=GoldTargetKind.ENTITY_CLUSTER,
            is_relevant=cluster.cluster_id in relevant_endpoints,
        )
        for cluster in clusters
    ]
    relevance.extend(
        GoldRelevanceAnnotation(
            target_id=event.event_id,
            target_kind=GoldTargetKind.EVENT,
            is_relevant=(
                event.event_id in relevant_endpoints
                and _event_overlaps_scope(
                    next(
                        item
                        for item in spec.events
                        if event_id_by_ref.get(item.event_ref) == event.event_id
                    ),
                    effective_scope,
                )
            ),
        )
        for event in events
    )
    relevance.extend(
        GoldRelevanceAnnotation(
            target_id=item.assertion_id,
            target_kind=GoldTargetKind.ASSERTION,
            is_relevant=assertion_relevance[item.assertion_id],
        )
        for item in assertions
    )

    rare_fact = next(
        (item for item in included_facts if item.role is NeutralFactRole.RARE_PIVOTAL),
        None,
    )
    rare_assertion_id = next(
        (
            assertion_id
            for assertion_id, fact in assertion_fact.items()
            if fact.role is NeutralFactRole.RARE_PIVOTAL
        ),
        None,
    )
    if require_rare and (rare_fact is None or rare_assertion_id is None):
        raise ValueError("projection lacks its registered rare-pivotal assertion")
    rare_is_pivotal = bool(rare_assertion_id is not None and assertion_relevance[rare_assertion_id])
    support_path: tuple[str, ...] = ()
    typed_support_path: RareSupportPath | None = None
    if rare_assertion_id is not None and rare_is_pivotal:
        assertion_id_by_fact = {
            fact.fact_id: assertion_id for assertion_id, fact in assertion_fact.items()
        }
        steps = tuple(
            RareSupportPathStep(
                source_fact_id=left_id,
                target_fact_id=right_id,
                source_assertion_id=assertion_id_by_fact[left_id],
                causal_assertion_id=causal_assertion_ids[(left_id, right_id)],
                target_assertion_id=assertion_id_by_fact[right_id],
                evidence_ids=narrative.causal_evidence_ids[(left_id, right_id)],
            )
            for left_id, right_id in spec.causal_dependencies
        )
        if len(steps) != 2:
            raise ValueError("rare-pivotal proof requires the exact two-edge causal chain")
        support_path = (
            steps[0].source_assertion_id,
            steps[0].causal_assertion_id,
            steps[0].target_assertion_id,
            steps[1].causal_assertion_id,
            steps[1].target_assertion_id,
        )
        typed_support_path = RareSupportPath(
            path_id=f"score.{context.context_id}.rare-support-path",
            query_id=context.context_id,
            rare_assertion_id=rare_assertion_id,
            outcome_assertion_id=steps[-1].target_assertion_id,
            impact_kind=spec.rare_impact_kind,
            steps=steps,  # type: ignore[arg-type]
        )
    annotations = tuple(
        GoldAssertionAnnotation(
            assertion_id=item.assertion_id,
            is_rare=item.assertion_id == rare_assertion_id,
            is_pivotal=item.assertion_id == rare_assertion_id and rare_is_pivotal,
            support_path_assertion_ids=(
                support_path if item.assertion_id == rare_assertion_id and rare_is_pivotal else ()
            ),
        )
        for item in assertions
    )

    communities, community_rationales, community_signature = _community_semantics(
        spec,
        context,
        policy,
        narrative,
        clusters,
        ref_to_cluster,
        event_id_by_ref,
        included_facts,
    )

    all_atoms = list(identity_atoms)
    first_ev = narrative.evidence[0]
    first_anchor = next(iter(sorted(_candidate_ids(first_ev))))
    focus_predicate = next(
        item for item in predicates if item.predicate_id == _predicate_id(context, focus_relation)
    )
    all_atoms.extend(
        (
            _semantic_atom(
                slot_key="abstraction/root",
                operator=ConstructionOperator.ABSTRACTION,
                signature=f"abstraction:{abstraction.value}",
                evidence_ids=(first_ev.evidence_id,),
                anchor_ids=(first_anchor,),
                narrative=narrative,
            ),
            _semantic_atom(
                slot_key="schema/focus-relation",
                operator=ConstructionOperator.SCHEMA_RELATION,
                signature=f"schema:{schema_variant}|relation:{focus_relation}",
                evidence_ids=focus_predicate.evidence_ids,
                narrative=narrative,
                object_ids=(focus_predicate.predicate_id,),
            ),
        )
    )
    focal_evidence = tuple(
        evidence_id
        for fact in included_facts
        if fact.event_ref == focal_event.event_ref
        for evidence_id in narrative.fact_evidence_ids.get(fact.fact_id, ())
    )
    all_atoms.append(
        _semantic_atom(
            slot_key="event/focal-representation",
            operator=ConstructionOperator.EVENT_REIFICATION,
            signature=(
                "event-object-with-qualified-roles"
                if focal_reified
                else "qualified-nary-event-relation"
            ),
            evidence_ids=focal_evidence,
            narrative=narrative,
            object_ids=(
                (event_id_by_ref[focal_event.event_ref],)
                if focal_reified
                else tuple(
                    item.assertion_id
                    for item in assertions
                    if item.predicate_id.startswith(
                        f"score.{context.context_id}.predicate.unreified-"
                    )
                    and set(item.evidence_ids).intersection(focal_evidence)
                )
            ),
        )
    )
    scope_signature = (
        f"{effective_scope.kind.value}:{effective_scope.point}:"
        f"{effective_scope.start}:{effective_scope.end}"
    )
    all_atoms.append(
        _semantic_atom(
            slot_key="qualification/story-scope",
            operator=ConstructionOperator.TEMPORAL_QUALIFICATION,
            signature=f"story-scope:{scope_signature}",
            evidence_ids=(first_ev.evidence_id,),
            anchor_ids=(first_ev.temporal_clues[0].clue_id,),
            narrative=narrative,
            object_ids=tuple(item.assertion_id for item in assertions),
        )
    )
    if context.viewpoint is None:
        epistemic_signature = "holder-frame:all-attributed-holders"
    else:
        epistemic_signature = f"holder-frame:{context.viewpoint.holder_id}"
    report_evidence = tuple(
        evidence_id
        for fact in included_facts
        if fact.commitment in {NeutralCommitment.REPORTED, NeutralCommitment.DENIED}
        for evidence_id in narrative.fact_evidence_ids[fact.fact_id]
    )
    all_atoms.append(
        _semantic_atom(
            slot_key="qualification/epistemic-frame",
            operator=ConstructionOperator.EPISTEMIC_QUALIFICATION,
            signature=epistemic_signature,
            evidence_ids=report_evidence,
            anchor_ids=(
                (context.viewpoint.holder_id,)
                if context.viewpoint is not None
                else _grounded_anchors_for_evidence(narrative, report_evidence, limit=1)
            ),
            narrative=narrative,
            object_ids=tuple(
                item.assertion_id
                for item in assertions
                if item.epistemic_scope is not None
                and (
                    viewpoint_holder_ref is None
                    or item.epistemic_scope.holder_id == ref_to_cluster[viewpoint_holder_ref]
                )
            ),
        )
    )
    # Explicitly consume every registered causal dependency and event predecessor.
    for dependency, evidence_ids in narrative.causal_evidence_ids.items():
        if dependency[0] in fact_by_id and dependency[1] in fact_by_id and evidence_ids:
            all_atoms.append(
                _semantic_atom(
                    slot_key=(
                        f"causal/{dependency[0].rsplit('.', 1)[-1]}-"
                        f"{dependency[1].rsplit('.', 1)[-1]}"
                    ),
                    operator=ConstructionOperator.SCHEMA_RELATION,
                    signature=f"causal-dependency:{dependency[0]}->{dependency[1]}",
                    evidence_ids=evidence_ids,
                    narrative=narrative,
                )
            )
    for event in spec.events:
        if event.causal_predecessor_ref is None:
            continue
        edge = (event.causal_predecessor_ref, event.event_ref)
        evidence_ids = narrative.temporal_evidence_ids.get(edge, ())
        if evidence_ids:
            all_atoms.append(
                _semantic_atom(
                    slot_key=f"event-predecessor/{event.event_ref.rsplit('.', 1)[-1]}",
                    operator=ConstructionOperator.TEMPORAL_QUALIFICATION,
                    signature=(
                        f"event-predecessor:{event.causal_predecessor_ref}->{event.event_ref}"
                    ),
                    evidence_ids=evidence_ids,
                    narrative=narrative,
                    object_ids=(temporal_assertion_ids[edge],),
                )
            )
    for persona in spec.personas:
        membership_fact = next(
            fact
            for fact in included_facts
            if fact.subject_ref == persona.persona_id and fact.relation == "member_of"
        )
        all_atoms.append(
            _semantic_atom(
                slot_key=f"home-collective/{persona.persona_id.rsplit('.', 1)[-1]}",
                operator=ConstructionOperator.CONTEXTUAL_TYPE,
                signature=f"home-collective:{persona.home_collective_id}",
                evidence_ids=narrative.fact_evidence_ids[membership_fact.fact_id],
                narrative=narrative,
            )
        )
    conflict_groups: dict[str, list[WorldFactSpec]] = defaultdict(list)
    for fact in included_facts:
        if fact.conflict_group is not None:
            conflict_groups[fact.conflict_group].append(fact)
    for group, facts in conflict_groups.items():
        evidence_ids = tuple(
            evidence_id
            for fact in facts
            for evidence_id in narrative.fact_evidence_ids[fact.fact_id]
        )
        all_atoms.append(
            _semantic_atom(
                slot_key=f"conflict/{group.rsplit('.', 1)[-1]}",
                operator=ConstructionOperator.EPISTEMIC_QUALIFICATION,
                signature="contested-holder-claims:"
                + "+".join(sorted(item.commitment.value for item in facts)),
                evidence_ids=evidence_ids,
                narrative=narrative,
            )
        )
    if len({item.slot_key for item in all_atoms}) != len(all_atoms):
        raise ValueError("semantic decision slot keys must be unique within a projection")

    relevant_facts = tuple(
        sorted(
            _fact_signature(fact)
            for fact in included_facts
            if _fact_overlaps_scope(fact, effective_scope)
            and effective_lens in fact.relevant_lenses
        )
    )
    causal_paths = tuple(
        sorted(
            f"{left}->{right}"
            for left, right in spec.causal_dependencies
            if left in fact_by_id and right in fact_by_id
        )
    )
    temporal_items = {f"{left}<{right}" for left, right in spec.temporal_precedence}
    temporal_items.update(
        f"{fact.fact_id}:signal<{fact.object_ref}:occurrence"
        for fact in included_facts
        if fact.relation == "signals_before"
    )
    temporal_order = tuple(sorted(temporal_items))
    states = tuple(
        sorted(
            _fact_signature(fact)
            for fact in included_facts
            if fact.relation in {"holds_office", "member_of", "defects_to"}
            and _fact_overlaps_scope(fact, effective_scope)
        )
    )
    answer_signature = AnswerSignature(
        answer_facts=relevant_facts,
        causal_paths=causal_paths,
        temporal_order=temporal_order,
        consequential_states=states,
        community_partition=community_signature,
    )
    invariant_evidence = (
        narrative.fact_evidence_ids[rare_fact.fact_id]
        if rare_fact is not None
        else (first_ev.evidence_id,)
    )
    invariant_anchors = _grounded_anchors_for_evidence(narrative, invariant_evidence, limit=2)
    review_status = (
        GoldReviewStatus.PENDING if selected_for_review else GoldReviewStatus.NOT_SELECTED
    )
    adjudication_status = (
        GoldAdjudicationStatus.PENDING
        if selected_for_review
        else GoldAdjudicationStatus.NOT_REQUIRED
    )
    projection = GoldContextualProjection(
        gold_projection_id=f"score.{context.context_id}.projection",
        split=spec.split,
        world_id=spec.world_id,
        query_id=context.context_id,
        local_schema=schema,
        entity_partition=clusters,
        events=tuple(events),
        qualified_assertions=tuple(assertions),
        relevance=tuple(relevance),
        assertion_annotations=annotations,
        communities=communities,
        signed_contrast_decisions=(),
        contrast_invariants=(
            GoldContrastInvariant(
                invariant_id=f"score.{context.context_id}.invariant.rare",
                anchor_ids=invariant_anchors,
                expected_signature=(
                    _fact_signature(rare_fact)
                    if rare_fact is not None
                    else "rare-fact-deleted-mutation"
                ),
                evidence_ids=invariant_evidence,
            ),
        ),
        matcher_revision=config.matcher_revision,
        review_status=review_status,
        adjudication_status=adjudication_status,
        compiled_at=_timestamp(config, 120 + context_index),
    )
    candidate = _alternative_view(projection)
    executable = _executable_alternative_values(projection)
    # The primary and inverse forms move as one atomic representation. No field-wise
    # Cartesian product is permitted.
    alternative = GoldConstraintAlternative(
        alternative_id=f"score.{context.context_id}.alternative.normalized",
        description=(
            "The same binary assertion may be encoded in inverse direction only when its "
            "endpoints are swapped; all partitions, evidence, temporal, and epistemic fields "
            "remain identical."
        ),
        constraints=(
            GoldMatchingConstraint(
                field_path=AlternativeField.JOINT_REPRESENTATION.value,
                operator=GoldConstraintOperator.ONE_OF,
                accepted_values=(
                    executable.primary_joint_representation_signature,
                    executable.alternate_joint_representation_signature,
                ),
            ),
        ),
    )
    alternatives = GoldAlternativeSet(
        alternative_set_id=f"score.{context.context_id}.alternatives",
        gold_projection_id=projection.gold_projection_id,
        constraint_alternatives=(alternative,),
        equivalence_rule=(
            "One closed joint representation: inverse direction requires swapped endpoints "
            "and preserves every other semantic field."
        ),
        matching_rule=(
            "Compare the complete canonical joint signature; partial hybrids are invalid."
        ),
        matcher_revision=config.matcher_revision,
        review_status=review_status,
        adjudication_status=adjudication_status,
        compiled_at=_timestamp(config, 124 + context_index),
    )
    if not evaluate_alternative_set(alternatives, candidate).matched:
        raise ValueError("primary gold candidate must satisfy its executable alternative set")
    return CompiledGold(
        projection=projection,
        alternatives=alternatives,
        semantic_atoms=tuple(sorted(all_atoms, key=lambda item: item.slot_key)),
        answer_signature=answer_signature,
        community_rationales=community_rationales,
        alternative_candidate=candidate,
        rare_support_path=typed_support_path,
    )


class ReviewCriterion(StrEnum):
    EVIDENCE_SUPPORT = "evidence_support"
    IDENTITY_PARTITIONS = "identity_partitions"
    EVENT_CHOICES = "event_choices"
    TEMPORAL_SCOPE = "temporal_scope"
    CONTRAST_DELTAS = "contrast_deltas"
    RARE_PIVOTAL_LABELS = "rare_pivotal_labels"
    COMMUNITIES = "communities"
    ALTERNATIVES = "alternatives"


class ReviewDisposition(StrEnum):
    AGREE = "agree"
    DISAGREE = "disagree"
    UNCERTAIN = "uncertain"


class AdjudicationDisposition(StrEnum):
    RETAIN = "retain"
    AMEND = "amend"
    EXCLUDE = "exclude"


class ReviewPromptItem(ImmutableRecord):
    review_item_id: Identifier
    blind_projection_id: Identifier
    criterion: ReviewCriterion
    prompt: str


class BlindReviewProjection(ImmutableRecord):
    blind_projection_id: Identifier
    context_label: Literal["A", "B", "C"]
    query: dict[str, Any]
    proposed_local_schema: dict[str, Any]
    proposed_identity_partitions: tuple[dict[str, Any], ...]
    proposed_events: tuple[dict[str, Any], ...]
    proposed_qualified_assertions: tuple[dict[str, Any], ...]
    proposed_relevance: tuple[dict[str, Any], ...]
    proposed_rare_pivotal_labels: tuple[dict[str, Any], ...]
    proposed_communities: tuple[dict[str, Any], ...]
    proposed_community_rationales: tuple[dict[str, Any], ...]
    proposed_answer_signature: dict[str, Any]
    proposed_rare_support_path: dict[str, Any] | None
    proposed_contrast_decisions: tuple[dict[str, Any], ...]
    proposed_contrast_invariants: tuple[dict[str, Any], ...]
    permissible_alternatives: dict[str, Any]
    review_items: tuple[ReviewPromptItem, ...]

    @model_validator(mode="after")
    def exact_criteria(self) -> Self:
        if {item.criterion for item in self.review_items} != set(ReviewCriterion):
            raise ValueError("each reviewed projection requires the exact eight criteria")
        if any(item.blind_projection_id != self.blind_projection_id for item in self.review_items):
            raise ValueError("review item projection binding changed")
        return self


class BlindReviewWorld(ImmutableRecord):
    blind_world_id: Identifier
    evidence: tuple[dict[str, Any], ...]
    projections: tuple[BlindReviewProjection, BlindReviewProjection, BlindReviewProjection]


class BlindIndependentReviewPackage(ImmutableRecord):
    package_id: Identifier
    selection_rule: Literal["one-seeded-world-per-difficulty-stratum"]
    lifecycle_state: Literal["draft_for_external_review"] = "draft_for_external_review"
    condition_blind: Literal[True] = True
    contains_method_outputs: Literal[False] = False
    reviewer_instructions: tuple[str, ...]
    worlds: tuple[BlindReviewWorld, BlindReviewWorld, BlindReviewWorld]

    @model_validator(mode="after")
    def exact_cartesian_rubric(self) -> Self:
        projections = [projection for world in self.worlds for projection in world.projections]
        if len(projections) != 9:
            raise ValueError("review package must contain exactly nine projections")
        items = [item for projection in projections for item in projection.review_items]
        expected = {
            (projection.blind_projection_id, criterion)
            for projection in projections
            for criterion in ReviewCriterion
        }
        actual = {(item.blind_projection_id, item.criterion) for item in items}
        if (
            len(items) != 72
            or actual != expected
            or len({item.review_item_id for item in items}) != 72
        ):
            raise ValueError("review package must contain the exact 9x8 Cartesian rubric")
        return self


class ReviewSelectionEntry(ImmutableRecord):
    difficulty: Difficulty
    world_id: Identifier
    blind_world_id: Identifier


class IndependentReviewSelectionManifest(ImmutableRecord):
    selection_seed: Annotated[int, Field(ge=0)]
    rule: Literal["uniform-one-within-each-difficulty-stratum"]
    selected: tuple[ReviewSelectionEntry, ReviewSelectionEntry, ReviewSelectionEntry]
    lifecycle_state: Literal["selected-before-condition-output"] = (
        "selected-before-condition-output"
    )
    condition_outputs_generated: Literal[False] = False


class IndependentReviewResponseItem(ImmutableRecord):
    review_item_id: Identifier
    blind_projection_id: Identifier
    criterion: ReviewCriterion
    disposition: ReviewDisposition
    notes: str


class ProposedAlternativeEdit(ImmutableRecord):
    blind_projection_id: Identifier
    rationale: str
    proposed_joint_representation_signatures: tuple[str, ...]


class IndependentReviewResponse(ImmutableRecord):
    package_hash: Sha256Digest
    reviewer_pseudonym: Identifier
    reviewed_at: AwareDatetime
    items: tuple[IndependentReviewResponseItem, ...]
    proposed_alternative_edits: tuple[ProposedAlternativeEdit, ...] = ()

    @model_validator(mode="after")
    def response_has_unique_cartesian_keys(self) -> Self:
        keys = [(item.blind_projection_id, item.criterion) for item in self.items]
        if len(self.items) != 72 or len(set(keys)) != 72:
            raise ValueError("response must contain 72 unique projection/criterion items")
        if len({item.review_item_id for item in self.items}) != 72:
            raise ValueError("response review item IDs must be unique")
        if len({item.blind_projection_id for item in self.proposed_alternative_edits}) != len(
            self.proposed_alternative_edits
        ):
            raise ValueError("proposed alternative edits must be unique by blind projection")
        return self


class ReviewAdjudicationItem(ImmutableRecord):
    review_item_id: Identifier
    disposition: AdjudicationDisposition
    rationale: str
    amended_artifact_hash: Sha256Digest | None = None

    @model_validator(mode="after")
    def amendment_has_hash(self) -> Self:
        if (self.disposition is AdjudicationDisposition.AMEND) != (
            self.amended_artifact_hash is not None
        ):
            raise ValueError("only amended adjudications carry a replacement artifact hash")
        return self


class ReviewAdjudication(ImmutableRecord):
    package_hash: Sha256Digest
    response_hash: Sha256Digest
    adjudicator_pseudonym: Identifier
    adjudicated_at: AwareDatetime
    items: tuple[ReviewAdjudicationItem, ...]


class ReviewProjectionBinding(ImmutableRecord):
    blind_projection_id: Identifier
    blind_projection_hash: Sha256Digest
    world_id: Identifier
    query_id: Identifier
    source_gold_projection_hash: Sha256Digest
    source_alternative_set_hash: Sha256Digest
    source_semantic_hash: Sha256Digest


class ReviewProjectionBindingManifest(ImmutableRecord):
    package_hash: Sha256Digest
    selection_manifest_hash: Sha256Digest
    entries: tuple[ReviewProjectionBinding, ...]

    @model_validator(mode="after")
    def exact_nine_unique_bindings(self) -> Self:
        if len(self.entries) != 9:
            raise ValueError("review binding manifest requires exactly nine projections")
        if len({item.blind_projection_id for item in self.entries}) != 9:
            raise ValueError("review binding projection IDs must be unique")
        if len({item.query_id for item in self.entries}) != 9:
            raise ValueError("review binding query IDs must be unique")
        return self


_REVIEW_WORKFLOW_FIELDS = frozenset(
    {
        "content_hash",
        "review_status",
        "adjudication_status",
        "independent_review_record_hash",
        "adjudication_record_hash",
    }
)


def _review_semantic_payload(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {
            str(key): _review_semantic_payload(child)
            for key, child in value.items()
            if key not in _REVIEW_WORKFLOW_FIELDS
        }
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [_review_semantic_payload(child) for child in value]
    return value


def reviewed_semantic_hash(gold: GoldContextualProjection, alternatives: GoldAlternativeSet) -> str:
    return canonical_sha256(
        {
            "gold": _review_semantic_payload(gold.model_dump(mode="python")),
            "alternatives": _review_semantic_payload(alternatives.model_dump(mode="python")),
        }
    )


class ReviewedProjectionArtifact(ImmutableRecord):
    blind_projection_id: Identifier
    source_gold_projection_hash: Sha256Digest
    source_alternative_set_hash: Sha256Digest
    response_hash: Sha256Digest
    adjudication_hash: Sha256Digest
    final_semantic_hash: Sha256Digest
    gold_projection: GoldContextualProjection
    alternatives: GoldAlternativeSet

    @model_validator(mode="after")
    def bind_final_semantics(self) -> Self:
        if self.gold_projection.gold_projection_id != self.alternatives.gold_projection_id:
            raise ValueError("reviewed gold and alternatives name different projections")
        if (
            reviewed_semantic_hash(self.gold_projection, self.alternatives)
            != self.final_semantic_hash
        ):
            raise ValueError("reviewed artifact semantic hash is not canonical")
        if self.gold_projection.review_status not in {
            GoldReviewStatus.REVIEWED,
            GoldReviewStatus.DISAGREEMENT_LOGGED,
        }:
            raise ValueError("reviewed artifact must carry completed review status")
        return self


class ReviewLifecycleError(RuntimeError):
    """Raised when a held-out review lifecycle binding is incomplete or inconsistent."""


def _package_review_items(
    package: BlindIndependentReviewPackage,
) -> dict[str, ReviewPromptItem]:
    return {
        item.review_item_id: item
        for world in package.worlds
        for projection in world.projections
        for item in projection.review_items
    }


def bind_review_response(
    package: BlindIndependentReviewPackage,
    response: IndependentReviewResponse,
) -> dict[str, IndependentReviewResponseItem]:
    """Bind a response to the exact package hash and exact 9x8 item identities."""

    if response.package_hash != package.content_hash:
        raise ReviewLifecycleError("review response package hash does not match the frozen package")
    expected = _package_review_items(package)
    actual = {item.review_item_id: item for item in response.items}
    if set(actual) != set(expected):
        missing = sorted(set(expected) - set(actual))
        extra = sorted(set(actual) - set(expected))
        raise ReviewLifecycleError(
            f"review response item IDs differ; missing={missing}, extra={extra}"
        )
    for item_id, item in actual.items():
        template = expected[item_id]
        if (
            item.blind_projection_id != template.blind_projection_id
            or item.criterion is not template.criterion
        ):
            raise ReviewLifecycleError(f"review item binding changed for {item_id}")
    known_projection_ids = {item.blind_projection_id for item in expected.values()}
    unknown_edits = {
        item.blind_projection_id for item in response.proposed_alternative_edits
    } - known_projection_ids
    if unknown_edits:
        raise ReviewLifecycleError(
            f"alternative edits name unknown blind projections: {sorted(unknown_edits)}"
        )
    return actual


def validate_adjudication(
    package: BlindIndependentReviewPackage,
    response: IndependentReviewResponse,
    adjudication: ReviewAdjudication,
) -> None:
    response_items = bind_review_response(package, response)
    if (
        adjudication.package_hash != package.content_hash
        or adjudication.response_hash != response.content_hash
    ):
        raise ReviewLifecycleError("adjudication hash lineage does not match package and response")
    required = {
        item_id
        for item_id, item in response_items.items()
        if item.disposition is not ReviewDisposition.AGREE
    }
    actual = {item.review_item_id for item in adjudication.items}
    if len(actual) != len(adjudication.items) or actual != required:
        raise ReviewLifecycleError(
            "adjudication must cover exactly every disagreement/uncertain item"
        )


class HeldOutSealEntry(ImmutableRecord):
    world_id: Identifier
    world_spec_hash: Sha256Digest
    model_eligible_artifact_hash: Sha256Digest
    query_context_hashes: tuple[Sha256Digest, Sha256Digest, Sha256Digest]
    gold_projection_hashes: tuple[Sha256Digest, Sha256Digest, Sha256Digest]
    alternative_set_hashes: tuple[Sha256Digest, Sha256Digest, Sha256Digest]


class HeldOutDraftSeal(ImmutableRecord):
    seal_id: Identifier
    lifecycle_state: Literal["pending_independent_review"] = "pending_independent_review"
    sealed_at: AwareDatetime
    generator_revision: str
    generator_source_hash: Sha256Digest
    runtime_source_hash: Sha256Digest
    compiler_dependency_hashes: dict[str, Sha256Digest]
    configuration_hash: Sha256Digest
    seed_manifest_hash: Sha256Digest
    review_package_hash: Sha256Digest
    review_binding_manifest_hash: Sha256Digest
    query_tuning_forbidden: Literal[True] = True
    condition_outputs_generated: Literal[False] = False
    entries: tuple[HeldOutSealEntry, ...]

    @model_validator(mode="after")
    def twelve_worlds(self) -> Self:
        if len(self.entries) != 12 or len({item.world_id for item in self.entries}) != 12:
            raise ValueError("draft held-out seal requires twelve unique worlds")
        if set(self.compiler_dependency_hashes) != {
            "synthetic_benchmark.py",
            "benchmark_runtime.py",
            "contracts.py",
            "metrics/alignment.py",
        }:
            raise ValueError("draft held-out seal must bind every benchmark semantic source")
        if (
            self.generator_source_hash != self.compiler_dependency_hashes["synthetic_benchmark.py"]
            or self.runtime_source_hash != self.compiler_dependency_hashes["benchmark_runtime.py"]
        ):
            raise ValueError("legacy source fields disagree with the full dependency hash map")
        return self


class FinalReviewedSeal(ImmutableRecord):
    seal_id: Identifier
    lifecycle_state: Literal["reviewed_adjudicated_frozen"] = "reviewed_adjudicated_frozen"
    draft_seal_hash: Sha256Digest
    package_hash: Sha256Digest
    binding_manifest_hash: Sha256Digest
    response_hash: Sha256Digest
    adjudication_hash: Sha256Digest
    reviewed_artifact_hashes: tuple[Sha256Digest, ...]
    reviewed_projection_hashes: tuple[Sha256Digest, ...]
    reviewed_alternative_hashes: tuple[Sha256Digest, ...]
    held_out_launch_authorized: Literal[True] = True
    condition_outputs_generated_before_review: Literal[False] = False

    @model_validator(mode="after")
    def exactly_nine_reviewed_hashes(self) -> Self:
        if any(
            len(values) != 9
            for values in (
                self.reviewed_artifact_hashes,
                self.reviewed_projection_hashes,
                self.reviewed_alternative_hashes,
            )
        ):
            raise ValueError("final reviewed seal must bind all nine reviewed artifacts")
        return self


class IndependentReviewGateError(RuntimeError):
    """Held-out condition execution cannot begin from a pending draft seal."""


def finalize_reviewed_seal(
    draft: HeldOutDraftSeal,
    package: BlindIndependentReviewPackage,
    bindings: ReviewProjectionBindingManifest,
    response: IndependentReviewResponse,
    adjudication: ReviewAdjudication,
    reviewed_artifacts: Sequence[ReviewedProjectionArtifact],
) -> FinalReviewedSeal:
    if draft.review_package_hash != package.content_hash:
        raise ReviewLifecycleError("draft seal does not bind this review package")
    if (
        draft.review_binding_manifest_hash != bindings.content_hash
        or bindings.package_hash != package.content_hash
    ):
        raise ReviewLifecycleError("draft/package do not bind this scorer review mapping")
    validate_adjudication(package, response, adjudication)
    package_projections = {
        projection.blind_projection_id: projection
        for world in package.worlds
        for projection in world.projections
    }
    binding_by_blind = {item.blind_projection_id: item for item in bindings.entries}
    artifact_by_blind = {item.blind_projection_id: item for item in reviewed_artifacts}
    if (
        len(artifact_by_blind) != len(reviewed_artifacts)
        or set(binding_by_blind) != set(package_projections)
        or set(artifact_by_blind) != set(binding_by_blind)
    ):
        raise ReviewLifecycleError("reviewed artifacts must cover the exact nine bound projections")
    draft_entries = {item.world_id: item for item in draft.entries}
    adjudications = {item.review_item_id: item for item in adjudication.items}
    ordered_artifacts: list[ReviewedProjectionArtifact] = []
    for binding in bindings.entries:
        projection = package_projections[binding.blind_projection_id]
        artifact = artifact_by_blind[binding.blind_projection_id]
        draft_entry = draft_entries.get(binding.world_id)
        if draft_entry is None:
            raise ReviewLifecycleError("review binding names a world outside the held-out seal")
        if (
            binding.blind_projection_hash != projection.content_hash
            or binding.source_gold_projection_hash not in draft_entry.gold_projection_hashes
            or binding.source_alternative_set_hash not in draft_entry.alternative_set_hashes
            or artifact.source_gold_projection_hash != binding.source_gold_projection_hash
            or artifact.source_alternative_set_hash != binding.source_alternative_set_hash
            or artifact.gold_projection.world_id != binding.world_id
            or artifact.gold_projection.query_id != binding.query_id
            or artifact.response_hash != response.content_hash
            or artifact.adjudication_hash != adjudication.content_hash
        ):
            raise ReviewLifecycleError(
                "reviewed artifact lineage differs from its sealed scorer gold"
            )
        projection_items = {item.review_item_id for item in projection.review_items}
        projection_adjudications = [
            item for item_id, item in adjudications.items() if item_id in projection_items
        ]
        if any(
            item.disposition is AdjudicationDisposition.EXCLUDE for item in projection_adjudications
        ):
            raise ReviewLifecycleError(
                "excluding a registered projection requires an explicit methodological amendment"
            )
        amendment_hashes = {
            item.amended_artifact_hash
            for item in projection_adjudications
            if item.disposition is AdjudicationDisposition.AMEND
        }
        if len(amendment_hashes) > 1:
            raise ReviewLifecycleError(
                "all amendments for one projection must bind one semantic replacement"
            )
        if amendment_hashes:
            if amendment_hashes != {artifact.final_semantic_hash}:
                raise ReviewLifecycleError(
                    "amendment hash does not bind the final reviewed semantics"
                )
        else:
            if binding.source_semantic_hash != artifact.final_semantic_hash:
                raise ReviewLifecycleError("retained review artifact changed unamended semantics")
        ordered_artifacts.append(artifact)
    return FinalReviewedSeal(
        seal_id=_opaque("finalseal", draft.content_hash, response.content_hash),
        draft_seal_hash=draft.content_hash,
        package_hash=package.content_hash,
        binding_manifest_hash=bindings.content_hash,
        response_hash=response.content_hash,
        adjudication_hash=adjudication.content_hash,
        reviewed_artifact_hashes=tuple(item.content_hash for item in ordered_artifacts),
        reviewed_projection_hashes=tuple(
            item.gold_projection.content_hash for item in ordered_artifacts
        ),
        reviewed_alternative_hashes=tuple(
            item.alternatives.content_hash for item in ordered_artifacts
        ),
    )


def require_independent_review_complete(
    draft: HeldOutDraftSeal,
    package: BlindIndependentReviewPackage | None,
    bindings: ReviewProjectionBindingManifest | None,
    response: IndependentReviewResponse | None,
    adjudication: ReviewAdjudication | None,
    reviewed_artifacts: Sequence[ReviewedProjectionArtifact] | None,
    final_seal: FinalReviewedSeal | None,
) -> None:
    if any(
        item is None
        for item in (package, bindings, response, adjudication, reviewed_artifacts, final_seal)
    ):
        raise IndependentReviewGateError(
            "held-out launch blocked: complete review lineage and reviewed artifacts are required"
        )
    assert package is not None
    assert bindings is not None
    assert response is not None
    assert adjudication is not None
    assert reviewed_artifacts is not None
    assert final_seal is not None
    try:
        expected = finalize_reviewed_seal(
            draft, package, bindings, response, adjudication, reviewed_artifacts
        )
    except ReviewLifecycleError as error:
        raise IndependentReviewGateError(str(error)) from error
    if (
        expected.content_hash != final_seal.content_hash
        or not final_seal.held_out_launch_authorized
    ):
        raise IndependentReviewGateError(
            "final reviewed seal was not reproduced from the complete reviewed lineage"
        )


class FixedFriendlyProof(ImmutableRecord):
    world_id: Identifier
    query_id: Identifier
    fixed_reference_semantic_hash: Sha256Digest
    query_dependent_semantic_hash: Sha256Digest
    equivalence_rule: Literal["exact-compiled-semantics-for-actual-query"] = (
        "exact-compiled-semantics-for-actual-query"
    )

    @model_validator(mode="after")
    def exact_match(self) -> Self:
        if self.fixed_reference_semantic_hash != self.query_dependent_semantic_hash:
            raise ValueError("fixed-friendly case is not actually construction-equivalent")
        return self


class ScorerWorldArtifact(ImmutableRecord):
    scorer_namespace: Literal["scorer_only"] = "scorer_only"
    world_spec: WorldSpec
    query_assignment: QueryAssignment
    gold_projections: tuple[
        GoldContextualProjection,
        GoldContextualProjection,
        GoldContextualProjection,
    ]
    alternatives: tuple[GoldAlternativeSet, GoldAlternativeSet, GoldAlternativeSet]
    semantic_atoms_by_query: dict[Identifier, tuple[SemanticAtom, ...]]
    answer_signatures_by_query: dict[Identifier, AnswerSignature]
    community_rationales_by_query: dict[Identifier, tuple[CommunityRationale, ...]]
    rare_support_paths_by_query: dict[Identifier, RareSupportPath | None]
    fact_evidence_ids: dict[Identifier, tuple[Identifier, ...]]
    fixed_friendly_proof: FixedFriendlyProof | None = None
    null_case_proof: NullCaseProof | None = None


class FactorCoverage(ImmutableRecord):
    factor: BenchmarkFactor
    world_ids: tuple[Identifier, ...]
    context_ids: tuple[Identifier, ...]


class EligibilityManifest(ImmutableRecord):
    selection_seed: Annotated[int, Field(ge=0)]
    rare_guard_candidate_context_ids: tuple[Identifier, ...]
    temporal_candidate_context_ids: tuple[Identifier, ...]
    holder_status_candidate_context_ids: tuple[Identifier, ...]
    community_candidate_context_ids: tuple[Identifier, ...]
    rare_guard_context_ids: tuple[Identifier, ...]
    temporal_epistemic_context_ids: tuple[Identifier, ...]
    temporal_case_context_ids: tuple[Identifier, ...]
    holder_status_context_ids: tuple[Identifier, ...]
    community_context_ids: tuple[Identifier, ...]

    @model_validator(mode="after")
    def temporal_epistemic_cases_are_explicit(self) -> Self:
        if len(self.rare_guard_context_ids) != 8:
            raise ValueError("rare-guard ablation must contain exactly eight contexts")
        if len(self.temporal_case_context_ids) != 4 or len(self.holder_status_context_ids) != 4:
            raise ValueError("temporal/epistemic ablation must contain four cases of each kind")
        combined = (*self.temporal_case_context_ids, *self.holder_status_context_ids)
        if len(set(combined)) != 8 or set(combined) != set(self.temporal_epistemic_context_ids):
            raise ValueError("temporal and holder-status cases must partition eight contexts")
        subset_pairs = (
            (self.rare_guard_context_ids, self.rare_guard_candidate_context_ids),
            (self.temporal_case_context_ids, self.temporal_candidate_context_ids),
            (self.holder_status_context_ids, self.holder_status_candidate_context_ids),
            (self.community_context_ids, self.community_candidate_context_ids),
        )
        if any(not set(selected).issubset(candidates) for selected, candidates in subset_pairs):
            raise ValueError("eligibility selection is outside its frozen candidate population")
        return self


class ReviewSelectionManifest(IndependentReviewSelectionManifest):
    pass


class ModelRoutingEntry(ImmutableRecord):
    world_id: Identifier
    split: BenchmarkSplit
    artifact_id: Identifier
    relative_prequery_stage_path: str
    relative_neutral_evidence_stage_path: str
    relative_query_stage_paths: tuple[str, str, str]
    artifact_hash: Sha256Digest
    neutral_evidence_artifact_hash: Sha256Digest
    evidence_equivalence_certificate_hash: Sha256Digest
    query_reveal_hashes: tuple[Sha256Digest, Sha256Digest, Sha256Digest]


class RejectedCandidateProvenance(ImmutableRecord):
    disposition: Literal["rejected-before-external-review"] = "rejected-before-external-review"
    rejected_manifest_hashes: tuple[Sha256Digest, ...]
    rejected_draft_or_seal_hashes: tuple[Sha256Digest, ...]
    rejected_review_package_hashes: tuple[Sha256Digest, ...] = ()
    audit_failures: tuple[str, ...]
    condition_outputs_generated: Literal[False] = False
    external_review_used: Literal[False] = False


class GeneratedFileRecord(ImmutableRecord):
    relative_path: str
    byte_count: Annotated[int, Field(ge=0)]
    sha256: Sha256Digest
    release_class: ReleaseClass
    namespace: Literal["condition_input", "model_visible", "scorer_only", "manifest"]


class SyntheticBenchmarkManifest(ImmutableRecord):
    generator_revision: str
    generator_source_hash: Sha256Digest
    runtime_source_hash: Sha256Digest
    compiler_dependency_hashes: dict[str, Sha256Digest]
    configuration_hash: Sha256Digest
    seed_manifest_hash: Sha256Digest
    development_world_count: Literal[4]
    held_out_world_count: Literal[12]
    held_out_context_count: Literal[36]
    neutral_evidence_artifact_count: Literal[16]
    evidence_equivalence_certificate_count: Literal[16]
    paraphrase_context_count: Literal[12]
    review_world_count: Literal[3]
    review_projection_count: Literal[9]
    review_lifecycle_state: Literal["pending_independent_review"] = "pending_independent_review"
    review_complete: Literal[False] = False
    condition_outputs_generated: Literal[False] = False
    held_out_launch_authorized: Literal[False] = False
    lens_counts: dict[LensFamily, int]
    story_scope_counts: dict[StoryScopeBlock, int]
    abstraction_counts: dict[AbstractionLevel, int]
    viewpoint_context_count: Literal[12]
    horizon_counts: dict[HorizonBlock, int]
    node_budget_counts: dict[str, int]
    factor_coverage: tuple[FactorCoverage, ...]
    eligibility_manifest_hash: Sha256Digest
    no_context_selection_hash: Sha256Digest
    query_sampling_audit_hash: Sha256Digest
    paraphrase_manifest_hash: Sha256Digest
    review_selection_manifest_hash: Sha256Digest
    review_package_hash: Sha256Digest
    review_binding_manifest_hash: Sha256Digest
    mutation_manifest_hash: Sha256Digest
    rejected_candidate_provenance_hash: Sha256Digest
    draft_seal_hash: Sha256Digest
    final_reviewed_seal_hash: None = None
    generated_files: tuple[GeneratedFileRecord, ...]


def select_independent_review_worlds(
    held_out_specs: Sequence[WorldSpec],
    seeds: SeedDerivationManifest,
) -> IndependentReviewSelectionManifest:
    selected = []
    review_seed = seed_for(seeds, SeedPurpose.REVIEW_SELECTION)
    for difficulty in Difficulty:
        eligible = sorted(
            (item for item in held_out_specs if item.difficulty is difficulty),
            key=lambda item: item.world_id,
        )
        if len(eligible) != 4:
            raise ValueError("review selection requires four worlds in each difficulty stratum")
        choice = random.Random(derive_seed(review_seed, f"review/{difficulty.value}")).choice(
            eligible
        )
        selected.append(
            ReviewSelectionEntry(
                difficulty=difficulty,
                world_id=choice.world_id,
                blind_world_id=_opaque("reviewworld", difficulty.value, review_seed),
            )
        )
    return IndependentReviewSelectionManifest(
        selection_seed=review_seed,
        rule="uniform-one-within-each-difficulty-stratum",
        selected=tuple(selected),
    )


def _blind_value(value: Any, replacements: Mapping[str, str]) -> Any:
    if isinstance(value, Mapping):
        return {
            str(key): _blind_value(child, replacements)
            for key, child in value.items()
            if key
            not in {
                "content_hash",
                "schema_version",
                "split",
                "world_id",
                "query_id",
                "scorer_namespace",
            }
        }
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [_blind_value(child, replacements) for child in value]
    if isinstance(value, str):
        result = value
        for original, replacement in sorted(
            replacements.items(), key=lambda item: len(item[0]), reverse=True
        ):
            result = result.replace(original, replacement)
        return result
    return value


_REVIEW_PROMPTS = {
    ReviewCriterion.EVIDENCE_SUPPORT: (
        "Are all proposed semantic objects supported by cited evidence?"
    ),
    ReviewCriterion.IDENTITY_PARTITIONS: (
        "Are holder identities and continuing-office partitions evidence-defensible?"
    ),
    ReviewCriterion.EVENT_CHOICES: "Are event-object versus qualified n-ary choices defensible?",
    ReviewCriterion.TEMPORAL_SCOPE: (
        "Are story, validity, discourse, revelation, and horizon distinctions correct?"
    ),
    ReviewCriterion.CONTRAST_DELTAS: (
        "Does the signed A/B delta exactly reflect a nonselection semantic change?"
    ),
    ReviewCriterion.RARE_PIVOTAL_LABELS: (
        "Does deleting the labeled fact alter the answer or registered support structure?"
    ),
    ReviewCriterion.COMMUNITIES: (
        "Do memberships and event participation support the community assignments?"
    ),
    ReviewCriterion.ALTERNATIVES: (
        "Are the executable permissible alternatives defensible and complete?"
    ),
}


def _blind_review_payload(value: Any, *, replacements: Mapping[str, str]) -> Any:
    return _blind_value(_without_hashes(value), replacements)


def build_blind_review_package(
    selection: IndependentReviewSelectionManifest,
    model_artifacts: Mapping[str, ModelEligibleWorldArtifact],
    contexts_by_world: Mapping[str, tuple[QueryContext, QueryContext, QueryContext]],
    scorer_artifacts: Mapping[str, ScorerWorldArtifact],
) -> BlindIndependentReviewPackage:
    package_id = _opaque("reviewpackage", selection.content_hash)
    worlds = []
    for selected in selection.selected:
        model = model_artifacts[selected.world_id]
        contexts = contexts_by_world[selected.world_id]
        scorer = scorer_artifacts[selected.world_id]
        replacements: dict[str, str] = {selected.world_id: selected.blind_world_id}
        for index, context in enumerate(contexts):
            replacements[context.context_id] = _opaque(
                "reviewquery", selected.blind_world_id, index
            )
            replacements[f"score.{context.context_id}"] = _opaque(
                "reviewproposal", selected.blind_world_id, index
            )
        projections = []
        for index, (context, gold, alternatives) in enumerate(
            zip(contexts, scorer.gold_projections, scorer.alternatives, strict=True)
        ):
            blind_projection_id = _opaque("reviewprojection", selected.blind_world_id, index)
            review_items = tuple(
                ReviewPromptItem(
                    review_item_id=_opaque("reviewitem", blind_projection_id, criterion.value),
                    blind_projection_id=blind_projection_id,
                    criterion=criterion,
                    prompt=_REVIEW_PROMPTS[criterion],
                )
                for criterion in ReviewCriterion
            )
            blind = partial(_blind_review_payload, replacements=replacements)
            projections.append(
                BlindReviewProjection(
                    blind_projection_id=blind_projection_id,
                    context_label=chr(65 + index),
                    query=blind(to_model_visible_query(context).model_dump(mode="python")),
                    proposed_local_schema=blind(gold.local_schema.model_dump(mode="python")),
                    proposed_identity_partitions=tuple(
                        blind(item.model_dump(mode="python")) for item in gold.entity_partition
                    ),
                    proposed_events=tuple(
                        blind(item.model_dump(mode="python")) for item in gold.events
                    ),
                    proposed_qualified_assertions=tuple(
                        blind(item.model_dump(mode="python")) for item in gold.qualified_assertions
                    ),
                    proposed_relevance=tuple(
                        blind(item.model_dump(mode="python")) for item in gold.relevance
                    ),
                    proposed_rare_pivotal_labels=tuple(
                        blind(item.model_dump(mode="python")) for item in gold.assertion_annotations
                    ),
                    proposed_communities=tuple(
                        blind(item.model_dump(mode="python")) for item in gold.communities
                    ),
                    proposed_community_rationales=tuple(
                        blind(item.model_dump(mode="python"))
                        for item in scorer.community_rationales_by_query[context.context_id]
                    ),
                    proposed_answer_signature=blind(
                        scorer.answer_signatures_by_query[context.context_id].model_dump(
                            mode="python"
                        )
                    ),
                    proposed_rare_support_path=(
                        blind(
                            scorer.rare_support_paths_by_query[context.context_id].model_dump(
                                mode="python"
                            )
                        )
                        if scorer.rare_support_paths_by_query[context.context_id] is not None
                        else None
                    ),
                    proposed_contrast_decisions=tuple(
                        blind(item.model_dump(mode="python"))
                        for item in gold.signed_contrast_decisions
                    ),
                    proposed_contrast_invariants=tuple(
                        blind(item.model_dump(mode="python")) for item in gold.contrast_invariants
                    ),
                    permissible_alternatives=blind(alternatives.model_dump(mode="python")),
                    review_items=review_items,
                )
            )
        evidence = tuple(_without_hashes(item.model_dump(mode="python")) for item in model.evidence)
        worlds.append(
            BlindReviewWorld(
                blind_world_id=selected.blind_world_id,
                evidence=evidence,
                projections=tuple(projections),
            )
        )
    package = BlindIndependentReviewPackage(
        package_id=package_id,
        selection_rule="one-seeded-world-per-difficulty-stratum",
        reviewer_instructions=(
            "Review only the supplied synthetic evidence and proposed annotations.",
            "Remain blind to evaluated conditions and method outputs.",
            "Return exactly one response for every immutable review_item_id.",
            "Cite opaque evidence identifiers for disagreements or proposed alternatives.",
            "Held-out execution remains blocked until response and adjudication are sealed.",
        ),
        worlds=tuple(worlds),
    )
    serialized = package.to_canonical_json().casefold()
    if (
        '"condition"' in serialized
        or '"prediction"' in serialized
        or '"method_output"' in serialized
    ):
        raise ValueError("blind package contains evaluated-condition output metadata")
    return package


def build_review_projection_bindings(
    selection: IndependentReviewSelectionManifest,
    package: BlindIndependentReviewPackage,
    contexts_by_world: Mapping[str, tuple[QueryContext, QueryContext, QueryContext]],
    scorer_artifacts: Mapping[str, ScorerWorldArtifact],
) -> ReviewProjectionBindingManifest:
    """Bind blind review IDs to the exact sealed scorer objects outside reviewer view."""

    entries: list[ReviewProjectionBinding] = []
    for selected, blind_world in zip(selection.selected, package.worlds, strict=True):
        scorer = scorer_artifacts[selected.world_id]
        contexts = contexts_by_world[selected.world_id]
        for blind, context, gold, alternatives in zip(
            blind_world.projections,
            contexts,
            scorer.gold_projections,
            scorer.alternatives,
            strict=True,
        ):
            entries.append(
                ReviewProjectionBinding(
                    blind_projection_id=blind.blind_projection_id,
                    blind_projection_hash=blind.content_hash,
                    world_id=selected.world_id,
                    query_id=context.context_id,
                    source_gold_projection_hash=gold.content_hash,
                    source_alternative_set_hash=alternatives.content_hash,
                    source_semantic_hash=reviewed_semantic_hash(gold, alternatives),
                )
            )
    return ReviewProjectionBindingManifest(
        package_hash=package.content_hash,
        selection_manifest_hash=selection.content_hash,
        entries=tuple(entries),
    )


def reviewed_artifact_from_original(
    *,
    binding: ReviewProjectionBinding,
    package: BlindIndependentReviewPackage,
    response: IndependentReviewResponse,
    adjudication: ReviewAdjudication,
    gold: GoldContextualProjection,
    alternatives: GoldAlternativeSet,
) -> ReviewedProjectionArtifact:
    """Create a status-updated artifact for an agreed or retained original proposal."""

    validate_adjudication(package, response, adjudication)
    blind = next(
        projection
        for world in package.worlds
        for projection in world.projections
        if projection.blind_projection_id == binding.blind_projection_id
    )
    response_by_item = {item.review_item_id: item for item in response.items}
    has_disagreement = any(
        response_by_item[item.review_item_id].disposition is not ReviewDisposition.AGREE
        for item in blind.review_items
    )
    review_status = (
        GoldReviewStatus.DISAGREEMENT_LOGGED if has_disagreement else GoldReviewStatus.REVIEWED
    )
    adjudication_status = (
        GoldAdjudicationStatus.ADJUDICATED
        if has_disagreement
        else GoldAdjudicationStatus.NOT_REQUIRED
    )
    gold_values = _without_hashes(gold.model_dump(mode="python"))
    gold_values.update(
        {
            "review_status": review_status.value,
            "adjudication_status": adjudication_status.value,
            "independent_review_record_hash": response.content_hash,
            "adjudication_record_hash": (adjudication.content_hash if has_disagreement else None),
        }
    )
    reviewed_gold = GoldContextualProjection.model_validate(gold_values)
    alternative_values = _without_hashes(alternatives.model_dump(mode="python"))
    alternative_values.update(
        {
            "review_status": review_status.value,
            "adjudication_status": adjudication_status.value,
        }
    )
    reviewed_alternatives = GoldAlternativeSet.model_validate(alternative_values)
    return ReviewedProjectionArtifact(
        blind_projection_id=binding.blind_projection_id,
        source_gold_projection_hash=binding.source_gold_projection_hash,
        source_alternative_set_hash=binding.source_alternative_set_hash,
        response_hash=response.content_hash,
        adjudication_hash=adjudication.content_hash,
        final_semantic_hash=reviewed_semantic_hash(reviewed_gold, reviewed_alternatives),
        gold_projection=reviewed_gold,
        alternatives=reviewed_alternatives,
    )


_PARAPHRASE_TEXT = {
    LensFamily.ALLEGIANCE_STATE_CHANGE: (
        "Trace only evidence-supported shifts in affiliation and consequential state."
    ),
    LensFamily.IDENTITY_KINSHIP: (
        "Resolve personal identity separately from continuity of an evidenced office."
    ),
    LensFamily.CAUSAL_CONSEQUENCE: (
        "Construct the supported chain by which the focal outcome came about."
    ),
    LensFamily.EVENT_PARTICIPATION_CONFLICT: (
        "Organize the focal occurrence by participants, roles, and location."
    ),
    LensFamily.MOVEMENT_TIME: (
        "Show relevant movement and ordering inside the declared story interval."
    ),
    LensFamily.KNOWLEDGE_BELIEF: (
        "Distinguish each holder's report or denial from narrative commitment."
    ),
}


def _rebuild_context(context: QueryContext, **updates: Any) -> QueryContext:
    values = _without_hashes(context.model_dump(mode="python"))
    values.update(updates)
    return QueryContext.model_validate(values)


def _balanced_context_selection(
    specs: Sequence[WorldSpec],
    contexts_by_world: Mapping[str, tuple[QueryContext, QueryContext, QueryContext]],
    seed: int,
) -> dict[str, int]:
    ordered = sorted(specs, key=lambda item: item.world_id)
    choices: dict[str, list[int]] = {}
    for spec in ordered:
        values = [0, 1, 2]
        random.Random(derive_seed(seed, f"balanced/{spec.world_id}")).shuffle(values)
        choices[spec.world_id] = values
    selected: dict[str, int] = {}
    counts: Counter[LensFamily] = Counter()

    def visit(index: int) -> bool:
        if index == len(ordered):
            return counts == Counter({lens: 2 for lens in LensFamily})
        spec = ordered[index]
        for context_index in choices[spec.world_id]:
            lens = _lens_for_context(contexts_by_world[spec.world_id][context_index])
            if counts[lens] >= 2:
                continue
            selected[spec.world_id] = context_index
            counts[lens] += 1
            if visit(index + 1):
                return True
            counts[lens] -= 1
            del selected[spec.world_id]
        return False

    if not visit(0):
        raise RuntimeError("unable to satisfy frozen one-per-world/two-per-lens selection")
    return selected


def select_paraphrases(
    held_out_specs: Sequence[WorldSpec],
    contexts_by_world: Mapping[str, tuple[QueryContext, QueryContext, QueryContext]],
    seeds: SeedDerivationManifest,
    config: BenchmarkConfiguration,
) -> ParaphraseSelectionManifest:
    seed = seed_for(seeds, SeedPurpose.PARAPHRASE)
    selected = _balanced_context_selection(held_out_specs, contexts_by_world, seed)
    pairs = []
    for spec in sorted(held_out_specs, key=lambda item: item.world_id):
        base = contexts_by_world[spec.world_id][selected[spec.world_id]]
        lens = _lens_for_context(base)
        pairs.append(
            ParaphrasePair(
                world_id=spec.world_id,
                base_context_id=base.context_id,
                base_context_hash=base.content_hash,
                paraphrase_context=_rebuild_context(
                    base,
                    context_id=_opaque("ctx", base.context_id, "paraphrase"),
                    wording=_PARAPHRASE_TEXT[lens],
                ),
                base_renderer_id=config.query_renderers[0],
                paraphrase_renderer_id=config.query_renderers[1],
            )
        )
    return ParaphraseSelectionManifest(selection_seed=seed, pairs=tuple(pairs))


def select_no_context_ablation(
    held_out_specs: Sequence[WorldSpec],
    contexts_by_world: Mapping[str, tuple[QueryContext, QueryContext, QueryContext]],
    seeds: SeedDerivationManifest,
) -> ANoContextSelectionManifest:
    seed = seed_for(seeds, SeedPurpose.A_NO_CONTEXT)
    selected = _balanced_context_selection(held_out_specs, contexts_by_world, seed)
    return ANoContextSelectionManifest(
        selection_seed=seed,
        entries=tuple(
            ANoContextSelectionEntry(
                world_id=spec.world_id,
                query_id=contexts_by_world[spec.world_id][selected[spec.world_id]].context_id,
                difficulty=spec.difficulty,
                lens=_lens_for_context(contexts_by_world[spec.world_id][selected[spec.world_id]]),
            )
            for spec in sorted(held_out_specs, key=lambda item: item.world_id)
        ),
    )


def mutate_source_fact(
    spec: WorldSpec,
    fact_id: str,
    *,
    replacement_relation: str,
) -> WorldSpec:
    values = _without_hashes(spec.model_dump(mode="python"))
    matched = False
    for fact in values["facts"]:
        if fact["fact_id"] == fact_id:
            fact["relation"] = replacement_relation
            matched = True
    if not matched:
        raise KeyError(fact_id)
    return WorldSpec.model_validate(values)


def mutate_query_lens(context: QueryContext, replacement_lens: LensFamily) -> QueryContext:
    wording, description = _LENS_TEXT[replacement_lens]
    return _rebuild_context(context, wording=wording, lens=description)


def mutate_query_wording(context: QueryContext, wording: str) -> QueryContext:
    """Change only surface wording while retaining the registered query semantics."""

    return _rebuild_context(context, wording=wording)


def mutate_world_horizon(spec: WorldSpec, replacement_horizon: HorizonBlock) -> WorldSpec:
    values = _without_hashes(spec.model_dump(mode="python"))
    values["horizon_block"] = replacement_horizon.value
    return WorldSpec.model_validate(values)


def remove_rare_pivotal_fact(spec: WorldSpec) -> WorldSpec:
    values = _without_hashes(spec.model_dump(mode="python"))
    removed_ids = {
        fact["fact_id"]
        for fact in values["facts"]
        if fact["role"] == NeutralFactRole.RARE_PIVOTAL.value
    }
    values["facts"] = [
        fact for fact in values["facts"] if fact["role"] != NeutralFactRole.RARE_PIVOTAL.value
    ]
    values["causal_dependencies"] = [
        edge for edge in values["causal_dependencies"] if not removed_ids.intersection(edge)
    ]
    values["rare_deletion_mutation"] = True
    return WorldSpec.model_validate(values)


def semantic_signatures_for_horizon(
    spec: WorldSpec,
    horizon: HorizonBlock,
) -> frozenset[str]:
    return frozenset(
        _fact_signature(fact)
        for fact in spec.facts
        if horizon is HorizonBlock.FINAL or fact.disclosure_order < 90
    )


def _changed_answer_components(before: AnswerSignature, after: AnswerSignature) -> tuple[str, ...]:
    fields = {
        "answer_facts": "answer_facts",
        "causal_paths": RareImpactKind.CAUSAL.value,
        "temporal_order": RareImpactKind.TEMPORAL.value,
        "consequential_states": RareImpactKind.STATE.value,
        "community_partition": RareImpactKind.COMMUNITY.value,
    }
    return tuple(
        output
        for field, output in fields.items()
        if getattr(before, field) != getattr(after, field)
    )


def _semantic_leaf_signatures(value: Any, path: str = "root") -> frozenset[str]:
    """Return an exact, path-qualified semantic leaf set for mutation deltas."""

    if hasattr(value, "model_dump"):
        value = value.model_dump(mode="json")
    if isinstance(value, Mapping):
        leaves: set[str] = set()
        for key in sorted(value):
            if key in _REVIEW_WORKFLOW_FIELDS or key in {"compiled_at", "created_at"}:
                continue
            if key == "source_artifact_hash":
                # Passage-byte lineage was added after the mutation oracle was
                # frozen.  Normalize this administrative value to its historical
                # null representation so the oracle continues to describe only
                # the registered semantic mutation, including added records.
                leaves.update(_semantic_leaf_signatures(None, f"{path}.{key}"))
                continue
            leaves.update(_semantic_leaf_signatures(value[key], f"{path}.{key}"))
        return frozenset(leaves)
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        leaves = set()
        for index, child in enumerate(value):
            leaves.update(_semantic_leaf_signatures(child, f"{path}[{index}]"))
        return frozenset(leaves)
    return frozenset((f"{path}={canonical_json(value)}",))


def _regenerate_world(
    spec: WorldSpec,
    assignment: QueryAssignment,
    config: BenchmarkConfiguration,
    seeds: SeedDerivationManifest,
) -> tuple[
    NarrativeProducts,
    tuple[QueryContext, QueryContext, QueryContext],
    tuple[CompiledGold, CompiledGold, CompiledGold],
]:
    narrative = realize_narrative(spec, config, seeds)
    contexts = compile_query_contexts(spec, assignment, narrative, config)
    products = tuple(
        compile_gold_projection(
            spec,
            context,
            index,
            narrative,
            config,
            policy=assignment.compiler_policies[index],
            selected_for_review=False,
            require_rare=not spec.rare_deletion_mutation,
        )
        for index, context in enumerate(contexts)
    )
    return narrative, contexts, products  # type: ignore[return-value]


def _regenerated_bundle_payload(
    narrative: NarrativeProducts,
    contexts: Sequence[QueryContext],
    products: Sequence[CompiledGold],
) -> dict[str, Any]:
    return {
        "snapshot": narrative.snapshot,
        "evidence": narrative.evidence,
        "contexts": tuple(contexts),
        "projections": tuple(item.projection for item in products),
        "semantic_atoms": tuple(item.semantic_atoms for item in products),
        "answers": tuple(item.answer_signature for item in products),
        "rare_support_paths": tuple(item.rare_support_path for item in products),
    }


def _registered_mutation_bundle_hash(bundle: Mapping[str, Any]) -> str:
    """Hash the preregistered semantic bundle using its frozen evidence view.

    Mutation proofs predate the administrative passage-byte binding now carried
    by neutral evidence.  Reconstructing only each evidence provenance/record
    content hash with that field set to its historical null value prevents a
    provenance hardening correction from masquerading as a semantic mutation.
    """

    evidence = bundle.get("evidence")
    if not isinstance(evidence, Sequence):
        raise TypeError("mutation bundle evidence must be a sequence")
    frozen_evidence: list[EvidenceRecord] = []
    for record in evidence:
        if not isinstance(record, EvidenceRecord):
            raise TypeError("mutation bundle contains a non-evidence record")
        provenance_payload = record.provenance.model_dump(
            mode="python",
            exclude={"content_hash", "source_artifact_hash"},
        )
        frozen_provenance = ProvenanceReference(
            **provenance_payload,
            source_artifact_hash=None,
        )
        record_payload = record.model_dump(
            mode="python",
            exclude={"content_hash", "provenance"},
        )
        frozen_evidence.append(
            EvidenceRecord(**record_payload, provenance=frozen_provenance)
        )
    frozen_bundle = dict(bundle)
    frozen_bundle["evidence"] = tuple(frozen_evidence)
    return canonical_sha256(frozen_bundle)


def build_mutation_manifest(
    development_specs: Sequence[WorldSpec],
    contexts_by_world: Mapping[str, tuple[QueryContext, QueryContext, QueryContext]],
    narratives: Mapping[str, NarrativeProducts],
    scorer_artifacts: Mapping[str, ScorerWorldArtifact],
    config: BenchmarkConfiguration,
    seeds: SeedDerivationManifest,
) -> MutationManifest:
    mutation_seed = seed_for(seeds, SeedPurpose.MUTATION)
    spec = sorted(development_specs, key=lambda item: item.world_id)[0]
    assignment = scorer_artifacts[spec.world_id].query_assignment
    safe_source_mutations = {
        f"{spec.world_id}.fact.causal.bridge": "consults_with",
        f"{spec.world_id}.fact.causal.outcome": "achieves_result_for",
        f"{spec.world_id}.fact.office.succession": "takes_office_after",
    }
    source_target = random.Random(derive_seed(mutation_seed, "source-target")).choice(
        sorted(safe_source_mutations)
    )
    changed = mutate_source_fact(
        spec,
        source_target,
        replacement_relation=safe_source_mutations[source_target],
    )
    before_narrative, before_contexts, before_products = _regenerate_world(
        spec, assignment, config, seeds
    )
    changed_narrative, changed_contexts, changed_products = _regenerate_world(
        changed, assignment, config, seeds
    )
    source_before = _semantic_leaf_signatures(
        _regenerated_bundle_payload(before_narrative, before_contexts, before_products)
    )
    source_after = _semantic_leaf_signatures(
        _regenerated_bundle_payload(changed_narrative, changed_contexts, changed_products)
    )
    context = contexts_by_world[spec.world_id][0]
    replacement_lens = random.Random(derive_seed(mutation_seed, "query-lens")).choice(
        tuple(item for item in LensFamily if item is not _lens_for_context(context))
    )
    mutated_context = mutate_query_lens(context, replacement_lens)
    original = before_products[0]
    query_changed = compile_gold_projection(
        spec,
        mutated_context,
        0,
        narratives[spec.world_id],
        config,
        policy=CompilerPolicy.QUERY_DEPENDENT,
        selected_for_review=False,
    )
    query_before = _semantic_leaf_signatures(
        {"context": context, "projection": original.projection, "atoms": original.semantic_atoms}
    )
    query_after = _semantic_leaf_signatures(
        {
            "context": mutated_context,
            "projection": query_changed.projection,
            "atoms": query_changed.semantic_atoms,
        }
    )
    replacement_horizon = (
        HorizonBlock.INTERMEDIATE
        if spec.horizon_block is HorizonBlock.FINAL
        else HorizonBlock.FINAL
    )
    horizon_spec = mutate_world_horizon(spec, replacement_horizon)
    assignment_values = _without_hashes(assignment.model_dump(mode="python"))
    assignment_values["horizon_block"] = replacement_horizon.value
    horizon_assignment = QueryAssignment.model_validate(assignment_values)
    horizon_narrative, horizon_contexts, horizon_products = _regenerate_world(
        horizon_spec, horizon_assignment, config, seeds
    )
    horizon_before = _semantic_leaf_signatures(
        _regenerated_bundle_payload(before_narrative, before_contexts, before_products)
    )
    horizon_after = _semantic_leaf_signatures(
        _regenerated_bundle_payload(horizon_narrative, horizon_contexts, horizon_products)
    )
    expectations = (
        MutationExpectation(
            mutation_id="mutation-source-regenerate-v3",
            kind=MutationKind.SOURCE_FACT,
            world_id=spec.world_id,
            target_id=source_target,
            removed_signatures=tuple(sorted(source_before - source_after)),
            added_signatures=tuple(sorted(source_after - source_before)),
            unaffected_signature_count=len(source_before & source_after),
        ),
        MutationExpectation(
            mutation_id="mutation-query-regenerate-v3",
            kind=MutationKind.QUERY_CONTEXT,
            world_id=spec.world_id,
            target_id=context.context_id,
            removed_signatures=tuple(sorted(query_before - query_after)),
            added_signatures=tuple(sorted(query_after - query_before)),
            unaffected_signature_count=len(query_before & query_after),
        ),
        MutationExpectation(
            mutation_id="mutation-horizon-regenerate-v3",
            kind=MutationKind.SPOILER_HORIZON,
            world_id=spec.world_id,
            target_id=f"{spec.world_id}.fact.late",
            removed_signatures=tuple(sorted(horizon_before - horizon_after)),
            added_signatures=tuple(sorted(horizon_after - horizon_before)),
            unaffected_signature_count=len(horizon_before & horizon_after),
        ),
    )
    if any(not item.removed_signatures or not item.added_signatures for item in expectations):
        raise ValueError("each regenerated mutation must produce a nonempty exact delta")
    rare_proofs: list[RareDeletionProof] = []
    for world_id, scorer in sorted(scorer_artifacts.items()):
        if scorer.world_spec.split is not BenchmarkSplit.HELD_OUT:
            continue
        mutated_spec = remove_rare_pivotal_fact(scorer.world_spec)
        original_narrative, original_contexts, original_products = _regenerate_world(
            scorer.world_spec, scorer.query_assignment, config, seeds
        )
        mutated_narrative, mutated_contexts, mutated_products = _regenerate_world(
            mutated_spec, scorer.query_assignment, config, seeds
        )
        world_proof_count = 0
        for index, (context_item, original_product, mutated_product) in enumerate(
            zip(original_contexts, original_products, mutated_products, strict=True)
        ):
            if original_product.rare_support_path is None:
                continue
            after = mutated_product.answer_signature
            before = scorer.answer_signatures_by_query[context_item.context_id]
            changed_components = _changed_answer_components(before, after)
            before_bundle = _regenerated_bundle_payload(
                original_narrative, (context_item,), (original_product,)
            )
            after_bundle = _regenerated_bundle_payload(
                mutated_narrative, (mutated_contexts[index],), (mutated_product,)
            )
            rare_proofs.append(
                RareDeletionProof(
                    world_id=world_id,
                    query_id=context_item.context_id,
                    impact_kind=scorer.world_spec.rare_impact_kind,
                    before_answer_hash=before.content_hash,
                    after_answer_hash=after.content_hash,
                    before_regenerated_bundle_hash=_registered_mutation_bundle_hash(
                        before_bundle
                    ),
                    after_regenerated_bundle_hash=_registered_mutation_bundle_hash(
                        after_bundle
                    ),
                    changed_components=changed_components,
                )
            )
            world_proof_count += 1
        if world_proof_count == 0:
            raise ValueError(f"held-out world has no eligible rare-pivotal query: {world_id}")
    manifest = MutationManifest(
        mutation_seed=mutation_seed,
        expectations=expectations,
        rare_deletion_proofs=tuple(rare_proofs),
    )
    if len(manifest.rare_deletion_proofs) < config.minimum_rare_ablation_context_count:
        raise ValueError("rare-deletion mutation proofs do not meet the registered minimum")
    return manifest


def _factor_coverage(
    held_out_specs: Sequence[WorldSpec],
    contexts_by_world: Mapping[str, tuple[QueryContext, QueryContext, QueryContext]],
) -> tuple[FactorCoverage, ...]:
    return tuple(
        FactorCoverage(
            factor=factor,
            world_ids=tuple(item.world_id for item in held_out_specs if factor in item.factors),
            context_ids=tuple(
                context.context_id
                for item in held_out_specs
                if factor in item.factors
                for context in contexts_by_world[item.world_id]
            ),
        )
        for factor in BenchmarkFactor
    )


def build_eligibility_manifest(
    held_out_specs: Sequence[WorldSpec],
    contexts_by_world: Mapping[str, tuple[QueryContext, QueryContext, QueryContext]],
    scorer_artifacts: Mapping[str, ScorerWorldArtifact],
    narratives: Mapping[str, NarrativeProducts],
    config: BenchmarkConfiguration,
    seeds: SeedDerivationManifest,
) -> EligibilityManifest:
    ordered = sorted(held_out_specs, key=lambda item: item.world_id)
    selection_seed = derive_seed(seed_for(seeds, SeedPurpose.QUERY), "eligibility-v3")
    context_meta = {
        context.context_id: (spec, index, context)
        for spec in ordered
        for index, context in enumerate(contexts_by_world[spec.world_id])
    }

    rare_candidates = tuple(
        sorted(
            context_id
            for spec in ordered
            for context_id, path in scorer_artifacts[
                spec.world_id
            ].rare_support_paths_by_query.items()
            if path is not None
        )
    )
    rare_by_world: dict[str, list[str]] = defaultdict(list)
    for context_id in rare_candidates:
        rare_by_world[context_meta[context_id][0].world_id].append(context_id)
    rare_world_order = list(sorted(rare_by_world))
    random.Random(derive_seed(selection_seed, "rare-worlds")).shuffle(rare_world_order)
    selected_rare_worlds = rare_world_order[:8]
    rare_cases = tuple(
        random.Random(derive_seed(selection_seed, f"rare-context/{world_id}")).choice(
            sorted(rare_by_world[world_id])
        )
        for world_id in selected_rare_worlds
    )

    temporal_candidates: list[str] = []
    holder_candidates: list[str] = []
    for context_id, (spec, index, context) in sorted(context_meta.items()):
        scorer = scorer_artifacts[spec.world_id]
        base = scorer.gold_projections[index]
        policy = scorer.query_assignment.compiler_policies[index]
        if context.viewpoint is None:
            replacement_scope = next(
                _scope(block) for block in StoryScopeBlock if _scope(block) != context.story_scope
            )
            perturbed = compile_gold_projection(
                spec,
                _rebuild_context(context, story_scope=replacement_scope),
                index,
                narratives[spec.world_id],
                config,
                policy=policy,
                selected_for_review=False,
            ).projection
            relevance = {item.target_id: item.is_relevant for item in base.relevance}
            base_temporal = {
                item.assertion_id: canonical_sha256(item.temporal_scope)
                for item in base.qualified_assertions
                if relevance[item.assertion_id]
            }
            changed_temporal = {
                item.assertion_id: canonical_sha256(item.temporal_scope)
                for item in perturbed.qualified_assertions
            }
            if any(changed_temporal.get(key) != value for key, value in base_temporal.items()):
                temporal_candidates.append(context_id)
        else:
            relevance = {item.target_id: item.is_relevant for item in base.relevance}
            relevant_denials = tuple(
                item
                for item in base.qualified_assertions
                if relevance[item.assertion_id]
                and item.epistemic_scope is not None
                and item.epistemic_scope.attitude is EpistemicAttitude.DENIED
            )
            if not relevant_denials:
                continue
            unframed = compile_gold_projection(
                spec,
                _rebuild_context(context, viewpoint=None),
                index,
                narratives[spec.world_id],
                config,
                policy=policy,
                selected_for_review=False,
            ).projection
            unframed_by_id = {item.assertion_id: item for item in unframed.qualified_assertions}
            if any(
                unframed_by_id[item.assertion_id].narrative_commitment
                is not item.narrative_commitment
                for item in relevant_denials
            ):
                holder_candidates.append(context_id)

    def balanced_four(
        candidates: Sequence[str], namespace: str, excluded_worlds: frozenset[str] = frozenset()
    ) -> tuple[str, str, str, str]:
        eligible = [
            context_id
            for context_id in candidates
            if context_meta[context_id][0].world_id not in excluded_worlds
        ]
        selected: list[str] = []
        used_worlds: set[str] = set()
        for difficulty in Difficulty:
            block = [
                context_id
                for context_id in eligible
                if context_meta[context_id][0].difficulty is difficulty
                and context_meta[context_id][0].world_id not in used_worlds
            ]
            if not block:
                raise ValueError(f"{namespace} eligibility lacks {difficulty.value} cases")
            choice = random.Random(
                derive_seed(selection_seed, f"{namespace}/{difficulty.value}")
            ).choice(sorted(block))
            selected.append(choice)
            used_worlds.add(context_meta[choice][0].world_id)
        remainder = [
            context_id
            for context_id in eligible
            if context_meta[context_id][0].world_id not in used_worlds
        ]
        if not remainder:
            raise ValueError(f"{namespace} eligibility lacks a fourth distinct world")
        extra = random.Random(derive_seed(selection_seed, f"{namespace}/extra")).choice(
            sorted(remainder)
        )
        selected.append(extra)
        return tuple(selected)  # type: ignore[return-value]

    holder_cases = balanced_four(holder_candidates, "holder")
    holder_worlds = frozenset(context_meta[item][0].world_id for item in holder_cases)
    temporal_cases = balanced_four(temporal_candidates, "temporal", holder_worlds)
    community_candidates = tuple(
        sorted(
            contexts_by_world[spec.world_id][0].context_id
            for spec in ordered
            if BenchmarkFactor.MEANINGFUL_COMMUNITIES in spec.factors
        )
    )
    return EligibilityManifest(
        selection_seed=selection_seed,
        rare_guard_candidate_context_ids=rare_candidates,
        temporal_candidate_context_ids=tuple(sorted(temporal_candidates)),
        holder_status_candidate_context_ids=tuple(sorted(holder_candidates)),
        community_candidate_context_ids=community_candidates,
        rare_guard_context_ids=rare_cases,
        temporal_epistemic_context_ids=(*temporal_cases, *holder_cases),
        temporal_case_context_ids=tuple(temporal_cases),
        holder_status_context_ids=tuple(holder_cases),
        community_context_ids=community_candidates,
    )


def validate_factor_quotas(
    config: BenchmarkConfiguration,
    coverage: Sequence[FactorCoverage],
    eligibility: EligibilityManifest,
) -> None:
    by_factor = {item.factor: item for item in coverage}
    for factor in BenchmarkFactor:
        if factor is BenchmarkFactor.NULL_NONSELECTION_CASE:
            required = 3
        elif factor is BenchmarkFactor.TEMPORAL_CHANGE:
            required = config.minimum_temporal_change_world_count
        else:
            required = config.minimum_factor_world_count
        if len(by_factor[factor].world_ids) < required:
            raise ValueError(f"factor quota failed: {factor.value}")
    if len(eligibility.rare_guard_context_ids) != 8:
        raise ValueError("A-NoRareGuard selection must contain exactly eight contexts")
    if len(eligibility.temporal_epistemic_context_ids) != 8:
        raise ValueError("A-NoTemporalEpistemic selection must contain exactly eight contexts")
    if len({item for item in eligibility.community_context_ids}) < 8:
        raise ValueError("community eligibility must span at least eight contexts")


@dataclass(frozen=True)
class BenchmarkBuild:
    configuration: BenchmarkConfiguration
    seeds: SeedDerivationManifest
    world_specs: tuple[WorldSpec, ...]
    query_sampling_audit: QuerySamplingAudit
    model_artifacts: Mapping[str, ModelEligibleWorldArtifact]
    neutral_evidence_artifacts: Mapping[str, NeutralEvidenceArtifact]
    evidence_equivalence_certificates: Mapping[str, EvidenceProjectionEquivalenceCertificate]
    query_reveals_by_world: Mapping[
        str, tuple[QueryRevealArtifact, QueryRevealArtifact, QueryRevealArtifact]
    ]
    scorer_artifacts: Mapping[str, ScorerWorldArtifact]
    contexts_by_world: Mapping[str, tuple[QueryContext, QueryContext, QueryContext]]
    narratives: Mapping[str, NarrativeProducts]
    paraphrases: ParaphraseSelectionManifest
    no_context_selection: ANoContextSelectionManifest
    eligibility: EligibilityManifest
    mutation_manifest: MutationManifest
    review_selection: IndependentReviewSelectionManifest
    review_package: BlindIndependentReviewPackage
    review_bindings: ReviewProjectionBindingManifest
    draft_seal: HeldOutDraftSeal
    factor_coverage: tuple[FactorCoverage, ...]
    rejected_candidate: RejectedCandidateProvenance


_REJECTED_CANDIDATE = RejectedCandidateProvenance(
    rejected_manifest_hashes=(
        "15dae90a236601bce3937ca6e5ac76ed4b67259da9783ec71a1e6de53380544d",
        "8b063a7af1c52156076dc6d3d436d34efb924018fe942187de6d44e736e519cf",
        "6d16cd8c4cb2f14926784a2cfe8e2e718a2797495ecc099677aac381103858e4",
        "23574bdbe02b990f4aa987b005b98b03e611d518c64508e2742fa6aba1f668d6",
        "01046b13fd0e3b50159de9ce8832ba2980f109a8f291aeccf2c78d03dcb12e15",
    ),
    rejected_draft_or_seal_hashes=(
        "67d8ed102477e400403becdc97bc1d258790c05ec1ee39d969104fcd808ade70",
        "3b8eadc5a65d2bf3a4df7c663b2393e1eebaefaab564dcb6c5b6e8294d363927",
        "28b5a7bd2c63e7007cedcfb0f114e9db689898d798b4aa6a2f35e5a2e59d58c5",
    ),
    rejected_review_package_hashes=(
        "52f75c452c583e77ffe0236c0ea24902d919829ea584817d59273ac4827da24e",
    ),
    audit_failures=(
        "32/32 contrast decisions had anchors disjoint from their cited evidence",
        "36/36 rare-pivotal support paths were graph-disconnected",
        "25 relevant event annotations were outside the query story scope",
        "arbitrary independent-review response IDs were accepted by count alone",
        "C1 evidence and all three future queries were co-located in one runtime object",
        "held-out worlds shared one parameterized generator skeleton with development",
        "fixed-friendly and null labels did not hold the actual registered query fixed",
        "schema type/relation evidence did not support the declared semantics",
        "temporal and epistemic decisions were decorative and no relevant denial was tested",
        "rare necessity was circular and its support path was untyped/undirected",
        "permissible alternatives admitted unsupported Cartesian hybrids",
        "mutation checks compared hand-selected signatures instead of regenerated outputs",
        "one gold projection exceeded its query node budget",
        "source/config provenance and independent-review replacement bindings were incomplete",
    ),
)


def _compiled_product_semantic_hash(product: CompiledGold) -> str:
    return canonical_sha256(
        {
            "projection": _review_semantic_payload(product.projection.model_dump(mode="python")),
            "alternatives": _review_semantic_payload(
                product.alternatives.model_dump(mode="python")
            ),
            "semantic_atoms": product.semantic_atoms,
            "answer_signature": product.answer_signature,
            "community_rationales": product.community_rationales,
            "rare_support_path": product.rare_support_path,
        }
    )


def compile_benchmark(config_path: Path = DEFAULT_CONFIG_PATH) -> BenchmarkBuild:
    config = load_benchmark_configuration(config_path)
    seeds = build_seed_manifest(config.root_seed)
    specs = build_world_specs(config, seeds)
    development = tuple(item for item in specs if item.split is BenchmarkSplit.DEVELOPMENT)
    held_out = tuple(item for item in specs if item.split is BenchmarkSplit.HELD_OUT)
    audit = sample_held_out_queries(held_out, seeds)
    assignments = {item.world_id: item for item in audit.assignments}
    assignments.update({item.world_id: item for item in _development_assignments(development)})
    review_selection = select_independent_review_worlds(held_out, seeds)
    reviewed_worlds = {item.world_id for item in review_selection.selected}
    model_artifacts: dict[str, ModelEligibleWorldArtifact] = {}
    neutral_evidence_artifacts: dict[str, NeutralEvidenceArtifact] = {}
    evidence_equivalence_certificates: dict[str, EvidenceProjectionEquivalenceCertificate] = {}
    query_reveals_by_world: dict[
        str, tuple[QueryRevealArtifact, QueryRevealArtifact, QueryRevealArtifact]
    ] = {}
    scorer_artifacts: dict[str, ScorerWorldArtifact] = {}
    contexts_by_world: dict[str, tuple[QueryContext, QueryContext, QueryContext]] = {}
    narratives: dict[str, NarrativeProducts] = {}
    for spec in specs:
        narrative = realize_narrative(spec, config, seeds)
        artifact = ModelEligibleWorldArtifact(
            artifact_id=_opaque("artifact", spec.world_id),
            snapshot=narrative.snapshot,
            evidence=tuple(to_model_visible_evidence(item) for item in narrative.evidence),
        )
        neutral_artifact = NeutralEvidenceArtifact(
            artifact_id=_opaque("neutral", spec.world_id),
            snapshot=narrative.snapshot,
            evidence=narrative.evidence,
        )
        equivalence = build_evidence_projection_equivalence_certificate(
            neutral_artifact,
            artifact,
            certificate_id=_opaque(
                "equivalence", neutral_artifact.content_hash, artifact.content_hash
            ),
        )
        model_artifacts[spec.world_id] = artifact
        neutral_evidence_artifacts[spec.world_id] = neutral_artifact
        evidence_equivalence_certificates[spec.world_id] = equivalence
        assignment = assignments[spec.world_id]
        contexts = compile_query_contexts(spec, assignment, narrative, config)
        products = [
            compile_gold_projection(
                spec,
                context,
                index,
                narrative,
                config,
                policy=assignment.compiler_policies[index],
                selected_for_review=spec.world_id in reviewed_worlds,
            )
            for index, context in enumerate(contexts)
        ]
        delta = derive_signed_changes(products[0].semantic_atoms, products[1].semantic_atoms)
        if not delta:
            raise ValueError("every A/B pair requires a semantic nonselection delta")
        products[0] = CompiledGold(
            projection=_rebuild_gold_with_delta(products[0].projection, delta),
            alternatives=products[0].alternatives,
            semantic_atoms=products[0].semantic_atoms,
            answer_signature=products[0].answer_signature,
            community_rationales=products[0].community_rationales,
            alternative_candidate=products[0].alternative_candidate,
            rare_support_path=products[0].rare_support_path,
        )
        products[1] = CompiledGold(
            projection=_rebuild_gold_with_delta(products[1].projection, delta),
            alternatives=products[1].alternatives,
            semantic_atoms=products[1].semantic_atoms,
            answer_signature=products[1].answer_signature,
            community_rationales=products[1].community_rationales,
            alternative_candidate=products[1].alternative_candidate,
            rare_support_path=products[1].rare_support_path,
        )
        null_proof = None
        if assignment.null_nonselection_context_index is not None:
            null_index = assignment.null_nonselection_context_index
            base = products[null_index]
            base_context = contexts[null_index]
            perturbed_context = mutate_query_wording(
                base_context,
                (
                    "Using the same declared lens, scope, abstraction, horizon, and viewpoint, "
                    "express the identical ontology-construction request in alternate wording."
                ),
            )
            perturbed = compile_gold_projection(
                spec,
                perturbed_context,
                null_index,
                narrative,
                config,
                policy=CompilerPolicy.FIXED_REFERENCE,
                selected_for_review=spec.world_id in reviewed_worlds,
            )
            base_hash = canonical_sha256(
                tuple(normalized_decision_dict(item) for item in base.semantic_atoms)
            )
            changed_hash = canonical_sha256(
                tuple(normalized_decision_dict(item) for item in perturbed.semantic_atoms)
            )
            null_proof = NullCaseProof(
                world_id=spec.world_id,
                base_query_id=base_context.context_id,
                perturbation="wording-only-under-fixed-reference",
                base_nonselection_signature=base_hash,
                perturbed_nonselection_signature=changed_hash,
            )
        fixed_friendly_proof = None
        if (
            spec.split is BenchmarkSplit.HELD_OUT
            and BenchmarkFactor.FIXED_ONTOLOGY_FRIENDLY in spec.factors
        ):
            friendly_index = 2
            if assignment.compiler_policies[friendly_index] is not CompilerPolicy.FIXED_REFERENCE:
                raise ValueError("fixed-friendly case must use the frozen reference compiler")
            query_dependent = compile_gold_projection(
                spec,
                contexts[friendly_index],
                friendly_index,
                narrative,
                config,
                policy=CompilerPolicy.QUERY_DEPENDENT,
                selected_for_review=spec.world_id in reviewed_worlds,
            )
            fixed_friendly_proof = FixedFriendlyProof(
                world_id=spec.world_id,
                query_id=contexts[friendly_index].context_id,
                fixed_reference_semantic_hash=_compiled_product_semantic_hash(
                    products[friendly_index]
                ),
                query_dependent_semantic_hash=_compiled_product_semantic_hash(query_dependent),
            )
        reveals = tuple(
            QueryRevealArtifact(
                reveal_id=_opaque("reveal", spec.world_id, index),
                evidence_artifact_hash=artifact.content_hash,
                query=to_model_visible_query(context),
                revealed_at=context.revealed_at,
            )
            for index, context in enumerate(contexts)
        )
        scan_model_payload(preconstruction_request_payload(artifact))
        for reveal in reveals:
            scan_model_payload(model_request_payload(artifact, reveal))
        query_reveals_by_world[spec.world_id] = reveals  # type: ignore[assignment]
        scorer_artifacts[spec.world_id] = ScorerWorldArtifact(
            world_spec=spec,
            query_assignment=assignment,
            gold_projections=tuple(item.projection for item in products),
            alternatives=tuple(item.alternatives for item in products),
            semantic_atoms_by_query={
                context.context_id: product.semantic_atoms
                for context, product in zip(contexts, products, strict=True)
            },
            answer_signatures_by_query={
                context.context_id: product.answer_signature
                for context, product in zip(contexts, products, strict=True)
            },
            community_rationales_by_query={
                context.context_id: product.community_rationales
                for context, product in zip(contexts, products, strict=True)
            },
            rare_support_paths_by_query={
                context.context_id: product.rare_support_path
                for context, product in zip(contexts, products, strict=True)
            },
            fact_evidence_ids={
                key: tuple(value) for key, value in narrative.fact_evidence_ids.items()
            },
            fixed_friendly_proof=fixed_friendly_proof,
            null_case_proof=null_proof,
        )
        contexts_by_world[spec.world_id] = contexts
        narratives[spec.world_id] = narrative
    paraphrases = select_paraphrases(held_out, contexts_by_world, seeds, config)
    no_context = select_no_context_ablation(held_out, contexts_by_world, seeds)
    eligibility = build_eligibility_manifest(
        held_out, contexts_by_world, scorer_artifacts, narratives, config, seeds
    )
    coverage = _factor_coverage(held_out, contexts_by_world)
    validate_factor_quotas(config, coverage, eligibility)
    mutations = build_mutation_manifest(
        development,
        contexts_by_world,
        narratives,
        scorer_artifacts,
        config,
        seeds,
    )
    review_package = build_blind_review_package(
        review_selection,
        model_artifacts,
        contexts_by_world,
        scorer_artifacts,
    )
    review_bindings = build_review_projection_bindings(
        review_selection, review_package, contexts_by_world, scorer_artifacts
    )
    source_hashes = benchmark_source_hashes()
    draft_seal = HeldOutDraftSeal(
        seal_id=_opaque("draftseal", config.content_hash, review_package.content_hash),
        sealed_at=_timestamp(config, 180),
        generator_revision=config.generator_revision,
        generator_source_hash=source_hashes["synthetic_benchmark.py"],
        runtime_source_hash=source_hashes["benchmark_runtime.py"],
        compiler_dependency_hashes=source_hashes,
        configuration_hash=config.content_hash,
        seed_manifest_hash=seeds.content_hash,
        review_package_hash=review_package.content_hash,
        review_binding_manifest_hash=review_bindings.content_hash,
        entries=tuple(
            HeldOutSealEntry(
                world_id=spec.world_id,
                world_spec_hash=spec.content_hash,
                model_eligible_artifact_hash=model_artifacts[spec.world_id].content_hash,
                query_context_hashes=tuple(
                    item.content_hash for item in contexts_by_world[spec.world_id]
                ),
                gold_projection_hashes=tuple(
                    item.content_hash for item in scorer_artifacts[spec.world_id].gold_projections
                ),
                alternative_set_hashes=tuple(
                    item.content_hash for item in scorer_artifacts[spec.world_id].alternatives
                ),
            )
            for spec in sorted(held_out, key=lambda item: item.world_id)
        ),
    )
    build = BenchmarkBuild(
        configuration=config,
        seeds=seeds,
        world_specs=specs,
        query_sampling_audit=audit,
        model_artifacts=model_artifacts,
        neutral_evidence_artifacts=neutral_evidence_artifacts,
        evidence_equivalence_certificates=evidence_equivalence_certificates,
        query_reveals_by_world=query_reveals_by_world,
        scorer_artifacts=scorer_artifacts,
        contexts_by_world=contexts_by_world,
        narratives=narratives,
        paraphrases=paraphrases,
        no_context_selection=no_context,
        eligibility=eligibility,
        mutation_manifest=mutations,
        review_selection=review_selection,
        review_package=review_package,
        review_bindings=review_bindings,
        draft_seal=draft_seal,
        factor_coverage=coverage,
        rejected_candidate=_REJECTED_CANDIDATE,
    )
    validate_compiled_benchmark(build)
    return build


class BenchmarkScientificAudit(ImmutableRecord):
    contrast_decision_count: Annotated[int, Field(ge=1)]
    invalid_contrast_anchor_count: Annotated[int, Field(ge=0)]
    rare_support_path_count: Annotated[int, Field(ge=1)]
    disconnected_rare_support_path_count: Annotated[int, Field(ge=0)]
    relevant_event_count: Annotated[int, Field(ge=1)]
    relevant_event_outside_scope_count: Annotated[int, Field(ge=0)]
    null_case_count: Literal[3]
    arbitrary_review_id_rejected: Literal[True]

    @model_validator(mode="after")
    def blocking_failures_are_zero(self) -> Self:
        if (
            self.invalid_contrast_anchor_count
            or self.disconnected_rare_support_path_count
            or self.relevant_event_outside_scope_count
        ):
            raise ValueError(
                "scientific audit contains a blocking semantic integrity failure: "
                f"invalid_anchors={self.invalid_contrast_anchor_count}, "
                f"disconnected_paths={self.disconnected_rare_support_path_count}, "
                f"events_outside_scope={self.relevant_event_outside_scope_count}"
            )
        return self


def scientific_audit(build: BenchmarkBuild) -> BenchmarkScientificAudit:
    contrast_count = 0
    invalid_anchors = 0
    invalid_anchor_details: list[str] = []
    rare_paths = 0
    disconnected_paths = 0
    relevant_events = 0
    outside_scope = 0
    for spec in build.world_specs:
        model = build.model_artifacts[spec.world_id]
        scorer = build.scorer_artifacts[spec.world_id]
        contexts = build.contexts_by_world[spec.world_id]
        evidence_by_id = {item.evidence_id: item for item in model.evidence}
        for context, gold in zip(contexts, scorer.gold_projections, strict=True):
            for decision in gold.signed_contrast_decisions:
                contrast_count += 1
                cited_candidates = set().union(
                    *(_candidate_ids(evidence_by_id[item]) for item in decision.evidence_ids)
                )
                if not set(decision.anchor_ids).issubset(cited_candidates):
                    invalid_anchors += 1
                    invalid_anchor_details.append(
                        f"{spec.world_id}:{decision.contrast_decision_id}:"
                        f"missing={sorted(set(decision.anchor_ids) - cited_candidates)}"
                    )
            assertion_by_id = {item.assertion_id: item for item in gold.qualified_assertions}
            for annotation in gold.assertion_annotations:
                if not (annotation.is_rare and annotation.is_pivotal):
                    continue
                rare_paths += 1
                typed_path = scorer.rare_support_paths_by_query[context.context_id]
                path = annotation.support_path_assertion_ids
                if (
                    typed_path is None
                    or not path
                    or path[0] != annotation.assertion_id
                    or path[-1] != typed_path.outcome_assertion_id
                    or tuple(
                        (item.source_fact_id, item.target_fact_id) for item in typed_path.steps
                    )
                    != spec.causal_dependencies
                    or any(
                        step.source_assertion_id not in assertion_by_id
                        or step.causal_assertion_id not in assertion_by_id
                        or step.target_assertion_id not in assertion_by_id
                        or not set(step.evidence_ids).issubset(evidence_by_id)
                        or assertion_by_id[step.causal_assertion_id].subject_id
                        != _assertion_directed_subject(assertion_by_id[step.source_assertion_id])
                        or assertion_by_id[step.causal_assertion_id].object_id
                        != _assertion_directed_subject(assertion_by_id[step.target_assertion_id])
                        for step in typed_path.steps
                    )
                ):
                    disconnected_paths += 1
            relevance = {item.target_id: item.is_relevant for item in gold.relevance}
            effective_scope = context.story_scope
            for event in gold.events:
                if not relevance[event.event_id]:
                    continue
                relevant_events += 1
                start, end = _time_bounds(effective_scope)
                occurrence_start, occurrence_end = _time_bounds(event.occurrence_time)
                if occurrence_start > end or occurrence_end < start:
                    outside_scope += 1
        # The exact annotation must be the exact semantic compiler diff.
        expected = derive_signed_changes(
            scorer.semantic_atoms_by_query[contexts[0].context_id],
            scorer.semantic_atoms_by_query[contexts[1].context_id],
        )
        if scorer.gold_projections[0].signed_contrast_decisions != expected:
            raise ValueError("A annotation is not the exact compiled A-to-B semantic diff")
        if scorer.gold_projections[1].signed_contrast_decisions != expected:
            raise ValueError("B annotation is not the exact compiled A-to-B semantic diff")
    null_count = sum(
        scorer.null_case_proof is not None
        for scorer in build.scorer_artifacts.values()
        if scorer.world_spec.split is BenchmarkSplit.HELD_OUT
    )
    # Regression for the former count-only response validation bug.
    arbitrary_rejected = False
    templates = _package_review_items(build.review_package)
    fake_items = tuple(
        IndependentReviewResponseItem(
            review_item_id=_opaque("arbitrary", index),
            blind_projection_id=template.blind_projection_id,
            criterion=template.criterion,
            disposition=ReviewDisposition.AGREE,
            notes="synthetic invalid binding probe",
        )
        for index, template in enumerate(templates.values())
    )
    try:
        fake = IndependentReviewResponse(
            package_hash=build.review_package.content_hash,
            reviewer_pseudonym="binding-probe",
            reviewed_at=build.configuration.frozen_at,
            items=fake_items,
        )
        bind_review_response(build.review_package, fake)
    except (ValueError, ReviewLifecycleError):
        arbitrary_rejected = True
    if not arbitrary_rejected:
        raise ValueError("arbitrary review response IDs were accepted")
    if invalid_anchor_details:
        raise ValueError(
            "invalid contrast anchor evidence binding: " + "; ".join(invalid_anchor_details)
        )
    return BenchmarkScientificAudit(
        contrast_decision_count=contrast_count,
        invalid_contrast_anchor_count=invalid_anchors,
        rare_support_path_count=rare_paths,
        disconnected_rare_support_path_count=disconnected_paths,
        relevant_event_count=relevant_events,
        relevant_event_outside_scope_count=outside_scope,
        null_case_count=null_count,
        arbitrary_review_id_rejected=True,
    )


def _world_structure_signature(spec: WorldSpec) -> str:
    """Normalize away names/IDs so template duplication cannot hide behind surfaces."""

    ref_map = {
        **{item.persona_id: f"person:{index}" for index, item in enumerate(spec.personas)},
        **{
            item.collective_id: f"collective:{index}" for index, item in enumerate(spec.collectives)
        },
        **{item.place_id: f"place:{index}" for index, item in enumerate(spec.places)},
        **{item.event_ref: f"event:{index}" for index, item in enumerate(spec.events)},
    }
    fact_map = {item.fact_id: f"fact:{index}" for index, item in enumerate(spec.facts)}
    return canonical_json(
        {
            "personas": tuple(
                (
                    ref_map[item.home_collective_id],
                    ref_map.get(item.predecessor_persona_id),
                    item.office_title is not None,
                )
                for item in spec.personas
            ),
            "events": tuple(
                (
                    tuple(ref_map[ref] for ref in item.participant_refs),
                    ref_map[item.place_ref],
                    item.story_position,
                    item.duration,
                    ref_map.get(item.causal_predecessor_ref),
                    item.focal,
                )
                for item in spec.events
            ),
            "facts": tuple(
                (
                    ref_map[item.subject_ref],
                    item.relation,
                    ref_map[item.object_ref],
                    item.story_position,
                    item.validity_end,
                    item.disclosure_order,
                    item.role.value,
                    item.commitment.value,
                    ref_map.get(item.holder_ref),
                    ref_map.get(item.event_ref),
                    tuple(lens.value for lens in item.relevant_lenses),
                    item.repetition_count,
                    item.conflict_group is not None,
                )
                for item in spec.facts
            ),
            "temporal": tuple(
                (ref_map[left], ref_map[right]) for left, right in spec.temporal_precedence
            ),
            "causal": tuple(
                (fact_map[left], fact_map[right]) for left, right in spec.causal_dependencies
            ),
        }
    )


def validate_compiled_benchmark(build: BenchmarkBuild) -> None:
    development = [item for item in build.world_specs if item.split is BenchmarkSplit.DEVELOPMENT]
    held_out = [item for item in build.world_specs if item.split is BenchmarkSplit.HELD_OUT]
    if len(development) != 4 or len(held_out) != 12:
        raise ValueError("benchmark split must remain four development and twelve held-out worlds")
    if Counter(item.difficulty for item in held_out) != Counter({item: 4 for item in Difficulty}):
        raise ValueError("held-out difficulty strata must be exactly 4/4/4")
    if Counter(item.horizon_block for item in held_out) != Counter(
        {HorizonBlock.INTERMEDIATE: 6, HorizonBlock.FINAL: 6}
    ):
        raise ValueError("held-out horizons must be balanced 6/6")
    world_ids = {item.world_id for item in build.world_specs}
    evidence_namespaces = (
        set(build.model_artifacts),
        set(build.neutral_evidence_artifacts),
        set(build.evidence_equivalence_certificates),
        set(build.narratives),
    )
    if any(namespace != world_ids for namespace in evidence_namespaces):
        raise ValueError("every world requires one model, neutral, and narrative artifact")
    held_structure_ids = {item.structure_template_id for item in held_out}
    dev_structure_ids = {item.structure_template_id for item in development}
    held_signatures = {_world_structure_signature(item) for item in held_out}
    dev_signatures = {_world_structure_signature(item) for item in development}
    if (
        len(held_structure_ids) != 12
        or held_structure_ids & dev_structure_ids
        or len(held_signatures) != 12
        or held_signatures & dev_signatures
    ):
        raise ValueError("held-out worlds must use twelve genuinely distinct unseen structures")

    validate_query_allocation(build.query_sampling_audit)
    replayed_sampling = sample_held_out_queries(held_out, build.seeds)
    if replayed_sampling.content_hash != build.query_sampling_audit.content_hash:
        raise ValueError("query allocation cannot be reproduced from its registered seeds")
    if len(build.query_sampling_audit.conditional_allocations) != 12:
        raise ValueError("every held-out query block needs a conditional allocation record")
    if len(build.paraphrases.pairs) != 12:
        raise ValueError("paraphrase subset must contain twelve contexts")
    if Counter(
        _lens_for_context(item.paraphrase_context) for item in build.paraphrases.pairs
    ) != Counter({item: 2 for item in LensFamily}):
        raise ValueError("paraphrase subset must be exactly balanced by lens")
    if any(
        (item.base_renderer_id, item.paraphrase_renderer_id) != build.configuration.query_renderers
        for item in build.paraphrases.pairs
    ):
        raise ValueError("paraphrase renderer provenance differs from configuration")

    context_lookup = {
        context.context_id: (spec.world_id, spec.difficulty, index, context)
        for spec in held_out
        for index, context in enumerate(build.contexts_by_world[spec.world_id])
    }
    if len({context_lookup[item][0] for item in build.eligibility.rare_guard_context_ids}) != 8:
        raise ValueError("rare-guard contexts must come from eight distinct worlds")
    temporal_epistemic_worlds = {
        context_lookup[item][0] for item in build.eligibility.temporal_epistemic_context_ids
    }
    if len(temporal_epistemic_worlds) != 8:
        raise ValueError("temporal/epistemic contexts must come from eight distinct worlds")
    if {context_lookup[item][1] for item in build.eligibility.temporal_case_context_ids} != set(
        Difficulty
    ) or {context_lookup[item][1] for item in build.eligibility.holder_status_context_ids} != set(
        Difficulty
    ):
        raise ValueError("both temporal and holder-status cases must span every difficulty")
    if any(
        context_lookup[item][3].viewpoint is not None
        for item in build.eligibility.temporal_case_context_ids
    ) or any(
        context_lookup[item][3].viewpoint is None
        for item in build.eligibility.holder_status_context_ids
    ):
        raise ValueError("temporal/epistemic eligibility case labels disagree with viewpoints")
    replayed_eligibility = build_eligibility_manifest(
        held_out,
        build.contexts_by_world,
        build.scorer_artifacts,
        build.narratives,
        build.configuration,
        build.seeds,
    )
    if replayed_eligibility.content_hash != build.eligibility.content_hash:
        raise ValueError("ablation eligibility cannot be reproduced from actual compiled cases")

    relevant_denial_count = 0
    coordinate_difference_count = 0
    fixed_friendly_count = 0
    for spec in build.world_specs:
        model = build.model_artifacts[spec.world_id]
        neutral = build.neutral_evidence_artifacts[spec.world_id]
        equivalence = build.evidence_equivalence_certificates[spec.world_id]
        narrative = build.narratives[spec.world_id]
        scorer = build.scorer_artifacts[spec.world_id]
        contexts = build.contexts_by_world[spec.world_id]
        reveals = build.query_reveals_by_world[spec.world_id]
        if neutral.snapshot != narrative.snapshot or neutral.evidence != narrative.evidence:
            raise ValueError("neutral evidence must be compiled directly from the narrative")
        verify_neutral_evidence_projection(neutral, model, equivalence)
        if len(contexts) != 3 or len(reveals) != 3 or len(scorer.gold_projections) != 3:
            raise ValueError("every world must have three isolated reveals and gold projections")
        if model.snapshot.sealed_at >= min(item.revealed_at for item in contexts):
            raise ValueError("query reveal must occur strictly after the query-blind evidence seal")
        if any(
            reveal.evidence_artifact_hash != model.content_hash
            or reveal.query.content_hash != to_model_visible_query(context).content_hash
            or reveal.revealed_at != context.revealed_at
            for reveal, context in zip(reveals, contexts, strict=True)
        ):
            raise ValueError("query reveal is not bound to one context and its sealed evidence")
        if "query" in preconstruction_request_payload(model):
            raise ValueError("pre-query request contains a revealed query")
        evidence_by_id = {item.evidence_id: item for item in model.evidence}
        coordinate_difference_count += sum(
            (left.discourse_position.passage_order - right.discourse_position.passage_order)
            * (
                build.narratives[spec.world_id].revelation_order_by_evidence_id[left.evidence_id]
                - build.narratives[spec.world_id].revelation_order_by_evidence_id[right.evidence_id]
            )
            < 0
            for left, right in itertools.combinations(model.evidence, 2)
        )
        if any(
            not any(
                clue.normalized_expression
                == "revelation-order-"
                + str(
                    build.narratives[spec.world_id].revelation_order_by_evidence_id[
                        evidence.evidence_id
                    ]
                )
                for clue in evidence.temporal_clues
            )
            for evidence in model.evidence
        ):
            raise ValueError("model-visible evidence omits its query-blind revelation coordinate")
        for index, (context, gold, reveal) in enumerate(
            zip(contexts, scorer.gold_projections, reveals, strict=True)
        ):
            request = model_request_payload(model, reveal)
            if request["snapshot_hash"] != model.snapshot.content_hash:
                raise ValueError("query request names the wrong evidence snapshot")
            node_count = len(gold.entity_partition) + len(gold.events)
            limits = {
                Difficulty.EASY: (10, 12),
                Difficulty.MEDIUM: (13, 16),
                Difficulty.HARD: (17, 20),
            }[spec.difficulty]
            if not limits[0] <= node_count <= limits[1] or node_count > context.budgets.node_budget:
                raise ValueError("gold projection violates its registered node budget")
            if len(gold.qualified_assertions) > context.budgets.assertion_budget:
                raise ValueError("gold projection violates its registered assertion budget")

            relevance = {item.target_id: item.is_relevant for item in gold.relevance}
            node_ids = {item.cluster_id for item in gold.entity_partition} | {
                item.event_id for item in gold.events
            }
            assertion_ids = {item.assertion_id for item in gold.qualified_assertions}
            predicate_ids = {item.predicate_id for item in gold.local_schema.predicates}
            known_semantic_objects = node_ids | assertion_ids | predicate_ids
            atoms = scorer.semantic_atoms_by_query[context.context_id]
            if any(not set(item.object_ids).issubset(known_semantic_objects) for item in atoms):
                raise ValueError("semantic atom names a nonexistent compiled semantic object")
            for slot in ("qualification/story-scope", "qualification/epistemic-frame"):
                atom = next(item for item in atoms if item.slot_key == slot)
                if not atom.object_ids:
                    raise ValueError(f"{slot} is decorative rather than assertion-level semantics")

            for local_type in gold.local_schema.contextual_types:
                kind = local_type.parent_upper_type.removeprefix("upper.")
                for evidence_id in local_type.evidence_ids:
                    evidence = evidence_by_id[evidence_id]
                    supported = any(
                        item.provisional_type == f"{kind}_candidate"
                        for item in evidence.mention_candidates
                    ) or (kind == "event" and bool(evidence.event_candidates))
                    if not supported:
                        raise ValueError(
                            "local type cites evidence with no matching candidate type"
                        )
            assertions_by_predicate: dict[str, set[str]] = defaultdict(set)
            for assertion in gold.qualified_assertions:
                assertions_by_predicate[assertion.predicate_id].update(assertion.evidence_ids)
            for predicate in gold.local_schema.predicates:
                if (
                    not predicate.evidence_ids
                    or set(predicate.evidence_ids)
                    != (assertions_by_predicate[predicate.predicate_id])
                ):
                    raise ValueError("local predicate evidence does not ground its assertions")
                if (
                    predicate.label == "causally enables"
                    and predicate.parent_upper_relation != "upper.causal"
                ):
                    raise ValueError("causal schema relation has the wrong upper relation")

            attributed = [
                assertion
                for assertion in gold.qualified_assertions
                if assertion.epistemic_scope is not None
            ]
            if not attributed or any(
                item.narrative_commitment
                not in {NarrativeCommitment.HOLDER_ATTRIBUTED, NarrativeCommitment.CONTESTED}
                for item in attributed
            ):
                raise ValueError("reported/denied propositions lost holder-level qualification")
            if context.viewpoint is not None:
                holder_cluster = next(
                    item.cluster_id
                    for item in gold.entity_partition
                    if context.viewpoint.holder_id in item.mention_candidate_ids
                )
                relevant_attributed = [item for item in attributed if relevance[item.assertion_id]]
                if not relevant_attributed or any(
                    item.epistemic_scope is None
                    or item.epistemic_scope.holder_id != holder_cluster
                    or item.narrative_commitment is not NarrativeCommitment.HOLDER_ATTRIBUTED
                    for item in relevant_attributed
                ):
                    raise ValueError("viewpoint query does not alter relevant holder assertions")
                relevant_denial_count += sum(
                    item.epistemic_scope is not None
                    and item.epistemic_scope.attitude is EpistemicAttitude.DENIED
                    for item in relevant_attributed
                )

            alternative_set = scorer.alternatives[index]
            if not evaluate_alternative_set(alternative_set, _alternative_view(gold)).matched:
                raise ValueError("executable gold alternative rejected its primary projection")
            if len(alternative_set.constraint_alternatives) != 1:
                raise ValueError("synthetic projection must have one closed joint alternative")
            constraint = alternative_set.constraint_alternatives[0].constraints
            if (
                len(constraint) != 1
                or constraint[0].field_path != AlternativeField.JOINT_REPRESENTATION
            ):
                raise ValueError("permissible alternatives permit an invalid Cartesian hybrid")
            primary, alternate = map(json.loads, constraint[0].accepted_values)
            differing = {key for key in primary if primary[key] != alternate[key]}
            if differing != {"assertion_direction_signature", "assertion_signature"}:
                raise ValueError("joint alternative changes fields beyond one inverse encoding")
            primary_assertion = json.loads(primary["assertion_signature"])
            alternate_assertion = json.loads(alternate["assertion_signature"])
            if not (
                primary_assertion["direction"] != alternate_assertion["direction"]
                and primary_assertion["subject_target_id"]
                == alternate_assertion["object_target_id"]
                and primary_assertion["object_target_id"]
                == alternate_assertion["subject_target_id"]
            ):
                raise ValueError("inverse assertion alternative is not direction/endpoint coupled")
            compile_alignment_alternatives(gold, alternative_set)

            path = scorer.rare_support_paths_by_query[context.context_id]
            rare_annotation = next(item for item in gold.assertion_annotations if item.is_rare)
            if (path is not None) != rare_annotation.is_pivotal:
                raise ValueError("rare pivotal label and typed support path disagree")
            if (
                path is not None
                and tuple((item.source_fact_id, item.target_fact_id) for item in path.steps)
                != spec.causal_dependencies
            ):
                raise ValueError("rare support path is not the registered directed causal chain")

            for rationale in scorer.community_rationales_by_query[context.context_id]:
                if not rationale.evidence_ids or not set(rationale.evidence_ids).issubset(
                    model.snapshot.eligible_evidence_ids
                ):
                    raise ValueError("community assignments must have evidence-based rationales")

            if (
                spec.split is BenchmarkSplit.HELD_OUT
                and BenchmarkFactor.FIXED_ONTOLOGY_FRIENDLY in spec.factors
                and scorer.query_assignment.compiler_policies[index]
                is CompilerPolicy.FIXED_REFERENCE
            ):
                proof = scorer.fixed_friendly_proof
                if (
                    proof is None
                    or proof.query_id != context.context_id
                    or proof.fixed_reference_semantic_hash != proof.query_dependent_semantic_hash
                ):
                    raise ValueError("fixed-friendly label lacks exact actual-query proof")
                fixed_friendly_count += 1

        if not any(path is not None for path in scorer.rare_support_paths_by_query.values()):
            raise ValueError("each held-out world must contribute a rare-pivotal denominator")
        expects_fixed_proof = (
            spec.split is BenchmarkSplit.HELD_OUT
            and BenchmarkFactor.FIXED_ONTOLOGY_FRIENDLY in spec.factors
        )
        if (scorer.fixed_friendly_proof is not None) != expects_fixed_proof:
            raise ValueError("fixed-friendly proof population disagrees with registered factors")
        atoms = tuple(atom for values in scorer.semantic_atoms_by_query.values() for atom in values)
        if sum(item.slot_key.startswith("causal/") for item in atoms) < len(
            spec.causal_dependencies
        ):
            raise ValueError("causal_dependencies were not consumed")
        predecessor_count = sum(item.causal_predecessor_ref is not None for item in spec.events)
        if (
            sum(item.slot_key.startswith("event-predecessor/") for item in atoms)
            < predecessor_count
        ):
            raise ValueError("event causal_predecessor_ref was not consumed")
        if sum(item.slot_key.startswith("home-collective/") for item in atoms) < len(spec.personas):
            raise ValueError("persona home_collective_id was not consumed")
        has_conflict_atom = any(item.slot_key.startswith("conflict/") for item in atoms)
        expects_conflict = BenchmarkFactor.INCOMPLETE_CONFLICTING_EVIDENCE in spec.factors
        if has_conflict_atom != expects_conflict:
            raise ValueError("fact conflict_group coverage disagrees with compiled semantics")

    if coordinate_difference_count == 0:
        raise ValueError("discourse and revelation orderings were accidentally conflated")
    if relevant_denial_count == 0:
        raise ValueError("no denied proposition is relevant under an actual holder viewpoint")
    if fixed_friendly_count < 4:
        raise ValueError("registered fixed-friendly queries do not yield usable actual projections")
    proof_queries = {item.query_id for item in build.mutation_manifest.rare_deletion_proofs}
    if not set(build.eligibility.rare_guard_context_ids).issubset(proof_queries):
        raise ValueError("rare-guard eligibility is not backed by regenerated deletion proofs")
    if len(build.review_package.worlds) != 3:
        raise ValueError("review package must contain three selected worlds")
    replayed_bindings = build_review_projection_bindings(
        build.review_selection,
        build.review_package,
        build.contexts_by_world,
        build.scorer_artifacts,
    )
    if (
        replayed_bindings.content_hash != build.review_bindings.content_hash
        or build.draft_seal.review_binding_manifest_hash != build.review_bindings.content_hash
    ):
        raise ValueError("blind review package is not bound to the sealed scorer gold")
    try:
        require_independent_review_complete(build.draft_seal, None, None, None, None, None, None)
    except IndependentReviewGateError:
        pass
    else:
        raise ValueError("pending draft review seal incorrectly authorized held-out launch")
    scientific_audit(build)


def _json_bytes(value: Any) -> bytes:
    return (canonical_json(value) + "\n").encode("utf-8")


def _namespace(
    relative_path: str,
) -> Literal["condition_input", "model_visible", "scorer_only", "manifest"]:
    first = Path(relative_path).parts[0]
    if first == CONDITION_INPUT_DIRECTORY:
        return "condition_input"
    if first == MODEL_VISIBLE_DIRECTORY:
        return "model_visible"
    if first == SCORER_ONLY_DIRECTORY:
        return "scorer_only"
    return "manifest"


def build_file_payloads(
    build: BenchmarkBuild,
) -> tuple[dict[str, bytes], SyntheticBenchmarkManifest]:
    payloads: dict[str, bytes] = {}
    routing: list[ModelRoutingEntry] = []
    runtime_artifacts = sorted(build.model_artifacts.items(), key=lambda item: item[1].artifact_id)
    for world_id, artifact in runtime_artifacts:
        neutral = build.neutral_evidence_artifacts[world_id]
        equivalence = build.evidence_equivalence_certificates[world_id]
        neutral_relative = (
            f"{CONDITION_INPUT_DIRECTORY}/{NEUTRAL_EVIDENCE_DIRECTORY}/{neutral.artifact_id}"
        )
        neutral_manifest = RuntimeStagingManifest(
            stage_id=_opaque("neutralstage", neutral.content_hash),
            stage_kind=RuntimeStageKind.NEUTRAL_EVIDENCE,
            artifact_hashes=(neutral.content_hash, equivalence.content_hash),
            file_names=("neutral_evidence.json", "equivalence.json"),
        )
        payloads[f"{neutral_relative}/neutral_evidence.json"] = _json_bytes(neutral)
        payloads[f"{neutral_relative}/equivalence.json"] = _json_bytes(equivalence)
        payloads[f"{neutral_relative}/manifest.json"] = _json_bytes(neutral_manifest)
        prequery_relative = f"{MODEL_VISIBLE_DIRECTORY}/prequery_stages/{artifact.artifact_id}"
        prequery_manifest = RuntimeStagingManifest(
            stage_id=_opaque("stage", artifact.content_hash, "prequery"),
            stage_kind=RuntimeStageKind.PREQUERY_EVIDENCE,
            artifact_hashes=(artifact.content_hash,),
            file_names=("evidence.json",),
        )
        payloads[f"{prequery_relative}/evidence.json"] = _json_bytes(artifact)
        payloads[f"{prequery_relative}/manifest.json"] = _json_bytes(prequery_manifest)
        query_stage_paths: list[str] = []
        reveals = build.query_reveals_by_world[world_id]
        for reveal in reveals:
            query_relative = f"{MODEL_VISIBLE_DIRECTORY}/query_stages/{reveal.reveal_id}"
            query_manifest = RuntimeStagingManifest(
                stage_id=_opaque("stage", reveal.content_hash, "query"),
                stage_kind=RuntimeStageKind.QUERY_REVEALED,
                artifact_hashes=(artifact.content_hash, reveal.content_hash),
                file_names=("evidence.json", "query.json"),
            )
            payloads[f"{query_relative}/evidence.json"] = _json_bytes(artifact)
            payloads[f"{query_relative}/query.json"] = _json_bytes(reveal)
            payloads[f"{query_relative}/manifest.json"] = _json_bytes(query_manifest)
            query_stage_paths.append(query_relative)
        routing.append(
            ModelRoutingEntry(
                world_id=world_id,
                split=build.scorer_artifacts[world_id].world_spec.split,
                artifact_id=artifact.artifact_id,
                relative_prequery_stage_path=prequery_relative,
                relative_neutral_evidence_stage_path=neutral_relative,
                relative_query_stage_paths=tuple(query_stage_paths),
                artifact_hash=artifact.content_hash,
                neutral_evidence_artifact_hash=neutral.content_hash,
                evidence_equivalence_certificate_hash=equivalence.content_hash,
                query_reveal_hashes=tuple(item.content_hash for item in reveals),
            )
        )
    for spec in sorted(build.world_specs, key=lambda item: item.world_id):
        payloads[f"{SCORER_ONLY_DIRECTORY}/{spec.split.value}/{spec.world_id}.json"] = _json_bytes(
            build.scorer_artifacts[spec.world_id]
        )
    payloads[f"{SCORER_ONLY_DIRECTORY}/routing/model_artifacts.json"] = _json_bytes(tuple(routing))
    payloads[f"{SCORER_ONLY_DIRECTORY}/sampling/held_out_query_allocation.json"] = _json_bytes(
        build.query_sampling_audit
    )
    payloads[f"{SCORER_ONLY_DIRECTORY}/sampling/paraphrase_selection.json"] = _json_bytes(
        build.paraphrases
    )
    payloads[f"{SCORER_ONLY_DIRECTORY}/sampling/a_no_context_selection.json"] = _json_bytes(
        build.no_context_selection
    )
    payloads[f"{SCORER_ONLY_DIRECTORY}/sampling/eligibility.json"] = _json_bytes(build.eligibility)
    payloads[f"{SCORER_ONLY_DIRECTORY}/mutations/expectations.json"] = _json_bytes(
        build.mutation_manifest
    )
    payloads[f"{SCORER_ONLY_DIRECTORY}/review/selection_manifest.json"] = _json_bytes(
        build.review_selection
    )
    payloads[f"{SCORER_ONLY_DIRECTORY}/review/blind_review_package.json"] = _json_bytes(
        build.review_package
    )
    payloads[f"{SCORER_ONLY_DIRECTORY}/review/scorer_bindings.json"] = _json_bytes(
        build.review_bindings
    )
    payloads[f"{SCORER_ONLY_DIRECTORY}/review/reviewer_response.schema.json"] = _json_bytes(
        IndependentReviewResponse.model_json_schema(mode="validation")
    )
    payloads[f"{SCORER_ONLY_DIRECTORY}/review/adjudication.schema.json"] = _json_bytes(
        ReviewAdjudication.model_json_schema(mode="validation")
    )
    payloads[f"{SCORER_ONLY_DIRECTORY}/review/final_reviewed_seal.schema.json"] = _json_bytes(
        FinalReviewedSeal.model_json_schema(mode="validation")
    )
    payloads[f"{SCORER_ONLY_DIRECTORY}/review/reviewed_projection.schema.json"] = _json_bytes(
        ReviewedProjectionArtifact.model_json_schema(mode="validation")
    )
    payloads[f"{SCORER_ONLY_DIRECTORY}/held_out/draft_seal.json"] = _json_bytes(build.draft_seal)
    payloads[f"{SCORER_ONLY_DIRECTORY}/provenance/rejected_candidate.json"] = _json_bytes(
        build.rejected_candidate
    )
    audit = scientific_audit(build)
    payloads[f"{SCORER_ONLY_DIRECTORY}/audits/scientific_integrity.json"] = _json_bytes(audit)
    payloads[f"{MANIFEST_DIRECTORY}/seed_manifest.json"] = _json_bytes(build.seeds)
    readme = (
        b"# Synthetic benchmark v3\n\n"
        b"Each directory below `condition_inputs/neutral_evidence` contains one query-blind "
        b"full-evidence artifact and a certificate proving its deterministic projection to "
        b"the corresponding model-visible artifact. It contains no query or scorer metadata. "
        b"Each directory below `model_visible/prequery_stages` is an exact evidence-only C1 "
        b"worker sandbox. Each directory below `model_visible/query_stages` contains the same "
        b"sealed evidence and exactly one revealed query. Workers receive one directory, never "
        b"the corpus root. Split routing, contexts, formal worlds, gold, mutations, reviewer "
        b"bindings, and the pending review package are scorer-only. The draft seal is not "
        b"authorization to run held-out conditions; a real independent response, typed "
        b"adjudication, and nine reviewed semantic artifacts must reproduce the final seal.\n"
    )
    payloads["README.md"] = readme
    generated = tuple(
        GeneratedFileRecord(
            relative_path=relative,
            byte_count=len(content),
            sha256=hashlib.sha256(content).hexdigest(),
            release_class=ReleaseClass.PUBLIC,
            namespace=_namespace(relative),
        )
        for relative, content in sorted(payloads.items())
    )
    assignments = build.query_sampling_audit.assignments
    held_out_specs = [item for item in build.world_specs if item.split is BenchmarkSplit.HELD_OUT]
    manifest = SyntheticBenchmarkManifest(
        generator_revision=build.configuration.generator_revision,
        generator_source_hash=build.draft_seal.generator_source_hash,
        runtime_source_hash=build.draft_seal.runtime_source_hash,
        compiler_dependency_hashes=build.draft_seal.compiler_dependency_hashes,
        configuration_hash=build.configuration.content_hash,
        seed_manifest_hash=build.seeds.content_hash,
        development_world_count=4,
        held_out_world_count=12,
        held_out_context_count=36,
        neutral_evidence_artifact_count=len(build.neutral_evidence_artifacts),
        evidence_equivalence_certificate_count=len(build.evidence_equivalence_certificates),
        paraphrase_context_count=12,
        review_world_count=3,
        review_projection_count=9,
        lens_counts=dict(Counter(lens for item in assignments for lens in item.lenses)),
        story_scope_counts=dict(
            Counter(scope for item in assignments for scope in item.story_scopes)
        ),
        abstraction_counts=dict(
            Counter(level for item in assignments for level in item.abstractions)
        ),
        viewpoint_context_count=len(assignments),
        horizon_counts=dict(Counter(item.horizon_block for item in held_out_specs)),
        node_budget_counts={
            str(key): value
            for key, value in Counter(item.node_budget for item in assignments).items()
        },
        factor_coverage=build.factor_coverage,
        eligibility_manifest_hash=build.eligibility.content_hash,
        no_context_selection_hash=build.no_context_selection.content_hash,
        query_sampling_audit_hash=build.query_sampling_audit.content_hash,
        paraphrase_manifest_hash=build.paraphrases.content_hash,
        review_selection_manifest_hash=build.review_selection.content_hash,
        review_package_hash=build.review_package.content_hash,
        review_binding_manifest_hash=build.review_bindings.content_hash,
        mutation_manifest_hash=build.mutation_manifest.content_hash,
        rejected_candidate_provenance_hash=build.rejected_candidate.content_hash,
        draft_seal_hash=build.draft_seal.content_hash,
        generated_files=generated,
    )
    payloads[f"{MANIFEST_DIRECTORY}/benchmark_manifest.json"] = _json_bytes(manifest)
    return payloads, manifest


class BenchmarkDriftError(RuntimeError):
    """Raised instead of overwriting or accepting a changed frozen benchmark."""


def materialize_benchmark(
    output_root: Path = DEFAULT_OUTPUT_ROOT,
    config_path: Path = DEFAULT_CONFIG_PATH,
) -> SyntheticBenchmarkManifest:
    build = compile_benchmark(config_path)
    payloads, manifest = build_file_payloads(build)
    if output_root.is_symlink():
        raise BenchmarkDriftError("synthetic output root cannot be a symlink")
    for relative, content in sorted(payloads.items()):
        target = output_root / relative
        if target.is_symlink():
            raise BenchmarkDriftError(f"refusing synthetic artifact symlink: {target}")
        if target.exists():
            if target.read_bytes() != content:
                raise BenchmarkDriftError(f"frozen artifact differs; refusing overwrite: {target}")
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(content)
    verify_materialized_benchmark(output_root, config_path)
    return manifest


def refresh_benchmark_lineage(
    output_root: Path = DEFAULT_OUTPUT_ROOT,
    config_path: Path = DEFAULT_CONFIG_PATH,
) -> SyntheticBenchmarkManifest:
    """Refresh source-bound draft lineage without changing benchmark contents.

    This operation is intentionally narrower than regeneration.  It is permitted
    only while the draft benchmark has exactly the expected file set and every
    non-lineage artifact is byte-identical to a fresh deterministic compilation.
    Any scientific-content drift, extra review artifact, or symlink fails closed.
    """

    build = compile_benchmark(config_path)
    payloads, manifest = build_file_payloads(build)
    if output_root.is_symlink():
        raise BenchmarkDriftError("synthetic output root cannot be a symlink")
    if not output_root.is_dir():
        raise BenchmarkDriftError("lineage refresh requires a materialized benchmark root")
    expected_files = set(payloads)
    actual_files = {
        str(path.relative_to(output_root)) for path in output_root.rglob("*") if path.is_file()
    }
    if actual_files != expected_files:
        raise BenchmarkDriftError(
            "lineage refresh requires the exact draft artifact set; "
            f"missing={sorted(expected_files - actual_files)}, "
            f"unexpected={sorted(actual_files - expected_files)}"
        )
    for relative, expected_content in sorted(payloads.items()):
        target = output_root / relative
        if target.is_symlink():
            raise BenchmarkDriftError(f"refusing synthetic artifact symlink: {target}")
        if relative in LINEAGE_REFRESH_PATHS:
            continue
        if target.read_bytes() != expected_content:
            raise BenchmarkDriftError(
                "lineage refresh cannot change scientific benchmark content: " f"{target}"
            )
    for relative in sorted(LINEAGE_REFRESH_PATHS):
        (output_root / relative).write_bytes(payloads[relative])
    verify_materialized_benchmark(output_root, config_path)
    return manifest


def verify_materialized_benchmark(
    output_root: Path = DEFAULT_OUTPUT_ROOT,
    config_path: Path = DEFAULT_CONFIG_PATH,
) -> None:
    manifest_path = output_root / MANIFEST_DIRECTORY / "benchmark_manifest.json"
    manifest = SyntheticBenchmarkManifest.model_validate_json(
        manifest_path.read_text(encoding="utf-8")
    )
    current_source_hashes = benchmark_source_hashes()
    if manifest.compiler_dependency_hashes != current_source_hashes:
        raise BenchmarkDriftError(
            "generated benchmark does not match all current benchmark semantic source hashes"
        )
    config = load_benchmark_configuration(config_path)
    if manifest.configuration_hash != config.content_hash:
        raise BenchmarkDriftError("generated benchmark does not match the supplied configuration")
    expected_files = {item.relative_path for item in manifest.generated_files} | {
        f"{MANIFEST_DIRECTORY}/benchmark_manifest.json"
    }
    actual_files = {
        str(path.relative_to(output_root)) for path in output_root.rglob("*") if path.is_file()
    }
    if actual_files != expected_files:
        raise BenchmarkDriftError(
            f"materialized file set drift; missing={sorted(expected_files - actual_files)}, "
            f"unexpected={sorted(actual_files - expected_files)}"
        )
    for record in manifest.generated_files:
        path = output_root / record.relative_path
        content = path.read_bytes()
        if (
            len(content) != record.byte_count
            or hashlib.sha256(content).hexdigest() != record.sha256
        ):
            raise BenchmarkDriftError(f"generated artifact hash/size drift: {path}")
    routing_raw = json.loads(
        (output_root / SCORER_ONLY_DIRECTORY / "routing/model_artifacts.json").read_text(
            encoding="utf-8"
        )
    )
    routing = tuple(ModelRoutingEntry.model_validate(item) for item in routing_raw)
    if len(routing) != 16 or len({item.world_id for item in routing}) != 16:
        raise BenchmarkDriftError("runtime routing must cover exactly sixteen unique worlds")
    for entry in routing:
        prequery_parts = Path(entry.relative_prequery_stage_path).parts
        neutral_parts = Path(entry.relative_neutral_evidence_stage_path).parts
        if (
            prequery_parts[:2] != (MODEL_VISIBLE_DIRECTORY, "prequery_stages")
            or len(prequery_parts) != 3
        ):
            raise BenchmarkDriftError("pre-query route escapes its isolated stage namespace")
        if (
            neutral_parts[:2] != (CONDITION_INPUT_DIRECTORY, NEUTRAL_EVIDENCE_DIRECTORY)
            or len(neutral_parts) != 3
        ):
            raise BenchmarkDriftError("neutral route escapes its condition-input namespace")
        if any(
            Path(relative).parts[:2] != (MODEL_VISIBLE_DIRECTORY, "query_stages")
            or len(Path(relative).parts) != 3
            for relative in entry.relative_query_stage_paths
        ):
            raise BenchmarkDriftError("query route escapes its isolated stage namespace")
        prequery_root = output_root / entry.relative_prequery_stage_path
        prequery_manifest = RuntimeStagingManifest.model_validate_json(
            (prequery_root / "manifest.json").read_text(encoding="utf-8")
        )
        evidence = load_staged_world(
            prequery_root / "evidence.json", prequery_root, prequery_manifest
        )
        if evidence.content_hash != entry.artifact_hash:
            raise BenchmarkDriftError("pre-query route names the wrong evidence artifact")
        neutral_root = output_root / entry.relative_neutral_evidence_stage_path
        neutral_manifest = RuntimeStagingManifest.model_validate_json(
            (neutral_root / "manifest.json").read_text(encoding="utf-8")
        )
        neutral, equivalence = load_staged_neutral_evidence(
            neutral_root,
            output_root / CONDITION_INPUT_DIRECTORY / NEUTRAL_EVIDENCE_DIRECTORY,
            neutral_manifest,
            evidence,
        )
        if (
            neutral.content_hash != entry.neutral_evidence_artifact_hash
            or equivalence.content_hash != entry.evidence_equivalence_certificate_hash
        ):
            raise BenchmarkDriftError("neutral route names the wrong evidence or certificate")
        for relative, reveal_hash in zip(
            entry.relative_query_stage_paths, entry.query_reveal_hashes, strict=True
        ):
            query_root = output_root / relative
            query_manifest = RuntimeStagingManifest.model_validate_json(
                (query_root / "manifest.json").read_text(encoding="utf-8")
            )
            query_evidence, reveal = load_staged_query(query_root, query_manifest)
            if (
                query_evidence.content_hash != entry.artifact_hash
                or reveal.content_hash != reveal_hash
            ):
                raise BenchmarkDriftError("query route differs from its sealed artifact hashes")
    draft = HeldOutDraftSeal.model_validate_json(
        (output_root / SCORER_ONLY_DIRECTORY / "held_out/draft_seal.json").read_text(
            encoding="utf-8"
        )
    )
    bindings = ReviewProjectionBindingManifest.model_validate_json(
        (output_root / SCORER_ONLY_DIRECTORY / "review/scorer_bindings.json").read_text(
            encoding="utf-8"
        )
    )
    package = BlindIndependentReviewPackage.model_validate_json(
        (output_root / SCORER_ONLY_DIRECTORY / "review/blind_review_package.json").read_text(
            encoding="utf-8"
        )
    )
    if (
        draft.review_package_hash != package.content_hash
        or draft.review_binding_manifest_hash != bindings.content_hash
        or bindings.package_hash != package.content_hash
        or manifest.review_binding_manifest_hash != bindings.content_hash
    ):
        raise BenchmarkDriftError("materialized independent-review lineage is inconsistent")


def assert_model_visible_path(
    path: Path,
    output_root: Path = DEFAULT_OUTPUT_ROOT,
) -> Path:
    stage_root = (output_root / MODEL_VISIBLE_DIRECTORY / "prequery_stages").resolve(strict=True)
    if path.is_symlink():
        raise GoldFirewallError("model artifact cannot be a symlink")
    resolved = path.resolve(strict=True)
    try:
        relative = resolved.relative_to(stage_root)
    except ValueError as error:
        raise GoldFirewallError("model input path is outside the pre-query stage root") from error
    if len(relative.parts) != 2 or relative.parts[1] != "evidence.json":
        raise GoldFirewallError("model input must be one evidence file in one isolated stage")
    return resolved


def load_model_eligible_world(
    path: Path,
    output_root: Path = DEFAULT_OUTPUT_ROOT,
) -> ModelEligibleWorldArtifact:
    safe = assert_model_visible_path(path, output_root)
    manifest = RuntimeStagingManifest.model_validate_json(
        (safe.parent / "manifest.json").read_text(encoding="utf-8")
    )
    return load_staged_world(safe, safe.parent, manifest)


def load_model_eligible_query(
    stage_path: Path,
    output_root: Path = DEFAULT_OUTPUT_ROOT,
) -> tuple[ModelEligibleWorldArtifact, QueryRevealArtifact]:
    stage_root = (output_root / MODEL_VISIBLE_DIRECTORY / "query_stages").resolve(strict=True)
    if stage_path.is_symlink():
        raise GoldFirewallError("query stage cannot be a symlink")
    resolved = stage_path.resolve(strict=True)
    try:
        relative = resolved.relative_to(stage_root)
    except ValueError as error:
        raise GoldFirewallError("query input is outside the isolated query stage root") from error
    if len(relative.parts) != 1 or not resolved.is_dir():
        raise GoldFirewallError("query input must be exactly one isolated stage directory")
    manifest = RuntimeStagingManifest.model_validate_json(
        (resolved / "manifest.json").read_text(encoding="utf-8")
    )
    return load_staged_query(resolved, manifest)
