"""Frozen contracts for the single 49-call Phase 4/5 GPU allocation.

The module has two deliberately separate surfaces.  The trusted compiler may read
the scorer-only *selection* manifests in order to freeze the registered subsets.
Its output, :class:`CombinedCallManifest`, contains only opaque runtime lineage and
model-eligible query semantics.  Model workers receive only the return value of
``CombinedCallSpec.model_visible_context`` and an independently verified evidence
packet; they can never resolve the selection manifests or scorer gold.

No model lifecycle or inference code lives here.  The production owner is in
``combined_gpu_production`` and is the sole component allowed to activate and stop
the one shared service.
"""

from __future__ import annotations

import hashlib
import math
from collections import Counter
from collections.abc import Mapping
from datetime import datetime
from enum import StrEnum
from pathlib import Path, PurePosixPath
from typing import Annotated, Literal, Self

from pydantic import AwareDatetime, Field, model_validator

from story_projection_onto.benchmark_runtime import scan_model_payload
from story_projection_onto.conditions.base import SCORED_PROJECTION_SCHEMA_HASH
from story_projection_onto.contracts import (
    ConditionName,
    Identifier,
    ImmutableRecord,
    ModelVisibleGenericConstructionContext,
    ModelVisibleQueryContext,
    QueryContext,
    RunOutcome,
    Sha256Digest,
    canonical_sha256,
    to_model_visible_context_for_condition,
)
from story_projection_onto.feedback_runtime import (
    FeedbackEpisodeKind,
    FeedbackProtocolConfiguration,
    load_feedback_protocol,
)
from story_projection_onto.held_out_primary import PublicStageReference
from story_projection_onto.synthetic_benchmark import (
    ANoContextSelectionManifest,
    EligibilityManifest,
    ParaphraseSelectionManifest,
    SeedDerivationManifest,
    SeedPurpose,
    SyntheticBenchmarkManifest,
    seed_for,
)

DEFAULT_CONFIGURATION_PATH = Path("configs/study/combined_gpu_block.json")
COMBINED_BASE_CALL_COUNT = 49
COMBINED_CALL_COUNTS: Mapping[str, int] = {
    "paraphrase_c2": 12,
    "scripted_feedback_c2": 6,
    "researcher_trace_c2": 3,
    "ablation_no_context": 12,
    "ablation_no_temporal_epistemic": 8,
    "ablation_no_rare_guard": 8,
}
ACTIVE_ABLATION_CONDITIONS = (
    ConditionName.A_NO_CONTEXT,
    ConditionName.A_NO_TEMPORAL_EPISTEMIC,
    ConditionName.A_NO_RARE_GUARD,
)


class CombinedBlockError(RuntimeError):
    """A frozen selection, prerequisite, or execution contract changed."""


class CombinedCallClass(StrEnum):
    PARAPHRASE_C2 = "paraphrase_c2"
    SCRIPTED_FEEDBACK_C2 = "scripted_feedback_c2"
    RESEARCHER_TRACE_C2 = "researcher_trace_c2"
    ABLATION_NO_CONTEXT = "ablation_no_context"
    ABLATION_NO_TEMPORAL_EPISTEMIC = "ablation_no_temporal_epistemic"
    ABLATION_NO_RARE_GUARD = "ablation_no_rare_guard"


class CombinedFileBinding(ImmutableRecord):
    """Exact tracked input read only by the trusted plan compiler."""

    relative_path: str = Field(min_length=1)
    file_sha256: Sha256Digest
    logical_content_hash: Sha256Digest | None = None

    @model_validator(mode="after")
    def bounded_relative_path(self) -> Self:
        path = PurePosixPath(self.relative_path)
        if path.is_absolute() or ".." in path.parts or "\\" in self.relative_path:
            raise ValueError("combined input binding must use a bounded relative path")
        return self


class CombinedBlockConfiguration(ImmutableRecord):
    configuration_id: Literal["combined-paraphrase-feedback-ablation-block-v1"]
    production_adapter_factory: Literal[
        "story_projection_onto.combined_gpu_factory:create_frozen_production_combined_bundle"
    ]
    output_root: Literal["artifacts/restricted/combined_gpu_block"]
    public_summary_path: Literal["artifacts/public/results/combined_gpu_block.json"]
    benchmark_manifest_path: str
    benchmark_manifest_file_sha256: Sha256Digest
    benchmark_manifest_hash: Sha256Digest
    seed_manifest_path: str
    seed_manifest_file_sha256: Sha256Digest
    seed_manifest_hash: Sha256Digest
    paraphrase_selection_path: str
    paraphrase_selection_file_sha256: Sha256Digest
    paraphrase_selection_hash: Sha256Digest
    no_context_selection_path: str
    no_context_selection_file_sha256: Sha256Digest
    no_context_selection_hash: Sha256Digest
    eligibility_selection_path: str
    eligibility_selection_file_sha256: Sha256Digest
    eligibility_selection_hash: Sha256Digest
    feedback_protocol_path: str
    feedback_protocol_file_sha256: Sha256Digest
    feedback_protocol_hash: Sha256Digest
    phase5_runner_configuration_path: str
    phase5_runner_configuration_file_sha256: Sha256Digest
    gpu_call_inventory_path: str
    gpu_call_inventory_file_sha256: Sha256Digest
    decoding_configuration_path: str
    decoding_configuration_file_sha256: Sha256Digest
    development_construction_path: str
    selected_model_repository: Literal["Qwen/Qwen3-8B-AWQ"]
    selected_model_revision: Literal["4da05a8edb55c6046cce958586c33b61da07bb79"]
    served_model_name: Literal["qwen3-8b-awq-fallback"]
    llm_seed_block: Literal[1]
    frozen_llm_seed: Literal[3864250958737859446]
    vllm_seed: Literal[1988649846]
    maximum_concurrency: Literal[1]
    model_load_count: Literal[1]
    base_call_count: Literal[49]
    paraphrase_call_count: Literal[12]
    scripted_feedback_call_count: Literal[6]
    researcher_trace_call_count: Literal[3]
    no_context_call_count: Literal[12]
    no_temporal_epistemic_call_count: Literal[8]
    no_rare_guard_call_count: Literal[8]
    base_call_p95_seconds: Literal[95]
    base_call_watchdog_seconds: Literal[150]
    model_load_p95_seconds: Literal[180]
    model_load_watchdog_seconds: Literal[300]
    combined_base_forecast_seconds: Literal[4655]
    combined_with_load_forecast_seconds: Literal[4835]
    activity_ceiling_seconds: Literal[4860]
    repair_reserve_class: Literal["reserve_short"]
    repair_watchdog_seconds: Literal[90]
    maximum_repairs_per_call: Literal[1]
    global_short_reserve_slot_count: Literal[4]
    scheduled_limit_seconds: Literal[32400]
    hard_limit_seconds: Literal[36000]
    protected_shutdown_margin_seconds: Literal[60]
    controller_owns_model_service_lifecycle: Literal[True]
    phase5_adapter_owns_model_service_lifecycle: Literal[False]
    append_only: Literal[True]
    runtime_namespace: Literal["gold_free"]
    public_payload_policy: Literal["hashes_counts_status_and_resource_measurements_only"]

    @model_validator(mode="after")
    def exact_registered_envelope(self) -> Self:
        paths = (
            self.benchmark_manifest_path,
            self.seed_manifest_path,
            self.paraphrase_selection_path,
            self.no_context_selection_path,
            self.eligibility_selection_path,
            self.feedback_protocol_path,
            self.phase5_runner_configuration_path,
            self.gpu_call_inventory_path,
            self.decoding_configuration_path,
            self.development_construction_path,
        )
        for value in paths:
            path = PurePosixPath(value)
            if path.is_absolute() or ".." in path.parts or "\\" in value:
                raise ValueError("combined configuration path must be safe and relative")
        counts = (
            self.paraphrase_call_count
            + self.scripted_feedback_call_count
            + self.researcher_trace_call_count
            + self.no_context_call_count
            + self.no_temporal_epistemic_call_count
            + self.no_rare_guard_call_count
        )
        if counts != self.base_call_count:
            raise ValueError("combined class counts do not sum to 49")
        if self.combined_base_forecast_seconds != (
            self.base_call_count * self.base_call_p95_seconds
        ):
            raise ValueError("combined base-call forecast changed")
        if self.combined_with_load_forecast_seconds != (
            self.combined_base_forecast_seconds + self.model_load_p95_seconds
        ):
            raise ValueError("combined load-inclusive forecast changed")
        if self.combined_with_load_forecast_seconds > self.activity_ceiling_seconds:
            raise ValueError("combined block exceeds its 1.35-hour activity ceiling")
        if self.scheduled_limit_seconds >= self.hard_limit_seconds:
            raise ValueError("scheduled limit must remain below the hard stop")
        if self.protected_shutdown_margin_seconds >= (
            self.hard_limit_seconds - self.scheduled_limit_seconds
        ):
            raise ValueError("shutdown margin consumes the registered contingency")
        if self.vllm_seed != self.frozen_llm_seed & (2**31 - 1):
            raise ValueError("combined vLLM seed differs from frozen LLM block 1")
        return self

    def tracked_bindings(self) -> tuple[CombinedFileBinding, ...]:
        return (
            CombinedFileBinding(
                relative_path=self.benchmark_manifest_path,
                file_sha256=self.benchmark_manifest_file_sha256,
                logical_content_hash=self.benchmark_manifest_hash,
            ),
            CombinedFileBinding(
                relative_path=self.seed_manifest_path,
                file_sha256=self.seed_manifest_file_sha256,
                logical_content_hash=self.seed_manifest_hash,
            ),
            CombinedFileBinding(
                relative_path=self.paraphrase_selection_path,
                file_sha256=self.paraphrase_selection_file_sha256,
                logical_content_hash=self.paraphrase_selection_hash,
            ),
            CombinedFileBinding(
                relative_path=self.no_context_selection_path,
                file_sha256=self.no_context_selection_file_sha256,
                logical_content_hash=self.no_context_selection_hash,
            ),
            CombinedFileBinding(
                relative_path=self.eligibility_selection_path,
                file_sha256=self.eligibility_selection_file_sha256,
                logical_content_hash=self.eligibility_selection_hash,
            ),
            CombinedFileBinding(
                relative_path=self.feedback_protocol_path,
                file_sha256=self.feedback_protocol_file_sha256,
                logical_content_hash=self.feedback_protocol_hash,
            ),
            CombinedFileBinding(
                relative_path=self.phase5_runner_configuration_path,
                file_sha256=self.phase5_runner_configuration_file_sha256,
            ),
            CombinedFileBinding(
                relative_path=self.gpu_call_inventory_path,
                file_sha256=self.gpu_call_inventory_file_sha256,
            ),
            CombinedFileBinding(
                relative_path=self.decoding_configuration_path,
                file_sha256=self.decoding_configuration_file_sha256,
            ),
        )


def _safe_bound_file(repository: Path, binding: CombinedFileBinding) -> bytes:
    root = repository.resolve(strict=True)
    current = root
    for part in PurePosixPath(binding.relative_path).parts:
        current = current / part
        if current.is_symlink():
            raise CombinedBlockError(f"symlinked combined input: {binding.relative_path}")
    resolved = current.resolve(strict=True)
    if not resolved.is_relative_to(root) or not resolved.is_file():
        raise CombinedBlockError(f"combined input escaped repository: {binding.relative_path}")
    raw = resolved.read_bytes()
    observed = hashlib.sha256(raw).hexdigest()
    if observed != binding.file_sha256:
        raise CombinedBlockError(f"combined input bytes changed: {binding.relative_path}")
    if len(raw) > 64 * 1024 * 1024:
        raise CombinedBlockError(f"combined input is unexpectedly large: {binding.relative_path}")
    return raw


def load_combined_configuration(
    repository: Path,
    path: Path = DEFAULT_CONFIGURATION_PATH,
) -> CombinedBlockConfiguration:
    relative = PurePosixPath(path.as_posix())
    if relative.is_absolute() or ".." in relative.parts:
        raise CombinedBlockError("combined configuration path must be repository-relative")
    source = repository.resolve(strict=True) / relative
    if source.is_symlink() or not source.is_file():
        raise CombinedBlockError("combined configuration must be a regular file")
    try:
        configuration = CombinedBlockConfiguration.model_validate_json(source.read_bytes())
    except Exception as error:
        raise CombinedBlockError(f"invalid combined configuration: {error}") from error
    for binding in configuration.tracked_bindings():
        _safe_bound_file(repository, binding)
    return configuration


class RegisteredCombinedSelections(ImmutableRecord):
    """Trusted-compiler view; this record must never enter a model payload."""

    benchmark: SyntheticBenchmarkManifest
    seeds: SeedDerivationManifest
    paraphrases: ParaphraseSelectionManifest
    no_context: ANoContextSelectionManifest
    eligibility: EligibilityManifest
    feedback: FeedbackProtocolConfiguration
    source_file_hashes: tuple[Sha256Digest, ...]
    model_visible: Literal[False] = False

    @model_validator(mode="after")
    def cross_manifest_hashes_match(self) -> Self:
        if (
            self.benchmark.seed_manifest_hash != self.seeds.content_hash
            or self.benchmark.paraphrase_manifest_hash != self.paraphrases.content_hash
            or self.benchmark.no_context_selection_hash != self.no_context.content_hash
            or self.benchmark.eligibility_manifest_hash != self.eligibility.content_hash
            or self.feedback.benchmark_draft_seal_hash != self.benchmark.draft_seal_hash
            or self.feedback.seed_manifest_hash != self.seeds.content_hash
        ):
            raise ValueError("combined selection manifests do not share one benchmark freeze")
        if len(self.source_file_hashes) != 6 or len(set(self.source_file_hashes)) != 6:
            raise ValueError("combined selections require six unique source-file hashes")
        llm_seed = seed_for(self.seeds, SeedPurpose.LLM_BLOCK_1)
        if llm_seed != self.feedback.llm_seed:
            raise ValueError("feedback and benchmark use different registered LLM seeds")
        return self


def combined_selection_registry_hash(
    selections: RegisteredCombinedSelections,
) -> Sha256Digest:
    """Digest the six frozen selection inputs without retaining scorer records."""

    return canonical_sha256(
        {
            "benchmark": selections.benchmark.content_hash,
            "seeds": selections.seeds.content_hash,
            "paraphrases": selections.paraphrases.content_hash,
            "no_context": selections.no_context.content_hash,
            "eligibility": selections.eligibility.content_hash,
            "feedback": selections.feedback.content_hash,
            "source_files": selections.source_file_hashes,
        }
    )


def load_registered_combined_selections(
    repository: Path,
    configuration: CombinedBlockConfiguration,
) -> RegisteredCombinedSelections:
    """Read the frozen subset manifests only in the trusted coordinator process."""

    bindings = configuration.tracked_bindings()[:6]
    raw = tuple(_safe_bound_file(repository, item) for item in bindings)
    try:
        benchmark = SyntheticBenchmarkManifest.model_validate_json(raw[0])
        seeds = SeedDerivationManifest.model_validate_json(raw[1])
        paraphrases = ParaphraseSelectionManifest.model_validate_json(raw[2])
        no_context = ANoContextSelectionManifest.model_validate_json(raw[3])
        eligibility = EligibilityManifest.model_validate_json(raw[4])
        feedback = load_feedback_protocol(
            repository.resolve(strict=True) / configuration.feedback_protocol_path
        )
    except Exception as error:
        raise CombinedBlockError(f"combined selection parsing failed: {error}") from error
    values = (benchmark, seeds, paraphrases, no_context, eligibility, feedback)
    expected = tuple(item.logical_content_hash for item in bindings)
    observed = tuple(item.content_hash for item in values)
    if observed != expected:
        raise CombinedBlockError("combined selection logical hashes changed")
    return RegisteredCombinedSelections(
        benchmark=benchmark,
        seeds=seeds,
        paraphrases=paraphrases,
        no_context=no_context,
        eligibility=eligibility,
        feedback=feedback,
        source_file_hashes=tuple(item.file_sha256 for item in bindings),
    )


class RestrictedArtifactPointer(ImmutableRecord):
    """Typed pointer to one restricted content-addressed predecessor object."""

    artifact_hash: Sha256Digest
    logical_content_hash: Sha256Digest
    object_kind: Identifier
    media_type: str = Field(min_length=1)
    release_class: Literal["restricted"] = "restricted"


class PublicEvidencePacketPointer(ImmutableRecord):
    """Exact pointer to a public synthetic evidence packet.

    Held-out packet materialization deliberately inherits the public release
    class of the synthetic evidence snapshot.  Keeping this type separate from
    :class:`RestrictedArtifactPointer` prevents a public packet from weakening
    the restricted-output contracts used everywhere else in this block.
    """

    artifact_hash: Sha256Digest
    logical_content_hash: Sha256Digest
    object_kind: Literal["evidence_packet"] = "evidence_packet"
    media_type: Literal[
        "application/vnd.story-projection.evidence-packet+json"
    ] = "application/vnd.story-projection.evidence-packet+json"
    release_class: Literal["public"] = "public"


class PrimaryC2Reference(ImmutableRecord):
    """Seed-1 primary result retained for paired scoring, never sent to the model."""

    call_id: Identifier
    call_spec_hash: Sha256Digest
    itt_record_hash: Sha256Digest
    outcome: RunOutcome
    output_artifact: RestrictedArtifactPointer | None = None
    condition_attempt_hash: Sha256Digest | None = None
    seed_block: Literal[1] = 1

    @model_validator(mode="after")
    def successful_output_is_explicit(self) -> Self:
        has_output = self.output_artifact is not None and self.condition_attempt_hash is not None
        if (self.outcome is RunOutcome.SUCCEEDED) != has_output:
            raise ValueError("primary C2 success and retained output lineage disagree")
        return self


class AblationPrequeryLineage(ImmutableRecord):
    """Condition-matching empty inventory sealed before any held-out query."""

    condition: Literal[
        ConditionName.A_NO_CONTEXT,
        ConditionName.A_NO_TEMPORAL_EPISTEMIC,
        ConditionName.A_NO_RARE_GUARD,
    ]
    preparation_hash: Sha256Digest
    inventory_hash: Sha256Digest
    receipt_hash: Sha256Digest
    preparation_artifact_hash: Sha256Digest
    inventory_artifact_hash: Sha256Digest
    completed_at: AwareDatetime


class CombinedQuerySource(ImmutableRecord):
    """Gold-free source for one already opened held-out context."""

    execution_id: Identifier
    held_out_call_manifest_hash: Sha256Digest
    held_out_execution_manifest_hash: Sha256Digest
    held_out_scorer_bridge_hash: Sha256Digest
    final_reviewed_seal_hash: Sha256Digest
    ablation_prequery_registry_hash: Sha256Digest
    unit_id: Identifier
    context_id: Identifier
    context: QueryContext
    prequery_stage: PublicStageReference
    query_stage: PublicStageReference
    prequery_barrier_hash: Sha256Digest
    prequery_barrier_sealed_at: AwareDatetime
    query_access_event_hash: Sha256Digest
    query_accessed_at: AwareDatetime
    packet_hash: Sha256Digest
    packet_artifact: PublicEvidencePacketPointer
    c2_seed1_preparation_hash: Sha256Digest
    c2_seed1_inventory_hash: Sha256Digest
    c2_seed1_receipt_hash: Sha256Digest
    c2_seed1_preparation_artifact_hash: Sha256Digest
    c2_seed1_inventory_artifact_hash: Sha256Digest
    c2_seed1_completed_at: AwareDatetime
    ablation_prequery_lineage: tuple[
        AblationPrequeryLineage,
        AblationPrequeryLineage,
        AblationPrequeryLineage,
    ]
    primary_c2: PrimaryC2Reference

    @model_validator(mode="after")
    def exact_base_context(self) -> Self:
        if self.context_id != self.context.context_id:
            raise ValueError("combined source context ID differs from typed context")
        if not self.query_stage.query_semantics_opened:
            raise ValueError("combined source query must already have audited semantic hashes")
        if (
            self.query_stage.query_context_hash != self.context.content_hash
            or self.query_stage.snapshot_hash != self.prequery_stage.snapshot_hash
            or self.query_stage.evidence_artifact_hash != self.prequery_stage.evidence_artifact_hash
            or self.query_stage.horizon_hash != self.context.spoiler_horizon.content_hash
            or self.query_stage.budget_hash != self.context.budgets.content_hash
            or self.packet_hash != self.packet_artifact.logical_content_hash
        ):
            raise ValueError("combined source changed context/evidence/horizon/budget lineage")
        if not (
            self.c2_seed1_completed_at < self.prequery_barrier_sealed_at < self.query_accessed_at
        ):
            raise ValueError("combined source C2/barrier/query chronology is not strict")
        if {item.condition for item in self.ablation_prequery_lineage} != set(
            ACTIVE_ABLATION_CONDITIONS
        ):
            raise ValueError("combined source requires all three ablation preparations")
        if any(
            item.completed_at >= self.prequery_barrier_sealed_at
            for item in self.ablation_prequery_lineage
        ):
            raise ValueError("ablation preparation did not precede the held-out barrier")
        return self

    def preparation_for(self, condition: ConditionName) -> AblationPrequeryLineage:
        for lineage in self.ablation_prequery_lineage:
            if lineage.condition is condition:
                return lineage
        raise CombinedBlockError(f"missing query-blind preparation for {condition.value}")


class ConfigurationDelta(ImmutableRecord):
    switch_name: Literal[
        "context_payload_mode",
        "temporal_epistemic_fields",
        "rare_guard",
    ]
    baseline_value: Literal["structured", "enabled"]
    ablated_value: Literal["generic", "disabled"]
    changed_field_count: Literal[1] = 1

    @model_validator(mode="after")
    def exact_switch_values(self) -> Self:
        expected = {
            "context_payload_mode": ("structured", "generic"),
            "temporal_epistemic_fields": ("enabled", "disabled"),
            "rare_guard": ("enabled", "disabled"),
        }[self.switch_name]
        if (self.baseline_value, self.ablated_value) != expected:
            raise ValueError("ablation delta changed more than its registered switch")
        return self


class OneSwitchFingerprint(ImmutableRecord):
    """Resolved comparison fields used by blocking one-switch audits."""

    model_manifest_hash: Sha256Digest
    model_revision: str
    tokenizer_hash: Sha256Digest
    packet_hash: Sha256Digest
    ordered_evidence_hash: Sha256Digest
    horizon_hash: Sha256Digest
    upper_ontology_hash: Sha256Digest
    budgets_hash: Sha256Digest
    seed_manifest_hash: Sha256Digest
    seed_block: Literal[1]
    vllm_seed: Annotated[int, Field(ge=0, le=2**31 - 1)]
    decoding_family_hash: Sha256Digest
    maximum_input_tokens: Literal[10240]
    maximum_output_tokens: Literal[2048]
    repair_attempt_budget: Literal[1]
    repair_policy_hash: Sha256Digest
    validator_hash: Sha256Digest
    scored_schema_hash: Literal[SCORED_PROJECTION_SCHEMA_HASH] = SCORED_PROJECTION_SCHEMA_HASH
    context_payload_mode: Literal["structured", "generic"] = "structured"
    temporal_epistemic_fields: Literal["enabled", "disabled"] = "enabled"
    rare_guard: Literal["enabled", "disabled"] = "enabled"


def assert_one_switch_only(
    baseline: OneSwitchFingerprint,
    ablated: OneSwitchFingerprint,
    delta: ConfigurationDelta,
) -> None:
    left = baseline.model_dump(mode="json", exclude={"content_hash"})
    right = ablated.model_dump(mode="json", exclude={"content_hash"})
    differences = {key for key in left if left[key] != right[key]}
    if differences != {delta.switch_name}:
        raise CombinedBlockError(
            f"one-switch audit expected {delta.switch_name}, observed {sorted(differences)}"
        )
    if (
        left[delta.switch_name] != delta.baseline_value
        or right[delta.switch_name] != delta.ablated_value
    ):
        raise CombinedBlockError("one-switch audit values differ from the frozen delta")


def _paraphrase_invariant(context: QueryContext) -> Sha256Digest:
    return canonical_sha256(
        {
            "lens": context.lens,
            "target": context.target,
            "story_scope": context.story_scope,
            "spoiler_horizon": context.spoiler_horizon,
            "viewpoint": context.viewpoint,
            "abstraction": context.abstraction,
            "budgets": context.budgets,
        }
    )


class CombinedCallSpec(ImmutableRecord):
    ordinal: Annotated[int, Field(ge=1, le=COMBINED_BASE_CALL_COUNT)]
    call_id: Identifier
    call_class: CombinedCallClass
    condition: ConditionName
    context_id: Identifier
    source: CombinedQuerySource
    selection_manifest_hash: Sha256Digest
    phase5_input_manifest_hash: Sha256Digest | None = None
    phase5_episode_id: Identifier | None = None
    phase5_episode_kind: FeedbackEpisodeKind | None = None
    paraphrase_context: QueryContext | None = None
    paraphrase_semantic_invariant_hash: Sha256Digest | None = None
    configuration_delta: ConfigurationDelta | None = None
    frozen_seed: Literal[3864250958737859446]
    vllm_seed: Literal[1988649846]
    seed_block: Literal[1]
    p95_seconds: Literal[95]
    watchdog_seconds: Literal[150]
    repair_reserve_class: Literal["reserve_short"]
    repair_watchdog_seconds: Literal[90]
    maximum_repair_attempts: Literal[1]
    construction_operations_permitted: Literal[True] = True

    @model_validator(mode="after")
    def exact_call_shape(self) -> Self:
        expected_condition = {
            CombinedCallClass.PARAPHRASE_C2: ConditionName.C2_LLM_QUERY,
            CombinedCallClass.SCRIPTED_FEEDBACK_C2: ConditionName.C2_LLM_QUERY,
            CombinedCallClass.RESEARCHER_TRACE_C2: ConditionName.C2_LLM_QUERY,
            CombinedCallClass.ABLATION_NO_CONTEXT: ConditionName.A_NO_CONTEXT,
            CombinedCallClass.ABLATION_NO_TEMPORAL_EPISTEMIC: (
                ConditionName.A_NO_TEMPORAL_EPISTEMIC
            ),
            CombinedCallClass.ABLATION_NO_RARE_GUARD: ConditionName.A_NO_RARE_GUARD,
        }[self.call_class]
        if self.condition is not expected_condition:
            raise ValueError("combined call condition differs from its registered class")
        if self.context_id != self.source.context_id:
            raise ValueError("combined call source belongs to another context")
        is_feedback = self.call_class in {
            CombinedCallClass.SCRIPTED_FEEDBACK_C2,
            CombinedCallClass.RESEARCHER_TRACE_C2,
        }
        feedback_values = (
            self.phase5_input_manifest_hash,
            self.phase5_episode_id,
            self.phase5_episode_kind,
        )
        if is_feedback != all(item is not None for item in feedback_values):
            raise ValueError("exactly Phase 5 calls require their input/episode binding")
        if not is_feedback and any(item is not None for item in feedback_values):
            raise ValueError("nonfeedback call cannot carry Phase 5 lineage")
        is_paraphrase = self.call_class is CombinedCallClass.PARAPHRASE_C2
        if is_paraphrase != (
            self.paraphrase_context is not None
            and self.paraphrase_semantic_invariant_hash is not None
        ):
            raise ValueError("exactly paraphrase calls require their replacement context")
        if is_paraphrase:
            assert self.paraphrase_context is not None
            if (
                self.paraphrase_context.context_id == self.source.context.context_id
                or self.paraphrase_context.wording == self.source.context.wording
                or _paraphrase_invariant(self.paraphrase_context)
                != _paraphrase_invariant(self.source.context)
                or self.paraphrase_semantic_invariant_hash
                != _paraphrase_invariant(self.source.context)
            ):
                raise ValueError("paraphrase changed a structured field or failed to reword")
        elif (
            self.paraphrase_context is not None
            or self.paraphrase_semantic_invariant_hash is not None
        ):
            raise ValueError("nonparaphrase call cannot carry paraphrase semantics")
        expected_switch = {
            CombinedCallClass.ABLATION_NO_CONTEXT: "context_payload_mode",
            CombinedCallClass.ABLATION_NO_TEMPORAL_EPISTEMIC: ("temporal_epistemic_fields"),
            CombinedCallClass.ABLATION_NO_RARE_GUARD: "rare_guard",
        }.get(self.call_class)
        if (self.configuration_delta is None) != (expected_switch is None):
            raise ValueError("ablation call requires exactly one declared delta")
        if (
            self.configuration_delta is not None
            and self.configuration_delta.switch_name != expected_switch
        ):
            raise ValueError("ablation call names the wrong single switch")
        if self.call_class is CombinedCallClass.RESEARCHER_TRACE_C2 and (
            self.phase5_episode_kind is not FeedbackEpisodeKind.RESEARCHER_TRACE
        ):
            raise ValueError("researcher trace call has the wrong episode kind")
        if self.call_class is CombinedCallClass.SCRIPTED_FEEDBACK_C2 and (
            self.phase5_episode_kind is not FeedbackEpisodeKind.SCRIPTED_KNOWN_ANSWER
        ):
            raise ValueError("scripted feedback call has the wrong episode kind")
        return self

    def model_visible_context(
        self,
    ) -> ModelVisibleQueryContext | ModelVisibleGenericConstructionContext:
        if self.call_class in {
            CombinedCallClass.SCRIPTED_FEEDBACK_C2,
            CombinedCallClass.RESEARCHER_TRACE_C2,
        }:
            raise CombinedBlockError("Phase 5 supplies its own revised model-visible context")
        context = self.paraphrase_context or self.source.context
        visible = to_model_visible_context_for_condition(context, self.condition)
        scan_model_payload(visible.model_dump(mode="json"))
        return visible


class CombinedRuntimeBinding(ImmutableRecord):
    """Immutable source/model/decoder identity shared by all 49 calls."""

    source_tree_association_hash: Sha256Digest
    selected_model_freeze_hash: Sha256Digest
    model_manifest_hash: Sha256Digest
    model_repository: Literal["Qwen/Qwen3-8B-AWQ"]
    model_revision: Literal["4da05a8edb55c6046cce958586c33b61da07bb79"]
    served_model_name: Literal["qwen3-8b-awq-fallback"]
    tokenizer_manifest_hash: Sha256Digest
    launcher_configuration_hash: Sha256Digest
    runtime_version: Literal["vllm-0.10.2"]
    decoding_configuration_file_sha256: Sha256Digest
    seed_manifest_hash: Sha256Digest
    validator_hash: Sha256Digest
    upper_ontology_hash: Sha256Digest
    repair_policy_hash: Sha256Digest
    prompt_hashes: Mapping[ConditionName, Sha256Digest]
    output_schema_hashes: Mapping[ConditionName, Sha256Digest]
    decoding_manifest_hashes: Mapping[ConditionName, Sha256Digest]
    decoding_family_hash: Sha256Digest
    capability_manifest_hashes: Mapping[ConditionName, Sha256Digest]
    maximum_input_tokens: Literal[10240]
    maximum_output_tokens: Literal[2048]
    repair_maximum_input_tokens: Literal[10752]
    repair_maximum_output_tokens: Literal[1536]
    structured_decoder: str = Field(min_length=1)

    @model_validator(mode="after")
    def all_active_conditions_are_bound(self) -> Self:
        required = {
            ConditionName.C2_LLM_QUERY,
            *ACTIVE_ABLATION_CONDITIONS,
        }
        for mapping in (
            self.prompt_hashes,
            self.output_schema_hashes,
            self.decoding_manifest_hashes,
            self.capability_manifest_hashes,
        ):
            if set(mapping) != required:
                raise ValueError("combined runtime mapping must bind four active conditions")
        if (
            self.output_schema_hashes[ConditionName.A_NO_TEMPORAL_EPISTEMIC]
            == (self.output_schema_hashes[ConditionName.C2_LLM_QUERY])
        ):
            raise ValueError("NoTemporalEpistemic requires its distinct output grammar")
        if (
            self.decoding_manifest_hashes[ConditionName.A_NO_TEMPORAL_EPISTEMIC]
            == (self.decoding_manifest_hashes[ConditionName.C2_LLM_QUERY])
        ):
            raise ValueError("NoTemporalEpistemic decoder must bind its distinct grammar")
        if (
            self.prompt_hashes[ConditionName.A_NO_CONTEXT]
            != self.prompt_hashes[ConditionName.C2_LLM_QUERY]
        ):
            raise ValueError("NoContext may change only the model-visible context")
        if (
            self.output_schema_hashes[ConditionName.A_NO_CONTEXT]
            != (self.output_schema_hashes[ConditionName.C2_LLM_QUERY])
        ):
            raise ValueError("NoContext output grammar must remain the C2 grammar")
        if (
            self.output_schema_hashes[ConditionName.A_NO_RARE_GUARD]
            != (self.output_schema_hashes[ConditionName.C2_LLM_QUERY])
        ):
            raise ValueError("NoRareGuard output grammar must remain the C2 grammar")
        if (
            self.prompt_hashes[ConditionName.A_NO_RARE_GUARD]
            == self.prompt_hashes[ConditionName.C2_LLM_QUERY]
        ):
            raise ValueError("NoRareGuard must remove its sole prompt guard")
        return self


class CombinedUpstreamGate(ImmutableRecord):
    """Hash-only authorization after accepted development and complete primary ITT."""

    development_execution_result_hash: Sha256Digest
    development_gate_passed: Literal[True]
    held_out_call_manifest_hash: Sha256Digest
    held_out_execution_manifest_hash: Sha256Digest
    held_out_scorer_bridge_hash: Sha256Digest
    held_out_itt_record_hashes: tuple[Sha256Digest, ...]
    held_out_final_schedule_snapshot_hash: Sha256Digest
    held_out_prequery_barrier_hash: Sha256Digest
    ablation_prequery_registry_hash: Sha256Digest
    phase5_input_manifest_hash: Sha256Digest
    final_reviewed_seal_hash: Sha256Digest
    global_accounting_id: Identifier
    actual_allocated_gpu_seconds_before_block: float = Field(ge=0.0)
    remaining_registered_p95_seconds_before_block: float = Field(ge=0.0)
    consumed_short_reserve_slots_before_block: Annotated[int, Field(ge=0, le=4)]
    verified_at: AwareDatetime
    held_out_runtime_closed: Literal[True] = True
    all_held_out_failures_included_in_itt: Literal[True] = True
    scorer_payload_visible_to_model: Literal[False] = False

    @model_validator(mode="after")
    def exact_primary_inventory(self) -> Self:
        if (
            len(self.held_out_itt_record_hashes) != 168
            or len(set(self.held_out_itt_record_hashes)) != 168
        ):
            raise ValueError("combined gate requires all 168 primary ITT records")
        if not math.isfinite(self.actual_allocated_gpu_seconds_before_block):
            raise ValueError("combined predecessor GPU counter is not finite")
        return self


class CombinedCallManifest(ImmutableRecord):
    manifest_id: Identifier
    configuration_hash: Sha256Digest
    registered_selections_hash: Sha256Digest
    runtime_binding_hash: Sha256Digest
    upstream_gate_hash: Sha256Digest
    phase5_input_manifest_hash: Sha256Digest
    calls: tuple[CombinedCallSpec, ...]
    base_call_forecast_seconds: Literal[4655]
    model_load_forecast_seconds: Literal[180]
    load_inclusive_forecast_seconds: Literal[4835]
    activity_ceiling_seconds: Literal[4860]
    model_load_count: Literal[1]
    maximum_concurrency: Literal[1]
    created_at: AwareDatetime
    runtime_namespace: Literal["gold_free"] = "gold_free"
    scorer_gold_in_model_payload: Literal[False] = False

    @model_validator(mode="after")
    def exact_49_call_schedule(self) -> Self:
        if len(self.calls) != COMBINED_BASE_CALL_COUNT:
            raise ValueError("combined manifest requires exactly 49 base calls")
        if tuple(item.ordinal for item in self.calls) != tuple(
            range(1, COMBINED_BASE_CALL_COUNT + 1)
        ):
            raise ValueError("combined calls must be a contiguous ordered schedule")
        if len({item.call_id for item in self.calls}) != COMBINED_BASE_CALL_COUNT:
            raise ValueError("combined call IDs must be unique")
        if Counter(item.call_class.value for item in self.calls) != Counter(COMBINED_CALL_COUNTS):
            raise ValueError("combined call-class counts changed")
        expected_order = (
            *([CombinedCallClass.PARAPHRASE_C2] * 12),
            *([CombinedCallClass.SCRIPTED_FEEDBACK_C2] * 6),
            *([CombinedCallClass.RESEARCHER_TRACE_C2] * 3),
            *([CombinedCallClass.ABLATION_NO_CONTEXT] * 12),
            *([CombinedCallClass.ABLATION_NO_TEMPORAL_EPISTEMIC] * 8),
            *([CombinedCallClass.ABLATION_NO_RARE_GUARD] * 8),
        )
        if tuple(item.call_class for item in self.calls) != expected_order:
            raise ValueError("combined call blocks are out of frozen order")
        if any(
            item.phase5_input_manifest_hash not in {None, self.phase5_input_manifest_hash}
            for item in self.calls
        ):
            raise ValueError("Phase 5 call binds another materialized input manifest")
        if sum(item.p95_seconds for item in self.calls) != self.base_call_forecast_seconds:
            raise ValueError("combined base-call p95 sum changed")
        if (
            self.base_call_forecast_seconds + self.model_load_forecast_seconds
            != self.load_inclusive_forecast_seconds
            or self.load_inclusive_forecast_seconds > self.activity_ceiling_seconds
        ):
            raise ValueError("combined block forecast exceeds the registered activity cap")
        return self


def _call(
    *,
    ordinal: int,
    call_class: CombinedCallClass,
    condition: ConditionName,
    source: CombinedQuerySource,
    selection_hash: str,
    configuration: CombinedBlockConfiguration,
    phase5_input_manifest_hash: str | None = None,
    episode_id: str | None = None,
    episode_kind: FeedbackEpisodeKind | None = None,
    paraphrase_context: QueryContext | None = None,
    delta: ConfigurationDelta | None = None,
) -> CombinedCallSpec:
    return CombinedCallSpec(
        ordinal=ordinal,
        call_id=f"combined-{ordinal:03d}-{call_class.value}",
        call_class=call_class,
        condition=condition,
        context_id=source.context_id,
        source=source,
        selection_manifest_hash=selection_hash,
        phase5_input_manifest_hash=phase5_input_manifest_hash,
        phase5_episode_id=episode_id,
        phase5_episode_kind=episode_kind,
        paraphrase_context=paraphrase_context,
        paraphrase_semantic_invariant_hash=(
            None if paraphrase_context is None else _paraphrase_invariant(source.context)
        ),
        configuration_delta=delta,
        frozen_seed=configuration.frozen_llm_seed,
        vllm_seed=configuration.vllm_seed,
        seed_block=configuration.llm_seed_block,
        p95_seconds=configuration.base_call_p95_seconds,
        watchdog_seconds=configuration.base_call_watchdog_seconds,
        repair_reserve_class=configuration.repair_reserve_class,
        repair_watchdog_seconds=configuration.repair_watchdog_seconds,
        maximum_repair_attempts=configuration.maximum_repairs_per_call,
    )


def compile_combined_call_manifest(
    *,
    configuration: CombinedBlockConfiguration,
    selections: RegisteredCombinedSelections,
    sources_by_context_id: Mapping[str, CombinedQuerySource],
    runtime_binding: CombinedRuntimeBinding,
    upstream_gate: CombinedUpstreamGate,
    created_at: datetime,
) -> CombinedCallManifest:
    """Compile the exact 49 calls, then discard scorer-only selection objects."""

    if created_at.tzinfo is None or created_at.utcoffset() is None:
        raise CombinedBlockError("combined manifest timestamp must be timezone-aware")
    if (
        configuration.content_hash == ""
        or runtime_binding.seed_manifest_hash != configuration.seed_manifest_hash
        or runtime_binding.model_repository != configuration.selected_model_repository
        or runtime_binding.model_revision != configuration.selected_model_revision
        or runtime_binding.served_model_name != configuration.served_model_name
        or runtime_binding.decoding_configuration_file_sha256
        != configuration.decoding_configuration_file_sha256
        or upstream_gate.phase5_input_manifest_hash == ""
    ):
        raise CombinedBlockError("combined configuration/runtime/upstream bindings differ")
    required_ids = {
        *(item.base_context_id for item in selections.paraphrases.pairs),
        *(item.query_id for item in selections.no_context.entries),
        *selections.eligibility.temporal_epistemic_context_ids,
        *selections.eligibility.rare_guard_context_ids,
        *(item.context_id for item in selections.feedback.scripted_episodes),
        *(item.context_id for item in selections.feedback.researcher_trace_slots),
    }
    missing = required_ids - set(sources_by_context_id)
    if missing:
        raise CombinedBlockError(f"combined selected contexts lack sources: {sorted(missing)}")
    selected_sources = tuple(sources_by_context_id[item] for item in sorted(required_ids))
    if created_at < upstream_gate.verified_at or any(
        created_at < source.query_accessed_at for source in selected_sources
    ):
        raise CombinedBlockError("combined manifest predates its frozen predecessor artifacts")
    if any(
        source.held_out_call_manifest_hash != upstream_gate.held_out_call_manifest_hash
        or source.held_out_execution_manifest_hash != upstream_gate.held_out_execution_manifest_hash
        or source.held_out_scorer_bridge_hash != upstream_gate.held_out_scorer_bridge_hash
        or source.final_reviewed_seal_hash != upstream_gate.final_reviewed_seal_hash
        or source.prequery_barrier_hash != upstream_gate.held_out_prequery_barrier_hash
        or source.ablation_prequery_registry_hash != upstream_gate.ablation_prequery_registry_hash
        for source in selected_sources
    ):
        raise CombinedBlockError("combined source lineage differs from the admitted held-out run")

    calls: list[CombinedCallSpec] = []
    ordinal = 1
    for pair in selections.paraphrases.pairs:
        source = sources_by_context_id[pair.base_context_id]
        if pair.base_context_hash != source.context.content_hash:
            raise CombinedBlockError("paraphrase base context hash changed")
        calls.append(
            _call(
                ordinal=ordinal,
                call_class=CombinedCallClass.PARAPHRASE_C2,
                condition=ConditionName.C2_LLM_QUERY,
                source=source,
                selection_hash=selections.paraphrases.content_hash,
                configuration=configuration,
                paraphrase_context=pair.paraphrase_context,
            )
        )
        ordinal += 1
    for episode in selections.feedback.scripted_episodes:
        calls.append(
            _call(
                ordinal=ordinal,
                call_class=CombinedCallClass.SCRIPTED_FEEDBACK_C2,
                condition=ConditionName.C2_LLM_QUERY,
                source=sources_by_context_id[episode.context_id],
                selection_hash=selections.feedback.content_hash,
                configuration=configuration,
                phase5_input_manifest_hash=upstream_gate.phase5_input_manifest_hash,
                episode_id=episode.episode_id,
                episode_kind=FeedbackEpisodeKind.SCRIPTED_KNOWN_ANSWER,
            )
        )
        ordinal += 1
    for episode in selections.feedback.researcher_trace_slots:
        calls.append(
            _call(
                ordinal=ordinal,
                call_class=CombinedCallClass.RESEARCHER_TRACE_C2,
                condition=ConditionName.C2_LLM_QUERY,
                source=sources_by_context_id[episode.context_id],
                selection_hash=selections.feedback.content_hash,
                configuration=configuration,
                phase5_input_manifest_hash=upstream_gate.phase5_input_manifest_hash,
                episode_id=episode.episode_id,
                episode_kind=FeedbackEpisodeKind.RESEARCHER_TRACE,
            )
        )
        ordinal += 1
    no_context_delta = ConfigurationDelta(
        switch_name="context_payload_mode",
        baseline_value="structured",
        ablated_value="generic",
    )
    for entry in selections.no_context.entries:
        calls.append(
            _call(
                ordinal=ordinal,
                call_class=CombinedCallClass.ABLATION_NO_CONTEXT,
                condition=ConditionName.A_NO_CONTEXT,
                source=sources_by_context_id[entry.query_id],
                selection_hash=selections.no_context.content_hash,
                configuration=configuration,
                delta=no_context_delta,
            )
        )
        ordinal += 1
    no_temporal_delta = ConfigurationDelta(
        switch_name="temporal_epistemic_fields",
        baseline_value="enabled",
        ablated_value="disabled",
    )
    for context_id in selections.eligibility.temporal_epistemic_context_ids:
        calls.append(
            _call(
                ordinal=ordinal,
                call_class=CombinedCallClass.ABLATION_NO_TEMPORAL_EPISTEMIC,
                condition=ConditionName.A_NO_TEMPORAL_EPISTEMIC,
                source=sources_by_context_id[context_id],
                selection_hash=selections.eligibility.content_hash,
                configuration=configuration,
                delta=no_temporal_delta,
            )
        )
        ordinal += 1
    no_rare_delta = ConfigurationDelta(
        switch_name="rare_guard",
        baseline_value="enabled",
        ablated_value="disabled",
    )
    rare_sources: list[CombinedQuerySource] = []
    for context_id in selections.eligibility.rare_guard_context_ids:
        source = sources_by_context_id[context_id]
        rare_sources.append(source)
        calls.append(
            _call(
                ordinal=ordinal,
                call_class=CombinedCallClass.ABLATION_NO_RARE_GUARD,
                condition=ConditionName.A_NO_RARE_GUARD,
                source=source,
                selection_hash=selections.eligibility.content_hash,
                configuration=configuration,
                delta=no_rare_delta,
            )
        )
        ordinal += 1
    if len({item.prequery_stage.snapshot_hash for item in rare_sources}) != 8:
        raise CombinedBlockError("NoRareGuard must span eight distinct worlds/snapshots")
    selection_hash = combined_selection_registry_hash(selections)
    return CombinedCallManifest(
        manifest_id=f"combined-manifest-{selection_hash[:20]}",
        configuration_hash=configuration.content_hash,
        registered_selections_hash=selection_hash,
        runtime_binding_hash=runtime_binding.content_hash,
        upstream_gate_hash=upstream_gate.content_hash,
        phase5_input_manifest_hash=upstream_gate.phase5_input_manifest_hash,
        calls=tuple(calls),
        base_call_forecast_seconds=configuration.combined_base_forecast_seconds,
        model_load_forecast_seconds=configuration.model_load_p95_seconds,
        load_inclusive_forecast_seconds=configuration.combined_with_load_forecast_seconds,
        activity_ceiling_seconds=configuration.activity_ceiling_seconds,
        model_load_count=configuration.model_load_count,
        maximum_concurrency=configuration.maximum_concurrency,
        created_at=created_at,
    )


def combined_context_policy_hash(
    context: QueryContext,
    condition: ConditionName,
) -> Sha256Digest:
    """Hash the condition-specific model context without exposing it to scorer code."""

    visible = to_model_visible_context_for_condition(context, condition)
    payload = visible.model_dump(mode="json")
    scan_model_payload(payload)
    return canonical_sha256(payload)


def paraphrase_base_and_variant_are_equivalent(
    base: QueryContext,
    variant: QueryContext,
) -> bool:
    return (
        base.content_hash != variant.content_hash
        and base.wording != variant.wording
        and _paraphrase_invariant(base) == _paraphrase_invariant(variant)
    )


__all__ = [
    "ACTIVE_ABLATION_CONDITIONS",
    "COMBINED_BASE_CALL_COUNT",
    "COMBINED_CALL_COUNTS",
    "AblationPrequeryLineage",
    "CombinedBlockConfiguration",
    "CombinedBlockError",
    "CombinedCallClass",
    "CombinedCallManifest",
    "CombinedCallSpec",
    "CombinedFileBinding",
    "CombinedQuerySource",
    "CombinedRuntimeBinding",
    "CombinedUpstreamGate",
    "ConfigurationDelta",
    "OneSwitchFingerprint",
    "PrimaryC2Reference",
    "PublicEvidencePacketPointer",
    "RegisteredCombinedSelections",
    "RestrictedArtifactPointer",
    "assert_one_switch_only",
    "combined_context_policy_hash",
    "combined_selection_registry_hash",
    "compile_combined_call_manifest",
    "load_combined_configuration",
    "load_registered_combined_selections",
    "paraphrase_base_and_variant_are_equivalent",
]
