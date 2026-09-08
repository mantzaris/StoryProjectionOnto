"""Typed, side-effect-free contracts for the controlled GPU LLM runtime.

This module intentionally contains no HTTP client, vLLM process management, or model
invocation.  Phase 1 first freezes the request boundary: decoding, seeds, packing,
capabilities, repair lineage, and accounting metadata must all be complete before a
request can be admitted to the GPU runner.
"""

from __future__ import annotations

import copy
import hashlib
from collections.abc import Iterable, Mapping, Sequence
from datetime import datetime
from enum import StrEnum
from pathlib import Path
from typing import Annotated, Literal, Self

from pydantic import (
    AwareDatetime,
    ConfigDict,
    Field,
    field_validator,
    model_validator,
)

from .contracts import (
    CONSTRUCTIVE_OPERATORS,
    FIXED_SELECT_ALLOWED_OPERATORS,
    BudgetAccounting,
    ConditionName,
    ConstructionOperator,
    DiscoursePosition,
    Entity,
    Event,
    FixedOntologyInput,
    ImmutableRecord,
    InstanceGraph,
    NarrativeCommitment,
    NoTemporalEpistemicOntologyDraft,
    OntologyDraft,
    QualifiedAssertion,
    RevelationPosition,
    StoryTime,
    TemporalKind,
    TemporalScope,
    ValidityTime,
    canonical_json,
    canonical_json_schema,
    canonical_sha256,
)

SCHEMA_VERSION = "1.0.0"
SHA256_PATTERN = r"^[0-9a-f]{64}$"
GIT_REVISION_PATTERN = r"^[0-9a-f]{40}$"

Sha256 = Annotated[str, Field(pattern=SHA256_PATTERN)]
GitRevision = Annotated[str, Field(pattern=GIT_REVISION_PATTERN)]
NonNegativeInt = Annotated[int, Field(ge=0)]
PositiveInt = Annotated[int, Field(gt=0)]


def canonical_json_bytes(value: object) -> bytes:
    """Serialize a JSON-compatible value without platform- or insertion-order drift."""

    return canonical_json(value, include_content_hash=False).encode("utf-8")


class RuntimeManifest(ImmutableRecord):
    """Immutable base for small content-addressable runtime records."""


# Runtime terminology aliases the canonical contract enums instead of creating a
# second, subtly different vocabulary.
LLMCondition = ConditionName
ConstructionCapability = ConstructionOperator
CONSTRUCTIVE_CAPABILITIES = CONSTRUCTIVE_OPERATORS
FIXED_SELECT_CAPABILITIES = FIXED_SELECT_ALLOWED_OPERATORS
_ALL_CAPABILITIES = frozenset(ConstructionOperator)

# vLLM 0.10.2's pinned XGrammar backend warns that these JSON Schema string
# validation keywords are unsupported and ignores them while compiling a guided
# grammar.  In xgrammar 0.1.23, sufficiently large schemas containing those
# keywords can instead fail during the intermediate EBNF conversion.  They are
# therefore removed only from the decoder-facing copy.  The canonical Pydantic
# validation schema and post-generation validation remain unchanged.
VLLM_XGRAMMAR_IGNORED_STRING_KEYWORDS = frozenset(
    {"format", "maxLength", "minLength", "pattern"}
)
_JSON_SCHEMA_MAP_OF_SCHEMAS = frozenset(
    {"$defs", "definitions", "dependentSchemas", "patternProperties", "properties"}
)
_JSON_SCHEMA_ARRAY_OF_SCHEMAS = frozenset(
    {"allOf", "anyOf", "oneOf", "prefixItems"}
)
_JSON_SCHEMA_SINGLE_SCHEMA = frozenset(
    {
        "additionalProperties",
        "contains",
        "contentSchema",
        "else",
        "if",
        "items",
        "not",
        "propertyNames",
        "then",
        "unevaluatedItems",
        "unevaluatedProperties",
    }
)

ABLATION_QUALIFICATION_REASON = (
    "qualification deliberately absent under A-NoTemporalEpistemic"
)

_NO_RARE_GUARD_PARAGRAPH = (
    "8. inspect every low-frequency item in the packet for answer necessity, state change,\n"
    "   identity consequences, temporal consequences, or causal reach. Preserve a one-off\n"
    "   fact when it is pivotal; never use frequency alone to prune it;\n"
)
_NO_RARE_GUARD_OPERATOR_PHRASE = "rare-evidence\n    preservation, "


def condition_output_model(
    condition: ConditionName,
) -> type[OntologyDraft] | type[NoTemporalEpistemicOntologyDraft]:
    """Return the model-authored JSON surface registered for a condition.

    Selection-only ID restrictions are layered onto the ordinary draft schema by the
    request packer.  The qualification ablation is different: its fields must be
    absent from the grammar itself, so it has a distinct immutable record type.
    """

    if condition is ConditionName.A_NO_TEMPORAL_EPISTEMIC:
        return NoTemporalEpistemicOntologyDraft
    return OntologyDraft


def vllm_xgrammar_decoder_schema(
    schema: Mapping[str, object],
) -> dict[str, object]:
    """Return a vLLM-0.10.2/XGrammar-compatible decoder-only schema copy.

    The transform is deliberately narrow and recursive.  It removes only the
    four string-validation keywords that the pinned backend documents at run
    time as ignored.  Structural constraints, enums, constants, array bounds,
    numeric bounds, and required/additional-property rules are preserved.  A
    deep copy prevents the compatibility surface from weakening the canonical
    schema used for deterministic validation after generation.
    """

    compatible = copy.deepcopy(dict(schema))

    def strip_ignored_keywords(value: object) -> None:
        if not isinstance(value, dict):
            return
        for keyword in VLLM_XGRAMMAR_IGNORED_STRING_KEYWORDS:
            value.pop(keyword, None)
        for keyword, child in value.items():
            if keyword in _JSON_SCHEMA_MAP_OF_SCHEMAS and isinstance(child, dict):
                for nested_schema in child.values():
                    strip_ignored_keywords(nested_schema)
            elif keyword in _JSON_SCHEMA_ARRAY_OF_SCHEMAS and isinstance(child, list):
                for nested_schema in child:
                    strip_ignored_keywords(nested_schema)
            elif keyword in _JSON_SCHEMA_SINGLE_SCHEMA:
                if isinstance(child, list):
                    for nested_schema in child:
                        strip_ignored_keywords(nested_schema)
                else:
                    strip_ignored_keywords(child)

    strip_ignored_keywords(compatible)
    return compatible


def base_condition_output_schema(condition: ConditionName) -> dict[str, object]:
    """Generate the condition's base constrained-decoding schema.

    Administrative token counts remain zero sentinels in model output and are
    replaced only from vLLM usage metadata after generation.
    """

    schema = canonical_json_schema(condition_output_model(condition))
    definitions = schema.get("$defs")
    if not isinstance(definitions, dict):
        raise ValueError("condition output schema lacks definitions")
    accounting = definitions.get("BudgetAccounting")
    if not isinstance(accounting, dict):
        raise ValueError("condition output schema lacks BudgetAccounting")
    properties = accounting.get("properties")
    if not isinstance(properties, dict):
        raise ValueError("BudgetAccounting schema lacks properties")
    properties["input_tokens"] = {"const": 0, "type": "integer"}
    properties["output_tokens"] = {"const": 0, "type": "integer"}
    return vllm_xgrammar_decoder_schema(schema)


def render_condition_system_prompt(root: Path, condition: ConditionName) -> str:
    """Resolve the frozen first-pass prompt and its registered single-switch overlay."""

    root = Path(root).resolve(strict=True)
    if condition is ConditionName.C1_LLM_PRE:
        relative = Path("prompts/c1_pre/prompt_v1.md")
    elif condition is ConditionName.A_FIXED_SELECT:
        relative = Path("prompts/fixed_select/prompt_v1.md")
    elif condition is ConditionName.A_NO_TEMPORAL_EPISTEMIC:
        relative = Path("prompts/ablations/no_temporal_epistemic_v1.md")
    else:
        relative = Path("prompts/c2_query/prompt_v1.md")
    prompt = (root / relative).read_text(encoding="utf-8")
    if condition is not ConditionName.A_NO_RARE_GUARD:
        return prompt
    if prompt.count(_NO_RARE_GUARD_PARAGRAPH) != 1:
        raise ValueError("C2 prompt no longer has the registered rare-guard paragraph")
    if prompt.count(_NO_RARE_GUARD_OPERATOR_PHRASE) != 1:
        raise ValueError("C2 prompt no longer has the registered rare checklist operator")
    return prompt.replace(_NO_RARE_GUARD_PARAGRAPH, "").replace(
        _NO_RARE_GUARD_OPERATOR_PHRASE,
        "",
    )


def normalize_no_temporal_epistemic_draft(
    raw_draft: NoTemporalEpistemicOntologyDraft,
) -> OntologyDraft:
    """Map the ablated raw surface to the common scorer representation.

    This is a fixed, fact-free adapter.  It never derives a time, holder, attitude,
    commitment, or proposition.  Instead it writes conspicuous unknown sentinels so
    the unchanged strict qualified-assertion scorer counts required qualifications as
    misses.  The exact raw model JSON remains the authoritative generation artifact.
    """

    def ordinary_fields(value: ImmutableRecord) -> dict[str, object]:
        return value.model_dump(
            mode="python",
            exclude={"schema_version", "content_hash"},
        )

    unknown_story_time = StoryTime(
        kind=TemporalKind.UNKNOWN,
        reason=ABLATION_QUALIFICATION_REASON,
    )
    unknown_validity_time = ValidityTime(
        kind=TemporalKind.UNKNOWN,
        reason=ABLATION_QUALIFICATION_REASON,
    )
    absent_scope = TemporalScope(
        story_time=unknown_story_time,
        validity_time=unknown_validity_time,
        discourse_position=DiscoursePosition(passage_order=0),
        revelation_position=RevelationPosition(
            revelation_order=0,
            label="qualification-ablated",
        ),
    )
    entities = tuple(
        Entity(
            **ordinary_fields(entity),
            temporal_state=unknown_story_time,
        )
        for entity in raw_draft.instance_graph.entities
    )
    events = tuple(
        Event(
            **ordinary_fields(event),
            occurrence_time=unknown_story_time,
        )
        for event in raw_draft.instance_graph.events
    )
    assertions = tuple(
        QualifiedAssertion(
            **ordinary_fields(assertion),
            proposition_content_id=None,
            temporal_scope=absent_scope,
            epistemic_scope=None,
            narrative_commitment=NarrativeCommitment.UNKNOWN,
        )
        for assertion in raw_draft.instance_graph.assertions
    )
    return OntologyDraft(
        contextual_interpretation=raw_draft.contextual_interpretation,
        local_schema=raw_draft.local_schema,
        instance_graph=InstanceGraph(
            entities=entities,
            events=events,
            proposition_contents=(),
            assertions=assertions,
        ),
        decisions=raw_draft.decisions,
        omissions=raw_draft.omissions,
        uncertainty_and_abstentions=raw_draft.uncertainty_and_abstentions,
        budget_accounting=BudgetAccounting.model_validate(
            raw_draft.budget_accounting.model_dump(
                mode="python", exclude={"schema_version", "content_hash"}
            )
        ),
    )


def _ordered_capabilities(
    values: Iterable[ConstructionOperator],
) -> tuple[ConstructionOperator, ...]:
    return tuple(sorted(values, key=lambda value: value.value))


def allowed_capabilities_for(condition: ConditionName) -> frozenset[ConstructionOperator]:
    """Return the immutable, registered capability set for a GPU pathway."""

    if condition is ConditionName.C1_LLM_PRE:
        # C1 constructs comprehensively before reveal; it does not perform query-time
        # selection or ranking.
        return CONSTRUCTIVE_CAPABILITIES | {ConstructionOperator.SUPPORTED_DESCRIPTION}
    if condition in {
        ConditionName.C2_LLM_QUERY,
        ConditionName.A_NO_CONTEXT,
        ConditionName.A_NO_RARE_GUARD,
    }:
        return _ALL_CAPABILITIES
    if condition is ConditionName.A_NO_TEMPORAL_EPISTEMIC:
        return _ALL_CAPABILITIES - {
            ConstructionOperator.TEMPORAL_QUALIFICATION,
            ConstructionOperator.EPISTEMIC_QUALIFICATION,
        }
    if condition is ConditionName.A_FIXED_SELECT:
        return FIXED_SELECT_CAPABILITIES
    raise AssertionError(f"unhandled LLM condition: {condition!r}")


class CapabilityManifest(RuntimeManifest):
    """Hashed allowlist used both by the prompt builder and mechanical validator."""

    condition: ConditionName
    grammar_mode: Literal["prequery_construction", "query_construction", "selection_only"]
    allowed: tuple[ConstructionOperator, ...]
    forbidden: tuple[ConstructionOperator, ...]

    @model_validator(mode="after")
    def enforce_registered_allowlist(self) -> Self:
        expected_allowed = allowed_capabilities_for(self.condition)
        expected_forbidden = _ALL_CAPABILITIES - expected_allowed
        if self.allowed != _ordered_capabilities(expected_allowed):
            raise ValueError(f"{self.condition.value} capability allowlist differs from protocol")
        if self.forbidden != _ordered_capabilities(expected_forbidden):
            raise ValueError(f"{self.condition.value} forbidden capabilities differ from protocol")
        expected_mode = {
            ConditionName.C1_LLM_PRE: "prequery_construction",
            ConditionName.C2_LLM_QUERY: "query_construction",
            ConditionName.A_NO_CONTEXT: "query_construction",
            ConditionName.A_NO_TEMPORAL_EPISTEMIC: "query_construction",
            ConditionName.A_NO_RARE_GUARD: "query_construction",
            ConditionName.A_FIXED_SELECT: "selection_only",
        }[self.condition]
        if self.grammar_mode != expected_mode:
            raise ValueError(f"{self.condition.value} must use {expected_mode!r} grammar")
        return self

    @classmethod
    def for_condition(cls, condition: ConditionName) -> CapabilityManifest:
        allowed = allowed_capabilities_for(condition)
        return cls(
            condition=condition,
            grammar_mode={
                ConditionName.C1_LLM_PRE: "prequery_construction",
                ConditionName.C2_LLM_QUERY: "query_construction",
                ConditionName.A_NO_CONTEXT: "query_construction",
                ConditionName.A_NO_TEMPORAL_EPISTEMIC: "query_construction",
                ConditionName.A_NO_RARE_GUARD: "query_construction",
                ConditionName.A_FIXED_SELECT: "selection_only",
            }[condition],
            allowed=_ordered_capabilities(allowed),
            forbidden=_ordered_capabilities(_ALL_CAPABILITIES - allowed),
        )

    def permits(self, capability: ConstructionOperator) -> bool:
        return capability in self.allowed


class DecodingPass(StrEnum):
    FIRST_PASS = "first_pass"
    REPAIR = "repair"


class DecodingManifest(RuntimeManifest):
    """Complete registered non-thinking vLLM generation decision.

    Stop-token values and schema/template hashes are deliberately required because they
    can only be frozen after inspecting the pinned tokenizer and decoder.
    """

    decoding_pass: DecodingPass
    temperature: float = Field(default=0.7, ge=0.0)
    top_p: float = Field(default=0.8, gt=0.0, le=1.0)
    top_k: int = Field(default=20, ge=-1)
    min_p: float = Field(default=0.0, ge=0.0, le=1.0)
    presence_penalty: float = Field(default=0.0)
    frequency_penalty: float = Field(default=0.0)
    repetition_penalty: float = Field(default=1.0, gt=0.0)
    n: Literal[1] = 1
    best_of: Literal[1] = 1
    beam_search: Literal[False] = False
    thinking_mode: Literal[False] = False
    ignore_eos: Literal[False] = False
    maximum_model_tokens: PositiveInt = 12_288
    maximum_input_tokens: PositiveInt
    maximum_output_tokens: PositiveInt
    seed: int = Field(ge=0, le=2**31 - 1)
    eos_token_id: NonNegativeInt
    end_of_turn_token_ids: tuple[NonNegativeInt, ...]
    stop_token_ids: tuple[NonNegativeInt, ...]
    chat_template_hash: Sha256
    output_schema_hash: Sha256
    structured_decoder: str = Field(min_length=1)
    tokenizer_revision: str = Field(min_length=1)

    @field_validator("end_of_turn_token_ids", "stop_token_ids")
    @classmethod
    def token_ids_are_nonempty_and_unique(cls, value: tuple[int, ...]) -> tuple[int, ...]:
        if not value:
            raise ValueError("token ID tuple must be nonempty")
        if len(value) != len(set(value)):
            raise ValueError("token IDs must not contain duplicates")
        return value

    @model_validator(mode="after")
    def enforce_registered_decoding(self) -> Self:
        registered_values = {
            "temperature": 0.7,
            "top_p": 0.8,
            "top_k": 20,
            "min_p": 0.0,
            "presence_penalty": 0.0,
            "frequency_penalty": 0.0,
            "repetition_penalty": 1.0,
        }
        for field_name, expected in registered_values.items():
            if getattr(self, field_name) != expected:
                raise ValueError(f"{field_name} must remain at registered value {expected}")
        if self.eos_token_id not in self.stop_token_ids:
            raise ValueError("stop_token_ids must include the pinned tokenizer EOS ID")
        missing_turn_ids = set(self.end_of_turn_token_ids) - set(self.stop_token_ids)
        if missing_turn_ids:
            raise ValueError("stop_token_ids must include every end-of-turn ID")
        if self.maximum_input_tokens + self.maximum_output_tokens > self.maximum_model_tokens:
            raise ValueError("input and output token caps exceed maximum_model_tokens")
        input_ceiling, output_ceiling = {
            DecodingPass.FIRST_PASS: (10_240, 2_048),
            DecodingPass.REPAIR: (10_752, 1_536),
        }[self.decoding_pass]
        # Explicit pre-held-out capacity candidate; historical record shape and
        # hashes remain unchanged. Activation still requires packing/admission.
        if (self.maximum_input_tokens, self.maximum_output_tokens) == (6_144, 6_144):
            input_ceiling, output_ceiling = 6_144, 6_144
        if (self.maximum_input_tokens, self.maximum_output_tokens) == (9_216, 3_072):
            input_ceiling, output_ceiling = 9_216, 3_072
        # CPU-prepared small retry candidate only. GuidedJSONRequest restricts
        # this pair to its diagnostic request ID; ordinary calls cannot use it.
        if (self.maximum_input_tokens, self.maximum_output_tokens) == (8_704, 3_584):
            input_ceiling, output_ceiling = 8_704, 3_584
        if self.maximum_input_tokens > input_ceiling:
            raise ValueError(
                f"{self.decoding_pass.value} input cap exceeds registered {input_ceiling}"
            )
        if self.maximum_output_tokens > output_ceiling:
            raise ValueError(
                f"{self.decoding_pass.value} output cap exceeds registered {output_ceiling}"
            )
        return self

    @property
    def comparison_family_hash(self) -> str:
        """Hash the shared decoding policy without condition grammar or paired seed.

        The exact manifest hash must bind the guided JSON schema and concrete seed for
        replay.  Those values legitimately differ for ``A-FixedSelect`` (restricted
        grammar) and across paired seed blocks.  Fairness audits instead compare this
        schema-independent family hash while checking schema and seed lineage in their
        own explicit fields.
        """

        payload = self.model_dump(
            mode="json",
            exclude={"content_hash", "output_schema_hash", "seed"},
        )
        return canonical_sha256(payload)

    @classmethod
    def first_pass(
        cls,
        *,
        seed: int,
        eos_token_id: int,
        end_of_turn_token_ids: Sequence[int],
        chat_template_hash: str,
        output_schema_hash: str,
        structured_decoder: str,
        tokenizer_revision: str,
        maximum_input_tokens: int = 10_240,
        maximum_output_tokens: int = 2_048,
    ) -> DecodingManifest:
        stop_ids = tuple(dict.fromkeys((eos_token_id, *end_of_turn_token_ids)))
        return cls(
            decoding_pass=DecodingPass.FIRST_PASS,
            maximum_input_tokens=maximum_input_tokens,
            maximum_output_tokens=maximum_output_tokens,
            seed=seed,
            eos_token_id=eos_token_id,
            end_of_turn_token_ids=tuple(end_of_turn_token_ids),
            stop_token_ids=stop_ids,
            chat_template_hash=chat_template_hash,
            output_schema_hash=output_schema_hash,
            structured_decoder=structured_decoder,
            tokenizer_revision=tokenizer_revision,
        )

    @classmethod
    def repair(
        cls,
        *,
        seed: int,
        eos_token_id: int,
        end_of_turn_token_ids: Sequence[int],
        chat_template_hash: str,
        output_schema_hash: str,
        structured_decoder: str,
        tokenizer_revision: str,
        maximum_input_tokens: int = 10_752,
        maximum_output_tokens: int = 1_536,
    ) -> DecodingManifest:
        stop_ids = tuple(dict.fromkeys((eos_token_id, *end_of_turn_token_ids)))
        return cls(
            decoding_pass=DecodingPass.REPAIR,
            maximum_input_tokens=maximum_input_tokens,
            maximum_output_tokens=maximum_output_tokens,
            seed=seed,
            eos_token_id=eos_token_id,
            end_of_turn_token_ids=tuple(end_of_turn_token_ids),
            stop_token_ids=stop_ids,
            chat_template_hash=chat_template_hash,
            output_schema_hash=output_schema_hash,
            structured_decoder=structured_decoder,
            tokenizer_revision=tokenizer_revision,
        )


SEED_NAMESPACES = (
    "world",
    "narrative",
    "query",
    "paraphrase",
    "llm",
    "layout",
    "leiden",
    "bootstrap",
)


def derive_seed(root_seed: int, namespace: str, *coordinates: str) -> int:
    """Derive a deterministic 31-bit seed without Python's randomized ``hash``."""

    if root_seed < 0:
        raise ValueError("root_seed must be nonnegative")
    if namespace not in SEED_NAMESPACES:
        raise ValueError(f"unregistered seed namespace: {namespace!r}")
    payload = {
        "domain": "story-projection-onto.seed.v1",
        "root_seed": root_seed,
        "namespace": namespace,
        "coordinates": list(coordinates),
    }
    return int.from_bytes(hashlib.sha256(canonical_json_bytes(payload)).digest()[:4], "big") & (
        2**31 - 1
    )


class SeedManifest(RuntimeManifest):
    """All registered pseudorandom streams derived from one public root seed."""

    root_seed: NonNegativeInt
    seed_block: NonNegativeInt
    scope: tuple[str, ...] = ()
    world_seed: NonNegativeInt
    narrative_seed: NonNegativeInt
    query_seed: NonNegativeInt
    paraphrase_seed: NonNegativeInt
    llm_seed: NonNegativeInt
    layout_seed: NonNegativeInt
    leiden_seed: NonNegativeInt
    bootstrap_seed: NonNegativeInt

    @model_validator(mode="after")
    def seeds_match_derivation(self) -> Self:
        coordinates = (str(self.seed_block), *self.scope)
        for namespace in SEED_NAMESPACES:
            expected = derive_seed(self.root_seed, namespace, *coordinates)
            if getattr(self, f"{namespace}_seed") != expected:
                raise ValueError(f"{namespace}_seed does not match deterministic derivation")
        return self

    @classmethod
    def from_root(
        cls,
        root_seed: int,
        *,
        seed_block: int,
        scope: Sequence[str] = (),
    ) -> SeedManifest:
        coordinates = (str(seed_block), *scope)
        seeds = {
            f"{namespace}_seed": derive_seed(root_seed, namespace, *coordinates)
            for namespace in SEED_NAMESPACES
        }
        return cls(
            root_seed=root_seed,
            seed_block=seed_block,
            scope=tuple(scope),
            **seeds,
        )


class PackingSection(RuntimeManifest):
    """One completely encoded section of a model request."""

    name: str = Field(min_length=1, pattern=r"^[a-z][a-z0-9_]*$")
    content_hash_value: Sha256 = Field(alias="section_content_hash")
    token_count: NonNegativeInt
    required: bool = True
    complete: Literal[True] = True

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        str_strip_whitespace=True,
        populate_by_name=True,
    )

    @property
    def section_content_hash(self) -> str:
        return self.content_hash_value


class PackingReport(RuntimeManifest):
    """Blocking proof that a request was packed completely and without truncation."""

    condition: LLMCondition
    tokenizer_revision: str = Field(min_length=1)
    maximum_model_tokens: PositiveInt
    maximum_input_tokens: PositiveInt
    reserved_output_tokens: PositiveInt
    input_token_count: NonNegativeInt
    sections: tuple[PackingSection, ...]
    required_section_names: tuple[str, ...]
    complete_evidence_snapshot: bool | None = None
    complete_evidence_packet: bool | None = None
    complete_sealed_ontology: bool | None = None
    truncation_applied: Literal[False] = False
    omitted_section_names: tuple[str, ...] = ()

    @field_validator("required_section_names", "omitted_section_names")
    @classmethod
    def names_are_unique(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if len(value) != len(set(value)):
            raise ValueError("packing section names must be unique")
        return value

    @model_validator(mode="after")
    def reject_incomplete_or_oversize_pack(self) -> Self:
        section_names = tuple(section.name for section in self.sections)
        if len(section_names) != len(set(section_names)):
            raise ValueError("sections must have unique names")
        if self.omitted_section_names:
            raise ValueError("packing reports may not omit sections")
        missing = set(self.required_section_names) - set(section_names)
        if missing:
            raise ValueError(f"required packing sections are missing: {sorted(missing)}")
        calculated_tokens = sum(section.token_count for section in self.sections)
        if self.input_token_count != calculated_tokens:
            raise ValueError("input_token_count must equal the sum of section token counts")
        if self.input_token_count > self.maximum_input_tokens:
            raise ValueError("complete input exceeds maximum_input_tokens; truncation is forbidden")
        if self.input_token_count + self.reserved_output_tokens > self.maximum_model_tokens:
            raise ValueError("input plus reserved output exceeds maximum_model_tokens")

        required_by_condition = {
            ConditionName.C1_LLM_PRE: {
                "system_prompt",
                "output_schema",
                "upper_ontology",
                "evidence_snapshot",
            },
            ConditionName.C2_LLM_QUERY: {
                "system_prompt",
                "output_schema",
                "upper_ontology",
                "query_context",
                "evidence_packet",
            },
            ConditionName.A_NO_CONTEXT: {
                "system_prompt",
                "output_schema",
                "upper_ontology",
                "query_context",
                "evidence_packet",
            },
            ConditionName.A_NO_TEMPORAL_EPISTEMIC: {
                "system_prompt",
                "output_schema",
                "upper_ontology",
                "query_context",
                "evidence_packet",
            },
            ConditionName.A_NO_RARE_GUARD: {
                "system_prompt",
                "output_schema",
                "upper_ontology",
                "query_context",
                "evidence_packet",
            },
            ConditionName.A_FIXED_SELECT: {
                "system_prompt",
                "output_schema",
                "upper_ontology",
                "query_context",
                "evidence_packet",
                "sealed_ontology",
            },
        }[self.condition]
        undeclared = required_by_condition - set(self.required_section_names)
        if undeclared:
            raise ValueError(f"required_section_names omit protocol sections: {sorted(undeclared)}")
        if self.condition is ConditionName.C1_LLM_PRE:
            if self.complete_evidence_snapshot is not True:
                raise ValueError("C1 must pack the complete evidence snapshot")
            if "query_context" in section_names:
                raise ValueError("C1 packing must remain query-blind")
            if (
                self.complete_evidence_packet is not None
                or self.complete_sealed_ontology is not None
            ):
                raise ValueError("C1 cannot claim query packet or sealed-input ontology packing")
        else:
            if self.complete_evidence_packet is not True:
                raise ValueError(f"{self.condition.value} must pack the complete evidence packet")
            if self.complete_evidence_snapshot is not None:
                raise ValueError("query-time request must not substitute an evidence snapshot")
            if self.condition is ConditionName.A_FIXED_SELECT:
                if self.complete_sealed_ontology is not True:
                    raise ValueError("A-FixedSelect must pack the complete sealed C1 ontology")
            elif self.complete_sealed_ontology is not None:
                raise ValueError("C2 must not receive a sealed ontology")
        return self

    @classmethod
    def build(
        cls,
        *,
        condition: LLMCondition,
        tokenizer_revision: str,
        maximum_model_tokens: int,
        maximum_input_tokens: int,
        reserved_output_tokens: int,
        sections: Sequence[PackingSection],
        required_section_names: Sequence[str],
        complete_evidence_snapshot: bool | None = None,
        complete_evidence_packet: bool | None = None,
        complete_sealed_ontology: bool | None = None,
    ) -> PackingReport:
        return cls(
            condition=condition,
            tokenizer_revision=tokenizer_revision,
            maximum_model_tokens=maximum_model_tokens,
            maximum_input_tokens=maximum_input_tokens,
            reserved_output_tokens=reserved_output_tokens,
            input_token_count=sum(section.token_count for section in sections),
            sections=tuple(sections),
            required_section_names=tuple(required_section_names),
            complete_evidence_snapshot=complete_evidence_snapshot,
            complete_evidence_packet=complete_evidence_packet,
            complete_sealed_ontology=complete_sealed_ontology,
        )


class SemanticFingerprint(RuntimeManifest):
    """Hash of one sealed semantic object, independent of display/layout state."""

    semantic_id: str = Field(min_length=1)
    semantic_kind: Literal[
        "schema",
        "entity",
        "event",
        "contextual_type",
        "predicate",
        "assertion",
        "proposition",
    ]
    semantic_hash: Sha256


class SealedOntologyInventory(RuntimeManifest):
    """The complete same-seed C1 semantic inventory exposed to FixedSelect."""

    seal_hash: Sha256
    seed_block: NonNegativeInt
    objects: tuple[SemanticFingerprint, ...]

    @model_validator(mode="after")
    def semantic_ids_are_unique(self) -> Self:
        ids = tuple(item.semantic_id for item in self.objects)
        if len(ids) != len(set(ids)):
            raise ValueError("sealed semantic IDs must be globally unique")
        return self

    @property
    def by_id(self) -> Mapping[str, SemanticFingerprint]:
        return {item.semantic_id: item for item in self.objects}


class ProposedOperation(RuntimeManifest):
    """Operation emitted by a constrained output for capability auditing."""

    operation_id: str = Field(min_length=1)
    capability: ConstructionCapability
    referenced_ids: tuple[str, ...] = ()
    created_ids: tuple[str, ...] = ()
    decided_at: AwareDatetime

    @field_validator("referenced_ids", "created_ids")
    @classmethod
    def ids_are_unique(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if len(value) != len(set(value)):
            raise ValueError("operation IDs must not contain duplicates")
        return value


class FixedSelectOutputAudit(RuntimeManifest):
    """Semantic inventory and operations parsed from one FixedSelect response."""

    source_seal_hash: Sha256
    seed_block: NonNegativeInt
    selected_objects: tuple[SemanticFingerprint, ...]
    semantic_reference_ids: tuple[str, ...] = ()
    operations: tuple[ProposedOperation, ...]

    @model_validator(mode="after")
    def selected_ids_are_unique(self) -> Self:
        ids = tuple(item.semantic_id for item in self.selected_objects)
        if len(ids) != len(set(ids)):
            raise ValueError("selected semantic IDs must be unique")
        if len(self.semantic_reference_ids) != len(set(self.semantic_reference_ids)):
            raise ValueError("semantic reference IDs must be unique")
        return self


class FixedSelectCapabilityError(ValueError):
    """Raised when selection-only output attempts semantic construction or mutation."""

    def __init__(self, violations: Sequence[str]) -> None:
        self.violations = tuple(violations)
        super().__init__("; ".join(self.violations))


def enforce_fixed_select_output(
    output: FixedSelectOutputAudit,
    sealed: SealedOntologyInventory,
) -> None:
    """Mechanically reject construction, new IDs, cross-seed input, or semantic edits."""

    violations: list[str] = []
    if output.source_seal_hash != sealed.seal_hash:
        violations.append("source_seal_hash does not match the complete sealed C1 ontology")
    if output.seed_block != sealed.seed_block:
        violations.append("FixedSelect and its sealed C1 ontology use different seed blocks")

    sealed_by_id = sealed.by_id
    for selected in output.selected_objects:
        original = sealed_by_id.get(selected.semantic_id)
        if original is None:
            violations.append(f"new semantic ID is forbidden: {selected.semantic_id}")
            continue
        if selected.semantic_kind != original.semantic_kind:
            violations.append(f"semantic kind changed for sealed ID: {selected.semantic_id}")
        if selected.semantic_hash != original.semantic_hash:
            violations.append(f"semantic content changed for sealed ID: {selected.semantic_id}")

    unknown_semantic_references = set(output.semantic_reference_ids) - set(sealed_by_id)
    if unknown_semantic_references:
        violations.append(
            "A-FixedSelect output contains references to unsealed semantic IDs: "
            f"{sorted(unknown_semantic_references)}"
        )

    for operation in output.operations:
        if operation.capability not in FIXED_SELECT_CAPABILITIES:
            violations.append(
                f"constructive operation is forbidden in A-FixedSelect: "
                f"{operation.capability.value}"
            )
        if operation.created_ids:
            violations.append(
                f"A-FixedSelect operation {operation.operation_id} emitted created IDs"
            )
        unknown_references = set(operation.referenced_ids) - set(sealed_by_id)
        if unknown_references:
            violations.append(
                f"A-FixedSelect operation {operation.operation_id} references unsealed IDs: "
                f"{sorted(unknown_references)}"
            )

    if violations:
        raise FixedSelectCapabilityError(violations)


def semantic_fingerprints_from_draft(
    draft: OntologyDraft,
) -> tuple[SemanticFingerprint, ...]:
    """Extract semantic (not display/description-ranking) hashes from a draft.

    FixedSelect may compress supported labels/descriptions and assign query relevance,
    so those fields do not enter the semantic fingerprint. Identity, schema, temporal
    and epistemic qualification, confidence, and provenance do enter it.
    """

    fingerprints: list[SemanticFingerprint] = []

    def append(
        semantic_id: str,
        semantic_kind: Literal[
            "schema",
            "entity",
            "event",
            "contextual_type",
            "predicate",
            "assertion",
            "proposition",
        ],
        payload: object,
    ) -> None:
        fingerprints.append(
            SemanticFingerprint(
                semantic_id=semantic_id,
                semantic_kind=semantic_kind,
                semantic_hash=canonical_sha256(payload),
            )
        )

    schema = draft.local_schema
    append(
        schema.schema_id,
        "schema",
        {"schema_id": schema.schema_id, "abstraction": schema.abstraction},
    )
    for contextual_type in schema.contextual_types:
        append(contextual_type.type_id, "contextual_type", contextual_type)
    for predicate in schema.predicates:
        append(predicate.predicate_id, "predicate", predicate)

    for entity in draft.instance_graph.entities:
        payload = entity.model_dump(
            mode="python",
            exclude={"content_hash", "label", "description", "description_assertion_ids"},
        )
        append(entity.entity_id, "entity", payload)
    for event in draft.instance_graph.events:
        payload = event.model_dump(
            mode="python",
            exclude={"content_hash", "label", "description", "description_assertion_ids"},
        )
        append(event.event_id, "event", payload)
    for proposition in draft.instance_graph.proposition_contents:
        append(proposition.proposition_content_id, "proposition", proposition)
    for assertion in draft.instance_graph.assertions:
        payload = assertion.model_dump(
            mode="python",
            exclude={
                "content_hash",
                "contextual_relevance",
                "why_matters",
                "why_matters_evidence_ids",
            },
        )
        append(assertion.assertion_id, "assertion", payload)
    return tuple(fingerprints)


def sealed_inventory_from_fixed_ontology(
    fixed_ontology: FixedOntologyInput,
    *,
    seed_block: int,
    source_draft: OntologyDraft,
) -> SealedOntologyInventory:
    """Build a complete inventory and verify it exactly matches the C1 seal."""

    if source_draft.local_schema != fixed_ontology.local_schema:
        raise ValueError("source draft local schema differs from fixed C1 ontology")
    if source_draft.instance_graph != fixed_ontology.instance_graph:
        raise ValueError("source draft instance graph differs from fixed C1 ontology")
    objects = semantic_fingerprints_from_draft(source_draft)
    object_ids = {item.semantic_id for item in objects}
    sealed_ids = set(fixed_ontology.construction_seal.sealed_object_ids)
    if object_ids != sealed_ids:
        missing = sorted(object_ids - sealed_ids)
        extra = sorted(sealed_ids - object_ids)
        raise ValueError(
            f"construction seal and complete semantic inventory differ; "
            f"unsealed={missing}, unexplained={extra}"
        )
    return SealedOntologyInventory(
        seal_hash=fixed_ontology.construction_seal.content_hash,
        seed_block=seed_block,
        objects=objects,
    )


def fixed_select_audit_from_draft(
    draft: OntologyDraft,
    *,
    source_seal_hash: str,
    seed_block: int,
) -> FixedSelectOutputAudit:
    """Convert the parsed common draft into the mechanical selection-only audit."""

    operations = tuple(
        ProposedOperation(
            operation_id=decision.decision_id,
            capability=decision.operator,
            referenced_ids=tuple(
                dict.fromkeys((*decision.input_object_ids, *decision.removed_object_ids))
            ),
            created_ids=decision.created_object_ids,
            decided_at=decision.decided_at,
        )
        for decision in draft.decisions
    )
    semantic_references: list[str] = []
    for predicate in draft.local_schema.predicates:
        semantic_references.extend(predicate.domain_type_ids)
        semantic_references.extend(predicate.range_type_ids)
    for entity in draft.instance_graph.entities:
        semantic_references.append(entity.contextual_type_id)
        semantic_references.extend(entity.description_assertion_ids)
    for event in draft.instance_graph.events:
        semantic_references.append(event.contextual_type_id)
        semantic_references.extend(event.description_assertion_ids)
    for proposition in draft.instance_graph.proposition_contents:
        semantic_references.append(proposition.predicate_id)
        semantic_references.extend(
            item for item in (proposition.subject_id, proposition.object_id) if item is not None
        )
        semantic_references.extend(role.object_id for role in proposition.roles)
    for assertion in draft.instance_graph.assertions:
        semantic_references.append(assertion.predicate_id)
        semantic_references.extend(
            item for item in (assertion.subject_id, assertion.object_id) if item is not None
        )
        semantic_references.extend(role.object_id for role in assertion.roles)
        if assertion.proposition_content_id is not None:
            semantic_references.append(assertion.proposition_content_id)
        if assertion.epistemic_scope is not None:
            semantic_references.append(assertion.epistemic_scope.holder_id)
    return FixedSelectOutputAudit(
        source_seal_hash=source_seal_hash,
        seed_block=seed_block,
        selected_objects=semantic_fingerprints_from_draft(draft),
        semantic_reference_ids=tuple(dict.fromkeys(semantic_references)),
        operations=operations,
    )


def enforce_fixed_select_draft(
    draft: OntologyDraft,
    *,
    sealed: SealedOntologyInventory,
    seed_block: int,
) -> None:
    """Apply the ID/operator/semantic-mutation gate to a parsed common draft."""

    enforce_fixed_select_output(
        fixed_select_audit_from_draft(
            draft,
            source_seal_hash=sealed.seal_hash,
            seed_block=seed_block,
        ),
        sealed,
    )


class RepairLineageMetadata(RuntimeManifest):
    """Audit metadata for the sole bounded semantic-repair attempt."""

    root_attempt_id: str = Field(min_length=1)
    base_attempt_id: str = Field(min_length=1)
    repair_attempt_id: str = Field(min_length=1)
    repair_number: Literal[1] = 1
    semantic_request_hash: Sha256
    base_output_hash: Sha256
    validation_record_hash: Sha256
    diagnostic_codes: tuple[str, ...]
    allowed_changes: tuple[Literal["validation_diagnostics", "maximum_output_tokens"], ...] = (
        "validation_diagnostics",
        "maximum_output_tokens",
    )

    @field_validator("diagnostic_codes")
    @classmethod
    def diagnostics_are_nonempty_and_unique(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if not value:
            raise ValueError("a repair must cite at least one validation diagnostic")
        if len(value) != len(set(value)):
            raise ValueError("repair diagnostic codes must be unique")
        return value

    @model_validator(mode="after")
    def repair_has_one_distinct_parent(self) -> Self:
        if self.root_attempt_id != self.base_attempt_id:
            raise ValueError("the only repair must descend directly from the base attempt")
        if self.repair_attempt_id == self.base_attempt_id:
            raise ValueError("repair_attempt_id must differ from base_attempt_id")
        if set(self.allowed_changes) != {"validation_diagnostics", "maximum_output_tokens"}:
            raise ValueError("repair may change only diagnostics and the registered output cap")
        return self


class RequestTimeoutClass(StrEnum):
    LONG = "long"
    STANDARD = "standard"
    SHORT = "short"


class GPURequestMetadata(RuntimeManifest):
    """Immutable admission metadata recorded before (not by) a real GPU request."""

    request_id: str = Field(min_length=1)
    attempt_id: str = Field(min_length=1)
    condition: LLMCondition
    backend: Literal["vllm_gpu"] = "vllm_gpu"
    model_repository: str = Field(min_length=1)
    model_revision: GitRevision
    tokenizer_revision: str = Field(min_length=1)
    quantization: Literal["awq"] = "awq"
    runtime: Literal["vllm"] = "vllm"
    runtime_version: str = Field(min_length=1)
    pytorch_version: str = Field(min_length=1)
    cuda_version: str = Field(min_length=1)
    driver_version: str = Field(min_length=1)
    hardware_manifest_hash: Sha256
    code_revision: GitRevision
    dirty_patch_hash: Sha256 | None = None
    prompt_hash: Sha256
    output_schema_hash: Sha256
    semantic_request_hash: Sha256
    evidence_snapshot_hash: Sha256 | None = None
    evidence_packet_hash: Sha256 | None = None
    sealed_ontology_hash: Sha256 | None = None
    decoding_manifest_hash: Sha256
    seed_manifest_hash: Sha256
    capability_manifest_hash: Sha256
    packing_report_hash: Sha256
    seed_block: NonNegativeInt
    decoding_pass: DecodingPass
    requested_at: AwareDatetime
    query_revealed_at: AwareDatetime | None = None
    timeout_class: RequestTimeoutClass
    watchdog_seconds: PositiveInt
    cpu_offload_gb: Literal[0] = 0
    request_concurrency: Literal[1] = 1
    allocated_gpu_seconds_before_request: float = Field(ge=0.0)
    repair_lineage_hash: Sha256 | None = None

    @model_validator(mode="after")
    def enforce_request_boundary(self) -> Self:
        timeout_ceiling = {
            RequestTimeoutClass.LONG: 240,
            RequestTimeoutClass.STANDARD: 150,
            RequestTimeoutClass.SHORT: 90,
        }[self.timeout_class]
        if self.watchdog_seconds > timeout_ceiling:
            raise ValueError(
                f"watchdog_seconds exceeds the {self.timeout_class.value} class ceiling"
            )
        if self.decoding_pass is DecodingPass.REPAIR and self.repair_lineage_hash is None:
            raise ValueError("repair decoding requires repair lineage metadata")
        if self.decoding_pass is DecodingPass.FIRST_PASS and self.repair_lineage_hash is not None:
            raise ValueError("first-pass request cannot carry repair lineage")

        if self.condition is ConditionName.C1_LLM_PRE:
            if self.query_revealed_at is not None or self.evidence_packet_hash is not None:
                raise ValueError("C1 GPU construction must remain before query reveal")
            if self.evidence_snapshot_hash is None:
                raise ValueError("C1 request requires the sealed evidence snapshot hash")
            if self.sealed_ontology_hash is not None:
                raise ValueError("C1 request cannot consume another ontology")
        else:
            if self.query_revealed_at is None:
                raise ValueError(f"{self.condition.value} requires a query-reveal timestamp")
            if self.requested_at < self.query_revealed_at:
                raise ValueError("query-time request was created before query reveal")
            if self.evidence_packet_hash is None:
                raise ValueError(f"{self.condition.value} requires a frozen evidence packet")
            if self.evidence_snapshot_hash is not None:
                raise ValueError(
                    "query-time request must reference the packet, not raw snapshot access"
                )
            if self.condition is ConditionName.A_FIXED_SELECT:
                if self.sealed_ontology_hash is None:
                    raise ValueError("A-FixedSelect requires the complete sealed C1 ontology")
            elif self.sealed_ontology_hash is not None:
                raise ValueError("C2 cannot consume a hidden prebuilt ontology")
        return self


def make_gpu_request_metadata(
    *,
    decoding: DecodingManifest,
    seeds: SeedManifest,
    capabilities: CapabilityManifest,
    packing: PackingReport,
    repair_lineage: RepairLineageMetadata | None = None,
    **metadata: object,
) -> GPURequestMetadata:
    """Bind separately stored manifests into one checked, content-addressed request row."""

    condition = metadata.get("condition")
    if condition != capabilities.condition or condition != packing.condition:
        raise ValueError("condition differs across request, capability, and packing manifests")
    if decoding.seed != seeds.llm_seed:
        raise ValueError("decoding seed must equal the deterministic seed-manifest LLM seed")
    if metadata.get("seed_block") != seeds.seed_block:
        raise ValueError("request seed_block differs from the seed manifest")
    if metadata.get("tokenizer_revision") != decoding.tokenizer_revision:
        raise ValueError("request and decoding manifests use different tokenizer revisions")
    if metadata.get("tokenizer_revision") != packing.tokenizer_revision:
        raise ValueError("request and packing manifests use different tokenizer revisions")
    if decoding.maximum_input_tokens != packing.maximum_input_tokens:
        raise ValueError("decoding and packing input caps differ")
    if decoding.maximum_output_tokens != packing.reserved_output_tokens:
        raise ValueError("decoding and packing output reservations differ")
    if decoding.maximum_model_tokens != packing.maximum_model_tokens:
        raise ValueError("decoding and packing model-token caps differ")
    if decoding.output_schema_hash != metadata.get("output_schema_hash"):
        raise ValueError("request and decoding output schema hashes differ")
    if decoding.decoding_pass is DecodingPass.REPAIR and repair_lineage is None:
        raise ValueError("repair request is missing repair lineage")
    if decoding.decoding_pass is DecodingPass.FIRST_PASS and repair_lineage is not None:
        raise ValueError("first-pass request cannot bind repair lineage")
    if (
        repair_lineage is not None
        and metadata.get("semantic_request_hash") != repair_lineage.semantic_request_hash
    ):
        raise ValueError("repair must retain the base attempt's semantic request hash")

    return GPURequestMetadata(
        **metadata,
        decoding_manifest_hash=decoding.content_hash,
        seed_manifest_hash=seeds.content_hash,
        capability_manifest_hash=capabilities.content_hash,
        packing_report_hash=packing.content_hash,
        decoding_pass=decoding.decoding_pass,
        repair_lineage_hash=repair_lineage.content_hash if repair_lineage else None,
    )


def utc_now() -> datetime:
    """Small injectable boundary for callers creating aware metadata timestamps."""

    return datetime.now().astimezone()
