"""Deterministic human-readable handoff for the blinded synthetic-gold review.

This scorer-only module renders the already sealed blind-review package.  It
does not derive gold, change any proposed interpretation, or create a reviewer
judgment.  Its output is restricted because it contains scorer-side gold.
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import os
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Any, Literal, Self

from pydantic import Field, model_validator

from story_projection_onto.contracts import (
    Identifier,
    ImmutableRecord,
    ReleaseClass,
    Sha256Digest,
    canonical_json,
    canonical_json_schema,
)
from story_projection_onto.synthetic_benchmark import (
    BlindIndependentReviewPackage,
    BlindReviewProjection,
    BlindReviewWorld,
    IndependentReviewResponse,
    ReviewCriterion,
)

DEFAULT_PACKAGE_PATH = Path("data/synthetic/scorer_only/review/blind_review_package.json")
DEFAULT_RESPONSE_SCHEMA_PATH = Path(
    "data/synthetic/scorer_only/review/reviewer_response.schema.json"
)
DEFAULT_OUTPUT_ROOT = Path("artifacts/restricted/scorer_only/independent_review_handoff")

PACKET_FILE = "review_packet.md"
RESPONSE_TEMPLATE_FILE = "review_response_template.tsv"
MANIFEST_FILE = "handoff_manifest.json"

_EXPECTED_FILE_NAMES = frozenset({PACKET_FILE, RESPONSE_TEMPLATE_FILE, MANIFEST_FILE})
_FORBIDDEN_CONDITION_MARKERS = (
    "c0_classical_pre",
    "c1_llm_pre",
    "c2_llm_query",
    "a_fixed_select",
    "a-fixedselect",
    "syn-test",
    "syn-dev",
    '"condition"',
    '"method_output"',
)


class IndependentReviewHandoffError(RuntimeError):
    """The human-readable scorer-only handoff cannot be safely produced."""


class HandoffFileBinding(ImmutableRecord):
    """Portable digest and size for one deterministic handoff file."""

    relative_path: Identifier
    media_type: Literal["text/markdown", "text/tab-separated-values"]
    size_bytes: Annotated[int, Field(gt=0)]
    file_sha256: Sha256Digest


class IndependentReviewHandoffManifest(ImmutableRecord):
    """Self-hashed proof that the worksheet is only a rendering of the package."""

    kind: Literal["independent_review_human_handoff"] = "independent_review_human_handoff"
    package_id: Identifier
    package_hash: Sha256Digest
    package_file_sha256: Sha256Digest
    response_schema_file_sha256: Sha256Digest
    selection_rule: Literal["one-seeded-world-per-difficulty-stratum"]
    world_count: Literal[3] = 3
    projection_count: Literal[9] = 9
    review_item_count: Literal[72] = 72
    condition_blind: Literal[True] = True
    contains_method_outputs: Literal[False] = False
    contains_reviewer_judgments: Literal[False] = False
    response_dispositions_prefilled: Literal[False] = False
    release_class: Literal[ReleaseClass.RESTRICTED] = ReleaseClass.RESTRICTED
    files: tuple[HandoffFileBinding, HandoffFileBinding]

    @model_validator(mode="after")
    def exact_file_inventory(self) -> Self:
        if {item.relative_path for item in self.files} != {
            PACKET_FILE,
            RESPONSE_TEMPLATE_FILE,
        }:
            raise ValueError("handoff manifest has an unexpected rendered-file inventory")
        return self


@dataclass(frozen=True, slots=True)
class IndependentReviewHumanHandoff:
    """In-memory deterministic handoff ready for append-only materialization."""

    packet_markdown: bytes
    response_template_tsv: bytes
    manifest: IndependentReviewHandoffManifest

    def file_bytes(self) -> dict[str, bytes]:
        return {
            PACKET_FILE: self.packet_markdown,
            RESPONSE_TEMPLATE_FILE: self.response_template_tsv,
            MANIFEST_FILE: (self.manifest.to_canonical_json() + "\n").encode("utf-8"),
        }


def _file_sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _markdown_text(value: object) -> str:
    text = "not specified" if value is None else str(value)
    return (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace("|", "\\|")
        .replace("\r", " ")
        .replace("\n", " ")
    )


def _code(value: object) -> str:
    return f"`{_markdown_text(value)}`"


def _joined_codes(values: Sequence[object]) -> str:
    return ", ".join(_code(value) for value in values) if values else "none"


def _decode_embedded_json(value: Any) -> Any:
    """Decode nested canonical-JSON strings for a readable, lossless review view."""

    if isinstance(value, str):
        candidate = value.strip()
        if (
            len(candidate) >= 2
            and (candidate[0], candidate[-1]) in {("{", "}"), ("[", "]")}
        ):
            try:
                return _decode_embedded_json(json.loads(value))
            except json.JSONDecodeError:
                return value
        return value
    if isinstance(value, Mapping):
        return {str(key): _decode_embedded_json(child) for key, child in value.items()}
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [_decode_embedded_json(child) for child in value]
    return value


def _readable_accepted_representation(value: object) -> str:
    return json.dumps(
        _decode_embedded_json(value),
        ensure_ascii=False,
        indent=2,
        sort_keys=True,
    )


def _time_label(value: Mapping[str, Any] | None) -> str:
    if value is None:
        return "not specified"
    if value.get("label"):
        return _markdown_text(value["label"])
    kind = value.get("kind", "unspecified")
    if kind == "point":
        return f"point {_markdown_text(value.get('point'))}"
    if kind == "interval":
        return (
            f"interval {_markdown_text(value.get('start'))} to {_markdown_text(value.get('end'))}"
        )
    if kind == "through_bound":
        return f"through {_markdown_text(value.get('end'))}"
    relation = value.get("relation")
    anchor = value.get("anchor_id")
    if relation or anchor:
        return f"{_markdown_text(kind)} ({_markdown_text(relation)} {_markdown_text(anchor)})"
    return _markdown_text(kind)


def _discourse_label(value: Mapping[str, Any] | None) -> str:
    if value is None:
        return "not specified"
    return (
        f"passage {value.get('passage_order', '?')}, "
        f"sentence {value.get('sentence_order', '?')}, "
        f"token {value.get('token_order', '?')}"
    )


def _revelation_label(value: Mapping[str, Any] | None) -> str:
    if value is None:
        return "not specified"
    label = value.get("label")
    order = value.get("revelation_order")
    if label:
        return f"{_markdown_text(label)} (order {order})"
    return f"revelation order {order}"


@dataclass(frozen=True, slots=True)
class _WorldLabels:
    mentions: Mapping[str, str]
    mention_evidence: Mapping[str, str]


def _world_labels(world: BlindReviewWorld) -> _WorldLabels:
    mentions: dict[str, str] = {}
    mention_evidence: dict[str, str] = {}
    for evidence in world.evidence:
        evidence_id = str(evidence["evidence_id"])
        for mention in evidence.get("mention_candidates", ()):
            mention_id = str(mention["candidate_id"])
            surface = _markdown_text(mention.get("surface", mention_id))
            provisional_type = _markdown_text(mention.get("provisional_type", "candidate"))
            mentions[mention_id] = f"{surface} ({provisional_type})"
            mention_evidence[mention_id] = evidence_id
    return _WorldLabels(mentions=mentions, mention_evidence=mention_evidence)


@dataclass(frozen=True, slots=True)
class _ProjectionLabels:
    targets: Mapping[str, str]
    predicates: Mapping[str, str]


def _projection_labels(
    projection: BlindReviewProjection,
    world_labels: _WorldLabels,
) -> _ProjectionLabels:
    targets: dict[str, str] = {}
    for partition in projection.proposed_identity_partitions:
        cluster_id = str(partition["cluster_id"])
        surfaces = sorted(
            {
                world_labels.mentions.get(str(mention_id), str(mention_id)).split(" (")[0]
                for mention_id in partition.get("mention_candidate_ids", ())
            }
        )
        targets[cluster_id] = " / ".join(surfaces) if surfaces else cluster_id
    for event in projection.proposed_events:
        targets[str(event["event_id"])] = _markdown_text(event.get("label", event["event_id"]))
    schema = projection.proposed_local_schema
    predicates = {
        str(predicate["predicate_id"]): _markdown_text(
            predicate.get("label", predicate["predicate_id"])
        )
        for predicate in schema.get("predicates", ())
    }
    return _ProjectionLabels(targets=targets, predicates=predicates)


def _target(value: object, labels: _ProjectionLabels) -> str:
    if value is None:
        return "not specified"
    identifier = str(value)
    human = labels.targets.get(identifier)
    return _code(identifier) if human is None else f"{human} ({_code(identifier)})"


def _render_evidence(world: BlindReviewWorld, labels: _WorldLabels) -> list[str]:
    lines = [
        f"### Evidence ({len(world.evidence)} records)",
        "",
        "The same evidence below applies to contexts A, B, and C in this blind world.",
        "",
    ]
    for index, evidence in enumerate(world.evidence, start=1):
        evidence_id = str(evidence["evidence_id"])
        lines.extend(
            [
                f"#### Evidence {index}: {_code(evidence_id)}",
                "",
                f"> {_markdown_text(evidence.get('text', ''))}",
                "",
                f"- Discourse position: {_discourse_label(evidence.get('discourse_position'))}",
                f"- Evidence confidence: {evidence.get('confidence', 'not specified')}",
            ]
        )
        mentions = evidence.get("mention_candidates", ())
        if mentions:
            rendered_mentions = []
            for mention in mentions:
                mention_id = str(mention["candidate_id"])
                rendered_mentions.append(
                    f"{labels.mentions.get(mention_id, mention_id)} ({_code(mention_id)})"
                )
            lines.append("- Mention candidates: " + "; ".join(rendered_mentions))
        relations = evidence.get("relation_phrase_candidates", ())
        if relations:
            rendered_relations = []
            for relation in relations:
                subject = labels.mentions.get(
                    str(relation.get("subject_mention_candidate_id")),
                    str(relation.get("subject_mention_candidate_id")),
                )
                object_ = labels.mentions.get(
                    str(relation.get("object_mention_candidate_id")),
                    str(relation.get("object_mention_candidate_id")),
                )
                rendered_relations.append(
                    f"{subject} — {_markdown_text(relation.get('surface_phrase'))} → {object_} "
                    f"({_code(relation.get('candidate_id'))}, confidence "
                    f"{relation.get('confidence', 'not specified')})"
                )
            lines.append("- Surface relation candidates: " + "; ".join(rendered_relations))
        event_candidates = evidence.get("event_candidates", ())
        if event_candidates:
            lines.append(
                "- Event candidates: "
                + "; ".join(
                    f"{_markdown_text(item.get('trigger_surface') or item.get('event_kind'))} "
                    f"({_code(item.get('candidate_id'))})"
                    for item in event_candidates
                )
            )
        clues = evidence.get("temporal_clues", ())
        if clues:
            lines.append(
                "- Temporal/revelation clues: "
                + "; ".join(
                    f"{_markdown_text(item.get('normalized_expression'))} "
                    f"({_code(item.get('clue_id'))}; targets "
                    f"{_joined_codes(item.get('target_candidate_ids', ()))})"
                    for item in clues
                )
            )
        lines.append("")
    return lines


def _render_query(projection: BlindReviewProjection, labels: _ProjectionLabels) -> list[str]:
    query = projection.query
    viewpoint = query.get("viewpoint") or {}
    story_scope = query.get("story_scope")
    horizon = query.get("spoiler_horizon") or {}
    budgets = query.get("budgets") or {}
    holder = viewpoint.get("holder_id")
    return [
        "#### Review context",
        "",
        f"- Question: {_markdown_text(query.get('wording'))}",
        f"- Target: {_markdown_text(query.get('target'))}",
        f"- Lens: {_markdown_text(query.get('lens'))}",
        f"- Abstraction: {_markdown_text(query.get('abstraction'))}",
        f"- Story-time scope: {_time_label(story_scope)}",
        f"- Viewpoint holder: {_target(holder, labels)}",
        f"- Attitude scope: {_markdown_text(viewpoint.get('attitude_scope'))}",
        f"- Spoiler/discourse horizon: {_discourse_label(horizon.get('max_discourse_position'))}",
        f"- Revelation horizon: {_revelation_label(horizon.get('max_revelation_position'))}",
        "- Object budgets: "
        f"nodes {budgets.get('node_budget', '?')}, assertions "
        f"{budgets.get('assertion_budget', '?')}",
        "",
    ]


def _render_local_schema(projection: BlindReviewProjection) -> list[str]:
    schema = projection.proposed_local_schema
    lines = [
        "##### Contextual schema",
        "",
        f"- Proposed abstraction: {_markdown_text(schema.get('abstraction'))}",
        f"- Schema ID: {_code(schema.get('schema_id'))}",
        "- Contextual types:",
    ]
    for item in schema.get("contextual_types", ()):
        lines.append(
            "  - "
            f"{_markdown_text(item.get('label'))} ({_code(item.get('type_id'))}; parent "
            f"{_code(item.get('parent_upper_type'))}): "
            f"{_markdown_text(item.get('definition'))}; evidence "
            f"{_joined_codes(item.get('evidence_ids', ()))}"
        )
    lines.append("- Contextual predicates:")
    for item in schema.get("predicates", ()):
        roles = item.get("role_names", ())
        role_text = f"; roles {', '.join(map(str, roles))}" if roles else ""
        lines.append(
            "  - "
            f"{_markdown_text(item.get('label'))} ({_code(item.get('predicate_id'))}; "
            f"arity {item.get('arity', '?')}; parent "
            f"{_code(item.get('parent_upper_relation'))}{role_text}): "
            f"{_markdown_text(item.get('definition'))}; evidence "
            f"{_joined_codes(item.get('evidence_ids', ()))}"
        )
    lines.append("")
    return lines


def _render_partitions(
    projection: BlindReviewProjection,
    world_labels: _WorldLabels,
    labels: _ProjectionLabels,
) -> list[str]:
    relevance = {
        str(item["target_id"]): bool(item["is_relevant"]) for item in projection.proposed_relevance
    }
    lines = ["##### Proposed identity partitions", ""]
    for index, partition in enumerate(projection.proposed_identity_partitions, start=1):
        cluster_id = str(partition["cluster_id"])
        lines.append(
            f"{index}. {_target(cluster_id, labels)} — answer-relevant: "
            f"{'yes' if relevance.get(cluster_id, False) else 'no'}"
        )
        for mention_id_raw in partition.get("mention_candidate_ids", ()):
            mention_id = str(mention_id_raw)
            lines.append(
                "   - "
                f"{world_labels.mentions.get(mention_id, mention_id)} ({_code(mention_id)}; "
                f"evidence {_code(world_labels.mention_evidence.get(mention_id, 'unknown'))})"
            )
    lines.append("")
    return lines


def _render_events(
    projection: BlindReviewProjection,
    labels: _ProjectionLabels,
) -> list[str]:
    relevance = {
        str(item["target_id"]): bool(item["is_relevant"]) for item in projection.proposed_relevance
    }
    lines = ["##### Proposed event objects", ""]
    if not projection.proposed_events:
        lines.extend(["No event object is proposed in this context.", ""])
        return lines
    for index, event in enumerate(projection.proposed_events, start=1):
        event_id = str(event["event_id"])
        lines.extend(
            [
                f"{index}. {_target(event_id, labels)}",
                f"   - Type: {_code(event.get('contextual_type_id'))}",
                f"   - Occurrence: {_time_label(event.get('occurrence_time'))}",
                f"   - Reification reason: {_markdown_text(event.get('reification_reason'))}",
                f"   - Description: {_markdown_text(event.get('description'))}",
                f"   - Evidence: {_joined_codes(event.get('evidence_ids', ()))}",
                f"   - Confidence/uncertainty: {event.get('confidence', '?')} / "
                f"{_markdown_text(event.get('uncertainty'))}",
                f"   - Answer-relevant: {'yes' if relevance.get(event_id, False) else 'no'}",
            ]
        )
    lines.append("")
    return lines


def _assertion_statement(
    assertion: Mapping[str, Any],
    labels: _ProjectionLabels,
) -> str:
    predicate_id = str(assertion.get("predicate_id"))
    predicate = labels.predicates.get(predicate_id, predicate_id)
    roles = assertion.get("roles", ())
    if roles:
        role_text = "; ".join(
            f"{_markdown_text(role.get('role'))}={_target(role.get('object_id'), labels)}"
            for role in roles
        )
        return f"{predicate} ({role_text})"
    return (
        f"{_target(assertion.get('subject_id'), labels)} — {predicate} → "
        f"{_target(assertion.get('object_id'), labels)}"
    )


def _render_assertions(
    projection: BlindReviewProjection,
    labels: _ProjectionLabels,
) -> list[str]:
    annotations = {
        str(item["assertion_id"]): item for item in projection.proposed_rare_pivotal_labels
    }
    relevance = {
        str(item["target_id"]): bool(item["is_relevant"]) for item in projection.proposed_relevance
    }
    lines = [
        f"##### Proposed qualified assertions ({len(projection.proposed_qualified_assertions)})",
        "",
    ]
    for index, assertion in enumerate(projection.proposed_qualified_assertions, start=1):
        assertion_id = str(assertion["assertion_id"])
        temporal = assertion.get("temporal_scope", {})
        epistemic = assertion.get("epistemic_scope")
        annotation = annotations.get(assertion_id, {})
        lines.extend(
            [
                f"{index}. **{_assertion_statement(assertion, labels)}**",
                f"   - Assertion ID/direction: {_code(assertion_id)} / "
                f"{_markdown_text(assertion.get('direction'))}",
                f"   - Evidence: {_joined_codes(assertion.get('evidence_ids', ()))}",
                f"   - Story time: {_time_label(temporal.get('story_time'))}",
                f"   - Validity time: {_time_label(temporal.get('validity_time'))}",
                f"   - Discourse position: {_discourse_label(temporal.get('discourse_position'))}",
                "   - Revelation position: "
                f"{_revelation_label(temporal.get('revelation_position'))}",
                f"   - Narrative commitment: "
                f"{_markdown_text(assertion.get('narrative_commitment'))}",
            ]
        )
        if epistemic is None:
            lines.append("   - Holder-relative epistemic status: none (world-level commitment)")
        else:
            lines.append(
                "   - Holder-relative epistemic status: "
                f"{_target(epistemic.get('holder_id'), labels)} — "
                f"{_markdown_text(epistemic.get('attitude'))}; holder time "
                f"{_time_label(epistemic.get('holder_relative_time'))}; evidence "
                f"{_joined_codes(epistemic.get('evidence_ids', ()))}"
            )
        lines.extend(
            [
                f"   - Confidence/contextual relevance: "
                f"{assertion.get('confidence', '?')} / "
                f"{assertion.get('contextual_relevance', '?')}",
                f"   - Marked answer-relevant: "
                f"{'yes' if relevance.get(assertion_id, False) else 'no'}",
                f"   - Rare/pivotal: {'yes' if annotation.get('is_rare') else 'no'} / "
                f"{'yes' if annotation.get('is_pivotal') else 'no'}; support assertions "
                f"{_joined_codes(annotation.get('support_path_assertion_ids', ()))}",
                f"   - Why it matters: {_markdown_text(assertion.get('why_matters'))}",
                f"   - Why-it-matters evidence: "
                f"{_joined_codes(assertion.get('why_matters_evidence_ids', ()))}",
            ]
        )
        provenance = assertion.get("provenance", ())
        if provenance:
            lines.append(
                "   - Provenance: "
                + "; ".join(
                    f"{_code(item.get('evidence_id'))} at "
                    f"{_code(item.get('locator'))} "
                    f"({_markdown_text(item.get('extraction_method'))})"
                    for item in provenance
                )
            )
    lines.append("")
    return lines


def _render_rare_path(
    projection: BlindReviewProjection,
) -> list[str]:
    path = projection.proposed_rare_support_path
    lines = ["##### Rare-pivotal support path", ""]
    if path is None:
        lines.extend(["No rare support path is proposed for this context.", ""])
        return lines
    lines.extend(
        [
            f"- Path ID: {_code(path.get('path_id'))}",
            f"- Impact kind: {_markdown_text(path.get('impact_kind'))}",
            f"- Rare assertion: {_code(path.get('rare_assertion_id'))}",
            f"- Outcome assertion: {_code(path.get('outcome_assertion_id'))}",
            "- Proposed causal steps:",
        ]
    )
    for step in path.get("steps", ()):
        lines.append(
            "  - "
            f"{_code(step.get('source_assertion_id'))} — "
            f"{_code(step.get('causal_assertion_id'))} → "
            f"{_code(step.get('target_assertion_id'))}; evidence "
            f"{_joined_codes(step.get('evidence_ids', ()))}"
        )
    lines.append("")
    return lines


def _render_communities(
    projection: BlindReviewProjection,
    labels: _ProjectionLabels,
) -> list[str]:
    members: dict[str, list[str]] = defaultdict(list)
    for item in projection.proposed_communities:
        members[str(item["community_id"])].append(str(item["anchor_id"]))
    rationales = {str(item["anchor_id"]): item for item in projection.proposed_community_rationales}
    lines = ["##### Proposed communities", ""]
    if not members:
        lines.extend(["No reviewed community partition is proposed for this context.", ""])
        return lines
    for community_id in sorted(members):
        lines.append(f"- Community {_code(community_id)}")
        for anchor_id in sorted(members[community_id]):
            rationale = rationales.get(anchor_id, {})
            lines.append(
                "  - "
                f"{_target(anchor_id, labels)}; basis "
                f"{_markdown_text(rationale.get('semantic_basis'))}; evidence "
                f"{_joined_codes(rationale.get('evidence_ids', ()))}"
            )
    lines.append("")
    return lines


def _render_contrasts(projection: BlindReviewProjection) -> list[str]:
    lines = ["##### Proposed contrast behavior", ""]
    if projection.proposed_contrast_decisions:
        lines.append("- Signed changes:")
        for item in projection.proposed_contrast_decisions:
            lines.append(
                "  - "
                f"{_markdown_text(item.get('operator'))} / "
                f"{_markdown_text(item.get('direction'))}: "
                f"{_markdown_text(item.get('expected_signature'))}; anchors "
                f"{_joined_codes(item.get('anchor_ids', ()))}; evidence "
                f"{_joined_codes(item.get('evidence_ids', ()))}"
            )
    else:
        lines.append("- Signed changes: none proposed for this context.")
    if projection.proposed_contrast_invariants:
        lines.append("- Invariants:")
        for item in projection.proposed_contrast_invariants:
            lines.append(
                "  - "
                f"{_markdown_text(item.get('expected_signature'))}; anchors "
                f"{_joined_codes(item.get('anchor_ids', ()))}; evidence "
                f"{_joined_codes(item.get('evidence_ids', ()))}"
            )
    else:
        lines.append("- Invariants: none proposed for this context.")
    lines.append("")
    return lines


def _render_answer_signature(projection: BlindReviewProjection) -> list[str]:
    signature = projection.proposed_answer_signature
    lines = ["##### Proposed answer-signature components", ""]
    for key in sorted(signature):
        values = signature[key]
        lines.append(f"- {_markdown_text(key.replace('_', ' ').title())}:")
        if isinstance(values, Sequence) and not isinstance(values, str):
            if values:
                lines.extend(f"  - {_code(value)}" for value in values)
            else:
                lines.append("  - none")
        else:
            lines.append(f"  - {_code(values)}")
    lines.append("")
    return lines


def _render_alternatives(projection: BlindReviewProjection) -> list[str]:
    alternatives = projection.permissible_alternatives
    lines = [
        "##### Proposed permissible alternatives",
        "",
        f"- Equivalence rule: {_markdown_text(alternatives.get('equivalence_rule'))}",
        f"- Matching rule: {_markdown_text(alternatives.get('matching_rule'))}",
    ]
    constraint_alternatives = alternatives.get("constraint_alternatives", ())
    if not constraint_alternatives:
        lines.append("- No constraint alternative is proposed.")
    for alternative in constraint_alternatives:
        alternative_id = str(alternative.get("alternative_id"))
        lines.extend(
            [
                f"- Alternative {_code(alternative_id)}: "
                f"{_markdown_text(alternative.get('description'))}",
            ]
        )
        for constraint in alternative.get("constraints", ()):
            field_path = str(constraint.get("field_path"))
            accepted = tuple(constraint.get("accepted_values", ()))
            lines.append(
                "- Constraint "
                f"{_code(field_path)} "
                f"{_markdown_text(constraint.get('operator'))}: "
                f"{len(accepted)} exact accepted representations."
            )
            for index, value in enumerate(accepted, start=1):
                source_value = str(value)
                fingerprint = hashlib.sha256(source_value.encode("utf-8")).hexdigest()
                lines.extend(
                    [
                        "",
                        f"###### Accepted representation {index} of {len(accepted)} for "
                        f"{_code(field_path)} in {_code(alternative_id)}",
                        "",
                        f"- Source-value SHA-256: {_code(fingerprint)}",
                        "- Decoded exact representation:",
                        "",
                        "```json",
                        *_readable_accepted_representation(source_value).splitlines(),
                        "```",
                    ]
                )
    lines.append("")
    return lines


def _render_review_questions(projection: BlindReviewProjection) -> list[str]:
    lines = [
        "#### Review questions",
        "",
        "For each item, choose exactly one of **agree**, **disagree**, or **uncertain** in "
        f"`{RESPONSE_TEMPLATE_FILE}`. For disagreement or uncertainty, explain the issue and "
        "cite the relevant opaque evidence IDs.",
        "",
    ]
    ordered = {criterion: index for index, criterion in enumerate(ReviewCriterion)}
    for item in sorted(projection.review_items, key=lambda value: ordered[value.criterion]):
        lines.append(
            f"{ordered[item.criterion] + 1}. **{_markdown_text(item.criterion.value)}** "
            f"({_code(item.review_item_id)}): {_markdown_text(item.prompt)}"
        )
    lines.append("")
    return lines


def _render_projection(
    projection: BlindReviewProjection,
    *,
    world_labels: _WorldLabels,
    projection_number: int,
) -> list[str]:
    labels = _projection_labels(projection, world_labels)
    lines = [
        f"### Projection {projection_number} of 9 — context {projection.context_label}",
        "",
        f"Blind projection ID: {_code(projection.blind_projection_id)}",
        "",
        "The following is the **proposed gold interpretation to review**, not an evaluated "
        "system output and not a reviewer decision.",
        "",
    ]
    lines.extend(_render_query(projection, labels))
    lines.extend(_render_local_schema(projection))
    lines.extend(_render_partitions(projection, world_labels, labels))
    lines.extend(_render_events(projection, labels))
    lines.extend(_render_assertions(projection, labels))
    lines.extend(_render_rare_path(projection))
    lines.extend(_render_communities(projection, labels))
    lines.extend(_render_contrasts(projection))
    lines.extend(_render_answer_signature(projection))
    lines.extend(_render_alternatives(projection))
    lines.extend(_render_review_questions(projection))
    return lines


def _render_packet(package: BlindIndependentReviewPackage) -> bytes:
    lines = [
        "# Condition-blind independent-review packet",
        "",
        "> Restricted scorer-only material. Keep this packet out of model inputs, public "
        "artifacts, and condition-output workspaces.",
        "",
        f"- Package ID: {_code(package.package_id)}",
        f"- Package content hash: {_code(package.content_hash)}",
        f"- Selection rule: {_markdown_text(package.selection_rule)}",
        "- Coverage: three seeded blind worlds, nine projections, eight questions per "
        "projection (72 responses total)",
        "- Condition blindness: yes; evaluated outputs included: no",
        "",
        "## Reviewer instructions",
        "",
    ]
    lines.extend(f"- {_markdown_text(item)}" for item in package.reviewer_instructions)
    lines.extend(
        [
            "- Review the proposal against the supplied evidence rather than trying to infer "
            "which construction pathway will later be evaluated.",
            "- Record disagreements and defensible alternatives; adjudication happens before "
            "condition outputs are opened.",
            "- This rendering does not prefill or imply any substantive judgment.",
            "",
            "## What the eight questions cover",
            "",
            "The questions are copied from the sealed package. Together they cover evidence "
            "support, identity partitions, event choices, temporal scope, contrast deltas, "
            "rare-pivotal labels, communities, and permissible alternatives.",
            "",
        ]
    )
    projection_number = 0
    for world_number, world in enumerate(package.worlds, start=1):
        labels = _world_labels(world)
        lines.extend(
            [
                f"## Blind world {world_number} of 3",
                "",
                f"Blind world ID: {_code(world.blind_world_id)}",
                "",
            ]
        )
        lines.extend(_render_evidence(world, labels))
        for projection in world.projections:
            projection_number += 1
            lines.extend(
                _render_projection(
                    projection,
                    world_labels=labels,
                    projection_number=projection_number,
                )
            )
    return ("\n".join(lines).rstrip() + "\n").encode("utf-8")


def _render_response_template(package: BlindIndependentReviewPackage) -> bytes:
    stream = io.StringIO(newline="")
    stream.write(f"# package_id\t{package.package_id}\n")
    stream.write(f"# package_hash\t{package.content_hash}\n")
    stream.write("# reviewer_pseudonym\t\n")
    stream.write("# reviewed_at_utc\t\n")
    stream.write("# allowed_dispositions\tagree | disagree | uncertain\n")
    writer = csv.writer(stream, delimiter="\t", lineterminator="\n")
    writer.writerow(
        (
            "world_number",
            "projection_number",
            "context_label",
            "blind_projection_id",
            "review_item_id",
            "criterion",
            "question",
            "disposition",
            "notes_and_evidence_ids",
        )
    )
    projection_number = 0
    for world_number, world in enumerate(package.worlds, start=1):
        for projection in world.projections:
            projection_number += 1
            for item in projection.review_items:
                writer.writerow(
                    (
                        world_number,
                        projection_number,
                        projection.context_label,
                        projection.blind_projection_id,
                        item.review_item_id,
                        item.criterion.value,
                        item.prompt,
                        "",
                        "",
                    )
                )
    return stream.getvalue().encode("utf-8")


def _load_package(path: Path) -> tuple[BlindIndependentReviewPackage, bytes]:
    supplied = Path(path)
    if supplied.is_symlink():
        raise IndependentReviewHandoffError("blind-review package cannot be a symbolic link")
    try:
        resolved = supplied.resolve(strict=True)
    except OSError as exc:
        raise IndependentReviewHandoffError("blind-review package does not exist") from exc
    if not resolved.is_file():
        raise IndependentReviewHandoffError("blind-review package must be a regular file")
    raw = resolved.read_bytes()
    package = BlindIndependentReviewPackage.model_validate_json(raw)
    canonical = (package.to_canonical_json() + "\n").encode("utf-8")
    if raw != canonical:
        raise IndependentReviewHandoffError("blind-review package bytes are not canonical")
    return package, raw


def _load_response_schema(path: Path) -> bytes:
    supplied = Path(path)
    if supplied.is_symlink():
        raise IndependentReviewHandoffError("reviewer-response schema cannot be a symbolic link")
    try:
        resolved = supplied.resolve(strict=True)
    except OSError as exc:
        raise IndependentReviewHandoffError("reviewer-response schema does not exist") from exc
    if not resolved.is_file():
        raise IndependentReviewHandoffError("reviewer-response schema must be a regular file")
    raw = resolved.read_bytes()
    try:
        observed = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise IndependentReviewHandoffError("reviewer-response schema is not valid JSON") from exc
    expected = canonical_json_schema(IndependentReviewResponse)
    if observed != expected or raw != (canonical_json(expected) + "\n").encode("utf-8"):
        raise IndependentReviewHandoffError(
            "reviewer-response schema differs from the exact typed response contract"
        )
    return raw


def build_independent_review_handoff(
    *,
    package_path: Path = DEFAULT_PACKAGE_PATH,
    response_schema_path: Path = DEFAULT_RESPONSE_SCHEMA_PATH,
) -> IndependentReviewHumanHandoff:
    """Render the exact blind package without adding any substantive judgment."""

    package, package_bytes = _load_package(package_path)
    schema_bytes = _load_response_schema(response_schema_path)
    packet = _render_packet(package)
    response = _render_response_template(package)
    rendered = (packet + response).decode("utf-8").casefold()
    forbidden = [marker for marker in _FORBIDDEN_CONDITION_MARKERS if marker in rendered]
    if forbidden:
        raise IndependentReviewHandoffError(
            "human handoff contains a forbidden condition/output marker: " + ", ".join(forbidden)
        )
    manifest = IndependentReviewHandoffManifest(
        package_id=package.package_id,
        package_hash=package.content_hash,
        package_file_sha256=_file_sha256(package_bytes),
        response_schema_file_sha256=_file_sha256(schema_bytes),
        selection_rule=package.selection_rule,
        files=(
            HandoffFileBinding(
                relative_path=PACKET_FILE,
                media_type="text/markdown",
                size_bytes=len(packet),
                file_sha256=_file_sha256(packet),
            ),
            HandoffFileBinding(
                relative_path=RESPONSE_TEMPLATE_FILE,
                media_type="text/tab-separated-values",
                size_bytes=len(response),
                file_sha256=_file_sha256(response),
            ),
        ),
    )
    return IndependentReviewHumanHandoff(
        packet_markdown=packet,
        response_template_tsv=response,
        manifest=manifest,
    )


def _require_restricted_output_root(path: Path) -> Path:
    supplied = Path(path)
    if supplied.is_symlink():
        raise IndependentReviewHandoffError("handoff output root cannot be a symbolic link")
    resolved = supplied.resolve(strict=False)
    if "restricted" not in resolved.parts or "scorer_only" not in resolved.parts:
        raise IndependentReviewHandoffError(
            "human review handoff must remain under an explicitly restricted scorer-only path"
        )
    return resolved


def _verify_exact_materialization(
    output_root: Path,
    expected: Mapping[str, bytes],
) -> Literal["already_exact"]:
    if output_root.is_symlink() or not output_root.is_dir():
        raise IndependentReviewHandoffError("existing handoff root is not a regular directory")
    children = tuple(output_root.iterdir())
    if {item.name for item in children} != _EXPECTED_FILE_NAMES:
        raise IndependentReviewHandoffError(
            "existing handoff has a partial, extra, or changed file inventory"
        )
    for item in children:
        if item.is_symlink() or not item.is_file():
            raise IndependentReviewHandoffError("handoff files must be regular non-symlink files")
        if item.read_bytes() != expected[item.name]:
            raise IndependentReviewHandoffError(
                f"existing handoff file differs from deterministic rendering: {item.name}"
            )
    return "already_exact"


def materialize_independent_review_handoff(
    handoff: IndependentReviewHumanHandoff,
    output_root: Path = DEFAULT_OUTPUT_ROOT,
) -> Literal["created", "already_exact"]:
    """Write the restricted rendering once; exact replay is the only accepted reuse."""

    destination = _require_restricted_output_root(output_root)
    expected = handoff.file_bytes()
    if destination.exists():
        return _verify_exact_materialization(destination, expected)
    destination.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    try:
        destination.mkdir(mode=0o700)
    except FileExistsError:
        return _verify_exact_materialization(destination, expected)
    try:
        for name in (PACKET_FILE, RESPONSE_TEMPLATE_FILE, MANIFEST_FILE):
            target = destination / name
            flags = os.O_CREAT | os.O_EXCL | os.O_WRONLY
            if hasattr(os, "O_NOFOLLOW"):
                flags |= os.O_NOFOLLOW
            descriptor = os.open(target, flags, 0o600)
            try:
                with os.fdopen(descriptor, "wb", closefd=True) as stream:
                    stream.write(expected[name])
                    stream.flush()
                    os.fsync(stream.fileno())
            except BaseException:
                raise
        directory_descriptor = os.open(destination, os.O_RDONLY)
        try:
            os.fsync(directory_descriptor)
        finally:
            os.close(directory_descriptor)
    except BaseException as exc:
        raise IndependentReviewHandoffError(
            "handoff materialization was interrupted; preserve the partial restricted tree"
        ) from exc
    return "created"


__all__ = [
    "DEFAULT_OUTPUT_ROOT",
    "DEFAULT_PACKAGE_PATH",
    "DEFAULT_RESPONSE_SCHEMA_PATH",
    "HandoffFileBinding",
    "IndependentReviewHandoffError",
    "IndependentReviewHandoffManifest",
    "IndependentReviewHumanHandoff",
    "build_independent_review_handoff",
    "materialize_independent_review_handoff",
]
