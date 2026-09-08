"""Proposed diagnostic adapter: administrative IDs, never semantic disambiguation.

Opt-in small diagnostics only, never ordinary/held-out execution. A field's declared reference type
can resolve cross-type reuse; an ambiguous untyped decision target cannot.
"""

from __future__ import annotations

import copy
from collections import Counter
from collections.abc import Mapping, Sequence
from typing import Any

from story_projection_onto.contracts import canonical_sha256

REVISION = "typed-local-identifiers-candidate-v2"
FROZEN_FIRST_REQUEST = "fc57ab003859a7bfd7c8075fb2f1b1bec6b102f970238f778c15d392dac83693"


def build_typed_small_request(fixture, tokenizer, tokenizer_manifest):
    """The previously prepared request, rebuilt from complete evidence, not outputs."""
    import json

    from story_projection_onto.contracts import canonical_json
    from story_projection_onto.gpu_runtime import ChatMessage
    from story_projection_onto.representation_diagnostic import _repack
    from story_projection_onto.semantic_generation import build_small_request, schema_guide

    base = build_small_request(fixture, tokenizer, tokenizer_manifest)
    # The prepared candidate was derived from the immutable canonical request,
    # whose key order and numeric spelling determine the readable field guide.
    schema = typed_identifier_schema(json.loads(canonical_json(base.output_schema)))
    instruction = base.messages[0].content.split("The following complete field guide")[0]
    begin = instruction.index("New local IDs start with n")
    end = instruction.index("Evidence", begin)
    instruction = instruction[:begin] + IDENTIFIER_INSTRUCTION + " " + instruction[end:]
    system = (
        instruction + "The following complete field guide matches the supplied grammar. "
        "It is syntax, not an answer.\n" + schema_guide(schema)
    )
    return _repack(
        base,
        (ChatMessage(role="system", content=system), *base.messages[1:]),
        schema,
        tokenizer,
        label="typed-small",
    )


# Independent short counters, not a shared counter the model must maintain.
PREFIXES = dict(
    schema="nS",
    type="nT",
    predicate="nR",
    entity="nE",
    event="nV",
    proposition="nP",
    assertion="nA",
    decision="nD",
)
RECORDS = (
    ("schema", "local_schema", "schema_id"),
    ("type", "local_schema/contextual_types", "type_id"),
    ("predicate", "local_schema/predicates", "predicate_id"),
    ("entity", "instance_graph/entities", "entity_id"),
    ("event", "instance_graph/events", "event_id"),
    ("proposition", "instance_graph/proposition_contents", "proposition_content_id"),
    ("assertion", "instance_graph/assertions", "assertion_id"),
    ("decision", "decisions", "decision_id"),
)
TARGETS = tuple(k for k in PREFIXES if k != "decision")
REFERENCE_KINDS = {
    "contextual_type_id": ("type",),
    "domain_type_ids": ("type",),
    "range_type_ids": ("type",),
    "predicate_id": ("predicate",),
    "subject_id": ("entity", "event"),
    "object_id": ("entity", "event"),
    "holder_id": ("entity",),
    "description_assertion_ids": ("assertion",),
    "proposition_content_id": ("proposition",),
    "input_object_ids": TARGETS,
    "created_object_ids": TARGETS,
    "removed_object_ids": TARGETS,
    # Relative anchors can include supplied candidates; no inferred coordinates.
    "anchor_id": TARGETS,
    "left_id": TARGETS,
    "right_id": TARGETS,
}
SUPPLIED_FIELDS = {
    "evidence_id",
    "evidence_ids",
    "why_matters_evidence_ids",
    "supported_mention_candidate_ids",
    "parent_upper_type",
    "parent_upper_relation",
}


def identifier_audit(generated: Mapping[str, Any], supplied_ids: Sequence[str] = ()) -> dict:
    """Return ALL declarations/references, not just the first collision."""
    declarations = []
    for kind, location, field in RECORDS:
        value = generated
        for part in location.split("/"):
            value = value[part]
        for index, record in enumerate(value if isinstance(value, list) else [value]):
            path = "/" + location + (f"/{index}" if isinstance(value, list) else "")
            declarations.append(dict(kind=kind, path=path, field=field, id=record[field]))
    declaration_paths = {d["path"] + "/" + d["field"] for d in declarations}
    references = []
    supplied = set(supplied_ids)

    def walk(value, path="", field=""):
        if path in declaration_paths:
            return
        if isinstance(value, dict):
            for k, v in value.items():
                walk(v, path + "/" + k, k)
        elif isinstance(value, list):
            for i, v in enumerate(value):
                walk(v, path + f"/{i}", field)
        elif isinstance(value, str) and (field in REFERENCE_KINDS or field in SUPPLIED_FIELDS):
            if field in SUPPLIED_FIELDS:
                references.append(dict(path=path, id=value, candidates=[], resolution="supplied"))
                return
            candidates = [
                d["path"]
                for d in declarations
                if d["id"] == value and d["kind"] in REFERENCE_KINDS[field]
            ]
            external = value in supplied and field in {
                "input_object_ids",
                "removed_object_ids",
                "anchor_id",
                "left_id",
                "right_id",
            }
            count = len(candidates) + int(external)
            references.append(
                dict(
                    path=path,
                    id=value,
                    candidates=candidates,
                    resolution="unique" if count == 1 else "unknown" if count == 0 else "ambiguous",
                    external=external,
                )
            )

    walk(generated)
    collisions = []
    for identity in sorted({d["id"] for d in declarations}):
        records = [d for d in declarations if d["id"] == identity]
        if len(records) > 1:
            counts = Counter(d["kind"] for d in records)
            collisions.append(
                dict(
                    id=identity, records=records, same_namespace=any(n > 1 for n in counts.values())
                )
            )
    return dict(
        revision=REVISION,
        source_hash=canonical_sha256(generated),
        declarations=declarations,
        references=references,
        collisions=collisions,
        normalizable=not any(c["same_namespace"] for c in collisions)
        and all(r["resolution"] in {"unique", "supplied"} for r in references),
    )


def normalize_identifiers(generated: Mapping[str, Any], supplied_ids: Sequence[str] = ()):
    """Copy and bijectively translate ONLY resolved record IDs/reference fields.

    Exact duplicate records are not collapsed. Free text is never substituted.
    On ambiguity, fail before returning a partially reconstructed object.
    """
    audit = identifier_audit(generated, supplied_ids)
    if not audit["normalizable"]:
        raise ValueError("identifier normalization would require a semantic choice", audit)
    value = copy.deepcopy(generated)
    canonical_ids = {
        d["path"]: "nC" + canonical_sha256((audit["source_hash"], d["kind"], d["id"]))[:40]
        for d in audit["declarations"]
    }

    def replace(path, new):
        parts = path.strip("/").split("/")
        target = value
        for part in parts[:-1]:
            target = target[int(part)] if isinstance(target, list) else target[part]
        target[int(parts[-1]) if isinstance(target, list) else parts[-1]] = new

    for d in audit["declarations"]:
        replace(d["path"] + "/" + d["field"], canonical_ids[d["path"]])
    for r in audit["references"]:
        if r["candidates"]:
            replace(r["path"], canonical_ids[r["candidates"][0]])
    return value, {
        **audit,
        "canonical_ids_by_record_path": canonical_ids,
        "normalized_hash": canonical_sha256(value),
        "scientific_acceptance": False,
    }


def typed_identifier_schema(schema: Mapping[str, Any]) -> dict:
    """Candidate grammar only; uniqueness/existence still require validation.

    No changes to descriptive/temporal/evidence content or general object limits.
    """
    value = copy.deepcopy(schema)
    definitions = value["$defs"]
    record_names = dict(
        LocalContextSchema="schema",
        LocalTypeDefinition="type",
        LocalPredicateDefinition="predicate",
        Entity="entity",
        Event="event",
        PropositionContent="proposition",
        QualifiedAssertion="assertion",
        OntologyDecision="decision",
    )
    declaration_fields = {kind: field for kind, _, field in RECORDS}

    def rule(kinds):
        return {
            "type": "string",
            "pattern": "^(?:" + "|".join(PREFIXES[k] for k in kinds) + ")[1-9][0-9]{0,3}$",
        }

    def walk(node, kind=None):
        if not isinstance(node, dict):
            return
        for alternative in node.get("anyOf", []):
            walk(alternative, kind)
        for field, child in node.get("properties", {}).items():
            if kind and field == declaration_fields[kind]:
                node["properties"][field] = rule((kind,))
            elif field in REFERENCE_KINDS:
                replacement = rule(REFERENCE_KINDS[field])
                if child.get("type") == "array":
                    # Preserve supplied-candidate alternatives in decision inputs.
                    previous = child.get("items", {})
                    enums = [a for a in previous.get("anyOf", []) if "enum" in a]
                    child["items"] = {"anyOf": [replacement, *enums]} if enums else replacement
                elif any(a.get("type") == "null" for a in child.get("anyOf", [])):
                    node["properties"][field] = {"anyOf": [replacement, {"type": "null"}]}
                else:
                    node["properties"][field] = replacement
            else:
                walk(child)
        if "items" in node:
            walk(node["items"])

    walk(value)
    for name, definition in definitions.items():
        walk(definition, record_names.get(name))
    # Pinned xgrammar ignores min/maxLength alongside a pattern. These TWO
    # canonical role-identifier fields already had that exact 1..192 bound;
    # encode the same language in the regex, not a new descriptive-text limit.
    for role in (
        definitions["RoleBinding"]["properties"]["role"],
        definitions["LocalPredicateDefinition"]["properties"]["role_names"]["items"],
    ):
        if role != {
            "type": "string",
            "pattern": r"^[A-Za-z0-9][A-Za-z0-9_.:/-]*$",
            "minLength": 1,
            "maxLength": 192,
        }:
            raise ValueError("canonical role identifier contract changed; inspect before adapting")
        role.clear()
        role.update(type="string", pattern=r"^[A-Za-z0-9][A-Za-z0-9_.:/-]{0,191}$")
    # Root LocalContextSchema is referenced; binary/nary alternatives keep their
    # record kind through the anyOf traversal above.
    return value


def reconstruct_typed(generated, *, evidence, upper, execution, small=False):
    """Proposed, opt-in diagnostic revision; never called by a live controller.

    Bind the source response, administrative translation and canonical adapter
    separately. All original scientific fields remain in the generated payload.
    """
    from jsonschema import Draft202012Validator

    from story_projection_onto.semantic_generation import (
        AdaptedSemanticDraft,
        reconstruct,
        semantic_schema,
    )

    schema = typed_identifier_schema(semantic_schema(evidence, upper, small=small))
    Draft202012Validator(schema).validate(generated)
    supplied = [
        i
        for e in evidence
        for i in (
            *(m.candidate_id for m in e.mention_candidates),
            *(m.candidate_id for m in e.event_candidates),
            *(m.candidate_id for m in e.relation_phrase_candidates),
            *(m.clue_id for m in e.temporal_clues),
        )
    ]
    translated, receipt = normalize_identifiers(generated, supplied)
    adapted = reconstruct(
        translated, evidence=evidence, upper=upper, execution=execution, small=small
    )
    return AdaptedSemanticDraft(
        adapted.draft,
        {
            **adapted.provenance,
            "interface_revision": REVISION,
            "generated_semantic_hash": canonical_sha256(generated),
            "generation_schema_hash": canonical_sha256(schema),
            "identifier_translation": receipt,
            "translated_payload_provenance": adapted.provenance,
        },
    )


IDENTIFIER_INSTRUCTION = (
    "Use independent typed local counters: nS1 schema, nT1 types, nR1 predicates, "
    "nE1 entities, nV1 events, nP1 proposition contents, nA1 assertions, nD1 decisions. "
    "Increment only within each kind; do not use one global numeric counter. "
    "Every reference must name the exact selected record including its prefix. "
    "Endpoints/role bindings use nE or nV; holders use nE; contextual types/domain/range "
    "use nT; predicates use nR; description support uses nA; epistemic content uses nP. "
    "Decision targets include the record's prefix; inputs may also cite supplied candidate IDs. "
    "Never invent supplied citations. Runtime assigns canonical administrative IDs by a "
    "bijective reference translation; it does not select endpoints, merge records "
    "or fill semantics."
)
