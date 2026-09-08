"""Frozen, conservative rules for TWO development diagnostics, never study scoring.

Positive support requires an explicit source witness and a recognized meaning.
Unrecognized paraphrases/definitions remain unresolved. This is deliberately not
a general natural-language entailment engine. Nothing here is model-visible.
"""

from __future__ import annotations

import re

from story_projection_onto.contracts import SUBSTANTIVE_CONSTRUCTION_OPERATORS
from story_projection_onto.scorer_only.acceptance_grounding import (
    _EXPECTED_EVIDENCE_TEXT,
    _MENTION_CONCEPTS,
)

REVISION = "small-source-bound-semantics-v3"
SUPPORTED, REJECTED, UNRESOLVED = "supported", "rejected", "unresolved"

# Source-bound alternatives, authored before new model responses. No hidden clock.
WITNESSES = {
    "arrival": ("ev-01", {"agent": "courier_lio", "location": "north_gate", "theme": "seal"}),
    "carrying": ("ev-01", {"agent": "courier_lio", "theme": "seal"}),
    "repair": ("ev-03", {"agent": "mechanic_ash", "patient": "river_pump"}),
    "remaining": ("ev-03", {"agent": "courier_lio", "location": "north_gate"}),
}
CONCEPT_TYPES = {
    "courier_lio": "person",
    "mechanic_ash": "person",
    "north_gate": "place",
    "seal": "entity",
    "river_pump": "entity",
}
NAMES = {
    "courier_lio": r"(?:courier )?lio",
    "mechanic_ash": r"(?:mechanic )?ash",
    "north_gate": r"(?:the )?north gate",
    "seal": r"(?:a |the )?(?:copper )?seal",
    "river_pump": r"(?:a |the )?(?:river )?pump",
}
ROLE_WORDS = {
    "agent": {"agent", "actor", "arriver", "courier", "carrier", "repairer", "mechanic", "person"},
    "location": {"location", "place", "destination", "site", "gate"},
    "theme": {"theme", "carried", "cargo", "seal"},
    "patient": {"patient", "repaired", "pump", "target"},
    "event": {"event", "arrival", "repair", "occurrence"},
}


def normalized(text: str) -> str:
    return " ".join(re.findall(r"[a-z0-9]+", text.lower()))


def combined(statuses) -> str:
    values = list(statuses)
    if REJECTED in values:
        return REJECTED
    return SUPPORTED if values and all(v == SUPPORTED for v in values) else UNRESOLVED


def observation(status: str, reason: str, **details) -> dict:
    return {"status": status, "reason": reason, **details}


def role_meaning(name: str) -> str | None:
    words = set(normalized(name).split())
    # 'arrival_location' and 'repair_agent' denote a participant, not the event.
    matches = [r for r, vocabulary in ROLE_WORDS.items() if r != "event" and words & vocabulary]
    if len(matches) == 1:
        return matches[0]
    if not matches and words & ROLE_WORDS["event"]:
        return "event"
    return None


def family_meaning(text: str) -> str | None:
    words = normalized(text).split()
    stems = {
        "arrival": ("arriv",),
        "carrying": ("carry", "carri"),
        "repair": ("repair",),
        "remaining": ("remain", "stay"),
    }
    families = [f for f, roots in stems.items() if any(w.startswith(roots) for w in words)]
    return families[0] if len(families) == 1 else None


def schema_description_support(text: str, *, predicate: bool) -> dict:
    """Whole-definition positive coverage, never a bag-of-words truth decision.

    Source roles/identity/upper-parent checks must ALSO pass. Other definitions
    remain unresolved, not false. This is not a production entailment engine.
    """
    value = normalized(text)
    noun = (
        r"(?:person|actor|courier|mechanic|place|gate|location|object|artifact|"
        r"seal|pump|event|arrival|repair)"
    )
    if not predicate:
        pattern = r"(?:a |an |the )?" + noun + r"(?: (?:entity|event|object|person|place|type))?"
        supported = re.fullmatch(pattern, value)
    else:
        patterns = [
            r"(?:a |an |the )?(?:actor|person)(?: s)? (?:observed )?arrival at (?:a |the )?place",
            r"(?:the )?arrival of (?:a |the )?person at (?:a |the )?place",
            r"(?:a |the )?(?:person|actor|place|object) (?:acting|serving) as "
            r"(?:agent|location|patient|theme) (?:in|of) (?:an? |the )?(?:arrival|repair) event",
            r"(?:arrival|repair) event with (?:explicit )?(?:agent|location|patient|theme)"
            r"(?: and (?:agent|location|patient|theme))+ roles",
            r"(?:an? |the )?(?:actor|person|carrier|mechanic) "
            r"(?:carries|carrying|repairs|repaired) (?:an? |the )?object",
            r"(?:an? |the )?(?:actor|person) (?:remains|remained|stays) at (?:a |the )?place",
        ]
        supported = any(re.fullmatch(pattern, value) for pattern in patterns)
    return observation(
        SUPPORTED if supported else UNRESOLVED,
        "recognized whole definition; positive source/role checks still required"
        if supported
        else "local definition outside frozen entailment coverage",
    )


def text_support(text: str, cited: set[str], *, focus: str | None = None) -> dict:
    """Positive source-span/simple-clause coverage, NOT bag-of-words entailment.

    Every clause must match a source span or a frozen ordered paraphrase. Unknown
    embellishments, negation and reversed arguments never receive positive credit.
    Focus constrains node text to its own referent. Assertion-link coverage is a
    SEPARATE check, so unmatched graph edges do not make supported prose false.
    """
    value = normalized(text)
    if not value:
        return observation(UNRESOLVED, "empty or unrecognized description")
    source_match = any(
        re.search(r"\b" + re.escape(value) + r"\b", normalized(_EXPECTED_EVIDENCE_TEXT[e]))
        for e in cited
    )
    mentioned = {c for c, pattern in NAMES.items() if re.search(r"\b" + pattern + r"\b", value)}
    if focus and focus not in mentioned and not focus.startswith("event:"):
        return observation(UNRESOLVED, "description does not establish its node referent")
    if source_match:
        return observation(SUPPORTED, "exact source-grounded phrase", concepts=sorted(mentioned))
    value = re.sub(
        r"^(?:the evidence (?:states|shows) that |this (?:shows|records) that )", "", value
    )
    clauses = re.split(r"\s+(?:and|while)\s+", value)
    atoms = []
    for clause in clauses:
        found = False
        for family, (evidence_id, roles) in WITNESSES.items():
            if evidence_id not in cited:
                continue
            agent = NAMES[roles["agent"]]
            if family in {"arrival", "remaining"}:
                verb = (
                    r"(?:arrived|arrives|arrival) at"
                    if family == "arrival"
                    else r"(?:remained|remains|stayed|stays) at"
                )
                pattern = agent + r" " + verb + r" " + NAMES[roles["location"]]
            elif family == "carrying":
                pattern = agent + r" (?:carried|carrying|carries) " + NAMES[roles["theme"]]
            else:
                pattern = agent + r" (?:repaired|repairs) " + NAMES[roles["patient"]]
            if re.fullmatch(pattern + (r"(?: at dawn)?" if evidence_id == "ev-01" else ""), clause):
                atoms += [family + "." + r for r in roles if r != "theme" or family == "carrying"]
                found = True
                break
        if not found:
            return observation(UNRESOLVED, "clause not covered by frozen positive entailment rules")
    return observation(
        SUPPORTED,
        "ordered source-bound clause",
        atoms=sorted(set(atoms)),
        concepts=sorted(mentioned),
    )


def time_support(value, *, axis: str, cited: set[str]) -> dict:
    if value.kind.value in {"point", "interval"}:
        return observation(
            REJECTED, "no supplied coordinate or intrinsic-duration witness", axis=axis
        )
    if value.kind.value == "unknown":
        label = normalized(value.label or "")
        if label and not (label == "dawn" and "ev-01" in cited and axis == "story"):
            return observation(UNRESOLVED, "time label is not supported for this axis", axis=axis)
        return observation(SUPPORTED, "explicit uncertainty; no invented bounds", axis=axis)
    if value.kind.value == "not_applicable":
        return observation(
            REJECTED, "occurrence/state validity applies; missing time is unknown", axis=axis
        )
    return observation(
        UNRESOLVED, "relative/partial-order meaning needs an explicit assessed witness", axis=axis
    )


def audit_small_semantics(draft, fixture) -> dict:
    """No generation, mutation, relaxed primary metric, or use of held-out gold."""
    evidence = {e.evidence_id: e for e in fixture.evidence}
    if (
        len(evidence) != 1
        or not set(evidence) <= {"ev-01", "ev-03"}
        or any(e.text != _EXPECTED_EVIDENCE_TEXT[k] for k, e in evidence.items())
    ):
        raise ValueError("reconciled small rules require one exact frozen development passage")
    types = {t.type_id: t for t in draft.local_schema.contextual_types}
    predicates = {p.predicate_id: p for p in draft.local_schema.predicates}
    schema_checks = {
        **{
            t.type_id: schema_description_support(t.definition, predicate=False)
            for t in types.values()
        },
        **{
            p.predicate_id: schema_description_support(p.definition, predicate=True)
            for p in predicates.values()
        },
    }
    graph = draft.instance_graph
    nodes = {n.entity_id: n for n in graph.entities} | {n.event_id: n for n in graph.events}
    referents, node_checks, temporal_checks = {}, {}, {}
    candidates = {m.candidate_id for e in evidence.values() for m in e.mention_candidates}
    for node_id, node in nodes.items():
        is_event = hasattr(node, "event_id")
        if is_event:
            family = family_meaning(node.label)
            concept = (
                "event:" + family
                if family in WITNESSES and WITNESSES[family][0] in evidence
                else None
            )
        else:
            meanings = {
                _MENTION_CONCEPTS.get(m)
                for m in node.supported_mention_candidate_ids
                if m in candidates
            }
            concept = next(iter(meanings)) if len(meanings) == 1 else None
        referents[node_id] = concept
        parent = types[node.contextual_type_id].parent_upper_type
        if concept is None:
            check = observation(UNRESOLVED, "no unique source-bound node identity")
        elif (is_event and parent != "event") or (
            not is_event and parent not in {"entity", CONCEPT_TYPES[concept]}
        ):
            check = observation(REJECTED, "node's upper type conflicts with its anchored identity")
        else:
            check = observation(
                SUPPORTED, "source-bound identity and compatible upper type", concept=concept
            )
        node_checks[node_id] = check
        temporal_checks[node_id] = time_support(
            node.occurrence_time if is_event else node.temporal_state,
            axis="story" if is_event else "validity",
            cited=set(node.evidence_ids),
        )

    assertion_checks = {}
    for a in graph.assertions:
        p = predicates[a.predicate_id]
        text = p.label + " " + p.definition
        family = family_meaning(text)
        roles = [role_meaning(r) for r in p.role_names]
        bindings = {}
        representation = None
        if a.roles:
            if a.direction != "forward":
                representation = "invalid_nary_direction"
            else:
                for r in a.roles:
                    role = role_meaning(r.role)
                    if role is None or role in bindings:
                        break
                    bindings[role] = referents.get(r.object_id)
                representation = "explicit_nary_roles" if len(bindings) == len(a.roles) else None
        else:
            ids = [a.subject_id, a.object_id]
            if a.direction == "inverse":
                ids.reverse()
            values = [referents.get(i) for i in ids]
            if len(roles) == 2 and None not in roles and len(set(roles)) == 2:
                bindings = dict(zip(roles, values, strict=True))
                representation = "explicit_binary_roles"
            elif family in WITNESSES:
                second = (
                    "location"
                    if family in {"arrival", "remaining"}
                    else "theme"
                    if family == "carrying"
                    else "patient"
                )
                # A direct subject→object relation is not implicitly an event-role edge.
                bindings = {"agent": values[0], second: values[1]}
                representation = "direct_binary"
        if "event" in bindings and bindings["event"] and bindings["event"].startswith("event:"):
            event_family = bindings["event"].split(":", 1)[1]
            if family is None:
                # Explicit participant-role names plus an explicit event relation,
                # not generic participation with a guessed role.
                if re.search(r"\b(?:event|participat\w*)\b", normalized(text)):
                    family = event_family
            elif family != event_family:
                representation = None
        if representation == "invalid_nary_direction":
            endpoint = observation(REJECTED, "named n-ary roles already define orientation")
        elif representation is None or family not in WITNESSES:
            endpoint = observation(
                UNRESOLVED, "predicate/role definition has no established equivalent witness"
            )
        else:
            witness_evidence, expected_roles = WITNESSES[family]
            expected = {**expected_roles, "event": "event:" + family}
            parent_ok = p.parent_upper_relation in (
                {"participates_in", "located_at"}
                if "event" in bindings
                else {"located_at"}
                if family in {"arrival", "remaining"}
                else {"related_to", "participates_in"}
            )
            # Local text saying 'person at place' cannot be overridden by a generic
            # participates_in parent/role spelling to make an event destination.
            definition = normalized(p.definition)
            definition_conflict = "event" in bindings and bool(
                re.search(r"(?:person|actor) at (?:a |the )?place", definition)
            )
            if (
                not parent_ok
                or definition_conflict
                or any(expected.get(r) != v for r, v in bindings.items())
            ):
                endpoint = observation(
                    REJECTED,
                    "wrong role binding, endpoint kind, or contradictory predicate definition",
                    bindings=bindings,
                )
            elif witness_evidence not in a.evidence_ids or set(a.evidence_ids) != {
                witness_evidence
            }:
                endpoint = observation(
                    REJECTED, "source witness not cited exclusively in supplied passage"
                )
            elif (
                "event" not in bindings
                and not {
                    "agent",
                    "location"
                    if family in {"arrival", "remaining"}
                    else "theme"
                    if family == "carrying"
                    else "patient",
                }
                <= bindings.keys()
            ):
                endpoint = observation(UNRESOLVED, "incomplete direct relation role signature")
            elif "event" in bindings and len(bindings) < 2:
                endpoint = observation(
                    UNRESOLVED, "event binding without explicit participant role"
                )
            else:
                endpoint = observation(
                    SUPPORTED,
                    "explicit source-bound representation",
                    representation=representation,
                    bindings=bindings,
                    atoms=sorted(family + "." + r for r in bindings if r != "event"),
                )
        scope = a.temporal_scope
        story = time_support(scope.story_time, axis="story", cited=set(a.evidence_ids))
        validity = time_support(scope.validity_time, axis="validity", cited=set(a.evidence_ids))
        latest = max(
            (evidence[e] for e in a.evidence_ids), key=lambda e: e.discourse_position.ordering_key
        )
        position_ok = (
            scope.discourse_position == latest.discourse_position
            # These standalone directly narrated clauses reveal their propositions
            # at their supplied passage position; never use it as a story clock.
            and scope.revelation_position.revelation_order
            == latest.discourse_position.passage_order
        )
        attribution_ok = (
            a.epistemic_scope is None
            and a.proposition_content_id is None
            and a.narrative_commitment.value == "world_committed"
        )
        assertion_checks[a.assertion_id] = {
            "endpoints_roles": endpoint,
            "story_time": story,
            "intrinsic_validity": validity,
            "discourse_revelation": observation(
                SUPPORTED if position_ok else REJECTED,
                "exact supplied disclosure positions required",
            ),
            "epistemic": observation(
                SUPPORTED if attribution_ok else REJECTED,
                "these supplied clauses are direct narrative assertions, not holder reports",
            ),
            "description_text": text_support(a.why_matters, set(a.why_matters_evidence_ids)),
        }
    # Labels and prose receive independent positive source support; graph links
    # must ALSO establish the referenced assertion and involve the described node.
    description_checks = {}
    for node_id, node in nodes.items():
        prose = text_support(node.description, set(node.evidence_ids), focus=referents[node_id])
        label = text_support(node.label, set(node.evidence_ids), focus=referents[node_id])
        if hasattr(node, "event_id") and family_meaning(node.label):
            label = observation(SUPPORTED, "source-bound event name")
        linked = [a for a in graph.assertions if a.assertion_id in node.description_assertion_ids]
        links = []
        for a in linked:
            involved = node_id in {a.subject_id, a.object_id, *(r.object_id for r in a.roles)}
            links.append(
                assertion_checks[a.assertion_id]["endpoints_roles"]["status"]
                if involved
                else REJECTED
            )
        linked_concepts = {
            referents.get(i)
            for a in linked
            for i in (a.subject_id, a.object_id, *(r.object_id for r in a.roles))
        }
        text_concepts = set(prose.get("concepts", ())) | set(label.get("concepts", ()))
        mapping = combined(links)
        if mapping == SUPPORTED and not text_concepts <= linked_concepts:
            mapping = UNRESOLVED
        description_checks[node_id] = {
            "text": prose,
            "label": label,
            "assertion_mapping": observation(
                mapping,
                "supporting assertions must involve node and cover described referents; "
                "no unmatched→false prose inference",
            ),
        }
    # A decision must name the actual artifact of the claimed operation. Existence
    # alone is insufficient if its referenced semantics are unresolved/rejected.
    target_status = {k: v["status"] for k, v in node_checks.items()}
    target_status.update(
        {k: combined(x["status"] for x in v.values()) for k, v in assertion_checks.items()}
    )
    for typ in types.values():
        target_status[typ.type_id] = combined(
            [schema_checks[typ.type_id]["status"]]
            + [target_status[k] for k, n in nodes.items() if n.contextual_type_id == typ.type_id]
        )
    for pred in predicates.values():
        target_status[pred.predicate_id] = combined(
            [schema_checks[pred.predicate_id]["status"]]
            + [
                target_status[a.assertion_id]
                for a in graph.assertions
                if a.predicate_id == pred.predicate_id
            ]
        )
    target_status[draft.local_schema.schema_id] = combined(
        target_status[k] for k in [*types, *predicates]
    )
    decision_checks = {}
    for d in draft.decisions:
        op = d.operator.value
        targets = set(d.created_object_ids)
        expected_kind = {
            "contextual_type": set(types),
            "schema_relation": set(predicates) | {draft.local_schema.schema_id},
            "event_reification": {e.event_id for e in graph.events},
            "temporal_qualification": set(assertion_checks),
        }.get(op)
        if d.operator not in SUBSTANTIVE_CONSTRUCTION_OPERATORS:
            decision_checks[d.decision_id] = observation(
                UNRESOLVED, "non-substantive report cannot certify construction"
            )
        elif expected_kind is None:
            decision_checks[d.decision_id] = observation(
                UNRESOLVED, "operation requires prerequisites outside these bounded rules"
            )
        elif not targets or not targets <= expected_kind or d.removed_object_ids:
            decision_checks[d.decision_id] = observation(
                REJECTED, "operation targets do not match what it claims to construct"
            )
        elif not d.input_object_ids or not set(d.evidence_ids) <= evidence.keys():
            decision_checks[d.decision_id] = observation(
                REJECTED, "construction lacks source inputs/citations"
            )
        else:
            decision_checks[d.decision_id] = observation(
                combined(target_status.get(t, UNRESOLVED) for t in targets),
                "operation matches materialized target kind and supported contents",
            )
    components = {
        "schema_description_support": combined(v["status"] for v in schema_checks.values()),
        "node_identity_types": combined(v["status"] for v in node_checks.values()),
        "endpoint_roles": combined(
            v["endpoints_roles"]["status"] for v in assertion_checks.values()
        ),
        "story_time": combined(
            [v["story_time"]["status"] for v in assertion_checks.values()]
            + [
                v["status"]
                for k, v in temporal_checks.items()
                if k in {e.event_id for e in graph.events}
            ]
        ),
        "intrinsic_validity": combined(
            [v["intrinsic_validity"]["status"] for v in assertion_checks.values()]
            + [
                v["status"]
                for k, v in temporal_checks.items()
                if k in {e.entity_id for e in graph.entities}
            ]
        ),
        "description_support": combined(
            [v["description_text"]["status"] for v in assertion_checks.values()]
            + [s["status"] for v in description_checks.values() for s in v.values()]
        ),
        "construction_decisions": combined(v["status"] for v in decision_checks.values()),
        "discourse_revelation_epistemic": combined(
            s["status"]
            for v in assertion_checks.values()
            for k, s in v.items()
            if k in {"discourse_revelation", "epistemic"}
        ),
    }
    if not any(d.operator in SUBSTANTIVE_CONSTRUCTION_OPERATORS for d in draft.decisions):
        components["construction_decisions"] = REJECTED
    if graph.proposition_contents:
        components["proposition_contents"] = UNRESOLVED
    return {
        "revision": REVISION,
        "scope": "two development small diagnostics only; no primary/gold change",
        "node_checks": node_checks,
        "schema_checks": schema_checks,
        "node_temporal_checks": temporal_checks,
        "assertion_checks": assertion_checks,
        "description_checks": description_checks,
        "decision_checks": decision_checks,
        "components": components,
        "overall_status": combined(components.values()),
        "all_supported": combined(components.values()) == SUPPORTED,
        "unknown_is_not_positive_support": True,
    }
