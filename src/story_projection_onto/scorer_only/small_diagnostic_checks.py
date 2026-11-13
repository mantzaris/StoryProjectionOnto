"""Separate diagnostic observations; never imported by model-facing packing.

Retain the original pilot audit unchanged. In particular, expose (do not repair)
its historical clock assumptions when the supplied small task lacks coordinates.
These checks do not claim exhaustive natural-language description entailment.
"""

from dataclasses import asdict

from story_projection_onto.contracts import SUBSTANTIVE_CONSTRUCTION_OPERATORS
from story_projection_onto.scorer_only.acceptance_grounding import (
    _KNOWN_FACTS,
    _MENTION_CONCEPTS,
    _assertion_fact,
    _event_concept,
    audit_acceptance_semantic_grounding,
)
from story_projection_onto.validate import validate_draft_structure


def reconciled_component_audit(draft, fixture, oracle_evidence):
    """Opt-in v3 diagnostic only. Retain legacy observations, not their clock gate."""
    from story_projection_onto.scorer_only.small_semantic_rules import audit_small_semantics

    legacy = component_audit(draft, fixture, oracle_evidence)
    corrected = audit_small_semantics(draft, fixture)
    structure_ok = (
        legacy["structural_valid"]
        and legacy["reference_integrity"]
        and legacy["required_nonempty_structure"]
        and legacy["event_connectivity"]
    )
    repairable = list(legacy["repairable_contract_diagnostics"])
    for assertion_id, checks in corrected["assertion_checks"].items():
        if checks["endpoints_roles"]["status"] == "rejected":
            repairable.append(
                {
                    "code": "predicate_role_mismatch",
                    "path": f"instance_graph.assertions.{assertion_id}.predicate_id",
                }
            )
        if any(
            checks[axis]["reason"] == "no supplied coordinate or intrinsic-duration witness"
            for axis in ("story_time", "intrinsic_validity")
        ):
            repairable.append(
                {
                    "code": "unsupported_temporal_precision",
                    "path": f"instance_graph.assertions.{assertion_id}.temporal_scope",
                }
            )
    return {
        **legacy,
        "revision": corrected["revision"],
        "legacy_observation_all_checks_pass": legacy["all_checks_pass"],
        "reconciled_science": corrected,
        "overall_status": corrected["overall_status"] if structure_ok else "rejected",
        "all_checks_pass": structure_ok and corrected["all_supported"],
        "repairable_contract_diagnostics": repairable,
        "scope": "authorized small-pilot rule reconciliation only; no study metric/gold change",
    }


def component_audit(draft, fixture, oracle_evidence):
    structural = validate_draft_structure(
        draft=draft,
        upper_ontology=fixture.upper_ontology,
        evidence=fixture.evidence,
        horizon=fixture.sealed_horizon,
        budgets=fixture.budgets,
        capabilities=fixture.capabilities,
    )
    errors = [d.model_dump(mode="json") for d in structural.diagnostics if d.severity == "error"]
    graph = draft.instance_graph
    predicates = {p.predicate_id: p for p in draft.local_schema.predicates}
    referents = {
        e.entity_id: frozenset(
            _MENTION_CONCEPTS[m]
            for m in e.supported_mention_candidate_ids
            if m in _MENTION_CONCEPTS
        )
        for e in graph.entities
    }
    for event in graph.events:
        concept, _ = _event_concept(
            label=event.label,
            description=event.description,
            reification_reason=event.reification_reason,
            evidence_ids=event.evidence_ids,
        )
        referents[event.event_id] = frozenset([concept]) if concept else frozenset()
    endpoint_checks = []
    for assertion in graph.assertions:
        predicate = predicates.get(assertion.predicate_id)
        fact = None
        reason = "unknown predicate"
        if predicate:
            fact, _, reason = _assertion_fact(
                assertion,
                predicate_parent=predicate.parent_upper_relation,
                predicate_text=predicate.label + " " + predicate.definition,
                referent_concepts=referents,
            )
        endpoint_checks.append(
            dict(
                assertion_id=assertion.assertion_id,
                endpoint_relation_citation_support="supported" if fact is not None else "unknown",
                matched_fact=None if fact is None else fact.fact_id,
                qualification_or_matching_reason=reason,
            )
        )
    footprints = {
        a.assertion_id: {a.subject_id, a.object_id, *(r.object_id for r in a.roles)} - {None}
        for a in graph.assertions
    }
    description_links = [
        dict(
            node_id=n.entity_id if hasattr(n, "entity_id") else n.event_id,
            support_assertions=list(n.description_assertion_ids),
            all_support_involves_node=all(
                (n.entity_id if hasattr(n, "entity_id") else n.event_id) in footprints.get(a, set())
                for a in n.description_assertion_ids
            ),
        )
        for n in (*graph.entities, *graph.events)
    ]
    disconnected = [
        e.event_id for e in graph.events if not any(e.event_id in f for f in footprints.values())
    ]
    # These TWO frozen passages supply neutral labels, not a numeric story clock
    # or intrinsic duration. This is a diagnostic support check, not a gold edit.
    temporal_issues = []
    if {e.evidence_id for e in fixture.evidence} <= {"ev-01", "ev-03"}:

        def inspect(value, path=""):
            if isinstance(value, dict):
                if value.get("kind") in {"point", "interval"}:
                    temporal_issues.append(
                        dict(
                            path=path,
                            value=value,
                            reason="supplied small evidence has no explicit "
                            "numeric coordinate or duration",
                        )
                    )
                for k, v in value.items():
                    inspect(v, path + "/" + k)
            elif isinstance(value, list):
                for i, v in enumerate(value):
                    inspect(v, path + "/" + str(i))

        inspect(draft.model_dump(mode="json"))
    try:
        audit = audit_acceptance_semantic_grounding(draft=draft, evidence=oracle_evidence)
        assessments = [asdict(a) for a in audit.assessments]
        legacy_complete = audit.complete
        audit_error = None
    except (KeyError, ValueError) as error:
        assessments, legacy_complete, audit_error = [], False, str(error)
    # Use the authoritative existing operator set. Include/exclude and rarity
    # checks must not accidentally certify substantive construction.
    nonselection = [d for d in draft.decisions if d.operator in SUBSTANTIVE_CONSTRUCTION_OPERATORS]
    decisions_supported = (
        bool(nonselection)
        and all(
            a["status"] == "supported"
            for a in assessments
            if a["record_kind"] == "ontology_decision"
        )
        and any(a["record_kind"] == "ontology_decision" for a in assessments)
    )
    reference_ok = not any(d["code"] == "unknown_reference" for d in errors)
    nonempty_ok = (
        2 <= len(graph.entities) + len(graph.events) <= 4 and 1 <= len(graph.assertions) <= 3
    )
    structure_ok = structural.validation_status == "accepted"
    # Fact-free violations of existing contract requirements, never expected
    # facts from _KNOWN_FACTS. Whitelisted codes may support a bounded repair.
    repairable = []
    if not nonselection:
        repairable.append(dict(code="substantive_decision_missing", path="decisions"))
    types = {t.type_id: t for t in draft.local_schema.contextual_types}
    for event in graph.events:
        typ = types.get(event.contextual_type_id)
        if typ is not None and typ.parent_upper_type != "event":
            repairable.append(
                dict(
                    code="event_type_incompatible",
                    path=f"instance_graph.events.{event.event_id}.contextual_type_id",
                )
            )
    endpoint_support = (
        "supported"
        if endpoint_checks
        and all(x["endpoint_relation_citation_support"] == "supported" for x in endpoint_checks)
        else "unknown"
    )
    return dict(
        revision="small-component-observations-v2",
        reference_integrity=reference_ok,
        structural_valid=structure_ok,
        required_nonempty_structure=nonempty_ok,
        substantive_operation_reported=bool(nonselection),
        materialized_objects=dict(
            types=len(draft.local_schema.contextual_types),
            predicates=len(predicates),
            entities=len(graph.entities),
            events=len(graph.events),
        ),
        structural_diagnostics=errors,
        endpoint_checks=endpoint_checks,
        endpoint_support=endpoint_support,
        endpoint_support_scope="legacy witness coverage only; unmatched is unknown, not false; "
        "definition/role consistency requires separate assessment",
        description_support_links=description_links,
        description_text_entailment="requires readable-output diagnostic review; "
        "links alone are not entailment",
        temporal_support_issues=temporal_issues,
        evidence_supported_temporal_bounds=not temporal_issues,
        disconnected_event_ids=disconnected,
        event_connectivity=not disconnected,
        construction_decision_grounding=decisions_supported,
        legacy_assessments=assessments,
        legacy_audit_complete=legacy_complete,
        legacy_audit_error=audit_error,
        repairable_contract_diagnostics=repairable,
        legacy_clock_assumptions={
            f.fact_id: f.story_point
            for f in _KNOWN_FACTS
            if f.required_evidence <= {e.evidence_id for e in fixture.evidence}
        },
        all_checks_pass=structure_ok
        and nonempty_ok
        and reference_ok
        and legacy_complete
        and not temporal_issues
        and not disconnected
        and decisions_supported,
        scope="small diagnostics only; no production acceptance or scorer amendment",
    )
