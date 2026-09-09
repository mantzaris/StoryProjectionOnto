"""Frozen references and evaluation for the compact-story exploratory demonstration."""

from __future__ import annotations

from story_projection_onto import compact_story as protocol

from .simple_ladder_v2 import citations_valid, count_matches, field_issues

RULE_VERSION = "compact-stories-v1-frozen-before-inference"


def story_reference(story_id):
    if story_id == "harbor":
        rows = [
            ("Lena", "owns", "Cobalt Compass", {}),
            ("Omar", "owns", "Amber Lantern", {}),
            ("Priya", "carries", "Cobalt Compass", dict(valid_from=0, valid_until=6)),
            ("Lena", "carries", "Amber Lantern", dict(valid_from=1, valid_until=5)),
            ("Lena", "located_in", "North Hall", dict(valid_from=1, valid_until=5)),
            ("Omar", "located_in", "East Garden", dict(valid_from=0, valid_until=3)),
            ("Priya", "located_in", "Stone Shed", dict(valid_from=3, valid_until=6)),
            ("Silver Flute", "located_in", "West Pier", dict(valid_from=2, valid_until=4)),
            (
                "Cobalt Compass",
                "located_in",
                "North Hall",
                dict(holder="Lena", attitude="believes"),
            ),
            ("Cobalt Compass", "located_in", "East Garden", {}),
            ("Amber Lantern", "located_in", "Stone Shed", dict(holder="Omar", attitude="believes")),
            ("Amber Lantern", "located_in", "West Pier", {}),
            ("Priya", "owns", "Silver Flute", {}),
            ("Lena", "located_in", "Stone Shed", dict(valid_from=7, valid_until=9)),
        ]
    elif story_id == "orchard":
        rows = [
            ("Vera", "owns", "Copper Book", {}),
            ("Soren", "owns", "Ivory Vase", {}),
            ("Jun", "carries", "Copper Book", dict(valid_from=9, valid_until=17)),
            ("Vera", "carries", "Ivory Vase", dict(valid_from=10, valid_until=16)),
            ("Jun", "located_in", "River Room", dict(valid_from=10, valid_until=14)),
            ("Soren", "located_in", "Hill Court", dict(valid_from=14, valid_until=18)),
            ("Vera", "located_in", "South Loft", dict(valid_from=11, valid_until=16)),
            ("Jade Drum", "located_in", "Elm Dock", dict(valid_from=12, valid_until=15)),
            ("Copper Book", "located_in", "South Loft", dict(holder="Soren", attitude="believes")),
            ("Copper Book", "located_in", "River Room", {}),
            ("Ivory Vase", "located_in", "Hill Court", dict(holder="Jun", attitude="believes")),
            ("Ivory Vase", "located_in", "Elm Dock", {}),
            ("Jun", "owns", "Jade Drum", {}),
            ("Jun", "located_in", "South Loft", dict(valid_from=18, valid_until=20)),
        ]
    else:
        raise ValueError("unknown frozen story")
    return [
        dict(subject=s, relation=r, object=o, evidence_ids=[f"S{i}"], **q)
        for i, (s, r, o, q) in enumerate(rows, 1)
    ]


TARGET_SENTENCES = {
    "all": list(range(1, 15)),
    "possession": [1, 2, 3, 4, 13],
    "locations": [5, 6, 7, 8],
    "belief": [9, 10, 11, 12],
}


def reference_for(story_id, task):
    all_facts = story_reference(story_id)
    return [all_facts[i - 1] for i in TARGET_SENTENCES[task]]


def references():
    return {k: [reference_for(c["story_id"], c["task"])] for k, c in protocol.cases().items()}


def frozen_rules():
    return {
        "version": RULE_VERSION,
        "relations": protocol.RELATIONS,
        "inverses": protocol.INVERSES,
        "endpoint_normalization": "case and whitespace; optional leading 'the' only for exact declared names; no inferred aliases",
        "citation_rule": "nonempty distinct supplied string IDs; exact support set for complete-fact match",
        "qualification_rule": "absence is not null/unknown string; exact explicit full bounds (never query clipping); holder and believes only for source attribution",
        "target_sentences": TARGET_SENTENCES,
        "matching": "one-to-one multiset, all predictions in denominators; bare triples ignore qualifications/citations; full keys retain both",
        "error_policy": "complete explicit-only source inventory distinguishes supported irrelevant from unsupported normalized relationships; unknown predicate/endpoint wording is unresolved, not accepted",
        "selection": "CPU record-only category/interval-overlap/attributed-subject filters; unknown intervals excluded from explicit-overlap question; all exclusions traced; no repair or deduplication",
        "progression": "80% precision and recall is descriptive exploratory criterion only; every independent request runs once if healthy and time remains",
        "registered_protocol_changed": False,
    }


def qualification(row, fields, story_id):
    return tuple(
        (f in row, protocol.endpoint(row.get(f), story_id) if f == "holder" else row.get(f))
        for f in fields
    )


def full_key(row, story_id):
    t = protocol.triple(row, story_id)
    if (
        t is None
        or field_issues(row)
        or not citations_valid(row, protocol.stories()[story_id]["evidence"])
    ):
        return None
    return (
        *t,
        tuple(sorted(row["evidence_ids"])),
        *qualification(row, ("valid_from", "valid_until", "holder", "attitude"), story_id),
    )


def evaluate_answer(
    story_id,
    task,
    parsed,
    *,
    _protocol=protocol,
    _targets=None,
    _source=None,
    _full_key=full_key,
    _field_issues=field_issues,
    _qualification=qualification,
):
    protocol, full_key, field_issues, qualification = (
        _protocol,
        _full_key,
        _field_issues,
        _qualification,
    )
    predictions = parsed["facts"] if parsed else []
    targets = reference_for(story_id, task) if _targets is None else _targets
    source = story_reference(story_id) if _source is None else _source
    evidence = protocol.stories()[story_id]["evidence"]
    full = count_matches(predictions, targets, lambda r: full_key(r, story_id))
    bare = count_matches(predictions, targets, lambda r: protocol.triple(r, story_id))
    correct = {i for i, _ in full["matches"]}
    known_names = {protocol.normalize(n) for n in protocol.stories()[story_id]["names"]}
    rows = []
    for i, row in enumerate(predictions):
        t = protocol.triple(row, story_id)
        issues = field_issues(row)
        candidates = [r for r in source if t is not None and protocol.triple(r, story_id) == t]
        cited = (
            [
                r
                for r in candidates
                if citations_valid(row, evidence)
                and set(row["evidence_ids"]) == set(r["evidence_ids"])
            ]
            if isinstance(row, dict)
            else []
        )
        ref = (cited or candidates or [None])[0]
        temporal = epistemic = None
        if ref is not None:
            temporal = qualification(row, ("valid_from", "valid_until"), story_id) == qualification(
                ref, ("valid_from", "valid_until"), story_id
            )
            epistemic = qualification(row, ("holder", "attitude"), story_id) == qualification(
                ref, ("holder", "attitude"), story_id
            )
            if not temporal:
                issues.append("missing, unsupported or wrong explicit temporal qualification")
            if not epistemic:
                issues.append("missing, unsupported or wrong holder/attitude qualification")
            if not cited:
                issues.append("citations do not equal the supporting source sentence set")
        source_supported = full_key(row, story_id) is not None and any(
            full_key(row, story_id) == full_key(r, story_id) for r in source
        )
        if i in correct:
            status = "correct_complete_fact"
        elif source_supported:
            in_target = any(full_key(row, story_id) == full_key(r, story_id) for r in targets)
            status = "duplicate" if in_target else "supported_but_irrelevant"
        elif t is None:
            status = "malformed_record"
        elif t[0] not in known_names or t[2] not in known_names or t[1] not in protocol.RELATIONS:
            status = "unresolved_wording"
        elif ref is not None:
            status = "qualification_or_evidence_error"
        else:
            status = "unsupported_explicit_relationship"
        rows.append(
            {
                "index": i,
                "fact": row,
                "status": status,
                "issues": issues,
                "citation_valid": citations_valid(row, evidence),
                "source_supported": source_supported,
                "temporal_correct": temporal,
                "epistemic_correct": epistemic,
                "source_reference": ref,
            }
        )
    return {
        "parseable": parsed is not None,
        "underlying": bare,
        "full": full,
        "rows": rows,
        "citation_valid": sum(r["citation_valid"] for r in rows),
        "source_supported": sum(r["source_supported"] for r in rows),
        "format_compliant": sum(not field_issues(r) for r in predictions),
        "confirmed_errors": [
            r
            for r in rows
            if r["status"]
            in ("qualification_or_evidence_error", "unsupported_explicit_relationship")
        ],
        "unresolved": [r for r in rows if r["status"] == "unresolved_wording"],
        "registered_acceptance": False,
    }


def evaluate(case_id, parsed):
    case = protocol.cases()[case_id]
    return evaluate_answer(case["story_id"], case["task"], parsed)
