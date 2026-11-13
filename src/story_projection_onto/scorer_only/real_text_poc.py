"""Frozen Codex-authored reference scope and rubric; not independent human review."""

import math

from .. import real_text_poc as p
from .simple_ladder_v2 import citations_valid

FIELDS = (
    "subject",
    "relation",
    "object",
    "evidence_ids",
    "valid_from",
    "valid_until",
    "holder",
    "attitude",
)
ATTITUDES = {
    "believes": ["believes", "believed", "belief"],
    "reported": ["reported", "reports", "says", "said"],
    "promises": ["promises", "promised"],
    "questions": ["questions", "questioned"],
    "considers": ["considers", "considered"],
}
RELATIONS = {
    "runs_over": ["runs_over", "ran_over", "running_over"],
    "awakens": ["awakens", "awakened", "wakes", "woke", "woke_up"],
    "catches": ["catches", "caught", "captures", "captured"],
    "releases": [
        "releases",
        "released",
        "lets_go",
        "let_go",
        "freed",
        "frees",
        "set_free",
        "spares",
        "spared",
    ],
    "bound_by": ["bound_by", "tied_by", "bound_with", "tied_with"],
    "gnaws": ["gnaws", "gnawed", "gnawed_through", "gnaws_through"],
    "would_repay_if_spared": ["would_repay_if_spared", "repay_if_spared", "repays_if_spared"],
    "can_help": ["can_help", "can_benefit", "can_confer_benefits_on"],
    "ridicules_help_from": ["ridicules_help_from", "ridiculed_help_from", "ridiculed_ability_of"],
    "sits_by": ["sits_by", "sat_by", "sitting_by", "sits_beside", "sitting_beside", "sat_beside"],
    "located_on": ["located_on", "on", "sits_on", "sitting_on", "sat_on", "is_on"],
    "reads": ["reads", "read", "reading", "was_reading"],
    "peeps_into": [
        "peeps_into",
        "peeped_into",
        "looks_into",
        "looked_into",
        "inspects",
        "inspected",
    ],
    "runs_past": [
        "runs_past",
        "ran_past",
        "runs_near",
        "ran_near",
        "ran_close_by",
        "runs_close_by",
    ],
    "lacks": ["lacks", "has_no", "contains_no", "does_not_have", "without"],
    "questions_usefulness_of": ["questions_usefulness_of", "wonders_about_use_of"],
    "considers_making": ["considers_making", "considered_making", "thinks_about_making"],
    "is_useful_to": ["is_useful_to", "useful_to"],
    "makes": ["makes", "making"],
    "visits": ["visits", "visited", "called_upon", "calls_upon", "called_on", "calls_on"],
    "converses_with": [
        "converses_with",
        "in_conversation_with",
        "talks_with",
        "talking_with",
        "talks_to",
        "talking_to",
    ],
    "pulls": ["pulls", "pulled", "pulled_into_room", "pulls_into_room"],
    "closes": ["closes", "closed", "shuts", "shut"],
    "rises_from": ["rises_from", "rose_from", "half_rose_from", "half_rises_from"],
    "greets": ["greets", "greeted", "gave_greeting_to", "bobs_greeting_to"],
    "is": ["is", "was", "is_in_state"],
    "partner_of": ["partner_of", "is_partner_of", "has_been_partner_of"],
    "helper_of": ["helper_of", "helps", "helped", "assists", "assisted", "is_helper_of"],
    "expected_useful_in": [
        "expected_useful_in",
        "will_be_useful_in",
        "would_be_useful_in",
        "will_help_with",
    ],
}
ALIASES = {
    "fable": {
        "a lion": "lion",
        "a mouse": "mouse",
        "some hunters": "hunters",
        "strong ropes": "ropes",
        "rope": "ropes",
        "his face": "lion's face",
        "lion’s face": "lion's face",
    },
    "alice": {
        "her sister": "alice's sister",
        "sister": "alice's sister",
        "alice’s sister": "alice's sister",
        "rabbit": "white rabbit",
        "a white rabbit": "white rabbit",
        "daisy chain": "daisy-chain",
        "a daisy-chain": "daisy-chain",
    },
    "holmes": {
        "mr. sherlock holmes": "holmes",
        "sherlock holmes": "holmes",
        "mr. holmes": "holmes",
        "i": "watson",
        "narrator": "watson",
        "dr. watson": "watson",
        "mr. watson": "watson",
        "mr. wilson": "wilson",
        "elderly gentleman": "wilson",
        "stout gentleman": "wilson",
        "red-haired gentleman": "wilson",
        "wilson’s case": "wilson's case",
        "busy": "engaged",
    },
}


def fact(s, r, o, span, holder=None, attitude=None):
    return dict(
        subject=s,
        relation=r,
        object=o,
        evidence_ids=[span],
        valid_from=None,
        valid_until=None,
        holder=holder,
        attitude=attitude,
    )


def concepts(sid):
    if sid == "fable":
        rows = [
            ("Mouse", "runs_over", "Lion's face", "S1"),
            ("Mouse", "awakens", "Lion", "S1"),
            ("Lion", "catches", "Mouse", "S2"),
            ("Lion", "releases", "Mouse", "S3"),
            ("hunters", "catches", "Lion", "S4"),
            ("Lion", "bound_by", "ropes", "S4"),
            ("Mouse", "gnaws", "ropes", "S5"),
            ("Mouse", "releases", "Lion", "S5"),
            ("Mouse", "would_repay_if_spared", "Lion", "S2", "Mouse", "promises"),
            ("Mouse", "can_help", "Lion", "S6", "Mouse", "reported"),
            ("Lion", "ridicules_help_from", "Mouse", "S6", "Mouse", "reported"),
        ]
    elif sid == "alice":
        rows = [
            ("Alice", "sits_by", "Alice's sister", "S1"),
            ("Alice", "located_on", "bank", "S1"),
            ("Alice's sister", "reads", "book", "S1"),
            ("Alice", "peeps_into", "book", "S1"),
            ("White Rabbit", "runs_past", "Alice", "S2"),
            ("book", "lacks", "pictures", "S1"),
            ("book", "lacks", "conversations", "S1"),
            ("Alice", "questions_usefulness_of", "book without pictures or conversations", "S1"),
            ("Alice", "considers_making", "daisy-chain", "S2"),
        ]
    else:
        rows = [
            ("Watson", "visits", "Holmes", "S1"),
            ("Holmes", "converses_with", "Wilson", "S1"),
            ("Holmes", "pulls", "Watson", "S2"),
            ("Holmes", "closes", "door", "S2"),
            ("Wilson", "rises_from", "chair", "S8"),
            ("Wilson", "greets", "Watson", "S8"),
            ("Holmes", "is", "engaged", "S4", "Watson", "believes"),
            ("Holmes", "is", "engaged", "S5", "Holmes", "reported"),
            ("Watson", "partner_of", "Holmes", "S7", "Holmes", "reported"),
            ("Watson", "helper_of", "Holmes", "S7", "Holmes", "reported"),
            ("Watson", "expected_useful_in", "Wilson's case", "S7", "Holmes", "believes"),
        ]
    groups = [[fact(*r)] for r in rows]
    if sid == "alice":
        groups[7].append(
            fact(
                "book without pictures or conversations",
                "is_useful_to",
                "Alice",
                "S1",
                "Alice",
                "questions",
            )
        )
        groups[8].append(fact("Alice", "makes", "daisy-chain", "S2", "Alice", "considers"))
    return groups


def targets(sid, task):
    groups = concepts(sid)
    indices = {
        "fable": {"actions": list(range(8)), "claims": [2, 3, 8, 9, 10]},
        "alice": {"actions": list(range(5)), "claims": list(range(5, 9))},
        "holmes": {"actions": list(range(6)), "claims": list(range(6, 11))},
    }
    return groups if task == "all" else [groups[i] for i in indices[sid][task]]


def reference_for(sid, task):
    return [g[0] for g in targets(sid, task)]


def references():
    return {k: [reference_for(c["story_id"], c["task"])] for k, c in p.cases().items()}


def endpoint(v, sid):
    if not isinstance(v, str):
        return None
    value = p.normalize(v).removeprefix("the ")
    return ALIASES[sid].get(value, value)


def named(value, vocabulary):
    value = p.normalize(value).replace(" ", "_") if isinstance(value, str) else None
    return next((key for key, vs in vocabulary.items() if value in vs), value)


def triple(row, sid):
    if not isinstance(row, dict) or not all(
        isinstance(row.get(k), str) and row[k].strip() for k in FIELDS[:3]
    ):
        return None
    return (
        endpoint(row["subject"], sid),
        named(row["relation"], RELATIONS),
        endpoint(row["object"], sid),
    )


def field_issues(row):
    if not isinstance(row, dict):
        return ["fact must be an object"]
    errors = ["required field missing: " + k for k in FIELDS if k not in row]
    errors += ["unexpected field: " + k for k in row if k not in FIELDS]
    for k in FIELDS[:3]:
        if not isinstance(row.get(k), str) or not row[k].strip():
            errors.append(k + " must be a nonempty string")
    for k in ("valid_from", "valid_until"):
        v = row.get(k)
        if v is not None and (
            isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v)
        ):
            errors.append(k + " must be null or finite numeric")
    if (row.get("holder") is None) != (row.get("attitude") is None):
        errors.append("holder and attitude must be jointly attributed or jointly null")
    if row.get("holder") is not None and (
        not isinstance(row["holder"], str) or not row["holder"].strip()
    ):
        errors.append("holder must be a readable nonempty name")
    if row.get("attitude") is not None and named(row["attitude"], ATTITUDES) not in ATTITUDES:
        errors.append("unsupported attitude label")
    return errors


def qualification(row, sid):
    return (
        row.get("valid_from"),
        row.get("valid_until"),
        endpoint(row.get("holder"), sid),
        named(row.get("attitude"), ATTITUDES),
    )


def equivalent(a, b, sid, complete):
    if triple(a, sid) is None or triple(a, sid) != triple(b, sid):
        return False
    if not complete:
        return True
    return (
        not field_issues(a)
        and citations_valid(a, p.stories()[sid]["evidence"])
        and set(a["evidence_ids"]) == set(b["evidence_ids"])
        and qualification(a, sid) == qualification(b, sid)
    )


def count(predictions, groups, sid, complete):
    # Deterministic augmenting paths avoid hash-seed-dependent tie assignments.
    links = [
        [
            j
            for j, group in enumerate(groups)
            if any(equivalent(pred, r, sid, complete) for r in group)
        ]
        for pred in predictions
    ]
    owner = {}

    def augment(i, seen):
        for j in links[i]:
            if j in seen:
                continue
            seen.add(j)
            if j not in owner or augment(owner[j], seen):
                owner[j] = i
                return True
        return False

    for i in range(len(predictions)):
        augment(i, set())
    matches = sorted((i, j) for j, i in owner.items())
    n, m, tp = len(predictions), len(groups), len(matches)
    precision, recall = tp / n if n else 0, tp / m if m else 0
    return dict(
        predicted=n,
        reference_count=m,
        true_positive=tp,
        precision=precision,
        recall=recall,
        f1=2 * precision * recall / (precision + recall) if precision + recall else 0,
        matches=matches,
        missing=[grp[0] for j, grp in enumerate(groups) if j not in {x[1] for x in matches}],
        incorrect_or_unmatched=[
            pred for i, pred in enumerate(predictions) if i not in {x[0] for x in matches}
        ],
    )


def evaluate_answer(sid, task, parsed):
    predictions = (parsed or {}).get("facts", [])
    groups, source = targets(sid, task), concepts(sid)
    full, bare = count(predictions, groups, sid, True), count(predictions, groups, sid, False)
    matched = {i for i, _ in full["matches"]}
    rows = []
    for i, row in enumerate(predictions):
        complete_source = [r for grp in source for r in grp if equivalent(row, r, sid, True)]
        bare_source = [r for grp in source for r in grp if equivalent(row, r, sid, False)]
        ref = next(
            (
                r
                for r in bare_source
                if isinstance(row, dict) and row.get("evidence_ids") == r["evidence_ids"]
            ),
            (bare_source or [None])[0],
        )
        status = (
            "correct_complete_fact"
            if i in matched
            else "supported_but_irrelevant"
            if complete_source
            and not any(equivalent(row, r, sid, True) for grp in groups for r in grp)
            else "duplicate"
            if complete_source
            else "qualification_or_evidence_error"
            if bare_source
            else "unresolved_wording"
        )
        issues = field_issues(row)
        temporal = (
            (row.get("valid_from"), row.get("valid_until")) == (None, None)
            if isinstance(row, dict)
            else False
        )
        if not temporal:
            issues.append(
                "unsupported numeric intrinsic validity: excerpt supplies no numeric bounds"
            )
        if status == "unresolved_wording":
            issues.append(
                "not matched by frozen annotations; absence is not proof of falsity; review under frozen rubric"
            )
        if status == "qualification_or_evidence_error":
            issues.append(
                "relationship matched but required attribution, evidence or fields did not"
            )
        rows.append(
            dict(
                index=i,
                fact=row,
                status=status,
                issues=issues,
                citation_valid=citations_valid(row, p.stories()[sid]["evidence"]),
                source_supported=bool(complete_source),
                source_reference=ref,
                temporal_correct=temporal,
                epistemic_correct=qualification(row, sid)[2:] == qualification(ref, sid)[2:]
                if ref
                else None,
            )
        )
    return dict(
        parseable=parsed is not None,
        full=full,
        underlying=bare,
        rows=rows,
        citation_valid=sum(r["citation_valid"] for r in rows),
        source_supported=sum(r["source_supported"] for r in rows),
        format_compliant=sum(not field_issues(r) for r in predictions),
        confirmed_errors=[
            r
            for r in rows
            if r["status"] == "qualification_or_evidence_error" or not r["temporal_correct"]
        ],
        unresolved=[r for r in rows if r["status"] == "unresolved_wording"],
        registered_acceptance=False,
    )


def evaluate(case_id, parsed):
    c = p.cases()[case_id]
    return evaluate_answer(c["story_id"], c["task"], parsed)


def frozen_rules():
    return dict(
        version="published-prose-poc-v1",
        author="Codex; not independent human review",
        aliases=ALIASES,
        relations=RELATIONS,
        attitudes=ATTITUDES,
        alternatives={s: concepts(s) for s in p.stories()},
        selection=p.SELECTION_TERMS,
        scope="Question-target references, not exhaustive annotation of every possible assertion. Query-blind recall is against their union. Strict precision is scoped reference-match precision, not a factual-truth rate.",
        strict="One-to-one maximum matching over frozen alternatives; all predictions in denominator. Exact cited span set, normalized endpoints/predicate, all eight fields and correct qualifiers required.",
        temporal="No numeric intrinsic validity in these excerpts. Once, shortly after, one day in autumn, past cases, and future modality are not numeric intervals. Compact null loses narrative order/duration; report source phrases.",
        ambiguity="Alice rhetorical usefulness question does not establish uselessness. Daisy-chain consideration is not a decision. Holmes addresses Wilson while describing Watson as this gentleman; past collaboration remains Holmes-reported. Wilson greeting target is a contextual pronoun inference; disclose sensitivity.",
        manual_rubric={
            "authorship": "Every manual judgment is by Codex after output, using this frozen rubric, not independent review; response-hash-bound and same for both methods.",
            "relationship": "1 only when participants/direction/predicate meaning are explicit or entailed by the supplied text; 0 for contradiction or invented fact; unresolved for ambiguous support. No credit from lack of contradiction.",
            "qualification": "Full semantic credit requires source-supported modality/holder, no belief as world truth, and no invented numerical validity. Sentence-valued believed proposition can retain meaning if believer, proposition and attitude are explicit, while still failing strict structure.",
            "support": "Citations must actually support the claim; valid ID alone is not support. Relevant supported alternative absent from references is a coverage limitation, not falsehood.",
            "relevance": "Evaluate the whole claim against the frozen question; supported extras stay in precision denominator. Relevance 1,0,unresolved separately from factual support.",
            "coverage": "Map to frozen concept indices only if semantically equivalent. Recall counts unique covered targets. Preserve unmatched supported relevant claims as reference-coverage flags; do not add targets after results.",
            "reporting": "Report relationship and qualified semantic support counts and unresolved denominators, not just perfect-output acceptance. Any manual semantic F1 uses all predictions and the original reference count.",
            "component_credit": "Report supported/partial/unsupported/contradicted/unresolved records, qualification and relevance separately. A composite record may explicitly cover multiple reference concepts, but receives at most one supported-record credit. Semantic coverage is unique explicit supported concept IDs, with the original target denominator. No semantic F1 is required; strict F1 remains untouched.",
        },
    )
