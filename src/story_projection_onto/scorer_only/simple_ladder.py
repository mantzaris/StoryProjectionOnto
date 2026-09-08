"""Frozen diagnostic references/scoring; never imported by the model request builder."""

import re

RULE_VERSION = "simple-ladder-v1"
SYNONYMS = {
    "carries": ["carry", "carrying"],
    "owns": ["own", "owner_of"],
    "located_in": ["is_in", "in", "located_at", "is_located_in"],
    "holds_office": ["holds", "holder_of", "serves_as"],
    "responsible_for": ["has_responsibility", "is_responsible_for"],
}


def fact(subject, relation, obj, evidence_id, **qualification):
    return dict(
        subject=subject, relation=relation, object=obj, evidence_id=evidence_id, **qualification
    )


def references():
    office = [
        fact("Ada", "holds_office", "Harbor Warden", "S1", valid_from=0, valid_until=3),
        fact("Bram", "holds_office", "Harbor Warden", "S2", valid_from=3, valid_until=6),
        fact("Harbor Warden", "responsible_for", "inspecting boats", "S3"),
        fact("Harbor Warden", "responsible_for", "maintaining beacons", "S4"),
    ]
    direct = office[:2] + [
        fact(person, "responsible_for", duty, [sid, ds], valid_from=a, valid_until=b)
        for person, sid, a, b in (("Ada", "S1", 0, 3), ("Bram", "S2", 3, 6))
        for duty, ds in (("inspecting boats", "S3"), ("maintaining beacons", "S4"))
    ]
    return {
        "1": [
            [
                fact("Mira", "carries", "lantern", "S1"),
                fact("Tomas", "owns", "lantern", "S2"),
                fact("Mira", "located_in", "courtyard", "S3"),
            ]
        ],
        "2": [
            [
                fact("Nadia", "carries", "compass", "S1"),
                fact("Oren", "owns", "compass", "S2"),
                fact("Oren", "located_in", "harbor", "S3"),
            ]
        ],
        "3": [[fact("Ivo", "owns", "key", "S1"), fact("Leda", "owns", "telescope", "S2")]],
        "4": [
            [
                fact("Ivo", "located_in", "observatory", "S3"),
                fact("Leda", "located_in", "garden", "S4"),
            ]
        ],
        "5": [
            [
                fact("Ada", "located_in", "atrium", "S1", valid_from=0, valid_until=2),
                fact("Ada", "located_in", "library", "S2", valid_from=2, valid_until=5),
            ]
        ],
        "6": [
            [
                fact("coin", "located_in", "drawer", "S1", holder="Rina", attitude="believes"),
                fact("coin", "located_in", "chest", "S2"),
            ]
        ],
        "7": [office, direct, office + direct[2:]],
        "8": [office, office + direct[2:]],
    }


def normalize(value):
    return " ".join(value.strip().casefold().split()) if isinstance(value, str) else value


def relation(value):
    value = normalize(value)
    if not isinstance(value, str):
        return value
    value = value.replace(" ", "_")
    return next(
        (
            canonical
            for canonical, alternatives in SYNONYMS.items()
            if value == canonical or value in alternatives
        ),
        value,
    )


def key(row, *, underlying=False):
    if not isinstance(row, dict) or not all(
        isinstance(row.get(k), str) and row[k].strip() for k in ("subject", "relation", "object")
    ):
        return None
    evidence = row.get("evidence_id")
    citations = [evidence] if isinstance(evidence, str) else evidence
    if (
        not isinstance(citations, list)
        or not citations
        or not all(isinstance(s, str) for s in citations)
    ):
        return None
    allowed = {
        "subject",
        "relation",
        "object",
        "evidence_id",
        "valid_from",
        "valid_until",
        "holder",
        "attitude",
    }
    if set(row) - allowed or len(citations) != len(set(citations)):
        return None
    base = (
        normalize(row["subject"]),
        relation(row["relation"]),
        normalize(row["object"]),
        tuple(sorted(citations)),
    )
    if underlying:
        return base
    bounds = tuple(row.get(k, "ABSENT") for k in ("valid_from", "valid_until"))
    if any(
        v != "ABSENT" and (isinstance(v, bool) or not isinstance(v, (int, float))) for v in bounds
    ):
        return None
    return base + bounds + tuple(normalize(row.get(k, "ABSENT")) for k in ("holder", "attitude"))


def counts(predictions, reference, *, underlying=False):
    remaining = list(range(len(reference)))
    correct, incorrect = [], []
    for row in predictions:
        k = key(row, underlying=underlying)
        found = next(
            (
                i
                for i in remaining
                if k is not None and k == key(reference[i], underlying=underlying)
            ),
            None,
        )
        if found is None:
            incorrect.append(row)
        else:
            remaining.remove(found)
            correct.append(row)
    tp, predicted, gold = len(correct), len(predictions), len(reference)
    p, r = tp / predicted if predicted else 0.0, tp / gold if gold else 0.0
    return {
        "correct": correct,
        "incorrect_or_unmatched": incorrect,
        "missing": [reference[i] for i in remaining],
        "true_positive": tp,
        "predicted": predicted,
        "reference_count": gold,
        "precision": p,
        "recall": r,
        "f1": 2 * p * r / (p + r) if p + r else 0.0,
    }


def evaluate(case_id, parsed):
    predictions = parsed["facts"] if parsed is not None else []
    alternatives = references()[case_id]
    # Frozen structural alternatives, not post-output synonym changes. Ties use listed order.
    scored = [counts(predictions, ref) for ref in alternatives]
    index = max(range(len(scored)), key=lambda i: (scored[i]["f1"], scored[i]["recall"]))
    full, base = scored[index], counts(predictions, alternatives[index], underlying=True)
    unresolved = []
    errors = []
    for row in full["incorrect_or_unmatched"]:
        k = key(row)
        if k is None:
            category = "malformed_or_contract_violation"
        elif key(row, underlying=True) in [key(x, underlying=True) for x in alternatives[index]]:
            category = "wrong_qualification_or_duplicate"
        elif relation(row["relation"]) not in SYNONYMS:
            category = "unresolved_relation_wording"
        else:
            category = "wrong_or_unmatched_binding_or_citation"
        (unresolved if category.startswith("unresolved") else errors).append(
            {"fact": row, "category": category}
        )
    return {
        "rule_version": RULE_VERSION,
        "matched_alternative": index,
        "full": full,
        "underlying": base,
        "qualification_correct_given_matched_relationship": full["true_positive"]
        / base["true_positive"]
        if base["true_positive"]
        else None,
        "errors": errors,
        "unresolved": unresolved,
        "diagnostic_level_supported": full["precision"] >= 0.8 and full["recall"] >= 0.8,
    }


def direct_gate(outcomes):
    direct = [o for o in outcomes if o["case_id"] in ("1", "2")]
    if len(direct) != 2:
        return False
    tp = sum(o["evaluation"]["full"]["true_positive"] for o in direct)
    n = sum(o["evaluation"]["full"]["predicted"] for o in direct)
    return n > 0 and tp / n >= 0.8 and tp / 6 >= 0.8


def toy_extract(evidence):
    """Sentence parser, not a stored-answer baseline and not registered C0."""
    result = []
    patterns = [
        (r"(.+) carries (?:a|the) (.+)\.", "carries"),
        (r"(.+) owns (?:a|the) (.+)\.", "owns"),
        (r"(.+) is in the (.+)\.", "located_in"),
    ]
    for sid, sentence in evidence.items():
        for pattern, predicate in patterns:
            match = re.fullmatch(pattern, sentence)
            if match:
                result.append(fact(match[1], predicate, match[2], sid))
                break
    return {"facts": result}
