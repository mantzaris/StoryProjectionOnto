"""Prospectively frozen v2 evaluation. Never re-scores original model outputs."""

from __future__ import annotations

import math

from .simple_ladder import SYNONYMS, normalize, relation
from .simple_ladder import references as original_references

RULE_VERSION = "simple-ladder-v2"


def references():
    old = original_references()
    mapping = {"1": "1", "2": "5", "3": "6", "4": "6", "5": "7", "6": "8", "7": "7", "8": "8"}
    changes = {
        "Rina": "Selin",
        "coin": "map",
        "drawer": "cabinet",
        "chest": "satchel",
        "Ada": "Celia",
        "Bram": "Dax",
        "Harbor Warden": "Orchard Steward",
        "inspecting boats": "checking irrigation",
        "maintaining beacons": "recording harvests",
    }
    result = {}
    for k, prior in mapping.items():
        alternatives = []
        for alt in old[prior]:
            rows = []
            for fact in alt:
                row = {field: value for field, value in fact.items() if field != "evidence_id"}
                row["evidence_ids"] = (
                    fact["evidence_id"]
                    if isinstance(fact["evidence_id"], list)
                    else [fact["evidence_id"]]
                )
                if k in ("4", "7", "8"):
                    row = {
                        field: changes.get(value, value) if isinstance(value, str) else value
                        for field, value in row.items()
                    }
                if k in ("7", "8"):
                    row = {
                        field: {0: 2, 3: 5, 6: 9}[value]
                        if field in ("valid_from", "valid_until")
                        else value
                        for field, value in row.items()
                    }
                rows.append(row)
            alternatives.append(rows)
        result[k] = alternatives
    return result


def aliases(case_id):
    """Only listed, unambiguous noun endpoints; no learned or inferred identity merging."""
    nouns = {
        "1": ["lantern", "courtyard"],
        "2": ["atrium", "library"],
        "3": ["coin", "drawer", "chest"],
        "4": ["map", "cabinet", "satchel"],
        "5": ["Harbor Warden"],
        "6": ["Harbor Warden"],
        "7": ["Orchard Steward"],
        "8": ["Orchard Steward"],
    }[case_id]
    result = {
        normalize(article + noun): normalize(noun)
        for noun in nouns
        for article in ("a ", "an ", "the ")
    }
    if case_id in ("5", "6", "7", "8"):
        office = normalize(nouns[0])
        result.update({"office of " + office: office, "the office of " + office: office})
    return result


def endpoint(value, case_id):
    value = normalize(value)
    return aliases(case_id).get(value, value) if isinstance(value, str) else None


def triple(row, case_id):
    if not isinstance(row, dict) or not all(
        isinstance(row.get(f), str) and row[f].strip() for f in ("subject", "relation", "object")
    ):
        return None
    return (
        endpoint(row["subject"], case_id),
        relation(row["relation"]),
        endpoint(row["object"], case_id),
    )


def citations_valid(row, evidence_ids):
    ids = row.get("evidence_ids") if isinstance(row, dict) else None
    return (
        isinstance(ids, list)
        and bool(ids)
        and all(isinstance(s, str) and s in evidence_ids for s in ids)
        and len(ids) == len(set(ids))
    )


def field_issues(row):
    if not isinstance(row, dict):
        return ["record must be an object"]
    errors = []
    for f in ("subject", "relation", "object"):
        if not isinstance(row.get(f), str) or not row[f].strip():
            errors.append(f + " must be a nonempty string")
    ids = row.get("evidence_ids")
    if not isinstance(ids, list) or not ids or not all(isinstance(s, str) for s in ids):
        errors.append("evidence_ids must be a nonempty list of supplied string IDs")
    allowed = {
        "subject",
        "relation",
        "object",
        "evidence_ids",
        "valid_from",
        "valid_until",
        "holder",
        "attitude",
    }
    if set(row) - allowed:
        errors.append("undeclared fields: " + ",".join(sorted(set(row) - allowed)))
    for a, b in (("valid_from", "valid_until"), ("holder", "attitude")):
        if (a in row) != (b in row):
            errors.append(f"{a} and {b} must appear together")
    for f in ("valid_from", "valid_until"):
        if f in row and (
            isinstance(row[f], bool)
            or not isinstance(row[f], (int, float))
            or not math.isfinite(row[f])
        ):
            errors.append(f + " must be a finite number")
    if (
        all(f in row and isinstance(row[f], (int, float)) for f in ("valid_from", "valid_until"))
        and row["valid_from"] >= row["valid_until"]
    ):
        errors.append("interval must have increasing bounds")
    if "holder" in row and (not isinstance(row["holder"], str) or not row["holder"].strip()):
        errors.append("holder must be a nonempty name")
    if "attitude" in row and row["attitude"] != "believes":
        errors.append("this diagnostic's attributed attitude is believes")
    return errors


def qualification(row, fields, case_id):
    values = []
    for f in fields:
        value = endpoint(row.get(f), case_id) if f == "holder" else normalize(row.get(f))
        if (
            f in row
            and f in ("valid_from", "valid_until")
            and (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
            )
        ):
            value = ("invalid_numeric_bound", repr(value))
        values.append((f in row, value))
    return tuple(values)


def full_key(row, case_id, evidence_ids):
    base = triple(row, case_id)
    if base is None or field_issues(row) or not citations_valid(row, evidence_ids):
        return None
    return (
        *base,
        tuple(sorted(row["evidence_ids"])),
        *qualification(row, ("valid_from", "valid_until", "holder", "attitude"), case_id),
    )


def count_matches(predictions, gold, key):
    remaining, matched, unmatched = list(range(len(gold))), [], []
    for index, row in enumerate(predictions):
        value = key(row)
        hit = next((j for j in remaining if value is not None and value == key(gold[j])), None)
        if hit is None:
            unmatched.append(index)
        else:
            remaining.remove(hit)
            matched.append((index, hit))
    tp, n, g = len(matched), len(predictions), len(gold)
    p, r = tp / n if n else 0.0, tp / g if g else 0.0
    return dict(
        true_positive=tp,
        predicted=n,
        reference_count=g,
        precision=p,
        recall=r,
        f1=2 * p * r / (p + r) if p + r else 0.0,
        matches=matched,
        incorrect_or_unmatched=[predictions[i] for i in unmatched],
        missing=[gold[j] for j in remaining],
    )


def evaluate(case_id, parsed):
    from story_projection_onto.simple_ladder_v2 import cases

    case = cases()[case_id]
    predictions = parsed["facts"] if parsed else []
    options = references()[case_id]
    scores = [
        count_matches(predictions, gold, lambda r: full_key(r, case_id, case["evidence"]))
        for gold in options
    ]
    chosen = max(range(len(scores)), key=lambda i: (scores[i]["f1"], scores[i]["recall"]))
    gold, full = options[chosen], scores[chosen]
    base = count_matches(predictions, gold, lambda r: triple(r, case_id))
    mapping = dict(base["matches"])
    complete_indices = {i for i, _ in full["matches"]}
    rows = []
    for i, row in enumerate(predictions):
        matched = gold[mapping[i]] if i in mapping else None
        issues = field_issues(row)
        shape_compliant = not issues
        # Duplicates can have good format despite failing one-to-one fact scoring.
        format_reference = next(
            (
                g
                for g in gold
                if triple(row, case_id) is not None and triple(row, case_id) == triple(g, case_id)
            ),
            None,
        )
        temporal, epistemic = None, None
        if matched is not None:
            for fields, label in (
                (("valid_from", "valid_until"), "temporal"),
                (("holder", "attitude"), "epistemic"),
            ):
                same = qualification(row, fields, case_id) == qualification(
                    matched, fields, case_id
                )
                if label == "temporal":
                    temporal = same
                else:
                    epistemic = same
        if format_reference is not None:
            for fields, label in (
                (("valid_from", "valid_until"), "temporal"),
                (("holder", "attitude"), "epistemic"),
            ):
                for f in fields:
                    if f in format_reference and f not in row:
                        issues.append("required " + f + " missing for matched relationship")
                if any(f in row for f in fields) and not any(f in format_reference for f in fields):
                    issues.append("unsupported " + label + " fields on this relationship")
        status = "correct_complete_fact" if i in complete_indices else "unmatched"
        meaning = []
        t = triple(row, case_id)
        if matched is not None:
            if temporal is False:
                meaning.append("temporal bounds missing, changed or unsupported")
            if epistemic is False:
                meaning.append("epistemic holder/attitude missing, changed or unsupported")
            if not citations_valid(row, case["evidence"]):
                meaning.append("invalid supplied-evidence reference")
            elif sorted(row["evidence_ids"]) != sorted(matched["evidence_ids"]):
                meaning.append("citations do not exactly support this frozen fact")
        elif t is not None:
            known = {
                endpoint(x[f], case_id)
                for alt in options
                for x in alt
                for f in ("subject", "object")
            }
            if t[1] not in SYNONYMS or t[0] not in known or t[2] not in known:
                status = "unresolved_wording_or_unsupported_endpoint"
            else:
                status = "unmatched_known_binding_or_duplicate"
            # Neither an unmatched expression nor absence of contradiction establishes support.
        rows.append(
            dict(
                index=i,
                fact=row,
                format_issues=issues,
                shape_compliant=shape_compliant,
                conditional_format_resolved=format_reference is not None,
                citation_valid=citations_valid(row, case["evidence"]),
                underlying_correct=matched is not None,
                temporal_correct=temporal,
                epistemic_correct=epistemic,
                complete_correct=i in complete_indices,
                assessment=status,
                meaning_issues=meaning,
            )
        )
    components = {}
    for label, fields in (
        ("temporal", ("valid_from", "valid_until")),
        ("epistemic", ("holder", "attitude")),
    ):
        target_count = sum(any(f in row for f in fields) for row in gold)
        applicable, correct, absence, absence_correct = 0, 0, 0, 0
        for r in rows:
            expected = gold[mapping[r["index"]]] if r["index"] in mapping else None
            row = r["fact"] if isinstance(r["fact"], dict) else {}
            if any(f in row for f in fields) or (
                expected is not None and any(f in expected for f in fields)
            ):
                applicable += 1
                correct += int(r[label + "_correct"] is True)
            elif expected is not None:
                absence += 1
                absence_correct += int(r[label + "_correct"] is True)
        components[label] = dict(
            correct=correct,
            applicable_predictions=applicable,
            target_count=target_count,
            precision=correct / applicable if applicable else None,
            target_recall=correct / target_count if target_count else None,
            correct_absence=absence_correct,
            assessed_absence=absence,
        )
    n = len(predictions)
    return dict(
        rule_version=RULE_VERSION,
        matched_alternative=chosen,
        full=full,
        underlying=base,
        parseable=parsed is not None,
        format_compliant=sum(
            not r["format_issues"] and r["conditional_format_resolved"] for r in rows
        ),
        format_unresolved=sum(
            not r["format_issues"] and not r["conditional_format_resolved"] for r in rows
        ),
        shape_compliant=sum(r["shape_compliant"] for r in rows),
        citation_valid=sum(r["citation_valid"] for r in rows),
        prediction_count=n,
        qualifications=components,
        records=rows,
        diagnostic_level_supported=full["precision"] >= 0.8 and full["recall"] >= 0.8,
    )


def frozen_rules():
    return {
        "version": RULE_VERSION,
        "synonyms": SYNONYMS,
        "endpoint_aliases": {k: aliases(k) for k in references()},
        "order": "all eight frozen cases once; semantic failure does not alter later cases",
        "criterion": "exploratory complete-fact precision and recall >=0.8; no registered acceptance",
        "full_matching": "all fields, exact evidence set, explicit qualification presence; all predictions retained; duplicates penalized",
        "underlying": "subject/relation/object only, independent of citation/qualification/format errors",
        "format": "basic shape separately; conditional required-field compliance only when a reference relationship resolves; unresolved conditional requirements are not passes",
        "qualifications": "separate applicable-prediction accuracy, target recall and correct absence; unmatched expressions unresolved, not passes",
        "office_alternatives": "same mediated or mixed organization permitted for both questions; person question additionally permits direct-person responsibilities; max frozen-alternative F1 then recall",
        "contrast_limit": "pairs cannot force a construction contrast because identical office-mediated graphs are allowed",
    }


def office_comparison(first, second, case_ids):
    def organization(parsed, case_id):
        facts = parsed["facts"] if parsed else []
        edges = {triple(row, case_id) for row in facts} - {None}
        office = "harbor warden" if case_id in ("5", "6") else "orchard steward"
        people = {"ada", "bram"} if case_id in ("5", "6") else {"celia", "dax"}
        office_links = any(s == office and p == "responsible_for" for s, p, _ in edges)
        person_links = any(s in people and p == "responsible_for" for s, p, _ in edges)
        kind = (
            "mixed"
            if office_links and person_links
            else "office-mediated"
            if office_links
            else "person-direct"
            if person_links
            else "unresolved"
        )
        return kind, edges

    a, ae = organization(first, case_ids[0])
    b, be = organization(second, case_ids[1])
    verdict = (
        "unresolved"
        if "unresolved" in (a, b)
        else "same graph organization and bindings"
        if ae == be
        else "different fact selection within same organization"
        if a == b
        else "different semantic organization"
    )
    return dict(
        first_organization=a,
        second_organization=b,
        verdict=verdict,
        only_first=sorted(ae - be),
        only_second=sorted(be - ae),
        limitations="Ignores cosmetic order, allowed aliases and qualification-only changes. Identical office-mediated answers are permitted by both frozen references; not a required construction contrast or efficacy test.",
    )
