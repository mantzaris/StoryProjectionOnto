"""Prospectively frozen null-aware scoring; historical v1 scores remain immutable."""

from story_projection_onto import compact_story_v2 as protocol

from . import compact_story as old
from .simple_ladder_v2 import citations_valid
from .simple_ladder_v2 import field_issues as old_field_issues

NULLABLE = ("valid_from", "valid_until", "holder", "attitude")


def explicit(row):
    return {**dict.fromkeys(NULLABLE), **row}


def story_reference(story_id):
    if story_id != "museum":
        return [explicit(r) for r in old.story_reference(story_id)]
    rows = [
        ("Pearl Box", "located_in", "Glass Room", dict(holder="Niko", attitude="believes")),
        ("Esme", "owns", "Ruby Medal", {}),
        ("Dario", "located_in", "Upper Court", dict(valid_from=20, valid_until=23)),
        ("Pearl Box", "located_in", "Brick Store", {}),
        ("Niko", "carries", "Ruby Medal", dict(valid_from=19, valid_until=26)),
        ("Onyx Bell", "located_in", "Cedar Gate", dict(valid_from=23, valid_until=27)),
        ("Ruby Medal", "located_in", "Cedar Gate", dict(holder="Esme", attitude="believes")),
        ("Dario", "owns", "Onyx Bell", {}),
        ("Esme", "located_in", "Glass Room", dict(valid_from=21, valid_until=25)),
        ("Ruby Medal", "located_in", "Upper Court", {}),
        ("Niko", "owns", "Pearl Box", {}),
        ("Niko", "located_in", "Brick Store", dict(valid_from=24, valid_until=28)),
        ("Esme", "carries", "Pearl Box", dict(valid_from=20, valid_until=24)),
        ("Dario", "located_in", "Cedar Gate", dict(valid_from=28, valid_until=30)),
    ]
    return [
        explicit(dict(subject=s, relation=r, object=o, evidence_ids=[f"S{i}"], **q))
        for i, (s, r, o, q) in enumerate(rows, 1)
    ]


def reference_for(story_id, task):
    ids = (
        {
            "all": list(range(1, 15)),
            "possession": [2, 5, 8, 11, 13],
            "locations": [3, 6, 9, 12],
            "belief": [1, 4, 7, 10],
        }
        if story_id == "museum"
        else old.TARGET_SENTENCES
    )[task]
    all_facts = story_reference(story_id)
    return [all_facts[i - 1] for i in ids]


def references():
    return {k: [reference_for(c["story_id"], c["task"])] for k, c in protocol.cases().items()}


def semantic_view(row):
    """Mechanical null-to-absence view only; never fill an omitted known value."""
    return (
        {k: v for k, v in row.items() if k not in NULLABLE or v is not None}
        if isinstance(row, dict)
        else row
    )


def field_issues(row):
    issues = old_field_issues(semantic_view(row))
    if isinstance(row, dict):
        issues += ["required explicit field missing: " + k for k in protocol.FIELDS if k not in row]
    return issues


def qualification(row, fields, story_id):
    return tuple(
        (
            row.get(f) is not None,
            protocol.endpoint(row.get(f), story_id) if f == "holder" else row.get(f),
        )
        for f in fields
    )


def full_key(row, story_id):
    t = protocol.triple(row, story_id)
    if (
        t is None
        or old_field_issues(semantic_view(row))
        or not citations_valid(row, protocol.stories()[story_id]["evidence"])
    ):
        return None
    return (*t, tuple(sorted(row["evidence_ids"])), *qualification(row, NULLABLE, story_id))


def evaluate_answer(story_id, task, parsed):
    return old.evaluate_answer(
        story_id,
        task,
        parsed,
        _protocol=protocol,
        _targets=reference_for(story_id, task),
        _source=story_reference(story_id),
        _full_key=full_key,
        _field_issues=field_issues,
        _qualification=qualification,
    )


def evaluate(case_id, parsed):
    case = protocol.cases()[case_id]
    return evaluate_answer(case["story_id"], case["task"], parsed)


def frozen_rules():
    return {
        **old.frozen_rules(),
        "version": "compact-story-v2-explicit-null",
        "qualification_rule": "null and missing nullable values both mean unspecified/not attributed; required-field compliance separately flags missing keys; known reference bounds/holders never inferred or filled",
        "format": "all eight keys required; null only for unspecified bounds and absent attribution",
        "syntax": "strict JSON rejects duplicate keys and non-JSON constants; only trailing commas outside strings removable after complete-document validation; edits separately logged",
        "selection": "same record-only category/overlap/attributed-subject filters; null key presence is NOT attribution; no query clipping",
        "museum_targets": {
            "possession": [2, 5, 8, 11, 13],
            "locations": [3, 6, 9, 12],
            "belief": [1, 4, 7, 10],
        },
        "historical_scoring": "unchanged v1 scorer for separately derived syntax-only recovery; no v2 normalization applied to old scores",
    }
