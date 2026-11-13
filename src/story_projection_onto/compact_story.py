"""Two frozen compact development stories; no reference answers in requests."""

from __future__ import annotations

import math
from dataclasses import replace
from types import SimpleNamespace

from .simple_ladder import prepare as prepare_simple
from .simple_ladder_v2 import FORMAT

BASELINE = 10070.219657
ALLOWANCE = 900.0
BLOCK_ID = "compact-stories-20260909"
CONFIG = {
    "block_id": BLOCK_ID,
    "historical_actual_seconds": BASELINE,
    "maximum_additional_seconds": ALLOWANCE,
    "global_maximum_seconds": BASELINE + ALLOWANCE,
    "maximum_new_starts": 1,
    "maximum_new_attempts": 8,
    "startup_seconds": 300,
    "live_checks_seconds": 15,
    "generation_seconds": 60,
    "validation_seconds": 10,
    "shutdown_seconds": 60,
    "guard_seconds": 5,
    "ordinary_execution_authorized": False,
    "complete_forecast_exception": "exploratory-development-only",
}


def stories():
    return {
        "harbor": {
            "title": "The harbor supplies",
            "names": [
                "Lena",
                "Omar",
                "Priya",
                "Cobalt Compass",
                "Amber Lantern",
                "Silver Flute",
                "North Hall",
                "East Garden",
                "Stone Shed",
                "West Pier",
            ],
            "window": [2, 4],
            "evidence": dict(
                zip(
                    (f"S{i}" for i in range(1, 15)),
                    [
                        "Lena owns Cobalt Compass.",
                        "Omar owns Amber Lantern.",
                        "Priya carries Cobalt Compass from day 0 up to but not including day 6.",
                        "Lena carries Amber Lantern from day 1 up to but not including day 5.",
                        "Lena is in North Hall from day 1 up to but not including day 5.",
                        "Omar is in East Garden from day 0 up to but not including day 3.",
                        "Priya is in Stone Shed from day 3 up to but not including day 6.",
                        "Silver Flute is in West Pier from day 2 up to but not including day 4.",
                        "Lena believes that Cobalt Compass is in North Hall.",
                        "In reality, Cobalt Compass is in East Garden.",
                        "Omar believes that Amber Lantern is in Stone Shed.",
                        "In reality, Amber Lantern is in West Pier.",
                        "Priya owns Silver Flute.",
                        "Lena is in Stone Shed from day 7 up to but not including day 9.",
                    ],
                    strict=True,
                )
            ),
        },
        "orchard": {
            "title": "The orchard delivery",
            "names": [
                "Jun",
                "Vera",
                "Soren",
                "Copper Book",
                "Ivory Vase",
                "Jade Drum",
                "River Room",
                "Hill Court",
                "South Loft",
                "Elm Dock",
            ],
            "window": [12, 15],
            "evidence": dict(
                zip(
                    (f"S{i}" for i in range(1, 15)),
                    [
                        "Vera owns Copper Book.",
                        "Soren owns Ivory Vase.",
                        "Jun carries Copper Book from day 9 up to but not including day 17.",
                        "Vera carries Ivory Vase from day 10 up to but not including day 16.",
                        "Jun is in River Room from day 10 up to but not including day 14.",
                        "Soren is in Hill Court from day 14 up to but not including day 18.",
                        "Vera is in South Loft from day 11 up to but not including day 16.",
                        "Jade Drum is in Elm Dock from day 12 up to but not including day 15.",
                        "Soren believes that Copper Book is in South Loft.",
                        "In reality, Copper Book is in River Room.",
                        "Jun believes that Ivory Vase is in Hill Court.",
                        "In reality, Ivory Vase is in Elm Dock.",
                        "Jun owns Jade Drum.",
                        "Jun is in South Loft from day 18 up to but not including day 20.",
                    ],
                    strict=True,
                )
            ),
        },
    }


def questions(story):
    start, end = story["window"]
    return {
        "possession": "Who owns or carries what? Return all and only directly narrated ownership and carrying relationships, retaining every stated interval.",
        "locations": f"Which directly narrated location relationships have an explicitly stated interval overlapping day {start} up to but not including day {end}? Return all of them with their full evidence-stated intervals, not clipped to the question window. Exclude undated locations and beliefs.",
        "belief": "Who explicitly believes what, and what reality is separately narrated about those same objects? Return each underlying believed relationship with its attribution and each matching narrated reality without attribution. Omit unrelated facts.",
    }


def cases():
    result = {}
    sources = stories()
    # Both pre-extractions complete or terminally fail before any contextual transmission.
    order = [(s, "all") for s in sources] + [
        (s, task) for s in sources for task in ("possession", "locations", "belief")
    ]
    for i, (story_id, task) in enumerate(order, 1):
        story = sources[story_id]
        result[str(i)] = {
            "case_id": str(i),
            "story_id": story_id,
            "task": task,
            "level": "query-blind" if task == "all" else "contextual",
            "evidence": story["evidence"],
            "question": "Extract every explicitly stated relationship in all supplied sentences, including each belief and separately narrated reality. Retain all explicitly stated intervals. Do not infer additional relationships."
            if task == "all"
            else questions(story)[task],
            "instructions": FORMAT,
            # Same cap for both approaches; 14-record fixture plus formatting margin.
            "output_allowance": 2048,
        }
    return result


def prepare(case, tokenizer, manifest):
    return replace(
        prepare_simple(case, tokenizer, manifest),
        request_id="simple-ladder-compact-story-" + case["case_id"],
    )


def next_case(outcomes):
    return str(len(outcomes) + 1) if len(outcomes) < 8 else None


def normalize(value):
    return " ".join(value.strip().casefold().split()) if isinstance(value, str) else None


# These rules are frozen before outputs and shared by selection and evaluation.
RELATIONS = {
    "owns": ["owns", "own", "owner_of"],
    "carries": ["carries", "carry", "carrying"],
    "located_in": ["located_in", "located_at", "is_in", "is_at", "in", "at", "is_located_in"],
}
INVERSES = {
    "owned_by": "owns",
    "is_owned_by": "owns",
    "carried_by": "carries",
    "is_carried_by": "carries",
}


def endpoint(value, story_id):
    text = normalize(value)
    for name in stories()[story_id]["names"]:
        canonical = normalize(name)
        if text in (canonical, "the " + canonical):
            return canonical
    return text


def triple(row, story_id):
    if not isinstance(row, dict) or not all(
        isinstance(row.get(k), str) and row[k].strip() for k in ("subject", "relation", "object")
    ):
        return None
    s, o = endpoint(row["subject"], story_id), endpoint(row["object"], story_id)
    r = normalize(row["relation"]).replace(" ", "_")
    if r in INVERSES:
        return o, INVERSES[r], s
    return s, next((k for k, values in RELATIONS.items() if r in values), r), o


def select(parsed, story_id, task, *, _stories=stories, _triple=triple, _attributed=None):
    """Record-only filter. No reference lookup, semantic repair, clipping or deduplication."""
    facts = parsed.get("facts", []) if isinstance(parsed, dict) else []

    def attributed(row):
        return (
            _attributed(row)
            if _attributed is not None
            else isinstance(row, dict) and ("holder" in row or "attitude" in row)
        )

    triple = _triple

    belief_subjects = {
        triple(r, story_id)[0] for r in facts if attributed(r) and triple(r, story_id) is not None
    }
    start, end = _stories()[story_id]["window"]
    selected, trace = [], []
    for i, row in enumerate(facts):
        t = triple(row, story_id)
        keep, reason = False, "unclassified record"
        if t:
            if task == "possession":
                keep = t[1] in ("owns", "carries") and not attributed(row)
                reason = "direct ownership/carrying category"
            elif task == "locations":
                a, b = row.get("valid_from"), row.get("valid_until")
                numeric = all(
                    isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)
                    for v in (a, b)
                )
                keep = (
                    t[1] == "located_in"
                    and not attributed(row)
                    and numeric
                    and a < b
                    and a < end
                    and b > start
                )
                reason = "direct location with explicit nonempty interval overlap; original bounds retained"
            elif task == "belief":
                keep = attributed(row) or (t[1] == "located_in" and t[0] in belief_subjects)
                reason = "attributed record or narrated location of an attributed subject"
            else:
                raise ValueError("unknown selection task")
        if keep:
            selected.append(row)
        trace.append({"source_index": i, "selected": bool(keep), "criterion": reason})
    return {"facts": selected}, trace


def admit(actual, starts, attempts, *, starting=False, generating=False, seconds):
    if not all(math.isfinite(x) for x in (actual, seconds)) or actual < BASELINE or seconds < 0:
        raise ValueError("invalid compact-story allocation")
    if starts + int(starting) > 1 or attempts + int(generating) > 8:
        raise ValueError("compact-story one-start/eight-request limit")
    end = actual + seconds + 60
    if end > BASELINE + ALLOWANCE - 5 or end > 33660 or end >= 36000:
        raise ValueError("compact-story shutdown/global allocation protection")
    return {
        "actual_seconds": actual,
        "envelope_end_seconds": end,
        "complete_forecast_exception": True,
        "ordinary_execution_authorized": False,
    }


def policy():
    return SimpleNamespace(
        BASELINE=BASELINE,
        ALLOWANCE=ALLOWANCE,
        BLOCK_ID=BLOCK_ID,
        STARTS=1,
        ATTEMPTS=8,
        GENERATION=60,
        config=CONFIG,
        admit=admit,
    )
