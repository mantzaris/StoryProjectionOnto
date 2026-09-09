"""Frozen compact v2 diagnostic batch; no scorer answers enter requests."""

from __future__ import annotations

import math
from dataclasses import replace
from types import SimpleNamespace

from .simple_ladder import cases as original_cases
from .simple_ladder import prepare as prepare_original

BASELINE = 9898.847944
ALLOWANCE = 417.260137
BLOCK_ID = "simple-synthetic-v2-20260908"
CONFIG = {
    "block_id": BLOCK_ID,
    "historical_actual_seconds": BASELINE,
    "maximum_additional_seconds": ALLOWANCE,
    "global_maximum_seconds": 10316.108081,
    "maximum_new_starts": 1,
    "maximum_new_attempts": 8,
    "startup_seconds": 240,
    "live_checks_seconds": 15,
    "generation_seconds": 35,
    "validation_seconds": 10,
    "shutdown_seconds": 60,
    "guard_seconds": 5,
    "historical_development_starts": 9,
    "historical_development_reservations": 25,
    "ordinary_execution_authorized": False,
    "complete_forecast_exception": "development-only",
}

FORMAT = """Return only JSON {"facts": [...]} without extra fields or explanation.
Each fact has subject, relation, object (short readable names), and evidence_ids (a nonempty list of exact supplied string IDs, e.g. ["S1"], never [1]). Use consistent names without leading a/an/the.
For belief, subject/relation/object express the underlying relationship; add holder (the named believer) and attitude="believes". Do not put a sentence inside object. Direct narration has no holder or attitude; do not infer anyone's knowledge.
For explicitly dated relationships requested by the question, include BOTH numeric valid_from and valid_until. Intervals are [valid_from, valid_until), start included and end excluded. Undated facts have neither bound; do not invent durations.
Formatting example only, NOT evidence: {"subject":"kite","relation":"located_in","object":"shed","evidence_ids":["Z9"],"holder":"Orla","attitude":"believes"}. Never copy these example facts or IDs."""

OFFICE_RULE = (
    " Choose direct person-responsibility links or an explicit office-mediated graph as "
    "appropriate. Keep the holder intervals and the office responsibilities. Any direct "
    "person-responsibility link inherits that person's explicitly stated holder interval "
    "and cites both source sentences. Do not add unrelated facts."
)


def cases():
    old = original_cases()
    fresh_office = {
        "S1": "Celia holds the office of Orchard Steward from day 2 up to but not including day 5.",
        "S2": "Dax holds the office of Orchard Steward from day 5 up to but not including day 9.",
        "S3": "The office of Orchard Steward is responsible for checking irrigation.",
        "S4": "The office of Orchard Steward is responsible for recording harvests.",
    }
    fresh_belief = {
        "S1": "Selin believes that the map is in the cabinet.",
        "S2": "In reality, the map is in the satchel.",
    }
    # Order, examples and questions fixed before inference; no output-dependent branching.
    spec = [
        ("1", "direct", "regression", "1", old["1"]["evidence"], old["1"]["question"], 384),
        ("2", "temporal", "regression", "5", old["5"]["evidence"], old["5"]["question"], 512),
        (
            "3",
            "epistemic",
            "original-revised",
            "6",
            old["6"]["evidence"],
            old["6"]["question"],
            512,
        ),
        ("4", "epistemic", "fresh", None, fresh_belief, old["6"]["question"], 512),
        (
            "5",
            "office-person",
            "original-revised",
            "7",
            old["7"]["evidence"],
            old["7"]["question"],
            1024,
        ),
        (
            "6",
            "office-continuity",
            "original-revised",
            "8",
            old["8"]["evidence"],
            old["8"]["question"],
            1024,
        ),
        ("7", "office-person", "fresh", None, fresh_office, old["7"]["question"], 1024),
        ("8", "office-continuity", "fresh", None, fresh_office, old["8"]["question"], 1024),
    ]
    return {
        k: dict(
            case_id=k,
            level=level,
            cohort=cohort,
            original_case=original,
            evidence=evidence,
            question=question,
            instructions=FORMAT + (OFFICE_RULE if level.startswith("office") else ""),
            output_allowance=cap,
        )
        for k, level, cohort, original, evidence, question, cap in spec
    }


def prepare(case, tokenizer, manifest):
    return replace(
        prepare_original(case, tokenizer, manifest),
        request_id="simple-ladder-v2-" + case["case_id"],
    )


def next_case(outcomes):
    return str(len(outcomes) + 1) if len(outcomes) < 8 else None


def admit(actual, starts, attempts, *, starting=False, generating=False, seconds):
    if not all(math.isfinite(x) for x in (actual, seconds)) or actual < BASELINE or seconds < 0:
        raise ValueError("invalid v2 diagnostic allocation")
    if starts + int(starting) > 1 or attempts + int(generating) > 8:
        raise ValueError("v2 one-start/eight-request limit")
    end = actual + seconds + CONFIG["shutdown_seconds"]
    if end > 10316.108081 - 5 or end > 33660 or end >= 36000:
        raise ValueError("v2 must protect shutdown and all historical allocation")
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
        GENERATION=35,
        config=CONFIG,
        admit=admit,
    )
