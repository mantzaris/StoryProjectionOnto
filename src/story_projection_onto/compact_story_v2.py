"""Explicit-null compact-story version; same evidence path, no gold in requests."""

from dataclasses import asdict, replace
from types import SimpleNamespace

from . import compact_story as old
from .gpu_runtime import ChatMessage
from .simple_ladder import prepare as prepare_simple

BASELINE = 10263.092129
ALLOWANCE = 700.0
BLOCK_ID = "compact-stories-v2-20260909"
CONFIG = {
    **old.CONFIG,
    "block_id": BLOCK_ID,
    "historical_actual_seconds": BASELINE,
    "maximum_additional_seconds": ALLOWANCE,
    "global_maximum_seconds": 10970.219657,
    "maximum_new_attempts": 12,
    "generation_seconds": 45,
}
RELATIONS, INVERSES = old.RELATIONS, old.INVERSES
normalize = old.normalize
questions = old.questions
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
FORMAT = """Return only JSON {"facts":[...]} answering the TASK. Every fact must explicitly contain all eight fields: subject, relation, object, evidence_ids, valid_from, valid_until, holder, attitude. Use short consistent endpoint names and a list of supplied string evidence IDs. No trailing commas.
For unstated bounds, use null; for direct narration, holder and attitude are null. Null means unspecified/not attributed here, never false, timeless, or lack of knowledge. For explicit intervals include the actual numeric start and end: [valid_from,valid_until), start included, end excluded. Never clip source bounds to the question window.
For belief, subject/relation/object express the underlying believed relationship; holder is its named believer and attitude is "believes". Do not put a proposition sentence inside object or use believes as the underlying relation. Do not infer knowledge or other relationships. In a contextual task return ONLY needed relationships, not every story fact.
Unrelated worked example (not evidence for your answer): Z1 Tavi carries Plum Token from day 2 up to but not including day 5. Z2 Miro believes Plum Token is in Blue Cabin. Z3 Lusa owns Red Bowl. Example TASK: return carrying and explicit belief only.
Example answer: {"facts":[{"subject":"Tavi","relation":"carries","object":"Plum Token","evidence_ids":["Z1"],"valid_from":2,"valid_until":5,"holder":null,"attitude":null},{"subject":"Plum Token","relation":"located_in","object":"Blue Cabin","evidence_ids":["Z2"],"valid_from":null,"valid_until":null,"holder":"Miro","attitude":"believes"}]}
Z3 is excluded because ownership does not answer that example TASK. Never copy example names, facts or IDs into your answer."""


def stories():
    result = old.stories()
    result["museum"] = {
        "title": "Museum preparations",
        "window": [22, 25],
        "names": [
            "Esme",
            "Dario",
            "Niko",
            "Ruby Medal",
            "Onyx Bell",
            "Pearl Box",
            "Glass Room",
            "Upper Court",
            "Brick Store",
            "Cedar Gate",
        ],
        "evidence": dict(
            zip(
                (f"S{i}" for i in range(1, 15)),
                [
                    "Niko believes that Pearl Box is in Glass Room.",
                    "Esme owns Ruby Medal.",
                    "Dario is in Upper Court from day 20 up to but not including day 23.",
                    "In reality, Pearl Box is in Brick Store.",
                    "Niko carries Ruby Medal from day 19 up to but not including day 26.",
                    "Onyx Bell is in Cedar Gate from day 23 up to but not including day 27.",
                    "Esme believes that Ruby Medal is in Cedar Gate.",
                    "Dario owns Onyx Bell.",
                    "Esme is in Glass Room from day 21 up to but not including day 25.",
                    "In reality, Ruby Medal is in Upper Court.",
                    "Niko owns Pearl Box.",
                    "Niko is in Brick Store from day 24 up to but not including day 28.",
                    "Esme carries Pearl Box from day 20 up to but not including day 24.",
                    "Dario is in Cedar Gate from day 28 up to but not including day 30.",
                ],
                strict=True,
            )
        ),
    }
    return result


def cases():
    result = {}
    sources = stories()
    order = [(s, "all") for s in sources] + [
        (s, t) for s in sources for t in ("possession", "locations", "belief")
    ]
    for i, (sid, task) in enumerate(order, 1):
        result[str(i)] = dict(
            case_id=str(i),
            story_id=sid,
            task=task,
            level="query-blind" if task == "all" else "contextual",
            evidence=sources[sid]["evidence"],
            instructions=FORMAT,
            question="Extract every explicitly stated relationship in all supplied sentences, including each belief and separately narrated reality. Retain all explicitly stated intervals. Do not infer additional relationships."
            if task == "all"
            else questions(sources[sid])[task],
            output_allowance=2048,
        )
    return result


def prepare(case, tokenizer, manifest):
    request = prepare_simple(case, tokenizer, manifest)
    messages = (
        ChatMessage("system", FORMAT),
        ChatMessage(
            "user",
            "TASK: "
            + case["question"]
            + "\n\nCOMPLETE EVIDENCE:\n"
            + "\n".join(f"{k}: {v}" for k, v in case["evidence"].items()),
        ),
    )
    tokens = tokenizer.apply_chat_template(
        [asdict(m) for m in messages],
        tokenize=True,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    return replace(
        request,
        request_id="simple-ladder-compact-story-v2-" + case["case_id"],
        messages=messages,
        rendered_input_token_count=len(tokens),
    )


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


def select(parsed, story_id, task):
    return old.select(
        parsed,
        story_id,
        task,
        _stories=stories,
        _triple=triple,
        _attributed=lambda row: (
            isinstance(row, dict)
            and (row.get("holder") is not None or row.get("attitude") is not None)
        ),
    )


def next_case(outcomes):
    return str(len(outcomes) + 1) if len(outcomes) < 12 else None


def admit(actual, starts, attempts, *, starting=False, generating=False, seconds):
    import math

    if not all(math.isfinite(v) for v in (actual, seconds)) or actual < BASELINE or seconds < 0:
        raise ValueError("invalid compact-story-v2 allocation")
    if starts + int(starting) > 1 or attempts + int(generating) > 12:
        raise ValueError("one start / twelve requests exhausted")
    end = actual + seconds + 60
    if end > min(BASELINE + ALLOWANCE, 10970.219657) - 5 or end > 33660 or end >= 36000:
        raise ValueError("shutdown or whole-session allocation limit")
    return dict(
        actual_seconds=actual,
        envelope_end_seconds=end,
        complete_forecast_exception=True,
        ordinary_execution_authorized=False,
    )


def policy():
    return SimpleNamespace(
        BASELINE=BASELINE,
        ALLOWANCE=ALLOWANCE,
        BLOCK_ID=BLOCK_ID,
        STARTS=1,
        ATTEMPTS=12,
        GENERATION=45,
        config=CONFIG,
        admit=admit,
    )
