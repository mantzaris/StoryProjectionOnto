"""Bounded published-prose extraction; no registered ontology contract or gold imports."""

import json
import math
from dataclasses import asdict, replace
from pathlib import Path
from types import SimpleNamespace

from . import compact_story_v2 as compact
from .gpu_runtime import ChatMessage

BASELINE = 10512.544560
ALLOWANCE = 450.0
BLOCK_ID = "real-text-poc-20260909"
CONFIG = {
    **compact.CONFIG,
    "block_id": BLOCK_ID,
    "historical_actual_seconds": BASELINE,
    "maximum_additional_seconds": ALLOWANCE,
    "global_maximum_seconds": 10970.219657,
    "maximum_new_attempts": 9,
    "startup_seconds": 240,
    "generation_seconds": 40,
}
FORMAT = (
    compact.FORMAT
    + """
This task is extraction from published narrative prose, not real-world verification. Use ONLY the supplied excerpt, never remembered book or plot information. Ordinary action relations are welcome. Resolve pronouns only when their referent is clear within this excerpt; otherwise preserve uncertainty in readable wording. Besides believes, use reported, promises, questions or considers only when that attribution is explicit. An external description of a speech/mental act may itself be a direct narrated relationship, but never assert its uncertain or promised content as accomplished reality. Do not invent numerical validity for 'once', 'later', 'that morning' or an event occurrence: leave unsupported bounds null. Keep answers concise; no inferred relationships."""
)
normalize = compact.normalize

QUESTIONS = {
    "fable": {
        "actions": "Which physical running/waking, capture, binding, rope-gnawing and release relationships involve the Lion, Mouse, hunters and ropes? Exclude dialogue, intentions, laughter and the moral.",
        "claims": "How does the Lion capture and release the Mouse, and what does the Mouse explicitly promise or claim about repayment, ability to help, and the Lion's earlier ridicule? Exclude hunters, other physical actions and unspoken intentions.",
    },
    "alice": {
        "actions": "Which sitting/location, reading/inspection and running relationships involve Alice, her sister, the book, bank and Rabbit? Exclude thoughts, feelings, appearance and book-content properties.",
        "claims": "What does the passage say about the book's absent content and Alice's thoughts about the usefulness of such books and making a daisy-chain? Exclude physical actions, locations, appearance and the Rabbit; preserve questions and consideration without treating them as decisions.",
    },
    "holmes": {
        "actions": "Which visit, conversation, pulling, door-closing, rising and greeting actions involve Watson, Holmes and the elderly gentleman? Exclude appearance and the content of dialogue.",
        "claims": "What does the dialogue explicitly convey about being engaged, past partnership/help in cases and expected help in the present case? Keep each claim or concern attributed to its speaker. Exclude physical movements, appearance, offers to wait and politeness about the arrival time.",
    },
}

# Prospective lexical categories only; they neither inspect scorer references nor add records.
SELECTION_TERMS = {
    "fable": {
        "actions": [
            "run",
            "ran",
            "wake",
            "waken",
            "catch",
            "caught",
            "captur",
            "bind",
            "bound",
            "tie",
            "gnaw",
            "free",
            "releas",
            "let_go",
            "spare",
        ],
        "claims": [
            "catch",
            "caught",
            "captur",
            "free",
            "releas",
            "let_go",
            "spare",
            "repay",
            "help",
            "benefit",
            "promis",
            "claim",
            "say",
            "said",
            "ridicul",
        ],
    },
    "alice": {
        "actions": [
            "sit",
            "sat",
            "beside",
            "read",
            "peep",
            "look",
            "inspect",
            "run",
            "ran",
            "near",
            "locat",
            "on_bank",
        ],
        "claims": [
            "lack",
            "contain",
            "has_no",
            "have_no",
            "without",
            "think",
            "thought",
            "consider",
            "wonder",
            "question",
            "useful",
            "daisy",
        ],
    },
    "holmes": {
        "actions": [
            "visit",
            "call",
            "convers",
            "talk",
            "pull",
            "close",
            "rose",
            "rise",
            "greet",
            "bob",
        ],
        "claims": [
            "engag",
            "partner",
            "help",
            "use",
            "afraid",
            "believ",
            "say",
            "said",
            "report",
            "claim",
            "assist",
        ],
    },
}


def stories():
    return json.loads(
        (Path(__file__).resolve().parents[2] / "data/published_prose/passages.json").read_text()
    )


def questions(story):
    sid = next(k for k, v in stories().items() if v["excerpt_sha256"] == story["excerpt_sha256"])
    return QUESTIONS[sid]


def cases():
    sources = stories()
    order = [(s, "all") for s in ("fable", "alice", "holmes")]
    order += [(s, t) for s in ("fable", "alice", "holmes") for t in QUESTIONS[s]]
    return {
        str(i): dict(
            case_id=str(i),
            story_id=sid,
            task=task,
            level="published_prose",
            evidence=sources[sid]["evidence"],
            instructions=FORMAT,
            question="Extract the explicitly supported relationships throughout this complete excerpt. No contextual question is supplied; distinguish narration from attributed or uncertain content."
            if task == "all"
            else QUESTIONS[sid][task],
            output_allowance=2048,
        )
        for i, (sid, task) in enumerate(order, 1)
    }


def prepare(case, tokenizer, manifest):
    q = compact.prepare(case, tokenizer, manifest)
    messages = (ChatMessage("system", FORMAT), q.messages[1])
    count = len(
        tokenizer.apply_chat_template(
            [asdict(m) for m in messages],
            tokenize=True,
            add_generation_prompt=True,
            enable_thinking=False,
        )
    )
    return replace(
        q,
        messages=messages,
        request_id="simple-ladder-real-text-" + case["case_id"],
        rendered_input_token_count=count,
    )


def select(parsed, story_id, task):
    selected, trace = [], []
    for i, row in enumerate((parsed or {}).get("facts", [])):
        relation = row.get("relation") if isinstance(row, dict) else None
        text = normalize(relation).replace(" ", "_") if isinstance(relation, str) else ""
        keep = any(term in text for term in SELECTION_TERMS[story_id][task])
        if story_id == "fable" and task == "claims":
            # Do not mistake Mouse freeing Lion for Lion releasing Mouse; this is entity filtering,
            # not a binding correction. Attributed content remains unaltered.
            physical = any(
                t in text
                for t in ("catch", "caught", "captur", "free", "releas", "let_go", "spare")
            )
            if physical:
                subject = row.get("subject")
                keep = (
                    isinstance(subject, str) and normalize(subject).removeprefix("the ") == "lion"
                )
        if keep:
            selected.append(row)
        trace.append(
            dict(
                source_index=i,
                selected=keep,
                criterion="frozen relation-substring category; fable claims additionally Lion subject for capture/release",
            )
        )
    return {"facts": selected}, trace


def next_case(outcomes):
    return str(len(outcomes) + 1) if len(outcomes) < 9 else None


def admit(actual, starts, attempts, *, starting=False, generating=False, seconds):
    if not all(math.isfinite(x) for x in (actual, seconds)) or actual < BASELINE or seconds < 0:
        raise ValueError("invalid published-prose allocation")
    if starts + int(starting) > 1 or attempts + int(generating) > 9:
        raise ValueError("published-prose one-start/nine-request limit")
    end = actual + seconds + 60
    if end > min(BASELINE + ALLOWANCE, 10970.219657) - 5 or end > 33660 or end >= 36000:
        raise ValueError("published-prose protected shutdown/global limit")
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
        ATTEMPTS=9,
        GENERATION=40,
        config=CONFIG,
        admit=admit,
    )
