"""Tiny, unconstrained development probes. No ontology contract or gold imports."""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from types import SimpleNamespace

from .contracts import canonical_sha256
from .gpu_runtime import FALLBACK_MODEL_REVISION, FALLBACK_SERVED_MODEL_NAME, ChatMessage
from .llm import DecodingManifest

BASELINE = 9741.500089
ALLOWANCE = 574.607992
BLOCK_ID = "simple-synthetic-ladder-20260908"
CONFIG = {
    "block_id": BLOCK_ID,
    "historical_actual_seconds": BASELINE,
    "maximum_additional_seconds": ALLOWANCE,
    "global_maximum_seconds": 10316.108081,
    "maximum_new_starts": 1,
    "maximum_new_attempts": 8,
    "startup_seconds": 300,
    "live_checks_seconds": 15,
    "generation_seconds": 45,
    "validation_seconds": 10,
    "shutdown_seconds": 60,
    "guard_seconds": 5,
    "historical_development_starts": 8,
    "historical_development_reservations": 17,
    "ordinary_execution_authorized": False,
    "complete_forecast_exception": "development-only",
}


def admit(actual, starts, attempts, *, starting=False, generating=False, seconds):
    if not all(math.isfinite(x) for x in (actual, seconds)) or actual < BASELINE or seconds < 0:
        raise ValueError("invalid ladder allocation")
    if starts + int(starting) > 1 or attempts + int(generating) > 8:
        raise ValueError("ladder start/request limit")
    end = actual + seconds + 60
    if end > BASELINE + ALLOWANCE - 5 or end > 33660 or end >= 36000:
        raise ValueError("ladder must protect shutdown and all historical allocation")
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
        GENERATION=45,
        config=CONFIG,
        admit=admit,
    )


DIRECT1 = ["Mira carries a lantern.", "Tomas owns the lantern.", "Mira is in the courtyard."]
DIRECT2 = ["Nadia carries a compass.", "Oren owns the compass.", "Oren is in the harbor."]
SELECT = [
    "Ivo owns a key.",
    "Leda owns a telescope.",
    "Ivo is in the observatory.",
    "Leda is in the garden.",
]
OFFICE = [
    "Ada holds the office of Harbor Warden from day 0 up to but not including day 3.",
    "Bram holds the office of Harbor Warden from day 3 up to but not including day 6.",
    "The office of Harbor Warden is responsible for inspecting boats.",
    "The office of Harbor Warden is responsible for maintaining beacons.",
]
BASIC = (
    'Return only JSON {"facts": [...]} with one object per relationship. Each object has '
    "only subject, relation, object, evidence_id (the sentence ID). Use readable endpoint "
    "names directly, without articles; do not invent identifiers. No explanation or inference."
)
TIME = (
    " Use valid_from and valid_until numeric day bounds only for explicitly dated relationships; "
    "intervals are [valid_from, valid_until), including the start and excluding the end. "
    "Do not infer duration for undated relationships."
)
OFFICE_FORMAT = (
    BASIC.replace("only subject", "subject").replace("No explanation or inference.", "")
    + TIME
    + " For a relationship supported jointly by multiple sentences, evidence_id "
    "is a list of their IDs; otherwise it is one ID. Choose direct person-responsibility "
    "links or an explicit office-mediated graph as appropriate. Office-mediated links "
    "must retain the office and its responsibilities. Direct person-responsibility "
    "links must retain the holder intervals and cite the holder and responsibility "
    "sentences. Do not add unrelated facts."
)


def cases():
    """Exactly eight frozen cases plus a conditional replacement control, never nine calls."""
    rows = [
        (
            "1",
            "direct",
            DIRECT1,
            "Extract every explicitly stated relationship and no inferred relationships.",
            BASIC,
            384,
        ),
        (
            "2",
            "direct",
            DIRECT2,
            "Extract every explicitly stated relationship and no inferred relationships.",
            BASIC,
            384,
        ),
        (
            "3",
            "selection",
            SELECT,
            "Who owns what? Return all and only ownership relationships.",
            BASIC,
            384,
        ),
        (
            "4",
            "selection",
            SELECT,
            "Where are the people? Return all and only location relationships.",
            BASIC,
            384,
        ),
        (
            "5",
            "temporal",
            [
                "Ada is in the atrium from day 0 up to but not including day 2.",
                "Ada is in the library from day 2 up to but not including day 5.",
            ],
            "State both location relationships with their explicitly given intervals.",
            BASIC.replace("only subject", "subject") + TIME,
            512,
        ),
        (
            "6",
            "epistemic",
            [
                "Rina believes that the coin is in the drawer.",
                "In reality, the coin is in the chest.",
            ],
            "State the believed location and the narrated real location, keeping belief separate from reality.",
            BASIC.replace("only subject", "subject")
            + " Add holder and attitude only to an attributed "
            "proposition: holder is the explicitly named believer and attitude is believes. "
            "Direct narrated reality has neither field. Do not turn belief into a global fact.",
            512,
        ),
        (
            "7",
            "ontology_contrast",
            OFFICE,
            "Person-centered question: who held the office when, and what responsibilities did each person have through it?",
            OFFICE_FORMAT,
            1024,
        ),
        (
            "8",
            "ontology_contrast",
            OFFICE,
            "Office-continuity question: represent the continuing office, its responsibilities, and the successive holders with their intervals.",
            OFFICE_FORMAT,
            1024,
        ),
        (
            "control",
            "plain_language_control",
            DIRECT1,
            "List every explicitly stated relationship in these sentences in plain language, one per line. Do not infer anything.",
            "Answer in plain language, not JSON. Use only the supplied evidence.",
            256,
        ),
    ]
    return {
        k: {
            "case_id": k,
            "level": level,
            "evidence": {f"S{i}": s for i, s in enumerate(ev, 1)},
            "question": question,
            "instructions": guide,
            "output_allowance": cap,
        }
        for k, level, ev, question, guide, cap in rows
    }


@dataclass(frozen=True)
class SimpleDiagnosticRequest:
    """Transport-compatible request with no condition, ontology packing or guided schema."""

    request_id: str
    messages: tuple[ChatMessage, ...]
    decoding: DecodingManifest
    rendered_input_token_count: int
    model_name: str = FALLBACK_SERVED_MODEL_NAME
    stream_response: bool = True
    diagnostic_raw_text: bool = True
    canonical_output_schema: None = None
    opaque_reference_aliases: None = None
    sealed_record_copies: None = None

    def __post_init__(self):
        if not self.request_id.startswith("simple-ladder-"):
            raise ValueError("simple ladder request identity required")
        if self.decoding.tokenizer_revision != FALLBACK_MODEL_REVISION:
            raise ValueError("pinned tokenizer required")
        if self.rendered_input_token_count > self.decoding.maximum_input_tokens:
            raise ValueError("complete ladder input exceeds declared cap")
        if self.rendered_input_token_count + self.decoding.maximum_output_tokens > 12288:
            raise ValueError("complete ladder request exceeds context")

    @property
    def request_hash(self):
        return canonical_sha256(self.wire_payload())

    def wire_payload(self):
        d = self.decoding
        return {
            "model": self.model_name,
            "messages": [asdict(m) for m in self.messages],
            "chat_template_kwargs": {"enable_thinking": False},
            **{
                k: getattr(d, k)
                for k in (
                    "temperature",
                    "top_p",
                    "top_k",
                    "min_p",
                    "presence_penalty",
                    "frequency_penalty",
                    "repetition_penalty",
                    "n",
                    "best_of",
                )
            },
            "use_beam_search": False,
            "ignore_eos": False,
            "max_tokens": d.maximum_output_tokens,
            "seed": d.seed,
            "stop_token_ids": list(d.stop_token_ids),
            "stream": True,
            "stream_options": {"include_usage": True, "continuous_usage_stats": False},
        }


def prepare(case, tokenizer, manifest):
    messages = (
        ChatMessage("system", case["instructions"]),
        ChatMessage(
            "user",
            "Evidence:\n"
            + "\n".join(f"{k}: {v}" for k, v in case["evidence"].items())
            + "\nQuestion: "
            + case["question"],
        ),
    )
    rendered = tokenizer.apply_chat_template(
        [asdict(m) for m in messages],
        tokenize=True,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    decoding = DecodingManifest.first_pass(
        seed=1988649846,
        eos_token_id=manifest.eos_token_id,
        end_of_turn_token_ids=manifest.end_of_turn_token_ids,
        chat_template_hash=manifest.chat_template_sha256,
        output_schema_hash=canonical_sha256({}),
        structured_decoder="none-simple-development-diagnostic",
        tokenizer_revision=manifest.tokenizer_revision,
        maximum_input_tokens=10240,
        maximum_output_tokens=case["output_allowance"],
    )
    return SimpleDiagnosticRequest(
        "simple-ladder-" + case["case_id"], messages, decoding, len(rendered)
    )


def parse_text(text):
    """Only trim surrounding whitespace and one exact Markdown fence; never complete JSON."""
    stripped = text.strip()
    normalization = "surrounding_whitespace" if stripped != text else "none"
    for prefix in ("```json\n", "```\n"):
        if stripped.startswith(prefix) and stripped.endswith("\n```"):
            stripped = stripped[len(prefix) : -4]
            normalization = "exact_surrounding_markdown_fence"
            break
    try:
        parsed = json.loads(stripped)
    except (ValueError, TypeError) as exc:
        return None, normalization, str(exc)
    if not isinstance(parsed, dict) or not isinstance(parsed.get("facts"), list):
        return None, normalization, "JSON must contain a facts list"
    return parsed, normalization, None
