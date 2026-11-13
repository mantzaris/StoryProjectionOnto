"""Focused source, packing, reference and resource controls for published prose only."""

import hashlib
import json
from dataclasses import asdict
from pathlib import Path

import pytest

from story_projection_onto import real_text_poc as p
from story_projection_onto.compact_syntax import recover
from story_projection_onto.scorer_only import real_text_poc as s
from tests.unit.test_development_demo import pinned


def test_exact_contiguous_sources_and_attribution():
    for story in p.stories().values():
        assert "".join(story["evidence"].values()) == story["excerpt"]
        assert hashlib.sha256(story["excerpt"].encode()).hexdigest() == story["excerpt_sha256"]
        assert 100 <= story["word_count"] <= 250
        assert (
            hashlib.sha256(Path(story["notice_file"]).read_bytes()).hexdigest()
            == story["notice_sha256"]
        )
        assert "Project Gutenberg" in Path(story["notice_file"]).read_text()
    assert p.stories()["fable"]["excerpt"].startswith("A LION")
    assert p.stories()["fable"]["excerpt"].endswith("on a Lion.”\r\n")


def test_frozen_requests_actual_template_and_complete_output_capacity():
    t, m = pinned.__wrapped__()
    qs = {k: p.prepare(c, t, m) for k, c in p.cases().items()}
    assert len(qs) == 9
    for k, q in qs.items():
        c = p.cases()[k]
        actual = t.apply_chat_template(
            [asdict(x) for x in q.messages],
            tokenize=True,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        assert len(actual) == q.rendered_input_token_count
        assert len(actual) < 1500 and len(actual) + 2048 <= 12288
        assert q.decoding.maximum_output_tokens == 2048 and "guided_json" not in q.wire_payload()
        assert all(v in q.messages[1].content for v in c["evidence"].values())
        assert len(t.encode(json.dumps({"facts": s.references()[k][0]}))) * 1.5 < 2048
        if c["task"] == "all":
            assert not any(
                question in q.messages[1].content
                for questions in p.QUESTIONS.values()
                for question in questions.values()
            )
        assert s.evaluate(k, {"facts": s.references()[k][0]})["full"]["f1"] == 1
    for sid in p.stories():
        cases = [c for c in p.cases().values() if c["story_id"] == sid]
        assert all(c["evidence"] == cases[0]["evidence"] for c in cases)


def test_component_credit_preserves_qualifications_and_wrong_direction_errors():
    b = s.concepts("holmes")[6][0]
    assert s.evaluate_answer("holmes", "claims", {"facts": [b]})["full"]["true_positive"] == 1
    for changed in (
        {"holder": None, "attitude": None},
        {"holder": "Holmes"},
        {"valid_from": 1, "valid_until": 2},
    ):
        row = {**b, **changed}
        e = s.evaluate_answer("holmes", "claims", {"facts": [row]})
        assert e["full"]["true_positive"] == 0 and e["underlying"]["true_positive"] == 1
    wrong = {**b, "subject": "Wilson"}
    assert (
        s.evaluate_answer("holmes", "claims", {"facts": [wrong]})["underlying"]["true_positive"]
        == 0
    )
    for group in s.concepts("alice")[7:]:
        for a in group:
            assert (
                s.evaluate_answer("alice", "claims", {"facts": [a]})["full"]["true_positive"] == 1
            )


def test_no_falsity_from_reference_absence_no_semantic_repair_by_filter():
    row = s.fact("Alice", "feels", "sleepy", "S2")
    e = s.evaluate_answer("alice", "actions", {"facts": [row]})
    assert e["rows"][0]["status"] == "unresolved_wording" and not e["confirmed_errors"]
    assert e["full"]["predicted"] == 1
    original = {"facts": [s.concepts("alice")[0][0], row]}
    selected, _ = p.select(original, "alice", "actions")
    assert (
        selected["facts"] == [original["facts"][0]] and selected["facts"][0] is original["facts"][0]
    )
    assert selected["facts"][0]["holder"] is None


def test_syntax_recovery_is_only_complete_document_comma_removal():
    good = '{"facts":[{"object":"comma,] within string"},]}'
    parsed, edits = recover(good)
    assert parsed["facts"][0]["object"] == "comma,] within string" and len(edits["edits"]) == 1
    for bad in ('{"facts":[{"object":"unfinished', '{"facts":[],"facts":[],}'):
        assert recover(bad)[0] is None


def test_session_caps_preserve_history_and_shutdown():
    from scripts.run_capacity_diagnostics import diagnostic_policy, stage_deadline

    assert diagnostic_policy("real-text-poc").config == p.CONFIG
    p.admit(p.BASELINE, 0, 0, starting=True, seconds=385)
    for kwargs in (
        dict(starts=1, attempts=0, starting=True),
        dict(starts=1, attempts=9, generating=True),
    ):
        with pytest.raises(ValueError):
            p.admit(p.BASELINE, seconds=1, **kwargs)
    with pytest.raises(ValueError):
        p.admit(p.BASELINE + 390, 1, 8, seconds=1)
    deadline, whole = stage_deadline(
        started=100,
        now_monotonic=400,
        prior_block_seconds=0,
        stage_seconds=240,
        semantic="real-text-poc",
    )
    assert whole == 545 and deadline == 485
