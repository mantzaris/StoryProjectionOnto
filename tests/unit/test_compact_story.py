"""Bounded compact story checks; authored references are CPU fixtures, never model results."""

import copy
import json
from dataclasses import asdict

import pytest

from story_projection_onto import compact_story as p
from story_projection_onto.scorer_only import compact_story as s
from tests.unit.test_development_demo import pinned as pinned_fixture


@pytest.fixture(scope="module")
def pinned():
    return pinned_fixture.__wrapped__()


def test_frozen_capacity_and_evidence_equality(pinned):
    tokenizer, manifest = pinned
    for k, c in p.cases().items():
        q = p.prepare(c, tokenizer, manifest)
        rendered = tokenizer.apply_chat_template(
            [asdict(m) for m in q.messages],
            tokenize=True,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        assert len(rendered) == q.rendered_input_token_count < 1000
        assert len(rendered) + q.decoding.maximum_output_tokens <= 12288
        assert "guided_json" not in q.wire_payload() and "response_format" not in q.wire_payload()
        complete = {"facts": s.references()[k][0]}
        # Deliberately ordinary spaced serialization, not optimistically whitespace-free.
        tokens = len(tokenizer.encode(json.dumps(complete), add_special_tokens=False))
        assert tokens * 1.25 < q.decoding.maximum_output_tokens
        assert s.evaluate(k, complete)["full"]["f1"] == 1
        for sentence in c["evidence"].values():
            assert sentence in q.messages[1].content
        if c["task"] == "all":
            for question in p.questions(p.stories()[c["story_id"]]).values():
                assert question not in q.messages[1].content
    for story_id, story in p.stories().items():
        assert len(story["evidence"]) == 14 and len(story["names"]) == 10
        assert (
            len(
                {json.dumps(c["evidence"]) for c in p.cases().values() if c["story_id"] == story_id}
            )
            == 1
        )
        assert sum("believes" in line for line in story["evidence"].values()) == 2


def test_selection_copies_records_without_repair_or_gold():
    for story_id in p.stories():
        collection = {"facts": s.story_reference(story_id)}
        before = copy.deepcopy(collection)
        for task in ("possession", "locations", "belief"):
            selected, trace = p.select(collection, story_id, task)
            assert s.evaluate_answer(story_id, task, selected)["full"]["f1"] == 1
            assert all(
                row is collection["facts"][t["source_index"]]
                for row, t in zip(
                    selected["facts"], [t for t in trace if t["selected"]], strict=True
                )
            )
        assert collection == before
    row = s.story_reference("harbor")[4]
    chosen, _ = p.select({"facts": [row, row]}, "harbor", "locations")
    assert chosen["facts"] == [row, row] and chosen["facts"][0]["valid_from"] == 1
    for a, b in ((0, 2), (4, 8)):
        assert not p.select(
            {"facts": [{**row, "valid_from": a, "valid_until": b}]}, "harbor", "locations"
        )[0]["facts"]


def test_errors_uncertainty_and_irrelevance_preserve_denominators():
    refs = s.reference_for("harbor", "possession")
    predicted = [*copy.deepcopy(refs), refs[0], s.story_reference("harbor")[4]]
    predicted[0]["holder"], predicted[0]["attitude"] = "Lena", "believes"
    predicted[2]["valid_until"] = 99
    predicted += [
        {**refs[1], "relation": "has a mysterious connection to"},
        {**refs[1], "subject": "Priya"},
    ]
    outcome = s.evaluate_answer("harbor", "possession", {"facts": predicted})
    assert outcome["full"]["predicted"] == 9
    assert outcome["underlying"]["true_positive"] == 5
    assert len(outcome["unresolved"]) == 1
    assert len(outcome["confirmed_errors"]) == 3
    assert "supported_but_irrelevant" in {r["status"] for r in outcome["rows"]}
    # Inverse relation translation never strips erroneous holder or temporal fields.
    inv = {
        **refs[0],
        "subject": refs[0]["object"],
        "object": refs[0]["subject"],
        "relation": "owned by",
    }
    assert s.full_key(inv, "harbor") == s.full_key(refs[0], "harbor")
    assert s.full_key({**inv, "holder": "Omar", "attitude": "believes"}, "harbor") != s.full_key(
        refs[0], "harbor"
    )


def test_malformed_records_never_crash_or_disappear():
    bad = [
        None,
        "sentence",
        {},
        {"subject": "Lena", "relation": "owns", "object": "Cobalt Compass", "evidence_ids": [{}]},
    ]
    result = s.evaluate_answer("harbor", "possession", {"facts": bad})
    assert result["full"]["predicted"] == 4 and result["full"]["true_positive"] == 0


def test_session_bounds_and_frozen_order():
    from scripts.run_capacity_diagnostics import diagnostic_policy, stage_deadline

    assert diagnostic_policy("compact-story").ALLOWANCE == 900
    p.admit(p.BASELINE, 0, 0, starting=True, seconds=835)
    for kwargs in (
        dict(starts=1, attempts=0, starting=True),
        dict(starts=1, attempts=8, generating=True),
    ):
        with pytest.raises(ValueError):
            p.admit(p.BASELINE, seconds=1, **kwargs)
    with pytest.raises(ValueError):
        p.admit(p.BASELINE + 840, 1, 1, seconds=1)
    deadline, whole = stage_deadline(
        started=100,
        now_monotonic=900,
        prior_block_seconds=0,
        stage_seconds=60,
        semantic="compact-story",
    )
    assert deadline == 935 and whole == 995
    assert [p.cases()[str(i)]["task"] for i in (1, 2)] == ["all", "all"]
    assert p.next_case([{}] * 8) is None
