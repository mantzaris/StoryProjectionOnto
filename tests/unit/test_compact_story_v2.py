"""Focused explicit-null/syntax-recovery controls, not model results."""

import copy
import json
from dataclasses import asdict
from pathlib import Path

import pytest

from story_projection_onto import compact_story as previous
from story_projection_onto import compact_story_v2 as p
from story_projection_onto.compact_syntax import recover, strict_parse
from story_projection_onto.scorer_only import compact_story_v2 as s
from tests.unit.test_development_demo import pinned as pinned_fixture


@pytest.fixture(scope="module")
def pinned():
    return pinned_fixture.__wrapped__()


def test_trailing_commas_only_outside_strings_with_exact_edits():
    raw = '{"facts":[{"subject":"É,] \\" ,}","relation":"owns","object":"x",},]}'
    parsed, info = recover(raw)
    assert parsed["facts"][0]["subject"] == 'É,] " ,}'
    assert not info["strict_parseable"] and info["applied"] and len(info["edits"]) == 2
    original = raw.encode()
    offsets = {e["byte_offset"] for e in info["edits"]}
    assert all(original[i : i + 1] == b"," for i in offsets)
    assert (
        bytes(b for i, b in enumerate(original) if i not in offsets).decode()
        == info["recovered_text"]
    )
    good = '{"facts":[{"object":"comma,] inside"}]}'
    assert recover(good)[1]["edits"] == []


@pytest.mark.parametrize(
    "bad",
    [
        '{"facts":[{"subject":"unfinished',
        '{"facts":[1,',
        '{"facts":[],"facts":[],}',
        '{"facts":[{"a":1,"a":2,},]}',
        '{"facts":[,]}',
        '{"facts":[1,,]}',
        '{"facts":[NaN]}',
    ],
)
def test_incomplete_duplicate_and_ambiguous_syntax_rejected(bad):
    assert strict_parse(bad)[0] is None
    value, info = recover(bad)
    assert value is None and not info["applied"] and not info["edits"]


def test_historical_recovery_uses_original_scorer():
    from scripts.report_compact_story_v2 import historical

    data = historical(Path("reports/tables/compact_story_results.json"))
    for r in data["derived"]:
        full = r["evaluation"]["full"]
        assert (full["true_positive"], full["predicted"], full["reference_count"]) == (3, 10, 5)
        assert full["precision"] == 0.3 and full["recall"] == 0.6 and full["f1"] == 0.4
        assert len(r["syntax_recovery"]["edits"]) == 1 and not r["new_model_output"]


def test_null_selection_preserves_intervals_and_rejects_invented_qualifications():
    for story in p.stories():
        source = {"facts": s.story_reference(story)}
        for task in ("possession", "locations", "belief"):
            selected, trace = p.select(source, story, task)
            assert s.evaluate_answer(story, task, selected)["full"]["f1"] == 1
            assert all(
                row is source["facts"][t["source_index"]]
                for row, t in zip(
                    selected["facts"], [t for t in trace if t["selected"]], strict=True
                )
            )
        assert len(p.select(source, story, "belief")[0]["facts"]) == 4
    row = copy.deepcopy(s.reference_for("harbor", "locations")[0])
    assert p.select({"facts": [row]}, "harbor", "locations")[0]["facts"][0]["valid_from"] == 1
    assert row["holder"] is None and row["attitude"] is None
    for field, value in (("valid_from", 2), ("holder", "Omar")):
        bad = {**row, field: value}
        assert s.full_key(bad, "harbor") != s.full_key(row, "harbor")
    unqualified = s.reference_for("harbor", "possession")[0]
    absent = {k: v for k, v in unqualified.items() if v is not None}
    assert s.full_key(absent, "harbor") == s.full_key(unqualified, "harbor")
    assert s.field_issues(absent) and not s.field_issues(unqualified)
    belief = s.reference_for("harbor", "belief")[0]
    assert belief["subject"] == "Cobalt Compass" and belief["holder"] == "Lena"
    assert s.full_key({**belief, "holder": None, "attitude": None}, "harbor") != s.full_key(
        belief, "harbor"
    )


def test_exact_requests_capacity_and_unchanged_old_evidence(pinned):
    t, m = pinned
    for k, c in p.cases().items():
        q = p.prepare(c, t, m)
        actual = t.apply_chat_template(
            [asdict(x) for x in q.messages],
            tokenize=True,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        assert q.rendered_input_token_count == len(actual) < 1500
        assert len(actual) + 2048 <= 12288 and "guided_json" not in q.wire_payload()
        capacity = len(
            t.encode(json.dumps({"facts": s.references()[k][0]}), add_special_tokens=False)
        )
        assert capacity * 1.5 < 2048
        assert q.messages[1].content.startswith("TASK: " + c["question"])
        for sentence in c["evidence"].values():
            assert sentence in q.messages[1].content
    for name, old in previous.stories().items():
        assert p.stories()[name] == old and p.questions(p.stories()[name]) == previous.questions(
            old
        )
    assert [p.cases()[str(i)]["task"] for i in (1, 2, 3)] == ["all"] * 3
    assert p.stories()["museum"]["evidence"]["S1"].startswith("Niko believes")


def test_seven_hundred_second_limit_and_no_thirteenth_request():
    from scripts.run_capacity_diagnostics import diagnostic_policy, stage_deadline

    assert diagnostic_policy("compact-story-v2").ATTEMPTS == 12
    p.admit(p.BASELINE, 0, 0, starting=True, seconds=635)
    for kw in (
        dict(starts=1, attempts=0, starting=True),
        dict(starts=1, attempts=12, generating=True),
    ):
        with pytest.raises(ValueError):
            p.admit(p.BASELINE, seconds=1, **kw)
    with pytest.raises(ValueError):
        p.admit(p.BASELINE + 640, 1, 1, seconds=1)
    end, whole = stage_deadline(
        started=100,
        now_monotonic=700,
        prior_block_seconds=0,
        stage_seconds=45,
        semantic="compact-story-v2",
    )
    assert end == 735 and whole == 795
    assert p.next_case([{}] * 12) is None
