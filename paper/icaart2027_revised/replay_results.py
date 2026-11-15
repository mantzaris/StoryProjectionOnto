"""Replay retained scores and apply the existing A filters to B outputs on CPU.

Run from any directory with Python 3.12, pydantic and zstandard installed.
No model, tokenizer, remote service or private run directory is needed.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "src"))

from story_projection_onto import compact_story_v2, real_text_poc
from story_projection_onto.compact_syntax import recover
from story_projection_onto.scorer_only import compact_story_v2 as compact_scorer
from story_projection_onto.scorer_only import real_text_poc as prose_scorer


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def json_value(value):
    return json.loads(json.dumps(value))


def calculate():
    rows, traces, hashes = [], [], {}
    checked_calls = checked_answers = 0
    for name, filename, protocol, scorer in [
        ("synthetic", "compact_story_v2_results", compact_story_v2, compact_scorer),
        ("prose", "real_text_proof_of_concept", real_text_poc, prose_scorer),
    ]:
        path = ROOT / "reports/tables" / (filename + ".json")
        hashes[str(path.relative_to(ROOT))] = sha(path)
        data = json.loads(path.read_text())
        assert data["references"] == json_value(scorer.references())
        assert data["rules"] == json_value(scorer.frozen_rules())
        for call in data["calls"]:
            parsed, receipt = recover(call["raw_text"])
            assert parsed == call["parsed"]
            assert not receipt["applied"], "Main batches must retain their original parse outcomes"
            assert json_value(scorer.evaluate(call["case_id"], parsed)) == call["evaluation"]
            checked_calls += 1
        for answer in data["results"]:
            original = scorer.evaluate_answer(answer["story_id"], answer["task"], answer["parsed"])
            assert json_value(original) == answer["evaluation"]
            if answer["approach"].startswith("A"):
                call = next(c for c in data["calls"] if c["case_id"] == answer["source_case_id"])
                selected = protocol.select(call["parsed"], answer["story_id"], answer["task"])[0] if call["parsed"] is not None else None
                assert selected == answer["parsed"]
            checked_answers += 1
        for answer in data["results"]:
            if not answer["approach"].startswith("B"):
                continue
            sid, task = answer["story_id"], answer["task"]
            a = next(r for r in data["results"] if r["story_id"] == sid and r["task"] == task and r["approach"].startswith("A"))
            call = next(c for c in data["calls"] if c["case_id"] == answer["source_case_id"])
            if answer["parsed"] is None:
                filtered = trace = score = None
            else:
                filtered, trace = protocol.select(answer["parsed"], sid, task)
                kept = [answer["parsed"]["facts"][t["source_index"]] for t in trace if t["selected"]]
                assert filtered == {"facts": kept}, "Filter must only select existing records"
                score = scorer.evaluate_answer(sid, task, filtered)["full"]
            row = {"dataset": name, "story": sid, "question": task}
            for label, parsed, evaluation in [
                ("A", a["parsed"], a["evaluation"]["full"]),
                ("B", answer["parsed"], answer["evaluation"]["full"]),
                ("B_filter", filtered, score),
            ]:
                for key in ("predicted", "true_positive", "reference_count", "precision", "recall", "f1"):
                    row[label + "_" + key] = evaluation[key] if parsed is not None else None
                row[label + "_status"] = "scored" if parsed is not None else "unavailable_parse_failure"
            rows.append(row)
            traces.append({"dataset": name, "story": sid, "question": task,
                           "source_call": call["case_id"], "response_sha256": call["response"]["response_sha256"],
                           "selection_trace": trace, "selected_output": filtered})
    for name in ["compact_story.py", "compact_story_v2.py", "real_text_poc.py", "compact_syntax.py",
                 "scorer_only/compact_story.py", "scorer_only/compact_story_v2.py",
                 "scorer_only/real_text_poc.py", "scorer_only/simple_ladder_v2.py"]:
        path = ROOT / "src/story_projection_onto" / name
        hashes[str(path.relative_to(ROOT))] = sha(path)
    return {"design": "Post hoc diagnostic: apply the original A selector unchanged to each B response; no new generation, scoring rules, reference answers or semantic judgments.",
            "original_calls_replayed": checked_calls, "original_question_answers_replayed": checked_answers,
            "source_sha256": hashes, "rows": rows, "selection_traces": traces}


def outputs(data):
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=list(data["rows"][0]), lineterminator="\n")
    writer.writeheader()
    writer.writerows(data["rows"])
    tex = [r"\begin{table}[t]", r"\caption{Follow-up filter check: strict F1 on saved synthetic answers. B+filter uses A's unchanged selection rules.}\label{tab:filter}",
           r"\centering\small", r"\begin{tabular}{llrr}\toprule", r"Story & Question & B & B+filter\\\midrule"]
    for row in data["rows"]:
        if row["dataset"] != "synthetic":
            continue
        label = {"possession": "Possession", "locations": "Location", "belief": "Belief/reality"}[row["question"]]
        tex.append(f"{row['story'].title()} & {label} & {row['B_f1']:.3f} & {row['B_filter_f1']:.3f}" + r" \\")
    tex += [r"\bottomrule\end{tabular}", r"\end{table}"]
    # Append one diagnostic column; every original A/B table cell stays literal.
    compact = (ROOT / "paper/icaart2027/tables/compact.tex").read_text()
    compact = re.sub(r"\\caption\{.*?\}\\label", lambda _: r"\caption{Synthetic strict reference agreement. A: pre-extract/select; B: contextual extraction; B+f: B with the unchanged fixed filters. $n$ counts all predictions.}\label", compact)
    compact = compact.replace(r"\centering\small", r"\centering\small\setlength{\tabcolsep}{4pt}")
    compact = compact.replace("{llrrrrrrrr}", "{llrrrrrrrrr}")
    compact = compact.replace(r"$F1_B$\\", r"$F1_B$ & $F1_{B+f}$\\")
    lines = compact.splitlines()
    added = 0
    for row in data["rows"]:
        if row["dataset"] != "synthetic":
            continue
        label = {"possession": "Possession", "locations": "Location", "belief": "Belief/reality"}[row["question"]]
        prefix = f"{row['story'].title()} & {label} &"
        index = next(i for i, line in enumerate(lines) if line.startswith(prefix))
        assert r" \\" in lines[index]
        lines[index] = lines[index].replace(r" \\", f" & {row['B_filter_f1']:.3f}" + r" \\", 1)
        added += 1
    assert added == 9
    return {"filter_check.json": json.dumps(data, indent=2, ensure_ascii=False) + "\n",
            "filter_check.csv": stream.getvalue(), "filter_check.tex": "\n".join(tex) + "\n",
            "compact.tex": "\n".join(lines) + "\n"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verify", action="store_true", help="Check saved diagnostic files without writing")
    args = parser.parse_args()
    data = calculate()
    for name, content in outputs(data).items():
        path = HERE / "tables" / name
        if args.verify:
            assert path.read_text() == content, name
        else:
            path.write_text(content)
    print(f"Replayed {data['original_calls_replayed']} calls and {data['original_question_answers_replayed']} question answers; original scores match. Checked all 15 B+filter cases, retaining parse failures as unavailable. GPU calls: 0.")


if __name__ == "__main__":
    main()
