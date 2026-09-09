"""Versioned report from immutable new calls and separately derived historical syntax recovery."""

import argparse
import csv
import hashlib
import html
import json
from pathlib import Path

from scripts.report_compact_story import anchors_for, build, graph
from story_projection_onto import compact_story_v2 as p
from story_projection_onto.compact_syntax import recover
from story_projection_onto.manifest import write_json_atomic
from story_projection_onto.scorer_only import compact_story as old_scorer
from story_projection_onto.scorer_only import compact_story_v2 as scorer


def historical(path):
    data = json.loads(path.read_text())
    derived = []
    for call_id in ("3", "6"):
        call = next(c for c in data["calls"] if c["case_id"] == call_id)
        parsed, recovery = recover(call["raw_text"])
        assert call["parsed"] is None and recovery["applied"]
        story = "harbor" if call_id == "3" else "orchard"
        derived.append(
            dict(
                category="historical_syntax_recovered",
                story_id=story,
                task="possession",
                approach="B contextual",
                source_case_id=call_id,
                source_response_hash=call["response"]["response_sha256"],
                raw_text=call["raw_text"],
                parsed=parsed,
                syntax_recovery=recovery,
                evaluation=old_scorer.evaluate_answer(story, "possession", parsed),
                registered_acceptance=False,
                new_model_output=False,
            )
        )
    return {
        "original_report_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "original_results": data["results"],
        "derived": derived,
    }


def facts_list(facts, evaluation=None):
    e = html.escape
    rows = (evaluation or {}).get("rows", [])
    return (
        "<ol>"
        + "".join(
            "<li><code>"
            + e(json.dumps(f, ensure_ascii=False))
            + "</code>"
            + ("<br>" + e(rows[i]["status"] + "; " + "; ".join(rows[i]["issues"])) if rows else "")
            + "</li>"
            for i, f in enumerate(facts)
        )
        + "</ol>"
    )


def render(run, output, history_path=Path("reports/tables/compact_story_results.json")):
    data = build(run, protocol=p, evaluator=scorer)
    data["protocol"] = "compact-story-v2-explicit-null"
    data["historical"] = historical(history_path)
    call_map = {c["case_id"]: c for c in data["calls"]}
    rows = []

    def table_row(record, category):
        ev = record["evaluation"]
        call = call_map.get(record["source_case_id"], {}) if category == "new_model" else {}
        recovery = record.get("syntax_recovery") or call.get("syntax_recovery") or {}
        strict = recovery.get("strict_parseable", record.get("parseable", False))
        parsed = record.get("parsed") is not None
        result = dict(
            category=category,
            story=record["story_id"],
            task=record["task"],
            approach=record["approach"],
            source_call=record["source_case_id"],
            strict_json=strict,
            after_syntax_recovery=parsed,
            recovered=bool(recovery.get("applied")),
            predictions=ev["full"]["predicted"],
            references=ev["full"]["reference_count"],
            correct=ev["full"]["true_positive"],
            citations_valid=ev["citation_valid"],
            required_field_compliant=ev["format_compliant"],
            missing_intervals=sum(
                r["source_reference"] is not None
                and r["source_reference"].get("valid_from") is not None
                and (r["fact"].get("valid_from") is None or r["fact"].get("valid_until") is None)
                for r in ev["rows"]
            ),
            temporal_errors=sum(r["temporal_correct"] is False for r in ev["rows"]),
            attribution_errors=sum(r["epistemic_correct"] is False for r in ev["rows"]),
            unsupported_relationships=sum(
                r["status"] == "unsupported_explicit_relationship" for r in ev["rows"]
            ),
            irrelevant=sum(r["status"] == "supported_but_irrelevant" for r in ev["rows"]),
            unresolved=len(ev["unresolved"]),
            input_tokens=(call.get("response") or {}).get("prompt_tokens"),
            output_tokens=(call.get("response") or {}).get("completion_tokens"),
            request_seconds=call.get("request_seconds"),
        )
        for prefix, key in (("bare", "underlying"), ("qualified", "full")):
            result.update({prefix + "_" + m: ev[key][m] for m in ("precision", "recall", "f1")})
        return result

    rows += [table_row(r, "historical_raw") for r in data["historical"]["original_results"]]
    rows += [table_row(r, "historical_syntax_recovered") for r in data["historical"]["derived"]]
    rows += [table_row(r, "new_model") for r in data["results"]]
    output.mkdir(parents=True, exist_ok=True)
    for folder in ("tables", "figures"):
        (output / folder).mkdir(exist_ok=True)
    write_json_atomic(data, output / "tables/compact_story_v2_results.json")
    with (output / "tables/compact_story_v2_results.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator="\n")
        w.writeheader()
        w.writerows(rows)
    md = [
        "# Compact story v2 — explicit-null exploratory extraction",
        "",
        "A = query-blind extraction plus fixed CPU selection; B = direct contextual extraction. Not registered C1/C2, scientific acceptance, or ontology-construction evidence.",
        "",
        "## Results and comparison",
        "",
        "| Story / version | Mean qualified F1 A | Mean qualified F1 B |",
        "|---|---|---|",
    ]
    for story in p.stories():
        for category in ("historical_raw", "new_model"):
            group = [r for r in rows if r["story"] == story and r["category"] == category]
            if group:
                means = {
                    a: sum(r["qualified_f1"] for r in group if r["approach"].startswith(a)) / 3
                    for a in ("A", "B")
                }
                md.append(f"| {story} / {category} | {means['A']:.3f} | {means['B']:.3f} |")
    md += [
        "",
        "Historical strict failures remain unchanged. Derived comma-only recovery is not a new model response and is not folded into the historical means above. All predictions remain in their applicable denominators.",
        "",
        "Null and omitted optional values are both unspecified/not attributed for semantic comparison; missing required keys are separately reported as format failures. Neither null nor omission supplies a missing known interval or believer. References, normalization and examples were frozen before inference.",
        "",
        "## Every question and approach",
        "",
        "| Category | Story / task | Approach | Strict / recovered parse | Bare P/R/F1 | Qualified P/R/F1 | Citations |",
        "|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        measures = [
            " / ".join(f"{r[prefix + '_' + m]:.3f}" for m in ("precision", "recall", "f1"))
            for prefix in ("bare", "qualified")
        ]
        md.append(
            f"| {r['category']} | {r['story']} / {r['task']} | {r['approach']} | {r['strict_json']} / {r['after_syntax_recovery']} | {measures[0]} | {measures[1]} | {r['citations_valid']}/{r['predictions']} |"
        )
    e = html.escape
    body = [
        '<!doctype html><meta charset="utf-8"><title>Compact story v2</title><style>body{font:16px system-ui;margin:24px;color:#192b3c}.compare{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:12px}.panel{border:1px solid #b8c4d0;padding:10px}svg{width:100%;font:14px system-ui}svg:fullscreen{background:white;width:100vw;height:100vh}pre,code{white-space:pre-wrap;overflow-wrap:anywhere}li{margin:12px 0}button{padding:8px}h2{border-top:2px solid #496c88;padding-top:15px}@media(max-width:1200px){.compare{grid-template-columns:1fr}}</style><h1>Compact story v2</h1><p>Reference, A selection, and B contextual answers share exact-name anchors. Enlarge graphs for labels. Edges: green complete target match; amber unresolved; red error/irrelevant. Null means unspecified/not attributed, not false or timeless.</p>'
    ]
    for sid, story in p.stories().items():
        md += (
            ["", "## " + story["title"], ""]
            + [f"- {k}: {v}" for k, v in story["evidence"].items()]
            + [""]
        )
        body += (
            ["<h2>" + e(story["title"]) + "</h2><ol>"]
            + [f"<li>{k}: {e(v)}</li>" for k, v in story["evidence"].items()]
            + ["</ol>"]
        )
        for task, question in p.questions(story).items():
            pair = [r for r in data["results"] if r["story_id"] == sid and r["task"] == task]
            refs = scorer.reference_for(sid, task)
            collections = [refs] + [(r["parsed"] or {}).get("facts", []) for r in pair]
            anchors = anchors_for(collections)
            md += [
                "### " + question,
                "",
                "Reference answers:",
                "```json",
                json.dumps(refs, indent=2),
                "```",
                "",
            ]
            body += [
                "<h3>"
                + e(question)
                + '</h3><div class="compare"><section class="panel"><h4>Reference</h4>',
                graph(refs, anchors),
                facts_list(refs),
                "</section>",
            ]
            for r in pair:
                ev = r["evaluation"]
                facts = (r["parsed"] or {}).get("facts", [])
                call = call_map.get(r["source_case_id"], {})
                recovery = call.get("syntax_recovery", {})
                md += [
                    r["approach"] + ":",
                    "",
                    f"Strict parse: {recovery.get('strict_parseable', False)}; after permitted recovery: {r['parseable']}.",
                    "",
                ]
                for row in ev["rows"]:
                    md += [
                        "- `"
                        + json.dumps(row["fact"])
                        + "` — "
                        + row["status"]
                        + "; "
                        + "; ".join(row["issues"])
                    ]
                md += ["", "Missing: `" + json.dumps(ev["full"]["missing"]) + "`.", ""]
                if not r["parseable"]:
                    md += [
                        "Raw response (not completed or repaired semantically):",
                        "```text",
                        r["raw_source_text"],
                        "```",
                        "",
                    ]
                body += [
                    f'<section class="panel" id="{sid}-{task}-{r["approach"][0]}"><h4>{e(r["approach"])}</h4>',
                    graph(facts, anchors, ev),
                    facts_list(facts, ev),
                    "<p>"
                    + e(
                        f"Qualified P/R/F1 {ev['full']['precision']:.3f}/{ev['full']['recall']:.3f}/{ev['full']['f1']:.3f}; strict JSON {recovery.get('strict_parseable', False)}; usable after recovery {r['parseable']}"
                    )
                    + "</p>",
                    "<details><summary>Raw source, exact recovery edits, missing facts and selection trace</summary><pre>"
                    + e(
                        json.dumps(
                            {
                                "raw_text": r["raw_source_text"],
                                "syntax_recovery": recovery,
                                "missing": ev["full"]["missing"],
                                "selection_trace": r["selection_trace"],
                            },
                            indent=2,
                        )
                    )
                    + "</pre></details></section>",
                ]
            body += ["</div>"]
        base = next(
            (
                c
                for c in data["calls"]
                if p.cases()[c["case_id"]]["story_id"] == sid
                and p.cases()[c["case_id"]]["task"] == "all"
            ),
            None,
        )
        facts = ((base or {}).get("parsed") or {}).get("facts", [])
        body += [
            f'<h3>Actual complete pre-extraction: {e(story["title"])}</h3><section class="panel" id="{sid}-preextract">',
            graph(facts, anchors_for([facts]), (base or {}).get("evaluation")),
            facts_list(facts, (base or {}).get("evaluation")),
            "</section>",
        ]
    md += ["## Historical syntax recovery (derived only)", ""]
    body += ["<h2>Historical syntax recovery — not new model success</h2>"]
    for r in data["historical"]["derived"]:
        ev = r["evaluation"]
        text = f"{r['story_id']}: {ev['full']['true_positive']} complete matches / {ev['full']['predicted']} predictions / {ev['full']['reference_count']} references. P/R/F1 {ev['full']['precision']:.3f}/{ev['full']['recall']:.3f}/{ev['full']['f1']:.3f}."
        md += [
            text,
            "",
            "Exact edit record: `" + json.dumps(r["syntax_recovery"]["edits"]) + "`.",
            "",
            "Recovered records (all retained):",
            "```json",
            json.dumps(r["parsed"], indent=2),
            "```",
            "",
        ]
        facts = r["parsed"]["facts"]
        body += [
            "<h3>" + e(text) + "</h3>",
            graph(facts, anchors_for([facts]), ev),
            facts_list(facts, ev),
            "<details><summary>Immutable parent and exact recovery</summary><pre>"
            + e(json.dumps(r, indent=2))
            + "</pre></details>",
        ]
    md += [
        "## Calls, allocation and limitations",
        "",
        "| Call | Story/task | Strict / after recovery | Finish | Tokens in/out | Seconds |",
        "|---|---|---|---|---|---|",
    ]
    for c in data["calls"]:
        cfg = p.cases()[c["case_id"]]
        resp = c.get("response") or {}
        rec = c["syntax_recovery"]
        md.append(
            f"| {c['case_id']} | {cfg['story_id']}/{cfg['task']} | {rec['strict_parseable']} / {rec['recovered_parseable']} | {resp.get('finish_reason')} | {resp.get('prompt_tokens', 0)}/{resp.get('completion_tokens', 0)} | {c['request_seconds']:.3f} |"
        )
    a = data["allocation"]
    md += [
        "",
        f"Executed {len(data['calls'])}/12 single attempts; no adaptive tuning. New allocation {a['new_allocated_seconds']:.6f}s; cumulative {a['actual_allocated_seconds']:.6f}s. Open allocation/service journals {a['open_allocations']}/{a['open_service_journals']}. Stop reason: {a['stop_reason']}.",
        "",
        "Three query-blind calls precede all contextual transmissions. Baseline call costs are shared across its three selections, not counted three times. Per-call raw responses, token counts and request times are in the canonical JSON. One pinned model, nonthinking sampling, 12,288 context, 2,048 output tokens for both approaches; no production-p95 inference from this batch.",
        "",
        "This small exploratory batch has no minimum score for display/completion. Harbor and Orchard are repeatedly developed material; Museum is fresh development evidence, not held-out generalization. Original scores, registered metrics, C0 competence and independent-review gates remain unchanged. No automatic follow-up service start.",
        "",
        "[Graphs](figures/compact_story_v2_comparison.html) · [CSV](tables/compact_story_v2_results.csv) · [Original report](COMPACT_STORY_RESULTS.md).",
        "",
        "Reproduce on CPU: `python -m scripts.report_compact_story_v2 --run <restricted-run> --output reports`.",
    ]
    (output / "COMPACT_STORY_V2_RESULTS.md").write_text("\n".join(md) + "\n")
    (output / "figures/compact_story_v2_comparison.html").write_text("\n".join(body))
    files = [
        "COMPACT_STORY_V2_RESULTS.md",
        "tables/compact_story_v2_results.csv",
        "tables/compact_story_v2_results.json",
        "figures/compact_story_v2_comparison.html",
    ]
    write_json_atomic(
        {f: hashlib.sha256((output / f).read_bytes()).hexdigest() for f in files},
        output / "tables/compact_story_v2_manifest.json",
    )
    return data


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True, type=Path)
    parser.add_argument("--output", type=Path, default=Path("reports"))
    args = parser.parse_args()
    render(args.run, args.output)
