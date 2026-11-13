"""Render every actual published-prose output; strict and semantic assessments stay separate."""

import argparse
import csv
import hashlib
import html
import json
import math
from pathlib import Path

from scripts.report_compact_story import anchors_for as endpoint_inventory
from scripts.report_compact_story import build
from scripts.report_compact_story import graph as base_graph
from scripts.report_compact_story_v2 import facts_list
from story_projection_onto import real_text_poc as p
from story_projection_onto.contracts import canonical_sha256
from story_projection_onto.manifest import write_json_atomic
from story_projection_onto.scorer_only import real_text_poc as scorer


def assessment_key(sid, fact):
    return canonical_sha256({"story": sid, "fact": fact})


def graph(facts, anchors, evaluation=None):
    # Display abbreviation only. Endpoint identity and full assertion text remain unchanged.
    return base_graph(facts, anchors, evaluation, max_label_lines=3)


def anchors_for(collections):
    """Stable exact-string grid avoids overlapping long prose-node labels on a circle."""
    names = sorted(endpoint_inventory(collections))
    columns = min(4, max(1, math.ceil(math.sqrt(len(names)))))
    rows = max(1, math.ceil(len(names) / columns))
    return {
        name: (
            100 + i % columns * (650 / max(1, columns - 1)),
            75 + i // columns * (430 / max(1, rows - 1)),
        )
        for i, name in enumerate(names)
    }


def triple_score(row, prefix):
    values = [row[prefix + "_" + k] for k in ("precision", "recall", "f1")]
    return "not measurable" if values[0] is None else "/".join(f"{v:.3f}" for v in values)


def assessments(data, path):
    catalog = json.loads(path.read_text()) if path.exists() else {}
    actual = {}
    for call in data["calls"]:
        sid = p.cases()[call["case_id"]]["story_id"]
        for index, fact in enumerate((call.get("parsed") or {}).get("facts", []), 1):
            actual.setdefault(assessment_key(sid, fact), []).append(
                dict(
                    case_id=call["case_id"],
                    fact_index=index,
                    response_sha256=call["response"]["response_sha256"],
                )
            )
    for key, note in catalog.items():
        assert key in actual, "manual assessment does not belong to a received record"
        assert note["source_responses"] == actual[key], "manual assessment response binding differs"
    base_records = [
        dict(
            story_id=p.cases()[c["case_id"]]["story_id"],
            task="all",
            source_case_id=c["case_id"],
            evaluation=scorer.evaluate(c["case_id"], c.get("parsed")),
        )
        for c in data["calls"]
        if p.cases()[c["case_id"]]["task"] == "all"
    ]
    for r in data["results"] + base_records:
        sid, task = r["story_id"], r["task"]
        notes = []
        for row in r["evaluation"]["rows"]:
            key = assessment_key(sid, row["fact"])
            if key in catalog:
                note = catalog[key]
                assert note["story_id"] == sid and note["fact"] == row["fact"]
                assert note["support"] in (
                    "supported",
                    "partial",
                    "unsupported",
                    "contradicted",
                    "unresolved",
                )
                assert all(0 <= i < len(scorer.concepts(sid)) for i in note["covered_concepts"])
                assert set(note["evidence_ids"]) <= set(p.stories()[sid]["evidence"])
            else:
                matches = [
                    i
                    for i, g in enumerate(scorer.concepts(sid))
                    if any(scorer.equivalent(row["fact"], a, sid, True) for a in g)
                ]
                note = dict(
                    story_id=sid,
                    fact=row["fact"],
                    support="supported" if matches else "unresolved",
                    qualifications="correct" if matches else "unresolved",
                    covered_concepts=matches,
                    relevant_to=[
                        t
                        for t in p.QUESTIONS[sid]
                        if any(
                            scorer.equivalent(row["fact"], a, sid, True)
                            for g in scorer.targets(sid, t)
                            for a in g
                        )
                    ],
                    evidence_ids=row["fact"].get("evidence_ids", [])
                    if isinstance(row["fact"], dict)
                    else [],
                    author="automatic match to frozen Codex reference" if matches else "unassessed",
                    note="Frozen reference match, not independent verification."
                    if matches
                    else "Unmatched wording awaits source-based assessment; not automatically false.",
                )
            notes.append({**note, "relevant": task in note["relevant_to"], "assessment_key": key})
        r["semantic_assessments"] = notes
    data["manual_assessment_catalog"] = catalog
    data["preextract_assessments"] = base_records


def render(run, output):
    data = build(run, protocol=p, evaluator=scorer)
    data["protocol"] = "published-narrative-prose-poc-v1"
    assessments(data, run / "semantic-assessments.json")
    calls = {c["case_id"]: c for c in data["calls"]}
    terminal = json.loads((run / "terminal.json").read_text())
    data["resources"] = {
        "sample_count": len(terminal["resource_samples"]),
        "sampled_peaks": {
            k: max((s[k] for s in terminal["resource_samples"]), default=0)
            for k in (
                "gpu_vram_bytes",
                "process_ram_bytes",
                "project_storage_bytes",
                "cpu_worker_count",
            )
        },
        "observed_violations": [
            s["violations"] for s in terminal["resource_samples"] if s["violations"]
        ],
        "measurement_scope": "Sampled project process peaks, not proof of an unobserved instantaneous maximum. Storage uses full censuses plus loss-detecting write tracking.",
    }
    table = []
    for r in data["results"]:
        ev = r["evaluation"]
        notes = r["semantic_assessments"]
        c = calls.get(r["source_case_id"], {})
        rec = c.get("syntax_recovery", {})
        target_ids = [
            i
            for i, g in enumerate(scorer.concepts(r["story_id"]))
            if g in scorer.targets(r["story_id"], r["task"])
        ]
        covered = set(
            i
            for n in notes
            if n["support"] in ("supported", "partial")
            for i in n["covered_concepts"]
        ) & set(target_ids)
        row = dict(
            story=r["story_id"],
            task=r["task"],
            approach=r["approach"],
            source_call=r["source_case_id"],
            executed=r["executed"],
            complete=r["transport_complete"],
            finish=r["finish_reason"],
            strict_json=rec.get("strict_parseable", False),
            after_syntax_recovery=r["parseable"],
            syntax_recovered=rec.get("applied", False),
            metric_status="scored" if r["parseable"] else "unavailable_parse_failure",
            predictions=ev["full"]["predicted"],
            references=ev["full"]["reference_count"],
            citations_valid=ev["citation_valid"],
            format_compliant=ev["format_compliant"],
            semantically_supported=sum(n["support"] == "supported" for n in notes),
            partially_supported=sum(n["support"] == "partial" for n in notes),
            unsupported=sum(n["support"] == "unsupported" for n in notes),
            contradicted=sum(n["support"] == "contradicted" for n in notes),
            unresolved=sum(n["support"] == "unresolved" for n in notes),
            qualification_errors=sum(n["qualifications"] == "incorrect" for n in notes),
            supported_irrelevant=sum(
                n["support"] == "supported" and not n["relevant"] for n in notes
            ),
            semantic_targets_covered=len(covered),
            semantic_coverage=len(covered) / len(target_ids),
            source_input_tokens=r["input_tokens"],
            source_output_tokens=r["output_tokens"],
            source_request_seconds=r["request_seconds"],
        )
        for prefix, key in (("relationship", "underlying"), ("strict_qualified", "full")):
            row.update(
                {
                    prefix + "_" + m: ev[key][m] if r["parseable"] else None
                    for m in ("precision", "recall", "f1")
                }
            )
        if not r["parseable"]:
            for key in ("predictions", "semantic_targets_covered", "semantic_coverage"):
                row[key] = None
        table.append(row)
    data["summary"] = {
        "executed_calls": len(data["calls"]),
        "strict_json_calls": sum(
            c.get("syntax_recovery", {}).get("strict_parseable", False) for c in data["calls"]
        ),
        "usable_calls": sum(c.get("parsed") is not None for c in data["calls"]),
        "recovered_calls": sum(
            c.get("syntax_recovery", {}).get("applied", False) for c in data["calls"]
        ),
        "input_tokens": sum(
            (c.get("response") or {}).get("prompt_tokens", 0) for c in data["calls"]
        ),
        "output_tokens": sum(
            (c.get("response") or {}).get("completion_tokens", 0) for c in data["calls"]
        ),
    }
    output.mkdir(exist_ok=True, parents=True)
    for folder in ("tables", "figures"):
        (output / folder).mkdir(exist_ok=True)
    data["table"] = table
    write_json_atomic(data, output / "tables/real_text_proof_of_concept.json")
    with (output / "tables/real_text_proof_of_concept.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(table[0]), lineterminator="\n")
        w.writeheader()
        w.writerows(table)
    md = [
        "# Published narrative prose: compact extraction proof of concept",
        "",
        "Exploratory extraction, not factual verification of real events, registered C1/C2 acceptance, or ontology-construction efficacy. A = query-blind extraction then fixed CPU selection; B = direct contextual extraction. References and manual assessments are Codex-authored, not independent human review.",
        "",
        f"Executed {data['summary']['executed_calls']}/9 calls: {data['summary']['usable_calls']} usable responses, {data['summary']['strict_json_calls']} strict JSON successes, {data['summary']['recovered_calls']} syntax recoveries. Total request tokens: {data['summary']['input_tokens']} input / {data['summary']['output_tokens']} output. All received outputs and failed raw responses are retained below.",
        "",
        "## Actual results",
        "",
        "| Passage / question | Method | Strict qualified P/R/F1 | Supported / partial / unresolved records | Semantic target coverage |",
        "|---|---|---|---|---|",
    ]
    for r in table:
        md.append(
            f"| {r['story']}/{r['task']} | {r['approach'][0]} | {triple_score(r, 'strict_qualified')} | {r['semantically_supported']}/{r['partially_supported']}/{r['unresolved']} of {r['predictions'] if r['predictions'] is not None else 'unavailable'} | {str(r['semantic_targets_covered']) + '/' + str(r['references']) if r['predictions'] is not None else 'unavailable'} |"
        )
    overview = run / "interpretation.md"
    if overview.exists():
        md += ["", overview.read_text(), ""]
    md += [
        "",
        "Strict scores measure agreement with the frozen scoped annotation, not a rate of factual truth. Every prediction remains in its applicable denominator. Semantic review can recognize explicit meaning in another representation without changing strict scores; partial support and ambiguous wording are not full correctness. Coverage counts unique explicitly preserved target concepts, not a new reference set.",
        "Failed parsing is a missing result, not a zero-precision empty graph. Table metrics are unavailable in that case; immutable raw call evaluations retain their original empty-input accounting. No malformed output is repaired beyond the frozen trailing-comma rule. Null bounds mean unsupported/unstated duration, not timelessness or falsehood.",
        "",
        "## Format, citations and errors",
        "",
        "| Passage/question | A/B | Strict / usable JSON | Format / citations / predictions | Bare relationship P/R/F1 | Unsupported / contradicted / qualification errors | Supported irrelevant |",
        "|---|---|---|---|---|---|---|",
    ]
    for r in table:
        md.append(
            f"| {r['story']}/{r['task']} | {r['approach'][0]} | {r['strict_json']}/{r['after_syntax_recovery']} | {r['format_compliant']}/{r['citations_valid']}/{r['predictions']} | {triple_score(r, 'relationship')} | {r['unsupported']}/{r['contradicted']}/{r['qualification_errors']} | {r['supported_irrelevant']} |"
        )
    for cid, note in data["manual_diagnosis"].get("cases", {}).items():
        md += ["", "### Failed-response manual diagnosis — call " + cid, "", note["syntax"], ""]
        md += ["- " + n for n in note["manual_findings"]]
        md += [""]
    e = html.escape
    body = [
        '<!doctype html><meta charset="utf-8"><title>Published prose proof of concept</title><style>body{font:16px system-ui;margin:24px;color:#192b3c}.compare{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:12px}.panel{border:1px solid #abc;padding:10px}pre{white-space:pre-wrap;overflow-wrap:anywhere;background:#f3f5f7;padding:15px}svg{width:100%}svg:fullscreen{background:white;width:100vw;height:100vh}code{white-space:pre-wrap;overflow-wrap:anywhere}li{margin:12px 0}h2{border-top:2px solid #456;padding-top:12px}@media(max-width:1200px){.compare{grid-template-columns:1fr}}</style><h1>Published narrative prose: exploratory extraction</h1><p>All outputs and errors are displayed. Green: strict reference match. Red: frozen structural/qualification mismatch or irrelevant record. Amber: unresolved automated matching, which may still preserve meaning; read the separate semantic assessment. Enlarge graphs for labels. No independent human review or general reliability claim.</p>'
    ]
    body += [
        "<p>Long labels are abbreviated visually only: hover or read the complete numbered assertions. Identical literal endpoint strings share anchors; aliases are not merged. Parse failures are unavailable results, not empty correct graphs.</p>"
    ]
    if overview.exists():
        body += ["<pre>" + e(overview.read_text()) + "</pre>"]
    body += [
        "<h2>All comparisons</h2><table border='1' cellpadding='6'><tr><th>Passage/question</th><th>Method</th><th>Strict P/R/F1</th><th>Semantic target coverage</th></tr>"
    ]
    for row in table:
        body += [
            "<tr><td>"
            + e(row["story"] + "/" + row["task"])
            + "</td><td>"
            + row["approach"][0]
            + "</td><td>"
            + triple_score(row, "strict_qualified")
            + "</td><td>"
            + (
                str(row["semantic_targets_covered"]) + "/" + str(row["references"])
                if row["predictions"] is not None
                else "unavailable"
            )
            + "</td></tr>"
        ]
    body += ["</table>"]
    for cid, note in data["manual_diagnosis"].get("cases", {}).items():
        body += [
            "<h3>Failed-response manual diagnosis — call "
            + e(cid)
            + "</h3><p>"
            + e(note["syntax"])
            + "</p><ul>"
        ]
        body += ["<li>" + e(n) + "</li>" for n in note["manual_findings"]]
        body += ["</ul>"]
    for sid in ("fable", "alice", "holmes"):
        source = p.stories()[sid]
        md += [
            "",
            "## " + source["title"] + " — " + source["section"],
            "",
            f"{source['author']}"
            + (f"; translated by {source['translator']}" if source["translator"] else "")
            + f". [Source]({source['source_url']}); retrieved {source['retrieval_date']}. Excerpt SHA-256 `{source['excerpt_sha256']}`; {source['word_count']} words.",
            "",
            "Exact contiguous excerpt (original line breaks retained):",
            "```text",
            source["excerpt"],
            "```",
            "",
            f"[Source attribution and full supplied license notices](../{source['notice_file']}). {source['rights']}",
            "",
            "Evidence spans (numbers are added labels; concatenation reproduces the exact excerpt):",
            "",
        ]
        body += [
            "<h2>" + e(source["title"] + " — " + source["section"]) + "</h2>",
            "<p>"
            + e(source["author"])
            + ' · <a href="'
            + e(source["source_url"])
            + '">Gutenberg source</a> · <a href="../../'
            + e(source["notice_file"])
            + '">Original attribution and license</a></p>',
            "<pre>" + e(source["excerpt"]) + "</pre>",
        ]
        for k, v in source["evidence"].items():
            md += ["**" + k + "**", "", "```text", v, "```", ""]
        body += [
            "<details><summary>Exact evidence spans and provenance</summary><pre>"
            + e(json.dumps(source, indent=2))
            + "</pre></details>"
        ]
        for task, question in p.QUESTIONS[sid].items():
            pair = [r for r in data["results"] if r["story_id"] == sid and r["task"] == task]
            refs = scorer.reference_for(sid, task)
            anchors = anchors_for([refs] + [(r["parsed"] or {}).get("facts", []) for r in pair])
            md += [
                "### " + question,
                "",
                "Codex-authored reference facts (alternatives in canonical JSON frozen rules):",
                "```json",
                json.dumps(refs, indent=2),
                "```",
                "",
            ]
            body += [
                "<h3>"
                + e(question)
                + '</h3><div class="compare"><section class="panel"><h4>Codex reference</h4>',
                graph(refs, anchors),
                facts_list(refs),
                "</section>",
            ]
            for r in pair:
                facts = (r["parsed"] or {}).get("facts", [])
                md += [
                    "#### " + r["approach"],
                    "",
                    "Actual model-authored records and source-based assessment:",
                    "",
                ]
                for i, (f, n) in enumerate(zip(facts, r["semantic_assessments"], strict=True), 1):
                    md += [
                        f"{i}. `{json.dumps(f)}` — {n['support']}; qualifiers {n['qualifications']}; relevant {n['relevant']}. {n['note']}",
                        "",
                    ]
                md += [
                    "Missing strict targets: `"
                    + json.dumps(r["evaluation"]["full"]["missing"])
                    + "`.",
                    "",
                ]
                if not r["parseable"]:
                    md += [
                        "Structural failure: " + str(r["failure"]),
                        "",
                        "Unusable/unexecuted raw response (not completed by CPU):",
                        "```text",
                        r["raw_source_text"],
                        "```",
                        "",
                    ]
                body += [
                    f'<section class="panel" id="{sid}-{task}-{r["approach"][0]}"><h4>'
                    + e(r["approach"])
                    + "</h4>",
                    graph(facts, anchors, r["evaluation"]),
                    facts_list(facts, r["evaluation"]),
                ]
                if not r["parseable"]:
                    body += [
                        "<p>Structural failure: "
                        + e(str(r["failure"]))
                        + "</p><pre>"
                        + e(r["raw_source_text"])
                        + "</pre>"
                    ]
                body += ["<h4>Separate semantic assessment</h4><ol>"]
                for n in r["semantic_assessments"]:
                    body += [
                        "<li>"
                        + e(
                            f"{n['support']}; qualifiers {n['qualifications']}; relevant {n['relevant']}. {n['note']}"
                        )
                        + "</li>"
                    ]
                body += [
                    "</ol><details><summary>Raw response, syntax edits, strict metrics and selection trace</summary><pre>"
                    + e(
                        json.dumps(
                            {
                                "record": r,
                                "syntax_recovery": calls.get(r["source_case_id"], {}).get(
                                    "syntax_recovery"
                                ),
                            },
                            indent=2,
                        )
                    )
                    + "</pre></details></section>"
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
        notes = next(
            (
                r["semantic_assessments"]
                for r in data["preextract_assessments"]
                if r["story_id"] == sid
            ),
            [],
        )
        md += ["### Complete query-blind collection (before filtering)", ""]
        for i, (fact, note) in enumerate(zip(facts, notes, strict=True), 1):
            md += [f"{i}. `{json.dumps(fact)}` — {note['support']}; {note['note']}", ""]
        body += [
            "<h3>Complete actual query-blind collection</h3>",
            graph(facts, anchors_for([facts])),
            facts_list(facts),
            "<details><summary>Exact source output</summary><pre>"
            + e((base or {}).get("raw_text", "Unexecuted"))
            + "</pre></details>",
        ]
        body += ["<h4>Pre-extraction semantic notes (including unselected records)</h4><ol>"]
        body += ["<li>" + e(n["support"] + ": " + n["note"]) + "</li>" for n in notes]
        body += ["</ol>"]
    md += [
        "## Costs and limits",
        "",
        "| Call | Passage/task | Finish | Input/output tokens | Request seconds |",
        "|---|---|---|---|---|",
    ]
    for c in data["calls"]:
        v = p.cases()[c["case_id"]]
        r = c.get("response") or {}
        md.append(
            f"| {c['case_id']} | {v['story_id']}/{v['task']} | {r.get('finish_reason')} | {r.get('prompt_tokens', 0)}/{r.get('completion_tokens', 0)} | {c['request_seconds']:.6f} |"
        )
    a = data["allocation"]
    peaks = data["resources"]["sampled_peaks"]
    md += [
        "",
        f"{len(data['calls'])}/9 calls executed; unexecuted IDs: {sorted(set(p.cases()) - set(calls))}. Allocation {a['new_allocated_seconds']:.6f} seconds new / {a['actual_allocated_seconds']:.6f} cumulative. Open allocation/service journals {a['open_allocations']}/{a['open_service_journals']}; stop reason {a['stop_reason']}.",
        f"Sampled peaks across {data['resources']['sample_count']} observations: GPU VRAM {peaks['gpu_vram_bytes'] / 2**30:.3f} GiB; process RAM {peaks['process_ram_bytes'] / 2**30:.3f} GiB; project storage {peaks['project_storage_bytes'] / 2**30:.3f} GiB; CPU workers {peaks['cpu_worker_count']}. Observed resource violations: {data['resources']['observed_violations']}. These are observed, not continuous peak guarantees.",
        "",
        "One pinned Qwen3-8B-AWQ revision `4da05a8edb55c6046cce958586c33b61da07bb79`, unchanged nonthinking decoding, 12,288 context and common 2,048 output reserve. Query-blind costs are shared across two selections; no adaptive retry. Samples do not establish p95, held-out generalization or full-study feasibility. Familiar published books may have appeared in training.",
        "",
        "Temporal limitation: the fable says “shortly after this”; Holmes says “one day in the autumn of last year” and discusses past cases and future help. Null numerical validity avoids fabrication but does not preserve this chronology or duration. Relative order, conditional promises, and detailed speech content exceed some uses of this compact representation.",
        "",
        "Coverage caveats: references cover the declared questions, not every defensible detail. Pronoun/greeting targets, rhetorical questions and vocative dialogue introduce interpretation uncertainty. Codex-authored judgments are exploratory, not independent review. Original synthetic/C0 results and registered gates are unchanged.",
        "",
        "[Graphs](figures/real_text_proof_of_concept.html) · [CSV](tables/real_text_proof_of_concept.csv).",
        "",
        "CPU reproduction: `python -m scripts.report_real_text_poc --run <restricted-run> --output reports`.",
    ]
    (output / "REAL_TEXT_PROOF_OF_CONCEPT.md").write_text("\n".join(md) + "\n")
    (output / "figures/real_text_proof_of_concept.html").write_text("\n".join(body))
    files = [
        "REAL_TEXT_PROOF_OF_CONCEPT.md",
        "tables/real_text_proof_of_concept.json",
        "tables/real_text_proof_of_concept.csv",
        "figures/real_text_proof_of_concept.html",
    ]
    write_json_atomic(
        {f: hashlib.sha256((output / f).read_bytes()).hexdigest() for f in files},
        output / "tables/real_text_proof_of_concept_manifest.json",
    )
    return data


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True, type=Path)
    parser.add_argument("--output", type=Path, default=Path("reports"))
    args = parser.parse_args()
    render(args.run, args.output)
