"""CPU-only manuscript publisher. Reads retained tables; never evaluates model outputs.

Run: python -m scripts.build_poc_manuscript
Edit paper/manuscript_source.md, not its numerically resolved output.
"""

# ruff: noqa: E501, RUF001
from __future__ import annotations

import csv
import html
import json
import re
import subprocess
from copy import deepcopy
from pathlib import Path

from markdown_it import MarkdownIt
from pdfrw import PdfReader, PdfWriter
from pdfrw.buildxobj import pagexobj
from pdfrw.toreportlab import makerl
from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    BaseDocTemplate,
    Flowable,
    KeepTogether,
    PageBreak,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
)

from scripts import build_paper_figures as old

ROOT = old.ROOT
OUT = ROOT / "paper"
SOURCES = {
    "ladder": "reports/tables/simple_synthetic_results.json",
    "ladder_v2": "reports/tables/simple_synthetic_v2_results.json",
    "compact": "reports/tables/compact_story_v2_results.json",
    "prose": "reports/tables/real_text_proof_of_concept.json",
}
FONT = Path(old.findfont(old.FontProperties(family="DejaVu Sans")))
FONT_BOLD = Path(old.findfont(old.FontProperties(family="DejaVu Sans", weight="bold")))
METRIC_REGISTRY = {}


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n")


def number(key, value, source, spec=""):
    rendered = format(value, spec) if spec else str(value)
    METRIC_REGISTRY[key] = {"value": value, "source": source, "display": rendered}
    return rendered


def csv_table(name, rows):
    with (OUT / "tables" / (name + ".csv")).open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator="\n")
        w.writeheader()
        w.writerows(rows)
    write_json(OUT / "tables" / (name + ".json"), rows)


def md_table(headers, rows):
    return "\n".join(
        [
            "| " + " | ".join(headers) + " |",
            "| " + " | ".join("---" for _ in headers) + " |",
            *("| " + " | ".join(str(x) for x in r) + " |" for r in rows),
        ]
    )


def collect_tables(data):
    c, r = data["compact"], data["prose"]
    compact_rows, display = [], []
    for story in ["harbor", "orchard", "museum"]:
        for task in ["possession", "locations", "belief"]:
            pair = [
                next(
                    x
                    for x in c["results"]
                    if x["story_id"] == story and x["task"] == task and x["approach"].startswith(a)
                )
                for a in "AB"
            ]
            row = {"story": story, "question": task}
            cells = [story.title(), task]
            for a, result in zip("AB", pair, strict=True):
                for k, v in result["evaluation"]["full"].items():
                    if not isinstance(v, (int, float)):
                        continue
                    row[a + "_" + k] = v
                    number(
                        f"{story}_{task}_{a}_{k}",
                        v,
                        f"compact/results/{story}/{task}/{a}/evaluation/full/{k}",
                        ".3f" if k in {"precision", "recall", "f1"} else "",
                    )
                ev = result["evaluation"]["full"]
                cells += [
                    str(ev["predicted"]),
                    "/".join(f"{ev[k]:.3f}" for k in ["precision", "recall", "f1"]),
                ]
            compact_rows.append(row)
            display.append(cells)
    csv_table("compact_comparison", compact_rows)
    csv_table("published_comparison", r["table"])
    components = []
    for x in c["results"]:
        ev = x["evaluation"]
        call = next(call for call in c["calls"] if call["case_id"] == x["source_case_id"])
        components.append(
            dict(
                story=x["story_id"],
                task=x["task"],
                approach=x["approach"],
                source_call=x["source_case_id"],
                strict_json=call["syntax_recovery"]["strict_parseable"],
                syntax_recovered=call["syntax_recovery"]["applied"],
                predictions=ev["full"]["predicted"],
                citation_valid=ev["citation_valid"],
                format_compliant=ev["format_compliant"],
                source_supported=ev["source_supported"],
                unresolved=len(ev["unresolved"]),
                irrelevant=sum(y["status"] == "supported_but_irrelevant" for y in ev["rows"]),
                bare_precision=ev["underlying"]["precision"],
                bare_recall=ev["underlying"]["recall"],
                bare_f1=ev["underlying"]["f1"],
            )
        )
    csv_table("compact_components", components)
    write_json(
        OUT / "tables/compact_assertion_assessments.json",
        [
            {
                k: x[k]
                for k in ["story_id", "task", "approach", "source_response_hash", "evaluation"]
            }
            for x in c["results"]
        ],
    )
    tables = {
        "compact": md_table(["Story", "Question", "A n", "A P/R/F1", "B n", "B P/R/F1"], display)
    }
    display = []
    for row in r["table"]:
        key = f"{row['story']}_{row['task']}_{row['approach'][0]}"
        for k, v in row.items():
            if isinstance(v, (float, int)) and not isinstance(v, bool):
                number(
                    key + "_" + k,
                    v,
                    f"prose/table/{key}/{k}",
                    ".3f" if isinstance(v, float) else "",
                )
        ok = row["metric_status"] == "scored"

        def fmt(k, row=row, ok=ok):
            return f"{row[k]:.3f}" if ok else "NA"

        status = (
            f"{row['semantically_supported']}/{row['partially_supported']}/{row['unsupported']}/{row['unresolved']}"
            if ok
            else "NA"
        )
        display.append(
            [
                row["story"].title() + " / " + row["task"],
                row["approach"][0],
                row["predictions"] if ok else "NA",
                "/".join(fmt("strict_qualified_" + k) for k in ["precision", "recall", "f1"]),
                status,
                f"{row['semantic_targets_covered']}/{row['references']}" if ok else "NA",
            ]
        )
    tables["prose"] = md_table(
        ["Passage / question", "Method", "n", "Strict P/R/F1", "S/P/X/U", "Coverage"], display
    )
    costs = []
    for key, title in [
        ("ladder", "Simple ladder"),
        ("ladder_v2", "Simple ladder v2"),
        ("compact", "Compact stories v2"),
        ("prose", "Published prose"),
    ]:
        d = data[key]
        acc = d.get("allocation", d.get("accounting"))
        calls = d.get("calls", d.get("rows"))

        def get(c, field):
            return c.get("response", {}).get(
                {"input_tokens": "prompt_tokens", "output_tokens": "completion_tokens"}.get(
                    field, field
                ),
                c.get(field, 0),
            )

        cost = dict(
            batch=title,
            calls=len(calls),
            input_tokens=sum(get(x, "input_tokens") for x in calls),
            output_tokens=sum(get(x, "output_tokens") for x in calls),
            request_seconds=sum(x["request_seconds"] for x in calls),
            allocated_seconds=acc["new_allocated_seconds"],
        )
        costs.append(cost)
        for k, v in cost.items():
            if k != "batch":
                number(
                    key + "_" + k,
                    v,
                    f"{key}/unique calls and allocation/new_allocated_seconds",
                    ".2f" if isinstance(v, float) else "",
                )
    csv_table("allocation", costs)
    tables["cost"] = md_table(
        ["Batch (not pooled)", "Calls", "Input / output tokens", "Request s", "Allocated s"],
        [
            [
                x["batch"],
                x["calls"],
                f"{x['input_tokens']} / {x['output_tokens']}",
                f"{x['request_seconds']:.2f}",
                f"{x['allocated_seconds']:.2f}",
            ]
            for x in costs
        ],
    )
    # Main text shows only the two comparison batches; S3 retains all four rows.
    cost_lines = tables["cost"].splitlines()
    tables["main_cost"] = "\n".join(cost_lines[:2] + cost_lines[-2:])
    number(
        "main_cost",
        sum(x["allocated_seconds"] for x in costs[2:]),
        "sum of compact and prose batch allocation only",
        ".2f",
    )
    number(
        "project_cost",
        r["allocation"]["actual_allocated_seconds"],
        "prose/allocation/actual_allocated_seconds",
        ".2f",
    )
    for key in ["compact", "prose"]:
        d = data[key]
        number(
            key + "_strict_json",
            sum(x["syntax_recovery"]["strict_parseable"] for x in d["calls"]),
            f"{key}/calls/syntax_recovery/strict_parseable",
        )
        for story in d["stories"]:
            s = d["stories"][story]
            number(
                story + "_sentences", len(s["evidence"]), f"{key}/stories/{story}/evidence count"
            )
            if "word_count" in s:
                number(story + "_words", s["word_count"], f"{key}/stories/{story}/word_count")
    # Historical annotations remain their own records, never rescored or pooled.
    write_json(OUT / "tables/historical_syntax_recovery.json", c["historical"])
    for key in ["ladder", "ladder_v2"]:
        csv_table(key, data[key]["rows"])
    # Derived descriptive within-story averages, not estimates over independent trials.
    for story in c["stories"]:
        for a in "AB":
            vals = [
                x["evaluation"]["full"]["f1"]
                for x in c["results"]
                if x["story_id"] == story and x["approach"].startswith(a)
            ]
            number(
                f"{story}_{a}_mean",
                sum(vals) / len(vals),
                f"arithmetic mean of compact/{story}/{a} three question F1 values",
                ".3f",
            )
    number(
        "compact_irrelevant_answers",
        sum(
            any(y["status"] == "supported_but_irrelevant" for y in x["evaluation"]["rows"])
            for x in c["results"]
            if x["approach"].startswith("B")
        ),
        "compact B evaluation rows containing supported_but_irrelevant",
    )
    # The interval summary is already a canonical, verified finding, not a new scorer.
    finding = next(s for s in c["findings"] if "emitted facts" in s)
    a, b = re.search(r"(\d+)/(\d+) emitted facts", finding).groups()
    number("interval_retained", int(a), "compact/findings: unique-call interval audit numerator")
    number("interval_eligible", int(b), "compact/findings: unique-call interval audit denominator")
    return tables


def subset(panel, ids):
    p = deepcopy(panel)
    p["records"] = [r for r in p["records"] if r["index"] in ids]
    p["omitted_count"] = p["original_count"] - len(p["records"])
    return p


def compact_source(p, story, ids, y, label):
    p.text(12, y, label, size=8.5, bold=True, color=old.BLUE)
    y += 14
    for eid in ids:
        p.text(12, y, eid, size=8.5, bold=True, color=old.BLUE)
        y = p.para(32, y, " ".join(story["evidence"][eid].split()), old.W - 46, size=8.5) + 3
    return y + 3


def mini_edge(p, panel, r, x, y, w, h=36):
    """Faceted graph; endpoint strings are untouched. IDs join only exact strings."""
    fact = r["fact"]
    color, symbol, style = old.STATUS[r["status"]]
    sw, ow = (66, 73) if w < 250 else (104, 218)
    cy = y + h / 2

    def box(cx, value, width):
        lines = old.wrap(value, width - 6, 8.5)
        bh = max(23, len(lines) * 11.05 + 6)
        p.rect(cx - width / 2, cy - bh / 2, width, bh, fc="#F3F6F8", ec="#9DADB7")
        p.text(cx, cy - len(lines) * 11.05 / 2, "\n".join(lines), size=8.5, align="center")

    box(x + sw / 2, fact["subject"], sw)
    box(x + w - ow / 2, fact["object"], ow)
    left, right = x + sw + 2, x + w - ow - 2
    arrow = old.FancyArrowPatch(
        (left, cy + 9),
        (right, cy + 9),
        arrowstyle="-|>",
        mutation_scale=9,
        color=color,
        linestyle=style,
        lw=1.1,
    )
    p.ax.add_patch(arrow)
    label = symbol + " " + fact["relation"].replace("_", " ")
    lines = old.wrap(label, right - left + 6, 8.5)
    t = p.text((left + right) / 2, y, "\n".join(lines), size=8.5, align="center", color=color)
    t.set_bbox(dict(facecolor="white", edgecolor="none", pad=0.3))
    t2 = p.text(
        (left + right) / 2,
        y + h - 9,
        " ".join(fact["evidence_ids"]),
        size=8.5,
        align="center",
        color=color,
    )
    old.register_edge(p, panel, r, [arrow, t, t2])
    if panel["id"] not in p.panels:
        p.panels.append(panel["id"])


def finish_compact(p, y, legend):
    p.text(12, y, legend, size=8.5, color=old.MUTED)
    h = y + 17
    assert h <= 180 / 25.4 * 72, (p.name, h)
    p.height = h
    p.fig.set_size_inches(old.W / 72, h / 72)
    p.ax.set_ylim(h, 0)
    return p


def compact_figures(data, panels):
    figures = []
    p = old.Plate(
        "figure_1_fable",
        "1 · A fable, an action request, two actual outputs",
        "A: text → pre-extraction → fixed selection    B: text + question → extraction",
        500,
    )
    y = compact_source(
        p,
        data["real"]["stories"]["fable"],
        ["S1", "S2", "S3", "S4", "S5"],
        54,
        "SOURCE EXCERPT · S1–S5; final speech S6 omitted here, supplied to model",
    )
    p.text(
        12,
        y,
        "REQUEST · Physical running, capture, binding, gnawing and release",
        size=9,
        bold=True,
    )
    y += 18
    a = subset(panels["fable_A"], [2, 3, 4, 5, 7, 8])
    b = panels["fable_B"]
    col = (old.W - 36) / 2
    p.text(12, y, "A · selected pre-extraction", bold=True)
    p.text(24 + col, y, "B · contextual extraction", bold=True)
    y += 14
    p.text(12, y, "6 of 8 records shown", size=8.5)
    p.text(24 + col, y, "All 5 records shown", size=8.5)
    y += 17
    for i in range(6):
        mini_edge(p, a, a["records"][i], 12, y, col, 35)
        if i < len(b["records"]):
            mini_edge(p, b, b["records"][i], 24 + col, y, col, 35)
        y += 38
    y = (
        p.para(
            12,
            y + 1,
            "A's spare loses conditional scope (and has a missing valid_until key). B omits the Lion's early capture and release of the Mouse. No missing edge is drawn.",
            old.W - 24,
            size=8.5,
        )
        + 5
    )
    figures.append(
        finish_compact(
            p, y, "+ supported   ~ partial   Facets repeat exact names; no identity repair."
        )
    )
    p = old.Plate(
        "figure_2_harbor",
        "2 · Source validity is not the query window",
        "A: fixed selection from pre-extraction    B: independent contextual extraction",
        500,
    )
    y = compact_source(
        p,
        data["synthetic"]["stories"]["harbor"],
        ["S5", "S6", "S7", "S8"],
        54,
        "SOURCE SUBSET · 4 of 14 sentences shown; both methods received all 14",
    )
    p.text(
        12,
        y,
        "REQUEST · Direct locations overlapping day [2, 4); preserve full bounds",
        bold=True,
        size=9,
    )
    y += 18
    p.text(12, y, "A · all 4 selected records", bold=True)
    p.text(24 + col, y, "B · 4 of 6 records below", bold=True)
    y += 18
    a = panels["harbor_A_locations"]
    b = subset(panels["harbor_B_locations"], [3, 4, 5, 6])
    for ra, rb in zip(a["records"], b["records"], strict=True):
        mini_edge(p, a, ra, 12, y, col, 32)
        mini_edge(p, b, rb, 24 + col, y, col, 32)
        y += 36
    y = (
        p.para(
            12,
            y,
            "I B also returns "
            + "; ".join(
                f"{r['fact']['subject']} → {r['fact']['relation']} → {r['fact']['object']} [{r['fact']['valid_from']}, {r['fact']['valid_until']}), {', '.join(r['fact']['evidence_ids'])}"
                for r in panels["harbor_B_locations"]["records"][:2]
            )
            + ". Supported but irrelevant: counted, not hidden errors.",
            old.W - 24,
            size=8.5,
            color=old.MUTED,
        )
        + 7
    )
    p.text(
        12, y, "Generated intervals · identical in A and B location records", size=8.5, bold=True
    )
    y += 17
    left, right = 170, old.W - 20
    unit = (right - left) / 6
    p.rect(left + 2 * unit, y - 2, 2 * unit, 77, fc="#E1EAF1", ec="none", radius=0)
    for i, r in enumerate(a["records"]):
        f = r["fact"]
        yy = y + i * 16
        p.text(12, yy, f"{f['subject']} [{f['valid_from']}, {f['valid_until']})", size=8.5)
        p.ax.plot(
            [left + f["valid_from"] * unit, left + f["valid_until"] * unit],
            [yy + 5] * 2,
            color=old.STATUS["supported"][0],
            lw=3,
        )
        p.ax.plot(
            left + f["valid_from"] * unit, yy + 5, "o", ms=4, color=old.STATUS["supported"][0]
        )
        p.ax.plot(
            left + f["valid_until"] * unit,
            yy + 5,
            "o",
            ms=4,
            mfc="white",
            mec=old.STATUS["supported"][0],
        )
    for n in range(7):
        p.text(left + n * unit, y + 66, str(n), size=8.5, align="center")
    y += 85
    figures.append(
        finish_compact(
            p, y, "+ supported   I irrelevant   Shading: query [2, 4); ○ excludes endpoint."
        )
    )
    p = old.Plate(
        "figure_3_alice",
        "3 · Actions and thoughts from the same passage",
        "B only · complete text + each question → independently generated records",
        500,
    )
    y = compact_source(
        p,
        data["real"]["stories"]["alice"],
        ["S1", "S2"],
        54,
        "SOURCE · complete supplied Alice passage",
    )
    p.text(
        12,
        y,
        "REQUEST 1 · Sitting, inspection/reading and running (display headline)",
        size=9,
        bold=True,
    )
    y += 17
    a = subset(panels["alice_B_actions"], [3, 4, 5])
    for r in a["records"]:
        mini_edge(p, a, r, 12, y, old.W - 24, 31)
        y += 34
    p.text(
        12,
        y,
        "3 of 5 action records shown; sitting-by-sister and sitting-by-bank omitted.",
        size=8.5,
        color=old.MUTED,
    )
    y += 17
    p.text(
        12, y, "REQUEST 2 · Book content and Alice's thoughts (display headline)", size=9, bold=True
    )
    y += 16
    b = panels["alice_B_claims"]
    for r, h in zip(b["records"], [29, 47, 47], strict=True):
        mini_edge(p, b, r, 12, y, old.W - 24, h)
        y += h + 3
    y = (
        p.para(
            12,
            y,
            "Reading belongs to the sister, not Alice. The thought remains a sentence object; its meaning is preserved without decomposition. Daisy-chain: holder=Alice, attitude=null; unresolved judgment versus consideration.",
            old.W - 24,
            size=8.5,
        )
        + 3
    )
    figures.append(
        finish_compact(
            p, y, "+ supported   x unsupported participant   ? unresolved   No semantic repair."
        )
    )
    return figures


def captions(panels):
    def q(k):
        return panels[k]["result"]["question"]

    return {
        "figure_1_fable": "Figure 1. Aesop's The Lion and the Mouse (Townsend translation). Exact executed question: “"
        + q("fable_A")
        + "” A selects from an actual pre-extraction; B is generated independently. All B action records and the A capture, conditional-spare, release, hunter-capture, gnawing and rescue records are shown. A's waking and rope-binding records are omitted only from display (two of eight); full graphs and all scores remain in the supplement. B preserves the later rescue but omits the earlier capture and release; its running relation loses the face-specific endpoint. A's spare claim loses conditional scope. S6, the final speech, is not displayed but was supplied. Faceted arrows reproduce exact generated endpoint strings; repeated names denote the same exact-string node. Undisplayed qualification values are null except the explicitly noted missing key. Colours/symbols reproduce Codex-authored assessments, not independent verification.",
        "figure_2_harbor": "Figure 2. Harbor location comparison. Exact executed question: “"
        + q("harbor_A_locations")
        + "” All four A location records and the four corresponding B records are displayed. B's other two records are reproduced in the labelled carrying callout and remain in its six-prediction denominator. Both methods preserve source intervals; the shaded question window selects overlap and does not redefine validity. Circles distinguish included starts from excluded ends. Only S5–S8 are printed; the model received the same complete fourteen-sentence story in every call. A is fixed selection; B is independent generation. All generated holder/attitude values in these records are null. This is an extraction/relevance comparison, not ontology construction.",
        "figure_3_alice": "Figure 3. Carroll's Alice opening passage, supplied unchanged to both B requests. Exact action question: “"
        + q("alice_B_actions")
        + "” Exact claims question: “"
        + q("alice_B_claims")
        + "” Three of five action records show inspection, incorrect assignment of reading to Alice, and the Rabbit's literal object ‘close by her’; the two sitting records remain in the full graph and score. All three claims records are shown. The sentence-valued thinks object preserves rhetorical meaning but not the requested decomposed attribution structure; it is not redrawn as a corrected proposition. The daisy-chain record lacks attitude alongside holder Alice and may turn consideration into a judgment. All generated bounds are null; other holder/attitude fields are null. Display line breaks and underscore-to-space predicate typography do not change identities, meanings or evaluations. Status labels reproduce the retained Codex assessment.",
    }


def main_captions():
    """Short review captions; exact questions and full display notes stay in S6."""
    return {
        "figure_1_fable": "Figure 1. Fable actions: A selects from a pre-extraction; B independently answers from text and question. Both capture the later rescue. B omits the earlier capture/release; A's spare relation loses conditional scope. Six of eight A records and all five B records are drawn; A's waking and binding records remain in the full output. Exact question and display notes: Supplement S6.",
        "figure_2_harbor": "Figure 2. Harbor locations: full generated intervals are distinct from the shaded query window. All four A location records and the four corresponding B records are drawn; B's two irrelevant carrying records appear in the callout and remain scored. A is fixed selection; B is independent extraction. Source display is S5–S8 of the complete story supplied. Exact question: Supplement S6.",
        "figure_3_alice": "Figure 3. Alice's actions and thoughts, independently extracted by B from the same passage. Reading is assigned to the wrong participant; a literal sentence-valued thought preserves rhetorical meaning, while the daisy-chain interpretation remains unresolved. Three of five action records and all three claims records are drawn; the two sitting records remain in the full output and score. Exact questions and display notes: Supplement S6.",
    }


class VectorFigure(Flowable):
    def __init__(self, path):
        super().__init__()
        self.xobj = pagexobj(PdfReader(str(path)).pages[0])
        self.width = float(self.xobj.BBox[2])
        self.height = float(self.xobj.BBox[3])

    def draw(self):
        self.canv.doForm(makerl(self.canv, self.xobj))


def pdf_styles():
    pdfmetrics.registerFont(TTFont("Paper", str(FONT)))
    pdfmetrics.registerFont(TTFont("PaperBold", str(FONT_BOLD)))
    pdfmetrics.registerFontFamily(
        "Paper", normal="Paper", bold="PaperBold", italic="Paper", boldItalic="PaperBold"
    )
    base = getSampleStyleSheet()
    styles = {}
    for name, size, leading in [
        ("body", 10.5, 14.2),
        ("caption", 9, 12),
        ("small", 8.7, 11.5),
        ("h1", 19, 24),
        ("h2", 14, 18),
        ("h3", 11.5, 15),
    ]:
        styles[name] = ParagraphStyle(
            name,
            parent=base["Normal"],
            fontName="PaperBold" if name.startswith("h") else "Paper",
            fontSize=size,
            leading=leading,
            spaceAfter=8,
            spaceBefore=10 if name.startswith("h") else 0,
            textColor=colors.HexColor(old.INK),
            keepWithNext=name.startswith("h"),
            alignment=TA_LEFT,
            splitLongWords=True,
            allowWidows=0,
            allowOrphans=0,
        )
    return styles


def inline(text):
    # Render only small, controlled Markdown inline syntax into ReportLab markup.
    text = html.escape(text)
    text = re.sub(
        r"\[([^\]]+)\]\(([^)]+)\)",
        lambda m: '<a href="' + m[2] + '" color="#315D82">' + m[1] + "</a>",
        text,
    )
    text = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", text)
    text = re.sub(r"`([^`]+)`", r"\1", text)
    return text


def make_pdf(markdown, path, *, figure_captions=None):
    styles = pdf_styles()
    flow = []
    tokens = MarkdownIt("commonmark").enable("table").parse(markdown)
    i = 0
    while i < len(tokens):
        t = tokens[i]
        if t.type == "heading_open":
            style = styles["h" + t.tag[1]]
            flow.append(Paragraph(inline(tokens[i + 1].content), style))
            i += 3
            continue
        if t.type == "paragraph_open":
            s = tokens[i + 1].content
            if s.startswith("!FIGURE:"):
                name = s.split(":", 1)[1]
                fig = VectorFigure(OUT / "figures" / (name + ".pdf"))
                # Compact artwork at native paper size; captions outside its bounding box.
                flow.append(
                    KeepTogether(
                        [
                            fig,
                            Spacer(1, 7),
                            Paragraph(inline(figure_captions[name]), styles["caption"]),
                        ]
                    )
                )
            else:
                paragraph = Paragraph(
                    inline(s), styles["caption"] if re.match(r"Table \d+\.", s) else styles["body"]
                )
                paragraph.keepWithNext = s.endswith(":") or bool(re.match(r"Table \d+\.", s))
                flow.append(paragraph)
            i += 3
            continue
        if t.type == "fence":
            code = [
                Paragraph(html.escape(line).replace(" ", "&#160;"), styles["small"])
                for line in t.content.rstrip().splitlines()
            ]
            if flow and isinstance(flow[-1], Paragraph) and flow[-1].getPlainText().endswith(":"):
                code.insert(0, flow.pop())
            flow.append(KeepTogether([*code, Spacer(1, 6)]))
            i += 1
            continue
        if t.type == "table_open":
            rows = []
            row = []
            i += 1
            while tokens[i].type != "table_close":
                if tokens[i].type == "tr_open":
                    row = []
                elif tokens[i].type == "inline":
                    row.append(Paragraph(inline(tokens[i].content), styles["small"]))
                elif tokens[i].type == "tr_close":
                    rows.append(row)
                i += 1
            n = len(rows[0])
            w = old.W
            widths = {
                6: [88, 60, 30, 145, 30, 151],
                7: [129, 39, 28, 141, 92, 75],
                5: [144, 37, 139, 92, 92],
            }.get(n, [w / n] * n)
            # Published table also has 6 columns, selected using its heading.
            if "Passage" in rows[0][0].text:
                widths = [128, 42, 27, 131, 90, 86]
            if rows[0][0].text == "Case":
                widths = [35, 190, 93, 93, 93]
            if n == 5 and rows[0][0].text == "Story":
                widths = [104, 64, 90, 90, 156]
            widths = [v * w / sum(widths) for v in widths]
            table = Table(rows, colWidths=widths, repeatRows=1, hAlign="LEFT")
            table.setStyle(
                TableStyle(
                    [
                        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#E8EFF3")),
                        ("LINEBELOW", (0, 0), (-1, 0), 0.8, colors.HexColor(old.BLUE)),
                        ("VALIGN", (0, 0), (-1, -1), "TOP"),
                        ("LEFTPADDING", (0, 0), (-1, -1), 4),
                        ("RIGHTPADDING", (0, 0), (-1, -1), 4),
                        ("TOPPADDING", (0, 0), (-1, -1), 5),
                        ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
                        (
                            "ROWBACKGROUNDS",
                            (0, 1),
                            (-1, -1),
                            [colors.white, colors.HexColor("#F7F9FA")],
                        ),
                    ]
                )
            )
            group = [table, Spacer(1, 10)]
            if flow and isinstance(flow[-1], Paragraph):
                if flow[-1].style.name.startswith("h") or re.match(
                    r"Table \d+\.", flow[-1].getPlainText()
                ):
                    group.insert(0, flow.pop())
                elif (
                    len(flow) >= 2
                    and isinstance(flow[-2], Paragraph)
                    and flow[-2].style.name.startswith("h")
                ):
                    group.insert(0, flow.pop())
                    group.insert(0, flow.pop())
            flow.append(KeepTogether(group))
            i += 1
            continue
        if t.type == "hr":
            flow.append(PageBreak())
        i += 1

    def footer(canvas, doc):
        canvas.setFont("Paper", 8.5)
        canvas.setFillColor(colors.HexColor(old.MUTED))
        canvas.drawString(
            16 / 25.4 * 72, 25, "Exploratory narrative-graph prototype · review manuscript"
        )
        canvas.drawRightString(A4[0] - 16 / 25.4 * 72, 25, str(doc.page))

    doc = BaseDocTemplate(
        str(path),
        pagesize=A4,
        leftMargin=16 / 25.4 * 72,
        rightMargin=16 / 25.4 * 72,
        topMargin=34,
        bottomMargin=42,
        title="Inspectable Narrative Graphs: An Exploratory Extraction Prototype",
        author="a.v.mantzaris",
        pageCompression=1,
        invariant=1,
    )
    # Remove the default six-point frame padding so 178-mm vector artwork stays native.
    from reportlab.platypus import Frame, PageTemplate

    frame = Frame(
        doc.leftMargin,
        doc.bottomMargin,
        doc.width,
        doc.height,
        leftPadding=0,
        rightPadding=0,
        topPadding=0,
        bottomPadding=0,
        id="body",
    )
    doc.addPageTemplates(PageTemplate(id="normal", frames=[frame], onPage=footer))
    doc.build(flow)


def bibliography(refs):
    excluded = {
        "id",
        "type",
        "label",
        "supports",
        "verified_full_text",
        "verification_note",
        "type_label",
    }
    bib = []
    md = []
    for r in refs:
        fields = [f"  {k} = {{{v}}}" for k, v in r.items() if k not in excluded]
        if r.get("type_label"):
            fields.append(f"  type = {{{r['type_label']}}}")
        bib.append("@" + r["type"] + "{" + r["id"] + ",\n" + ",\n".join(fields) + "\n}")

        def names(value):
            # BibTeX keeps family/given order; readable bibliography uses given family.
            parts = []
            for name in value.replace("{", "").replace("}", "").split(" and "):
                if name == "others":
                    parts.append("et al.")
                elif ", " in name:
                    family, given = name.split(", ", 1)
                    parts.append(given + " " + family)
                else:
                    parts.append(name)
            return "; ".join(parts)

        author = names(r.get("author", r.get("editor", "")))
        if "editor" in r and "author" not in r:
            author += " (eds.)"
        if r.get("translator"):
            author += "; translated by " + names(r["translator"])
        venue = r.get(
            "journal",
            r.get(
                "booktitle", r.get("institution", r.get("publisher", r.get("archivePrefix", "")))
            ),
        )
        detail = "".join(
            [
                f" {r['volume']}" if "volume" in r else "",
                f"({r['number']})" if "number" in r else "",
                f": {r['pages'].replace('--', '–')}" if "pages" in r else "",
            ]
        )
        sentences = [
            f"{author} ({r.get('year', 'n.d.')}). [{r['title']}]({r['url']})",
            venue + detail,
            r.get("type_label", ""),
            r.get("note", ""),
        ]
        md.append(". ".join(s.rstrip(". ") for s in sentences if s.strip()) + ".")
    (OUT / "references.bib").write_text("\n\n".join(bib) + "\n")
    return "\n\n".join(x.rstrip() for x in md)


def build():
    (OUT / "figures").mkdir(parents=True, exist_ok=True)
    (OUT / "tables").mkdir(exist_ok=True)
    data = {k: old.read(ROOT / v) for k, v in SOURCES.items()}
    source_hashes = {v: old.sha(ROOT / v) for v in SOURCES.values()}
    # Verify the retained report/table seals before deriving manuscript material.
    for fn in ["compact_story_v2_manifest", "real_text_proof_of_concept_manifest"]:
        for path, digest in old.read(ROOT / "reports/tables" / (fn + ".json")).items():
            assert old.sha(ROOT / "reports" / path) == digest, (path, "retained result changed")
    _, figdata, panels, hashes = old.load_inputs()
    source_hashes.update(hashes)
    tables = collect_tables(data)
    refs = old.read(OUT / "reference_sources.json")
    reference_md = bibliography(refs)
    caps = main_captions()
    full_caps = captions(panels)
    plates = compact_figures(figdata, panels)
    figure_manifest = []
    for p in plates:
        old.validate_plate(p)
        stem = OUT / "figures" / p.name
        p.fig.savefig(stem.with_suffix(".pdf"), metadata=old.PDF_META)
        p.fig.savefig(
            stem.with_suffix(".svg"),
            metadata={"Date": None, "Creator": "StoryProjectionOnto manuscript builder"},
        )
        fonts = old.embed_svg(stem.with_suffix(".svg"), p)
        subprocess.run(
            [
                "pdftoppm",
                "-r",
                "400",
                "-png",
                "-singlefile",
                str(stem.with_suffix(".pdf")),
                str(stem),
            ],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
        )
        figure_manifest.append(
            dict(
                name=p.name,
                width_mm=178,
                height_mm=p.height / 72 * 25.4,
                minimum_font_pt=8.5,
                displayed_edges=p.edges,
                display_callouts=panels["harbor_B_locations"]["records"][:2]
                if p.name == "figure_2_harbor"
                else [],
                panels=p.panels,
                font_hashes=fonts,
                caption=caps[p.name],
                detailed_caption=full_caps[p.name],
            )
        )
        old.plt.close(p.fig)
    (OUT / "figures/DEJAVU_FONT_LICENSE.txt").write_bytes(
        (ROOT / "configs/visualization/DEJAVU_FONT_LICENSE.txt").read_bytes()
    )
    template = (OUT / "manuscript_source.md").read_text()
    # Model/settings sourced from frozen original model metadata and common main packing.
    for k, v in data["ladder"]["model"].items():
        if k != "template_sha256":
            number("model_" + k, v, "ladder/model/" + k)
    decoding = old.read(ROOT / "configs/study/decoding.json")
    for k in ["temperature", "top_p", "top_k"]:
        number(k, decoding[k], "configs/study/decoding.json/" + k)
    for k in ["maximum_context", "reserved_output_tokens"]:
        vals = {p[k] for d in [data["compact"], data["prose"]] for p in d["packing"].values()}
        assert len(vals) == 1
        number(k, vals.pop(), "common compact and prose packing/" + k, ",")
    missing = set(re.findall(r"\{\{(\w+)\}\}", template)) - set(METRIC_REGISTRY)
    assert not missing, missing
    rendered = re.sub(r"\{\{(\w+)\}\}", lambda m: METRIC_REGISTRY[m[1]]["display"], template)
    for k, v in tables.items():
        rendered = rendered.replace("!TABLE:" + k, v)
    for r in refs:
        rendered = rendered.replace(
            "[@" + r["id"] + "]", "([" + r["label"] + "](" + r["url"] + "))"
        )
    assert "[@" not in rendered
    rendered += "\n\n## References\n\n" + reference_md + "\n"
    markdown = rendered
    for name, cap in caps.items():
        markdown = markdown.replace("!FIGURE:" + name, f"![{name}](figures/{name}.png)\n\n" + cap)
    (OUT / "PROOF_OF_CONCEPT.md").write_text(markdown)
    (OUT / "FIGURE_CAPTIONS.md").write_text(
        "# Figure captions and exact-question notes\n\n## Main-paper captions\n\n"
        + "\n\n".join(caps.values())
        + "\n\n## Complete questions and display accounting (Supplement S6)\n\n"
        + "\n\n".join(full_caps.values())
        + "\n"
    )
    make_pdf(rendered, OUT / "PROOF_OF_CONCEPT.pdf", figure_captions=caps)
    supplementary(data, tables, panels)
    write_json(OUT / "tables/numerical_registry.json", METRIC_REGISTRY)
    prose_only = re.sub(
        r"!FIGURE:[^\n]+|!TABLE:[^\n]+", "", re.sub(r"```.*?```", "", template, flags=re.S)
    )
    prose_only = re.sub(r"^Table \d+\..*$", "", prose_only, flags=re.M)
    prose_only = re.sub(r"\{\{(\w+)\}\}", lambda m: METRIC_REGISTRY[m[1]]["display"], prose_only)
    word_count = len(re.findall(r"\b[\w’'-]+\b", prose_only))
    manifest = dict(
        version="exploratory-manuscript-v2-editorial",
        scope="Retained outputs only; no new inference, scoring changes or registered-study acceptance",
        generator="python -m scripts.build_poc_manuscript",
        source_artifacts=source_hashes,
        source_checkpoint="e76d52807ab2004d1e644c6f25a49b8b41fd214a",
        editorial_parent_checkpoint="3329a6e3ba8d8538d69c7e701a630932bfcf129e",
        editorial_note="Tighter narrative and captions; unchanged figure artwork, retained outputs, tables, assessments and scoring. Main cost display selects the two comparison batches; supplement retains all batch costs. No minimum word-count target.",
        body_word_count_excluding_tables_figures_references_and_code=word_count,
        word_count_note="Approximate whitespace/punctuation-tokenized prose count; table captions also excluded; headings and AI-assistance declaration included.",
        reference_verification_date="2026-09-09",
        references=refs,
        figures=figure_manifest,
        selection_rationale={
            "fable": "Readable capture/rescue example, baseline advantage and partial conditional representation",
            "harbor": "Explicit intervals plus contextual over-selection, not a success-only example",
            "alice": "Thought meaning preserved despite structural mismatch, with unsupported reading participant",
        },
        display_transformations=[
            "Whitespace reflow of source only; selected evidence IDs and omissions explicitly labelled",
            "Exact endpoint strings; no alias merging, semantic repair or record aggregation",
            "Predicate underscores replaced by spaces for typography only",
            "Faceted arrows repeat exact identities; native 178-mm artwork, at least 8.5-point labels",
            "Harbor B carrying extras shown in a labelled callout; all denominators unchanged",
        ],
        numerical_registry="tables/numerical_registry.json",
        retained_supplement_pages={
            "reports/PAPER_FIGURES.pdf": len(
                PdfReader(str(ROOT / "reports/PAPER_FIGURES.pdf")).pages
            ),
            "reports/figures/paper/FULL_OUTPUT_SUPPLEMENT.pdf": len(
                PdfReader(str(ROOT / "reports/figures/paper/FULL_OUTPUT_SUPPLEMENT.pdf")).pages
            ),
        },
    )
    manifest["inputs"] = {
        str(p.relative_to(ROOT)): old.sha(p)
        for p in [
            Path(__file__),
            OUT / "manuscript_source.md",
            OUT / "README.md",
            OUT / "requirements.txt",
            OUT / "reference_sources.json",
            OUT / "AUTHOR_REVIEW.md",
            ROOT / "configs/study/decoding.json",
            ROOT / "scripts/build_paper_figures.py",
            ROOT / "reports/PAPER_FIGURES.pdf",
            ROOT / "reports/figures/paper/FULL_OUTPUT_SUPPLEMENT.pdf",
        ]
    }
    index = "<!doctype html><meta charset='utf-8'><title>Compact manuscript figures</title><style>body{font:16px system-ui;max-width:900px;margin:2em auto}img{width:100%}</style><h1>Compact main-paper figures</h1><p>Retained exploratory outputs. <a href='../PROOF_OF_CONCEPT.pdf'>Manuscript PDF</a> · <a href='../SUPPLEMENTARY_MATERIAL.pdf'>Supplement</a> · <a href='../../reports/figures/paper/index.html'>Existing interactive evidence-linked comparisons</a></p>"
    for f in figure_manifest:
        n = f["name"]
        index += f"<h2>{html.escape(n)}</h2><p><a href='{n}.pdf'>PDF</a> · <a href='{n}.svg'>SVG</a> · <a href='{n}.png'>400-dpi PNG</a></p><img src='{n}.svg' alt='{html.escape(n)}'><p>{html.escape(f['caption'])}</p>"
    (OUT / "figures/index.html").write_text(index + "\n")
    manifest["outputs"] = {
        str(p.relative_to(OUT)): old.sha(p)
        for p in sorted(OUT.rglob("*"))
        if p.is_file()
        and p.name not in {"manuscript_manifest.json", "verification.json"}
        and "inspection" not in p.parts
    }
    write_json(OUT / "manuscript_manifest.json", manifest)
    print(
        json.dumps(
            {
                "body_words": word_count,
                "main_pages": len(PdfReader(str(OUT / "PROOF_OF_CONCEPT.pdf")).pages),
                "figure_heights_mm": [round(f["height_mm"], 1) for f in figure_manifest],
            },
            indent=2,
        )
    )


def supplementary(data, tables, panels):
    text = (
        "# Supplementary material: retained exploratory narrative graphs\n\nThis supplement preserves the eight original review plates and the full-output appendix without altering their PDF pages. The new main figures use labelled display subsets; all evaluated records and historical failures remain available. References and semantic assessments are Codex-authored, not independent human review.\n\n## Contents and evaluation versions\n\nThe pages below provide detailed canonical tables and exact executed questions. The original eight-plate review document follows, then its full-output appendix, including unparseable Holmes outputs shown as raw text. Source passages, reference alternatives, and all calls are also available in the immutable repository reports linked in the manuscript manifest.\n\n## S1. Compact stories v2: fresh outputs only\n\n"
        + tables["compact"]
        + "\n\n## S2. Published prose: separate strict and semantic outcomes\n\nS/P/X/U denotes supported, partial, unsupported and unresolved records. Coverage is unique explicit target concepts. NA is unavailable parsing, not an empty graph.\n\n"
        + tables["prose"]
        + "\n\n## S3. Distinct batch costs\n\nService allocation includes loading, checks, inference and shutdown. Request times alone exclude those costs. Costs may be added; accuracy across incompatible versions may not.\n\n"
        + tables["cost"]
        + "\n\nThe two main comparison batches used "
        + METRIC_REGISTRY["main_cost"]["display"]
        + " allocated seconds. Historical whole-project allocation at their conclusion was "
        + METRIC_REGISTRY["project_cost"]["display"]
        + " seconds, including earlier attempts and failures. No new inference was performed for this manuscript. Small-batch timing does not establish production-tail latency or feasibility of the larger registered study.\n\n"
        + "## S4. Development context\n\nThe original simple ladder recovered its tested direct facts, contextual selections and explicit intervals before encountering belief decomposition and citation/interval failures. The revised ladder retained its regression successes and handled original and fresh simple beliefs; office examples remained imperfect. The office questions permit the same office-mediated organization and do not establish an ontology-construction contrast. Earlier unsuccessful comprehensive-interface attempts motivated simplification but are not compact-interface results. Scores below remain separate by version.\n\n"
    )
    for key in ["ladder", "ladder_v2"]:
        rows = data[key]["rows"]
        text += (
            "\n\n### "
            + key.replace("_", " ")
            + " — unchanged development scores\n\n"
            + md_table(
                ["Case", "Task", "Precision", "Recall", "F1"],
                [
                    [
                        r["case_id"],
                        r.get("task", r.get("level")),
                        f"{r['precision']:.3f}",
                        f"{r['recall']:.3f}",
                        f"{r['f1']:.3f}",
                    ]
                    for r in rows
                ],
            )
        )
    text += "\n\n## S5. Historical syntax recovery is not new generation\n\nEarlier Harbor/Orchard possession outputs failed strict parsing. Their separately retained comma-only derivations are below. Original strict failures are unchanged and excluded from the fresh table. Full edit records and original scores are reproduced unchanged in tables/historical_syntax_recovery.json.\n\n"
    text += md_table(
        ["Story", "Matches", "Predictions", "References", "P/R/F1"],
        [
            [
                x["story_id"],
                x["evaluation"]["full"]["true_positive"],
                x["evaluation"]["full"]["predicted"],
                x["evaluation"]["full"]["reference_count"],
                "/".join(
                    f"{x['evaluation']['full'][k]:.3f}" for k in ["precision", "recall", "f1"]
                ),
            ]
            for x in data["compact"]["historical"]["derived"]
        ],
    )
    text += "\n\n## S6. Exact requests, display notes and source attribution\n\nQuestion headlines in the artwork abbreviate the executed requests below. A means fixed selection from an actual query-blind extraction; B is independently generated from text plus question. These are comparisons, not a transformation from A into B.\n"
    for i, cap in enumerate(captions(panels).values(), 1):
        text += f"\n\n### Figure {i}: complete question and display notes\n\n" + cap
    for story in data["prose"]["stories"].values():
        text += (
            "\n\n### "
            + story["title"]
            + "\n\n"
            + story["author"]
            + ("; translated by " + story["translator"] if story.get("translator") else "")
            + ". "
            + story["section"]
            + ". [Source]("
            + story["source_url"]
            + "). Retrieval: "
            + story["retrieval_date"]
            + ". Excerpt SHA-256: "
            + story["excerpt_sha256"]
            + ".\n\n"
            + story["rights"]
            + " Notices: "
            + story["notice_file"]
            + ".\n\n"
        )
        for eid, span in story["evidence"].items():
            text += "**" + eid + "** " + " ".join(span.split()) + "\n\n"
    text += (
        "\n\n## S7. Reproducibility and author review\n\nThe pinned model is "
        + METRIC_REGISTRY["model_repository"]["display"]
        + ", revision "
        + METRIC_REGISTRY["model_revision"]["display"]
        + ", seed "
        + METRIC_REGISTRY["model_seed"]["display"]
        + ". Full model/template metadata and call settings originate in the frozen manifests. All outputs, timestamps, request hashes and assessments are linked by manuscript_manifest.json. Figure edge manifests preserve original record dictionaries and statuses. Reference details were checked against primary sources on 9 September 2026.\n\n"
        + "[AUTHOR_REVIEW.md](AUTHOR_REVIEW.md) provides the prioritized evidence-linked judgment questions and the single list of unresolved publication details. It is a request for author decisions, not a completed human review. Any subsequent annotation or score change must be separately versioned. [README.md](README.md) gives the portable regeneration command. The original visual-review and full-output pages follow unchanged.\n\n"
    )
    (OUT / "SUPPLEMENTARY_MATERIAL.md").write_text(text.rstrip() + "\n")
    front = OUT / "supplement_front.pdf"
    make_pdf(text, front)
    writer = PdfWriter()
    for path in [
        front,
        ROOT / "reports/PAPER_FIGURES.pdf",
        ROOT / "reports/figures/paper/FULL_OUTPUT_SUPPLEMENT.pdf",
    ]:
        writer.addpages(PdfReader(str(path)).pages)
    writer.write(str(OUT / "SUPPLEMENTARY_MATERIAL.pdf"))


if __name__ == "__main__":
    build()
