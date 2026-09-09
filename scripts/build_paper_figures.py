"""Render paper-sized vector figures from immutable compact-result tables; CPU only.

Run: python -m scripts.build_paper_figures
No model, controller, scorer or remote-service imports are used.
"""

# Human-facing captions, exact quoted prose and inline HTML retain their typography.
# ruff: noqa: E501, RUF001

from __future__ import annotations

import argparse
import base64
import hashlib
import html
import io
import json
import os
import subprocess
import textwrap
from functools import lru_cache
from pathlib import Path
from xml.etree import ElementTree as ET

os.environ.setdefault("MPLCONFIGDIR", "/tmp/storyprojection-paper-mpl")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from fontTools import subset
from fontTools.ttLib import TTFont
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.font_manager import FontProperties, findfont
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle
from matplotlib.textpath import TextToPath

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/visualization/paper_figures.json"
W = 178 / 25.4 * 72
INK, MUTED, BLUE, PAPER = "#142D40", "#4C5B68", "#315D82", "#F4F7F8"
STATUS = {
    "supported": ("#14776E", "+", "solid"),
    "partial": ("#986007", "~", (0, (5, 2))),
    "unsupported": ("#B33C3D", "x", (0, (2, 2))),
    "unresolved": ("#735094", "?", (0, (6, 2, 1, 2))),
    "irrelevant": ("#606B79", "I", (0, (7, 3))),
}
matplotlib.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 9,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
        "svg.hashsalt": "story-projection-paper-v1",
        "axes.unicode_minus": False,
    }
)
PDF_META = {
    "Creator": "StoryProjectionOnto / retained-result figure builder",
    "CreationDate": None,
    "ModDate": None,
}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


@lru_cache(maxsize=20000)
def width(text, size=9, bold=False):
    return TextToPath().get_text_width_height_descent(
        text,
        FontProperties(family="DejaVu Sans", size=size, weight="bold" if bold else "normal"),
        False,
    )[0]


def wrap(text, maximum, size=9, bold=False):
    """Line breaks only; preserve every token, with no ellipsis or semantic shortening."""
    lines = []
    for paragraph in str(text).split("\n"):
        line = ""
        for word in paragraph.split():
            trial = (line + " " + word).strip()
            if line and width(trial, size, bold) > maximum * 0.95:
                lines.append(line)
                line = word
            else:
                line = trial
        lines.append(line)
    return lines


class Plate:
    def __init__(self, name, title, subtitle, height):
        self.name, self.height = name, height
        self.fig = plt.figure(figsize=(W / 72, height / 72), facecolor="white")
        self.ax = self.fig.add_axes([0, 0, 1, 1])
        self.ax.set(xlim=(0, W), ylim=(height, 0))
        self.ax.axis("off")
        self.text_items, self.edges, self.panels, self.numeric_claims = [], [], [], []
        self.text(16, 14, title, size=13, bold=True)
        self.text(16, 33, subtitle, size=8.5, color=MUTED)
        self.ax.plot([16, W - 16], [47, 47], color="#C7D3DA", lw=0.8)

    def text(self, x, y, value, *, size=9, bold=False, color=INK, align="left", gid=None):
        assert size >= 8.5
        t = self.ax.text(
            x,
            y,
            value,
            fontsize=size,
            weight="bold" if bold else "normal",
            color=color,
            va="top",
            ha=align,
            linespacing=1.3,
            zorder=8,
        )
        if gid:
            t.set_gid(gid)
        self.text_items.append(t)
        return t

    def para(self, x, y, value, maximum, *, size=9, bold=False, color=INK, gid=None):
        lines = wrap(value, maximum, size, bold)
        self.text(x, y, "\n".join(lines), size=size, bold=bold, color=color, gid=gid)
        return y + len(lines) * size * 1.3

    def rect(self, x, y, w, h, *, fc=PAPER, ec="#D8E0E5", lw=0.7, radius=4):
        patch = FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle=f"round,pad=0,rounding_size={radius}",
            facecolor=fc,
            edgecolor=ec,
            lw=lw,
            zorder=0,
        )
        self.ax.add_patch(patch)
        return patch

    def source(self, story, y, *, selected=None, label="SOURCE · complete evidence shown", gap=4):
        self.text(24, y + 9, label, bold=True, size=9)
        cursor = y + 27
        for eid in sorted(story["evidence"], key=lambda s: int(s[1:])):
            if selected is not None and eid not in selected:
                continue
            self.text(24, cursor, eid, size=8.5, color=BLUE, bold=True)
            cursor = (
                self.para(
                    48,
                    cursor,
                    " ".join(story["evidence"][eid].split()),
                    W - 80,
                    size=9,
                    gid=f"source-{self.name}-{eid}",
                )
                + gap
            )
        self.rect(16, y, W - 32, cursor - y + 5)
        return cursor + 12

    def question(self, y, title, question, *, x=16, w=None):
        w = W - 32 if w is None else w
        self.text(x + 9, y + 8, title, size=9, bold=True, color=BLUE)
        end = self.para(x + 9, y + 23, question, w - 18, size=9)
        self.rect(x, y, w, end - y + 10, fc="#EDF4FA", ec="#BDCEDD")
        return end + 17

    def note(self, y, heading, message, *, color=MUTED):
        self.text(24, y + 8, heading, bold=True, color=color)
        end = self.para(24, y + 24, message, W - 64, size=9)
        self.rect(16, y, W - 32, end - y + 9, fc="#FAFAF8")
        return end + 16

    def legend(self, y):
        x = 16
        for state, title in (
            ("supported", "supported"),
            ("partial", "partial"),
            ("unsupported", "unsupported"),
            ("unresolved", "unresolved"),
            ("irrelevant", "irrelevant"),
        ):
            color, symbol, _ = STATUS[state]
            self.text(x, y, symbol + " " + title, size=8.5, color=color)
            x += width(symbol + " " + title, 8.5) + 13
        self.text(
            16,
            y + 15,
            "Status: retained assessment. A citation badge is a link, not proof of support.",
            size=8.5,
            color=MUTED,
        )
        self.text(
            16,
            y + 29,
            "Unlabelled qualifiers are null (unstated), not timeless. Missing keys are marked.",
            size=8.5,
            color=MUTED,
        )

    def flow(self, y, method):
        labels = (
            ["Text", "Query-blind extraction", "Fixed selection + question"]
            if method == "A"
            else ["Text + question", "Direct contextual extraction"]
        )
        widths = [52, 152, 195] if method == "A" else [153, 226]
        x = 16
        for i, (label, w) in enumerate(zip(labels, widths, strict=True)):
            self.rect(x, y, w, 25, fc="white", ec="#A4B6C3")
            self.text(x + w / 2, y + 7, label, size=8.5, align="center", color=BLUE)
            if i < len(labels) - 1:
                self.ax.add_patch(
                    FancyArrowPatch(
                        (x + w + 3, y + 12.5),
                        (x + w + 25, y + 12.5),
                        arrowstyle="-|>",
                        mutation_scale=9,
                        color=BLUE,
                        lw=0.8,
                    )
                )
            x += w + 28
        return y + 34

    def finish(self, y):
        self.legend(y)
        self.height = y + 54
        self.fig.set_size_inches(W / 72, self.height / 72)
        self.ax.set_ylim(self.height, 0)
        return self


def load_inputs():
    config = read(CONFIG)
    data = {key: read(ROOT / value) for key, value in config["sources"].items()}
    hashes = {}
    for key in config["sources"]:
        manifest = (
            ROOT
            / "reports/tables"
            / (
                "real_text_proof_of_concept_manifest.json"
                if key == "real"
                else "compact_story_v2_manifest.json"
            )
        )
        for rel, digest in read(manifest).items():
            assert sha(ROOT / "reports" / rel) == digest, "Retained result changed"
            hashes[str(Path("reports") / rel)] = digest
    panels = {}
    for key, selection in config["panels"].items():
        d = data[selection["dataset"]]
        if "call" in selection:
            c = next(c for c in d["calls"] if c["case_id"] == selection["call"])
            facts = c["parsed"]["facts"]
            notes = next(
                r["semantic_assessments"]
                for r in d["preextract_assessments"]
                if r["story_id"] == selection["story"]
            )
            r = None
        else:
            r = next(
                r
                for r in d["results"]
                if r["story_id"] == selection["story"]
                and r["task"] == selection["task"]
                and r["approach"][0] == selection["method"]
            )
            c = next(c for c in d["calls"] if c["case_id"] == r["source_case_id"])
            facts = r["parsed"]["facts"]
            notes = r.get("semantic_assessments")
        indices = selection.get("indices_1based", list(range(1, len(facts) + 1)))
        records = []
        for i in indices:
            f = facts[i - 1]
            if notes:
                n = notes[i - 1]
                state = n["support"]
                if state == "contradicted":
                    state = "unsupported"
                if r and n["support"] == "supported" and not n["relevant"]:
                    state = "irrelevant"
                note = n["note"]
            else:
                ev = r["evaluation"]["rows"][i - 1]
                state = {
                    "correct_complete_fact": "supported",
                    "supported_but_irrelevant": "irrelevant",
                    "unsupported": "unsupported",
                }.get(ev["status"], "unresolved")
                note = ev["status"]
            records.append(dict(index=i, fact=f, status=state, note=note))
        panels[key] = dict(
            id=key,
            dataset=selection["dataset"],
            story=selection["story"],
            records=records,
            original_count=len(facts),
            omitted_count=len(facts) - len(indices),
            selection=selection,
            result=r,
            source_call=c["case_id"],
            response_sha256=c["response"]["response_sha256"],
            request_hash=c["request_hash"],
        )
    return config, data, panels, hashes


def qualification(f):
    values = []
    if f.get("valid_from") is not None or f.get("valid_until") is not None:
        values.append(f"[{f.get('valid_from')}, {f.get('valid_until')})")
    if "valid_until" not in f:
        values.append("valid_until MISSING")
    if f.get("holder") is not None or f.get("attitude") is not None:
        values.append(f"holder={f.get('holder') or 'null'}; attitude={f.get('attitude') or 'null'}")
    return "; ".join(values)


def edge_label(record):
    f, state = record["fact"], record["status"]
    symbol = STATUS[state][1]
    return f"{symbol} {record['index']} {f['relation'].replace('_', ' ')}", " ".join(
        f["evidence_ids"]
    ) + (" · " + qualification(f) if qualification(f) else "")


def metric(plate, panel, x, y):
    r = panel["result"]
    if not r:
        return
    ev = r["evaluation"]["full"]
    txt = f"Strict qualified F1 {ev['f1']:.3f} · {ev['predicted']} predictions"
    plate.text(x, y, txt, size=8.5, color=MUTED)
    plate.numeric_claims.append(
        dict(panel=panel["id"], metric="strict_qualified_f1", value=ev["f1"])
    )


def register_edge(plate, panel, record, artists):
    eid = f"edge-{plate.name}-{panel['id']}-{record['index']}"
    ids = []
    for i, artist in enumerate(artists):
        gid = eid + "-" + str(i)
        artist.set_gid(gid)
        ids.append(gid)
    plate.edges.append(
        dict(
            id=eid,
            artist_ids=ids,
            panel=panel["id"],
            story=panel["story"],
            dataset=panel["dataset"],
            index=record["index"],
            fact=record["fact"],
            status=record["status"],
            note=record["note"],
        )
    )


def node(plate, x, y, name, *, w=82):
    lines = wrap(name, w - 10, 9)
    h = max(26, len(lines) * 11.7 + 10)
    box = plate.rect(x - w / 2, y - h / 2, w, h, fc="#F1F5F7", ec="#8CA6B6")
    box.set_zorder(4)
    plate.text(x, y - len(lines) * 11.7 / 2, "\n".join(lines), align="center", size=9)
    return box


def network(plate, panel, origin, positions, routes=None, *, node_width=80, yscale=1):
    """Small curated layouts; all selected records appear once, never matched or repaired."""
    ox, oy = origin
    routes = routes or {}
    positions = {k: (x, y * yscale) for k, (x, y) in positions.items()}
    routes = {k: (r * yscale, x, y * yscale) for k, (r, x, y) in routes.items()}
    actual = {f["fact"][k] for f in panel["records"] for k in ("subject", "object")}
    assert actual <= set(positions), (panel["id"], actual - set(positions))
    boxes = {}
    for name in sorted(actual):
        x, y = positions[name]
        boxes[name] = node(plate, ox + x, oy + y, name, w=node_width)
    for record in panel["records"]:
        f = record["fact"]
        sx, sy = positions[f["subject"]]
        tx, ty = positions[f["object"]]
        rad, lx, ly = routes.get(record["index"], (0, (sx + tx) / 2, (sy + ty) / 2 - 10))
        # Explicit curves plus a hand-placed label for each actual parallel assertion.
        color, _, style = STATUS[record["status"]]
        arrow = FancyArrowPatch(
            (ox + sx, oy + sy),
            (ox + tx, oy + ty),
            connectionstyle=f"arc3,rad={rad}",
            arrowstyle="-|>",
            mutation_scale=10,
            lw=1.05,
            color=color,
            linestyle=style,
            patchA=boxes[f["subject"]],
            patchB=boxes[f["object"]],
            shrinkA=2,
            shrinkB=2,
            zorder=2,
        )
        plate.ax.add_patch(arrow)
        a, b = edge_label(record)
        text = plate.text(ox + lx, oy + ly, a + "\n" + b, size=8.5, align="center", color=color)
        text.set_bbox(dict(facecolor="white", edgecolor="none", alpha=0.96, pad=1.2))
        register_edge(plate, panel, record, [arrow, text])
    plate.panels.append(panel["id"])


def lanes(plate, panel, y, *, row_gap=7):
    """Assertion-faceted graph; repeated glyphs retain the same exact-string identity."""
    for record in panel["records"]:
        f = record["fact"]
        sw, ow = 105, 220
        sl = wrap(f["subject"], sw - 10, 9)
        ol = wrap(f["object"], ow - 10, 9)
        first, second = edge_label(record)
        label = wrap(first, 128, 8.5) + wrap(second, 128, 8.5)
        h = max(len(sl) * 11.7 + 10, len(ol) * 11.7 + 10, len(label) * 11.05 + 14, 43)
        cy = y + h / 2
        node(plate, 16 + sw / 2, cy, f["subject"], w=sw)
        node(plate, W - 16 - ow / 2, cy, f["object"], w=ow)
        color, _, style = STATUS[record["status"]]
        arrow = FancyArrowPatch(
            (16 + sw + 3, cy + h / 2 - 9),
            (W - 19 - ow, cy + h / 2 - 9),
            arrowstyle="-|>",
            mutation_scale=10,
            color=color,
            lw=1,
            linestyle=style,
        )
        plate.ax.add_patch(arrow)
        t = plate.text(
            (16 + sw + W - 16 - ow) / 2,
            y + 3,
            "\n".join(label),
            size=8.5,
            color=color,
            align="center",
        )
        register_edge(plate, panel, record, [arrow, t])
        y += h + row_gap
    plate.panels.append(panel["id"])
    return y


def fable_overview(data, panels):
    p = Plate(
        "figure_1a_fable_overview",
        "Figure 1a · From prose to a requested view",
        "Published narrative · Aesop, The Lion and the Mouse · exploratory extraction",
        686,
    )
    y = p.source(data["real"]["stories"]["fable"], 57)
    y = p.question(
        y, "USER REQUEST · exact executed action question", panels["fable_A"]["result"]["question"]
    )
    y = p.flow(y, "A")
    p.text(16, y, "Before the request: actual query-blind extraction", bold=True)
    p.text(
        16,
        y + 15,
        "S2–S3 detail: 5 of 12 records; full graph in the supplement.",
        size=8.5,
        color=MUTED,
    )
    positions = {"A LION": (75, 88), "a Mouse": (397, 88)}
    routes = {
        2: (-0.42, 236, 4),
        3: (-0.20, 236, 39),
        4: (0, 236, 76),
        5: (0.22, 236, 111),
        6: (0.43, 236, 148),
    }
    network(p, panels["fable_pre_detail"], (16, y + 34), positions, routes)
    y += 224
    y = p.note(
        y,
        "Preserved limitation",
        "S2 is a conditional plea to spare the Mouse. The generated 'spare' record loses that scope and misspells valid_until as 'valid until'. Capture and release are useful; the error is not removed.",
    )
    return p.finish(y + 2)


def fable_comparison(data, panels):
    p = Plate(
        "figure_1b_fable_comparison",
        "Figure 1b · Two actual action-focused outputs",
        "Same complete fable and exact question as Figure 1a · no semantic repair",
        716,
    )
    y = p.question(57, "USER REQUEST", panels["fable_A"]["result"]["question"])
    y = p.flow(y, "A")
    p.text(16, y, "A · Fixed selection from the actual pre-extraction", bold=True)
    metric(p, panels["fable_A"], 16, y + 15)
    pos_a = {
        "A LION": (135, 110),
        "a Mouse": (365, 110),
        "some hunters": (58, 15),
        "strong ropes": (58, 205),
        "the rope": (410, 205),
    }
    routes_a = {
        1: (-0.6, 254, 11),
        2: (-0.25, 254, 51),
        3: (0, 252, 94),
        4: (0.30, 256, 137),
        5: (0.1, 79, 52),
        6: (-0.1, 74, 153),
        7: (-0.12, 395, 154),
        8: (-0.58, 257, 187),
    }
    network(p, panels["fable_A"], (16, y + 39), pos_a, routes_a, node_width=80)
    y += 262
    y = p.flow(y, "B")
    p.text(16, y, "B · Direct contextual extraction", bold=True)
    metric(p, panels["fable_B"], 16, y + 15)
    pos_b = {"Lion": (135, 75), "Mouse": (365, 75), "hunters": (58, 4), "ropes": (250, 175)}
    routes_b = {
        1: (0.50, 253, 14),
        2: (0.1, 76, 24),
        3: (0, 144, 130),
        4: (0, 362, 130),
        5: (0.28, 252, 91),
    }
    network(p, panels["fable_B"], (16, y + 39), pos_b, routes_b, yscale=0.85)
    end = p.note(
        y + 211,
        "Missing target information · not model edges",
        "B omits the Lion catching and releasing the Mouse; 'running over Lion' loses the face-specific endpoint. Passive release/capture wording preserves meaning but remains a strict-match miss.",
    )
    return p.finish(end + 2)


def timeline(p, panels, y):
    p.text(16, y, "Generated intervals stay intact; the question window is separate", bold=True)
    x0, x1 = 228, W - 24
    scale = (x1 - x0) / 6
    rows = panels["harbor_A_possession"]["records"][2:4] + panels["harbor_A_locations"]["records"]
    top = y + 35
    p.ax.add_patch(
        Rectangle(
            (x0 + 2 * scale, top - 6), 2 * scale, 6 * 17 + 7, fc="#DDE7EF", ec="none", zorder=0
        )
    )
    for d in range(7):
        p.text(x0 + d * scale, y + 20, str(d), align="center", size=8.5, color=MUTED)
    for i, r in enumerate(rows):
        f = r["fact"]
        cy = top + i * 17
        p.text(
            16,
            cy - 5,
            f"{f['evidence_ids'][0]}  {f['subject']} · {f['relation'].replace('_', ' ')}",
            size=8.5,
        )
        a, b = f["valid_from"], f["valid_until"]
        p.ax.plot([x0 + a * scale, x0 + b * scale], [cy, cy], color=STATUS["supported"][0], lw=2.3)
        p.ax.plot(x0 + a * scale, cy, "o", ms=3.5, color=STATUS["supported"][0])
        p.ax.plot(x0 + b * scale, cy, "o", ms=3.5, mfc="white", mec=STATUS["supported"][0])
        p.numeric_claims.append(
            dict(
                panel="timeline", record=r["index"], evidence_ids=f["evidence_ids"], interval=[a, b]
            )
        )
    p.text(
        16,
        top + 108,
        "Shading: request [2,4). Filled start / open end: generated [start,end). Units: days.",
        size=8.5,
        color=MUTED,
    )


def harbor_fixed(data, panels):
    p = Plate(
        "figure_2a_harbor_requests",
        "Figure 2a · One story, two information needs",
        "Synthetic development · Harbor · A: pre-extract/select",
        796,
    )
    y = p.source(data["synthetic"]["stories"]["harbor"], 57, gap=1)
    y = p.flow(y, "A")
    half = (W - 44) / 2
    qa = panels["harbor_A_possession"]["result"]["question"]
    qb = panels["harbor_A_locations"]["result"]["question"]
    end_a = p.question(y, "REQUEST 1 · possession", qa, w=half)
    end_b = p.question(y, "REQUEST 2 · locations", qb, x=28 + half, w=half)
    y = max(end_a, end_b)
    p.text(16, y, "Actual selected ownership / carrying", bold=True, size=8.5)
    p.text(28 + half, y, "Actual selected timed locations", bold=True, size=8.5)
    metric(p, panels["harbor_A_possession"], 16, y + 15)
    metric(p, panels["harbor_A_locations"], 28 + half, y + 15)
    pos1 = {
        "Lena": (38, 25),
        "Omar": (38, 100),
        "Priya": (38, 175),
        "Cobalt Compass": (190, 25),
        "Amber Lantern": (190, 100),
        "Silver Flute": (190, 175),
    }
    routes1 = {
        1: (0, 114, 8),
        2: (0, 114, 83),
        3: (-0.10, 110, 113),
        4: (0.08, 115, 48),
        5: (0, 114, 158),
    }
    network(p, panels["harbor_A_possession"], (16, y + 40), pos1, routes1, node_width=70)
    pos2 = {
        "Lena": (38, 20),
        "Omar": (38, 75),
        "Priya": (38, 130),
        "Silver Flute": (38, 185),
        "North Hall": (190, 20),
        "East Garden": (190, 75),
        "Stone Shed": (190, 130),
        "West Pier": (190, 185),
    }
    routes2 = {i: (0, 114, (i - 1) * 55 + 3) for i in range(1, 5)}
    network(p, panels["harbor_A_locations"], (28 + half, y + 40), pos2, routes2, node_width=70)
    p.para(
        16,
        y + 247,
        "Each view is a fixed selection from the same actual query-blind collection. All returned records are shown; see Figure 2c for the un-clipped intervals and Figure 2b for independent contextual outputs.",
        W - 32,
        size=8.5,
    )
    return p.finish(y + 285)


def harbor_timeline(data, panels):
    p = Plate(
        "figure_2c_harbor_intervals",
        "Figure 2c · Full validity versus request window",
        "Harbor · exact model-generated bounds from the two fixed selections in Figure 2a",
        430,
    )
    y = p.source(
        data["synthetic"]["stories"]["harbor"],
        57,
        selected=["S3", "S4", "S5", "S6", "S7", "S8"],
        label="SOURCE DETAIL · S1–S2 and S9–S14 omitted here",
        gap=2,
    )
    p.para(
        16,
        y,
        "Display-only source excerpt; the model received all 14 sentences shown in Figure 2a. The location request is overlap with [2,4), not a request to rewrite validity bounds.",
        W - 32,
        size=9,
    )
    timeline(p, panels, y + 49)
    return p.finish(y + 216)


def harbor_direct(data, panels):
    p = Plate(
        "figure_2b_harbor_contextual",
        "Figure 2b · Correct intervals, excess information",
        "Same complete Harbor evidence and exact requests as Figure 2a · B: direct contextual extraction",
        714,
    )
    y = p.flow(59, "B")
    p.text(16, y, "Request 1 · Actual possession answer, including all extras", bold=True)
    metric(p, panels["harbor_B_possession"], 16, y + 16)
    pos = {
        "Lena": (65, 35),
        "Omar": (65, 145),
        "Priya": (65, 255),
        "Cobalt Compass": (235, 35),
        "Amber Lantern": (235, 145),
        "Silver Flute": (235, 255),
        "North Hall": (416, 35),
        "East Garden": (416, 110),
        "Stone Shed": (416, 190),
        "West Pier": (416, 275),
    }
    routes = {
        1: (0, 153, 12),
        2: (0, 153, 122),
        3: (-0.08, 120, 185),
        4: (0.06, 152, 72),
        5: (0, 327, 263),
        6: (0, 153, 232),
        7: (-0.34, 326, -5),
        8: (-0.19, 330, 77),
        9: (-0.05, 330, 193),
        10: (0.60, 327, 222),
    }
    network(p, panels["harbor_B_possession"], (16, y + 49), pos, routes, node_width=78, yscale=0.85)
    y += 329
    p.text(16, y, "Request 2 · Actual location answer, including carrying extras", bold=True)
    metric(p, panels["harbor_B_locations"], 16, y + 16)
    pos2 = {
        "Lena": (61, 35),
        "Omar": (61, 92),
        "Priya": (61, 149),
        "Silver Flute": (61, 205),
        "North Hall": (260, 35),
        "East Garden": (260, 92),
        "Stone Shed": (260, 149),
        "West Pier": (260, 205),
        "Cobalt Compass": (425, 149),
        "Amber Lantern": (425, 35),
    }
    routes2 = {
        1: (-0.22, 344, 104),
        2: (-0.27, 346, 3),
        3: (0, 161, 17),
        4: (0, 161, 74),
        5: (0, 161, 131),
        6: (0, 161, 187),
    }
    network(p, panels["harbor_B_locations"], (16, y + 39), pos2, routes2, node_width=78, yscale=0.8)
    end = p.para(
        16,
        y + 230,
        "I = source-supported but irrelevant to this request. All 10 possession and all 6 location predictions remain in the denominators; fewer displayed nodes would not establish improvement.",
        W - 32,
        size=9,
    )
    return p.finish(end + 15)


def alice_actions(data, panels):
    p = Plate(
        "figure_3a_alice_actions",
        "Figure 3a · A useful action graph with a role error",
        "Published narrative · Lewis Carroll, Alice’s Adventures in Wonderland, Chapter I · B output",
        619,
    )
    y = p.source(data["real"]["stories"]["alice"], 57)
    y = p.question(y, "REQUEST 1 · actions", panels["alice_B_actions"]["result"]["question"])
    p.text(16, y, "Text + action question → direct contextual extraction", bold=True)
    metric(p, panels["alice_B_actions"], 16, y + 15)
    pos = {
        "Alice": (80, 64),
        "her sister": (280, 4),
        "bank": (280, 73),
        "the book": (435, 73),
        "White Rabbit": (80, 158),
        "close by her": (392, 158),
    }
    routes = {
        1: (0.04, 184, 6),
        2: (0, 184, 60),
        3: (-0.25, 359, 27),
        4: (0.26, 325, 107),
        5: (0, 245, 143),
    }
    network(p, panels["alice_B_actions"], (16, y + 37), pos, routes, node_width=78)
    y += 224
    end = p.note(
        y,
        "Genuine error and lost specificity",
        "The text assigns reading to Alice's sister; B instead says Alice was reading. 'Sitting by bank' is less specific than 'on the bank'. The Rabbit's 'close by her' wording is retained, not silently converted into an Alice node.",
    )
    return p.finish(end + 2)


def alice_claims(data, panels):
    p = Plate(
        "figure_3b_alice_thoughts",
        "Figure 3b · Thought preserved as sentence-valued nodes",
        "Same full Alice passage · independent B claims output, not a transformation of Figure 3a",
        680,
    )
    y = p.source(data["real"]["stories"]["alice"], 57)
    y = p.question(
        y, "REQUEST 2 · claims and thoughts", panels["alice_B_claims"]["result"]["question"]
    )
    p.text(16, y, "Text + claims question → independent contextual extraction", bold=True)
    metric(p, panels["alice_B_claims"], 16, y + 15)
    y = lanes(p, panels["alice_B_claims"], y + 37)
    end = p.note(
        y + 2,
        "Meaning and required structure are separate",
        "The combined absence and sentence-valued thought preserve 3/4 target meanings in the retained Codex assessment (strict F1 = 0). These are actual object strings, not reconstructed propositions. The daisy-chain record lacks attitude and may imply a settled judgment.",
    )
    return p.finish(end + 2)


def holmes(data, panels):
    p = Plate(
        "figure_s1_holmes",
        "Supplementary S1 · Failure and ambiguity",
        "Published narrative · Arthur Conan Doyle, The Red-Headed League · original failures retained",
        779,
    )
    y = p.source(data["real"]["stories"]["holmes"], 57)
    r = next(
        r
        for r in data["real"]["results"]
        if r["story_id"] == "holmes" and r["task"] == "actions" and r["approach"][0] == "B"
    )
    y = p.question(y, "ACTION REQUEST · exact executed question", r["question"])
    p.text(
        16,
        y,
        "Not an empty graph: two complete responses fail JSON parsing",
        bold=True,
        color=STATUS["unsupported"][0],
    )
    calls = {c["case_id"]: c for c in data["real"]["calls"]}
    y += 23
    for cid, label in [("3", "A · query-blind collection"), ("8", "B · action answer")]:
        raw = calls[cid]["raw_text"]
        p.text(24, y + 8, label, bold=True, size=9)
        p.text(24, y + 25, "Retained final bytes: …" + raw[-54:], size=8.5)
        p.text(24, y + 42, calls[cid]["parse_error"], size=8.5, color=STATUS["unsupported"][0])
        p.rect(16, y, W - 32, 61, fc="#FDF4F1", ec="#D9AAA6")
        y += 69
    y = p.question(
        y,
        "CLAIMS REQUEST · exact executed question",
        panels["holmes_B_claim_detail"]["result"]["question"],
    )
    p.text(16, y, "B · One actual claim (record 2 of 3); not a reconstructed failure", bold=True)
    y = lanes(p, panels["holmes_B_claim_detail"], y + 25)
    end = p.note(
        y,
        "Reference reading and limitation · not model output",
        "The frozen reading treats Mr. Wilson as the addressee and Watson as 'this gentleman'. Local vocative ambiguity remains unresolved. Independently, the generated past partnership has null holder/attitude instead of preserving Holmes's report.",
    )
    return p.finish(end + 2)


def captions(data, panels):
    f1a, f1b = [panels[k]["result"]["evaluation"]["full"]["f1"] for k in ("fable_A", "fable_B")]
    hp, hl = [
        panels[k]["result"]["evaluation"]["full"]["f1"]
        for k in ("harbor_B_possession", "harbor_B_locations")
    ]
    return {
        "figure_1a_fable_overview": (
            "Figure 1a. Aesop's complete The Lion and the Mouse (Townsend translation, Project Gutenberg #21), with the executed action question. The pre-request panel contains every actual query-blind record citing S2 or S3: five of twelve records, including the incomplete 'spare' claim and underdefined laughter object. Seven undisplayed records remain in the full-output supplement. The pipeline arrow denotes the actual A computation, not a derivation of B. Status colours/symbols reproduce the retained Codex assessment; this is exploratory extraction, not registered C1/C2 acceptance."
        ),
        "figure_1b_fable_comparison": (
            f"Figure 1b. Complete A and B answers to the same fable action question shown in Figure 1a. A selects eight existing records; B independently generates five records from the complete text plus question. Strict qualified-fact F1 is {f1a:.3f} for A and {f1b:.3f} for B, with full output denominators. B captures rescue relationships but omits the Lion's earlier capture and release of the Mouse, and loses the face-specific running endpoint. Some inverse wording is semantically supported while failing the frozen strict matcher. Missing reference content appears only in the callout. Literal names and parallel assertions remain distinct."
        ),
        "figure_2a_harbor_requests": (
            "Figure 2a. The complete synthetic Harbor story and its exact executed possession and interval-location questions. Both graphs are the complete A selections from one actual query-blind collection, not independent reconstructions. All five possession and four location predictions match their respective references (strict qualified-fact F1 = 1.000 for each). The location view preserves full generated intervals rather than clipping them to the [2,4) question window. This illustrates fixed projection and explicit qualification, not novel ontology construction. Full pre-extraction and the other approach are retained in the supplement and Figure 2b."
        ),
        "figure_2b_harbor_contextual": (
            f"Figure 2b. Complete independent B contextual answers to the two Harbor questions in Figure 2a. The ten-record possession answer contains five source-supported but irrelevant location facts; the six-record location answer contains two irrelevant carrying facts. Their strict qualified F1 values are {hp:.3f} and {hl:.3f}. These facts are visibly marked I and remain in all metric denominators. Correct intervals do not establish correct selection. The two graphs are not obtained by filtering A, and their layouts or node counts are not evidence of ontology construction."
        ),
        "figure_2c_harbor_intervals": (
            "Figure 2c. Full model-generated bounds from A's Harbor possession and location selections. The six quoted source sentences support the six timeline bars; the model received all fourteen sentences, not this display excerpt. Filled starts and open ends denote [start,end); the shaded [2,4) region is solely the user's location window. Carrying bars are included to show the ownership/possession request, not to declare them relevant location answers. Source order and full bounds are retained; no onset or endpoint is inferred or clipped. The exact questions appear in Figure 2a and the caption companion."
        ),
        "figure_3a_alice_actions": (
            "Figure 3a. The complete opening Alice passage (Carroll, Project Gutenberg #11) and B's exact executed action question. All five actual assertions are drawn. Correct inspection and Rabbit movement coexist with an unsupported reader assignment: the source says the sister reads, but the graph says Alice reads. 'Sitting by bank' is less specific than the narrated on-bank location. Strict qualified F1 is 0.400. The phrase 'close by her' remains its own literal endpoint rather than being silently normalized to Alice. Source-based status is the retained Codex assessment, not an independent review or a new score."
        ),
        "figure_3b_alice_thoughts": (
            "Figure 3b. The same complete Alice evidence with the independently executed B claims question. All three actual records are shown as assertion-lane graphs; sentence-valued objects remain model-authored node labels. The combined missing-content object and rhetorical thought preserve three of four reference concepts in the retained Codex semantic assessment, although strict qualified F1 remains 0.000. The daisy-chain record supplies holder Alice with null attitude and may turn consideration into a positive judgment. No decomposed proposition or missing qualification is inserted. Semantic coverage is not strict F1 or scientific acceptance."
        ),
        "figure_s1_holmes": (
            "Supplementary Figure S1. The complete Holmes excerpt (Doyle, The Red-Headed League, Project Gutenberg #1661), exact action and claims questions, and retained failures. The A pre-extraction and B action answer terminate with an extra closing brace; no empty or reconstructed successful graph is drawn. The single displayed usable claim is B claims record two of three, chosen to expose the locally ambiguous vocative participant reading and absent attribution. The other two claims and complete failed strings are in the supplement. The reference reading is explicitly not model output. All original strict-parser failures and scoped scores remain unchanged."
        ),
    }


def validate_plate(p):
    p.fig.canvas.draw()
    renderer = p.fig.canvas.get_renderer()
    bad = []
    for t in p.text_items:
        b = t.get_window_extent(renderer).transformed(p.ax.transData.inverted())
        x0, x1 = sorted((b.x0, b.x1))
        y0, y1 = sorted((b.y0, b.y1))
        if x0 < 5 or x1 > W - 5 or y0 < 4 or y1 > p.height - 5:
            bad.append(dict(text=t.get_text(), bounds=[x0, y0, x1, y1]))
    if bad:
        raise ValueError(f"Text outside {p.name}: {bad}")


def embed_svg(svg_path, p):
    """Keep searchable text and embed the installed DejaVu font subsets for portability."""
    ns = "http://www.w3.org/2000/svg"
    ET.register_namespace("", ns)
    ET.register_namespace("xlink", "http://www.w3.org/1999/xlink")
    root = ET.parse(svg_path).getroot()
    defs = root.find(f"{{{ns}}}defs")
    if defs is None:
        defs = ET.SubElement(root, f"{{{ns}}}defs")
    characters = "".join(t.get_text() for t in p.text_items)
    family = "Paper-" + p.name
    # Each inline SVG has its own subset. Never let a later plate override an
    # earlier plate's font or clipping identifiers in the interactive document.
    id_map = {
        el.get("id"): p.name + "-" + el.get("id")
        for el in root.iter()
        if el.get("id") and not el.get("id").startswith(("edge-", "source-"))
    }
    for el in root.iter():
        for key, value in list(el.attrib.items()):
            if key == "id" and value in id_map:
                el.set(key, id_map[value])
            else:
                value = value.replace("DejaVu Sans", family)
                for old, new in id_map.items():
                    value = value.replace("url(#" + old + ")", "url(#" + new + ")")
                    if value == "#" + old:
                        value = "#" + new
                el.set(key, value)
    css = []
    font_hashes = {}
    for weight in ("normal", "bold"):
        path = Path(findfont(FontProperties(family="DejaVu Sans", weight=weight)))
        font_hashes[weight] = sha(path)
        font = TTFont(path, recalcTimestamp=False)
        if "FFTM" in font:
            del font["FFTM"]
        options = subset.Options()
        options.recalc_timestamp = False
        sub = subset.Subsetter(options=options)
        sub.populate(text=characters)
        sub.subset(font)
        font.flavor = "woff"
        buf = io.BytesIO()
        font.save(buf)
        css.append(
            "@font-face{font-family:'"
            + family
            + "';font-weight:"
            + weight
            + ";src:url(data:font/woff;base64,"
            + base64.b64encode(buf.getvalue()).decode()
            + ") format('woff');}"
        )
    ET.SubElement(defs, f"{{{ns}}}style").text = "\n".join(css)
    ET.SubElement(root, f"{{{ns}}}metadata").text = (
        ROOT / "configs/visualization/DEJAVU_FONT_LICENSE.txt"
    ).read_text()
    for item in p.edges:
        for gid in item["artist_ids"]:
            g = next(node for node in root.iter() if node.get("id") == gid)
            g.set("data-record", item["id"])
            g.set("role", "button")
            g.set("tabindex", "0")
            ET.SubElement(g, f"{{{ns}}}title").text = (
                item["status"]
                + ": "
                + json.dumps(item["fact"], ensure_ascii=False)
                + ". Citation link only; "
                + item["note"]
            )
    ET.ElementTree(root).write(svg_path, encoding="utf-8", xml_declaration=True)
    return font_hashes


def full_supplement(data, panels, path):
    """Full retained outputs in an explicit edge-lane display, not repaired graphs."""
    output = []
    # Complete pre-extractions plus every selected main output; add both Alice A views.
    for dataset, story, cid in [("real", "fable", "1"), ("synthetic", "harbor", "1")]:
        d = data[dataset]
        c = next(c for c in d["calls"] if c["case_id"] == cid)
        notes = next(
            (
                r["semantic_assessments"]
                for r in d.get("preextract_assessments", [])
                if r["story_id"] == story
            ),
            None,
        )
        records = []
        for i, f in enumerate(c["parsed"]["facts"], 1):
            if notes:
                state = notes[i - 1]["support"]
                note = notes[i - 1]["note"]
            else:
                state = (
                    "supported"
                    if c["evaluation"]["rows"][i - 1]["status"] == "correct_complete_fact"
                    else "unresolved"
                )
                note = c["evaluation"]["rows"][i - 1]["status"]
            records.append(dict(index=i, fact=f, status=state, note=note))
        output.append(
            dict(
                id=story + "_pre_full",
                dataset=dataset,
                story=story,
                records=records,
                source_call=cid,
                response_sha256=c["response"]["response_sha256"],
                question="Query-blind extraction: no user question supplied.",
            )
        )
    wanted = {
        (v["dataset"], v["story"], v.get("task"), v.get("method"))
        for v in read(CONFIG)["panels"].values()
        if "task" in v
    }
    wanted.update({("real", "alice", "actions", "A"), ("real", "alice", "claims", "A")})
    for dataset, story, task, method in sorted(wanted):
        r = next(
            r
            for r in data[dataset]["results"]
            if r["story_id"] == story and r["task"] == task and r["approach"][0] == method
        )
        if not r["parseable"]:
            continue
        records = []
        for i, f in enumerate(r["parsed"]["facts"], 1):
            if "semantic_assessments" in r:
                n = r["semantic_assessments"][i - 1]
                state = n["support"]
                note = n["note"]
                if state == "supported" and not n["relevant"]:
                    state = "irrelevant"
            else:
                status = r["evaluation"]["rows"][i - 1]["status"]
                state = {
                    "correct_complete_fact": "supported",
                    "supported_but_irrelevant": "irrelevant",
                }.get(status, "unresolved")
                note = status
            records.append(dict(index=i, fact=f, status=state, note=note))
        output.append(
            dict(
                id=f"{story}_{task}_{method}_full",
                dataset=dataset,
                story=story,
                records=records,
                source_call=r["source_case_id"],
                response_sha256=r["source_response_hash"],
                question=r["question"],
            )
        )
    with PdfPages(path, metadata=PDF_META) as pdf:
        for panel in output:
            p = Plate(
                "supp_" + panel["id"],
                "Full output · " + panel["id"].replace("_", " "),
                "Assertion-lane graph · repeated endpoint strings denote the same node; every record retained",
                1100,
            )
            y = p.question(57, "EXECUTED QUESTION / PRE-QUERY STATUS", panel["question"])
            end = lanes(p, panel, y)
            p.finish(end + 8)
            validate_plate(p)
            pdf.savefig(p.fig)
            plt.close(p.fig)
        for cid in ("3", "8"):
            c = next(c for c in data["real"]["calls"] if c["case_id"] == cid)
            p = Plate(
                "raw_holmes_" + cid,
                "Unparseable raw output · Holmes "
                + ("pre-extraction" if cid == "3" else "action answer"),
                "Not an empty or repaired graph · original JSON failure remains",
                1000,
            )
            y = p.para(16, 59, c["parse_error"], W - 32, color=STATUS["unsupported"][0]) + 15
            # Word wrapping preserves the entire raw string in the public JSON and inspector;
            # here hard wrapping changes only display line breaks, not fields or syntax.
            raw = "\n".join(
                textwrap.wrap(
                    c["raw_text"], width=92, break_long_words=True, break_on_hyphens=False
                )
            )
            y = p.para(16, y, raw, W - 32, size=8.5) + 15
            p.finish(y)
            validate_plate(p)
            pdf.savefig(p.fig)
            plt.close(p.fig)
    return output


def write_index(out, plates, data, captions_by_name, appendix):
    e = html.escape
    body = [
        """<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>StoryProjectionOnto · Paper figures</title>
<style>body{font:16px/1.5 system-ui;color:#142d40;margin:24px auto;max-width:1400px;padding:0 18px}a{color:#315d82}nav a{margin-right:16px}.plate{max-width:760px}.plate svg{width:100%;height:auto}.layout{display:grid;grid-template-columns:minmax(500px,780px) minmax(320px,1fr);gap:30px}.source{position:sticky;top:16px;max-height:92vh;overflow:auto;background:#f4f7f8;padding:18px;border:1px solid #ccd6de}.evidence{padding:5px;border-left:4px solid transparent}.cited{background:#fff1b5;border-color:#735094}.selected text{font-weight:bold!important;text-decoration:underline}.selected path{stroke-width:2!important}[data-record]{cursor:pointer}pre{white-space:pre-wrap;overflow-wrap:anywhere;font:13px/1.45 ui-monospace}section{border-top:1px solid #ccd6de;margin:28px 0;padding-top:18px}summary{cursor:pointer}#inspector{background:#fff;border:1px solid #bccad5;padding:12px}.tag{font-weight:bold}.caption{max-width:900px}@media(max-width:950px){.layout{display:block}.source{position:static;max-height:none}.plate{max-width:100%}}</style>
<h1>Paper figures · exploratory extraction</h1><p>Retained model outputs only. No new generation or rescoring. Click or keyboard-select an edge/label to highlight its <em>cited</em> evidence and inspect the exact record. Citation highlighting is not a support verdict; the retained assessment is shown separately. Symbols: + supported, ~ partial, x unsupported, ? unresolved, I source-supported but irrelevant.</p>
<nav><a href="../../PAPER_FIGURES.pdf">Review PDF</a><a href="../../PAPER_FIGURE_CAPTIONS.md">Captions + exact questions</a><a href="FULL_OUTPUT_SUPPLEMENT.pdf">All selected outputs in full</a><a href="../real_text_proof_of_concept.html">Original prose comparison</a><a href="../compact_story_v2_comparison.html">Original synthetic comparison</a><a href="figure_manifest.json">Manifest</a></nav><div class="layout"><main>"""
    ]
    mapping = {}
    for plate in plates:
        name = plate.name
        svg = (out / (name + ".svg")).read_text()
        svg = svg[svg.index("<svg") :]
        body += [
            f'<section id="{name}"><h2>{e(name.replace("_", " "))}</h2><p><a href="{name}.pdf">PDF</a> · <a href="{name}.svg">SVG</a> · <a href="{name}.png">400 dpi PNG</a></p><div class="plate">'
            + svg
            + '</div><p class="caption">'
            + e(captions_by_name[name])
            + "</p></section>"
        ]
        mapping.update({row["id"]: row for row in plate.edges})
    body += [
        "<section><h2>Complete output records and all qualifications</h2><p>Same immutable facts as the full-output PDF; no record is omitted from these lists.</p>"
    ]
    for panel in appendix:
        body += [
            "<details><summary>"
            + e(panel["id"])
            + f" · {len(panel['records'])} records</summary><pre>"
            + e(json.dumps(panel, ensure_ascii=False, indent=2))
            + "</pre></details>"
        ]
    body += [
        "</section></main><aside><div class='source'><h2>Cited evidence</h2><p id='help'>Select a graph edge. Highlighting links to the source; it does not verify the generated claim.</p><div id='inspector' aria-live='polite'>No edge selected.</div>"
    ]
    for dataset, story in [
        ("real", "fable"),
        ("synthetic", "harbor"),
        ("real", "alice"),
        ("real", "holmes"),
    ]:
        source = data[dataset]["stories"][story]
        body += [
            f'<details class="story" id="story-{dataset}-{story}"><summary>{e(source["title"])}</summary>'
        ]
        for eid, text in sorted(source["evidence"].items(), key=lambda kv: int(kv[0][1:])):
            # Character references preserve the original CR/LF in the DOM without
            # introducing trailing whitespace in the generated HTML source file.
            source_text = e(text).replace("\r", "&#13;").replace("\n", "&#10;")
            body += [
                f'<div class="evidence" id="evidence-{dataset}-{story}-{eid}"><b>{e(eid)}</b> {source_text}</div>'
            ]
        if "source_url" in source:
            body += [
                '<p><a href="'
                + e(source["source_url"])
                + '">Original source and attribution</a> · <a href="../../../'
                + e(source["notice_file"])
                + '">Retained source notices</a></p>'
            ]
        body += ["</details>"]
    body += [
        "</div></aside></div><script type='application/json' id='edge-data'>"
        + json.dumps(mapping, ensure_ascii=False).replace("<", "\\u003c")
        + "</script>"
    ]
    body += [
        """<script>
const records=JSON.parse(document.getElementById('edge-data').textContent);
function choose(id){const r=records[id];document.querySelectorAll('.cited,.selected').forEach(el=>el.classList.remove('cited','selected'));document.querySelectorAll('[data-record]').forEach(el=>{if(el.dataset.record===id)el.classList.add('selected')});document.querySelectorAll('.story').forEach(el=>el.open=false);document.getElementById('story-'+r.dataset+'-'+r.story).open=true;r.fact.evidence_ids.forEach(e=>document.getElementById('evidence-'+r.dataset+'-'+r.story+'-'+e)?.classList.add('cited'));const box=document.getElementById('inspector');box.replaceChildren();const title=document.createElement('strong');title.textContent='Assessed status: '+r.status+' — citation link only';const note=document.createElement('p');note.textContent=r.note;const pre=document.createElement('pre');pre.textContent=JSON.stringify(r.fact,null,2);box.append(title,note,pre);}
document.querySelectorAll('[data-record]').forEach(el=>{el.addEventListener('click',()=>choose(el.dataset.record));el.addEventListener('keydown',event=>{if(event.key==='Enter'||event.key===' '){event.preventDefault();choose(el.dataset.record)}})});
</script></html>"""
    ]
    (out / "index.html").write_text("\n".join(body))


def build(output_root):
    config, data, panels, source_hashes = load_inputs()
    out = output_root / "figures/paper"
    out.mkdir(parents=True, exist_ok=True)
    (out / "DEJAVU_FONT_LICENSE.txt").write_bytes(
        (ROOT / "configs/visualization/DEJAVU_FONT_LICENSE.txt").read_bytes()
    )
    fns = [
        fable_overview,
        fable_comparison,
        harbor_fixed,
        harbor_direct,
        harbor_timeline,
        alice_actions,
        alice_claims,
        holmes,
    ]
    plates = [fn(data, panels) for fn in fns]
    caps = captions(data, panels)
    manifest = dict(
        version=config["version"],
        scope=config["scope"],
        generation_command="python -m scripts.build_paper_figures",
        generator_sha256=sha(__file__),
        configuration_sha256=sha(CONFIG),
        source_artifacts=source_hashes,
        selection_rationale=config["selection_rationale"],
        transformations=config["transformations"],
        width_mm=config["width_mm"],
        minimum_font_pt=config["minimum_font_pt"],
        png_dpi=config["png_dpi"],
        matplotlib_version=matplotlib.__version__,
        panels=panels,
        figures=[],
    )
    for p in plates:
        validate_plate(p)
        p.fig.savefig(out / (p.name + ".pdf"), metadata=PDF_META)
        p.fig.savefig(
            out / (p.name + ".svg"), metadata={"Date": None, "Creator": PDF_META["Creator"]}
        )
        fonts = embed_svg(out / (p.name + ".svg"), p)
        subprocess.run(
            [
                "pdftoppm",
                "-r",
                str(config["png_dpi"]),
                "-png",
                "-singlefile",
                str(out / (p.name + ".pdf")),
                str(out / p.name),
            ],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
        )
        manifest["figures"].append(
            dict(
                name=p.name,
                width_pt=W,
                height_pt=p.height,
                font_sha256=fonts,
                panels=p.panels,
                displayed_edges=p.edges,
                numeric_claims=p.numeric_claims,
                caption=caps[p.name],
            )
        )
    appendix = full_supplement(data, panels, out / "FULL_OUTPUT_SUPPLEMENT.pdf")
    manifest["full_supplement"] = appendix
    write_index(out, plates, data, caps, appendix)
    md = [
        "# Paper figure captions",
        "",
        "All figures: exploratory extraction from retained outputs, not registered C1/C2 acceptance or evidence of novel ontology construction. 178 mm width; text at least 8.5 pt. References and published-prose semantic assessments are Codex-authored, not independent human review.",
        "",
    ]
    for name, caption in caps.items():
        md += [
            "## " + name,
            "",
            caption,
            "",
            f"[PDF](figures/paper/{name}.pdf) · [SVG](figures/paper/{name}.svg) · [PNG](figures/paper/{name}.png)",
            "",
        ]
    md += ["## Exact executed questions", ""]
    seen = set()
    for panel in panels.values():
        r = panel["result"]
        if r and (panel["story"], r["question"]) not in seen:
            md += ["### " + panel["story"] + " / " + r["task"], "", r["question"], ""]
            seen.add((panel["story"], r["question"]))
    action = next(
        r["question"]
        for r in data["real"]["results"]
        if r["story_id"] == "holmes" and r["task"] == "actions"
    )
    md += ["### holmes / actions", "", action, "", "## Selection and display rationale", ""]
    for story, why in config["selection_rationale"].items():
        md += [f"- **{story}:** {why}"]
    md += [
        "",
        "All main panels are complete except the five-record fable pre-extraction detail and the single Holmes claim. Full outputs remain in [the supplement](figures/paper/FULL_OUTPUT_SUPPLEMENT.pdf) and [the interactive index](figures/paper/index.html). Undisplayed records still count in the original metrics. Graphs and metrics are copied from the canonical retained tables, never rescored for figure selection.",
        "",
        "Predicate underscores become spaces and evidence whitespace is reflowed only for typography. Exact strings, malformed fields, request/response identities and source hashes are in the manifest and interactive inspector. Assertion-lane panels repeat glyphs for identical endpoint strings; they do not create or merge semantic nodes.",
        "",
        "Regenerate: `python -m scripts.build_paper_figures`.",
        "",
    ]
    (output_root / "PAPER_FIGURE_CAPTIONS.md").write_text("\n".join(md))
    with PdfPages(output_root / "PAPER_FIGURES.pdf", metadata=PDF_META) as pdf:
        for p in plates:
            old = p.height
            cap_lines = wrap(caps[p.name], W - 32, 9)
            p.height = old + len(cap_lines) * 11.7 + 34
            p.fig.set_size_inches(W / 72, p.height / 72)
            p.ax.set_ylim(p.height, 0)
            p.para(16, old + 12, caps[p.name], W - 32, size=9)
            validate_plate(p)
            pdf.savefig(p.fig)
            plt.close(p.fig)
    files = (
        list(out.glob("*.pdf"))
        + list(out.glob("*.svg"))
        + list(out.glob("*.png"))
        + [
            out / "DEJAVU_FONT_LICENSE.txt",
            out / "index.html",
            output_root / "PAPER_FIGURES.pdf",
            output_root / "PAPER_FIGURE_CAPTIONS.md",
        ]
    )
    manifest["files"] = {str(f.relative_to(output_root)): sha(f) for f in sorted(files)}
    (out / "figure_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    )
    assert all(sha(ROOT / path) == digest for path, digest in source_hashes.items()), (
        "Source artifact changed"
    )
    print(
        json.dumps(
            {
                "plates": len(plates),
                "full_output_graphs": len(appendix),
                "output": str(out),
                "source_artifacts_unchanged": True,
            }
        )
    )
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=ROOT / "reports")
    build(parser.parse_args().output_root)
