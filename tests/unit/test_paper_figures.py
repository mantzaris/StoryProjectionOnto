"""Display-only integrity checks against retained outputs, never new scoring."""

import html
import re
import subprocess
from collections import Counter
from xml.etree import ElementTree as ET

import pytest

from scripts import build_paper_figures as p

OUT = p.ROOT / "reports/figures/paper"


@pytest.fixture(scope="module")
def inputs():
    return p.load_inputs()


@pytest.fixture(scope="module")
def manifest():
    return p.read(OUT / "figure_manifest.json")


def test_retained_sources_and_all_exports_have_unchanged_hashes(inputs, manifest):
    assert manifest["source_artifacts"] == inputs[3]
    assert manifest["generator_sha256"] == p.sha(p.__file__)
    assert manifest["configuration_sha256"] == p.sha(p.CONFIG)
    for path, digest in manifest["files"].items():
        assert p.sha(p.ROOT / "reports" / path) == digest


def test_selected_records_exact_no_alias_merging_or_semantic_repair(inputs, manifest):
    _, data, panels, _ = inputs
    assert manifest["panels"] == panels
    for panel in panels.values():
        if panel["result"]:
            originals = panel["result"]["parsed"]["facts"]
        else:
            originals = next(
                c["parsed"]["facts"]
                for c in data[panel["dataset"]]["calls"]
                if c["case_id"] == panel["source_call"]
            )
        for record in panel["records"]:
            assert record["fact"] == originals[record["index"] - 1]
        assert len(originals) == len(panel["records"]) + panel["omitted_count"]
    assert panels["fable_pre_detail"]["omitted_count"] == 7
    assert panels["holmes_B_claim_detail"]["omitted_count"] == 2
    assert all(
        panel["omitted_count"] == 0
        for k, panel in panels.items()
        if k not in {"fable_pre_detail", "holmes_B_claim_detail"}
    )
    assert panels["fable_pre_detail"]["records"][2]["fact"]["valid until"] is None


def test_every_graph_record_is_drawn_once_with_actual_citations(inputs, manifest):
    _, data, panels, _ = inputs
    for figure in manifest["figures"]:
        expected = Counter(
            (key, r["index"]) for key in figure["panels"] for r in panels[key]["records"]
        )
        actual = Counter((r["panel"], r["index"]) for r in figure["displayed_edges"])
        assert actual == expected
        root = ET.parse(OUT / (figure["name"] + ".svg")).getroot()
        ids = [el.get("id") for el in root.iter() if el.get("id")]
        assert len(ids) == len(set(ids))
        for r in figure["displayed_edges"]:
            assert set(r["fact"]["evidence_ids"]) <= set(
                data[r["dataset"]]["stories"][r["story"]]["evidence"]
            )
            assert all(gid in ids for gid in r["artist_ids"])


def test_numeric_claims_use_full_denominators_and_actual_intervals(inputs, manifest):
    panels = inputs[2]
    for figure in manifest["figures"]:
        for claim in figure["numeric_claims"]:
            if claim["panel"] == "timeline":
                candidates = [
                    r["fact"]
                    for key in ("harbor_A_possession", "harbor_A_locations")
                    for r in panels[key]["records"]
                    if r["fact"]["evidence_ids"] == claim["evidence_ids"]
                ]
                assert len(candidates) == 1
                assert claim["interval"] == [
                    candidates[0]["valid_from"],
                    candidates[0]["valid_until"],
                ]
            else:
                assert (
                    claim["value"] == panels[claim["panel"]]["result"]["evaluation"]["full"]["f1"]
                )
    for key, n, f1 in [("harbor_B_possession", 10, 2 / 3), ("harbor_B_locations", 6, 0.8)]:
        ev = panels[key]["result"]["evaluation"]["full"]
        assert ev["predicted"] == n
        assert ev["f1"] == pytest.approx(f1)
    table = next(
        r
        for r in inputs[1]["real"]["table"]
        if r["story"] == "alice" and r["task"] == "claims" and r["approach"].startswith("B")
    )
    assert table["semantic_coverage"] == 0.75


def test_exact_executed_questions_and_original_words_retained(inputs):
    captions = (p.ROOT / "reports/PAPER_FIGURE_CAPTIONS.md").read_text()
    for panel in inputs[2].values():
        if panel["result"]:
            assert panel["result"]["question"] in captions
    root = ET.parse(OUT / "figure_1a_fable_overview.svg").getroot()
    texts = [el.text for el in root.iter() if el.tag.endswith("}text") and el.text]
    joined = " ".join(texts)
    for value in inputs[1]["real"]["stories"]["fable"]["evidence"].values():
        assert " ".join(value.split()) in joined
    index = (OUT / "index.html").read_text()
    for dataset, story in [
        ("real", "fable"),
        ("real", "alice"),
        ("real", "holmes"),
        ("synthetic", "harbor"),
    ]:
        for eid, value in inputs[1][dataset]["stories"][story]["evidence"].items():
            pattern = rf'id="evidence-{dataset}-{story}-{eid}"><b>{eid}</b> (.*?)</div>'
            match = re.search(pattern, index, re.DOTALL)
            assert match is not None
            assert html.unescape(match.group(1)) == value


def test_holmes_failures_never_become_empty_or_reconstructed_graphs(inputs, manifest):
    calls = {c["case_id"]: c for c in inputs[1]["real"]["calls"]}
    for cid in ("3", "8"):
        assert calls[cid]["parsed"] is None
        assert calls[cid]["raw_text"].endswith("]}}")
        assert "Extra data" in calls[cid]["parse_error"]
    figure = next(f for f in manifest["figures"] if f["name"] == "figure_s1_holmes")
    assert {r["panel"] for r in figure["displayed_edges"]} == {"holmes_B_claim_detail"}
    assert "not model output" in figure["caption"]


def test_full_supplement_preserves_every_record_and_reference_identity(inputs, manifest):
    assert len(manifest["full_supplement"]) == 13
    for panel in manifest["full_supplement"]:
        d = inputs[1][panel["dataset"]]
        if "pre_full" in panel["id"]:
            expected = next(
                c["parsed"]["facts"] for c in d["calls"] if c["case_id"] == panel["source_call"]
            )
        else:
            _, task, approach, _ = panel["id"].split("_")
            expected = next(
                r["parsed"]["facts"]
                for r in d["results"]
                if r["story_id"] == panel["story"]
                and r["task"] == task
                and r["approach"].startswith(approach)
            )
        assert [r["fact"] for r in panel["records"]] == expected


def test_pdf_and_svg_paper_dimensions_searchable_embedded_fonts(manifest):
    assert len(manifest["figures"]) == 8
    all_ids = []
    for f in manifest["figures"]:
        root = ET.parse(OUT / (f["name"] + ".svg")).getroot()
        assert float(root.get("width").removesuffix("pt")) == pytest.approx(178 / 25.4 * 72)
        all_ids.extend(el.get("id") for el in root.iter() if el.get("id"))
        for el in root.iter():
            if el.tag.endswith("}text"):
                sizes = re.findall(r"([\d.]+)px", el.get("style", ""))
                assert sizes and all(float(size) >= 8.5 for size in sizes)
        svg = (OUT / (f["name"] + ".svg")).read_text()
        assert "data:font/woff;base64," in svg
        assert "font-family:'Paper-" + f["name"] in svg
        fonts = subprocess.check_output(["pdffonts", str(OUT / (f["name"] + ".pdf"))], text=True)
        assert "DejaVuSans" in fonts
        for line in fonts.splitlines()[2:]:
            assert re.search(r"yes\s+yes\s+yes", line)
        text = subprocess.check_output(
            ["pdftotext", str(OUT / (f["name"] + ".pdf")), "-"], text=True
        )
        assert "unresolved" in text and "citation badge" in text
    assert len(all_ids) == len(set(all_ids))
    info = subprocess.check_output(
        ["pdfinfo", str(p.ROOT / "reports/PAPER_FIGURES.pdf")], text=True
    )
    assert re.search(r"Pages:\s+8", info)
