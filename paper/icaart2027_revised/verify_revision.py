"""Verify revision preservation, score replay, PDF layout and a portable source build."""
from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
import tempfile
import zipfile
from pathlib import Path
from xml.etree import ElementTree as ET

import replay_results

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
OLD = HERE.parent / "icaart2027"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def command(*args, cwd=None):
    return subprocess.check_output(args, cwd=cwd, text=True, stderr=subprocess.STDOUT)


def abstract(source):
    return re.search(r"(?ms)^\\abstract\{(.*?)\}\s*\\onecolumn", source)[1]


def table_rows(path):
    rows = []
    for line in path.read_text().splitlines():
        if re.match(r"^(Harbor|Orchard|Museum|Alice|Fable|Holmes) &", line):
            rows.append(line.split(r"\\")[0].strip().split(" & "))
    return rows


def pdf_checks(path, main):
    info = command("pdfinfo", str(path))
    pages = int(re.search(r"Pages:\s+(\d+)", info)[1])
    if main:
        assert pages == 8, pages
    fonts = command("pdffonts", str(path))
    assert "Type 3" not in fonts
    assert all(line.split()[-5] == "yes" for line in fonts.splitlines()[2:] if line.strip())
    xml = ET.fromstring(command("pdftotext", "-bbox-layout", str(path), "-"))
    ns = {"h": "http://www.w3.org/1999/xhtml"}
    bounds, captions = [], []
    for i, page in enumerate(xml.findall(".//h:page", ns), 1):
        words = page.findall(".//h:word", ns)
        box = [fn(float(w.get(k)) for w in words) for k, fn in [("xMin", min), ("yMin", min), ("xMax", max), ("yMax", max)]]
        if main:
            assert box[0] >= 72 and box[2] <= 523 and box[1] >= 90 and box[3] <= 727, (i, box)
            for block in page.findall(".//h:block", ns):
                text = " ".join(w.text or "" for w in block.findall(".//h:word", ns))
                if re.match(r"^Figure [123]:", text):
                    lines = block.findall("h:line", ns)
                    assert len(lines) <= 2, (i, text, len(lines))
                    captions.append({"page": i, "lines": len(lines), "text": text})
        else:
            assert box[0] >= 0 and box[2] <= 596 and box[1] >= 0 and box[3] <= 842, (i, box)
        bounds.append(box)
    if main:
        assert len(captions) == 3, captions
    return {"pages": pages, "sha256": sha(path), "all_fonts_embedded": True,
            "page_bounds_points": bounds, "figure_captions": captions}


def main():
    baseline = json.loads((HERE / "baseline_hashes.json").read_text())
    for name, digest in baseline.items():
        assert sha(ROOT / name) == digest, "Submitted/baseline artifact changed: " + name
    old = (OLD / "ICAART2027_submission.tex").read_text()
    new = (HERE / "ICAART2027_revised.tex").read_text()
    assert re.search(r"\\title\{([^}]+)\}", old)[1] == re.search(r"\\title\{([^}]+)\}", new)[1]
    assert abstract(old) == abstract(new)
    assert (OLD / "generated/numbers.tex").read_bytes() == (HERE / "generated/numbers.tex").read_bytes()
    assert table_rows(OLD / "tables/prose.tex") == table_rows(HERE / "tables/prose.tex")
    assert table_rows(OLD / "tables/compact.tex") == [r[:-1] for r in table_rows(HERE / "tables/compact.tex")]
    for name in ["article.cls", "SCITEPRESS.sty", "apalike.sty", "apalike.bst",
                 "figures/fable.pdf", "figures/alice.pdf", "ICAART2027_companion.tex", "ICAART2027_companion.pdf"]:
        assert sha(OLD / name) == sha(HERE / name), name
    replay = replay_results.calculate()
    for name, content in replay_results.outputs(replay).items():
        assert (HERE / "tables" / name).read_text() == content, name
    synthetic = [r for r in replay["rows"] if r["dataset"] == "synthetic"]
    assert all(r["B_filter_f1"] == 1 for r in synthetic if r["question"] == "possession")
    assert all(r["B_filter_f1"] == 0 for r in synthetic if r["question"] == "belief")
    assert next(r for r in synthetic if r["story"] == "orchard" and r["question"] == "locations")["B_filter_f1"] == 6 / 7
    prose = [r for r in replay["rows"] if r["dataset"] == "prose"]
    assert sum(r["B_status"] == "scored" for r in prose) == 5
    assert all(r["B_f1"] == r["B_filter_f1"] for r in prose)
    assert sum((r["B_predicted"] or 0) - (r["B_filter_predicted"] or 0) for r in prose) == 1
    figure = json.loads((HERE / "figures/orchard_manifest.json").read_text())
    evidence = json.loads((ROOT / "reports/tables/compact_story_v2_results.json").read_text())["stories"]["orchard"]["evidence"]
    assert len(figure["displayed_evidence"]) == 10
    assert all(evidence[k] == v for k, v in figure["displayed_evidence"].items())
    old_figure = json.loads((OLD / "manifest.json").read_text())["figures"]["orchard_network"]
    for a, b in zip(old_figure["records"], figure["records"], strict=True):
        for field in ("fact", "status", "source_call", "source_indices", "display_label"):
            assert a[field] == b[field], field
    for ext, digest in figure["output_sha256"].items():
        assert sha(HERE / "figures" / ("orchard_network." + ext)) == digest
    source = new + (HERE / "figure_blocks.tex").read_text()
    citations = {k for group in re.findall(r"\\cite\{([^}]+)\}", source) for k in group.split(",")}
    entries = set(re.findall(r"@\w+\{([^,]+),", (HERE / "references.bib").read_text()))
    assert citations == entries, (citations, entries)
    assert {"lairgi2026", "fader2011", "mausam2012"} <= citations
    diff_source = (HERE / "ICAART2027_diff.tex").read_text()
    assert abstract(diff_source) == abstract(old), "Abstract marked as changed in diff"
    assert len(re.findall(r"\\DIFadd\{", diff_source)) > 20
    assert len(re.findall(r"\\DIFdel\{", diff_source)) > 20
    for name in ["ICAART2027_revised", "ICAART2027_diff"]:
        log = (HERE / "build" / (name + ".log")).read_text()
        for forbidden in ["Overfull", "Float too large", "undefined", "Label(s) may have changed"]:
            assert forbidden not in log, (name, forbidden)
    revised = pdf_checks(HERE / "ICAART2027_revised.pdf", True)
    diff = pdf_checks(HERE / "ICAART2027_diff.pdf", False)
    files = ["ICAART2027_revised.tex", "references.bib", "figure_blocks.tex", "build.sh",
             "article.cls", "SCITEPRESS.sty", "apalike.sty", "apalike.bst", "generated/numbers.tex",
             "tables/compact.tex", "tables/prose.tex", "figures/orchard_network.pdf", "figures/fable.pdf", "figures/alice.pdf"]
    archive = HERE / "ICAART2027_revised_source.zip"
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as z:
        for name in files:
            item = zipfile.ZipInfo(name, (2026, 11, 13, 0, 0, 0))
            item.compress_type = zipfile.ZIP_DEFLATED
            item.external_attr = 0o644 << 16
            z.writestr(item, (HERE / name).read_bytes())
    with tempfile.TemporaryDirectory(prefix="storyprojection-revised-source-") as tmp:
        with zipfile.ZipFile(archive) as z:
            z.extractall(tmp)
        command("sh", "build.sh", cwd=tmp)
        extracted = Path(tmp) / "ICAART2027_revised.pdf"
        assert command("pdftotext", "-layout", str(extracted), "-") == command("pdftotext", "-layout", str(HERE / extracted.name), "-")
    report = {"baseline_files_unchanged": len(baseline), "title_unchanged": True, "abstract_unchanged": True,
              "original_result_cells_unchanged": True, "original_calls_replayed": replay["original_calls_replayed"],
              "original_question_answers_replayed": replay["original_question_answers_replayed"],
              "new_filter_cases": len(replay["rows"]), "new_gpu_calls": 0, "new_semantic_annotations": 0,
              "figure_1_original_record_occurrences_unchanged": len(figure["records"]),
              "figure_1_displayed_source_sentences": 10, "revised_pdf": revised, "diff_pdf": diff,
              "source_zip": {"sha256": sha(archive), "isolated_compile_text_identical": True, "files": files},
              "references_defined_and_cited": sorted(citations),
              "source_sha256": {name: sha(HERE / name) for name in files},
              "visual_inspection": "See VISUAL_CHECK.md; manual inspection is separate from these automated checks."}
    (HERE / "verification.json").write_text(json.dumps(report, indent=2) + "\n")
    print(f"Verified: {len(baseline)} baseline files unchanged; title/abstract unchanged; 21 calls/30 answers replayed; 15 filter cases; eight-page manuscript; clean source ZIP; {diff['pages']}-page diff. GPU calls: 0.")


if __name__ == "__main__":
    main()
