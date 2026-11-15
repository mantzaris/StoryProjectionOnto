"""Compare both revisions with the submission using the same source-token diff.

Writes only comparison.json and COMPARISON.md beside this script. No third-party
packages, PDF parsing, model calls or GPU are needed. Run after final source edits.
"""
from __future__ import annotations

import difflib
import hashlib
import json
import re
from pathlib import Path


HERE = Path(__file__).resolve().parent
PAPER = HERE.parent
ORIGINAL = PAPER / "icaart2027"
VERSIONS = {
    "v1": PAPER / "icaart2027_revised",
    "v2": HERE,
}
MAIN_NAMES = {
    "original": "ICAART2027_submission.tex",
    "v1": "ICAART2027_revised.tex",
    "v2": "ICAART2027_revised_v2.tex",
}
COMPONENTS = {
    "main_tex": ("Main manuscript source", None),
    "figure_definitions": ("Figure captions and definitions", "figure_blocks.tex"),
    "synthetic_table": ("Synthetic results table", "tables/compact.tex"),
    "prose_table": ("Prose results table", "tables/prose.tex"),
    "bibliography": ("Bibliography source", "references.bib"),
}


def digest(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def source(path: Path) -> dict:
    raw = path.read_bytes()
    return {
        "path": str(path.relative_to(PAPER.parent)),
        "sha256": digest(raw),
        "text": raw.decode("utf-8"),
    }


def token_diff(old: str, new: str) -> dict:
    """Whitespace changes alone are ignored; punctuation and TeX remain literal."""
    old_tokens = re.findall(r"\S+", old)
    new_tokens = re.findall(r"\S+", new)
    changes = {"added": 0, "deleted": 0, "unchanged": 0}
    matcher = difflib.SequenceMatcher(None, old_tokens, new_tokens, autojunk=False)
    for tag, i, j, k, m in matcher.get_opcodes():
        if tag == "equal":
            changes["unchanged"] += j - i
        else:
            changes["deleted"] += j - i
            changes["added"] += m - k
    assert changes["unchanged"] + changes["deleted"] == len(old_tokens)
    assert changes["unchanged"] + changes["added"] == len(new_tokens)
    return {
        "original_tokens": len(old_tokens),
        "revision_tokens": len(new_tokens),
        **changes,
        "total_edits": changes["added"] + changes["deleted"],
    }


def field(text: str, name: str) -> str:
    # These manuscript fields have nested TeX braces, so use balanced delimiters.
    marker = "\\" + name + "{"
    start = text.index(marker) + len(marker)
    depth = 1
    for index in range(start, len(text)):
        if text[index] in "{}" and (index == 0 or text[index - 1] != "\\"):
            depth += 1 if text[index] == "{" else -1
            if depth == 0:
                return text[start:index]
    raise ValueError(f"Unclosed {name} field")


def reduction(first: int, second: int) -> float:
    if first == 0:
        return 0.0
    return 100 * (first - second) / first


def calculate() -> dict:
    originals = {}
    revisions = {name: {} for name in VERSIONS}
    inputs = {}
    for key, (_, relative) in COMPONENTS.items():
        original = source(ORIGINAL / (relative or MAIN_NAMES["original"]))
        originals[key] = original["text"]
        inputs[original["path"]] = original["sha256"]
        for name, folder in VERSIONS.items():
            current = source(folder / (relative or MAIN_NAMES[name]))
            inputs[current["path"]] = current["sha256"]
            revisions[name][key] = token_diff(original["text"], current["text"])
    unchanged_fields = {}
    for name, folder in VERSIONS.items():
        current_text = (folder / MAIN_NAMES[name]).read_text()
        unchanged_fields[name] = {
            key: field(originals["main_tex"], key) == field(current_text, key)
            for key in ("title", "abstract")
        }
        revisions[name]["aggregate"] = {
            metric: sum(component[metric] for component in revisions[name].values())
            for metric in ("original_tokens", "revision_tokens", "added", "deleted", "unchanged", "total_edits")
        }
    shared = {}
    for relative in (
        "tables/filter_check.json", "figures/orchard_network.pdf",
        "figures/fable.pdf", "figures/alice.pdf", "generated/numbers.tex",
    ):
        hashes = {name: digest((folder / relative).read_bytes()) for name, folder in VERSIONS.items()}
        shared[relative] = {"identical_between_revisions": hashes["v1"] == hashes["v2"], "sha256": hashes}
    return {
        "schema_version": 1,
        "method": {
            "tokenization": "Each maximal non-whitespace sequence (Python regex \\S+) is one source token. Case, punctuation, TeX commands, comments and table syntax are retained.",
            "alignment": "difflib.SequenceMatcher with autojunk=False, separately for each source file; identical method for both revisions.",
            "counts": "Every non-equal span contributes removed original tokens and added revision tokens. Total edits = added + deleted. Moves can count as deletion plus addition.",
            "aggregate": "Sum of main TeX, figure definitions/captions, two main tables and bibliography. Inputs are counted separately, without double-counting expanded content.",
            "exclusions": "PDFs, figure artwork, companion, build scripts, supporting documentation and extra diagnostic tables are not included in source-token edit totals.",
            "limitations": "These are source tokens, not linguistic word counts or changed PDF lines. The alignment is not a proof of minimum edit distance. Smaller counts do not establish readability, reviewer satisfaction, or a globally minimal revision.",
        },
        "versions": revisions,
        "reduction_v2_vs_v1_percent": {
            key: reduction(revisions["v1"][key]["total_edits"], revisions["v2"][key]["total_edits"])
            for key in ("main_tex", "aggregate")
        },
        "unchanged_fields": unchanged_fields,
        "shared_artifacts": shared,
        "input_sha256": inputs,
    }


def markdown(report: dict) -> str:
    versions = report["versions"]
    rows = []
    for key, label in [*((key, value[0]) for key, value in COMPONENTS.items()), ("aggregate", "**Total of these sources**")]:
        a, b = versions["v1"][key], versions["v2"][key]
        rows.append(f"| {label} | {a['deleted']} | {a['added']} | {a['total_edits']} | {b['deleted']} | {b['added']} | {b['total_edits']} |")
    reduction_main = report["reduction_v2_vs_v1_percent"]["main_tex"]
    reduction_all = report["reduction_v2_vs_v1_percent"]["aggregate"]
    title_abstract = all(all(fields.values()) for fields in report["unchanged_fields"].values())
    same_assets = all(asset["identical_between_revisions"] for asset in report["shared_artifacts"].values())
    fields_note = "Both revisions preserve the original title and abstract exactly." if title_abstract else "At least one title or abstract differs; inspect `comparison.json` before using either version."
    assets_note = ("Both versions use identical saved-output filter results, Figure 1 artwork, fable/Alice artwork and numerical macros. This comparison makes no new model calls."
                   if same_assets else "Some checked result or figure artifacts differ between revisions; see `shared_artifacts` in `comparison.json`.")
    return f"""# Comparison of the two revision attempts

Both columns compare their revision directly against the original submission, not against each other. {fields_note}

| Source | v1 removed | v1 added | v1 total edits | v2 removed | v2 added | v2 total edits |
|---|---:|---:|---:|---:|---:|---:|
{chr(10).join(rows)}

By this measure, v2 has **{reduction_main:.1f}% fewer main-manuscript source-token edits** and **{reduction_all:.1f}% fewer edits across all five source components** than v1. A negative percentage would mean more edits. These counts measure additions plus deletions, not manuscript length.

## Method and limits

Each whitespace-delimited source token is compared literally, using Python's `difflib.SequenceMatcher(..., autojunk=False)` independently for each file. Whitespace-only changes are ignored; TeX commands, punctuation, comments and table syntax remain included. A replaced token counts as one removal and one addition, and a moved passage can count on both sides. These are source-token counts, not linguistic word counts or changed PDF lines; the algorithm does not establish a mathematically minimal edit sequence.

The total includes the main manuscript, figure definitions/captions, both main tables and bibliography, with no double-counting of expanded inputs. It excludes artwork, the companion, extra diagnostic tables, scripts and supporting documents. The file hashes and full counts are in [comparison.json](comparison.json). Rerun after any source change:

```sh
python3 paper/icaart2027_revised_v2/compare_versions.py
```

{assets_note}

Fewer changes alone do not establish reviewer satisfaction or better writing. Compare the [v1 diff](../icaart2027_revised/ICAART2027_diff.pdf), [v2 diff](ICAART2027_diff_v2.pdf) and [v2 reviewer response](REVIEW_RESPONSE.md) to assess the retained explanations and scope. Both attempts retain the limitation that the saved-output comparison is not independent validation on new texts.
"""


def main() -> None:
    report = calculate()
    (HERE / "comparison.json").write_text(json.dumps(report, indent=2) + "\n")
    (HERE / "COMPARISON.md").write_text(markdown(report))
    for name in ("v1", "v2"):
        main_edits = report["versions"][name]["main_tex"]["total_edits"]
        total_edits = report["versions"][name]["aggregate"]["total_edits"]
        print(f"{name}: {main_edits} main-source edits; {total_edits} edits across five source components")
    print("Wrote comparison.json and COMPARISON.md.")


if __name__ == "__main__":
    main()
