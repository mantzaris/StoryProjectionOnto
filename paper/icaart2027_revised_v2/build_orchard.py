"""Expand Figure 1's source excerpt; retain the exact original graph records and layout."""
from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "paper/icaart2027"))
import build_assets as b
import network_figures as n


def main():
    b.HERE = HERE  # Only the new revision directory receives generated artwork.
    data = {key: json.loads((ROOT / path).read_text()) for key, path in [
        ("compact", "reports/tables/compact_story_v2_results.json"),
        ("prose", "reports/tables/real_text_proof_of_concept.json"),
    ]}
    panels = n.load_panels(data)
    p = b.make_plate("orchard_network")
    p.text(5, 2, "Orchard · input excerpts and extracted relationships", size=10, bold=True)
    full_evidence = data["compact"]["stories"]["orchard"]["evidence"]
    # Expand the earlier three-sentence excerpt without crowding the graph.
    ids = ["S1", "S2", "S3", "S4", "S13", "S9", "S10", "S11", "S12", "S14"]
    evidence = [(eid, full_evidence[eid]) for eid in ids]
    width = (b.W - 24) / 2
    bottoms = []
    for column in range(2):
        y = 20
        for eid, text in evidence[column * 5:(column + 1) * 5]:
            y = p.para(5 + column * (width + 14), y, eid + " " + text, width, size=8.5) + 2
        bottoms.append(y)
    y = max(bottoms) + 3
    p.text(5, y + 4, "Model extraction before the question · 10 nodes / 14 records", size=9, bold=True)
    n.draw_network(b, p, panels["orchard_pre"], (0, y + 28), n.ORCHARD, n.ORCHARD_ROUTES,
                   yscale=.70, categories=n.CATEGORIES, broad=True)
    y += 270
    p.text(5, y, "Fixed selection: ownership/carrying · 6 nodes / 5 records", size=9, bold=True)
    n.draw_network(b, p, panels["orchard_A"], (0, y + 8), n.ORCHARD,
                   {**n.ORCHARD_ROUTES, "S4": (0, 252, 150), "S13": (0, 43, 190)},
                   yscale=.35, categories=n.CATEGORIES)
    y += 124
    y = p.para(5, y, "Blue: person. Cream: object. Grey: place (display categories). loc = located in. "
               "Dashed top edges show beliefs; + marks supported selected facts. Unprinted bounds and attribution are null. Input excerpted; all 14 sentences supplied.",
               b.W - 10, size=8.5) + 4
    meta = b.save_plate(p, y, "Ten original input sentences displayed in two columns (previously three). All graph records, directions, qualifications, status labels and relative graph positions unchanged.")
    original = json.loads((ROOT / "paper/icaart2027/manifest.json").read_text())["figures"]["orchard_network"]
    for key in ("records",):
        # Absolute drawing coordinates move down; all scientific content is unchanged.
        for old, new in zip(original[key], meta[key], strict=True):
            for field in ("fact", "status", "source_call", "source_indices", "display_label"):
                assert old[field] == new[field], field
    meta["displayed_evidence"] = dict(evidence)
    (HERE / "figures/orchard_manifest.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False) + "\n")
    print("Figure 1: 10 of 14 input sentences; 19 unchanged displayed records; no inference.")


if __name__ == "__main__":
    main()
