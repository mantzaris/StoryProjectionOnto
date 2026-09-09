# Exploratory proof-of-concept manuscript

[Review manuscript](PROOF_OF_CONCEPT.pdf) · [Editable Markdown](PROOF_OF_CONCEPT.md) · [Compact figures](figures/index.html) · [Supplement](SUPPLEMENTARY_MATERIAL.pdf)

This manuscript describes retained **exploratory extraction** results. It does not report completed registered C1/C2 acceptance, an independently reviewed benchmark, a full-novel study, or participant evaluation. No GPU is used by this publishing toolchain. Historical outputs, assessments and scores are unchanged.

## Regenerate

From the repository root, in the existing publishing Python environment:

```bash
/tmp/spo-refresh-venv/bin/python -m scripts.build_poc_manuscript
```

Portable equivalent: `python -m scripts.build_poc_manuscript`, with [requirements.txt](requirements.txt) installed and Poppler's `pdftoppm`, `pdftotext`, `pdfinfo` and `pdffonts` available. No TeX distribution is required. The script reuses the existing vector-figure renderer; ReportLab embeds the figure PDFs without rasterizing or scaling their text. pdfrw combines existing supplementary pages. PDF creation timestamps are fixed for byte reproducibility, not reported as experiment times.

Edit [manuscript_source.md](manuscript_source.md) for prose. `{{name}}` placeholders resolve numerical claims from retained artifacts and are listed with provenance in [numerical_registry.json](tables/numerical_registry.json). `!TABLE:name` and `!FIGURE:name` insert generated content. `[@key]` links to verified primary references. Edit [reference_sources.json](reference_sources.json) for bibliography metadata and citation-support notes; `references.bib` is generated. Do not hand-edit generated scores or regenerate experimental results for a manuscript change.

## Assets and scope

- Three main figures are 178 mm wide, under 180 mm high, with text at least 8.5 points. PDF, searchable embedded-font SVG and 400-dpi PNG are under `figures/`. Exact questions and display omissions are in `FIGURE_CAPTIONS.md` and the manifest.
- `SUPPLEMENTARY_MATERIAL.pdf` combines the new explanatory front matter with all eight pages of the unchanged `reports/PAPER_FIGURES.pdf`, then the unchanged full-output appendix. `supplement_front.pdf` and its Markdown are editable-toolchain intermediates, intentionally retained for reproducibility.
- `tables/compact_comparison.*` contains only fresh compact-story v2 scores; `compact_components.*` and `compact_assertion_assessments.json` preserve citation, format, relationship, qualification, relevance and unresolved details. `published_comparison.*` retains unavailable parser outcomes separately from scored records.
- The original ladders, their revised scoring policy, historical comma-only recoveries, and new calls are separate tables. Costs sum unique calls and separately recorded service allocations, not duplicated selected views.
- [manuscript_manifest.json](manuscript_manifest.json) binds source artifacts, selected output records, assessment statuses, captions, transformations and generated assets to hashes. Main-paper display subsets do not change any denominator. The existing interactive source-linked graphs remain at `reports/figures/paper/index.html`.

## Focused checks

```bash
/tmp/spo-refresh-venv/bin/python -m pytest -q tests/test_poc_manuscript.py
```

The checks cover retained-source hashes, strict-versus-recovered parsing, exact table values, disjoint cost accounting, literal figure records/statuses, native figure size, vector embedding, source text, bibliography coverage, numerical substitutions, all-page text bounds, and unchanged appended supplementary page streams. Rendering inspection is recorded separately in `verification.json`; CPU checks are not model evaluation or independent semantic review.

## Concrete author-review items before submission

- Confirm publication name and affiliation. The draft uses only the repository's established `a.v.mantzaris` copyright identity; it invents neither an affiliation nor coauthors.
- Supply funding and conflict-of-interest declarations and approve the AI-assistance statement.
- Review the Codex-authored reference scope and semantic assessments, particularly sentence-valued thought credit and the ambiguous Holmes vocative. These are disclosed limitations, not independent human validation. Changes would require separately versioned annotations and scores, not silent figure edits.
- Confirm publication venue and local rights requirements for the Gutenberg excerpts. Exact source metadata and supplied notices remain in the existing source bundle. Familiar literary material is not evidence of unseen-data generalization.

No manuscript build starts or stops remote services. The preserved last experiment accounting reports vLLM stopped and journals closed; this CPU task makes no fresh claim about pod state.
