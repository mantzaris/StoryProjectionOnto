# Exploratory proof-of-concept manuscript

[Review manuscript](PROOF_OF_CONCEPT.pdf) · [Editable Markdown](PROOF_OF_CONCEPT.md) · [Compact figures](figures/index.html) · [Supplement](SUPPLEMENTARY_MATERIAL.pdf)

Conference-specific local adaptation: [ICAART 2027 Position Paper package](icaart2027/README.md). It uses the official LaTeX template and preserves this general manuscript. Read its author actions and public-posting warning before any public push or submission.

This manuscript describes retained **exploratory extraction** results. It does not report completed registered C1/C2 acceptance, an independently reviewed benchmark, a full-novel study, or participant evaluation. No GPU is used by this publishing toolchain. Historical outputs, assessments and scores are unchanged.

## Regenerate

From the repository root, using Python 3.12 and the declared publishing dependencies:

```bash
python -m pip install -r paper/requirements.txt
python -m scripts.build_poc_manuscript
```

The second command regenerates the complete manuscript and figure set. Poppler's `pdftoppm`, `pdftotext`, `pdfinfo` and `pdffonts` must be on PATH. No TeX distribution is required. An existing local environment can also run `/tmp/spo-refresh-venv/bin/python -m scripts.build_poc_manuscript`; that temporary path is not a portability requirement. The script reuses the vector-figure renderer; ReportLab embeds the figure PDFs without rasterizing or scaling their text. pdfrw combines supplementary pages. PDF creation timestamps are fixed for byte reproducibility, not experiment times.

Edit [manuscript_source.md](manuscript_source.md) for prose. `{{name}}` placeholders resolve numerical claims from retained artifacts and are listed with provenance in [numerical_registry.json](tables/numerical_registry.json). `!TABLE:name` and `!FIGURE:name` insert generated content. `[@key]` links to verified primary references. Edit [reference_sources.json](reference_sources.json) for bibliography metadata and citation-support notes; `references.bib` is generated. Do not hand-edit generated scores or regenerate experimental results for a manuscript change.

Short main captions and complete figure notes are maintained in `main_captions()` and `captions()` in `scripts/build_poc_manuscript.py`. The complete notes, including exact questions and display-subset accounting, appear in `FIGURE_CAPTIONS.md` and Supplement S6. [AUTHOR_REVIEW.md](AUTHOR_REVIEW.md) is a human-editable review sheet, hashed as a source input and never overwritten by the generator. Record decisions there; any later annotation or scoring change requires a separate version.

## Assets and scope

- Three main figures are 178 mm wide, under 180 mm high, with text at least 8.5 points. PDF, searchable embedded-font SVG and 400-dpi PNG are under `figures/`. Exact questions and display omissions are in `FIGURE_CAPTIONS.md` and the manifest.
- `SUPPLEMENTARY_MATERIAL.pdf` combines the new explanatory front matter with all eight pages of the unchanged `reports/PAPER_FIGURES.pdf`, then the unchanged full-output appendix. `supplement_front.pdf` and its Markdown are editable-toolchain intermediates, intentionally retained for reproducibility.
- `tables/compact_comparison.*` contains only fresh compact-story v2 scores; `compact_components.*` and `compact_assertion_assessments.json` preserve citation, format, relationship, qualification, relevance and unresolved details. `published_comparison.*` retains unavailable parser outcomes separately from scored records.
- The original ladders, their revised scoring policy, historical comma-only recoveries, and fresh calls are separate tables. Main Table 3 displays only the two comparison batches; Supplement S3 and `tables/allocation.*` retain all four batches and distinguish whole-project allocation. Costs sum unique calls, not duplicated selected views.
- [manuscript_manifest.json](manuscript_manifest.json) binds source artifacts, selected output records, assessment statuses, captions, transformations and generated assets to hashes. Main-paper display subsets do not change any denominator. The existing interactive source-linked graphs remain at `reports/figures/paper/index.html`.

## Focused checks

```bash
python -m pytest -q tests/test_poc_manuscript.py
```

Use a development environment with pytest installed for the focused checks (pytest is not required to build the PDFs). Checks cover retained-source hashes, parser outcomes, exact table values, disjoint cost accounting, literal figure records/statuses, native size and vector embedding, bibliography coverage, numerical substitutions, all-page text bounds, unchanged appended pages, and verbatim review-sheet records/evidence. Rendering inspection is recorded separately in `verification.json`; CPU checks are not model evaluation or independent semantic review.

## Author review and editorial provenance

[AUTHOR_REVIEW.md](AUTHOR_REVIEW.md) is the single prioritized list of judgment requests and unresolved publication details. It links exact evidence, generated records, and frozen assessments. The editorial revision builds on `3329a6e`, retaining the three compact figure artworks and eight original plates. It shortens the narrative and captions without changing outcomes or requiring a minimum word count. Historical output and table hashes remain verified by the manifest.

No manuscript build starts or stops remote services. The preserved last experiment accounting reports vLLM stopped and journals closed; this CPU task makes no fresh claim about pod state.
