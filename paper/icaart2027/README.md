# ICAART 2027 local Position Paper package

[Anonymous main PDF](ICAART2027_submission.pdf) · [Main LaTeX](main.tex) · [Local companion PDF](ICAART2027_companion.pdf) · [Author actions](author_actions.md) · [Requirements](submission_requirements.md)

**No submission, email, registration or push is performed by these commands.** The main PDF is the intended anonymous submission artifact, subject to author review and eligibility clarification. The companion and its data are locally prepared anonymous supporting material; separate upload permission is unconfirmed. The source ZIP is for compilation/reproducibility, not an assertion that initial-review source upload is required.

## Portable build

From the repository root:

```bash
python -m pip install -r paper/requirements.txt
python paper/icaart2027/build_assets.py
sh paper/icaart2027/build.sh
python paper/icaart2027/verify.py
```

This is the complete retained-result regeneration sequence. Python 3.12 and the pinned existing publishing dependencies are used. A local example substitutes `/tmp/spo-refresh-venv/bin/python` for `python`; that path is not required. No model packages, downloads, GPU access or scoring runs are needed. The first published `data/requests.json` exports only hash-matched project request payloads; subsequent builds use that safe retained file, not restricted backups.

LaTeX dependencies: pdfLaTeX, BibTeX, `pslatex`, `fontenc`, `inputenc`, `epsfig`, `subcaption`, `calc`, AMS packages, `multicol`, `algorithm2e`, `footmisc`, `booktabs`, `url`, `hyperref`, `geometry`, `mathptmx`, `courier`, `fvextra`, and `ragged2e`. They were already installed in TeX Live 2019/Debian; no TeX installation was performed. Typical Debian/Ubuntu TeX Live package groups are `texlive-latex-base`, `texlive-latex-recommended`, `texlive-latex-extra`, `texlive-fonts-recommended` and `texlive-science`. Use `kpsewhich package.sty` to identify a missing dependency instead of installing a large distribution blindly. Poppler `pdftotext`, `pdfinfo`, `pdffonts`, `pdftoppm` is required for verification/rendering.

For an already-generated source directory, compile without Python or repository access:

```bash
sh build.sh --submission-only
```

The clean ZIP contains only the main source, generated inputs, three figure PDFs, bibliography, four official formatting files and build command. It excludes companion, author notes, paths, source history, logs, old examples, the downloaded archive, and prior PDFs. Unzip into a new directory and use that command; it does not contact the conference.

## Editing and provenance

The conference [references.bib](references.bib) is now directly editable and is no longer overwritten from the earlier general manuscript. [REFERENCE_AUDIT.md](REFERENCE_AUDIT.md) records primary verification and the contribution-to-evidence mapping.

Figure regeneration requires licensed local **Times New Roman** regular and bold faces. Both were already installed. The renderer requires these exact faces without fallback, uses them for wrapping and drawing, embeds document subsets in PDFs, and records font hashes. Editable SVGs retain searchable text with local font references without redistributing font software. PDFs and PNGs are portable viewing artifacts. Recompiling the source ZIP uses the embedded figure PDFs and does not require these local figure-generation fonts.

Edit `main.tex`, `abstract.tex`, `figure_blocks.tex` and `companion.tex`. Numerical commands and tables are generated from retained canonical tables by `build_assets.py`; do not transcribe or change scores. The figure renderer reuses the established exact-record display helpers, but uses the actual template width and new layouts. PDF/SVG/400-dpi PNG figures are in `figures/`. Literal model values, assessment statuses and omitted display indices are in `manifest.json`. The three compact historical figures, general manuscript, general supplement and eight original plates are untouched.

## Network figures and concise companion

The main paper now has three figures: Orchard's full model extraction and actual fixed-selection slice, the fable action comparison, and Alice's action/thought comparison. The Harbor timeline moves to the companion. The two main numerical tables and all scores are unchanged.

`network_figures.py` is a small adapter to the existing Matplotlib renderer, not a new graph framework. Each panel has one visual node per literal endpoint string and one arrow per retained record. Author-chosen coordinates and node categories are display annotations. Directed, parallel edges retain citations, intervals and holder/attitude fields. Holder badges do not add nodes or relationships. The main Orchard broad extraction is not labelled as gold or retrospectively given a query-specific assessment. Fixed-selection membership is checked against actual call 2, while the contextual ownership panel uses independent call 7. The manifest records source indices, identities, coordinates, statuses and complete facts. All network panels are complete, with zero display omissions.

The current companion has seven pages rather than 36, with four supplementary figures: Orchard A/B ownership networks, Orchard belief/reality selection, Harbor intervals, and complete fable action networks. It does not repeat the main figures unchanged. Edit its narrative in `companion_source.tex`. Selected literal examples and numerical summaries are generated into `generated/companion_facts.tex`. Regeneration no longer emits the old raw-record dump or rewrites the complete evidence files.

Complete evidence remains in [data/retained_outputs.json](data/retained_outputs.json), exact payloads in [data/requests.json](data/requests.json), and rules in [data/evaluation_rules.json](data/evaluation_rules.json). Full semantic assessments remain in the unchanged [published-prose result artifact](../../reports/tables/real_text_proof_of_concept.json), alongside the [compact-story results](../../reports/tables/compact_story_v2_results.json). The [reproducibility index](generated/reproducibility_index.json) gives field paths and hashes. The deletion of the generated 1,918-line PDF dump does not delete these records. Strict matching, partial meaning and failures are explained once in the reader-oriented companion.

The 36-page companion and its old generated source are preserved at commit `48c0def1bfe8880029d88e9539462302d77833c7`. To inspect the archival PDF without creating a second current supplement, run this read-only command from the repository root:

```bash
git show 48c0def1bfe8880029d88e9539462302d77833c7:paper/icaart2027/ICAART2027_companion.pdf | pdftotext -layout - -
```

The normal build commands above regenerate every current PDF, SVG and PNG, both documents and the clean main-paper source ZIP. Focused checks:

```bash
python -m pytest -q tests/unit/test_icaart_networks.py tests/unit/test_paper_figures.py
python paper/icaart2027/verify.py
```

`vendor/` preserves the official archive and example for local provenance only. Never edit the supplied class/style/BibTeX files to alter layout. `verification.json` records template hashes, numerical consistency, anonymity, embedded fonts, page/abstract/character counts, text bounds and source-package reproduction. `rendered/` is ignored and contains inspection images. Build timestamps are fixed for reproducibility, not represented as experiment times. Companion raw records and requests are under `data/`; Gutenberg notices remain attached there.

## Before public distribution

Read `author_actions.md` and the **unsent** secretariat inquiry. The general manuscript and initial conference package already occur in the public upstream history, but no assumption about ICAART eligibility follows. The conference prohibits public posting of submitted manuscripts while under review. A normal push of this branch to its public upstream would publish this revision; the local commit is not authorization to do so.
