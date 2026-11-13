# ICAART 2027 local Position Paper package

[Anonymous main PDF](ICAART2027_submission.pdf) · [Main LaTeX](ICAART2027_submission.tex) · [Local companion PDF](ICAART2027_companion.pdf) · [Companion LaTeX](ICAART2027_companion.tex) · [Author actions](author_actions.md) · [Requirements](submission_requirements.md)

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

LaTeX dependencies: pdfLaTeX, BibTeX, `pslatex`, `fontenc`, `inputenc`, `epsfig`, `subcaption`, `calc`, AMS packages, `multicol`, `algorithm2e`, `footmisc`, `booktabs`, `url`, `hyperref`, `hypcap`, `geometry`, `mathptmx`, `courier`, `fvextra`, and `ragged2e`. They were already installed in TeX Live 2019/Debian; no TeX installation was performed. Typical Debian/Ubuntu TeX Live package groups are `texlive-latex-base`, `texlive-latex-recommended`, `texlive-latex-extra`, `texlive-fonts-recommended` and `texlive-science`. Use `kpsewhich package.sty` to identify a missing dependency instead of installing a large distribution blindly. Poppler `pdftotext`, `pdfinfo`, `pdffonts`, `pdftoppm` is required for verification/rendering.

For an already-generated source directory, compile without Python or repository access:

```bash
sh build.sh --submission-only
```

The clean ZIP contains the root `ICAART2027_submission.tex`, generated inputs, three figure PDFs, bibliography, four official formatting files and build command. It excludes companion, author notes, paths, source history, logs, old examples, the downloaded archive, and prior PDFs. Unzip into a new directory and use that command; it does not contact the conference.

Both root sources compile under their own basenames: `ICAART2027_submission.tex` produces `ICAART2027_submission.pdf`, and `ICAART2027_companion.tex` produces `ICAART2027_companion.pdf`. The build keeps auxiliary files in `build/` and copies the completed PDFs to this directory without renaming them or overriding job names. To rebuild both PDFs from the existing assets, run `sh paper/icaart2027/build.sh` from the repository root. No Python or asset regeneration is needed for a manuscript-only edit.

### Single-source companion

`ICAART2027_companion.tex` is the complete, authoritative editable source. Its currently displayed prose, commands, numerical macros, tables and captions are inline. It has no external content or bibliography inputs. Compilation requires only standard installed LaTeX packages, the unchanged supplied `article.cls`, and these four images in `figures/`: `orchard_comparison.pdf`, `orchard_beliefs.pdf`, `harbor.pdf` and `fable_networks.pdf`.

For sharing or uploading source, use [ICAART2027_companion_source.zip](ICAART2027_companion_source.zip), not the `.tex` file alone. The archive contains the current edited source, its class and all four actual figure PDFs with their `figures/` paths intact. No placeholders are used. Extract the whole archive before compiling.

From the repository root, this command resolves image paths by changing to the document directory:

```bash
latexmk -pdf -cd paper/icaart2027/ICAART2027_companion.tex
```

In an extracted companion source package, use `latexmk -pdf -cd ICAART2027_companion.tex`. A complete root document must end with `\end{document}`. If compilation still fails, inspect the first `!` error in `ICAART2027_companion.log`; a missing image and an end-of-file emergency stop are different failures.

Build and verify only the companion, preserving the submission and any ongoing manuscript edits:

```bash
sh paper/icaart2027/build.sh --companion-only
python paper/icaart2027/verify.py --companion-only
```

For a minimal copied directory containing the single root source, class and four figure PDFs, run the following twice from that directory:

```bash
pdflatex -no-shell-escape -interaction=nonstopmode -halt-on-error ICAART2027_companion.tex
```

The companion check regenerates the companion source ZIP and tests that exact extracted package in isolation. It checks inline values against retained records without rewriting them and compares text and every rendered page with the built PDF. A diagnostic listing is checked if displayed; its omission is recorded rather than restoring text removed by the author. It writes `companion_verification.json`. For a source-only reorganization, add `--compare-with /path/to/prior-companion.pdf` to compare against the preceding PDF as well. Companion-only commands leave the main submission, its source ZIP and its last full-package verification record unchanged. The existing `ICAART2027_source.zip` contains only the main submission, not the companion.

## Editing and provenance

The conference [references.bib](references.bib) is now directly editable and is no longer overwritten from the earlier general manuscript. [REFERENCE_AUDIT.md](REFERENCE_AUDIT.md) records primary verification and the contribution-to-evidence mapping.

Figure regeneration requires licensed local **Times New Roman** regular and bold faces. Both were already installed. The renderer requires these exact faces without fallback, uses them for wrapping and drawing, embeds document subsets in PDFs, and records font hashes. Editable SVGs retain searchable text with local font references without redistributing font software. PDFs and PNGs are portable viewing artifacts. Recompiling the source ZIP uses the embedded figure PDFs and does not require these local figure-generation fonts.

Edit `ICAART2027_submission.tex` (including its inline abstract), `figure_blocks.tex` and `ICAART2027_companion.tex`. These files are not generated or overwritten from an earlier manuscript. Asset generation checks that they remain unchanged. The main paper's numerical commands and tables are generated from retained canonical tables by `build_assets.py`; do not transcribe or change scores. Companion values are inline and checked, never restored from generated fragments. The figure renderer reuses the established exact-record display helpers, but uses the actual template width and new layouts. PDF/SVG/400-dpi PNG figures are in `figures/`. Literal model values, assessment statuses and omitted display indices are in `manifest.json`. The three compact historical figures, general manuscript, general supplement and eight original plates are untouched.

## Network figures and concise companion

The main paper now has three figures: Orchard's full model extraction and actual fixed-selection slice, the fable action comparison, and Alice's action/thought comparison. The Harbor timeline moves to the companion. The two main numerical tables and all scores are unchanged.

`network_figures.py` is a small adapter to the existing Matplotlib renderer, not a new graph framework. Each panel has one visual node per literal endpoint string and one arrow per retained record. Author-chosen coordinates and node categories are display annotations. Directed, parallel edges retain citations, intervals and holder/attitude fields. Holder badges do not add nodes or relationships. The main Orchard broad extraction is not labelled as gold or retrospectively given a query-specific assessment. Fixed-selection membership is checked against actual call 2, while the contextual ownership panel uses independent call 7. The manifest records source indices, identities, coordinates, statuses and complete facts. All network panels are complete, with zero display omissions.

The current edited companion has six pages, four captioned figures (S1–S4) and two captioned tables (S1–S2). Clickable references target the beginning of each float; exact questions and scores stay with their figures. The figures show Orchard A/B ownership networks, Orchard belief/reality selection, Harbor intervals, and complete fable action networks. It does not repeat the main figures unchanged. All content is editable directly in `ICAART2027_companion.tex`. The former companion prose, macro and listing fragments have been inlined and removed after checking that no other document uses them. The shared `generated/numbers.tex` remains for the main paper only. Regeneration neither recreates companion fragments nor rewrites the complete evidence files.

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
