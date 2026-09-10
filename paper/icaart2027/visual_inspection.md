# Publishing verification — 9 September 2026

The final main PDF has **8 A4 pages**, a **164-word abstract**, and **25,224 non-whitespace characters** by UTF-8 PDF text extraction. Searchable figure text is included. Counting all figure text a second time gives a deliberately conservative bound of **27,451**, leaving 12,549 below 40,000 even under that overcount. The page limit and character range are separate checks; no paid extra pages or modified template margins are used. These are local counts, not a claim about PRIMORIS's exact implementation.

The main PDF was rendered with Poppler at 1,500 pixels page height. All pages were inspected for overall layout and reading order, with enlarged checks of the three native-width graphs and final bibliography/appendix page. The 36-page companion was likewise rendered and every page inspected; corrected source-order pages and the final failure-diagnosis pages were reinspected after regeneration. Contact sheets and individual page renders are reproducible local inspection artifacts under ignored `rendered/`, not reviewer attachments.

Main-page inspection:

- 1: anonymous title block, keywords, 164-word abstract and two-column introduction; no identifying author block or page number.
- 2: fable artwork at native 158.0134 mm, correct directed edges and partial-status strokes; caption retains the executed question and omitted-record accounting.
- 3: compact record example, null/interval/attribution interpretation, model settings and two distinct processing paths.
- 4: Harbor source intervals, included/excluded endpoint markers, distinct shaded query window, and both irrelevant carrying assertions; no interval clipping or hidden predictions.
- 5: evaluation definitions and mixed results, preserving parse failures and Codex annotation provenance.
- 6: Alice's actual thought object, wrong reading participant and unresolved daisy-chain record; exact two questions and display omissions in caption.
- 7: both numerical tables, unavailable Holmes views, costs and discussion; no table overflow or missing row.
- 8: conclusion, non-identifying substantive AI disclosure, author-date bibliography and appendix after references. Tool/preprint identifiers and model hash are readable and no author affiliation or repository link appears.

Corrections made during local preparation: used ordinary `[!t]` float placement so full-width figures no longer collected after the text; reflowed artwork at the conference width without reducing its 8.5-point minimum text; replaced repeated overflowing companion JSON dumps with literal-field records and explicit source-record mappings; used a breakable verbatim display for exact system messages and unparseable responses; embedded Courier instead of bitmap typewriter fonts; restored numeric evidence order after identifying lexical JSON-key ordering. Reconstructed user-message strings are checked against the exact retained payloads, so display order is not assumed from canonical JSON serialization. No output, annotation, matcher or score was repaired.

Automated checks (`python paper/icaart2027/verify.py`) pass for original formatting-file hashes, unchanged canonical/report artifacts, figure-record and assessment correspondence, actual request hashes, all-page text bounds, embedded fonts without Type 3 glyphs, anonymity scans and blank Author metadata, citation/entry correspondence, independent character/page/abstract limits, and isolated ZIP compilation with identical extracted PDF text. Final TeX logs contain no overfull boxes or undefined references. The companion's page-number footer is checked separately from body margins.

Additional focused checks: `python -m pytest -q tests/unit/test_paper_figures.py` — **8 passed**; publishing-source `ruff` checks for syntax/undefined-name classes — passed. No experimental suites or model calls were run. The general manuscript, canonical results, previous compact figures and eight original plates remain unchanged; only their README gains a link to this separate package.

Remaining uncertainty is editorial/policy, not an unfinished build: semantic author-review decisions, full author declarations, prior public-manuscript eligibility, remote presentation, separate supplementary-upload permission and anonymous AI-disclosure placement. These are listed in `author_actions.md`; the secretariat inquiry is unsent. Visual inspection is Codex work, not independent human review or conference certification.
