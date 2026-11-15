# Revision checks — 13 November 2026

The eight-page revised manuscript and twelve-page diff were rendered and visually inspected by Codex. This is a layout and revision check, not independent human validation of the reference answers or semantic assessments.

## PDF inspection

- All eight manuscript pages were inspected using 110 dpi renders. The title and abstract fit as before. The expanded Figure 1 input appears in two columns; labels, arrows and qualifiers remain readable. Tables, short captions, references and the Appendix fit within the page margins.
- All twelve diff pages were inspected using 105 dpi renders. Additions appear in blue with underlining; deletions appear in red with strikethrough. The abstract has no change marks. The added result column and new references are marked. The final two pages show the original and revised Figure 1 at the manuscript's full width.
- Inspection caught an empty heading left by `latexdiff` when it removed the separate AI disclosure section. The diff builder now exposes the heading text before comparison. The corrected pages 9–10 were rendered and inspected again; the removed heading is visible in red. The revised manuscript discloses AI assistance in Section 4.1.
- Automated PDF checks confirm embedded fonts, no Type 3 fonts, content within page bounds, and at most two rendered lines in each main figure caption. Both LaTeX logs are free of overfull boxes, oversized floats, undefined references and pending label changes. Artwork is reduced only in the diff's marked-up pages to accommodate deleted captions.

## Evidence and build checks

`verification.json` records the final PDF hashes and detailed results. Its checks confirm:

- All 92 baseline files retain their original hashes. The title and abstract are unchanged, character for character; original result cells are unchanged.
- All 21 original calls and 30 question-specific answers reproduce their saved strict scores from retained outputs.
- The follow-up filter analysis covers all 15 contextual cases, including the unavailable Holmes action answer. It uses the original selectors and adds no semantic labels or model calls.
- All 19 displayed record occurrences in Figure 1 retain their facts, directions, qualifications and status. Its ten displayed source sentences match the original evidence.
- The source ZIP compiles in an isolated temporary directory and produces the same extracted manuscript text. All cited bibliography entries are defined.

The focused regression suite finished with **35 passed and 4 skipped**:

```sh
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src:. python -m pytest -q -p no:cacheprovider \
  tests/unit/test_compact_story_v2.py \
  tests/unit/test_compact_story_v2_results.py \
  tests/unit/test_real_text_poc.py \
  tests/unit/test_real_text_poc_results.py \
  tests/unit/test_icaart_networks.py \
  tests/unit/test_paper_figures.py
```

Two skips require a locally cached pinned tokenizer; two require excluded private run backups. The public retained-output replay passes without those resources. These checks used Python 3.12 with the publishing dependencies and pytest; no GPU was used.
