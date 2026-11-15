# Revision v2 of ICAART 2027 paper 82

- [Revised manuscript](ICAART2027_revised_v2.pdf) — eight pages.
- [Diff PDF](ICAART2027_diff_v2.pdf) — blue underlined additions and red struck deletions; the final two pages show Figure 1 before and after.
- [Editable manuscript](ICAART2027_revised_v2.tex).
- [Portable LaTeX package](ICAART2027_revised_v2_source.zip).
- [Response to the reviews](REVIEW_RESPONSE.md) — each request and the corresponding change, including limits of the additional analysis.
- [Unchanged companion](ICAART2027_companion.pdf).
- [Verification notes](VISUAL_CHECK.md) and [machine-readable checks](verification.json).

The title and abstract are unchanged, character for character. Original A/B results, reference answers, assessments and raw outputs are unchanged. All changes are confined to this new directory. No inference or GPU work was performed. The new B+filter column is a follow-up CPU analysis of the retained outputs, not a new generation experiment.

The current v2 includes Alexander V. Mantzaris as the sole author, with the supplied school, UCF affiliation, address and email in the conference's `authorname`, `affiliation` and `email` macros. The address was verified against the [official UCF school page](https://sciences.ucf.edu/sdmss/) and corrected to 4000 Central Florida Blvd, Orlando, FL 32816, USA. PDF author metadata is also set. Adding the author block required displaying the three figures at 95% of their previous width to retain eight pages; manuscript text, captions, artwork files, body fonts and margins are unchanged by this author update. The diff and source ZIP include this update. The template format follows the [official conference template](https://icaart.scitevents.org/Templates.aspx?y=2027).

This is a separate editing attempt starting from the original submitted TeX. It retains more original wording and keeps the AI-use section in place. The verified filter analysis, expanded Figure 1, bibliography additions and build utilities are reused from v1. Neither the submitted package nor v1 was edited. See [comparison with v1](COMPARISON.md) for reproducible edit counts; fewer changes alone do not establish scientific adequacy.

For the requested priority of minimal edits, v2 is the recommended version: it retains the necessary clarifications with fewer rewritten passages. V1 gives more expansive explanations. Both versions use the same evidence and retain the same limits on broader validation.

## Build and reproduce

Compile the revised manuscript from the repository root:

```sh
sh paper/icaart2027_revised_v2/build.sh
```

This needs pdfLaTeX, BibTeX and the same LaTeX packages as the original package. It uses the included figures and does not need Python, a GPU or model files. Intermediate files go in `build/`.

To replay the original strict scores and reproduce the new filter comparison, use Python 3.12 with the two small dependencies in `requirements-replay.txt`:

```sh
python -m pip install -r paper/icaart2027_revised_v2/requirements-replay.txt
python paper/icaart2027_revised_v2/replay_results.py --verify
```

The script checks all 21 calls and 30 question-specific answers against their saved evaluations. It applies the original selectors to all 15 contextual answer cases; the malformed Holmes action output stays unavailable. It verifies exact selected-record membership, preserves duplicates and missing fields, and writes no new semantic judgments. Without `--verify`, it regenerates `tables/filter_check.json`, `.csv`, `.tex`, and the added column in `tables/compact.tex`. The JSON includes full selection traces, unrounded scores and input/code hashes. It requires the repository's public files, not the excluded private run backups or tokenizer.

Figure 1 can be regenerated with the publishing dependencies in `paper/requirements.txt` and the local Times New Roman fonts required by the existing renderer:

```sh
python paper/icaart2027_revised_v2/build_orchard.py
```

This reuses the original graph renderer. It shows ten exact source sentences instead of three, retaining all 19 displayed records, their directions, labels, qualifications and status. Fable and Alice artwork is copied unchanged. The short captions replace the earlier long captions; original display details remain in `FIGURE_DETAILS.md` and the retained figure manifests.

After building the revised PDF, generate the diff (requires `latexdiff`, the publishing dependencies and the same local figure fonts):

```sh
python paper/icaart2027_revised_v2/build_diff.py
python paper/icaart2027_revised_v2/verify_revision.py
python paper/icaart2027_revised_v2/compare_versions.py
```

The diff compares against the current submitted sources in `paper/icaart2027`, including expanded figure captions, table inputs and compiled bibliography entries. It reduces artwork only in the marked-up text pages to make room for deleted captions. Two additional pages preserve the full-size original and revised Figure 1. `verify_revision.py` checks the unchanged baseline, title/abstract, original table values, replay outputs, fonts, page bounds, caption lengths, references and isolated source-ZIP compilation. It writes `verification.json` and a portable `ICAART2027_revised_v2_source.zip`.

## Evidence locations

| Material | Repository path |
|---|---|
| Exact prompts, complete evidence and decoding settings | `paper/icaart2027/data/requests.json` |
| Raw/parsed model responses and reference answers | `paper/icaart2027/data/retained_outputs.json` |
| Reference alternatives, matching rules and semantic rubric | `paper/icaart2027/data/evaluation_rules.json` |
| Full synthetic results | `reports/tables/compact_story_v2_results.json` |
| Full prose results, per-record judgments and their response hashes | `reports/tables/real_text_proof_of_concept.json` |
| Existing questions for author review | `paper/AUTHOR_REVIEW.md` |
| New filter results and exact selected records | `paper/icaart2027_revised_v2/tables/filter_check.json` |

The reference answers and semantic judgments were prepared with Codex, not independently validated by human annotators. Replaying them checks computational consistency; it does not validate their interpretation of the prose. No new human review, additional model run, larger dataset or comparison against an external extraction system is claimed.

The seed documents the original sampling setup, not a guarantee of identical fresh generations. The [vLLM 0.10.2 documentation](https://docs.vllm.ai/en/v0.10.2/usage/reproducibility.html) explicitly limits reproducibility and does not guarantee it for the online serving API used here.

## Reference checks

New citations were checked against primary sources on 13 November 2026:

- [ATOM, EACL 2026, pages 950–966](https://aclanthology.org/2026.findings-eacl.49/) — the work specifically requested by Reviewer 2.
- [ReVerb, EMNLP 2011, pages 1535–1545](https://aclanthology.org/D11-1142/).
- [OLLIE, EMNLP-CoNLL 2012, pages 523–534](https://aclanthology.org/D12-1048/).

This is a local revision package. Nothing has been submitted or pushed; the new revision files are not yet available at the public repository URL. The manuscript links the existing public repository and retained evidence. Distribute this package with the revision or publish its files when submitting the final version.
