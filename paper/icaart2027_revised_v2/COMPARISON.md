# Comparison of the two revision attempts

Both columns compare their revision directly against the original submission, not against each other. Both revisions preserve the original title and abstract exactly.

| Source | v1 removed | v1 added | v1 total edits | v2 removed | v2 added | v2 total edits |
|---|---:|---:|---:|---:|---:|---:|
| Main manuscript source | 900 | 1178 | 2078 | 522 | 734 | 1256 |
| Figure captions and definitions | 239 | 29 | 268 | 242 | 32 | 274 |
| Synthetic results table | 26 | 38 | 64 | 26 | 38 | 64 |
| Prose results table | 39 | 7 | 46 | 39 | 7 | 46 |
| Bibliography source | 0 | 145 | 145 | 0 | 145 | 145 |
| **Total of these sources** | 1204 | 1397 | 2601 | 829 | 956 | 1785 |

By this measure, v2 has **39.6% fewer main-manuscript source-token edits** and **31.4% fewer edits across all five source components** than v1. A negative percentage would mean more edits. These counts measure additions plus deletions, not manuscript length.

## Method and limits

Each whitespace-delimited source token is compared literally, using Python's `difflib.SequenceMatcher(..., autojunk=False)` independently for each file. Whitespace-only changes are ignored; TeX commands, punctuation, comments and table syntax remain included. A replaced token counts as one removal and one addition, and a moved passage can count on both sides. These are source-token counts, not linguistic word counts or changed PDF lines; the algorithm does not establish a mathematically minimal edit sequence.

The total includes the main manuscript, figure definitions/captions, both main tables and bibliography, with no double-counting of expanded inputs. It excludes artwork, the companion, extra diagnostic tables, scripts and supporting documents. The file hashes and full counts are in [comparison.json](comparison.json). Rerun after any source change:

```sh
python3 paper/icaart2027_revised_v2/compare_versions.py
```

Both versions use identical saved-output filter results, Figure 1 artwork, fable/Alice artwork and numerical macros. This comparison makes no new model calls.

Fewer changes alone do not establish reviewer satisfaction or better writing. Compare the [v1 diff](../icaart2027_revised/ICAART2027_diff.pdf), [v2 diff](ICAART2027_diff_v2.pdf) and [v2 reviewer response](REVIEW_RESPONSE.md) to assess the retained explanations and scope. Both attempts retain the limitation that the saved-output comparison is not independent validation on new texts.
