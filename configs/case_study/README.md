# Restricted first-novel case configuration

`indexing.json` freezes only query-blind segmentation and FTS settings. It does
not contain a corpus path, novel text, windows, annotations, or query outputs.

The source path and enclosing restricted root must be supplied explicitly to
`scripts/prepare_novel_case.py`; both must remain under an ignored restricted
directory such as `.local_data/restricted/`. The script will not search for the
novel and requires an affirmative lawful-copy attestation. It writes the index
and its manifest only inside that restricted root and refuses to overwrite
existing files.

The four windows, eight bounded contexts, fixed per-window horizons, and one
separate operational FTS query are intentionally absent until they can be
preregistered from the lawfully supplied source. Their typed contract is
`CaseStudyPreregistration` in `story_projection_onto.novel_case`.
