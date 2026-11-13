# StoryProjectionOnto
Projecting stories onto temporal ontological maps

## Authoritative plans

Implementation and scientific decisions are governed only by:

- `plan_notes/METHODOLOGICAL_PLAN_QUERY_DEPENDENT_TEMPORAL_ONTOLOGY.md`
- `plan_notes/IMPLEMENTATION_PLAN_QUERY_DEPENDENT_TEMPORAL_ONTOLOGY.md`

Other files in `plan_notes/` are retained as historical provenance and do not
control the current study.

## Reproducing the corrected synthetic inputs

Scorer artifacts are generated into an ignored, scorer-only namespace; model
workers receive only an allowlisted evidence/query stage, never the corpus root.
Before running tests or a CPU calibration from a fresh checkout:

```bash
python scripts/generate_synthetic_benchmark.py
chmod 700 data/synthetic/scorer_only
python scripts/generate_synthetic_benchmark.py --verify-only
```

This does not authorize inference. The current temporal-reference amendment,
replacement human-review package, and execution gates are described in
`docs/TEMPORAL_REFERENCE_AMENDMENT.md`, `docs/INDEPENDENT_REVIEW_HANDOFF.md`, and
`RUN_STATUS.md`. Earlier scorer artifacts remain preserved in Git history.
