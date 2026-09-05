# Independent review completion handoff

This workflow completes—but never performs—the mandatory independent review of
the nine condition-blind held-out projections. The reviewer and adjudicator must
author their records outside the model runtime. Test fixtures are not scientific
reviews and cannot authorize a held-out run.

## Inputs and roles

1. Give the external reviewer only
   `data/synthetic/scorer_only/review/blind_review_package.json` and the generated
   reviewer-response schema. Do not give them condition output.
2. Save their completed `IndependentReviewResponse` in a restricted location.
   Notes and the reviewer pseudonym are restricted by default.
3. A separately acting adjudicator receives the response and writes a
   `ReviewAdjudication`. Its required `adjudicator_pseudonym` must differ from
   the response's `reviewer_pseudonym`; the validator compares them
   case-insensitively and fails closed. Every `DISAGREE` or `UNCERTAIN` item
   must be resolved; agreed items must not be included.
4. `RETAIN` keeps the original sealed scorer semantics. `AMEND` requires a
   `ReviewAmendmentBundle` containing the complete amended
   `GoldContextualProjection` and `GoldAlternativeSet`. Each AMEND item must name
   the replacement's canonical `final_semantic_hash`. `EXCLUDE` is not executable
   under the registered method and blocks pending a methodological amendment.

The bundle schema can be generated with
`ReviewAmendmentBundle.model_json_schema()`. Workflow status/hash fields in an
amended projection are normalized by the completion runtime; its substantive
semantic hash must reproduce exactly and must differ from the sealed source.

## Validate, then materialize

Run from the repository root in the frozen project environment:

```bash
python scripts/complete_independent_review.py \
  --response /restricted/input/reviewer_response.json \
  --adjudication /restricted/input/adjudication.json \
  --validate-only

python scripts/complete_independent_review.py \
  --response /restricted/input/reviewer_response.json \
  --adjudication /restricted/input/adjudication.json \
  --amendments /restricted/input/amendments.json \
  --output-root artifacts/restricted/scorer_only/independent_review \
  --materialize
```

Omit `--amendments` when there are no AMEND decisions. `--dry-run` performs the
same validation and prints the planned logical file inventory without writing.
The default output is ignored, restricted scorer-only storage outside the
immutable `data/synthetic` tree.

Materialization is atomic. A complete byte-identical tree is an idempotent
resume. Partial trees, extra files, changed bytes, symlinks, stale staging paths,
or replayed hashes fail closed and are preserved for audit rather than replaced.

## Held-out execution gate

Held-out orchestration must call
`require_materialized_independent_review_complete()`. The loader re-parses the
72 response bindings, adjudication, nine scorer-source bindings, nine reviewed
artifacts, draft seal, final seal, and self-hashed completion manifest. It calls
the canonical `require_independent_review_complete` gate and reproduces every
hash. It never trusts `held_out_launch_authorized` as a standalone boolean.

The persisted tree contains reviewer notes and scorer gold and therefore must
remain restricted. Never place it in prompts, model staging, reports, or a public
artifact bundle.
