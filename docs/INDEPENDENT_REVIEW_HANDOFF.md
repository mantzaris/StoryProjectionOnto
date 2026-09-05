# Independent review completion handoff

This workflow completes—but never performs—the mandatory independent review of
the nine condition-blind held-out projections. The authoritative plans require
one independent, condition-blind review, followed by adjudication and freezing
of disagreements and permissible alternatives before condition outputs. They do
not require a second reviewer or an institutionally independent adjudicator, and
they do not prohibit the project researcher or user from adjudicating after
another human has authored the substantive independent review. Both records
must be authored outside the model runtime. Test fixtures are not scientific
reviews and cannot authorize a held-out run.

## Inputs and roles

1. Materialize the deterministic human-readable handoff from the sealed package:

   ```bash
   PYTHONPATH=src python scripts/render_independent_review_handoff.py \
     --output-root artifacts/restricted/scorer_only/independent_review_handoff \
     --materialize
   ```

   Give the independent human reviewer only the condition-blind
   `review_packet.md` and
   `review_response_template.tsv` from that restricted directory. The sealed
   `data/synthetic/scorer_only/review/blind_review_package.json` and generated
   reviewer-response schema remain the exact machine-readable references. Do
   not give the reviewer condition output. The TSV is a blank human worksheet,
   not an executable `IndependentReviewResponse`; conversion to the validated
   JSON contract occurs after the human has completed it.
2. Save their completed `IndependentReviewResponse` in a restricted location.
   Notes and the reviewer pseudonym are restricted by default.
3. An adjudicator receives the response and writes a `ReviewAdjudication`. The
   project researcher or user may fill this role. The current validator requires
   its `adjudicator_pseudonym` to differ from the response's
   `reviewer_pseudonym`; it compares them case-insensitively and fails closed.
   Distinct pseudonyms are an implementation mechanism for auditable role
   separation, not a claim that the plans require a second reviewer or an
   institutionally independent adjudicator. Every `DISAGREE` or `UNCERTAIN`
   item must be resolved; agreed items must not be included.
4. `RETAIN` keeps the original sealed scorer semantics. `AMEND` requires a
   `ReviewAmendmentBundle` containing the complete amended
   `GoldContextualProjection` and `GoldAlternativeSet`. Each AMEND item must name
   the replacement's canonical `final_semantic_hash`. `EXCLUDE` is not executable
   under the registered method and blocks pending a methodological amendment.

The bundle schema can be generated with
`ReviewAmendmentBundle.model_json_schema()`. Workflow status/hash fields in an
amended projection are normalized by the completion runtime; its substantive
semantic hash must reproduce exactly and must differ from the sealed source.

The rendering is scorer-only and append-only. It displays all three selected
blind worlds, all nine projections, their shared evidence, proposed identity,
event, qualified-assertion, temporal/epistemic, rare-pivotal, contrast,
community, and alternative structures, and the package's exact 72 questions.
It contains no reviewer disposition, adjudication, evaluated condition output,
or held-out launch authorization. A byte-identical rerun is accepted; partial,
extra, changed, or non-restricted output is rejected without replacement.

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
