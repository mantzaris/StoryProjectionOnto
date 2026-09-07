# C0 development competence: denominator audit

No gold, metric scope or competence threshold is changed by this audit.
Implementation plan §13.2 and methodology §20 require ≥.85 precision and ≥.70
recall on **directly stated development qualified assertions**, explicit fixture
family coverage and valid citations. Methodology §12 separately defines contextual
strict F1. Neither plan says that every prediction missing from a context-filtered
reference is factually unsupported.

## What the code currently computes

`_assess_c0` evaluates 12 post-query projections. `build_alignment_plan` first
excludes `gold.relevance=False` assertions. `_direct_assertion_ids` further
restricts targets to evidence found in `fact_evidence_ids`. Every predicted
assertion, including supported background and causal assertions excluded by that
map, remains in the precision denominator. It yields 31/281 precision and 31/84
recall in the preserved corrected-v4 calibration. This is not an unsupported-fact
rate or a clean query-blind extraction-competence measurement.

The 214 predictions lacking an eligible direct target reconcile as **210** with
references marked context-irrelevant and **4** with relevant references excluded
by the direct filter. Those are eligibility counts, not support judgments. There
is no evidence here that all 214 lack reference annotations: their evidence does
occur in the unfiltered annotations.

## Bounded substantive audit

Selection was frozen before reading examples: lowest two SHA-256 ranks within
each of twelve development contexts, seed `c0-denominator-audit-20260907-v1`.
The 24 retained examples are classified as 22 supported/context-irrelevant,
one supported/relevant but erroneously outside the direct-fact map, and one
unsupported identity merge. Zero sampled cases require inventing a missing
reference; none is a within-projection duplicate. These are assistant development
error findings, **not the independent held-out review**, and not extrapolated
truth labels for the other 190 predictions. Full graph equivalence and all rich
description clauses still require strict validation.

Readable examples:

- Evidence: Cyra Cedar belongs to Cedar Guild at story step 1. C0 emits that
  membership with unknown intrinsic validity. The causal query's reference
  explicitly retains it as irrelevant background. Counting it as contextual
  precision loss can be legitimate; calling it unsupported extraction is not.
- Evidence explicitly states that Galen Cedar's earlier action enabled Cyra
  Cedar's later action at step 5. C0 emits `enabled`; the causal query already has
  relevant `causal.00` gold. That evidence comes from `causal_dependencies`, not
  `WorldSpec.facts`, so `_direct_assertion_ids` excludes an explicitly stated fact.
- Evidence says Galen Silver participates in **Silver Turn 2**, with validity
  explicitly 2–4. C0 points to its merged **Silver Turn 1** entity containing
  both events' mentions. The identity normalizer dropped digits. A general CPU
  repair preserves numeric identity tokens (also Gate 1/2, R2/R3), without
  altering query-time creation permissions, gold or temporal matching.

## Proposed scope clarification (requires approval; not applied)

Keep contextual strict F1 unchanged: all displayed assertions enter its precision
denominator, so supported-but-irrelevant content remains a contextual false
positive, not a claim of factual falsity.

For the separate **direct-extraction competence gate**, assess sealed query-blind
C0 preontologies once per development world against a complete, evidence-grounded
direct-assertion reference, independent of query relevance. Include all C0 emitted
qualified assertions in that declared evaluation scope, including unsupported
ones; do not filter predictions merely because they lack a gold match. Explicit
causal-dependency and precedence witness sentences must be eligible alongside
`WorldSpec.facts`. Resolve missing reference coverage before scoring, without
inventing hidden qualifications. Preserve one-to-one matching, evidence/identity/
relation/direction/holder correctness, essential story and intrinsic validity, and
the unchanged .85/.70 thresholds. Report contextual performance separately.

This is a clarification/amendment to the implemented competence target universe,
not a numerical shortcut or a claim C0 now passes. It requires auditing the direct
reference/alternatives before new competence results can be certified. The
numeric-identity extraction repair is independent and can be measured under the
unchanged old scoring scope in the meantime.

Restricted audit: `artifacts/restricted/c0-denominator-sample-v2.json` and
`c0-denominator-judgments-v1.json`. The reproducible export script refuses held-out
scorer files and never updates calibration, reference or scoring artifacts.
