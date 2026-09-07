# C0 development competence: denominator audit

## Approved amendment and actual evaluations — 2026-09-07

The user approved `development-query-blind-direct-extraction-v1` and
`source-bound-direct-witness-v1`. The integrated assessor now routes competence
to the four **sealed preconstructions**, once per world. Its previous final-
projection/direct-filter function remains available for exact historical replay.
No C0 extraction implementation changed in this amendment.

| Evaluation | Strict TP / emitted predictions | Strict TP / eligible references | F1 |
|---|---:|---:|---:|
| Preserved historical competence (final projections plus incomplete direct filter) | 29/269 = .107807 | 29/84 = .345238 | Historical record preserved |
| Approved query-blind extraction competence | 53/154 = .344156 | 53/114 = .464912 | .395522 |
| Unchanged contextual strict projection evaluation | 29/269 = .107807 | 29/91 = .318681 | .161111 |

Extraction **fails** the unchanged .85 precision/.70 recall thresholds. Explicit
family coverage is .80 (no strictly matched event-role family), evidence-reference
validity 1.00. All 154 emitted preconstruction assertions remain in extraction
precision. All 269 final assertions remain in contextual precision. In particular,
**73** final assertions strictly match a direct reference when relevance is
ignored but are excluded by that query's gold relevance; they remain contextual
false positives. They are not labeled unsupported merely because they are irrelevant.
The recall denominators 84 and 91 differ because the historical competence filter
omitted eligible contextual causal/precedence targets; the contextual definition
itself has not changed.

### Reference construction, coverage and scientific boundaries

Before opening predictions, the scorer re-renders each development world's
query-blind narrative from its frozen source specification. Every evidence ID and
content hash must match the shared evidence. Direct eligibility comes from the
generator's fact, explicit causal-witness and explicit precedence-witness maps,
not C0 output, target labels or loose citation overlap. Every supplied evidence
record must belong to one source-bound witness family. Each of the three existing
reference organizations must cover every direct witness exactly once; missing or
ambiguous coverage fails before scoring. This passes for **114 direct witnesses**
across four worlds. Repeated mentions of one fact remain one witness target.

The existing gold organizations and their already-declared alternatives provide
permissible representations, with contextual relevance ignored **only for this
extraction scope**. No endpoint, role, relation, epistemic or intrinsic-validity
alternative is invented. Existing strict matching operates within each frozen
organization; deterministic one-to-one matching across their source-witness IDs
then prevents crediting one prediction or witness multiple times across contexts.
The three organizations are not three independent observations or denominators.

Strict matching still rejects wrong temporal bounds. Unknown and not-applicable
remain distinct; query windows do not create intrinsic bounds. The held-out gold,
benchmark evidence, hypotheses, thresholds and human-review package are unchanged.
This development-only scoring sidecar does not substitute for independent review.

### Remaining concrete development errors

Source-sharing nearest-reference comparisons cover the 101 unmatched predictions:
93 have strict signature differences; eight have an exact candidate but lose
one-to-one/duplicate competition. Field counts overlap: subject alignment 75,
object alignment 65, role bindings 50, predicate naming 41, epistemic qualification
five. No validity-only difference occurs in these nearest comparisons; that is
not proof of temporal correctness for every prediction.

- “Galen Cedar's earlier action enabled Cyra Cedar's later action at story step
  5”: C0 `enabled` versus reference `causally_enables`; this closest comparison
  differs only in predicate normalization. It is a concrete lexical-mapping issue,
  not absence of an eligible causal witness.
- “Doran Cedar participates in Cedar Turn 1,” with explicit duration 3–4: C0
  emits a binary relation to an entity representation of Turn 1; the reference
  requires event/participant role bindings to the reified event. The intrinsic
  interval agrees; event identity/organization and roles do not.
- “Doran Cedar reported that Evin Cedar plans to leave Cedar Guild”: the report
  attitude, holder-relative step and non-global commitment agree, but C0's holder
  cannot align to the expected entity. The epistemic error is holder identity,
  not a license to treat the reported proposition as global truth.

The same unchanged predictions produced both new evaluations, so the numerical
change is **not a C0 implementation improvement**. General extraction or lexical
mapping fixes must be separately versioned and assessed. No scores are certified
as competent merely because this scope amendment improves their appearance.

Immutable result, four source-bound maps, four world scores and detailed error
comparisons: `artifacts/restricted/c0-extraction-scope-v2/assessment.json` and its
12 manifest-listed sidecars. Source hashes and the original 17 calibration-file
hashes are recorded/verified. CPU scoring and diagnosis: **3.439123 seconds**;
no construction repeated, zero GPU. The first scoring-only `c0-extraction-scope-v1`
and the old calibration remain preserved.

Verification: **89 focused C0/alignment/temporal/scorer/firewall tests** and
**47 interface/controller tests** pass locally. After deployment at `e841572`,
**15 focused CPU tests** pass on the pod, which reproduces both evaluations
exactly in **2.976049 seconds**. No GPU allocation during this CPU verification.

## Historical scope proposal and prior audit

The preserved current result is **29/269 precision, 29/84 recall, failed**.
This CPU interface change does not rescore it, change a threshold, filter a
prediction, or change a reference. The older 31/281 record below is historical.

| Quantity | Prediction universe | Appropriate treatment of supported background |
|---|---|---|
| Query-blind extraction competence | Proposed: four sealed preconstructions, once per world | Assess against a complete direct-evidence reference, without query relevance |
| Context-dependent projection precision | Current twelve final post-query projections | A supported but context-excluded assertion can be a false positive |
| Unsupported-extraction rate | Explicit evidence-support audit of predictions | Absence from contextual gold is not proof of unsupportedness |
| Reference eligibility/coverage | Audit of target inclusion, separately from predictions | Do not repair coverage by dropping predictions |

### Verified execution route and authoritative scope

`DevelopmentScientificAssessmentProvider.__call__` loads preconstructions to
verify provenance, but passes `_load_cpu_projections(...)` into `_assess_c0`.
`_load_cpu_projections` requires exactly twelve C0 and twelve C1 **final
projections**, keyed by condition, unit and query ordinal. `_assess_c0` reads
`item.projection.instance_graph`, not `ConditionPreparation.ontology_draft`.
It obtains query-specific gold, applies `build_alignment_plan`'s relevance
filter, then `_direct_assertion_ids`' evidence filter, and adds **all** emitted
assertions to `predicted_count`. Strict matched assertions form the numerator.
There is no route that currently computes this gate from four preconstructions.

Methodology §12 defines contextual strict precision and requires endpoint/role,
predicate, direction, essential story/intrinsic-validity, holder and supporting
evidence agreement. That definition justifies the all-projection denominator;
supported but irrelevant assertions must not be removed to improve it.
Methodology §20(4) and implementation §13.2 name the separate competence target
as “directly stated development qualified assertions,” with .85/.70 thresholds
and fixture-family coverage. They do **not** expressly specify whether that
competence denominator is preconstruction or contextual final output. C0's
pre-query construction requirement (§4/§7.1) does not by itself settle the
evaluation universe. Consequently a move to preconstruction scoring is an
explicit scope clarification/amendment, not a silent routing bug fix.

The causal/precedence eligibility issue is narrower: the generator explicitly
writes separate witness sentences and compiles their supported assertions, but
`ScorerWorldArtifact.fact_evidence_ids` serializes only `WorldSpec.facts`, not
`NarrativeProducts.causal_evidence_ids` or `temporal_evidence_ids`.
`_direct_assertion_ids` cannot see those witness maps. The four observed relevant
causal targets are excluded for that reason, not because the prediction is
unsupported. The proposed correction is to materialize a complete **source-bound
direct-witness eligibility map**, including those explicit witness families,
and use it condition-independently. Do not infer directness from matching a C0
prediction, from an assertion's name, or merely from citation overlap. This
requires a declared reference-eligibility revision and new scoring identity;
the existing records remain immutable. No such target-universe change is
activated in this CPU diagnostic-interface patch.

Recommendation: approve the separate preconstruction competence scope and the
source-bound eligibility-map revision together after reference coverage review.
Retain contextual strict F1, every emitted projection assertion in its precision
denominator, all strict temporal rules, and the existing failed calibration as
a separately labeled historical result. Neither support-only filtering nor an
improved-looking recalculation can establish baseline competence.

## Earlier audit evidence and preserved calibration history

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

## Actual repair and recalibration

Numeric identity repair initially exposed a 31-node preconstruction against the
unchanged 30-node budget in development world four. That failed run is preserved
as `c0-calibration-numeric-v5`. Inspection found three spurious “Occurred Before”
events: the neutral index's ordering hints had been treated as new events with
the existing events as agents. A general occurrence-order rule now keeps the
explicit ordering assertion but does not reify a third event. No evidence,
mandatory temporal relation, budget or threshold was removed/changed. A focused
fixture tests the retained relation and absence of the spurious event.

Final source checkpoint `0262fca` completed four preconstructions and twelve
structurally valid development projections in **76.215800437 CPU seconds**, GPU
zero. Under the **unchanged** disputed competence scope: **29/269 precision
(0.107806691)**, **29/84 recall (0.345238095)**, family coverage **0.80**, valid
evidence **1.00**. It still fails, and scores did not improve versus the retained
31/281 and 31/84 result. General bug fixes are not certified by favorable scores.
Remaining comparable-pair errors include identity, roles, relation normalization
and epistemic-holder matching; no temporal mismatch in those comparable pairs
is proof of correctness for every emitted assertion.

**40 focused C0/temporal tests pass.** All 17 listed calibration artifacts match
their manifest hashes; all seven recorded source hashes match checkpoint
`0262fca`. The launch environment accidentally recorded the literal label
`checkpoint` instead of a revision; the separate restricted verification receipt
resolves the source by hashes and preserves the original metadata. No output,
timestamp or source record was retrospectively replaced.

Final calibration and preserved failure are backed up under
`artifacts/restricted/representation-backup.maHyBy/`. This audit proposes the
competence-scope decision above; it does not authorize altered scoring or declare
the integrated development gate complete.
