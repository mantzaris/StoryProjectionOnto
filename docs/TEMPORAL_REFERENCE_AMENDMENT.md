# Approved temporal-reference correction (2026-09-07)

This explicit pre-held-out benchmark-semantic amendment implements the user's
approval of `docs/C0_VALIDITY_CORRECTION_PROPOSAL.md`. The authoritative plans
are unchanged. Their distinction between story/event time, intrinsic validity,
and query visibility is restored; query-window clipping was an implementation
error, not a licensed temporal inference.

## General rule

- A query changes visibility/relevance, never an assertion's intrinsic bounds.
  The shared non-mutating `query_time_visibility` routine is used by reference
  relevance, CPU fixed selection and the condition-independent graph display.
- An observation at step N proves neither onset at N nor indefinite persistence.
  Unstated durations remain `unknown`; unknown duration outside an observation
  window is retained as uncertain, not asserted persistent or certainly excluded.
- Exact office tenure and event participation/duration are registered temporal
  targets. Their onset and end are now explicitly stated in the same query-blind
  evidence given to every condition. An explicitly unknown tenure end remains
  unknown; event duration is parsed from supporting text, not latent metadata.
- Other relations' latent finite endpoints are not required truth. Their
  intrinsic validity is unknown unless explicit evidence supplies it. Auxiliary
  answer-identity signatures no longer encode unspoken latent duration.
- Causal/precedence relations between fixed occurrences have `not_applicable`
  independent validity, not a truth interval equal to the gap between events.
  Story ordering remains essential. `unknown` and `not_applicable` are distinct.

Gold compilation and C0 independently parse the explicit evidence. No model-side
codec or C2 validator fills dates or other missing semantics. C0 qualifies before
query reveal and never invents post-query temporal bounds. Strict assertion
matching still checks essential story and validity fields: invented or clipped
bounds and an inappropriate unknown/not-applicable value fail.

The legacy decision slot `qualification/story-scope` now describes actual
qualified contextual relations and their roles, not a copied viewport string.
Changing visibility alone is **not** credited as a constructive ontology change.
All contrast pairs must still have a real nonselection compiler delta.
The eight A-NoTemporalEpistemic cases remain four duration cases with relevant,
explicit, bounded intrinsic validity and four holder-status cases. Eligibility
no longer demands the erroneous query-clipping behavior. Temporal-change world
quotas, event reification, precedence, essential temporal scoring, and mutation
tests remain required.

## Preservation, replacement and review

The complete previous benchmark, original configuration and development-call
manifest, and old review instructions are preserved under
`artifacts/restricted/temporal-reference-amendment.SOjSCF/predecessor-benchmark`
and its parent directory. Earlier failed C0 calibration and field comparisons
are unchanged. Prior tracked artifacts also remain in Git history.

The replacement generator is `semantic-query-compiler-v4-intrinsic-validity`,
with index revision `synthetic-query-blind-index-v4-explicit-durations`.
Formal worlds, seeds, queries, sample sizes, hypotheses, conditions, final object
budgets and scientific thresholds are unchanged. Shared evidence, references,
temporal decision signatures and temporal eligibility bindings are revised by
the general compiler, not by condition outputs or individual C0 predictions.

All predecessor evidence seals, gold hashes, review packages, review responses
and derived preontology/packing/acceptance certificates are stale for the new
inputs. None authorizes the replacement. Updated held-out scorer artifacts and
the new nine-projection condition-blind review package stay ignored/restricted;
previously tracked scorer files are removed from the index, not erased from disk
or history. Human judgments must refer to the replacement package. The human
review gate remains closed; no old review is carried forward.

The revised input lengths must be remeasured before any future ordinary LLM
acceptance/development execution. This CPU correction is not a token-feasibility
or model-acceptance result. The fifth small diagnostic used its frozen original
request before these changes and cannot validate the revised benchmark.
