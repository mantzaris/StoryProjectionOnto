# Synthetic benchmark boundaries

The benchmark is deterministically compiled by
`scripts/generate_synthetic_benchmark.py` from
`configs/study/synthetic_benchmark.json`. The compiler creates four development
worlds and twelve sealed held-out worlds, with three contexts per world. The
held-out allocation is balanced over easy, medium, and hard strata and includes
the registered temporal, epistemic, merge/split, event-reification, schema,
abstraction, conflict, community, rare-pivotal, fixed-friendly, and null cases.

## Semantic compilation

World facts and query contexts are typed scorer inputs. Story scope, viewpoint,
and abstraction are compiler inputs rather than descriptive tags. Each contextual
gold projection is compiled from those inputs, and its canonical nonselection
decisions are represented as `(slot_key, operator, signature, anchors, evidence)`
atoms, with temporal and epistemic decisions linked to the assertion IDs they
qualify. Signed A/B decisions are the exact diff of those atoms; substitutions use
the canonical `old=>new` signature. Every decision anchor resolves to a candidate
or temporal clue in its cited evidence.

Rare-pivotal facts have one witness. Their typed, directed support paths begin at
the rare assertion, follow two registered causal-dependency assertions, and end at
the outcome assertion. Only context-relevant rare facts are pivotal. Each eligible
case has a fresh source-regeneration deletion proof showing a change in the answer,
the full generated bundle, and the registered causal, temporal, state, or community
component. Permissible alternatives are closed atomic joint representations: the
only supplied alternate couples an inverse direction with swapped endpoints while
holding partition, evidence, time, and holder semantics fixed. Field-wise Cartesian
hybrids are invalid.

## Gold firewall

The generated tree has two deliberately separate namespaces:

- Each child of `data/synthetic/model_visible/prequery_stages/` is a complete C1
  job sandbox containing exactly `evidence.json` and `manifest.json`; it is
  incapable of carrying a query. Each child of
  `data/synthetic/model_visible/query_stages/` contains exactly the same sealed
  evidence, one `query.json`, and its manifest. A worker receives one child
  directory, never the corpus root. File allowlists and content hashes are local
  to that job, and both keys and string values are scanned for split, scorer,
  gold, pair, and expected-result leakage.
- `data/synthetic/scorer_only/` contains formal world semantics, split routing,
  contextual gold, executable alternatives, sampling records, mutation proofs,
  and review material. The model runtime loader rejects paths outside the exact
  one-job stage and does not import the scorer compiler.

Held-out surface renderers are absent from development data. Gold and scorer IDs
may identify the split inside `scorer_only`; actual model request payloads must not
contain `syn-test`, `syn-dev`, split/world/pair labels, gold fields, or expected
metadata. The verifier rejects missing, changed, symlinked, or unmanifested files.

## Independent-review lifecycle

Exactly one held-out world is selected within each difficulty stratum before any
condition output. The condition-blind draft package contains all nine projections
and a fixed 9-by-8 Cartesian rubric. A response must bind the exact package hash,
review item IDs, projection IDs, and criteria. A typed adjudication must cover
exactly every disagreement or uncertain response. A final seal additionally binds
the nine reviewed gold/alternative artifacts and their canonical semantic hashes.
The scorer-only binding manifest connects blind projection IDs to the exact sealed
gold and alternative hashes; retained semantics must hash to the original, and an
adjudicated replacement must hash to the declared amendment.

Generation deliberately emits only `held_out/draft_seal.json`, the scorer binding,
and schemas for the response, reviewed replacement, adjudication, and final seal.
It does not emit a response or pretend that its own checks are the independent
review. The manifest remains
`pending_independent_review`, `review_complete=false`, and
`held_out_launch_authorized=false`; a draft seal cannot launch held-out condition
execution. Unit-test responses are contract fixtures and are not scientific
reviews.

## Frozen auxiliary selections and provenance

The A-NoContext manifest selects exactly twelve contexts: one per held-out world,
two per lens family, and four per difficulty. It is frozen before outputs. The
paraphrase and ablation eligibility selections are likewise seed-derived from
explicit candidate populations. Temporal candidates require a changed assertion
validity scope; holder candidates require a relevant denied proposition whose
narrative commitment changes when viewpoint is removed; rare candidates require a
typed support path and a successful regenerated deletion proof.

Fixed-friendly is not assigned from a label alone. In four registered held-out
worlds, context C is constrained to the predeclared actor-level reference profile,
and the scorer records exact equality between independently compiled fixed-reference
and query-dependent semantic hashes for that actual query. The three null proofs
change wording only while holding lens, scope, abstraction, horizon, and viewpoint
constant.

The first benchmark candidate failed the internal scientific audit before
external review. Its manifest/seal hashes and the measured failures are retained
only in `scorer_only/provenance/rejected_candidate.json`; none of its artifacts is
presented as reviewed or valid, and no condition output was generated from it.

The top-level manifest records hashes for the compiler, runtime, contracts, and
alignment semantics plus the exact configuration,
content hashes for all generated files, root-seed derivations, factor coverage,
allocations, review selection, mutation proofs, rejected-candidate provenance,
and the pending draft seal. Regeneration into an empty directory is
byte-identical. Existing differing files are never overwritten.
