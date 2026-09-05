# Implementation checklist

This checklist maps the seven implementation phases to the authoritative
methodological and engineering plans. A checked item requires a persisted,
hash-addressed artifact or a passing test; prose-only completion does not count.

## Global invariants

- [x] Keep `C0 ClassicalPre`, `C1 LLMPre`, `C2 LLMQuery`, and
  `A-FixedSelect` as separate implementations behind one interface.
- [x] Keep the pre-query `EvidenceIndex` free of finalized entities, events,
  predicates, qualified truth assertions, query relevance, and scorer gold.
- [x] Require C0/C1 construction seals and a C2 empty pre-query inventory plus
  post-query construction certificate.
- [x] Mechanically forbid new identity, schema, predicates, events,
  abstraction, and qualifications in `A-FixedSelect`.
- [x] Preserve distinct story time, validity time, discourse order,
  proposition-revelation position, spoiler horizon, and holder epistemic scope.
- [x] Apply identical evidence packets, horizons, upper ontology, final object
  budgets, display budgets, repair policy, and paired LLM seeds.
- [x] Keep scorer gold physically and logically outside model-visible paths.
- [x] Retain invalid, refused, timed-out, and unrepaired attempts in
  intention-to-treat results.
- [ ] Never publish model weights, private paths, novel prose, reconstructive
  offsets, or restricted indexes.

## Phase 1 — contracts, resources, and GPU pilot

- [x] Define immutable versioned records, JSON Schemas, canonical JSON, and
  SHA-256 hashing for evidence, context, ontology, validation, revisions,
  visualization, run manifests, and all certificates.
- [x] Implement explicit model-visible allowlists and boundary/capability tests.
- [x] Implement storage preflight: at most 25 GB occupied, at least 5 GB
  headroom, at most 30 GB allocation, all project-controlled paths counted.
- [x] Implement monotonic GPU-service accounting, eight allocation-event
  inventory, 278 attempt slots, reserve tiers, 9-hour admission, and hard stop
  before 10 actual hours.
- [x] Implement append-only SQLite ledger, content-addressed compressed blobs,
  atomic writes, resume, DAG verification, release classes, decoding/seed
  manifests, packing reports, and repair lineage.
- [x] Add hand-authored C1/C2/FixedSelect fixtures and unit/property/integration
  tests before model contact.
- [x] Freeze the smallest practical Python environment; cap study execution at
  8 CPU workers and project process RAM below 25 GB.
- [x] Inventory immutable container CUDA/Python/PyTorch/vLLM separately from
  project-controlled writable files.
- [x] Preflight storage, configure one shared model cache, fetch and hash only
  Qwen3-14B-AWQ revision `1a6fe1ecf891437a270cce11ad54d796c4f56ce0`,
  verify Apache-2.0, and retain no second model snapshot. This is the completed
  historical primary-model gate: its terminal rejection and authorized cache
  replacement are preserved, and the 14B snapshot is no longer retained.
- [ ] Complete the active Qwen3-8B-AWQ fallback micro-pilot that replaces the
  normal eight-call primary-model block under the registered symmetric fallback;
  validate packing, structured C1/C2/FixedSelect output, grounding, operators,
  horizon integrity, restart/resume, p50/nearest-rank p95, VRAM <23 GB, and
  process RAM <25 GB. V8 reached the healthy pinned service but failed closed
  during service adoption before any inference call or accepted output. Its
  bounded EngineCore identity repair passes the runtime suite, but a fresh GPU
  start is not admitted while the all-in schedule exceeds 9 hours. V3 through
  v8 remain preserved terminal provenance and are nonexecutable.
- [ ] Admit the study only if the complete measured forecast is <=9 hours
  (operational target approximately <=8.25 hours); otherwise execute only the
  single permitted symmetric 7B/8B AWQ fallback procedure or stop.

## Phase 2 — synthetic benchmark and blinded review

- [x] Generate four development and twelve sealed held-out `WorldSpec`s with
  three contexts each and approximately 10–20 gold entity/event nodes.
- [x] Satisfy the exact lens, story-scope, abstraction, viewpoint, horizon,
  node-budget, difficulty, factor, rare-pivotal, temporal, and community quotas.
- [x] Include identical-evidence nonselection contrasts, pregraph-friendly/null
  cases, conflict/uncertainty, beliefs/reports, distractors, and permissible
  alternatives.
- [x] Record deterministic root-seed derivations, renderer choices, candidate
  sets, rejections, paraphrases, contrast proofs, and mutation-test deltas.
- [x] Generate the condition-blind independent-review package for the seeded
  easy/medium/hard worlds and all nine projections.
- [ ] Freeze reviewer decisions/adjudication before any held-out condition run.

## Phase 3 — conditions and primary comparisons

- [x] Implement a credible CPU C0 with spaCy NER, dependencies,
  alias/coreference, events, relations, temporal rules, evidence provenance,
  pre-query sealing, and fixed query-time projection.
- [ ] Development-calibrate C0 on the four development worlds and freeze its
  competence record before held-out reveal (all registered fixture families,
  at least 0.85 directly stated qualified-assertion precision, at least 0.70
  recall, and valid evidence for every assertion). This unchecked item is an
  execution/calibration gate; it does not mean the C0 implementation is absent
  or incomplete.
- [x] Implement query-blind GPU C1 preconstruction reused across three contexts.
- [x] Implement active post-query GPU C2 construction with merge/split, event
  reification, local schema/relation, abstraction, temporal/epistemic, rare
  guard, evidence-grounded descriptions, and certificates.
- [x] Implement query-time GPU `A-FixedSelect` over the complete matching-seed
  C1 graph with construction operations rejected by grammar and validator.
- [ ] Pass evidence equality, gold firewall, seals, certificates, packing,
  capability, horizon, budget, and development competence gates.
- [ ] Run 24 development calls, freeze prompt/schema/runtime/threshold behavior,
  recompute the complete forecast, and only then reveal held-out inputs.
- [ ] Run and seal 24 C1, 72 C2, and 72 FixedSelect held-out outputs with full
  timing/token/failure lineage.

## Phase 4 — metrics, statistics, and ablations

- [x] Implement contextual-node and strict qualified-assertion P/R/F1;
  six-family ontology-decision macro F1; contrastive change/collapse;
  temporal, grounding, citation, unsupported, rare-pivotal, and support-path
  measures.
- [x] Implement declared degree, relation-neighborhood, native-schema, and
  canonical-mapped entropy with denominators/undefined cases visible.
- [x] Implement node/edge/density/isolate/component, label overlap, irrelevant
  load, discoverability, and two-part crossing-opportunity measures.
- [x] Implement Leiden-CPM, AMI, purity, conductance, fragmentation/merging,
  cluster/modularity descriptions, and cross-seed AMI/VI.
- [x] Implement 12-world aggregation, two one-sided paired t-tests, Holm,
  two-sided paired intervals, all 4,096 sign flips, 10,000 world bootstraps,
  corrected paired effects, C2–C0, gated C2–FixedSelect, and rare noninferiority.
- [ ] Run only the frozen 12 `A-NoContext`, 8 `A-NoTemporalEpistemic`, and
  8 `A-NoRareGuard` one-switch overlays plus the 12 paraphrases.

## Phase 5 — visualization and feedback

- [x] Build the thin local Cytoscape.js page with rich grounded node/assertion
  labels, time/epistemic/confidence/evidence detail, filtering, and stable layout.
- [x] Implement only `REFINE_CONTEXT` and `REQUEST_MERGE_SPLIT`, with
  condition-independent anchors and per-condition resolutions.
- [ ] Run six preregistered known-answer scripts and three researcher traces;
  report replay, latency, graph diff, capability limits, and only applicable
  gold-based metrics without usability claims.

## Phase 6 — bounded first-novel case study

- [ ] Obtain the exact lawful local path only when all possible preceding work
  is complete; keep it ignored and restricted.
- [ ] Query-blindly index one *A Game of Thrones* copy into passage IDs and FTS5.
- [ ] Preregister four windows, one horizon and two contrastive contexts per
  window; run C0, four C1 prebuilds, and eight C2 constructions with one seed.
- [ ] Run one separately labeled full-index BM25/FTS C2 demonstration and audit
  evidence, relevance, time, rare facts, organization, and the planned subset.
- [ ] Use only high-level paraphrases and opaque evidence IDs publicly.

## Phase 7 — results and release

- [ ] Regenerate every paper number from immutable canonical result tables and
  enforce report/table consistency.
- [ ] Produce `reports/RESULTS_REPORT.md`, `reports/RESULTS_REPORT.pdf`, canonical
  tables, publication figures, hash manifests, reproducibility/resource reports,
  and a safe public bundle below 2 GB.
- [ ] Include all required primary/secondary, mechanism, rare, entropy/clutter,
  community, paraphrase, ablation, feedback, case, resource, failure, error,
  negative-result, limitation, and claim-boundary sections.
- [ ] Include frozen-rule tutorial, rare-pivotal, temporal/epistemic, held-out,
  counterexample/failure, and safe narrative illustrations with fixed anchors.
- [ ] Run unit/property/integration/regression/UI, firewall/equality/certificate,
  accounting/resume, public-scan, PDF visual-inspection, and consistency checks.
- [ ] Stop vLLM, retain the pod, record local/remote hashes, commits, complete
  call/time/RAM/VRAM/storage accounting, deviations, and incomplete outputs.
