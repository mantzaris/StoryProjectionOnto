# Implementation Plan V2 for Admissibility-Constrained Transition-Bundle Projection

This is a planning and specification document. It does not create software, annotate text, or run experiments. It translates the fixed-benchmark methodology in `SOLO_RESEARCHER_CONTEXTUAL_NARRATIVE_PROJECTION_PLAN_V2.md` into a deterministic, locally executable Paper 1 program, while superseding targeted engineering details in `IMPLEMENTATION_PLAN_ADMISSIBILITY_CONSTRAINED_PROJECTION.md`.

## 1. Executive implementation decision

Paper 1 will be implemented as a small Python research package over manually validated, episode-bounded oracle graphs. It will compile typed narrative records into a finite root-and-atom representation; construct independently sealed, method-visible 18-root candidate pools; enumerate every permitted root subset; compute semantic closure once; verify semantic admissibility and query responsiveness separately; and select exact optima under fixed componentwise budgets. Proposed-Full, its direct transition-priority ablation, and all guard-wrapped comparators will read the same cached candidate rows. They may differ only in declared scoring and tie rules.

The primary structural comparator is renamed the **Exact Root-Incidence Prize-Collecting Forest baseline**, abbreviated **ERIPCF**. It is inspired by prize-collecting Steiner methods, but it is not unrestricted PCST: it enumerates root subsets, admits no arbitrary Steiner vertices, and computes an exact forest cost on a frozen root-incidence graph. The primary comparison is Proposed-Full versus guard-ERIPCF. Frequency and personalized PageRank (PPR) are secondary conventional and query-seeded comparators. Proposed-NoTransitionPriority isolates the proposed mechanism.

The fixed study contains:

- four held-out *A Game of Thrones* (AGOT) episode bundles;
- one public-domain development bundle;
- one different public-domain held-out bundle;
- one researcher-authored hidden control used only for contamination and verifier controls;
- exactly 25 selectable roots in each of the six natural-text bundles, hence 150 natural-text roots, plus at most 18 hidden-control roots;
- 24 base query families producing 32 answerable contexts: 6 development and 26 held-out;
- 10 primary horizon-rare consequential (HRC) contexts, two in each of the five real held-out bundles;
- 10 typed feasibility or adversarial probes, including two invalid-context probes;
- exactly 18 roots per primary candidate pool;
- root caps of 4, 6, and 8, with development-derived non-root caps;
- eight method variants: Proposed-Full, Proposed-NoTransitionPriority, and native and guard-wrapped frequency, PPR, and ERIPCF;
- no required LLM condition, human study, extraction pipeline, graph database, interactive visualization, or GPU.

The decisive methodological firewall is temporal as well as structural. HRC targets and their initial gold support identifiers are sealed first in restricted evaluation storage. Candidate pools are then generated in an isolated process that can read only a sanitized method bundle and restricted `PoolQueryView`. Pool generation cannot read the target seal, gold outcomes, support labels, rarity, consequence, or final relevance. Target and support availability are computed only after the target and pool seals independently exist. A failed availability gate is a benchmark-construction failure, not permission to repair a pool or replace a target.

Canonical JSONL remains authoritative for records, selections, certificates, manifests, and release artifacts. A local SQLite database is the deterministic, resumable transient store for the millions of exact-search rows. One mandatory public-development performance pilot must justify any runtime or storage forecast before held-out annotation begins.

The default required program is forecast at 108 optimistic, 142 expected, and 206 conservative researcher hours before manuscript writing. The conservative tail is genuine. Held-out material may be opened only after annotation-rate and exact-search pilots re-estimate the protected core at no more than 160 hours; otherwise predeclared secondary work is removed or the project stops. Automated compute time is logged separately and is never treated as free researcher verification.

## 2. Scientific traceability to the V2 research plan

The scientific source of truth is `SOLO_RESEARCHER_CONTEXTUAL_NARRATIVE_PROJECTION_PLAN_V2.md`. Its planning-time SHA-256 is:

```text
a8d63400a1184e0f03c1d5fd4bcd0f6e0261cb59c9dbb146081be846cc5d7a2d
```

The predecessor implementation plan, retained as historical rationale, has planning-time SHA-256:

```text
7c520294ff4e67a4eb88d3a6274afe4cfb2d93e809e37b9b858270f2adaf46bc
```

Both embedded hashes must be recomputed before the first build. This V2 implementation plan must also be checked against the raw SHA-256 reported externally in the final planning-session handoff. A mismatch blocks work until explained. This V2 implementation specification controls execution details where it explicitly corrects the predecessor. It does not silently alter the V2 scientific hypothesis. The research plan should eventually receive a narrow V2.1 textual correction for independent target and pool sealing, ERIPCF terminology, canonical evidence routes, endpoint semantics, and equal-cap ablation fairness. That documentary correction is not a prerequisite for building if this plan and its delivery hash are frozen together.

### 2.1 Non-negotiable scientific invariants

1. Confirmatory projection begins from manually validated oracle records; extraction quality is outside Paper 1.
2. The hidden narrative is excluded from superiority and transfer evidence.
3. The public-domain held-out bundle is unavailable for handbook changes, budgets, coefficients, templates, response schemas, generator rules, or debugging.
4. Semantic admissibility and task response are distinct nested predicates.
5. Guard-wrapped methods share the same pool, canonical routes, closure, guards, response schema, cost calculator, and componentwise caps.
6. Safety is validated through fixtures, corruptions, and guard ablations. Passing guards by construction is not credited as selector utility.
7. The primary rarity variable is manually audited HorizonRarity. LocalRarity is only sensitivity information.
8. Consequence, target identity, `GoldSup`, diagnostic membership, and final relevance never enter a candidate generator or selector.
9. Proposed-Full and Proposed-NoTransitionPriority differ only in the transition-bundle priority coordinate.
10. Exactness means complete optimization over the frozen 18-root, canonical-route problem. It is not a scalability claim or equivalence to unrestricted Steiner retrieval.
11. All primary claims concern the preregistered fixed benchmark. There is no population inference over the novel or narrative genre.
12. A textual anchor licenses an attributed claim under the frozen annotation policy; it does not establish unrestricted fictional-world truth.
13. No held-out miss may trigger target replacement, pool repair, context rewriting, budget change, or method retuning.

### 2.2 Scientific traceability matrix

| Scientific claim | Responsible module | Input artifact | Output artifact | Required test | Evaluation metric | Failure consequence |
|---|---|---|---|---|---|---|
| Horizon safety | `verification/horizon.py` | frozen context, closure, anchors, presentation | structured horizon verdict | inclusive boundary, flashback, fact leak, label leak | leak count by severity | contract gate fails |
| Evidence grounding | `verification/evidence.py` | assertions, canonical route, anchors | evidence witness and verdict | missing, wrong-holder, post-horizon, alternative-route fixtures | evidence precision/completeness; unsupported rate | contract gate fails |
| Epistemic fidelity | `verification/epistemic.py` | proposition, assertion, holder, mode, polarity | qualification verdict | rumor promotion, holder deletion, disputed and negative cases | epistemic violations; rumor promotion | contract gate fails |
| Temporal satisfiability | `temporal/solver.py` | full normalized constraint set | SAT result, entailments, deterministic core | all endpoint cases and multiple-core fixture | inconsistency and non-entailment counts | contract gate fails or temporal display is withheld |
| Identity safety | `verification/identity.py` | entity decisions and dependencies | identity witness | disputed merge, dangling reference, post-horizon alias | identity-error count | contract gate fails |
| Support completeness | `verification/support.py` | response claim and method-visible proof schema | `GuardSup` witness | delete-one-required-atom cases | guard-support pass; gold-support recall scored separately | contract or utility interpretation fails |
| Budget compliance | `costing/budget.py` | closure and fixed presentation | nine-coordinate realized cost | exact-boundary, shared-atom, word accounting | budget violations and realized costs | projection is inadmissible |
| Exact root optimization | `projection/exhaustive.py` | sealed 18-root pool and canonical routes | complete mask ledger and winner | naive enumeration, known optima, deterministic resume | expected mask count; optimum audit | zero-gap claim withdrawn; confirmatory run stops |
| Exact ERIPCF cost | `baselines/root_incidence_prize_collecting_forest.py` | sealed incidence graph, root mask, query seeds, rho rule | canonical forest certificate | exhaustive tiny forests, equal MSTs, no hidden Steiner roots | exact forest score and connector audit | comparator invalid; primary comparison stops |
| Transition-priority mechanism | `projection/objectives.py` | common feasible table | Full and ablation selections | one-coordinate objective regression | paired HRC recall difference at equal caps | no mechanism claim if ablation matches Full |
| Horizon rarity | `annotation/mention_audit.py` | frozen search packets and reviewed candidates | horizon mention ledger | occurrence identity, duplicates, delayed audit | HorizonCount and category stability | use locally singleton or query-necessary terminology |
| Pool independence | `candidates/seal.py` and isolated worker | sanitized pool capsule only | frozen pool and access ledger | forbidden-field, canary, gold-mutation invariance | leakage-audit pass | protocol failure; benchmark cannot proceed |
| Target and support availability | `experiments/availability.py` | independently sealed target and pool manifests | immutable availability assessment | dual-seal and no-write tests | TAR, canonical SAR, any-route SAR | construction failure; no repair |
| Fixed-benchmark superiority | `experiments/decision_gates.py` | sealed 10-HRC results | decision-gate table | hand-computed miniature benchmark | support-complete recall difference, wins, bundle effects | confirmatory claim unsupported |
| Public-domain transfer | `reporting/slices.py` | independent public held-out results | separate transfer table | split-isolation test | held-out public paired difference | conclusion restricted to AGOT cases |
| Annotation uncertainty | `reporting/uncertainty.py` | presealed licensed oracle alternatives | primary, conservative, permissive scores | two-state and three-state fixtures | scoring envelope and gate stability | qualify or withdraw interpretation-sensitive claim |

Each module name is a planned responsibility, not software created by this document.

## 3. Pre-implementation methodological decisions to freeze

The following defaults resolve implementation ambiguities before development. A **methodological** decision affects the scientific object and may not be changed after design freeze. A **replaceable engineering** decision may change only if invariant tests prove byte-equivalent canonical outputs or a logged defect protocol authorizes a versioned rerun.

### 3.1 Independent target and candidate-pool sealing

**V2 requirement identified:** the benchmark has 10 HRC targets, common 18-root pools, gold supports, and post-pool availability measures.

**Ambiguity:** the predecessor constructed targets, supports, and pools in one phase. A one-researcher process could therefore use target knowledge, even accidentally, to make a pool pass its availability gate.

**Frozen methodological interpretation:** the benchmark follows this irreversible state machine:

```text
DESIGN_FROZEN
  -> HELDOUT_METHOD_GRAPH_FROZEN
  -> CONTEXTS_FROZEN
  -> CANONICAL_ROUTES_SEALED
  -> TARGET_GOLD_SEALED
  -> POOL_INPUT_CAPSULES_SEALED
  -> POOL_WORKER_OUTPUTS_SEALED
  -> POOLS_SEALED
  -> AVAILABILITY_JOINED
       -> BENCHMARK_READY
            -> CORRUPTIONS_SEALED
            -> EXPERIMENT_MANIFEST_FROZEN
            -> SELECTOR_RESULTS_SEALED
       -> BENCHMARK_CONSTRUCTION_FAILED
            -> STOP_PRIMARY
```

After held-out method graphs and contexts are fixed, a method-only route compiler writes `CanonicalRouteSealManifest`. A restricted gold-authoring process then selects the 10 HRC targets using the preregistered eligibility and discourse-order rule. It freezes the context-to-target root mapping, complete target-transition atoms, initial acceptable `GoldSup` structures, consequence and rarity labels, diagnostic labels, and licensed annotation alternatives in `TargetGoldSealManifest`. A public commitment may disclose its digest and counts, never target IDs.

Only after that target seal does a separate compiler create a `PoolInputCapsule` containing the sanitized `MethodBundle`, a `PoolQueryView`, response schema, canonical-route seal, generator configuration, global seed, and code/schema hashes. The capsule schema rejects target IDs, gold supports, consequence, rarity, final relevance, diagnostic membership, HRC status, outcome split flags, and unrestricted `OracleBundle` objects. The candidate worker accepts `PoolQueryView`, never the richer `TypedQueryContext`.

Pool generation runs under an Ubuntu mount namespace using `bubblewrap`: immutable code and capsule are read-only, output is the only writable mount, the restricted gold directory and repository root are absent, the home directory is empty, and networking is disabled. The pool process receives neither the target manifest nor its digest, and the digest cannot influence its seed. If this isolation cannot be demonstrated, pool freezing stops. This is a small local access boundary, not distributed infrastructure.

The worker first emits `PoolWorkerOutputManifest`, which references only its capsule and generated bytes and serves as the independent immutable content seal for the pool. After the worker has exited, the controller verifies that the target commitment already occurs in the append-only freeze ledger, then creates `PoolSealManifest` binding worker-output and capsule hashes plus the prior target commitment as chronology metadata. This wrapper cannot alter pool bytes; the target digest was never an algorithm input. Only then may an evaluation-only process read both seals and write an `AvailabilityAssessmentManifest`. It has no write access to graphs, contexts, targets, routes, or pools. It computes target availability and both canonical-route and any-licensed-route support availability. Failure of a predeclared availability or selection-pressure requirement is reported as `BENCHMARK_CONSTRUCTION_FAILED`; gold access by the pool worker is `PROTOCOL_FAILURE`. Neither authorizes repair, target replacement, context rewriting, denominator reduction, query deletion, or candidate substitution. The failure branch stops primary selection; only preregistered construction-failure diagnostics may run. A new benchmark would require a new version and preregistration.

Required tests include state-transition enforcement; capsule forbidden-field rejection; static import separation; an inaccessible gold-directory canary; symlink-escape rejection; byte-identical pools after arbitrary gold-label mutation; refusal to join before two valid seals; invalidation after any postseal edit; absence of a repair operation; and a synthetic missing-target case that deterministically fails. This is a required V2.1 clarification.

### 3.2 Primary candidate-pool generator

**V2 requirement identified:** each answerable context must recreate selection pressure with 18 legitimate roots.

**Ambiguity:** quota categories such as rare irrelevant and incomplete gold support can leak the outcome.

**Frozen methodological interpretation:** the primary generator is deterministic and context-visible. It horizon-filters same-work roots; rejects scenario or unresolved-identity incompatibility; scores only focal-entity overlap, requested root/event type, story/discourse-window compatibility, specified holder, and response-schema type compatibility; takes the six highest scores with opaque-ID ties; and fills to 18 by SHA-256 ordering of `(global_seed, context_id, root_id)`. A frozen, outcome-blind exchange may ensure representation of event, transition, and assertion root kinds. No seed is searched. The quota-balanced pool is optional and diagnostic only.

The candidate worker has no access to consequence, rarity, target, initial support, final relevance, method scores, or test outcomes. TargetAvailabilityRecall is computed after independent seals, before projection. This preserves the scientific question while making a pool miss visible. Required V2.1 correction.

### 3.3 Development-derived budgets

Root caps are analytically fixed at 4, 6, and 8. Non-root caps are derived only from the public-domain development bundle. One canonical development context for each of four response schemas supplies its Pareto-minimal admissible response cost vectors at root cap 8. For each task, choose the vector minimizing, in order, roots, evidence cues, assertions, transitions, the sum of remaining components, and stable atom IDs. For every non-root coordinate, low is the upper median (third sorted value among four task vectors), medium is the maximum, and high is medium plus the upper-median marginal cost of one complete development transition-and-evidence bundle.

The calibration command emits a review report and `budgets.v1.yaml`. All coordinates are integers. The published numerical values in the research plan are provisional envelopes, not empirical values. If a coordinate exceeds an envelope or a task has no admissible response at cap 8, calibration stops for development-only protocol review; it is never clipped and no held-out record is consulted. Nearby sensitivity caps are generated mechanically and frozen, but are optional. This is a methodological clarification.

### 3.4 ERIPCF terminology, activation penalty, and parameter fairness

The comparator called guard-PCST in the research V2 becomes **guard-ERIPCF**. ERIPCF is a study-specific benchmark object, not a claim to implement standard unrestricted PCST. Section 16 gives its exact definition.

The virtual query-root activation penalty is not left arbitrary. For context `C`, define exact rational semantic connector costs `c_C(e)` using the frozen medium non-root caps. Let

$$
L_C=\operatorname{lcm}\{B^{med}_i:i\ne root\},\qquad
\delta_C=1/L_C,
$$

and

$$
\rho_C=\begin{cases}
\max_{e\in E_C}c_C(e)+\delta_C,&E_C\ne\varnothing,\\
\delta_C,&E_C=\varnothing.
\end{cases}
$$

Thus any one declared semantic edge is strictly cheaper than activating another unseeded component, while a long expensive path may still lose to a new component. The rule is frozen before held-out data are opened; each value is computed from the sealed method-visible pool and budget only, then committed before selection. It uses no gold and adds no tuning runs. The existing 12-cell development grid tunes only relevance prize, slot prize, and connector multiplier. Required V2.1 terminology correction.

### 3.5 Direct-ablation cost fairness

Proposed-Full and Proposed-NoTransitionPriority receive identical componentwise caps. The ablation removes only the transition-completeness objective coordinate. Full is not required to use no more realized cost on every coordinate. Realized costs and efficiency are reported, and an optional matched-realized-cost sensitivity cannot replace the equal-permitted-budget result. Required V2.1 correction.

### 3.6 Corruption isolation

Every corruption records its expected full guard-failure signature. A case labeled `isolated` must pass all non-target guards when its target guard is disabled. If logical dependence makes that impossible, it is labeled `multi_guard` and assessed against its complete signature. Development corruptions and templates are frozen before held-out use; held-out corruption instances are sealed before verifier execution. A failed held-out test remains in the confirmatory record after any correction. This is a methodological clarification.

### 3.7 Annotation uncertainty

Each outcome-relevant `AnnotationAlternative` group stores a curator-primary interpretation and at most one additional licensed interpretation before selector execution. Paper 1 permits at most three such groups per query, hence at most eight combined oracle states. Selection never changes. Scoring is curator-primary, conservative (success in every licensed state), and permissive (success in any licensed state). This is a methodological clarification.

### 3.8 Canonical evidence routes

The predecessor retained alternatives for two roots selected by opaque identifier order. That rule is arbitrary and removed.

For the confirmatory problem, every root has exactly one context-and-horizon-specific **canonical route**. Annotation licenses alternatives, but a method-only compiler chooses routes immediately after contexts freeze and seals them before target sealing and pool generation. A route is licensed only if it supports the same typed root assertion with the same proposition, holder, epistemic mode, polarity, scenario, temporal claim, and response meaning. Materially different claims become separate assertions or roots, not routes.

First collapse routes that are identical under the frozen canonical normalizer. The supported normalizations are stable atom ordering, duplicate-atom removal, converse temporal normalization, reduced rationals, sorted assertion qualifications, sorted anchor sets, and exclusion of anchor identity from otherwise identical closure. The class signature contains normalized closure, temporal-constraint bitset, epistemic signature, response-slot signature, component-cost vector, and display-word cost. Within a class, choose by smallest maximum anchor discourse position, then lexicographically smallest sorted anchor-position tuple, then route-content hash. Eliminate componentwise-dominated remaining classes when all scientific qualifications are identical. Select the canonical class lexicographically by:

1. fewest added semantic atoms;
2. fewest evidence cues;
3. shortest fixed-template display word count;
4. earliest maximum discourse position;
5. canonical route-content hash.

This rule is budget-independent and never sees target identity, rarity, consequence, `GoldSup`, final relevance, or outcome results. Its policy and per-context choices are frozen and hashed in `CanonicalRouteSealManifest`. Target availability stays root-level. Canonical-route support availability is the primary construction gate; any-licensed-route support availability is a postseal diagnostic upper bound. A canonical miss cannot be repaired by switching routes. Optional route sensitivity holds the selected mask fixed for Full, the ablation, and guard-ERIPCF on the 10 HRC medium-budget cases. It enumerates the licensed nondominated route product when it is at most 64; otherwise it tests the canonical assignment, every single-root substitution, and the componentwise minimum- and maximum-cost route assignments. It never reselects. This is a required V2.1 clarification and narrows exactness to the declared canonical-route problem.

### 3.9 Feasibility calibration

One complete public-development pilot must traverse annotation, horizon audit, queries, target-style records, pool isolation, closure, all guards, temporal reasoning, budgets, exact enumeration, ERIPCF, Proposed-Full, the direct ablation, metrics, and reporting. It logs active researcher minutes by task. A separate one-context computational performance pilot occurs as soon as exact search and ERIPCF are correct and before held-out annotation. Section 15 defines its thresholds.

Held-out opening requires an expected remaining forecast no greater than 145 hours total and a conservative minimal-core forecast no greater than 160 hours after permitted scope reductions. This gate is methodological. The particular profiler or SQLite query plan is replaceable engineering.

## 4. Recommended technology stack

| Concern | Selected option | Reason it is sufficient |
|---|---|---|
| Runtime | Python 3.12, exact patch pinned | Readable solo maintenance and adequate for 18 roots |
| Environment | `pyproject.toml` plus `uv.lock` | One lock, fast reproducible local environment, no service |
| Typed models | Pydantic v2 strict models, frozen after validation | One source for runtime checks and generated JSON Schema |
| Graph authoring | NetworkX `MultiDiGraph` | Typed parallel relations and easy inspection at this scale |
| Enumeration representation | Python integer bitsets and compact indexed arrays | Transparent exact enumeration of at most 18 roots |
| Temporal reasoning | pinned `z3-solver`, quantifier-free linear rational arithmetic | Exact inequalities, disjunctions, SAT, and entailment |
| Configuration | YAML with pinned safe loader; canonical JSON for hashes | Human-readable entry without ambiguous freeze serialization |
| Canonical records | JSONL; CSV for flat public tables | Diffable and broadly reproducible |
| Transient exact-search store | Python `sqlite3`, one database per context | Indexed, transactional, resumable, local, and dependency-light |
| Testing | `pytest` and Hypothesis | Unit, integration, regression, and invariant/property testing |
| CLI | standard-library `argparse` | Stable typed command boundary without another framework |
| Static quality | Ruff and mypy in strict project settings | Fast consistency checks for a small typed codebase |
| Tables and figures | standard CSV generation and Matplotlib for a few static figures | No notebook or GUI is authoritative |

All dependency versions, Z3 settings, SQLite library version, locale, and Python hash seed are recorded. Exact comparisons use integers or reduced `Fraction` values, never binary floats. PageRank alone iterates in floating point and is quantized under a frozen rule before comparison.

The scientific ontology remains serialization-neutral. Strict JSONL is the Paper 1 exchange format because it is easy to validate and hash. The same qualified assertions could later be serialized with RDF named graphs, reification or RDF 1.2 triple terms, OWL-Time, and PROV-O; ordinary unqualified RDF triples are not alleged to be incapable of temporal modeling. No RDF store or serialization mapping is required to test the projection contract.

A graph database would add serialization and query semantics without benefiting a maximum 18-root search. Distributed processing is unnecessary because contexts are independent and local multiprocessing is sufficient if the performance pilot permits it. A frontend, web API, cloud service, GPU stack, and LLM dependency would broaden neither the fixed-benchmark claim nor the minimal Paper 1 evidence.

## 5. Repository structure

The future repository will use this minimal tree. This tree is a specification only; it is not created in this planning session.

```text
narrative-projection/
|-- .python-version
|-- pyproject.toml
|-- uv.lock
|-- README.md
|-- LICENSE
|-- CITATION.cff
|-- .gitignore
|-- src/narrative_projection/
|   |-- __init__.py
|   |-- cli.py
|   |-- models/
|   |   |-- common.py
|   |   |-- narrative.py
|   |   |-- annotation.py
|   |   |-- sealing.py
|   |   `-- experiments.py
|   |-- serialization/
|   |   |-- canonical.py
|   |   |-- identifiers.py
|   |   `-- hashing.py
|   |-- access/
|   |   |-- visibility.py
|   |   |-- bundle_access.py
|   |   `-- pool_sandbox.py
|   |-- graph/
|   |   |-- compile.py
|   |   |-- integrity.py
|   |   `-- closure.py
|   |-- temporal/
|   |   |-- normalize.py
|   |   |-- endpoints.py
|   |   `-- solver.py
|   |-- verification/
|   |   |-- common.py
|   |   |-- evidence.py
|   |   |-- horizon.py
|   |   |-- epistemic.py
|   |   |-- temporal.py
|   |   |-- identity.py
|   |   |-- provenance.py
|   |   |-- support.py
|   |   `-- budget.py
|   |-- candidates/
|   |   |-- primary.py
|   |   |-- controlled_challenge.py
|   |   |-- capsule.py
|   |   |-- seal.py
|   |   `-- leakage_audit.py
|   |-- costing/
|   |   |-- presentation.py
|   |   `-- budget.py
|   |-- projection/
|   |   |-- cache_store.py
|   |   |-- feasible_table.py
|   |   |-- exhaustive.py
|   |   |-- objectives.py
|   |   |-- evidence_routes.py
|   |   `-- certificates.py
|   |-- baselines/
|   |   |-- frequency.py
|   |   |-- personalized_pagerank.py
|   |   `-- root_incidence_prize_collecting_forest.py
|   |-- annotation/
|   |   |-- compile.py
|   |   |-- mention_audit.py
|   |   |-- reannotation.py
|   |   `-- freeze.py
|   |-- experiments/
|   |   |-- design_freeze.py
|   |   |-- manifests.py
|   |   |-- availability.py
|   |   |-- runner.py
|   |   |-- corruption.py
|   |   `-- decision_gates.py
|   `-- reporting/
|       |-- metrics.py
|       |-- uncertainty.py
|       |-- tables.py
|       `-- figures.py
|-- schemas/
|   |-- generated/
|   `-- schema_manifest.json
|-- config/
|   |-- templates/
|   |-- development/
|   `-- frozen/
|-- data/
|   |-- public/
|   |   |-- public_domain_development/
|   |   |-- public_domain_heldout/
|   |   |-- hidden_control_release/
|   |   `-- fixtures/
|   |-- safe_restricted_manifests/
|   `-- README.md
|-- .local/                         # ignored: path map and private key IDs only
|-- annotation/
|   |-- handbook/
|   |-- templates/
|   |-- issue_ledgers/
|   |-- mention_audit_packets/
|   `-- reannotation_packets/
|-- tests/
|   |-- unit/
|   |-- property/
|   |-- integration/
|   |-- regression/
|   `-- fixtures/
|       |-- public/
|       |-- temporal/
|       |-- exact_optima/
|       `-- corruptions_development/
|-- corruptions/
|   |-- templates/
|   |-- development/
|   `-- sealed_manifests/
|-- manifests/
|   |-- development/
|   |-- design_frozen/
|   |-- heldout_graphs/
|   |-- target_gold_restricted/
|   |-- pool_capsules/
|   |-- pools/
|   |-- availability/
|   |-- experiment/
|   `-- release/
|-- preregistration/
|-- audit/
|   |-- change_ledger/
|   |-- runtime_ledger/
|   |-- access_ledger/
|   `-- leakage_ledger/
|-- artifacts/
|   |-- transient_sqlite/           # ignored and replaceable
|   |-- runs/
|   |-- metrics/
|   |-- tables/
|   |-- figures/
|   `-- release/
|-- docs/
`-- scripts/
```

An external restricted data root, never nested in Git, contains `private_agot_text/`, `private_agot_indexes/`, `private_agot_locators/`, `restricted_oracle_records/`, `target_gold/`, and `private_run_logs/`. HMAC keys live only in an operating-system keyring or a separate secrets location outside repository, corpus, and derived-data roots; ignored local configuration may contain a key locator, never key bytes. Git contains code, schemas, handbook, templates, public-domain data and annotations, eventually releasable hidden-control data, safe configurations and digest commitments, preregistration, safe aggregate results, and release documentation.

The conceptual `.gitignore` denies `.local/`, all restricted roots, raw AGOT text, local full-text indexes, exact offsets, HMAC keys, credentials, `.env`, virtual environments, tool caches, SQLite/WAL files, prose-bearing logs, unredacted oracle exports, and large transient artifacts. Release uses an allowlist, not an ignore list alone. A scanner rejects private paths, raw text fields, exact copyrighted offsets, secret material, and sampled protected n-grams.

## 6. Data-access and copyright boundaries

Four visibility classes are mandatory:

| Class | Typical content | Git and release rule |
|---|---|---|
| `PUBLIC` | code, schemas, public-domain text and annotations, synthetic fixtures, safe aggregates, hidden story after unsealing | versioned and releasable |
| `RESTRICTED_DERIVED` | AGOT oracle atoms, detailed supports, alias maps, exact local locators | access-controlled local storage; release only after legal review and redaction |
| `PRIVATE_RAW` | lawfully acquired copyrighted text, indexes, exact passage bytes | local only; never in Git, tests, logs, or public artifacts |
| `PUBLIC_COMMITMENT` | hashes of safe artifacts, opaque release IDs, counts, approved coarse bibliographic locators | public if release audit passes |

Ordinary salted hashes are not an acceptable protection for short copyrighted passages. A user with the same corpus can dictionary-match a short span even when a salt is known. The plan instead uses three different integrity mechanisms for different purposes:

1. The complete lawfully held edition file receives a local SHA-256 edition checksum. It may be published only after rights review as an edition fingerprint; it is not evidence of lawful access.
2. Each restricted evidence span receives a random opaque `span_id`, private exact offsets, a canonicalization-version identifier, and HMAC-SHA-256 over the canonical span bytes. The 256-bit random key is stored outside the repository and every corpus or derived-data root with restrictive permissions, preferably in an operating-system key store. `hmac_key_id`, algorithm, and span-HMAC value remain in the restricted anchor ledger; the key never enters a data artifact. A public artifact normally contains none of these, except that a generic algorithm label may be documented without an anchor value.
3. Public AGOT artifacts use an opaque release anchor ID, edition metadata, a legally approved coarse locator, and safe counts or length buckets. They contain no plaintext excerpt, exact offset, public salt, short-span digest, or reconstruction-friendly alias packet.

HMAC protects against unaided offline matching only while its key remains private. It does not prove the truth of a claim, authorship, lawful possession, or correctness of a locator. An authorized researcher with the same edition can verify the edition checksum and coarse anchors. Exact span identity is independently reproducible only with lawful access to the same edition plus controlled access to the restricted locator mapping and, when HMAC comparison is required, the private verification procedure. Public users cannot reproduce exact AGOT alignment. The paper must say so. Public-domain and hidden-control pipelines remain fully reproducible.

Raw ingestion is read-only. Visibility propagates monotonically: a derived artifact takes the most restrictive visibility of any source unless a reviewed redaction transformation produces a separate public record. The release validator rejects raw text fields, private paths, exact copyrighted offsets, HMAC keys, AGOT span-HMAC values, unapproved aliases, unredacted rationales, post-horizon descriptions, or protected n-grams. Tests use public-domain or synthetic text only. Converting prose into graph records is not presumed to authorize redistribution.

## 7. Typed data schemas

Pydantic models are strict, versioned, and immutable after freeze. All records share a common envelope:

`record_id`, `schema_version`, `record_version`, `scope_id`, optional `work_id`, `bundle_ids`, `status`, `visibility`, `provenance_ids`, `content_hash`, and audit timestamps. `status` is one of `draft`, `reviewed`, `frozen`, `retired`. Audit timestamps are excluded from scientific content hashes. Optional scientific values use explicit tagged states `known`, `unknown`, `unresolved`, `disputed`, or `not_applicable`; a missing JSON member does not silently mean unknown. Every reference must resolve within the bound graph and work unless the schema explicitly authorizes otherwise.

### 7.1 Narrative, evidence, and graph schemas

| Schema | Required fields | Optional fields and controlled values | Core validation and visibility |
|---|---|---|---|
| `Entity` | `entity_id`, `entity_kind`, `identity_status`, internal label | kinds: person, group, place, object, office, other; alias records; licensed alternatives | aliases unique in edition and scope; disputed identities remain distinct; AGOT labels restricted |
| `EventOccurrence` | `event_id`, controlled type, participant-role pairs, scenario, temporal extent, at least one mention anchor | trigger, transition links, occurrence-coreference decisions | same occurrence requires compatible participants, state change, scenario, and time; repeated narration is not a new event |
| `TemporalExtent` | `extent_id`, coordinate, interval kind, start and end bound states | exact rational values, lower/upper bounds, allowed disjunction | event point requires equal bounds; a known state-validity extent must be proper; `unknown_duration` permits equality; coordinate mixing is invalid |
| `State` | `state_id`, bearer, dimension, value, scenario, validity extent | presented status, attribution, dispute group | identity is bearer-dimension-value-validity specific; dimensions are location, possession, role, allegiance, office, relationship, knowledge, survival, goal, other |
| `Transition` | `transition_id`, bearer, dimension, before state, after state, trigger or explicit unknown, assertion and route links | restricted consequence and rarity packet references | before and after have the same bearer/dimension and differ unless a licensed unknown boundary is explicit |
| `Proposition` | `proposition_id`, normalized predicate, ordered typed arguments, scenario | abstract safe gloss, content alternative group | content identity is independent of who asserts it; there is no unrestricted truth boolean |
| `Assertion` | `assertion_id`, proposition, holder/source, epistemic mode, polarity, curator status, discourse/revelation position, evidence routes | nested assertion, validity horizon | modes: presented_as_established, observed, believed, reported, rumored, remembered, dreamed, prophesied, inferred, denied, refuted, disputed, hypothetical, unknown; status: supported_under_policy, refuted_under_policy, both, undetermined |
| `EvidenceAnchor` | `anchor_id`, work version, opaque span ID, discourse position, rights status, provenance | private locator, normalization version, HMAC algorithm/key ID/value, byte length; public text only for public works | AGOT plaintext prohibited; locator is edition-valid; an anchor supports an attributed presentation, not world truth |
| `MentionAuditRecord` | `audit_id`, target transition, candidate anchor, packet, classification, contribution, pass, reason code | ambiguity note and duplicate cluster | classifications: same_depiction, same_report, same_recollection, same_summary, consequence_only, same_type_other_occurrence, repeated_relation, ambiguous, nonmatch, post_horizon, duplicate |
| `ProvenanceRecord` | `provenance_id`, activity, responsible agent, input IDs/hashes, protocol/tool version, derivation type | safe note and issue ID | version derivation acyclic; Codex assistance recorded but researcher verification remains required |
| `TemporalConstraint` | `constraint_id`, coordinate, left extent, relation or allowed set, right extent or bound, scenario, evidence/provenance | confidence state and provenance-group IDs | supported predicates and endpoints follow Section 12; `after` normalizes to converse `before`; unsupported Allen enums fail validation |
| `SupportStructure` | `support_id`, query claim or slot, atom IDs, root IDs, route IDs, support role, sufficiency status | delete-one minimality record | roles: method_visible_rule, gold_reference, posthoc_accepted, decoy; gold roles are evaluation-only |
| `SelectableRoot` | `root_id`, root kind, payload, base atom IDs, method-visible features, redundancy class, display template | route class references and baseline-specific visibility flags | serialized method view cannot contain target, consequence, rarity, `GoldSup`, final relevance, or diagnostic fields |
| `RootDependency` | source root/atom, target atom, type, mandatory flag, horizon-nonincreasing claim | alternative group and provenance | types: identity, participant_role, before_state, after_state, proposition, assertion_status, evidence, provenance, temporal, support, display; finite positive dependency graph |
| `EvidenceAlternative` | `route_id`, supported root/claim, anchors, added atoms, epistemic signature, temporal signature, response signature, cost vector, licensing state | incompatibilities and restricted rationale | alternatives must support the same typed claim; non-equivalent claims become separate roots |
| `CanonicalEvidenceRoutePolicy` | version, eligibility rule, equivalence signature, dominance rule, ordered tie key | no corpus outcomes | policy and choices freeze before target sealing; opaque root-ID order is prohibited |
| `AnnotationAlternative` | group, target record/field, curator-primary value, licensed values, evidence/rationale, freeze state | reconciliation note | at most three outcome-relevant groups per query and two values per group |

The epistemic fields are deliberately minimal. An anchor can show that the text presents a report without making the reported proposition curator-established. `presented_as_established` records narrative presentation under the declared horizon, not metaphysical truth. Unreliable narration, mistaken observation, rumor, memory distortion, and competing assertions remain attributed alternatives. Deep arbitrary attitude nesting is deferred, but one assertion may cite another as its content to represent a report of a report.

### 7.2 Query, pool, projection, and execution schemas

| Schema | Required fields | Optional or enumerated fields | Core validation and visibility |
|---|---|---|---|
| `TypedQueryContext` | ID, split, safe natural-language or abstract query, task schema, focal entities/types, horizon, requested budget, response schema | story window, discourse window, holder/viewpoint, ambiguity alternatives; primary/secondary/probe flags | rich evaluation record; only an explicit projection of permitted fields may reach the pool worker |
| `PoolQueryView` | ID, safe query text/task, focal entities/types, story/discourse windows, holder, horizon, response-schema reference | no split or evaluation flags | exact allowlisted view for candidate generation; target, HRC, diagnostic, split, relevance, rarity, consequence, and outcome fields are impossible by schema |
| `ResponseSchema` | ID, task type, named typed slots, cardinalities, qualification and evidence rules, `GuardSup` version | licensed unknown/disputed forms and abstention classes | contains no target root IDs; four task types are state change, temporal relation, epistemic distinction, and evidence-backed explanation |
| `CandidatePool` | ID, context, generator version, eligible-universe hash, 18 root IDs, seed, method-score trace, `structural_pressure_statistics`, access-ledger hash, content hash | separate optional challenge metadata | structural statistics contain only counts, density, components, and budget ratios; outcome-aware pressure appears only after evaluation join |
| `BudgetVector` | ID, level, nine positive integer caps, grammar version, derivation and hash | sensitivity parent | low no greater than medium no greater than high componentwise |
| `Projection` | ID, run, root mask and IDs, canonical route signature, closed-atom hash, response fills, realized cost, objective tuple, verifier IDs, status | display record and certificate | selected roots are in pool; closure, costs, and verdicts are recomputed rather than trusted |
| `AbstentionCertificate` | ID, class, graph/context/pool/budget hashes, exhaustive-table hash, machine witness, validation state | public redacted explanation and Pareto deficits | classes: INVALID_CONTEXT, QUERY_AMBIGUOUS, CONTENT_ABSENT, CONTRACT_BLOCKED, BUDGET_BLOCKED, ERROR_OR_UNKNOWN |
| `CorruptionCase` | ID, split, base hash, deterministic mutation delta, clean/corrupt flag, isolation mode, target guards, expected full signature, severity, hash | multi-guard rationale | frozen case cannot change after verifier output |
| `ExperimentManifest` | ID, design, graph, context, canonical-route, target, worker-output, pool, availability, budget, method, verifier, solver, corruption, ERIPCF-instance and environment hashes; seeds; splits | deviation ledger | final binding manifest is created only after independent seals; append-only |
| `RunRecord` | ID, manifest, method and parameter hashes, context, pool, budget, timing, peak memory, status, result/certificate IDs, log hashes | retry lineage and machine description | retries append, never overwrite; deterministic repeats link to original |
| `MetricRecord` | ID, metric/version, run/context/bundle/slice, numerator, denominator, value, scoring policy, input hashes | licensed interpretation ID | scoring policies: curator_primary, conservative, permissive; aggregates retain source IDs |

### 7.3 Sealing, storage, and access-control schemas

| Schema | Purpose and mandatory content |
|---|---|
| `DesignFreezeManifest` | pre-held-out ontology, handbook, schemas, code interface, generator, routes policy, query templates, budgets, baseline grid, verifier semantics, corruption templates, seeds, source-document hashes |
| `HeldoutMethodGraphManifest` | hashes of sanitized held-out method graphs and their immutable root/atom catalogs |
| `ContextSealManifest` | every held-out typed context and response schema, without target IDs |
| `CanonicalRouteSealManifest` | policy hash and every context-root canonical route choice, compiled from method-only graphs and contexts before target sealing |
| `TargetGoldSealManifest` | restricted HRC target mappings, transition bundles, initial `GoldSup`, rarity, consequence, diagnostics, alternatives, prior manifest hashes |
| `PoolInputCapsuleManifest` | exact allowlisted files and hashes readable by the isolated pool worker; explicitly no target-seal digest |
| `PoolWorkerOutputManifest` | worker-produced pools, traces, capsule/code/seed hashes, access log and sandbox attestation; references no target artifact |
| `PoolSealManifest` | immutable worker output and capsule hashes plus a controller-added prior target commitment and append-only ledger position; target reference is chronology metadata added only after worker exit |
| `AvailabilityAssessmentManifest` | both independent seal hashes, TAR, canonical SAR, any-route SAR, pressure findings, ready/failure state; no write authority upstream |
| `GoldSupportExtensionManifest` | one masked post-hoc scoring extension, accepted/rejected novel paths, original seal reference; cannot alter pools or runs |
| `ERIPCFInstance` | context, incidence graph, query seeds, connector bundles/costs, delta, rho, medium-budget and rule hashes; freezes before scoring and is referenced by the final experiment manifest |
| `CacheRecord` | canonical cache key, value hash, scientific version hashes, result pointer, collision-verification bytes |
| `CheckpointRecord` | context, enumeration mode, mask range, expected/committed rows, batch digest, status, manifest hash |

The storage API exposes `OracleBundle`, `MethodBundle`, `EvaluationGold`, `PoolInputCapsule`, and `ProjectionResult` as noninterchangeable containers. Selector signatures accept only a method bundle, typed context, budget, pool, and frozen method configuration. Candidate-generator signatures accept only a pool capsule. Evaluation-only modules are the sole readers of `EvaluationGold`. Static imports, filesystem allowlists, Pydantic extra-field rejection, and runtime access ledgers enforce these boundaries.

## 8. Identifier, serialization, and versioning strategy

Internal IDs use UUIDv5 over a project namespace and canonical tuple `(artifact_type, work_version_id, bundle_id, opaque_local_key)`. The local key comes from a ledger and contains no name, event label, outcome, spoiler, rarity class, or diagnostic category. Public release IDs use a second namespace over the internal ID; the mapping remains restricted for AGOT. Root bit positions are the sorted opaque root IDs in the sealed pool. IDs do not depend on file order.

Canonical JSON is UTF-8 with sorted keys, normalized line endings, fixed enumeration strings, arrays sorted only where the schema declares order irrelevant, and rational values serialized as reduced `numerator/denominator` strings. Scientific hashes exclude their own hash field, signatures, filesystem paths, wall-clock timestamps, and nonsemantic notes. SHA-256 binds records, record sets, graphs, contexts, routes, pools, budgets, tables, manifests, and results. Hash hits are verified against canonical bytes rather than trusted solely by digest.

Versions have distinct meanings: `schema_version` changes field semantics; `record_version` changes an instance; `handbook_version` changes annotation rules; `graph_version` freezes records; and `route_policy_version`, `closure_version`, `temporal_semantics_version`, `verifier_version`, `objective_version`, `metric_version`, and `manifest_version` freeze scientific behavior. A frozen record is never edited in place. A schema migration preserves the source bytes, supplies a deterministic mapping, and demonstrates semantic equivalence. A scientific change is a protocol deviation and requires a complete affected rerun from the immutable original.

## 9. Annotation and graph-freeze workflow

No annotation application is built. The researcher uses structured YAML worksheets and CSV audit ledgers, with immediate schema and graph validation and canonical JSONL compilation. This is faster and more inspectable than a custom interface for 150 roots.

The default corpus is fixed as follows:

| Bundle role | Count | Default source | Focal text | Selectable roots |
|---|---:|---|---:|---:|
| AGOT held-out | 4 | one episode bundle from each discourse quartile of *A Game of Thrones* | 1,800 to 2,400 words plus at most 600 words of distant prerequisites | 25 each |
| Public development | 1 | a suitable episode from Wilkie Collins's *The Moonstone* | same bounds | 25 |
| Public held-out | 1 | a suitable episode from Arthur Conan Doyle's *The Hound of the Baskervilles* | same bounds | 25 |
| Hidden control | 1 | researcher-authored, unpublished until unsealing | comparable episode length | at most 18 |

The public held-out source is not opened for task-specific inspection until the design and public-development pilot are frozen. A public title may be replaced before design freeze only for an independently documented rights or episode-suitability failure; after freeze it cannot be substituted.

### 9.1 Annotation pass order

For each natural bundle the researcher performs:

1. Freeze the focal boundary under the discourse-stratified eligibility rule and add only prerequisite passages necessary for one declared competency question.
2. Record evidence anchors and mention spans without interpreting event identity.
3. Resolve entities and occurrence-coreference decisions, preserving unresolved alternatives.
4. Annotate events and participant roles.
5. Annotate states and explicit transitions.
6. Create propositions and attributed assertions with minimal epistemic fields.
7. Add story-time and state-validity constraints separately from discourse and revelation positions.
8. Add evidence, provenance, identity, support, and display dependencies.
9. Define and validate licensed evidence-route alternatives and the global canonical-route policy; do not choose a context-specific route yet.
10. Draft response schemas, competency questions, and initial support structures in the appropriate method-visible or gold container.
11. Run schema, referential, visibility, temporal, closure, rights, and 25-root-count validation.

The compiler never invents a value to make a record validate. Issues use `unknown`, `unresolved`, or `disputed` and a versioned issue ledger. Interpretive fields are visibly distinct from source-verifiable anchors, participants, and discourse positions.

### 9.2 Delayed intra-rater stability

After at least 14 full days, exactly 24 of the 150 natural roots are reannotated from masked, randomly ordered packets. The stratified set includes at least six transitions, six assertions, every natural bundle, all major epistemic modes, and all primary rare-consequence candidates if there are fewer than six. It omits first-pass labels and stable IDs and uses fresh review IDs. The sample and random seed freeze before export.

The second pass computes entity and event coreference F1, evidence-anchor overlap F1, categorical or weighted kappa for state/epistemic fields, temporal-relation agreement, support-atom F1, and route-choice agreement. Reconciliation preserves first pass, second pass, adopted interpretation, evidence, and handbook rule. The separate horizon re-audit covers eight audited transitions after the same 14-day minimum.

### 9.3 Freeze and correction policy

The held-out method graph becomes immutable after its schema, graph, temporal, visibility, route-license, and root-count checks pass and its digest enters `HeldoutMethodGraphManifest`. Contexts freeze next. The method-only route compiler then writes `CanonicalRouteSealManifest`. Targets and initial scoring gold freeze in restricted storage next. Pools freeze separately afterward. No older rule that waits to freeze graph records until pools are created remains valid.

After freeze:

- a factual correction fixes a demonstrable transcription or locator error in a new version, preserves the original, logs the deviation, and reruns every affected result;
- a schema migration must prove semantic equivalence and retain both serializations;
- an interpretation alternative affects sensitivity only if licensed before selector execution;
- a post-result interpretation is reported separately and cannot replace confirmatory gold;
- the sole allowed support-family extension is the one masked, method-blind novel-path review described later;
- every Codex-assisted transformation receives a human diff, validation-summary review, and logged source check.

## 10. Horizon-rarity audit workflow

The tooling retrieves candidate passages; only the researcher classifies and counts them. At most 32 master transitions are audited, including all 10 HRC targets and diagnostic contrasts. Eight are completely re-audited after at least 14 days. No full-book ontology is required.

Each `MentionSearchPacket` contains an opaque transition and work version; canonical names and aliases; abstract event descriptions and paraphrases; role-sensitive participant combinations; state dimension and before/after values; likely source/speaker combinations; lexical stems, spelling variants, and exclusions; horizon discourse position; permitted local files; and tool/version hashes. Local exact, normalized lexical, participant-proximity, and lemmatized searches are required. A local semantic retriever may propose candidates only if frozen during development; its output never determines a count.

A candidate co-refers with the same occurrence only when story-time compatibility, core participants and roles, scenario, state dimension, and before/after values jointly support identity. The ledger distinguishes repeated depiction, report, recollection, or summary of the same occurrence from a consequence, another event of the same type, a repeated relation between the same participants, and an overlapping search duplicate. The first four classifications contribute one per distinct discourse anchor. Consequences, other occurrences, repeated relations, duplicates, and post-horizon hits contribute zero. Ambiguous hits remain licensed alternatives rather than forced matches.

`HorizonCount` sums verified co-referring anchors at or before the inclusive query horizon. `LocalCount` applies the same decisions only inside the focal episode. Primary categories are singleton (`HorizonCount=1`), intermediate (`=2`), and repeated (`>=3`). Local categories are sensitivity labels. The audit ledger records every query form, hit, duplicate cluster, classification, rationale code, and horizon decision. Post-horizon hits remain restricted audit rows with zero contribution.

The delayed-audit gate is mention-decision F1 at least 0.85 and identical rarity category for at least seven of eight targets. Failure withdraws the broad horizon-rarity wording and narrows the contribution to locally singleton or query-necessary support structures. AGOT indexes, windows, aliases, and prose remain private. Public release includes schemas, safe abstractions, counts, opaque anchors, and full public-domain ledgers.

## 11. Candidate-pool construction and independent availability

### 11.1 Primary pool

For each answerable context, the isolated worker receives exactly one sealed capsule and its allowlisted `PoolQueryView`; it cannot deserialize a full `TypedQueryContext`. It:

1. retains same-work selectable roots whose root and visible mandatory dependencies are at or before the horizon;
2. rejects incompatible scenario scope or unresolved identity under method-visible rules;
3. computes the frozen coarse query score from only focal entities, requested types, story/discourse window, holder, and response-schema compatibility;
4. selects the six highest scores, using sorted opaque IDs for ties;
5. fills to 18 by the hash order of `(global_seed, context_id, root_id)` over the remaining universe;
6. performs only the frozen outcome-blind root-kind exchange if event, transition, or assertion roots are absent;
7. writes the ordered roots, score trace, eligible-universe hash, seed, canonical route IDs, and structural statistics.

If fewer than 18 roots are eligible, generation fails. A root from another already annotated bundle of the same work may be used if it is horizon-safe. Cross-work roots are prohibited. No method-specific pool is allowed.

The generator reports, but does not optimize against, root count, component counts, root-incidence edges, edge density, connected components, structural centrality distribution, and each budget-to-pool ratio. Outcome-aware pressure checks occur only after the two seals exist. A primary context should have a pool-to-medium-root-cap ratio of exactly 3:1, at least nine roots later scored irrelevant or contextual, at least one frequent distractor, one rare nonconsequential structure, one competing incomplete response, one structurally central but irrelevant root, and no target identifiable only by a unique event type or identifier. Principal semantic components must exceed corresponding medium caps by at least 3:1 where the eligible graph contains enough components. Failure is reported; it is never repaired.

### 11.2 Optional controlled challenge pool

Only after the required study is complete may a separate diagnostic pool apply outcome-aware quotas for rare nonconsequential, frequent weak, unrelated transitions, repeated assertions, central irrelevant, incomplete competing support, and focal structures. Category precedence is fixed in that order, each root appears once, and every root is a genuine horizon-safe annotated same-work structure. The challenge pool cannot substitute for a failed primary pool and is the first analysis dropped under time pressure.

### 11.3 Postseal availability

Let `R_q` be the independently sealed pool, `t_q` the independently sealed HRC target, `eta_star` the frozen canonical routes, and `GoldSup_q` the initial acceptable support family. Only the evaluation join computes:

$$
TargetAvailable_q=\mathbf 1[t_q\in R_q],
$$

$$
CanonicalSupportAvailable_q=\mathbf 1[\exists Q\in GoldSup_q:Q\subseteq cl_C(R_q,\eta^*)],
$$

and the diagnostic upper bound

$$
AnyRouteSupportAvailable_q=\mathbf 1[\exists Q\in GoldSup_q,\eta\in Licensed_q:Q\subseteq cl_C(R_q,\eta)].
$$

`TargetAvailabilityRecall`, `CanonicalSupportAvailabilityRecall`, and `AnyRouteSupportAvailabilityRecall` average these indicators over a declared slice. The primary construction gate requires target and canonical support availability of 1.00 on all 10 HRC contexts. If any-route availability passes while canonical availability fails, the result is specifically a route-policy benchmark-construction failure. The pool remains unchanged.

`GoldSup` refers to evidence-equivalence requirements and licensed route classes, not to one accidental anchor ID. For the any-route diagnostic, inspect only route classes for roots and evidence slots occurring in the candidate `GoldSup` structure; do not form a Cartesian product over all 18 roots. Dominated but licensed classes remain eligible because this is an upper bound. Enumerate exactly when the relevant product is at most 64. Above 64, solve the finite route-existence question with an exact Boolean satisfiability encoding and a deterministic witness; never sample. A solver error yields availability `ERROR`, not a guessed result.

### 11.4 Leakage-audit checklist

Before a pool seal is accepted, automated and human audits confirm that the worker neither reads nor derives:

- target IDs or `GoldSup` membership;
- consequence or rarity values;
- diagnostic-cell or HRC membership;
- final relevance grades;
- selector scores, projections, or method identity;
- held-out aggregate outcomes;
- target-seal hash or restricted filesystem metadata.

The worker's mount list, file-open ledger, capsule schema validation, process command, environment, network isolation, and output hashes are archived. A mutation test changes every gold label while keeping the capsule fixed and requires identical pool bytes. This is a controlled dense graph assembled from bounded annotation, not a complete novel graph.

## 12. Temporal satisfiability implementation

### 12.1 Coordinates and interval convention

Paper 1 uses one explicit interval convention for semantic temporal reasoning. Story-time event extents and state-validity extents are **closed rational intervals** `[s,e]` with `s <= e`. An event point has `s = e`; a proper event interval has `s < e`; an unknown-duration event asserts only `s <= e` plus recorded bounds. A known state-validity extent MUST be proper with `s < e`; `unknown_duration` permits only `s <= e`; a point state is invalid. This choice makes endpoint sharing observable and avoids silently mixing half-open state intervals with closed event intervals. If future work needs instantaneous state records or half-open database-validity semantics, it will be a new temporal-semantics version, not an implementation optimization.

Discourse positions, revelation positions, and spoiler horizons are separate integer coordinates rather than temporal intervals. Horizon eligibility is inclusive: position `p` is visible exactly when `p <= H`. Story time never determines horizon eligibility. Extraction and curation timestamps are provenance metadata and never constrain story time.

The supported relation fragment is:

$$
\begin{aligned}
before(i,j)&: e_i<s_j,\\
after(i,j)&: before(j,i),\\
equal(i,j)&:s_i=s_j\land e_i=e_j,\\
overlaps(i,j)&:s_i<s_j<e_i<e_j,\\
during(i,j)&:s_j<s_i\land e_i<e_j,\\
contains(i,j)&:during(j,i),\\
intersects(i,j)&:s_i\le e_j\land s_j\le e_i.
\end{aligned}
$$

`overlaps` is directional. Input alias `overlapped_by(i,j)` normalizes to canonical `overlaps(j,i)` and is never stored as a separate predicate. `intersects` is symmetric and coarse. Two closed intervals sharing exactly one endpoint intersect, but they are not `before`, `overlaps`, `during`, `contains`, or `equal`. This boundary pattern corresponds to Allen `meets` or `met-by`, which Paper 1 does not name. Likewise `starts`, `started-by`, `finishes`, and `finished-by` are unsupported names. Same-start or same-end unequal intervals entail `intersects`, not strict `during`, `contains`, or `equal`.

An input annotation or query using an unsupported relation name is a schema validation error. It is never mapped silently to `unknown`, `equal`, or `before`. Exact endpoint bounds may imply an unsupported Allen boundary pattern without error, but the response can state only an entailed supported relation, often `intersects`, or a declared unresolved allowed set. Natural-language "overlap" must be mapped deliberately to directional `overlaps` or coarse `intersects`. `unknown` contributes no relation constraint. An allowed disjunction is a normalized logical `Or` over supported predicates.

Point-boundary behavior is explicit for events. A point at the start or end of a proper interval intersects it but is not strictly during it. A point strictly inside is both during and intersecting. Equal points satisfy `equal` and `intersects`. `before` always implies nonintersection. Point state-validity extents are rejected as stated above.

Typed story windows are closed `[a,b]`; discourse windows are inclusive integer ranges `[d_0,d_1]`. Candidate filtering excludes a root for a story window only when the temporal theory entails that its extent is disjoint from the window. A temporally unresolved root remains eligible but receives no window-match relevance point. A response that asserts occurrence in the window must entail intersection; when its response schema specifically asks for full containment, it must entail containment. Discourse eligibility and match use the explicit inclusive range. Tests cover exact starts/ends, unknown bounds, entailed disjointness, possible intersection, and required containment.

Flashbacks have earlier story intervals and later discourse anchors. A memory, report, rumor, dream, prophecy, or hypothetical is an assertion in its own epistemic mode and scenario; its discourse position does not establish the asserted event. Dream, prophecy, and hypothetical scenarios do not share constraints with `actual_story` unless a typed cross-scenario relation is explicitly annotated. Repeated narration creates additional mentions or assertions around the same occurrence, not a second occurrence by default.

### 12.2 Solver translation and entailment

The temporal compiler assigns rational endpoint variables, emits interval-kind axioms and exact/bounded observations, normalizes `after` into reversed `before` and `overlapped_by` into reversed `overlaps`, canonicalizes symmetric operands, sorts disjunction members, and partitions constraints by scenario. The solver uses pinned Z3, quantifier-free linear rational arithmetic, no timeout for confirmatory cases, fixed seeds, and disabled parallel solving. All inputs are exact integers or reduced rationals.

`sat` means only that the selected constraints have a model. A displayed relation is **entailed** only when the base set plus the negation of that relation is unsatisfiable. A merely possible relation cannot be displayed as established. If several alternatives remain possible, the output uses a licensed allowed set or `unknown`. `unknown` returned by the solver is operational `ERROR_OR_UNKNOWN`, never a proof of satisfiability or infeasibility. Entailment is not attempted when the base set is inconsistent.

### 12.3 Deterministic inconsistency certificates

The implementation must not minimize an arbitrary solver-selected unsatisfiable core. It operates over the complete relevant constraint set: every normalized constraint in the selected closure plus every premise used by a displayed temporal claim.

First normalize only the declared equivalent forms: inverse aliases, reduced rationals, symmetric operand order, scenario identifiers, duplicate predicates, and ordered disjunctions. Formulas identical under this frozen canonical normalizer are coalesced into one constraint with a sorted list of provenance IDs. No general logical-equivalence prover is assumed. Sort the canonical constraints by their complete semantic serialization hash, using the normalized tuple as a collision tie check.

If the full set is unsatisfiable, initialize `K` to the complete sorted set. Visit constraints in that fixed order. For each `c`, solve `K minus {c}` in a fresh deterministic solver. If the remainder is still unsatisfiable, permanently remove `c`; if satisfiable, retain it; if unknown, return operational error and no certificate. Finally verify that `K` is unsatisfiable and that deleting any one retained member makes it satisfiable. Archive the full-set hash, sorted `K`, every deletion decision, solver/version/settings, and verification results.

The result is a deterministic **deletion-minimal** core: no retained member can be removed while preserving inconsistency. It is not guaranteed to have minimum cardinality. Fixed canonical ordering resolves cases with multiple possible minimal cores.

### 12.4 Temporal tests

Unit fixtures cover unequal and equal points; point-before-point; a point at each proper-interval boundary; a point strictly inside; a gap; a shared endpoint; directional overlap and its converse; equal proper intervals; strict containment; same-start and same-end unequal intervals; unknown duration; allowed disjunctions; unsupported-name rejection; inclusive horizon boundary; and a flashback whose story and discourse orders differ.

Property tests check before/after and during/contains converses; `overlapped_by(i,j) iff overlaps(j,i)`; equal/intersects symmetry; `before -> not intersects`; and agreement between solver output and direct endpoint predicates over a bounded integer domain. Certificate tests cover input-order permutations, repeated processes, solver seeds 0, 1, and 42, equivalent serialization, duplicate formulas with different provenance, and a graph with several possible unsatisfiable cores. The normalized core and deletion trace must be identical across test seeds; complete certificate bytes are compared only under the frozen production seed because audit metadata records that seed. Every emitted core must be unsatisfiable and every single deletion satisfiable.

## 13. Semantic closure and admissibility engine

### 13.1 Roots, atoms, routes, and closure

A **semantic atom** is the smallest canonical record charged or required as one unit by the projection theory, such as an entity identity, event, role participation, state, assertion, evidence cue, provenance witness, or displayed relation. A **selectable root** is a decision-bearing event, transition, assertion, or contextual relation whose selection invokes mandatory semantic dependencies. A **transition bundle** is the transition root plus bearer and dimension, warranted before and after states, trigger when asserted, temporal and epistemic qualifications, provenance, and its canonical evidence route.

For pool roots `R`, root mask `X`, and the frozen canonical route assignment `eta_star`, closure `cl_C(X,eta_star)` is the least fixed point containing each root's base atoms and every mandatory dependency licensed under context `C`. Dependencies include:

- entity identity for every reference;
- event participant and role atoms;
- transition bearer, dimension, before state, after state, trigger when asserted, and qualification;
- assertion proposition, holder/source, mode, polarity, curator status, scenario, and revelation position;
- proposition predicate and argument identities, without an unqualified truth flag;
- evidence-route anchors and support atoms;
- provenance activities and derivation links;
- temporal premises required for displayed temporal statements;
- method-visible `GuardSup` dependencies for displayed answer claims;
- static presentation facts and word-counted phrases.

Dependencies are never trimmed to fit horizon or budget. Shared atoms are charged once by canonical identity. Cycles are permitted; finite positive dependencies guarantee a fixed point. Paper 1 permits the union fast path only when the compiler certifies that every closure rule has exactly one positive antecedent and that no pair-conditioned, response-conditioned, negative, or aggregate atom is synthesized during closure. Under that schema fragment, closure is the bitwise union of precomputed single-root canonical closures. A held-out graph that violates the fragment is rejected; response feasibility remains outside closure. Tests must also prove equality with the reference worklist on every fixture and development graph. A later Horn-rule extension requires a new closure version and reference evaluator, not an unchecked optimization.

### 13.2 Admissibility verifiers

One common guard wrapper runs all verifiers. Each returns `PASS`, `FAIL`, or `ERROR`; violation category; severity `CRITICAL`, `MAJOR`, or `MINOR`; implicated IDs; a concise explanation; machine-readable witnesses; and input/version hashes.

| Verifier | Pass condition | Observable violation |
|---|---|---|
| Evidence grounding | each displayed answer claim has a selected licensed canonical route whose anchors and atoms are in closure and within horizon | missing, mismatched, ineligible, or post-horizon evidence |
| Horizon and label safety | every semantic atom, evidence anchor, revealed identity, and descriptor fact is at or before `H` | fact, identity, anchor, or label crosses horizon |
| Epistemic fidelity | holder, mode, polarity, curator status, and scenario equal the selected assertion; unqualified establishment is policy-licensed | holder loss, polarity change, modal flattening, rumor promotion |
| Temporal validity | base constraints are satisfiable and every displayed specific relation is asserted or entailed | contradiction or merely possible relation displayed as settled |
| Identity safety | each reference uses one licensed identity decision and disputed alternatives remain distinct | unsafe merge, dangling entity, hidden-identity leak |
| Provenance completeness | every displayed atom has an acyclic derivation to annotation/curation and an anchor or explicit curator rule | absent or cyclic provenance |
| Support completeness | each displayed answer claim instantiates the task's method-visible proof schema with all mandatory atoms | missing state, qualification, evidence, link, or response premise |
| Budget compliance | the postclosure and postpresentation cost vector is componentwise within the common cap | any coordinate exceeds its cap |

Critical violations are post-horizon disclosure, unsupported answer claim, rumor-to-fact promotion, mistaken identity, answer-affecting temporal inconsistency, and missing answer support. A harmless rendering defect is not promoted to a semantic violation. Severity is reported by category; no weighted score permits several minor defects to compensate for a critical one.

Verifier order is structural integrity, horizon, identity, evidence, provenance, epistemic, temporal, support, and budget. Production diagnostics may stop early, but confirmatory execution never does: it preserves every detectable category and dependency-induced multi-guard signature. Native outputs are checked by the same components after selection; guards report but do not repair them.

## 14. Admissibility, response feasibility, and certificates

Let `G` be a frozen temporal-epistemic narrative graph, `C` a typed context, `B` a nine-coordinate budget, `X` selected roots, and `S=cl_C(X,eta_star)` with its deterministic presentation mapping. Define:

$$
\mathcal A_{adm}(G,C,B)=\{S:\ Closure\land Horizon\land Evidence\land Epistemic
\land Temporal\land Identity\land Provenance\land GuardSupDisplayed\land Cost(S)\preceq B\}.
$$

`GuardSupDisplayed` requires support for every answer claim actually displayed; it does not require the graph to answer the query. Let `ResponseSchema_C(S)` mean that required response slots, cardinalities, qualifications, and evidence witnesses are filled. Then:

$$
\mathcal A_{resp}(G,C,B)=\{S\in\mathcal A_{adm}(G,C,B):ResponseSchema_C(S)\}.
$$

A safe but nonresponsive graph lies in `A_adm` but not `A_resp`; it must not be presented as an answer. Selection optimizes only over `A_resp`. If that family is empty, the exhaustive evaluator, not a heuristic, assigns one of these classes:

| Class | Exact meaning | Required certificate |
|---|---|---|
| `INVALID_CONTEXT` | typed fields or relation names violate the frozen schema | validation errors and context hash |
| `QUERY_AMBIGUOUS` | multiple licensed parses require materially different response schemas and the context has no declared disambiguation policy | alternatives and the violated uniqueness rule |
| `CONTENT_ABSENT` | no horizon-eligible root/atom family in the full declared 18-root universe can fill a required semantic slot even before nonbudget guards | missing slot/type witness |
| `CONTRACT_BLOCKED` | a response-shaped candidate exists, but every such candidate violates at least one nonbudget semantic guard | failed requirements and candidate witness classes |
| `BUDGET_BLOCKED` | at least one nonbudget-admissible responsive candidate exists, but none fits `B` | Pareto-minimal component deficit vectors and witness masks |
| `ERROR_OR_UNKNOWN` | solver, data, or execution state prevents a proof | structured error; never scored correct |

Additional budget is meaningful only for `BUDGET_BLOCKED`. Its certificate reports all nondominated deficit vectors relative to `B`; it does not call an arbitrary scalar increment minimal. Content absence, ambiguity, contradiction, missing in-horizon evidence, or an unsafe identity cannot be repaired by a "minimal additional budget". A heuristic's failure to return an answer never establishes infeasibility.

Certificates recompute from graph, context, pool, route-policy, closure, response-schema, verifier, and budget hashes. They contain no gold outcome data. The public explanation is a redacted rendering of the machine witness, not an LLM summary.

## 15. Exact enumeration, cache hierarchy, and transient storage

### 15.1 Search space and exactness boundary

Root positions 0 through 17 follow the sealed pool's sorted opaque IDs. A Python integer mask represents each subset. For answerable contexts, enumerate every mask with population count at most the high root cap 8:

$$
\sum_{k=0}^{8}{18\choose k}=106{,}762.
$$

The 32 answerable contexts therefore produce at most 3,416,384 root-mask rows. Eight typed probes enumerate all `2^18 = 262,144` masks, adding 2,097,152 rows. Two invalid-context probes stop at validation. With one canonical route per root, the confirmatory maximum is therefore **5,513,536 unique semantic root-mask rows**, not the predecessor's approximately 22 million mask-route rows. Native ERIPCF additionally needs raw-materialization features only at medium on six development and 10 HRC contexts. Its root cap 6 gives `sum_{k=0}^6 C(18,k) = 31,180`, hence at most 498,880 derived raw rows keyed to those same masks. Optional route sensitivity evaluates only fixed winning masks and is not part of either count.

Every mask row records closure, nonbudget guards, response slots, cost vector, structural features, and objective-ready statistics whether or not it fits high budget. Low, medium, and high are filters of this single table; they never trigger new semantic evaluation. Exactness is claimed only if every expected mask appears exactly once and the winning row is chosen by full exact comparison. An optional performance optimization may prune before materializing only when a formal equivalence test demonstrates that the complete canonical row ledger and winner remain recoverable; the default is full enumeration.

### 15.2 Cache layers and execution frequency

Cache keys are SHA-256 values over canonical scientific bytes, never Python object hashes.

| Layer | Canonical key | Value | Frequency |
|---|---|---|---|
| Graph compilation | graph, schema, dependency-rule, route-policy hashes | atom indices, adjacency, single-root closures, normalized temporal catalog | once per graph version |
| Context compilation | graph, pool, context, response, presentation, verifier hashes | 18 root positions, seeds, feature vectors, budget-independent tables | once per context |
| Root mask | context-compile hash plus `root_mask` | union of root base and mandatory dependency bitsets | once per root mask |
| Selected route | SHA-256 of sorted `(root_id, canonical_route_id)` pairs | route-set signature and route deltas | once per selected root set; primary route is fixed |
| Expanded closure | graph, context-semantics, root mask, route-set hash | atom bitset, closure signature, cost input, temporal and presentation signatures | once per unique mask/route signature |
| Temporal SAT/core | temporal-semantics, solver-config, scenario partition, normalized constraint-bitset hash | SAT state or deterministic core | once per unique temporal bitset |
| Temporal entailment | preceding key plus displayed-relation signature | entailed, not entailed, or error | once per temporal/display pair |
| Response feasibility | response-schema, context-restriction, closure signature | filled slots, witnesses, responsive flag | once per unique closure/schema |
| Nonbudget admissibility | verifier bundle, context guard, closure and presentation signatures | full predicate vector and witnesses | once per unique closure/presentation |
| Budget verdict | cost-calculator, budget, closure, presentation | component vector and pass/fail | once per row per budget, a cheap comparison |
| Root-incidence structure | pool, incidence rule, connector bundle/cost-rule hashes | 18 by 18 rational direct-edge matrix and sorted edge catalog | once per context |
| ERIPCF forest | mode (`closed` or `raw`), incidence-graph, query-seed bitset, root mask, available-edge or materialization signature, rho hash | exact cost, components, canonical MST edges | once per structural key |
| Method score | common-row hash and method-config hash | objective tuple | once per method and row, with no semantic recomputation |

The root-incidence matrix never uses shortest paths through unselected roots. A missing direct declared connection is infinity. Full, the ablation, all wrapped baselines, all budget levels, and the 12 ERIPCF parameter settings reuse the same semantic rows. Selector scoring must produce zero new closure, SAT, entailment, response, verifier, or forest records.

### 15.3 SQLite inner-loop store

Use one local SQLite database per context and final manifest, through Python's pinned standard-library `sqlite3`. SQLite is selected over compressed JSONL because millions of rows need indexed filters, uniqueness, transactions, checkpoints, and interruption recovery. It is selected over Parquet because the workload needs keyed cache lookups and incremental atomic commits, not only scans. It introduces no server. SQLite is replaceable engineering; its page-level byte hash is not a scientific artifact.

Required `STRICT` tables are:

- `enumeration_meta(key PRIMARY KEY, canonical_value_json)`;
- `canonical_route(root_id PRIMARY KEY, route_id, route_set_hash, policy_hash)`;
- `closure_cache(closure_signature PRIMARY KEY, atom_bitset BLOB, route_set_hash, temporal_signature, presentation_signature, nine cost columns, row_hash)`;
- `temporal_cache(temporal_signature PRIMARY KEY, constraint_bitset BLOB, status, deterministic_core_json, result_hash)`;
- `temporal_entailment(temporal_signature, relation_signature, status, result_hash, PRIMARY KEY(temporal_signature, relation_signature))`;
- `verifier_cache(verifier_key PRIMARY KEY, predicate_bits, severity_bits, result_json, result_hash)`;
- `response_cache(response_key PRIMARY KEY, slot_bits BLOB, response_feasible, witness_json, result_hash)`;
- `forest_cache(forest_key PRIMARY KEY, root_mask INTEGER, connector_edge_bitset BLOB, cost_numerator INTEGER, cost_denominator INTEGER, component_count INTEGER, sorted_edges_json, result_hash)`;
- `mask_evaluation(context_hash, root_mask INTEGER, route_set_hash, root_count, closure_signature, semantic_admissible, response_feasible, nine integer cost columns, frozen feature columns, forest_key, row_hash, PRIMARY KEY(context_hash, root_mask, route_set_hash)) WITHOUT ROWID`;
- `native_mask_evaluation(context_hash, root_mask INTEGER, raw_materialization_hash, nine raw cost columns, native_forest_key, raw_cap_flags, row_hash, PRIMARY KEY(context_hash, root_mask)) WITHOUT ROWID`;
- `progress(context_hash, mode, batch_start, batch_end, expected_rows, committed_rows, batch_digest, status, PRIMARY KEY(context_hash, mode, batch_start))`.

Foreign keys link mask rows to cache records. Scientific cache keys and row hashes are unique. Indexes cover `(context_hash,response_feasible,semantic_admissible,root_count)`, closure and temporal signatures, forest keys, and the ordered budget scan beginning with context, response flag, root count, and the nine cost coordinates. The exact index list freezes after the performance pilot, but indexes may change later if canonical exports and winners remain identical.

Enumeration uses WAL, `foreign_keys=ON`, `synchronous=FULL`, and one writer per context. Each transaction covers exactly 4,096 consecutive numeric masks, all induced cache inserts, and one completed checkpoint row. `INSERT OR ABORT` prevents silent duplicates. On resume, the process verifies manifest, schema, route-policy, context, and code hashes; validates counts and batch digests for committed ranges; and restarts the first absent range. At most one uncommitted batch is lost. Finalization performs integrity and foreign-key checks, checkpoints and truncates WAL, and exports with explicit `ORDER BY root_mask, route_set_hash` plus stable subordinate order.

Canonical results, projections, certificates, manifests, audit ledgers, and release data remain JSONL. An ordered SQLite scan streams every inner-loop row into an enumeration digest and count, but it does not persist a second full 5.5-million-row JSONL copy by default. Only winners, certificates, declared audit samples, metrics, and manifests become canonical JSONL. A full row export is optional, restricted, and counts against the performance-pilot storage ceiling. JSONL uses canonical keys, reduced rationals, and no semantic timestamps. The ordered digest and row count enter the manifest; SQLite page bytes do not. Common features are stored once, and millions of redundant per-method scores are not retained.

### 15.4 Exact search algorithm and tests

For each allowed mask, the runner obtains the precomputed root union, adds canonical route dependencies, computes or retrieves expanded closure, retrieves temporal and other guard results, fills response slots, computes exact costs, and writes one row. For contexts requiring native ERIPCF, it also computes the canonical raw forest, materializes its connector bundles, and writes raw costs and a mode-separated forest key to `native_mask_evaluation`. This adds derived raw records for the six development and 10 HRC medium contexts, not new root masks. Budget prechecks may avoid expensive downstream checks only when the row still records the logically implied result and equivalence to the reference order is tested. Evidence alternatives are absent from the primary inner loop because the canonical route is fixed.

Within each method and budget, candidates are compared by exact objective tuples and deterministic ties. If no response-feasible row exists, the certificate engine analyzes the complete table. Runtime, CPU, peak memory, cache hits and misses, solver calls, forest calls, transaction timing, database size, and export hashes are logged.

Correctness tests compare against naive enumeration on pools of at most 10 roots, known-optimum fixtures, and an independent integer or exhaustive formulation on tiny cases. Property tests cover deterministic repeats, stable input permutations, budget-family monotonicity, closure equality, and complete mask counts. Infeasible fixtures cover every certificate class. No report may say "zero optimality gap" unless the mask ledger is complete and both reference checks pass.

### 15.5 Mandatory one-context performance pilot

Before held-out annotation, select the public-development context with the greatest method-visible sum of dependency count, temporal-constraint count, and incidence-edge count, with stable-ID ties. Its pool must contain 18 roots. Run:

1. a cold cap-8 enumeration of 106,762 masks;
2. a cold unrestricted stress pass of 262,144 masks on the same context;
3. the medium-cap native raw-materialization table and every selector plus the ERIPCF development grid from persisted rows;
4. two warm repeats;
5. forced interruption after at least two batches followed by resume.

Measure wall and CPU time by closure, Z3 SAT, entailment, verifier, response, closed/raw forest, SQLite, and export; masks and unique closures per second; peak RSS; database, WAL, and export bytes; unique closure and temporal signatures; cache hits/misses by layer; Z3 and MST calls; raw-materialization work; transaction and checkpoint time; recomputed masks after interruption; and cold/warm output hashes. A warm run uses a fresh progress/output namespace while retaining immutable content-addressed caches, so its denominator is actual cacheable key requests rather than skipped completed batches. The interruption test uses a separate fresh database. Structural expectations are exact: a unique key is computed once; low/medium and selector/grid passes create no new semantic cache entries; warm key hits are at least 99.9 percent with zero new closure/Z3/entailment/forest values; resume recomputes at most one batch; and every canonical output hash matches.

Resource acceptance targets, not advance runtime claims, are: cap-8 cold pass at most 30 wall minutes; unrestricted pass at most 75 minutes; selector and grid scoring at most 2 minutes; peak RSS at most 4 GiB; cap-8 database at most 512 MiB; unrestricted database at most 1.25 GiB. Extrapolate total planned compute from observed time per mask, unique closure, temporal signature, and forest key, using the actual context mix and a 1.5 safety factor. The forecast must be no more than 48 CPU-hours and 24 GiB of simultaneous transient storage.

A correctness mismatch, nondeterministic hash, missing mask, or invalid resume is an immediate stop. Cap-8 above 45 minutes, unrestricted above 120 minutes, RSS above 8 GiB, per-context database above 2 GiB, projected storage above 30 GiB, or safety-adjusted compute above 48 CPU-hours triggers one engineering-only optimization cycle. Permitted changes are bitsets, precomputed single-root closure, memoization, incremental subset sums, transaction batching, SQLite indexes/pragmas consistent with durability, streaming export, and process-level parallelism across contexts. Pools, routes, guards, budgets, objectives, exact mask coverage, and ties may not change. If limits still fail, remove optional challenge, paraphrase, native-secondary, and route-sensitivity work or revise nonprimary context scope before held-out opening. A heuristic cannot replace confirmatory exact search, and a timeout is never infeasibility.

No 6-to-30 CPU-hour or other full-run estimate is claimed before this pilot reports empirical evidence and machine specifications.

## 16. Exact Root-Incidence Prize-Collecting Forest baseline

### 16.1 Restricted structural object

For context `C`, let `H_C=(R_C,E_C)` be the frozen undirected root-incidence graph over exactly the 18 candidate roots. An edge exists only for a declared, method-visible semantic connection, such as a shared participant, explicit event-state link, causal/support relation, or temporal/state-chain adjacency. Every edge has a frozen connector bundle `D_e`. No unselected root, latent entity, shortest path through another root, or arbitrary Steiner vertex may be introduced.

For a common closed row with selected roots `X`, edge `e` is available only if both endpoints are in `X` and every atom in `D_e` is already in the row's closure. Connector atoms are charged once by the common cost calculator. ERIPCF does not receive free semantic material. In native raw mode, an edge is eligible only when its endpoints and connector bundle are root-level horizon-safe; the canonical forest is chosen and its connector bundle union is then materialized and charged against raw caps. A cap failure makes that mask unavailable rather than causing a different unmanifested forest search. Raw and closed modes have separate cache keys. If a future formulation allows a wrapped forest edge to add new atoms, those atoms must first be added to the common row and charged for every method; that would be a new manifested baseline version.

Using the medium budget, define the exact rational connector score

$$
c_C(e)=\sum_{i\ne root}{cost_i(D_e)\over B_i^{med}}.
$$

The activation penalty `rho_C` follows Section 3.4. Add a scoring-only virtual vertex `q_0`. It has a zero-cost edge to each method-visible query-seed root and an edge of cost `rho_C` to every other selected root. Virtual edges are neither semantic atoms nor display items.

For root set `X`, let `A_C(X)` be the augmented graph induced by `X union {q_0}` using available semantic edges and the virtual edges. Define

$$
Conn_{\rho_C}(X)=\min_{T\text{ spans }A_C(X)}\sum_{e\in T}w(e).
$$

The minimum is the exact MST cost. Equivalently, the semantic edges form a forest that pays `rho_C` for each component with no query seed. Components containing different query seeds may remain semantically disconnected because `q_0` represents only their common query context. If there is no query seed, each retained semantic component requires one activation. A long semantic path can cost more than an activation even though every individual semantic edge costs less than `rho_C`.

This is related to prize-collecting Steiner formulations because it trades root prizes against connection cost. It is not unrestricted PCST or prize-collecting Steiner forest: all root subsets are enumerated; the graph contains only declared candidate roots; arbitrary Steiner vertices are absent; slot utility is nonadditive; and semantic closure, response feasibility, and vector budgets sit outside the forest objective. A result against guard-ERIPCF supports only a claim against this reproducible compact structural comparator.

### 16.2 Prize function, parameters, and exact ties

Let `rhat_C(r)` be frozen method-visible relevance and `Slots_C(r)` the method-visible response slots a root can help fill. ERIPCF scores a common row by

$$
J_{ERIPCF}(X)=
\lambda_r\sum_{r\in X}\widehat r_C(r)
+\lambda_s\left|\bigcup_{r\in X}Slots_C(r)\right|
-\lambda_c Conn_{\rho_C}(X).
$$

Evidence-only atoms receive no prize. Event, transition, assertion, and contextual-relation roots use the same generic rule. There is no transition-completeness bonus. Development tuning uses exactly 12 tuples:

$$
\lambda_r\in\{1,2\},\quad
\lambda_s\in\{1,2\},\quad
\lambda_c\in\{1/2,1,2\}.
$$

For a row at requested budget `B`, normalized realized cost is

$$
J_{cost}(S,B)=\sum_i {cost_i(S)\over B_i}.
$$

At medium budget on all six public-development contexts, choose the parameter tuple lexicographically by: number responsive; number with a complete accepted development support; summed frozen development relevance; smaller exact aggregate `sum_{C in Dev} J_cost(S_{theta,C},B^med)`; then lexical parameter tuple. This tuning and the deterministic rho rule freeze before held-out data are opened.

Kruskal's algorithm compares exact rational weights and breaks edge ties by weight, numeric edge-kind rank (`query_seed=0`, `semantic=1`, `activation=2`), sorted endpoint IDs, and connector-bundle hash. Candidate ties are resolved by larger ERIPCF score, larger slot-union coverage, larger relevance sum, lower connector cost, lower `J_cost(S,B)` at the requested budget, fewer roots, then lexical root and route vector. Thus identical frozen input yields the same forest certificate and winner.

Guard-ERIPCF scores only common response-feasible rows. Native ERIPCF enumerates horizon-filtered raw root subsets, constructs the same restricted forest, materializes its declared connector bundles, charges raw component caps, and is then evaluated by the common guards without repair. Its raw cache key and table are distinct from closed rows. The word `exact` refers to exhaustive root-subset comparison and exact MST cost in this restricted object.

Before any selector score is computed, an `ERIPCFInstance` freezes the context's incidence graph, query-seed set, connector bundles and rational costs, delta, rho, edge ordering, and medium-budget/rule hashes. The final experiment manifest references every instance digest. A transient cache alone is not a scientific freeze artifact.

Tests include no, one, and multiple query seeds; empty semantic-edge graph; line, star, cycle, and disconnected graphs; two unseeded components; a direct edge cheaper than rho; a path more expensive than activation; exact `delta` and rho; equal-cost MSTs; rational comparison; input-order invariance; connector materialization and charging; absence of hidden Steiner roots; gold-mutation invariance; and agreement with exhaustive edge-subset forest enumeration on tiny graphs.

## 17. Proposed objective and direct ablation

The context-visible relevance of root `r` is

$$
\widehat r_C(r)=\min(3,I_F(r)+I_K(r)+I_W(r)),
$$

where `I_F` indicates a focal-entity match, `I_K` task-compatible root/event type or response slot, and `I_W` compatibility with all specified story/discourse/holder restrictions. An unspecified restriction contributes neither a match nor a penalty. The feature trace is archived. Outcome relevance is separate gold and cannot be imported.

Let `Z_C` be method-visible candidate transition bundles, `K_C` the frozen optional and mandatory response-slot set, and `roots(S)` selected roots. Define:

$$
\begin{aligned}
J_{TB}(S,C)&=\sum_{z\in Z_C}\widehat r_C(z)I[z\subseteq S],\\
J_{rel}(S,C)&=\sum_{r\in roots(S)}\widehat r_C(r),\\
J_{cov}(S,C)&=\sum_{k\in K_C}I[S\text{ fills }k],\\
J_{red}(S)&=\sum_{\{r,s\}}I[r,s\text{ share a redundancy class}],\\
J_{cost}(S,B)&=\sum_i cost_i(S)/B_i.
\end{aligned}
$$

All arithmetic is exact. Proposed-Full selects from `A_resp` by the lexicographic tuple

$$
\langle J_{TB},J_{rel},J_{cov},-J_{red},-J_{cost},-J_{tie}\rangle.
$$

Proposed-NoTransitionPriority reads the identical row and removes only the first coordinate:

$$
\langle J_{rel},J_{cov},-J_{red},-J_{cost},-J_{tie}\rangle.
$$

`J_tie` is the stable root and route vector and applies last. Both variants have identical allowed caps; no realized-cost dominance is required. Every coordinate is reported, and optional matched-realized-cost scoring is nonconfirmatory.

Mechanism evidence requires Proposed-Full to exceed the ablation on medium-budget horizon-rare, support-complete consequential recall, without a budget violation. The desired diagnostic pattern is no observed decrease in frequent-consequential recall and no observed increase in rare-nonconsequential selection. Those small balanced cells remain descriptive rather than formal pass/fail gates; a contrary pattern weakens interpretation and must be discussed. The paired mechanism summary reports wins, losses, and ties. If the ablation matches Full, any gain over guard-ERIPCF cannot be attributed to transition-complete priority.

Objective code is a pure function over frozen row columns. It cannot import evaluation-gold models. A regression fixture differing in exactly one complete transition bundle demonstrates the only intended ranking difference.

## 18. Horizon-filtered native and guard-wrapped baselines

All native methods start from the same reasonable root-level horizon filter. They are not deliberately given later narrative information. They select and materialize their ordinary raw structures under the same component caps, after which the common verifier reports omissions without repairing them. Native results are secondary practical comparisons, not evidence that Full's utility objective is safer.

Every guard-wrapped method optimizes over the same `A_resp` table with identical canonical routes, closure, presentation grammar, cost calculator, vector caps, response schema, and guard versions. Passing the contract is therefore validation of the wrapper and verifier, not an independent selector advantage.

### 18.1 Frequency

Frequency is the number of distinct verified in-horizon `EvidenceAnchor` records that depict or present that particular root occurrence or assertion. It is not event-type frequency, consequence, diagnostic class, or the named HorizonRarity label. When roots share an occurrence, a documented feature record copies the occurrence count.

Native frequency sorts descending, greedily adds roots while raw caps permit, and uses opaque IDs for ties. It does not skip a higher-ranked feasible root merely to improve semantic coverage. Guard-frequency maximizes summed frequency over common response-feasible rows, then lower normalized cost and the common stable tie. Only frequency baselines receive this raw count.

### 18.2 Personalized PageRank

PPR uses the frozen undirected root-incidence graph. Edge weight is the number of distinct present link types among shared participant, explicit event-state/assertion link, causal/support link, and temporal/state-chain adjacency, hence an integer from 1 through 4. Duplicate instances do not inflate it. Personalization is uniform over roots matching focal entities, task-compatible types, or the specified holder; when empty it is uniform over all 18 roots. Damping is 0.85, dangling mass returns to personalization, node order is sorted, tolerance is `1e-12`, and the maximum is 1,000 iterations.

Scores are quantized to 12 decimal places and stored as integers scaled by `10^12`; opaque IDs break remaining ties. Nonconvergence is `ERROR_OR_UNKNOWN`. Native PPR greedily materializes ranked roots under raw caps. Guard-PPR maximizes summed PPR over common response-feasible rows, then lower cost and stable tie.

### 18.3 ERIPCF and fairness assertions

Native and guard-ERIPCF use Section 16 exactly. Guard-ERIPCF is the primary comparator. Every wrapped result must assert equal hashes for candidate pool, canonical routes, common row table, budget, response schema, evidence universe, verifier bundle, presentation grammar, and tie-rule version. The only allowed differences are the manifested score function and selected row. There is one common guard wrapper, cost calculator, experiment runner, and result schema. Any mismatch invalidates the paired comparison.

The method names used in artifacts are exactly:

```text
proposed_full
proposed_no_transition_priority
native_frequency
guard_frequency
native_ppr
guard_ppr
native_eripcf
guard_eripcf
```

No artifact calls ERIPCF `PCST` except a documented literature and V2 crosswalk.

## 19. Budget calibration, matching, and presentation scope

The budget vector is

$$
B=(B_{root},B_{entity},B_{event},B_{transition},B_{state},B_{assertion},B_{relation},B_{evidence},B_{word}).
$$

All coordinates count unique postclosure, postpresentation atoms. The root caps are low 4, medium 6, and high 8. Non-root caps follow Section 3.3 and are generated from the public-development graph only. `budgets.v1.yaml` includes source graph, route-policy, response-schema, closure, cost, and presentation hashes; task minima; selected vectors; transition marginal; final caps; and generation command.

The researcher reviews every integer, confirms componentwise monotonicity, and freezes canonical JSON plus SHA-256 before opening held-out texts. Held-out commands cannot write the budget directory. Sensitivity budgets are deterministic children of the frozen primary budget and cannot be added after results.

Requested costs are the cap; realized costs are recomputed from each closure and display record. Every paired method receives the exact same cap hash, including Full and its ablation. Shared atoms count once. Words follow a fixed whitespace-token grammar over static identity labels, relation phrases, epistemic qualifiers, time qualifiers, and evidence cues. No adaptive descriptors or layout-area optimization appears in Paper 1.

The study evaluates **semantic projection under count and word proxies**, not visualization layout or graph readability. Claims about visual comprehension are prohibited. Required budget sensitivity is Full versus guard-ERIPCF on the 10 HRC contexts at low, medium, and high. Optional single-coordinate perturbations are dropped first.

## 20. Corruption-based verifier validation

The corruption bank has 18 public-development or hidden-control development cases; 45 sealed held-out corruptions, five per primary category; 10 held-out clean controls; and 45 target-guard-disabled evaluations. The nine categories are evidence omission, fact/label horizon leak, epistemic promotion, temporal inconsistency or non-entailment, unsafe identity merge, support incompleteness, provenance omission, budget overrun, and response/closure corruption. Multi-guard dependencies are expected for some cases.

Each `CorruptionCase` stores base projection hash, deterministic mutation delta, clean/corrupt status, target guards, expected full predicate signature, expected non-target passes, isolation mode, severity, and template/code hash. An isolated case must pass every non-target guard when the target guard is disabled. Otherwise the case is explicitly `multi_guard`, with a preregistered complete signature. A missing-evidence mutation, for example, may properly fail evidence, support, and response.

Development cases guide verifier construction. Mutation templates, category counts, selection rules, and expected-signature rules freeze before held-out material is opened. After held-out oracle graphs exist, a deterministic process instantiates and seals cases without running the verifier. Clean controls and corruptions are immutable before test execution.

Report per-category sensitivity and specificity, full-signature accuracy, target-guard removal effect, and severity. A miss is preserved. If it reveals an implementation defect, the correction gets a new verifier version, the original and rerun both remain, all cases rerun, and the preregistered claim is qualified. It is never silently converted into a development case. Guard ablations are fault-localization experiments, not selector variants.

## 21. Annotation-uncertainty sensitivity

Before selector execution, outcome-relevant ambiguities are frozen as small alternative-state groups. For each fixed projection, scoring runs against:

- the curator-primary oracle;
- conservative semantics, requiring success under every licensed combined state;
- permissive semantics, allowing success under any licensed combined state.

No selector, pool, route, support structure, or certificate is regenerated. Results report the entire envelope and whether fixed-benchmark and mechanism conclusions change. If more than three groups or two values per group would be required for a query, it is flagged during context validation and may be replaced only under the preregistered rule **before** `ContextSealManifest`. After context sealing, it remains with the recorded ambiguity or triggers the declared context failure; it is never replaced after target, availability, or method results.

## 22. Experiment manifests and freeze protocol

The predecessor's single overloaded manifest is replaced by staged immutable manifests. The exact order is:

1. Create public development records, hidden verifier fixtures, preliminary schemas, and audit templates.
2. Run the full public-development pipeline, annotation-rate pilot, exactness checks, performance pilot, budget calibration, and ERIPCF parameter selection.
3. Revise only development artifacts and rerun the complete pilot until gates pass.
4. Freeze ontology/handbook, schema interfaces, route policy, context and query templates, generator, seeds, budgets, ERIPCF grid winner and rho rule, objectives, tie rules, guards, temporal semantics, corruption templates, metrics, and code/environment interfaces in `DesignFreezeManifest`. Preregister it.
5. Only now open the five real held-out episode sources. Annotate them and freeze sanitized method graphs in `HeldoutMethodGraphManifest`.
6. Instantiate and freeze all 26 held-out contexts and response schemas in `ContextSealManifest` without target IDs.
7. Compile every context-root canonical route from only held-out method graphs and contexts; freeze `CanonicalRouteSealManifest`.
8. In restricted evaluation storage, select all 10 HRC targets and freeze their root IDs, transition bundles, initial `GoldSup`, rarity/consequence records, diagnostic labels, and alternatives in `TargetGoldSealManifest`.
9. Compile and seal method-only pool capsules. Validate that target and gold paths are absent.
10. Run the isolated candidate worker. It writes `PoolWorkerOutputManifest`, which references only its capsule; archive access attestations.
11. After worker exit, the controller verifies the prior target commitment in the append-only ledger and freezes `PoolSealManifest`. The target hash is chronology metadata, never a worker input.
12. In a read-only evaluation join, calculate target availability, canonical support availability, any-route availability, and pressure diagnostics. Write `AvailabilityAssessmentManifest` and either `BENCHMARK_READY` or `BENCHMARK_CONSTRUCTION_FAILED`. The failure branch stops superiority execution and has no repair path.
13. On the ready branch only, instantiate and seal held-out corruption cases from the frozen templates without invoking verifier outcomes. Compile and freeze each `ERIPCFInstance`.
14. Create the final `ExperimentManifest`, binding every prior digest, code revision, lock file, solver settings, SQLite/schema versions, machine profile, seed, split, corruption seal, and ERIPCF instance.
15. Run all methods and seal selector results. No confirmatory code or configuration change follows.
16. Present a deidentified mixture of genuinely novel support paths from all eligible primary methods for one blinded, frozen-rubric review. Write one scoring-only `GoldSupportExtensionManifest`.
17. Rescore all frozen projections once under the extended support family and annotation alternatives; generate tables; seal final outputs.

The canonical-route and target seals must predate pool execution; the pool seal must predate availability calculation. Filesystem creation times are not proof. The append-only freeze ledger and hash chain prove order. The worker output points only backward to the capsule; the controller adds the already existing target commitment only after worker exit. The pool capsule intentionally omits the target digest so it cannot be a hidden input.

Code defects found after design freeze may be fixed only when a public or synthetic regression test demonstrates the defect without inspecting aggregate held-out outcomes. The old code, result, and manifest remain; the fix receives a new version; every affected run is repeated; and the deviation ledger explains scientific impact. Tuning, semantic-rule changes, gold reinterpretation, or case deletion are not defect repair.

Held-out opening is allowed only when public-development schema and annotation gates, temporal boundary tests, exact-search cross-checks, cache/resume performance thresholds, corruption development tests, budget calibration, baseline tuning, source-document hashes, and workload forecasts all pass. Full evaluation begins only when target and pool seals are independently valid, availability and pressure gates pass, corruption cases are sealed, and common-table fairness hashes match.

## 23. Minimal experiment matrix

The common exact table is not a method run. It is generated once for each of 32 answerable contexts with root cap 8 and once without a root cap for eight typed probes; two invalid contexts stop during validation. Low and medium results filter the existing rows.

| Block | Method settings | Contexts | Budgets | Records |
|---|---:|---:|---:|---:|
| ERIPCF development grid | 12 | 6 development | medium | 72 |
| Final development check | all 8 variants | 6 development | medium | 48 |
| Held-out medium comparison | Full, NoTransitionPriority, guard-ERIPCF, guard-PPR, guard-frequency | 26 held-out | medium | 130 |
| Primary budget sensitivity | Full and guard-ERIPCF | 10 HRC | low and high; medium already above | 40 |
| Native practical comparison | native frequency, PPR, ERIPCF | 10 HRC | medium | 30 |
| Feasibility/certificate comparison | Full and guard-ERIPCF | 10 probes | declared budget or validation | 20 |
| **Final selector outputs excluding grid** |  |  |  | **268** |
| **All selector and parameter evaluations** |  |  |  | **340** |

The required verifier archive adds 18 development/hidden cases, 55 held-out full-verifier cases (45 corruptions and 10 clean controls), and 45 target-guard-disabled cases, or 118 records. The planned evaluation archive therefore has **458 required evaluation records**. These records reuse common rows and are not independent statistical observations.

The confirmatory utility slice is Full versus guard-ERIPCF at medium on 10 HRC contexts. The mechanistic slice is Full versus NoTransitionPriority on the same cases. Full versus guard-ERIPCF at low and high is required budget sensitivity. Frequency/PPR and natives remain secondary.

The optional controlled pool, paraphrases, route sensitivity, one-coordinate budgets, and extra figures are not in the required 458-record total and MUST remain inactive when the gate is tight; this prevents scope growth but does not reduce the stated required estimate. Actual protected-core reductions, declared before held-out opening, are: remove guard-frequency and guard-PPR from the 16 non-HRC held-out contexts (32 outputs); then remove Proposed-NoTransitionPriority from those 16 (16 outputs), retaining Full and guard-ERIPCF at medium. All eight methods remain represented on development and the 10 HRC slice. A final reduction may shrink initial distinct transition audits from at most 32 to 20, of which eight still receive an additional delayed re-audit. The 10 HRC cases, three Full/guard-ERIPCF budgets, medium Full/ablation on HRC, horizon rarity for HRC, independent public held-out case, corruption core, and exact search are never dropped without a new preregistration and narrower paper.

## 24. Metrics, fixed-benchmark reporting, and result generation

Metric definitions are centralized and versioned. They consume immutable projections and evaluation gold; selector and candidate modules cannot import them.

### 24.1 Availability, admissibility, response, and support

For a declared context set `Q`:

$$
TAR={1\over |Q|}\sum_q TargetAvailable_q,
\quad
CSAR={1\over |Q|}\sum_q CanonicalSupportAvailable_q,
\quad
ASAR={1\over |Q|}\sum_q AnyRouteSupportAvailable_q.
$$

For a returned projection `S_mq`:

$$
Adm_{mq}=I[S_{mq}\in\mathcal A_{adm}],\qquad
Resp_{mq}=I[S_{mq}\in\mathcal A_{resp}].
$$

`AdmissibilityRate` and `ResponseRate` are reported separately. For guard-wrapped methods, admissibility is expected by construction and is an execution invariant. `ResponseFamilyExists` is established from complete enumeration and is distinct from a method returning a response. Correct abstention requires the exact class and a recomputable certificate. `ERROR_OR_UNKNOWN` is always incorrect.

Let `t_q` be the target transition, `TB(t_q)` its complete transition bundle, and `Q_q` the accepted support family after the single masked extension. Define:

$$
Y_{mq}=I[TB(t_q)\subseteq S_{mq}]\ I[\exists Q\in\mathcal Q_q:Q\subseteq S_{mq}].
$$

`HorizonRareSupportCompleteRecall` is the mean of `Y` over the 10 HRC contexts, conditional on the output being admissible. Its paired difference Full minus guard-ERIPCF is the primary utility outcome. `ASCPR` is the violation-zeroed combined summary:

$$
ASCPR_m={1\over |Q|}\sum_q Y_{mq}Adm_{mq}.
$$

It is not the only outcome and cannot conceal the reason for failure. Complete support preservation is `I[exists Q subseteq S]`; fractional preservation `|Q intersect S|/|Q|` is secondary. Evidence precision is displayed answer claims with a licensed route divided by displayed answer claims. Evidence completeness is claims retaining every required anchor and qualification divided by displayed answer claims. Unsupported assertion rate is one minus evidence precision for factual answer claims.

### 24.2 Violations, diagnostics, and efficiency

The pipeline reports temporal inconsistency, nonentailed temporal display, holder/status error, rumor promotion, horizon fact leak, horizon label leak, identity error, provenance omission, support omission, and budget overrun separately by critical, major, and minor severity. `ViolationOutputRate` uses returned outputs as denominator; a claim-level rate uses displayed claims when meaningful. No severity-weighted total trades a critical spoiler or epistemic promotion against minor presentation defects.

`RareNonconsequentialFalseProtection` is selected horizon-singleton nonconsequential transitions divided by available items in that diagnostic cell. `FrequentConsequentialRecall` is complete-bundle recall for horizon-repeated consequential transitions. The balanced diagnostic has 20 transition-context tuples, five in each rarity-by-consequence cell, and remains descriptive. It is not a confirmatory gate.

The natural-frame description takes the first five eligible audited tuples by discourse order from each of the five real held-out bundles, 25 total, with no probability weight or population interval. It reports per-bundle and equal-bundle proportions. At most seven prespecified cell-filling tuples may supplement the diagnostic audit, keeping the master audit at no more than 32. Local-rarity versions substitute LocalCount and are sensitivity results only.

Outcome relevance is the mean frozen 0-to-3 curator grade of selected roots, with empty outputs separate. Redundancy is the proportion of selected root pairs in a common redundancy class, zero for fewer than two roots. Root compression is `1 - |X|/18`. Component compression is reported coordinatewise against the eligible closed-pool inventory. Every realized budget coordinate is reported, along with HRC successes per selected root, evidence cue, and word. No confirmatory composite combines these efficiencies.

### 24.3 Fixed-benchmark decision rule

The fixed-benchmark superiority claim requires preregistered conditions 1 through 8 below. The separate transition-priority mechanism claim additionally requires condition 9.

1. Target availability and canonical-route support availability are both 1.00 on the 10 HRC contexts; otherwise the benchmark construction fails before selector superiority is assessed.
2. The corruption contract has no critical miss under the declared confirmatory verifier version, and all guard-wrapped HRC outputs are admissible and responsive.
3. Proposed-Full exceeds guard-ERIPCF in medium-budget HRC recall by at least 0.20, equivalent to at least two net successes among 10 cases.
4. The paired medium result has at least three Full wins and at most one Full loss.
5. Full's within-bundle difference is positive in at least three of the five held-out real-text bundles.
6. The aggregate difference remains positive after deleting any one held-out bundle.
7. No one bundle contributes more than 50 percent of Full's positive-gain count; if there is no gain, concentration is `NA` and the claim fails.
8. The independent public-domain held-out bundle has a nonnegative paired difference.
9. Proposed-Full exceeds Proposed-NoTransitionPriority on HRC recall without using a different cap. If it does not, the transition-priority mechanism claim fails even if the main comparator result is positive.

Safety is lexically prior to utility: relevance, compression, or recall cannot compensate for a critical violation. Conditions 3 through 8 are deterministic benchmark criteria, not estimates about the novel or genre. Frequent-consequential and rare-nonconsequential cells are reported beside the decision table but are not converted into an additional binary gate because each has only five items.

### 24.4 Robustness and novel support paths

Report every query result, exact wins/losses/ties, bundle means, low/medium/high curves, and leave-one-bundle-out differences. Positive-gain concentration is:

$$
Concentration=\max_b{
\sum_{q\in b}\max(Y_{Full,q}-Y_{ERIPCF,q},0)
\over
\sum_q\max(Y_{Full,q}-Y_{ERIPCF,q},0)}.
$$

After systems are sealed, collect previously unlisted support paths from Full, the ablation, and guard-ERIPCF at medium on the 10 HRC contexts. Deduplicate by atom-set and qualification signature and cap the review packet at 30, using a frozen hash order if needed. Mix and relabel them so the researcher cannot see method identity. Apply the frozen sufficiency rubric: fully sufficient, partially sufficient, valid but redundant, unsupported, temporally invalid, epistemically invalid, post-horizon, or irrelevant. This produces one scoring-only extension. It cannot tune a method, alter the original availability calculation, or remove an initial gold support.

Annotation alternatives yield curator-primary, conservative, and permissive versions. Route sensitivity, if retained, holds winning root masks fixed. Paraphrase stability and the controlled pool are optional.

### 24.5 Statistical reporting without population inference

The unit is a preregistered query-context in a fixed bundle, not a random draw from a narrative population. There is no sign-flip test, p-value, bootstrap population interval, normal-theory confidence interval, Horvitz-Thompson estimator, or claim of representativeness. Report exact paired counts, absolute differences, denominators, per-bundle results, leave-one-bundle-out values, and contribution concentration. The AGOT and public-domain held-out slices remain separate.

Intra-rater agreement and corruption sensitivity/specificity receive exact descriptive denominators and, when useful, exact binomial intervals clearly labeled as measurement precision for the fixed fixture set rather than genre inference. Null and negative results are fully tabulated. Missing records are never imputed; they are categorized as protocol failure, construction failure, or execution error and remain in denominator rules specified before execution.

### 24.6 Automated result artifacts

One reporting command, operating on sealed JSONL results and metric definitions, must generate:

- canonical JSONL `MetricRecord` output;
- CSV per-query audits of roots, closure, support, violations, routes, and costs;
- per-bundle and separate AGOT/public summaries;
- target and canonical/any-route support availability tables;
- fixed-benchmark gate and mechanism-ablation tables;
- low/medium/high sensitivity and realized-cost tables;
- native and guard-wrapped comparisons;
- corruption confusion and expected-signature matrices;
- certificate class and validation tables;
- rarity, route, and annotation-uncertainty sensitivity tables;
- leave-one-bundle-out and contribution-concentration tables;
- a reproducibility manifest connecting each output row to input and run hashes;
- a small set of paper-ready static figures for paired outcomes, budget curves, and component costs.

No notebook or copied spreadsheet cell is authoritative. Paper tables have a source-data hash or sidecar; any typeset copy is automatically compared with canonical CSV.

## 25. Testing and code-quality strategy

### 25.1 Unit tests

Unit tests cover strict schema validation; referential and visibility integrity; canonical identifiers, JSON and hashes; HMAC behavior on synthetic text; closure; route input-order, multi-anchor ties, dominance and allowed context/horizon changes; component costs and word counts; horizon filtering; every endpoint predicate; temporal normalization and entailment; deterministic inconsistency cores; epistemic qualification; support validation; candidate scoring and forbidden fields; seal transitions; canonical-versus-any-route availability; root incidence and rho; closed and raw ERIPCF forests; each objective coordinate; exact ties; metric formulas; and every certificate class.

### 25.2 Property-based tests

Hypothesis tests the following scientific invariants:

- closure is extensive, monotone, and idempotent;
- union of precomputed single-root closures equals reference worklist closure under the declared positive dependency fragment;
- a larger componentwise budget never removes a member from the feasible family;
- selection is a subset of the sealed eligible roots;
- every admissible displayed answer claim has a guard-valid support witness;
- canonical output is invariant to semantically irrelevant input order and repeated execution;
- horizon safety is inherited through dependencies marked and validated as horizon-nonincreasing;
- before/after and during/contains are converses, equal/intersects are symmetric, and before excludes intersection;
- deterministic temporal cores are invariant to input order, solver seed, and equivalent serialization;
- changing evaluation gold while fixing a pool capsule cannot change pool output;
- changing target/gold fields while fixing method inputs cannot change canonical route choices;
- a route choice changes only when a declared route, context, or horizon input changes;
- increasing the mask cap preserves every previously enumerated mask;
- exact objective comparison is transitive and total after final ties.

### 25.3 Integration tests

Integration tests run a public synthetic query from typed context through route seal, pool capsule, isolated generation, closure, temporal checks, common rows, all eight methods, projection or abstention, metrics, and canonical report. They cover all three budgets, every baseline, an invalid context, each infeasibility class, corruption rejection, exact known optima, native raw connector materialization, interruption/resume, masked path extension, annotation alternatives, and public release with the private root unavailable.

Independent-sealing integration tests verify that a target seal predates a pool run; the worker cannot see a gold canary; a symlink cannot escape the capsule; availability cannot run until both seals exist; a failed availability case has no repair transition; and a target or pool change invalidates downstream manifests.

### 25.4 Regression and performance tests

Freeze known closures, route choices, temporal SAT/entailment/core results, violation signatures, response slots, cost vectors, mask counts, objective values, closed and native-raw ERIPCF connectors/MSTs, winning projections, certificate payloads, aggregate metrics, SQLite resume results, and canonical export hashes. Cache invalidation fixtures change route-policy, context-horizon, temporal-semantics, and presentation hashes; raw and closed forest modes must never collide. Tiny independent exhaustive forests, raw-materialization alternatives, and root selectors serve as reference oracles.

The performance pilot is a scientific acceptance test, not a continuous test on every commit. A smaller performance fixture guards against accidental order-of-magnitude regression. Coverage targets are at least 90 percent statement coverage in closure, temporal, verification, costing, exact search, and ERIPCF modules and at least 80 percent across the package; every violation enum, objective coordinate, tie branch, and certificate class must execute. Passing a percentage cannot compensate for a missing invariant test.

### 25.5 Code quality

All code uses type hints, descriptive names, small single-purpose modules, and docstrings explaining scientific semantics. Validated records are immutable where practical. Logging is structured and redacted. There is no hidden global state, implicit seed, locale-dependent sorting, or duplicated guard logic. Enumerations, metric definitions, guard wrapper, cost calculator, experiment runner, and result schema each have one canonical implementation.

The highest-risk modules receive paired reference tests and human review: temporal normalization/core generation, route canonicalization, closure compilation, exact enumeration/cache resume, candidate capsule isolation, ERIPCF incidence and MST cost, budget presentation accounting, and final metric gates.

## 26. Reproducibility and release

The public release is organized around what can actually be reproduced:

- ontology profile, annotation handbook, competency questions, schemas, static presentation grammar, and all configurations;
- code, lock file, environment metadata, deterministic seeds, and command documentation;
- complete public-development and public-held-out text references, annotations, contexts, pools, corruptions, selections, metrics, and reports where the source editions permit it;
- the hidden narrative and controls after unsealing;
- target-independent generator fixtures and leakage tests;
- safe query-context abstractions, budget derivation, objective and baseline definitions;
- canonical public manifests, run records, tables, and figures;
- legally reviewed aggregate and redacted AGOT derivatives, with opaque IDs and coarse anchors;
- hashes and counts sufficient to detect a changed restricted artifact, but not to reconstruct its text.

Restricted AGOT oracle graphs, exact locators, aliases, target/support mappings, and audit windows remain private unless a later rights decision permits controlled access. The HMAC key remains secret and is never released; an authorized verification service or controlled local procedure may compare values without disclosing it. A `REPRODUCIBILITY_LIMITS.md` must explain that public users can rerun the complete algorithms on the public and hidden corpora but cannot reconstruct or independently verify exact AGOT spans without lawful same-edition access and restricted alignment information.

Every public command runs with the private root absent. A release allowlist enumerates files rather than trusting `.gitignore`. The release audit scans paths, schema fields, canonical JSON values, logs, repository history, archives, tables, and figure metadata. It also checks known private n-gram canaries. Synthetic HMAC tests verify canonicalization version, key-ID mismatch, deterministic integrity under a fixed test key, secret exclusion, and rejection of exact offsets, raw text, or a span-HMAC value inserted into a public export.

The final reproducibility manifest records source-document digests, code commit, lock digest, Python/Z3/SQLite versions, operating-system and CPU summary, schema and handbook versions, all seal hashes, canonical result hashes, commands, seeds, known deviations, and the public/private reproduction boundary. A clean checkout must reproduce public canonical outputs from one documented command sequence without network access after environment installation.

## 27. Phased implementation roadmap

The roadmap is sequential. Hours are expected active researcher hours and sum to 142; automated compute is separate. A phase passes only when both software and scientific acceptance criteria hold.

| Phase | Inputs and dependencies | Tasks | Outputs | Required tests and acceptance | Expected hours | Stop or revise condition |
|---|---|---|---|---|---:|---|
| 1. Authority, repository, environment | this plan, research V2; none | verify source hashes and recent cited methods; initialize repository, lock and quality config | source-document manifest, empty package, environment report | hashes match; clean install and public smoke test | 3 | unexplained source mismatch or unpinnable core dependency |
| 2. Typed schemas, IDs, and access boundary | Phase 1 | implement common envelope, schemas, canonical serialization, identifiers, visibility, HMAC utility on synthetic text, access classes | generated JSON Schemas, public valid/invalid fixtures | deterministic bytes/IDs; extra-field rejection; no private path access | 5 | schema cannot express core distinctions without methodology change |
| 3. Graph integrity and closure | Phase 2 | graph compilation, dependency validation, reference worklist, bitset closure | compiled public fixtures and closure certificates | extensiveness, monotonicity, idempotence, union/reference equality | 6 | ambiguous dependency semantics or closure mismatch |
| 4. Temporal, epistemic, and other guards | Phases 2-3 | endpoint normalization, Z3 translation, entailment, deterministic cores, guard result API | verifier bundle and temporal reports | all boundary, seed, multiple-core, rumor, identity and evidence fixtures pass | 7 | nondeterministic certificate, solver unknown, or guard overlap undocumented |
| 5. Budget, response, and certificates | Phases 3-4 | presentation costing, response schemas, nested families, exact abstention witnesses | cost/response records and certificate fixtures | vector boundaries and every abstention class independently recompute | 4 | task success remains conflated with semantic safety |
| 6. Exact store and search correctness | Phases 3-5 | SQLite schema, mask enumeration, caches, resume, exact selector and reference solver | complete small-instance tables and ordered exports | known optima, mask counts, resume, cache uniqueness, byte-identical exports | 7 | missing/duplicate row or unverified pruning |
| 7. Frequency, PPR, and ERIPCF | Phases 5-6 | native/wrapped baseline logic, root incidence, rho, exact MST, parameter-grid interface | eight method interfaces and forest certificates | tiny exhaustive forests, PPR/frequency fixtures, common-wrapper hashes | 7 | ERIPCF uses hidden Steiner node or comparator is not exact |
| 8. Full objective and direct ablation | Phases 6-7 | implement common features and two lexicographic tuples | Full/ablation fixtures | sole intended coordinate difference and equal caps verified | 2 | any extra ablation difference or gold import |
| 9. Corruption framework | Phases 4-8 | mutation schemas, expected signatures, guard-disable harness | 18 development/hidden cases and sealed template rules | isolation or multi-guard signatures correct | 4 | corruption cannot be attributed under declared signature |
| 10. Annotation and rarity tools | Phases 2-4 | worksheets, compiler, issue ledger, search packets, audit and reannotation exporters | handbook-linked templates and public examples | round-trip, mask, duplicate, horizon, release-safety tests | 4 | tooling invents or drops scientific values |
| 11. One-context performance pilot | Phases 6-8 | run cap-8/stress/warm/resume benchmark and profile | performance report and empirical extrapolation | Section 15 thresholds and exact hashes pass | 3 | correctness failure stops; resource excess triggers one optimization cycle |
| 12. Complete public-development pilot | Phases 1-11 | annotate 25 roots and hidden controls; build six contexts; audit; calibrate budgets; tune ERIPCF; run/report end to end | development graph, budgets, parameters, timings, complete pilot report | all scientific checks, query types, corruptions, reporting and annotation-rate measures pass | 12 | end-to-end gap, inadequate annotation stability, or infeasible task schema |
| 13. Reforecast, preregister, and design freeze | Phases 11-12 | calculate three scenarios, apply drop ladder if needed, freeze all design inputs | preregistration and `DesignFreezeManifest` | expected total <=145 and conservative reduced core <=160 | 4 | forecast remains above ceiling; do not open held-out text |
| 14. Held-out corpus and oracle work | Phase 13 | follow frozen exposure order; annotate first 25-root AGOT bundle and reforecast before opening the other three AGOT bundles and, last, independent public held-out; complete 125 roots, horizon audits, delayed reannotation and method graphs | five held-out oracle/method graphs, audit ledgers, first-bundle gate, stability report | first-bundle forecast passes; 25 roots each; graph/audit/stability checks; no scientific retuning | 50 | do not open another bundle if projected ceiling fails; reliability or rights failure stops |
| 15. Context, route, and target-gold seals | Phase 14 | freeze 26 contexts; compile and seal method-only canonical routes; select 10 HRC targets and initial supports in gold workspace | context, route, and target manifests | route input/gold-mutation tests, target eligibility, support rubric, alternatives and chronology pass | 5 | route or target rule requires discretionary outcome-aware repair |
| 16. Isolated pools and availability | Phase 15 | compile capsules; sandbox pool worker; seal pools; read-only availability join | pool/access/availability manifests | canary and mutation tests; 18 roots; TAR/CSAR and pressure gates | 4 | gold access is protocol failure; availability miss is construction failure |
| 17. Corruption seal and confirmatory execution | Phase 16 | instantiate held-out corruptions from frozen templates; seal cases and expected signatures before any verifier output; freeze ERIPCF instances and final manifest; run exact tables, methods, probes and corruptions | corruption seal, final manifest, immutable run and projection records | no verifier output predates corruption seal; deterministic rerun, fairness hashes, no missing record | 4 | postfreeze change, cache inconsistency, premature verifier output, or unresolved error |
| 18. Masked support review and final scoring | Phase 17 | deidentify <=30 novel paths; one frozen review; rescore alternatives once | gold extension and final metric records | method masking, rubric consistency, no reselection | 3 | method identity exposed or review attempts to tune systems |
| 19. Analysis and manuscript outputs | Phase 18 | generate all paired, bundle, budget, corruption and sensitivity outputs | canonical tables, figures, gate report | hand-check sample, hash-linked source tables, no population statistics | 4 | copied/manual result or denominator mismatch |
| 20. Reproducibility and rights package | Phase 19 | clean public rerun, allowlist scan, copyright audit, documentation | public package and restricted archive manifest | private root absent; release scan and full public reproduction pass | 4 | any protected text/key/path leaks or public rerun fails |

Phase 15 is deliberately split conceptually: target gold is finished and sealed before Phase 16 creates a pool capsule. Phase 16's evaluator joins two read-only seals only after pool freezing. No command combines target selection and candidate construction.

## 28. Realistic workload, pilots, and contingency

### 28.1 Three-point researcher-hour estimate

The following estimates include researcher review of Codex-produced code. They do not count unattended CPU time as labor.

| Activity | Optimistic | Expected | Conservative |
|---|---:|---:|---:|
| Literature verification and scientific trace check | 1 | 2 | 3 |
| Minimal ontology and annotation handbook | 2 | 3 | 5 |
| Corpus selection and boundary rules | 1 | 2 | 4 |
| Schemas, IDs, graph representation, access and copyright controls | 5 | 6 | 9 |
| Closure and admissibility verifier implementation | 5 | 7 | 10 |
| Temporal checker, entailment, and deterministic cores | 5 | 6 | 9 |
| Exact optimization, caches, SQLite, and resume | 7 | 8 | 12 |
| Native frequency and PPR baselines | 2 | 3 | 5 |
| Guard wrapper and exact ERIPCF baseline | 3 | 4 | 6 |
| Proposed-Full and direct ablation | 1 | 2 | 3 |
| Automated testing and debugging not counted above | 4 | 5 | 8 |
| Annotation and horizon-audit tooling | 3 | 4 | 6 |
| Public-development first-pass annotation | 5 | 5 | 7 |
| Full development/hidden pilot, queries, supports, calibration | 5 | 5 | 8 |
| Performance pilot and workload reforecast | 2 | 3 | 5 |
| Preregistration and design freeze | 2 | 3 | 4 |
| Held-out boundary audits | 1 | 2 | 3 |
| First-pass annotation of 125 held-out roots | 25 | 30 | 38 |
| Horizon-level rarity audits, up to 32 total | 8 | 12 | 16 |
| Delayed reannotation of 24 roots and eight rarity audits | 4 | 6 | 8 |
| Query and typed-context construction | 2 | 3 | 5 |
| Initial support-path construction and masking preparation | 2 | 3 | 5 |
| Dense pools, capsules, sealing, and availability audit | 3 | 4 | 5 |
| Oracle runs, corruption evaluation, and result validation | 3 | 4 | 6 |
| Masked novel-path review | 1 | 2 | 4 |
| Statistical and sensitivity analyses | 3 | 4 | 7 |
| Release packaging, documentation, and copyright audit | 3 | 4 | 5 |
| **Required total** | **108** | **142** | **206** |

The expected case leaves 18 hours below the hard ceiling. The conservative prior exceeds it by 46 hours and is not hidden. Therefore the plan is conditionally feasible, not guaranteed feasible. Optional controlled pools, paraphrases, route sensitivity, one-coordinate budget perturbations, extra figures, a human pilot, and an LLM condition are excluded from the required total; declining those options prevents scope growth but saves zero hours from the table above.

### 28.2 Annotation-rate pilot

Partition the 25 public-development roots into five consecutive five-root work blocks. Log active minutes separately for anchors, entity/event/state/assertion annotation, temporal/dependency work, validation/correction, query/support work, and shared setup allocated pro rata. Let `a_bar` be mean active minutes per root and `a_max5` the slowest block rate. Forecast held-out first-pass annotation at:

$$
r_{root,expected}=\max(14,1.10a_{bar}),
$$

$$
r_{root,conservative}=\max(17,1.25a_{bar},a_{max5}).
$$

Bundle-boundary cost remains separate. Query, support, pool, horizon-audit, reannotation, novel-path, and release rates are likewise taken from distinct development activities, not folded into root minutes. Codex wall time is not annotation time.

### 28.3 Forecast equation and gates

For scenario `s` in optimistic, expected, or conservative:

```text
H_s = H_spent
    + H_fixed_remaining_s
    + r_root_s * N_roots_remaining
    + r_audit_s * N_audits_remaining
    + r_query_s * N_contexts_remaining
    + r_support_s * N_supports_remaining
    + r_pool_s * N_pools_remaining
    + r_path_s * N_paths_forecast
    + H_release_s
    + H_defect_s
```

Rates are hours per unit, and only remaining units are counted. The performance pilot supplies actual debugging/review effort and machine time; the annotation pilot supplies human rates. The forecast is rerun before held-out opening and after the first held-out 25-root bundle.

Held-out opening requires expected total no greater than 145 and conservative minimal core no greater than 160 after the declared scope ladder. `DesignFreezeManifest` fixes an exposure order: first one predeclared AGOT bundle, then the remaining three AGOT bundles, then the independent public held-out bundle. After the first 25-root AGOT bundle, the forecast is rerun; another source is not opened if the protected remainder exceeds 160. This resource-only checkpoint cannot change scientific settings except through scope reductions already declared before any held-out source was opened.

The scope ladder is:

1. keep controlled challenge pools, paraphrases, route sensitivity, one-coordinate budgets, and optional figures inactive; this prevents growth but saves zero from the stated required estimate;
2. retain the already fixed cap of 30 deduplicated novel paths from Full, the ablation, and guard-ERIPCF at medium on the 10 HRC contexts;
3. remove guard-frequency and guard-PPR from the 16 non-HRC held-out contexts, eliminating 32 secondary outputs;
4. remove Proposed-NoTransitionPriority from those 16 contexts, eliminating 16 secondary outputs while retaining Full and guard-ERIPCF;
5. reduce initial distinct transition audits from at most 32 to 20, including every HRC target; eight of those 20 still receive a separate delayed repeat audit.

Pilot timings, not advance guesses, quantify the hours saved by steps 3 through 5. Applied reductions receive a revised run-count manifest and preregistration before held-out opening.

The following are protected: seven-bundle allocation, 150 natural roots unless a new narrower paper is preregistered, independent public held-out test, all 10 HRC cases, horizon rarity for all HRC targets, target-before-pool sealing, guard-ERIPCF, Proposed-NoTransitionPriority, all three Full/guard-ERIPCF budgets, primary corruption tests, and exact search. If the empirically conservative protected core still exceeds 160 hours, work stops before further held-out exposure and the paper is narrowed; estimates are not edited to force a go decision.

Automated compute time is reported separately after the performance pilot. Researcher supervision, defect triage, and output review remain in the human forecast.

## 29. Risk register

| Risk | Likely effect | Prevention or detection | Required response |
|---|---|---|---|
| Research and implementation semantics diverge | invalid construct or hidden method advantage | source hashes, traceability matrix, common schemas and wrappers | stop; document V2.1 correction before code continues |
| Target knowledge leaks into pools | benchmark target is preselected | gold-free capsule, mount isolation, access ledger, mutation/canary tests | protocol failure; do not report superiority |
| Candidate pool misses target/support | benchmark cannot test intended failure mode | TAR/CSAR only after independent seals | construction failure; no repair; publish narrower protocol result or rebuild under a new preregistration |
| Canonical route suppresses a valid support | apparent selector failure is route policy | any-route availability and fixed-mask route sensitivity | report route-policy failure; never switch primary route post hoc |
| ERIPCF is mistaken for standard PCST | overstated baseline and novelty | exact name, crosswalk, formal definition, no hidden Steiner nodes | revise all claims; result is only against ERIPCF |
| Exact search is too slow or large | ceiling exceeded or incomplete enumeration | cache hierarchy, SQLite, mandatory stress/resume pilot | one semantics-preserving optimization cycle, then drop secondary work or stop |
| Cache bug changes scientific rows | false exactness or unfair methods | canonical keys, byte verification, reference enumeration, cold/warm hashes | invalidate table and all dependent results |
| Temporal core is nondeterministic | different certificates for same graph | full-set deletion-minimal procedure and invariance tests | stop certificate claims and fix before held-out opening |
| Endpoint semantics are misapplied | contradiction or entailment error | explicit closed intervals, boundary fixtures, direct-predicate properties | stop temporal evaluation until corrected |
| Guard corruptions overlap | misleading sensitivity attribution | expected full signatures and `multi_guard` labels | report joint signature; do not call isolated |
| Textual support is promoted to fictional truth | epistemic invalidity | assertion-holder/status model and wording policy | contract failure; restrict claims to textual support |
| One-researcher annotation drift | unstable rarity/support/consequence gold | frozen handbook, masked delayed passes, alternatives | narrow construct or stop if gates fail |
| Public held-out choices influence design | invalid transfer evidence | access chronology and design manifest | remove transfer claim; treat as development only under a new version |
| Copyrighted text or key leaks | legal and ethical breach | external roots, HMAC, allowlist release, history scan | halt release, rotate key, purge through reviewed process, document incident |
| Workload exceeds ceiling | incomplete or rushed benchmark | two formal pilots and second reforecast after first held-out bundle | apply drop ladder; stop if protected core exceeds 160 |
| Postfreeze defect invites tuning | optimistic result bias | immutable original runs and public-fixture defect requirement | version, rerun all affected cases, disclose deviation |
| Fixed benchmark is overgeneralized | unsupported publication claim | automated wording checklist and separate bundle tables | limit conclusion to tested cases |

## 30. Stop, revise, and go gates

| Gate | Go | Revise | Stop or redirect |
|---|---|---|---|
| Source authority | both embedded hashes and this plan's external delivery hash match; precedence is recorded | explain benign line-ending migration with canonical copy | unexplained content change |
| Schema and ontology | every competency question fits minimal records and public fixtures validate | simplify optional fields before development freeze | core distinctions require a materially different ontology |
| Closure | reference and bitset closure agree; all invariants pass | optimize or correct before any corpus work | dependency semantics remain ambiguous |
| Temporal reasoning | all endpoint, entailment, and deterministic-core tests pass | fix solver translation on public fixtures | repeated nondeterminism or `unknown` on required fragment |
| Exactness and performance | complete mask counts, reference optima, resume hashes, and resource targets pass | one allowed optimization cycle | still above abort thresholds or any semantic mismatch |
| Development annotation | 25 roots valid; rate forecast passes; delayed pilot agreement adequate | refine handbook and repeat on development only | expected or conservative protected core cannot fit 160 |
| Budget and comparator | four task minima exist; caps, ERIPCF parameters and rho freeze without held-out data | simplify development response schema and rerun entire pilot | calibration requires held-out information or arbitrary clipping |
| Design freeze | all configs, templates, code interfaces, tests, seeds and drop decisions have hashes | complete missing public tests | held-out source was opened early for tuning |
| Held-out oracle | five method graphs have 25 roots each; rights and stability checks pass | logged correction before target sealing | severe reliability, provenance, or rights failure |
| Independent seals | target gold predates capsule/pool; worker accesses method-only data | repair access tooling before pool execution, with no pool retained | worker saw gold or target information |
| Availability and pressure | TAR=CSAR=1.00 on 10 HRC and preregistered density checks pass | none within the sealed benchmark | mark construction failure; no target/pool/context repair |
| Contract validation | expected corruption signatures and clean specificity pass; no critical miss | versioned defect repair with original results retained | unresolved critical guard defect |
| Utility result | all deterministic fixed-benchmark criteria pass | publish qualified mixed result | main superiority unsupported; use negative-result paper |
| Mechanism result | Full exceeds ablation under equal caps; descriptive diagnostics show no contrary pattern | report the contrary diagnostic pattern and narrow mechanism interpretation | no transition-priority claim if the primary ablation does not improve |
| Release | private-root-absent rerun and allowlist scan pass | redact a derivative and rerun release tests | protected content/key leakage or nonreproducible public pipeline |

The strongest positive Paper 1 conclusion is bounded: on this sealed fixed benchmark, admissibility-constrained transition-bundle projection preserved complete evidential support for more horizon-rare consequential transitions than guard-ERIPCF under equal componentwise budgets, with the direct ablation supporting transition priority as the mechanism. It is not a claim about all chapters, all narratives, visual comprehension, or extraction.

If utility superiority fails but the contract, exact benchmark, independent sealing, corruption validation, and ablation evidence are sound, the strongest defensible negative-result paper characterizes when transition-bundle priority does and does not improve support preservation under dense controlled candidate pools. If canonical route or candidate availability fails, the paper redirects to a benchmark-construction and verifier protocol with no superiority claim. If the guard or exactness core fails, confirmatory publication waits.

## 31. Definition of done for Paper 1 implementation

Implementation is complete only when all of the following are true:

- source-document hashes and precedence are recorded;
- every schema and referential/visibility rule validates on public fixtures;
- deterministic identifiers, canonical JSON/JSONL, hashes, and HMAC synthetic fixtures pass;
- no copyrighted text, exact locator, AGOT span-HMAC value, or secret appears in Git or public artifacts;
- graph closure invariants and reference/bitset equivalence pass;
- all temporal endpoints, entailments, scenario partitions, and deterministic full-set core tests pass;
- the common guard wrapper returns complete structured signatures;
- exact enumeration has the expected masks, matches independent small optima, resumes within one batch, and exports deterministically;
- the mandatory one-context performance pilot and projected compute/storage gates pass;
- ERIPCF uses only frozen root-incidence edges, exact MSTs, deterministic rho/ties, and no hidden Steiner roots;
- development tuning and budget calibration are reviewed and frozen;
- the complete public-development pipeline and workload reforecast pass before held-out opening;
- held-out method graphs, contexts, and method-only canonical routes seal before HRC target selection;
- HRC target and initial support manifest predates pool input and execution;
- pool-worker access records demonstrate that evaluation gold and target metadata were absent;
- pool seal predates target/support availability calculation;
- candidate leakage audit, 18-root pressure checks, TAR, and canonical SAR pass, or construction failure is reported without repair;
- held-out corruptions produce their sealed expected signatures and clean controls pass;
- all paired methods share pool, route, table, budget, guard, response, cost, and presentation hashes;
- all required runs complete deterministically or retain explicit error records;
- one masked novel-path review is completed without method identity and changes scoring only;
- curator-primary, conservative, and permissive annotation scoring is generated;
- fixed-benchmark, bundle, leave-one-out, budget, rarity, violation, and ablation tables are produced automatically;
- every postfreeze correction and rerun is preserved in the deviation ledger;
- a private-root-absent public-domain reproduction succeeds from the final manifest;
- the final release allowlist and copyright audit pass;
- active researcher effort remains within 160 hours, or the project stopped at its declared gate rather than silently shrinking.

No requirement is satisfied merely because a command exits successfully. Each has a scientific witness, test report, human review record, and content hash.

## 32. Targeted change log from the predecessor implementation plan

| Change | Exact correction and reason |
|---|---|
| Target and pool order | split the overloaded freeze into target-gold-first seal, gold-free capsule, independently sealed pool, and postseal availability join; prohibited all repair |
| Access enforcement | added mount-isolated candidate worker, forbidden-field schema, access ledger, canary/symlink/gold-mutation tests, and explicit state machine |
| Structural comparator | renamed exact root-incidence PCST to ERIPCF and explicitly denied equivalence to unrestricted PCST/Steiner retrieval |
| Activation penalty | replaced opaque rho with `max direct edge cost + 1/lcm(caps)`, derived from method-visible data and frozen without gold |
| Exact computation | replaced repeated method-level reasoning with root-mask, route, closure, temporal, response, guard, budget, incidence, forest, and score cache layers |
| Transient storage | retained canonical JSONL but selected resumable per-context SQLite for inner-loop rows, with schemas, indexes, transactions, checkpoints, and ordered export |
| Performance evidence | added mandatory cap-8/unrestricted/warm/resume pilot, resource ceilings, cache expectations, extrapolation, and abort thresholds; removed advance CPU-hour claim |
| Temporal certificates | replaced arbitrary Z3-core minimization with deterministic deletion minimization over the full sorted constraint set and invariance tests |
| Endpoint semantics | fixed closed interval and inclusive horizon conventions, point/shared-boundary behavior, supported predicates, unsupported Allen handling, and boundary tests |
| Evidence routes | removed opaque-ID-based alternative truncation; added one method-visible canonical route for every root plus postselection sensitivity |
| Workload | replaced a 144-hour point estimate with 108/142/206 estimates, annotation/performance reforecast gates, a protected core, and ordered scope reductions |
| Copyright integrity | replaced salted short-span digests with private keyed HMAC, opaque IDs, coarse locators, and an explicit limit on AGOT reproducibility |
| Completion contract | updated traceability, tests, phase order, gates, and definition of done, and bounded the first build turn |

## 33. Final build contract for the next Codex session

This section is normative. `MUST`, `MUST NOT`, and `MAY` have their ordinary requirements meaning.

### 33.1 Authority and precedence

1. The scientific authority is `/home/resort/Downloads/ontology_tmp/SOLO_RESEARCHER_CONTEXTUAL_NARRATIVE_PROJECTION_PLAN_V2.md`, SHA-256 `a8d63400a1184e0f03c1d5fd4bcd0f6e0261cb59c9dbb146081be846cc5d7a2d` at this planning freeze.
2. The historical implementation source is `/home/resort/Downloads/ontology_tmp/IMPLEMENTATION_PLAN_ADMISSIBILITY_CONSTRAINED_PROJECTION.md`, SHA-256 `7c520294ff4e67a4eb88d3a6274afe4cfb2d93e809e37b9b858270f2adaf46bc`.
3. This document, `/home/resort/Downloads/ontology_tmp/IMPLEMENTATION_PLAN_ADMISSIBILITY_CONSTRAINED_PROJECTION_V2.md`, is the authoritative execution specification wherever it explicitly supersedes item 2. Its raw full-file SHA-256 is the digest reported in the final planning-session handoff. A file cannot embed its own ordinary raw SHA-256 without changing that digest; therefore the next session MUST recompute it and compare it with the external handoff, then copy it into the repository's `source_documents` manifest before any edit.
4. If item 1 and this document conflict on scientific scope, item 1 wins unless this document identifies the issue as a targeted V2.1 correction. Engineering choices in this document win over item 2.
5. Any hash mismatch MUST block building until the user or an auditable benign transformation resolves it.

### 33.2 Methodological invariants

The build MUST preserve exactly seven bundles: four held-out AGOT, one public development, one independent public held-out, and one hidden control excluded from efficacy. It MUST preserve 25 roots in each of six natural bundles, no more than 18 hidden roots, 24 families, 32 answerable contexts, 10 HRC targets, 10 probes, 18-root pools, and root caps 4/6/8 unless a pre-heldout preregistered scope reduction explicitly narrows a nonprimary slice.

`A_adm` MUST remain distinct from `A_resp`. Proposed-Full, Proposed-NoTransitionPriority, and guard-ERIPCF MUST remain the protected primary mechanism/comparator set. Native and wrapped frequency/PPR/ERIPCF MUST use the declared horizon filter and common costs. All wrapped methods MUST share pools, canonical routes, closure, guards, response schemas, presentation, and caps. The hidden story MUST NOT contribute to superiority or transfer.

Targets and initial gold supports MUST be sealed before a pool is generated. The candidate worker MUST NOT be able to read the target seal, outcome labels, `GoldSup`, rarity, consequence, final relevance, diagnostic membership, or unrestricted oracle. Pools MUST seal before TAR/CSAR/ASAR are computed. An availability miss MUST become benchmark-construction failure, never repair. Primary rarity MUST be HorizonRarity verified by a human audit. Textual evidence MUST remain attributed and MUST NOT be converted into unrestricted fictional-world truth.

Confirmatory enumeration MUST be exact for all declared root masks and canonical routes. ERIPCF MUST be named accurately, use only direct frozen root-incidence edges, and MUST NOT add hidden Steiner nodes. Full and its ablation MUST differ only in transition priority and receive equal permitted caps. Safety MUST be evaluated through controlled corruption and verifier tests, not credited as selector utility. Conclusions MUST remain fixed-benchmark claims.

### 33.3 Permitted engineering optimizations

The build MAY use bitsets, packed arrays, precomputed single-root closure, memoization, incremental subset sums, deterministic batching, SQLite index or checkpoint changes, canonical streaming export, profiling, and process-level parallelism across contexts. Incremental Z3 use is permitted only if SAT, entailment, and deterministic certificate bytes match the reference fresh-solver results. An internal data structure MAY be replaced when public fixtures, property tests, known optima, and canonical output hashes demonstrate equivalence.

Before design freeze, a permitted optimization requires a reviewed benchmark and regression diff. After design freeze, it is allowed only under the defect protocol and requires versioned full affected reruns. Performance alone never authorizes a scientific change.

### 33.4 Forbidden shortcuts

The build MUST NOT use approximate or incomplete confirmatory enumeration; label ERIPCF as standard PCST; generate a pool with target/gold access; repair a missed target or support; replace a target; rewrite a context after availability; tune a seed; tune budgets, rho, routes, objectives, query templates, or coefficients on held-out data; select evidence routes by opaque root IDs; drop closure dependencies to fit a cap; use possible temporal relations as entailed; use an arbitrary solver core as the final certificate; treat a timeout as infeasibility; duplicate guard or cost logic in a baseline; let an LLM add facts or adjudicate truth; expose AGOT prose, exact locators, HMAC keys, or AGOT span-HMAC values; delete failed runs or corruptions; or change gold after outcomes except through the one masked scoring-only extension.

Codex-generated code MUST receive researcher review. Passing tests MUST NOT be inferred merely from generation.

### 33.5 Mandatory phase order and verification

1. Verify all three source-document identities and initialize repository/environment.
2. Build configuration, strict schemas, canonical serialization/hashing, opaque IDs, visibility propagation, access controls, and public fixtures. Run schema, determinism, referential, access, and leak tests and stop for review.
3. Build graph integrity, closure, temporal/epistemic guards, response families, budgets, certificates, SQLite exact enumeration, ERIPCF, baselines, objectives, and corruptions in that dependency order. Each phase MUST pass its public reference, invariant, and known-optimum tests.
4. Run the one-context performance pilot and complete public-development/hidden pilot. Record machine and human rates. Calibrate budgets and ERIPCF parameters only on development.
5. Apply the scope ladder if needed, preregister, and create `DesignFreezeManifest`.
6. Only after the held-out-opening gate, annotate and freeze held-out method graphs, then freeze contexts and compile/seal method-only canonical routes.
7. In gold-only storage, select and seal all HRC target IDs, initial supports, outcomes, rarity/consequence, diagnostics, and alternatives.
8. Compile method-only capsules and verify the forbidden-field/access boundary. Run the isolated pool worker, which emits a target-free worker manifest. After it exits, the controller attaches the prior target commitment as chronology metadata and seals pools.
9. Only after both independent seals exist, run the read-only availability and pressure join. Enter `BENCHMARK_READY` or `BENCHMARK_CONSTRUCTION_FAILED`; the failure branch stops primary evaluation and has no repair path.
10. On the ready branch, seal corruption instances and ERIPCF instances, then the final `ExperimentManifest`; verify common-table fairness.
11. Execute all required systems, preserve every result, conduct one masked novel-path review, rescore once, and generate tables.
12. Run the private-root-absent reproduction, release allowlist and copyright audit, then seal the package.

After every numbered phase, the researcher MUST review a diff, public test report, invariant/coverage report, active minutes, access report when relevant, and artifact hashes. A phase cannot advance merely because code runs.

### 33.6 Conditions for opening data and beginning evaluation

Held-out data MUST remain unopened until source hashes match; schemas and access tests pass; endpoint and deterministic-core tests pass; exact search matches independent small instances; the performance/resume pilot passes or an allowed optimization cycle succeeds; the complete development pipeline passes; annotation and workload forecasts satisfy the 145-hour expected and 160-hour conservative-core gates; budgets, generator, route policy, ERIPCF grid and rho rule, objectives, ties, corruption templates, response schemas, and seeds are frozen; a held-out exposure order is recorded; and preregistration is committed.

Full evaluation MUST NOT begin until held-out method graph, context, and canonical-route seals exist; target gold, worker output, and pool seals exist in the required order; candidate-worker access logs show method-only reads; target and canonical support availability and selection pressure pass or the benchmark is declared failed; held-out corruptions and ERIPCF instances are sealed; the final manifest validates; and all guard-wrapped comparisons share identical scientific hashes.

### 33.7 Completion

Completion requires every item in Section 31, including exact mask coverage, deterministic SQLite resume/export, deterministic temporal core invariance, guard-ERIPCF freeze, corruption signatures, independent seal chronology, masked path review, uncertainty scores, automatic paper tables, copyright-safe release, public-domain reproduction, and effort within the ceiling. A construction failure or negative result can complete a narrower paper only when it is preserved and interpreted under Section 30; it cannot be relabeled success.

### 33.8 Exact scope of the first build turn

The next Codex session's first build turn MUST do only the following:

1. Recompute and record the three source-document hashes, stopping on mismatch.
2. Initialize the repository, Python 3.12 configuration, `pyproject.toml`, lock workflow, quality settings, public documentation skeleton, and copyright-safe `.gitignore`/release allowlist skeleton.
3. Implement the common record envelope and strict typed schemas, including visibility and seal-state models, without scientific inference logic.
4. Implement canonical JSON/JSONL serialization, SHA-256 artifact hashing, deterministic UUID namespaces and opaque identifiers, and keyed-HMAC utilities tested only on synthetic text.
5. Implement bundle visibility propagation and an initial deny-by-default access layer for public, restricted-derived, private-raw, and public-commitment classes.
6. Generate JSON Schemas and public synthetic valid/invalid fixtures.
7. Add and run unit tests for deterministic serialization/hashing/IDs, strict extra-field rejection, referential and visibility errors, bundle access denial, HMAC key/normalization mismatch, and public-export rejection of private fields including a synthetic stand-in for an AGOT span-HMAC value.
8. Produce a Phase 2 acceptance report with hashes, tests, coverage, researcher-review checklist, and active minutes, then stop for human review.

The first build turn MUST NOT ingest or search copyrighted text; create real annotations, target manifests, candidate pools, or gold; implement closure, temporal solving, enumeration, ERIPCF, selectors, or experimental metrics; tune any parameter; open held-out public material; or run an experiment. Its purpose is to establish the trustworthy typed and access-controlled foundation on which all later scientific behavior depends.
