# Implementation Plan for Admissibility-Constrained Narrative Projection

This document translates `SOLO_RESEARCHER_CONTEXTUAL_NARRATIVE_PROJECTION_PLAN_V2.md` into a reproducible solo-researcher implementation program. It specifies software and research-execution work only. It does not annotate the study corpus, implement the system, run an experiment, or modify the methodology files.

## 1. Executive implementation decision

Paper 1 should be implemented as a small, deterministic Python research package operating on manually validated, episode-bounded oracle graphs. The package will compile typed records into a finite root-and-atom graph, enumerate every permitted subset of at most 18 selectable roots, compute semantic closure and evidence alternatives, verify admissibility and response feasibility, and select the exact lexicographic optimum. One immutable high-cap candidate table per answerable context, with exact low- and medium-budget filters, will be shared by Proposed-Full, Proposed-NoTransitionPriority, and all guard-wrapped baselines. This shared table is the central fairness mechanism.

The required implementation will use six natural-text bundles and one control bundle exactly as V2 specifies: four held-out AGOT bundles, one public-domain development bundle, one independent public-domain held-out bundle, and one researcher-authored hidden control. The target annotation size is exactly 25 defensible selectable roots per natural bundle, or 150 roots total; this lies within V2's range of 25 to 27 and reduces the implementation forecast. Each primary candidate pool contains exactly 18 roots. There are 24 base query families, 32 answerable contexts, 10 primary horizon-rare consequential contexts, 10 feasibility probes, three root caps of 4, 6, and 8, three core baseline families, and one direct objective ablation.

The primary candidate generator will be deterministic and method-independent. It may use only frozen typed context fields and the horizon-eligible annotated graph. It will not use consequence, rarity, gold support, final relevance judgments, diagnostic cells, or system outcomes. The quota-balanced 18-root construction in V2 becomes an optional controlled diagnostic, not the primary pool. Target availability is measured after pools are frozen and is never repaired with gold knowledge.

Paper 1 requires no LLM, GPU, human study, extraction pipeline, graph database, interactive visualization, or web service. The required run matrix is reduced from V2's exhaustive secondary matrix while preserving every confirmatory comparison and scientific control. Planned researcher effort is 144 hours, with 16 hours of contingency under the 160-hour hard ceiling. A complete public-development pilot occurs before any held-out annotation and can trigger further removal of optional work.

## 2. Scientific traceability to the V2 research plan

The V2 file is the scientific source of truth. The implementation shall record its SHA-256 digest in every frozen experiment manifest. At planning time that digest is `a8d63400a1184e0f03c1d5fd4bcd0f6e0261cb59c9dbb146081be846cc5d7a2d`; it must be recomputed before repository initialization and any discrepancy investigated.

The following scientific invariants are not negotiable:

1. Extraction and projection remain separate. Confirmatory selection starts from manually validated oracle records.
2. Semantic admissibility and query responsiveness are nested but different properties.
3. Guard-wrapped methods share eligibility, closure, verifiers, response schemas, evidence choices, budgets, and candidate pools.
4. Proposed-Full differs from Proposed-NoTransitionPriority only by the first objective coordinate.
5. Guard-PCST is the primary comparator. Frequency and PPR remain secondary.
6. HorizonRarity, consequence labels, final relevance, diagnostic categories, and `GoldSup` never enter selection.
7. The primary rarity claim uses manually verified mentions over the available pre-horizon text, not only the episode.
8. Safety is validated with corruptions and verifier tests. It is not credited as an objective-level advantage over a guard-wrapped baseline.
9. All caps are componentwise, identical across methods, and applied after semantic closure and presentation mapping.
10. Confirmatory search is exact only within the frozen 18-root universe. There is no scalability claim.
11. The held-out public work is not used for design, tuning, query-template revision, budget calibration, or verifier repair.
12. The hidden narrative is a control and test source, not evidence of external validity.
13. Results support a fixed-benchmark case claim only. No population inference, reader-comprehension claim, or unrestricted fictional-world truth claim is permitted.

### Scientific traceability matrix

| Scientific claim | Responsible module | Frozen input artifact | Output artifact | Required test | Evaluation metric | Consequence of failure |
|---|---|---|---|---|---|---|
| Horizon safety | `verification/horizon.py` | contexts, anchors, closure graph | structured horizon verifier results | boundary, flashback, label-leak, and post-horizon fixtures | leak count and severity | Contract gate fails; no utility superiority claim |
| Evidence grounding | `verification/evidence.py` | assertions, alternatives, anchors, proof rules | evidence verdict and witness | missing, mismatched, and alternative-anchor fixtures | evidence precision/completeness; unsupported-claim rate | Contract gate fails |
| Epistemic fidelity | `verification/epistemic.py` | assertions, propositions, response rendering | status/polarity/holder verdict | rumor promotion, holder deletion, disputed-status fixtures | epistemic violations and rumor-promotion errors | Contract gate fails |
| Temporal satisfiability | `temporal/solver.py` | normalized temporal constraints | SAT status, entailments, minimized unsat core | Allen-relation oracle and contradiction fixtures | temporal violation count; solver agreement | Contract gate fails or temporal claim removed |
| Identity safety | `verification/identity.py` | entity decisions, role dependencies | merge/reference verdict | mistaken-merge and unresolved-identity fixtures | identity-error count | Contract gate fails |
| Support completeness | `verification/support.py` | response schema and method-visible proof rules | `GuardSup` proof witness | delete-one-required-atom fixtures | guard-support pass; complete gold-support recall separately | Contract or utility interpretation fails |
| Budget compliance | `costing/budget.py` | closure, fixed templates, budget vector | exact component cost vector | coordinate-boundary and shared-atom tests | budget violation count; realized costs | Output inadmissible |
| Exactness | `projection/exhaustive.py` | 18-root pool, alternatives, feasible table | enumeration ledger and winner | naive small-instance and independent maximizer checks | mask/variant coverage; optimum audit | Remove zero-gap claim; stop confirmatory runs |
| Transition-priority mechanism | `projection/objectives.py` | common feasible table and frozen context features | Full and NoTransitionPriority selections | objective-coordinate regression fixture | paired HRC difference at equal caps | No mechanism claim if ablation matches Full |
| Horizon rarity | `annotation/mention_audit.py` | search packets and reviewed passage candidates | mention ledger and counts | duplicate, horizon, occurrence-identity, delayed re-audit checks | HorizonCount, category stability, mention-decision F1 | Replace horizon-rare with locally singleton/query-necessary |
| Fixed-benchmark superiority | `experiments/decision_gates.py` | sealed results for 10 primary contexts | gate table | hand-computed miniature benchmark | HRSC difference, wins/losses, bundle consistency | H1 unsupported; publish boundary result if otherwise valid |
| Public-domain transfer | `reporting/slices.py` | sealed independent public bundle results | separate public transfer table | split-isolation test | public-bundle paired difference | Restrict conclusion to AGOT cases |
| Annotation uncertainty | `metrics/uncertainty.py` | licensed alternative oracle states | primary, conservative, permissive scores | two-state and three-state fixtures | score envelope and decision stability | Qualify or withdraw interpretation-sensitive claim |

Every module name in this matrix is a planned responsibility, not an instruction to create the software in this session.

## 3. Pre-implementation methodological decisions to freeze

These decisions preserve V2's hypothesis while resolving details that otherwise permit held-out discretion. Items marked "V2.1 correction" should eventually receive a small textual correction in the research plan. The correction is documentary only and must not be made during this task.

### 3.1 Candidate-pool independence

**V2 requirement:** every answerable context has an 18-root controlled-dense pool assembled with quotas such as focal support, rare irrelevant structures, and incomplete support.

**Ambiguity:** several quotas require gold relevance, rarity, or support knowledge. Using them to build the only primary pool can perform part of the target selection before a method runs.

**Frozen interpretation:** the primary pool is created by a deterministic context-visible generator. For a context, it first horizon-filters all already annotated roots from the same work. It assigns each eligible root a coarse score using only focal-entity overlap, declared root or event type, story/discourse-window compatibility, assertion holder, and response-schema compatibility. It selects exactly six highest-scoring roots, with stable-ID ties, then fills to exactly 18 by a manifest-seeded hash ordering of all remaining eligible roots. A deterministic type-balancing pass may exchange roots only to ensure that event, transition, and assertion roots are represented; it cannot inspect any outcome label. All changes and scores are logged.

After the pool is sealed, outcome annotation measures target and complete-support availability. A failed availability or density gate is reported as a benchmark-construction failure; the held-out pool is not repaired. The V2 quota-balanced pool is retained only as an optional secondary challenge set. **V2.1 correction: required**, because V2 currently describes the quota pool as primary.

### 3.2 Budget derivation

**V2 requirement:** root caps are 4, 6, and 8; other caps are derived from public development task minima, while a numerical table is presented as an envelope.

**Ambiguity:** the table can be mistaken for already calibrated values, and "median" is underspecified for four task types.

**Frozen interpretation:** root caps stay analytically fixed at 4, 6, and 8. One canonical public-development base context from each of the four task schemas is used. At root cap 8, exact enumeration returns all Pareto-minimal admissible response cost vectors. For each task, choose the vector minimizing, in order, roots, evidence cues, assertions, transitions, total remaining components, and stable atom IDs. The low non-root cap is the componentwise upper median, meaning the third sorted integer among the four task vectors. Medium is their componentwise maximum. High equals medium plus the componentwise upper-median marginal cost of one complete development transition/evidence bundle. All values are integers. The published V2 numbers are provisional maximum envelopes, not empirically derived values.

Calibration writes `config/frozen/budgets.v1.yaml`, a review report, and a canonical-JSON digest. If a derived coordinate exceeds the V2 envelope, development stops for a protocol-level review; it is not clipped and held-out data are not consulted. Nearby sensitivity budgets are generated mechanically before freeze. **V2.1 clarification: advisable.**

### 3.3 Baseline parameter fairness

**V2 requirement:** guard-PCST is the strongest comparator, but its coefficients and structural details are not fully fixed.

**Frozen interpretation:** development-only guard-PCST tuning uses a 12-cell grid: relevance-prize multiplier in `{1,2}`, response-slot prize in `{1,2}`, and connector-cost multiplier in `{1/2,1,2}`. Exact rational arithmetic is used. Each setting runs at medium budget on all six public-development contexts. Selection is lexicographic by: number of responsive development contexts; complete accepted support count; summed frozen relevance grade; lower realized normalized cost; and lexical parameter tuple. Evidence-only atoms receive no prize. Event, transition, and assertion roots use the same generic prize rule, with no transition-completeness bonus. Forest handling is fixed, not separately tuned. The winning tuple is frozen before held-out material is opened.

PPR uses damping 0.85. Personalization is uniform over roots matching focal entities, task-compatible types, or the specified holder; if there is no match it is uniform over all eligible roots. The root graph is an undirected weighted projection of frozen explicit semantic links and shared non-evidence dependencies. Rows are normalized; dangling mass returns to the personalization vector. Stable identifiers break ties.

Frequency is the number of distinct, verified, in-horizon EvidenceAnchor records presenting that particular root occurrence or assertion. It is the frequency baseline's raw count, not the named HorizonRarity category, consequence, a diagnostic category, or broad event-type frequency. Ties use stable identifiers. Native and wrapped forms use identical frequency counts, and other selectors cannot read it. **V2.1 clarification: advisable.**

### 3.4 Direct-ablation cost fairness

**V2 requirement:** Section 19 gives both methods identical caps but additionally demands that Full use no more realized cost on every coordinate.

**Ambiguity:** lexicographic transition protection may legitimately use more of an allowed coordinate. Realized-cost dominance is not required for a fair equal-cap comparison and can reject the mechanism by definition.

**Frozen interpretation:** Full and NoTransitionPriority receive exactly the same vector cap. Every realized coordinate and efficiency ratio is reported. No output-dependent rebudgeting occurs. Low, medium, and high results provide primary cost sensitivity; a matched-realized-cost analysis is optional and cannot replace the equal-cap result. **V2.1 correction: required.**

### 3.5 Isolated verifier corruptions

**V2 requirement:** 45 held-out corruptions plus 10 clean controls, with individual guard-removal runs.

**Ambiguity:** removing one piece of evidence can correctly trigger evidence, support, and response predicates, so not every mutation is logically isolated.

**Frozen interpretation:** every `CorruptionCase` stores an expected full predicate-failure signature. An isolated case must pass every non-target guard when the target guard is disabled. When logical overlap is unavoidable, the case is explicitly marked `multi_guard` and its full expected signature is tested. Development cases are authored on the public-development and hidden bundles. Held-out mutation templates, expected rules, and counts are frozen and hashed before being instantiated on held-out clean projections. A held-out miss remains a confirmatory failure. A repair becomes a new development regression test and cannot erase the sealed result. **V2.1 clarification: advisable.**

### 3.6 Annotation uncertainty

**V2 requirement:** unresolved and disputed interpretations remain represented, but scoring sensitivity is not operationally complete.

**Frozen interpretation:** an `AnnotationAlternative` group stores one curator-primary value and one or more licensed alternatives before system execution. At most three outcome-relevant groups per query and two alternatives per group are admitted to Paper 1, yielding at most eight oracle interpretations. System selections and candidate pools remain fixed. The scorer reports curator-primary success, conservative success under every licensed interpretation, and permissive success under at least one. Selection is never rerun against an alternative oracle. **V2.1 clarification: advisable.**

### 3.7 Feasibility calibration and scope reduction

**V2 requirement:** work must stop below 160 hours, but implementation labor was estimated only coarsely.

**Frozen interpretation:** one complete public-development pilot must cover annotation, horizon audit, contexts, candidate pools, closure, every verifier, budget calibration, exact search, guard-PCST tuning, Full, the ablation, metrics, and report generation. The researcher logs minutes by activity. The forecast is:

```text
projected_total_hours = spent_hours
                      + remaining_fixed_hours
                      + max(planned_root_minutes, observed_root_minutes) * remaining_roots / 60
                      + max(planned_query_minutes, observed_query_minutes) * remaining_contexts / 60
                      + max(planned_pool_minutes, observed_pool_minutes) * remaining_pools / 60
                      + max(planned_audit_minutes, observed_audit_minutes) * remaining_audit_passes / 60
                      + max(planned_reannotation_minutes, observed_reannotation_minutes) * remaining_reannotation_roots / 60
                      + max(planned_path_minutes, observed_path_minutes) * forecast_distinct_paths / 60
```

Proceed unchanged at a forecast of at most 145 hours. Between 145 and 160, remove optional challenge pools, paraphrases, native probe runs, secondary low/high runs, one-coordinate sensitivity budgets, and nonessential figures, in that order. A forecast above 160 triggers a protocol revision before held-out annotation. The public held-out bundle, HorizonRarity audit, guard-PCST, NoTransitionPriority, primary verifier corruptions, and 10 primary contexts cannot be cut. **V2.1 addition: advisable.**

### 3.8 Exact PCST terminology

V2 calls the structural comparator PCST while adapting it to semantic roots and a common response-feasible table. The implementation will call it **exact root-incidence PCST** and define its score completely in Section 16. It is an exact prize-collecting connected-root or forest objective over the 18 candidate roots, not an invocation of an approximate PCST package. No unselected semantic root may appear as an uncharged Steiner node. A V2.1 clarification would improve terminology but does not change the comparator.

## 4. Recommended technology stack

The primary stack is deliberately small.

| Concern | Choice | Decision rationale |
|---|---|---|
| Language | Python 3.12, exact patch pinned at repository creation | Mature scientific ecosystem, readable solo maintenance, adequate for at most 18 roots |
| Typed records | Pydantic v2 models in strict mode, frozen after validation | Field validation, discriminated unions, referential prechecks, and generated JSON Schema from one semantic source |
| Semantic graph | NetworkX `MultiDiGraph` for authoring and inspection; compiled integer-index arrays and bitsets for enumeration | NetworkX preserves typed parallel relations; compact arrays avoid object overhead in exact search |
| Temporal reasoning | `z3-solver` using quantifier-free linear rational arithmetic | Exact strict inequalities, disjunctions, entailment checks, and unsatisfiable witnesses without floating-point ambiguity |
| Exact subset search | Python integers as root and atom bitsets, `fractions.Fraction` for objective values | Complete 18-bit enumeration is transparent and independently auditable |
| Configuration | YAML parsed with pinned PyYAML safe loading; canonical JSON for hashing | YAML is readable; safe loading rejects executable tags; canonical JSON removes serialization ambiguity from freezes |
| Records/results | JSONL for typed records and projections; CSV for flat metrics and ledgers | Both are diffable and broadly reproducible; Parquet is unnecessary at this scale |
| Tables/figures | Python `csv` plus Matplotlib | The standard library is sufficient for small metric tables; Matplotlib provides deterministic static figures without an interactive stack |
| CLI | Typer with descriptive subcommands | A discoverable local command surface while keeping one process and no service layer |
| Tests | pytest, Hypothesis, coverage.py | Example, integration, property, and regression tests in one conventional framework |
| Static quality | Ruff and mypy | Centralized style checks and type consistency for high-risk scientific logic |
| Environment | `.python-version`, `pyproject.toml`, `uv.lock`, `uv lock --check`, and `uv sync --locked` | Verify that the checked-in lock matches project metadata, then install exactly that lock without containers |

Dependency versions, including Python, Z3, NetworkX, Pydantic, and test tools, must be locked, not loosely ranged, at the development freeze. Pydantic's generated JSON Schemas are committed. Scientific records are validated with strict coercion disabled. Objective weights and connector costs are integers or rational strings, never binary floating point.

Every boundary model uses the equivalent of `ConfigDict(strict=True, frozen=True, extra="forbid")`. Cross-record semantics remain in the graph validator so field errors and scientific referential errors have stable, separate categories.

JSONL is the implementation exchange format, not a change to V2's serialization-neutral theory. The typed model could later be serialized as RDF event/assertion structures with temporal and provenance vocabularies, but RDF export is not part of Paper 1.

A graph database would add migration, query, and deployment work without improving an exhaustive 18-root experiment. Distributed processing and GPU libraries are unnecessary. A frontend, annotation platform, API, container orchestration layer, or production database would consume the solo budget and create no Paper 1 evidence. A small structured-file workflow and local commands are sufficient. Relevant primary documentation includes the [Pydantic model documentation](https://docs.pydantic.dev/latest/concepts/models/), [NetworkX MultiDiGraph documentation](https://networkx.org/documentation/stable/reference/classes/multidigraph.html), [Z3 Python API](https://z3prover.github.io/api/html/namespacez3py.html), and [uv project locking and synchronization documentation](https://docs.astral.sh/uv/concepts/projects/sync/).

## 5. Repository structure

The repository should use the following complete minimal tree. Directories marked `LOCAL ONLY` must be ignored by Git; restricted derived data may instead live in a separately access-controlled repository whose public repository contains only hashes and schemas.

```text
narrative-projection/
|-- .python-version
|-- pyproject.toml
|-- uv.lock
|-- README.md
|-- LICENSE
|-- CITATION.cff
|-- .gitignore
|-- src/
|   `-- narrative_projection/
|       |-- __init__.py
|       |-- cli.py
|       |-- models/
|       |   |-- common.py
|       |   |-- narrative.py
|       |   |-- annotation.py
|       |   `-- experiments.py
|       |-- graph/
|       |   |-- compile.py
|       |   |-- integrity.py
|       |   `-- closure.py
|       |-- temporal/
|       |   |-- normalize.py
|       |   `-- solver.py
|       |-- verification/
|       |   |-- common.py
|       |   |-- evidence.py
|       |   |-- horizon.py
|       |   |-- epistemic.py
|       |   |-- temporal.py
|       |   |-- identity.py
|       |   |-- provenance.py
|       |   |-- support.py
|       |   `-- budget.py
|       |-- candidates/
|       |   |-- primary.py
|       |   |-- controlled_challenge.py
|       |   `-- leakage_audit.py
|       |-- costing/
|       |   |-- presentation.py
|       |   `-- budget.py
|       |-- projection/
|       |   |-- feasible_table.py
|       |   |-- exhaustive.py
|       |   |-- objectives.py
|       |   `-- certificates.py
|       |-- baselines/
|       |   |-- frequency.py
|       |   |-- personalized_pagerank.py
|       |   `-- root_incidence_pcst.py
|       |-- annotation/
|       |   |-- compile.py
|       |   |-- mention_audit.py
|       |   |-- reannotation.py
|       |   `-- freeze.py
|       |-- experiments/
|       |   |-- manifest.py
|       |   |-- runner.py
|       |   |-- corruption.py
|       |   `-- decision_gates.py
|       `-- reporting/
|           |-- metrics.py
|           |-- uncertainty.py
|           |-- tables.py
|           `-- figures.py
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
|   |-- safe_restricted_manifests/         # hashes and legally safe metadata only
|   `-- README.md
|-- .local/                                # LOCAL ONLY and ignored
|   `-- paths.yaml                         # logical corpus IDs -> external restricted roots
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
|   |-- frozen/
|   `-- release/
|-- preregistration/
|-- audit/
|   |-- change_ledger/
|   |-- runtime_ledger/
|   `-- leakage_ledger/
|-- artifacts/
|   |-- runs/                               # transient details ignored unless frozen and safe
|   |-- metrics/
|   |-- tables/
|   |-- figures/
|   `-- release/
|-- docs/
`-- scripts/
```

The external location referenced by `.local/paths.yaml` has the fixed conceptual tree `private_agot_text/`, `private_agot_locators/`, `restricted_agot_records/`, `transient_candidates/`, and `private_run_logs/`. It is never nested in the Git worktree. Git should contain source, tests, schemas, the handbook, templates, public-domain and eventually releasable hidden-control material, frozen public configurations and hashes, preregistration, safe aggregate results, and release documentation. It must ignore `.local/`, raw AGOT text, local locators that expose licensed editions, restricted reconstruction-ready records, credentials, `.env` files, virtual environments, Python/tool caches, solver caches, temporary candidate passages, raw logs containing text, unredacted annotation exports, and large replaceable run caches. A pre-release leakage scan checks paths, forbidden hashes, and sampled private n-grams. No credential is required by Paper 1.

## 6. Data-access and copyright boundaries

Data access has four explicit visibility classes:

| Class | Contents | Git/release policy |
|---|---|---|
| `PUBLIC` | Code, schemas, public-domain text/annotations, safe fixtures, final hidden story after unsealing, safe aggregate results | Versioned in public Git |
| `RESTRICTED_DERIVED` | AGOT oracle records, detailed support paths, internal aliases, locators that enable reconstruction | Local access-controlled storage or restricted repository; only approved derivatives released |
| `PRIVATE_RAW` | Lawfully acquired copyrighted text and full-text search indexes | Local only, ignored by Git, never copied into test fixtures or logs |
| `PUBLIC_HASH_ONLY` | SHA-256 digests, opaque IDs, counts, safe chapter/section anchors approved for release | Public Git and manifest |

Raw text ingestion is read-only. A local corpus locator maps a `work_version_id` to a path and edition metadata. That locator is absent from public manifests. EvidenceAnchor records for AGOT store an opaque passage ID, discourse index, a salted local span digest, span length, and a lawful coarse locator; they do not store quoted text. Public-domain anchors may include the text needed for reproducible fixtures. Unit and integration tests use public-domain or synthetic passages only.

The compiler must propagate the most restrictive visibility of its inputs. A projection containing any restricted atom remains restricted even if its labels are abstract. Public reporting operates from a redacted metric table and opaque release IDs. A release validator fails if an output contains a private path, a raw passage field, an unapproved alias, an unredacted rationale, a post-horizon descriptor, or an AGOT text fragment above a conservatively small configured threshold. Converting prose into graph records is not presumed to authorize redistribution.

## 7. Typed data schemas

All records use a common envelope: `record_id`, `schema_version`, `record_version`, `scope_id`, optional `work_id`, zero or more `bundle_ids`, `status` (`draft`, `reviewed`, `frozen`, `retired`), `visibility`, `provenance_ids`, `created_at`, `modified_at`, and `content_hash`. Narrative records require `work_id`; global schemas, budgets, and manifests use a study `scope_id` and explicit `not_applicable` work scope. Frozen scientific records are immutable. Optional values use explicit `unknown`, `unresolved`, `disputed`, or `not_applicable` states rather than missing strings. References must resolve within the frozen graph version and must not cross works unless the schema explicitly permits it.

### 7.1 Narrative and annotation records

| Schema | Required fields | Optional fields and enumerations | Principal validation and visibility rules |
|---|---|---|---|
| `Entity` | `entity_id`, `entity_kind`, `identity_status`, `internal_label`, alias records | kinds: person, group, place, object, office, other; status: resolved, disputed, unresolved; candidate alternative IDs | aliases are unique within an edition and time scope; a disputed merge cannot be silently resolved; AGOT labels are restricted |
| `EventOccurrence` | `event_id`, controlled `event_type`, participant-role pairs, `scenario_scope`, at least one mention anchor, temporal-object ID | optional trigger/transition links and occurrence-equivalence decisions; scopes: actual_story, presented_occurrence, dream, prophecy, hypothetical | participant IDs resolve; repeated mentions link to one occurrence only under the frozen identity rule; evidence does not itself assert world truth |
| `State` | `state_id`, bearer, dimension, typed value, scenario scope, validity temporal object | status may be presented, attributed, disputed, unknown; dimensions are location, possession, role, allegiance, office, relationship, knowledge, survival, goal, other | bearer and referenced value resolve; state identity is bearer-dimension-value-validity specific |
| `Transition` | `transition_id`, bearer, dimension, before-state, after-state, triggering event or explicit unknown, assertion/evidence route | consequence packet ID and rarity-audit ID are restricted outcome fields, never method-visible | before and after share bearer/dimension and differ unless an explicit unknown boundary is licensed; dependencies are complete |
| `Proposition` | `proposition_id`, normalized predicate, ordered typed arguments, scenario scope | normalized abstract gloss; content alternatives | identity is content, independent of holder; no unrestricted truth flag is allowed |
| `Assertion` | `assertion_id`, proposition, holder/source, epistemic mode, polarity, curator status, discourse/revelation position, evidence alternative IDs | modes: presented_as_established, observed, believed, reported, rumored, remembered, dreamed, prophesied, inferred, denied, refuted, disputed, hypothetical, unknown; polarity: positive, negative, unresolved; curator status: supported_under_policy, refuted_under_policy, both, undetermined | holder may be a typed narrator/source identifier; status cannot be promoted by evidence presence alone; nested reports use linked assertions |
| `EvidenceAnchor` | `anchor_id`, work version, locator, discourse index, span digest, span length, rights status | public text only for public-domain/synthetic records; story-time hint is optional | locator lies within edition; digest and offsets agree locally; AGOT text field is prohibited |
| `MentionAuditRecord` | `audit_id`, target transition, candidate anchor, search-packet ID, classification, count contribution, review pass, rationale code | classifications: same_depiction, same_report, same_recollection, same_summary, consequence_only, same_type_other_occurrence, repeated_relation, ambiguous, nonmatch, post_horizon, duplicate | only first four add one mention; duplicates and post-horizon hits add zero; AGOT rationale text remains restricted |
| `ProvenanceRecord` | `provenance_id`, activity type, responsible agent, timestamp, input IDs/hashes, protocol/tool version, derivation type | safe note, issue ID | derivation graph must be acyclic at version level; Codex assistance is recorded as tool use but never substitutes for researcher verification |
| `TemporalConstraint` | `constraint_id`, coordinate system, left object, relation/allowed relation set, right object or bound, assertion status, scenario scope | systems: story, state_validity, discourse, revelation; relations: before, equal, overlaps, intersects, during, contains, unknown, allowed_set, lower_bound, upper_bound | after is normalized to converse-before; relation semantics are Section 12 semantics; evidence/provenance required for annotated constraints |
| `SupportStructure` | `support_id`, query/claim or response slot, atom IDs, root IDs, evidence route, support role, sufficiency status | roles: method_visible_rule_instance, gold_reference, posthoc_accepted, decoy; statuses: full, partial, valid_redundant, unsupported, temporally_invalid, epistemically_invalid, post_horizon, irrelevant | gold and post-hoc roles are outcome-only; minimality is recorded as delete-one results, not assumed |
| `SelectableRoot` | `root_id`, root kind, payload record ID, base atom IDs, method-visible feature record, redundancy class, display-template ID | kinds: event, transition, assertion, contextual_relation; evidence alternatives; structural-centrality value only if baseline-visible and development-frozen | no consequence, HorizonRarity, `GoldSup`, final relevance, or diagnostic field may occur in the method-visible serialization |
| `RootDependency` | source root/atom, target atom, dependency type, mandatory flag, horizon-nonincreasing claim | types: identity, participant_role, before_state, after_state, proposition, assertion_status, evidence, provenance, temporal, support, display; alternative-group ID | mandatory graph is finite; alternative dependencies are resolved by `eta`; claimed horizon inheritance is validated |
| `EvidenceAlternative` | `alternative_id`, root or answer claim, evidence anchor IDs, support atom IDs, choice group, licensing status | incompatibility IDs and rationale | at most two primary roots per pool may have choices and at most two choices each; all atoms obey the horizon |
| `AnnotationAlternative` | `alternative_group_id`, target record/field, curator-primary value, licensed values, rationale/evidence, freeze timestamp | interpretation labels and reconciliation note | at most three outcome-relevant groups per query and two options per group; alternatives freeze before selections |

### 7.2 Query, projection, and experiment records

| Schema | Required fields | Optional fields and enumerations | Principal validation and visibility rules |
|---|---|---|---|
| `TypedQueryContext` | `context_id`, split, natural-language or safe abstract query, task schema, focal entities/types, horizon, budget/detail level, response-schema ID | story window, discourse window, epistemic holder, ambiguity alternatives; flags for primary, secondary, probe, paraphrase | only minimal context fields are method-visible; held-out outcome labels are stored separately |
| `ResponseSchema` | `response_schema_id`, task type, named slots, slot cardinality/types, qualification requirements, evidence requirements, GuardSup rule version | licensed unknown/disputed response forms and permitted abstention classes | slots cannot contain gold target IDs; four task types match V2 |
| `CandidatePool` | `pool_id`, context, generator version, eligible-universe hash, selected 18 root IDs, seed, method-visible score trace, pressure statistics, leakage-audit hash, frozen digest | controlled-challenge category assignments in a separate restricted outcome file | selected IDs are unique, same-work, horizon-safe at root level; no label-leak fields |
| `BudgetVector` | `budget_id`, level, nine integer caps, presentation grammar version, derivation record and freeze hash | sensitivity parent and changed coordinate | all main-study caps are positive; low is no greater than medium, which is no greater than high componentwise |
| `Projection` | `projection_id`, run ID, selected-root mask and IDs, evidence choices, closed atom IDs/hash, response-slot fills, realized cost vector, objective tuple, verifier-result IDs, result status | deterministic display record and abstention-certificate ID | selected roots belong to pool; closure hash recomputes; no cap or guard result is trusted from serialized input |
| `AbstentionCertificate` | `certificate_id`, class, graph/context/budget hashes, enumeration-table hash, machine witness, public redacted explanation, validation status | minimal failed-guard sets, temporal core, Pareto budget deltas | classes: INVALID_CONTEXT, CONTENT_ABSENT, CONTRACT_BLOCKED, BUDGET_BLOCKED, ERROR_OR_UNKNOWN; last is never scored correct |
| `CorruptionCase` | `case_id`, split, base-projection hash, signed mutation delta, clean/corrupt flag, target guards, expected full signature, expected non-target passes, severity, case hash | allowed multi-guard rationale, seed | sealed case cannot change after verifier output; private content remains referenced by opaque IDs |
| `ExperimentManifest` | `manifest_id`, V2 hash, code revision, environment-lock hash, all graph/config/query/pool/budget/verifier/solver/corruption hashes, seeds, split IDs, freeze timestamp | protocol-deviation ledger link | canonical and append-only; no held-out outcome is an input to a development parameter hash |
| `RunRecord` | `run_id`, manifest, method and parameter hashes, context, pool, budget, start/end, runtime, peak memory, status, projection/certificate ID, log hashes | machine description and retry lineage | retry cannot overwrite a run; deterministic reruns receive new IDs linked to the original |
| `MetricRecord` | `metric_record_id`, metric name/version, run/context/bundle/slice, numerator, denominator, value, scoring policy, input hashes | uncertainty interpretation ID, confidence flag when denominator zero | scoring policies: curator_primary, conservative, permissive; aggregate rows retain source run IDs |

JSON Schema validates field shapes. A second graph-level validator enforces referential integrity, type-compatible links, split boundaries, visibility propagation, dependency completeness, and the absence of outcome-only fields from method-visible exports.

The storage API exposes four noninterchangeable containers. `OracleBundle` is the curator's complete episode record. `MethodBundle` contains only method-visible records, dependencies, contexts, proof schemas, candidate pools, and permitted baseline features. `EvaluationGold` contains targets, accepted supports, rarity, consequence, diagnostic cells, and final relevance; only evaluation modules may import it. `ProjectionResult` contains selections, closure, evidence choices, costs, verifier results, and certificates. Selector function signatures accept only `MethodBundle`, `TypedQueryContext`, `BudgetVector`, and a frozen method configuration. Static import tests and run-time access ledgers enforce this firewall.

## 8. Identifier and versioning strategy

Identifiers must be stable, opaque, and non-spoiling. Internal IDs use UUIDv5 over a project namespace and canonical tuple `(artifact_type, work_version_id, bundle_id, opaque_local_key)`. The local key is assigned from a ledger and never contains a character name, event label, chapter outcome, or diagnostic class. Public release IDs use a second UUIDv5 namespace over the internal ID; the mapping remains restricted for AGOT. Root bit positions are assigned by sorted opaque root ID within a pool and are stored in the pool manifest.

Canonical serialization is UTF-8 JSON with sorted keys, fixed enumeration strings, rational numbers as reduced `numerator/denominator` strings, normalized line endings, and no timestamps in content digests. A record digest excludes its own `content_hash`, signatures, and nonsemantic audit timestamps. SHA-256 identifies each record set, graph, pool, table, manifest, and result. A scientific version change creates a new immutable record and provenance link; it never edits a frozen object in place.

Versions are separated as follows:

- `schema_version` changes field semantics or validation.
- `record_version` changes an annotation or configuration instance.
- `handbook_version` changes interpretation rules.
- `graph_version` fixes a compiled set of record versions.
- `verifier_version`, `objective_version`, and `metric_version` fix scientific behavior.
- `manifest_version` binds every preceding version and the environment.

Post-freeze schema migration must preserve the old canonical file, supply a deterministic migration record, and show that scientific content hashes map one-to-one. A semantic change is a protocol deviation and requires rescoring from the immutable original; it cannot be disguised as migration.

## 9. Annotation workflow

The annotation workflow uses structured YAML worksheets for researcher entry, CSV ledgers for repetitive audits, and validated canonical JSONL for experiments. Building a custom annotation application is out of scope. A local editor with schema validation is sufficient. The annotation compiler must never infer a missing scientific value merely to make a record valid.

The fixed corpus allocation is four held-out AGOT bundles, one from each discourse quartile; one development bundle from Wilkie Collins's *The Moonstone*; one independent held-out bundle from Arthur Conan Doyle's *The Hound of the Baskervilles*; and one researcher-authored hidden control. Each natural bundle contains a 1,800 to 2,400 word focal episode plus no more than 600 words of distant prerequisite anchors, with a hard 3,000 word annotation boundary. The hidden control is 1,800 to 2,200 words. Each natural bundle must yield exactly 25 defensible roots for the implementation target, without padding. If a candidate episode cannot do so, the next episode selected by the frozen pre-result discourse rule replaces it before contexts or pools exist. Across six natural bundles this yields 150 roots; the hidden control has at most 18.

### 9.1 Pass order

For each bundle, use the following order:

1. **Boundary and edition pass:** freeze the lawful work version, focal episode boundary, prerequisite anchors, discourse indices, and a boundary memorandum. Record why the episode is query-sufficient but not book-exhaustive.
2. **Anchor pass:** create EvidenceAnchor records before interpreting claims. Public-domain text may be included; AGOT uses locators and hashes only.
3. **Entity and identity pass:** identify entities, aliases, unresolved identities, and explicit non-merges.
4. **Occurrence pass:** identify EventOccurrence records and decide whether repeated descriptions refer to the same occurrence or different occurrences.
5. **State and transition pass:** create before and after states, validity constraints where needed, and transitions. Unknown boundaries remain explicit.
6. **Proposition and assertion pass:** separate content from the holder/source's stance, polarity, epistemic mode, and curator status.
7. **Temporal pass:** add story-time, validity, discourse, and revelation constraints without collapsing their coordinate systems.
8. **Dependency and provenance pass:** create root dependencies, evidence alternatives, and derivation records; compile the method-visible graph.
9. **Outcome pass after episode freeze:** construct queries, consequence packets, final relevance labels, horizon audits, reference supports, and diagnostic labels in separate gold files.
10. **Validation and freeze pass:** run schema, graph, visibility, rights, temporal, dependency, and competency-query checks. Resolve only recorded issues.

Each pass writes a validation report. An issue ledger assigns `OPEN`, `RESOLVED`, `LICENSED_ALTERNATIVE`, or `DEFERRED_OUT_OF_SCOPE`. A record cannot freeze while an error-level issue is open. `unknown` is valid when the source and policy do not support a more specific value.

### 9.2 Delayed stability check

After at least 14 full days, export a masked, randomly ordered packet containing exactly 24 of the 150 natural roots. This exceeds the rounded-up 15 percent minimum and preserves V2's explicit floor of 24. It must include at least six transitions, six assertions, all primary rare-consequence candidates if there are fewer than six, and examples from every natural bundle. The packet omits first-pass labels and stable root IDs and uses fresh review IDs. The random seed and stratification are frozen before export.

Reannotation results are linked through a sealed mapping only after the second pass. Compute pairwise coreference F1, evidence-anchor overlap F1, categorical or weighted kappa where V2 requires it, temporal-relation agreement, and support-atom F1. The horizon re-audit is separate. Reconciliation retains first pass, second pass, adopted value, handbook rule, and rationale. It cannot silently overwrite the first pass.

### 9.3 Freeze and change policy

Held-out episode records become scientifically immutable when all five held-out bundles pass validation, their graph hashes enter a signed freeze manifest, and the 26 held-out contexts and pools have been created but not scored. Thereafter:

- A **factual correction** fixes a demonstrable transcription or locator error. It creates a new version, preserves the old one, and triggers all affected reruns plus a protocol-deviation entry.
- A **schema migration** changes representation without changing meaning. A migration equivalence report is required.
- An **interpretation alternative** is added only if it was licensed before method outputs; after output it is a sensitivity note, not a new primary oracle.
- A **post-result change** never replaces confirmatory data. It is reported and analyzed separately.
- A **protocol deviation** states who discovered it, when, which artifacts and metrics it affects, and whether the fixed-benchmark gate remains interpretable.

The researcher verifies every Codex-assisted transformation by reviewing diffs, validation summaries, and a stratified source sample. Tool generation does not reduce logged researcher time.

## 10. Horizon-rarity audit workflow

The rarity tool retrieves candidates; the researcher determines co-reference and the count. It audits at most 32 master transitions, followed by complete delayed re-audits of eight after 14 days. It does not construct a full-book ontology.

### 10.1 Search packet

Each transition receives one frozen `MentionSearchPacket` with:

- opaque transition and work-version IDs;
- canonical entity aliases and edition-specific variants;
- short abstract event descriptions and manually enumerated paraphrases;
- unordered and role-sensitive participant combinations;
- state dimension and before/after value descriptors;
- likely holder/source combinations for reports and recollections;
- lexical stems, spelling variants, and exclusion terms;
- horizon discourse index and permitted local files;
- search-tool and packet version.

The local workflow executes exact and normalized lexical search, proximity search over participant aliases, and lemmatized search. A small local semantic retriever is optional during development only; if retained, its frozen model hash and recall-audit role are recorded, and it only proposes candidates. It cannot classify a hit. The required path must remain executable with lexical, lemmatized, and participant-combination searches on CPU.

### 10.2 Candidate review and counting

For every proposed passage, the researcher sees a fixed local context window and classifies it using `MentionAuditRecord`. A hit co-refers only if story time, core participants, state dimension, and before/after values are compatible with the same occurrence. `same_depiction`, `same_report`, `same_recollection`, and `same_summary` each count once per distinct discourse anchor. A consequence, a different event of the same type, or the same participant relation at another time does not count. Duplicate overlapping search hits collapse by work, normalized span, and digest. A vague allusion counts only when the frozen identity rule licenses the occurrence without access to rarity category.

The audit ledger records every query form, hit, duplicate cluster, manual classification, reason code, horizon decision, and count contribution. Post-horizon hits remain in the restricted audit ledger with contribution zero so exclusion is auditable. `HorizonCount` is the sum of verified contributions at positions no later than the query horizon. `LocalCount` uses the same rules but restricts anchors to the episode bundle.

Primary categories are horizon singleton (`HorizonCount=1`), intermediate (`=2`), and horizon repeated (`>=3`). Local categories are sensitivity labels only. The delayed eight-transition re-audit repeats search and review, not merely rechecks saved labels. The V2 gate remains mention-decision F1 at least 0.85 and identical rarity category for at least seven of eight. Failure replaces the broad claim with locally singleton or query-necessary support terminology.

AGOT search indexes, candidate windows, aliases, and prose-bearing logs remain outside Git. The public audit release contains packet schemas, safe query abstractions, counts, opaque locators/hashes, and the fully reproducible public-domain ledgers.

## 11. Candidate-pool construction

Two modules build two scientifically distinct pools. Neither module is inside a selector; all methods receive the same frozen pool ID.

### 11.1 Primary method-independent pool

Inputs are a frozen `MethodBundle`, one `TypedQueryContext`, generator configuration, and seed. The generator performs these steps in a fixed order:

1. Restrict to selectable roots from the same work whose root-level discourse/revelation position and visible dependencies do not exceed the horizon. Roots from other already annotated bundles in that work are eligible; cross-work roots are not.
2. Exclude roots whose scenario scope or unresolved identity makes them incompatible with the typed context under method-visible rules.
3. Compute the coarse context score only from focal-entity overlap, task-compatible root/type, specified story or discourse window, holder/viewpoint, and response-schema compatibility. Outcome-only imports are structurally unavailable.
4. Select exactly six roots with the highest coarse score, using sorted opaque IDs for ties. Fewer than six eligible roots is a hard pool-construction failure.
5. Order the remaining universe by SHA-256 of `(manifest_seed, context_id, root_id)` and add roots until 18 are present.
6. If a root kind is absent, apply a frozen minimal type-balance exchange using the same hash order. No outcome label is consulted.
7. Compile closure-preview counts, component counts, edge density, budget-to-pool ratios, method-visible score distribution, and the full selection trace.

The pool compiler also enforces V2's evidence-choice bound. If more than two selected roots have multiple licensed routes, the two smallest opaque root IDs retain at most two selectable routes and every other root uses its deterministic canonical route, chosen by lowest development-frozen component cost and then alternative ID. All routes remain in the oracle audit record, and the restriction is reported in target/support availability metrics.

The seed is global and preregistered; it is not searched. If fewer than 18 eligible roots exist, the context fails before evaluation. The generator never inserts a gold target after availability is known.

For a target transition (t_q) and accepted support family \(\mathcal Q_q\), define:

$$
TargetAvailable_q=\mathbb 1[t_q\in R(Pool_q)],
$$

$$
SupportAvailable_q=\mathbb 1[\exists Q\in\mathcal Q_q,\eta:\ Q\subseteq cl_C(R(Pool_q),\eta)].
$$

`TargetAvailabilityRecall` and `SupportAvailabilityRecall` are the means of these indicators over a declared slice. They are computed only after the candidate pool is immutable and are outcome metrics, never generator inputs. Both must equal 1.00 on the 10 primary HRC contexts for the planned superiority claim. A failure is reported and invokes the benchmark-construction fallback; no target-aware repair is permitted.

Minimum primary-pool pressure is checked after gold remains sealed from selectors:

- 18 roots and exactly the same pool for every method;
- at least a 3:1 root ratio relative to the medium cap of six;
- at least nine roots later judged irrelevant or merely contextual;
- at least one frequent distractor, one rare nonconsequential structure, one incomplete competing response, and one structurally central but query-irrelevant root;
- same-type alternatives so the target is not identified by a unique event type or ID;
- compiled components at least three times the medium cap for the principal available categories.

These are benchmark diagnostics, not generator quotas. Failure is disclosed and cannot be fixed after outcomes or method results are examined.

### 11.2 Optional controlled challenge pool

If required work is complete, a second module may reproduce the V2 quota-balanced 18-root pool as a controlled diagnostic. It may use sealed labels to populate mutually exclusive categories, but all methods remain blind to those category fields. Category precedence is: incomplete competing support, rare nonconsequential, frequent weak interaction, unrelated transition, repeated low-information assertion, central irrelevant, then focal/potential support. A root is assigned once; short categories are filled by stable discourse order without relabeling. Each root remains a genuine same-work, horizon-eligible annotated structure with its actual evidence and status. No synthetic nonsense is allowed.

An incomplete competing response is a genuine root or root set that can fill an answer head or one response slot but lacks at least one method-visible qualification or support requirement; it is semantically valid as a partial structure and is not deliberately corrupted. The challenge pool is frozen before method execution and never substitutes for the primary pool. Its results are descriptive and can be dropped when the workload forecast exceeds 145 hours.

### 11.3 Leakage-audit checklist

The compiler must show that candidate construction neither reads nor derives:

- gold-support membership or target IDs;
- consequence score or dimensions;
- HorizonRarity or LocalRarity category;
- balanced diagnostic cell;
- final human relevance grade;
- any selector score or projection;
- held-out aggregate results;
- method identity or a predicted winner.

Enforcement has three layers: outcome files live in a separate directory; candidate modules cannot import evaluation-gold model classes; and a run-time access ledger records every input hash. A planted forbidden-field test must fail. A manual pre-freeze checklist reviews score traces for all 32 answerable pools.

The resulting object is explicitly a controlled dense candidate graph assembled from bounded oracle annotations. It is not described as a complete book-scale network, even when roots from several AGOT bundles are reused.

## 12. Temporal satisfiability implementation

Temporal reasoning uses a frozen endpoint semantics over quantifier-free linear rational arithmetic. For each story-time or state-validity interval (i), create rational variables (s_i,e_i) and require (s_i\le e_i). Instantaneous events permit equality; records marked proper intervals require (s_i<e_i). Exact dates or story ordinals become equalities. Approximate dates become rational lower and upper bounds. Unknown boundaries add no bound beyond interval validity.

The supported normalized relations are:

$$
\begin{aligned}
before(i,j)&: e_i<s_j,\\
equal(i,j)&: s_i=s_j\land e_i=e_j,\\
overlaps(i,j)&: s_i<s_j<e_i<e_j,\\
during(i,j)&: s_j<s_i\land e_i<e_j,\\
contains(i,j)&: during(j,i).
\end{aligned}
$$

`after(i,j)` is normalized to `before(j,i)`. Allen `overlaps` is directional; its converse is stored as a reversed relation. If a curator means any nonempty temporal intersection, the distinct relation `intersects` is used and encoded as (s_i<e_j\land s_j<e_i). An allowed disjunction is an explicit logical `Or` over supported relations. `unknown` contributes no ordering proposition and cannot be displayed as a specific relation.

Story time, state validity, discourse order, and revelation order remain separate coordinates. Discourse and revelation positions are fixed integer indices used for eligibility, not solved story times. A flashback has an early story interval but a later evidence/discourse anchor. Memory, rumor, and report are assertions whose proposition may refer to an occurrence; the assertion's revelation time does not assert the occurrence's truth. Dream, prophecy, and hypothetical occurrences use separate scenario scopes. Constraints do not cross a scenario boundary unless an explicit typed mapping is annotated.

The solver translation sorts constraints by stable ID, uses exact integer/rational literals, fixes the Z3 version, sets all available SAT/SMT random seeds to zero, disables internal parallel solving, and has no benchmark timeout. `sat` establishes consistency only. A displayed temporal relation is entailed only if the base constraints plus its negation are unsatisfiable. If several relations remain possible, the display must use the frozen allowed set or `unknown`; mere possibility is not rendered as fact. `unknown` from Z3 becomes `ERROR_OR_UNKNOWN`, never admissibility or infeasibility.

Every constraint uses a named tracking literal. On unsatisfiability, the tool obtains a core and deletion-minimizes it in stable constraint-ID order. The archived temporal report includes normalized constraints, input hash, solver version/settings, SAT status, entailed relations, and the deterministic deletion-minimal core. It does not claim a minimum-cardinality core.

Small abstract fixtures include: an event ending before a later state begins; a flashback narrated after the current episode; two intervals satisfying a disjunction of before or overlaps; a rumor assertion whose discourse time is known but event time is unknown; and a contradictory cycle. No copyrighted passage appears in fixtures.

## 13. Closure and admissibility engine

The canonical graph distinguishes selectable roots from semantic atoms. A selected root is a decision variable. Its closure contains the records required to interpret and display it. Evidence choices are resolved before closure, not treated as an unresolved existential.

### 13.1 Closure compilation

Each pool assigns stable integer bit positions to its 18 roots and all eligible atoms. `RootDependency` edges compile to sorted adjacency arrays. For a root mask (X) and compatible evidence-choice assignment \(\eta\), a deterministic worklist computes the least fixed point. Cycles are allowed; finite positive dependencies guarantee termination. The cache key is `(graph_hash, context_hash, root_mask, eta_bits)` and the value contains the atom bitset, closure hash, presentation hash, cost vector, and preliminary failure vector.

Required dependency rules are:

- **Identity:** every referenced entity and licensed identity decision.
- **Participant role:** event, role label, and participant entity.
- **Transition:** bearer/dimension, triggering event when asserted, warranted before and after states, qualification, time constraints, provenance, and chosen evidence/support.
- **State:** bearer, value referent, validity constraints, attribution when not curator-established under policy, evidence, and provenance.
- **Assertion:** proposition, holder/source, mode, polarity, curator status, revelation position, provenance, and chosen evidence.
- **Proposition:** predicate and argument identities; it never imports an unqualified truth claim.
- **Temporal:** all constraints needed for a displayed relation or validity claim.
- **Support:** the atoms required by the method-visible response proof schema.
- **Presentation:** every fact used by a static label, relation, evidence cue, or word-counted phrase.

Dependencies are never trimmed to fit a budget or horizon. Shared atoms are counted once by canonical ID. The root coordinate counts decision roots only; every other coordinate is computed after closure and deterministic presentation mapping.

### 13.2 Separate verifiers

One common wrapper invokes these components for every method:

1. structural and referential integrity;
2. horizon and label non-leakage;
3. identity safety;
4. evidence grounding;
5. provenance completeness;
6. epistemic fidelity;
7. temporal satisfiability and displayed-relation entailment;
8. method-visible support completeness (`GuardSup`);
9. componentwise budget compliance.

Each component returns a `VerifierResult` with `PASS`, `FAIL`, or `ERROR`; category; severity (`CRITICAL`, `MAJOR`, or `MINOR`); implicated record IDs; a short human-readable explanation; and machine-readable details. Critical failures include post-horizon disclosure, unsupported answer claim, rumor-to-fact promotion, mistaken identity, answer-affecting temporal contradiction, and incomplete answer support. Provenance and budget failures remain separately visible. Cosmetic formatting is not an admissibility category.

| Verifier | Exact pass condition | Observable failure |
|---|---|---|
| Evidence grounding | Every displayed answer claim has one selected licensed EvidenceAlternative whose anchors and support atoms are in closure and no later than the horizon | Missing, mismatched, ineligible, or post-horizon evidence |
| Horizon safety | Every semantic atom, evidence anchor, revealed identity, and fact used in a display phrase has discourse/revelation position no later than (H) | Fact, identity, evidence, or descriptor crosses (H) |
| Epistemic fidelity | Displayed holder/source, mode, polarity, curator status, and scenario equal the selected assertion; unqualified establishment is licensed by the frozen policy | Holder loss, polarity change, rumor/belief promotion, or modal flattening |
| Temporal satisfiability | Selected constraints are SAT and each displayed specific temporal relation is asserted or entailed | Unsatisfiable constraint set or merely possible relation displayed as settled |
| Identity safety | Every role/reference resolves to one licensed entity decision; disputed or unresolved alternatives remain distinct and no post-horizon alias is used | Unsafe merge, dangling identity, or identity disclosure |
| Provenance completeness | Every displayed semantic atom has a provenance chain to an annotation/curation activity and source anchor or explicit curator rule | Missing or cyclic derivation witness |
| Support completeness | Each displayed answer claim instantiates its task's method-visible GuardSup schema with every mandatory atom present | Missing transition state, qualification, answer link, or required proof atom |
| Budget compliance | The post-closure, post-presentation nine-vector is componentwise no greater than the exact shared cap | Any coordinate exceeds its cap, even if all others are below |

Experiments do not early-terminate. They preserve all detectable categories and the exact predicate vector even when one failure suffices to reject a projection. Interactive early termination is unnecessary. A native baseline is evaluated by the same verifiers after selection, but the verifier does not repair it.

`GuardSup` is a finite proof-schema checker using only method-visible atoms. For transition explanation it requires the transition, warranted states, qualification, and evidence route. For temporal ordering it requires both occurrences, an asserted or entailed relation, and evidence. For epistemic discrimination it requires proposition, assertion, holder, mode, polarity, and evidence. For state tracing it requires an ordered, qualified transition chain. `GoldSup` remains a sealed scoring family. An `attacks` edge alone is not proof of a negated proposition.

## 14. Response feasibility and certificates

The implementation represents the two nested families exactly:

$$
\mathcal A_{adm}(G,C,B)=\{S: Closure, Horizon, Evidence, Epistemic, Temporal, Identity, Provenance, GuardSup, Budget\},
$$

$$
\mathcal A_{resp}(G,C,B)=\{S\in\mathcal A_{adm}: Resp_\kappa(S,C)\}.
$$

A safe but nonresponsive projection belongs to the first family only. A returned answer succeeds only in the second. Exact enumeration establishes whether either family is empty; failure of a heuristic does not.

The certificate class precedence is:

1. `INVALID_CONTEXT`: the raw request cannot compile to one licensed typed context because of missing or contradictory required fields. It is a pre-projection result.
2. `CONTENT_ABSENT`: no horizon-eligible answer-head root exists even when semantic guards and budgets are ignored.
3. `CONTRACT_BLOCKED`: candidate answer heads exist, but \(\mathcal A_{resp}(G,C,B_\infty)\) is empty.
4. `BUDGET_BLOCKED`: the unbounded response family is nonempty but the requested-budget family is empty.
5. `ERROR_OR_UNKNOWN`: an operational failure, never a correct abstention.

For contract blockage, enumerate candidate response closures at (B_\infty), group their full failed-guard signatures, and return all inclusion-minimal signatures in stable lexical order. Temporal failures include the deletion-minimal unsat core. For budget blockage, compute every Pareto-minimal feasible cost vector (c(S)) and delta \(\max(0,c(S)-B)\); archive the frontier and expose one lexicographically selected witness. A "minimal additional budget" statement is valid only for this class.

If an answer-head candidate exists but all its evidence is missing, post-horizon, disputed beyond the response policy, or mutually inconsistent, the class is `CONTRACT_BLOCKED`, not `CONTENT_ABSENT`. A method that abstains while the exact table contains a response-feasible row commits false abstention regardless of its internal search failure.

Certificate validation recomputes the class from the frozen enumeration table and ignores the prose explanation. A public certificate redacts post-horizon and restricted IDs; a protected audit certificate retains the full witness. The 10 probes are fixed as two invalid-context, two content-absent, four contract-blocked spanning temporal, epistemic, identity/support, and horizon causes, and two budget-blocked cases, while retaining two probes per real held-out bundle. The two invalid-context probes do not create feasible tables.

## 15. Exact subset search

Each 18-root pool is mapped to an integer mask. Confirmatory enumeration visits masks in numeric order, filters by `bit_count() <= K`, and expands only the evidence alternatives present in that mask. At most two roots have alternatives and each has at most two, so a mask yields at most four \(\eta\) variants. Incompatible alternatives fail explicitly.

The exact subset counts are:

| Root cap | Root masks | Maximum mask-choice cases |
|---:|---:|---:|
| 4 | 4,048 | 16,192 |
| 6 | 31,180 | 124,720 |
| 8 | 106,762 | 427,048 |
| Unbounded 18 | 262,144 | 1,048,576 |

One high-cap table per answerable context is sufficient because low and medium are exact filters of its rows by the frozen vector caps. There are 32 answerable tables. Eight typed feasibility probes require unbounded enumeration; two invalid-context probes stop before enumeration. The conservative maximum is therefore 13,665,536 high-cap candidate-choice rows plus 8,388,608 unbounded probe rows, or 22,054,144. These are computational rows, not experimental observations. Closure, temporal signatures, and verifier results are cached; contexts can run as separate local CPU processes, but rows and final outputs are sorted canonically.

Enumeration streams one context at a time. It retains response-feasible rows in canonical gzip-compressed JSONL with a fixed zero timestamp, retains aggregated failed-signature counts and certificate witnesses, and updates a rolling digest over every visited candidate-choice row. It does not keep all invalid rows in RAM. A full deterministic rerun can verify the rolling digest. This bounds memory by one context and a small cache while preserving the common table and exact coverage evidence. Transient full tables may stay local; their hashes, certificates, and regeneration commands are permanent.

Safe prechecks may reject an incompatible evidence assignment, a root mask above eight, or a closure lower bound already exceeding a cap. Every precheck is tested to have no false exclusion. Confirmatory zero-gap certification still accounts for every expected mask and evidence variant. No objective-based branch-and-bound pruning is allowed unless independently proven and preregistered; the default is complete enumeration.

Each table row stores root mask, evidence choice, closure hash, exact cost vector, verifier predicate vector, response slots, all objective coordinates, and stable tie key. Full and every wrapped baseline read this immutable table hash. Objectives use integers and `Fraction`; normalized cost is exact. Ties choose the lexicographically smallest sorted `(root_id, alternative_id)` vector.

An enumeration certificate stores `n`, cap, expected and observed mask/variant counts, invalid/admissible/responsive row counts, table hash, maximizing row, runner-up objective tuple, and error status. An independent table scanner must confirm no response-feasible row outranks the winner. Zero optimality gap is claimed only when all counts, hashes, and independent checks pass.

Correctness tests compare the optimized worklist/enumerator with naive set closure and enumeration on fixtures with at most 10 roots; compare a simplified objective with an independent Z3 optimization encoding where practical; exhaust known-optimum and known-infeasible fixtures; permute input storage order; test deterministic repeated processes; and check budget-family monotonicity.

## 16. Exact PCST scoring

"PCST" in this benchmark is an exact prize-collecting root-incidence objective adapted to semantic display roots. It is not a claim that an off-the-shelf PCST approximation was run unchanged.

For each pool, construct a frozen undirected incidence graph (H_C). Its vertices are the 18 selectable roots plus a scoring-only query vertex (q_0). A root-root edge exists only for a declared, method-visible connection: shared participant, explicit event-state/assertion link, explicit causal/support relation, or compatible temporal/state-chain adjacency. The edge records the smallest declared connector atom bundle. Every connector atom must already be in the candidate projection's closure and is charged by the common componentwise cost calculator. An unselected semantic root cannot be used as a hidden Steiner vertex.

For a response-feasible selected root set (X), let (H_C[X\cup\{q_0\}]) contain root-root edges whose connector bundle is present. Query-seed roots receive a zero-cost edge to (q_0). To permit an explicit forest rather than silent disconnection, every nonseed root has a virtual edge to (q_0) with component activation cost

$$
\rho=1+\max_{e\in E(H_{development})} cost(e),
$$

derived and frozen on development, with \(\rho=1\) if the development incidence graph has no root-root edge. Each disconnected semantic component therefore pays one activation cost. Edge cost is the exact medium-budget normalized component cost of its connector bundle. The exact connector cost `Conn(X)` is the rational minimum spanning-tree cost on this induced graph including (q_0), computed by deterministic Kruskal ordering. This is also a rooted prize-collecting forest cost because virtual edges activate otherwise disconnected components. It is infinity only when a malformed graph lacks even a virtual edge.

The candidate score is

$$
J_{PCST}(S)=\lambda_r\sum_{r\in X}\widehat r_C(r)
+\lambda_s\left|\bigcup_{r\in X} Slots_C(r)\right|
-\lambda_c Conn(X).
$$

The set union prevents response-slot double counting. Evidence-only atoms have zero prize. Transition, event, and assertion roots use the same rule. The three multipliers use the 12-cell development grid in Section 3.3. Stable root IDs break exact ties. Guard-PCST maximizes this score over the same immutable response-feasible table as Full. Native horizon-filtered PCST enumerates root subsets under raw component caps, materializes and charges its chosen connector bundles, but does not add the semantic closure or guard repairs.

Because global root-subset selection is exhaustively enumerated and every induced connector cost is an exact MST/forest value, the benchmark comparator is exact under this declared root-incidence formulation. It is not the unrestricted node-weighted PCST problem and it cannot use off-pool Steiner roots. Tests include hand-solved line, star, cycle, duplicate-slot, disconnected-forest, and stable-tie fixtures; exhaustive edge-subset comparison on tiny graphs; connector charge checks; and input-order invariance. If these checks fail, the comparator becomes `ERROR_OR_UNKNOWN`; an approximation must not be relabeled exact.

## 17. Proposed objectives and direct ablation

The typed context is compiled manually and frozen. For every root, the selector computes a method-visible relevance score without gold labels:

$$
\widehat r_C(r)=\min(3,I_F(r)+I_K(r)+I_W(r)),
$$

where (I_F=1) for a focal-entity match, (I_K=1) for compatibility with a requested event/state type or response-schema slot, and (I_W=1) when all applicable story/discourse/holder restrictions match. A restriction that is not specified contributes neither a match nor a penalty. The feature trace is stored so every score can be audited. Final curator relevance is a scoring label and is absent from this function.

A method-visible candidate transition bundle (z\in\mathcal Z_C) contains one transition root plus the closure atoms for bearer/dimension, warranted before and after states, scenario and epistemic qualification, temporal constraints needed for the stated change, provenance, and a guard-valid evidence route. It is a structural ontology pattern, not an accepted gold support path. \(\widehat r_C(z)\) is the score of its transition root. Required response slots determine membership in \(\mathcal A_{resp}\); \(K_C\) may additionally contain frozen optional explanatory slots, so coverage is not necessarily constant. Define:

$$
\begin{aligned}
J_{TB}(S,C)&=\sum_{z\in\mathcal Z_C}\widehat r_C(z)I[z\subseteq S],\\
J_{rel}(S,C)&=\sum_{r\in roots(S)}\widehat r_C(r),\\
J_{cov}(S,C)&=\sum_{k\in K_C}I[S\text{ fills response slot }k],\\
J_{red}(S)&=\sum_{\{r,s\}}I[r,s\text{ share a frozen redundancy class}],\\
J_{cost}(S,B)&=\sum_i c_i(S)/B_i.
\end{aligned}
$$

All divisions are exact rational operations. Proposed-Full selects the response-feasible row maximizing:

$$
\langle J_{TB},J_{rel},J_{cov},-J_{red},-J_{cost},-J_{tie}\rangle.
$$

`J_tie` is the stable sorted root/evidence-choice vector and is used only after substantive coordinates tie. Proposed-NoTransitionPriority uses the identical row table and objective implementation but removes only (J_{TB}):

$$
\langle J_{rel},J_{cov},-J_{red},-J_{cost},-J_{tie}\rangle.
$$

The two methods receive identical caps. Realized costs may differ and are reported coordinate by coordinate. No requirement that Full consume no more of every coordinate is imposed. Mechanism evidence at medium budget requires at least two net HRC successes over the ablation, at least three wins, and at most one loss. It additionally checks that the gain does not arise from a cap violation, that frequent-consequential recall does not decrease by two or more items, and that rare-nonconsequential selection does not increase by two or more items. The small diagnostic cells qualify interpretation rather than form a binary gate. If Full and the ablation tie, any advantage over PCST is not attributed to transition priority.

Objective code is a pure function over one feasible-table row and frozen configuration. It cannot import corpus gold models. A regression fixture differs in exactly one complete transition bundle and demonstrates the only intended ranking difference between Full and the ablation.

## 18. Native and guard-wrapped baselines

All eight declared variants are implemented, but the minimal matrix does not run every secondary variant at every budget and context.

### 18.1 Common horizon filter and native materialization

Every native baseline begins with the same reasonable root-level horizon filter as the candidate generator. It never receives known post-horizon roots. The no-horizon form exists only as a verifier fault diagnostic. A native output materializes its selected roots, their direct identity endpoints, and any explicit connector atoms the native method uses. It is charged by the common cost calculator and fixed presentation grammar but is not automatically repaired with missing evidence, epistemic qualification, temporal dependencies, provenance, or `GuardSup` atoms. The common verifier evaluates its practical violations after selection.

This design avoids two straw men: native systems are not deliberately given later plot information, and their outputs are not forced to behave like the proposed semantic closure. Native failures are secondary practical comparisons, not evidence that Full's objective itself is safer.

### 18.2 Frequency

The frozen frequency score is the number of distinct verified in-horizon anchors depicting or presenting the particular root occurrence or assertion. When several roots share an occurrence, counts attach to the canonical occurrence and are copied through a documented feature record. Native frequency sorts descending, adds roots greedily while raw component caps remain satisfied, and uses stable IDs for ties. It does not skip a higher root to find a semantically complementary lower one unless the higher root would violate a cap. Guard-frequency maximizes summed frequency over the common response-feasible table, then lower cost and stable tie.

Because frequency is the defining baseline signal, only frequency selectors may read this raw count. They do not receive the named HorizonRarity category, consequence, target, support, or relevance labels. Full, the ablation, PPR, and PCST receive method views from which the frequency feature is absent. This exception is explicit rather than allowing a general method bundle to expose all baseline features.

### 18.3 Personalized PageRank

PPR operates on the frozen root-incidence graph before guard closure. An edge weight is the count of distinct present link types among shared participant, explicit event-state/assertion link, causal/support link, and temporal/state-chain adjacency, so weights are integers 1 through 4 and duplicate instances do not accumulate. The graph is undirected and rows are normalized by weighted degree. The personalization distribution is uniform over focal/type/holder seed roots; if empty, it is uniform over all 18 roots. Damping is 0.85, dangling mass uses the personalization vector, node input order is sorted, convergence tolerance is `1e-12`, and the maximum is 1,000 iterations. Failure returns `ERROR_OR_UNKNOWN`. Scores are quantized to 12 decimal places for stable ordering, with opaque-ID ties.

Native PPR greedily materializes highest-scoring roots under raw caps. Guard-PPR maximizes summed PPR score over the common response-feasible table, then lower cost and stable tie.

The archived PPR score is the quantized value multiplied by (10^{12}) and stored as an integer, so downstream row comparison is exact even though PageRank iteration uses floating point. The unquantized development trace is retained for convergence diagnostics only.

### 18.4 PCST

Native and wrapped PCST use the exact Section 16 graph, prizes, connector rules, and development-frozen parameters. Native PCST maximizes its score over horizon-eligible raw root/connector projections under the same vector caps, with no semantic repair. Guard-PCST maximizes over the common response-feasible table. It is the primary comparator.

### 18.5 Fairness checks

For every wrapped run, assert equality of candidate-pool hash, feasible-table hash, budget hash, response-schema hash, evidence-choice universe, verifier version, presentation grammar, and tie-rule version. The only allowed differences are objective/score and selected row. A common guard wrapper and cost calculator are called by identifier, never duplicated in a baseline module. Any mismatch invalidates the comparison.

## 19. Budget calibration and freezing

The budget vector is

$$
B=(B_{root},B_{entity},B_{event},B_{transition},B_{state},B_{assertion},B_{relation},B_{evidence},B_{word}).
$$

The fixed root caps are 4, 6, and 8. Non-root caps are generated by one development-only calibration command after all four public-development task types can produce an admissible response at cap 8.

### 19.1 Calibration inputs and calculation

Inputs are the frozen development graph, its four canonical base contexts, response schemas, closure/verifier versions, evidence alternatives, and presentation grammar. Variants are excluded. Exact enumeration obtains Pareto-minimal cost vectors for each task. The canonical vector per task is chosen lexicographically by roots, evidence cues, assertions, transitions, sum of other components, and stable atom IDs.

For each non-root coordinate:

- low is the upper median, the third value after sorting four task minima;
- medium is their maximum;
- high is medium plus the upper-median marginal cost of a complete transition/evidence bundle observed in development.

Round only where the source quantity is an integer count, which it always is. The algorithm emits candidate vectors, chosen vectors, task feasibility, transition marginals, and comparison with V2's provisional envelope. If a task has no cap-8 admissible response or a chosen coordinate exceeds the envelope, calibration stops. The researcher may simplify a development response schema or episode before any held-out source is opened, document the change, rerun the full pilot, and recalibrate. Values are never clipped to fit the plan.

### 19.2 Review, freeze, and sensitivity

The generated file is `config/frozen/budgets.v1.yaml`. A researcher checks that every integer matches the calibration report and that low is componentwise no greater than medium and medium no greater than high. The canonical JSON form, inputs, command version, and review record are hashed into the experiment manifest. Held-out data are inaccessible to the calibration command and cannot raise or lower a cap.

Requested costs are the declared caps; realized costs are recomputed from the projection after closure and fixed presentation. Every method receives the same cap hash. Words use a frozen whitespace-token grammar over static identity-preserving templates, evidence cues, and edge phrases. Adaptive descriptors and layout area are absent. The study therefore evaluates semantic projection under display proxies, not visual readability.

Required sensitivity is Full versus guard-PCST on the 10 primary contexts at the adjacent low and high levels. Optional one-coordinate perturbations are generated mechanically from development by plus or minus one canonical bundle marginal and are dropped first under time pressure.

## 20. Corruption-study implementation

The corruption bank validates the contract implementation. It does not show that Full's selection objective is safer than a wrapped comparator.

### 20.1 Development and sealed banks

Development and hidden-control fixtures contain at least one corrupt and one clean case for each of nine categories: evidence, mandatory closure/support, horizon fact, epistemic promotion, temporal contradiction, identity merge, budget, provenance, and label leakage. These 18 cases may be revised while developing verifiers.

Before held-out clean projections are generated, freeze one transformation template per category, the expected guard logic, severity, and method for selecting a target atom. For each held-out bundle, create two canonical medium-budget clean oracle responses from frozen response schemas before selector execution. The clean bases may use gold to ensure known validity because they test the verifier, not selection, and are never selector inputs. Instantiate exactly one category case per bundle, yielding 45 corruptions, and retain the two clean bases, yielding 10 controls. Templates select targets by stable eligible-ID order, never by verifier behavior. Every case is a signed delta with base hash, before/after hashes, mutation version, target IDs, expected failed guards, guards expected to pass, permitted multi-guard explanation, and case hash.

Examples are precise transformations: remove or mismatch one answer evidence anchor; remove one non-evidence mandatory closure atom; insert an otherwise valid post-horizon root or descriptor; remove a rumor holder or change its display status; insert a constraint contradicting a known temporal chain; merge two entities with an explicit non-merge decision; lower one budget coordinate below a clean realized cost; remove provenance; or add a post-horizon display phrase without a semantic fact.

### 20.2 Isolation and evaluation

Before verifier output is viewed, manually inspect the mutation delta. An isolated case must have exactly the expected target failure when the target guard is disabled and every declared non-target guard must pass. If evidence deletion necessarily also fails `GuardSup` and response, mark the case `multi_guard` and require that full signature. Do not claim isolation.

The full verifier runs on 45 corruptions and 10 controls. Then each target guard is disabled for its five cases, producing 45 targeted reruns; the expected predicate vector, not merely acceptance, is compared. Report per-category sensitivity, clean specificity, full confusion matrix, severity, and signature agreement. A missed sealed corruption remains a failed gate. Any fix creates a new verifier version and development regression case; confirmatory recovery requires a newly preregistered held-out bank rather than overwriting the miss.

All templates, held-out case hashes, clean-base hashes, expected signatures, and verifier versions enter the frozen manifest. Public-domain/synthetic cases are released completely; AGOT mutations are released as safe abstract deltas and hashes.

## 21. Annotation-uncertainty sensitivity

The annotation compiler creates alternative oracle states before method outputs. Each state is a set of field-level substitutions licensed by `AnnotationAlternative` groups. Substitutions may affect assertion status, temporal relation, occurrence identity, consequence dimension, or membership of an accepted support structure; they cannot change candidate selection, method-visible dependencies, or the chosen projection after the fact.

For each run and metric (M), compute:

$$
M_{primary}=M(O_0),\qquad
M_{conservative}=\min_{O\in\mathcal O}M(O),\qquad
M_{permissive}=\max_{O\in\mathcal O}M(O),
$$

where the order is reversed for error metrics. For binary success, conservative requires success in every licensed interpretation and permissive requires success in at least one. The scorer enumerates at most eight states per query, records the exact state hashes, and never calls a selector. The fixed-benchmark decision is reported under curator-primary scoring and is considered interpretation-robust only if its direction and safety gate remain unchanged under conservative scoring. A result that exists only under permissive scoring is exploratory.

Tests use fixtures in which alternatives affect no score, only support membership, and a temporal or epistemic qualification. They verify that the selection hash is identical across all three scoring passes.

## 22. Experiment manifest and freeze protocol

Configuration has four levels:

1. immutable schema/default enumerations in source;
2. human-edited development YAML for corpus, methods, budgets, queries, verifiers, solver, corruption templates, and seeds;
3. generated canonical JSON configurations with all defaults explicit;
4. one write-once `ExperimentManifest` binding their hashes.

The manifest contains: V2 digest; handbook, schema, ontology, graph, query, response, pool, gold-seal, and annotation-alternative hashes; corpus editions and rights classes; split membership; budget derivation and vectors; objective order and rational weights; PCST grid and winner; PPR and frequency rules; verifier and corruption versions; temporal semantics and solver settings; presentation grammar; all seeds; run-matrix identifiers; code revision; Python and dependency-lock hashes; external restricted-snapshot hashes; freeze time; and protocol-deviation ledger location.

Human-edited configuration is divided into `corpora.yaml`, `graphs.yaml`, `queries.yaml`, `candidate_pools.yaml`, `budgets.yaml`, `methods.yaml`, `verifiers.yaml`, `temporal_solver.yaml`, `corruptions.yaml`, `run_matrix.yaml`, and a local ignored `paths.yaml`. Development, frozen, and release-redacted variants live in separate directories. After freeze, command-line options may select only manifest-declared contexts, methods, budgets, and output roots; they cannot override a scientific field.

The 24 base families and 32 answerable contexts are allocated exactly as follows: four base plus two variants on public development; four base plus one necessary horizon or holder variant in each of four AGOT bundles, giving 20; and four base plus two necessary variants on the independent public held-out bundle, giving six. Two distinct base families and transitions from each real held-out bundle form the 10 HRC contexts. The 10 probes are separate, two per real held-out bundle. Six paraphrases are optional and never part of the required count.

The minimal CLI has eight commands: `validate` checks schemas, references, rights, and graph integrity; `compile` creates method/gold-separated snapshots; `calibrate` derives development budgets and baseline parameters; `freeze` emits a write-once manifest; `enumerate` generates and certifies one common table; `run` executes declared selectors; `evaluate` loads sealed gold, scores, and generates reports; and `audit-release` builds and scans the public package. No notebook or ad hoc script can write canonical scientific outputs.

The mandatory order is:

1. Create and validate development artifacts and hidden-control fixtures.
2. Run the complete public-development pilot and log actual labor.
3. Revise only development-permitted components and repeat the pilot as needed.
4. Freeze ontology profile, schemas, handbook, code interfaces, parameters, budgets, query templates, response schemas, corruption templates, tests, and scope.
5. Emit canonical configuration, hash the manifest, commit it, and timestamp the preregistration commitment.
6. Open and annotate the held-out public material and restricted AGOT episodes.
7. Finalize held-out oracle records, contexts, candidate pools, support references, alternatives, and their hashes.
8. Seal evaluation gold and outcome labels without running selectors.
9. Generate common tables and run all systems from the frozen matrix.
10. Export unlisted support paths, mix and mask them, and review once under the frozen rubric.
11. Add accepted paths to a versioned scoring-only gold extension and rescore once; selections and feasible tables remain unchanged.
12. Generate final tables, lock confirmatory artifacts, and prohibit further confirmatory changes.

Held-out public material is not used in steps 1 through 5. The researcher may know the literary work, so hash sealing prevents artifact modification but not cognitive familiarity; the limitation is disclosed.

After freeze, a code defect may be fixed only if a failing test or independent invariant demonstrates divergence from the frozen specification. Preserve old code/results, create a defect report, increment implementation version, add a regression test, rerun every affected method, and report whether conclusions change. No coefficient, budget, query, pool rule, response schema, outcome rubric, or tie rule can change under the label "bug fix." If the specification itself is wrong, the confirmatory study stops or is explicitly re-preregistered before any results are interpreted.

## 23. Minimal run matrix

An exact table is a shared computational artifact, not a method-specific run. The pipeline creates 32 high-cap tables for answerable contexts and eight unbounded tables for typed probes. Two invalid-context probes produce parse records without enumeration. Low and medium feasible families are exact filters of high-cap answerable tables.

### 23.1 Required selector outputs

| Block | Methods | Contexts | Budgets | Output records |
|---|---:|---:|---:|---:|
| PCST development grid | 12 parameter settings | 6 development | Medium | 72 |
| Final development check | All 8 variants | 6 development | Medium | 48 |
| Held-out medium comparison | Full, NoTransitionPriority, guard-PCST, guard-PPR, guard-frequency | 26 held-out | Medium | 130 |
| Primary budget sensitivity | Full and guard-PCST | 10 HRC | Low and high | 40 |
| Native practical comparison | Native frequency, PPR, PCST | 10 HRC | Medium | 30 |
| Feasibility/certificate comparison | Full and guard-PCST | 10 probes | Requested budget or invalid-context stage | 20 |
| **Final selector outputs, excluding tuning grid** |  |  |  | **268** |
| **All selector/parameter evaluations** |  |  |  | **340** |

The 10 medium-budget primary Full and guard-PCST runs are included in the 130 held-out outputs, so they are not double-counted. The grid is development-only and cannot be reopened after held-out annotation.

The held-out medium block contains 10 primary HRC contexts and 16 secondary answerable contexts. Development has six contexts; probes have their own 10-record slice; corruptions are verifier cases rather than narrative query observations. Optional controlled-dense diagnostics use separate pool IDs and cannot be combined with primary results.

### 23.2 Verifier records and total

The required verifier set is 18 development/hidden unit cases, 55 sealed full-verifier cases, and 45 target-guard-removal cases, or 118. Together with 340 selector/parameter evaluations, the archive contains 458 required evaluation records. This number does not turn repeated rows or corruption variants into independent statistical units.

Required primary comparisons are Full versus guard-PCST at medium on 10 HRC contexts; Full versus NoTransitionPriority at medium on those same contexts; Full versus guard-PCST at low/high as budget sensitivity; and the corruption/control bank. Guard-frequency and guard-PPR on all 26 medium contexts and native families on 10 HRC contexts are secondary.

Optional, in drop order, are: six paraphrase pairs (12 outputs); quota-balanced challenge pools for Full, ablation, and guard-PCST at medium (30); ablation low/high runs (20); native probe outputs (30); and any secondary low/high expansion. No optional output is needed for H1. Every omitted block is declared before aggregate held-out results are opened.

## 24. Metrics and result pipeline

Metric definitions live in one versioned module. They consume immutable projections and sealed gold; selectors cannot import them.

### 24.1 Availability, safety, response, and support

For declared context set (Q):

$$
TAR={1\over |Q|}\sum_q TargetAvailable_q,\qquad
SAR={1\over |Q|}\sum_q SupportAvailable_q.
$$

For returned projection (S_{mq}):

$$
Adm_{mq}=I[S_{mq}\in\mathcal A_{adm}],\qquad
Resp_{mq}=I[S_{mq}\in\mathcal A_{resp}].
$$

`AdmissibilityRate` and `ResponseRate` are their means on feasible contexts. `ResponseFamilyExists` is computed directly from the exact table and is distinct from whether a method returned a response. A correct abstention requires both the correct class and a certificate that recomputes. Certificate accuracy is reported by class and overall, with `ERROR_OR_UNKNOWN` always incorrect.

Let (t_q) be the primary transition, (TB(t_q)) its complete transition bundle, and \(\mathcal Q_q\) the accepted full support family after one masked review. Then:

$$
Y_{mq}=I[TB(t_q)\subseteq S_{mq}]\ I[\exists Q\in\mathcal Q_q:Q\subseteq S_{mq}].
$$

`HorizonRareSupportCompleteRecall` is the mean of (Y) over the 10 HRC contexts. `ASCPR` is the mean of (Y\times Adm). The co-primary decision first requires the safety/response gates, then compares HRC recall; relevance cannot compensate for a violation.

For a support structure (Q), fractional preservation is \(|Q\cap S|/|Q|\); complete support preservation is the indicator that at least one accepted (Q) is a subset of (S). Evidence precision is the proportion of displayed answer claims whose chosen anchor is licensed under the scoring oracle. Evidence completeness is the proportion of answer claims retaining every anchor/qualification required by an accepted route. Unsupported assertion rate is one minus evidence precision for factual answer claims.

### 24.2 Violation, diagnostic, and efficiency metrics

The pipeline counts temporal inconsistency, nonentailed temporal display, epistemic-holder/status error, rumor promotion, horizon fact leakage, horizon label leakage, identity error, provenance omission, support omission, and budget overrun separately by severity. A critical leak is never averaged with a minor display defect.

For each category, `ViolationOutputRate` is the number of returned outputs containing at least one violation divided by returned outputs in the declared slice. `ViolationClaimRate` uses displayed answer claims as the denominator where a claim-level denominator exists. Severity is reported as separate critical, major, and minor counts; no weighted total is used to trade one critical disclosure against several harmless defects.

`RareNonconsequentialFalseProtection` is selected horizon-singleton nonconsequential diagnostic transitions divided by available transitions in that cell. `FrequentConsequentialRecall` is complete-bundle recall for horizon-repeated consequential transitions. Both use five-item diagnostic cells only descriptively. Local-rarity versions substitute `LocalCount`; they never redefine the primary target.

The balanced diagnostic contains 20 audited transition-context tuples, five in each rarity-by-consequence cell. The natural-frame description starts from the first five eligible audited tuples in discourse order from each real held-out bundle, 25 total, with no probability weights or population interval. Neither small set is a confirmatory gate.

Outcome relevance is the mean frozen 0 to 3 curator grade of selected roots, with empty output reported separately. Redundancy is the number of selected root pairs sharing a redundancy class divided by \({|X|\choose2}\), defined as zero for fewer than two roots. Root compression is (1-|X|/18). Component compression is reported coordinatewise against the eligible closed-pool inventory. Realized cost records every one of nine coordinates; efficiency reports HRC successes per selected root, evidence cue, and word, without creating a confirmatory composite.

### 24.3 Fixed-benchmark summaries

For Full minus comparator, report every query's paired binary difference, exact win/loss/tie counts, per-bundle mean difference, and leave-one-bundle-out difference. Positive-gain concentration is

$$
Concentration=\max_b {\sum_{q\in b}\max(Y_{Full,q}-Y_{PCST,q},0)\over
\sum_q\max(Y_{Full,q}-Y_{PCST,q},0)},
$$

and is undefined, reported as `NA`, when there is no positive gain. Report AGOT and public held-out slices separately. Do not compute a population p-value or confidence interval.

Annotation uncertainty produces curator-primary, conservative, and permissive versions without changing selection. Budget sensitivity reports low, medium, and high as a curve. Local rarity, controlled challenge, and paraphrase stability are clearly secondary or optional.

The confirmatory decision table implements V2 literally. Full and guard-PCST must both be admissible and responsive on all 10 HRC contexts; Full must exceed guard-PCST HRC recall by at least 0.20, have at least three wins and at most one loss, improve the within-bundle mean in at least three of five held-out bundles, retain a positive aggregate difference after deleting any one bundle, receive no more than 50 percent of its positive gain from one bundle, and have a nonnegative public-held-out difference. The verifier must reject all 45 corruptions, accept all 10 controls, classify all 10 probes correctly, and show no critical violation. These deterministic conditions are evaluated together; no p-value replaces them.

### 24.4 Automated artifacts

One result command generates, without manual copying:

- canonical JSONL `MetricRecord` files;
- CSV per-query audit tables with roots, closure, support, violations, and costs;
- per-bundle and AGOT/public summaries;
- fixed-benchmark decision-gate table;
- budget sensitivity and realized-cost tables;
- Full-versus-PCST and Full-versus-ablation paired tables;
- native/wrapped practical comparison;
- corruption confusion and signature matrices;
- certificate-class table;
- target/support availability table;
- rarity and uncertainty sensitivity tables;
- leave-one-bundle-out and contribution-concentration tables;
- a reproducibility manifest linking every row to run/input hashes;
- paper-ready static figures limited to budget-outcome curves, per-query paired results, and component-cost profiles.

Generated manuscript tables include a source-data hash in a footnote or sidecar. Any edited presentation copy is checked against the canonical CSV. No notebook is authoritative for a metric.

## 25. Testing strategy

Testing follows a pyramid, but scientific invariants outrank a cosmetic coverage percentage. Tests use public-domain and synthetic fixtures only unless they run locally against a restricted manifest.

### 25.1 Unit tests

Unit tests cover:

- strict schema validation, explicit unknown states, forbidden extras, and every conditional field rule;
- opaque identifier generation, duplicate detection, cross-reference type compatibility, and version immutability;
- closure worklist behavior, evidence-choice compatibility, shared-atom deduplication, and dependency cycles;
- each of nine budget coordinates, fixed presentation word counts, and boundary equality;
- horizon filtering for facts, evidence, dependencies, labels, flashbacks, and discourse/revelation positions;
- temporal relation translation, converse normalization, allowed disjunctions, exact/approximate bounds, scenario separation, entailment, and minimized cores;
- assertion holder, mode, polarity, disputed status, and rumor-promotion detection;
- each `GuardSup` response rule and delete-one support failure;
- primary and challenge candidate rules, stable sampling, pressure statistics, and forbidden label access;
- relevance, transition-completeness, redundancy, cost, PPR, frequency, and PCST objective coordinates;
- deterministic tie-breaking and rational comparison;
- every metric formula, zero denominator, `NA`, slice, and uncertainty direction.

Temporal relation tests enumerate endpoints over a small integer domain and compare the Z3 encoding with direct mathematical predicates. Metamorphic checks confirm that adding constraints cannot turn UNSAT into SAT and removing constraints cannot turn SAT into UNSAT.

### 25.2 Property-based tests

Hypothesis generates finite dependency graphs, compatible evidence alternatives, temporal bounds, and budgets. Required properties are:

$$
X\subseteq cl(X),\qquad X\subseteq Y\Rightarrow cl(X)\subseteq cl(Y),\qquad cl(cl(X))=cl(X),
$$

with compatible \(\eta\) where choices matter. Also test worklist-order invariance; union preservation under the declared compatible reachability conditions; horizon inheritance when every edge is horizon-nonincreasing; post-closure horizon rejection when that premise is false; projection roots as a subset of eligible pool roots; every admissible displayed answer claim having a guard-valid support witness; and no previously feasible projection leaving the feasible family when a budget coordinate increases.

Determinism properties permute file order, dictionary order, equivalent constraint order, and process count while requiring identical semantic hashes. Exact-search properties compare table winners with an independent full scan. Candidate properties prove that changing gold labels while method inputs stay fixed cannot change a pool or projection.

### 25.3 Integration tests

Integration tests exercise:

- one public development query from validated records through graph compilation, pool generation, exact table, every selector, verification, metrics, and tables;
- all eight method variants at each budget on small fixtures;
- all certificate classes, including false-abstention rejection and tampered-certificate failure;
- clean and corrupted projections with expected full signatures;
- exact optimum and infeasible fixtures;
- development budget calibration and PCST parameter freeze;
- a sealed manifest run followed by one scoring-only novel-path extension;
- public release generation and a planted private-text leak;
- a clean environment reproduction of the public pipeline.

### 25.4 Regression tests

Freeze known projection IDs, closure hashes, violation signatures, temporal SAT/UNSAT and entailment outcomes, exact mask counts, objective values, PCST connector forests, certificate frontiers, candidate pools, aggregate miniature-benchmark metrics, and result-table hashes. A scientifically intentional change must update the fixture and its protocol rationale together.

Core modules for closure, temporal translation, verifiers, exact enumeration, PCST, certificates, candidate generation, costing, and scoring should reach at least 90 percent branch coverage. Overall 80 percent is an informational floor, not a release substitute. Every listed invariant and every corruption category must pass even if a percentage target is met.

The highest-risk modules are temporal entailment, alternative-aware closure, exact table completeness, PCST connector charging, infeasibility certificates, candidate leakage firewall, annotation-alternative scoring, and release redaction. They require independent fixtures and researcher review of both successful and failing traces.

### 25.5 Code quality requirements

All source uses type hints, descriptive names, small single-responsibility modules, and docstrings that explain scientific semantics rather than restating syntax. Validated records are immutable where practical. Enumerations and metric definitions are centralized. Structured logs use opaque IDs and hashes. There is no hidden global state, implicit current corpus, unrecorded randomness, or dependence on filesystem ordering. Seeds are explicit, sorting is deterministic, and exact numeric values avoid floats.

There is one common guard wrapper, one cost calculator, one experiment runner, and one result schema. Baselines cannot copy admissibility logic. Notebooks may inspect generated data but cannot produce canonical records, metrics, or tables.

## 26. Reproducibility and release

Reproducibility has two honest tiers.

**Complete public reproduction** includes source, environment lock, generated schemas, public-development and public-held-out texts and oracle graphs, the released hidden narrative and truth ledger after unsealing, query contexts, pools, budgets, corruption fixtures, all method configurations, exact table certificates, metrics, safe run records, tests, and table/figure generation. A clean clone runs with the locked environment and regenerates public result hashes on CPU.

**Restricted AGOT alignment reproduction** includes code, schemas, safe contexts, opaque IDs, edition fingerprints, hashes of restricted snapshots, lawful locator policy, counts, budgets, configuration, safe aggregate results, and legally reviewed derivatives. An authorized researcher with the same edition can populate the external restricted root and verify its hash. The public archive does not reconstruct the novel or assume that plot triples are freely redistributable.

At development freeze, run the environment lock consistency check and synchronize exactly from the lock. Record Python, package, operating-system, and Z3 versions. Export a standards-based lock representation and software bill of materials if supported by the frozen environment. Parallelism is across contexts only; each context is deterministic and results are sorted before hashing. Run the final public pipeline twice in separate processes and require identical semantic output, table, and manifest hashes. Timestamps and machine telemetry live outside semantic digests.

The release contains code licensing separately from data rights notices. Public-domain source editions are identified and checksummed. Original handbook, annotations, and hidden story may receive a declared open license. Third-party text never inherits the code license. A release audit applies a positive allowlist, scans Git history and the release archive for forbidden paths/fields/private n-grams, checks visibility propagation, and verifies that every published table can be regenerated from published or hash-resolved inputs.

Public artifacts to preserve include the ontology profile, annotation handbook and templates, JSON Schemas, public and hidden annotations, query and response schemas, candidate-generator definition and seeds, budget calibration procedure and public report, baseline definitions, corruption templates and public fixtures, frozen manifests, exactness certificates, machine-readable metrics, aggregate AGOT results approved for release, and protocol-deviation ledger. Local caches and raw enumeration rows may be omitted if their certificates and deterministic regeneration path are retained.

## 27. Phased implementation roadmap

No phase passes merely because its code executes. The acceptance column states the scientific gate. Hours are researcher time and include review of Codex-generated work. Automated compute is reported separately in Section 28.

| Phase | Hours | Inputs and dependencies | Tasks | Outputs and tests | Acceptance, stop, or revise condition |
|---:|---:|---|---|---|---|
| 1. Repository and environment setup | 3 | V2 and this plan | Recompute source hashes; verify scientific status ledger; initialize package, lock strategy, rights boundaries, logging, and quality configuration | Repository skeleton, environment smoke test, source-hash record | Accept only if clean setup needs no private data; stop if V2 digest differs without explanation |
| 2. Typed schemas and public fixtures | 7 | Phase 1; minimal ontology | Implement strict models, generated JSON Schemas, canonical serialization, IDs, visibility, graph-level reference checks, and abstract fixtures | Valid/invalid fixture suite and schema manifest | Every requested schema and invalid case validates predictably; revise fields before graph work |
| 3. Graph integrity and closure | 5 | Phase 2 | Compile immutable graph snapshot, bit indices, dependencies, evidence choices, and fixed-point closure | Closure module, hashes, unit/property tests | Extensiveness, idempotence, compatible monotonicity, and order invariance pass |
| 4. Temporal and epistemic verifiers | 8 | Phases 2-3 | Implement normalized temporal semantics, Z3 translation, entailment, deterministic core minimization, assertion/status and horizon separation | Temporal/epistemic reports and exhaustive endpoint fixtures | All relation oracle and rumor/status tests pass; `unknown` never passes silently |
| 5. Budget and response feasibility | 5 | Phases 3-4 | Implement remaining guards, presentation costs, response schemas, nested families, and certificates | Common verifier, cost vectors, class fixtures | Safe/nonresponsive and each infeasibility class distinguish correctly; all guards return structured results |
| 6. Exact search | 7 | Phases 3-5 | Implement bitmask enumeration, alternative expansion, caching, exact objectives, table/certificate format, independent scanner | Exact table on small public fixtures | Mask counts, naive comparison, known optima, budget monotonicity, and repeated hashes agree |
| 7. Baselines | 7 | Phases 5-6 | Implement frequency, deterministic PPR, root-incidence connector scorer, native materialization, common wrapper, and development grid logic | Native/wrapped baseline tests and hand-known connector cases | No selector reads gold; exact PCST fixtures and feasible-table hash equality pass |
| 8. Proposed-Full and ablation | 2 | Phases 6-7 | Implement method-visible features, lexicographic Full, and one-coordinate ablation | Objective regression fixtures | Only transition-priority coordinate differs; equal-cap assertions pass |
| 9. Corruption bank framework | 4 | Phases 4-8 | Implement signed mutations, signatures, clean controls, guard-disable harness, and confusion reports | 18 development/hidden category cases | Isolated cases pass non-target guards; overlapping cases declare exact multi-guard signatures |
| 10. Annotation and horizon-audit tooling | 6 | Phases 2-5 | Finalize handbook/templates, YAML/CSV compiler, issue ledger, masked reannotation packets, search packets, deduplication, and freeze tools | Public synthetic workflow and leakage tests | Researcher can annotate and audit without hand-editing canonical JSON; raw text never enters logs |
| 11. Complete public-development pilot | 13 | Phases 1-10 | Annotate exactly 25 public roots; author/finalize hidden control; build six contexts/pools/supports; run two rarity audits; calibrate budgets; tune PCST; run all methods, corruption fixtures, metrics, and reports; log minutes | End-to-end public report, actual-rate ledger, provisional frozen configs | All four task types, 18-root pressure, exactness, and reports work; otherwise revise development components only |
| 12. Feasibility and workload reforecast | 1 | Phase 11 actuals | Apply Section 28 equation and protected-scope rules | Signed forecast and scope decision | Go at <=145 required hours; revise before held-out at 145-160; stop if still >160 |
| 13. Methodology/configuration freeze | 3 | Phases 11-12 | Finalize preregistration, canonical configs, tests, budgets, parameter winner, templates, run matrix, and hashes | Write-once manifest and timestamped commitment | Held-out files remain unopened; any mutable scientific setting blocks progress |
| 14. Held-out annotation and stability | 46 | Phase 13; lawful access | Select/freeze five episode frames; annotate exactly 25 roots each; run up to 30 remaining master horizon audits plus eight delayed re-audits; reannotate 24 stratified natural roots after 14 days; reconcile and compile alternatives | Five graph snapshots, audit/stability reports, immutable issue ledger | V2 reliability and rarity gates pass; otherwise use declared fallback or stop superiority claim |
| 15. Held-out queries and candidate-pool freeze | 9 | Phase 14 records; frozen templates | Build 26 contexts, response schemas, one to three support references, alternatives, 10 probes, 10 HRC targets, and 18-root primary pools; confirm one guard-valid reference response fits the high cap for every answerable context; run pressure/leakage checks; seal gold | Context/pool/gold manifests and target-availability report | Counts agree, high-budget answerability is witnessed, no gold leakage occurs, and TAR/SAR are 1.00 for HRC; no target-aware repair is allowed |
| 16. Experiment execution | 5 | Phases 13 and 15 | Generate the 26 held-out answer tables and eight typed-probe tables plus two invalid-context records; retain 72 tuning and 48 final-development outputs from Phase 11, then run 220 held-out/probe outputs plus 100 sealed verifier records, yielding the declared 340-selector archive; rerun a deterministic sample; keep aggregates sealed | Run records, certificates, runtime/memory ledger | Complete counts, no errors, identical common-table hashes, and deterministic reruns; otherwise fix only demonstrated implementation defects |
| 17. Masked path review | 4 | Phase 16 projections; frozen rubric | Deduplicate at most 78 elected paths, target <=36 distinct; mix with decoys, deidentify method, classify once | Masked decision ledger and scoring-only gold extension | Blinding and sufficiency rubric documented; selections/tables unchanged; overflow invokes HRC-only review fallback |
| 18. Final scoring and sensitivity | 5 | Phases 16-17; sealed gold | Run one rescore, uncertainty modes, fixed-benchmark gates, bundle/LOO/budget/rarity analyses, corruption reports | Canonical metrics and decision table | Every context is reported; no population inference; null/negative results retained |
| 19. Artifact packaging | 3 | Phase 18 | Run rights/leakage audit, clean public reproduction, hash archives, write reproduction documentation | Public and restricted-tier packages and audit report | No copyrighted text leak; public bundle reproduces; manifest resolves every result |
| 20. Manuscript integration outputs | 1 | Phase 19 | Generate final tables, static figures, source-data sidecars, and methods checklist | Paper-ready artifact directory | Every number is generated and hash-linked; no manual numerical copying |
|  | **144** |  | **Required researcher total** |  | **16 hours remain before the 160-hour hard stop** |

Dependencies are strictly ordered except that documentation can be updated alongside a phase. In particular, held-out annotation cannot begin before Phase 13, method execution cannot begin before held-out gold is sealed, and scoring cannot load gold until projections are immutable.

## 28. Realistic workload and contingency

The 144-hour estimate is conservative relative to a prototype-only build but still assumes disciplined scope. It corresponds to 18 weeks at eight hours per week or 24 weeks at six, so it satisfies the hour ceiling but not a literal 16-week calendar at the lower weekly allocation. Calendar duration must expand rather than hide labor.

### 28.1 Activity-level audit

| Required activity | Researcher hours |
|---|---:|
| Literature/status verification and V2 traceability | 1 |
| Repository, environment, configuration, and rights-boundary design | 2 |
| Minimal ontology, handbook, typed schemas, and public fixtures | 6 |
| Graph representation, integrity, and closure | 5 |
| Temporal solver and epistemic/horizon verification | 8 |
| Evidence, identity, provenance, budget, response, and certificates | 5 |
| Exact enumeration, caching, and independent exactness checks | 7 |
| Horizon-filtered native frequency/PPR/PCST | 3 |
| Common guards, exact connector scoring, and PCST development selection | 4 |
| Proposed objective and direct ablation | 2 |
| Corruption framework, signatures, and development cases | 4 |
| Annotation and horizon-audit tooling | 4 |
| Cross-cutting automated tests, debugging, and leakage fixtures | 3 |
| Complete public-development and hidden-control pilot | 13 |
| Workload reforecast | 1 |
| Preregistration, configuration freeze, and manifest | 3 |
| Held-out corpus selection and boundary audits | 3 |
| First-pass annotation of 125 held-out roots | 27 |
| Horizon audit passes, up to 32 initial total and 8 delayed total | 12 |
| Delayed reannotation of 24 natural roots and reconciliation | 5 |
| Held-out query, context, response, and support construction | 5 |
| Dense primary candidate-pool assembly and leakage/pressure checks | 4 |
| Confirmatory runs and corruption execution oversight | 3 |
| Deterministic rerun, runtime, and manifest audit | 2 |
| Masked novel-path review | 4 |
| Fixed-benchmark, budget, rarity, and uncertainty analyses | 4 |
| Automatic final tables and figures | 1 |
| Reproducibility documentation and packaging | 3 |
| **Required total** | **144** |
| **Unallocated contingency** | **16** |
| **Hard ceiling** | **160** |

The first-pass estimate assumes about 13 minutes per root. Codex-generated code and transformations remain inside the listed review and testing hours; they do not count as free. Automated compute is estimated at 6 to 30 CPU hours for tables, probes, repeated runs, and reports, with at most 3 hours of active oversight already counted. If compute exceeds 48 hours or memory exceeds ordinary workstation capacity, optimize caching and serialization without changing the search space; a heuristic substitution is not allowed for confirmatory results.

### 28.2 Pilot forecast equation

At the end of Phase 11, record actual elapsed researcher minutes for each public root, query/support pair, pool, horizon-audit pass, reannotation item, exact-table review, and masked development path. Let `spent_hours` include every activity completed. The conservative forecast is:

```text
projected_total_hours = spent_hours
                      + remaining_fixed_hours
                      + max(planned_root_minutes, observed_root_minutes) * remaining_roots / 60
                      + max(planned_query_minutes, observed_query_minutes) * remaining_queries / 60
                      + max(planned_pool_minutes, observed_pool_minutes) * remaining_pools / 60
                      + max(planned_audit_minutes, observed_audit_minutes) * remaining_audit_passes / 60
                      + max(planned_reannotation_minutes, observed_reannotation_minutes) * remaining_reannotation_roots / 60
                      + max(planned_path_minutes, observed_path_minutes) * forecast_distinct_paths / 60
```

At the planned checkpoint the maximum remaining counts are 125 roots, 26 contexts, 26 pools, 38 audit passes after two development pilots, 24 reannotation roots, and 36 distinct new paths after deduplication. `remaining_fixed_hours` includes freeze, boundary review, experiment oversight, analysis, and packaging not represented by rates. Planned rates are calculated from the 144-hour table and printed by the forecast command; they are not hidden constants.

- **Go:** projected required work is at most 145 hours.
- **Revise once:** 145 to 160 hours. First drop every optional run. Then restrict masked path review to the three medium-budget methods on the 10 HRC contexts, remove nonprimary native outputs and nonprimary guard-frequency/PPR outputs, and remove optional context variants or secondary figures only through a documented V2.1/preregistration correction made before held-out annotation.
- **Stop or redirect:** the forecast remains above 160, or a required protected control would need removal.

Protected work is the independent public held-out test, primary horizon audits, guard-PCST, NoTransitionPriority, the primary verifier bank, exact HRC search at three budgets for Full/guard-PCST, and dense method-independent pools. The contingency is for bounded defects or slower primary work, not new features.

## 29. Risk register

| Risk | Likelihood/impact | Early indicator | Mitigation | Required fallback |
|---|---|---|---|---|
| Candidate generator misses target | Medium/high | TAR or SAR below 1 on sealed HRC | Method-visible top-six plus seeded fill; report availability before selection | Benchmark-construction failure; no gold repair; narrow to available-query result |
| Pool lacks real pressure | Medium/high | Fewer than nine irrelevant/contextual roots or weak component ratios | Reuse same-work annotated roots and audit pressure after freeze | Report controlled-density failure; optional challenge cannot rescue primary claim |
| Annotation takes longer than 13 minutes/root | High/high | Development or first 25 held-out roots exceed rate | Immediate forecast; remove secondary outputs | V2.1 scope reduction before results or stop above 160 |
| Horizon search misses co-referring mention | Medium/high | Delayed audit changes categories | Multiple search modes, full hit ledger, eight re-audits | Use locally singleton/query-necessary terminology |
| Single-curator interpretation unstable | Medium/high | V2 stability thresholds fail | Handbook, masked delayed pass, licensed alternatives | Drop consequence/pivotality interpretation or superiority claim as specified |
| Gold leaks into pools/selectors | Low/critical | Import/access ledger or planted-field test fails | Physical/type-layer firewall and hash audit | Invalidate affected pools/runs and re-preregister before rerun |
| Exact table too slow or incomplete | Medium/critical | Count mismatch, timeout, or missing variant | Bitset closure, context-level parallelism, caching | Optimize exact implementation; do not substitute heuristic; drop zero-gap claim if unresolved |
| Temporal semantics incorrectly encoded | Medium/critical | Endpoint oracle mismatch or unstable core | Direct mathematical fixtures and independent entailment checks | Stop contract study until corrected and rerun all affected artifacts |
| PCST comparator is unfair or mislabeled | Medium/high | Hidden connector, prize double count, or approximation | Root-incidence definition, exact MST tests, development grid | Rename/narrow comparator; no primary claim until exact scorer passes |
| Corruptions are not isolated | High/medium | Non-target guards fail with target disabled | Store full expected signatures and multi-guard status | Report joint signature; do not claim isolated sensitivity |
| Held-out verifier miss | Medium/critical | Any of 45 accepted or clean control rejected | Seal expected signatures and preserve miss | Contract gate fails; repaired case is development only |
| Novel valid support path omitted | Medium/medium | Many unlisted paths | One masked cross-method review and rescore | HRC-only review if volume exceeds forecast; other scores remain lower bounds |
| Public transfer is negative | Medium/medium | Hound paired difference below zero | Strict independent split and separate reporting | Restrict conclusion to four AGOT bundles |
| Restricted text leaks | Low/critical | Release scanner or Git-history check finds material | External restricted root, allowlist release, safe fixtures | Stop release, remediate history privately, obtain rights review |
| Result changes across reruns | Low/critical | Semantic hash mismatch | Exact rationals, sorted IDs, locked dependencies, explicit seeds | Stop scoring until nondeterminism is found and all affected runs repeated |
| Scope creep | High/high | LLM, UI, RDF export, or extra corpora enter backlog | Paper 2 list and 160-hour ledger | Delete/defer feature; protected controls take priority |

The sole researcher owns every risk and logs reviews. There is no assumed external adjudicator or engineer.

## 30. Stop, revise, and go gates

| Gate | Go | Revise | Stop or claim consequence |
|---|---|---|---|
| Source trace and lawful access | V2 hash verified; editions lawful | Correct locator policy before annotation | No AGOT implementation without lawful access |
| Schemas and competency coverage | Four response types represented without optional ontology inflation | Simplify/defer fields | Stop if minimal graph cannot express required answers |
| Public pilot | Full local pipeline and all method families complete | Development-only changes and repeat | Do not open held-out data until complete |
| Workload | Forecast <=145 | 145-160: execute declared secondary reductions before held-out | >160 after reduction: redirect scope |
| Exactness | Counts, independent maximizer, connector tests pass | Fix implementation before freeze | No zero-gap or H1 run |
| Candidate independence | Leakage audit passes | Correct generator before pool freeze | Affected benchmark invalid |
| Target/pressure availability | HRC TAR=1, SAR=1, density criteria pass | No held-out gold repair; document design failure | Narrow/fallback benchmark; do not claim selection loss for unavailable target |
| Annotation stability | V2 thresholds pass | One handbook-consistent pre-result reconciliation | Use local/query-necessary fallback or stop construct claim |
| Horizon rarity | F1 >=0.85 and 7/8 category stability | Mark uncertain | Remove horizon-rarity claim |
| Contract implementation | 45/45 corruptions rejected, 10/10 controls accepted, expected signatures agree | Preserve miss and diagnose | No confirmatory utility interpretation |
| Feasibility probes | 10/10 correct classes and valid certificates | Correct demonstrated code defect under protocol | No abstention/certificate claim |
| Utility | V2 H1 fixed-benchmark rule passes | Report mixed/null exact results | H1 unsupported, not hidden |
| Mechanism | Full beats ablation by frozen pattern without unacceptable diagnostic tradeoff | Comparator-only interpretation | No transition-priority attribution |
| Uncertainty | Direction stable under conservative scoring | Qualify as curator-dependent | Withdraw robust interpretation |
| Public transfer | Nonnegative public held-out difference | AGOT-only result | Negative transfer is an explicit boundary result |
| Release | Public reproduction and leakage audit pass | Repair release packaging without changing results | No public artifact release until safe |

A revise decision never authorizes favorable episode replacement, target insertion, budget adjustment from test outcomes, or objective retuning.

## 31. Definition of done for Paper 1 implementation

Implementation is complete only when all of the following are true:

- all required public fixtures pass in a clean locked environment;
- every requested schema validates and all cross-record references resolve;
- closure extensiveness, monotonicity under compatible choices, idempotence, and order invariance pass;
- temporal relation, disjunction, entailment, horizon-separation, and contradiction fixtures pass;
- exact search matches naive and independent small-instance checks and accounts for every expected mask/choice;
- exact root-incidence PCST matches hand and exhaustive connector fixtures;
- guard-PCST is selected on public development, reviewed, hashed, and frozen;
- every wrapped selector consumes the identical feasible-table, guard, cost, and budget hashes;
- corruption cases produce their expected isolated or multi-guard signatures and clean controls pass;
- candidate-pool leakage audit and planted-label firewall tests pass;
- the complete public-development pipeline runs from annotation compilation through paper tables;
- the pilot forecast is within 160 hours after any preregistered scope reduction;
- held-out oracle, alternatives, contexts, pools, and gold are sealed before any method output;
- all primary and required sensitivity runs finish without operational unknowns and repeat deterministically;
- all elected novel support paths are deduplicated and reviewed under method masking within the frozen scope;
- curator-primary, conservative, and permissive scoring is generated without changing selections;
- every metric, decision gate, paper table, and figure is generated automatically from hashed runs;
- no raw or reconstruction-ready copyrighted text appears in Git or the public release;
- a fresh public clone reproduces the public-domain and released hidden-control outputs from the frozen manifest;
- all deviations, defects, reruns, negative gates, and claim reductions remain in the final audit ledger.

Code completion without these scientific conditions is not completion.

## 32. First execution prompt for the next Codex session

Use the following bounded prompt to start implementation without reopening the methodology:

> Read the complete files `SOLO_RESEARCHER_CONTEXTUAL_NARRATIVE_PROJECTION_PLAN_V2.md` and `IMPLEMENTATION_PLAN_ADMISSIBILITY_CONSTRAINED_PROJECTION.md`. Verify and record their SHA-256 hashes. Execute only Phases 1 and 2 of the implementation plan: create the minimal repository and locked Python project, establish public/private data boundaries, implement the strict typed schema layer and canonical serialization, generate JSON Schemas, and add public abstract valid/invalid fixtures plus unit tests. Do not access or copy AGOT text, annotate any study episode, build selectors, tune parameters, run experiments, create an LLM or UI feature, or modify either plan. Use `apply_patch` for edits, keep restricted paths outside Git, run the declared quality and schema tests, and report created files, test results, unresolved schema ambiguities, actual researcher-review time, and whether the Phase 2 acceptance gate passes. Stop after Phase 2 and request a scientific review before beginning graph closure.

That first session should favor a small, inspectable foundation. It must not preempt the public-development pilot or create held-out artifacts before the freeze sequence permits them.
