# Implementation Plan: Conference Study of Active Query-Dependent Temporal Ontology Construction

**Document status:** authoritative engineering plan; this document does not implement software.
**Methodological authority:** `METHODOLOGICAL_PLAN_QUERY_DEPENDENT_TEMPORAL_ONTOLOGY.md`.
**Target:** one academic researcher, one RTX 4090 (24 GB), approximately 31 GB RAM, 8 vCPUs, at most 9 scheduled and 10 actual allocated GPU hours, and at most 30 GB project-controlled writable storage.
**Planning date:** 2026-09-03.

## Conference scope boundary

This plan implements only the thin vertical slice needed for a defensible conference paper. Deferred work includes novels two through five; series-wide inference; recruited participants or usability experiments; multi-user and collaborative editing; branching histories and elaborate undo/redo; hosted or production deployment; authentication; distributed workers; durable service queues and Server-Sent Events; multiple LLM backends; second-model, prompt-matrix, or thinking-mode studies; the previous broad ablation registry; multiple community algorithms; large layout/screenshot batteries; vector databases; nested belief worlds; full bitemporal curation; extensive RDF 1.2 work; alternative graph stores; and duplicated notebook/reporting pipelines.

The minimum application is a local single-user graph page with synchronous or simple-polled jobs. The minimum corpus is one locally supplied lawful novel. The minimum test suite protects scientific boundaries and reproducibility without becoming a production certification program.

Historical repository plans remain superseded where they specify a complete oracle graph, exhaustive deterministic subset selection, fixed templates, CPU-only execution, or deferral of the LLM, narrative ingestion, feedback, and visualization. Their useful patterns—typed artifacts, gold isolation, provenance, temporal validation, resource gates, and resumable records—remain.

## Non-negotiable contribution

This scope-control section is immutable. No resource fallback may remove:

1. active query-time GPU LLM ontology construction;
2. separately implemented C0, C1, and C2;
3. `A-FixedSelect`;
4. temporal qualified assertions;
5. contrastive known-answer contextual projections;
6. rare-pivotal preservation and evaluation; or
7. the bounded first-novel narrative case study.

If these do not fit, the runner stops before the scientific run. During Phase 1 it may symmetrically change the one model; before freeze it may reduce a common query-blind evidence/sequence window through a documented amendment. It may not shrink the mandatory 12/8/8 ablation subsets, reduce `A-FixedSelect`, turn C2 into fixed-graph retrieval, use a labeler as the treatment, or substitute a CPU template pipeline.

## 1. Thin architecture and data flow

The system has a strict pre-query/query-time boundary:

```text
PRE-QUERY
local novel or public synthetic text
  -> rights check and passage segmentation
  -> mention/alias/event candidates and temporal clues
  -> immutable EvidenceSnapshot D_z + SQLite FTS index
       |                         |
       +-> C0 CPU build          +-> C1 GPU build
       |      sealed O_pre^C0         sealed O_pre^C1
       +-> C2 PreQueryInventory (no ontology)

QUERY REVEAL
QueryContext q + frozen packet E_q
       |                 |                    |
       v                 v                    v
C0 fixed project   C1 fixed project     C2 GPU constructs O_q
       |                 |                    |
       +-----------------+--------------------+
                         v
       CPU schema/evidence/time/lineage/budget validation
                         v
       compressed artifact + SQLite ledger + graph DTO
                         v
              local Cytoscape.js page
                         |
       condition-independent UserRevision / per-condition resolution
                         |
         C0/C1 capability response; C2 GPU regeneration
```

`A-FixedSelect` uses the C2 query-time GPU route but receives the complete required C1 serialization and a restricted grammar. Every arrow writes a small immutable artifact or a reference to one. Evidence and preontologies are stored once and referenced by SHA-256; a run record does not duplicate them.

The implementation distinguishes:

- stable `UpperOntology`;
- query-time `LocalContextSchema`;
- query-specific `InstanceGraph`;
- first-class `QualifiedAssertion` records; and
- renderer-only `VisualizationState`.

Creating or changing the last object is not ontology construction. C2 must create or revise nonselection objects in the middle three after query reveal.

## 2. Hardware and hard resource enforcement

### 2.1 Hardware envelope

The reference host has one NVIDIA RTX 4090 with 24 GB VRAM, approximately 31 GB usable system RAM, and 8 vCPUs. Only one model server and one generation request run at a time. Main-model CPU offload is prohibited. CPU validation and compression may overlap only when measured RAM remains below 25 GB, preserving approximately 6 GB for the OS and safety.

The runner samples GPU allocation start/end, service uptime, VRAM, RAM, disk use, request duration, and failure state. GPU time means allocated device/service wall time, not successful token-generation time. Loading, warm-up, schema probes, failures, timeouts, repairs, and restarts count.

### 2.2 Complete writable-storage budget

All project-controlled writable files share one quota root. The base OS or immutable container layer may be reported separately only when it is physically outside this writable allocation.

| Category | Peak allocation | Included material |
|---|---:|---|
| One pinned quantized model and tokenizer | 11 GB | AWQ weights, tokenizer, configuration, chat template; exactly one snapshot |
| Locked environment and project runtime/download cache | 5 GB | virtual environment, compiled wheels, vLLM/PyTorch runtime files, remaining project-owned caches |
| Source, datasets, private corpus, and indexes | 2 GB | repository source, synthetic worlds/gold, one restricted novel, passage map, SQLite FTS |
| Compressed generations and traces | 2 GB | requests, raw responses, repairs, diagnostics and validation traces as `jsonl.zst` |
| Ledger, metrics, schemas, and graphs | 1 GB | SQLite, compact Parquet/CSV tables, manifests, projection JSON, geometry |
| UI assets and selected figures | 1 GB | one Cytoscape bundle, styles, visual fixtures, no more than 12 retained paper/test PNGs |
| Regenerable temporary space | 3 GB | atomic partial files, decompression buffers, quarantine and release staging |
| Required free headroom | 5 GB | untouchable during every phase |
| **Peak writable allocation** | **30 GB** | **at most 25 GB occupied plus 5 GB free** |

One environment variable such as `STORY_PROJECTION_CACHE` points Hugging Face, Transformers, and vLLM references to the same cache. The manifest records the single 40-character model revision. It never downloads both `main` and that revision or copies weights into the repository. Package-download and build caches are pruned after the locked environment and hashes verify. A rejected model snapshot is removed before a fallback snapshot is downloaded.

Raw generations use zstd-compressed JSONL. SQLite stores metadata and content hashes, not duplicate payloads. Parquet is used only for compact metric tables when it is smaller than compressed JSONL. The system never stores logits, token-probability banks, attentions, hidden states, KV-cache dumps, videos, every layout seed, or a screenshot per projection. Copyrighted text exists once in the restricted input directory and is referenced by opaque IDs.

Before model download and every phase, `StoragePreflight` sums current bytes, declared phase growth, largest atomic temporary file, quarantine allowance, and release staging. Model-shard `.incomplete` files and Xet/blob staging are charged within the 11 GB model category and must replace, not duplicate, the final cache object; the 3 GB temporary category never has to hold a second 4.99 GB shard. It refuses work if projected occupancy exceeds 25 GB or actual/projected free space falls below 5 GB. Regenerable partial downloads, decompression buffers, invalid unverified temporaries, and redundant package downloads may be removed. The only copy of a verified scientific artifact is never deleted until its hash is checked and it is copied to genuinely external durable storage. Every project-controlled path, mount, cache, staging area, and same-machine archive counts toward the quota; an external archive is reported separately and never licenses a duplicate local copy. The public reproduction bundle targets less than 2 GB and excludes weights and copyrighted text.

## 3. GPU model, runtime, and decoding

### 3.1 Recommended model

Use one pinned [`Qwen/Qwen3-14B-AWQ`](https://huggingface.co/Qwen/Qwen3-14B-AWQ) snapshot with vLLM. The official checkpoint is a four-bit AWQ form of the 14.8B Qwen3 instruction model and occupies approximately 9.99 GB before runtime files; the storage category permits 11 GB. The verified candidate planning revision is [`1a6fe1ecf891437a270cce11ad54d796c4f56ce0`](https://huggingface.co/Qwen/Qwen3-14B-AWQ/tree/1a6fe1ecf891437a270cce11ad54d796c4f56ce0), but Phase 1 must download it once, hash every file, confirm its Apache-2.0 license, and freeze the exact snapshot. Never use a moving branch for a scientific run.

The pilot must demonstrate:

- peak VRAM below 23 GB without CPU weight offload;
- aggregate measured project process RAM below 25 GB;
- schema-constrained output for C1, C2, and `A-FixedSelect`;
- complete-packet/complete-sealed-graph packing;
- p50 and nearest-rank p95 service-wall latency by call class;
- clean restart and seed/config logging; and
- sufficient grounding and operator behavior on development fixtures.

If 14B fails during the Phase-1 acceptance pilot, record the artifact and remove its snapshot, then install one pinned 7B/8B open-weight AWQ model. The fallback micro-pilot is bounded to one reserved load event, one long C1-form call, two standard C2-form calls, one short `A-FixedSelect` call, and at most one short repair. These consume the predeclared reserve rather than creating uncounted work. The runner recomputes the complete remaining p95 schedule and proceeds only if it remains at most 9 hours. The fallback then freezes for C1, C2, `A-FixedSelect`, and LLM ablations. After Phase 1, model failure or an excessive forecast causes an amendment or abort, never an unbudgeted model switch or development rerun.

### 3.2 Sequence and runtime settings

Initial settings are:

| Setting | Conference default |
|---|---|
| vLLM tensor parallelism | 1 |
| batch/concurrency | 1 |
| activation/runtime dtype | float16 where supported by pinned checkpoint/runtime |
| `max_model_len` | 12,288 total tokens |
| first-pass input/output maxima | 10,240 / 2,048 |
| repair input/output maxima | 10,752 / 1,536 |
| target evidence allocation | at most 6,144 input tokens |
| GPU memory utilization | start at 0.88, freeze after pilot |
| prefix/speculative decoding | disabled for primary timing |
| thinking mode | disabled |

Prompts, schema, evidence, and—only for `A-FixedSelect`—the complete required sealed graph must jointly fit. `PackingReport` rejects a request rather than truncating it. If development content does not fit, reduce and refreeze the query-blind world/window for all conditions. Held-out condition-specific truncation is an integrity failure.

### 3.3 Complete decoding decision

The primary first-pass configuration is:

- `temperature=0.7`;
- `top_p=0.8`;
- `top_k=20`;
- `min_p=0.0` (disabled);
- `presence_penalty=0.0`;
- `frequency_penalty=0.0`;
- `repetition_penalty=1.0`;
- `n=1`, `best_of=1`, no beam search;
- non-thinking chat-template control;
- exact EOS/end-of-turn IDs from the pinned tokenizer; and
- one recorded seed per seed block.

Temperature, top-p, and top-k follow the model's non-thinking guidance; min-p and penalties are explicitly disabled rather than silently omitted or misrepresented as part of a “complete model-card recommendation.” C1, C2, and `A-FixedSelect` share these settings. Repairs differ only in the declared shorter output limit and receive the same semantic budget policy. Model weights, tokenizer, chat template, quantization, vLLM, PyTorch, CUDA/driver, and structured decoder freeze after the Phase-1 acceptance pilot; the one study prompt revision and output schema freeze after the later 24 development calls and before held-out reveal.

## 4. Minimal technology and local CPU workflow

### 4.1 Selected stack

| Concern | Selection | Reason |
|---|---|---|
| Language/environment | Python 3.12 with `uv` lock | one typed implementation and compact reproducibility |
| Contracts | Pydantic 2 and JSON Schema | validation plus constrained generation |
| LLM | one vLLM/PyTorch/Hugging Face backend | no backend matrix or duplicate weights |
| C0/evidence NLP | spaCy CPU tokenizer/NER/dependencies plus deterministic alias, event, relation, and temporal rules | compact credible classical baseline without a Java service |
| Retrieval | SQLite FTS5/BM25 | primary units use all evidence; one small full-novel retriever |
| Graphs | NetworkX for conversions/metrics; `python-igraph` plus `leidenalg` only for Leiden | small graphs and one community implementation |
| Storage | SQLite plus content-addressed `jsonl.zst`; compact Parquet only when smaller | resume and deduplication under 30 GB |
| Analysis | NumPy, SciPy, pandas | exact permutation, bootstrap, tables |
| Local UI | thin FastAPI process plus one vendored plain-JavaScript Cytoscape.js page | typed local endpoints without production infrastructure |
| Tests | pytest and a small Hypothesis set | one test runner; bounded property coverage |

There is no vector database, sentence-transformer index, FAISS cache, CoreNLP server, Infomap package, alternate renderer, R analysis duplicate, or notebook-owned scientific logic. A small RDF 1.1-compatible export is optional and may use a simple deterministic serializer; RDFLib and RDF 1.2 experimentation are not conference-critical dependencies.

### 4.2 Local CPU development

Without loading the LLM, the researcher can:

1. generate four tiny development worlds and all gold;
2. ingest synthetic or locally authorized text;
3. run C0, SQLite retrieval, schemas, temporal checks, metric fixtures, ledger, compression, and UI DTO creation;
4. inject hand-written response fixtures at the parser boundary to test C1/C2 orchestration without creating a second backend;
5. run the CPU test suite on three 10-node fixtures in under ten minutes and less than 8 GB RAM; and
6. prepare immutable GPU request manifests.

Only records with `backend=vllm_gpu`, pinned hardware/model manifests, and measured budget entries count as scientific C1/C2/`A-FixedSelect` output.

### 4.3 Compact repository layout

```text
configs/             dev/ study/ ablations/ visualization/
prompts/             c1_pre/ c2_query/ fixed_select/ repair/
schemas/             jsonschema/ upper_ontology/
src/story_projection_onto/
  contracts.py       evidence.py       temporal.py
  conditions/        c0.py c1.py c2.py fixed_select.py
  llm.py             validate.py       benchmark.py
  metrics/           alignment.py contrastive.py rare.py
                     entropy.py clutter.py community.py feedback.py
  experiment.py      analysis.py       store.py          api.py
ui/                  index.html app.js cytoscape.min.js style.css
tests/               unit/ property/ integration/ e2e/ gpu/
data/synthetic/      public generated worlds and gold
.local_data/         ignored restricted first-novel input/index
artifacts/           blobs/ restricted/ public/
```

Exploratory notebooks may read frozen tables but cannot define metrics or paper values.

## 5. Typed records and invariants

Every record has a schema version and content hash. Unknown, not-applicable, horizon-withheld, and invalid are distinct.

| Record | Essential fields and invariants |
|---|---|
| `Document` / `Passage` | corpus/edition, chapter/order, restricted text handle, offsets/hash, discourse coordinate, rights class; public artifact has no prose |
| `EvidenceSpan` | passage, token/character span, restricted text pointer/hash, discourse position, provenance/confidence; offsets resolve |
| `MentionCandidate` | evidence/offsets, surface hash, provisional type, candidate aliases/coreference scores; no final entity ID |
| `EventCandidate` / `RelationPhraseCandidate` | evidence and mention endpoints/triggers, surface form, uncertainty; no final event/predicate |
| `TemporalClue` | evidence, normalized expression or partial-order candidate, target candidates, confidence; does not finalize assertion time |
| `EvidenceSnapshot` | query-blind world/window, one registered horizon, eligible evidence IDs, index/config hash, pre-query seal; no contextual ontology |
| `EvidencePacket` | snapshot hash, ordered evidence IDs, all-evidence or FTS method, ranks/scores, token count, horizon rejections; one hash shared across conditions |
| `UpperOntology` | small stable primitive entity/event/relation/time/epistemic vocabulary |
| `LocalContextSchema` | projection-local contextual types and predicates, definitions, arity/domain/range/roles, parent upper term, abstraction |
| `Entity` | projection ID, supported mentions/aliases, contextual type/role, abstraction, temporal state, uncertainty, evidence |
| `Event` | projection ID/type, occurrence interval/order, reification reason, evidence; participants/causes are qualified assertions |
| `TemporalScope` | separate story/event time, validity bounds/partial order, discourse position, proposition-revelation position, precision and unknown reason |
| `PropositionContent` | nonasserted content ID, normalized relation and endpoints/roles, evidence anchors, temporal content; becomes world truth only through a separate world-committed assertion |
| `EpistemicScope` | optional single holder, attitude, holder-relative time, `PropositionContent` reference, evidence; never endorses that content globally |
| `QualifiedAssertion` | proposition/content reference, endpoints or n-ary roles, local predicate/direction, time, optional epistemic scope, explicit narrative commitment, confidence, evidence/provenance, relevance and supported `why_matters` |
| `ValidationRecord` | independent structural `validation_status`, evidence-support status, temporal determination/contradiction status, commitment check, diagnostics and repair lineage |
| `QueryContext` | wording, lens/target, story scope, spoiler horizon, viewpoint, abstraction, node/assertion/display budgets; no gold, split, pair, expected-effect, or condition-comparison IDs |
| `ModelVisibleQueryContext` | allowlisted view of `QueryContext` containing only task semantics and budgets; serialization proves experiment/world/split/pair/gold fields absent |
| `OntologyDecision` | include/exclude, merge/split, type, schema/relation, event reification, abstraction, qualification, evidence and post-query timestamp |
| `OntologyProjection` | context/snapshot/packet hashes, upper/local schema, entities/events/assertions, decisions, omissions, validation, certificate, budget/run IDs |
| `GoldContextualProjection` | scorer-only world/query IDs, local schema, anchor-based entity partition, events, qualified assertions, relevance, rare/pivotal flags, communities, signed contrast decisions/invariants and review/adjudication status |
| `GoldAlternativeSet` | permissible projection IDs or constraint-based alternatives, equivalence/matching rule and matcher revision; never model-visible |
| `FeedbackAnchor` | condition-independent evidence/mention IDs, normalized requested semantic signature, optional time/role constraint; no projection-local ID or gold |
| `UserRevision` | condition-independent action, anchors, typed requested change, rationale/evidence, sequence, before/after context hashes; no condition result |
| `FeedbackResolution` | runner-only receiving condition, its resolved IDs/status, capability result, before/after projection IDs, resolver/config/seed; not serialized as a cross-condition map |
| `ModelVisibleRevision` | allowlisted shared revision plus at most the receiving condition's own resolution; no other-condition output/status |
| `VisualizationNode` / `VisualizationAssertion` | projection-object references plus contextual labels, role/type, time, uncertainty/holder status, evidence badge, direction/roles, provenance and supported `why_matters`; no new facts |
| `VisualizationState` | projection hash, one layout/style/font/viewport/seed, coordinates, labels/visibility, temporal filter; semantic hash unchanged by renderer actions |
| `RunManifest` | code/env/model/prompt/schema/config/seed/input/output hashes, timings/resources, parents, release class, failure/retry lineage |

Pipeline `validation_status`, narrative commitment, evidence support, and temporal underdetermination are orthogonal. A structurally accepted record may faithfully represent a disputed belief; an unsupported world assertion still fails evidence policy.

## 6. Structured LLM contracts and condition interface

`PreconstructionRequest` is used by C1 and contains the sealed snapshot/evidence, small upper ontology, prebuild budget, and exact runtime/config identifiers—never a held-out query, context, query-derived order, or revision. `ConstructionRequest` is used only by C2 and `A-FixedSelect`; it contains `ModelVisibleQueryContext`, complete frozen packet contents, small upper ontology, budgets, permitted capabilities, exact runtime/config identifiers, and model-visible revision history when applicable. Neither contract contains a `GoldContextualProjection`, rare/pivotal labels, experiment pair/group IDs, expected effects, or another condition's projection.

`OntologyDraft` must return:

1. contextual interpretation;
2. local types and relation definitions;
3. entity assemblies and merge/split decisions;
4. events and reification decisions;
5. qualified assertions with roles, time, holder status, confidence, evidence, relevance, and supported descriptions;
6. omissions, uncertainty, and abstentions;
7. construction decisions/certificate material; and
8. token/object budget accounting.

Free prose outside typed fields is not trusted. Every factual clause in labels or `why_matters` maps to an accepted supported assertion. The parser may normalize Unicode and ordering, but semantic repair returns to the LLM.

The common interface is:

```text
prepare(snapshot, prequery_config) -> sealed artifact | audited inventory
produce(prequery_ref, packet, ModelVisibleQueryContext,
        list[ModelVisibleRevision], run_config) -> OntologyProjection
```

Capabilities are enforced outside the condition:

- C0: CPU preconstruction; no query-time creation;
- C1: GPU preconstruction; deterministic sealed-ID projection;
- C2: no preontology; GPU query-time creation and regeneration;
- `A-FixedSelect`: C2 GPU path but sealed C1 IDs and selection-only grammar.

C0/C1 projection output must have complete pre-query lineage. C2 must have a query-reveal time earlier than every nonselection decision. `A-FixedSelect` rejects new IDs, merges/splits, schema/predicates, reification, abstraction, and qualification. A condition-specific grammar difference is permitted only in a hashed capability manifest; scored schema meanings and budgets remain common.

For seed block \(r\), `A-FixedSelect` must consume the sealed C1 preontology created with the same seed block \(r\); cross-seed or best-of artifact selection is prohibited.

One repair is permitted per artifact. The global 16-call fallback/repair/failure/timeout/rerun pool is pre-split into four long slots at 240 seconds, eight standard slots at 150 seconds, and four short slots at 90 seconds. C1 and long case attempts use the long watchdog; C2/feature-ablation and ordinary case attempts use the standard watchdog; `A-FixedSelect` and short structured repairs use the short watchdog. A failed base attempt is still charged to its original class; an additional repair or identical retry consumes the matching reserve slot. Slots cannot be borrowed into a longer timeout class. A condition-blocked execution order and protected per-class slots prevent a late condition from losing all repair access. If a class pool is exhausted, remaining invalid outputs stay invalid unless the separately controlled hard contingency and remaining-time admission check authorize recovery.

## 7. Evidence, temporal validation, and provenance

### 7.1 Neutral evidence and compact C0

spaCy supplies tokenization, NER, dependencies, and sentence boundaries on CPU. Deterministic rules propose normalized names/titles, alias pairs, local pronoun links, event triggers, participant candidates, surface relations, and relative/exact temporal clues. All are provisional in the shared index.

C0 alone turns these candidates into a query-blind ontology using frozen development dictionaries, type/relation patterns, within-window coreference rules, event templates, and partial-order/validity logic. Its query-time scorer combines lexical/context match, time compatibility, support, and typed path continuity. It is credible for common explicit relations but expected to struggle with ambiguous identity, implicit causality, or contextual schema. Those limitations are documented rather than engineered to make it fail.

Primary synthetic and bounded case snapshots use `retrieval.mode=all_admissible`; no embedding model is required. The one complete-novel demonstration uses SQLite FTS5/BM25 with a frozen tokenizer, query, top-k/token cap, and ranks.

### 7.2 Deterministic validation

CPU checks enforce:

- schema/referential integrity and object budgets;
- evidence IDs, offsets, packet membership, horizon, and release class;
- every label clause's assertion/evidence mapping;
- C0/C1 lineage and C2 construction timing;
- interval bound order and partial-order acyclicity;
- story, validity, discourse, revelation, and spoiler separation;
- holder required for belief/report/denial;
- no attributed proposition becoming world truth without a separate assertion;
- no causal relation inferred from temporal precedence alone; and
- complete graph/packet packing.

The conference temporal validator uses explicit interval rules and NetworkX DAG reachability rather than a large general solver dependency. It reports `valid`, `underdetermined`, or a typed contradiction diagnostic without inventing missing order. Repair diagnostics name invalid fields but do not propose new facts.

### 7.3 Provenance

The provenance branches are:

```text
Document -> Passage -> EvidenceSpan -> candidates -> EvidenceSnapshot
  C0 -> deterministic prebuild -> validation
  C1 -> PreconstructionRequest -> raw draft -> validation/repair
  C2/A-FixedSelect -> EvidencePacket -> ConstructionRequest -> raw draft -> validation/repair
-> OntologyProjection -> metric/VisualizationState
-> UserRevision -> per-condition FeedbackResolution
-> ModelVisibleRevision -> next condition output
```

Every link stores parent hashes. `ConstructionSeal` proves C0/C1 completion before query reveal. `PreQueryInventory` proves C2 had no ontology. `ConstructionCertificate` lists post-query decisions and evidence. Public manifests replace restricted paths with opaque IDs.

## 8. Cache, ledger, and recovery

### 8.1 Content-addressed cache

| Namespace | Key essentials | Boundary |
|---|---|---|
| `index` | corpus/window/horizon, segmenter/NLP/rule config | query absent; candidates provisional |
| `c0_pre` | snapshot, C0 rules/vocabulary | query absent; sealed |
| `c1_pre` | snapshot, model/tokenizer/AWQ/runtime/prompt/schema/seed/prebuild budget | query absent; sealed and reused |
| `packet` | snapshot, condition-independent context/revision, retrieval mode/config | identical hash for all conditions |
| `fixed_project` | sealed ontology, packet, context/revision, own resolution, projector/capability config, seed/budget | sealed IDs only |
| `c2_query` | snapshot manifest, packet, full context/shared history, own resolution, model/prompt/schema/seed/budget, optional same-branch parent | no pre-query or cross-context ontology |
| `metric` / `visual` | projection plus metric or renderer config | condition label cannot change formula |

C2 may reference only its exact same-context post-reveal parent. That `ParentProjectionRef` is separate from shared `UserRevision` history. It cannot be the sole source of a claim; packet evidence remains required. Changing any input/config invalidates downstream entries.

### 8.2 SQLite ledger

SQLite tables cover study, job, input, model call, validation, projection, feedback, metric, visualization, resource sample, storage sample, and failure. Model-call rows include prompt/completion tokens, allocated GPU seconds, construction world, served-context count, prebuild/query-time role, and retry class. This supports per-world construction total, amortized per-context cost, first-query and complete three-query workloads, and C1-prebuild-plus-`A-FixedSelect` accounting. Large payloads are compressed blobs referenced by hash. Final rows are append-only. The job state is:

```text
planned -> prequery_sealed -> query_revealed -> generated
-> validated/repaired -> finalized -> scored -> rendered
```

Job identity derives from inputs/config. Reruns never overwrite. Raw invalid output and failure duration are retained. Metric workers stream one projection/blob at a time into incremental SQLite aggregates and compact result tables; they neither load nor copy the full study. A verification command rehashes the DAG, checks missing parents, duplicate payloads, release classes, GPU time, and storage totals.

### 8.3 Safe resume

Each generation writes one bounded temporary blob, verifies it, atomically renames it, then commits the ledger row. An interrupted file is quarantined within the 3 GB temp budget. OOM or timeout attempts count toward actual GPU time and the global retry pool. A retry may lower concurrency or restart the identical stack, but cannot change evidence, prompt, model, token caps, or seed. A semantic change creates a superseding run.

## 9. Synthetic and narrative data implementation

### 9.1 Synthetic generator

Implement four development and 12 test `WorldSpec` records. The test split has four easy 10--12-node, four medium 13--16-node, and four hard 17--20-node gold projections. Each world has three contexts; A/B are a nonselection contrast over identical evidence, while C adds another lens or a fixed-ontology-friendly/null case.

The generator covers temporal change, merge/split, reification, relation/abstraction differences, frequent irrelevant edges, rare pivotal facts, evidence conflict, holder belief/report, causal chains, distractors, and gold communities. Each factor's frozen quota follows the methodology, including a positive rare-pivotal denominator in every world, at least eight rare-ablation contexts, at least eight temporal/epistemic-ablation contexts, and community gold in at least eight contexts from eight worlds. Density is controlled through frequent irrelevant and distractor assertions without expanding the gold projection past 20 nodes.

Separate deterministic stages create world truth, narrative realization, evidence fixtures, structured query, and the typed `GoldContextualProjection`/`GoldAlternativeSet`. The gold format carries anchor-based partitions, local schema, entities/events, qualified assertions, decision signature, relevance, rare/pivotal labels, community labels, signed contrast changes/invariants, alternatives, matcher revision, and review state in a scorer-only namespace. The study LLM never compiles gold. Mutation tests change a source fact or query and assert the exact gold delta.

One test world is selected by recorded random seed within each easy/medium/hard stratum for mandatory independent condition-blind review, covering three worlds and nine of 36 projections. Review must finish and disagreements/alternatives must be frozen before held-out condition execution.

`QuerySampler` materializes the methodology's exact blocked lens, story-scope, abstraction, viewpoint, horizon, and node-budget allocations; it stores the finite eligible choices, conditional probabilities, accepted assignment, rejection reasons, and root seed. The 12-context paraphrase subset contains one context per world and runs C2 once under one registered seed. The primary C1/C2/`A-FixedSelect` run uses two seeds. Root-seed derivation records separate world, narrative, query, paraphrase, LLM, layout, Leiden, and bootstrap seeds.

### 9.2 Frozen ablation configurations

Each ablation is a hashed overlay on its parent C2 configuration. An invariant-manifest test requires identical model, packet, seed, decoding, input/output and object budgets, repair policy, and validator version except for the named switch.

- `A-NoContext` uses one preselected context per world, balanced by difficulty and lens. It replaces model-visible wording, lens, target, story scope, viewpoint, and abstraction with a generic evidence-grounded-construction request. The packet, runner-only horizon enforcement, upper ontology, and neutral budgets remain. The principal diagnostic is ontology-decision macro F1.
- `A-NoTemporalEpistemic` uses eight preselected eligible contexts spanning difficulty and both temporal and holder-status cases. It removes those prompt capabilities and output fields, and its capability validator rejects them; the ordinary full-gold scorer counts required missing qualifications as misses. The principal diagnostic is essential temporal-qualification accuracy.
- `A-NoRareGuard` uses eight preselected rare-pivotal contexts from eight distinct worlds. It removes only the gold-blind instruction and draft checklist requiring inspection of low-frequency evidence for answer necessity, state change, and causal reach. It retains all evidence, normal grounding checks, and output budgets and never exposes scorer labels. The principal diagnostic is rare-pivotal qualified-assertion recall.

Selection manifests are frozen before outputs. A one-switch integration test diffs the fully resolved configurations and fails on any undeclared delta.

### 9.3 First-novel case

Ingest one lawful local copy of *A Game of Thrones*. Store one text file or normalized restricted source, chapter/passages, hashes, and an FTS index. Select four windows before output inspection. Each window has one fixed discourse/spoiler horizon and two contrastive queries sharing the same snapshot; spoiler variation may occur across windows. This preserves exactly four C1 calls, one per window, with one seed.

Run C0/C1/C2 on eight bounded all-admissible packets. Run one additional complete-index FTS query through C2 only as an operational demonstration. Store ranks and omitted evidence, and label it noncausal. Public artifacts contain no prose, detailed offsets, or reconstructive index.

## 10. Metric and analysis modules

`metrics/alignment.py` implements contextual node P/R/F1, strict qualified-assertion P/R/F1, essential temporal accuracy, citation validity, grounding precision, unsupported rate, and ontology-decision macro F1 with visible merge/split, type, relation/schema, event, abstraction, and temporal/epistemic components.

`metrics/contrastive.py` aligns signed nonselection changes on evidence/mention anchors, returns decision-change F1 and collapse rate, and scores the 12 paraphrase pairs. Selection-only change still counts as collapse when gold requires an ontological decision.

`metrics/rare.py` uses scorer-only rarity/pivotality labels, returns qualified-assertion recall and support-path survival, and verifies that changing gold labels cannot change run artifacts.

`metrics/entropy.py` uses one unit-weight entity-event skeleton and implements degree-histogram, degree-mass, local relation-neighborhood, native local-schema relation entropy, and canonical-mapped relation entropy with frozen bins plus `OTHER`. Native and canonical vocabulary sizes and `OTHER` rate are stored. Von Neumann entropy is included only if its development fixture and schedule gate pass before test scoring.

`metrics/clutter.py` records node/edge count, density, isolates/components, label-overlap geometry, irrelevant visible load, rare-pivotal discoverability, crossing opportunity count \(N\), zero-opportunity indicator, and conditional rate \(C/N\) only when \(N>0\). It never imputes 1 or 0 for no opportunity. Invalid/empty outputs receive semantic failure results and `NA` geometry.

`metrics/community.py` runs only Leiden-CPM at one development-frozen resolution/algorithm seed, plus CPU sensitivity at half/double resolution. It returns AMI and purity on a fixed gold-anchor universe, descriptive cluster count/modularity, unweighted mean conductance, fragmentation/merging error, and mention-aligned AMI/variation-of-information stability across the two LLM construction seeds. Purity uses \(N^{-1}\sum_k\max_j|C_k\cap G_j|\) and is never interpreted without fragmentation/cluster count. A small condition-blind `community_review.jsonl.zst` rubric records evidence-based semantic coherence and interpretability for selected examples. There is no Infomap or broad algorithm-seed sweep.

`metrics/feedback.py` computes post-minus-pre strict F1, target-change recall, edit locality, unsupported-change count, and rare-pivotal recall change only for the six known-answer scripts. The three researcher traces record request, resolution, graph diff, latency, and replay/hash success; their gold-dependent fields are `NA`. `capability_limited` rate is reported for both sets. These are descriptive episodes, not participant outcomes.

`analysis.py` implements the fixed world-mean formula, one-sided paired-`t` primary tests, two-sided 95% paired-`t` intervals with 11 degrees of freedom, all 4,096 sign-flip sensitivity assignments, small-sample standardized effects, and a 10,000-resample world bootstrap sensitivity with warnings. Holm covers the two primary C2--C1 endpoints; C2--`A-FixedSelect` is a gated mechanism test. C2--C0 is secondary and entropy two-sided exploratory. Contexts, seeds, nodes, and edges are never treated as independent worlds.

## 11. Minimal graph and feedback implementation

The FastAPI process exposes only:

- load an immutable projection/visual DTO;
- retrieve authorized evidence metadata;
- submit one of two typed revision actions;
- synchronously enqueue/run or poll one local C2 job; and
- load before/after projection diff.

There is no login, remote deployment, multi-user state, SSE, distributed queue, or public service. The plain Cytoscape page shows contextual node role/type, informative assertion labels, time validity, uncertainty/holder status, evidence badges, detail/evidence panels, one story-time or spoiler control, and a revision diff. Renderer-only pan/zoom/focus/hide does not change semantic hashes. One layout configuration/seed is used. At most 12 screenshots are retained.

The two actions are:

- `REFINE_CONTEXT`: change lens or story-time/spoiler scope; and
- `REQUEST_MERGE_SPLIT`: request merge/separation of evidence-anchored mentions/entities.

Six synthetic scripts—three of each action—run one seed and one regeneration. Three researcher-driven interface traces each issue one action and one separately counted C2 regeneration. Shared `UserRevision` and anchors are frozen independently of condition output; each condition gets its own gold-blind resolution. C0/C1 reproject or return `capability_limited`. The UI records request, resolution, call, output, diff, latency, and replay hashes. It demonstrates implementation, not usability.

## 12. Exact GPU call and time budget

### 12.1 Call inventory

Every line is in the call-budget manifest. A call is counted when the GPU request begins, even if it fails. `gpu_session_start` meters allocated loading/warm-up and is included in the same formula.

The inventory exposes rather than hides the timing-treatment asymmetry: C1 makes 24 comprehensive held-out prebuild calls and amortizes each seed/world artifact across three contexts, while C2 makes 72 context-specific calls. Identical per-call caps do not make total opportunities identical. The runner therefore emits prompt/completion tokens and GPU seconds per construction world, amortized served context, first-query workload, full three-query workload, and C1-prebuild-plus-FixedSelect workload. This is reported as a limitation and treatment cost, not interpreted as an efficiency win.

| Class | Count | Provisional admitted p95 seconds | Planned seconds |
|---|---:|---:|---:|
| `gpu_session_start` across pilot and phases | 8 | 180 | 1,440 |
| Acceptance C1-form calls | 2 | 180 | 360 |
| Acceptance C2-form calls | 3 | 120 | 360 |
| Acceptance `A-FixedSelect` calls | 2 | 90 | 180 |
| Acceptance repair probe | 1 | 90 | 90 |
| Development C1 preontologies | 4 | 180 | 720 |
| Development C2 contexts | 12 | 120 | 1,440 |
| Development `A-FixedSelect` | 4 | 90 | 360 |
| Development mandatory-ablation probes | 3 | 120 | 360 |
| Development repair probe | 1 | 90 | 90 |
| Test C1 preontologies: 12 worlds × 2 seeds | 24 | 180 | 4,320 |
| Test C2 primary: 36 contexts × 2 seeds | 72 | 95 | 6,840 |
| Test `A-FixedSelect`: 36 × 2 | 72 | 72 | 5,184 |
| C2 paraphrases: 12 × 1 | 12 | 95 | 1,140 |
| C2 scripted feedback | 6 | 95 | 570 |
| C2 researcher-interface traces | 3 | 95 | 285 |
| `A-NoContext`: 12 × 1 | 12 | 95 | 1,140 |
| `A-NoTemporalEpistemic`: 8 × 1 | 8 | 95 | 760 |
| `A-NoRareGuard`: 8 × 1 | 8 | 95 | 760 |
| Case C1: four windows × one seed | 4 | 240 | 960 |
| Case C2: eight contexts × one seed | 8 | 150 | 1,200 |
| Complete-index C2 demonstration | 1 | 150 | 150 |
| Reserved long fallback/repair/failure/timeout/rerun slots | 4 | 240 | 960 |
| Reserved standard fallback/repair/failure/timeout/rerun slots | 8 | 150 | 1,200 |
| Reserved short fallback/repair/failure/timeout/rerun slots | 4 | 90 | 360 |
| **Inference-attempt total** | **278** |  | **29,789** |
| **Including eight load/allocation events** | **286 accounting events** |  | **31,229** |

The admission-ceiling forecast is:

\[
\texttt{planned\_gpu\_hours}
=\frac{\sum_c N_c\,p95_c}{3600}
=\frac{31{,}229}{3600}
=8.6747\ \text{hours}.
\]

After the pilot, every provisional p95 is replaced by the measured nearest-rank p95 for that class, with a conservative proxy from the longer observed class when sample count is small. The exact manifest calculation must remain at or below 9.00 hours. Faster pilot results do not authorize extra science automatically.

### 12.2 Activity ceilings and hard stop

| Activity | Maximum scheduled allocation |
|---|---:|
| Runtime/model acceptance and development | 1.25 h |
| C1 synthetic preconstruction | 1.25 h |
| C2 primary synthetic construction | 1.95 h |
| `A-FixedSelect` | 1.49 h |
| Paraphrase, feedback/traces, and three limited ablations | 1.35 h |
| Bounded first-novel study | 0.70 h |
| Bounded fallback, repairs, failures, timeouts, reruns | 0.70 h |
| Unallocated scheduled admission slack | 0.31 h |
| **Scheduled envelope** | **9.00 h** |
| Locked unallocated contingency | 1.00 h |
| **Absolute hard maximum** | **10.00 h** |

The eight starts are allocated as three within acceptance/development and one each to C1, C2, `A-FixedSelect`, the combined paraphrase/feedback/ablation block, and the case block; their detailed seconds fit the rounded activity caps above. The runner maintains `actual_allocated_gpu_seconds` monotonically across sessions. Before held-out reveal it refuses the main run when the p95 forecast exceeds 9 hours. Before every call it checks actual used time plus p95 remaining required work. Optional work stops first, but none of the declared 278 calls is silently deleted; a protocol amendment or abort occurs if common pre-freeze reductions cannot admit the mandatory manifest. The one-hour contingency is unlocked only for essential failure recovery after the declared reserve is exhausted, with a recorded incident. It is never used for additional ablations, models, prompts, or samples. The service is shut down before actual time reaches 10 hours.

The 16-call global pool includes the bounded fallback micro-pilot, all validation repairs, and additional attempts after a failed, timed-out, or deliberately rerun base call. A failed base call remains counted in its original row; its successor consumes a reserve row. The 240/150/90-second values are both forecast ceilings and watchdog limits for reserve calls. Unused slots do not become new samples and cannot move into a longer class. If a class pool is exhausted, condition-blind scheduling pauses and records remaining outputs as incomplete unless the controlled contingency is invoked.

## 13. Tests and acceptance gates

### 13.1 Bounded test strategy

Unit tests cover Pydantic constraints, canonical hashes, span/evidence resolution, rights classes, capability enforcement, cache keys, seed derivation, interval/partial-order rules, status separation, each metric formula, GPU/storage accounting, and ledger transitions.

Six small Hypothesis families cover ID/order permutation, empty/singleton/disconnected/multiedge graphs, cyclic/unknown temporal cases, alternative gold, rare-label firewall, and storage/cache deduplication. This is sufficient property coverage; no combinatorial renderer matrix is planned.

Integration tests cover:

1. synthetic ingest through C0/C1/C2/`A-FixedSelect` fixture parsing;
2. C1 `prepare` serialization and access logs containing no query, context, pair, or expected-effect fields;
3. C2 pre-query inventory and request namespaces being unable to resolve C0/C1 finalized IDs, predicates, events, assertions, hidden ontologies, or an unlogged full index;
4. model-visible context stripping world/split/pair/gold/expected-effect metadata;
5. byte-identical packet hashes and equal input/output/object/display/repair budgets across eligible conditions;
6. gold-import denial, C0/C1 seal lineage, and C2 query-time certificate;
7. attributed false belief not becoming world truth;
8. invalid ID/label/temporal output and one repair;
9. two-part zero-crossing-opportunity behavior;
10. native versus canonical relation entropy and cross-seed community stability;
11. fixed gold-anchor community omissions and purity's singleton-inflation fixture;
12. ablation one-switch-only configuration diffs;
13. feedback anchor/resolution isolation, applicable script metrics, trace `NA` policy, and C2 regeneration;
14. interrupted job resume and content deduplication; and
15. public release scan rejecting synthetic copyright canaries/private paths.

A compact regression suite freezes canonical hashes and expected metric rows for six representative gold/condition fixtures, including one C2 failure and one fixed-ontology-friendly case. One local end-to-end UI smoke path covers query/context, graph, evidence, time/spoiler control, typed revision, C2 rebuild, ledger entry, and diff. It uses an existing system browser if available; the project does not download a browser bank. Selected figure geometry receives manual review.

### 13.2 GPU gate

The normal eight-call block tests C1, C2, `A-FixedSelect`, and repair forms on hand-authored fixtures in Phase 1. If it rejects 14B, the finally selected fallback instead must pass the declared one-C1/two-C2/one-FixedSelect micro-pilot and any triggered repair, covering the same gate properties. The remaining 24 development calls occur only after Phase 2 creates the development worlds and Phase 3 implements the condition pathways; the selected model's accepted Phase-1 calls plus those development calls finalize the p50/p95 forecast before held-out launch. The gate requires:

- model/checksum/license and one-cache verification;
- no offload, VRAM below 23 GB, process RAM below 25 GB;
- input/output/complete-graph packing;
- valid constrained generation for all three LLM pathways;
- exact decoding metadata;
- p50/p95 and load-time measurements;
- restart/resume;
- evidence/provenance integrity and zero programmed horizon leaks; and
- recalculated forecast at most 9 hours.

Readiness thresholds guide engineering but do not require C2 to outperform C1. Before reveal, C0 must cover every explicit entity, binary-relation, event-role, story-time, and validity fixture family; reach at least 0.85 precision and 0.70 recall on directly stated development qualified assertions; and have 100% valid evidence references. Query-blind C1 must be schema-valid, exercise every construction operator, have 100% valid evidence IDs, at least 0.95 grounding precision, and at least 0.75 recall against the union of supported development-context gold after its preontology is sealed. C2 operator presence is audited, and programmed horizon leaks must be zero. No gate conditions on C2 beating a baseline.

### 13.3 Pre-case gate

The first-novel phase begins only after synthetic schemas/gold/review, fairness manifests, metrics, feedback replay, storage headroom, GPU forecast, and release scan pass. Test outputs remain intention-to-treat even if invalid. No case call begins if the remaining case allocation plus required repair reserve would exceed the scheduled or hard timer.

## 14. Seven phases and researcher effort

The project uses seven genuine phases rather than renaming the previous nine-phase program.

| Phase | Deliverables | Active researcher hours | Scheduled GPU cap |
|---|---|---:|---:|
| 1. Contracts, compact schemas, and GPU pilot | terminology, records, storage/GPU preflight, pinned model/runtime, hand-authored request fixtures, eight acceptance calls | 20 | 0.38 h |
| 2. Small synthetic benchmark and independent audit | 4 development/12 test worlds, 36 contexts, gold alternatives, 12 paraphrases, mandatory three-world review | 28 | 0 |
| 3. C0, C1, C2, and `A-FixedSelect` | compact C0, common interface, 24 development calls, final freeze, then 24/72/72 primary calls | 36 | 5.57 h |
| 4. Essential metrics and limited ablations | alignment, rare, entropy/clutter, Leiden, statistics, 12/8/8 ablations and 12 paraphrases | 22 | 1.06 h |
| 5. Minimal visualization and scripted feedback | local graph page, two actions, six scripts, three researcher traces | 18 | 0.29 h |
| 6. Bounded first-novel case study | one complete index, four windows/eight queries, one FTS demonstration, safe annotations | 16 | 0.70 h |
| 7. Analysis and conference artifacts | centrally budgeted fallback/repairs/reruns, exact tests/CIs, errors, paper figures, <2 GB public bundle | 20 | 0.70 h |
| Cross-phase admission slack | no new calls; absorbs rounding and measured p95 uncertainty | 0 | 0.30 h |
| **Total** |  | **160 hours** | **9.00 h scheduled** |

The Phase-7 GPU value is a central reserve accounting row: reserve calls may physically occur in an earlier phase, but they are charged there once and never duplicated. Plan three to five months part-time. Unattended CPU time is measured separately and does not inflate active hours; researcher supervision, annotation, debugging, and review do. Scope is reforecast after Phase 1 and before the held-out run. The target range is 120--180 hours; if the 160-hour plan forecasts above 180, optional polish and detailed case annotation are removed before a non-negotiable component.

## 15. Failure strategies

| Risk | Detection | Bounded response |
|---|---|---|
| 14B Phase-1 memory/latency failure | VRAM/p95/packing gate | delete it and symmetrically pilot one pinned 7--8B AWQ model before freeze |
| Invalid structured output | development failure rate | simplify staging/schema without removing semantic obligations; one repair and global pool |
| Full packet does not fit | packing report | shrink/refreeze query-blind world/window for every condition |
| C2 acts as selector | no nonselection certificates/contrast changes | fix prompt on development; do not relabel selection as construction |
| Weak C0/C1 | development error audit | improve compact rules or query-blind prompt within frozen time; retain honest failures |
| Rare-pivotal loss | development fixtures or test result | preserve evidence/method-visible guards; report negative result rather than optimize sparsity |
| Retrieval confound | missing gold evidence | retain all-evidence primary regime; label one FTS demo separately |
| GPU forecast above 9 h | call manifest | stop; before freeze reduce the common evidence/sequence budget and reforecast, or amend/abort without dropping any mandatory call; no post-Phase-1 model switch |
| Storage above 25 GB | preflight | prune caches/temp/duplicate payloads, compress/transfer verified artifacts; keep 5 GB free |
| Copyright scan hit | release preflight | quarantine and rebuild safe bundle; never publish prose/index |
| UI schedule overrun | phase reforecast | keep only graph, detail/evidence, time control, two revisions, diff |
| Null result | primary estimates | retain and publish benchmark, costs, mechanisms, and counterexamples |

Forbidden responses are CPU-only C2, a hidden complete ontology, asymmetric evidence or models, silent truncation, deletion of rare facts, dropping invalid outputs, GPU time outside the authorized scheduled/contingency budget or beyond 10 hours, or removing the first-novel demonstration.

## 16. Definition of done

### Phase 1

Done when the schemas and terminology distinguish upper ontology, local schema, instance graph, qualified assertion, and view; the finally selected model/runtime is verified and frozen; GPU and storage meters work; complete decoding parameters are recorded; and that model passes either the normal eight-call acceptance block or the declared fallback micro-pilot, including all packing/restart/resource gates. Prompts and final p95 values remain development-only until the later 24 calls.

### Phase 2

Done when four development and 12 sealed test worlds contain three contexts each, 10--20-node gold, at least one nonselection contrast per world, required factors, alternatives, rare/community labels, deterministic seeds/mutations, 12 fixed paraphrases, and condition-blind independent review of all nine projections in three selected test worlds.

### Phase 3

Done when C0/C1/C2 and `A-FixedSelect` share interfaces/packets/budgets; C0 and query-blind C1 pass their frozen competence gates; the 24 post-benchmark development calls freeze prompts and the final p95 manifest before reveal; C1 then seals 24 test preontologies and reuses them; C2 creates 72 post-query outputs and records nonselection decisions; FixedSelect creates 72 lineage-valid outputs with all constructive operators rejected; failures remain recorded; and the phase stays within 5.57 GPU hours.

### Phase 4

Done when the two primary endpoints, rare safeguard, contrast/paraphrase metrics, both relation-entropy views, other reduced entropy/clutter metrics, two-part crossing rule, Leiden/AMI/purity/conductance/frag/merge and cross-seed stability, paired-`t` plus exact sensitivity analysis, and three one-switch limited ablations reproduce from the ledger with tested undefined cases.

### Phase 5

Done when one local Cytoscape page shows rich grounded nodes/assertions, evidence and time/spoiler state, and before/after differences; both revision actions serialize; six scripts and three researcher traces resolve per condition and replay; known-answer metrics regenerate for scripts while trace gold fields remain `NA`; C2 reconstructs; C0/C1 capability limits are explicit; and no usability claim is emitted.

### Phase 6

Done when the complete first novel is locally query-blindly indexed without duplication; four same-horizon windows and eight contexts are frozen; C0/C1/C2 run with one seed; one full-index FTS C2 demonstration is separately labeled; annotations and qualitative traces are complete; and public staging contains no protected/reconstructive material.

### Phase 7

Done when the p95 and actual GPU ledgers remain at or below 9 scheduled/10 hard hours; occupied/peak storage remains at or below 25/30 GB; all 12 world-level paired estimates, exact tests, effects and intervals regenerate; secondary and negative results remain; selected figures are traceable; and the public bundle is verified below 2 GB.

## 17. Traceability matrix

| Scientific requirement | Modules/config | Required artifacts | Metrics/tests |
|---|---|---|---|
| RQ1 / `H1-semantic` | `conditions/*`, `metrics/alignment.py`, common budgets | 36-context paired C1/C2 projections, C0 outputs, gold alternatives, packets | primary strict assertion F1 C2--C1 paired-`t`; required secondary C2--C0; node F1, grounding/unsupported, exact sensitivity |
| RQ1 / `H1-organization` | `contracts.py`, C2 decisions, `metrics/contrastive.py` | local schemas, certificates, decision signatures | primary decision macro F1/components C2--C1; required secondary C2--C0; post-query/lineage tests |
| RQ1 / `H1-mechanism` | `fixed_select.py`, capability grammar | 72 paired FixedSelect outputs, complete packing reports, contrast deltas | gated C2--FixedSelect decision F1, change F1/collapse, novel-ID/operator rejection |
| RQ1 robustness | `metrics/contrastive.py`, paraphrase config | 12 fixed paraphrase pairs, same-evidence contrast pairs | signature divergence, strict-F1 change, invariant/change accuracy |
| Rare noncompensable safeguard | `metrics/rare.py`, scorer firewall | positive world denominators, rare/pivotal labels, support paths, `A-NoRareGuard` | one-sided lower bound, qualified rare recall, gold-mutation test, no sparsity-only success |
| RQ2 / `H2-profile` | `metrics/entropy.py`, `clutter.py`, one renderer | named skeleton, native/canonical mappings, frozen geometry | degree/local/native/canonical entropy; opportunity + conditional crossing; visible load; empty graph fails |
| RQ3 / `H3-organization` | `community.py`, Leiden config | fixed gold anchors, partitions, three resolutions, blind review rubric | gold AMI/purity, count/modularity, conductance, frag/merge, cross-seed AMI/VI, omission/singleton fixtures |
| Feature ablations | hashed config overlays, selection manifests | 12 `NoContext`, 8 `NoTemporalEpistemic`, 8 `NoRareGuard` outputs | declared principal diagnostic per ablation, one-switch config test |
| HITL demonstration | `api.py`, revision contracts, `metrics/feedback.py`, C2 | six scripts, three traces, per-condition resolutions/diffs | script-only pre/post F1, target/locality/unsupported/rare; trace operational fields with gold `NA`; capability, latency, replay |
| Narrative transfer | `evidence.py`, FTS, conditions, release scan | four window snapshots, eight outputs/condition, one operational packet | descriptive alignment/rare/error results, no causal claim for full-index query |
| Compute/fairness/storage integrity | `experiment.py`, `store.py` | 278-call manifest with 4/8/4 reserve tiers, eight session events, token/time fields, 30 GB allocation | p95 <=9 h launch gate, actual <10 h stop, C1/C2 amortized-cost report, free >=5 GB, dedup/hash checks |
| Reproducibility/copyright | ledger, manifests, release scanner | compressed raw attempts, safe public bundle, restricted hashes | DAG verification, retry preservation, no prose/private path/model weights |

The traceability table is itself frozen in Phase 1. A missing artifact blocks its associated claim; it does not invite an unplanned substitute.
