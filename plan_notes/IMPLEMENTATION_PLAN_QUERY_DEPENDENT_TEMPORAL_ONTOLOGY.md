# Implementation Plan: Active Query-Dependent Temporal Ontology Construction

**Document status:** executable engineering plan; no software is implemented by this document.  
**Methodological authority:** `METHODOLOGICAL_PLAN_QUERY_DEPENDENT_TEMPORAL_ONTOLOGY.md`.  
**Target researcher:** one academic researcher, with bounded annotation/participant assistance.  
**Target hardware:** one NVIDIA RTX 4090 (24 GB VRAM), approximately 31 GB system RAM, and 8 vCPUs.  
**Planning date:** 2026-09-03.

## Scope, invariants, and correction of the historical implementation direction

Historical plans in this repository describe a deterministic CPU optimizer over a complete oracle graph and explicitly remove or postpone the LLM, GPU, narrative ingestion, user interaction, and interactive presentation. Their good engineering ideas—typed immutable artifacts, provenance checks, ledgers, temporal validation, and retry-safe execution—are reused. Their fixed-oracle architecture is not.

This implementation has a hard boundary:

\[
D_z=I(X_{\le z}),\quad M_z=\operatorname{Manifest}(D_z),\quad
O_{\mathrm{pre},z}=F_{\mathrm{pre}}(D_z),\quad
P_{\mathrm{pre}}=\operatorname{Project}(O_{\mathrm{pre},z},E_q,q,h),\quad
O_{q,h}=F_{\mathrm{query}}(M_z,E_q,q,h),\quad
V=\operatorname{Render}(O).
\]

`D_z` is a provisional `EvidenceSnapshot` at registered discourse/spoiler horizon \(z\), containing evidence with discourse coordinate \(\delta(e)\le z\), and `D` denotes the snapshot family; `M_z` is its immutable metadata/hash/access manifest, not an unrestricted evidence channel; `q` is a structured `QueryContext` with spoiler horizon \(\sigma_q=z\); `h` is an ordered `UserRevision` history. C0 and C1 construct `O_pre,z` before a held-out lens/question and may only project sealed ontology IDs. C2 owns no complete pre-query ontology: after `q` arrives, the RTX 4090-hosted active LLM receives `M_z` and the logged packet contents `E_q`, constructs `O_q`, and reconstructs `O_{q,h}` after semantic revisions. Retrieval, relabeling, summarization, UI filtering, and layout are never aliases for ontology construction.

The minimum publishable implementation includes active GPU inference, C0/C1/C2, a temporal qualified-assertion model, the 10--20-node known-answer synthetic benchmark, seeded contrastive contexts, metrics and ablations, structured human refinement, an interactive rich graph, local narrative ingestion, the bounded Game of Thrones case study, and paper-ready ledgers. None is deferred to an unspecified later project.

## 1. System architecture and data flow

### 1.1 Two-time architecture

```text
                         PRE-QUERY / QUERY-BLIND
local X or public synthetic X
        |
        v
rights check -> segmentation -> mentions/candidates -> temporal clues
        |               -> embeddings/BM25 -> provenance
        +------------------- EvidenceSnapshot family {D_z} ---------+
        |                                                            |
        |            +-----------------------+                       |
        |            |                       |                       |
        v            v                       |                       |
C0 CPU builder   C1 GPU LLM builder          |                       |
        |            |                       |                       |
 sealed O_pre^C0  sealed O_pre^C1            |                       |
        |            |                       |                       |
========+============+=======================+==== QUERY ARRIVES ========
        |            |                       |
        +------ common evidence packet E_q <-+--- QueryContext q
        |            |                       |
 CPU project/rank  CPU project/rank          v
        |          (IDs immutable)      C2 GPU LLM constructs O_q
        |            |                       |
        +------------+-----------------------+
                     v
         schema/evidence/temporal/budget validators (CPU)
                     |
       validated ontology/version + construction/lineage certificate
                     |
          metrics artifacts       visualization DTO
                     |                    |
                     v                    v
              run ledger         Cytoscape.js application
                                          |
                              typed UserRevision h
                                          |
                  C0/C1 reselect only; C2 GPU reconstructs O_{q,h}
```

The required `A-FixedSelect` branch reuses C2's query-time GPU pipeline on sealed C1 IDs with every construction operator disabled; it is omitted from the diagram only for legibility.

Each arrow produces an immutable artifact with a content hash. Pre-query and query-time jobs use separate command groups and cache namespaces. A “query reveal” record closes C0/C1 construction and snapshots the C2 pre-query inventory. The experiment runner refuses to score a run if ordering, lineage, or evidence-access invariants fail.

### 1.2 Components and ownership

| Component | Role | Execution |
|---|---|---|
| Rights-aware ingester | local files, edition manifest, chapters/passages, release class | CPU |
| Evidence preparer | spans, mention/alias/coreference candidates, temporal clues, open relation phrases, uncertainty | CPU |
| Retriever | frozen FTS/BM25 and embedding search; emits identical `EvidencePacket` per query | CPU |
| C0 builder/projector | conventional query-blind ontology plus fixed-ID projection | **CPU only** |
| C1 constructor/projector | query-blind comprehensive ontology, then deterministic fixed-ID query projection | **GPU LLM required** for pre-query construction; CPU for the primary projector |
| C2 constructor/refiner | context-dependent identity/schema/event/assertion construction and feedback regeneration | **GPU LLM required at query time** |
| Structured-output layer | JSON-schema grammar and parsing; no semantic invention | GPU decoding + CPU parse |
| Validators | schema, IDs, evidence support, horizon, temporal satisfiability, lineage, budgets | deterministic CPU |
| Artifact/ledger manager | hashing, caching, manifests, resume, atomic finalization | CPU/storage |
| Metrics/statistics | alignment, rare facts, entropy, clutter, communities, paired analysis | CPU |
| API/UI | versioned projection delivery, feedback, evidence access, temporal interaction | CPU/browser; invokes C2 GPU jobs |

The classical baseline alone is CPU-only. The project as a whole is explicitly not CPU-only.

## 2. Required GPU inference and deterministic CPU validation

### 2.1 Essential GPU calls

The following calls are part of the minimum study and must execute on the RTX 4090 or a documented equivalent:

1. **C1 pre-query construction:** query-blind chunk extraction and consolidation into each replicate's sealed `O_pre`.
2. **`A-FixedSelect` query-time control:** the C2 GPU prompt/tool/evidence/repair pipeline performs selection, compression, and evidence-faithful descriptions constrained to `O_pre` IDs, with create/merge/split/schema/event/abstraction/qualification operators disabled. Primary C1 itself uses the deterministic fixed projector.
3. **C2 initial construction:** after `q` and `E_q` exist, the LLM interprets context, assembles/separates entities, defines contextual types and predicates, chooses event structures and abstraction, adds temporal/epistemic scope, and identifies potentially pivotal assertions using only method-visible evidence signals.
4. **C2 reconstruction after `UserRevision`:** every semantic refinement produces a fresh version from the registered `D_z` manifest/hash, the contents of `E_q`, `q`, and full structured `h`, not unrestricted snapshot access or unlogged chat state.
5. **Required ablations/sensitivity:** exactly `A-FixedSelect`, `A-NoContext`, `A-ShuffledContext`, `A-NoTemporalEpistemic`, `A-NoRareGuard`, `A-NoFeedback`, and the balanced-subset `A-Prompt`; later operator/model diagnostics are secondary.

These calls cannot be replaced by templates, prerecorded gold, CPU heuristics, or a label-only model and still count as the proposed condition. Replayed outputs are allowed only in local tests, never as scientific C1/C2 runs.

### 2.2 CPU responsibilities and limits

Deterministic CPU code may:

- ingest, segment, hash, and classify release permissions;
- propose mentions, aliases, coreference links, temporal expressions, and surface relation candidates without sealing final ontology choices;
- retrieve and deduplicate evidence;
- implement the complete C0 baseline;
- parse JSON, validate Pydantic schemas and referential integrity, normalize harmless syntax, and reject invalid objects;
- check temporal/epistemic constraints, spoiler horizons, provenance, evidence offsets, lineage, and size/token budgets;
- compute graph conversions, matching, entropy, clutter, communities, statistics, hashes, and reports;
- render visualization DTOs and serve the interface; and
- orchestrate/retry/resume jobs.

CPU validation may not invent a missing entity, relation, event, temporal scope, evidence citation, or explanation; choose an ontological abstraction on C2's behalf; or silently repair a substantive contradiction. A validator returns structured diagnostics for at most two counted LLM repair calls. Persistent failure remains a failed run.

## 3. Hardware envelope and resource policy

The reference machine is one NVIDIA GeForce RTX 4090 with 24 GB GDDR6X, approximately 31 GB usable system RAM, and 8 vCPUs. NVIDIA's official specification confirms the 24 GB capacity ([RTX 4090 specifications](https://www.nvidia.com/en-us/geforce/graphics-cards/40-series/rtx-4090/)). The design assumes no second GPU and no large-memory CPU offload.

Operational rules:

- run one vLLM model server and one generation job at a time; parallelize CPU validation/metrics only within an explicit memory cap;
- reserve GPU headroom for KV cache and structured decoding rather than loading the largest possible weight file;
- start with a 20,480-token total sequence cap: at most 16,384 input tokens and at most 4,096 output tokens; size primary packets near 8k evidence tokens and reserve up to about 4k for the complete sealed-graph serialization in `A-FixedSelect`, leaving the remainder for prompts/schema; a blocking per-unit packing preflight, not silent truncation, decides admissibility;
- keep at least 20% (approximately 6.2 GB) of the 31 GB system RAM free during inference, so aggregate measured use stays below approximately 24.8 GB; memory-map immutable model/cache files and stream Parquet rather than holding all books/graphs in memory;
- record peak VRAM/RAM, input/output tokens, queue time, generation time, and OOM diagnostics for every call;
- require at least 300 GB free disk before full experiments to cover the 100--220 GB estimated live set plus temporary, quarantine, and safety headroom; and
- use an uninterruptible or checkpoint-aware run policy; no job assumes a multi-day uninterrupted session.

The official 4090 target is a constraint, not a reason to remove the LLM. If memory pressure occurs, reduce context/batch size or use the approved smaller active model fallback.

## 4. Local CPU development and test workflow

Most engineering work must be possible without occupying the GPU:

1. install the locked base/dev environment with `uv` and use Python 3.12;
2. run schemas, synthetic generation, C0, validators, metrics, RDF round trips, ledgers, API, and UI contract tests on tiny public fixtures;
3. use `FakeLLMBackend` for deterministic hand-authored responses and `ReplayLLMBackend` for immutable, rights-safe recorded outputs;
4. run unit/property tests and a three-world CPU smoke configuration with 10-node graphs, one context pair, and no real generation;
5. validate UI behavior against static DTO fixtures and browser end-to-end tests;
6. enqueue GPU acceptance manifests locally, then execute them only on the reference GPU; and
7. promote an artifact to a scientific result only if its ledger says `backend=vllm_gpu`, contains the pinned model/quantization/hardware manifest, and passes capability/access checks.

Mocks prove orchestration and contracts, not the research hypothesis. Continuous integration may run the CPU suite; a separately labeled manual/nightly GPU suite validates the real C1/C2 path.

## 5. Model and inference-runtime decision

### 5.1 Recommended default

Use **Qwen3-14B-AWQ** as the primary open-weight instruction model, served locally by vLLM with constrained JSON Schema output. Its official model card identifies a 14.8B-parameter Apache-2.0 model and an official four-bit AWQ variant supported by vLLM ([Qwen3-14B](https://huggingface.co/Qwen/Qwen3-14B), [Qwen3-14B-AWQ](https://huggingface.co/Qwen/Qwen3-14B-AWQ)). A 14.8B BF16 checkpoint requires roughly 29.6 GB for weights alone, before KV cache/runtime overhead, so BF16 is not realistic on 24 GB. Four-bit AWQ leaves materially more room for structured decoding and the chosen context window.

The candidate AWQ repository revision observed during planning is `1a6fe1ecf891437a270cce11ad54d796c4f56ce0`; Phase 1 must resolve and verify the full snapshot, store all file hashes, and replace the candidate only through a written model-change decision. Scientific configurations must use a 40-character commit, never `main` or an unpinned alias. The tokenizer and chat template come from the same pinned snapshot.

Use non-thinking mode for the primary structured-output study because it offers bounded latency and avoids mixing variable reasoning-token budgets with C1/C2 timing. Pin and test the exact chat-template control (`enable_thinking=false` or its accepted runtime equivalent) identically for C1/C2. Start the development pilot with the model-card non-thinking sampling recommendation: temperature 0.7, top-p 0.8, top-k 20, three seed-blocked samples. Do not assume greedy decoding is preferred; the official card warns against it for Qwen3. A thinking-mode ablation is optional and must match reasoning/output budgets across C1/C2.

### 5.2 Runtime and structured output

vLLM is selected rather than a custom Transformers generation loop because it provides efficient single-GPU serving, continuous request metadata, and JSON-schema-constrained structured output from Pydantic models ([vLLM structured-output documentation](https://docs.vllm.ai/en/latest/features/structured_outputs/)). The accepted vLLM, PyTorch, Transformers, CUDA toolkit, NVIDIA driver, compiler, and structured-decoding backend are locked together after the hardware pilot; APIs are not assumed stable across versions.

Transformers remains a development/reference dependency for tokenizer checks and one-request diagnostic generation. It is not a second untracked scientific backend. Every run records the runtime and kernel determinism flags.

### 5.3 Fallback ladder that preserves the contribution

1. lower vLLM concurrency/KV cache without changing the scientific packet; if development packing still fails, apply the predeclared query-blind window/packet policy symmetrically, refreeze every affected condition, and score any gold-support loss afterward rather than consulting gold—held-out or condition-specific truncation is forbidden;
2. keep AWQ but reduce maximum context/output length identically for C1/C2 after measuring truncation risk, regenerate fairness manifests, and retain the same final scored ontology budgets;
3. switch to a pinned open-weight 7--8B instruction model in BF16 or approved 4/8-bit quantization, rerunning **both C1 and C2** and labeling the model change;
4. reduce optional diagnostics and then registered mandatory-ablation subset sizes only after a new power/resource analysis, while retaining all primary worlds, three conditions, three primary stochastic seed blocks, contrastive/paraphrase tests, and the variance subset; or
5. use a compatible 4-bit GPTQ checkpoint only if AWQ runtime correctness fails and a structured-output equivalence pilot passes.

A 32B four-bit model is not the default because weight, quantization metadata, KV cache, and runtime overhead leave too little reliable margin on 24 GB. Cloud-only proprietary inference, CPU-only C2, fixed templates, and label-only LLM calls are not acceptable fallbacks for the minimum study.

## 6. Revision and environment pinning

Each scientific run manifest must contain:

- model repository, exact commit, individual weight/config/tokenizer/chat-template hashes, license, and local cache digest;
- quantization method (`AWQ`), bit width, group size/zero-point/kernel settings as read from the snapshot, runtime implementation, and calibration provenance supplied by the publisher;
- tokenizer class/version, vocabulary hash, special tokens, and rendered chat-template hash;
- system/task/repair prompt IDs, semantic versions, content hashes, and fully rendered prompt hash;
- Pydantic/JSON Schema version and hash, upper-ontology vocabulary hash, validator rule-set hash, graph-conversion hash, and metric configuration hash;
- Python lockfile hash; exact vLLM, PyTorch, Transformers, Pydantic, CUDA/driver and OS/container revisions;
- decoding mode, temperature, top-p, top-k, min/max tokens, stop tokens, repetition settings, seed, and structured-output backend;
- hardware model/UUID, visible device, peak resource values, and nondeterminism flags; and
- Git commit plus dirty-tree patch hash. Final paper runs require a tagged clean code revision, although this planning task creates no commit.

Configurations reference immutable prompt/schema/model revisions; changing any invalidates downstream caches. The run ledger stores the rendered inputs and raw outputs so exact artifact reproduction does not depend on a moving external repository. Reproducibility means identical recorded inputs and quantified rerun variance, not a false promise of cross-driver byte identity.

## 7. Coherent minimal technology stack

| Concern | Selected technology | Justification |
|---|---|---|
| Language/environment | Python 3.12, `uv`, `pyproject.toml`/`uv.lock` | one typed application language and reproducible lock |
| Typed contracts | Pydantic 2 + generated JSON Schema | runtime validation and direct vLLM grammar |
| GPU inference | PyTorch + Hugging Face tokenizer + vLLM | pinned local open-weight inference and constrained output |
| Conventional NLP | pinned Stanford CoreNLP server/models for NER, coreference, OpenIE, and SUTime, wrapped from Python | credible, documented CPU C0 primitives; candidate outputs remain defeasible in each `D_z` |
| Evidence retrieval | SQLite FTS5/BM25 plus pinned small sentence-transformer embeddings and FAISS-CPU | transparent lexical baseline plus semantic recall; one frozen result packet shared by conditions |
| Internal graph | Pydantic assertion/event objects; NetworkX conversions for small-graph algorithms | typed, inspectable, sufficient for 10--20-node primary graphs |
| Communities | `python-igraph` + `leidenalg`; pinned Infomap package for sensitivity | reliable Leiden/CPM and distinct-objective robustness |
| Constraints/export | Z3 for temporal satisfiability; RDFLib for RDF 1.1-compatible export | deterministic temporal diagnostics and interoperable output |
| Artifacts | canonical JSONL for objects/events, Parquet for metric tables, SHA-256 manifests | diffable source artifacts plus efficient analysis |
| Ledger/cache | SQLite in WAL mode plus immutable content-addressed files | transactional queue/resume on a solo machine without a server database |
| API | FastAPI + Server-Sent Events for job progress | typed Python boundary and simple one-way progress stream |
| UI | TypeScript, Vite, Cytoscape.js | mature interactive compound/event graphs, styles, filtering, and extensions |
| Tests | pytest, Hypothesis, Playwright, Vitest | unit/property/integration plus real browser behavior |
| Analysis | NumPy, SciPy, pandas/Polars, scikit-learn, statsmodels; R only if a prespecified mixed model requires it | reproducible metrics and hierarchical analysis without duplicative stacks |

CoreNLP and embedding snapshots are pinned exactly like the LLM. A neural embedding retriever run on CPU does not make C0 an LLM baseline; retrieval is shared and query-time ontology construction remains rule-based in C0. If CoreNLP proves operationally unsuitable, the only allowed replacement is a versioned conventional CPU NLP stack applied equally to indexing and C0, evaluated before test lock.

## 8. Package and directory structure

The intended repository layout is:

```text
pyproject.toml
uv.lock
configs/
  dev/  acceptance/  study/  ablations/  visualization/
prompts/
  c1_pre/  fixed_select/  c2_construct/  c2_repair/  shared/
schemas/
  jsonschema/  upper_ontology/  rdf_mappings/
src/story_projection_onto/
  contracts/       documents.py evidence.py ontology.py context.py
                   visualization.py experiments.py
  ingest/          rights.py documents.py segment.py release_scan.py
  index/           mentions.py candidates.py temporal.py embeddings.py
                   retrieval.py manifest.py
  conditions/      base.py classical_pre.py llm_pre.py llm_query.py
                   fixed_select.py capabilities.py
  llm/             backend.py vllm_backend.py fake_backend.py replay_backend.py
                   prompts.py structured.py repair.py accounting.py
  ontology/        construct.py normalize.py matching.py graph_views.py rdf_export.py
  validate/        schema.py evidence.py temporal.py epistemic.py horizon.py
                   lineage.py budgets.py labels.py
  benchmark/       worlds.py narratives.py queries.py paraphrases.py
                   splits.py mutations.py review.py
  scoring/         contracts.py gold_compile.py gold_store.py firewall.py scorer.py
  metrics/         alignment.py contrastive.py rare_pivotal.py entropy.py
                   clutter.py community.py stability.py cost.py
  experiments/     config.py runner.py ledger.py cache.py jobs.py resume.py
                   manifests.py statistics.py reports.py
  api/             app.py routes.py jobs.py evidence.py revisions.py
ui/
  src/             graph/ timeline/ evidence/ revisions/ metrics/ api/
tests/
  unit/ property/ integration/ regression/ e2e/ gpu_acceptance/
data/
  synthetic/       public source worlds, evidence, queries, and gold
  case_study_meta/ public non-text manifests and safe annotations
.local_data/       ignored copyrighted inputs and reconstructive indexes
artifacts/
  public/ restricted/ release_staging/
scripts/           explicit CLI entry points only; no hidden notebook pipeline
notebooks/         exploratory views that read frozen artifacts, never canonical logic
```

The Python package, not notebooks, owns all scientific logic. `data/synthetic` is fully releasable. `.local_data` and `artifacts/restricted` are denied by the release allowlist. The case-study manifest stores opaque source IDs and hashes without protected text.

## 9. Typed domain schemas

All IDs are namespaced UUIDs or content-derived IDs; all records include `schema_version`, `created_by_stage`, and `artifact_hash`. Optional values distinguish `unknown`, `not_applicable`, and `withheld_by_horizon` rather than using an ambiguous null.

| Schema | Required core fields and invariants |
|---|---|
| `EvidenceIndexManifest` | corpus/segmenter/candidate/retrieval revisions, document/passages and content hashes, snapshot IDs, embedding/FTS handles, rights classes, forbidden-field audit; contains no final entity partition, predicates/types, event identity, query relevance/schema, or qualified assertion |
| `EvidenceSnapshot` | `snapshot_id`, parent index, narrative window, discourse/spoiler horizon \(z\), eligible passage/evidence IDs satisfying \(\delta(e)\le z\), monotonic-parent hash, created-before-query seal; no ontology objects or proposition-level revelation annotations |
| `DocumentRecord` | `document_id`, corpus/edition ID, ordered chapter IDs, language, local URI (restricted), content hash, license/rights class, ingestion time; text never appears in public form |
| `PassageRecord` | `passage_id`, document/chapter/order, local text reference, start/end offsets, text hash, discourse position, release class |
| `EvidenceSpan` | `evidence_id`, passage ID, sentence/token/character offsets, local quote or restricted pointer, hash, discourse coordinate \(\delta(e)\), retrieval eligibility, provenance and extraction confidence; offsets must resolve inside the passage and the span itself carries no proposition-level revelation timestamp |
| `Mention` | `mention_id`, evidence ID, offsets, surface hash/text by release class, provisional mention type, features, confidence; it is not a final entity |
| `AliasCandidate` / `CoreferenceCandidate` | ordered mention IDs, surface relation, score/source, uncertainty and rationale code; explicitly defeasible, no global entity ID |
| `EventMentionCandidate` | one or more evidence/mention IDs, surface trigger, candidate participant mentions/time clues, source/score/uncertainty; no canonical event ID, role semantics, or reified event assertion |
| `RelationPhraseCandidate` | evidence and endpoint mention candidates, surface phrase/direction, extractor source/score; no finalized predicate, truth status, domain/range, or query relevance |
| `TemporalClue` | evidence ID, normalized expression/partial order, target candidate IDs, story/discourse hypothesis, precision and confidence; not a finalized assertion scope |
| `EvidencePacket` | packet ID, `D_z` hash, `q`/horizon hash, ordered evidence IDs, retrieval method/scores/ranks, token count, rejected horizon items, seed; identical packet hash is supplied across conditions |
| `Entity` | projection-local ID, canonical name, supported aliases/mention IDs, contextual type(s), role summary, abstraction level, temporal-state references, uncertainty, evidence availability; merge/split provenance required |
| `Event` | projection-local ID, contextual type/description, story interval/order, evidence IDs, confidence/relevance, and reification decision. Participant-role, causal, consequence, and belief/report/denial semantics are qualified `Assertion`/role records with their own scope/provenance, not unqualified event properties |
| `RevelationAnnotation` | proposition-content/signature ID, first licensed discourse coordinate \(\tau_r\) or partial-order constraint, evidence IDs, derivation (`generator`, `annotator`, or rule), rationale, precision/confidence; belongs to a proposition, never to a raw evidence span |
| `TemporalScope` | story/event time, validity interval, Allen/partial-order constraints, discourse position/window, optional `RevelationAnnotation` reference, precision and `unknown` reason; axes cannot be placed in one generic timestamp |
| `PropositionContent` | stable content ID/hash plus normalized subject/predicate/object or n-ary role proposition and local-schema references; denotes what is said/believed without asserting that it is true |
| `EpistemicScope` | holder entity/source, attitude (`known`, `believed`, `reported`, `rumored`, `denied`, `uncertain`), embedded `PropositionContent` ID, holder-relative validity and evidence; it does not globally endorse the embedded proposition, and the active spoiler horizon remains context metadata |
| `Provenance` | evidence IDs and hashes, index/retrieval/run IDs, constructor/model/prompt/config IDs, derivation/repair parents, curation/ingestion timestamps, actor/release class |
| `Assertion` | assertion ID and `PropositionContent` reference; readable description/direction; temporal and optional epistemic scope; narrative commitment (`world_asserted`, `attributed_only`, `contested`, `hypothetical`, `unknown`) separate from pipeline `validation_status` (`accepted`, `rejected`, `invalid`) and temporal/epistemic diagnostic/precision state; confidence; evidence/provenance; contextual relevance and supported `why_matters`. An epistemically scoped assertion never entails a separate world assertion of its embedded content |
| `LocalSchemaElement` | contextual type or predicate ID, label, formal definition, arity/domain/range/role constraints, abstraction level, parent upper-vocabulary ID, evidence/examples; cannot define away provenance requirements |
| `QueryContext` | natural-language wording, lens, target, story-time scope, discourse window, spoiler horizon `spoiler_horizon_sigma` \(\sigma_q\), epistemic viewpoint, abstraction, node/assertion/display budgets, requested evidence behavior, semantic version/hash; contains no split, contrast, paraphrase-group, condition, expected-delta, or gold IDs |
| `ExperimentUnitEnvelope` | runner/scorer-only unit ID, query-context hash, `base_context_id`, `surface_wording_id`, `paraphrase_group_id`, split/world/condition/seed blocks, paired-unit handles, and scorer artifact references; never serialized into an LLM request |
| `ModelVisibleQueryContext` | explicit prompt DTO containing only the wording and semantic fields copied from `QueryContext`; excludes all storage hashes, unit/group/split/condition IDs, gold handles, and expected changes |
| `ConstructorRunView` | execution-only model/prompt/runtime revisions, seed, budgets, permitted capabilities, and evidence/snapshot handles needed by the condition; excludes analysis models, ablation expected-effect metadata, gold/scorer paths, pair/group IDs, and held-out split labels |
| `OntologyDecision` | operator type (`merge`, `split`, `type`, `predicate`, `reify`, `abstract`, `qualify`, `include/exclude`), inputs/outputs, context reason code, evidence IDs, construction time |
| `OntologyProjection` | projection ID, condition, `D_z`/packet/query/revision hashes, local schema, entities/events/assertions, decisions, omissions/abstentions, validation status, construction/lineage certificate, budgets and run provenance |
| `FeedbackAnchor` | condition-independent admissible evidence/mention IDs, normalized semantic signature, optional role/story-time constraints and anchor-schema version; contains no gold, condition ID, projection-local ID, or resolved target |
| `UserRevision` | condition-independent revision ID and parent revision, action, one or more `FeedbackAnchor` records, typed requested change/constraint, rationale/evidence, actor, sequence/timestamp, prior/result context hashes; contains no condition resolution, actual before/after projection value, or model output and is append-only and branchable |
| `FeedbackResolution` | runner-only revision ID, receiving condition, optional clicked source-local ID for UI provenance, that condition's resolved target IDs, resolver status (`resolved`, `ambiguous`, `absent`), separate capability response (`applied`, `capability_limited`, `not_applicable`), before/after projection IDs, resolver/model/prompt/config/seed and diagnostics; never serialized as an all-condition map |
| `ModelVisibleRevision` | allowlisted projection of one `UserRevision`: action, anchors, requested change, rationale/evidence and sequence; may add only the receiving condition's own resolved target IDs and resolver status when needed, and never exposes a condition label, any other condition's ID/output/status, a capability result, or a scorer field |
| `VisualizationNode` | projection object ID, short supported label, role/type/temporal-state fields, confidence/evidence indicators, zoom tiers, style tokens, semantic hash |
| `VisualizationAssertion` | assertion ID, source/target or event ports, supported label, time/epistemic/confidence/provenance/relevance badges, bundling membership, semantic hash |
| `VisualizationState` | projection hash, renderer/style/layout/font/viewport/seed, visible/filter/bundle states, node coordinates, crossings/overlaps, horizon; semantic and renderer hashes separate |
| `EvidencePanel` | assertion/node ID, authorized evidence metadata and local excerpt handles, support status, temporal/provenance explanation, release rules; horizon guard required |

Gold extensions add `GoldAlternativeSet`, contextual relevance labels, rare/common and pivotal/non-pivotal labels, community partitions or soft co-assignment, and explicit invariant/change sets between contrastive contexts. Validation schemas are published independently of private text.

CoreNLP clusters are decomposed into scored, defeasible mention-pair candidates before entering `D_z`; OpenIE triples become surface `RelationPhraseCandidate` records and are never accepted assertions. A boundary classifier rejects any evidence artifact with canonical/global entity IDs, finalized predicates/types, reified event identity, query-specific relevance/schema, or qualified truth assertions. Gold schemas and storage are isolated under `scoring/`; production condition modules have a static import deny-rule and filesystem allowlist that prevent access.

## 10. Structured LLM input and output contracts

### 10.1 Construction request

`ConstructionRequest` is the only scientific input to a constructor. It contains:

- condition and permitted capability set;
- immutable corpus/index/snapshot hashes and the ordered `EvidencePacket` records with IDs; on case data the LLM receives the packet, never unrestricted `D_z` contents;
- the explicit `ModelVisibleQueryContext` DTO and ordered `ModelVisibleRevision` history derived from condition-independent `UserRevision` records for C2, absent during C1 preconstruction; a model-visible revision may contain only the receiving condition's own frozen resolution when necessary, never the cross-condition resolution table;
- the small shared upper ontology and allowed primitive temporal/epistemic terms, not a hidden gold/final schema;
- node, event, assertion, evidence, input/output token, and display budgets;
- requirements for citations, abstention, uncertainty, horizon, and consideration of method-visible low-frequency/state-change/causal-necessity signals; no scorer-only rare/pivotal label;
- prior ontology only for a declared C2 refinement version or C1 fixed selection, with capability rules explicit; and
- the allowlisted `ConstructorRunView` of model/prompt/schema revisions, budgets, capabilities, and seed.

C1 preconstruction requests contain no held-out query fields or query-derived retrieval order; primary C1 projection is deterministic and has no selection prompt. `A-FixedSelect` requests contain the model-visible semantic `q`, complete `E_q`, and the complete required serialization of the sealed ontology, but the JSON grammar permits only selections and supported display descriptions. C2 requests expose candidate mentions/coreference and temporal clues as alternatives, never as settled entity/event/relation decisions. The prompt serializer has a field allowlist and snapshot test proving that base-context, paraphrase-group, split, pair, condition-comparison, expected-delta/effect, gold, and analysis IDs are absent. A separate access test proves that the constructor's run view cannot reach the full experiment/ablation configuration.

### 10.2 Ontology draft

`OntologyDraft` returns:

1. a concise structured interpretation of lens, horizon, viewpoint, and abstraction;
2. locally defined contextual types and predicates with normalized definitions;
3. entity assemblies with evidence mention IDs, aliases, contextual roles, and explicit merge/split decisions;
4. event records with identity, boundaries, occurrence time, evidence, and reification decisions, plus references to separate qualified participant-role, causal, and consequence assertion IDs; those semantics cannot bypass assertion-level time, confidence, evidence, and provenance;
5. `PropositionContent` records and qualified assertions with predicate/direction, narrative commitment separate from validation status, all applicable temporal and epistemic axes, confidence, relevance, evidence IDs, and supported `why_matters` text; an attributed belief/report never silently asserts its embedded proposition as world fact;
6. contextual node labels/role summaries and edge descriptions, each linked to supporting assertion/evidence IDs;
7. explicit omissions, unresolved conflicts, alternative interpretations, and abstentions;
8. construction decisions and parent lineage needed for operational timing tests; and
9. budget accounting.

The contract requests concise auditable decision reasons, not private chain-of-thought. No free-form prose outside schema fields is trusted. Grammar-constrained JSON prevents syntax classes of error; Pydantic and semantic validators still decide whether referenced objects exist and claims are admissible.

### 10.3 Repair contract

`RepairRequest` contains the original immutable request hash, raw draft hash, machine-readable diagnostics, remaining repair budget, and an instruction to change only invalid fields. `RepairResponse` repeats the full draft plus a field-level change log. At most two repairs are allowed and all tokens/time count toward the condition. CPU code may sort arrays, canonicalize Unicode, or calculate hashes, but any semantic substitution returns to the LLM. A third failure finalizes `invalid`, preserving all attempts.

## 11. Evidence grounding and hallucination controls

Grounding is enforced at five layers, none of which delegates final scientific scoring to the evaluated model:

1. **Access layer:** build a query-blind `EvidenceSnapshot D_z` for each registered discourse/spoiler horizon \(z\), admitting only evidence with \(\delta(e)\le z\). All conditions use the identical snapshot hash; later passages are physically absent from the request and local evidence API. C0/C1 preconstruct separately from each \(D_z\) before the lens/question is revealed, so future evidence cannot influence their entity or schema decisions.
2. **Reference layer:** every semantic object and factual UI string cites existing evidence IDs. Offsets, passage hashes, release class, horizon eligibility, and retrieved-packet membership are validated.
3. **Structural-support layer:** assertion endpoints/events must connect to cited mentions; merge/split decisions list mention evidence; times and epistemic holders cite relevant clues; labels and `why_matters` fields cite validated assertions rather than arbitrary prose.
4. **Semantic-support layer:** after finalization, the isolated scorer evaluates synthetic support with generator entailment rules; these rules and results cannot create repair diagnostics or flow back into a condition. Case-study assertions receive deterministic pre-gold structural diagnostics and blind manual scoring on the registered sample. A separate entailment model or LLM judge may be exploratory only and cannot be the sole ground-truth arbiter.
5. **Counterfactual layer:** entity-name permutations, fabricated evidence IDs, swapped temporal clues, and beyond-horizon canaries verify that the system follows supplied evidence rather than pretrained recollection.

Gold lives in a scorer-only artifact namespace and process role. Constructors, retrievers, projectors, prompt renderers, validators that generate repair diagnostics, UI generation routes, and condition caches cannot import or read `GoldProjectionBundle`, relevance gold, or rare/pivotal labels. CI mutates those labels and requires all pre-score request/output hashes to remain identical. Any method-visible “pivotality” safeguard is computed from supplied evidence and query signals (proposed causal reach, state change, answer necessity, and uncertainty), never copied from gold.

For the primary timing experiment, `D_z` means the complete admissible small synthetic world or a preregistered bounded case-study window/horizon, capped before query revelation to fit the common input envelope; `retrieval.mode=all_admissible`, so `E_q` contains every record in that scored snapshot for all conditions. The full books still receive neutral indexing to locate and provenance-link windows. A separately labeled retrieval-realism analysis may query the full horizon index with a common bounded `E_q`; because C0/C1 can have broader preconstruction exposure in that regime, it is not used for the central construction-timing causal claim and the asymmetry is reported. `A-FullEvidence` diagnoses retrieval loss there.

Three orthogonal status axes are stored, never collapsed: pipeline `validation_status = accepted | rejected | invalid`; narrative `commitment = world_asserted | attributed_only | contested | hypothetical | unknown`; and `evidence_support_status = sufficient | insufficient | disputed | unknown`. Temporal/epistemic precision and diagnostics are separate again, including `determinate | underdetermined | unsatisfiable | not_applicable`; an explicitly uncertain but structurally legal assertion can therefore be `validation_status=accepted` and `temporal_diagnostic=underdetermined`. Thus a structurally accepted assertion may faithfully represent a contested or false character belief, while an unsupported world claim still fails support policy. Uncertainty is preferable to fabrication. Citation validity, grounding precision, unsupported-assertion rate, abstention, label/summary violations, and horizon leaks are stored separately. A citation existing in the packet is necessary but not sufficient for support.

Generated UI text follows a claim inventory: each factual clause maps to one or more assertion IDs with `validation_status=accepted` and to evidence IDs. Here “accepted” means pipeline validity, not that attributed/contested content is true; the UI displays narrative commitment and support status separately. A deterministic clause splitter plus reviewed fixtures detects unmapped clauses; the release UI suppresses ungrounded text rather than showing it. Suppression is logged and does not retroactively improve the ontology's scientific score.

## 12. Temporal and epistemic constraint validation

The internal representation is the same hybrid qualified-assertion/event model selected in the methodology. `validate/temporal.py` translates exact/interval/ordinal clues into difference and interval constraints; Z3 checks satisfiability and returns a minimal diagnostic set where practical. Supported relations include before/after, meets, overlaps, during/contains, starts/finishes, equal, open/closed bounds, and explicitly unknown bounds.

Validators enforce:

- event occurrence time and assertion validity are distinct fields;
- discourse order is derived from passage order, not substituted for story time;
- proposition-level revelation \(\tau_r\) and the separate query spoiler horizon \(\sigma_q\) restrict accessible assertions/evidence but are not character belief;
- character/source `known`, `believed`, `reported`, `rumored`, `denied`, and `uncertain` states have a holder and temporal scope when used;
- epistemically scoped content is represented as the holder's attitude toward a `PropositionContent`; it is not globally true unless a separate `world_asserted` assertion exists, and narrative commitment/contestation is never inferred from pipeline validation status;
- a state cannot have an inverted interval, and mutually exclusive states cannot overlap unless explicitly contested;
- causal edges cannot be inferred from `before` alone and must cite causal evidence or be marked a hypothesis;
- relative time constraints do not become false exact dates;
- projection filtering cannot leave an assertion pointing to a withheld event/entity; and
- ingestion/curation timestamps remain provenance and never order fictional events.

For a total proposition-revelation order, admissibility is \(\tau_r(a)\le\sigma_q\). For a partial order, an assertion is admissible only when the validator can prove its proposition-level revelation is not later than the context cutoff; incomparable/unknown revelation is withheld in the primary strict policy and included only in a labeled permissive sensitivity. Evidence membership is independently checked as \(\delta(e)\le z=\sigma_q\). A display-only time scrub leaves semantic/context hashes unchanged, while changing query story time or epistemic viewpoint creates a `UserRevision` and reconstruction/reprojection.

The validator can prove inconsistency or incompleteness under the encoded rules; it cannot establish the truth of a contested interpretation. Diagnostics distinguish `schema_error`, `unsatisfiable`, `underdetermined`, and `horizon_violation`. An `underdetermined` scope with uncertainty can be valid. Property tests compare known Allen-relation compositions and story/discourse/revelation counterexamples.

RDF export uses RDF 1.1-compatible resources for assertions/events, n-ary role nodes, OWL-Time terms, and PROV-O links. A versioned project vocabulary supplies discourse/proposition-revelation positions, epistemic attitudes, narrative commitment, contextual relevance/`why_matters`, confidence, and construction decisions that those standards do not fully define. An epistemically embedded `PropositionContent` is exported as the object of a qualified attitude assertion and its base triple is **not** emitted as asserted; only a distinct `world_asserted` record can license that triple/assertion resource. Named graphs may group source/run provenance. An experimental mapping targets the pinned W3C “RDF 1.2 Concepts and Abstract Data Model,” Candidate Recommendation Snapshot of 7 April 2026; RDF 1.1 remains the portable baseline. Round-trip tests require preservation of all internal IDs, time axes, epistemic scopes, narrative commitment, confidence, relevance, and evidence; export never becomes the authoritative store.

## 13. End-to-end provenance

The provenance chain is:

```text
DocumentRecord -> PassageRecord -> EvidenceSpan -> Mention/TemporalClue
 -> EvidenceIndex/Snapshot -> EvidencePacket -> ConstructionRequest
 -> raw OntologyDraft -> repairs/validation -> OntologyProjection
 -> graph conversion / metric result -> VisualizationState
 -> condition-independent UserRevision -> per-condition FeedbackResolution
 -> receiving-condition ModelVisibleRevision -> next ConstructionRequest
 -> next OntologyProjection
```

Every link stores parent hashes and the responsible stage. Pre-query `ConstructionSeal` records snapshot hash, constructor, end time, and ontology hash for C0/C1. C2's `PreQueryInventory` proves that only index, retrieval, prompt, model, and validation artifacts existed before `QueryReveal`; its `ConstructionCertificate` records the post-reveal start/end, all `OntologyDecision` records, evidence packet, revision history, and output hash.

For C0/C1, `LineageCertificate` requires every output entity, event, schema element, and assertion to resolve to a sealed pre-query ID. Contextual descriptions may quote or combine existing supported fields but may not state a new proposition. For C2, every locally created atom resolves to evidence and a post-query decision rather than another condition's ontology. Provenance validation is condition-blind and blocking.

Restricted case-study provenance contains local path handles in an encrypted or permission-limited manifest; public derivatives contain opaque IDs and hashes. The public release scanner fails on protected text, embeddings, non-allowlisted extensions, private paths, secrets, and excerpts over the policy threshold.

## 14. Caching without invalidating construction timing

All caches are content-addressed, read-only after finalization, and namespaced by stage and condition.

| Cache | Key includes | Critical exclusion/invariant |
|---|---|---|
| `index` | corpus/edition hash, horizon snapshot, segmenter/candidate/embedder revisions, config | query/lens absent; candidates remain provisional |
| `retrieval` | `D_z` hash, full structured `q`, condition-independent `UserRevision` request-history hash/feedback stage, page cursor/requested direction, retrieval revision, horizon, top-k/token cap, seed | one packet reused across conditions; no condition-specific resolution or retrieval |
| `c0_pre` | `D_z`, C0 code/rules/upper-vocabulary/config | query and feedback absent; immutable after seal |
| `c1_pre` | `D_z`, exact model/tokenizer snapshot, AWQ config, rendered chat template/non-thinking control, vLLM/structured-decoder/kernel/runtime, preprompt/schema/validator, decoding seed, comprehensive prebuild budget | query and feedback absent; immutable after seal |
| `fixed_project` | sealed ontology, packet, `q`, condition-independent `UserRevision` history hash, receiving-condition-only `FeedbackResolution` hash and parent projection/version where used, projector or `A-FixedSelect` model/tokenizer/AWQ/runtime/chat-template/prompt/schema, capability-policy hash, input/output/repair budgets, seed | output IDs must be a subset/faithful view of sealed IDs; feedback stages cannot collide and no other-condition resolution enters the key |
| `c2_query` | `D_z` manifest and packet, complete `q`, condition-independent `h`, receiving-C2-only resolution hash when used, optional declared same-branch `ParentProjectionRef`, exact model/tokenizer/AWQ/runtime/chat-template/structured-decoder/kernel, prompt/schema/validator revisions, budgets, decoding seed | no other-condition resolution/ID, prequery or cross-context ontology; same-branch parent cannot be the sole semantic source |
| `metrics` | projection/gold/graph-conversion/metric config and code revisions | condition label cannot change formula |
| `visual` | projection hash, renderer/style/layout/font/viewport/seed | no semantic mutation; hidden and absent remain distinct |

Any change to evidence, horizon, prompt, schema, model, quantization, decoder, upper ontology, validator, user history, receiving-condition resolution, capability policy, parent projection, or budget invalidates downstream entries. A semantically equivalent paraphrase intentionally has a distinct raw `q` cache key so stability is observed rather than hidden by reuse. C2 refinements may reference only the exact same-context, post-reveal parent through a typed `ParentProjectionRef`; its hash appears separately in the receiving-condition request/cache key and runner-only resolution/lineage, never in shared `h`, and the response remains a complete projection reconsidered against the full packet/history. There is no global `world -> canonical ontology` cache in the C2 namespace.

A cache audit rejects any prequery/cross-context `OntologyProjection`, recognizes only the narrow same-branch parent exception, rejects every gold object, checks pre-/post-reveal timestamps and access logs, and verifies that parent-only claims cannot survive without evidence. Cache hits are recorded in cost results. Deleting a failed cache entry is prohibited; supersession creates a new manifest.

## 15. Comparable implementations of C0, C1, and C2

### 15.1 Shared interface and capability enforcement

Each condition implements:

```text
prepare(snapshot: EvidenceSnapshot, prequery_config) -> PreQueryArtifact | None
produce(prequery_ref: SealedOntologyRef | PreQueryInventoryRef,
        packet: EvidencePacket, context: ModelVisibleQueryContext,
        history: list[ModelVisibleRevision],
        run_config: ConstructorRunView) -> OntologyProjection
```

`ConditionCapabilities` declares `uses_llm_prequery`, `uses_llm_query`, `may_create_new_entity_after_reveal`, `may_merge_split_after_reveal`, `may_define_schema_after_reveal`, `may_reify_event_after_reveal`, `may_add_qualification_after_reveal`, and `may_reconstruct_after_revision`. The runner and validator, not the condition itself, enforce the declaration. C0/C1 can serialize existing entities but cannot create them after reveal. The runner derives `ModelVisibleQueryContext` and `ConstructorRunView` through explicit field allowlists; the condition cannot inspect `ExperimentUnitEnvelope`, full ablation/analysis configuration, pair/group IDs, or expected effects. Identical public interfaces, evidence packets, scored-output budgets, projection schemas, and metric paths prevent condition-specific evaluation code.

### 15.2 C0 `ClassicalPre`

Before queries, C0 processes each registered `D_z` on CPU. It consumes the exact stored, pinned CoreNLP NER/pairwise-coreference/OpenIE-surface/SUTime candidates shared in that snapshot. Any additional C0-only pass is condition-owned ontology construction, is logged, and receives no evidence unavailable to C1/C2. Deterministic rules and development-only dictionaries build entities, events, n-ary roles, normalized upper-vocabulary predicates, validity/occurrence orders, attributed speech/belief records, and assertion-level provenance. Global alias/coreference thresholds and relation mappings are tuned on development worlds, documented, and frozen.

At query time, a deterministic relevance scorer combines lexical/embedding query similarity, temporal/horizon eligibility, typed-path coverage, and evidence support. A budgeted connected-selection routine balances relevance, answer-path continuity, and rare-fact candidates; it renders only sealed ontology IDs. Labels use precomputed names/descriptions and evidence snippets, not generated facts. This is a credible KG baseline, not frequency-only pruning. It remains CPU-only and cannot reconstruct ontology semantics.

### 15.3 C1 `LLMPre`

Before query reveal, C1 uses the primary GPU model to construct a comprehensive ontology for each `D_z` and stochastic replicate. Synthetic worlds fit in one structured request. Case-study windows use query-blind passage chunks followed by hierarchical consolidation; chunking, token totals, duplicate resolution, and every intermediate graph are recorded. Consolidation sees no lens, query wording, query-specific retrieval ranking, or test gold.

The preregistered primary C1 query path uses the same deterministic projector as C0 over the sealed LLM ontology, avoiding an unaccounted second semantic construction call. It may rank, prune, and render rich descriptions already present in `O_pre`, but all output atoms retain lineage. `A-FixedSelect` is the strengthened query-time control: it runs the same model, evidence packet, structured attention pipeline, and identical maximum input/output/repair caps as C2, while its grammar and validator restrict outputs to selecting and support-preservingly describing C1 IDs. Realized tokens are reported. C2 must be compared with both primary C1 and `A-FixedSelect` so neither general LLM use nor superior query-time selection explains the result.

### 15.4 C2 `LLMQuery`

C2's `prepare` returns only the audited neutral snapshot inventory. `produce` executes after `QueryReveal` and calls the GPU model with the `D_z` hash/manifest, the contents of frozen `E_q`, `q`, and `h`; it does not expose unrestricted case-study snapshot contents. The output can create projection-local entities/events/schema/assertions and is required to record decisions for relevance, merge/split, contextual typing, abstraction, predicate definition, event reification, temporal/epistemic qualification, method-visible potentially pivotal material, and contextual presentation.

On feedback, C2 reconstructs under the new context/history and outputs a semantic diff from its parent; it may preserve supported objects but cannot rely on hidden chat state. GPU LLM inference is indispensable. CPU validation can reject or request repair, not replace the active construction.

### 15.5 Fairness assertions

For every scored unit, a `FairnessManifest` verifies identical `D_z`, `E_q`, horizon, retrieval top-k/evidence-token cap, upper vocabulary, **final scored** node/event/assertion/display caps, maximum input/output tokens, repair allowance, feedback-step budget, validator/metric versions, common projection-schema field semantics, and prompt/evidence token accounting. C1 and C2 share the exact model/tokenizer/AWQ/runtime/chat-template control and three seed blocks. C0/C1 have separately declared comprehensive prebuild caps so a credible canonical ontology is not forced into C2's final projection size. Condition-specific prompt/grammar hashes are not falsely required to be identical: an allowlisted `GrammarDifferenceManifest` must show that differences implement only timing/capability restrictions, including `A-FixedSelect`'s intentionally stricter ID-subset grammar, while the scored projection schema remains comparable. Any unregistered field-semantic, evidence, or budget difference blocks scoring.

`A-FixedSelect` uses deterministic sealed-graph/evidence packing and identical maximum input/output/repair caps to C2, with realized use reported. Before any query-time model call for a scored unit, a blocking `PackingReport` verifies `prompt/schema + complete E_q + complete required sealed-ontology serialization <= max_input`; neither evidence nor graph may be truncated. Primary bounded windows are frozen against this stricter envelope using worst-case development serialization. A development failure causes a common smaller window or symmetric input-cap amendment after the 4090 pilot; a held-out failure is an integrity failure. Different prequery versus query-time prompts and unavoidable comprehensive/per-context costs are explicit; reports give prebuild, query, total, and amortized cost at 1/4/16/observed queries plus a labeled total-token-matched sensitivity.

### 15.6 Frozen ablation registry

| ID | Scope | Executable change and artifact |
|---|---:|---|
| `A-FixedSelect` | all 96 canonical contexts × 3 seeds | C2 query-time pipeline over sealed C1 IDs; all construction operators disabled; `fixed_select` projection/capability report |
| `A-NoContext` | balanced 48 contexts × 3 | remove lens/structured context but retain generic question; ablation projection |
| `A-ShuffledContext` | paired balanced 48 × 3 | substitute another world's compatible-looking context without changing evidence; mapping manifest |
| `A-NoTemporalEpistemic` | factor-eligible 48 × 3 | prohibit temporal/epistemic fields; ablation projection |
| `A-NoRareGuard` | rare-factor 48 × 3 | remove only method-visible state-change/causal/answer-necessity preservation signals; gold remains firewalled |
| `A-NoFeedback` | one context/world × 3 | withhold or sham the same registered revision at matched reconstruction budget; compare with supplied-revision branch |
| `A-Prompt` | balanced 24 contexts × 3, both C1/C2 | one predeclared independently worded prompt template with identical contract/budgets; prompt-sensitivity artifact |
| `A-WeakGrounding`, `A-FixedSchema`, `A-NoMergeSplit`, `A-NoEventReify`, `A-FullEvidence` | 24 eligible contexts × 3 each as resources permit | diagnostic operator/retrieval artifacts |
| `A-ModelSize` | optional balanced 24 × 3, both C1/C2 | pinned smaller active model; model-sensitivity artifact |

The first six and `A-Prompt` are mandatory. `configs/ablations/registry.yaml` freezes the selected context IDs, seeds, evidence/output/repair budgets, expected capability change (scorer-side), and analysis before execution; only its allowlisted `ConstructorRunView` reaches a condition. `A-FixedSelect` alone is confirmatory mechanistic: C2--`A-FixedSelect` uses a separate Holm family for ontology-decision macro F1 and contrastive decision-change F1 superiority, with -0.05 strict-F1 and rare-recall non-inferiority safeguards. It is excluded from the exploratory-ablation BH pool; failure forbids attributing a C2--C1 gain specifically to construction freedom. A five-replicate variance subset uses 24 balanced contexts across 12 worlds: seed blocks 1--3 are primary and blocks 4--5 add 48 C2 outputs plus corresponding C1 snapshot/projection replicates. C1 variance and `A-Prompt` preontologies are each built once per unique `(world, D_z, seed, prompt)` and reused across matching contexts, preserving the shared-preartifact dependency. Every ablation has a config-schema test, capability test, ledger tag, and traceable analysis row.

## 16. Synthetic world and narrative generator

### 16.1 Fixed scale and factors

Implement eight development and 24 sealed test worlds. Test difficulty tiers have eight worlds each and gold contextual projections of 10--12, 13--16, and 17--20 entity/event nodes; underlying semantic worlds remain in the 10--20-node initial regime while mentions/assertions can be larger. Every world has four base contexts, two semantically equivalent phrasings per context (one canonical and one paraphrase), and at least one gold nonselection contrastive pair.

The initial execution inventory, before repairs/feedback, is fixed as follows:

| Artifact or output | Planned count |
|---|---:|
| Structured test contexts | 24 × 4 = 96 |
| Canonical C2 initial outputs | 96 × 3 paired seed blocks = 288 |
| Paraphrase C2 outputs | stratified 48 × 3 = 144 |
| C1 preontologies | at most 24 worlds × 3 horizon snapshots × 3 seeds = 216; shared across same-snapshot contexts |
| C1 fixed projections | 288 canonical + 144 paraphrase |
| C0 projections | 96 canonical + 48 paraphrase, each counted once |
| `A-FixedSelect` outputs | 96 canonical × 3 = 288 |
| Layout/community execution | one frozen primary seed plus five-seed sensitivity per valid projection |

Scripted feedback uses one registered context per world, three seed blocks, a supplied-versus-sham revision, and one to three frozen steps; the manifest computes the exact extra call count before execution. C0 is never copied to create artificial replicate observations.

`WorldSpec` is representation-neutral and contains actors/personas, collectives, places, items, events, states, reports/beliefs, causal dependencies, partial temporal order, disclosure order, aliases, and evidence witnesses. Six generator families cover allegiance/betrayal, kinship/identity, causal incident/consequence, travel/location, conflict/participation, and rumor/revelation. At least two structural templates and one surface narrative renderer are completely held out from prompt/rule tuning.

Factor combinators introduce temporal changes, n-ary participation, context-dependent merge/split, aliases/coreference, relation ambiguity, distractors, frequent irrelevant facts, rare pivotal facts, incomplete/competing evidence, character knowledge, proposition revelation order, causal chains, and alternative abstraction. Difficulty increases interacting factors, ambiguity, chain length, and distractor ratio—not only size.

A pre-generation coverage constraint requires at least: 24 contexts with merge/split, 48 with temporal change, 24 with epistemic conflict, 48 with rare-pivotal facts, 32 with frequent-irrelevant material, 24 with incomplete/competing evidence or justified abstention, 24 with context-dependent abstraction, 24 with causal chains, and 48 with meaningful nontrivial hard gold communities. There are at least 48 rare-pivotal qualified assertions overall, and every factor occurs in every difficulty tier. Rejection sampling must meet the frozen realized distribution; it cannot search seeds after observing condition output.

### 16.2 Generation pipeline and anti-circularity

```text
root seed -> WorldSpec -> validated world states/partial orders
          -> independent narrative renderer(s) -> passages/evidence
          -> structured QueryContexts -> GoldCompiler
          -> gold projections / alternatives / contrast deltas / communities
```

Gold is compiled from formal world/query semantics, never from the study LLM. Semantic gold IDs are stripped from visible text. The generator includes fixed-ontology-friendly null cases, legitimate shared ontologies across contexts, unnecessary-reification traps, frequency traps, false-causality traps, and abstention cases. Mutation tests flip facts, horizons, identities, or contexts and assert the exact expected gold delta.

All templates and a stratified output sample receive manual review; test gold receives a delayed condition-blind second pass, and at least 20% is offered for independent review if feasible. Review disagreements and permissible alternatives are data. World/template/renderer splits, manifests, code, and seeds are public.

## 17. Seeded random and contrastive query generator

`QueryGenerator` samples typed records, then `ParaphraseRenderer` realizes language. The frozen held-out distribution is balanced 10% each over political allegiance, family/kinship, causal responsibility, knowledge/belief, event participation, loyalty change, geographic movement, conflict, pivotal-event consequences, and time/spoiler horizon. Conditional tables implement the methodology defaults:

- temporal scope: point/interval/before-or-after/unbounded at 0.25 each;
- spoiler horizon: early/middle/late-full at 0.35/0.35/0.30;
- abstraction: event/episode/role/group at 0.35/0.25/0.25/0.15; and
- epistemic viewpoint: reader-admissible/omniscient-evidence/named-character/not-applicable conditional on lens.

Invalid combinations use logged rejection sampling with integer counts stratified across difficulty tiers so the realized 96-context table approximates the 10% targets transparently. Each model-visible context stores a target node budget sampled from the difficulty stratum independently of the subsequently compiled gold size. Expected construction operators, contrast deltas/proofs, target schema, and all relevance/rare labels live only in scorer-side `GoldProjectionBundle`. `ContrastSet` contains same-world/evidence context IDs, gold invariant decisions, signed required additions/removals/substitutions/merges/splits/reifications/qualifications, and a proof that at least one nonselection operator differs. `ParaphraseSet` holds the structured context fixed and varies only wording. Serialization tests prove that `ConstructionRequest` cannot contain any scorer field.

A root seed deterministically derives world, narrative, query, paraphrase, retrieval, LLM, layout, community, and bootstrap seeds through a documented hash-based derivation. Python, NumPy, PyTorch, FAISS, igraph/Leiden, Infomap, and browser-layout seeds are recorded separately. Rejection counts and library versions are part of the artifact.

## 18. Gold contextual-projection format

`GoldProjectionBundle` contains:

- `WorldSpec` and visible narrative/evidence manifest;
- canonical mentions, allowable mention partitions, entities, events, and temporal/epistemic assertions;
- `QueryContext` and its target local schema, abstraction, and projection;
- gold relevant/irrelevant nodes/events/assertions and evidence links;
- `RarePivotalLabel` with independently computed evidence frequency and pivotal rationale;
- gold communities where meaningful plus permitted partitions or soft pairwise co-assignment;
- contrastive invariant/change sets and paraphrase equivalence groups;
- `GoldAlternativeSet` with graph-level alternatives and local equivalence rules; and
- review status, generator/version/seeds, validation report, and release class.

The scorer aligns a prediction to every allowed alternative using a frozen Hungarian/constraint matcher and chooses the best admissible score without inspecting condition name. Relation synonyms, time tolerances, and abstraction equivalences are explicit tables, not post hoc judgment. The format supports exact and partial gold; case-study incomplete gold marks unannotated fields rather than counting them negative.

## 19. Metric implementations

All metric functions accept immutable prediction, gold, graph-conversion/matcher configuration, and return `MetricResult(value, numerator, denominator, direction, status, diagnostics, artifact_hash)`. Macro aggregation over world-context units is primary.

`metrics/alignment.py` implements:

- node and strict/relaxed qualified-assertion precision/recall/F1;
- relation-type macro F1/agreement and contextual type/schema/abstraction agreement;
- separate story/event-time, validity, discourse-order, revelation-order, and epistemic holder/status accuracy; bounded interval IoU, point tolerance, and allowed-set Allen/partial-order F1 cover unknown/open bounds;
- B-cubed and mention-pair entity merge/split scores;
- event identity, participant-role, causal-link, and reification-decision scores;
- normalized typed graph-edit distance with frozen edit costs, divided by delete-all-predicted plus insert-all-gold cost and capped at 1;
- evidence-ID validity (valid/all cited IDs), evidence coverage (recovered gold-relevant assertions having at least one valid cited supporting span/all gold-relevant assertions), reviewed grounding precision (supported/all adjudicated predicted facts), unsupported factual-clause rate (unsupported/all adjudicated predicted clauses), spoiler-leak rate, and confidence calibration;
- rare-pivotal qualified-assertion/node/evidence recall and complete support-chain survival; and
- irrelevant-information rate with an explicit predicted-component denominator.

`OntologyDecisionMacroF1` is the unweighted macro average of F1 for the frozen decision families `merge_split`, `contextual_type`, `schema_relation`, `event_reification`, `abstraction`, and `temporal_epistemic_qualification`. Its exact per-family rule is: gold empty/prediction empty -> `NA`; gold nonempty/prediction empty -> 0; gold empty/prediction nonempty -> 0; otherwise calculate F1. Component scores and applicability counts remain visible.

`metrics/contrastive.py` aligns paired signatures on gold mention/span identity and converts them to typed signed operations (`add`, `remove`, `substitute`, `merge`, `split`, `reify/de-reify`, `qualify`). It computes decision-change macro F1 over the union of gold-active and prediction-active nonselection families with the four-case empty-family rule, plus invariant-preservation F1. Primary ontological collapse occurs whenever the gold nonselection delta is nonempty but the predicted nonselection delta is empty—even if visible nodes changed; near-identical total-signature collapse at distance ≤0.01 is a secondary diagnostic. Invariant F1 has a -0.05 C2--C1 non-inferiority safeguard. The module also emits world-resampled distance calibration. It separately tests paraphrase strict-F1 non-inferiority at 0.05 and requires the one-sided upper 95% confidence bound for mean normalized signature divergence to be below 0.10. Arbitrary divergence earns no credit.

`metrics/rare_pivotal.py` reads scorer-only rarity from evidence/generator annotations—not output frequency—and pivotality from gold answer/causal/state/community effects. Gold compilation freezes an at-least-48-context `rare_eligible` manifest before runs. Per-context recall is computed only where the gold rare-pivotal denominator is positive; other contexts return `NA` plus denominator 0. Eligible invalid/missing outputs score 0, scores are macro-averaged within world, and inference resamples worlds. The module emits the complete rare/common × pivotal/non-pivotal table and budget survival curves. The run-side rare guard receives none of these labels. Rare-pivotal recall is a noncompensable gate.

Case-study expert preferences are stored per rubric dimension and analyzed separately from automatic metrics. No LLM self-judge supplies a confirmatory semantic or coherence score.

## 20. Entropy and clutter implementations

`ontology/graph_views.py` produces named, hashed conversions. The primary topology is an unweighted undirected entity-event incidence skeleton; parallel roles/assertions collapse for simple topology, self-loops are excluded but remain in assertion metrics, isolates remain, and horizon filtering occurs first. Unknown-time assertions enter a slice when they may overlap it; a strict-known-time sensitivity excludes them. Unit-count/relevance-weighted skeletons, the directed qualified-assertion multigraph, and entity-only projection are separately named. Every entropy result records the view.

`metrics/entropy.py` implements the full preregistered, two-sided panel:

- degree-histogram entropy from \(p_k=|\{v:d(v)=k\}|/|V|\), normalized by \(\log n\) when defined;
- unweighted degree-mass entropy from \(d_v/(2m)\), normalized by \(\log n\), plus separately named weighted strength-mass sensitivity \(s_v/\sum_u s_u\);
- local incident-relation entropy where \(p_{v,r}=n_{v,r}/\sum_r n_{v,r}\), reporting mean, degree-weighted mean, median, and isolate fraction;
- relation-type entropy over qualified assertion records, using a condition-blind frozen mapper into upper-ontology/development-schema bins plus one `OTHER` bin for any unmatched held-out C2 predicate, with normalization against that complete fixed bin count;
- edge-relevance entropy with identically calibrated nonnegative weights, normalization by \(\log m\), and `NA` for zero total;
- von Neumann entropy using the primary unit-weight undirected entity-event skeleton, \(L=D-A\), \(\rho=L/\operatorname{tr}(L)\), raw and normalized by \(\log(n-1)\) where defined; the evaluator-relevance-weighted skeleton is a separately named sensitivity; and
- community-size membership entropy normalized by \(\log c\), labeled only as balance.

Use natural logs, define \(0\log0=0\), and always return raw entropy, support size, and effective diversity \(e^H\). Clip numerical eigenvalues only inside a frozen tolerance and return `NA` for zero-denominator/edgeless cases. Empty, complete regular, singleton, disconnected, directed, multiedge, unknown-predicate, and zero-weight fixtures have known regression expectations. Every entropy contrast is two-sided and belongs to the required secondary exploratory BH family; every table includes size, density, connectivity, fidelity, relevance, and rare-pivotal recall.

`metrics/clutter.py` separates semantic and renderer inputs. Semantic measures include node/assertion count, named density, isolates/components, maximum/mean degree, degree Gini/centralization, endpoint coverage/unreachable-pair rate, and temporally/type-admissible unweighted path ambiguity for reachable registered endpoint pairs with maximum length four. A disconnected answer pair cannot receive a favorable zero-ambiguity value.

Renderer metrics use headless Chromium screenshots and Cytoscape geometry under pinned viewport, device-pixel ratio, font, stylesheet, zoom, layout algorithm/version, label policy, and seed. They include:

- raw crossings and crossing rate = unordered nonadjacent edge pairs crossing at least once divided by eligible unordered **nonadjacent** edge pairs;
- label-label and label-node intersection count/area;
- node-node, node-edge, and label-edge occlusion;
- congestion/bundle load; and
- visible information components (nodes, assertion strokes, labels, words, glyphs, open panels) plus a development-frozen weighted load index.

`IrrelevantVisibleLoad` is the frozen weighted number of visible marks whose semantic source is gold-irrelevant, divided by total visible-mark weight; simple component rates are also shown so weights cannot hide a result. The frozen unbundled overview with one stroke per rendered assertion edge is primary. In the bundled sensitivity, a bundle stroke's weight is allocated among underlying assertions in proportion to their frozen unbundled weights, and its irrelevant share is the sum allocated to gold-irrelevant assertions; label clauses are attributed to their assertion IDs, with mixed or unmapped clauses counted as irrelevant.

Before runs, the scorer freezes `crossing_eligible` contexts whose gold unbundled overview has at least one eligible nonadjacent edge pair. On this fixed subset, predictions with a positive denominator use the rate above; a valid prediction with zero eligible pairs retains raw `NA` but receives primary `failure_adjusted_crossing_rate=1` and `zero_opportunity=true`. The module also returns every numerator/denominator, the zero-opportunity rate, and an exposure-weighted pooled sensitivity. Contexts outside the fixed subset are descriptive, never selected by condition output. One layout seed/configuration is confirmatory and five frozen seeds form the sensitivity block. Hidden, filtered, and semantically absent objects have distinct statuses. Evidence discoverability tasks count interactions required to reveal relevant and rare-pivotal support. Invalid or empty scientific outputs contribute semantic zeros and failure rates but have `NA` renderer metrics; they cannot win by zero clutter. H2a is tested only after strict-F1 and rare-recall C2--C1 lower 95% bounds exceed -0.05 and grounding precision is at least 0.95.

## 21. Community detection and cluster evaluation

`metrics/community.py` uses the primary unit-weight entity-event skeleton on the frozen set of at least 48 contexts with reviewed nontrivial hard disjoint gold communities. The scorer freezes a query-relevant gold vertex universe `U_g` keyed by mention/evidence anchors. It maps each anchor to a predicted vertex/community or to its own unique `ABSENT:<anchor>` label; it never drops an unmatched gold vertex. Extra predicted vertices are reported through node precision/irrelevance and a union-universe sensitivity. On that basis it:

1. runs Leiden with the Constant Potts Model at a development-frozen resolution and one primary seed, recording library/objective/weights/seed;
2. reports community count/size, modularity under the configuration-model null with \(\gamma=1\), internal density, per-community conductance with standard volume counting internal weight twice/boundary once, and its unweighted mean (lower favorable); per-community separation counts internal and boundary weights once, and graph separation is the unweighted mean over nonzero-volume communities (higher favorable), with zero-volume exclusions/rate recorded;
3. compares meaningful gold partitions using AMI as primary, arithmetic-normalized NMI secondary, purity plus inverse purity, and explicit pairwise errors: fragmentation = gold-same pairs predicted different divided by all gold-same pairs, merging = gold-different pairs predicted same divided by all gold-different pairs; a zero denominator is `NA` with the pair count reported, while over-/under-cluster counts remain descriptive;
4. obtains semantic coherence from synthetic gold categories or a frozen human rubric, never the same LLM's opinion;
5. repeats community seeds on each fixed ontology for algorithmic stability; and
6. compares predictions to gold with AMI/NMI/purity/pairwise errors on the same completed `U_g`; compares ontologies across LLM samples by first reporting mention-aligned vertex coverage/Jaccard, then AMI/variation of information on common aligned entities as a sensitivity; and runs a union-universe sensitivity with explicit `EXTRA` status.

A five-seed algorithmic block, preregistered resolution sweep, and Infomap provide sensitivity without extra confirmatory choices. Missing-anchor, extra-vertex, singleton, zero-volume, disconnected, all-one-cluster, all-singleton, and partially annotated partitions have explicit fixtures/flags and pinned library behavior. Enumerated hard alternatives use best admissible matching; soft co-assignment gold uses pairwise Brier loss and remains secondary. At 10--20 nodes, exact partition displays accompany scores. Raw community count is descriptive and community membership entropy is not treated as quality.

## 22. Experiment runner and configuration system

Pydantic settings load layered, immutable YAML: base -> dataset -> condition -> ablation -> run. Unknown keys are fatal; environment variables may point to secrets/private data but are resolved to redacted manifests. A fully rendered configuration is hashed before execution.

Required configuration groups are:

```text
dataset.{corpus_id,snapshot_id,split,world_ids,horizon,release_class}
query.{distribution_version,context_ids,paraphrase_mode,root_seed}
retrieval.{backend,index_revision,top_k,evidence_token_cap,seed}
condition.{id,capabilities,prequery_seal,projector}
model.{repo,revision,tokenizer_revision,quantization,runtime,chat_template}
decoding.{mode,temperature,top_p,top_k,max_input,max_output,seed_block}
prompt.{construct_id,select_id,repair_id,schema_revision}
budget.{nodes,events,assertions,visible_marks,repairs,feedback_steps}
validation.{ruleset,evidence_policy,temporal_policy,lineage_policy}
metrics.{matcher,graph_views,entropy_panel,community,clutter,rare_gate}
visualization.{renderer,layout,viewport,font,style,seed_block}
analysis.{estimands,mixed_models,bootstrap,multiplicity,missing_policy}
```

The runner state machine is `planned -> prequery_prepared -> sealed -> query_revealed -> generating -> validating/repairing -> finalized -> metric_complete -> rendered -> analyzed`. Transitions are transactional. Job IDs derive from config/input hashes; reruns are idempotent and never overwrite an artifact. A query cannot be revealed before the required C0/C1 seals and C2 inventory snapshot exist.

Study manifests enumerate paired condition runs before execution. The scheduler interleaves conditions/replicates to reduce temperature/driver-time confounds while respecting that C1 pregraphs precede queries. It records invalid outputs, repair attempts, timeouts, OOMs, cancellations, and cache hits as outcomes.

`analysis.{estimands,mixed_models,bootstrap,multiplicity,missing_policy}` is an executable contract, not report metadata. Core RQ1--RQ3 jobs admit only the 96 canonical-wording, initial-history projections; paraphrases route only to H1c safeguards and later feedback stages only to HITL estimands. The frozen analysis job first averages each stochastic condition over its three registered seed blocks within world/context, keeps deterministic C0 as one observation, and produces C2--C1 before C2--C0 paired contrasts. It resamples and permutes at the world level; emits raw differences, 95% intervals, standardized paired effects, and scale-appropriate odds/rate/rank-biserial effects; and fits the methodology's world/base-context/wording/shared-preartifact mixed-effects sensitivities with the prescribed fallback order. Holm is applied within each RQ's confirmatory family and separately to the two-endpoint C2--`A-FixedSelect` mechanistic family. Benjamini--Hochberg applies only to labeled secondary exploratory entropy and exploratory ablation families and explicitly excludes `A-FixedSelect`. The schema freezes node/rare/strict-fidelity/invariant margins at 0.05, grounding at 0.03 plus its 0.95 absolute floor, H1b component margins at 0.10, and paraphrase divergence's one-sided upper 95% bound at 0.10; it also freezes two-sided entropy tests, the joint clutter/fidelity/rare gate, rare/crossing-eligible manifests, and intention-to-treat invalid-output rules before query reveal. A dry run on synthetic fixtures must reproduce every planned table column and flag any unregistered endpoint, direction, exclusion, or multiplicity family.

## 23. Run ledger and artifact manifest

SQLite tables store `study`, `job`, `input_artifact`, `model_call`, `validation`, `repair`, `projection`, `feedback`, `metric`, `visualization`, `failure`, and `resource_sample`. Large JSONL/Parquet/screenshots live in content-addressed paths referenced by hash. The ledger uses foreign keys, WAL, checksums, and append-only triggers for finalized rows.

Every finalized job records:

- code commit/dirty patch hash, lockfile/container/OS/driver/CUDA versions and hardware UUID;
- corpus/edition/rights class, `D_z` and `E_q` hashes, eligible/rejected evidence IDs, and query reveal time;
- condition/capabilities, C0 rules or model/tokenizer/AWQ/runtime revisions, prompt/schema/config/upper-ontology hashes;
- all root/derived seeds and determinism flags;
- exact rendered LLM inputs, raw outputs, token accounting, timings, finish reason, VRAM/RAM, validator/repair traces;
- sealed preontology or post-query construction/lineage certificate, final ontology and semantic-diff hashes;
- condition-independent user revisions, per-condition feedback resolutions, model-visible-revision hashes, and every intermediate projection;
- matcher/graph conversion/metric/community/layout versions and outputs; and
- public/restricted release class, retry/supersession links, failure code, and human review status.

`ArtifactManifest` lists path, media type, byte size, SHA-256, schema version, parents, generator stage, release class, and expected retention. A verification command rehashes the entire DAG, checks ledger/file agreement, proves no final result depends on an unmanifested file, and creates a redacted public manifest. Experiment reports are generated from ledger queries rather than copied notebook values.

## 24. Interactive graph application design

The application is a local-first FastAPI service with a TypeScript/Cytoscape.js client. FastAPI serves immutable ontology versions, renderer DTOs, permitted evidence, experiment/session metadata, and typed revision endpoints. Long C2 reconstructions enter the same durable GPU queue as batch experiments; Server-Sent Events report state without holding an HTTP worker. The browser never calls the LLM directly.

Application state has three deliberately separate layers:

1. **semantic state:** an immutable validated `OntologyProjection` and parent/diff history;
2. **context state:** the `QueryContext` plus append-only `UserRevision` history that can cause reconstruction; and
3. **renderer state:** positions, pan/zoom, temporary visibility, selection, opened panels, bundles, and styles that cannot change semantics.

Routes that change context create a revision and a new job. Routes that pan, zoom, highlight, or bundle update only local renderer state. Changing the spoiler horizon creates a new evidence-snapshot/packet selection from a preregistered horizon and triggers C2 reconstruction; C0/C1 can only reproject their separately sealed ontology for that horizon. The interface states this limitation rather than suggesting equal semantic capabilities.

The app supports side-by-side or animated projection diffs, version history, branch from revision, undo/redo as new events, and export of a sanitized session manifest. Stable coordinates are seeded from shared surviving semantic IDs, but the layout may add/remove/reposition objects when C2 legitimately restructures the ontology.

Accessibility is part of acceptance: keyboard navigation, focus order, non-color status encodings, sufficient contrast, text alternatives for graph selections, reduced-motion mode, and a tabular assertion view. Spoiler warnings precede local evidence access.

## 25. Rich node, assertion, evidence, and temporal UI

### 25.1 Overview and progressive disclosure

The default overview shows only canonical name, short contextual role, primary current type, concise supported relation labels, temporal-change markers, uncertainty, and evidence badges. Four frozen semantic-zoom tiers reveal, in order:

1. community hulls, focal nodes/events, and bundled relationship families;
2. named nodes/events, essential edges, and method-visible “potentially pivotal” markers derived from evidence/context signals—not scorer gold;
3. aliases, contextual types, event participant roles, validity intervals, confidence, and relation direction; and
4. full assertion metadata, provenance, evidence locations, alternative interpretations, and revision lineage.

Focus-plus-context dims nonfocus elements while preserving discoverability. Edge bundling groups only renderer strokes; clicking a bundle lists every underlying assertion. No assertion is deleted to improve the drawing. Method-visible potentially pivotal markers remain discoverable at the overview unless the active horizon legitimately withholds their evidence; scorer-only rare-pivotal labels are used only for evaluation and never enter the application request.

### 25.2 Node and assertion detail

The node panel exposes canonical name, aliases, supporting mention IDs, contextual role, projection-local type, abstraction, temporal state history, relation-sensitive summary, uncertainty/contested flags, evidence count/availability, merge/split lineage, and revision diff. It distinguishes a contextual entity from provisional index mentions.

The assertion panel exposes the readable evidence-supported description, normalized predicate and definition, direction, source/target or event-role map, event/causal role, story/event time, validity, discourse/revelation positions, applicable character epistemic state, the separately active query spoiler horizon, narrative commitment, evidence-support status, confidence/calibration status, contextual relevance, `why_matters`, evidence locations, and construction/validation provenance. Each factual clause resolves to assertion/evidence IDs with `validation_status=accepted`; the visible commitment/status makes clear that validation acceptance is not narrative truth.

### 25.3 Time, perspective, evidence, and horizon

A time rail can show story/event intervals and validity bars; a parallel discourse/revelation rail prevents them from being conflated. The time slider changes the active temporal slice. The spoiler-horizon control switches among registered evidence snapshots and cannot expose future aliases, labels, summaries, node sizes, grouping, tooltips, or evidence counts. A viewpoint selector distinguishes reader-admissible propositions from a named character's knowledge/belief/report state.

The evidence drawer shows chapter/passage metadata and, only in a local authorized session, a minimal supporting excerpt. It explains whether evidence directly supports, reports, contradicts, or merely orders the assertion. Public mode shows safe locators/hashes and synthetic text only. “Request more evidence” follows the deterministic expansion policy in Section 26 rather than unconstrained model search.

### 25.4 Semantic versus renderer evaluation

The UI can load a fixed ontology into multiple renderer configurations to evaluate semantic zoom, focus, bundling, and temporal presentation without changing graph truth. Conversely, condition comparisons use one locked renderer/style/viewport/font/seed. Headless snapshots and geometry logs drive clutter metrics; task accuracy/time and human ratings are separate outcomes. The interface makes no usability claim merely because crossings decrease.

## 26. Human-feedback event model

`UserRevision.action` is one of the typed actions in the table below. All object-targeting payloads use one or more condition-independent `FeedbackAnchor` records. `UserRevision` is the shared request and contains no projection-local or resolved ID. The API may place a clicked projection-local ID only in the source condition's runner-only `FeedbackResolution` provenance. The experiment runner resolves the anchor independently per condition with a frozen gold-blind resolver; it never forwards one condition's local ID, resolution, output, or capability result to another condition.

| Action | Typed payload | Semantic behavior |
|---|---|---|
| `REFINE_QUERY` | revised text/lens/target fields | C2 reconstructs; C0/C1 reproject sealed IDs |
| `SET_STORY_TIME` | story interval/partial-order target | keep the same evidence snapshot, update `q`; C2 reconstructs and C0/C1 reproject |
| `SET_DISCOURSE_WINDOW` | narration-position window | update `q` within current admissible snapshot; semantic change reconstructs C2 |
| `SET_SPOILER_HORIZON` | registered discourse/spoiler cutoff \(\sigma_q\) | switch all conditions to matching sealed `D_z/O_pre,z`; C2 constructs for it |
| `SET_EPISTEMIC_VIEWPOINT` | reader-admissible, omniscient-evidence, or named holder | update `q`; C2 reconstructs holder-specific semantics and fixed conditions reproject |
| `EXPAND` / `COMPRESS` | target node/assertion/detail budgets | bounded reconstruction versus fixed projection |
| `MARK_RELEVANCE` | anchor plus relevant/irrelevant/uncertain | adds preference constraint; never changes evidence truth |
| `CORRECT_ASSERTION` | anchor, corrected structured value, evidence IDs, dispute flag | C2 repairs/reconstructs; fixed conditions may select an existing correction or log resolution/capability failure |
| `MERGE_CONCEPTS` / `SPLIT_CONCEPT` | anchors/mention partitions and optional rationale | C2 identity reconstruction; forbidden capability is explicit in C0/C1 |
| `REQUEST_EVIDENCE` | anchor, current packet, requested direction/count | deterministic admissible packet expansion, then equal condition access |
| `SET_ABSTRACTION` | event/episode/role/group level and budget | C2 changes ontology organization; fixed conditions reselect only |
| `REIFY_EVENT` / `CLARIFY_RELATION` | anchors and desired semantic distinction | advanced C2 construction; fixed conditions return logged resolution/capability response |

Every shared revision contains the `UserRevision` fields specified in Section 9, and every condition receives the same anchor-level request. The resolver emits one condition-specific `FeedbackResolution` with `resolved`, `ambiguous`, or `absent` before capability handling; the receiving constructor can see at most its own allowlisted resolution. A prohibited operation is not skipped: C0/C1 record a separate `capability_limited` response with their best legal projection, which is scored. The complete cross-condition resolution table remains runner/scorer-only.

Scripted cross-condition runs freeze anchors and requested changes before any condition output, using only admissible evidence and the registered feedback script. A session-specific clicked ID is provenance in the source condition's `FeedbackResolution`, not shared request content. An ad hoc human revision observed under one condition is not treated as matched feedback for another condition unless a condition-blind normalization step can reproduce the same frozen anchor and requested change; otherwise it remains qualitative/session-specific evidence.

Evidence expansion cannot privilege C2. `REQUEST_EVIDENCE` applies a frozen next-page rule to the shared retriever, creates `EvidencePacket_{k+1}`, and exposes it to all conditions. It cannot cross the active `D_z` horizon. This is analyzed as a separate feedback stage with matched retrieval calls; the initial benchmark never repeatedly retrieves until correct.

Scripted synthetic policies define a nonempty set of gold-required target changes \(T\), an invariant region \(U\), and the aligned predicted semantic diff \(\widehat{\Delta}\). Report:

- recovery gain = post-revision minus pre-revision strict F1;
- target-change recall = \(|\widehat{\Delta}\cap T|/|T|\);
- revision efficiency = successfully corrected gold decision/assertion count per semantic revision, and token efficiency = that count per 1,000 generated LLM tokens; token efficiency is `NA` for zero-token CPU paths;
- edit locality = \(|\widehat{\Delta}\cap T|/|\widehat{\Delta}|\); if the predicted diff is empty while \(T\) is nonempty, both locality and target-change recall are 0, while the excluded both-empty diagnostic case is `NA`;
- nonlocal regression rate = formerly correct items in \(U\) made incorrect divided by formerly correct items in \(U\); return `NA` and denominator 0 when there was no formerly correct item;
- rare-pivotal recall and unsupported-rate change; and
- replay success, latency, capability-limited rate, and revision count.

If the optional formative human study is run, it uses a within-subject, counterbalanced C1-versus-C2 design; C0 is exploratory if session length permits. Participants receive matched tasks and a maximum of three semantic revisions per task. Primary descriptive outcomes are answer correctness and time; secondary outcomes are revision count/type, evidence inspections, confidence on a fixed scale, a brief standardized mental-effort/workload measure selected before ethics review, trust calibration, and qualitative utility. Eight to twelve participants support process evidence only; a powered study is required for stronger population-level usability claims. The minimum study instead requires scripted revisions plus documented condition-blind researcher/expert traces through the same event model.

## 27. Unit, integration, property, regression, and end-to-end tests

### 27.1 Unit tests

Unit coverage includes every Pydantic invariant; ID/hash canonicalization; rights classifications; span offsets; context/horizon parsing; capability rules; evidence, lineage, budget and label validation; temporal relations; graph conversions; each metric equation and undefined case; config merging; seed derivation; cache keys; ledger transitions; and public-manifest redaction.

### 27.2 Property tests

Hypothesis strategies generate empty, singleton, complete, regular, disconnected, directed, multiedge, cyclic temporal, open/unknown interval, zero-relevance, one/all-singleton community, and alternative-gold graphs. Properties include:

- permutation invariance to safe ID/order renaming;
- metric bounds and identity for identical graphs;
- adding a provably unsupported assertion cannot improve strict assertion precision/F1 or grounding precision, cannot reduce unsupported-assertion rate, and fails the admissibility gate; no such monotonicity is asserted for recall, entropy, or community metrics;
- rare-pivotal labels independent of predicted frequency;
- evidence-index boundary classifier rejects canonical partitions, finalized predicates/events/assertions, and CoreNLP clusters not decomposed into defeasible pairs;
- production condition imports/filesystem accesses cannot reach `scoring/` or gold, and mutating any gold/relevance/rare label leaves every pre-score request/output hash unchanged;
- rendered prompts contain only `ModelVisibleQueryContext` and the allowlisted `ConstructorRunView`: base/paraphrase/pair/split IDs, condition-comparison labels, expected ablation effects, analysis configuration, and scorer handles are absent and inaccessible;
- graph conversion conserves the declared assertion/event incidence;
- temporal constraints preserve story/discourse/revelation separation and detect known contradictions;
- RDF round trips preserve qualifiers/provenance;
- two contexts over the same evidence can yield different gold decision signatures, while paraphrases share one signature;
- condition-independent feedback anchors resolve reproducibly without gold; shared `UserRevision` serialization contains no condition outcomes, and each condition receives at most its own resolution, never another condition's projection-local target ID or capability result;
- no C0/C1 query-time object lacks sealed lineage and no C2 cache key omits `q`/`h`;
- deterministic seed derivation and generator mutation locality; and
- no renderer action changes a semantic hash.

### 27.3 Integration and regression tests

Integration tests cover ingest -> `D_z` -> common packet -> each condition -> validation -> metrics -> UI DTO -> revision/replay. Fixtures include rare once-mentioned causal links, wrong merge/split traps, n-ary events, conflicting beliefs, flashback/revelation separation, early spoiler horizons, fixed-ontology capability failures, two repairs then failure, resumable jobs, and public-release scans.

Regression artifacts pin known small-graph entropy values, community partitions, normalized graph edits, crossing/overlap geometry, and previously found semantic failures. Scientific outputs are not broadly snapshot-tested byte-for-byte across GPU versions; semantic canonicalization and tolerance-aware expectations are used.

### 27.4 End-to-end tests

Playwright drives query entry, horizon and time controls, node/assertion/evidence panels, semantic zoom, focus, bundle expansion, revision actions, projection diff, history/undo, capability messages, accessibility navigation, and local/public evidence modes. It checks that future information never leaks through text or visual metadata, that rich labels contain only clauses tied to `validation_status=accepted` assertions, and that attributed/contested/hypothetical commitment is visibly distinguished from world assertion.

An experimental-integrity suite intentionally plants a hidden/cross-context `OntologyProjection` in the C2 cache, a future passage in an early snapshot, unrestricted `D_z` content in a case C2 request, a new C1 predicate, unequal packets, an inflated C2 budget, an evaluation/group ID in the rendered prompt, forbidden analysis/ablation metadata in `ConstructorRunView`, a scorer-gold import/access attempt, a gold-label mutation, an attributed false belief exported as an asserted base fact, and a label-only unsupported fact. Each must fail before scoring; only an explicit same-branch post-reveal parent reference is allowed for refinement.

## 28. CPU smoke tests and GPU acceptance tests

### 28.1 CPU smoke suite

The `configs/dev/cpu_smoke.yaml` suite uses three public worlds, 10--12 nodes, one contrast pair and paraphrase pair, C0 plus Fake/Replay backends, two renderer seeds, and all validators/metrics. It must finish in under 10 minutes on 8 vCPUs with no network and less than 8 GB RAM. It exercises every phase boundary, manifest, failure state, and UI contract but produces no C1/C2 research result.

### 28.2 Staged GPU model-acceptance suite

GPU acceptance is deliberately split so Phase 4 does not depend on Phase 5 software.

**Phase 4 runtime/C1 gate.** Run the real Qwen3-14B-AWQ/vLLM stack on query-blind development snapshots with three seeds. Require exact snapshot/checksum loading and JSON-schema decoding on the 24 GB 4090; peak VRAM below 23 GB and aggregate system-memory use below approximately 24.8 GB (at least 6.2 GB free), with no weight offload; at least 95% schema-valid C1 preontologies after no more than two repairs; 100% valid evidence IDs and zero programmed horizon leaks; reviewed grounding precision at least 0.95 and unsupported factual-clause rate at most 0.05; C1 query-blind seals and lineage immutability at 100%; three-seed completion, restart/resume, ledger/artifact verification, measured nondeterminism; and p95 latency/projected C1 hours inside Section 32. This gate freezes the candidate runtime/model for C1 while allowing a symmetric change if the later full gate fails.

**Phase 5 C2/operator/fixed-selection gate.** After C2 and `A-FixedSelect` exist, rerun the shared runtime checks and require at least 95% schema-valid C2 projections after no more than two repairs; 100% valid evidence IDs; zero programmed horizon leaks; grounding precision at least 0.95; unsupported factual-clause rate at most 0.05; C2 ontology-decision macro F1 at least 0.80, primary ontological-collapse rate at most 0.20, and rare-pivotal qualified-assertion recall at least 0.85; demonstrated post-query merge/split, contextual schema, event reification, abstraction, and temporal/epistemic changes; `A-FixedSelect` packing success and prohibited-operator rejection at 100%; three-seed completion/resume; and projected full-study GPU hours inside Section 32.

Thresholds are readiness gates, not claims that C2 is superior. They may be revised once on development data with a written rationale before held-out execution. If the full gate requires a model/runtime change, follow Section 5.3 and rerun **both** Phase 4 C1 and Phase 5 C2/`A-FixedSelect` acceptance under the new stack.

### 28.3 Synthetic-to-case acceptance

In addition to the full Phase 5 GPU gate, C0 must pass the frozen development reference suite at 100% schema/evidence validity, strict assertion F1 ≥0.55, temporal-relation F1 ≥0.60, and rare-pivotal recall ≥0.60. C1 preontologies must have schema validity ≥0.95 and comprehensive gold-assertion recall ≥0.70. These thresholds may change once with written development-only rationale; failure triggers baseline repair and rerun or stops the comparison, never admission of a weak baseline.

All gold/review/split/mutation checks, full metric edge cases, condition/fairness/cache audits, scorer isolation and gold-mutation invariance, `ConstructionRequest` gold-field exclusion, C2 packet-only access, scripted revisions, browser tests, statistical-analysis dry run, and public-release scan must pass. C2 need not beat C1 to continue; an integrity-valid null moves to the case study as a registered negative-transfer test.

## 29. Reproducibility and deterministic verification

Deterministic CPU stages use sorted canonical inputs, explicit locale/timezone, recorded thread counts, seeded random libraries, stable JSON serialization, floating-point tolerances, and content hashes. Each metric stores numerator/denominator and graph-view hash so results can be independently recomputed. Generator and config replay must reproduce byte-identical public artifacts on the locked platform.

GPU generation is controlled but acknowledged as potentially non-bitwise-deterministic. Exact weights, prompts, schemas, seeds, runtime/kernels, drivers, hardware, inputs, outputs, and resource logs make a run auditable; three replicates quantify variation. A rerun verifier distinguishes `byte_identical`, `semantically_equivalent`, `metric_equivalent_within_tolerance`, and `different` rather than declaring all seeded runs deterministic.

Reproduction bundles contain a public environment lock/container recipe, synthetic data/gold, commands/configs, prompt/schema/upper-vocabulary versions, ledgers and manifests, metric tables, analysis scripts, safe UI captures, and a citation/license inventory. Restricted bundles add local case-study hashes and mappings, never distributed text. Each paper table/figure carries an artifact manifest ID and regenerates from the frozen ledger.

## 30. Failure recovery and resumable GPU jobs

Jobs are small and resumable at `request`, `raw_generation`, `validation`, `repair_n`, `projection`, `metrics`, and `render` boundaries. Each stage writes to a unique temporary path, fsyncs where appropriate, verifies its hash, then atomically renames to its content-addressed final path and commits the ledger transition. An interrupted temporary file is quarantined, never assumed complete.

Failure policies are explicit:

- **OOM:** capture resource state; retry once with the same scientific input and a preregistered lower concurrency/KV setting that does not change evidence/output budgets. Otherwise mark failed and schedule the approved model/config block.
- **timeout/service crash:** restart the pinned server, preserve partial logs, and retry under the same seed/config with a linked attempt ID; final attempt policy is fixed.
- **invalid structure/semantics:** at most two diagnostic repair calls; persistent failure becomes an intention-to-treat failure.
- **evidence/horizon/lineage violation:** block immediately; do not “clean” the scientific output into validity.
- **corrupt/mismatched artifact:** quarantine, recompute from verified parents, and retain the incident record.
- **operator/user cancellation:** mark canceled, preserve checkpoints, and resume only with the same job ID or create a declared superseding job.

The queue supports pause between requests, disk-space checks, thermal/resource monitoring, and completion notifications. No retry silently changes a prompt, model, context length, quantization, evidence packet, or budget.

## 31. Phased milestones and dependency ordering

The work proceeds through nine gates in this exact order:

| Phase | Dependency | Principal deliverables | Gate before next phase |
|---|---|---|---|
| 1. Formal specification and schemas | none | notation, capabilities, Pydantic/JSON schemas, temporal semantics, configs, ledger skeleton, model pilot, minimal scorer/matcher kernels | contracts, boundary, and acceptance-metric tests pass |
| 2. Synthetic benchmark | Phase 1 | generator, eight development/24 test worlds, query distribution, gold alternatives, review/mutation logs, acceptance reference fixtures | benchmark freeze and audit pass |
| 3. Classical pre-query baseline | 1--2 | credible CPU C0, sealed snapshot construction, deterministic projector | C0 end-to-end and error review pass |
| 4. LLM pre-query condition | 1--3 | pinned Qwen/vLLM, C1 query-blind builder, consolidation, sealed lineage | Phase 4 runtime/C1 GPU gate, C1 competence and immutability pass |
| 5. Active query-dependent LLM condition | 1--4 | C2 constructor, decisions/certificates, repairs, revision reconstruction, `A-FixedSelect` | full Phase 5 C2/operator/fixed-selection, non-collapse, packing and fairness gates pass |
| 6. Automated metrics and ablations | 1--5 | alignment, entropy/clutter/community, rare guard, runner, preregistered analyses/ablations | edge-case tests and analysis dry run pass |
| 7. Interactive HITL visualization | 1--6 | FastAPI/Cytoscape app, rich panels/time/horizon, revisions/diffs, scripted and formative protocol | UI, replay, evidence, accessibility and ethics gates pass |
| 8. Game of Thrones case study | synthetic acceptance + 7 | local rights-aware index, 12 windows/48 contexts, curated subsets, registered runs | release/privacy and case artifact audit pass |
| 9. Statistical analysis and paper artifacts | completed paired runs | frozen models/intervals/effects, robustness, tables/figures, public/restricted bundles | independent manifest-to-claim audit passes |

Implementation does not start on copyrighted text to compensate for a failed synthetic method. Later phases may feed bug fixes back to earlier code, but any scientific change increments versions, invalidates caches, and requires rerunning affected frozen conditions.

Phase 8's registered default corpus is locally supplied English text of *A Game of Thrones*, *A Clash of Kings*, *A Storm of Swords*, *A Feast for Crows*, and *A Dance with Dragons*. Television scripts, fan wikis, and reference books are excluded. Exact editions/hashes are private-manifest inputs; the full neutral index locates 12 preregistered bounded windows, while central timing comparisons use the common all-admissible window snapshots described in Section 11. No raw text or reconstructive embedding enters a public artifact.

### Minimum publishable study and optional extensions

The minimum is Phases 1--9 with active C2 GPU construction, C0/C1, all controlled synthetic requirements, the preregistered metric panel, `A-FixedSelect`, a working rich interactive refinement loop with scripted tests and documented condition-blind researcher/expert refinement traces, local narrative ingestion, 12-window bounded case study, and paired analysis. The 8--12-person formative study is optional unless making a human-usability claim. A null C2 result is publishable if integrity holds.

Optional extensions are nested epistemic worlds, full bitemporal curation, a 32B/multi-model comparison, alternative temporal RDF stores, multiple narrative corpora, automatic LLM judging, a powered usability trial, collaborative multi-user editing, and hosted deployment. They cannot consume effort before the minimum's active LLM, UI/HITL, and case-study paths work.

## 32. Estimated resources and researcher effort

These are planning ranges to replace with measured Phase 1/4 rates. GPU-hour estimates include repairs and 20% rerun overhead but not the final contingency reserve.

Before execution, the runner materializes a worksheet that distinguishes **preontology jobs** from the GPU model calls inside them. Current mandatory maxima before repair are: up to 216 primary synthetic C1 jobs/calls (synthetic snapshots fit one call), plus up to 48 extra C1 variance-seed jobs; 288 canonical + 144 paraphrase + 48 extra variance-seed C2 calls; 288 `A-FixedSelect` calls; 4 × 144 calls for `A-NoContext`, `A-ShuffledContext`, `A-NoTemporalEpistemic`, and `A-NoRareGuard`; \(144t\) supplied-versus-sham feedback calls for \(t\in\{1,2,3\}\) frozen steps; 72 alternate-prompt C2 calls plus at most 72 C1 alternate-prompt **jobs**, each built once per unique `(world, D_z, seed, prompt)` and reused for all matching deterministic projections; and 144 initial case-study C2 calls. The case study has at most 108 C1 preontology jobs, not 108 model calls: every job expands into its recorded query-blind chunk-extraction and consolidation subcalls. Exact unique snapshot reuse, chunks/job, optional diagnostics, repair-rate multiplier, and case feedback are resolved in the sealed manifest. GPU hours are \(\sum_j N_j\times\) measured p50/p95 seconds for each actual call class, not guessed from a global average.

| Phase | RTX 4090 hours | Disk added | Active researcher hours | Notes |
|---|---:|---:|---:|---|
| 1. Specification/schemas/model pilot | 4--8 | 35--50 GB | 35 | model/environment snapshots dominate disk |
| 2. Synthetic benchmark | 2--5 | 2--5 GB | 75 | generation/gold/review mostly CPU/manual |
| 3. C0 | 0 | 2--5 GB | 40 | approximately 20--40 CPU hours including tuning |
| 4. C1 | 15--25 | 10--20 GB | 40 | preontologies, raw calls, consolidation |
| 5. C2 | 20--35 | 10--25 GB | 55 | 288 canonical, 144 paraphrase, and 48 extra variance-seed outputs plus development |
| 6. Metrics/ablations | 25--50 | 15--30 GB | 60 | frozen mandatory ablation registry dominates GPU; 30--70 CPU hours |
| 7. UI/HITL | 8--15 | 5--15 GB | 65 | scripted runs, researcher/expert traces, and optional formative-session outputs |
| 8. Case study | 25--45 | 25--50 GB restricted | 90 | ingestion/annotation and 48 contexts × replicates |
| 9. Analysis/artifacts | 6--12 | 10--20 GB | 50 | sensitivity reruns and release assembly |
| **Planned total** | **105--195** | **114--220 GB before deduplication** | **510 hours** | reserve 25% schedule/GPU contingency |

Content-addressing and Parquet should keep expected live storage near 100--220 GB, but require at least 300 GB free for temporary/quarantine headroom. Peak accepted usage is below 23 GB VRAM with at least 6.2 GB system RAM free; 8 vCPUs are sufficient because GPU generation is serialized. Plan roughly 9--12 months part-time for 510 active hours plus contingency, annotation scheduling, expert review, and overnight runs.

The runner produces an empirical forecast after development: calls × measured median/p95 seconds plus observed repair/OOM rates. If projected GPU time exceeds 200 hours before contingency, first reduce optional diagnostic/model-size subsets, then reduce mandatory-ablation subset sizes through a documented power/coverage amendment or evidence context through a method-visible recall analysis; never remove primary worlds, conditions, active construction, the prompt sensitivity, or the contrastive/paraphrase tests merely to meet the estimate.

## 33. Risks and fallback strategies

| Risk | Early indicator | Fallback that preserves the research question |
|---|---|---|
| 14B AWQ OOM/slow | VRAM >23 GB, high p95 | lower concurrency/context with support-retention proof; then matched 7--8B active model for C1/C2 |
| Structured output unstable | >5% invalid after repairs | simplify schema into staged calls with same semantic obligations; pin alternate grammar/runtime; keep failures |
| Context too long | truncation/support loss | deterministic evidence compression with IDs, smaller registered windows, hierarchical context-neutral indexing |
| C2 behaves like selector | no post-query decisions/high collapse | revise construction task on development, add operator-specific staged calls; do not relabel retrieval as success |
| C1 weak from consolidation | lineage duplicates/inconsistency | improve query-blind map/reduce and give credible budget; publish residual error |
| C0 weak | obvious missed common structures | improve frozen conventional rules/CoreNLP thresholds on development; no intentional degradation |
| Retrieval confound | gold evidence absent | primary synthetic/bounded-case all-evidence runs; `A-FullEvidence` only in the secondary full-index retrieval-realism case analysis; identical packet and recall reports |
| Hallucination | low grounding/label leaks | stricter evidence clause mapping, abstention, repair; never let CPU fabricate support |
| Temporal inconsistency | unsatisfiable outputs | staged temporal extraction/construction plus Z3 diagnostic repair; retain uncertainty |
| Rare pivotal loss | recall below 0.85 dev | preserve candidates/evidence budget, explicit guard and `A-NoRareGuard`; do not optimize sparsity alone |
| High stochastic variance | wide within-context spread | add development/variance samples, tighten prompt/decoder, report uncertainty; worlds remain analysis units |
| Community metrics unstable | resolution-dependent results | frozen Leiden primary, full sweep/Infomap sensitivity, exact small-graph inspection |
| UI becomes schedule sink | Phase 7 overrun | keep minimal Cytoscape feature set: graph, time/horizon, evidence, revision/diff; cut cosmetic polish, not interaction |
| Copyright leakage | release scan hit | quarantine artifact, strengthen allowlist/redaction, publish only synthetic/safe aggregates |
| Annotation burden | pilot >planned time | retain 48 high-level units, reduce detailed gold subset with preregistered precision analysis; do not remove case study |
| Recruitment delay | ethics/scheduling lag | complete scripted HITL and expert self-audit; postpone only powered study, not functioning UI or formative protocol |
| Null/negative result | C2 not superior | preserve preregistration, analyze mechanisms/cost/counterexamples, publish benchmark and negative finding |

Forbidden fallbacks are a deterministic C2, a complete hidden ontology, CPU-only scientific inference, cosmetic LLM labels, removal of time/evidence, arbitrary free-form gold, or deferring narrative ingestion, human refinement, and interactive presentation.

## 34. Definition of done for each major phase

### Phase 1 — Formal specification and schemas

Done when all typed contracts and condition capabilities are documented/generated; story time, validity, discourse, revelation, horizon, epistemic state, and curation provenance are distinct; model/runtime/hardware pilot configuration is pinned; minimal matching/strict-F1/temporal/rare/collapse acceptance kernels pass known fixtures; evidence-versus-ontology, scorer isolation, and query-reveal tests pass; public/restricted artifact policy and ledger migrations pass; and an independent requirements trace finds no unowned requirement.

### Phase 2 — Synthetic benchmark

Done when eight development and 24 sealed test worlds exist with 10--20-node gold projections, all controlled factors/difficulty tiers and numeric coverage quotas; 96 base contexts, the exact call inventory, contrastive nonselection deltas, paraphrases, declared distributions and seeds; complete world/evidence/entity/event/time/gold/rare/community/alternative artifacts inside the scorer firewall; held-out templates/renderers; manual review and mutation/temporal/determinism/gold-invariance tests; and a release bundle that reproduces without copyrighted data.

### Phase 3 — C0 ClassicalPre

Done when C0 constructs credible query-blind ontologies for every `D_z` on CPU, supports events/time/provenance and tuned development mappings, seals immutable IDs before query reveal, projects within equal scored budgets, passes lineage/evidence/horizon tests, reaches 100% schema/evidence validity, strict assertion F1 ≥0.55, temporal F1 ≥0.60, and rare-pivotal recall ≥0.60 on frozen development references, runs end-to-end, and has a documented blind error audit. Failure stops or repairs the baseline; it never creates a favorable weak comparison.

### Phase 4 — C1 LLMPre

Done when the exact model/tokenizer/AWQ/vLLM/prompt/schema revisions run within resource limits; query-blind single-call and map/reduce construction work; three sealed preontology replicates per world/horizon contain no query information; primary deterministic projection operates; 100% output atoms have lineage; C1 reaches ≥0.95 preontology schema validity and ≥0.70 comprehensive assertion recall on frozen development references; and the Phase 4 runtime/C1 GPU gate, resume, cost, and stochastic logs pass. `A-FixedSelect` is completed with the C2 query pipeline in Phase 5.

### Phase 5 — C2 LLMQuery

Done when C2 owns no pre-query ontology; the active GPU model constructs only after `q`; every operator (selection, merge/split, type, relation/schema, event reification, abstraction, temporal/epistemic qualification, method-visible pivotality handling, supported descriptions) is demonstrated and scored; `A-FixedSelect` runs with construction operators disabled and passes complete-packet/complete-graph packing; refinements reconstruct from structured `h`; construction certificates/cache audits pass; and the full Phase 5 development validity/grounding/non-collapse/rare gate passes.

### Phase 6 — Automated metrics and ablations

Done when every methodology metric has a tested implementation with formulas, denominators, direction, graph view and undefined-case policy; rare recall is a blocking co-outcome; all mandatory ablations run through the same interface; paired manifests and primary/multiplicity/missing-output policies are frozen; edge-case/property tests pass; and a synthetic analysis dry run regenerates all planned tables from the ledger.

### Phase 7 — Interactive HITL visualization

Done when the local app shows rich progressive nodes/assertions, separate story/discourse/revelation controls, spoiler-safe evidence, semantic zoom/focus/bundles, diffs/history, and every feedback action; semantic and renderer states remain separate; scripted revisions replay and documented condition-blind researcher/expert traces are preserved; C0/C1 capability limits are visible; clutter snapshots, accessibility, browser, privacy and release tests pass; and an ethics-ready optional formative-study protocol exists (approval/exemption is required only before recruitment).

### Phase 8 — Game of Thrones case study

Done when lawful local English editions of the five registered *A Song of Ice and Fire* novels are ingested with chapter/passage provenance but no text is tracked; 12 registered windows and 48 contexts cover the named narrative phenomena; all high-level and at least 24 detailed annotations are reviewed; evidence snapshots prevent spoilers equally; three conditions/replicates and feedback cases complete or retain failures; quantitative/qualitative/expert artifacts are ledger-linked; and public staging contains no protected/reconstructive content.

### Phase 9 — Statistical analysis and paper-ready artifacts

Done when locked scripts perform world-clustered paired inference, effect sizes/CIs, mixed-effects sensitivity, Holm/BH handling, non-inferiority/equivalence, all entropy/community/ablation sensitivity, and intention-to-treat failure analysis; every claim maps to a manifest/table/figure; negative results and compute costs are retained; public and restricted reproduction bundles verify; citations/licenses/ethics are current; and a final cross-document/requirement/claim audit passes.

## 35. Research-to-software traceability matrix

The table names the normative implementation hook for every hypothesis. Configuration keys are from Section 22; artifacts are immutable ledger objects. `WB paired` means world-blocked paired bootstrap/permutation, with hierarchical models as sensitivity analyses.

| RQ / hypothesis | Software modules | Controlling configuration | Required artifacts | Confirmatory metrics / analysis | Blocking tests |
|---|---|---|---|---|---|
| **RQ1 / H1a semantic alignment** | `conditions/*`, `metrics/alignment.py`, `validate/evidence.py`, `ontology/matching.py` | `condition.*`, `budget.*`, `metrics.matcher`, `validation.evidence_policy` | paired `OntologyProjection`, `GoldAlternativeSet`, evidence review, fairness manifest | strict assertion-F1 superiority; node-F1 lower CI >-0.05, grounding lower CI >-0.03 and absolute C2 grounding ≥0.95 safeguards; C2--C1 then C2--C0, WB paired CI/Holm | same `D_z/E_q`; matcher permutation; empty prediction; unsupported labels; C0/C1 lineage |
| **RQ1 / H1b ontological organization** | `contracts/ontology.py`, `conditions/llm_query.py`, `metrics/alignment.py`, `metrics/contrastive.py` | `condition.capabilities`, `prompt.construct_id`, `metrics.matcher` | `OntologyDecision`, construction certificate, local schema, gold operator labels | ontology-decision macro-F1 superiority; each applicable component lower CI >-0.10; event/merge/type/schema/abstraction/qualification effects; WB paired | four-case empty-family rule; post-query timestamp; required operator fixtures; hidden-ontology/cache injection |
| **RQ1 / H1c appropriate sensitivity/invariance** | `benchmark/queries.py`, `benchmark/paraphrases.py`, `metrics/contrastive.py` | `query.distribution_version`, `query.paraphrase_mode`, `decoding.seed_block` | `ContrastSet`, `ParaphraseSet`, aligned decision signatures | upward union-family signed decision-change F1 and downward primary ontological-collapse rate with Holm; invariant/paraphrase correctness lower CIs >-0.05 and divergence upper 95% CI <0.10 | four-case family rule; selection-only collapse fixture; arbitrary-divergence rejection; wording-only invariant; grouped resampling |
| **RQ2 / H2a constrained clutter** | `metrics/clutter.py`, `ui/src/graph`, `validate/budgets.py` | `visualization.*`, `budget.visible_marks`, `metrics.clutter`, `metrics.rare_gate` | semantic graph, `VisualizationState`, screenshot/geometry log, gold relevance, `crossing_eligible` manifest | irrelevant visible load and failure-adjusted crossing rate on frozen unbundled overview; raw/normalized overlap, occlusion, path ambiguity; claim only with fidelity/rare gates | fixed renderer/font/viewport; layout seed; bundle attribution; zero-opportunity rule; unreachable paths; empty graph cannot win; hidden ≠ absent |
| **RQ2 / H2b entropy profile** | `ontology/graph_views.py`, `metrics/entropy.py` | `metrics.graph_views`, `metrics.entropy_panel` | hashed graph conversions, complete entropy table | all degree/local/relation/relevance/von-Neumann/community-size entropies, required secondary exploratory two-sided WB paired/BH family | toy known values; frozen `OTHER`; primary VNE view; empty/singleton/complete/disconnected/zero-weight; no metric omission |
| **RQ2 / H2c rare-pivotal preservation** | `scoring/gold_compile.py`, `metrics/rare_pivotal.py`, `validate/evidence.py` | `metrics.rare_gate`, `budget.*`, `validation.evidence_policy` | scorer-only `rare_eligible` manifest/labels, support paths, budget curves | eligible-context/world-macro \(\Delta_R=R_{C2}-R_{C1}\), lower 95% CI > -0.05 primary; corrected C2--C0 required; grounding/fidelity floor | positive denominators; noneligible `NA`; eligible invalid=0; gold mutation invariant; once-only pivotal regression; `A-NoRareGuard` |
| **RQ3 / H3 contextual communities** | `metrics/community.py`, `ontology/graph_views.py`, `scoring/gold_compile.py` | `metrics.community.{algorithm,objective,resolution,seeds}`, `metrics.graph_views` | fixed gold-anchor `U_g`, eligible hard/soft partitions, detected partitions, co-assignment matrices, frozen rubrics | AMI, unweighted mean conductance, once-counted separation, pairwise fragmentation/merge error; NMI/purity/coherence required secondary, interpretability human-rated, modularity/count descriptive; WB paired/Holm | unique `ABSENT` anchors; extra-vertex report; zero-denominator/volume rules; one/all-singleton/soft cases; rubric independence; algorithm versus ontology stability |
| **Construction-freedom mechanism / RQ1** | `conditions/fixed_select.py`, `metrics/alignment.py`, `metrics/contrastive.py`, `experiments/statistics.py` | `A-FixedSelect`, mechanistic Holm family, matched budgets/seeds | fixed-select projection, capability/packing reports, paired C2 projections | C2 superiority on ontology-decision macro F1 and contrastive change F1; strict-F1/rare-recall lower CIs >-0.05 | full packet/graph fit; all novel operators rejected; sealed-ID lineage; excluded from exploratory BH |
| **HITL mechanism supporting RQ1--3** | `api/revisions.py`, `conditions/*`, `metrics/alignment.py`, UI revisions | `budget.feedback_steps`, `validation.*`, `model/prompt/seed` | anchors/resolution logs, append-only revisions, before/after projections/diffs, capability responses | recovery gain, target-change recall, revisions/correction, corrections/1k generated tokens, edit locality, nonlocal regression, rare/unsupported deltas, latency | condition-independent anchor; empty-diff rule; zero-token/nonlocal-denominator `NA`; full replay; deterministic evidence expansion; C2 GPU regeneration |
| **Prompt/other-ablation robustness / all RQs** | `experiments/runner.py`, `configs/ablations/registry.yaml` | seven mandatory IDs and diagnostics, selected context IDs, budgets/seeds | ablation projections, capability reports, prompt variant, supplied/sham revisions | registered within-unit deltas; secondary exploratory BH excluding `A-FixedSelect`; full negative results | config coverage, operator disable, same budget/evidence, prompt independence, anchor equality |
| **Experimental integrity / all RQs** | `experiments/runner.py`, `ledger.py`, `cache.py`, `conditions/capabilities.py`, `validate/lineage.py`, `scoring/firewall.py` | all rendered config hashes | prequery seals, query reveal, C2 inventory/certificate, access/fairness/cache/gold-firewall manifests | failure/repair/cost reports and intention-to-treat inclusion | unequal corpus/budget, future evidence, unrestricted C2 snapshot, novel C1 ID, cross-context C2 cache, gold import/mutation, unmanifested artifact all fail |

This matrix is maintained as a testable artifact in Phase 1. A module, metric, or UI feature that cannot be traced to a research question is optional; a hypothesis row without passing artifacts and tests cannot support a claim.
