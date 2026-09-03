# Methodological Plan: Conference Study of Active Query-Dependent Temporal Ontology Construction

**Document status:** authoritative scientific plan for the conference-paper study.
**Companion:** `IMPLEMENTATION_PLAN_QUERY_DEPENDENT_TEMPORAL_ONTOLOGY.md`.
**Planning date:** 2026-09-03.
**Resource envelope:** one RTX 4090 (24 GB), approximately 31 GB RAM, 8 vCPUs, at most 9 scheduled and 10 actual allocated GPU hours, and at most 30 GB project-controlled writable storage.

## Conference scope boundary

This revision turns the broader research program into one bounded conference paper. It retains the causal comparison needed to study construction timing, but removes breadth that cannot be defended under the resource ceiling.

The conference minimum defers the remaining four published *A Song of Ice and Fire* novels; book-series-wide inference; participant recruitment and usability experiments; powered workload, trust, or cognition claims; collaborative editing; branching histories and elaborate undo/redo; production deployment, authentication, distributed queues, and Server-Sent Events; multi-model and prompt matrices; thinking-mode experiments; the broad ablation registry; multi-algorithm and multi-seed community batteries; large screenshot banks; nested belief worlds; complete bitemporal curation; extensive RDF 1.2 experimentation; alternative graph stores; and broad ontology-language comparisons. These are future work, not hidden acceptance criteria.

The retained interface is a thin local demonstration, not a public service. The retained Game of Thrones study covers one lawfully supplied novel and eight bounded queries, not the series. The main evidence is a controlled synthetic benchmark plus descriptive narrative transfer evidence. Claims are model-specific and benchmark-bounded.

Historical plans are retained only as provenance. Their drift toward a complete oracle graph, deterministic fixed-template projection, CPU-only execution, and deferred LLM, narrative, interaction, and visualization work is explicitly superseded. This plan keeps their useful ideas—typed artifacts, gold isolation, provenance, temporal validation, and reproducible ledgers—without allowing a prebuilt graph to stand in for active post-query construction.

## Non-negotiable contribution

This scope-control section is immutable. Scope reduction must not remove or weaken any of the following:

1. active query-time ontology construction by a GPU-hosted LLM;
2. separately implemented `C0 ClassicalPre`, `C1 LLMPre`, and `C2 LLMQuery`;
3. `A-FixedSelect` as the mechanistic control for query-time selection without construction freedom;
4. temporal qualified assertions rather than timeless triples;
5. contrastive, known-answer projections over the same evidence;
6. explicit preservation and evaluation of rare but pivotal facts; and
7. a bounded narrative case study using locally supplied *A Game of Thrones* text.

If the GPU or storage pilot cannot support these items, the study stops or changes the common model/window size symmetrically. It must not fall back to the superseded CPU-only fixed-graph project.

## 1. Executive summary

The paper asks whether a narrative ontology should be fully organized before the user's information need is known or constructed after the query, time horizon, perspective, and abstraction are supplied. Dense comprehensive narrative graphs often obscure the relationships needed for a particular question. Ordinary pruning can reduce density, but it can also remove a once-mentioned fact that changes a causal chain, allegiance, identity, or revelation. Frequency alone is therefore not a valid importance criterion.

The method separates a query-blind **EvidenceIndex** from a constructed ontology. The index may contain passages, span identifiers, mention and alias candidates, temporal clues, surface relation candidates, provenance, uncertainty, and a compact retrieval index. It may not contain the final entity partition, event graph, local relation vocabulary, or a complete canonical ontology secretly used by C2. Once the query is known, C2's active LLM decides which mentions to merge or separate, which events to reify, which contextual types and relations to use, what abstraction is appropriate, which temporal or holder-level epistemic qualifications matter, which rare evidence is pivotal, and how supported nodes and assertions should be described. Deterministic CPU code validates rather than invents these decisions.

Three conditions isolate the effect. C0 constructs a conventional ontology on CPU before queries. C1 uses the same LLM as C2 but constructs before queries. C2 constructs after the query. The primary contrast is C2 versus C1; C2 versus C0 is secondary. `A-FixedSelect` uses C2's query-time LLM pathway over C1's sealed ontology while forbidding new semantic objects, isolating construction freedom from better query-time attention or selection.

The controlled benchmark has four development worlds and 12 held-out worlds. Each test world supplies three QueryContexts, including a contrastive pair, for 36 primary world-context units. Gold contextual projections contain approximately 10--20 entity/event nodes. C1 and C2 use two paired seed blocks, but the 12 worlds—not contexts, seeds, nodes, or edges—are the independent units. A 12-context paraphrase subset tests wording stability. At least three complete test worlds (25%) receive mandatory condition-blind independent gold review.

The transfer demonstration query-blindly indexes one local lawful copy of *A Game of Thrones*, selects four preregistered windows, and uses two contexts per window. One seed is used. One additional query retrieves a frozen bounded packet from the complete first-novel index and is labeled an operational retrieval demonstration rather than a clean causal comparison. No copyrighted prose, reconstructive embeddings, or detailed text artifacts are redistributed.

The conference minimum retains a small interactive graph and nine feedback regenerations: six scripted known-answer synthetic corrections and three researcher-driven interface traces. It makes no usability claim. All GPU activity—including loading, pilots, development, repairs, failures, ablations, case calls, and reruns—is metered. The provisional inventory contains 278 inference-call slots and eight load/allocation events, forecasts 8.67 hours at the admission ceilings, reserves a scheduled envelope of 9 hours, and enforces an absolute 10-hour stop. Writable occupied storage targets 25 GB inside a 30 GB allocation.

## 2. Problem, definitions, and notation

Let \(X\) be the legally available corpus and \(I\) a query-blind indexing process. For a registered discourse or spoiler horizon \(z\),

\[
D_z=I(X_{\le z}), \qquad M_z=\operatorname{Manifest}(D_z).
\]

`D_z` is an evidence snapshot; `M_z` is its immutable hash and access manifest. A structured query

\[
q=(\text{wording},\text{lens},\text{target},\tau_q,\sigma_q,\kappa_q,\alpha_q,B_q)
\]

contains story-time scope \(\tau_q\), spoiler horizon \(\sigma_q\), optional epistemic viewpoint \(\kappa_q\), desired abstraction \(\alpha_q\), and output budget \(B_q\). A deterministic retriever records one packet \(E_q=R(D_z,q)\). In the primary synthetic and bounded-window case comparisons, \(E_q\) contains all admissible evidence in the small snapshot; retrieval is not the treatment.

Five semantic layers must remain distinct:

- \(U\): a small, stable upper ontology of primitive categories and temporal/epistemic terms;
- \(S_q\): the query-time local contextual schema, including contextual types and relation definitions;
- \(K_q\): the query-specific entity/event instance graph and qualified assertions using \(S_q\);
- \(O_q=(U,S_q,K_q)\): the constructed contextual ontology; and
- \(V_q=\operatorname{Render}(O_q)\): the rendered network view, whose layout and visibility do not change semantics.

This terminology prevents ordinary instance or subgraph selection from being called ontology construction. Ontological decisions include the mention-to-entity partition, event boundaries, local types, relation definitions, event reification, abstraction, and temporal/epistemic qualifications. Relevance selection is recorded but cannot alone establish construction.

Pre-query construction followed by projection is:

\[
O_{\mathrm{pre},z}=F_{\mathrm{pre}}(D_z),\qquad
P_{\mathrm{pre}}(q,h)=\operatorname{Project}(O_{\mathrm{pre},z},E_q,q,h).
\]

Query-dependent construction is:

\[
O_{q,h}=F_{\mathrm{query}}(M_z,E_q,q,h),\qquad
V_{q,h}=\operatorname{Render}(O_{q,h}),
\]

where \(h\) is an ordered history of condition-independent `UserRevision` records. C0/C1 output atoms must inherit a sealed pre-query ID. C2 receives packet contents only after query reveal and may create locally defined semantic objects grounded in that packet. It cannot read a hidden preconstructed ontology or an unlogged full-index channel.

An assertion is a qualified record rather than a timeless triple:

\[
a=\langle s,p,o/\mathbf r,\tau_s,\tau_v,\delta,\tau_r,\kappa,\gamma,\Pi,\eta_q,\omega_q\rangle .
\]

Here \(\mathbf r\) is an optional n-ary role map; \(\tau_s\) is event/story time; \(\tau_v\) is a validity interval or partial order; \(\delta\) is discourse position; \(\tau_r\) is proposition revelation position; \(\kappa\) is an optional holder-level epistemic status; \(\gamma\) is confidence; \(\Pi\) is provenance; \(\eta_q\) is contextual relevance; and \(\omega_q\) is a supported reason the assertion matters. Unknown values remain explicit.

## 3. Evidence index, construction, projection, and rendering

The four stages produce separately hashed artifacts.

| Stage | Query known? | Permitted | Prohibited |
|---|---:|---|---|
| Evidence preparation | no | documents/passages/spans; mention, alias and coreference candidates; temporal clues; surface relation phrases; uncertainty; provenance; FTS terms | finalized entities, event ontology, predicates, query relevance, or comprehensive C2 graph |
| Ontology construction | C0/C1 no; C2 yes | entities, events, schema, merge/split decisions, qualified assertions, abstraction, contextual descriptions | importing another condition's ontology or scorer gold |
| Fixed projection | yes | select, rank, compress, and faithfully describe sealed objects | new identity, predicate, event, abstraction, or qualification |
| Rendering | yes | coordinates, style, filtering, evidence panels, temporal controls | factual additions or unrecorded semantic mutation |

Candidate coreference and OpenIE-like phrases remain defeasible. If preprocessing makes a semantic decision that cannot be avoided, that decision is documented, shared by all conditions, and excluded from the claimed query-time contribution. A boundary test rejects canonical entity IDs, finalized predicates, reified events, or qualified truth assertions from C2's pre-query namespace.

C2 must pass four operational non-collapse tests:

1. a pre-query inventory contains no contextual ontology;
2. a post-query construction certificate records nonselection decisions;
3. contrastive contexts with a gold nonselection difference are scored on typed decision changes, not displayed membership alone; and
4. `A-FixedSelect` receives the same query-time pathway and packet but is mechanically unable to create those changes.

## 4. Experimental conditions

### C0 ClassicalPre

C0 is a compact but credible deterministic CPU pipeline. Before test queries, it applies conventional NER/dependency parsing, normalization, alias/coreference heuristics, event and temporal patterns, a small upper vocabulary, and assertion-level provenance to each world/window. Query time uses fixed relevance scoring, typed-path continuity, support-aware selection, and rendering. It cannot create entities, merge/split, predicates, event structures, abstractions, or temporal/epistemic qualifications after reveal. Development rules are tuned on the four development worlds, frozen, and reported with their limitations.

### C1 LLMPre

C1 uses the exact model, tokenizer, quantization, runtime, structured decoder, and decoding family used by C2. For each test world and seed, it constructs one comprehensive ontology before the three QueryContexts are revealed; that sealed ontology is reused across all three. Each case-study window has one sealed C1 ontology reused for its two contexts. Query-time projection is deterministic and restricted to sealed IDs. No extra snapshot is created unless a distinct primary horizon truly requires it; the chosen conference design uses one shared horizon per world/window.

### C2 LLMQuery

C2 owns only the neutral index before reveal. After receiving \(q\) and frozen \(E_q\), the GPU LLM constructs \(S_q\), \(K_q\), and the qualified assertions. It records relevance, identity, schema/relation, event, abstraction, temporal/epistemic, rare-preservation, and description decisions. At least the merge/split, event-reification, relation/abstraction, and temporal-qualification operators must be demonstrated in held-out gold cases. After permitted feedback, it reconstructs from \(q\), \(E_q\), and structured \(h\). CPU validation may reject or request one bounded repair; it cannot supply missing semantics.

### `A-FixedSelect`

`A-FixedSelect` runs for all 36 primary synthetic contexts and both seeds: 72 calls in the frozen conference design. If the p95 forecast cannot accommodate them, Phase 1 may switch the common model or pre-freeze work may reduce the common evidence/sequence window; otherwise the study stops. It never weakens this mechanistic comparison. It uses C2's query-time model pathway, packet, maximum input/output limits, and repair policy while enforcing C1's sealed IDs and disabling:

- new entities and merge/split;
- schema creation or new predicates;
- event reification;
- abstraction changes; and
- new temporal or epistemic qualifications.

Its grammar permits selection, compression, and support-preserving descriptions only. A blocking packing report proves that the complete packet and required sealed graph fit without truncation.

## 5. Research questions, endpoints, and hypotheses

### RQ1: context alignment

Does construction after the query yield a more context-aligned ontology than construction before the query?

- **H1-semantic:** C2 has higher world-averaged strict qualified-assertion F1 than C1. This is the single primary semantic endpoint.
- **H1-organization:** C2 has higher world-averaged ontology-decision macro F1 than C1. This is the single primary ontology-organization endpoint.
- **H1-mechanism:** C2 has higher ontology-decision macro F1 than `A-FixedSelect`, supported by contrastive decision-change F1 and construction certificates. Without this result, gains cannot be attributed specifically to construction freedom.
- C2 versus C0 is a required secondary contrast for all retained endpoints.

Rare-pivotal qualified-assertion recall is a noncompensable safeguard, not another optimization target. C2 cannot support a favorable claim if its world-level difference from C1 crosses the preregistered substantive loss margin, initially \(-0.05\). Evidence grounding and unsupported-assertion results must also accompany both primary endpoints.

### RQ2: entropy and clutter

How does query-dependent construction change graph entropy and visual clutter while preserving fidelity and rare-pivotal information?

No entropy direction is called inherently better. The entropy panel and clutter measures are required secondary reports. A constrained simplification statement is permitted only when strict fidelity, grounding, and rare-pivotal safeguards pass. Fewer nodes or zero crossings caused by an empty or invalid graph is failure.

**H2-profile (secondary, two-sided):** C2 and C1 differ in their registered entropy-and-clutter profile after semantic validity and rare-pivotal preservation are made visible. The study does not predict that every entropy measure decreases. The narrower directional expectation is that C2 reduces irrelevant visible load; crossings and overlaps are interpreted conditionally on content-bearing valid outputs.

### RQ3: contextual communities

How does query-dependent construction affect community organization?

AMI, purity, conductance, fragmentation, and merging error are required secondary outcomes. Cluster count and modularity are descriptive. Neither more nor fewer communities is presumed superior. With 12 independent worlds, RQ2/RQ3 findings are bounded structural evidence rather than broad population claims.

**H3-organization (secondary):** relative to C1, C2 increases gold-aligned AMI and reduces fragmentation/merging error on eligible contexts; conductance, modularity, community count, and cross-seed stability are reported without treating any raw count as an inherent target.

## 6. Focused related work and research gap

Classical ontology learning derives concepts, taxonomies, relations, and instances from text, commonly for a reusable corpus-level artifact [R1--R3]. Contextual ontology work demonstrates that knowledge and mappings may be local to a context, and query-driven approaches can learn or select contextual modules [R4--R5]. These traditions motivate C0 and the evidence boundary, but do not by themselves isolate whether narrative identity, event, schema, and qualification decisions should be made before or after a user's context.

Temporal RDF, Allen interval relations, OWL-Time, and annotated-fluent models show why mutable relations cannot be represented adequately as timeless triples [R6--R9]. SEM, EventKG, event-centric extraction, and narrative ontologies motivate explicit events, participant roles, narrative order, and provenance [R10--R13]. Named graphs, PROV-O, and grounded perspective representations motivate source-aware and epistemically qualified assertions [R14--R16].

Interactive ontology tools and human-in-the-loop ontology engineering show the value of inspectable corrections [R17--R19]. The conference study narrows this idea to a reproducible typed revision and a few auditable reconstruction traces, not a usability trial.

LLMs4OL, SPIRES, and SAC-KG demonstrate LLM-assisted ontology or knowledge-graph construction [R20--R22]. RAG and GraphRAG retrieve or summarize from indexes and graphs built before a question [R23--R24]. Dynamic-ontology work for agents is close in spirit and prevents a broad priority claim [R25]. The novel target here is the controlled C1-versus-C2 timing contrast for temporal narrative ontology decisions, with `A-FixedSelect` isolating selection.

Dynamic-graph, semantic-zoom, and topological-fisheye work motivates focus-plus-context and progressive disclosure [R26--R27, R33]. Graph-readability studies caution that crossings and label clutter affect tasks but do not themselves prove usability [R28]. Graph entropy has inequivalent definitions, and community detection is sensitive to objective and resolution [R29--R32, R34--R35]. Accordingly, this study uses a small declared panel, a two-part crossing analysis, fixed rendering, and restrained claims.

The contribution is not “an LLM extracts a graph,” the first contextual ontology, or the first dynamic graph. It is a compact benchmark and experiment for active, post-query temporal ontology construction with known contrastive answers, evidence grounding, rare-fact safeguards, and a narrative transfer demonstration.

## 7. Temporal and epistemic representation

The internal representation remains a typed hybrid of contextual entities, event objects, local schema elements, proposition contents, and qualified assertion records. A binary assertion is still first class. An event is reified when it has multiple roles, independent duration, causality, contested occurrence, or query relevance. N-ary roles carry their own evidence and qualification.

The conference minimum supports:

- entity and event identity;
- binary and n-ary relations;
- story/event time;
- relation or state validity intervals and partial orders;
- discourse and proposition-revelation positions;
- a query spoiler horizon;
- evidence provenance and confidence;
- contextual relevance and supported descriptions; and
- one holder-level epistemic attitude (`known`, `believed`, `reported`, `denied`, `uncertain`) where applicable.

Story time, validity, discourse order, revelation, and spoiler horizon remain distinct. An attributed proposition is not asserted as world truth unless a separate world-committed assertion exists. Pipeline validation (`accepted`, `rejected`, `invalid`), narrative commitment, evidence sufficiency, and temporal underdetermination are independent statuses.

The minimum implements exact, bounded, relative, partial-order, and unknown time plus a tested subset of Allen relations needed by generated cases. Nested belief, possible-world semantics, exhaustive bitemporal correction history, and broad ontology-language experiments are deferred. A small RDF 1.1-compatible n-ary/provenance export may be included as an interoperability example if it does not displace primary work; RDF export is not an experimental condition.

Every displayed assertion must expose or make discoverable its normalized relation, direction/role, temporal validity, applicable holder status, confidence, provenance, evidence location, contextual relevance, and why it matters. Generated text cannot add a factual clause without an accepted, supported assertion.

## 8. Synthetic benchmark

### 8.1 Size, units, and splits

The benchmark contains four development worlds and 12 sealed test worlds. Each test world has three base QueryContexts, giving 36 primary world-context units. Two of the three form at least one contrastive pair over exactly the same evidence and require at least one nonselection change. The third supplies another lens, a fixed-ontology-friendly case, or both. Gold contextual projections contain approximately 10--20 entity/event nodes and controlled extra assertion/distractor density.

Two paired LLM seed blocks are used for C1, C2, and `A-FixedSelect`. A C1 preontology is built once per world and seed, then reused. Scores are averaged within world across contexts and seeds before primary inference. The independent sample size is 12 worlds—not 36 contexts, 72 seeded outputs, nodes, assertions, or edges.

The 12-context paraphrase subset contains one context per test world, balanced across lens and difficulty. It runs one registered seed for C2; C0/C1 deterministic projections reuse existing preontologies. Paraphrases hold every structured field constant. This is a stability check, not an independent efficacy sample.

### 8.2 Representation-neutral generation

`WorldSpec` describes actors/personas, collectives, places, events, states, reports, aliases, causal dependencies, partial temporal order, disclosure order, and evidence witnesses without using the C2 output schema. Separate deterministic compilers produce narrative passages, neutral evidence records, QueryContexts, and gold contextual projections. Gold IDs never enter model-visible text.

Across the 12 worlds the frozen coverage matrix must include:

- temporal relationship changes;
- context-dependent mention merge or split;
- event reification differences;
- relation-definition or abstraction differences;
- frequent but irrelevant relationships;
- rare, once-mentioned pivotal relationships;
- a small but nontrivial holder-belief/report subset;
- incomplete or competing evidence with legitimate uncertainty; and
- cases in which a comprehensive pre-query ontology is sufficient or preferable.

Every factor appears in at least three held-out worlds. Every test world contributes at least one rare-pivotal gold assertion across its three contexts, and at least eight contexts qualify for `A-NoRareGuard`. Temporal changes appear in at least six worlds, with at least eight contexts eligible for `A-NoTemporalEpistemic`. At least eight contexts spanning at least eight distinct worlds have meaningful gold communities. Density/distractor levels are stratified so graph clutter remains measurable without inflating gold beyond 20 nodes.

### 8.3 Gold and independent audit

Each bundle preserves the underlying world, narrative evidence, mentions, permissible entity partitions, entities/events, temporal and epistemic assertions, query-specific local schema and graph, relevant/irrelevant components, rare/common and pivotal/non-pivotal labels, contrastive signed changes/invariants, evidence support, and community assignments where meaningful. `GoldAlternativeSet` lists defensible equivalent structures or local matching rules.

Gold is compiled from formal world/query semantics, never from the evaluated LLM. Mutation tests change a fact, context, or horizon and require the exact expected gold delta. Held-out generator structures and at least one surface renderer are excluded from development prompt/rule tuning. Null and pregraph-friendly cases prevent the benchmark from encoding that C2 must always win.

Independent review is mandatory. A recorded seed selects one world at random within each difficulty stratum, so a second reviewer covers three held-out worlds (25% of worlds and nine projections) while blind to method outputs and condition identities. The reviewer checks evidence support, identity partitions, event choices, temporal scope, contrast deltas, rare-pivotal labels, communities, and alternatives. Disagreements are logged and adjudicated before condition outputs are opened. If a reviewer is unavailable, held-out execution does not begin.

### 8.4 Query sampling

The generator samples typed contexts before producing language. The 36-context lens multiset contains exactly six instances of each compact family: allegiance/state change, identity/kinship, causal consequence, event participation/conflict, movement/time, and knowledge/belief. A registered seed permutes that multiset within easy/medium/hard blocks, subject to three distinct lenses per world and the frozen factor quotas; this is balanced random allocation, not an unknown free-form prompt distribution.

Within the 36 slots, story-time scope is blocked to 12 point, 12 interval, and 12 through-bound contexts. Abstraction is blocked to 12 actor-level, 12 event/role-level, and 12 collective/causal-chain contexts. Twelve contexts request a holder viewpoint and 24 request world-level commitment. Each world uses one query-blind evidence snapshot and spoiler horizon for all three primary contexts; across test worlds, six horizons end at an intermediate disclosure cut and six at the world-final cut, randomized within difficulty. Eligible value choices within a block are uniform over the finite choices declared by `WorldSpec`. Node budgets are balanced among 10, 15, and 20. The compiler records the candidate set, probability or blocked allocation, every rejection, the registered seed, and the realized distribution; it never resamples based on model output.

Contrastive pairs must change a gold entity partition, schema/relation, event reification, abstraction, or temporal/epistemic qualification; merely choosing different existing nodes is insufficient. A held-out natural-language renderer is selected uniformly from the eligible renderer set. The 12-context paraphrase subset uses a different renderer while keeping all structured fields and the gold signature fixed.

## 9. Bounded *A Game of Thrones* case study

The only copyrighted source in the conference minimum is one locally supplied lawful English copy of *A Game of Thrones*. The complete novel is segmented and indexed query-blindly using chapter/passage IDs and SQLite full-text search. The raw text remains in one restricted local location and is never copied into run artifacts.

Four windows are preregistered on evidence and narrative criteria before condition outputs. Each has two same-horizon contexts, giving eight case query units, and is chosen to support a contrastive pair involving identity, allegiance, event participation/causality, temporal state, or belief/revelation. For the clean transfer comparison, all conditions receive all admissible evidence in the bounded window; C1 creates one preontology per window and reuses it for both contexts. C1 and C2 use one primary seed. Results are descriptive transfer evidence, not a powered second experiment.

All eight units receive evidence, relevance, temporal, rare-pivotal, and high-level organization review. Four receive more detailed assertion/event matching with permissible alternatives. A second knowledgeable reader reviews the selected paper examples where available; disagreement is preserved rather than treated as model error automatically.

One extra preregistered operational query runs C2 once over a frozen bounded packet returned by SQLite FTS/BM25 from the complete first-novel index. Its retrieval ranks, omissions, and evidence coverage are reported. It is explicitly not part of the same-evidence construction-timing causal comparison. The remaining four novels, extensive annotation, and series-wide conclusions are future work.

Public artifacts contain code, prompts, schemas, synthetic data, seeds, aggregate metrics, and safe opaque or chapter-level locators. They exclude prose, reconstructive embeddings, raw snippets, detailed offset maps that enable reconstruction, and model artifacts containing protected passages.

## 10. Minimal structured human refinement

The conference interaction mechanism retains two semantic action types:

1. `REFINE_CONTEXT`: patch the lens or story-time/spoiler scope; and
2. `REQUEST_MERGE_SPLIT`: request a specified mention/entity merge or separation.

These actions are sufficient to show that context and identity changes trigger genuine reconstruction. Other actions—including free correction batteries, evidence pagination, elaborate relevance editing, branching, collaborative histories, and undo/redo—are deferred.

A `UserRevision` is a condition-independent request containing typed intent, evidence/mention-based `FeedbackAnchor` records, rationale, sequence, and context hashes. It contains no projection-local ID or condition result. A frozen gold-blind resolver creates a separate `FeedbackResolution` for each condition. The receiving condition may see only its own resolution. C2 regenerates on the GPU; C0/C1 reproject sealed objects or return an explicit `capability_limited` response.

Six preregistered synthetic contexts receive one scripted known-answer revision each using one seed. The targets include three context/time refinements and three merge/split requests. Report post-minus-pre strict F1, target-change recall, edit locality, unsupported changes, rare-pivotal change, capability-limited rate, latency, and replay success. Three additional researcher-driven local-interface traces are recorded, each with one semantic action and one C2 regeneration. They demonstrate operability and auditability only. No participant, workload, trust, cognition, or usability-superiority claim is permitted.

## 11. Minimal rich visualization

A single local Cytoscape.js page presents the validated ontology. The overview exposes contextual node name/role/type, concise informative assertion labels, temporal validity, confidence or epistemic/uncertainty status, and evidence availability. A detail panel exposes aliases, relation definition, event roles, provenance, evidence locator, and `why_matters`. A story-time or spoiler control updates context, and a before/after diff marks added, removed, merged, split, or requalified objects.

Progressive disclosure prevents rich metadata from becoming immediate clutter. Pan, zoom, focus, evidence expansion, and temporary hiding are renderer state; they cannot count as ontology construction. One frozen layout configuration and seed is used for metrics. Screenshots are retained only for selected test fixtures and paper figures. The interface is local, single-user, and synchronous; production deployment and comprehensive accessibility certification are outside the conference claim.

## 12. Context-alignment and preservation metrics

All component matching is condition-blind and uses gold mention/evidence anchors plus permissible alternatives.

The retained RQ1 panel is:

- contextual node precision, recall, and F1;
- strict qualified-assertion precision, recall, and F1;
- ontology-decision macro F1;
- contrastive decision-change F1 and ontological-collapse rate;
- paraphrase signature divergence and strict-F1 change on 12 contexts;
- evidence citation validity and grounding precision;
- unsupported factual-assertion rate;
- rare-pivotal qualified-assertion recall; and
- essential temporal-qualification accuracy.

Strict assertion matching requires aligned endpoints or event roles, normalized predicate, direction, essential story/validity scope, applicable epistemic status, and a valid supporting evidence reference. The primary semantic endpoint is macro strict assertion F1. Elaborate relaxed hierarchies, generalized graph-edit-distance cost panels, confidence calibration suites, and many overlapping temporal variants are deferred.

Ontology-decision macro F1 is the unweighted mean of `merge_split`, `contextual_type`, `schema_relation`, `event_reification`, `abstraction`, and `temporal_epistemic_qualification` over the union of gold-active and prediction-active component families. Both empty is `NA`; exactly one empty is 0; otherwise F1 is computed. Thus a spurious activation cannot disappear merely because its gold family is empty. Every component and its denominator remain visible so a favorable macro cannot hide the absence of construction operators.

For a contrastive pair, signed additions, removals, substitutions, merges, splits, reifications, abstraction changes, and qualifications are aligned on the common evidence-mention basis. Decision-change F1 rewards only gold-required changes. Primary collapse occurs when the gold nonselection delta is nonempty but the predicted nonselection delta is empty, even if visible nodes differ. Paraphrase stability is interpreted conditional on correctness.

Rarity and pivotality are annotated independently before model runs. Rarity is initially one independent mention or the bottom evidence-frequency quartile. A fact is pivotal if removing it changes the gold answer, a necessary causal/temporal chain, a consequential state, a community assignment, or a registered revelation. The scorer-only labels never enter prompts. A rare-pivotal hit requires correct participants, relation, essential qualification, and supporting evidence. Report common/rare by pivotal/nonpivotal results and complete support-path survival. Frequency may be a method-visible signal but never the sole importance rule.

Every world has a positive rare-pivotal denominator pooled across its three contexts. For condition \(c\), world \(w\), and LLM seed \(r\),

\[
R^{\mathrm{RP}}_{cwr}=
\frac{\sum_{q\in w}\mathrm{TP}^{\mathrm{RP}}_{cqr}}
{\sum_{q\in w}|A^{\mathrm{RP}}_q|},
\qquad
\bar R^{\mathrm{RP}}_{cw}=\tfrac12\sum_{r=1}^{2}R^{\mathrm{RP}}_{cwr}.
\]

C0 uses its single deterministic value. The registered safeguard bound is the one-sided 95% paired-`t` lower bound over the 12 world differences, accompanied by a world-bootstrap sensitivity interval. It passes only when the lower bound for `C2-C1` is greater than \(-0.05\); the same safeguard accompanies any construction-freedom claim against `A-FixedSelect`. Failure is reported as failure of the claim gate, not converted into evidence of noninferiority by a nonsignificant harm test.

## 13. Entropy and clutter

All semantic graph measures use a named conversion. The primary topology is an unweighted undirected entity-event incidence skeleton; parallel assertion records are retained for relation counts, self-loops are excluded from topology, and isolates remain.

The reduced entropy panel contains:

1. degree-histogram entropy \(H_{\mathrm{dh}}=-\sum_k p_k\log p_k\);
2. degree-mass entropy \(H_{\mathrm{dm}}=-\sum_v d_v/(2m)\log[d_v/(2m)]\);
3. mean local relation-neighborhood entropy over incident **canonical** relation bins;
4. **canonical-mapped relation entropy**, the global assertion-frequency entropy over a frozen upper/development vocabulary plus `OTHER`; and
5. **native local-schema relation entropy**, the global assertion-frequency entropy over the projection's declared native predicate IDs before unmatched relations are collapsed into `OTHER`.

Isolates are excluded from the mean local entropy but reported separately. Native entropy is normalized by `log(number of active native types)` and canonical entropy by `log(number of frozen canonical bins including OTHER)`; a one-category distribution has normalized value 0 and an empty assertion set is `NA`. Degree entropies report their occupied-bin or positive-degree denominators explicitly. Von Neumann entropy on the unit-weight skeleton is retained only if development fixtures validate its numerical implementation and interpretation without additional GPU work; otherwise it is reported as a deferred metric before test scoring, not selected based on results. Raw values, normalized values, support sizes, vocabulary sizes, `OTHER` rates, node/edge counts, and undefined cases are reported. Entropy comparisons are two-sided secondary exploratory outcomes with no “lower is better” assumption.

Direct clutter measures are node count, assertion/edge count, density, isolates/components, edge crossings, label-overlap count/area, irrelevant visible load, and interactions needed to discover a rare-pivotal assertion. Renderer configuration, viewport, font, labels, and one layout seed are fixed.

Crossings use a two-part analysis. For each structurally valid, content-bearing projection—without filtering on achieved semantic score—let \(N\) be the number of eligible unordered nonadjacent edge pairs and \(C\) the number that cross:

1. report the opportunity indicator \(Z=\mathbf 1[N>0]\), opportunity count \(N\), and paired world-level differences in opportunity incidence/count; and
2. when \(N>0\), report \(C/N\), its numerator/denominator, the paired condition comparison on units where both members have \(N>0\), the retained pair count, discordant one-condition-only opportunity counts, and a pooled exposure-weighted sensitivity over all positive-opportunity outputs.

When \(N=0\), conditional crossing rate is `NA`, never automatically 1 or 0. A zero-opportunity graph may be appropriately simple or may have deleted necessary content; strict F1, node coverage, validity, and rare-pivotal recall decide which. Invalid or empty outputs receive semantic failure scores and no favorable clutter value.

## 14. Community analysis

Leiden with the Constant Potts Model is the only conference-minimum community algorithm. Its resolution and one algorithm seed are selected on development gold, documented, and frozen. A single limited sensitivity uses one lower and one higher resolution; Infomap, broad algorithm-seed sweeps, and alternative algorithms are deferred.

On contexts with meaningful reviewed gold communities, report cluster count and modularity descriptively, AMI, purity, unweighted mean conductance, and pairwise fragmentation and merging error. Purity is \(N^{-1}\sum_k\max_j|C_k\cap G_j|\) on the fixed gold-anchor universe and is interpreted only beside cluster count and fragmentation because singleton clusters can inflate it. Cross-construction-seed stability uses mention-aligned AMI and variation of information. Omitted vertices receive unique `ABSENT` labels; extra vertices remain visible in node precision. Zero denominators/volumes are `NA` with counts. A small condition-blind evidence-based rubric inspects semantic coherence and interpretability without becoming a confirmatory endpoint.

## 15. Fairness, stochasticity, and experimental integrity

For each primary unit, C0/C1/C2 receive the same evidence snapshot, packet, horizon, upper ontology, final node/assertion/display budgets, validation rules, and scorer. C1/C2 share one pinned model stack, maximum input/output limits, structured-output policy, and two seed blocks. Different prompts/grammars are allowed only where construction timing and capability require them and are enumerated in a grammar-difference manifest. C2 cannot receive a larger packet, output budget, repair allowance, or adaptive retrieval.

C1 is allowed a credible comprehensive preconstruction budget, but its cost is counted. A sealed C1 artifact is reused across the three contexts for its world/seed. C0 competence is reviewed on development data and cannot be intentionally weakened. `A-FixedSelect` packing must include the complete required sealed graph and packet without silent truncation.

Construction timing creates an unavoidable compute asymmetry: per seed and world C1 makes one comprehensive prebuild call amortized across three contexts, whereas C2 makes three context-specific calls. Per-call model, decoding, context cap, structured-output rules, repair eligibility, and final per-projection object/display budgets are matched, but total generation opportunities are not identical. The ledger therefore reports prompt/completion tokens and allocated GPU seconds as construction total per world, amortized per served context, first-query workload, and complete three-query workload; `A-FixedSelect` reports both its inherited C1 prebuild and query-time cost. No claim treats C2--C1 as a compute-efficiency comparison.

Two paired seed blocks measure model variability without pretending to create 24 worlds. C0 is deterministic and is not copied as independent output. A smaller paraphrase subset and component-level error analysis replace a broad prompt/model matrix. Results therefore apply to the pinned model and prompt only.

Scorer gold is process- and path-isolated. Model-visible context excludes split, world, contrast, expected-effect, and gold IDs. C0/C1 lineage certificates, C2 post-query construction certificates, packet equality, query-reveal order, cache namespaces, budget equality, and label/evidence support are blocking audits. Invalid, refused, timed-out, and unrepaired outputs remain in intention-to-treat semantic results.

## 16. GPU and storage feasibility

### 16.1 Exact scheduled GPU inventory

The implementation plan owns execution details; this table fixes the scientific call counts. A “call” is an attempted generation, successful or not. Eight separately timed model-load/allocation events are also included.

| Call class | Count |
|---|---:|
| Acceptance and development generations, including two repair probes | 32 |
| Held-out C1 synthetic preontologies: 12 worlds × 2 seeds | 24 |
| Held-out C2 primary: 36 contexts × 2 seeds | 72 |
| `A-FixedSelect`: 36 contexts × 2 seeds | 72 |
| C2 paraphrase subset: 12 contexts × 1 seed | 12 |
| C2 scripted feedback regenerations | 6 |
| C2 researcher-interface trace regenerations | 3 |
| `A-NoContext`: 12 contexts × 1 seed | 12 |
| `A-NoTemporalEpistemic`: 8 eligible contexts × 1 seed | 8 |
| `A-NoRareGuard`: 8 rare-pivotal contexts × 1 seed | 8 |
| Case C1 window preontologies | 4 |
| Case C2 bounded-window queries | 8 |
| Complete-index operational C2 query | 1 |
| Global bounded fallback/repair/failure/timeout/rerun attempts | 16 |
| **Maximum scheduled inference attempts** | **278** |
| Separately metered model-load/allocation events | **8** |

The manifest computes

\[
\texttt{planned\_gpu\_hours}
=\frac{\sum_c N_c\,p95_c}{3600},
\]

where \(p95_c\) is measured allocated service-wall seconds for class \(c\), and model-load/allocation is a class. The 16 reserve slots are divided into four long calls capped at 240 seconds, eight standard calls capped at 150 seconds, and four short calls capped at 90 seconds. They cover a bounded fallback micro-pilot as well as scientific repairs or retries; they cannot be borrowed across timeout classes. The provisional admission ceilings yield exactly 31,229 seconds (8.6747 hours). The activity rows below are rounded upper allocations; the call-level manifest governs admission.

| Activity | Maximum scheduled GPU time |
|---|---:|
| Runtime acceptance and development | 1.25 h |
| C1 synthetic preconstruction | 1.25 h |
| C2 primary synthetic construction | 1.95 h |
| `A-FixedSelect` | 1.49 h |
| Paraphrase, feedback, interface traces, and three limited ablations | 1.35 h |
| Bounded first-novel case study | 0.70 h |
| Bounded fallback, repairs, failures, timeouts, and reruns | 0.70 h |
| Unallocated scheduled admission slack | 0.31 h |
| **Scheduled ceiling** | **9.00 h** |
| Unallocated hard contingency | 1.00 h |
| **Absolute maximum** | **10.00 h** |

The eight load events reconcile as three within acceptance/development and one each within C1, C2, `A-FixedSelect`, the combined paraphrase/feedback/ablation block, and the case-study block. The runner refuses the main run if the p95 forecast exceeds 9 hours. Actual allocated GPU time, including load, failed attempts, timeouts, repair calls, and any fallback pilot, decrements a monotonic hard counter. Optional work stops first. No scope change may remove C0/C1/C2, `A-FixedSelect`, contrastive queries, temporal representation, or rare-pivotal evaluation. The process stops before 10 hours even if outputs remain incomplete.

### 16.2 Complete writable-storage allocation

The cap includes every project-controlled writable byte on the GPU machine.

| Category | Peak allocation |
|---|---:|
| One pinned quantized model snapshot and tokenizer | 11 GB |
| Locked environment and project-controlled runtime/download cache | 5 GB |
| Source, synthetic data, restricted first novel, FTS index | 2 GB |
| Compressed raw generations and repair/validation traces | 2 GB |
| SQLite ledger, metrics, schemas, manifests, and graph artifacts | 1 GB |
| Selected UI assets, visual fixtures, screenshots, and paper figures | 1 GB |
| Regenerable temporary working space | 3 GB |
| Required free headroom | 5 GB |
| **Peak writable allocation** | **30 GB** |

Occupied storage targets at most 25 GB at every phase, leaving 5 GB free. One shared Hugging Face/Transformers/vLLM cache holds only the pinned snapshot; download and package caches are pruned after verification. Evidence packets and C1 ontologies are content-addressed and referenced rather than copied into each run. Raw output is zstd-compressed JSONL; compact tables use Parquet only when smaller. No logits, attention, hidden states, KV cache dumps, reconstructive embeddings, or screenshot banks are stored. A preflight runs before download and each phase and pauses if projected or actual free space would fall below 5 GB. Every project-controlled path, mount, cache, staging area, and same-machine archive counts; transferring a verified artifact to genuinely external durable storage is reported separately and does not authorize a second local copy. A base immutable OS/container image may be reported outside this table only if it is genuinely outside project-controlled writable storage.

### 16.3 Seven-phase execution envelope

The implementation plan owns deliverables and accounting, but the scientific order is fixed: (1) contracts, compact schemas, and GPU acceptance pilot; (2) four development and 12 test worlds plus independent audit; (3) C0/C1/C2/`A-FixedSelect`, development calibration, and primary runs; (4) essential metrics and limited ablations; (5) minimal visualization and scripted feedback; (6) bounded first-novel case study; and (7) analysis and conference artifacts. The active-researcher estimate is 160 hours within a 120--180-hour, three-to-five-month target. This is a seven-phase conference study, not the superseded nine-phase program.

## 17. Statistical analysis

The 12 worlds are the independent units. For a primary metric \(Y\), condition \(c\), and world \(w\), the frozen estimand is

\[
\bar Y_{cw}=\frac13\sum_{q\in w}
\left(\frac12\sum_{r=1}^{2}Y_{cq r}\right),
\qquad
d_w=\bar Y_{C2,w}-\bar Y_{C1,w},
\qquad
\widehat\Delta=\frac1{12}\sum_w d_w .
\]

C0 uses its one deterministic value rather than duplicating it across seeds. Invalid, timed-out, or unrepaired output with nonempty gold scores 0 on the two primary endpoints and rare recall; mathematically undefined structural metrics are `NA`. The point estimate is the paired world-mean difference. Also report all 12 differences, the paired median, small-sample corrected standardized paired effect, a two-sided 95% paired-`t` interval with 11 degrees of freedom, and a registered 10,000-resample world-bootstrap sensitivity interval with explicit small-sample caution.

The confirmatory family contains exactly two C2--C1 one-sided paired-`t` superiority tests: strict qualified-assertion F1 and ontology-decision macro F1. Holm controls these two at \(\alpha=.05\). Exact sign-flip tests over all \(2^{12}=4096\) world-level sign assignments are registered sensitivity analyses, with their symmetry/exchangeability assumption stated rather than selected after seeing results. The required C2--C0 contrasts are secondary estimates for the same endpoints. Only if the organization endpoint passes its C2--C1 gate is C2 versus `A-FixedSelect` tested on ontology-decision macro F1 as the mechanistic gate; that result is required to attribute a gain to construction freedom. Rare-pivotal noninferiority is a separate claim safeguard, not superiority evidence.

With only 12 clusters, asymptotic mixed-effects claims are avoided. A secondary random-intercept model may be reported only if it converges stably and agrees in sign with the paired estimate; it cannot replace world-level inference. Contrastive change, collapse, paraphrase, entropy, clutter, community, and ablation metrics are required secondary reports. The entropy family uses two-sided Benjamini--Hochberg adjustment if inferential \(p\)-values are shown; otherwise secondary panels emphasize paired estimates and intervals.

Pilot power simulation uses development variance only. The registered conference design remains 12 test worlds and the fixed call manifest. If precision is inadequate, the study reduces claim strength rather than adding post hoc worlds, contexts, or seeds or presenting repeated measures as independent power.

Case-study outcomes are descriptive: per-unit metrics, evidence audits, condition differences, costs, and selected qualitative traces. No combined synthetic-plus-case population estimate is calculated. Null and negative findings remain valid.

## 18. Reduced ablations

The mandatory conference set is:

| Ablation | Scope | Purpose |
|---|---:|---|
| `A-FixedSelect` | 36 contexts × 2 seeds | construction freedom versus query-time selection |
| `A-NoContext` | balanced 12 × 1 seed | use of structured lens/context |
| `A-NoTemporalEpistemic` | eligible 8 × 1 seed | value of qualified representation |
| `A-NoRareGuard` | rare-pivotal 8 × 1 seed | whether simplification deletes pivotal evidence |

Subsets and configuration deltas are frozen before held-out outputs. `A-NoContext` selects one context per world, balanced across difficulty and lens, and replaces wording, lens, target, story scope, viewpoint, and abstraction with one generic construction request; only the identical packet, hidden horizon enforcement, upper ontology, and object/display budgets remain. `A-NoTemporalEpistemic` selects eight eligible contexts across difficulty and removes temporal/epistemic prompt capabilities and output fields; the unchanged full gold treats required missing qualifications as misses. `A-NoRareGuard` selects eight rare-pivotal contexts from eight distinct worlds and removes only the gold-blind C2 instruction/check to inspect low-frequency evidence for answer necessity, state change, or causal reach. It does not remove evidence or reveal scorer labels. Each uses the same model, packet, seed, decoding, output budget, and repair rule as its parent C2 run, carries one config-delta hash, and is checked by a one-switch-only test. Principal diagnostics are decision F1 for `A-NoContext`, essential temporal qualification accuracy for `A-NoTemporalEpistemic`, and rare-pivotal recall for `A-NoRareGuard`.

`A-ShuffledContext`, `A-NoFeedback`, `A-WeakGrounding`, `A-FixedSchema`, `A-NoMergeSplit`, `A-NoEventReify`, `A-FullEvidence`, `A-Prompt`, `A-ModelSize`, a second model, and thinking mode are deferred. One prompt revision is frozen after development. Prompt and model dependence are validity limitations. Optional calls are not added to the frozen conference run merely because an early phase runs quickly.

## 19. Error analysis, validity, ethics, and reproducibility

The compact error taxonomy covers evidence/retrieval omission, over/under-merge, wrong schema/relation, event-boundary or role error, story/discourse/revelation confusion, epistemic holder or commitment error, causal overclaim, unsupported description, wrong abstraction, rare-pivotal loss, contrastive collapse, paraphrase instability, invalid structure, and renderer hiding. Every held-out failure receives one or more codes; examples are selected without favoring C2.

Principal validity threats and mitigations are:

- **generator bias:** representation-neutral worlds, held-out templates, pregraph-friendly cases, alternatives, mutation tests, and mandatory independent review;
- **small independent sample:** world-level paired analysis, exact tests, intervals, no population claim;
- **one model/prompt:** symmetric pinning and explicitly model-specific conclusions;
- **retrieval confounding:** all admissible evidence in primary snapshots; full-index retrieval labeled operational only;
- **weak baselines:** a frozen development competence protocol and public rules/configuration;
- **C2 selection masquerade:** construction certificate, contrastive decision metrics, lineage audit, and `A-FixedSelect`;
- **metric opportunism:** two primary endpoints, one rare safeguard, frozen secondary panel, and all outcomes reported;
- **copyright:** one local source, no copied packets/prose/embeddings in public artifacts, release allowlist and scan;
- **visual proxy error:** structural metrics do not establish usability; and
- **resource attrition:** preflight budgets, resumable attempts, explicit missing-output accounting, and hard stops.

All runs record code revision, dirty patch if any, environment, model/tokenizer/AWQ/runtime/chat-template/prompt/schema revisions, full decoding parameters, hardware, seeds, packet hashes, outputs, repairs, failures, timings, memory, ontology certificates, metrics, and feedback. Synthetic data, gold, prompts, code, aggregate outputs, and safe figures form the public reproduction bundle, targeted below 2 GB without weights or copyrighted text.

Human participant research is absent. The researcher traces contain no participant data. If participant work is later added, it requires a new ethics determination and protocol. The system does not claim authoritative fictional-world truth; contested evidence remains qualified.

## 20. Acceptance criteria, claims, and open decisions

Held-out execution begins only after:

1. the four development worlds, 12 sealed test worlds, 36 contexts, contrast proofs, gold alternatives, factor quotas, and all nine projections in the mandatory three-world independent review are completed, adjudicated, and frozen;
2. C0/C1/C2 and `A-FixedSelect` pass evidence equality, timing, lineage/certificate, packing, and gold-firewall tests;
3. the active model demonstrates merge/split, schema/relation, event, abstraction, temporal/epistemic, and supported-description decisions;
4. C0 exercises every registered explicit entity, binary relation, event-role, story-time, and validity fixture family on development data, attains at least 0.85 precision and 0.70 recall on directly stated qualified assertions, and has valid evidence for every assertion; these thresholds are frozen before test reveal and do not require C0 to lose;
5. query-blind C1 is schema-valid, exercises every construction operator, has 100% valid evidence IDs, at least 0.95 grounding precision, and at least 0.75 recall against the union of development contexts' supported gold assertions after its preontology is sealed;
6. evidence-ID validity is 100%, development grounding precision is at least 0.95, and programmed horizon leakage is zero;
7. rare-pivotal test fixtures cannot be deleted to improve sparsity;
8. the two-part crossing metric, native/canonical relation entropy, Leiden metrics, and missing/empty cases pass known fixtures;
9. the p50/p95 pilot produces a manifest forecast no greater than 9 hours;
10. projected occupied storage is no greater than 25 GB with 5 GB free;
11. revision resolution and C2 regeneration replay exactly from manifests; and
12. the release dry run contains no protected prose or reconstructive artifacts.

The first-novel phase begins only after the synthetic run closes, timing/gold/firewall audits pass, registered metrics regenerate from immutable artifacts, and the blinded error review is complete. C2 need not win to proceed; an integrity failure blocks transfer until corrected within the frozen protocol or reported as a failed study.

If supported, the paper may claim bounded improvements on the registered synthetic benchmark under one pinned model and prompt; evidence consistent with a construction-timing mechanism; specified changes in entropy, clutter, or community organization subject to fidelity; an auditable interaction mechanism; and descriptive transfer to eight first-novel cases. It may not claim general LLM superiority, population-level narrative generalization, human usability/cognition/trust benefits, complete novel truth, book-series coverage, or inherent superiority from fewer nodes, lower entropy, fewer crossings, or a particular cluster count.

The remaining decisions are evidence-gated rather than unspecified:

| Decision | Recommended default | Alternative and tradeoff | Evidence required to change |
|---|---|---|---|
| Model | pinned Qwen3-14B-AWQ | one pinned 7--8B AWQ model; faster/smaller but potentially weaker ontology decisions | Phase-1 VRAM, RAM, packing, latency, or structured-output failure; change all LLM conditions symmetrically |
| Sequence budget | 12,288 total: 10,240 input + 2,048 output; repair 10,752 + 1,536 | lower common total/window; less evidence or output capacity | no-truncation packing and p95 admission test |
| Community setting | one development-frozen Leiden-CPM resolution | another single resolution; different granularity | development gold AMI/conductance, fixed before test |
| Rare loss margin | absolute 0.05 | tighter prespecified margin; harder safeguard | substantive development analysis, never held-out outcomes |
| Von Neumann entropy | include only after validated CPU fixture | omit; smaller entropy panel | numerical/interpretive validation before test scoring |

None of these choices may remove the non-negotiable contribution.

## 21. Verified references

1. **[R1]** Alexander Maedche and Steffen Staab. “Ontology Learning for the Semantic Web.” *IEEE Intelligent Systems* 16(2):72--79, 2001. [doi:10.1109/5254.920602](https://doi.org/10.1109/5254.920602).
2. **[R2]** Philipp Cimiano and Johanna Völker. “Text2Onto: A Framework for Ontology Learning and Data-Driven Change Discovery.” *NLDB 2005*, LNCS 3513, pp. 227--238. [doi:10.1007/11428817_21](https://doi.org/10.1007/11428817_21).
3. **[R3]** Wilson Wong, Wei Liu, and Mohammed Bennamoun. “Ontology Learning from Text: A Look Back and into the Future.” *ACM Computing Surveys* 44(4), Article 20, 2012. [doi:10.1145/2333112.2333115](https://doi.org/10.1145/2333112.2333115).
4. **[R4]** Paolo Bouquet, Fausto Giunchiglia, Frank van Harmelen, Luciano Serafini, and Heiner Stuckenschmidt. “Contextualizing Ontologies.” *Journal of Web Semantics* 1(4):325--343, 2004. [doi:10.1016/j.websem.2004.07.001](https://doi.org/10.1016/j.websem.2004.07.001).
5. **[R5]** Nesrine Ben Mustapha, Marie-Aude Aufaure, Hajer Baazaoui Zghal, and Henda Ben Ghézala. “Query-driven approach of contextual ontology module learning using web snippets.” *Journal of Intelligent Information Systems* 45(1):61--94, 2015. [doi:10.1007/s10844-013-0263-6](https://doi.org/10.1007/s10844-013-0263-6).
6. **[R6]** James F. Allen. “Maintaining Knowledge about Temporal Intervals.” *Communications of the ACM* 26(11):832--843, 1983. [doi:10.1145/182.358434](https://doi.org/10.1145/182.358434).
7. **[R7]** Claudio Gutierrez, Carlos A. Hurtado, and Alejandro A. Vaisman. “Introducing Time into RDF.” *IEEE Transactions on Knowledge and Data Engineering* 19(2):207--218, 2007. [doi:10.1109/TKDE.2007.34](https://doi.org/10.1109/TKDE.2007.34).
8. **[R8]** Simon Cox and Chris Little, editors. “Time Ontology in OWL.” W3C Recommendation, 2017. [Official Recommendation](https://www.w3.org/TR/2017/REC-owl-time-20171019/).
9. **[R9]** José M. Giménez-García, Antoine Zimmermann, and Pierre Maret. “NdFluents: An Ontology for Annotated Statements with Inference Preservation.” *The Semantic Web -- 14th International Conference, ESWC 2017*, pp. 638--654. [doi:10.1007/978-3-319-58068-5_39](https://doi.org/10.1007/978-3-319-58068-5_39).
10. **[R10]** Willem Robert van Hage, Véronique Malaisé, Roxane Segers, Laura Hollink, and Guus Schreiber. “Design and use of the Simple Event Model (SEM).” *Journal of Web Semantics* 9(2):128--136, 2011. [doi:10.1016/j.websem.2011.03.003](https://doi.org/10.1016/j.websem.2011.03.003).
11. **[R11]** Simon Gottschalk and Elena Demidova. “EventKG: A Multilingual Event-Centric Temporal Knowledge Graph.” *The Semantic Web -- 15th International Conference, ESWC 2018*, pp. 272--287. [doi:10.1007/978-3-319-93417-4_18](https://doi.org/10.1007/978-3-319-93417-4_18).
12. **[R12]** Marco Rospocher, Marieke van Erp, Piek Vossen, Antske Fokkens, Itziar Aldabe, German Rigau, Aitor Soroa, Thomas Ploeger, and Tessel Bogaard. “Building Event-Centric Knowledge Graphs from News.” *Journal of Web Semantics* 37--38:132--151, 2016. [doi:10.1016/j.websem.2015.12.004](https://doi.org/10.1016/j.websem.2015.12.004).
13. **[R13]** Carlo Meghini, Valentina Bartalesi, and Daniele Metilli. “Representing Narratives in Digital Libraries: The Narrative Ontology.” *Semantic Web* 12(2):241--264, 2021. [doi:10.3233/SW-200421](https://doi.org/10.3233/SW-200421).
14. **[R14]** Jeremy J. Carroll, Christian Bizer, Pat Hayes, and Patrick Stickler. “Named Graphs, Provenance and Trust.” *Proceedings of the 14th International Conference on World Wide Web*, pp. 613--622, 2005. [doi:10.1145/1060745.1060835](https://doi.org/10.1145/1060745.1060835).
15. **[R15]** Timothy Lebo, Satya Sahoo, and Deborah McGuinness, editors. “PROV-O: The PROV Ontology.” W3C Recommendation, 2013. [Official Recommendation](https://www.w3.org/TR/2013/REC-prov-o-20130430/).
16. **[R16]** Antske Fokkens, Piek Vossen, Marco Rospocher, Rinke Hoekstra, and Willem Robert van Hage. “GRaSP: Grounded Representation and Source Perspective.” *Proceedings of the Workshop Knowledge Resources for the Socio-Economic Sciences and Humanities associated with RANLP 2017*, pp. 19--25. [doi:10.26615/978-954-452-040-3_003](https://doi.org/10.26615/978-954-452-040-3_003).
17. **[R17]** Blaž Fortuna, Marko Grobelnik, and Dunja Mladenić. “OntoGen: Semi-automatic Ontology Editor.” *Human Interface and the Management of Information: Interacting in Information Environments (HCII 2007)*, LNCS 4558, pp. 309--318. [doi:10.1007/978-3-540-73354-6_34](https://doi.org/10.1007/978-3-540-73354-6_34).
18. **[R18]** Tania Tudorache, Csongor Nyulas, Natalya F. Noy, and Mark A. Musen. “WebProtégé: A Collaborative Ontology Editor and Knowledge Acquisition Tool for the Web.” *Semantic Web* 4(1):89--99, 2013. [doi:10.3233/SW-2012-0057](https://doi.org/10.3233/SW-2012-0057).
19. **[R19]** Saleema Amershi, Maya Cakmak, W. Bradley Knox, and Todd Kulesza. “Power to the People: The Role of Humans in Interactive Machine Learning.” *AI Magazine* 35(4):105--120, 2014. [doi:10.1609/aimag.v35i4.2513](https://doi.org/10.1609/aimag.v35i4.2513).
20. **[R20]** Hamed Babaei Giglou, Jennifer D'Souza, and Sören Auer. “LLMs4OL: Large Language Models for Ontology Learning.” *The Semantic Web -- ISWC 2023*, pp. 408--427. [doi:10.1007/978-3-031-47240-4_22](https://doi.org/10.1007/978-3-031-47240-4_22).
21. **[R21]** J. Harry Caufield, Harshad Hegde, Vincent Emonet, Nomi L. Harris, Marcin P. Joachimiak, Nicolas Matentzoglu, HyeongSik Kim, Sierra A. T. Moxon, Justin T. Reese, Melissa A. Haendel, Peter N. Robinson, and Christopher J. Mungall. “Structured Prompt Interrogation and Recursive Extraction of Semantics (SPIRES): a method for populating knowledge bases using zero-shot learning.” *Bioinformatics* 40(3):btae104, 2024. [doi:10.1093/bioinformatics/btae104](https://doi.org/10.1093/bioinformatics/btae104).
22. **[R22]** Hanzhu Chen, Xu Shen, Qitan Lv, Jie Wang, Xiaoqi Ni, and Jieping Ye. “SAC-KG: Exploiting Large Language Models as Skilled Automatic Constructors for Domain Knowledge Graph.” *Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)*, pp. 4345--4360, 2024. [doi:10.18653/v1/2024.acl-long.238](https://doi.org/10.18653/v1/2024.acl-long.238).
23. **[R23]** Patrick Lewis, Ethan Perez, Aleksandra Piktus, Fabio Petroni, Vladimir Karpukhin, Naman Goyal, Heinrich Küttler, Mike Lewis, Wen-tau Yih, Tim Rocktäschel, Sebastian Riedel, and Douwe Kiela. “Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks.” *Advances in Neural Information Processing Systems 33*, 2020. [Official record](https://papers.neurips.cc/paper/2020/hash/6b493230205f780e1bc26945df7481e5-Abstract.html).
24. **[R24]** Darren Edge, Ha Trinh, Newman Cheng, Joshua Bradley, Alex Chao, Apurva Mody, Steven Truitt, Dasha Metropolitansky, Robert Osazuwa Ness, and Jonathan Larson. “From Local to Global: A Graph RAG Approach to Query-Focused Summarization.” arXiv:2404.16130, 2024, revised 2025 (preprint). [doi:10.48550/arXiv.2404.16130](https://doi.org/10.48550/arXiv.2404.16130).
25. **[R25]** Xiaohui Zhang, Zequn Sun, Chengyuan Yang, Yuanning Cui, Lingbing Guo, and Wei Hu. “Toward Effective and Reliable LLM Agents via Dynamic Ontology.” arXiv:2608.22974, 2026 (preprint). [doi:10.48550/arXiv.2608.22974](https://doi.org/10.48550/arXiv.2608.22974).
26. **[R26]** Fabian Beck, Michael Burch, Stephan Diehl, and Daniel Weiskopf. “A Taxonomy and Survey of Dynamic Graph Visualization.” *Computer Graphics Forum* 36(1):133--159, 2017. [doi:10.1111/cgf.12791](https://doi.org/10.1111/cgf.12791).
27. **[R27]** Vitalis Wiens, Steffen Lohmann, and Sören Auer. “Semantic Zooming for Ontology Graph Visualizations.” *Proceedings of the Knowledge Capture Conference (K-CAP 2017)*, Article 4, pp. 1--8. [doi:10.1145/3148011.3148015](https://doi.org/10.1145/3148011.3148015).
28. **[R28]** Helen C. Purchase. “Metrics for Graph Drawing Aesthetics.” *Journal of Visual Languages & Computing* 13(5):501--516, 2002. [doi:10.1006/jvlc.2002.0232](https://doi.org/10.1006/jvlc.2002.0232).
29. **[R29]** Matthias Dehmer and Abbe Mowshowitz. “A History of Graph Entropy Measures.” *Information Sciences* 181(1):57--78, 2011. [doi:10.1016/j.ins.2010.08.041](https://doi.org/10.1016/j.ins.2010.08.041).
30. **[R30]** Kartik Anand and Ginestra Bianconi. “Entropy Measures for Networks: Toward an Information Theory of Complex Topologies.” *Physical Review E* 80:045102(R), 2009. [doi:10.1103/PhysRevE.80.045102](https://doi.org/10.1103/PhysRevE.80.045102).
31. **[R31]** Vincent A. Traag, Ludo Waltman, and Nees Jan van Eck. “From Louvain to Leiden: Guaranteeing Well-Connected Communities.” *Scientific Reports* 9:5233, 2019. [doi:10.1038/s41598-019-41695-z](https://doi.org/10.1038/s41598-019-41695-z).
32. **[R32]** Nguyen Xuan Vinh, Julien Epps, and James Bailey. “Information Theoretic Measures for Clusterings Comparison: Variants, Properties, Normalization and Correction for Chance.” *Journal of Machine Learning Research* 11:2837--2854, 2010. [Official record](https://www.jmlr.org/papers/v11/vinh10a.html).
33. **[R33]** Emden R. Gansner, Yehuda Koren, and Stephen C. North. “Topological Fisheye Views for Visualizing Large Graphs.” *IEEE Transactions on Visualization and Computer Graphics* 11(4):457--468, 2005. [doi:10.1109/TVCG.2005.66](https://doi.org/10.1109/TVCG.2005.66).
34. **[R34]** M. E. J. Newman and M. Girvan. “Finding and Evaluating Community Structure in Networks.” *Physical Review E* 69:026113, 2004. [doi:10.1103/PhysRevE.69.026113](https://doi.org/10.1103/PhysRevE.69.026113).
35. **[R35]** Santo Fortunato and Marc Barthélemy. “Resolution Limit in Community Detection.” *Proceedings of the National Academy of Sciences* 104(1):36--41, 2007. [doi:10.1073/pnas.0605965104](https://doi.org/10.1073/pnas.0605965104).
