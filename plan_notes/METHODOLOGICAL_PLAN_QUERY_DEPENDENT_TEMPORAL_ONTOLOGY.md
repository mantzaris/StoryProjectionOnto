# Methodological Plan: Active Query-Dependent Temporal Ontology Construction

**Document status:** authoritative research-method plan for the proposed study.  
**Companion document:** `IMPLEMENTATION_PLAN_QUERY_DEPENDENT_TEMPORAL_ONTOLOGY.md`.  
**Planning date:** 2026-09-03.

## Scope correction and relationship to earlier plans

Earlier plans in this repository are retained as historical material, but they do not specify this study. They progressively narrowed the project to a deterministic, CPU-only optimization over a complete oracle graph: the graph was fixed before a query, a query selected or pruned nodes and edges, presentation was static or cosmetic, and an LLM, GPU inference, narrative-text processing, end-user refinement, and an interactive graph were excluded or deferred. That design can still inform deterministic validation, provenance discipline, and one family of baselines. It cannot answer the present research questions because it makes the proposed treatment a fixed-graph projection problem.

This plan corrects that drift. The minimum publishable study requires an active LLM on an RTX 4090-class GPU to perform substantive semantic construction after a query and context are known. It also requires local narrative ingestion, temporal and conditional epistemic representation, structured human refinement, and interactive graph presentation. A CPU component remains important for indexing, validation, constraint checking, graph computation, and the classical baseline, but it complements rather than replaces query-time LLM construction.

## 1. Executive summary

The project asks whether a useful ontology for complex narrative evidence should be fully constructed before a user's information need is known, or instead constructed conditionally after the query, temporal horizon, perspective, and desired abstraction are supplied. The motivating problem is that a comprehensive narrative graph can be so dense that it obscures the relationships that matter. Ordinary pruning may reduce density, but frequency-based pruning can erase a unique yet causally pivotal event. The alternative studied here is not merely a better filter: it is a query-conditioned act of ontology engineering.

The study separates a neutral, pre-query **EvidenceIndex** from a constructed ontology. The index contains passages, locations, mentions, alias and coreference candidates, temporal clues, embeddings, provenance, and uncertain candidate semantic material. It does not contain a sealed global entity partition, canonical event graph, or final relation vocabulary that C2 can secretly retrieve. After a structured **QueryContext** is known, the active LLM in C2 decides which mentions denote the same contextual entity, which distinctions must remain separate, which entities and events matter, which events should be reified, what abstraction and relation types are appropriate, what temporal or epistemic qualifications apply, and how rich nodes and assertions should be described. Deterministic software validates evidence references, temporal constraints, budgets, schemas, and provenance.

Three conditions isolate the causal contrast:

- **C0 ClassicalPre:** a credible CPU pipeline constructs a conventional ontology before test queries are known; query time permits retrieval, pruning, and rendering only.
- **C1 LLMPre:** the same LLM class used by C2 constructs a canonical ontology before test queries are known; query time permits selection, ranking, faithful relabeling, and rendering only.
- **C2 LLMQuery:** the evidence index exists before the query, but the GPU-hosted active LLM constructs and may reconstruct the ontology after the query and structured human revisions are known.

The central comparison is C2 versus C1, because both use an LLM and differ principally in construction timing. C2 versus C0 measures improvement over a classical approach. A fixed-ontology, query-time LLM selection ablation prevents an apparent gain from being attributed merely to a better graph selector.

The first controlled benchmark uses small synthetic narrative worlds with 10--20 gold nodes per contextual scenario, multiple contexts over the same evidence, known temporal and epistemic answers, rare-pivotal facts, and permissible alternative ontologies. The benchmark precedes a locally ingested *A Song of Ice and Fire* / Game of Thrones books case study, for which copyrighted text is never redistributed. Evaluation combines semantic alignment, evidence grounding, rare-pivotal recall, a preregistered panel of entropy measures, direct visual-clutter measures, community quality, stochastic stability, scripted feedback recovery, and later expert assessment. Lower entropy, fewer nodes, or fewer communities are never sufficient definitions of success.

The minimum publishable contribution includes all three conditions, active query-time GPU inference, temporal ontology construction, the known-answer synthetic benchmark, contrastive contextual projections, automated analysis, a working interactive refinement interface, and a bounded narrative case study. Optional extensions may deepen nested belief logic, add models or corpora, or support a powered usability trial; they may not displace these central components.

## 2. Precise problem statement

Let a corpus provide descriptions of people, collectives, places, objects, events, states, claims, and changes. A single corpus-level graph must simultaneously accommodate many possible questions: kinship, allegiance, causal responsibility, knowledge, movement, conflict, or the consequences of an event at a particular story or spoiler horizon. A representation optimized for one lens can be misleading for another. For example, two aliases may be safely merged for a geographic-movement view but need separate mention or persona nodes for a mistaken-identity question; a battle may be a binary `opposed` edge in one view but an event with participants, causes, and consequences in another.

The target problem is to construct a small, rich, evidence-grounded temporal ontology whose **ontological organization** is appropriate to a specified context. Organization includes the entity partition, contextual typing, abstraction level, event boundaries and reification choices, relation vocabulary, temporal and epistemic qualification, salient rare facts, community organization, and contextual descriptions. It does not mean only choosing a subset of already canonical nodes.

The empirical question is whether construction after context is known improves this organization relative to (a) a credible classical graph constructed beforehand and (b) a graph constructed beforehand by an LLM. The treatment succeeds only if it improves alignment and/or organization while maintaining factual support and rare-pivotal information, and if any visual simplification is not achieved by producing an empty or impoverished graph.

The study is explicitly not:

- ordinary passage retrieval, GraphRAG, or question answering over a fixed graph;
- a test of whether an LLM can generate attractive labels;
- a chatbot wrapped around a precomputed ontology;
- deterministic graph pruning presented as ontology induction;
- an attempt to redistribute or reconstruct copyrighted novels; or
- a claim that one entropy value, cluster count, or drawing aesthetic measures usability.

## 3. Definitions and notation

### 3.1 Objects

| Symbol or term | Definition |
|---|---|
| \(X\) | Raw, legally available narrative documents. For the case study these are locally supplied and never released. |
| \(I\) | Query-blind indexing and evidence-preparation procedure. |
| \(D_z=I(X_{\le z})\) | Versioned, query-blind **EvidenceSnapshot** at registered discourse/spoiler horizon \(z\), containing only passages/evidence whose discourse coordinate \(\delta(e)\le z\): documents, spans, mentions, candidates, clues, embeddings, provenance, and uncertainty, but no complete hidden ontology for C2. \(D\) denotes the family of snapshots. For a scored unit the matching snapshot uses \(z=\sigma_q\). |
| \(M_z=\operatorname{Manifest}(D_z)\) | Immutable snapshot metadata, hashes, schema/revision, and access handle; it is not unrestricted evidence content. |
| \(q\) | A structured **QueryContext**: natural-language wording plus lens, story-time range, discourse window, spoiler horizon \(\sigma_q\), epistemic viewpoint, abstraction preference, and output budget. These fields remain distinct and may be absent when inapplicable. |
| \(h=(r_1,\ldots,r_t)\) | Ordered, replayable history of condition-independent structured **UserRevision** events. Resolution against each condition's projection is stored separately. The empty history is \(\varnothing\). |
| \(E_q=R(D_z,q)\) | Deterministically recorded evidence packet retrieved from the matching \(D_z\), including retrieval scores and omissions. On the controlled synthetic benchmark, all admissible evidence is supplied so retrieval cannot explain the primary result. |
| \(O\) | A constructed ontology: contextual schema plus entities, events, qualified assertions, provenance, and presentation semantics. |
| \(A(O)\) | An ontology-decision signature: relevant entities/events/assertions, entity equivalence partition, contextual types, relation definitions, event-reification decisions, abstraction choices, temporal/epistemic qualifications, and evidence-grounded contextual role/label semantics. Labels alone never establish non-collapse. |
| \(V\) | A renderer-specific visual state produced from an ontology without changing its semantic content. |
| \(G(O)\) | A declared graph view of \(O\) used for a particular structural metric. Different valid reductions must be named. |

“Query” henceforth means the full structured `QueryContext`, not only its question string. “Projection” means a context-appropriate ontology output. For C0 and C1 it is a projection *from* a fixed ontology. For C2 it is a newly constructed contextual ontology and must not be interpreted as fixed-subgraph selection.

### 3.2 Construction timing

Pre-query construction followed by selection is:

\[
O_{\mathrm{pre},z} = F_{\mathrm{pre}}(D_z), \qquad
P_{\mathrm{pre}}(q,h)=\operatorname{Project}(O_{\mathrm{pre},z},E_q,q,h), \qquad
V_{\mathrm{pre}}=\operatorname{Render}(P_{\mathrm{pre}}).
\]

Query-dependent construction is:

\[
O_{q,h}=F_{\mathrm{query}}(M_z,E_q,q,h), \qquad
V_{q,h}=\operatorname{Render}(O_{q,h}).
\]

The decisive difference is not whether \(P_{\mathrm{pre}}\) and \(O_{q,h}\) contain different nodes. In C0/C1 every semantic atom in the output must inherit an immutable lineage to the horizon-matched \(O_{\mathrm{pre},z}\); `Project` cannot change \(A(O_{\mathrm{pre},z})\). In C2, \(A(O_{q,h})\) is produced after \(q\) and may include a context-appropriate entity partition, schema, event structure, and qualified assertions not present as ontological objects before the query, provided all claims are supported by evidence in \(E_q\subseteq D_z\). The constructor receives \(M_z\) plus the packet contents, not an unlogged read channel over \(D_z\); in the primary controlled regime \(E_q\) is all admissible evidence in the bounded snapshot. C0/C1 snapshots are sealed for every preregistered \(z\) before any lens or question wording is revealed, so future evidence cannot influence an early-horizon preontology.

### 3.3 Assertion semantics

An assertion is not treated as a timeless triple. Its conceptual form is:

\[
a=\langle s,p,o/\mathbf{r},\tau_s,\tau_v,\delta,\tau_r,\kappa,
\gamma,\Pi,\eta_q,\omega_q\rangle,
\]

where \(s,p,o\) are a subject, predicate, and object (or \(\mathbf{r}\), an n-ary event-role map); \(\tau_s\) is story/event time; \(\tau_v\) is an assertion-validity interval or partial order; \(\delta\) is discourse position; \(\tau_r\) is the proposition-level first annotated discourse position licensed to reveal the proposition; \(\kappa\) is optional holder-specific knowledge, belief, report, denial, or uncertainty; \(\gamma\) is confidence; \(\Pi\) is evidence and run provenance; \(\eta_q\) is contextual relevance; and \(\omega_q\) explains why the assertion matters in this context. An evidence span carries \(\delta(e)\), not its own proposition-level revelation timestamp. The query's spoiler horizon \(\sigma_q\) is context metadata, not an assertion timestamp: an assertion is reader-admissible only when \(\tau_r(a)\le\sigma_q\) under the registered partial-order policy. If \(\kappa\) attributes a proposition to a holder, that embedded proposition is not thereby endorsed as a world fact; world commitment requires a separate unscoped assertion, and contested/hypothetical/attributed-only commitment is recorded independently of pipeline validation status. Not every field is populated for every assertion, but the model can carry each one and every displayed assertion has evidence, temporal scope (including `unknown`), confidence, provenance, and contextual relevance.

## 4. Separating indexing, construction, projection, and rendering

The four operations are separate artifacts with separate schemas and hashes.

| Operation | Knows the test query? | Permitted output | Forbidden shortcut |
|---|---:|---|---|
| Evidence indexing \(I(X_{\le z})\) | No | horizon-specific passage/span IDs; mention and alias candidates; provisional coreference links; temporal expressions/order clues; quotations or local references; embeddings; retrieval index; uncertainty and provenance; open relation phrases as uncommitted candidates | a final global entity partition, fixed event ontology, query-specific schema, or a complete canonical ontology stored for C2 |
| Ontology construction \(F\) | C0/C1: no; C2: yes | contextual or pre-query entities, events, relation definitions, types, qualified assertions, merge/split decisions, abstraction and labels | silently importing semantic decisions from another condition |
| Fixed-ontology projection | Yes | selection, ranking, visibility, and support-preserving labels over immutable pre-query IDs | new entities, merges/splits, types, predicates, event boundaries, roles, or qualifications |
| Rendering | Yes | layout, styles, zoom levels, filters, bundles, evidence panels, and temporal display | adding factual content or mutating the ontology without a recorded revision |

### 4.1 The evidence-index boundary

Candidate material in \(D\) is deliberately defeasible. A mention detector may propose that two spans corefer, but it does not seal them as one entity. An open-information extractor may retain a surface relation phrase, but it does not establish a canonical predicate. A temporal tagger may identify “three days later,” but it does not decide which contextual assertion it qualifies. This neutral boundary is essential: preprocessing can make evidence accessible without resolving all of the ontological questions whose timing is under study.

The index schema and a pre-query artifact audit enforce the boundary. Any artifact containing globally canonical entity IDs, finalized relation semantics, fully reified events, or an answer-specific graph is classified as an ontology and cannot be an input to C2. Registered snapshots \(D_z\) are monotone evidence views, but a condition may read only the snapshot matching the scored horizon. If a useful preprocessor necessarily makes a semantic decision, that decision is declared, shared by every condition, and excluded from claims about query-time construction.

### 4.2 Operational non-collapse tests

C2 is accepted as query-dependent construction only if all of the following hold:

1. **Pre-query audit:** a machine-readable inventory proves that C2 possesses \(D\), retrieval structures, prompts, and validators before a test query, but no complete world ontology or query gold.
2. **Capability and lineage audit:** every C0/C1 output entity, predicate, event, and assertion maps to a sealed horizon-matched pre-query object; labels and summaries may paraphrase but cannot create facts. C2 emits a construction certificate, timestamped after \(q\), that enumerates its entity/event/assertion relevance, merge/split, type, abstraction, relation, event, qualification, and grounded contextual-role/label decisions.
3. **Contrastive structural test:** two contexts over identical evidence have gold differences in at least one nonselection decision in \(A(O)\). Scoring examines those decisions, not only visible node membership.
4. **Collapse-rate test:** when gold decision signatures differ, the rate at which a method produces ontologically isomorphic or decision-equivalent outputs is reported.
5. **Paraphrase test:** semantically equivalent queries should produce equivalent decision signatures within a prespecified tolerance. Sensitivity to genuine context and invariance to wording are evaluated together.
6. **Fixed-selection ablation:** `A-FixedSelect` gives the query-time LLM the complete required serialization of the C1 ontology, the same \(E_q\), and identical maximum input/output/repair caps to C2, but forbids new ontological atoms. A blocking preflight verifies that prompt + full packet + sealed graph fit without truncation; primary windows are sized against this stricter envelope. A C2 advantage over this ablation is evidence against “better selection” as the explanation.
7. **Access-log test:** signed input manifests demonstrate that C2, C1, and C0 use the same admissible evidence and spoiler horizon for each scored unit.

Failure of any boundary audit is an experimental-integrity failure, not a minor implementation bug.

## 5. Experimental conditions

### 5.1 C0: ClassicalPre

`ClassicalPre` is a credible conventional, CPU-only ontology/KG pipeline. Before test queries are revealed, it consumes the exact shared provisional candidates in each \(D_z\) and performs deterministic resolution, canonicalization, and normalization to construct a reusable horizon-specific ontology. Any C0-only extraction is classified as part of its ontology constructor, not as shared indexing. Rules support entities, events, n-ary participant roles, common temporal relations, provenance, and a documented domain-neutral upper vocabulary. Development queries and training scenarios may be used to design the pipeline, but held-out query contexts and wording are unavailable during construction.

At query time, C0 can retrieve evidence, rank and prune the sealed graph, apply a fixed layout policy, and render rich metadata already supported by its assertions. It cannot create a new relation definition, split or merge an entity, alter an event boundary, or add a temporal/epistemic distinction. This is not a bag-of-words straw man; its extraction quality, rules, and thresholds are tuned on development data, and failures are reported honestly.

### 5.2 C1: LLMPre

`LLMPre` uses the same open-weight backbone, tokenizer, quantization, structured-output mechanism, and broadly comparable prompting conventions as C2. Before held-out queries are revealed, the GPU LLM constructs a comprehensive ontology separately from every registered \(D_z\). Long inputs use query-blind, recorded chunking and consolidation. Each stochastic replicate has its own sealed \(O_{\mathrm{pre},z}^{(r)}\), and no later-horizon evidence can influence an earlier snapshot.

The preregistered primary C1 query path uses the same deterministic budgeted projector as C0 to select, rank, compress, and render objects in that sealed ontology. Its output contract rejects novel entity partitions, types, relation types, event structures, or temporal/epistemic qualifications. This condition controls for the benefit of using an LLM for construction at all. It also represents a strong design common in LLM-assisted KG construction: build once, retrieve many times. The separate `A-FixedSelect` ablation adds a query-time LLM under C2-matched orchestration while retaining this immutability.

### 5.3 C2: LLMQuery

`LLMQuery` has only the neutral horizon-matched \(D_z\) before a test query. After receiving \(q\), the same GPU LLM interprets the context and performs substantive semantic work over the one frozen \(E_q\): it selects evidence; assembles or separates contextual entities; chooses local types and abstraction; defines and instantiates relation types; reifies events when roles, causality, or time require it; assigns temporal and conditional epistemic scope; identifies potentially rare-pivotal facts from method-visible evidence signals; and generates evidence-limited contextual roles and descriptions. It cannot adaptively retrieve until correct in the initial experiment. After each structured user revision, it reconstructs \(O_{q,h}\) or an explicitly scoped part of it while retaining full version history.

The LLM is therefore a necessary causal component of C2, not an optional labeler or post-processor. Deterministic CPU routines constrain and validate its output but cannot fill in semantic objects that the LLM failed to construct. An output still invalid after the bounded repair policy counts as a failure.

### 5.4 Comparison matrix

| Property | C0 ClassicalPre | C1 LLMPre | C2 LLMQuery |
|---|---|---|---|
| Ontology constructed after test context | No | No | **Yes** |
| LLM used for construction | No | Yes, pre-query | **Yes, query-time** |
| Required accelerator | CPU | RTX 4090-class GPU | RTX 4090-class GPU |
| Entity merge/split at query time | Forbidden | Forbidden | Permitted and scored |
| New contextual schema/event structure at query time | Forbidden | Forbidden | Permitted and scored |
| Query-time temporal/epistemic qualification | only selection of existing | only selection of existing | Permitted and scored |
| Human revision | fixed-ontology reselection or explicit capability failure | fixed-ontology reselection or explicit capability failure | ontology regeneration |
| Evidence and output budgets | matched/recorded | matched/recorded | matched/recorded |

The central confirmatory contrast is C2--C1. C2--C0 is secondary but required. C1--C0 characterizes the general benefit or harm of LLM preconstruction. `A-FixedSelect` runs the **C2 query-time prompt/tool/evidence/repair pipeline** against C1's sealed ontology while disabling create, merge/split, schema, event, abstraction, and qualification operators and enforcing C1 IDs. Thus C2--`A-FixedSelect` isolates construction freedom, while `A-FixedSelect`--C1 diagnoses query-time orchestration/attention. It is a required ablation, not a fourth primary condition.

The C2--`A-FixedSelect` contrast is the one confirmatory mechanistic ablation and is excluded from the exploratory-ablation BH pool. On the registered canonical/contrastive units, C2 must show Holm-controlled one-sided superiority on ontology-decision macro F1 and contrastive decision-change F1, while strict assertion F1 and rare-pivotal recall each satisfy a -0.05 non-inferiority safeguard. Without that result, even a C2--C1 gain cannot be claimed to arise from construction freedom rather than superior query-time attention/selection. All other ablation effects are secondary exploratory unless separately labeled.

## 6. Research questions and hypotheses

### RQ1: Context alignment

**Question.** Does constructing the ontological map after receiving the user's query and context produce a more context-aligned ontology than a classical ontology and an LLM-generated ontology constructed before the query?

- **H1a (semantic alignment):** C2 will have higher strict qualified-assertion F1 than C1 and C0, with C2--C1 primary. Contextual node F1 and evidence-grounding precision are required safeguards: in the primary C2--C1 contrast their lower 95% bounds must exceed non-inferiority margins of -0.05 and -0.03 respectively, and C2 grounding precision must be at least 0.95.
- **H1b (ontological organization):** C2 will have higher six-family ontology-decision macro F1 for entity partitions, contextual types/relations and abstraction, event reification, and temporal/epistemic qualification. Every component uses a -0.10 non-inferiority margin so a macro improvement cannot conceal a large family-level regression. This hypothesis is not satisfied by changing only the displayed-node set.
- **H1c (appropriate sensitivity):** across contrastive contexts on the same evidence, C2 will improve signed ontological-decision-change F1 and reduce primary ontological-collapse rate relative to fixed ontologies while invariant-preservation F1 remains non-inferior with margin 0.05; across equivalent paraphrases, its strict-F1 change will have a lower 95% confidence bound above -0.05 and the one-sided upper 95% confidence bound for mean normalized decision-signature edit divergence will be below 0.10. All margins are frozen after development and before test.

### RQ2: Entropy and visual clutter

**Question.** Does query-dependent ontology construction change or reduce connectivity entropy and visual clutter while preserving contextually relevant, rare, and pivotal information?

- **H2a (clutter under constraints):** at matched semantic and display budgets, C2 will reduce preregistered direct overview clutter measures, particularly irrelevant visible load, edge crossings, label overlap, and path ambiguity, conditional on meeting fidelity and rare-pivotal-recall floors.
- **H2b (entropy profile):** C2 will change the multivariate entropy profile. This is a required, preregistered **secondary exploratory family**, tested two-sided with Benjamini--Hochberg control; it is not a confirmatory superiority endpoint. Even relevance-weighted or local entropy can legitimately rise or fall as irrelevant material is removed. Directional simplification claims use direct irrelevant-mass and rendering measures, not entropy as a proxy.
- **H2c (information preservation):** C2's rare-pivotal qualified-assertion recall will be non-inferior to C1 in the preregistered primary contrast and separately compared with C0 under multiplicity control, using an initially proposed absolute margin of 0.05. The margin must be frozen after the development pilot and before held-out scoring.

An empty graph is maximally bad regardless of a favorable entropy or crossing count. H2 claims require joint reporting of relevance, fidelity, coverage, rare-pivotal recall, node/edge counts, and isolates.

### RQ3: Cluster and community structure

**Question.** How does query-dependent construction affect the number, quality, and organization of graph communities?

- **H3 (contextual organization):** C2 will have higher gold adjusted mutual information (AMI) and separation and lower unweighted mean community conductance and pairwise fragmentation/merging error than C1 and C0. Internal semantic coherence and interpretability are required secondary outcomes, not unspecified confirmatory gates. Raw community count and modularity are descriptive or two-sided; neither more nor fewer clusters is inherently desirable.

For all hypotheses, the ontology/query unit, confirmatory metric family, direction, equivalence or non-inferiority margin, and exclusions are preregistered. Results may refute the hypotheses without invalidating the benchmark or case-study contribution.

| Hypothesis | Confirmatory endpoint(s) | Decision rule |
|---|---|---|
| H1a | strict qualified-assertion F1; node F1 and grounding precision are jointly required safeguards | C2--C1 one-sided strict-F1 superiority after Holm; node-F1 lower 95% bound \(>-0.05\), grounding lower bound \(>-0.03\), and C2 grounding \(\ge 0.95\); C2--C0 reported/tested second |
| H1b | six-family ontology-decision macro F1 | C2--C1 one-sided superiority; each gold-eligible component lower 95% bound \(>-0.10\); construction certificate must contain post-query nonselection decisions |
| H1c | signed decision-change F1 and primary ontological-collapse rate; invariant F1, paraphrase correctness, and normalized signature divergence safeguards | Holm-controlled superiority for change F1 and downward collapse; invariant and paraphrase-correctness lower bounds \(>-0.05\); divergence upper 95% bound \(<0.10\) |
| H2a | irrelevant-visible-load rate and failure-adjusted normalized crossing rate on the frozen eligible subset | C2--C1 one-sided downward superiority only when strict-fidelity and rare-pivotal gates pass; raw opportunities, overlap/occlusion/path load are required secondary outcomes |
| H2b | complete entropy panel | required secondary exploratory, two-sided paired family with BH control; “different,” never automatically “better” |
| H2c | rare-pivotal qualified-assertion recall on the frozen positive-denominator eligible subset | \(\Delta_R=R_{C2}-R_{C1}\), lower 95% CI \(>-0.05\); separate corrected C2--C0 estimate/test |
| H3 | AMI on completed gold-anchor universe, unweighted mean conductance, separation, pairwise fragmentation/merging error | directional paired tests with Holm on the frozen community-eligible subset; anchor coverage, coherence, and interpretability are required secondary reports |

## 7. Focused related-work synthesis and research gap

### 7.1 Ontology learning from text

Classical ontology learning extracts concepts, taxonomies, relations, and instances from corpora, often to form reusable domain ontologies. Maedche and Staab provide an early framework for ontology learning from text; Text2Onto added confidence-aware, change-oriented support; later surveys systematized the pipeline [R1--R4]. These works motivate C0 and the neutral candidate index, but their usual target is corpus-level reuse rather than repeated construction of identity, event, relation, and abstraction decisions after an individual narrative context is known.

### 7.2 Contextual, perspective-dependent, and query-driven ontologies

Contextual ontology research already demonstrates that knowledge can be local to a context and connected through mappings [R5]. Ben Mustapha et al. learn contextual ontology modules from queries and web snippets [R6]. Optique instead demonstrates industrial ontology-based data access, ontology/mapping bootstrapping, and end-user query formulation at scale [R7]. Therefore this project will not claim to invent contextual or query-driven ontologies. Its narrower question is whether *construction timing* improves temporal narrative projections under controlled same-evidence, same-LLM comparisons, including changes to identity, event structure, and epistemic scope rather than module retrieval alone.

### 7.3 Temporal, event-centric, epistemic, and provenance-aware representation

Allen's interval algebra, temporal RDF research, OWL-Time, and multidimensional fluents show why a timeless triple is inadequate for changing facts, interval relations, validity time, and multidimensional annotations [R8--R12]. SEM, EventKG, event-centric extraction, and narrative ontologies represent events, roles, temporal relations, and narrative structure [R13--R17]. Named graphs, n-ary relation patterns, PROV-O, GRaSP-style grounded perspectives, and formal belief-query work support assertion-level qualification, provenance, and source viewpoints [R18--R21, R48]. These approaches guide the data model; they typically define and populate a representation before querying rather than making pre- versus post-context construction the experimental treatment.

### 7.4 Human-in-the-loop ontology engineering

Text2Onto, OntoGen, WebProtégé, active ontology matching, and interactive-machine-learning work establish that users can inspect, correct, and steer semantic models [R2, R22--R25]. The present loop differs from a one-time curator workflow: every end-user action becomes a versioned context revision and can trigger reproducible C2 reconstruction. A scripted synthetic protocol separates automatic recoverability from claims that require human participants.

### 7.5 LLM-assisted construction and retrieval/graph augmentation

LLMs4OL evaluates LLMs on ontology-learning tasks, while SPIRES and SAC-KG show structured LLM-based knowledge-base or graph population [R26--R28]. These support an active semantic role for the model but do not isolate whether a narrative ontology should be built before or after a query. RAG retrieves from a prebuilt passage index; GraphRAG constructs an entity graph and community summaries during indexing and then queries that completed index [R29--R30]. Those are important comparators, but a query-time answer or subgraph is not by itself a query-constructed ontology.

A recent preprint on dynamic ontologies for agents (OaK) is especially close prior art: it constructs a task-conditioned ontology kernel from training examples, freezes its schema and functions before unseen-query inference, and then instantiates a data-dependent graph for each query [R31]. It prevents any defensible broad claim that dynamic or query-specific graph construction is new, while differing from C2's post-query ontology decisions. Its target is agent task execution, not the controlled C0/C1/C2 construction-timing contrast for temporal narratives, contrastive gold projections, human visual refinement, or joint rare-pivotal, entropy, clutter, and community evaluation. It is discussed as recent non-peer-reviewed work, not ignored or treated as settled evidence.

### 7.6 Dynamic visualization, entropy, readability, and communities

Dynamic-graph surveys and focus-plus-context work distinguish temporal change, mental-map preservation, overview/detail interaction, and layout animation [R32--R35]. Empirical graph-drawing studies show that crossings and other aesthetics can influence task performance, but structural proxies do not prove usability [R36--R38]. Graph entropy has many inequivalent definitions [R39--R41]; community detection has resolution-limit, connectivity, and chance-sensitive comparison problems [R42--R47]. These literatures require a panel of preregistered measures, fixed renderer conditions, chance-adjusted comparisons, stability analyses, and cautious interpretation.

### 7.7 Research gap and bounded novelty claim

Existing work separately addresses text-to-ontology learning, contextual modules, temporal and event representation, provenance and perspectives, interactive ontology engineering, LLM graph construction, retrieval over fixed graphs, and dynamic graph display. The proposed contribution is their integration into a controlled study of **active LLM-based temporal narrative ontology construction after context is known**, where the model may alter identity partitions, event reification, abstraction, relation semantics, temporal/epistemic qualification, and contextual descriptions; structured human revisions regenerate that ontology; and construction timing is isolated with C0, C1, and C2 over identical evidence.

The novelty claim is the method-plus-benchmark-plus-evaluation design. It is not “the first LLM to extract a graph,” “the first query-driven ontology,” or “the first dynamic ontology.” A focused review cannot prove universal priority, so all priority language remains bounded.

## 8. Temporal and epistemic representation decision

### 8.1 Alternatives considered

| Representation | Strength | Limitation for this study | Decision |
|---|---|---|---|
| Plain RDF triples | interoperable and simple | cannot attach independent time, belief, confidence, and evidence to a changing assertion | reject as primary |
| RDF quads/named graphs | useful grouping and provenance | graph name alone does not express all per-assertion temporal/epistemic axes or n-ary roles | export/grouping option |
| RDF-star (2021 Community Group report) | compact quoted-triple annotation | not a W3C Recommendation and semantics/tooling are not identical to RDF 1.2 | legacy experimental export only |
| RDF 1.2 triple terms and reifiers | statement reification integrated into the current RDF specification track | Candidate Recommendation Snapshot as of this plan; tooling remains uneven | experimental export, pinned status |
| Classical reification | portable in RDF 1.1 | verbose and easy to misuse | compatibility export |
| N-ary relation/event objects | natural participant roles, causality, and time | more nodes and explicit conversion rules | required where semantics warrant |
| Property graph with assertion records | pragmatic typed properties and visualization | semantics can become tool-specific | useful implementation view |
| Hybrid typed assertion/event model with RDF-compatible export | rich internal semantics plus interoperability | requires explicit, tested mappings | **selected** |

The authoritative internal model is a typed hybrid of contextual entities, event objects, and qualified assertion records. Binary assertions are first-class records, not bare edges; an event is reified when it has multiple roles, its own duration, causes/consequences, contested occurrence, or independent relevance. A deterministic export maps records to RDF 1.1-compatible n-ary/reification patterns, OWL-Time, and PROV-O. A separately versioned RDF 1.2 mapping may use reifiers/triple terms, but no scientific result depends on draft-standard tooling.

### 8.2 Time and knowledge axes

| Axis | Meaning | Minimum-study status and rationale |
|---|---|---|
| Story/event time | when an event occurs in the represented world, possibly as an interval or partial order | **essential** for temporal change |
| Validity interval | when a state or relation holds; distinct from the event that established it | **essential** for allegiance, location, possession, status |
| Discourse/narration order | where and in what order the text narrates evidence | **essential** for narrative provenance and flashbacks |
| Revelation time | when a reader is first licensed to know a claim | **essential** for evaluating revelation and spoiler behavior |
| Reader spoiler horizon | admissible discourse/revelation boundary for a query | **essential** evidence-admissibility and evaluation constraint, not a security claim |
| Character/source knowledge or belief | who knows, believes, reports, denies, or doubts a proposition and when | schema support is **essential**; population is required for epistemic contexts, not every assertion |
| Ingestion/curation time | when evidence or an annotation entered the system | **required provenance**, not a narrative time axis in primary analyses |
| Nested beliefs/possible worlds | beliefs about others' beliefs, counterfactual worlds | optional extension; high modeling and annotation cost |
| Full bitemporal correction history | transaction-time semantics for every correction | optional; versioned provenance covers the minimum study |

Times can be exact, interval-bounded, ordinal, relative, or unknown. The minimum algebra implements Allen/OWL-Time interval relations—before/after, meets, overlaps, during/contains, starts/finishes, and equal—and separately defined application-level lower/upper uncertainty bounds, without pretending that the fictional calendar is complete. Story time, validity, discourse order, and proposition-level revelation are never collapsed into one timestamp. The model-level reader-admissible proposition set is computed from annotated \(\tau_r\) values and the context's spoiler horizon \(\sigma_q\); actual reader knowledge is not inferred. Character belief remains a separate qualified claim.

Every displayed assertion exposes or can reveal its temporal scope, epistemic holder/status when applicable, narrative commitment, confidence, evidence locations, and query relevance. Conflicting reports coexist as separately sourced propositions rather than being flattened into a single fact. An attributed proposition is a first-class content object but is not globally asserted; RDF export must represent the attitude/qualification without emitting the embedded base triple as true unless a separate world-asserted record exists. Narrative commitment or contestation is distinct from pipeline schema/validation status. Pipeline validation is `accepted`, `rejected`, or `invalid`; an explicitly uncertain but structurally legal scope can be accepted while carrying a separate `underdetermined` temporal diagnostic/precision state. Constraint violations are rejected; the CPU validator does not silently decide contested narrative truth.

## 9. Synthetic benchmark design

### 9.1 Purpose, scale, and splits

The synthetic benchmark supplies known answers before the project makes claims from a copyrighted narrative whose complete gold ontology is unknowable. Its purpose is not to mimic every literary phenomenon; it is to identify whether the three conditions recover controlled contextual distinctions under repeatable evidence.

The initial benchmark contains eight development worlds and 24 sealed test worlds. The test set is balanced across three prespecified difficulty tiers (eight worlds each): easy projections of 10--12 gold entity/event nodes, medium projections of 13--16, and hard projections of 17--20. Each underlying world and its scored gold projections remain in the requested 10--20-node regime; surface mentions and assertion records may be more numerous. Deliberately empty-answer diagnostic queries, if used, are marked separately and excluded from the 10--20-node primary set.

Each test world has four base QueryContexts (96 world-context units), at least one contrastive pair, and two semantically equivalent phrasings per base context. Thus the same evidence supports multiple appropriate ontologies and paraphrase-invariance tests. Three paired seed-block construction replicates per canonical base context are planned. A stratified 48-context subset also runs its second phrasing under all three seed blocks; all second phrasings are generated and released even when not executed. Counts are frozen after a resource and power-simulation pilot, without observing held-out condition outcomes.

The planned initial-output inventory, before repairs and feedback, is:

| Item | Count and dependence |
|---|---|
| Structured test contexts | 24 worlds × 4 = 96 |
| Canonical C2 outputs | 96 × 3 seed blocks = 288 |
| Paraphrase C2 outputs | 48 stratified contexts × 3 = 144; paired with their canonical outputs |
| C1 preconstruction calls | at most 24 worlds × 3 registered horizon snapshots × 3 seeds = 216; one preontology is reused across same-snapshot contexts and modeled as shared |
| C1 fixed projections | 288 canonical + 144 paraphrase, derived from the appropriate three preontologies |
| Deterministic C0 projections | 96 canonical + 48 paraphrase; never copied as independent replicates |
| `A-FixedSelect` | 96 canonical × 3 = 288 query-time calls; additional ablations use registered subsets |
| Scripted feedback | one registered context per world × three seed blocks × supplied-versus-sham revision, with one to three fixed steps; exact call count declared by the frozen scripts |
| Layout/community seeds | one primary seed/configuration plus a fixed five-seed sensitivity set per valid projection; these are not LLM replicates |

Worlds are divided by stable IDs into `development`, `test`, and an optional public challenge split. Lexical variants, narrative renderers, and generator templates are split as groups, not individual queries, preventing nearly identical templates from appearing in development and test. At least two structural template families and one narrative surface renderer are wholly held out from prompt and rule tuning.

Before test generation, a coverage matrix requires at least 24 contexts with a gold merge/split decision, 48 with temporal change, 24 with epistemic conflict, 48 with rare-pivotal facts, 32 with frequent irrelevant facts, 24 with incomplete/competing evidence or a justified abstention, 24 with a context-dependent abstraction, 24 with a causal chain, and 48 with meaningful nontrivial gold communities. Each required factor appears in every difficulty tier, and registered combinations—not seed shopping—satisfy overlapping quotas. The rare-pivotal set must contain at least 48 qualified assertions overall. Community-eligible contexts have a hard disjoint partition with 2 through \(n-1\) groups for the primary AMI analysis; soft/alternative groups are secondary.

### 9.2 Representation-neutral source of truth

Generation starts from a representation-neutral world specification rather than from the C2 output schema. It defines actors, possible aliases/personas, groups, places, events, state transitions, reports, evidence witnesses, partial temporal orders, causal dependencies, and disclosure order. A separate deterministic compiler emits:

1. narrative passages containing explicit, implicit, ambiguous, repeated, and conflicting evidence;
2. a neutral evidence-index fixture;
3. query contexts from a declared grammar and distribution; and
4. gold contextual projections and allowed equivalence classes.

The compiler is independent of the LLM prompt. It can realize the same world in multiple narrative orders and surface forms. Gold semantic IDs never appear in model-visible text or evidence IDs.

### 9.3 Controlled factors

Each world's manifest declares the presence and level of the following factors:

| Factor | Controlled realization |
|---|---|
| Temporal change | allegiance, location, possession, office, or status changes with exact, bounded, or partial ordering |
| Events and participants | binary events and n-ary events with agent, patient, beneficiary, location, cause, and consequence roles |
| Contextual merge/separation | aliases that should merge in one context; role, persona, claimant, or mistaken-identity distinctions that should remain separate in another |
| Alias/coreference difficulty | names, titles, pronouns, descriptions, reused titles, and deliberately ambiguous candidates |
| Relation ambiguity | surface phrases compatible with different normalized predicates until context disambiguates the intended level |
| Distractors | entities, events, and evidence unrelated to the sampled context |
| Frequent irrelevant facts | repeated routine relations that must not dominate solely through frequency |
| Rare pivotal facts | once-mentioned facts that change an answer, causal chain, state, or community assignment |
| Incomplete/competing evidence | unknown endpoints, incompatible reports, uncertain claims, and explicit abstention cases |
| Character knowledge | holder-specific knowledge, belief, report, denial, and mistaken belief |
| Proposition revelation order | claims narrated after their story-time occurrence, proposition-level \(\tau_r\), and spoiler-horizon exclusions |
| Causal chains | direct, mediated, enabling, and merely chronological relations |
| Abstraction | event-level, episode-level, role-level, and aggregate group views with more than one valid granularity where declared |
| Multiple contexts | allegiance, kinship, causality, epistemic, participation, loyalty change, movement, conflict, consequence, and horizon views over the same evidence |

Difficulty varies factor count, evidence ambiguity, length of causal/temporal chains, distractor ratio, relation granularity, and number of permissible alternatives. It does not merely add nodes.

### 9.4 Required preserved artifacts

Every scenario bundle preserves, with hashes and schema versions:

- the complete underlying world state and generation trace;
- narrative evidence and ordered document/passage/span IDs;
- gold mention-to-entity alternatives, entities, and events;
- gold temporal assertions, partial orders, validity intervals, and epistemic propositions;
- each query-specific contextual ontology, local schema, and decision signature;
- relevant and irrelevant nodes, events, and qualified assertions;
- rare-pivotal labels with independent rarity and pivotality rationales;
- gold contextual community assignments when meaningful, including permissible partitions or soft co-assignment constraints;
- permissible alternative entities, predicates, event reifications, and abstraction choices; and
- generator, world, query, paraphrase, evidence-order, and gold-compiler seeds.

Where a single exact graph would be arbitrary, a `GoldAlternativeSet` enumerates equivalent graphs or declares local equivalence rules. Scoring uses the best admissible matching chosen without reference to condition identity. Alternative freedom is narrow and documented; it cannot excuse unsupported claims.

### 9.5 Avoiding a benchmark that simply favors C2

The benchmark could become circular if every gold graph is generated by the same assumptions as the treatment. The following safeguards are mandatory:

- include null cases where all contexts legitimately share an ontology, cases where a good comprehensive ontology is sufficient, and contexts for which a pre-query graph should win on stability or cost;
- derive gold from world/query semantics, not by asking the study LLM or reusing its prompt/schema;
- hold out generator structures, not only names and prose;
- include adversarial cases for over-merging, unnecessary event reification, hallucinated causality, wrong abstraction, frequency bias, and inappropriate context sensitivity;
- manually inspect every template and a stratified sample of every generated split; conduct a delayed, condition-blind second pass over all test gold and seek an independent review of at least 20% if feasible;
- run mutation tests proving that changed world facts, horizons, or contexts cause the intended gold change and only that change;
- validate gold temporal constraints and provenance mechanically;
- retain cases with multiple defensible ontologies rather than forcing a single preferred serialization; and
- publish the entire synthetic generator, splits, manifests, review log, and gold rules so others can construct counter-benchmarks.

The benchmark is a controlled test bed, not proof of general narrative understanding. Case-study disagreements are analyzed rather than forced into synthetic exact-match assumptions.

## 10. Random and contrastive query generation

### 10.1 Declared sampling distribution

Queries are sampled from structured context records before their natural-language realizations are generated. For the 96 held-out base contexts, use a balanced distribution over ten lens families: political allegiance (10%), family/kinship (10%), causal responsibility (10%), knowledge or belief (10%), event participation (10%), changes in loyalty (10%), geographic movement (10%), conflicts (10%), consequences of a pivotal event (10%), and story-time/spoiler horizon (10%). Rounding is stratified across worlds so every lens appears in each difficulty tier. Within a lens:

- temporal scope is `point`, `interval`, `before/after event`, or `unbounded` with probabilities 0.25 each;
- spoiler horizon is `early`, `middle`, `late/full` with probabilities 0.35, 0.35, and 0.30, subject to a valid answer;
- abstraction is `specific event`, `episode`, `role`, or `group-level` with probabilities 0.35, 0.25, 0.25, and 0.15;
- epistemic viewpoint is `reader-admissible`, `omniscient evidence`, `named character`, or `not applicable`, conditional on lens, with the conditional table stored in the generator manifest; and
- target size budget is sampled uniformly inside the tier's node range and then held equal across conditions.

This distribution is a recommended starting point, not a covert post hoc choice. Invalid combinations are resampled with the rejection reason logged, and the realized strata must satisfy the frozen coverage matrix; seeds cannot be searched for favorable condition outputs. Development pilots can revise probabilities; the version and final distribution are frozen before test generation. Free-form phrasings are surface realizations of these known records, never the source of gold truth.

### 10.2 Contrastive sets

Each world includes at least two contexts that require a nonselection change in \(A(O)\), such as:

- a kinship projection that merges titles with a person versus a mistaken-identity projection that represents claimed and actual identities separately;
- an allegiance snapshot represented as a state assertion versus a causal query that reifies the pledge or betrayal event and its consequences;
- an omniscient causal view versus a character-belief view containing incompatible propositions;
- a before/after loyalty comparison with different validity intervals;
- an event-participation view centered on individual roles versus a political view abstracting participants to factions; or
- an early spoiler horizon that withholds a revealed identity versus a later horizon that licenses it.

The gold package explicitly records invariant decisions, required additions, deletions, substitutions, merges/splits, and reifications between contexts. Merely selecting disjoint node subsets is insufficient for a primary contrastive pair.

### 10.3 Paraphrases and seed hierarchy

Each structured context has exactly two stored phrasings for the initial benchmark: one canonical realization and one paraphrase; some second phrasings are human-authored during the blind review. The registered 48-context subset executes both phrasings, and all structured fields remain constant. The seed hierarchy prevents accidental coupling:

\[
s_{\mathrm{root}} \rightarrow
(s_{\mathrm{world}},s_{\mathrm{narrative}},s_{\mathrm{query}},s_{\mathrm{paraphrase}},
s_{\mathrm{retrieval}},s_{\mathrm{LLM}},s_{\mathrm{layout}},s_{\mathrm{community}},s_{\mathrm{bootstrap}}).
\]

Every derived seed and pseudorandom-library version is recorded. Contrastive responsiveness is evaluated against gold-required differences; arbitrary output variability receives no credit. Paraphrase stability is evaluated conditional on correctness so two identically wrong graphs do not appear successful.

## 11. Game of Thrones books case-study design

### 11.1 Legal and reproducible ingestion

The main narrative case study follows the synthetic acceptance gate. “Game of Thrones books case study” here means locally supplied English text of the first five *A Song of Ice and Fire* novels—*A Game of Thrones*, *A Clash of Kings*, *A Storm of Swords*, *A Feast for Crows*, and *A Dance with Dragons*—not television scripts, fan wikis, or reference books. Exact lawful editions may differ and are recorded/frozen before ingestion; changing the five-volume scope requires a resource and comparability amendment. An ingestion manifest records edition metadata, file hashes, chapter boundaries, and passage segmentation. Raw files, passages, embeddings that permit reconstruction, and long quotations remain in an ignored, access-controlled local store. They are never committed, bundled with artifacts, or redistributed.

Public reproducibility relies on code, schemas, prompts, synthetic data, configurations, seeds, non-infringing aggregate results, and cautious annotations such as opaque or chapter-level locators where lawful. A local alignment command lets another authorized researcher recreate private passage IDs from the same edition. Public tests use synthetic fixtures and do not require the novels.

### 11.2 Case selection

The recommended bounded design uses 12 registered narrative windows distributed across the books and discourse positions, with four contextual queries per window (48 context units). Windows are selected on development criteria before condition outputs are inspected and must cover:

- character aliases and titles;
- uncertain or partially ordered story time;
- changing allegiances and locations;
- conflicting knowledge, beliefs, reports, and rumors;
- events narrated after their story-time occurrence;
- early and late spoiler horizons;
- rare but pivotal events or revelations; and
- at least two contexts per window that induce distinct entity, event, relation, abstraction, or temporal organizations.

Full local books are neutrally indexed, but the primary timing comparison uses a frozen, query-blind, bounded narrative-window/horizon snapshot \(D_{w,z}\) that fits the common evidence envelope. For that comparison \(E_q\) contains all admissible evidence in \(D_{w,z}\) for every condition, eliminating both future-evidence and retrieval-recall advantages. A separately labeled retrieval-realism analysis may use the full horizon index and a common bounded packet; because C0/C1 can then have broader preconstruction exposure, it is not evidence for the central construction-timing causal claim and the asymmetry is reported. Queries span allegiance, family, responsibility, knowledge, participation, loyalty change, movement, conflict, and consequences rather than reproducing one template.

### 11.3 Annotation and analysis

All 48 units receive manually curated relevance, horizon, alias, rare-pivotal, and high-level organization annotations. A stratified subset of at least 24 receives detailed assertion, temporal, event-role, and permissible-alternative annotation; final counts are fixed after an annotation-time pilot. A second qualified reader reviews a stratified subset where resources permit, with disagreements preserved and adjudication rules reported. Annotators see evidence within the declared horizon and are blind to condition identity.

Quantitative outputs include alignment against the curated subset, unsupported-assertion audits, rare-pivotal recall, contrastive sensitivity, entropy/clutter/community profiles, cost, latency, and stochastic stability. Qualitative cases include successful contextual restructuring, fixed-ontology advantages, unsupported causal or epistemic inferences, horizon leaks, and ontology changes following human revisions. Pairwise expert preference uses anonymized, order-randomized projections and a rubric separating factual support, contextual usefulness, organization, and readability. Structural measures alone do not justify usability claims.

The case study does not assume that fictional narration has one objectively complete ontology. Alternative readings, unreliable reports, and uncertain chronology are represented rather than adjudicated by model confidence. Complete gold coverage is neither claimed nor required for the synthetic-first causal comparison.

## 12. Human-in-the-loop protocol

### 12.1 Interaction cycle

The loop is explicit and versioned:

\[
(D_z,E_q,q,h_t,O_{q,h_t},V_t) \xrightarrow{\text{user action }r_{t+1}}
h_{t+1} \xrightarrow{\text{C2 reconstructs on GPU}}
O_{q,h_{t+1}} \xrightarrow{\text{validate/render}} V_{t+1}.
\]

Supported semantic actions are:

1. refine the natural-language question or structured lens;
2. select or change story-time or the discourse-based spoiler horizon;
3. request expansion or compression with a target budget;
4. mark a node, event, or assertion relevant or irrelevant;
5. correct or dispute an assertion and optionally attach evidence;
6. request an entity/concept merge or split;
7. request more evidence or an explanation of missing evidence;
8. change the desired abstraction level; and
9. request event reification/de-reification or relation clarification when advanced controls are enabled.

Pan, zoom, filter, temporary hide, and bundle are renderer actions and do not reconstruct an ontology. Semantic actions create a condition-independent `UserRevision`. Cross-condition replay never assumes projection-local IDs are equal: each revision carries a `FeedbackAnchor` made from admissible evidence/mention IDs, a normalized semantic signature, and optional story-time/role constraints. The shared revision stores its ID, parent revision, action, anchor, typed requested change/constraint, actor, order and timestamp, rationale/evidence, and prior/resulting context hashes; it contains no condition ID, projection-local object ID, resolved target, actual before/after projection value, model output, or condition outcome.

A frozen gold-blind resolver maps the anchor separately in each condition. A distinct runner-only `FeedbackResolution` records the revision ID, condition, optional clicked source ID for UI provenance, that condition's resolved target IDs, `resolved`/`ambiguous`/`absent` status, any subsequent `capability_limited` response, before/after projection IDs, resolver/model/prompt/config revisions, and seed. A constructor receives the condition-independent history rendered as `ModelVisibleRevision` records and, when needed, only the receiving condition's own resolution—not an all-condition map or another condition's local ID. Histories branch, support undo through a new event, and are never overwritten.

For scripted cross-condition evaluation, anchors and requested changes are frozen before condition outputs from admissible evidence and the registered feedback script; only the click-provenance field may be session-specific and it is never replayed. Ad hoc human revisions from one displayed condition are not silently reused as another condition's matched feedback treatment unless a condition-blind coordinator can normalize them to the same predeclared anchor/request.

C0 and C1 receive the same anchor-level request, not another condition's local object ID, but capability enforcement limits them to reselection or faithful descriptions of sealed objects. Anchor-resolution failure and inability to satisfy a requested split or new event structure are measurable outcomes, not silently bypassed. C2 regenerates from the registered \(D_z\) manifest/hash, the frozen evidence packet \(E_q\), \(q\), and \(h\), never from unrestricted snapshot contents or untracked conversational memory.

An evidence request cannot become privileged adaptive search for C2. In controlled runs it either names evidence already in \(E_q\), or invokes a frozen “next packet” retrieval policy within the same \(D_z\); the new packet/version is then exposed to every condition and scored as a distinct feedback stage. A user-supplied correction may cite only admissible snapshot evidence. Changing a spoiler control switches all conditions to the separately prepared \(D_z/O_{\mathrm{pre},z}\); C2 regenerates, whereas C0/C1 only reproject their matching sealed ontology and log any capability limitation.

### 12.2 Evaluation layers

- **Automatic initial evaluation:** all conditions answer the frozen synthetic queries without feedback.
- **Scripted refinement:** fixed policies issue one to three known corrections or context changes. Every condition receives the same `FeedbackAnchor` request and must return a legal response, anchor-resolution outcome, or explicit capability failure. Let \(T\) be the nonempty set of gold-required target changes, \(U\) the invariant region, and \(\widehat{\Delta}\) the aligned predicted semantic diff. Report recovery gain (post-minus-pre strict F1); target-change recall \(|\widehat{\Delta}\cap T|/|T|\); edit locality \(|\widehat{\Delta}\cap T|/|\widehat{\Delta}|\); successful target corrections per semantic revision and per 1,000 generated LLM tokens as two separate efficiency measures; nonlocal regression (formerly correct items in \(U\) made wrong divided by formerly correct items in \(U\)); rare-pivotal/unsupported changes; interaction count; latency; and replay. If the predicted diff is empty while \(T\) is nonempty, target-change recall and locality both score 0; the both-empty case is `NA` and excluded because registered scripts require a target change. If there are no formerly correct items in \(U\), nonlocal regression is `NA` and its zero denominator is reported. Token efficiency is `NA` for a zero-token CPU path and never imputed as infinite. `A-NoFeedback` receives a withheld or sham revision at matched reconstruction budget and is scored against the same post-revision target; initial performance is a separate baseline.
- **Documented researcher/expert refinement cases:** the minimum study includes scripted replay plus a small, condition-blind set of evidence-grounded refinement traces performed with the functioning interface. These establish that real actions can be represented and audited, not general usability.
- **Later formative human/expert evaluation:** after the interactive application passes synthetic tests and ethics review, approximately 8--12 consented participants with adequate narrative familiarity may complete counterbalanced C1-versus-C2 tasks. Outcomes include task correctness, completion time, number/type of revisions, confidence calibration, perceived mental effort/workload, evidence use, and qualitative explanation. This is required before any human-usability claim but is optional for the minimum methods paper.
- **Optional powered study:** a later sample size determined by a preregistered power simulation can support stronger usability claims. The functioning interface, feedback model, scripted evaluation, and documented refinement cases remain part of the minimum study.

Factual corrections are analyzed separately from legitimate context changes: the former repair an error relative to a fixed target, while the latter change the intended target ontology. Participant study materials must receive institutional ethics review or exemption determination, informed consent, data minimization, and a withdrawal procedure before recruitment.

## 13. Rich node, edge, and graph presentation

The ontology must remain informative without placing all information on screen at once. Each **node or event** can expose:

- canonical name and supported aliases;
- short contextual role and relation-sensitive summary;
- type in the current projection and, where useful, the distinction from corpus-level candidates;
- current temporal state or event interval;
- confidence/uncertainty and contested status;
- evidence availability/count and spoiler-safe preview; and
- projection ID and change history.

Each **edge/assertion object** can expose:

- a human-readable, evidence-grounded relation description;
- normalized predicate, direction, participant/event or causal role;
- story-time and validity scope;
- epistemic holder/status and assertion revelation position when applicable, with the active query spoiler horizon shown separately;
- confidence, provenance, passage location, and evidence-access action; and
- a supported explanation of why it matters for the current context.

The default overview displays concise names, contextual roles, selected relation phrases, uncertainty glyphs, and temporal cues. Semantic zoom progressively reveals aliases, typed properties, temporal intervals, and evidence badges. Focus-plus-context dims rather than deletes surrounding structure. A timeline/time slider filters validity and event occurrence; a separate spoiler control changes the discourse-based evidence horizon \(\sigma_q\) and triggers regeneration when semantics change. An evidence drawer shows local, authorized passages; a projection-diff view shows merge/split, schema, assertion, and temporal changes after feedback. Edge bundling is permitted for overview readability only if every underlying assertion remains individually discoverable.

Stable positions across revisions aid mental-map comparison, but layout stability may not prevent C2 from making a justified semantic change. Generated labels, summaries, tooltips, and “why it matters” text are included in hallucination audits. No future fact may leak through a tooltip, node size, layout group, alias list, or evidence count beyond the selected horizon.

Semantic evaluation operates on validated \(O\) before rendering. Renderer-specific evaluation operates on a versioned \(V\) with fixed viewport, fonts, style, label policy, layout algorithm, and seeds. This separation prevents visual filtering from being mistaken for ontology construction.

## 14. Context-alignment metrics

The full artifact key is world/window × structured base QueryContext × surface wording × condition × paired seed block × feedback stage. The **core confirmatory RQ1--RQ3 dataset is frozen to the 96 canonical-wording contexts at initial history \(h=\varnothing\)**; the three stochastic seed blocks are averaged within unit as specified in Section 20. Paraphrase executions enter only H1c invariance safeguards and labeled robustness summaries, while post-revision stages enter only HITL estimands unless a separate analysis is preregistered. Macro-averaging over world-context units is primary; micro-averages are secondary. Surface paraphrases and LLM samples are repeated measures, not new worlds. Deterministic C0 contributes one output per wording and is never copied three times as independent evidence. A deterministic matcher independent of condition identity aligns a prediction with every permissible gold alternative and uses the best allowed match.

### 14.1 Component alignment

For matched predicted and gold component sets, report precision, recall, and F1 separately for nodes and qualified assertions:

\[
P=\frac{|\widehat{Y}\cap Y|}{|\widehat{Y}|},\qquad
R=\frac{|\widehat{Y}\cap Y|}{|Y|},\qquad
F_1=\frac{2PR}{P+R}.
\]

Primary node identity matching uses gold mention/span correspondences and allowed alias mappings rather than label strings or contextual type; type and abstraction are scored independently after identity matching. Strict assertion matching requires matched endpoints or event roles, normalized predicate, direction, story/validity scope, and applicable epistemic status. Relaxed assertion F1 gives partial credit under frozen predicate hierarchy and time tolerances; it is secondary and never replaces strict scoring. Primary gold sets are nonempty. For diagnostics, both sets empty scores 1, exactly one empty scores 0, and every such case is flagged; this convention cannot make an invalid/empty primary output look good.

The complete alignment panel includes:

- contextual node precision/recall/F1 and strict/relaxed assertion precision/recall/F1;
- macro relation-type agreement/F1, including false invention of predicates;
- separate story/event-time, validity-interval, discourse-order, revelation-order, and epistemic holder/status scores: bounded interval IoU; point-event correctness within a frozen tolerance; and macro F1 over allowed Allen/partial-order relation sets for open, uncertain, or unknown time;
- mention-level entity merge/split performance using B-cubed precision/recall/F1 and pairwise merge-decision F1;
- event identity, participant-role, causal-link, and reification-decision F1;
- contextual type, schema-element, and abstraction-level agreement;
- normalized typed graph-edit distance with preregistered costs for node, relation, qualification, split/merge, and event edits; divide minimum edit cost by the cost of deleting the complete prediction and inserting the complete gold, cap at 1, and version the cost table;
- evidence-citation validity (valid cited IDs/all cited IDs), evidence-grounding precision (supported adjudicated predicted factual components/all adjudicated predicted factual components), and evidence coverage (gold relevant assertions with at least one cited supporting span/all gold relevant assertions);
- unsupported-assertion rate (unsupported adjudicated factual clauses/all adjudicated predicted factual clauses), including labels and summaries; it is not assumed to be exactly one minus grounding precision when contested/unadjudicated clauses exist;
- rare-pivotal qualified-assertion recall; and
- irrelevant-information rate \(|\widehat{Y}_{\mathrm{irrelevant}}|/|\widehat{Y}|\), with empty denominators declared.

Evidence-grounding precision requires that a cited span actually supports the assertion under a blind rubric or synthetic entailment rule; a syntactically valid citation is insufficient. Spoiler-leak rate is horizon-ineligible factual clauses or evidence references divided by all displayed factual clauses/references, with a required value of zero on programmed tests. Confidence calibration (Brier score and reliability plots on synthetic correctness labels) is diagnostic.

**Ontology-decision macro F1** is the unweighted mean of the defined-family F1 values for `merge_split`, `contextual_type`, `schema_relation`, `event_reification`, `abstraction`, and `temporal_epistemic_qualification`. The four-case rule is fixed per family: gold empty/prediction empty is `NA`; gold nonempty/prediction empty is 0; gold empty/prediction nonempty is 0; otherwise calculate F1. This prevents invented decisions in an inapplicable family from disappearing from the macro. Component results are always reported. Contextual role/label relevance and factual grounding are secondary decision-signature measures and can demonstrate context response, but never alone prove construction.

### 14.2 Context sensitivity and paraphrase invariance

Let \(A_g(q_i)\) and \(A_m(q_i)\) be gold and method decision signatures. For a contrastive pair \((q_i,q_j)\), construct signed sets of required additions, removals, substitutions, merges/splits, reifications, and qualification changes:

\[
\Delta_g=A_g(q_j)\ominus A_g(q_i),\qquad
\Delta_m=A_m(q_j)\ominus A_m(q_i).
\]

The operator \(\ominus\) first aligns evidence mentions across contexts, then emits typed signed operations (`add`, `remove`, `substitute`, `merge`, `split`, `reify/de-reify`, `qualify`) relative to that common mention basis. **Contrastive decision-change F1** is the unweighted macro F1 between \(\Delta_m\) and \(\Delta_g\) over the union of gold-active and prediction-active nonselection operator families, using the same four-case empty-family rule as ontology-decision macro F1. **Invariant preservation F1** scores decisions that gold says must remain unchanged and is a non-inferiority safeguard with lower 95% bound greater than -0.05 for C2--C1. These two metrics prevent arbitrary changes from earning credit. **Primary ontological-collapse rate** is the proportion of pairs with a nonempty gold nonselection delta for which the predicted nonselection delta is empty; changing only selected/visible nodes therefore still counts as collapse. A stricter near-identical-output diagnostic additionally requires normalized total signature edit distance at most 0.01. A supplementary distance-calibration analysis correlates normalized predicted and gold typed edit counts, resampling at world level.

For paraphrases \(q\) and \(q'\) with the same structured context, report aligned decision agreement, normalized signature edit divergence, graph similarity, output variance, and the difference in strict gold alignment. Paraphrase stability is interpreted conditional on correctness; equivalent failures are not celebrated. H1c uses a 0.05 correctness non-inferiority margin and requires the one-sided upper 95% confidence bound for mean normalized divergence to be below 0.10, both rules frozen after development.

For the case study, incomplete gold is supplemented by blinded pairwise expert preference on factual support, contextual alignment, organization, temporal/epistemic clarity, and evidence utility. Preference does not replace synthetic known-answer metrics.

## 15. Entropy and visual-clutter measures

### 15.1 Graph conventions

Every metric names its graph conversion. The primary semantic skeleton is an **unweighted** undirected entity-event incidence graph: contextual entities and reified events are vertices; event roles and binary assertions are unit edges; parallel qualified assertions are retained for typed-count measures and collapsed for simple topology. A unit-count weighted skeleton, an evaluator-relevance weighted sensitivity graph, a projected entity-only graph, and a directed assertion multigraph are separately named sensitivity views. Self-loops are excluded from topology but retained in assertion metrics; isolates remain; horizon filtering precedes conversion; an assertion with unknown time is included in a slice when it may overlap the slice and is excluded only in a strict-known-time sensitivity; disconnected components are retained. Graph-conversion code and hashes are constant across conditions.

Natural logarithms are used and \(0\log 0=0\). Raw entropy, probability-support size, and effective diversity \(e^H\) are always reported. Degree-histogram entropy is normalized by \(\log n\); degree/strength-mass and edge-relevance entropy by \(\log n\) and \(\log m\), respectively. Local and relation-type entropy use a relation-bin universe frozen from the upper ontology plus development schemas and one explicit `OTHER` bin; a condition-blind versioned mapper sends any held-out contextual predicate without a frozen match to `OTHER`. Relation entropy is normalized by the log of that complete frozen bin count, so a new C2 label cannot expand its own denominator. Community-size entropy uses \(\log c\). A normalization is `NA` when its denominator is zero. Undefined cases are `NA`, not conveniently set to zero. Node/edge counts, isolates, components, and density always accompany entropy.

### 15.2 Preregistered entropy panel

1. **Degree-histogram entropy**
   \[
   H_{\mathrm{deg\mbox{-}hist}}=-\sum_k p_k\log p_k,
   \quad p_k=|\{v:d(v)=k\}|/|V|.
   \]
   Total degree is primary; in/out variants are sensitivities.
2. **Degree-mass entropy**, kept distinct from the histogram measure,
   \[
   H_{\mathrm{deg\mbox{-}mass}}=-\sum_v \frac{d_v}{2m}\log\frac{d_v}{2m},
   \]
   on the unweighted primary topology. Weighted sensitivity uses strength \(s_v/\sum_u s_u\) and a distinct name.
3. **Local neighborhood relation entropy**
   \[
   H_v=-\sum_r p_{v,r}\log p_{v,r},\quad p_{v,r}=n_{v,r}/\sum_r n_{v,r},
   \]
   where \(n_{v,r}\) counts incident typed assertion/role records; summarize by mean, incident-count-weighted mean, median, and distribution. Isolates are excluded from \(H_v\) and reported separately.
4. **Relation-type entropy**, from predicate frequencies over qualified assertion records, with raw and normalized values.
5. **Edge-relevance entropy**
   \[
   H_w=-\sum_e q_e\log q_e,\quad q_e=w_e/\sum_j w_j,
   \]
   using nonnegative relevance weights calibrated by the same scorer-only gold rule for evaluation, or a separately labeled frozen method-visible estimator for deployment, identically across conditions. Zero total weight is `NA`.
6. **Von Neumann graph entropy.** The primary calculation uses the unit-weight undirected entity-event skeleton defined in Section 15.1; the evaluator-relevance-weighted skeleton is a separately labeled sensitivity. For either declared symmetrized simple skeleton,
   \[
   L=D-A,\quad \rho=L/\operatorname{tr}(L),\quad
   S_{VN}=-\operatorname{tr}(\rho\log\rho)=-\sum_i\lambda_i\log\lambda_i.
   \]
   Report raw \(S_{VN}\) and \(S_{VN}/\log(n-1)\) when \(n>2\) and at least one edge exists. Tiny negative eigenvalues are clipped only within a fixed numerical tolerance; edgeless graphs are `NA` and fail the content gate.
7. **Community-membership entropy**, \(-\sum_c |c|/n\log(|c|/n)\), is reported only as partition-size balance, never as community quality.

No measure may be dropped because it moves unfavorably. Regular empty and complete graphs can both have zero degree-histogram entropy, illustrating why “lower is better” is false. Interpretation is joint with fidelity, relevance, coverage, size, connectivity, direct clutter, and rare-pivotal recall. All entropy comparisons are two-sided and belong to the registered secondary exploratory BH family.

### 15.3 Structural and rendered clutter

Direct semantic-graph measures are node count, qualified-assertion/edge count, directed and undirected density under named conversions, isolate/component count, maximum and mean degree, degree Gini/centralization, and path ambiguity. Endpoint coverage and unreachable-pair rate are reported first. For reachable, gold-required endpoint pairs, path ambiguity is the number of additional equally short unweighted, type-admissible paths beyond the first, at length at most four under the active temporal slice. Disconnection never scores as low ambiguity.

The confirmatory renderer state is the frozen **unbundled** overview with one visible stroke per rendered assertion edge; the bundled overview is a required sensitivity and UI mode. This makes crossings and mark relevance directly attributable. In the bundled sensitivity, a mixed bundle's stroke weight is divided among its underlying assertions in proportion to their frozen unbundled mark weights, and its irrelevant contribution is the sum assigned to gold-irrelevant assertions; bundle-label words are attributed clause-by-clause to their assertion sources, with mixed or unmapped clauses counted as irrelevant. The primary `IrrelevantVisibleLoad` never receives a benefit merely because several edges were grouped.

Rendered overview measures are:

- raw edge crossings excluding shared endpoints and normalized crossing rate (unordered nonadjacent edge pairs that cross at least once/eligible unordered nonadjacent edge pairs), with curved/polyline conventions declared;
- count and viewport-normalized intersection area of label-label and label-node overlap;
- node-node, node-edge, and label-edge occlusion;
- edge congestion/bundle load;
- visible information load: visible nodes, assertion strokes, labels, label words, glyphs, and open panels, reported separately; a weighted index frozen on development is secondary; **irrelevant visible load** is the gold-irrelevant visible-mark weight divided by all visible-mark weight; and
- task-specific discoverability of relevant and rare-pivotal assertions, including number of interactions to reveal evidence.

The scorer freezes a condition-independent `crossing_eligible` subset before runs: contexts whose gold unbundled overview has at least one eligible nonadjacent edge pair. For a prediction on that subset, the raw normalized rate is `NA` when it creates zero eligible pairs, but the primary failure-adjusted endpoint is set to 1 with a `zero_opportunity` flag so deleting all crossing opportunities cannot disappear from analysis. Raw numerator/denominator, zero-opportunity rate, and an exposure-weighted pooled-rate sensitivity are reported. Contexts outside the frozen subset are descriptive only for crossing; no condition-specific complete-case subset is allowed.

Viewport size, device-pixel ratio, font files and metrics, stylesheet, label policy, zoom, and layout algorithm/version are fixed. One frozen primary layout seed supplies confirmatory results; a declared five-seed block is sensitivity analysis. Bundled and unbundled views are scored separately, with unbundled designated primary above. Hidden by progressive disclosure and absent from the ontology are different states. Renderer comparisons hold the ontology fixed; semantic-condition comparisons use the same renderer configuration. Invalid/empty scientific outputs contribute to failure and semantic-loss analyses but receive no favorable zero-clutter value; clutter is analyzed among schema-valid content-bearing outputs under the joint fidelity/rare gate. Structural and visual measures motivate a later task study but do not establish human usability.

## 16. Cluster and community measures

The primary algorithm is Leiden optimizing the Constant Potts Model on the frozen unit-weight entity-event skeleton, at a resolution selected on development data and then locked. Report modularity at a declared configuration-model null and \(\gamma=1\), even though it is not the selection objective. A preregistered resolution sweep and Infomap sensitivity analysis assess dependence on algorithmic objective but create no additional confirmatory opportunities. Event-to-entity projections, directed/weighted choices, library versions, and seeds are explicit. The frozen community-eligible subset contains at least 48 synthetic contexts with a reviewed nontrivial hard disjoint gold partition; overlapping or soft gold is secondary.

Gold-alignment metrics use a fixed scorer-side vertex universe \(U_g\) of query-relevant entity/event anchors defined by gold mention/evidence IDs before condition output. Each anchor inherits the community label of its matched predicted vertex; an unmatched anchor receives its own unique `ABSENT:<anchor>` label, so omitted difficult vertices cannot disappear or spuriously co-cluster. Extra predicted vertices remain visible in node precision/irrelevance measures and a union-universe sensitivity. Anchor coverage and counts are always reported. AMI, NMI, purity, and the pairwise errors below operate on this same completed \(U_g\), not a condition-specific intersection.

For each projection report:

- number and size distribution of communities (descriptive only);
- modularity and internal density;
- per-community conductance
  \[
  \phi(S)=\frac{\operatorname{cut}(S,\bar S)}
  {\min(\operatorname{vol}(S),\operatorname{vol}(\bar S))},
  \]
  plus the **unweighted mean across communities** as balanced conductance (lower is better); standard volume counts internal edge weight twice and boundary weight once. For each detected community, separation is internal edge weight counted once divided by internal-once plus boundary-once edge weight; the reported graph-level separation is the unweighted mean over detected nonzero-volume communities (higher is better). Singletons with boundary edges have zero internal separation; zero-volume components are flagged and excluded from both means with their rate reported;
- semantic coherence from gold contextual categories on synthetic data and a frozen human rubric on the case study, not the evaluated LLM's self-score;
- alignment with meaningful gold groups using AMI as primary and arithmetic-normalized NMI as secondary;
- purity together with inverse purity and pairwise fragmentation/merging error, because purity alone rewards overclustering. On aligned eligible vertices let \(S_g\) be unordered pairs assigned to the same gold community and \(D_g\) pairs assigned to different gold communities. Fragmentation error is \(|\{(u,v)\in S_g:\widehat c_u\ne\widehat c_v\}|/|S_g|\); merging error is \(|\{(u,v)\in D_g:\widehat c_u=\widehat c_v\}|/|D_g|\). A zero denominator is `NA` with its pair count reported, never zero; over-/under-cluster counts remain descriptive;
- interpretable naming and explanation judged independently from topological score; and
- co-assignment matrices and variation of information/AMI stability.

Two sources of instability are separated: community-algorithm seeds on a fixed ontology and repeated LLM constructions that create different ontologies. The former is summarized within each graph; the latter is part of condition performance. When vertex sets differ across LLM outputs, report mention-aligned vertex coverage/Jaccard first. Confirmatory gold alignment uses the fixed completed \(U_g\) policy above; AMI/variation of information on only common aligned entities is explicitly a sensitivity, as is a union universe that gives extra vertices an `EXTRA` state. At 10--20 nodes, exact partition inspection and chance correction accompany asymptotic measures. Primary gold is nonoverlapping; enumerated hard alternatives use best admissible matching, and soft co-assignment targets use pairwise Brier loss. One-cluster, all-singleton, partially annotated, and empty cases are flagged and interpreted under the pinned metric-library convention rather than pooled unexamined.

Neither a lower nor higher cluster count is hypothesized. High modularity can reflect an inappropriate partition, and modularity has a resolution limit and many near-optimal solutions. The H3 confirmatory decision uses its four frozen structural endpoints; coherence, stability, exact small-graph inspection, and interpretability remain mandatory secondary reports and can limit interpretation, but they are not an unspecified post hoc significance gate.

## 17. Preserving rare but pivotal information

Rarity and importance are annotated independently. A fact's **rarity** is determined before condition outputs from its number of independent evidence mentions and generator stratum; the initial synthetic definition is one independent mention or the preregistered bottom frequency quartile. A fact is **pivotal in context** when removing it changes a correct answer, breaks a gold causal/temporal chain, changes a consequential state or community assignment, or erases a designated revelation. A frequent fact may be pivotal; a rare fact may be irrelevant. The full 2×2 table (rare/common × pivotal/non-pivotal) is scorer-only gold and is never visible to a constructor, retriever, projector, prompt builder, or repair loop.

The co-primary preservation metric is **rare-pivotal qualified-assertion recall**: a hit requires correct entities/event roles, relation, essential temporal/epistemic scope, and valid supporting evidence. Before condition execution, the scorer freezes the at-least-48-context `rare_eligible` subset for which the gold denominator is positive. Recall is computed per eligible world-context and macro-averaged within world before world-clustered inference; noneligible contexts are `NA` with zero denominator reported, never treated as perfect or zero. An invalid/missing output on an eligible unit scores zero under intention to treat. Also report rare-pivotal node recall, evidence recall, precision on included rare facts, false inclusion of rare-irrelevant facts, and survival curves as node/edge/display budgets tighten. For causal chains, report whether every necessary link survives and the proportion of complete answer-support paths.

Rare-pivotal recall is noncompensable. A condition cannot be declared preferable on H2 because it is sparse, has fewer crossings, or has a favorable entropy value if it falls below the preregistered recall floor or non-inferiority margin. Multiobjective/Pareto plots and constrained comparisons make that tradeoff visible. Frequency may inform retrieval, but it can never be the sole importance score. Production safeguards use only method-visible signals—novel state change, causal reach, query-answer necessity proposed from supplied evidence, and uncertainty—not scorer gold. `A-NoRareGuard` removes those signals/constraints only. A gold-firewall mutation test must prove that changing scorer-only rare/pivotal labels cannot change any condition output.

## 18. Experimental controls and fairness

### 18.1 Matched information and capabilities

For each scored unit, all conditions receive the same corpus/index revision \(D_z\), horizon-eligible evidence, passage segmentation, retrieval manifest, upper vocabulary, ontology and display budgets, temporal rules, and provenance format; they are **evaluated against** the same scorer-only gold. No constructor, retriever, projector, prompt builder, validator/repair loop, renderer, or selector can read gold. C0/C1 construct each \(O_{\mathrm{pre},z}\) from exactly the matching \(D_z\), before any lens/question is revealed. In the primary synthetic and bounded-window case analyses, the complete admissible snapshot fits in the common input and \(E_q=D_z\)'s eligible evidence, eliminating retrieval recall and prior-exposure confounds. In the secondary full-index retrieval regime, C2's model can read only the common frozen packet; broader preconstructor exposure is explicitly asymmetric and excluded from the central causal estimand.

C1 and C2 use the same backbone, tokenizer, quantization, inference runtime, scored projection schema/field semantics, validation policy, maximum input/output tokens, repair allowance, decoding families, and replicate seed blocks wherever their different construction timing permits. Prompt and constrained-decoding grammars may differ only where timing/capability requires it and every difference is enumerated in an allowlisted grammar-difference manifest; `A-FixedSelect` is intentionally stricter. Prompt length and evidence tokens are reported separately so “comparable” is inspectable. C2 may not have a larger node/edge budget, unlimited revisions, privileged annotations, or a better evidence corpus.

Before any primary unit is admitted, the `A-FixedSelect` packing preflight must prove that its prompt/schema, complete common \(E_q\), and complete required sealed-ontology serialization fit the shared input cap. No condition may silently truncate evidence or graph content. If a unit fails during development, reduce and refreeze the query-blind narrative window for every condition or adopt a symmetric larger input allowance after the GPU pilot; a held-out failure is retained as an integrity failure rather than repaired condition-specifically.

C0 receives serious development effort: a standard CPU NLP pipeline, event and temporal rules, an upper ontology, provenance, and development-set tuning. It is not deliberately deprived of structures needed by the task. It cannot use an LLM because its purpose is a classical CPU comparison, but all non-LLM deterministic validators and display features are available equally.

### 18.2 Output and compute budgets

Two budget regimes are reported:

1. **quality-matched per projection:** identical maximum evidence tokens, ontology nodes, qualified assertions, visible marks, and query-time generation tokens; and
2. **operational/amortized:** actual preconstruction plus query-time GPU/CPU seconds, tokens, energy proxy, and peak memory, amortized over 1, 4, 16, and the observed number of queries per world.

C1 necessarily pays a query-blind comprehensive-construction cost and C2 pays repeated query-time construction. Equalizing only one call would favor one architecture. Therefore pre-query and query-time compute are both reported, along with total and amortized cost. A secondary compute-matched analysis caps total generated tokens per world; it does not replace the primary semantic comparison. C1 is allowed a credible comprehensive budget and query-blind consolidation, while C2 cannot repeatedly request evidence until it sees the answer.

### 18.3 Integrity controls

- Split by world and generator family; never tune on test queries or gold.
- Seal C0/C1 ontologies separately for every registered \(D_z\) before revealing held-out QueryContexts and record their hashes/timestamps.
- Store C2 pre-query artifact inventories and construction certificates.
- Enforce a scorer-only gold process/account boundary; a mutation of gold labels must leave construction artifacts byte-identical.
- Serialize model-visible context and run controls through an allowlist that excludes world/split, contrast/paraphrase-group, pair/condition-comparison, expected-effect, and scorer IDs; analysis blocking metadata must not cue a constructor.
- Validate that every C0/C1 output semantic ID has pre-query lineage and that C2 has no cross-context ontology cache.
- Use identical matching, graph conversion, layout, and metric implementations across condition labels.
- Randomize anonymized output order for manual/expert evaluation.
- Count invalid, timed-out, refused, and unrepaired outputs under intention-to-treat rules rather than dropping them.
- Freeze primary metrics and directions before held-out execution; publish the full entropy and community panels.
- Audit every generated label, summary, tooltip, and explanation for evidence support.
- Require C0 development competence on a frozen reference suite (schema/evidence validity 100%, strict assertion F1 at least 0.55, temporal-relation F1 at least 0.60, rare-pivotal recall at least 0.60) and C1 preontology schema validity at least 95% with comprehensive gold-assertion recall at least 0.70. These readiness values may be changed once before acceptance data; failure triggers development repair or stops comparison, never a favorable weak baseline.

These controls directly prevent fixed-graph retrieval masquerading as construction, unequal evidence, output-budget advantages, metric cherry-picking, rare-fact deletion, a weak C0, and confounding “LLM use” with “LLM use after the query.”

## 19. LLM stochasticity, repeated runs, and decoding controls

The planned default is the same pinned open-weight 14B-class, four-bit model for C1 and C2 on one 24 GB GPU; the engineering plan recommends Qwen3-14B-AWQ subject to the Phase 1 acceptance pilot. The exact model and tokenizer repository revisions, weight checksums, quantization configuration, chat template, inference runtime/CUDA stack, prompt and JSON-schema hashes, stopping rules, and decoding parameters are frozen in the run ledger.

The primary study uses three paired seed-block replicates \(r\in\{1,2,3\}\) per canonical world/context. For C1, replicate \(r\) creates one query-blind preontology per world/horizon and reuses it across matching queries; this shared dependency is retained in the analysis. C2 uses the same seed block for each contextual construction. Seed block is a fixed nuisance/blocking factor, not a poorly estimated three-level random population. C0 is deterministic and is not falsely replicated as independent data. One primary community/layout seed is frozen, with a separate five-seed sensitivity block.

Recommended decoding begins in the model's non-thinking mode with documented sampling (temperature 0.7, top-p 0.8, top-k 20) and schema-constrained generation; the exact values are a model-specific open decision until the development pilot. Greedy decoding is not assumed stable or optimal. A deterministic-seeming seed does not guarantee bitwise identity across GPU kernels, drivers, or runtime versions. The study therefore records artifacts and quantifies empirical variation rather than claiming universal determinism.

A five-replicate subset estimates whether three samples materially understate variance. The registered 48-context paraphrase subset and, if resources permit, one smaller-model sensitivity are run as repeated measures. Prompt variants and samples are blocking factors, not independent worlds. Raw outputs, validator failures, bounded repairs, token counts, latency, and peak VRAM are retained. Intention-to-treat handling is outcome-specific as defined below; valid-output-only analysis is secondary.

## 20. Statistical analysis plan

### 20.1 Estimands and primary endpoints

The principal estimand is the paired mean condition difference over the declared distribution of the 96 canonical-wording synthetic contexts at initial \(h=\varnothing\). C2--C1 is tested first; C2--C0 follows. Paraphrase and feedback outputs are excluded from these core rows and analyzed only under H1c safeguards and HITL estimands respectively. Primary endpoints are:

- RQ1: strict qualified-assertion F1; ontology-decision macro F1; and, on contrastive pairs, signed decision-change F1 plus primary ontological-collapse rate. Node F1 and grounding precision are H1a safeguards; invariant-preservation F1 and paraphrase correctness/divergence are H1c non-inferiority/equivalence safeguards;
- RQ2: rare-pivotal qualified-assertion recall (co-primary safety outcome), irrelevant-visible-load rate, and normalized edge-crossing rate at the frozen overview; the complete entropy panel is a preregistered two-sided secondary exploratory family;
- RQ3: gold AMI, unweighted mean community conductance, separation, and fragmentation/merging error on the frozen community-eligible subset.

Relaxed assertion F1, temporal/coreference/event submetrics, GED, individual clutter components, NMI, purity, modularity, cluster count, internal semantic coherence, and interpretability are required secondary measures. The distinction between primary, safeguard, and secondary does not permit omission of an unfavorable measure.

The separate mechanistic estimand is C2--`A-FixedSelect` on the same canonical units (and contrastive-pair subset where appropriate): ontology-decision macro F1 and contrastive decision-change F1 are its two Holm-controlled superiority endpoints; strict assertion F1 and rare-pivotal recall use -0.05 non-inferiority safeguards. It supports a construction-freedom attribution, not a fourth condition ranking.

### 20.2 Paired and hierarchical analysis

Conditions are paired on the same world, QueryContext, evidence, budget, permissible gold set, and seed block. Base queries, contrasts, paraphrases, and C1 projections sharing a preontology are not independent. Primary inference for composite F1, AMI, conductance, entropy, and normalized clutter uses paired world-clustered bootstrap confidence intervals and world-level randomization/permutation tests. Stochastic-condition scores are first averaged across the three registered seed blocks for the primary world-context contrast; C0 is compared once, not cloned into three observations.

Hierarchical models are prespecified sensitivity/heterogeneity analyses. A representative long-form structure is:

\[
g(\mathbb{E}[Y]) = \beta_0 + \beta_1\mathrm{Condition}
+\beta_2\mathrm{Difficulty}+\beta_3\mathrm{Lens}
+\beta_4(\mathrm{Condition}\times\mathrm{Difficulty})
+u_{\mathrm{world}}+u_{\mathrm{base\ context(world)}}
+u_{\mathrm{wording(base\ context)}}+u_{\mathrm{shared\ preartifact(condition,world,z,seed)}}
+\beta_{\mathrm{seed\ block}},
\]

with random condition slopes only when estimable. The shared-preartifact term applies where a preontology is reused, especially C1. Constituent correct/incorrect counts use binomial or beta-binomial mixed models; overdispersed count outcomes use negative-binomial models; approximately continuous unbounded outcomes use robust linear models; ordinal ratings use cumulative-link models. Composite F1 and possibly negative AMI are not forced into binomial/beta likelihoods. Crossing counts are analyzed through the frozen normalized rate with world-paired resampling, with raw counts diagnostic. If a maximal model fails, simplify in order: remove interaction, remove random slope, combine sparse lens levels, then retain random intercepts for world and base context. Primary clustered resampling remains valid regardless.

Resampling occurs at the world level. Replicate outputs, phrasings, nodes, and edges are never treated as independent worlds. For every primary contrast report the paired raw difference, 95% confidence interval, standardized paired effect (with small-sample correction where appropriate), and a scale-appropriate effect such as odds ratio, rate ratio, or rank-biserial correlation. Medians and distribution plots accompany skewed measures.

### 20.3 Multiplicity, directions, and missing outcomes

Use \(\alpha=0.05\) and a gatekeeping sequence: test C2--C1 before C2--C0 for each RQ, while reporting the C2--C0 estimate/interval even if the gate prevents a confirmatory rejection. Holm controls family-wise error within each of the three separately declared RQ confirmatory families and separately within the two-endpoint C2--`A-FixedSelect` mechanistic family; no claim of study-wide FWER is made. Benjamini--Hochberg controls the clearly marked secondary exploratory entropy family and exploratory ablation families, excluding `A-FixedSelect`. Metric-level directions are exactly those in the Section 6 table: entropy, modularity, cluster count, and secondary coherence are two-sided; strict/decision/change F1, AMI, and separation are directional upward; H2a's irrelevant-load/crossing rates, collapse, conductance, fragmentation, and merging error are directional downward; rare recall, node F1, grounding, invariant F1, and paraphrase correctness use their stated non-inferiority rules; signature divergence uses its upper-bound equivalence rule. Report raw and adjusted values, not significance labels alone.

The 0.05 rare-recall/node/strict-fidelity/invariant margins, 0.03 grounding margin, 0.10 H1b component margin, and 0.10 paraphrase-signature bound are provisional defaults justified and frozen using development data and substantive loss analysis, not test results. Paraphrase divergence passes only when its one-sided upper 95% confidence bound is below 0.10; its point mean alone is insufficient. A clutter-superiority claim requires both lower 95% confidence bounds \(R_{C2}-R_{C1}>-0.05\) for rare-pivotal recall and strict assertion F1, plus grounding precision at least 0.95. Missing/invalid/OOM/horizon-failing outputs receive semantic precision/recall/F1 and rare recall zero when gold is nonempty, and contribute to failure/repair-rate analyses. Entropy and renderer metrics are `NA` for invalid or content-empty outputs and analyzed conditionally under the joint validity/fidelity/rare gate; they never receive favorable zero clutter. This two-part intention-to-treat policy and a secondary completed-output analysis are both reported.

### 20.4 Power and case-study inference

Before generating the sealed test split, simulate power across plausible within-world correlations and effect sizes using development residuals. Increase the number of worlds rather than treating more LLM samples as independent if power is inadequate. No retrospective “observed power” is used. The 24-world/96-context design is a resource-aware starting point.

The Game of Thrones analysis is paired and hierarchical where annotations permit, but incomplete gold and purposive windows limit population inference. Report estimates and intervals as evidence about the registered case-study set. Expert preferences use a preregistered ordinal or Bradley--Terry mixed model if sample size supports it; otherwise report descriptive uncertainty. A formative 8--12-person interaction study does not support a general usability-superiority claim.

## 21. Ablation studies

The following ablations are registered before the main run. The first six and `A-Prompt` are mandatory; the model-size sensitivity and other diagnostics may use smaller stratified subsets to control cost.

| Ablation | Change | What it identifies |
|---|---|---|
| `A-FixedSelect` | C2 query-time prompt/tool/evidence/repair path, but all create/merge/split/schema/event/abstraction/qualification operators are disabled and sealed C1 IDs enforced | selection/attention benefit versus genuine construction freedom |
| `A-NoContext` | remove lens and structured context while retaining a generic question | contribution of explicit context |
| `A-ShuffledContext` | pair evidence with another world's compatible-looking context | context use and leakage/canned-output detection |
| `A-NoTemporalEpistemic` | prohibit temporal and epistemic fields while keeping evidence | value of qualified representation |
| `A-NoRareGuard` | remove method-visible causal/state-change/answer-necessity preservation signals; scorer-only gold stays firewalled | whether sparsity gains come from losing rare-pivotal facts |
| `A-NoFeedback` | score the same post-revision target with the structured revision supplied versus withheld/sham at matched reconstruction budget | contribution, locality, and cost of human refinement |
| `A-WeakGrounding` | omit mandatory evidence IDs but retain post hoc audit | grounding constraint's effect on support and coverage |
| `A-FixedSchema` | freeze relation vocabulary and types while permitting contextual instances | contribution of schema/relation construction |
| `A-NoMergeSplit` | freeze mention partition | contribution of contextual identity decisions |
| `A-NoEventReify` | allow only binary assertions | contribution of event/n-ary modeling |
| `A-FullEvidence` | all eligible evidence versus deterministic retrieval on selected hard cases | retrieval bottleneck versus construction failure |
| `A-Prompt` | independently worded, preregistered prompt on a balanced C1/C2 subset with the same contract/budget | required prompt dependence check; prevents a one-prompt conclusion |
| `A-ModelSize` | smaller compatible active model on a balanced C1/C2 subset | optional model-size dependence |

Temporal-only and epistemic-only ablations may be separated if the joint ablation is ambiguous. Renderer progressive disclosure and edge bundling are evaluated as visualization factors on fixed semantic ontologies; they are never counted as construction ablations.

## 22. Error taxonomy and qualitative analysis

Every failed or partially correct output receives zero or more prespecified codes:

| Family | Examples |
|---|---|
| Index/retrieval | missing passage; wrong span; alias candidate absent; horizon-ineligible evidence retrieved |
| Entity identity | over-merge; under-merge; persona/title confused; unsupported split |
| Relevance/coverage | distractor retained; answer-support node omitted; graph empty or oversized |
| Relation/schema | wrong predicate; reversed direction; inappropriate granularity; invented schema meaning |
| Event structure | missing/unnecessary reification; wrong event boundary; participant/role error |
| Temporal | story time confused with discourse/revelation; invalid interval; missed state change; false exact date |
| Epistemic | belief reported as fact; wrong holder; rumor flattened; reader/character knowledge confused; spoiler leak |
| Causal | chronology treated as causation; pivotal link missing; consequence attached to wrong event |
| Grounding | nonexistent evidence ID; cited passage not supportive; label/summary adds an unsupported claim |
| Abstraction | too concrete, too aggregate, inconsistent across nodes, or unresponsive to context |
| Rare information | rare-pivotal deletion; frequency dominance; rare-irrelevant distraction |
| Stochastic | high sample variance; paraphrase sensitivity; canned graph across contrasts |
| Constraint/runtime | invalid JSON/schema; repair semantic drift; timeout; OOM; truncated output |
| Interaction | feedback ignored, over-applied, nonlocal regression, unreplayable context state |
| Rendering | overlap/occlusion; hidden pivotal fact; unstable layout; horizon leak in tooltip/style |

A stratified blind sample is double-coded where feasible. Report code frequencies by condition, examples selected before knowing whether they favor C2, and complete case traces from evidence through output, validation, metrics, and feedback. Qualitative analysis seeks counterexamples and fixed-ontology advantages, not only demonstrations. Prompt edits prompted by an error are restricted to development data and recorded.

## 23. Validity threats and mitigations

| Threat | Consequence | Mitigation |
|---|---|---|
| Synthetic-generator bias | gold rewards the chosen representation | representation-neutral worlds, null/pregraph-friendly cases, held-out generator families, alternative golds, mutation tests, public generator |
| Curator subjectivity | case-study “gold” reflects one reading | evidence-bounded rubric, alternatives/uncertainty, blind second review subset, agreement and disagreements reported |
| Model pretraining contamination | apparent narrative knowledge is not evidence-grounded | evidence-only output contract, counterfactual synthetic names/facts, entity-permutation canaries, support audits, spoiler access logs |
| Evidence/index leakage | C2 or a condition sees privileged facts/gold | shared hashed \(D_z\), frozen packets, horizon snapshots, scorer-only gold firewall, pre-query inventory and lineage tests |
| C2 as hidden subgraph retrieval | wrong causal treatment | evidence-index boundary, construction certificate, structural contrast metrics, `A-FixedSelect`, cache audit |
| Compute asymmetry | performance reflects budget | same C1/C2 model/config, matched semantic budgets, total/amortized compute analyses, report all tokens/latency/VRAM |
| Weak baseline | inflated treatment effect | credible tuned C0, strong C1, public configurations and baseline error analysis |
| LLM stochasticity | unstable point estimates | paired seed-block samples, raw artifacts, world-clustered inference, seed/runtime pinning, paraphrase tests |
| Prompt/model dependence | narrow result | frozen prompt plus preregistered prompt/model sensitivity; bounded claims |
| Metric cherry-picking | favorable but misleading conclusion | preregistered panel/directions, multiplicity correction, publish all outcomes |
| Metric validity | entropy/clutter not usefulness | joint fidelity/rare gates; semantic-render separation; task/expert evaluation |
| Community resolution/connectivity | arbitrary cluster claim | frozen Leiden setting, resolution sweep, Infomap, algorithm versus ontology stability |
| Renderer confound | layout differences credited to semantics | fixed visual stack/seeds and semantic-versus-renderer experiments |
| Small graph/sample sizes | noisy asymptotics and low power | exact inspection, chance correction, clustered intervals, power simulation, cautious claims |
| Copyright restrictions | incomplete public reproducibility | fully public synthetic study; local-only text; hashes/locators; public code/prompts/aggregate artifacts |
| External validity | one fantasy series/general synthetic worlds | explicit domain boundary, varied synthetic factors; optional later corpora, no universal claim |
| Human-study demand effects | biased preferences | anonymization, counterbalancing, standardized training, preregistered rubric, disclose hypotheses appropriately |

## 24. Copyright, privacy, reproducibility, and research ethics

The repository contains only reproducible code and non-infringing artifacts. Synthetic narratives and worlds use an explicit permissive license. Game of Thrones text is supplied locally by the researcher; `.gitignore`, pre-commit content scans, artifact allowlists, passage-length tests, and manual release review prevent redistribution. Released locators should be no more granular than legally and scientifically necessary, and no embedding or annotation export may enable reconstruction of protected prose.

Every experiment has an append-only ledger containing code revision, environment lock and container metadata, OS/driver/CUDA/runtime versions, hardware, corpus and index hashes, model/tokenizer/quantization revisions and licenses, prompt/schema/config hashes, all seed derivations, evidence packet hashes, raw structured outputs, validation and repair traces, ontology and visualization hashes, feedback events, metrics, failures, latency, and memory. Public manifests replace restricted paths with opaque IDs and cryptographic hashes. Secrets, account names, or private filesystem paths are redacted.

Model weights and software are used under their licenses; exact revisions and license files/links are recorded rather than redistributing weights. Case-study quotations in publications are short, necessary, and reviewed under the applicable legal/institutional policy. The project does not assert that model output is authoritative literary fact and documents contested interpretations.

Human sessions begin only after applicable institutional review or an exemption determination. Participants consent to interaction logging and recording choices, may withdraw, and need not disclose sensitive demographics. Study logs use pseudonymous IDs, minimize free text, define retention/deletion rules, and separate consent records. Accessibility and spoiler warnings are part of the interface protocol.

Reproducibility has two levels: **full public reproduction** of schemas, synthetic generation, conditions, metrics, and analyses; and **authorized local reproduction** of the copyrighted case study using edition hashes and the ingestion pipeline. Claims are labeled according to which level supports them.

## 25. Expected claims and explicitly prohibited overclaims

If supported, the study may claim:

- on the registered synthetic distribution, C2 improved specified context-alignment or organization outcomes relative to C1/C0 under the recorded model and budgets;
- the timing-controlled C2--C1 difference is consistent with a benefit (or harm) from post-query ontology construction, not merely LLM use;
- C2 produced gold-required merge/split, relation, event, abstraction, and qualification changes and did not collapse to fixed selection;
- particular entropy/community structures changed and direct clutter improved or worsened subject to fidelity and rare-pivotal preservation;
- structured revisions repaired specified errors under scripted or formative human conditions;
- the method transferred, with described limitations, to selected locally evaluated narrative windows; and
- the benchmark, schemas, ledgers, and interface form a reproducible platform for further tests.

The project must not claim:

- first-ever query-driven, contextual, dynamic, LLM, or narrative ontology construction;
- that LLMs generally outperform classical ontology engineering;
- that lower entropy, fewer nodes/edges, higher modularity, or fewer/more communities is inherently better;
- usability, cognitive benefit, or trust from structural graph metrics alone;
- complete truth, causality, chronology, or character knowledge beyond the licensed evidence;
- complete coverage of the novels or generalization to all narratives, users, models, or domains;
- byte-identical GPU determinism from seeds;
- that a faithful label can introduce an unsupported fact;
- that missing rare facts are an acceptable price for a sparse graph without reporting the failure; or
- that retrieval, pruning, relabeling, summarization, or layout is query-dependent ontology construction.

Null or negative results remain valid contributions if the boundary audits, benchmark, and analysis are sound.

## 26. Acceptance criteria for moving to the case study

The synthetic phase is a gate, not a demand that C2 win. Proceed to the copyrighted case study only when all criteria pass on frozen development/acceptance fixtures:

1. All schemas, gold projections, permissible alternatives, temporal constraints, query distributions, factor/denominator coverage, splits, and review logs are versioned; generator mutation and determinism tests pass; scorer gold is process-isolated and changing gold labels cannot change a run input/output.
2. All three conditions and `A-FixedSelect` run end-to-end through the same interface; equality of \(D_z/E_q\), horizon, budgets, model hashes where applicable, C0/C1 lineage, expected grammar-difference manifests, C2 pre-query-boundary tests, and full-packet/full-sealed-graph packing without truncation pass on 100% of audited artifacts.
3. Contrastive fixtures require and detect post-query identity, schema/relation, event, abstraction, and temporal/epistemic changes; C2's development ontological-decision F1 is at least 0.80 and primary ontological-collapse rate at most 0.20.
4. C0 passes the frozen reference competencies (100% schema/evidence validity, strict assertion F1 ≥0.55, temporal F1 ≥0.60, rare-pivotal recall ≥0.60); C1 preontology schema validity is ≥0.95 and comprehensive assertion recall ≥0.70; at least 95% of development C1/C2 calls yield schema-valid projections after no more than two recorded repairs. Failure triggers development repair or stops the comparison; every remaining failure is retained.
5. Evidence-ID validity is 100%, evidence-grounding precision is at least 0.95 on the reviewed development sample, unsupported factual text is at most 0.05, and programmed spoiler-horizon leakage is zero.
6. C2 rare-pivotal qualified-assertion recall is at least 0.85 on development data; rare recall is never replaced by a sparsity score. The final non-inferiority margin is frozen.
7. Temporal validator soundness tests, inconsistent-world diagnostics, and story/discourse/revelation cross-axis tests pass; no validator silently invents an ordering.
8. The full metric panel handles empty, singleton, complete, disconnected, directed, multiedge, zero-weight, unseen-predicate/`OTHER`, pairwise-community zero-denominator, and alternative-gold fixtures; attributed false beliefs never become world facts; all formulas/configurations are frozen.
9. Three-run seed replay completes, empirical stochastic variation is summarized, and intention-to-treat failure policy is implemented.
10. Every scripted feedback action replays to the same input manifest and expected semantic delta; the interactive graph passes evidence, horizon, rich-node/edge, and projection-diff end-to-end tests.
11. A pilot completes within the RTX 4090/31 GB RAM/8-vCPU envelope, with resumable jobs and sufficient storage; the estimated full case run fits the documented resource budget.
12. The public-release dry run contains no copyrighted passage, reconstructive embedding, secret, or private path.

Thresholds 0.80, 0.20, 0.95, 0.05, and 0.85 are recommended operational readiness defaults, not inferential success criteria. They may be changed once, with written rationale, before acceptance data are examined. If C2 passes integrity/capability gates but does not outperform C1, the case study may proceed as a registered negative-transfer test; the null is not hidden.

## 27. Open decisions with recommended defaults

| Decision | Recommended default | Alternative and tradeoff | Evidence required to change |
|---|---|---|---|
| Primary LLM | Qwen3-14B-AWQ, 4-bit AWQ, non-thinking structured generation | 7--8B model lowers memory/cost but may reduce semantic capability; another 14B changes external validity | Phase 1 JSON validity, grounding, VRAM, latency, and operator-accuracy pilot on development only |
| Evidence packet | all admissible evidence for synthetic and primary bounded case windows, initially about an 8k evidence-token cap so the stricter `A-FixedSelect` envelope can also carry the sealed graph; top-24 retrieval only in the secondary realism analysis | larger windows raise coverage but can break the common input cap; smaller packets require smaller query-blind windows to avoid a retrieval confound | blocking full-packet/full-graph packing preflight plus 4090 context/latency pilot, frozen before test |
| Local schema policy | small fixed upper ontology plus LLM-defined contextual relation/types with definitions | fully open schema increases expressivity/matching cost; fixed schema weakens RQ1 | development inter-annotator/matcher reliability and `A-FixedSchema` effect |
| Community detector | Leiden-CPM with development-frozen resolution, plus resolution sweep and Infomap sensitivity | modularity-Leiden is familiar but resolution-limited | synthetic recovery and stability across gold partitions, not best test score |
| Rare threshold | one independent mention or bottom frequency quartile; pivotality separately gold-labeled | fixed absolute corpus count is easier but corpus-size dependent | development sensitivity and annotation reliability |
| Non-inferiority margin | 0.05 absolute rare-pivotal recall | tighter protects more but may be underpowered; wider tolerates meaningful loss | substantive loss analysis and pre-test power simulation |
| Gold alternatives | enumerated alternatives plus local equivalence rules | one canonical graph is simpler but penalizes defensible structure | blind manual audit and matcher reliability |
| RDF export | portable RDF 1.1 n-ary/reification + OWL-Time/PROV-O; experimental RDF 1.2 mapping | RDF 1.2 reifiers are cleaner but Candidate Recommendation/tooling risk | finalized standard status and round-trip conformance in the pinned toolchain |
| Case-study annotation | 12 windows × 4 contexts; detailed gold for 24 stratified units | fewer detailed units save time but weaken quantitative transfer analysis | annotation-time and agreement pilot before condition outputs |
| Optional formative study | 8--12 counterbalanced participants if a human-usability claim is pursued | minimum retains scripted plus documented refinement traces; a powered study costs substantially more | institutional approval, recruitment feasibility, pilot variance, and formal power analysis |
| Nested epistemics | single holder/status/time in minimum study | recursive belief worlds add realism but annotation and reasoning complexity | repeated failure on registered epistemic queries that cannot be represented otherwise |

These are genuine scientific or resource choices. None permits removal of the active query-time LLM, GPU inference, three conditions, temporal construction, contrastive synthetic benchmark, structured human refinement, narrative ingestion, or interactive graph from the minimum study.

## 28. Verified references

All entries below were checked against DOI landing pages, publisher/proceedings records, or official standards documentation during preparation. Reviews and preprints are identified as such. Sources whose bibliographic identity could not be verified were omitted rather than presented as established evidence.

1. **[R1]** Alexander Maedche and Steffen Staab. “Ontology Learning for the Semantic Web.” *IEEE Intelligent Systems* 16(2):72--79, 2001. [doi:10.1109/5254.920602](https://doi.org/10.1109/5254.920602).
2. **[R2]** Philipp Cimiano and Johanna Völker. “Text2Onto: A Framework for Ontology Learning and Data-Driven Change Discovery.” *Natural Language Processing and Information Systems (NLDB 2005)*, LNCS 3513, pp. 227--238, 2005. [doi:10.1007/11428817_21](https://doi.org/10.1007/11428817_21).
3. **[R3]** Philipp Cimiano, Andreas Hotho, and Steffen Staab. “Learning Concept Hierarchies from Text Corpora using Formal Concept Analysis.” *Journal of Artificial Intelligence Research* 24:305--339, 2005. [doi:10.1613/JAIR.1648](https://doi.org/10.1613/JAIR.1648).
4. **[R4]** Wilson Wong, Wei Liu, and Mohammed Bennamoun. “Ontology Learning from Text: A Look Back and into the Future.” *ACM Computing Surveys* 44(4), Article 20, 2012 (review). [doi:10.1145/2333112.2333115](https://doi.org/10.1145/2333112.2333115).
5. **[R5]** Paolo Bouquet, Fausto Giunchiglia, Frank van Harmelen, Luciano Serafini, and Heiner Stuckenschmidt. “Contextualizing Ontologies.” *Journal of Web Semantics* 1(4):325--343, 2004. [doi:10.1016/j.websem.2004.07.001](https://doi.org/10.1016/j.websem.2004.07.001).
6. **[R6]** Nesrine Ben Mustapha, Marie-Aude Aufaure, Hajer Baazaoui Zghal, and Henda Ben Ghézala. “Query-driven approach of contextual ontology module learning using web snippets.” *Journal of Intelligent Information Systems* 45(1):61--94, 2015 (online-first 2013). [doi:10.1007/s10844-013-0263-6](https://doi.org/10.1007/s10844-013-0263-6).
7. **[R7]** Evgeny Kharlamov, Ernesto Jiménez-Ruiz, Dmitriy Zheleznyakov, Dimitris Bilidas, Martin Giese, Peter Haase, Ian Horrocks, Herald Kllapi, Manolis Koubarakis, Özgür L. Özçep, Mariano Rodríguez-Muro, Riccardo Rosati, Michael Schmidt, Rudolf Schlatte, Ahmet Soylu, and Arild Waaler. “Optique: Towards OBDA Systems for Industry.” *The Semantic Web: ESWC 2013 Satellite Events*, LNCS 7955, pp. 125--140, 2013. [doi:10.1007/978-3-642-41242-4_11](https://doi.org/10.1007/978-3-642-41242-4_11).
8. **[R8]** James F. Allen. “Maintaining Knowledge about Temporal Intervals.” *Communications of the ACM* 26(11):832--843, 1983. [doi:10.1145/182.358434](https://doi.org/10.1145/182.358434).
9. **[R9]** Claudio Gutierrez, Carlos A. Hurtado, and Alejandro A. Vaisman. “Introducing Time into RDF.” *IEEE Transactions on Knowledge and Data Engineering* 19(2):207--218, 2007. [doi:10.1109/TKDE.2007.34](https://doi.org/10.1109/TKDE.2007.34).
10. **[R10]** Simon Cox and Chris Little, editors. “Time Ontology in OWL.” W3C Recommendation, 19 October 2017. [Official dated Recommendation](https://www.w3.org/TR/2017/REC-owl-time-20171019/).
11. **[R11]** José M. Giménez-García, Antoine Zimmermann, and Pierre Maret. “NdFluents: An Ontology for Annotated Statements with Inference Preservation.” *The Semantic Web (ESWC 2017)*, LNCS 10249, pp. 638--654, 2017. [doi:10.1007/978-3-319-58068-5_39](https://doi.org/10.1007/978-3-319-58068-5_39).
12. **[R12]** Boris Motik. “Representing and Querying Validity Time in RDF and OWL: A Logic-Based Approach.” *Journal of Web Semantics* 12--13:3--21, 2012. [doi:10.1016/j.websem.2011.11.004](https://doi.org/10.1016/j.websem.2011.11.004).
13. **[R13]** Willem Robert van Hage, Véronique Malaisé, Roxane Segers, Laura Hollink, and Guus Schreiber. “Design and use of the Simple Event Model (SEM).” *Journal of Web Semantics* 9(2):128--136, 2011. [doi:10.1016/j.websem.2011.03.003](https://doi.org/10.1016/j.websem.2011.03.003).
14. **[R14]** Simon Gottschalk and Elena Demidova. “EventKG: A Multilingual Event-Centric Temporal Knowledge Graph.” *The Semantic Web (ESWC 2018)*, LNCS 10843, pp. 272--287, 2018. [doi:10.1007/978-3-319-93417-4_18](https://doi.org/10.1007/978-3-319-93417-4_18).
15. **[R15]** Marco Rospocher, Marieke van Erp, Piek Vossen, Antske Fokkens, Itziar Aldabe, German Rigau, Aitor Soroa, Thomas Ploeger, and Tessel Bogaard. “Building Event-Centric Knowledge Graphs from News.” *Journal of Web Semantics* 37--38:132--151, 2016. [doi:10.1016/j.websem.2015.12.004](https://doi.org/10.1016/j.websem.2015.12.004).
16. **[R16]** Carlo Meghini, Valentina Bartalesi, and Daniele Metilli. “Representing Narratives in Digital Libraries: The Narrative Ontology.” *Semantic Web* 12(2):241--264, 2021. [doi:10.3233/SW-200421](https://doi.org/10.3233/SW-200421).
17. **[R17]** Anas Fahad Khan, Andrea Bellandi, Giulia Benotto, Francesca Frontini, Emiliano Giovannetti, and Marianne Reboul. “Leveraging a Narrative Ontology to Query a Literary Text.” *7th Workshop on Computational Models of Narrative (CMN 2016)*, OASIcs 53, Article 10, pp. 10:1--10:10, 2016. [doi:10.4230/OASIcs.CMN.2016.10](https://doi.org/10.4230/OASIcs.CMN.2016.10).
18. **[R18]** Jeremy J. Carroll, Christian Bizer, Pat Hayes, and Patrick Stickler. “Named Graphs, Provenance and Trust.” *Proceedings of WWW '05*, pp. 613--622, 2005. [doi:10.1145/1060745.1060835](https://doi.org/10.1145/1060745.1060835).
19. **[R19]** Natasha Noy and Alan Rector, editors. “Defining N-ary Relations on the Semantic Web.” W3C Working Group Note, 12 April 2006 (informative Note, not a Recommendation). [Official W3C Note](https://www.w3.org/TR/2006/NOTE-swbp-n-aryRelations-20060412/).
20. **[R20]** Timothy Lebo, Satya Sahoo, and Deborah McGuinness, editors. “PROV-O: The PROV Ontology.” W3C Recommendation, 30 April 2013. [Official dated Recommendation](https://www.w3.org/TR/2013/REC-prov-o-20130430/).
21. **[R21]** Antske Fokkens, Piek Vossen, Marco Rospocher, Rinke Hoekstra, and Willem Robert van Hage. “GRaSP: Grounded Representation and Source Perspective.” *Proceedings of the Workshop Knowledge Resources for the Socio-Economic Sciences and Humanities associated with RANLP 2017*, pp. 19--25, Varna: INCOMA, 2017. [doi:10.26615/978-954-452-040-3_003](https://doi.org/10.26615/978-954-452-040-3_003).
22. **[R22]** Blaž Fortuna, Marko Grobelnik, and Dunja Mladenić. “OntoGen: Semi-automatic Ontology Editor.” *Human Interface and the Management of Information. Interacting in Information Environments*, Proceedings Part II, LNCS 4558, pp. 309--318, 2007. [doi:10.1007/978-3-540-73354-6_34](https://doi.org/10.1007/978-3-540-73354-6_34).
23. **[R23]** Tania Tudorache, Csongor Nyulas, Natalya F. Noy, and Mark A. Musen. “WebProtégé: A Collaborative Ontology Editor and Knowledge Acquisition Tool for the Web.” *Semantic Web* 4(1):89--99, 2013. [doi:10.3233/SW-2012-0057](https://doi.org/10.3233/SW-2012-0057).
24. **[R24]** Feng Shi, Juanzi Li, Jie Tang, Guotong Xie, and Hanyu Li. “Actively Learning Ontology Matching via User Interaction.” *The Semantic Web (ISWC 2009)*, LNCS 5823, pp. 585--600, 2009. [doi:10.1007/978-3-642-04930-9_37](https://doi.org/10.1007/978-3-642-04930-9_37).
25. **[R25]** Saleema Amershi, Maya Cakmak, W. Bradley Knox, and Todd Kulesza. “Power to the People: The Role of Humans in Interactive Machine Learning.” *AI Magazine* 35(4):105--120, 2014. [doi:10.1609/aimag.v35i4.2513](https://doi.org/10.1609/aimag.v35i4.2513).
26. **[R26]** Hamed Babaei Giglou, Jennifer D'Souza, and Sören Auer. “LLMs4OL: Large Language Models for Ontology Learning.” *The Semantic Web (ISWC 2023)*, LNCS 14265, pp. 408--427, 2023. [doi:10.1007/978-3-031-47240-4_22](https://doi.org/10.1007/978-3-031-47240-4_22).
27. **[R27]** J. Harry Caufield, Harshad Hegde, Vincent Emonet, Nomi L. Harris, Marcin P. Joachimiak, Nicolas Matentzoglu, HyeongSik Kim, Sierra A. T. Moxon, Justin T. Reese, Melissa A. Haendel, Peter N. Robinson, and Christopher J. Mungall. “Structured Prompt Interrogation and Recursive Extraction of Semantics (SPIRES): a method for populating knowledge bases using zero-shot learning.” *Bioinformatics* 40(3):btae104, 2024. [doi:10.1093/bioinformatics/btae104](https://doi.org/10.1093/bioinformatics/btae104).
28. **[R28]** Hanzhu Chen, Xu Shen, Qitan Lv, Jie Wang, Xiaoqi Ni, and Jieping Ye. “SAC-KG: Exploiting Large Language Models as Skilled Automatic Constructors for Domain Knowledge Graph.” *Proceedings of ACL 2024*, pp. 4345--4360, 2024. [doi:10.18653/v1/2024.acl-long.238](https://doi.org/10.18653/v1/2024.acl-long.238).
29. **[R29]** Patrick Lewis, Ethan Perez, Aleksandra Piktus, Fabio Petroni, Vladimir Karpukhin, Naman Goyal, Heinrich Küttler, Mike Lewis, Wen-tau Yih, Tim Rocktäschel, Sebastian Riedel, and Douwe Kiela. “Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks.” *Advances in Neural Information Processing Systems 33*, 2020. [Official NeurIPS record](https://papers.neurips.cc/paper/2020/hash/6b493230205f780e1bc26945df7481e5-Abstract.html).
30. **[R30]** Darren Edge, Ha Trinh, Newman Cheng, Joshua Bradley, Alex Chao, Apurva Mody, Steven Truitt, Dasha Metropolitansky, Robert Osazuwa Ness, and Jonathan Larson. “From Local to Global: A Graph RAG Approach to Query-Focused Summarization.” arXiv:2404.16130, 2024, revised 2025 (preprint, not peer reviewed). [doi:10.48550/arXiv.2404.16130](https://doi.org/10.48550/arXiv.2404.16130).
31. **[R31]** Xiaohui Zhang, Zequn Sun, Chengyuan Yang, Yuanning Cui, Lingbing Guo, and Wei Hu. “Toward Effective and Reliable LLM Agents via Dynamic Ontology.” arXiv:2608.22974, submitted 24 August 2026 (preprint, not peer reviewed). [doi:10.48550/arXiv.2608.22974](https://doi.org/10.48550/arXiv.2608.22974).
32. **[R32]** Fabian Beck, Michael Burch, Stephan Diehl, and Daniel Weiskopf. “A Taxonomy and Survey of Dynamic Graph Visualization.” *Computer Graphics Forum* 36(1):133--159, 2017 (review; version of record online 2016). [doi:10.1111/cgf.12791](https://doi.org/10.1111/cgf.12791).
33. **[R33]** Jae-wook Ahn, Catherine Plaisant, and Ben Shneiderman. “A Task Taxonomy for Network Evolution Analysis.” *IEEE Transactions on Visualization and Computer Graphics* 20(3):365--376, 2014. [doi:10.1109/TVCG.2013.238](https://doi.org/10.1109/TVCG.2013.238).
34. **[R34]** Emden R. Gansner, Yehuda Koren, and Stephen C. North. “Topological Fisheye Views for Visualizing Large Graphs.” *IEEE Transactions on Visualization and Computer Graphics* 11(4):457--468, 2005. [doi:10.1109/TVCG.2005.66](https://doi.org/10.1109/TVCG.2005.66).
35. **[R35]** Vitalis Wiens, Steffen Lohmann, and Sören Auer. “Semantic Zooming for Ontology Graph Visualizations.” *Proceedings of the Knowledge Capture Conference (K-CAP 2017)*, Article 4, pp. 1--8, 2017. [doi:10.1145/3148011.3148015](https://doi.org/10.1145/3148011.3148015).
36. **[R36]** Helen C. Purchase. “Metrics for Graph Drawing Aesthetics.” *Journal of Visual Languages & Computing* 13(5):501--516, 2002. [doi:10.1006/jvlc.2002.0232](https://doi.org/10.1006/jvlc.2002.0232).
37. **[R37]** Colin Ware, Helen Purchase, Linda Colpoys, and Matthew McGill. “Cognitive Measurements of Graph Aesthetics.” *Information Visualization* 1(2):103--110, 2002. [doi:10.1057/palgrave.ivs.9500013](https://doi.org/10.1057/palgrave.ivs.9500013).
38. **[R38]** Mohammad Ghoniem, Jean-Daniel Fekete, and Philippe Castagliola. “On the Readability of Graphs Using Node-Link and Matrix-Based Representations: A Controlled Experiment and Statistical Analysis.” *Information Visualization* 4(2):114--135, 2005. [doi:10.1057/palgrave.ivs.9500092](https://doi.org/10.1057/palgrave.ivs.9500092).
39. **[R39]** Matthias Dehmer and Abbe Mowshowitz. “A History of Graph Entropy Measures.” *Information Sciences* 181(1):57--78, 2011 (review). [doi:10.1016/j.ins.2010.08.041](https://doi.org/10.1016/j.ins.2010.08.041).
40. **[R40]** Kartik Anand and Ginestra Bianconi. “Entropy Measures for Networks: Toward an Information Theory of Complex Topologies.” *Physical Review E* 80:045102(R), 2009. [doi:10.1103/PhysRevE.80.045102](https://doi.org/10.1103/PhysRevE.80.045102).
41. **[R41]** Lin Han, Francisco Escolano, Edwin R. Hancock, and Richard C. Wilson. “Graph Characterizations from von Neumann Entropy.” *Pattern Recognition Letters* 33(15):1958--1967, 2012. [doi:10.1016/j.patrec.2012.03.016](https://doi.org/10.1016/j.patrec.2012.03.016).
42. **[R42]** M. E. J. Newman and M. Girvan. “Finding and Evaluating Community Structure in Networks.” *Physical Review E* 69:026113, 2004. [doi:10.1103/PhysRevE.69.026113](https://doi.org/10.1103/PhysRevE.69.026113).
43. **[R43]** Santo Fortunato. “Community Detection in Graphs.” *Physics Reports* 486(3--5):75--174, 2010 (review). [doi:10.1016/j.physrep.2009.11.002](https://doi.org/10.1016/j.physrep.2009.11.002).
44. **[R44]** Vincent A. Traag, Ludo Waltman, and Nees Jan van Eck. “From Louvain to Leiden: Guaranteeing Well-Connected Communities.” *Scientific Reports* 9:5233, 2019. [doi:10.1038/s41598-019-41695-z](https://doi.org/10.1038/s41598-019-41695-z).
45. **[R45]** Santo Fortunato and Marc Barthélemy. “Resolution Limit in Community Detection.” *Proceedings of the National Academy of Sciences* 104(1):36--41, 2007. [doi:10.1073/pnas.0605965104](https://doi.org/10.1073/pnas.0605965104).
46. **[R46]** Nguyen Xuan Vinh, Julien Epps, and James Bailey. “Information Theoretic Measures for Clusterings Comparison: Variants, Properties, Normalization and Correction for Chance.” *Journal of Machine Learning Research* 11:2837--2854, 2010. [Official JMLR record](https://www.jmlr.org/papers/v11/vinh10a.html).
47. **[R47]** Martin Rosvall and Carl T. Bergstrom. “Maps of Random Walks on Complex Networks Reveal Community Structure.” *Proceedings of the National Academy of Sciences* 105(4):1118--1123, 2008. [doi:10.1073/pnas.0706851105](https://doi.org/10.1073/pnas.0706851105).
48. **[R48]** Xinyi Pan, Daniel Hernández, Philipp Seifer, Ralf Lämmel, and Steffen Staab. “eSPARQL: Representing and Reconciling Agnostic and Atheistic Beliefs in RDF-star Knowledge Graphs.” *The Semantic Web -- ISWC 2024*, LNCS 15232, pp. 155--172, 2024. [doi:10.1007/978-3-031-77850-6_9](https://doi.org/10.1007/978-3-031-77850-6_9).
49. **[R49]** Richard Cyganiak, David Wood, and Markus Lanthaler, editors. “RDF 1.1 Concepts and Abstract Syntax.” W3C Recommendation, 25 February 2014. [Official specification](https://www.w3.org/TR/2014/REC-rdf11-concepts-20140225/).
50. **[R50]** Olaf Hartig, Pierre-Antoine Champin, Gregg Kellogg, and Andy Seaborne, editors. “RDF-star and SPARQL-star.” W3C RDF-DEV Community Group Final Report, 17 December 2021 (not a Recommendation). [Official report](https://www.w3.org/2021/12/rdf-star.html).
51. **[R51]** Gregg Kellogg, Olaf Hartig, Pierre-Antoine Champin, and Andy Seaborne, editors. “RDF 1.2 Concepts and Abstract Data Model.” W3C Candidate Recommendation Snapshot, 7 April 2026 (not a Recommendation as of this plan). [Official specification](https://www.w3.org/TR/2026/CR-rdf12-concepts-20260407/).

### Standards-status note

RDF 1.1 remains the latest RDF Recommendation used for the portable export [R49]. The 2021 RDF-star report is a Community Group Final Report, not a Recommendation [R50]. As of 2026-09-03, RDF 1.2 Concepts and Abstract Data Model is a Candidate Recommendation Snapshot, not a finalized Recommendation [R51]. These status distinctions are intentional.
