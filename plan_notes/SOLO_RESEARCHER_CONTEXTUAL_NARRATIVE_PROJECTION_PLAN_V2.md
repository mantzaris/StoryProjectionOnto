# Admissibility-Constrained Transition-Bundle Projection: A Solo-Researcher Fixed-Benchmark Plan

Literature status checked through 2 September 2026. Publication-status labels distinguish standards, peer-reviewed papers, and preprints.

## 1. Executive decision

Paper 1 will study one sharply defined algorithmic object: **admissibility-constrained transition-bundle projection under componentwise semantic display budgets**. Given a manually validated, episode-bounded temporal-epistemic graph and a typed query context, the method selects a small closed projection. Every displayed answer claim must remain within the evidence horizon, retain its evidence and epistemic qualification, be temporally satisfiable, preserve identity, and fit identical component caps. Within that admissible family, the proposed objective gives lexicographic priority to complete, query-relevant state-transition bundles.

This is a fixed-benchmark oracle-graph study, not an extraction, population, or visualization study. Proposed-Full is compared primarily with guard-PCST; frequency and PPR are secondary. Every baseline has a horizon-filtered native form and a fully guard-wrapped form. Proposed-NoTransitionPriority isolates the mechanism.

The corpus has seven bounded bundles: four held-out *A Game of Thrones* (AGOT) bundles, one public-domain development bundle, one different public-domain held-out bundle, and one hidden control. The control is not efficacy evidence. Paper 1 requires neither a human nor an LLM study. Required work is 150 researcher hours, leaving 10 hours before the hard stop.

Any conclusion is limited to the frozen cases; it cannot cover all AGOT, all fiction, reader comprehension, or unrestricted fictional-world truth.

## 2. Revised Paper 1 scope

**Required:** minimal ontology and context formalization; exact controlled-dense projection; verifier corruptions; six natural oracle bundles; horizon rarity audits; three native/wrapped baseline families; direct ablation; three budgets; and per-query reporting.

**Required sensitivity:** all three frozen budget levels. **Optional after completion:** one-component budget perturbations and six paraphrases. **Deferred:** extraction, adaptive labels, feedback, human and LLM studies, profiles, ontology learning, and large-graph heuristics. Full-novel annotation and broad visualization comparisons require additional personnel.

Layout is fixed. Paper 1 evaluates semantic selection and word/component costs, not readability or visual insight.

## 3. Research problem and fixed-benchmark research question

Ordinary graph pruning favors repeated or structurally central relations. A single capture, appointment, transfer, discovery, betrayal, or death can instead change later states and be necessary to explain an answer. Query retrieval can also return a relevant event while omitting its before-state, after-state, qualification, or evidence. These are different failures: one concerns utility, the other semantic admissibility.

The fixed-benchmark question is:

> Under identical componentwise budgets and an identical temporal, epistemic, evidence, identity, and horizon contract, does transition-bundle priority improve support-complete retention of query-relevant, horizon-rare consequential state changes over guard-PCST on the five real-text held-out bundles?

Identical budgets cap roots, entities, events, transitions, states, assertions, relations, evidence cues, and words. Identical contract means shared eligibility, closure, verifier, and response schema. Support completeness retains all indispensable atoms under a frozen rubric. Horizon rarity uses all verified pre-horizon mentions. Consequence uses source-verifiable change and downstream effects independent of gold support. Held-out means all rubrics, templates, coefficients, budgets, and baseline settings are frozen first.

## 4. Claims, non-claims, and contribution type

The contribution combines: (1) closed-root selection under vector budgets, semantic guards, and a response schema; (2) a verifier with typed failure certificates; (3) transition-bundle priority; (4) a matched-guard, controlled-dense benchmark protocol; and (5) an AGOT case study with one independent public-domain test.

The paper does not claim the first narrative ontology, temporal narrative graph, contextual graph, GraphRAG retriever, spoiler filter, event salience measure, or evidence-grounded selector. It does not establish fictional-world truth from textual occurrence. It does not estimate expected performance over the novel or narrative genre. It does not test extraction accuracy, human comprehension, layout, scalability beyond the root cap, or an LLM's contribution.

H1 is that Full passes the contract and the Section 21 advantage rule over guard-PCST. H2 asks whether removing only transition priority removes that advantage. Native violations, diagnostic cells, budgets, and public transfer are secondary.

## 5. Updated literature and novelty matrix

Most components have precedents: GOLEM, Narrative Ontology, and Drammar; GRaSP/CRMinf; SEM/OWL-Time; temporal KGQA/TimeR4; NWM, NKW, Context-KG, Zep, and GraphRAG (preprints); peer-reviewed E2RAG, G-Retriever, fine-grained GraphRAG, QAFD-RAG, and Story Ribbons; and GLIMPSE/personalized summarization. Salience, turning-point, and kernel-event studies operationalize importance beyond raw frequency.

The provisional gap is the evaluated conjunction of a verifier-backed contract, matched guards and vector budgets, complete transition/evidence selection, and horizon rare-consequence diagnostics. Usefulness must be demonstrated.

Codes are Y (central), P (partial/adjacent), and N (absent from the contribution). Grouping is avoided where it would transfer features between works.

| Work | Narrative ontology | Story time | Discourse order | Revelation/horizon | Epistemic holder/status | Evidence/provenance |
|---|---:|---:|---:|---:|---:|---:|
| GOLEM (peer reviewed) | Y | P | P | P | P | P |
| Narrative Ontology (peer reviewed) | Y | Y | Y | P | N | P |
| Drammar (peer reviewed) | Y | P | P | N | Y | P |
| GRaSP / CRMinf (workshop / standard) | N | N | P | N | Y | Y |
| NWM (preprint) | Y | Y | P | Y | P | Y |
| NKW (preprint) | Y | P | P | P | P | Y |
| E2RAG (peer reviewed) | N | Y | N | N | N | P |
| Temporal KGQA / TimeR4 (peer reviewed) | N | Y | N | P | N | P |
| Proposed Paper 1 | Y | Y | Y | Y | Y | Y |

| Work | Query projection | Hard spoiler rule | Semantic verifier | Matched guards | Componentwise display budget | Complete answer support | Complete transition bundle |
|---|---:|---:|---:|---:|---:|---:|---:|
| NWM | Y | Y | P | P | P | P | N |
| NKW | Y | P | P | N | P | P | P |
| E2RAG | Y | N | P | N | N | N | N |
| G-Retriever (peer reviewed) | Y | N | N | N | N | N | N |
| GraphRAG (preprint) | Y | N | N | N | N | N | N |
| Fine-grained GraphRAG (peer reviewed) | Y | N | N | N | N | N | N |
| QAFD-RAG (peer reviewed) | Y | N | P | N | N | N | N |
| GLIMPSE / personalized KG summary | Y | N | N | N | N | N | N |
| Context-KG (preprint) | Y | N | N | N | N | N | N |
| Story Ribbons (peer reviewed) | P | N | N | N | N | N | N |
| Proposed Paper 1 | Y | Y | Y | Y | Y | Y | Y |

| Work | Horizon rarity | Rare-consequence protection | Controlled dense distractors | Frozen benchmark | Human correction | Graph visualization | Human reasoning evaluation |
|---|---:|---:|---:|---:|---:|---:|---:|
| NWM / NKW / E2RAG | N | N | N | Y | N | N | N |
| G-Retriever / GraphRAG / fine-grained GraphRAG | N | N | N | Y | N | N | N |
| QAFD-RAG | N | N | P | Y | N | N | N |
| GLIMPSE / personalized summary | N | N | N | Y | N | N | N |
| Context-KG | N | N | N | Y | N | Y | P |
| Story Ribbons | N | N | N | Y | N | Y | P |
| Proposed Paper 1 | Y | Y | Y | Y | N | N | N |

Partial cells matter. NWM has strict chapter cutoff and matched evidence budget, not a general semantic verifier or display vector. NKW has channel caps and an actor/scope/polarity/time audit. E2RAG is text-grounded without formal provenance; QAFD guarantees relevant-subgraph recovery under assumptions, not admissibility. Context-KG preferences and Story Ribbons customization are not canonical correction. Fixed benchmarks are established. The possible novelty is the optimization/verification conjunction and diagnostic protocol; the proposed row is a plan, not a priority claim.

## 6. Minimal ontology and epistemic policy

### Serialization-neutral graph

The oracle is a typed finite graph

$$
G=(V,R,\tau,\alpha,\Pi),
$$

where \(V\) contains records, \(R\) typed relations, \(\tau\) record types, \(\alpha\) typed attributes, and \(\Pi\) provenance/evidence links. The core record types are:

| Type | Identity criterion | Required content |
|---|---|---|
| Entity | Same intended diegetic referent under a versioned alias decision | identifier, type, aliases used in audit |
| EventOccurrence | Same particular occurrence, compatible participants and story-time constraints | type, participants, story interval/order |
| State | Same bearer, dimension, value, and validity extent | bearer, dimension, value, interval if known |
| Transition | Same occurrence changing one state dimension from before to after | trigger/event, before-state, after-state |
| Proposition | Same normalized content, independent of who voices it | predicate/content and arguments |
| Assertion | One holder/source taking a stance toward one proposition in a discourse occurrence | proposition, holder/source, status, polarity, horizon |
| EvidenceAnchor | One lawful passage locator or span record | work/version, passage locator, discourse position |

Mentions are created only for identity or rarity audits. Deep discourse acts, themes, intentions, counterfactual structures, and nested attitudes are deferred; required nested reports become linked attributed assertions. Unknown, disputed, unresolved, and not-applicable values are permitted.

Core relations include participant, hasBefore, hasAfter, expresses, heldBy, evidencedBy, supports, attacks, before, overlaps, validDuring, revealedAt, and derivedFrom. Every selected root has provenance. A displayed binary edge is explicitly a presentation shortcut over an event, state, transition, or assertion bundle; it is never treated as the canonical fact.

### Epistemic policy

An evidence anchor establishes that text presents something, not unrestricted fictional-world truth. Each assertion receives the smallest applicable status set:

- *presented-as-established* by a narrator or source at the horizon;
- *observed*, *believed*, *reported*, *rumored*, *remembered*, *inferred*, *denied*, *refuted*, *disputed*, or *hypothetical*;
- polarity positive, negative, or unresolved;
- curator status *supported-under-policy*, *refuted-under-policy*, *both*, or *undetermined*.

Narrator presentation, character belief, and curator-established-under-policy are distinct from omniscient truth. Unresolved alternatives remain separate. Rumor promotion means displaying \(p\) without holder/status when only rumor, report, belief, or dispute is eligible.

Temporal fields are story interval/order, discourse and revelation positions, and required validity intervals. Relations are before, after (normalized), equal, overlaps, during, contains, and unknown; disjunctions are allowed sets. A projection is inconsistent iff no interval/order assignment satisfies all selected constraints. Flashbacks/memories separate discourse mention from occurrence; dreams, prophecies, and hypotheticals remain modal assertions.

The model is serialization-neutral. RDF can realize it with event nodes, n-ary relations, named graphs or RDF 1.2 reifiers, OWL-Time, PROV-O, and Web Annotation. RDF is not atemporal, but an unqualified triple omits these contexts; a quad's fourth term is a graph name, not automatically time.

## 7. Minimal typed context

A context is

$$
C=(q,\kappa,F,W_s,W_d,h,H,\ell),
$$

where \(q\) is a natural-language query, \(\kappa\) one of four task schemas, \(F\) focal entities or event/state types, \(W_s\) an optional story-time window, \(W_d\) an optional discourse/revelation window, \(h\) an optional epistemic holder/viewpoint, \(H\) the spoiler/evidence horizon, and \(\ell\) the detail level mapped to a budget. The four schemas are:

1. transition explanation: what changed and why;
2. temporal ordering: which occurrence preceded or overlapped another;
3. epistemic discrimination: who believed, reported, disputed, or learned what;
4. relationship/state tracing: how role, allegiance, possession, location, office, relation, knowledge, or survival changed.

Themes, expertise, learned preferences, user history, random lenses, and longitudinal personalization are out of scope. Context compilation is manual and frozen. An ambiguity record lists two plausible parses when the text of \(q\) cannot determine one; it is not silently resolved by the selector.

## 8. Admissibility family and response-feasible family

### Atoms, roots, and closure

Let \(U_C\) be horizon-eligible atoms and \(R_C\subseteq U_C\) selectable roots. A root is an event, transition, assertion, or contextual relation. A choice assignment \(\eta\) selects one of the root's declared evidence/support alternatives. Dependencies \(D_C(r,\eta)\) include identity, interpretive participants, assertion status, required time constraints, provenance, and that chosen evidence. For \(X\subseteq R_C\), \(\operatorname{cl}_C(X,\eta)\) is the least dependency fixed point. The display is its lossless presentation mapping.

### Two nested families

Let \(\mathbf c(S)\) be the component cost vector and \(\mathbf B\) the cap vector. Semantic admissibility is:

$$
\mathcal A_{\mathrm{adm}}(G,C,\mathbf B)=
\{S=\operatorname{cl}_C(X,\eta),\;X\subseteq R_C:
\mathbf c(S)\preceq\mathbf B,\;
Evid(S), Hor_H(S), Epi(S), Temp(S), Id(S), Prov(S), GuardSup(S)\}.
$$

\(Evid\) grounds claims; \(Hor_H\) excludes post-horizon content and labels; \(Epi\) preserves holder/status/polarity; \(Temp\) requires satisfiability; \(Id\) blocks unsafe merges; and \(Prov\) records provenance. \(GuardSup\) is a frozen, method-visible proof-schema check for every displayed answer claim. It differs from outcome-only \(GoldSup\); an attacks edge alone never proves a negation. Reproducibility is operator-level: frozen versions of \(G,C,\mathbf B,\theta,\eta\) and tie rules must return the same \(S\).

The response-feasible family is:

$$
\mathcal A_{\mathrm{resp}}(G,C,\mathbf B)=
\{S\in\mathcal A_{\mathrm{adm}}(G,C,\mathbf B): Resp_\kappa(S,C)\}.
$$

\(Resp_\kappa\) demands typed slots but does not reveal the gold target identity or \(GoldSup\); any method-visible candidate may fill them. Thus a graph may be safe but nonresponsive, or responsive yet miss the scored target. Correct abstention establishes an empty response family; false abstention occurs when it is nonempty.

Failure classes are mutually exclusive. INVALID_CONTEXT is a pre-projection raw-query result. With \(\mathbf B_\infty\) removing display caps: CONTENT_ABSENT means the required in-horizon candidate content/evidence set is empty; CONTRACT_BLOCKED means candidates exist but \(\mathcal A_{\mathrm{resp}}(G,C,\mathbf B_\infty)=\varnothing\); BUDGET_BLOCKED means that unbounded family is nonempty but the requested-budget family is empty. Contract certificates give minimal failed-guard sets. Budget certificates return all Pareto-minimal increment vectors, or one reproducibly lexicographic witness. ERROR_OR_UNKNOWN is solver status, never an infeasibility certificate.

## 9. Formal projection objective

The proposed operator is

$$
P(G,C,\mathbf B;\theta)\rightarrow
\begin{cases}
S^*,&\mathcal A_{\mathrm{resp}}\ne\varnothing,\\
\operatorname{Abstain}(\gamma),&\mathcal A_{\mathrm{resp}}=\varnothing,
\end{cases}
$$

with

$$
S^*=\operatorname*{lex\,arg\,max}_{S\in\mathcal A_{\mathrm{resp}}}
\langle J_{TB}(S,C),J_{rel}(S,C),J_{cov}(S,C),
-J_{red}(S),-J_{cost}(S),-J_{tie}(S)\rangle .
$$

Let \(\mathcal Z_C\) be method-visible candidate transition bundles and \(K_C\) response slots. With deterministic typed-match score \(\widehat r_C(r)\in\{0,1,2,3\}\):

$$
\begin{aligned}
J_{TB}&=\sum_{z\in\mathcal Z_C}\widehat r_C(z)\mathbb 1[z\subseteq S],&
J_{rel}&=\sum_{r\in roots(S)}\widehat r_C(r),\\
J_{cov}&=\sum_{k\in K_C}\mathbb 1[S\text{ fills }k],&
J_{red}&=\sum_{\{r,s\}\subseteq roots(S)}\mathbb 1[r,s\text{ share a frozen redundancy class}],\\
J_{cost}&=\sum_i c_i(S)/B_i.&
\end{aligned}
$$

A transition bundle contains the transition, bearer/dimension, warranted before/after states, qualification, and guard-valid evidence. \(J_{tie}\) orders stable root identifiers. PCST root prize is \(2\widehat r_C(r)\) plus the number of newly covered slots; connector cost is its exact normalized incremental component cost. All rational definitions freeze on development. Rarity, consequence, and \(GoldSup\) are absent.

Neither HorizonRarity, consequence labels, diagnostic cell membership, nor held-out gold support paths is an input to the selector. This prevents the evaluation labels from leaking into the optimization. PCST may select a compact connected root structure and is both the strongest comparator and a connector formulation; it does not receive inferior guards.

At most two roots per candidate graph may have alternative evidence structures, and each has at most two alternatives. Thus a root subset expands to at most four evidence variants. A new support structure may receive masked validation under Section 14 without changing selection.

## 10. Formal propositions and complexity

Paper 1 will prove only the following abstract claims:

1. **Least closure.** For fixed \(\eta\), monotone dependencies on finite atoms make \(\operatorname{cl}_C(\cdot,\eta)\) extensive, monotone, and idempotent. For reachability dependencies and compatible choices, closure preserves union.
2. **Horizon-inheritance proposition.** If every dependency edge is horizon-nonincreasing and every root is horizon-eligible, its closure is horizon-safe. If this condition fails, horizon safety must be rechecked after closure.
3. **Verifier composition.** Sound category verifiers jointly establish \(\mathcal A_{\mathrm{adm}}\); this does not establish annotation or implementation correctness.
4. **Exactness.** Complete roots/choices, deterministic verification, exact connector scoring, and exhaustive lexicographic comparison imply zero gap for that finite instance.
5. **Budget monotonicity.** If \(\mathbf B\preceq\mathbf B'\), then both admissible and response-feasible families under \(\mathbf B\) are subsets of their counterparts under \(\mathbf B'\).
6. **Complexity.** Response-feasible selection is NP-hard by restriction from knapsack when closures are disjoint and from prize-collecting Steiner tree when connectivity is required. This is a property of the abstract problem, not a scalability result.
7. **Restricted approximation.** With disjoint unit-cost roots, no connectivity, and monotone submodular relevance/coverage, greedy selection has the standard \(1-1/e\) guarantee under a cardinality constraint. Mandatory overlapping closures, vector caps, temporal/epistemic feasibility, lexicographic transition priority, and Steiner connectivity invalidate that direct guarantee.

With exactly 18 roots and fixed caps 4, 6, and 8, there are 4,048, 31,180, and 106,762 subsets (262,144 unrestricted). At most four \(\eta\) variants per subset yield at most 427,048 high-budget cases per context. Enumeration recomputes closure, guards, and costs. For PCST scoring, every candidate receives an exact minimum connector forest within the 18-root universe, with every connector atom charged to \(\mathbf c(S)\). All wrapped methods reuse the table; larger, unbounded, or heuristic-connector instances are ineligible for a zero-gap claim.

Formal proof correctness, implementation correctness, oracle correctness, empirical utility, and future scalability are reported separately.

## 11. Corpus and independent development/test allocation

The corpus contains exactly seven bundles:

| Allocation | Work | Role | Primary superiority evidence? |
|---|---|---|---:|
| 4 bundles | *A Game of Thrones* | Held-out central case study, one from each discourse quartile | Yes |
| 1 bundle | Wilkie Collins, *The Moonstone* | Public-domain development only | No |
| 1 bundle | Arthur Conan Doyle, *The Hound of the Baskervilles* | Independent public-domain held-out transfer | Yes |
| 1 bundle | Researcher-authored hidden narrative | Contamination controls, exact chronology, verifier tests | No |

The development work fixes every handbook, template, budget, coefficient, tie break, and baseline setting. The distinct-author public test stays sealed until these are timestamped. The 1,800 to 2,200 word hidden story is never external-validity evidence.

Each natural-text bundle contains a 1,800 to 2,400 word focal episode plus at most 600 words of distant prerequisite anchors, with a hard 3,000-word annotated cap. Across six natural bundles, the annotated text is approximately 15,000 to 18,000 words; lawful locators rather than copyrighted passages are released for AGOT.

Eligible episodes contain an explicit state change, at least two perspectives/sources, a time-revelation distinction, and a multi-item explanation. One is seed-selected per AGOT discourse quartile. Public bundles use the same rule. Stratification is coverage, not representativeness.

Across AGOT, coverage includes rare and frequent consequential events, rumor/mistaken knowledge, revelation/story-order divergence, a typed state change, and multi-step explanation. These do not justify outcome-dependent replacement.

## 12. Episode annotation and dense candidate-pool construction

Each natural bundle has at most 27 roots: approximately 11 events, 7 transitions, and 9 assertions, with about 14 states, 18 to 22 anchors, and 10 to 14 entities. This is a cap, not a quota. A bundle below 25 defensible roots is replaced before queries; no padding is allowed. Six bundles yield at most 162 roots; the hidden control has at most 18.

An episode bundle is a focal passage plus prerequisite anchors and a boundary memorandum. It is sufficiently complete when its four competency queries are answerable where expected, each answer has one complete support structure, and backward/forward audit finds no omitted in-horizon passage that changes the answer or qualification. Sufficiency is query-relative, not book-exhaustive.

Every one of the 32 answerable natural-text contexts receives a controlled-dense graph of exactly 18 roots drawn only from annotated, same-work, horizon-safe material. A family may reuse its graph for a context variant only if all 18 roots remain eligible:

- 4 focal or potentially supporting roots;
- 3 frequent but weakly relevant interactions;
- 3 unrelated state transitions;
- 3 repeated low-information assertions;
- 2 horizon-rare but irrelevant structures;
- 2 structurally central but query-irrelevant roots;
- 1 plausible competing but incomplete support structure.

At least 9 roots are irrelevant or merely contextual. Same-type distractors prevent trivial type matching. All retain true participants, times, status, and evidence; cross-work and nonsense records are prohibited. Reused records must be same-work, pre-horizon, and interpretable. Assembly never uses method ranks.

Selection pressure is mandatory: the 18-root pool is three times the medium root cap and more than twice the high cap. Closed candidates contain at least 9 events, 6 transitions, 6 assertions, 12 evidence cues, and 210 potential words, each three times medium. Overlap through closure permits these counts. Before any run, a failing family is replaced by the next eligible family under a frozen discourse-order rule; counts stay fixed and no method result is known. The release reports components, density \(2|E|/[|V|(|V|-1)]\), pool ratios, distractor classes, and source. This is controlled density, not a complete book graph.

## 13. Horizon-level and local rarity

Two noninterchangeable variables are recorded for every audited transition \(e\):

$$
LocalCount(e)=\#\{\text{verified co-referring mentions in its episode bundle}\},
$$

$$
HorizonCount(e,H)=\#\{\text{verified co-referring mentions in the same work at discourse positions}\le H\}.
$$

LocalRarity is a sensitivity variable. The primary rare-consequence analysis uses HorizonRarity: *horizon singleton* if \(HorizonCount=1\), *horizon repeated* if \(HorizonCount\ge3\), and *intermediate* if the count is 2. The gap keeps borderline cases out of the rare-versus-frequent diagnostic. Rarity concerns mentions of a particular transition occurrence, not the frequency of its broad event type.

A passage co-refers to the same transition only when it depicts, reports, recalls, or summarizes the same occurrence, with compatible story time, core participants, state dimension, and before/after values. Repeated descriptions, later recollections, reports, and summaries count as additional mentions of that occurrence. A downstream consequence does not. A different occurrence of the same type does not. Repeated relations between the same participants at different times do not. A vague allusion counts only when the frozen identity rubric licenses that occurrence without relying on the target label.

For at most 32 transitions, the researcher audits the full available horizon without building a full ontology:

1. form a frozen search packet containing canonical names and aliases, event descriptions and paraphrases, participant combinations, state dimensions and values, source/speaker combinations, and salient lexical forms;
2. search all lawfully available text through \(H\) with lexical, lemmatized, and semantic candidate retrieval;
3. manually inspect every candidate hit and a fixed context window;
4. classify it as same-occurrence depiction, report/recollection/summary, consequence only, same-type different occurrence, repeated relation, ambiguous, or nonmatch;
5. retain passage locators, decisions, and reasons in an audit ledger.

Search proposes candidates; manual review establishes counts. After 14 days, eight stratified transitions are fully re-audited. Required stability is mention-decision F1 at least 0.85 and identical rarity category for at least 7 of 8. Checks reject post-horizon hits, bad aliases, and duplicate spans. Allocation: 12 hours.

If stability fails, the paper drops the horizon-rare claim and uses the narrower phrase *locally singleton within the annotated episode*. It may still report uncertain horizon counts descriptively, but cannot present them as verified long-narrative rarity.

## 14. Consequence, relevance, and support

### Intrinsic consequence

Consequence is deliberately separated from rarity, query relevance, and answer support. A transition receives a frozen source-verifiable score:

$$
Conseq(e)=M(e)+P(e)+D(e),\qquad M,P,D\in\{0,1,2\}.
$$

- \(M\), state-change magnitude: 0 no durable value change; 1 one bounded change in knowledge, possession, location, role, relation, goal, allegiance, office, or status; 2 survival change or simultaneous change in at least two such dimensions.
- \(P\), persistence: 0 reversed within the episode; 1 persists beyond it; 2 remains operative at two later verified checkpoints by the frozen consequence horizon \(H_c\).
- \(D\), downstream consequence: 0 none verified by \(H_c\); 1 one later event/state explicitly depends on or responds to it; 2 at least two distinct consequences.

A consequential transition scores at least 4 with \(M\ge1,D\ge1\), and every nonzero item has a locator. For primary tuples \(H_c=H\), so later spoilers cannot define the target. Consequence is rated from deidentified source packets before HorizonCount is revealed and before any method run. Query wording, gold paths, prominence, centrality, and frequency are excluded.

### Query relevance and support

Query relevance is independently labeled 0 irrelevant, 1 contextual, 2 directly addresses a requested slot, or 3 indispensable to at least one accepted response. The selector sees frozen non-gold query features used to approximate relevance, but not the held-out support verdict.

A gold support structure \(Q\) is a closed atom set sufficient for one scored answer claim. It contains necessary transitions/states, qualification, time, and evidence. It is complete when removing an atom makes the claim unsupported, changes its reading, or empties a required slot. Only one to three evident references are listed. These define \(GoldSup\) for scoring, never \(GuardSup\) or selector feasibility.

Unlisted medium-budget paths from Full, NoTransitionPriority, and guard-PCST on the 26 held-out contexts enter a deidentified randomized mixture with decoys, at most 78 elected paths before deduplication. Blinded review labels each fully sufficient, partial, valid-redundant, unsupported, temporally invalid, epistemically invalid, post-horizon, or irrelevant. Fully sufficient paths extend \(GoldSup\), after which outcome scores are recomputed once. Feasible tables and selections never change. Other methods/budgets receive lower-bound scores against prelisted gold, or an optional frozen sample.

### Diagnostics

The audit unit is a tuple \((e,C,H,G_C)\): transition, context, horizon, and frozen candidate graph. One master set starts with the first five eligible tuples by discourse order in each real held-out bundle, 25 total, then adds at most 7 prespecified cell-filling tuples. All 10 distinct HRC targets come from this set. The balanced diagnostic selects 20 audited tuples, including five of the HRC tuples for its rare-consequential cell when available; all cells remain descriptive. Natural-frame results use the original 25 without weighting, intervals, or representativeness.

If consequence self-stability is inadequate, the fallback construct is *horizon-rare, query-necessary support transition*. The paper then makes no general pivotality or consequence claim.

## 15. Solo annotation and stability protocol

The sole researcher is both annotator and curator, which prevents inter-annotator agreement and requires explicit safeguards. Before held-out annotation, the researcher writes a compact handbook using the public development bundle and hidden control. It defines identity, event boundaries, transition dimensions, assertion statuses, temporal relations, evidence adequacy, consequence, relevance, and support sufficiency, with positive, negative, unknown, and disputed examples.

Workflow:

1. freeze lawful editions, bundle boundaries, and source locators;
2. annotate entities, occurrences, states/transitions, propositions/assertions, temporal/epistemic qualifiers, and evidence in separate passes;
3. run structural checks after each pass;
4. construct queries and reference support structures only after episode records are frozen;
5. seal held-out labels before system results;
6. after 14 days, independently reannotate 15 percent of roots, stratified by type, bundle, epistemic ambiguity, and rarity/consequence category;
7. reconcile disagreements by the frozen rubric and retain both versions plus reasons.

The second pass covers at least 24 roots, including 6 transitions, 6 assertions, and all rare-consequence candidates if fewer than 6. Horizon re-audit is scored separately.

Agreement measures are intra-rater, not inter-rater: entity/event coreference pairwise F1; categorical Cohen's kappa for transition dimension and epistemic status; weighted kappa for consequence dimensions; temporal-relation agreement; evidence-anchor overlap F1; and support-structure atom F1. Go criteria are coreference and evidence F1 at least 0.85, temporal/epistemic agreement at least 0.80, weighted consequence kappa at least 0.70, and support atom F1 at least 0.80. Prevalence-adjusted raw agreement is also reported because kappa can be unstable.

Automated checks cover dependencies, evidence, identity, state types, temporal cycles, shortcuts, horizons, and budgets. They cannot validate interpretation. All changes are versioned and method-blind.

## 16. Query and context construction

There are exactly 24 base query families, four per natural bundle, one for each task schema in Section 7. Context variants are added only when they test a necessary contrast:

- public development: 4 base contexts plus 2 variants, used only for design;
- four AGOT held-out bundles: 4 bases plus 1 horizon or holder variant each, 20 contexts;
- public held-out bundle: 4 bases plus one viewpoint and one temporal-window variant, 6 contexts.

There are 32 contexts: 6 development and 26 held-out. Ten, two per real held-out bundle, form the HRC benchmark. The pair in each bundle comes from distinct base families and targets distinct transition occurrences; variants cannot double-count one target. Before method execution each must be answerable, target an audited consequential horizon singleton, require multiple evidence items, and pass pressure. Failure invokes the query-necessary fallback, not favorable replacement.

Ten additional adversarial probes, two per real held-out bundle, test content absence, contract conflict, budget blockage, or unresolved ambiguity. They are outside HRC recall. If time remains, six paraphrases compare only Full and guard-PCST at medium budget; they are not required contexts or families.

Each frozen query has required response slots, allowed attributed answers, horizon, and ambiguity record. Public held-out content is opened only after schemas and parameters freeze. Opaque identifiers conceal diagnostic labels.

## 17. Budget derivation

The budget is a vector

$$
\mathbf B=(B_r,B_e,B_v,B_t,B_s,B_a,B_d,B_x,B_w),
$$

for roots, entities, events, transitions, states, assertions, display relations, evidence cues, and words. A fixed rendering grammar, viewport, and area are identical across methods; layout is not optimized or evaluated.

Six development contexts cannot support empirical quantiles. Root caps are fixed a priori at 4, 6, and 8, preserving exact-search and pressure calculations. For every other coordinate, exact development search finds a minimal admissible response cost per task schema. Low is their componentwise median and may correctly make some tasks BUDGET_BLOCKED; medium is their maximum; high adds one canonical transition/evidence bundle. These maximum envelopes apply before held-out opening:

| Level | Roots | Entities | Events | Transitions | States | Assertions | Display relations | Evidence cues | Words |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Low | 4 | 5 | 2 | 1 | 2 | 1 | 5 | 2 | 40 |
| Medium | 6 | 8 | 3 | 2 | 4 | 2 | 8 | 4 | 70 |
| High | 8 | 11 | 5 | 3 | 6 | 4 | 12 | 6 | 100 |

The derived vector is frozen. If a task minimum exceeds an envelope, simplify the schema or episode before sealing; test evidence never raises caps. Sensitivity uses adjacent levels and, if time permits, one-component changes around medium.

All methods share the vector. Post-closure requested and realized costs are reported; any cap violation invalidates output. Equal root count alone is insufficient matching.

## 18. Horizon-filtered native and guard-wrapped baselines

Three reproducible baseline families are required:

| Family | Native ranking/structure | Guard-wrapped form | Role |
|---|---|---|---|
| Frequency | root/event mention or relation frequency, stable tie break | ranks the identical response-feasible table by frequency | Conventional pruning |
| PPR | personalized PageRank from focal roots on the eligible graph | ranks the identical response-feasible table by summed PPR utility | Simple query-seeded method |
| PCST | prizes from frozen query relevance and costs from component burden; compact connected roots | chooses the best PCST-scored member of the identical response-feasible table | Strong primary comparator |

Every native baseline first receives a reasonable horizon eligibility filter. It is not deliberately supplied later plot information. Native outputs then use the method's ordinary selection behavior and raw caps, without forced evidence/epistemic/temporal/support closure; their practical violations are secondary results. Guard-wrapped variants receive the same candidate universe, semantic closure, guards, response schema, evidence alternatives, exact feasible table, component caps, and deterministic tie breaks as Proposed-Full.

Guard-PCST is the primary comparator because it directly addresses compact query-focused connected selection and is reproducible at this scale. Frequency and PPR are secondary. PCST connector paths count against every applicable cap. A no-horizon variant and individual no-guard variants are fault diagnostics only, never primary comparators.

GLIMPSE, GraphRAG variants, QAFD-RAG, Context-KG, NWM, NKW, and Story Ribbons remain closest work. Some have public artifacts, but their objectives and output types would require nontrivial, potentially unfaithful same-guard/vector adaptation within the solo limit. Excluding them narrows comparative claims; it is not an artifact-unavailability claim.

## 19. Direct objective ablation

Proposed-NoTransitionPriority uses the same exact feasible table, query features, relevance, coverage, redundancy, cost, guards, budgets, and stable tie break as Proposed-Full. Its objective is

$$
\operatorname*{lex\,arg\,max}_{S\in\mathcal A_{\mathrm{resp}}}
\langle J_{rel},J_{cov},-J_{red},-J_{cost},-J_{tie}\rangle .
$$

The only removed term is \(J_{TB}\). This makes it a direct mechanistic ablation, not a weaker safety system.

Mechanism evidence requires at least 2 net HRC successes, at least 3 wins, and at most 1 loss over NoTransitionPriority. Under the common frozen budget, Full's aggregate realized cost must be no greater on every one of the nine coordinates; no outcome-dependent rebudgeting is allowed. Balanced-cell frequent-consequence loss or rare-irrelevant increase qualifies interpretation but is not a five-item confirmatory gate. If the ablation matches Full, transition priority receives no credit.

## 20. Verifier corruption study

Guard-wrapped admissibility is true by construction if the implementation is correct. It is therefore contract validation, not independent evidence that Proposed-Full is safer than another wrapped selector.

Development and hidden bundles supply unit tests. The five real held-out bundles supply a sealed set of nine single-fault projections each:

1. evidence anchor removed or mismatched;
2. mandatory support/closure atom removed;
3. post-horizon event, identity, or descriptor inserted;
4. rumor/belief promoted to established or holder removed;
5. contradictory temporal edge inserted;
6. mistaken entity merge inserted;
7. one component budget exceeded;
8. provenance removed;
9. post-horizon label leakage inserted.

The total is 45 corruptions plus 10 clean controls. Frozen transformations are manually checked for a single defect. The gate is 45/45 rejected and 10/10 accepted; per-category sensitivity and clean specificity are reported. Critical defects are horizon leakage, unsupported answer, epistemic promotion, identity corruption, answer-affecting temporal contradiction, and incomplete support. Budget, provenance, and label defects remain separate; formatting is outside scope.

Each guard is removed alone for its five targeted held-out corruptions, giving 45 targeted reruns. A manual audit covers every Full and guard-PCST primary output. If the sealed verifier misses a case, that failure is retained; repaired cases become development data and cannot restore the confirmatory contract gate without a new preregistered bank.

## 21. Primary fixed-benchmark evaluation

### Co-primary quantities

For method \(m\) and context \(q\):

- \(Adm_{mq}=1\) only when a returned \(S\) belongs to \(\mathcal A_{\mathrm{adm}}\); abstention on these feasible contexts scores 0;
- \(Resp_{mq}=1\) when a feasible query has an admissible schema-complete response;
- \(Y_{mq}=1\) when the target HRC transition, warranted before/after states, temporal/epistemic qualification, and one accepted complete support structure are present;
- \(Abst_{mq}=1\) when an adversarial probe receives the correct feasibility class and a valid certificate.

Admissibility rate and support-complete HRC recall conditional on passing the admissibility gate are co-primary:

$$
AR_m={1\over10}\sum_q Adm_{mq},\qquad
HRSC_m={1\over10}\sum_q Y_{mq}.
$$

Because the gate requires \(AR=1\), conditional and unconditional HRSC coincide in the confirmatory comparison. The combined violation-zeroed summary is

$$
ASCPR_m={1\over10}\sum_q Y_{mq}\mathbb 1[Adm_{mq}=1].
$$

Correct infeasibility/abstention is \(10^{-1}\sum_q Abst_{mq}\) on the separate probes. Relevance never compensates for a critical violation.

### Frozen success rule

Proposed-Full supports H1 only if all conditions hold:

1. both Full and guard-PCST produce admissible, responsive outputs for all 10 feasible primary contexts;
2. \(HRSC_{Full}-HRSC_{PCST}\ge0.20\), a deliberately strict threshold of at least 2 net successes;
3. paired counts show at least 3 Full wins and at most 1 loss;
4. Full's within-bundle mean is greater than PCST's in at least 3 of the 5 real held-out bundles;
5. deleting any one bundle leaves a positive aggregate difference;
6. no bundle supplies more than 50 percent of \(\sum_q\max(Y_{Full,q}-Y_{PCST,q},0)\);
7. the public held-out bundle has a nonnegative difference;
8. the verifier passes 45/45 corruptions and 10/10 controls, all 10 probes receive the correct class and certificate, and neither method has a critical violation.

The 0.20 threshold is deliberately stricter than the smallest discrete 0.10 difference: it requires two net successes plus cross-bundle safeguards. All outcomes and costs are published.

The five-item diagnostic cells do not gate H1. However, a loss of at least 2 frequent-consequential items or at least 2 extra rare-nonconsequential selections versus guard-PCST is an unacceptable observed tradeoff: the paper must qualify the positive claim rather than generalize "protection."

The primary public-domain result being positive supports limited cross-work transfer; zero permits an AGOT-centered conclusion; negative confines the conclusion to the four AGOT bundles. The hidden narrative never enters superiority or external-validity results.

## 22. Diagnostic and sensitivity analyses

Secondary metrics are:

- support-complete recall for the 20 balanced diagnostic transitions;
- false protection of horizon-rare nonconsequential items;
- recall of horizon-repeated consequential items;
- LocalRarity versions of the above;
- exact query relevance, the mean frozen 0-3 grade of selected roots;
- redundancy, the proportion of selected roots marked semantically duplicative;
- compression and realized vector costs;
- response and certificate accuracy;
- complete-support-structure recall, giving credit to post-hoc accepted novel structures;
- optional paraphrase Jaccard stability for Full and guard-PCST only;
- temporal, epistemic, evidence, identity, horizon, provenance, and budget violation counts by severity.

Analyses are repeated at low, medium, and high budgets. The medium budget is primary. Low and high are sensitivity curves, not extra opportunities to declare success. Horizon versus local rarity is reported side by side. Results from frequency and PPR, native and wrapped, explain practical tradeoffs but do not replace the Full-versus-guard-PCST test.

The eight variants are Full, NoTransitionPriority, and native/wrapped frequency, PPR, and PCST. Required archives contain \(8\times32\times3=768\) answerable-context runs, \(8\times10=80\) medium-budget probe runs, 55 full-verifier corruption/control runs, and 45 targeted guard-removal reruns: 948 cases excluding development unit tests. Optional paraphrases add 12 runs. Novel paths are reviewed once; no insight, readability, recall, or workload coding occurs.

## 23. Statistical reporting without unsupported population inference

The six natural bundles are too few, partly conditioned, and not sampled from one common population. Paper 1 therefore uses fixed-benchmark inference:

- publish every paired binary and component-cost result;
- report exact win/loss/tie counts;
- report arithmetic differences and risk differences, not \(p\)-values;
- show each bundle and leave-one-bundle-out direction;
- show the fraction of positive gain contributed by each bundle;
- report budget sensitivity and the two rarity definitions;
- report AGOT and public held-out results separately.

There is no sign-flip test, bootstrap population interval, normal-theory confidence interval, Horvitz-Thompson estimate, or language such as "representative of the novel." The balanced 5-by-4 diagnostic is descriptive. The natural frame reports within-bundle and equal-bundle proportions without variance claims. The public held-out result is independent transfer evidence for one work, not genre generalization.

If a future multi-bundle study supplies a probability-sampling frame, preregistered independent annotations, and enough clusters, population inference can be reconsidered. It is not retrofitted here.

## 24. Copyright, contamination, and reproducibility

AGOT uses lawful access. Unless legal review permits, releases omit passages and reconstruction-ready graphs, using safe locators, hashes, abstract labels, counts, and restricted alignment instructions. Triples are not automatically redistributable.

Releasable artifacts include the ontology/handbook, contexts, definitions, verifier corruptions, budgets, baselines, exact results, seeds, public annotations, and hidden story. The independent public held-out annotation is fully reproducible.

Contamination controls use the hidden narrative to create exact chronology, altered object transfers, relationship swaps, false claims, and post-horizon facts. A system is memory-reliant if it prefers a familiar or planted account over supplied evidence. Because the primary condition has no LLM, these controls chiefly validate the graph and verifier in Paper 1. If an LLM is later tested, names are anonymized and entities permuted; unsupported recalled plot receives zero evidence support and a critical violation when displayed.

Reproducibility has two tiers: complete reproduction on public-domain and hidden material, and restricted alignment reproduction for AGOT. All held-out freezes are timestamped before results. The public development and public held-out roles cannot be exchanged after inspection.

## 25. Threats to validity

| Threat | Consequence | Mitigation and residual limit |
|---|---|---|
| One annotator | Bias, no community agreement | Frozen handbook, delayed reannotation, masked path review; claims remain curator-relative |
| Query-conditioned oracle | Easier than a book graph | Legitimate 18-root pressure pools; still only controlled density |
| Four AGOT bundles | Poor novel-wide coverage | Quartile coverage and full disclosure; no novel inference |
| One public test | Weak transfer evidence | Strict independence and full release; one-work claim only |
| Consequence rubric | Missed literary importance | Source-backed independent dimensions, stability gate, fallback |
| Horizon search | False singleton | Multiple searches, manual verification, re-audit, claim-removal gate |
| Exact small graphs | No scalability evidence | Hard 18-root scope; heuristics deferred |
| Hand-set objective | Development overfit | All settings frozen before held-out data |
| Wrapped baselines | Altered native behavior | Report both; credit safety to contract |
| Display proxies | Unknown cognitive burden | No readability claim |
| Literary uncertainty | Evidence not world truth | Preserve attribution and unresolved status |
| Researcher familiarity | Tacit AGOT memory | Locators, seals, public test, hidden controls |

The fixed benchmark can reveal whether the proposed objective works in these cases and why. It cannot estimate a population effect, validate reader usefulness, or show that oracle results survive extraction errors.

## 26. Realistic workload

The realistic required total is 150 hours, about 19 weeks at 8 hours per week or 25 weeks at 6. This exceeds the aspirational 16-week calendar but remains below the non-negotiable 160-hour ceiling; pretending otherwise would underprice annotation. Ten hours remain for one bounded correction.

| Required stage | Hours |
|---|---:|
| Recent-literature verification and status ledger | 4 |
| Minimal ontology and annotation handbook | 7 |
| Corpus selection, lawful editions, boundary audits | 6 |
| Hidden narrative, truth ledger, and control annotation | 6 |
| First-pass annotation of up to 162 natural roots | 35 |
| Targeted horizon audits, up to 32 transitions plus 8 re-audits | 12 |
| Delayed 15 percent reannotation and reconciliation | 5 |
| Query, context, response-schema construction | 4 |
| Reference support structures | 4 |
| Dense distractor-pool assembly and pressure checks | 6 |
| Canonical graph representation and validators | 4 |
| Closure, evidence, provenance, identity, budget verifiers | 6 |
| Temporal satisfiability checker | 3 |
| Exact enumeration, certificates, zero-gap checks | 6 |
| Horizon-filtered native baselines | 3 |
| Guard-wrapped baselines, including guard-PCST | 3 |
| Direct NoTransitionPriority ablation | 2 |
| Automated tests, researcher verification, debugging | 7 |
| Corruption-study creation and audit | 5 |
| Frozen oracle runs and result ledger | 4 |
| Masked novel-path review and one outcome rescore | 5 |
| Fixed-benchmark and sensitivity analyses | 5 |
| Preregistration | 4 |
| Reproducibility documentation and packaging | 4 |
| **Required total** | **150** |
| **Contingency before 160-hour stop** | **10** |

At 13 minutes per root, 162 roots require about 35 hours. Automation still requires review. At 100 logged hours, forecast completion; optional paraphrases and perturbations never consume required time. Stop at 160 without dropping guard-PCST, the direct ablation, public test, rarity audit, or pressure gate.

No human recruitment or LLM evaluation appears in this table. Either would require a separately approved Paper 2 budget.

## 27. Stop or go gates

The execution sequence is singular and gated:

| Order | Gate | Go | Revise | Stop or redirect |
|---:|---|---|---|---|
| 1 | Handbook | Four schemas usable | Simplify fields | Competencies remain unanswerable |
| 2 | Stability | Section 15 passes | One development revision | Core labels remain unstable |
| 3 | Rarity | F1 0.85; 7/8 categories | Mark uncertain | Use local/query-necessary fallback |
| 4 | Pressure | 18 roots; 3:1 medium components | Use a frozen context | Lacking two HRC cases triggers fallback |
| 5 | Exactness | Enumeration checks agree | Correct before unsealing | Drop zero-gap claim |
| 6 | Contract | 45/45 reject; 10/10 accept | Retain failure; diagnose | No confirmatory utility interpretation |
| 7 | Utility | Section 21 passes | Report mixed result | H1 unsupported |
| 8 | Mechanism | Section 19 passes | Comparator result only | No mechanism claim |
| 9 | Transfer | Public result nonnegative | AGOT-only claim | Report negative boundary |
| 10 | Resource | At most 160 hours | Drop optional work | Stop expansion |

Only one handbook revision is allowed after a stability failure, and it occurs before held-out outcomes. A failed gate narrows the claim; it does not authorize replacement with a favorable episode.

## 28. Negative-result paper

The strongest defensible negative-result paper reports a preregistered benchmark, verifier, and matched-guard comparison showing that transition priority did not improve upon guard-PCST once closure, evidence, horizon, and vector budgets were equalized. It can document which apparent advantages of native pruning disappear under common guards, whether horizon rarity differs from local rarity, and where exact feasibility fails under tight budgets. This would challenge the assumption that a bespoke pivotal-event objective is necessary.

If the consequence construct fails but support annotation is stable, the paper becomes **admissibility-constrained preservation of horizon-rare query-necessary support structures**. If the verifier fails, no selector superiority is reported; the contribution becomes a failure analysis and revised contract specification. If the public transfer is negative, conclusions are restricted to the four AGOT bundles. These are publishable boundary results only if all cases and protocol deviations are disclosed.

## 29. Deferred Paper 2 work

Paper 2 may study end-to-end literary extraction, heuristic scaling, and human reasoning. A powered human experiment would compare Proposed-Full with guard-PCST using closed questions on answer, temporal/epistemic qualification, evidence choice, time, confidence, and workload; its sample size would follow a pilot-based power analysis. A bounded LLM could compile natural-language queries into typed candidates or propose evidence-backed descriptors, with deterministic guards. Neither is retrospectively added to Paper 1.

Also deferred are adaptive labels, visualization layout, semantic zoom, expert insight coding, delayed recall, feedback learning, canonical corrections by users, ontology evolution, nested attitudes, themes, profiles, continual learning, multi-novel benchmarks, and broad domain transfer. Full-book extraction and exhaustive horizon annotation require additional personnel.

## 30. Venue-category assessment

The most defensible Paper 1 category is a **digital humanities computational methods case study**: a small, interpretable literary corpus, explicit hermeneutic limits, formal selection problem, and fully reported case results. A Semantic Web or applied-ontology venue is plausible only if the formal semantics, competency-question evaluation, and proofs are deepened; the present ontology is intentionally minimal. An NLP or information-retrieval venue would commonly expect more independent texts, automatic extraction, or a larger query benchmark. A visualization/HCI venue would normally require a developed presentation technique and powered human evaluation, which Paper 1 expressly lacks.

The venue claim should emphasize auditable method and literary interpretation boundaries, not scale. No specific journal is selected before results and artifact quality are known.

## 31. Minimum viable paper specification

**Object:** admissibility-constrained transition-bundle projection under componentwise semantic display budgets.

**Data:** four held-out AGOT bundles, one public development bundle, one independent public held-out bundle, and one hidden verifier/contamination control. Six natural bundles contain no more than 162 roots total; every context graph has exactly 18 roots.

**Queries:** 24 base families and 32 contexts; 10 feasible HRC primary contexts and 10 separate probes; three budgets. Six paraphrases are optional.

**Methods:** Proposed-Full; Proposed-NoTransitionPriority; frequency, PPR, and PCST in horizon-filtered native and guard-wrapped forms. Guard-PCST is the sole primary comparator.

**Evidence:** at most 32 horizon audits; 15 percent delayed reannotation after 14 days; 55-case corruption set; exact enumeration; masked novel-path validation; complete case reporting.

**Decision:** pass contract first, then require a 0.20 HRC advantage, at least 3 wins and at most 1 loss, majority-bundle improvement, positive leave-one-bundle-out direction, no dominant bundle, and nonnegative public transfer. No population inference.

**Claim boundary:** fixed benchmark only; no extraction, human, LLM, visualization, whole-novel, or genre claim. Required effort is 150 hours with a 160-hour stop.

**Strongest positive conclusion:** On these five held-out real-text bundles, under identical verified guards and component caps, transition-bundle priority preserved at least two more complete horizon-rare consequential support structures than guard-PCST, with distributed gains, nonnegative public transfer, and no critical violation. This is a benchmark result, not a population estimate.

## 32. Change log from the previous solo plan

| Previous issue | V2 correction |
|---|---|
| Episode-only rarity | Primary HorizonRarity plus targeted work-to-horizon mention audit; LocalRarity retained only as sensitivity |
| Small, overly relevant graph | Exact 18-root controlled-dense pool, legitimate reused distractors, and 3:1 component pressure |
| Six-cluster population-style test | Deterministic fixed-benchmark effect and consistency rules; no sign-flip test or population interval |
| Hidden story treated as transfer evidence | Hidden story restricted to controls; separate sealed public-domain held-out work added |
| Safety and answer success conflated | Nested \(\mathcal A_{\mathrm{adm}}\) and \(\mathcal A_{\mathrm{resp}}\), typed abstention and certificates |
| Wrapped safety credited to selector | Corruption and guard-ablation contract study; utility compared only under shared guards |
| Native baseline leakage straw man | Every native baseline is horizon-filtered; no-horizon output is only a diagnostic |
| Mechanism not isolated | Same-guard, same-budget NoTransitionPriority ablation |
| Broad algorithmic novelty | Specific optimization object plus verifier/benchmark contribution; component novelty disclaimed |
| Tautological theorem emphasis | Modest composition claim plus closure, inheritance, exactness, budget, hardness, and restricted-case results |
| Evidence equated with truth | Attributed, horizon-relative, curator-policy statuses and unresolved fictional-world truth |
| Arbitrary budgets | Analytic development-task minima and frozen vector envelopes |
| Unsupported prevalence weighting | Conditioned-frame raw and equal-bundle descriptions only |
| Missing computational labor | Separate verifier, temporal, optimization, baseline, debugging, corruption, and packaging hours |
| Human and LLM scope | Both removed from default Paper 1 |

## 33. Final internal-consistency audit

- **Corpus:** 7 bundles; efficacy uses 4 AGOT plus 1 public held-out. Development and hidden control are excluded. Six natural bundles have at most 162 roots.
- **Tasks:** 24 families, 32 contexts, 10 HRC primary cases, and 10 probes. Six paraphrases are optional.
- **Rarity/density:** at most 32 horizon audits; local rarity is secondary. Primary pools have 18 roots, 3:1 medium pressure, and exact caps 4/6/8.
- **Fairness:** native frequency/PPR/PCST are horizon-filtered; all wrapped selectors share one feasible table. Guard-PCST is primary.
- **Evidence:** 45 corruptions plus 10 controls; 10 feasible utility cases; 20 diagnostic transitions with no confirmatory role.
- **Bounds:** no population inference, extraction, human, LLM, adaptive-label, layout, or unrestricted-truth claim.
- **Resources/release:** 150 required plus 10 reserve hours; public/hidden artifacts are reproducible and AGOT derivatives are legally restricted.

Repeat this audit after inserting actual locators and derived budgets; propagate every changed count before unsealing.

## 34. References

- Allen, James F. 1983. "Maintaining Knowledge about Temporal Intervals." *Communications of the ACM* 26(11):832-843. [DOI](https://doi.org/10.1145/182.358434).
- CIDOC CRM-SIG. 2026. *CRMinf 1.2.1: An Extension of CIDOC CRM to Support Argumentation*. Official specification. [Specification](https://cidoc-crm.org/extensions/crminf/html/CRMinf_v1.2.1.html).
- Cox, Simon, and Chris Little, eds. 2017. *Time Ontology in OWL*. W3C Recommendation. [Specification](https://www.w3.org/TR/owl-time/).
- Damiano, Rossana, Vincenzo Lombardo, and Antonio Pizzo. 2019. "The Ontology of Drama." *Applied Ontology* 14(1):79-118. [DOI](https://doi.org/10.3233/AO-190204).
- Edge, Darren, Ha Trinh, Newman Cheng, Joshua Bradley, Alex Chao, Apurva Mody, Steven Truitt, Dasha Metropolitansky, Robert Osazuwa Ness, and Jonathan Larson. 2024. "From Local to Global: A Graph RAG Approach to Query-Focused Summarization." Preprint. [arXiv](https://arxiv.org/abs/2404.16130).
- Fokkens, Antske, Piek Vossen, Marco Rospocher, Rinke Hoekstra, and Willem Robert van Hage. 2017. "GRaSP: Grounded Representation and Source Perspective." RANLP workshop paper. [DOI](https://doi.org/10.26615/978-954-452-040-3_003).
- Goemans, Michel X., and David P. Williamson. 1995. "A General Approximation Technique for Constrained Forest Problems." *SIAM Journal on Computing* 24(2):296-317. [DOI](https://doi.org/10.1137/S0097539793242618).
- He, Xiaoxin, Yijun Tian, Yifei Sun, Nitesh V. Chawla, Thomas Laurent, Yann LeCun, Xavier Bresson, and Bryan Hooi. 2024. "G-Retriever: Retrieval-Augmented Generation for Textual Graph Understanding and Question Answering." *NeurIPS 2024*. [Proceedings](https://proceedings.neurips.cc/paper_files/paper/2024/hash/efaf1c9726648c8ba363a5c927440529-Abstract-Conference.html).
- Hong, Yubin, Chaofan Li, Jingyi Zhang, and Yingxia Shao. 2025. "Context-Aware Fine-Grained Graph RAG for Query-Focused Summarization." *CIKM 2025*, 4802-4806. [DOI](https://doi.org/10.1145/3746252.3760935).
- Jeh, Glen, and Jennifer Widom. 2003. "Scaling Personalized Web Search." *WWW 2003*, 271-279. [DOI](https://doi.org/10.1145/775152.775191).
- Lebo, Timothy, Satya Sahoo, and Deborah McGuinness, eds. 2013. *PROV-O: The PROV Ontology*. W3C Recommendation. [Specification](https://www.w3.org/TR/prov-o/).
- Liu, Zhengzhong, Chenyan Xiong, Teruko Mitamura, and Eduard Hovy. 2018. "Automatic Event Salience Identification." *EMNLP 2018*, 1226-1236. [DOI](https://doi.org/10.18653/v1/D18-1154).
- Meghini, Carlo, Valentina Bartalesi, and Daniele Metilli. 2021. "Representing Narratives in Digital Libraries: The Narrative Ontology." *Semantic Web* 12(2):241-264. [DOI](https://doi.org/10.3233/SW-200421).
- Nemhauser, George L., Laurence A. Wolsey, and Marshall L. Fisher. 1978. "An Analysis of Approximations for Maximizing Submodular Set Functions - I." *Mathematical Programming* 14:265-294. [DOI](https://doi.org/10.1007/BF01588971).
- Papalampidi, Pinelopi, Frank Keller, and Mirella Lapata. 2019. "Movie Plot Analysis via Turning Point Identification." *EMNLP-IJCNLP 2019*, 1707-1717. [DOI](https://doi.org/10.18653/v1/D19-1180).
- Perera, Rumali, Xiaoqi Wang, and Han-wei Shen. 2026. "Context-KG: Context-Aware Knowledge Graph Visualization with User Preferences and Ontological Guidance." Preprint. [arXiv](https://arxiv.org/abs/2604.10384).
- Pianzola, Federico, Luotong Cheng, Franziska Pannach, Xiaoyan Yang, and Luca Scotti. 2025. "The GOLEM Ontology for Narrative and Fiction." *Humanities* 14(10):193. [DOI](https://doi.org/10.3390/h14100193).
- Qian, Xinying, Ying Zhang, Yu Zhao, Baohang Zhou, Xuhui Sui, Li Zhang, and Kehui Song. 2024. "TimeR4: Time-aware Retrieval-Augmented Large Language Models for Temporal Knowledge Graph Question Answering." *EMNLP 2024*, 6942-6952. [DOI](https://doi.org/10.18653/v1/2024.emnlp-main.394).
- Rasmussen, Preston, Pavlo Paliychuk, Travis Beauvais, Jack Ryan, and Daniel Chalef. 2025. "Zep: A Temporal Knowledge Graph Architecture for Agent Memory." Preprint. [arXiv](https://arxiv.org/abs/2501.13956).
- Safavi, Tara, Caleb Belth, Lukas Faber, Davide Mottin, Emmanuel Muller, and Danai Koutra. 2019. "Personalized Knowledge Graph Summarization: From the Cloud to Your Pocket." *ICDM 2019*, 528-537. [DOI](https://doi.org/10.1109/ICDM.2019.00063).
- Saifullah, Mohammad, Thomas Kornmaier, Taaha Kazi, Vasu Sharma, Aditya Sanjiv Kanade, and Aanand Kumar Yadav. 2026. "Narrative World Model: Narratology-Grounded Writer Memory for Long-Form Fiction." Preprint. [arXiv](https://arxiv.org/abs/2607.05577).
- Sanderson, Robert, Paolo Ciccarese, and Benjamin Young, eds. 2017. *Web Annotation Data Model*. W3C Recommendation. [Specification](https://www.w3.org/TR/annotation-model/).
- Saxena, Apoorv, Soumen Chakrabarti, and Partha Talukdar. 2021. "Question Answering Over Temporal Knowledge Graphs." *ACL-IJCNLP 2021*, 6663-6676. [DOI](https://doi.org/10.18653/v1/2021.acl-long.520).
- Sharma, Anshu Kiran, Miguel Castiblanco-Melendez, Alejandro Morales, and Mark A. Finlayson. 2026. "Once upon a Kernel: Extracting Important Events from Narratives." *LREC 2026*, 6009-6021. [ACL Anthology](https://aclanthology.org/2026.lrec-1.477/).
- Tian, Qiuyu, Fengyi Chen, Yiding Li, et al. 2026. "Narrative Knowledge Weaver: Narrative-Centric Retrieval-Augmented Reasoning for Long-Form Text Understanding." Preprint. [arXiv](https://arxiv.org/abs/2606.05724).
- van Hage, Willem Robert, Veronique Malaise, Roxane Segers, Laura Hollink, and Guus Schreiber. 2011. "Design and Use of the Simple Event Model (SEM)." *Journal of Web Semantics* 9(2):128-136. [DOI](https://doi.org/10.1016/j.websem.2011.03.003).
- W3C RDF & SPARQL Working Group. 2026. *RDF 1.2 Concepts and Abstract Data Model*. Candidate Recommendation Snapshot. [Specification](https://www.w3.org/TR/rdf12-concepts/).
- Yeh, Catherine, Tara Menon, Robin Singh Arya, Helen He, Moira Weigel, Fernanda Viegas, and Martin Wattenberg. 2026. "Story Ribbons: Reimagining Storyline Visualizations with Large Language Models." *IEEE Transactions on Visualization and Computer Graphics* 32(1):736-746. [DOI](https://doi.org/10.1109/TVCG.2025.3634265).
- Zhang, Ze Yu, Zitao Li, Yaliang Li, Bolin Ding, and Bryan Kian Hsiang Low. 2026. "Respecting Temporal-Causal Consistency: Entity-Event Knowledge Graph for Retrieval-Augmented Generation." *EACL 2026*, 2017-2054. [DOI](https://doi.org/10.18653/v1/2026.eacl-long.90).
- Zhou, Zhuoping, Davoud Ataee Tarzanagh, Sima Didari, et al. 2026. "Query-Aware Flow Diffusion for Graph-Based RAG with Retrieval Guarantees." *ICLR 2026*. [Proceedings](https://proceedings.iclr.cc/paper_files/paper/2026/hash/584f32ccf76d73b85fa2053e795df697-Abstract-Conference.html).
