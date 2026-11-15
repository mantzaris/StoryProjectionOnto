# Response to reviews: ICAART 2027 paper 82

This revision keeps the title and abstract unchanged. It retains the original model outputs and A/B scores and concentrates on definitions, evaluation clarity, reproducibility, figure presentation and the interpretation of the comparison. A small additional comparison uses existing outputs on CPU. It is explicitly described as a follow-up analysis, not as new model generation or independent validation.

## Reviewer 1

**Contribution and inconsistent framing.** Sections 1 and 6 now describe the work as an exploratory implementation and comparison, not a new extraction algorithm. “Workflow” is used consistently for the proposed approach. The Introduction defines the limited meaning of narrative and temporal ontology used here. The Discussion states that neither completeness nor logical consistency is guaranteed. The abstract already describes a workflow and has been preserved.

**Belief modeling and Orchard S11.** Section 3.1 now walks through the actual example: Ivory Vase → Hill Court, with holder Jun and attitude `believes`. Jun is a text name attached to the relationship, referring to the same named character as the graph node; it is not another edge or a separately resolved entity identifier. S12's Ivory Vase → Elm Dock record has null attribution and represents narration. The paper explains why a belief and narration can disagree without treating both as the same factual assertion.

**Companion and repository access.** The Appendix now gives the actual public repository URL and package location. The revision README identifies exact requests, raw outputs, reference answers, matching rules, semantic assessments and the companion, and supplies an executable CPU replay command. The original public data are accessible; these new revision files remain local until distributed or published.

**More results, stronger validation/discussion, current references and future work.** Table 1 adds the B+filter comparison using the original filters on all nine saved synthetic contextual answers. The same check is also run on all six prose contextual cases, preserving the unparseable answer. Section 5 reports both improvements and failures; Sections 6–7 state the limits and next experiments. ATOM, ReVerb and OLLIE are added to Related Work. This adds a comparative diagnostic, not new texts, seeds, models or an independently annotated benchmark. Those broader validation requests remain limitations rather than being claimed complete.

## Reviewer 2

**How evaluation was performed; origin and availability of the correct answers.** Section 4 now states that the answers were prepared for this study with Codex before the runs, and are not a published gold standard. The same reference answers and rules score both approaches. The evaluation order is explicit: parse the response, calculate strict matches, then report the separate semantic judgments. All reference records, permitted wording alternatives and per-record assessments are available at the paths in the README.

**Who distinguished good and bad answers, and when; subjectivity.** Section 4 states that Codex assessed outputs after generation using a rubric set before the runs. It defines supported, partial, unsupported and unresolved records, explains coverage, and gives examples of a correct meaning missed by strict matching and an incorrect participant. Reference creation, assessment and other AI assistance are disclosed together. There is no claim of independent human review. The Discussion explicitly recognizes the resulting limitation.

**ATOM and the choice of TextRunner.** Related Work now includes the requested ATOM paper, with its verified authors, venue and pages, and explains its dynamic graph construction and observation/validity-time distinction. TextRunner is identified as historical context, not a chosen baseline. ReVerb and OLLIE supply later OpenIE context. The paper explains that its experiment compares question timing using one LLM; it makes no performance claim against those other systems.

**Meaning of speech acts.** Section 3.1 says that the represented relationship describes an utterance or promise; the extraction does not infer its fulfillment or a resulting legal/social state change.

**Nonzero temperature and the seed.** Model settings are moved to Evaluation. The text now says that the seed documents sampling rather than guaranteeing identical fresh server responses. The replay command reproduces scores from the saved responses. The README links the exact vLLM 0.10.2 reproducibility documentation.

**More raw text in Figure 1.** The source excerpt expands from three to ten verbatim sentences in two columns, covering ownership/carrying and belief/narration. All fourteen sentences were supplied to the model. The diagram's records, directions and scores are unchanged. The complete input remains in the public evidence files.

**Harbor, Orchard and Museum.** Sections 3–4 identify them as synthetic stories created for this study, not external datasets, books or organizations.

**Holmes's vocative and author review.** The unexplained term is replaced with a plain description of uncertainty about whom a phrase addresses. Alice's consideration is distinguished from a decision. The paper says how these unresolved readings affect coverage instead of merely leaving them “open to author review.”

**“Frozen alternatives,” strict F1 and semantic assessment.** The text explains that alternatives are acceptable names or relation wordings listed before generation. It distinguishes matching that list from factual correctness and identifies the assessor. It also clarifies the existing difference between synthetic and prose matching of missing fields; the scoring code and original scores have not changed.

**“Separately recorded recovery” and unclear evaluation paragraphs.** Parsing is defined in plain language. The only permitted repair is explained as removing a trailing comma before a closing bracket/brace. Neither reported batch used that repair; the extra-brace failures remain failures with unavailable scores. The unnecessary equation naming an evaluation case is replaced by a sentence.

**More comparative evidence and clearer conclusions.** The B+filter check is applied without changing the filters or tuning them to results. It improves possession/location answers but loses belief/reality matches. The conclusions compare the observed strengths and weaknesses directly and specify independent annotation and longer unseen texts as next steps. No new generalization claim is made.

## Reviewer 3

**Motivation and definitions.** The Introduction explains the tradeoff between an overly broad graph and omissions from question-focused extraction. Narrative ontology and temporal ontology are defined within this implementation's limited scope. Section 3 explains attribution, holder and attitude with the S11/S12 example; Section 4 defines parseable output. Section 3.2 distinguishes category/interval filters for synthetic stories from lists of relation words for prose.

**Related Work and unclear contribution statement.** TextRunner's historical role is clarified where it is introduced, alongside ReVerb, OLLIE and SPIRES. ATOM is added with the specific difference in task. The concluding Related Work paragraph now says directly that the views compare answers and let readers check each relationship against its source. The vague “cited systems” sentence is removed.

**Query window example.** Section 3.1 explains that asking about [3,4) retains the carrying record valid during [2,5), because it overlaps the requested period; the filter does not change the source's stated duration.

**Experiment settings belong in Evaluation.** The model, sampling parameters, hardware, context/output limits and seed caveat appear in Section 4.1, not in the approach description.

**Orchard's unexplained name.** Orchard is explicitly introduced as an invented test story before describing its graph.

**Large figures and long captions.** All three main figure captions are reduced to at most two rendered lines. Figure 1 uses two columns for the expanded source excerpt; the graph itself remains unchanged. Display counts remain in Section 3.3, with full questions and other original notes available in the data/companion and `FIGURE_DETAILS.md`. The fable and Alice artwork is preserved to avoid unnecessary redraws or changes to the displayed evidence.

**Paper structure paragraph.** The final Introduction paragraph gives the section order.

**Meaningful comparison and future work.** The Discussion now contrasts reuse and missing facts in A with question-specific content, extra facts and per-question model calls in B. The Conclusion states the concrete advantages observed, the incorrect reading participant in Alice, and the next validation steps. The general statement that visualization “provides a concrete basis” is removed.

## Scope and computation

The supplied review checklists request more experiments/comparisons, but do not specify a mandatory new dataset or model. The revision adds the smallest relevant comparison that can be performed on the existing evidence. It does **not** establish that the reviewers' broader requests for validation are fully satisfied. Larger studies or repeated generation would need fresh inference; running the existing vLLM setup would require a suitable GPU. Independent human annotation requires people, not a GPU. This revision used neither new inference nor new semantic labels.
