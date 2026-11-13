# Inspectable Narrative Graphs: An Exploratory Extraction Prototype

a.v.mantzaris

## Abstract

Narrative readers need different views of the same text: actions, locations, possessions, or a character's beliefs. We present a working prototype for extracting evidence-linked relationships and visually inspecting their meaning, qualifications, and errors. We compare query-blind extraction followed by fixed selection (A) with independent contextual extraction (B), using three short synthetic stories and three published narrative passages. Strict qualified-fact agreement is reported separately from semantic preservation, relevance, and unresolved interpretation, using Codex-authored references and assessments. The story batch produced {{compact_strict_json}} parseable outputs from {{compact_calls}} calls. A matched every possession and interval-location target without extras; B retained irrelevant facts. On the fable action question, strict F1 was {{fable_actions_A_strict_qualified_f1}} for A and {{fable_actions_B_strict_qualified_f1}} for B. Alice's contextual claims answer preserved {{alice_claims_B_semantic_targets_covered}} of {{alice_claims_B_references}} semantic targets despite zero strict F1. Prose parsing succeeded in {{prose_strict_json}} of {{prose_calls}} calls. These examples demonstrate useful extraction and inspectable partial meaning alongside attribution errors and omissions. The contribution is an exploratory comparison and source-linked inspection workflow, with mixed results rather than evidence of general reliability or contextual superiority.

## 1. Introduction

A narrative contains more relationships than a reader needs for a particular question. An ownership view, a location map, and a summary of beliefs answer different information needs. A graph makes these relationships explicit, but a plausible arrow can also hide an error: looking into a book is not the same as reading it, and believing that an object is somewhere does not establish its location. Useful narrative graphs therefore need both evidence links and ways to inspect the limits of an extraction.

Our prototype combines compact relationship extraction, question-specific selection, and coordinated text–graph displays. Readable endpoints, source citations, validity bounds, and attribution make individual records interpretable. The display preserves actual outputs, including partial representations, unsupported assignments, and unresolved interpretations. We compare selecting records from a query-blind extraction with independently extracting an answer from the text and question. Strict agreement, semantic preservation, and relevance remain separate measures, so useful content is visible even when the graph is imperfect.

Three questions guide the study: what relationships and qualifications survive as evidence becomes less synthetic; whether contextual extraction yields more relevant answers than fixed selection; and what visual inspection reveals beyond strict tuple agreement. The evidence comprises small, purposively selected development examples. Query-dependent ontology construction motivates the broader project, but the demonstrated scope here is extraction and selection. The registered ontology comparisons and full-novel study remain future work.

## 2. Related work

Open information extraction introduced relational tuple extraction without a separately trained extractor for every target relation [@banko2007]. More recent schema-guided language-model approaches, such as SPIRES, use prompting and normalization to populate knowledge bases [@caufield2024]. Our prototype builds on this view of extracted relationships as structured candidate knowledge. Its contribution is the comparison of two question-handling procedures and an inspection workflow that connects their records, evidence, and measured errors. The compact interface uses ordinary JSON rather than a comprehensive ontology schema.

Qualifications determine how those candidate relationships should be read. OWL-Time distinguishes instants, intervals, durations, and temporal ordering [@time2017]; GRaSP connects content to sources and perspectives [@fokkens2017]. These distinctions motivate our separation of source-supported validity from a query window and of narrated relationships from attributed beliefs. The compact fields cover explicit bounds and simple attribution, while leaving richer temporal and nested-perspective structures outside their expressive range.

Selection, filtering, and coordinated views are established operations in visual analysis [@heer2012]. We apply them to expose the connection between an extracted relationship and its cited source, while showing assessed support separately. This integration matters because valid syntax and plausible labels are insufficient evidence of faithful extraction. The examples make inspectable both a relationship whose wording differs from a reference and a superficially similar relationship assigned to the wrong participant.

## 3. System and methods

### 3.1 Evidence and compact records

Each request receives the complete selected story or excerpt, divided into numbered evidence spans. Literary wording and dialogue are retained. Outputs use eight fields: `subject`, `relation`, `object`, `evidence_ids`, `valid_from`, `valid_until`, `holder`, and `attitude`. Endpoints, predicates, qualifications, and citations are model-authored. Evidence identifiers link to supplied spans; a valid identifier does not establish support.

This authored formatting example, not an experimental output, represents an explicitly narrated carrying interval:

```json
{"subject":"Nara", "relation":"carries", "object":"map",
 "evidence_ids":["X1"], "valid_from":2, "valid_until":5,
 "holder":null, "attitude":null}
```

Intervals include their start and exclude their end. Source-supported intrinsic validity states when the relationship holds. A question's overlapping visibility window selects that record without changing its bounds. An occurrence or observation time, by contrast, does not automatically establish onset or duration. Null bounds mean that these fields supply no supported duration, not that the fact is timeless or false. Relative phrases such as “shortly after” are not converted into numerical bounds.

For belief, subject–relation–object expresses the believed relationship, `holder` identifies the believer, and `attitude` specifies belief. Direct narration uses null attribution fields; this says nothing about whether a participant knows the narrated fact. Missing fields remain distinct from explicit nulls, and no missing qualification is supplied from the reference. Pronouns are resolved only where the excerpt makes the referent clear. Display nodes join identical endpoint strings, without inferred alias merging.

### 3.2 Pre-extraction and contextual extraction

Approach A generates a query-blind fact collection, frozen before any contextual request is transmitted. Fixed CPU rules then select existing records by requested relation category or entity, explicit interval overlap where applicable, and attribution versus narration. The rules neither invent facts nor repair meanings. The synthetic selectors use possession, location, and belief categories; prose selectors use passage-appropriate action and claim categories with limited entity tests.

Approach B independently receives the same complete evidence and a particular question. It generates a contextual collection without consuming A's graph. Both procedures use the same compact representation and authoring conventions. Requests, selectors, and call order were frozen before each batch; each request was attempted once, with all query-blind calls preceding contextual calls. There was no adaptive repair. A failed pre-extraction did not block independent contextual requests.

### 3.3 Model and visualization

The runs use {{model_repository}} in nonthinking mode [@qwen2025] [@qwenmodel], served by vLLM [@kwon2023] on one NVIDIA RTX 4090. Temperature is {{temperature}}, top-p {{top_p}}, and top-k {{top_k}}. Each main-batch request reserves {{reserved_output_tokens}} completion tokens within a {{maximum_context}}-token context. Ordinary JSON is requested without schema-constrained decoding. The pinned snapshot, seed, and complete request provenance are supplied in Supplement S7.

Graph panels show directed, verb-labelled relationships, evidence badges, and generated qualifications. Corresponding panels use consistent positions where practical. Colours, symbols, and line styles distinguish supported, partial, unsupported, unresolved, and source-supported but irrelevant records. Selecting an edge highlights its citation, not a claim of verified support. Sentence-valued objects remain literal nodes. Invalid syntax is displayed as retained text rather than an empty graph. Labelled display subsets preserve all scoring denominators; full outputs accompany the figures.

## 4. Evaluation

### 4.1 Materials and references

Harbor, Orchard, and Museum each contain {{harbor_sentences}} short synthetic evidence sentences. Three questions address possession, interval-dependent location, and belief contrasted with narration. Harbor and Orchard had been used during development; Museum introduced different ordering and bindings, frozen before this batch. Two earlier simple ladders provide development context in Supplement S4. Their evaluation versions and historical syntax recoveries are not pooled with the main results.

The prose comparison uses the complete Lion and Mouse fable in Townsend's translation ({{fable_words}} words), the opening of Alice's Adventures in Wonderland ({{alice_words}} words), and the opening of The Red-Headed League in The Adventures of Sherlock Holmes ({{holmes_words}} words) [@aesop] [@carroll] [@doyle]. These contiguous passages were selected before execution for understandable participants, actions, and dialogue or thought. Two bounded questions per passage address actions and claims. Both receive identical complete evidence. Exact passages, questions, source attribution, retrieval metadata, and notices are retained in the supplement. This is extraction from published fiction, not verification of real events.

Reference facts, acceptable alternatives, normalization, and question scope were frozen before generation. References and subsequent response-linked semantic assessments were authored by Codex, not by independent human reviewers. Synthetic source inventories distinguish relevant targets from supported but irrelevant facts. Prose references are question-scoped: unmatched wording may reflect granularity or coverage limits rather than falsity. No inter-annotator agreement estimate is available.

### 4.2 Separate measures of agreement and meaning

Strict qualified-fact scoring uses one-to-one matching against frozen target alternatives. A complete match requires the prescribed relationship, citations, and temporal and attribution qualifications. Precision divides matches by every emitted prediction in the answer; recall divides them by reference targets; F1 is their harmonic mean. Duplicate, incorrect, irrelevant, and unresolved predictions stay in the denominator, and omissions reduce recall. Bare relationship scores ignore qualifications and citations. Strict F1 is agreement with this annotation contract, not the proportion of factually true statements.

The prose rubric separately records semantic support, partial support, unsupported content, unresolved interpretation, and contextual relevance. Semantic target coverage counts distinct target meanings explicitly preserved. A combined object may express several targets while remaining one prediction. Partial meaning does not receive full target credit, and lack of a known contradiction is not evidence of support. These component measures expose useful information without supplying missing semantics.

### 4.3 Syntax failures and reporting

Strict parsing rejects malformed JSON and ambiguous duplicate keys. The only permitted syntax recovery removes a trailing comma before a closing delimiter outside quoted strings, preserving the raw response and exact edit. It cannot complete a prefix, remove extra braces, or add fields. Unparseable answers have unavailable graph scores, not zero-quality empty graphs.

Results are descriptive, by story or passage and question. Reused pre-extractions, individual records, and calls are not independent experimental units; no significance tests are applied. Within-story averages summarize views only. All executed responses, including failures, remain available for inspection.

## 5. Results

### 5.1 Synthetic stories: preserved intervals, uneven selection and attribution

All {{compact_calls}} compact-story requests were strictly parseable. A achieves strict F1 {{harbor_locations_A_f1}} on every possession and interval-location question (Table 1). B usually retrieves the requested physical relationships but over-selects: all {{compact_irrelevant_answers}} contextual answers contain irrelevant predictions. Orchard's contextual location answer also omits a target, reducing recall to {{orchard_locations_B_recall}}.

Table 1. Compact-story v2 results. A = pre-extract/select; B = contextual extraction. n includes every prediction; P/R/F1 measure strict qualified-fact agreement. Each story has five possession, four location, and four belief/reality targets. Historical recoveries are excluded.

!TABLE:compact

Across emitted facts with identifiable explicitly timed sources, {{interval_retained}} of {{interval_eligible}} retain the full interval, including irrelevant records. This conditional measure is not complete temporal recall. Harbor illustrates the distinction: both approaches preserve requested location intervals, but B adds carrying facts (Figure 2).

Belief structure is less consistent. Orchard's pre-extraction decomposes beliefs correctly, yielding full agreement for A. Harbor and Museum place the believer in the subject and a whole proposition in the object, disrupting selection. Contextual belief answers also contain sentence-valued objects and extensive over-selection. Mean question F1 for A/B is {{harbor_A_mean}}/{{harbor_B_mean}} in Harbor, {{orchard_A_mean}}/{{orchard_B_mean}} in Orchard, and {{museum_A_mean}}/{{museum_B_mean}} in Museum. The fresh Museum story thus reproduces the relevance limitation seen in the earlier material.

### 5.2 Published prose: useful meaning beyond strict matches

The prose batch completed {{prose_calls}} calls, of which {{prose_strict_json}} were strictly parseable; none required permitted syntax recovery. The Holmes pre-extraction and contextual action answer contain an extra terminal closing brace. This leaves both A views and the B action view unavailable (Table 2). The parseable Holmes claims answer contains partial and unresolved interpretations, with no fully covered target.

Table 2. Published prose. S/P/X/U counts supported, partial, unsupported, and unresolved records; coverage counts explicitly preserved semantic targets. Both are Codex assessments. NA means parsing was unavailable, not that the model returned an empty graph.

!TABLE:prose

The fable favours A: it preserves {{fable_actions_A_semantic_targets_covered}} of {{fable_actions_A_references}} action targets semantically, compared with B's {{fable_actions_B_semantic_targets_covered}}, and has higher strict F1. Yet A loses the scope of a conditional plea, while some supported passive formulations fail strict matching. Alice illustrates a different benefit: B's claims preserve {{alice_claims_B_semantic_targets_covered}} of {{alice_claims_B_references}} target meanings versus A's {{alice_claims_A_semantic_targets_covered}}, despite both having zero strict F1. Its action answer assigns reading to the wrong participant. Increased coverage and substantive error coexist.

### 5.3 Cost

The two main batches used {{main_cost}} allocated GPU seconds (Table 3). Allocation includes loading, checks, idle service time, inference, and shutdown; summed request time measures only the calls. Earlier ladder costs and whole-project allocation are reported separately in Supplement S3. These short batches do not estimate production-tail latency or the feasibility of a larger study.

Table 3. Main comparison costs. Each generation is counted once, including pre-extractions reused for several questions. Request time and total service allocation are distinct.

!TABLE:main_cost

## 6. Reading the graphs

The figures connect source wording to actual output, making three different limitations visible. In the fable (Figure 1), the later rescue is captured by both methods, but B omits the earlier capture and release. A's unqualified spare relation also obscures that its cited sentence is a conditional plea. Missing reference information is explained outside the model graph.

Harbor (Figure 2) separates when a relationship holds from when the user asks to see it. Generated intervals extend beyond the shaded query window and remain intact. B's extra carrying assertions are visible in a labelled callout and count as relevance errors, rather than disappearing from the illustration.

Alice (Figure 3) separates partial representation from wrong meaning. The sentence-valued thinks object preserves a rhetorical thought without the requested decomposition. By contrast, assigning reading to Alice instead of her sister changes a participant. The daisy-chain record remains unresolved: considering whether something is worthwhile need not mean judging it worthwhile, and its holder lacks a corresponding attitude. Exact executed questions and complete display accounting accompany the figures in Supplement S6 and the figure notes.

!FIGURE:figure_1_fable

!FIGURE:figure_2_harbor

!FIGURE:figure_3_alice

## 7. Discussion and limitations

The prototype's practical contribution is to make extraction quality inspectable along several axes. Strict scoring exposes deviations from a declared contract; source-linked assessment distinguishes those deviations from loss of meaning. A supported passive construction or sentence-valued thought can be useful without receiving a strict match. Conversely, an attractive graph can contain unsupported participants or omit important relationships. Neither a single score nor graph size alone captures this difference.

Fixed selection works well when the pre-extraction and selector vocabulary fit the question, but it cannot recover omissions or repair attribution. Its exact subject filter rejects “A LION” while expecting “Lion” in the fable claims view, discarding useful records. Some Alice actions are embedded in feeling records excluded by the action selector. The comparison therefore measures these particular extraction and selection implementations, including their limitations.

The examples are small, purposively selected, and partly reused during development. Shared AI assistance in implementation, reference annotation, and assessment creates possible correlated blind spots. Published passages may have appeared in model training. The representation cannot fully express nested perspectives, event structure, or relative temporal ordering. No participant study measures whether the display improves understanding. Together these limits preclude general reliability, unseen-data, and usability claims.

The next steps are focused: author and independent review of semantic judgments and reference scope; prospectively frozen fresh passages; and improvements to participant, attribution, and relevance handling tested without answer-specific hints. Richer query-dependent ontology construction remains a subsequent research question, requiring evidence beyond differences between extracted views.

## 8. Conclusion

We demonstrate extraction of evidence-linked narrative relationships, comparison of fixed selection with independent contextual extraction, and visual inspection of qualifications and errors. Explicit synthetic intervals are preserved in the observed timed records; published prose yields useful action and thought information alongside omissions, wrong participants, and syntax failures. Fixed selection performs better on several questions, while contextual extraction preserves additional semantic targets in Alice's claims. The resulting prototype and measured examples show why strict agreement, semantic preservation, relevance, and unresolved interpretation should be reported together, without collapsing imperfect but inspectable graphs into a single acceptance verdict.

## AI assistance, attribution, and availability

Codex assisted implementation, manuscript preparation, visualization, reference annotation, and semantic assessment. The author is responsible for reviewing the text and judgments. Publication details await author confirmation in the accompanying author-review sheet. Editable sources, retained-output links, vector figures, tables, source attribution, and Project Gutenberg notices accompany this manuscript.
