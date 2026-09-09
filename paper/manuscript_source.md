# Inspectable Narrative Graphs: An Exploratory Extraction Prototype

a.v.mantzaris

## Abstract

Narrative readers need different views of the same text: who carries an object, where participants are located, or what somebody believes rather than what the narrator states. We present an exploratory prototype that extracts compact, evidence-linked graphs and makes both useful information and errors inspectable. We compare query-blind extraction followed by fixed selection (A) with independent, direct contextual extraction (B). Two simple diagnostic ladders provide development context; the main evaluation uses three short synthetic stories and three published narrative passages. References and semantic assessments are Codex-authored. The corrected story batch produced {{compact_strict_json}} parseable outputs from {{compact_calls}} calls. Fixed selection achieved perfect strict qualified-fact agreement on every ownership/possession and interval-location view, whereas contextual answers retained irrelevant facts. On the fable action question, strict F1 was {{fable_actions_A_strict_qualified_f1}} for A and {{fable_actions_B_strict_qualified_f1}} for B. Alice's contextual claims answer preserved {{alice_claims_B_semantic_targets_covered}} of {{alice_claims_B_references}} semantic targets despite zero strict F1. Published-prose parsing succeeded in {{prose_strict_json}} of {{prose_calls}} calls; dialogue attribution remained difficult. These small development demonstrations establish inspectable extraction, not general reliability, user benefit, or query-dependent ontology construction.

## 1. Introduction

A narrative contains more relationships than a reader needs for any particular question. An ownership view, a map of locations, and a summary of competing beliefs may all be useful, but they answer different information needs. A graph can expose participants and connections that are difficult to compare across paragraphs. It can also conceal a serious error behind a plausible arrow. Reading an object and merely looking into it are different actions; believing that an object is somewhere does not establish its location.

We study this practical boundary between extraction and inspection. The prototype produces small fact collections with readable endpoints, source citations, optional validity bounds, and explicit attribution. Its visualization links records to the supplied text and displays omissions, partial meaning, unsupported assignments, and unresolved interpretations. The immediate objective is not a perfectly reconstructed story world. It is an output whose useful and defective parts can be examined without silently repairing the model's answer.

The broader project asks how a user's context might change the ontology itself: which identities, events, abstractions, and qualified assertions should exist. That objective motivates the present work but is not its demonstrated contribution. The experiments here compare two extraction procedures using a shared compact representation. They neither establish contextual identity formation nor implement an accepted registered ontology-construction comparison. The larger benchmark, independent review, and full-novel transfer study remain uncompleted work.

Three questions guide this exploratory account. Can a compact interface recover relationships and explicit qualifications as evidence becomes less synthetic? Does direct contextual extraction produce more relevant views than selecting an existing extraction? Finally, what does visual inspection reveal that strict tuple agreement alone misses? We answer descriptively, including results where the simpler selection baseline performs better. The contribution is a reproducible prototype and an inspectable collection of measured examples and failures, rather than a superiority claim.

## 2. Related work

Open information extraction treats text as a source of relational tuples without requiring a separately trained extractor for each target relation. Banko and colleagues introduced this approach for web-scale extraction [@banko2007]. Our compact subject–relation–object records share that relational emphasis, but add source identifiers and selected temporal and epistemic fields. We do not evaluate web-scale extraction or claim a new open-extraction algorithm. The narrative setting instead makes relevance, attribution, and incomplete representations particularly visible.

Language models can populate structured knowledge representations through prompted extraction. SPIRES uses schema-guided prompting and normalization for knowledge-base population [@caufield2024]. This work provides an important precedent for treating model output as structured candidate knowledge rather than only prose. Our demonstrated interface is deliberately smaller: ordinary JSON, readable endpoints, and no comprehensive local ontology schema. Its simplicity does not eliminate semantic errors. The retained experiments show why successful serialization, correct relationships, and complete qualifications must be assessed separately.

Temporal representation also requires more than attaching a date to an edge. OWL-Time distinguishes instants, intervals, durations, and ordering relationships [@time2017]. Here we preserve explicitly supplied interval bounds while separating those bounds from a question's visibility window. We do not implement the full temporal ontology. In particular, the compact numeric fields cannot faithfully express every relative ordering phrase in literature, and the absence of bounds should not be read as unrestricted validity.

Source perspective is a related but distinct problem. GRaSP grounds information in sources and represents perspectives on content [@fokkens2017]. Our holder and attitude fields are a limited practical counterpart, not an implementation of GRaSP. They permit a belief to coexist with differing narration, but do not provide a complete treatment of nested reports, modal scope, or competing interpretations. A model may preserve a thought in a sentence-valued object while failing the chosen decomposed graph format.

Visual analysis research identifies selection, filtering, and coordinated views as important interactive operations [@heer2012]. We use these operations to connect an extracted relationship with its cited evidence and assessed status. A citation highlight is a provenance link, not proof of support. The current interface is an inspection aid demonstrated through examples; there is no participant study, evidence of improved task performance, or measured usability claim.

## 3. System and methods

### 3.1 Evidence and compact records

Each request receives the complete selected story or excerpt, divided into numbered evidence spans. Literary wording is retained, including dialogue; numbering does not turn it into simplified synthetic sentences. Outputs use eight fields: `subject`, `relation`, `object`, `evidence_ids`, `valid_from`, `valid_until`, `holder`, and `attitude`. Endpoints and predicates remain model-authored. Evidence references must name supplied spans, but syntactically valid references do not establish that the cited text supports the relationship.

The following formatting illustration is authored for explanation, not an experimental output. It expresses a directly narrated carrying interval for unrelated participants:

```json
{"subject":"Nara", "relation":"carries", "object":"map",
 "evidence_ids":["X1"], "valid_from":2, "valid_until":5,
 "holder":null, "attitude":null}
```

The interval convention includes the start and excludes the end. These are source-supported intrinsic validity bounds for the relationship, not occurrence timestamps. If evidence explicitly states that a relationship holds over that interval, an overlapping query window selects the record without changing its bounds. By contrast, observing an action at one moment does not establish its onset or duration. Null bounds mean that these fields do not supply a supported duration; they mean neither timeless truth nor falsehood. Relative phrases such as “shortly after” are not converted into invented numbers.

For an attributed belief, the underlying subject, relation, and object describe what is believed; `holder` identifies the believer and `attitude` records the belief. Direct narration uses null attribution fields. This does not assert that a participant lacks knowledge of narrated events. Missing keys are reported separately from explicit nulls. Neither selection nor scoring fills missing qualifications from reference answers. Pronouns remain unresolved unless the supplied text makes their referents clear; display nodes join only identical endpoint strings.

### 3.2 Two extraction procedures

Approach A first asks the model for a query-blind collection of directly supported facts. Each actual output is frozen before any contextual request is transmitted. A deterministic selector then keeps existing records by the requested relation category or entity, by explicit interval overlap where requested, and by attribution versus narration. The selector neither invents facts nor repairs their meaning. A malformed belief representation can therefore interfere with later selection even if its sentence-valued object contains useful information.

Approach B independently receives the same complete evidence plus a particular question and generates a contextual collection. It does not consume A's graph. Both procedures use the same scientific authoring conventions and compact fields. Question-specific outputs are not computational transformations of one another. The synthetic selectors use ownership, carrying, location, and attribution categories; published-prose selectors use passage-appropriate action or claim categories and limited entity tests fixed before generation. Their brittleness is part of the observed baseline, not hidden by retrospective corrections.

All requests within each main batch were frozen before execution and attempted once, with query-blind calls preceding contextual calls. There was no adaptive repair within those batches. A failed pre-extraction did not prevent independent contextual requests. This comparison concerns extraction and selection, not the registered C1/C2 conditions or a mechanically sealed ontology-construction experiment.

### 3.3 Model and inspection interface

The retained runs use {{model_repository}}, revision `{{model_revision}}`, served locally by vLLM on one NVIDIA RTX 4090. Qwen3 supports thinking and nonthinking modes; these requests use the nonthinking setting [@qwen2025]. The pinned AWQ snapshot follows the official model interface [@qwenmodel]. The retained configuration uses temperature {{temperature}}, top-p {{top_p}}, top-k {{top_k}}, and seed {{model_seed}}. The main batches reserve {{reserved_output_tokens}} completion tokens within a {{maximum_context}}-token context. Ordinary JSON is requested without the production guided-decoding grammar. vLLM supplies the serving infrastructure, not a correctness guarantee [@kwon2023].

The display renders each actual record as a directed, verb-labelled connection, with evidence badges and generated qualifications. Corresponding panels use consistent positions where practical. Colours are paired with symbols and line styles: supported, partial, unsupported, unresolved, or source-supported but irrelevant. Sentence-valued objects remain literal nodes. Invalid syntax is displayed as retained text, never as an intentionally empty graph. Full outputs accompany compact paper subsets, whose omitted record counts are explicit and do not affect scoring denominators.

## 4. Evaluation design

### 4.1 Materials and development provenance

The main synthetic comparison contains Harbor, Orchard, and Museum, each with {{harbor_sentences}} short evidence sentences. Questions ask about ownership/possession, locations overlapping a specified interval, and belief contrasted with narration. Every question receives identical complete evidence within its story. Harbor and Orchard were previously used during development; Museum was a fresh counterpart with different ordering and bindings, frozen before this batch. All are purposively constructed development examples, not a random sample or held-out efficacy set.

The published-text demonstration uses the complete Lion and Mouse fable in Townsend's translation ({{fable_words}} words), the opening of Alice's Adventures in Wonderland ({{alice_words}} words), and the opening of The Red-Headed League in The Adventures of Sherlock Holmes ({{holmes_words}} words) [@aesop] [@carroll] [@doyle]. These contiguous passages were selected before model execution for understandable participants, actions, and dialogue or thought. Two bounded questions per passage target actions and claims. Source URLs, exact excerpts, retrieval timestamps, hashes, and applicable notices are retained. The task is extraction from published fiction, not verification of real-world events.

Two earlier simple ladders supply development context. They progress from direct facts and contextual selection to explicit intervals, belief, and person/office questions. Their revised formats and normalization policies are separate evaluation versions. We do not pool their scores with the story or prose comparisons. Likewise, syntax-recovered historical responses remain derived records, not newly generated successes. Earlier unsuccessful attempts to use a much larger ontology contract motivated simplification but are not silently reclassified as compact-interface results.

### 4.2 References and assessment

Reference facts, acceptable alternatives, normalization rules, selectors, questions, and call order were frozen before each batch. References were authored by Codex from the supplied evidence, not by independent human reviewers. For synthetic stories, the explicit source inventory also distinguishes relevant targets from supported but irrelevant statements. Published references are scoped to the question; an unmatched prediction is not automatically false, because wording, granularity, and reference coverage can be limiting factors.

Strict qualified-fact scoring uses one-to-one matching against frozen target alternatives. A complete match requires the prescribed relationship, evidence references, and temporal and attribution qualifications. Precision divides complete matches by all emitted predictions in the applicable answer; recall divides them by the reference targets. F1 is their harmonic mean. Duplicate, incorrect, irrelevant, and unresolved predictions stay in the denominator; omitted targets reduce recall. Bare relationship scores separately ignore qualifications and citations in the main experiments. Strict F1 measures agreement with this annotation contract, not the proportion of factually true statements.

Published outputs also receive response-linked, Codex-authored semantic assessments under the declared rubric. These separate full support, partial support, unsupported content, unresolved interpretation, and contextual relevance. Semantic target coverage counts distinct target meanings explicitly preserved, without inventing missing semantics. One combined object may explicitly express multiple targets while remaining one predicted record. Partial support does not become full correctness, and absent evidence of contradiction is not positive support. There is no independent review or inter-annotator agreement estimate.

### 4.3 Failure handling and reporting units

Strict parsing rejects malformed JSON and ambiguous duplicate keys. The only semantic-neutral syntax recovery removes a trailing comma immediately before a closing delimiter outside quoted strings, preserving raw bytes and an edit record. It does not complete a prefix, remove extra braces, or add missing fields. Unparseable answers have unavailable graph scores, not a zero-quality empty graph. Every executed call and failed response remains inspectable.

Results are reported by story or passage and question, with no significance tests. Reused pre-extractions, repeated development questions, individual records, and calls are not independent experimental units. Descriptive within-story averages summarize views only. Small-call timing is not a reliable production throughput distribution. These restrictions allow useful examples to be reported without treating an exploratory debugging history as a confirmatory experiment.

## 5. Results

### 5.1 Compact stories: intervals work better than selection and belief structure

The compact-story v2 batch completed {{compact_calls}} requests, all strictly parseable. Table 1 reports every fresh answer after selection. A achieves strict F1 {{harbor_locations_A_f1}} on all possession and interval-location questions. B generally returns the requested physical relationships but includes extra source-supported records. All {{compact_irrelevant_answers}} contextual answers contain irrelevant predictions. Orchard's contextual location answer additionally omits a target, giving recall {{orchard_locations_B_recall}}. A smaller graph is not inherently better; here the advantage is supported by retained target matches and explicit relevance scope.

Table 1. Fresh compact-story v2 results only. A = pre-extract/select; B = direct contextual extraction. n counts every prediction; P/R/F1 are strict qualified-fact scores. References contain five possession, four location, and four belief/reality targets per story. Historical syntax recoveries are excluded.

!TABLE:compact

Interval preservation is encouraging but bounded. The existing audit finds {{interval_retained}} of {{interval_eligible}} emitted facts with identifiable explicitly timed sources preserve the full interval, including irrelevant records. This is conditional on emission and source identification, not complete temporal recall. Harbor illustrates the difference (Figure 2): both methods retain the requested location intervals, but B also returns carrying facts. No generated interval is clipped merely to make it resemble the query window.

Belief is less stable. Orchard's pre-extraction decomposes the supplied beliefs correctly, supporting A's perfect belief/reality view. Harbor and Museum instead place the believer in the subject and a whole proposition in the object, disrupting the fixed belief selector. Contextual belief answers similarly contain sentence-valued objects and extensive over-selection. Descriptive mean question F1 for A/B is {{harbor_A_mean}}/{{harbor_B_mean}} in Harbor, {{orchard_A_mean}}/{{orchard_B_mean}} in Orchard, and {{museum_A_mean}}/{{museum_B_mean}} in Museum. The fresh Museum example shows the same relevance limitation, not evidence of generalization to a population.

### 5.2 Published prose: useful meaning and substantial losses

The prose batch completes {{prose_calls}} calls, with {{prose_strict_json}} strictly parseable outputs and no permitted syntax recoveries. Table 2 separates strict agreement from semantic assessment. The Holmes pre-extraction and contextual action answer both contain an extra terminal closing brace, despite a completed response. Consequently, both A views and the B action view are unavailable. The usable Holmes claims answer has partial and unresolved content but no fully covered target; speaker roles remain unreliable.

Table 2. Published-prose results. S/P/X/U counts supported, partial, unsupported, and unresolved records; coverage counts explicitly preserved semantic targets, both assessed by Codex. NA indicates unavailable parsing, not an empty prediction. These measures do not replace strict scores or imply independent review.

!TABLE:prose

The fable provides a baseline advantage. A preserves {{fable_actions_A_semantic_targets_covered}} of {{fable_actions_A_references}} action targets semantically, compared with B's {{fable_actions_B_semantic_targets_covered}}. Its strict F1 is also higher. Yet A contains a conditional-spare claim presented without its conditional scope, while some valid passive formulations fail strict matching. Alice shows the opposite kind of partial benefit: B's claims preserve {{alice_claims_B_semantic_targets_covered}} of {{alice_claims_B_references}} target meanings against A's {{alice_claims_A_semantic_targets_covered}}, though both strict scores remain zero. The action answer still assigns reading to the wrong participant. More contextual coverage does not erase that substantive error.

### 5.3 Development context and cost

The first simple ladder succeeds on its tested direct extraction, selection, and explicit-interval cases before encountering belief decomposition and citation/interval failures. The revised ladder preserves its regression successes and handles the original and fresh simple belief cases, while office examples remain imperfect. Their questions permit the same office-mediated organization, so high factual agreement would not establish a construction contrast. Neither ladder supports a general reliability claim. Their complete original scores remain separate in the supplement.

Table 3 reports the allocated service cost of each presented batch separately from historical whole-project consumption. The two main batches account for {{main_cost}} allocated seconds; the preserved project total at their conclusion is {{project_cost}} seconds, including earlier development and failures. Allocation includes loading, checks, idle service time, inference, and shutdown. No new inference was performed for this manuscript. These measurements neither estimate production p95 latency nor establish feasibility of the original full registered study.

Table 3. Costs of distinct retained batches. Request seconds sum actual calls; allocation includes the surrounding service lifetime. Accuracy across versions is not pooled.

!TABLE:cost

## 6. Qualitative inspection

Figure 1 makes the computational distinction visible. A's arrows are selected from an earlier extraction; B is independently generated from the fable and action question. The late hunter-capture and Mouse-rescue sequence is useful in both. B loses the Lion's earlier capture and release of the Mouse, while A's spare assertion turns a conditional plea into an unqualified relation. The graph marks that partial representation without drawing a corrective reference edge into the model output.

Figure 2 links Harbor's explicit source sentences, question, output records, and intervals. The shaded query window is a selection criterion. The longer generated location intervals remain intact, so the display does not suggest that entering the question window caused a relationship to begin. B's additional carrying facts are reproduced in a labelled callout and remain scoring errors for relevance. A and B share evidence, not a derivation arrow between their graphs.

Figure 3 juxtaposes two Alice requests. The action output correctly preserves peeping into the book but assigns reading to Alice rather than the sister. The claims output captures absent book content and Alice's rhetorical thought. Its sentence-valued object is shown literally: mental meaning can survive while the requested decomposition fails. The daisy-chain statement remains unresolved because considering whether something is worthwhile is not necessarily judging it worthwhile. A holder without an attitude further leaves the required qualification incomplete.

!FIGURE:figure_1_fable

!FIGURE:figure_2_harbor

!FIGURE:figure_3_alice

## 7. Discussion and limitations

The most consistent lesson is that usable syntax, relevant selection, and faithful meaning are different achievements. Compact records enable inspection and ordinary scoring where a comprehensive interface proved difficult, but the reduced burden does not make the model reliably obey scope or attribution conventions. Conversely, a strict mismatch may be a supported passive construction or a combined object preserving multiple ideas. Reporting only strict F1 would obscure those distinctions; reporting only attractive graphs would obscure errors and omissions.

Fixed selection has a practical advantage when its extracted records and vocabulary fit the task. It also has a ceiling: it cannot recover omitted information or repair malformed attribution. In the fable claims view, an exact subject filter rejects “A LION” while expecting “Lion,” discarding useful existing records. Alice's pre-extracted action information can likewise be embedded in a feeling record excluded by the selector. These implementation limitations prevent interpreting the comparison as a broad test of classical versus contextual reasoning.

The materials are tiny, selected for interpretability, and repeatedly developed in part. Reference authorship and assessment share an AI-assisted workflow with implementation, creating potential blind spots and correlated judgments. Published passages may be familiar from model training. The compact format lacks event structure, nested perspective, and many relative temporal distinctions. No participant evaluation establishes that the visualization improves understanding. Resource histories include unsuccessful engineering paths, and the presented short batches cannot certify the larger study's timing or scientific gates.

The next research steps are therefore specific: obtain independent review of reference scope and semantic judgments; test prospectively frozen fresh passages; improve participant, attribution, and relevance handling without answer-specific hints; and compare representation choices under balanced resources. Richer query-dependent ontology construction should be evaluated only after its own acceptance and packing requirements are met. It remains an empirical objective, not a conclusion inferred from differently shaped extraction graphs.

## 8. Conclusion

This prototype produces inspectable, evidence-linked narrative graphs with measured successes and failures. Explicit synthetic intervals are often preserved in emitted records, but contextual extraction over-selects facts and attribution remains fragile. Published prose yields useful action and thought information alongside omissions, wrong participants, and syntax failures. The retained comparison supports neither universal reliability nor contextual superiority. Its value is a reproducible demonstration in which strict agreement, semantic preservation, relevance, and uncertainty remain visible rather than collapsed into one acceptance verdict.

## AI assistance, attribution, and availability

Codex assisted implementation, manuscript preparation, visualization, reference annotation, and semantic assessment. Those annotations are not independent human review; the author remains responsible for publication decisions. Repository author identity is retained as a.v.mantzaris, pending confirmation of publication name and affiliation. Funding and conflict declarations require author completion. Source attribution and Project Gutenberg notices accompany the excerpts. Editable text, vector figures, tables, retained-output links, and regeneration instructions are provided with the manuscript; no novel corpus or private runtime material is added to the paper bundle.
