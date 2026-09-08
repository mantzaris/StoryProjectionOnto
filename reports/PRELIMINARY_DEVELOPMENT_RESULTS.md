# Preliminary development results

Exploratory development only. One preselected easy world and two contrasting contexts over identical evidence; no significance testing, held-out efficacy, production acceptance or p95 claim.

## Results

All base attempts and prepared repairs are retained below. Scores describe canonical outputs even when other validation failed; they are not registered accepted-study results. Missing canonical outputs have no scored draft. Separate acceptance_gated columns apply the existing invalid-output rule to failed LLM attempts (strict F1=0), without declaring every received statement false. Not-transmitted and not-attempted repairs are not model outputs. Absence of a strict match is not proof of an invented claim.

| Condition / context | Attempt | Schema / canonical / structure | Scientific acceptance | Strict P / R / F1 | Finish / input : output tokens | Request s |
|---|---|---|---|---|---|---|
| C0 / 1 | base | True / True / True | False | 0.5000 / 0.6667 / 0.5714 | CPU / 0 : 0 | — |
| C0 / 2 | base | True / True / True | False | 0.0714 / 0.1429 / 0.0952 | CPU / 0 : 0 | — |
| C1 / query-blind | base | False / False / False | False | — / — / — | length / 7603 : 4096 | 94.6517 |
| C2 / 1 | base | False / False / False | False | — / — / — | length / 7839 : 4096 | 102.0061 |
| C2 / 2 | base | False / False / False | False | — / — / — | length / 7827 : 4096 | 86.3009 |
| C1 / query-blind | repair | — / — / — | False | — / — / — | not_transmitted / — : — | — |
| C2 / 1 | repair | — / — / — | False | — / — / — | not_attempted / — : — | — |
| C2 / 2 | repair | — / — / — | False | — / — / — | not_attempted / — : — | — |
| A-FixedSelect / 1 | blocked | — / — / — | False | — / — / — | blocked / — : — | — |
| A-FixedSelect / 2 | blocked | — / — / — | False | — / — / — | blocked / — : — | — |

## Available contextual measures

These use the existing scoring definitions. Unmatched reference grounding is a reference-coverage result, not proof of factual falsity. No complete C2 draft exists to score conditionally; its acceptance-gated strict F1 is zero.

| Condition / context | Node F1 | Essential temporal accuracy | Reference grounding | Rare-pivotal recall | Complete rare support path |
|---|---|---|---|---|---|
| C0 / 1 | 0.8889 | 0.6667 | 0.5000 | 1.0000 | 0.0000 |
| C0 / 2 | 0.4706 | 0.1429 | 0.0714 | — | — |

## Evidence, contexts and actual graphs

The [interactive comparison](figures/preliminary_development_comparison.html) includes all supplied synthetic prose, both contexts, every available generated record, readable assertion bindings and complete qualifications. Truncated outputs show only complete recovered JSON members, explicitly not reconstructed/accepted graphs. Identical labels use fixed visual anchors; assertion junctions are display-only, not invented event semantics.

[Machine-readable measures](tables/preliminary_development_results.csv) · [Graph records](tables/preliminary_development_graphs.json)

## Allocation and gates

Historical allocation 6716.108081 s; this phase 858.564408 s; cumulative 7574.672489 s. Open GPU/service journals: 0/0. The phase ceiling is 3,600 s. The global scheduled/hard limits remain 33,660/36,000 s; no complete-study admission is claimed.

Service starts: 2; transmitted generations: 3; reserved attempts: 4. Both authorized starts were used; unused time does not authorize a third start. No targeted model repair was transmitted. vLLM is stopped; the pod remains intact.

Peak sampled GPU VRAM / process RAM / project occupancy (bytes): 22793945088 / 7109718016 / 16912713216. These are sampled peaks, not continuous maximum guarantees. Terminal full storage checks passed.

Remaining registered inventory is preserved in the restricted run manifest; it has not been removed or reset. A few development calls cannot establish production throughput. Held-out execution remains independently reviewed and gated.

## Interpretation and limitations

C0 rows reuse actual, unchanged, hash-verified CPU projections; no authored graph substitutes for an LLM output. Its existing extraction gate is passed=True: precision 104/122, recall 104/114, 5/5 fixture families, evidence-reference validity 100.0%. Its separate aggregate contextual P=54/275, R=54/91, F1=0.295082 is unchanged. These two-context values are a subset, not a replacement gate.

Scientific acceptance requires positive source support; unresolved matching/description coverage is not a pass. The conservative description recognizer only certifies direct source substrings and does not call unmatched paraphrases false. FixedSelect cannot run without an actual accepted C1 seal and intact packing. C2 runs independently of C1. Output-cap failures are capacity failures, not evidence of universal model incapability.

- C0 context 1 base: Exploratory source/description assessment includes unresolved matches; see HTML
- C0 context 2 base: Exploratory source/description assessment includes unresolved matches; see HTML
- C1 context None base: vLLM response is not one guided JSON choice
- C2 context 1 base: vLLM response is not one guided JSON choice
- C2 context 2 base: vLLM response is not one guided JSON choice
- C1 context None repair: Missing repair flag rejected the prepared call before a GPU event; subsequent bookkeeping masked the original exception. CPU-reproduced diagnosis; no new model response.
- C2 context 1 repair: Prepared repair not executed after the controller failure; no replacement output.
- C2 context 2 repair: Prepared repair not executed after the controller failure; no replacement output.
- A-FixedSelect context 1 base: No accepted sealed C1; not attempted
- A-FixedSelect context 2 base: No accepted sealed C1; not attempted

Pinned model: Qwen/Qwen3-8B-AWQ at `4da05a8edb55c6046cce958586c33b61da07bb79`, 12,288 total tokens; vLLM 0.10.2 / XGrammar, no fallback, whitespace restriction enabled. No model change or multi-call graph construction. Each repair replaces one failed base; no semantic merging across outputs.

The prepared C1 repair is model-query-blind, but it was not transmitted. Its intended timing was after the initial C2 bases and could not retrospectively satisfy the registered physical pre-query barrier. No production adoption or complete acceptance is claimed.

## Remaining registered work

Unvalidated legacy remaining-work proxy: 40162.013213 s; all-in with preserved actual allocation: 47736.685702 s. This is not a calibrated production forecast. Failed completion speeds receive no credit. The original nine-hour target was not met; the amended scheduled ceiling remains unchanged.

The fallback gate still needs its C1, two C2, FixedSelect and conditional repair forms plus restart/resume validation. These calls remain earmarked within reserve rows, not added twice. Superseded historical 14B acceptance rows do not mean fallback acceptance passed.

| Remaining class | Count |
|---|---|
| gpu_session_start | 5 |
| development_c1 | 4 |
| development_c2 | 12 |
| development_fixed_select | 4 |
| development_ablation | 3 |
| development_repair | 1 |
| test_c1 | 24 |
| test_c2 | 72 |
| test_fixed_select | 72 |
| paraphrase_c2 | 12 |
| scripted_feedback_c2 | 6 |
| researcher_trace_c2 | 3 |
| ablation_no_context | 12 |
| ablation_no_temporal_epistemic | 8 |
| ablation_no_rare_guard | 8 |
| case_c1 | 4 |
| case_c2 | 8 |
| case_full_index_c2 | 1 |
| reserve_long | 1 |
| reserve_standard | 8 |
| reserve_short | 4 |

No GPU output was scientifically accepted. This development configuration failed within the used allowance. The semantic repairs remain model-untested because of the controller error; their failure or success cannot be inferred. No further diagnostic phase is automatically initiated.

## What changed and what was wrong

The two actual C0 projections retain the same named nodes. Context 2 adds an office-holding assertion and a greeting to the context-1 selection, without forming a separate continuing-office node. Its lower contextual score is a projection/relevance result, not a reversal of the extraction-competence gate.

The received C2 prefixes differ: context 1 reports selection with an excessive created-entity inventory; context 2 reports contextual-type and relation operations with mixed created record kinds. Neither completed its schema/entity declarations, so these surface differences cannot establish correct contrastive ontology construction.

For a readable source example, evidence e1 directly narrates Fara Cedar's membership in Cedar Circle at story step 1. Both C2 prefixes nevertheless attach holder-level knowledge and intrinsic point validity. Narration does not establish a participant's knowledge, and observation time does not establish intrinsic onset or duration. These are source-only findings on complete received assertion records, not full canonical validation. Missing endpoint/type declarations remain unresolved; the CPU does not fill them from descriptions.

| Received LLM prefix | Complete entity / event / assertion records | Source-only confirmed defect categories | Construction claims |
|---|---|---|---|
| C1 / query-blind | 8 / 0 / 1 | unsupported_attribution | supported_description (claims 0 created node IDs) |
| C2 / 1 | 0 / 0 / 9 | unsupported_attribution, unsupported_intrinsic_precision | selection (claims 100 created node IDs) |
| C2 / 2 | 0 / 0 / 6 | unsupported_attribution, unsupported_intrinsic_precision | contextual_type (claims 50 created node IDs); schema_relation (claims 0 created node IDs) |

Counts describe complete JSON members received before truncation, not complete graphs. A zero received-node count does not mean the model authored an empty graph. C2's excessive creation claims and repeated unsupported epistemic form show that output capacity is not the only remaining issue. The prepared semantic feedback was not model-tested.
