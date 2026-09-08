# Preliminary development results

Exploratory development only. One preselected easy world and two contrasting contexts over identical evidence; no significance testing, held-out efficacy, production acceptance or p95 claim.

## Results

All base attempts and prepared repairs are retained below. Scores describe canonical outputs even when other validation failed; they are not registered accepted-study results. Missing canonical outputs have no scored draft. Separate acceptance_gated columns apply the existing invalid-output rule to failed LLM attempts (strict F1=0), without declaring every received statement false. Not-transmitted and not-attempted repairs are not model outputs. Absence of a strict match is not proof of an invented claim.

| Condition / context | Attempt | Schema / canonical / structure | Scientific acceptance | Strict P / R / F1 | Finish / input : output tokens | Request s |
|---|---|---|---|---|---|---|
| C0 / 1 | base | True / True / True | False | 0.5000 / 0.6667 / 0.5714 | CPU / 0 : 0 | — |
| C0 / 2 | base | True / True / True | False | 0.0714 / 0.1429 / 0.0952 | CPU / 0 : 0 | — |
| C1 / query-blind | base | — / — / — | False | — / — / — | length / 7603 : 4096 | 94.6517 |
| C2 / 1 | base | — / — / — | False | — / — / — | length / 7839 : 4096 | 102.0061 |
| C2 / 2 | base | — / — / — | False | — / — / — | length / 7827 : 4096 | 86.3009 |
| C1 / query-blind | repair | — / — / — | False | — / — / — | blocked / — : — | 1.2170 |
| C2 / 1 | repair | — / — / — | False | — / — / — | length / 7142 : 5120 | 141.0353 |
| C2 / 2 | repair | — / — / — | False | — / — / — | length / 7130 : 5120 | 112.3729 |
| C1 / query-blind | stage A base | — / — / — | False | — / — / — | length / 5996 : 5120 | 143.1609 |
| C2 / 1 | stage A base | — / — / — | False | — / — / — | length / 5824 : 5120 | 90.6464 |
| C2 / 2 | stage A base | — / — / — | False | — / — / — | length / 5795 : 5120 | 83.5846 |
| C2 / 2 | stage A repair | — / — / — | False | — / — / — | length / 5863 : 6144 | 147.1145 |
| C1 / query-blind | repair | — / — / — | False | — / — / — | not_transmitted / — : — | — |
| A-FixedSelect / 1 | blocked | — / — / — | False | — / — / — | blocked / — : — | — |
| A-FixedSelect / 2 | blocked | — / — / — | False | — / — / — | blocked / — : — | — |

## Available contextual measures

These use the existing scoring definitions. Unmatched reference grounding is a reference-coverage result, not proof of factual falsity. Conditional draft scores are available only when canonical reconstruction completes; failed LLM attempts also retain separate acceptance-gated scores.

| Condition / context | Node F1 | Essential temporal accuracy | Reference grounding | Rare-pivotal recall | Complete rare support path |
|---|---|---|---|---|---|
| C0 / 1 | 0.8889 | 0.6667 | 0.5000 | 1.0000 | 0.0000 |
| C0 / 2 | 0.4706 | 0.1429 | 0.0714 | — | — |

## Evidence, contexts and actual graphs

The [interactive comparison](figures/preliminary_development_comparison.html) includes all supplied synthetic prose, both contexts, every available generated record, readable assertion bindings and complete qualifications. Truncated outputs show only complete recovered JSON members, explicitly not reconstructed/accepted graphs. Identical labels use fixed visual anchors; assertion junctions are display-only, not invented event semantics.

[Machine-readable measures](tables/preliminary_development_results.csv) · [Graph records](tables/preliminary_development_graphs.json)

## Allocation and gates

Historical allocation 6716.108081 s; this phase 2267.141867 s; cumulative 8983.249948 s. Open GPU/service journals: 0/0. The phase ceiling is 3,600 s. The global scheduled/hard limits remain 33,660/36,000 s; no complete-study admission is claimed.

Service starts: 6; transmitted HTTP requests: 10; responses with generation tokens: 9; reserved attempts: 11. The continuation allows at most 6 total starts within the same phase. Transmitted parent repairs: 3. vLLM is stopped; the pod remains intact. No further phase is initiated.

Peak sampled GPU VRAM / process RAM / project occupancy (bytes): 22793945088 / 7201918976 / 17042163200. These are sampled peaks, not continuous maximum guarantees. Terminal full storage checks passed.

Remaining registered inventory is preserved in the restricted run manifest; it has not been removed or reset. A few development calls cannot establish production throughput. Held-out execution remains independently reviewed and gated.

## Interpretation and limitations

C0 rows reuse actual, unchanged, hash-verified CPU projections; no authored graph substitutes for an LLM output. Its existing extraction gate is passed=True: precision 104/122, recall 104/114, 5/5 fixture families, evidence-reference validity 100.0%. Its separate aggregate contextual P=54/275, R=54/91, F1=0.295082 is unchanged. These two-context values are a subset, not a replacement gate.

Scientific acceptance requires positive source support; unresolved matching/description coverage is not a pass. The conservative description recognizer only certifies direct source substrings and does not call unmatched paraphrases false. FixedSelect cannot run without an actual accepted C1 seal and intact packing. C2 runs independently of C1. Output-cap failures are capacity failures, not evidence of universal model incapability.


### Continuation capacity and execution correction

The same C1/C2 development allocation is 7,168 input / 5,120 output within 12,288 total tokens. The named nested representation and lossless adapter are unchanged. Full evidence and essential source-grounded feedback are retained; prior responses remain complete in restricted lineage rather than being duplicated in model input.

Existing graph and single-kind creation budgets were added to the grammar. The first transmitted C1 repair encountered a server-integration failure: vLLM returned HTTP 200 with a streaming rejection of `uniqueItems`, despite passing standalone XGrammar compilation. No generation occurred. This preflight coverage gap is an implementation error, not a model-semantic failure. C1 was not repeated. The last single-response start used the actual vLLM validator before allocation and removed only that unsupported decoder keyword; identical post-validation uniqueness constraints remained. Both C2 repairs were then transmitted on one service, independently of C1, with no blind base repetition or further repair. Authored capacity fits did not predict successful complete generation.
- C0 context 1 base: Exploratory source/description assessment includes unresolved matches; see HTML
- C0 context 2 base: Exploratory source/description assessment includes unresolved matches; see HTML
- C1 context None base: vLLM response is not one guided JSON choice
- C2 context 1 base: vLLM response is not one guided JSON choice
- C2 context 2 base: vLLM response is not one guided JSON choice
- C1 context None repair: server emitted a streaming error event
- C2 context 1 repair: vLLM response is not one guided JSON choice
- C2 context 2 repair: vLLM response is not one guided JSON choice
- C1 context None base: vLLM response is not one guided JSON choice
- C2 context 1 base: vLLM response is not one guided JSON choice
- C2 context 2 base: vLLM response is not one guided JSON choice
- C2 context 2 repair: vLLM response is not one guided JSON choice
- C1 context None repair: Missing repair flag rejected the prepared call before a GPU event; subsequent bookkeeping masked the original exception. CPU-reproduced diagnosis; no new model response.
- A-FixedSelect context 1 base: No accepted sealed C1; not attempted
- A-FixedSelect context 2 base: No accepted sealed C1; not attempted

### Allocated service intervals

| Start | Allocated seconds | Cumulative seconds | Stop reason |
|---|---:|---:|---|
| 1 | 572.116016 | 7288.224097 | completed |
| 2 | 286.448392 | 7574.672489 | workload_exception |
| 3 | 197.182472 | 7771.854961 | unsafe_or_unresolved_transport |
| 4 | 428.570203 | 8200.425164 | completed |
| 5 | 468.509993 | 8668.935157 | completed |
| 6 | 314.314791 | 8983.249948 | completed |

Pinned model: Qwen/Qwen3-8B-AWQ at `4da05a8edb55c6046cce958586c33b61da07bb79`, 12,288 total tokens; vLLM 0.10.2 / XGrammar, no fallback, whitespace restriction enabled. No model change. Historical rows used single-response construction. Separately labeled staged rows use the explicitly amended A/B/C protocol, not successful execution of the original protocol.

C1 repair requests remain model-query-blind. A late repair/seal cannot retrospectively satisfy the registered physical pre-query barrier. The old untransmitted reservation remains separately reported. No production adoption or complete acceptance is claimed.

## Remaining registered work

Unvalidated legacy remaining-work proxy: 40162.013213 s; all-in with preserved actual allocation: 49145.263161 s. This is not a calibrated production forecast. Failed completion speeds receive no credit. The original nine-hour target was not met; the amended scheduled ceiling remains unchanged.

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

No GPU output was scientifically accepted. This development configuration failed within the used allowance. Structural completion, confirmed semantic errors and unresolved assessments are distinguished below. No further diagnostic phase is automatically initiated.

## What changed and what was wrong

The two actual C0 projections retain the same named nodes. Context 2 adds an office-holding assertion and a greeting to the context-1 selection, without forming a separate continuing-office node. Its lower contextual score is a projection/relevance result, not a reversal of the extraction-competence gate.

The original base C2 prefixes differ: context 1 reports selection with an excessive created-entity inventory; context 2 reports contextual-type and relation operations with mixed created record kinds. Neither completed its schema/entity declarations, so these surface differences cannot establish correct contrastive ontology construction.

For a readable source example, evidence e1 directly narrates Fara Cedar's membership in Cedar Circle at story step 1. Both original C2 base prefixes nevertheless attach holder-level knowledge and intrinsic point validity. Narration does not establish a participant's knowledge, and observation time does not establish intrinsic onset or duration. These are source-only findings on complete received assertion records, not full canonical validation. Missing endpoint/type declarations remain unresolved; the CPU does not fill them from descriptions.

| Received LLM prefix | Complete entity / event / assertion records | Source-only confirmed defect categories | Construction claims |
|---|---|---|---|
| C1 / query-blind / base | 8 / 0 / 1 | unsupported_attribution | supported_description (claims 0 created node IDs) |
| C2 / 1 / base | 0 / 0 / 9 | unsupported_attribution, unsupported_intrinsic_precision | selection (claims 100 created node IDs) |
| C2 / 2 / base | 0 / 0 / 6 | unsupported_attribution, unsupported_intrinsic_precision | contextual_type (claims 50 created node IDs); schema_relation (claims 0 created node IDs) |
| C2 / 1 / repair | 0 / 0 / 12 | unsupported_intrinsic_precision | include_exclude (claims 25 created node IDs); include_exclude (claims 0 created node IDs); include_exclude (claims 0 created node IDs); include_exclude (claims 0 created node IDs); schema_relation (claims 0 created node IDs) |
| C2 / 2 / repair | 0 / 0 / 6 | unsupported_intrinsic_precision | include_exclude (claims 25 created node IDs); include_exclude (claims 0 created node IDs); include_exclude (claims 0 created node IDs); include_exclude (claims 0 created node IDs); schema_relation (claims 0 created node IDs); include_exclude (claims 0 created node IDs); include_exclude (claims 0 created node IDs); include_exclude (claims 0 created node IDs); schema_relation (claims 0 created node IDs) |
| C1 / query-blind / base | 14 / 11 / 0 |  |  |
| C2 / 1 / base | 10 / 0 / 0 |  |  |
| C2 / 2 / base | 10 / 0 / 0 |  |  |
| C2 / 2 / repair | 10 / 0 / 0 |  |  |

Counts describe complete JSON members received before truncation, not complete graphs. A zero received-node count does not mean the model authored an empty graph. C2's excessive creation claims and repeated unsupported epistemic form show that output capacity is not the only remaining issue. These historical prefixes remain failed regardless of the separately reported repairs.

## Parent repairs: completion versus scientific assessment

No checker was weakened. Unmatched paraphrases are unresolved, not false. A generated repair is a replacement model output, not a CPU-completed prefix. Server rejection before generation is an integration failure, not a model verdict. The HTML shows every assertion, cited source passage and unresolved check.

| Condition / context | HTTP / JSON complete | Object budget | Confirmed source-semantic categories | Grounding / description unresolved | Scientific accepted |
|---|---|---|---|---|---|
| C1 / query-blind | None / False | — | none established by source-only checks | — / — | False |
| C2 / 1 | True / False | — | unsupported_intrinsic_precision | — / — | False |
| C2 / 2 | True / False | — | unsupported_intrinsic_precision | — / — | False |

### Before/after received structure

Counts for truncated bases describe complete received members only. Different counts are not themselves evidence of semantic improvement.

| Condition / context | Base received nodes / assertions | Repair received nodes / assertions | Repair failure stage |
|---|---|---|---|
| C1 / query-blind | 8 / 1 | 0 / — | decoding |
| C2 / 1 | 0 / 9 | 0 / 12 | decoding |
| C2 / 2 | 0 / 6 | 0 / 6 | decoding |

### Observable semantic changes in the repairs

Offline prefix inspection only. Counts cover received records, not complete ontologies or an estimate of repair effectiveness.

| Context | Attributed records, base → repair | Claimed unique nodes / ceiling | Identical assertion groups except ID |
|---|---|---|---|
| 1 | 9 → 0 | 25 / 10 | [] |
| 2 | 6 → 0 | 25 / 10 | [] |

The observed unsupported holder-attribution form was removed from the received repair assertions. Intrinsic point validity remains unsupported for observation-only evidence. Both repairs use `include_exclude` to claim more distinct nodes than the unchanged ceiling permits. The frozen grammar bounds actual graph objects and single-kind constructive operators, but mixed-kind creation lists remain cross-field post-checks. This unencoded path allowed excessive claims and repeated administrative content; the capacity repair did not eliminate that expansion. No final graph was available for formal checking.

Readable example selection: first received assertion citing the first supplied evidence record, in each repaired context (not selected for favorable performance). The HTML contains every other received assertion.

Supplied evidence: At story step 1, Fara Cedar member of Cedar Circle.

Context 1, `nA1`: `nE2 → nR1 → nE1`. Generated explanation: Establishes the affiliation of Fara Cedar with Cedar Circle at story step 1.

Generated temporal fields: `{"discourse_position":{"passage_order":0,"sentence_order":0,"token_order":0},"revelation_position":{"label":null,"revelation_order":4},"story_time":{"kind":"point","label":"story step 1","point":1},"validity_time":{"kind":"point","label":"story step 1","point":1}}`. Epistemic scope: `null`.

Context 2, `nA1`: `nE1 → nR1 → nE1`. Generated explanation: Fara Cedar is a member of the Cedar Circle at story step 1.

Generated temporal fields: `{"discourse_position":{"passage_order":0,"sentence_order":0,"token_order":0},"revelation_position":{"label":null,"revelation_order":0},"story_time":{"kind":"point","label":"story step 1","point":1},"validity_time":{"kind":"point","label":"story step 1","point":1}}`. Epistemic scope: `null`.

No entity or predicate declarations were received in either repair. Consequently endpoint meanings, role/type compatibility, description mapping and substantive construction correctness remain unresolved. The context-2 first assertion uses the same ID at both endpoints despite a description naming a person and a collective; it cannot be repaired by assigning labels from prose. Unmatched descriptions are not automatically false, but neither are they positively grounded by absence of a known contradiction.

## Staged exploratory development

A owns schema and graph objects; B owns qualified assertions; C owns descriptions and construction reporting. Only assembled canonical outputs receive contextual draft scores. Intermediate completion is not canonical or scientific success. C1 stages finish or fail before the new C2 query-bearing transmissions. C2 contexts do not inherit C1 or each other's records. No checker changed.

New reservations: 4. Additional staged allocation: 782.824784 s. Complete canonical stage-C outputs: 0; scientific accepts: 0.

| Condition/context/stage | Nodes/assertions available | Confirmed source defects | Grounding/description unresolved | Failure stage |
|---|---|---|---|---|
| C1/None/A base | 25/0 | none established | —/— | decoding |
| C2/1/A base | 10/0 | none established | —/— | decoding |
| C2/2/A base | 10/0 | none established | —/— | decoding |
| C2/2/A repair | 10/0 | none established | —/— | decoding |

The HTML provides the actual stage inventories, full assembled graphs where available, and each assertion alongside its cited evidence. Earlier failed rows remain unchanged evidence; no authored fixture is reported as GPU output.

### Construction completion (not just successful HTTP)

| Condition/context | A / B / C valid | Canonical / scientific | Calls / repairs | Input / output tokens | Strict F1 |
|---|---|---|---|---|---|
| C1/None | False / False / False | False / False | 1 / 0 | 5996 / 5120 | — |
| C2/1 | False / False / False | False / False | 1 / 0 | 5824 / 5120 | — |
| C2/2 | False / False / False | False / False | 2 / 1 | 11658 / 11264 | — |

Unreached B/C stages are blocked by prerequisite or intact-packing failure, not model-authored empty graphs. Prefix node counts describe only complete received members. No CPU-created assertions are drawn. Stage-C scores, if present, are conditional draft scores and are separate from scientific acceptance.

### Observed Stage A expansion

| Context/attempt | Duplicate reference occurrences | Complete types / predicates | Exactly repeated predicate records except ID |
|---|---|---|---|
| None/08 | 0 | 0 / 0 | [] |
| 1/09 | 0 | 4 / 2 | [] |
| 2/10 | 141 | 2 / 0 | [] |
| 2/11 | 0 | 10 / 24 | [["nR17", "nR18", "nR19", "nR20", "nR21", "nR22", "nR23", "nR24"]] |

Repeated predicates are an observed expansion pattern, not proof of an additional semantic contradiction or a universal model limitation. No tighter schema-object scientific budget was invented to force completion. Removing duplicate references alone did not establish complete staged construction.
