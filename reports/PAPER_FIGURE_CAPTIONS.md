# Paper figure captions

All figures: exploratory extraction from retained outputs, not registered C1/C2 acceptance or evidence of novel ontology construction. 178 mm width; text at least 8.5 pt. References and published-prose semantic assessments are Codex-authored, not independent human review.

## figure_1a_fable_overview

Figure 1a. Aesop's complete The Lion and the Mouse (Townsend translation, Project Gutenberg #21), with the executed action question. The pre-request panel contains every actual query-blind record citing S2 or S3: five of twelve records, including the incomplete 'spare' claim and underdefined laughter object. Seven undisplayed records remain in the full-output supplement. The pipeline arrow denotes the actual A computation, not a derivation of B. Status colours/symbols reproduce the retained Codex assessment; this is exploratory extraction, not registered C1/C2 acceptance.

[PDF](figures/paper/figure_1a_fable_overview.pdf) · [SVG](figures/paper/figure_1a_fable_overview.svg) · [PNG](figures/paper/figure_1a_fable_overview.png)

## figure_1b_fable_comparison

Figure 1b. Complete A and B answers to the same fable action question shown in Figure 1a. A selects eight existing records; B independently generates five records from the complete text plus question. Strict qualified-fact F1 is 0.625 for A and 0.308 for B, with full output denominators. B captures rescue relationships but omits the Lion's earlier capture and release of the Mouse, and loses the face-specific running endpoint. Some inverse wording is semantically supported while failing the frozen strict matcher. Missing reference content appears only in the callout. Literal names and parallel assertions remain distinct.

[PDF](figures/paper/figure_1b_fable_comparison.pdf) · [SVG](figures/paper/figure_1b_fable_comparison.svg) · [PNG](figures/paper/figure_1b_fable_comparison.png)

## figure_2a_harbor_requests

Figure 2a. The complete synthetic Harbor story and its exact executed possession and interval-location questions. Both graphs are the complete A selections from one actual query-blind collection, not independent reconstructions. All five possession and four location predictions match their respective references (strict qualified-fact F1 = 1.000 for each). The location view preserves full generated intervals rather than clipping them to the [2,4) question window. This illustrates fixed projection and explicit qualification, not novel ontology construction. Full pre-extraction and the other approach are retained in the supplement and Figure 2b.

[PDF](figures/paper/figure_2a_harbor_requests.pdf) · [SVG](figures/paper/figure_2a_harbor_requests.svg) · [PNG](figures/paper/figure_2a_harbor_requests.png)

## figure_2b_harbor_contextual

Figure 2b. Complete independent B contextual answers to the two Harbor questions in Figure 2a. The ten-record possession answer contains five source-supported but irrelevant location facts; the six-record location answer contains two irrelevant carrying facts. Their strict qualified F1 values are 0.667 and 0.800. These facts are visibly marked I and remain in all metric denominators. Correct intervals do not establish correct selection. The two graphs are not obtained by filtering A, and their layouts or node counts are not evidence of ontology construction.

[PDF](figures/paper/figure_2b_harbor_contextual.pdf) · [SVG](figures/paper/figure_2b_harbor_contextual.svg) · [PNG](figures/paper/figure_2b_harbor_contextual.png)

## figure_2c_harbor_intervals

Figure 2c. Full model-generated bounds from A's Harbor possession and location selections. The six quoted source sentences support the six timeline bars; the model received all fourteen sentences, not this display excerpt. Filled starts and open ends denote [start,end); the shaded [2,4) region is solely the user's location window. Carrying bars are included to show the ownership/possession request, not to declare them relevant location answers. Source order and full bounds are retained; no onset or endpoint is inferred or clipped. The exact questions appear in Figure 2a and the caption companion.

[PDF](figures/paper/figure_2c_harbor_intervals.pdf) · [SVG](figures/paper/figure_2c_harbor_intervals.svg) · [PNG](figures/paper/figure_2c_harbor_intervals.png)

## figure_3a_alice_actions

Figure 3a. The complete opening Alice passage (Carroll, Project Gutenberg #11) and B's exact executed action question. All five actual assertions are drawn. Correct inspection and Rabbit movement coexist with an unsupported reader assignment: the source says the sister reads, but the graph says Alice reads. 'Sitting by bank' is less specific than the narrated on-bank location. Strict qualified F1 is 0.400. The phrase 'close by her' remains its own literal endpoint rather than being silently normalized to Alice. Source-based status is the retained Codex assessment, not an independent review or a new score.

[PDF](figures/paper/figure_3a_alice_actions.pdf) · [SVG](figures/paper/figure_3a_alice_actions.svg) · [PNG](figures/paper/figure_3a_alice_actions.png)

## figure_3b_alice_thoughts

Figure 3b. The same complete Alice evidence with the independently executed B claims question. All three actual records are shown as assertion-lane graphs; sentence-valued objects remain model-authored node labels. The combined missing-content object and rhetorical thought preserve three of four reference concepts in the retained Codex semantic assessment, although strict qualified F1 remains 0.000. The daisy-chain record supplies holder Alice with null attitude and may turn consideration into a positive judgment. No decomposed proposition or missing qualification is inserted. Semantic coverage is not strict F1 or scientific acceptance.

[PDF](figures/paper/figure_3b_alice_thoughts.pdf) · [SVG](figures/paper/figure_3b_alice_thoughts.svg) · [PNG](figures/paper/figure_3b_alice_thoughts.png)

## figure_s1_holmes

Supplementary Figure S1. The complete Holmes excerpt (Doyle, The Red-Headed League, Project Gutenberg #1661), exact action and claims questions, and retained failures. The A pre-extraction and B action answer terminate with an extra closing brace; no empty or reconstructed successful graph is drawn. The single displayed usable claim is B claims record two of three, chosen to expose the locally ambiguous vocative participant reading and absent attribution. The other two claims and complete failed strings are in the supplement. The reference reading is explicitly not model output. All original strict-parser failures and scoped scores remain unchanged.

[PDF](figures/paper/figure_s1_holmes.pdf) · [SVG](figures/paper/figure_s1_holmes.svg) · [PNG](figures/paper/figure_s1_holmes.png)

## Exact executed questions

### fable / actions

Which physical running/waking, capture, binding, rope-gnawing and release relationships involve the Lion, Mouse, hunters and ropes? Exclude dialogue, intentions, laughter and the moral.

### harbor / possession

Who owns or carries what? Return all and only directly narrated ownership and carrying relationships, retaining every stated interval.

### harbor / locations

Which directly narrated location relationships have an explicitly stated interval overlapping day 2 up to but not including day 4? Return all of them with their full evidence-stated intervals, not clipped to the question window. Exclude undated locations and beliefs.

### alice / actions

Which sitting/location, reading/inspection and running relationships involve Alice, her sister, the book, bank and Rabbit? Exclude thoughts, feelings, appearance and book-content properties.

### alice / claims

What does the passage say about the book's absent content and Alice's thoughts about the usefulness of such books and making a daisy-chain? Exclude physical actions, locations, appearance and the Rabbit; preserve questions and consideration without treating them as decisions.

### holmes / claims

What does the dialogue explicitly convey about being engaged, past partnership/help in cases and expected help in the present case? Keep each claim or concern attributed to its speaker. Exclude physical movements, appearance, offers to wait and politeness about the arrival time.

### holmes / actions

Which visit, conversation, pulling, door-closing, rising and greeting actions involve Watson, Holmes and the elderly gentleman? Exclude appearance and the content of dialogue.

## Selection and display rationale

- **fable:** User-specified accessible complete fable. The S2-S3 pre-extraction detail includes correct capture/release, an unsupported qualification and ambiguous laughter. The complete action selections retain mixed outcomes and the stronger fixed-selection score.
- **harbor:** Chosen over Orchard because all required explicit intervals are present in both location answers. The example isolates intact duration from relevance: contextual answers additionally return irrelevant facts. This is a display rationale based on retained results, not prospective statistical sampling.
- **alice:** User-specified prose with physical action and rhetorical thought. Both direct contextual outputs are shown completely, including a wrong reading participant and the undecomposed thought objects.
- **holmes:** User-specified failure example. Both strict syntax failures are retained; the selected usable claim illustrates locally ambiguous participant interpretation and missing attribution, not universal model inability.

All main panels are complete except the five-record fable pre-extraction detail and the single Holmes claim. Full outputs remain in [the supplement](figures/paper/FULL_OUTPUT_SUPPLEMENT.pdf) and [the interactive index](figures/paper/index.html). Undisplayed records still count in the original metrics. Graphs and metrics are copied from the canonical retained tables, never rescored for figure selection.

Predicate underscores become spaces and evidence whitespace is reflowed only for typography. Exact strings, malformed fields, request/response identities and source hashes are in the manifest and interactive inspector. Assertion-lane panels repeat glyphs for identical endpoint strings; they do not create or merge semantic nodes.

Regenerate: `python -m scripts.build_paper_figures`.
