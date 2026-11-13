# Simple synthetic capability ladder

Exploratory pedagogical development diagnostics—not registered C0/C1/C2 results, canonical ontologies, or held-out evidence.

Model: `Qwen/Qwen3-8B-AWQ@4da05a8edb55c6046cce958586c33b61da07bb79`. Nonthinking; seed 1988649846; temperature 0.7; top-p 0.8; top-k 20; no guided schema.

All eight requests, conservative matching rules, reference alternatives and the conditional plain-language control were frozen before inference. No guided JSON schema, expected answer, ontology contract or model repair was transmitted. Identical endpoint strings alone define graph nodes.

Direct-fact progression requires combined precision and recall ≥80%. Later levels use the same exploratory P/R criterion on complete qualified facts. Every emitted fact remains in the denominator; duplicate, malformed and unresolved facts are not silently removed. Matching permits case/whitespace normalization and the frozen relation synonyms only. An unmatched wording is not automatically a false statement.

| Case | Level | Finish | Correct / extracted / reference | Precision | Recall | F1 | Input / output tokens | Seconds |
|---|---|---|---|---|---|---|---|---|
| 1 | direct | stop | 3 / 3 / 3 | 1.000 | 1.000 | 1.000 | 108 / 94 | 2.745 |
| 2 | direct | stop | 3 / 3 / 3 | 1.000 | 1.000 | 1.000 | 109 / 92 | 1.915 |
| 3 | selection | stop | 2 / 2 / 2 | 1.000 | 1.000 | 1.000 | 121 / 62 | 1.814 |
| 4 | selection | stop | 2 / 2 / 2 | 1.000 | 1.000 | 1.000 | 122 / 63 | 1.699 |
| 5 | temporal | stop | 2 / 2 / 2 | 1.000 | 1.000 | 1.000 | 164 / 88 | 2.164 |
| 6 | epistemic | stop | 0 / 2 / 2 | 0.000 | 0.000 | 0.000 | 152 / 69 | 1.906 |
| 7 | ontology_contrast | stop | 0 / 4 / 4 | 0.000 | 0.000 | 0.000 | 283 / 204 | 4.203 |
| 8 | ontology_contrast | stop | 2 / 4 / 4 | 0.500 | 0.500 | 0.500 | 285 / 176 | 3.588 |

Highest tested level meeting the declared exploratory criterion: explicit temporal intervals. First observed scored error: case 6. This does not establish model reliability, production ontology capacity, construction efficacy or p95 runtime.

Parseable JSON fact lists: 8/8. The scores require the supplied evidence IDs; “underlying” drops temporal/epistemic qualifications but still checks citations. The manual notes below distinguish contract failures from factual errors without changing the frozen scores.

Not executed: control. Stop reason: `completed`.

New allocated GPU time: **157.347855 s**. Cumulative: **9898.847944 s**. Open allocation/service journals: 0/0. Startup, idle time, monitoring, generation and shutdown are included; request times are not total allocation.

Sampled peaks (bytes): VRAM 22569549824, process RAM 7202799616, project storage 17175356928.

## Evidence and actual answers

### Case 1: direct

S1: Mira carries a lantern.  
S2: Tomas owns the lantern.  
S3: Mira is in the courtyard.  

Extract every explicitly stated relationship and no inferred relationships.

Actual extracted relationships (verbatim JSON is in the HTML):

- Mira → **carries** → lantern; `{"evidence_id": "S1"}`
- Tomas → **owns** → lantern; `{"evidence_id": "S2"}`
- Mira → **is in** → courtyard; `{"evidence_id": "S3"}`

Complete facts: 3/3; underlying relationships: 3/3.

Unmatched predictions: 0; missing complete reference facts: 0; automatically unresolved: 0. Full fact-level lists are in the HTML/JSON.

### Case 2: direct

S1: Nadia carries a compass.  
S2: Oren owns the compass.  
S3: Oren is in the harbor.  

Extract every explicitly stated relationship and no inferred relationships.

Actual extracted relationships (verbatim JSON is in the HTML):

- Nadia → **carries** → compass; `{"evidence_id": "S1"}`
- Oren → **owns** → compass; `{"evidence_id": "S2"}`
- Oren → **is in** → harbor; `{"evidence_id": "S3"}`

Complete facts: 3/3; underlying relationships: 3/3.

Unmatched predictions: 0; missing complete reference facts: 0; automatically unresolved: 0. Full fact-level lists are in the HTML/JSON.

### Case 3: selection

S1: Ivo owns a key.  
S2: Leda owns a telescope.  
S3: Ivo is in the observatory.  
S4: Leda is in the garden.  

Who owns what? Return all and only ownership relationships.

Actual extracted relationships (verbatim JSON is in the HTML):

- Ivo → **owns** → key; `{"evidence_id": "S1"}`
- Leda → **owns** → telescope; `{"evidence_id": "S2"}`

Complete facts: 2/2; underlying relationships: 2/2.

Unmatched predictions: 0; missing complete reference facts: 0; automatically unresolved: 0. Full fact-level lists are in the HTML/JSON.

### Case 4: selection

S1: Ivo owns a key.  
S2: Leda owns a telescope.  
S3: Ivo is in the observatory.  
S4: Leda is in the garden.  

Where are the people? Return all and only location relationships.

Actual extracted relationships (verbatim JSON is in the HTML):

- Ivo → **in** → observatory; `{"evidence_id": "S3"}`
- Leda → **in** → garden; `{"evidence_id": "S4"}`

Complete facts: 2/2; underlying relationships: 2/2.

Unmatched predictions: 0; missing complete reference facts: 0; automatically unresolved: 0. Full fact-level lists are in the HTML/JSON.

### Case 5: temporal

S1: Ada is in the atrium from day 0 up to but not including day 2.  
S2: Ada is in the library from day 2 up to but not including day 5.  

State both location relationships with their explicitly given intervals.

Actual extracted relationships (verbatim JSON is in the HTML):

- Ada → **in** → atrium; `{"evidence_id": "S1", "valid_from": 0, "valid_until": 2}`
- Ada → **in** → library; `{"evidence_id": "S2", "valid_from": 2, "valid_until": 5}`

Complete facts: 2/2; underlying relationships: 2/2.

Unmatched predictions: 0; missing complete reference facts: 0; automatically unresolved: 0. Full fact-level lists are in the HTML/JSON.

### Case 6: epistemic

S1: Rina believes that the coin is in the drawer.  
S2: In reality, the coin is in the chest.  

State the believed location and the narrated real location, keeping belief separate from reality.

Actual extracted relationships (verbatim JSON is in the HTML):

- Rina → **believes** → the coin is in the drawer; `{"evidence_id": "S1"}`
- the coin → **is in** → the chest; `{"evidence_id": "S2"}`

Complete facts: 0/2; underlying relationships: 0/2.

Unmatched predictions: 2; missing complete reference facts: 2; automatically unresolved: 1. Full fact-level lists are in the HTML/JSON.

Manual diagnosis (not rescoring): The output conveys the supplied distinction between Rina's belief and narrated reality, but not in the requested qualified-fact representation. It writes Rina --believes--> 'the coin is in the drawer', omitting holder and attitude fields and leaving the believed location inside a text object. The narrated location has the right meaning, but 'the coin' and 'the chest' violate the requested article-free endpoint convention and do not match the frozen names. The zero exact-match score is not evidence that the model reversed reality and belief or invented these facts. The scorer's unresolved relation-wording category does not fully describe this observable representation failure; no score was changed. Qualification accuracy conditional on a matched relationship is unavailable here, not zero proof of factual understanding.

### Case 7: ontology_contrast

S1: Ada holds the office of Harbor Warden from day 0 up to but not including day 3.  
S2: Bram holds the office of Harbor Warden from day 3 up to but not including day 6.  
S3: The office of Harbor Warden is responsible for inspecting boats.  
S4: The office of Harbor Warden is responsible for maintaining beacons.  

Person-centered question: who held the office when, and what responsibilities did each person have through it?

Actual extracted relationships (verbatim JSON is in the HTML):

- Ada → **holds_office** → Harbor Warden; `{"evidence_id": [1], "valid_from": 0, "valid_until": 3}`
- Bram → **holds_office** → Harbor Warden; `{"evidence_id": [2], "valid_from": 3, "valid_until": 6}`
- Harbor Warden → **has_responsibility** → inspecting boats; `{"evidence_id": [3]}`
- Harbor Warden → **has_responsibility** → maintaining beacons; `{"evidence_id": [4]}`

Complete facts: 0/4; underlying relationships: 0/4.

Unmatched predictions: 4; missing complete reference facts: 4; automatically unresolved: 0. Full fact-level lists are in the HTML/JSON.

Manual diagnosis (not rescoring): The generated office-mediated factual bindings and stated holder intervals agree with the supplied passage. Every citation is a numeric singleton such as [1], whereas the evidence IDs are strings such as S1. The frozen evaluator deliberately does not invent this reference translation; all facts remain unmatched under the citation-inclusive score. This is a reference-format failure, not demonstrated falsity of the depicted office relationships.

### Case 8: ontology_contrast

S1: Ada holds the office of Harbor Warden from day 0 up to but not including day 3.  
S2: Bram holds the office of Harbor Warden from day 3 up to but not including day 6.  
S3: The office of Harbor Warden is responsible for inspecting boats.  
S4: The office of Harbor Warden is responsible for maintaining beacons.  

Office-continuity question: represent the continuing office, its responsibilities, and the successive holders with their intervals.

Actual extracted relationships (verbatim JSON is in the HTML):

- Harbor Warden → **has_responsibility** → inspecting boats; `{"evidence_id": ["S3"]}`
- Harbor Warden → **has_responsibility** → maintaining beacons; `{"evidence_id": ["S4"]}`
- Ada → **holds_office** → Harbor Warden; `{"evidence_id": ["S1"]}`
- Bram → **holds_office** → Harbor Warden; `{"evidence_id": ["S2"]}`

Complete facts: 2/4; underlying relationships: 4/4.

Unmatched predictions: 2; missing complete reference facts: 2; automatically unresolved: 0. Full fact-level lists are in the HTML/JSON.

Manual diagnosis (not rescoring): The office responsibilities, holder bindings and citations are correct under the frozen matcher. Both holder intervals are omitted despite explicit evidence and the question requesting them. Thus the underlying relationship score passes while complete qualified-fact scoring loses the holder assertions. These are omissions of supported precision, not invented bounds. The office topology is the same office-mediated organization as the person-centered answer; this pair does not demonstrate a change of ontology organization.

## Toy template baseline

This three-pattern sentence parser is not registered C0; it reads the actual sentences and does not look up stored answers.

Case 1: 3/3 precision, 3/3 recall; F1=1.000.
Case 2: 3/3 precision, 3/3 recall; F1=1.000.

Reference answers and every full scoring record are in the accompanying HTML and machine-readable JSON. The synthetic sources and expected answers are diagnostic material, separate from the original benchmark. C0 extraction competence, contextual results, historical failures, global ceilings and held-out review remain unchanged.

Graph comparison: [simple_synthetic_comparison.html](figures/simple_synthetic_comparison.html).
