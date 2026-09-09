# Compact story v2 — explicit-null exploratory extraction

A = query-blind extraction plus fixed CPU selection; B = direct contextual extraction. Not registered C1/C2, scientific acceptance, or ontology-construction evidence.

## Results and comparison

The corrected batch produced 12/12 strict-JSON responses, with 0 syntax recoveries. All graphs are displayed regardless of accuracy.

Source-interval handling improved: 79/79 emitted facts with an identifiable explicitly timed source retain its full interval. This counts unique call outputs, including irrelevant predictions, not reused selections. Missing facts still count against recall.

Contextual relevance remains weak: 9/9 contextual answers contain source-supported but irrelevant predictions. Higher new F1 is not proof of improved selection: repaired format and qualifications also change the measured outcome. There was no adaptive tuning, matched replication or causal attribution of the gain to an individual prompt change.

The fresh Museum story shows the same limitation: usable interval/possession facts, over-selection, and sentence-valued belief objects. The Orchard pre-extraction decomposes beliefs correctly, but its contextual belief answer does not. These are development observations, not general reliability or ontology-construction evidence.

| Story / version | Mean qualified F1 A | Mean qualified F1 B |
|---|---|---|
| harbor / historical_raw | 0.200 | 0.074 |
| harbor / new_model | 0.667 | 0.563 |
| orchard / historical_raw | 0.200 | 0.119 |
| orchard / new_model | 1.000 | 0.497 |
| museum / new_model | 0.667 | 0.556 |

Historical strict failures remain unchanged. Derived comma-only recovery is not a new model response and is not folded into the historical means above. All predictions remain in their applicable denominators.

Null and omitted optional values are both unspecified/not attributed for semantic comparison; missing required keys are separately reported as format failures. Neither null nor omission supplies a missing known interval or believer. References, normalization and examples were frozen before inference.

## Inspectable examples and error interpretation

The examples below are an exhaustive view of the main failure types, not extra model attempts. Manual belief-format findings are response-hash-bound and do not alter frozen scores. An unresolved sentence-valued object is not automatically false: it fails the requested decomposition even when its prose states the supplied belief correctly.

| Story | Pre-extraction complete matches / emitted | Pre-extraction qualification example |
|---|---|---|
| harbor | 12/14 | `{"attitude": "believes", "evidence_ids": ["S9"], "holder": null, "object": "Cobalt Compass is in North Hall", "relation": "believes", "subject": "Lena", "valid_from": null, "valid_until": null}` |
| orchard | 14/14 | `{"attitude": "believes", "evidence_ids": ["S9"], "holder": "Soren", "object": "South Loft", "relation": "located_in", "subject": "Copper Book", "valid_from": null, "valid_until": null}` |
| museum | 12/14 | `{"attitude": "believes", "evidence_ids": ["S1"], "holder": null, "object": "Pearl Box is in Glass Room", "relation": "believes", "subject": "Niko", "valid_from": null, "valid_until": null}` |

Fixed selection only selects the supplied records. When the model uses the believer as subject instead of the believed object, the belief filter selects that person's locations and misses the intended object realities; it does not repair the representation. All selected errors remain in precision denominators.

## Error counts (new outputs after selection)

Temporal/attribution mismatches require a recognized source triple. Zero does not certify unresolved rows. Unpaired attribution means exactly one of holder/attitude is null; it is separate from an invented holder. Bare scores ignore qualifications and citations; qualified scores require their correct values. Supplied-ID validity alone does not establish citation support.

| Story/task | Method | Key/value format | Valid citations | Missing intervals | Temporal mismatches | Attribution mismatches / unpaired | Unsupported relations | Irrelevant | Unresolved |
|---|---|---|---|---|---|---|---|---|---|
| harbor/possession | A | 5/5 | 5/5 | 0 | 0 | 0 / 0 | 0 | 0 | 0 |
| harbor/possession | B | 10/10 | 10/10 | 0 | 0 | 0 / 0 | 0 | 5 | 0 |
| harbor/locations | A | 4/4 | 4/4 | 0 | 0 | 0 / 0 | 0 | 0 | 0 |
| harbor/locations | B | 6/6 | 6/6 | 0 | 0 | 0 / 0 | 0 | 2 | 0 |
| harbor/belief | A | 3/5 | 5/5 | 0 | 0 | 0 / 2 | 0 | 3 | 2 |
| harbor/belief | B | 14/14 | 14/14 | 0 | 0 | 0 / 0 | 0 | 10 | 2 |
| orchard/possession | A | 5/5 | 5/5 | 0 | 0 | 0 / 0 | 0 | 0 | 0 |
| orchard/possession | B | 12/12 | 12/12 | 0 | 0 | 0 / 0 | 0 | 7 | 0 |
| orchard/locations | A | 4/4 | 4/4 | 0 | 0 | 0 / 0 | 0 | 0 | 0 |
| orchard/locations | B | 5/5 | 5/5 | 0 | 0 | 0 / 0 | 0 | 2 | 0 |
| orchard/belief | A | 4/4 | 4/4 | 0 | 0 | 0 / 0 | 0 | 0 | 0 |
| orchard/belief | B | 13/13 | 13/13 | 0 | 0 | 0 / 0 | 0 | 9 | 2 |
| museum/possession | A | 5/5 | 5/5 | 0 | 0 | 0 / 0 | 0 | 0 | 0 |
| museum/possession | B | 11/13 | 13/13 | 0 | 0 | 0 / 2 | 0 | 6 | 2 |
| museum/locations | A | 4/4 | 4/4 | 0 | 0 | 0 / 0 | 0 | 0 | 0 |
| museum/locations | B | 5/5 | 5/5 | 0 | 0 | 0 / 0 | 0 | 1 | 0 |
| museum/belief | A | 2/4 | 4/4 | 0 | 0 | 0 / 2 | 0 | 2 | 2 |
| museum/belief | B | 12/14 | 14/14 | 0 | 0 | 0 / 2 | 0 | 10 | 2 |

## Every question and approach

| Category | Story / task | Approach | Strict / recovered parse | Bare P/R/F1 | Qualified P/R/F1 | Citations |
|---|---|---|---|---|---|---|
| historical_raw | harbor / possession | A pre-extract/select | True / True | 1.000 / 1.000 / 1.000 | 0.600 / 0.600 / 0.600 | 5/5 |
| historical_raw | harbor / possession | B contextual | False / False | 0.000 / 0.000 / 0.000 | 0.000 / 0.000 / 0.000 | 0/0 |
| historical_raw | harbor / locations | A pre-extract/select | True / True | 0.000 / 0.000 / 0.000 | 0.000 / 0.000 / 0.000 | 0/0 |
| historical_raw | harbor / locations | B contextual | True / True | 0.200 / 0.250 / 0.222 | 0.000 / 0.000 / 0.000 | 5/5 |
| historical_raw | harbor / belief | A pre-extract/select | True / True | 0.000 / 0.000 / 0.000 | 0.000 / 0.000 / 0.000 | 5/5 |
| historical_raw | harbor / belief | B contextual | True / True | 0.143 / 0.500 / 0.222 | 0.143 / 0.500 / 0.222 | 14/14 |
| historical_raw | orchard / possession | A pre-extract/select | True / True | 1.000 / 1.000 / 1.000 | 0.600 / 0.600 / 0.600 | 5/5 |
| historical_raw | orchard / possession | B contextual | False / False | 0.000 / 0.000 / 0.000 | 0.000 / 0.000 / 0.000 | 0/0 |
| historical_raw | orchard / locations | A pre-extract/select | True / True | 0.000 / 0.000 / 0.000 | 0.000 / 0.000 / 0.000 | 0/0 |
| historical_raw | orchard / locations | B contextual | True / True | 0.364 / 1.000 / 0.533 | 0.091 / 0.250 / 0.133 | 11/11 |
| historical_raw | orchard / belief | A pre-extract/select | True / True | 0.000 / 0.000 / 0.000 | 0.000 / 0.000 / 0.000 | 0/0 |
| historical_raw | orchard / belief | B contextual | True / True | 0.143 / 0.500 / 0.222 | 0.143 / 0.500 / 0.222 | 14/14 |
| historical_syntax_recovered | harbor / possession | B contextual | False / True | 0.500 / 1.000 / 0.667 | 0.300 / 0.600 / 0.400 | 10/10 |
| historical_syntax_recovered | orchard / possession | B contextual | False / True | 0.500 / 1.000 / 0.667 | 0.300 / 0.600 / 0.400 | 10/10 |
| new_model | harbor / possession | A pre-extract/select | True / True | 1.000 / 1.000 / 1.000 | 1.000 / 1.000 / 1.000 | 5/5 |
| new_model | harbor / possession | B contextual | True / True | 0.500 / 1.000 / 0.667 | 0.500 / 1.000 / 0.667 | 10/10 |
| new_model | harbor / locations | A pre-extract/select | True / True | 1.000 / 1.000 / 1.000 | 1.000 / 1.000 / 1.000 | 4/4 |
| new_model | harbor / locations | B contextual | True / True | 0.667 / 1.000 / 0.800 | 0.667 / 1.000 / 0.800 | 6/6 |
| new_model | harbor / belief | A pre-extract/select | True / True | 0.000 / 0.000 / 0.000 | 0.000 / 0.000 / 0.000 | 5/5 |
| new_model | harbor / belief | B contextual | True / True | 0.143 / 0.500 / 0.222 | 0.143 / 0.500 / 0.222 | 14/14 |
| new_model | orchard / possession | A pre-extract/select | True / True | 1.000 / 1.000 / 1.000 | 1.000 / 1.000 / 1.000 | 5/5 |
| new_model | orchard / possession | B contextual | True / True | 0.417 / 1.000 / 0.588 | 0.417 / 1.000 / 0.588 | 12/12 |
| new_model | orchard / locations | A pre-extract/select | True / True | 1.000 / 1.000 / 1.000 | 1.000 / 1.000 / 1.000 | 4/4 |
| new_model | orchard / locations | B contextual | True / True | 0.600 / 0.750 / 0.667 | 0.600 / 0.750 / 0.667 | 5/5 |
| new_model | orchard / belief | A pre-extract/select | True / True | 1.000 / 1.000 / 1.000 | 1.000 / 1.000 / 1.000 | 4/4 |
| new_model | orchard / belief | B contextual | True / True | 0.154 / 0.500 / 0.235 | 0.154 / 0.500 / 0.235 | 13/13 |
| new_model | museum / possession | A pre-extract/select | True / True | 1.000 / 1.000 / 1.000 | 1.000 / 1.000 / 1.000 | 5/5 |
| new_model | museum / possession | B contextual | True / True | 0.385 / 1.000 / 0.556 | 0.385 / 1.000 / 0.556 | 13/13 |
| new_model | museum / locations | A pre-extract/select | True / True | 1.000 / 1.000 / 1.000 | 1.000 / 1.000 / 1.000 | 4/4 |
| new_model | museum / locations | B contextual | True / True | 0.800 / 1.000 / 0.889 | 0.800 / 1.000 / 0.889 | 5/5 |
| new_model | museum / belief | A pre-extract/select | True / True | 0.000 / 0.000 / 0.000 | 0.000 / 0.000 / 0.000 | 4/4 |
| new_model | museum / belief | B contextual | True / True | 0.143 / 0.500 / 0.222 | 0.143 / 0.500 / 0.222 | 14/14 |

## The harbor supplies

- S1: Lena owns Cobalt Compass.
- S2: Omar owns Amber Lantern.
- S3: Priya carries Cobalt Compass from day 0 up to but not including day 6.
- S4: Lena carries Amber Lantern from day 1 up to but not including day 5.
- S5: Lena is in North Hall from day 1 up to but not including day 5.
- S6: Omar is in East Garden from day 0 up to but not including day 3.
- S7: Priya is in Stone Shed from day 3 up to but not including day 6.
- S8: Silver Flute is in West Pier from day 2 up to but not including day 4.
- S9: Lena believes that Cobalt Compass is in North Hall.
- S10: In reality, Cobalt Compass is in East Garden.
- S11: Omar believes that Amber Lantern is in Stone Shed.
- S12: In reality, Amber Lantern is in West Pier.
- S13: Priya owns Silver Flute.
- S14: Lena is in Stone Shed from day 7 up to but not including day 9.

### Who owns or carries what? Return all and only directly narrated ownership and carrying relationships, retaining every stated interval.

Reference answers:
```json
[
  {
    "valid_from": null,
    "valid_until": null,
    "holder": null,
    "attitude": null,
    "subject": "Lena",
    "relation": "owns",
    "object": "Cobalt Compass",
    "evidence_ids": [
      "S1"
    ]
  },
  {
    "valid_from": null,
    "valid_until": null,
    "holder": null,
    "attitude": null,
    "subject": "Omar",
    "relation": "owns",
    "object": "Amber Lantern",
    "evidence_ids": [
      "S2"
    ]
  },
  {
    "valid_from": 0,
    "valid_until": 6,
    "holder": null,
    "attitude": null,
    "subject": "Priya",
    "relation": "carries",
    "object": "Cobalt Compass",
    "evidence_ids": [
      "S3"
    ]
  },
  {
    "valid_from": 1,
    "valid_until": 5,
    "holder": null,
    "attitude": null,
    "subject": "Lena",
    "relation": "carries",
    "object": "Amber Lantern",
    "evidence_ids": [
      "S4"
    ]
  },
  {
    "valid_from": null,
    "valid_until": null,
    "holder": null,
    "attitude": null,
    "subject": "Priya",
    "relation": "owns",
    "object": "Silver Flute",
    "evidence_ids": [
      "S13"
    ]
  }
]
```

A pre-extract/select:

Strict parse: True; after permitted recovery: True.

- `{"attitude": null, "evidence_ids": ["S1"], "holder": null, "object": "Cobalt Compass", "relation": "owns", "subject": "Lena", "valid_from": null, "valid_until": null}` — correct_complete_fact; 
- `{"attitude": null, "evidence_ids": ["S2"], "holder": null, "object": "Amber Lantern", "relation": "owns", "subject": "Omar", "valid_from": null, "valid_until": null}` — correct_complete_fact; 
- `{"attitude": null, "evidence_ids": ["S3"], "holder": null, "object": "Cobalt Compass", "relation": "carries", "subject": "Priya", "valid_from": 0, "valid_until": 6}` — correct_complete_fact; 
- `{"attitude": null, "evidence_ids": ["S4"], "holder": null, "object": "Amber Lantern", "relation": "carries", "subject": "Lena", "valid_from": 1, "valid_until": 5}` — correct_complete_fact; 
- `{"attitude": null, "evidence_ids": ["S13"], "holder": null, "object": "Silver Flute", "relation": "owns", "subject": "Priya", "valid_from": null, "valid_until": null}` — correct_complete_fact; 

Missing: `[]`.

Manual diagnosis of source output (not rescoring): S9 and S11 are represented as person --believes--> proposition sentence, with null holder and attitude believes. This violates the explicit decomposition and holder/attitude pairing instructions. The sentences do express the supplied beliefs in prose; this is not evidence of invented beliefs. The record-only selector consequently uses the people as belief subjects, selects their locations and misses the narrated object locations. No CPU decomposition or selection repair was applied.

B contextual:

Strict parse: True; after permitted recovery: True.

- `{"attitude": null, "evidence_ids": ["S1"], "holder": null, "object": "Cobalt Compass", "relation": "owns", "subject": "Lena", "valid_from": null, "valid_until": null}` — correct_complete_fact; 
- `{"attitude": null, "evidence_ids": ["S2"], "holder": null, "object": "Amber Lantern", "relation": "owns", "subject": "Omar", "valid_from": null, "valid_until": null}` — correct_complete_fact; 
- `{"attitude": null, "evidence_ids": ["S3"], "holder": null, "object": "Cobalt Compass", "relation": "carries", "subject": "Priya", "valid_from": 0, "valid_until": 6}` — correct_complete_fact; 
- `{"attitude": null, "evidence_ids": ["S4"], "holder": null, "object": "Amber Lantern", "relation": "carries", "subject": "Lena", "valid_from": 1, "valid_until": 5}` — correct_complete_fact; 
- `{"attitude": null, "evidence_ids": ["S8"], "holder": null, "object": "West Pier", "relation": "located_in", "subject": "Silver Flute", "valid_from": 2, "valid_until": 4}` — supported_but_irrelevant; 
- `{"attitude": null, "evidence_ids": ["S13"], "holder": null, "object": "Silver Flute", "relation": "owns", "subject": "Priya", "valid_from": null, "valid_until": null}` — correct_complete_fact; 
- `{"attitude": null, "evidence_ids": ["S5"], "holder": null, "object": "North Hall", "relation": "is_in", "subject": "Lena", "valid_from": 1, "valid_until": 5}` — supported_but_irrelevant; 
- `{"attitude": null, "evidence_ids": ["S6"], "holder": null, "object": "East Garden", "relation": "is_in", "subject": "Omar", "valid_from": 0, "valid_until": 3}` — supported_but_irrelevant; 
- `{"attitude": null, "evidence_ids": ["S7"], "holder": null, "object": "Stone Shed", "relation": "is_in", "subject": "Priya", "valid_from": 3, "valid_until": 6}` — supported_but_irrelevant; 
- `{"attitude": null, "evidence_ids": ["S14"], "holder": null, "object": "Stone Shed", "relation": "is_in", "subject": "Lena", "valid_from": 7, "valid_until": 9}` — supported_but_irrelevant; 

Missing: `[]`.

### Which directly narrated location relationships have an explicitly stated interval overlapping day 2 up to but not including day 4? Return all of them with their full evidence-stated intervals, not clipped to the question window. Exclude undated locations and beliefs.

Reference answers:
```json
[
  {
    "valid_from": 1,
    "valid_until": 5,
    "holder": null,
    "attitude": null,
    "subject": "Lena",
    "relation": "located_in",
    "object": "North Hall",
    "evidence_ids": [
      "S5"
    ]
  },
  {
    "valid_from": 0,
    "valid_until": 3,
    "holder": null,
    "attitude": null,
    "subject": "Omar",
    "relation": "located_in",
    "object": "East Garden",
    "evidence_ids": [
      "S6"
    ]
  },
  {
    "valid_from": 3,
    "valid_until": 6,
    "holder": null,
    "attitude": null,
    "subject": "Priya",
    "relation": "located_in",
    "object": "Stone Shed",
    "evidence_ids": [
      "S7"
    ]
  },
  {
    "valid_from": 2,
    "valid_until": 4,
    "holder": null,
    "attitude": null,
    "subject": "Silver Flute",
    "relation": "located_in",
    "object": "West Pier",
    "evidence_ids": [
      "S8"
    ]
  }
]
```

A pre-extract/select:

Strict parse: True; after permitted recovery: True.

- `{"attitude": null, "evidence_ids": ["S5"], "holder": null, "object": "North Hall", "relation": "located_in", "subject": "Lena", "valid_from": 1, "valid_until": 5}` — correct_complete_fact; 
- `{"attitude": null, "evidence_ids": ["S6"], "holder": null, "object": "East Garden", "relation": "located_in", "subject": "Omar", "valid_from": 0, "valid_until": 3}` — correct_complete_fact; 
- `{"attitude": null, "evidence_ids": ["S7"], "holder": null, "object": "Stone Shed", "relation": "located_in", "subject": "Priya", "valid_from": 3, "valid_until": 6}` — correct_complete_fact; 
- `{"attitude": null, "evidence_ids": ["S8"], "holder": null, "object": "West Pier", "relation": "located_in", "subject": "Silver Flute", "valid_from": 2, "valid_until": 4}` — correct_complete_fact; 

Missing: `[]`.

Manual diagnosis of source output (not rescoring): S9 and S11 are represented as person --believes--> proposition sentence, with null holder and attitude believes. This violates the explicit decomposition and holder/attitude pairing instructions. The sentences do express the supplied beliefs in prose; this is not evidence of invented beliefs. The record-only selector consequently uses the people as belief subjects, selects their locations and misses the narrated object locations. No CPU decomposition or selection repair was applied.

B contextual:

Strict parse: True; after permitted recovery: True.

- `{"attitude": null, "evidence_ids": ["S3"], "holder": null, "object": "Cobalt Compass", "relation": "carries", "subject": "Priya", "valid_from": 0, "valid_until": 6}` — supported_but_irrelevant; 
- `{"attitude": null, "evidence_ids": ["S4"], "holder": null, "object": "Amber Lantern", "relation": "carries", "subject": "Lena", "valid_from": 1, "valid_until": 5}` — supported_but_irrelevant; 
- `{"attitude": null, "evidence_ids": ["S5"], "holder": null, "object": "North Hall", "relation": "located_in", "subject": "Lena", "valid_from": 1, "valid_until": 5}` — correct_complete_fact; 
- `{"attitude": null, "evidence_ids": ["S6"], "holder": null, "object": "East Garden", "relation": "located_in", "subject": "Omar", "valid_from": 0, "valid_until": 3}` — correct_complete_fact; 
- `{"attitude": null, "evidence_ids": ["S7"], "holder": null, "object": "Stone Shed", "relation": "located_in", "subject": "Priya", "valid_from": 3, "valid_until": 6}` — correct_complete_fact; 
- `{"attitude": null, "evidence_ids": ["S8"], "holder": null, "object": "West Pier", "relation": "located_in", "subject": "Silver Flute", "valid_from": 2, "valid_until": 4}` — correct_complete_fact; 

Missing: `[]`.

### Who explicitly believes what, and what reality is separately narrated about those same objects? Return each underlying believed relationship with its attribution and each matching narrated reality without attribution. Omit unrelated facts.

Reference answers:
```json
[
  {
    "valid_from": null,
    "valid_until": null,
    "holder": "Lena",
    "attitude": "believes",
    "subject": "Cobalt Compass",
    "relation": "located_in",
    "object": "North Hall",
    "evidence_ids": [
      "S9"
    ]
  },
  {
    "valid_from": null,
    "valid_until": null,
    "holder": null,
    "attitude": null,
    "subject": "Cobalt Compass",
    "relation": "located_in",
    "object": "East Garden",
    "evidence_ids": [
      "S10"
    ]
  },
  {
    "valid_from": null,
    "valid_until": null,
    "holder": "Omar",
    "attitude": "believes",
    "subject": "Amber Lantern",
    "relation": "located_in",
    "object": "Stone Shed",
    "evidence_ids": [
      "S11"
    ]
  },
  {
    "valid_from": null,
    "valid_until": null,
    "holder": null,
    "attitude": null,
    "subject": "Amber Lantern",
    "relation": "located_in",
    "object": "West Pier",
    "evidence_ids": [
      "S12"
    ]
  }
]
```

A pre-extract/select:

Strict parse: True; after permitted recovery: True.

- `{"attitude": null, "evidence_ids": ["S5"], "holder": null, "object": "North Hall", "relation": "located_in", "subject": "Lena", "valid_from": 1, "valid_until": 5}` — supported_but_irrelevant; 
- `{"attitude": null, "evidence_ids": ["S6"], "holder": null, "object": "East Garden", "relation": "located_in", "subject": "Omar", "valid_from": 0, "valid_until": 3}` — supported_but_irrelevant; 
- `{"attitude": "believes", "evidence_ids": ["S9"], "holder": null, "object": "Cobalt Compass is in North Hall", "relation": "believes", "subject": "Lena", "valid_from": null, "valid_until": null}` — unresolved_wording; holder and attitude must appear together
- `{"attitude": "believes", "evidence_ids": ["S11"], "holder": null, "object": "Amber Lantern is in Stone Shed", "relation": "believes", "subject": "Omar", "valid_from": null, "valid_until": null}` — unresolved_wording; holder and attitude must appear together
- `{"attitude": null, "evidence_ids": ["S14"], "holder": null, "object": "Stone Shed", "relation": "located_in", "subject": "Lena", "valid_from": 7, "valid_until": 9}` — supported_but_irrelevant; 

Missing: `[{"valid_from": null, "valid_until": null, "holder": "Lena", "attitude": "believes", "subject": "Cobalt Compass", "relation": "located_in", "object": "North Hall", "evidence_ids": ["S9"]}, {"valid_from": null, "valid_until": null, "holder": null, "attitude": null, "subject": "Cobalt Compass", "relation": "located_in", "object": "East Garden", "evidence_ids": ["S10"]}, {"valid_from": null, "valid_until": null, "holder": "Omar", "attitude": "believes", "subject": "Amber Lantern", "relation": "located_in", "object": "Stone Shed", "evidence_ids": ["S11"]}, {"valid_from": null, "valid_until": null, "holder": null, "attitude": null, "subject": "Amber Lantern", "relation": "located_in", "object": "West Pier", "evidence_ids": ["S12"]}]`.

Manual diagnosis of source output (not rescoring): S9 and S11 are represented as person --believes--> proposition sentence, with null holder and attitude believes. This violates the explicit decomposition and holder/attitude pairing instructions. The sentences do express the supplied beliefs in prose; this is not evidence of invented beliefs. The record-only selector consequently uses the people as belief subjects, selects their locations and misses the narrated object locations. No CPU decomposition or selection repair was applied.

B contextual:

Strict parse: True; after permitted recovery: True.

- `{"attitude": null, "evidence_ids": ["S1"], "holder": null, "object": "Cobalt Compass", "relation": "owns", "subject": "Lena", "valid_from": null, "valid_until": null}` — supported_but_irrelevant; 
- `{"attitude": null, "evidence_ids": ["S2"], "holder": null, "object": "Amber Lantern", "relation": "owns", "subject": "Omar", "valid_from": null, "valid_until": null}` — supported_but_irrelevant; 
- `{"attitude": null, "evidence_ids": ["S3"], "holder": null, "object": "Cobalt Compass", "relation": "carries", "subject": "Priya", "valid_from": 0, "valid_until": 6}` — supported_but_irrelevant; 
- `{"attitude": null, "evidence_ids": ["S4"], "holder": null, "object": "Amber Lantern", "relation": "carries", "subject": "Lena", "valid_from": 1, "valid_until": 5}` — supported_but_irrelevant; 
- `{"attitude": null, "evidence_ids": ["S5"], "holder": null, "object": "North Hall", "relation": "located_in", "subject": "Lena", "valid_from": 1, "valid_until": 5}` — supported_but_irrelevant; 
- `{"attitude": null, "evidence_ids": ["S6"], "holder": null, "object": "East Garden", "relation": "located_in", "subject": "Omar", "valid_from": 0, "valid_until": 3}` — supported_but_irrelevant; 
- `{"attitude": null, "evidence_ids": ["S7"], "holder": null, "object": "Stone Shed", "relation": "located_in", "subject": "Priya", "valid_from": 3, "valid_until": 6}` — supported_but_irrelevant; 
- `{"attitude": null, "evidence_ids": ["S8"], "holder": null, "object": "West Pier", "relation": "located_in", "subject": "Silver Flute", "valid_from": 2, "valid_until": 4}` — supported_but_irrelevant; 
- `{"attitude": null, "evidence_ids": ["S13"], "holder": null, "object": "Silver Flute", "relation": "owns", "subject": "Priya", "valid_from": null, "valid_until": null}` — supported_but_irrelevant; 
- `{"attitude": "believes", "evidence_ids": ["S9"], "holder": "Lena", "object": "Cobalt Compass is in North Hall", "relation": "believes", "subject": "Lena", "valid_from": null, "valid_until": null}` — unresolved_wording; 
- `{"attitude": null, "evidence_ids": ["S10"], "holder": null, "object": "East Garden", "relation": "located_in", "subject": "Cobalt Compass", "valid_from": null, "valid_until": null}` — correct_complete_fact; 
- `{"attitude": "believes", "evidence_ids": ["S11"], "holder": "Omar", "object": "Amber Lantern is in Stone Shed", "relation": "believes", "subject": "Omar", "valid_from": null, "valid_until": null}` — unresolved_wording; 
- `{"attitude": null, "evidence_ids": ["S12"], "holder": null, "object": "West Pier", "relation": "located_in", "subject": "Amber Lantern", "valid_from": null, "valid_until": null}` — correct_complete_fact; 
- `{"attitude": null, "evidence_ids": ["S14"], "holder": null, "object": "Stone Shed", "relation": "located_in", "subject": "Lena", "valid_from": 7, "valid_until": 9}` — supported_but_irrelevant; 

Missing: `[{"valid_from": null, "valid_until": null, "holder": "Lena", "attitude": "believes", "subject": "Cobalt Compass", "relation": "located_in", "object": "North Hall", "evidence_ids": ["S9"]}, {"valid_from": null, "valid_until": null, "holder": "Omar", "attitude": "believes", "subject": "Amber Lantern", "relation": "located_in", "object": "Stone Shed", "evidence_ids": ["S11"]}]`.

Manual diagnosis of source output (not rescoring): The holders Lena and Omar and attitude believes are explicit and evidence-supported. However, the subjects are the people and the objects are whole proposition sentences, not the requested underlying relationships. The frozen matcher leaves these rows unresolved; manual inspection identifies a representation failure, not false attribution.


## The orchard delivery

- S1: Vera owns Copper Book.
- S2: Soren owns Ivory Vase.
- S3: Jun carries Copper Book from day 9 up to but not including day 17.
- S4: Vera carries Ivory Vase from day 10 up to but not including day 16.
- S5: Jun is in River Room from day 10 up to but not including day 14.
- S6: Soren is in Hill Court from day 14 up to but not including day 18.
- S7: Vera is in South Loft from day 11 up to but not including day 16.
- S8: Jade Drum is in Elm Dock from day 12 up to but not including day 15.
- S9: Soren believes that Copper Book is in South Loft.
- S10: In reality, Copper Book is in River Room.
- S11: Jun believes that Ivory Vase is in Hill Court.
- S12: In reality, Ivory Vase is in Elm Dock.
- S13: Jun owns Jade Drum.
- S14: Jun is in South Loft from day 18 up to but not including day 20.

### Who owns or carries what? Return all and only directly narrated ownership and carrying relationships, retaining every stated interval.

Reference answers:
```json
[
  {
    "valid_from": null,
    "valid_until": null,
    "holder": null,
    "attitude": null,
    "subject": "Vera",
    "relation": "owns",
    "object": "Copper Book",
    "evidence_ids": [
      "S1"
    ]
  },
  {
    "valid_from": null,
    "valid_until": null,
    "holder": null,
    "attitude": null,
    "subject": "Soren",
    "relation": "owns",
    "object": "Ivory Vase",
    "evidence_ids": [
      "S2"
    ]
  },
  {
    "valid_from": 9,
    "valid_until": 17,
    "holder": null,
    "attitude": null,
    "subject": "Jun",
    "relation": "carries",
    "object": "Copper Book",
    "evidence_ids": [
      "S3"
    ]
  },
  {
    "valid_from": 10,
    "valid_until": 16,
    "holder": null,
    "attitude": null,
    "subject": "Vera",
    "relation": "carries",
    "object": "Ivory Vase",
    "evidence_ids": [
      "S4"
    ]
  },
  {
    "valid_from": null,
    "valid_until": null,
    "holder": null,
    "attitude": null,
    "subject": "Jun",
    "relation": "owns",
    "object": "Jade Drum",
    "evidence_ids": [
      "S13"
    ]
  }
]
```

A pre-extract/select:

Strict parse: True; after permitted recovery: True.

- `{"attitude": null, "evidence_ids": ["S1"], "holder": null, "object": "Copper Book", "relation": "owns", "subject": "Vera", "valid_from": null, "valid_until": null}` — correct_complete_fact; 
- `{"attitude": null, "evidence_ids": ["S2"], "holder": null, "object": "Ivory Vase", "relation": "owns", "subject": "Soren", "valid_from": null, "valid_until": null}` — correct_complete_fact; 
- `{"attitude": null, "evidence_ids": ["S3"], "holder": null, "object": "Copper Book", "relation": "carries", "subject": "Jun", "valid_from": 9, "valid_until": 17}` — correct_complete_fact; 
- `{"attitude": null, "evidence_ids": ["S4"], "holder": null, "object": "Ivory Vase", "relation": "carries", "subject": "Vera", "valid_from": 10, "valid_until": 16}` — correct_complete_fact; 
- `{"attitude": null, "evidence_ids": ["S13"], "holder": null, "object": "Jade Drum", "relation": "owns", "subject": "Jun", "valid_from": null, "valid_until": null}` — correct_complete_fact; 

Missing: `[]`.

Manual diagnosis of source output (not rescoring): The Orchard pre-extraction supplies object --located_in--> place with the named believer in holder for both beliefs. Narrated realities remain separate with null holders. This satisfies the requested representation on this output, without establishing reliability.

B contextual:

Strict parse: True; after permitted recovery: True.

- `{"attitude": null, "evidence_ids": ["S1"], "holder": null, "object": "Copper Book", "relation": "owns", "subject": "Vera", "valid_from": null, "valid_until": null}` — correct_complete_fact; 
- `{"attitude": null, "evidence_ids": ["S2"], "holder": null, "object": "Ivory Vase", "relation": "owns", "subject": "Soren", "valid_from": null, "valid_until": null}` — correct_complete_fact; 
- `{"attitude": null, "evidence_ids": ["S3"], "holder": null, "object": "Copper Book", "relation": "carries", "subject": "Jun", "valid_from": 9, "valid_until": 17}` — correct_complete_fact; 
- `{"attitude": null, "evidence_ids": ["S4"], "holder": null, "object": "Ivory Vase", "relation": "carries", "subject": "Vera", "valid_from": 10, "valid_until": 16}` — correct_complete_fact; 
- `{"attitude": null, "evidence_ids": ["S5"], "holder": null, "object": "River Room", "relation": "located_in", "subject": "Jun", "valid_from": 10, "valid_until": 14}` — supported_but_irrelevant; 
- `{"attitude": null, "evidence_ids": ["S6"], "holder": null, "object": "Hill Court", "relation": "located_in", "subject": "Soren", "valid_from": 14, "valid_until": 18}` — supported_but_irrelevant; 
- `{"attitude": null, "evidence_ids": ["S7"], "holder": null, "object": "South Loft", "relation": "located_in", "subject": "Vera", "valid_from": 11, "valid_until": 16}` — supported_but_irrelevant; 
- `{"attitude": null, "evidence_ids": ["S8"], "holder": null, "object": "Elm Dock", "relation": "located_in", "subject": "Jade Drum", "valid_from": 12, "valid_until": 15}` — supported_but_irrelevant; 
- `{"attitude": null, "evidence_ids": ["S10"], "holder": null, "object": "River Room", "relation": "located_in", "subject": "Copper Book", "valid_from": null, "valid_until": null}` — supported_but_irrelevant; 
- `{"attitude": null, "evidence_ids": ["S12"], "holder": null, "object": "Elm Dock", "relation": "located_in", "subject": "Ivory Vase", "valid_from": null, "valid_until": null}` — supported_but_irrelevant; 
- `{"attitude": null, "evidence_ids": ["S13"], "holder": null, "object": "Jade Drum", "relation": "owns", "subject": "Jun", "valid_from": null, "valid_until": null}` — correct_complete_fact; 
- `{"attitude": null, "evidence_ids": ["S14"], "holder": null, "object": "South Loft", "relation": "located_in", "subject": "Jun", "valid_from": 18, "valid_until": 20}` — supported_but_irrelevant; 

Missing: `[]`.

### Which directly narrated location relationships have an explicitly stated interval overlapping day 12 up to but not including day 15? Return all of them with their full evidence-stated intervals, not clipped to the question window. Exclude undated locations and beliefs.

Reference answers:
```json
[
  {
    "valid_from": 10,
    "valid_until": 14,
    "holder": null,
    "attitude": null,
    "subject": "Jun",
    "relation": "located_in",
    "object": "River Room",
    "evidence_ids": [
      "S5"
    ]
  },
  {
    "valid_from": 14,
    "valid_until": 18,
    "holder": null,
    "attitude": null,
    "subject": "Soren",
    "relation": "located_in",
    "object": "Hill Court",
    "evidence_ids": [
      "S6"
    ]
  },
  {
    "valid_from": 11,
    "valid_until": 16,
    "holder": null,
    "attitude": null,
    "subject": "Vera",
    "relation": "located_in",
    "object": "South Loft",
    "evidence_ids": [
      "S7"
    ]
  },
  {
    "valid_from": 12,
    "valid_until": 15,
    "holder": null,
    "attitude": null,
    "subject": "Jade Drum",
    "relation": "located_in",
    "object": "Elm Dock",
    "evidence_ids": [
      "S8"
    ]
  }
]
```

A pre-extract/select:

Strict parse: True; after permitted recovery: True.

- `{"attitude": null, "evidence_ids": ["S5"], "holder": null, "object": "River Room", "relation": "located_in", "subject": "Jun", "valid_from": 10, "valid_until": 14}` — correct_complete_fact; 
- `{"attitude": null, "evidence_ids": ["S6"], "holder": null, "object": "Hill Court", "relation": "located_in", "subject": "Soren", "valid_from": 14, "valid_until": 18}` — correct_complete_fact; 
- `{"attitude": null, "evidence_ids": ["S7"], "holder": null, "object": "South Loft", "relation": "located_in", "subject": "Vera", "valid_from": 11, "valid_until": 16}` — correct_complete_fact; 
- `{"attitude": null, "evidence_ids": ["S8"], "holder": null, "object": "Elm Dock", "relation": "located_in", "subject": "Jade Drum", "valid_from": 12, "valid_until": 15}` — correct_complete_fact; 

Missing: `[]`.

Manual diagnosis of source output (not rescoring): The Orchard pre-extraction supplies object --located_in--> place with the named believer in holder for both beliefs. Narrated realities remain separate with null holders. This satisfies the requested representation on this output, without establishing reliability.

B contextual:

Strict parse: True; after permitted recovery: True.

- `{"attitude": null, "evidence_ids": ["S3"], "holder": null, "object": "Copper Book", "relation": "carries", "subject": "Jun", "valid_from": 9, "valid_until": 17}` — supported_but_irrelevant; 
- `{"attitude": null, "evidence_ids": ["S4"], "holder": null, "object": "Ivory Vase", "relation": "carries", "subject": "Vera", "valid_from": 10, "valid_until": 16}` — supported_but_irrelevant; 
- `{"attitude": null, "evidence_ids": ["S5"], "holder": null, "object": "River Room", "relation": "located_in", "subject": "Jun", "valid_from": 10, "valid_until": 14}` — correct_complete_fact; 
- `{"attitude": null, "evidence_ids": ["S6"], "holder": null, "object": "Hill Court", "relation": "located_in", "subject": "Soren", "valid_from": 14, "valid_until": 18}` — correct_complete_fact; 
- `{"attitude": null, "evidence_ids": ["S8"], "holder": null, "object": "Elm Dock", "relation": "located_in", "subject": "Jade Drum", "valid_from": 12, "valid_until": 15}` — correct_complete_fact; 

Missing: `[{"valid_from": 11, "valid_until": 16, "holder": null, "attitude": null, "subject": "Vera", "relation": "located_in", "object": "South Loft", "evidence_ids": ["S7"]}]`.

### Who explicitly believes what, and what reality is separately narrated about those same objects? Return each underlying believed relationship with its attribution and each matching narrated reality without attribution. Omit unrelated facts.

Reference answers:
```json
[
  {
    "valid_from": null,
    "valid_until": null,
    "holder": "Soren",
    "attitude": "believes",
    "subject": "Copper Book",
    "relation": "located_in",
    "object": "South Loft",
    "evidence_ids": [
      "S9"
    ]
  },
  {
    "valid_from": null,
    "valid_until": null,
    "holder": null,
    "attitude": null,
    "subject": "Copper Book",
    "relation": "located_in",
    "object": "River Room",
    "evidence_ids": [
      "S10"
    ]
  },
  {
    "valid_from": null,
    "valid_until": null,
    "holder": "Jun",
    "attitude": "believes",
    "subject": "Ivory Vase",
    "relation": "located_in",
    "object": "Hill Court",
    "evidence_ids": [
      "S11"
    ]
  },
  {
    "valid_from": null,
    "valid_until": null,
    "holder": null,
    "attitude": null,
    "subject": "Ivory Vase",
    "relation": "located_in",
    "object": "Elm Dock",
    "evidence_ids": [
      "S12"
    ]
  }
]
```

A pre-extract/select:

Strict parse: True; after permitted recovery: True.

- `{"attitude": "believes", "evidence_ids": ["S9"], "holder": "Soren", "object": "South Loft", "relation": "located_in", "subject": "Copper Book", "valid_from": null, "valid_until": null}` — correct_complete_fact; 
- `{"attitude": null, "evidence_ids": ["S10"], "holder": null, "object": "River Room", "relation": "located_in", "subject": "Copper Book", "valid_from": null, "valid_until": null}` — correct_complete_fact; 
- `{"attitude": "believes", "evidence_ids": ["S11"], "holder": "Jun", "object": "Hill Court", "relation": "located_in", "subject": "Ivory Vase", "valid_from": null, "valid_until": null}` — correct_complete_fact; 
- `{"attitude": null, "evidence_ids": ["S12"], "holder": null, "object": "Elm Dock", "relation": "located_in", "subject": "Ivory Vase", "valid_from": null, "valid_until": null}` — correct_complete_fact; 

Missing: `[]`.

Manual diagnosis of source output (not rescoring): The Orchard pre-extraction supplies object --located_in--> place with the named believer in holder for both beliefs. Narrated realities remain separate with null holders. This satisfies the requested representation on this output, without establishing reliability.

B contextual:

Strict parse: True; after permitted recovery: True.

- `{"attitude": null, "evidence_ids": ["S2"], "holder": null, "object": "Ivory Vase", "relation": "owns", "subject": "Soren", "valid_from": null, "valid_until": null}` — supported_but_irrelevant; 
- `{"attitude": null, "evidence_ids": ["S3"], "holder": null, "object": "Copper Book", "relation": "carries", "subject": "Jun", "valid_from": 9, "valid_until": 17}` — supported_but_irrelevant; 
- `{"attitude": null, "evidence_ids": ["S4"], "holder": null, "object": "Ivory Vase", "relation": "carries", "subject": "Vera", "valid_from": 10, "valid_until": 16}` — supported_but_irrelevant; 
- `{"attitude": null, "evidence_ids": ["S5"], "holder": null, "object": "River Room", "relation": "located_in", "subject": "Jun", "valid_from": 10, "valid_until": 14}` — supported_but_irrelevant; 
- `{"attitude": null, "evidence_ids": ["S6"], "holder": null, "object": "Hill Court", "relation": "located_in", "subject": "Soren", "valid_from": 14, "valid_until": 18}` — supported_but_irrelevant; 
- `{"attitude": null, "evidence_ids": ["S7"], "holder": null, "object": "South Loft", "relation": "located_in", "subject": "Vera", "valid_from": 11, "valid_until": 16}` — supported_but_irrelevant; 
- `{"attitude": null, "evidence_ids": ["S8"], "holder": null, "object": "Elm Dock", "relation": "located_in", "subject": "Jade Drum", "valid_from": 12, "valid_until": 15}` — supported_but_irrelevant; 
- `{"attitude": "believes", "evidence_ids": ["S9"], "holder": "Soren", "object": "Copper Book is in South Loft", "relation": "believes", "subject": "Soren", "valid_from": null, "valid_until": null}` — unresolved_wording; 
- `{"attitude": null, "evidence_ids": ["S10"], "holder": null, "object": "River Room", "relation": "located_in", "subject": "Copper Book", "valid_from": null, "valid_until": null}` — correct_complete_fact; 
- `{"attitude": "believes", "evidence_ids": ["S11"], "holder": "Jun", "object": "Ivory Vase is in Hill Court", "relation": "believes", "subject": "Jun", "valid_from": null, "valid_until": null}` — unresolved_wording; 
- `{"attitude": null, "evidence_ids": ["S12"], "holder": null, "object": "Elm Dock", "relation": "located_in", "subject": "Ivory Vase", "valid_from": null, "valid_until": null}` — correct_complete_fact; 
- `{"attitude": null, "evidence_ids": ["S13"], "holder": null, "object": "Jade Drum", "relation": "owns", "subject": "Jun", "valid_from": null, "valid_until": null}` — supported_but_irrelevant; 
- `{"attitude": null, "evidence_ids": ["S14"], "holder": null, "object": "South Loft", "relation": "located_in", "subject": "Jun", "valid_from": 18, "valid_until": 20}` — supported_but_irrelevant; 

Missing: `[{"valid_from": null, "valid_until": null, "holder": "Soren", "attitude": "believes", "subject": "Copper Book", "relation": "located_in", "object": "South Loft", "evidence_ids": ["S9"]}, {"valid_from": null, "valid_until": null, "holder": "Jun", "attitude": "believes", "subject": "Ivory Vase", "relation": "located_in", "object": "Hill Court", "evidence_ids": ["S11"]}]`.

Manual diagnosis of source output (not rescoring): The named holders Soren and Jun are correct, but the believed relationships remain proposition strings under the relation believes. This differs from the successful decomposition in this story's query-blind output. No new equivalence was introduced to award these rows a match.


## Museum preparations

- S1: Niko believes that Pearl Box is in Glass Room.
- S2: Esme owns Ruby Medal.
- S3: Dario is in Upper Court from day 20 up to but not including day 23.
- S4: In reality, Pearl Box is in Brick Store.
- S5: Niko carries Ruby Medal from day 19 up to but not including day 26.
- S6: Onyx Bell is in Cedar Gate from day 23 up to but not including day 27.
- S7: Esme believes that Ruby Medal is in Cedar Gate.
- S8: Dario owns Onyx Bell.
- S9: Esme is in Glass Room from day 21 up to but not including day 25.
- S10: In reality, Ruby Medal is in Upper Court.
- S11: Niko owns Pearl Box.
- S12: Niko is in Brick Store from day 24 up to but not including day 28.
- S13: Esme carries Pearl Box from day 20 up to but not including day 24.
- S14: Dario is in Cedar Gate from day 28 up to but not including day 30.

### Who owns or carries what? Return all and only directly narrated ownership and carrying relationships, retaining every stated interval.

Reference answers:
```json
[
  {
    "valid_from": null,
    "valid_until": null,
    "holder": null,
    "attitude": null,
    "subject": "Esme",
    "relation": "owns",
    "object": "Ruby Medal",
    "evidence_ids": [
      "S2"
    ]
  },
  {
    "valid_from": 19,
    "valid_until": 26,
    "holder": null,
    "attitude": null,
    "subject": "Niko",
    "relation": "carries",
    "object": "Ruby Medal",
    "evidence_ids": [
      "S5"
    ]
  },
  {
    "valid_from": null,
    "valid_until": null,
    "holder": null,
    "attitude": null,
    "subject": "Dario",
    "relation": "owns",
    "object": "Onyx Bell",
    "evidence_ids": [
      "S8"
    ]
  },
  {
    "valid_from": null,
    "valid_until": null,
    "holder": null,
    "attitude": null,
    "subject": "Niko",
    "relation": "owns",
    "object": "Pearl Box",
    "evidence_ids": [
      "S11"
    ]
  },
  {
    "valid_from": 20,
    "valid_until": 24,
    "holder": null,
    "attitude": null,
    "subject": "Esme",
    "relation": "carries",
    "object": "Pearl Box",
    "evidence_ids": [
      "S13"
    ]
  }
]
```

A pre-extract/select:

Strict parse: True; after permitted recovery: True.

- `{"attitude": null, "evidence_ids": ["S2"], "holder": null, "object": "Ruby Medal", "relation": "owns", "subject": "Esme", "valid_from": null, "valid_until": null}` — correct_complete_fact; 
- `{"attitude": null, "evidence_ids": ["S5"], "holder": null, "object": "Ruby Medal", "relation": "carries", "subject": "Niko", "valid_from": 19, "valid_until": 26}` — correct_complete_fact; 
- `{"attitude": null, "evidence_ids": ["S11"], "holder": null, "object": "Pearl Box", "relation": "owns", "subject": "Niko", "valid_from": null, "valid_until": null}` — correct_complete_fact; 
- `{"attitude": null, "evidence_ids": ["S13"], "holder": null, "object": "Pearl Box", "relation": "carries", "subject": "Esme", "valid_from": 20, "valid_until": 24}` — correct_complete_fact; 
- `{"attitude": null, "evidence_ids": ["S8"], "holder": null, "object": "Onyx Bell", "relation": "owns", "subject": "Dario", "valid_from": null, "valid_until": null}` — correct_complete_fact; 

Missing: `[]`.

Manual diagnosis of source output (not rescoring): S1 and S7 use sentence-valued objects and believes as the relation, with null holder and attitude believes. The prose corresponds to the stated beliefs, but decomposition and holder pairing fail. Fixed selection uses Niko and Esme as attributed subjects instead of the believed objects and misses the object-reality targets. This fresh story reproduces the limitation without changing scoring or repairing the graph.

B contextual:

Strict parse: True; after permitted recovery: True.

- `{"attitude": "believes", "evidence_ids": ["S1"], "holder": null, "object": "Pearl Box is in Glass Room", "relation": "believes", "subject": "Niko", "valid_from": null, "valid_until": null}` — unresolved_wording; holder and attitude must appear together
- `{"attitude": null, "evidence_ids": ["S2"], "holder": null, "object": "Ruby Medal", "relation": "owns", "subject": "Esme", "valid_from": null, "valid_until": null}` — correct_complete_fact; 
- `{"attitude": null, "evidence_ids": ["S3"], "holder": null, "object": "Upper Court", "relation": "is in", "subject": "Dario", "valid_from": 20, "valid_until": 23}` — supported_but_irrelevant; 
- `{"attitude": null, "evidence_ids": ["S5"], "holder": null, "object": "Ruby Medal", "relation": "carries", "subject": "Niko", "valid_from": 19, "valid_until": 26}` — correct_complete_fact; 
- `{"attitude": null, "evidence_ids": ["S6"], "holder": null, "object": "Cedar Gate", "relation": "is in", "subject": "Onyx Bell", "valid_from": 23, "valid_until": 27}` — supported_but_irrelevant; 
- `{"attitude": "believes", "evidence_ids": ["S7"], "holder": null, "object": "Ruby Medal is in Cedar Gate", "relation": "believes", "subject": "Esme", "valid_from": null, "valid_until": null}` — unresolved_wording; holder and attitude must appear together
- `{"attitude": null, "evidence_ids": ["S9"], "holder": null, "object": "Glass Room", "relation": "is in", "subject": "Esme", "valid_from": 21, "valid_until": 25}` — supported_but_irrelevant; 
- `{"attitude": null, "evidence_ids": ["S12"], "holder": null, "object": "Brick Store", "relation": "is in", "subject": "Niko", "valid_from": 24, "valid_until": 28}` — supported_but_irrelevant; 
- `{"attitude": null, "evidence_ids": ["S13"], "holder": null, "object": "Pearl Box", "relation": "carries", "subject": "Esme", "valid_from": 20, "valid_until": 24}` — correct_complete_fact; 
- `{"attitude": null, "evidence_ids": ["S8"], "holder": null, "object": "Onyx Bell", "relation": "owns", "subject": "Dario", "valid_from": null, "valid_until": null}` — correct_complete_fact; 
- `{"attitude": null, "evidence_ids": ["S14"], "holder": null, "object": "Cedar Gate", "relation": "is in", "subject": "Dario", "valid_from": 28, "valid_until": 30}` — supported_but_irrelevant; 
- `{"attitude": null, "evidence_ids": ["S10"], "holder": null, "object": "Upper Court", "relation": "is in", "subject": "Ruby Medal", "valid_from": null, "valid_until": null}` — supported_but_irrelevant; 
- `{"attitude": null, "evidence_ids": ["S11"], "holder": null, "object": "Pearl Box", "relation": "owns", "subject": "Niko", "valid_from": null, "valid_until": null}` — correct_complete_fact; 

Missing: `[]`.

Manual diagnosis of source output (not rescoring): The possession answer unnecessarily includes both belief sentences. They also lack the requested relationship decomposition and have null holders despite attitude believes. The frozen matcher leaves wording unresolved; the observed format failure and contextual irrelevance are separately noted, without calling the belief sentences false.

### Which directly narrated location relationships have an explicitly stated interval overlapping day 22 up to but not including day 25? Return all of them with their full evidence-stated intervals, not clipped to the question window. Exclude undated locations and beliefs.

Reference answers:
```json
[
  {
    "valid_from": 20,
    "valid_until": 23,
    "holder": null,
    "attitude": null,
    "subject": "Dario",
    "relation": "located_in",
    "object": "Upper Court",
    "evidence_ids": [
      "S3"
    ]
  },
  {
    "valid_from": 23,
    "valid_until": 27,
    "holder": null,
    "attitude": null,
    "subject": "Onyx Bell",
    "relation": "located_in",
    "object": "Cedar Gate",
    "evidence_ids": [
      "S6"
    ]
  },
  {
    "valid_from": 21,
    "valid_until": 25,
    "holder": null,
    "attitude": null,
    "subject": "Esme",
    "relation": "located_in",
    "object": "Glass Room",
    "evidence_ids": [
      "S9"
    ]
  },
  {
    "valid_from": 24,
    "valid_until": 28,
    "holder": null,
    "attitude": null,
    "subject": "Niko",
    "relation": "located_in",
    "object": "Brick Store",
    "evidence_ids": [
      "S12"
    ]
  }
]
```

A pre-extract/select:

Strict parse: True; after permitted recovery: True.

- `{"attitude": null, "evidence_ids": ["S3"], "holder": null, "object": "Upper Court", "relation": "is_in", "subject": "Dario", "valid_from": 20, "valid_until": 23}` — correct_complete_fact; 
- `{"attitude": null, "evidence_ids": ["S6"], "holder": null, "object": "Cedar Gate", "relation": "located_in", "subject": "Onyx Bell", "valid_from": 23, "valid_until": 27}` — correct_complete_fact; 
- `{"attitude": null, "evidence_ids": ["S9"], "holder": null, "object": "Glass Room", "relation": "is_in", "subject": "Esme", "valid_from": 21, "valid_until": 25}` — correct_complete_fact; 
- `{"attitude": null, "evidence_ids": ["S12"], "holder": null, "object": "Brick Store", "relation": "is_in", "subject": "Niko", "valid_from": 24, "valid_until": 28}` — correct_complete_fact; 

Missing: `[]`.

Manual diagnosis of source output (not rescoring): S1 and S7 use sentence-valued objects and believes as the relation, with null holder and attitude believes. The prose corresponds to the stated beliefs, but decomposition and holder pairing fail. Fixed selection uses Niko and Esme as attributed subjects instead of the believed objects and misses the object-reality targets. This fresh story reproduces the limitation without changing scoring or repairing the graph.

B contextual:

Strict parse: True; after permitted recovery: True.

- `{"attitude": null, "evidence_ids": ["S3"], "holder": null, "object": "Upper Court", "relation": "located_in", "subject": "Dario", "valid_from": 20, "valid_until": 23}` — correct_complete_fact; 
- `{"attitude": null, "evidence_ids": ["S6"], "holder": null, "object": "Cedar Gate", "relation": "located_in", "subject": "Onyx Bell", "valid_from": 23, "valid_until": 27}` — correct_complete_fact; 
- `{"attitude": null, "evidence_ids": ["S9"], "holder": null, "object": "Glass Room", "relation": "located_in", "subject": "Esme", "valid_from": 21, "valid_until": 25}` — correct_complete_fact; 
- `{"attitude": null, "evidence_ids": ["S12"], "holder": null, "object": "Brick Store", "relation": "located_in", "subject": "Niko", "valid_from": 24, "valid_until": 28}` — correct_complete_fact; 
- `{"attitude": null, "evidence_ids": ["S13"], "holder": null, "object": "Pearl Box", "relation": "carries", "subject": "Esme", "valid_from": 20, "valid_until": 24}` — supported_but_irrelevant; 

Missing: `[]`.

### Who explicitly believes what, and what reality is separately narrated about those same objects? Return each underlying believed relationship with its attribution and each matching narrated reality without attribution. Omit unrelated facts.

Reference answers:
```json
[
  {
    "valid_from": null,
    "valid_until": null,
    "holder": "Niko",
    "attitude": "believes",
    "subject": "Pearl Box",
    "relation": "located_in",
    "object": "Glass Room",
    "evidence_ids": [
      "S1"
    ]
  },
  {
    "valid_from": null,
    "valid_until": null,
    "holder": null,
    "attitude": null,
    "subject": "Pearl Box",
    "relation": "located_in",
    "object": "Brick Store",
    "evidence_ids": [
      "S4"
    ]
  },
  {
    "valid_from": null,
    "valid_until": null,
    "holder": "Esme",
    "attitude": "believes",
    "subject": "Ruby Medal",
    "relation": "located_in",
    "object": "Cedar Gate",
    "evidence_ids": [
      "S7"
    ]
  },
  {
    "valid_from": null,
    "valid_until": null,
    "holder": null,
    "attitude": null,
    "subject": "Ruby Medal",
    "relation": "located_in",
    "object": "Upper Court",
    "evidence_ids": [
      "S10"
    ]
  }
]
```

A pre-extract/select:

Strict parse: True; after permitted recovery: True.

- `{"attitude": "believes", "evidence_ids": ["S1"], "holder": null, "object": "Pearl Box is in Glass Room", "relation": "believes", "subject": "Niko", "valid_from": null, "valid_until": null}` — unresolved_wording; holder and attitude must appear together
- `{"attitude": "believes", "evidence_ids": ["S7"], "holder": null, "object": "Ruby Medal is in Cedar Gate", "relation": "believes", "subject": "Esme", "valid_from": null, "valid_until": null}` — unresolved_wording; holder and attitude must appear together
- `{"attitude": null, "evidence_ids": ["S9"], "holder": null, "object": "Glass Room", "relation": "is_in", "subject": "Esme", "valid_from": 21, "valid_until": 25}` — supported_but_irrelevant; 
- `{"attitude": null, "evidence_ids": ["S12"], "holder": null, "object": "Brick Store", "relation": "is_in", "subject": "Niko", "valid_from": 24, "valid_until": 28}` — supported_but_irrelevant; 

Missing: `[{"valid_from": null, "valid_until": null, "holder": "Niko", "attitude": "believes", "subject": "Pearl Box", "relation": "located_in", "object": "Glass Room", "evidence_ids": ["S1"]}, {"valid_from": null, "valid_until": null, "holder": null, "attitude": null, "subject": "Pearl Box", "relation": "located_in", "object": "Brick Store", "evidence_ids": ["S4"]}, {"valid_from": null, "valid_until": null, "holder": "Esme", "attitude": "believes", "subject": "Ruby Medal", "relation": "located_in", "object": "Cedar Gate", "evidence_ids": ["S7"]}, {"valid_from": null, "valid_until": null, "holder": null, "attitude": null, "subject": "Ruby Medal", "relation": "located_in", "object": "Upper Court", "evidence_ids": ["S10"]}]`.

Manual diagnosis of source output (not rescoring): S1 and S7 use sentence-valued objects and believes as the relation, with null holder and attitude believes. The prose corresponds to the stated beliefs, but decomposition and holder pairing fail. Fixed selection uses Niko and Esme as attributed subjects instead of the believed objects and misses the object-reality targets. This fresh story reproduces the limitation without changing scoring or repairing the graph.

B contextual:

Strict parse: True; after permitted recovery: True.

- `{"attitude": "believes", "evidence_ids": ["S1"], "holder": null, "object": "Pearl Box is in Glass Room", "relation": "believes", "subject": "Niko", "valid_from": null, "valid_until": null}` — unresolved_wording; holder and attitude must appear together
- `{"attitude": null, "evidence_ids": ["S4"], "holder": null, "object": "Brick Store", "relation": "located_in", "subject": "Pearl Box", "valid_from": null, "valid_until": null}` — correct_complete_fact; 
- `{"attitude": null, "evidence_ids": ["S2"], "holder": null, "object": "Ruby Medal", "relation": "owns", "subject": "Esme", "valid_from": null, "valid_until": null}` — supported_but_irrelevant; 
- `{"attitude": null, "evidence_ids": ["S5"], "holder": null, "object": "Ruby Medal", "relation": "carries", "subject": "Niko", "valid_from": 19, "valid_until": 26}` — supported_but_irrelevant; 
- `{"attitude": null, "evidence_ids": ["S10"], "holder": null, "object": "Upper Court", "relation": "located_in", "subject": "Ruby Medal", "valid_from": null, "valid_until": null}` — correct_complete_fact; 
- `{"attitude": "believes", "evidence_ids": ["S7"], "holder": null, "object": "Ruby Medal is in Cedar Gate", "relation": "believes", "subject": "Esme", "valid_from": null, "valid_until": null}` — unresolved_wording; holder and attitude must appear together
- `{"attitude": null, "evidence_ids": ["S3"], "holder": null, "object": "Upper Court", "relation": "is_in", "subject": "Dario", "valid_from": 20, "valid_until": 23}` — supported_but_irrelevant; 
- `{"attitude": null, "evidence_ids": ["S6"], "holder": null, "object": "Cedar Gate", "relation": "located_in", "subject": "Onyx Bell", "valid_from": 23, "valid_until": 27}` — supported_but_irrelevant; 
- `{"attitude": null, "evidence_ids": ["S8"], "holder": null, "object": "Onyx Bell", "relation": "owns", "subject": "Dario", "valid_from": null, "valid_until": null}` — supported_but_irrelevant; 
- `{"attitude": null, "evidence_ids": ["S9"], "holder": null, "object": "Glass Room", "relation": "is_in", "subject": "Esme", "valid_from": 21, "valid_until": 25}` — supported_but_irrelevant; 
- `{"attitude": null, "evidence_ids": ["S11"], "holder": null, "object": "Pearl Box", "relation": "owns", "subject": "Niko", "valid_from": null, "valid_until": null}` — supported_but_irrelevant; 
- `{"attitude": null, "evidence_ids": ["S12"], "holder": null, "object": "Brick Store", "relation": "is_in", "subject": "Niko", "valid_from": 24, "valid_until": 28}` — supported_but_irrelevant; 
- `{"attitude": null, "evidence_ids": ["S13"], "holder": null, "object": "Pearl Box", "relation": "carries", "subject": "Esme", "valid_from": 20, "valid_until": 24}` — supported_but_irrelevant; 
- `{"attitude": null, "evidence_ids": ["S14"], "holder": null, "object": "Cedar Gate", "relation": "is_in", "subject": "Dario", "valid_from": 28, "valid_until": 30}` — supported_but_irrelevant; 

Missing: `[{"valid_from": null, "valid_until": null, "holder": "Niko", "attitude": "believes", "subject": "Pearl Box", "relation": "located_in", "object": "Glass Room", "evidence_ids": ["S1"]}, {"valid_from": null, "valid_until": null, "holder": "Esme", "attitude": "believes", "subject": "Ruby Medal", "relation": "located_in", "object": "Cedar Gate", "evidence_ids": ["S7"]}]`.

Manual diagnosis of source output (not rescoring): Both belief rows retain sentence-valued objects, relation believes and null holders. The two narrated realities are correctly separate, but the attributed facts do not follow the required decomposed format. Other source-supported facts are also included despite being irrelevant to the belief question.

## Historical syntax recovery (derived only)

harbor: 3 complete matches / 10 predictions / 5 references. P/R/F1 0.300/0.600/0.400.

Exact edit record: `[{"byte_offset": 885, "character_offset": 885, "removed": ",", "replacement": ""}]`.

Recovered records (all retained):
```json
{
  "facts": [
    {
      "subject": "Lena",
      "relation": "owns",
      "object": "Cobalt Compass",
      "evidence_ids": [
        "S1"
      ]
    },
    {
      "subject": "Omar",
      "relation": "owns",
      "object": "Amber Lantern",
      "evidence_ids": [
        "S2"
      ]
    },
    {
      "subject": "Priya",
      "relation": "carries",
      "object": "Cobalt Compass",
      "evidence_ids": [
        "S3"
      ]
    },
    {
      "subject": "Lena",
      "relation": "carries",
      "object": "Amber Lantern",
      "evidence_ids": [
        "S4"
      ]
    },
    {
      "subject": "Lena",
      "relation": "located_in",
      "object": "North Hall",
      "evidence_ids": [
        "S5"
      ]
    },
    {
      "subject": "Omar",
      "relation": "located_in",
      "object": "East Garden",
      "evidence_ids": [
        "S6"
      ]
    },
    {
      "subject": "Priya",
      "relation": "located_in",
      "object": "Stone Shed",
      "evidence_ids": [
        "S7"
      ]
    },
    {
      "subject": "Silver Flute",
      "relation": "located_in",
      "object": "West Pier",
      "evidence_ids": [
        "S8"
      ]
    },
    {
      "subject": "Priya",
      "relation": "owns",
      "object": "Silver Flute",
      "evidence_ids": [
        "S13"
      ]
    },
    {
      "subject": "Lena",
      "relation": "located_in",
      "object": "Stone Shed",
      "evidence_ids": [
        "S14"
      ]
    }
  ]
}
```

orchard: 3 complete matches / 10 predictions / 5 references. P/R/F1 0.300/0.600/0.400.

Exact edit record: `[{"byte_offset": 835, "character_offset": 835, "removed": ",", "replacement": ""}]`.

Recovered records (all retained):
```json
{
  "facts": [
    {
      "subject": "Vera",
      "relation": "owns",
      "object": "Copper Book",
      "evidence_ids": [
        "S1"
      ]
    },
    {
      "subject": "Soren",
      "relation": "owns",
      "object": "Ivory Vase",
      "evidence_ids": [
        "S2"
      ]
    },
    {
      "subject": "Jun",
      "relation": "carries",
      "object": "Copper Book",
      "evidence_ids": [
        "S3"
      ]
    },
    {
      "subject": "Vera",
      "relation": "carries",
      "object": "Ivory Vase",
      "evidence_ids": [
        "S4"
      ]
    },
    {
      "subject": "Jun",
      "relation": "is_in",
      "object": "River Room",
      "evidence_ids": [
        "S5"
      ]
    },
    {
      "subject": "Soren",
      "relation": "is_in",
      "object": "Hill Court",
      "evidence_ids": [
        "S6"
      ]
    },
    {
      "subject": "Vera",
      "relation": "is_in",
      "object": "South Loft",
      "evidence_ids": [
        "S7"
      ]
    },
    {
      "subject": "Jade Drum",
      "relation": "is_in",
      "object": "Elm Dock",
      "evidence_ids": [
        "S8"
      ]
    },
    {
      "subject": "Jun",
      "relation": "owns",
      "object": "Jade Drum",
      "evidence_ids": [
        "S13"
      ]
    },
    {
      "subject": "Jun",
      "relation": "is_in",
      "object": "South Loft",
      "evidence_ids": [
        "S14"
      ]
    }
  ]
}
```

## Calls, allocation and limitations

| Call | Story/task | Strict / after recovery | Finish | Tokens in/out | Seconds |
|---|---|---|---|---|---|
| 1 | harbor/all | True / True | stop | 660/605 | 10.943 |
| 2 | orchard/all | True / True | stop | 671/615 | 10.896 |
| 3 | museum/all | True / True | stop | 678/615 | 10.758 |
| 4 | harbor/possession | True / True | stop | 651/422 | 7.605 |
| 5 | harbor/locations | True / True | stop | 678/255 | 5.285 |
| 6 | harbor/belief | True / True | stop | 667/607 | 10.843 |
| 7 | orchard/possession | True / True | stop | 662/524 | 9.670 |
| 8 | orchard/locations | True / True | stop | 691/224 | 4.735 |
| 9 | orchard/belief | True / True | stop | 678/580 | 9.434 |
| 10 | museum/possession | True / True | stop | 669/572 | 9.411 |
| 11 | museum/locations | True / True | stop | 698/226 | 4.933 |
| 12 | museum/belief | True / True | stop | 685/615 | 10.207 |

Executed 12/12 single attempts; no adaptive tuning. New allocation 249.452431s; cumulative 10512.544560s. Open allocation/service journals 0/0. Stop reason: completed.

Three query-blind calls precede all contextual transmissions. Baseline call costs are shared across its three selections, not counted three times. Per-call raw responses, token counts and request times are in the canonical JSON. One pinned model, nonthinking sampling, 12,288 context, 2,048 output tokens for both approaches; no production-p95 inference from this batch.

Unique calls used 8088 input / 5860 completion tokens and 104.720401 request seconds. Sampled VRAM/RAM/project-storage peaks (bytes): 22636658688 / 7185870848 / 17337667072. Sampling is not a guarantee that no between-sample peak occurred.

Pinned Qwen/Qwen3-8B-AWQ revision `4da05a8edb55c6046cce958586c33b61da07bb79`; vLLM 0.10.2, unchanged nonthinking template and sampling. Actual input tokens span the counts above and remain within context with the output reserve. The authored complete-output capacity check was not model output or a throughput estimate.

This small exploratory batch has no minimum score for display/completion. Harbor and Orchard are repeatedly developed material; Museum is fresh development evidence, not held-out generalization. Original scores, registered metrics, C0 competence and independent-review gates remain unchanged. No automatic follow-up service start.

[Graphs](figures/compact_story_v2_comparison.html) · [CSV](tables/compact_story_v2_results.csv) · [Original report](COMPACT_STORY_RESULTS.md).

Reproduce on CPU: `python -m scripts.report_compact_story_v2 --run <restricted-run> --output reports`.
