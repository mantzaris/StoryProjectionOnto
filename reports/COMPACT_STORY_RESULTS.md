# Compact story results — exploratory development

A = query-blind pre-extraction followed by fixed CPU selection. B = direct contextual extraction. These are not registered C1/C2 results or evidence of ontology construction.

| Story / question | Approach | Complete / parseable | Valid citations | Bare P/R/F1 | Qualified P/R/F1 |
|---|---|---|---|---|---|
| harbor / possession | A pre-extract/select | True / True | 5/5 | 1.000 / 1.000 / 1.000 | 0.600 / 0.600 / 0.600 |
| harbor / possession | B contextual | True / False | 0/0 | 0.000 / 0.000 / 0.000 | 0.000 / 0.000 / 0.000 |
| harbor / locations | A pre-extract/select | True / True | 0/0 | 0.000 / 0.000 / 0.000 | 0.000 / 0.000 / 0.000 |
| harbor / locations | B contextual | True / True | 5/5 | 0.200 / 0.250 / 0.222 | 0.000 / 0.000 / 0.000 |
| harbor / belief | A pre-extract/select | True / True | 5/5 | 0.000 / 0.000 / 0.000 | 0.000 / 0.000 / 0.000 |
| harbor / belief | B contextual | True / True | 14/14 | 0.143 / 0.500 / 0.222 | 0.143 / 0.500 / 0.222 |
| orchard / possession | A pre-extract/select | True / True | 5/5 | 1.000 / 1.000 / 1.000 | 0.600 / 0.600 / 0.600 |
| orchard / possession | B contextual | True / False | 0/0 | 0.000 / 0.000 / 0.000 | 0.000 / 0.000 / 0.000 |
| orchard / locations | A pre-extract/select | True / True | 0/0 | 0.000 / 0.000 / 0.000 | 0.000 / 0.000 / 0.000 |
| orchard / locations | B contextual | True / True | 11/11 | 0.364 / 1.000 / 0.533 | 0.091 / 0.250 / 0.133 |
| orchard / belief | A pre-extract/select | True / True | 0/0 | 0.000 / 0.000 / 0.000 | 0.000 / 0.000 / 0.000 |
| orchard / belief | B contextual | True / True | 14/14 | 0.143 / 0.500 / 0.222 | 0.143 / 0.500 / 0.222 |

## What the comparison shows

Contextual extraction did not improve overall qualified accuracy on either story. It frequently repeated unrelated narrative facts and omitted qualifications. Both contextual possession responses ended normally but contain an invalid trailing comma; no syntax repair was applied. The only correctly qualified contextual interval is Jade Drum in Elm Dock [12,15), alongside an unqualified duplicate and unrelated records. Belief relationships were not decomposed as requested, even when the embedded sentence preserved understandable belief content. This is a representation failure, not proof that every unmatched belief string is factually false.

Unparseable answers retain the frozen zero-score answer-failure convention. Their per-fact denominator is not measurable; zero parsed records does not mean no candidate text was emitted. No trailing-comma removal or semantic repair was applied. Empty CPU selections are separate: their omission/zero-recall scores are measured outputs of the frozen filter.

harbor: mean question-level qualified F1 is 0.200 for A and 0.074 for B (lower). This describes these three answers, not a general reliability or significance result.

orchard: mean question-level qualified F1 is 0.200 for A and 0.119 for B (lower). This describes these three answers, not a general reliability or significance result.

Differences in target score can reflect relevance selection, omissions, or qualification errors; consult the assertion-level statuses below. Supported but irrelevant facts remain precision errors. Unrecognized wording remains unresolved, not automatically false or accepted. All predictions and duplicate records stay in their applicable denominators.

## Three readable examples

- Evidence: S3: Priya carries Cobalt Compass from day 0 up to but not including day 6.
  Generated: Priya — carries → Cobalt Compass; evidence_ids=[S3]; no bounds
  Assessment: Correct bare relationship, missing explicit [0,6) qualification. CPU selection preserves the error.

- Evidence: S8: Jade Drum is in Elm Dock from day 12 up to but not including day 15.
  Generated: Jade Drum — located_in → Elm Dock; evidence_ids=[S8]; valid_from=12, valid_until=15
  Assessment: One complete contextual temporal fact is correct; an earlier unqualified duplicate and other wrong/irrelevant records remain in the denominator.

- Evidence: S1: Vera owns Copper Book.
  Generated: Jun — owns → Copper Book; evidence_ids=[S1]
  Assessment: Known endpoint substitution, unsupported by the cited sentence, not an unresolved paraphrase.

## Readable stories and answers

### The harbor supplies

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

#### possession: Who owns or carries what? Return all and only directly narrated ownership and carrying relationships, retaining every stated interval.

Reference answers:

- Lena — owns → Cobalt Compass; evidence=["S1"]
- Omar — owns → Amber Lantern; evidence=["S2"]
- Priya — carries → Cobalt Compass [0, 6); evidence=["S3"]
- Lena — carries → Amber Lantern [1, 5); evidence=["S4"]
- Priya — owns → Silver Flute; evidence=["S13"]

A pre-extract/select:

Manual source-output diagnosis (not rescoring): All 14 source sentences receive records. Twelve bare relationships match; all seven explicit intervals are missing. The two belief records use the person as subject, believes as predicate and only the object name as object, omitting the believed location. Five of fourteen complete qualified facts match. The fixed belief filter consequently selects the generated believer-subjects' locations, not the missing object-centered proposition; no CPU correction is applied.

- Lena — owns → Cobalt Compass; evidence=["S1"] — correct_complete_fact
- Omar — owns → Amber Lantern; evidence=["S2"] — correct_complete_fact
- Priya — carries → Cobalt Compass; evidence=["S3"] — qualification_or_evidence_error; missing, unsupported or wrong explicit temporal qualification
- Lena — carries → Amber Lantern; evidence=["S4"] — qualification_or_evidence_error; missing, unsupported or wrong explicit temporal qualification
- Priya — owns → Silver Flute; evidence=["S13"] — correct_complete_fact

Missing complete facts: [{"subject": "Priya", "relation": "carries", "object": "Cobalt Compass", "evidence_ids": ["S3"], "valid_from": 0, "valid_until": 6}, {"subject": "Lena", "relation": "carries", "object": "Amber Lantern", "evidence_ids": ["S4"], "valid_from": 1, "valid_until": 5}]

B contextual:

Manual source-output diagnosis (not rescoring): Completed response, finish stop, not truncated. The final facts-list comma before ] makes JSON invalid. The raw text visibly contains ten records, including unrelated locations and no temporal bounds. Formal per-fact metrics are unavailable for this unparseable response; the frozen answer-failure convention assigns zero scores, not an assertion that the model emitted zero text or zero candidate records.

No complete parseable source response. Raw received output:
```text
{"facts": [{"subject":"Lena","relation":"owns","object":"Cobalt Compass","evidence_ids":["S1"]},{"subject":"Omar","relation":"owns","object":"Amber Lantern","evidence_ids":["S2"]},{"subject":"Priya","relation":"carries","object":"Cobalt Compass","evidence_ids":["S3"]},{"subject":"Lena","relation":"carries","object":"Amber Lantern","evidence_ids":["S4"]},{"subject":"Lena","relation":"located_in","object":"North Hall","evidence_ids":["S5"]},{"subject":"Omar","relation":"located_in","object":"East Garden","evidence_ids":["S6"]},{"subject":"Priya","relation":"located_in","object":"Stone Shed","evidence_ids":["S7"]},{"subject":"Silver Flute","relation":"located_in","object":"West Pier","evidence_ids":["S8"]},{"subject":"Priya","relation":"owns","object":"Silver Flute","evidence_ids":["S13"]},{"subject":"Lena","relation":"located_in","object":"Stone Shed","evidence_ids":["S14"]},]}
```

Missing complete facts: [{"subject": "Lena", "relation": "owns", "object": "Cobalt Compass", "evidence_ids": ["S1"]}, {"subject": "Omar", "relation": "owns", "object": "Amber Lantern", "evidence_ids": ["S2"]}, {"subject": "Priya", "relation": "carries", "object": "Cobalt Compass", "evidence_ids": ["S3"], "valid_from": 0, "valid_until": 6}, {"subject": "Lena", "relation": "carries", "object": "Amber Lantern", "evidence_ids": ["S4"], "valid_from": 1, "valid_until": 5}, {"subject": "Priya", "relation": "owns", "object": "Silver Flute", "evidence_ids": ["S13"]}]

#### locations: Which directly narrated location relationships have an explicitly stated interval overlapping day 2 up to but not including day 4? Return all of them with their full evidence-stated intervals, not clipped to the question window. Exclude undated locations and beliefs.

Reference answers:

- Lena — located_in → North Hall [1, 5); evidence=["S5"]
- Omar — located_in → East Garden [0, 3); evidence=["S6"]
- Priya — located_in → Stone Shed [3, 6); evidence=["S7"]
- Silver Flute — located_in → West Pier [2, 4); evidence=["S8"]

A pre-extract/select:

Manual source-output diagnosis (not rescoring): All 14 source sentences receive records. Twelve bare relationships match; all seven explicit intervals are missing. The two belief records use the person as subject, believes as predicate and only the object name as object, omitting the believed location. Five of fourteen complete qualified facts match. The fixed belief filter consequently selects the generated believer-subjects' locations, not the missing object-centered proposition; no CPU correction is applied.


Missing complete facts: [{"subject": "Lena", "relation": "located_in", "object": "North Hall", "evidence_ids": ["S5"], "valid_from": 1, "valid_until": 5}, {"subject": "Omar", "relation": "located_in", "object": "East Garden", "evidence_ids": ["S6"], "valid_from": 0, "valid_until": 3}, {"subject": "Priya", "relation": "located_in", "object": "Stone Shed", "evidence_ids": ["S7"], "valid_from": 3, "valid_until": 6}, {"subject": "Silver Flute", "relation": "located_in", "object": "West Pier", "evidence_ids": ["S8"], "valid_from": 2, "valid_until": 4}]

B contextual:

Manual source-output diagnosis (not rescoring): The interval-location answer returns undated reality facts, one unqualified target location, and two beliefs stripped of attribution. None has an explicit interval. Silver Flute in West Pier has correct bare endpoints but lacks [2,4). Belief citations S9/S11 cannot support global narrated location claims.

- Cobalt Compass — located_in → East Garden; evidence=["S10"] — supported_but_irrelevant
- Amber Lantern — located_in → West Pier; evidence=["S12"] — supported_but_irrelevant
- Silver Flute — located_in → West Pier; evidence=["S8"] — qualification_or_evidence_error; missing, unsupported or wrong explicit temporal qualification
- Cobalt Compass — located_in → North Hall; evidence=["S9"] — qualification_or_evidence_error; missing, unsupported or wrong holder/attitude qualification
- Amber Lantern — located_in → Stone Shed; evidence=["S11"] — qualification_or_evidence_error; missing, unsupported or wrong holder/attitude qualification

Missing complete facts: [{"subject": "Lena", "relation": "located_in", "object": "North Hall", "evidence_ids": ["S5"], "valid_from": 1, "valid_until": 5}, {"subject": "Omar", "relation": "located_in", "object": "East Garden", "evidence_ids": ["S6"], "valid_from": 0, "valid_until": 3}, {"subject": "Priya", "relation": "located_in", "object": "Stone Shed", "evidence_ids": ["S7"], "valid_from": 3, "valid_until": 6}, {"subject": "Silver Flute", "relation": "located_in", "object": "West Pier", "evidence_ids": ["S8"], "valid_from": 2, "valid_until": 4}]

#### belief: Who explicitly believes what, and what reality is separately narrated about those same objects? Return each underlying believed relationship with its attribution and each matching narrated reality without attribution. Omit unrelated facts.

Reference answers:

- Cobalt Compass — located_in → North Hall; holder=Lena, attitude=believes; evidence=["S9"]
- Cobalt Compass — located_in → East Garden; evidence=["S10"]
- Amber Lantern — located_in → Stone Shed; holder=Omar, attitude=believes; evidence=["S11"]
- Amber Lantern — located_in → West Pier; evidence=["S12"]

A pre-extract/select:

Manual source-output diagnosis (not rescoring): All 14 source sentences receive records. Twelve bare relationships match; all seven explicit intervals are missing. The two belief records use the person as subject, believes as predicate and only the object name as object, omitting the believed location. Five of fourteen complete qualified facts match. The fixed belief filter consequently selects the generated believer-subjects' locations, not the missing object-centered proposition; no CPU correction is applied.

- Lena — located_in → North Hall; evidence=["S5"] — qualification_or_evidence_error; missing, unsupported or wrong explicit temporal qualification
- Omar — located_in → East Garden; evidence=["S6"] — qualification_or_evidence_error; missing, unsupported or wrong explicit temporal qualification
- Lena — believes → Cobalt Compass; holder=Lena, attitude=believes; evidence=["S9"] — unresolved_wording
- Omar — believes → Amber Lantern; holder=Omar, attitude=believes; evidence=["S11"] — unresolved_wording
- Lena — located_in → Stone Shed; evidence=["S14"] — qualification_or_evidence_error; missing, unsupported or wrong explicit temporal qualification

Missing complete facts: [{"subject": "Cobalt Compass", "relation": "located_in", "object": "North Hall", "evidence_ids": ["S9"], "holder": "Lena", "attitude": "believes"}, {"subject": "Cobalt Compass", "relation": "located_in", "object": "East Garden", "evidence_ids": ["S10"]}, {"subject": "Amber Lantern", "relation": "located_in", "object": "Stone Shed", "evidence_ids": ["S11"], "holder": "Omar", "attitude": "believes"}, {"subject": "Amber Lantern", "relation": "located_in", "object": "West Pier", "evidence_ids": ["S12"]}]

B contextual:

Manual source-output diagnosis (not rescoring): The belief answer repeats all fourteen story facts. Only the two separately narrated object locations match the four-target question. The two belief statements remain embedded in object strings without required holder/attitude; seven unrelated time-qualified source facts lose their bounds. Repeated irrelevant content harms contextual precision even where a bare relationship is supported elsewhere in the story.

- Lena — owns → Cobalt Compass; evidence=["S1"] — supported_but_irrelevant
- Omar — owns → Amber Lantern; evidence=["S2"] — supported_but_irrelevant
- Priya — carries → Cobalt Compass; evidence=["S3"] — qualification_or_evidence_error; missing, unsupported or wrong explicit temporal qualification
- Lena — carries → Amber Lantern; evidence=["S4"] — qualification_or_evidence_error; missing, unsupported or wrong explicit temporal qualification
- Lena — is_in → North Hall; evidence=["S5"] — qualification_or_evidence_error; missing, unsupported or wrong explicit temporal qualification
- Omar — is_in → East Garden; evidence=["S6"] — qualification_or_evidence_error; missing, unsupported or wrong explicit temporal qualification
- Priya — is_in → Stone Shed; evidence=["S7"] — qualification_or_evidence_error; missing, unsupported or wrong explicit temporal qualification
- Silver Flute — is_in → West Pier; evidence=["S8"] — qualification_or_evidence_error; missing, unsupported or wrong explicit temporal qualification
- Lena — believes → Cobalt Compass is in North Hall; evidence=["S9"] — unresolved_wording
- Cobalt Compass — is_in → East Garden; evidence=["S10"] — correct_complete_fact
- Omar — believes → Amber Lantern is in Stone Shed; evidence=["S11"] — unresolved_wording
- Amber Lantern — is_in → West Pier; evidence=["S12"] — correct_complete_fact
- Priya — owns → Silver Flute; evidence=["S13"] — supported_but_irrelevant
- Lena — is_in → Stone Shed; evidence=["S14"] — qualification_or_evidence_error; missing, unsupported or wrong explicit temporal qualification

Missing complete facts: [{"subject": "Cobalt Compass", "relation": "located_in", "object": "North Hall", "evidence_ids": ["S9"], "holder": "Lena", "attitude": "believes"}, {"subject": "Amber Lantern", "relation": "located_in", "object": "Stone Shed", "evidence_ids": ["S11"], "holder": "Omar", "attitude": "believes"}]

Query-blind source collection (actual, before CPU selection):
```json
{"facts": [{"subject":"Lena","relation":"owns","object":"Cobalt Compass","evidence_ids":["S1"]},{"subject":"Omar","relation":"owns","object":"Amber Lantern","evidence_ids":["S2"]},{"subject":"Priya","relation":"carries","object":"Cobalt Compass","evidence_ids":["S3"]},{"subject":"Lena","relation":"carries","object":"Amber Lantern","evidence_ids":["S4"]},{"subject":"Lena","relation":"located_in","object":"North Hall","evidence_ids":["S5"]},{"subject":"Omar","relation":"located_in","object":"East Garden","evidence_ids":["S6"]},{"subject":"Priya","relation":"located_in","object":"Stone Shed","evidence_ids":["S7"]},{"subject":"Silver Flute","relation":"located_in","object":"West Pier","evidence_ids":["S8"]},{"subject":"Lena","relation":"believes","object":"Cobalt Compass","evidence_ids":["S9"],"holder":"Lena","attitude":"believes"},{"subject":"Cobalt Compass","relation":"located_in","object":"East Garden","evidence_ids":["S10"]},{"subject":"Omar","relation":"believes","object":"Amber Lantern","evidence_ids":["S11"],"holder":"Omar","attitude":"believes"},{"subject":"Amber Lantern","relation":"located_in","object":"West Pier","evidence_ids":["S12"]},{"subject":"Priya","relation":"owns","object":"Silver Flute","evidence_ids":["S13"]},{"subject":"Lena","relation":"located_in","object":"Stone Shed","evidence_ids":["S14"]}]}
```

### The orchard delivery

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

#### possession: Who owns or carries what? Return all and only directly narrated ownership and carrying relationships, retaining every stated interval.

Reference answers:

- Vera — owns → Copper Book; evidence=["S1"]
- Soren — owns → Ivory Vase; evidence=["S2"]
- Jun — carries → Copper Book [9, 17); evidence=["S3"]
- Vera — carries → Ivory Vase [10, 16); evidence=["S4"]
- Jun — owns → Jade Drum; evidence=["S13"]

A pre-extract/select:

Manual source-output diagnosis (not rescoring): All 14 sentences receive records; twelve bare relationships and five complete facts match. Seven explicit intervals are absent. Belief records embed entire propositions in object strings and omit holder/attitude. The frozen filter therefore cannot identify any attributed records and returns an empty belief answer. Unknown predicate/endpoint strings retain their frozen unresolved status; the manual format failure is separate.

- Vera — owns → Copper Book; evidence=["S1"] — correct_complete_fact
- Soren — owns → Ivory Vase; evidence=["S2"] — correct_complete_fact
- Jun — carries → Copper Book; evidence=["S3"] — qualification_or_evidence_error; missing, unsupported or wrong explicit temporal qualification
- Vera — carries → Ivory Vase; evidence=["S4"] — qualification_or_evidence_error; missing, unsupported or wrong explicit temporal qualification
- Jun — owns → Jade Drum; evidence=["S13"] — correct_complete_fact

Missing complete facts: [{"subject": "Jun", "relation": "carries", "object": "Copper Book", "evidence_ids": ["S3"], "valid_from": 9, "valid_until": 17}, {"subject": "Vera", "relation": "carries", "object": "Ivory Vase", "evidence_ids": ["S4"], "valid_from": 10, "valid_until": 16}]

B contextual:

Manual source-output diagnosis (not rescoring): Like the first story's possession response, this is a normally completed answer with ten visible record objects followed by an invalid trailing comma. No closing JSON or semantic content was manufactured; formal answer failure is preserved. The raw records also include irrelevant locations and omit all time bounds.

No complete parseable source response. Raw received output:
```text
{"facts": [{"subject":"Vera","relation":"owns","object":"Copper Book","evidence_ids":["S1"]},{"subject":"Soren","relation":"owns","object":"Ivory Vase","evidence_ids":["S2"]},{"subject":"Jun","relation":"carries","object":"Copper Book","evidence_ids":["S3"]},{"subject":"Vera","relation":"carries","object":"Ivory Vase","evidence_ids":["S4"]},{"subject":"Jun","relation":"is_in","object":"River Room","evidence_ids":["S5"]},{"subject":"Soren","relation":"is_in","object":"Hill Court","evidence_ids":["S6"]},{"subject":"Vera","relation":"is_in","object":"South Loft","evidence_ids":["S7"]},{"subject":"Jade Drum","relation":"is_in","object":"Elm Dock","evidence_ids":["S8"]},{"subject":"Jun","relation":"owns","object":"Jade Drum","evidence_ids":["S13"]},{"subject":"Jun","relation":"is_in","object":"South Loft","evidence_ids":["S14"]},]}
```

Missing complete facts: [{"subject": "Vera", "relation": "owns", "object": "Copper Book", "evidence_ids": ["S1"]}, {"subject": "Soren", "relation": "owns", "object": "Ivory Vase", "evidence_ids": ["S2"]}, {"subject": "Jun", "relation": "carries", "object": "Copper Book", "evidence_ids": ["S3"], "valid_from": 9, "valid_until": 17}, {"subject": "Vera", "relation": "carries", "object": "Ivory Vase", "evidence_ids": ["S4"], "valid_from": 10, "valid_until": 16}, {"subject": "Jun", "relation": "owns", "object": "Jade Drum", "evidence_ids": ["S13"]}]

#### locations: Which directly narrated location relationships have an explicitly stated interval overlapping day 12 up to but not including day 15? Return all of them with their full evidence-stated intervals, not clipped to the question window. Exclude undated locations and beliefs.

Reference answers:

- Jun — located_in → River Room [10, 14); evidence=["S5"]
- Soren — located_in → Hill Court [14, 18); evidence=["S6"]
- Vera — located_in → South Loft [11, 16); evidence=["S7"]
- Jade Drum — located_in → Elm Dock [12, 15); evidence=["S8"]

A pre-extract/select:

Manual source-output diagnosis (not rescoring): All 14 sentences receive records; twelve bare relationships and five complete facts match. Seven explicit intervals are absent. Belief records embed entire propositions in object strings and omit holder/attitude. The frozen filter therefore cannot identify any attributed records and returns an empty belief answer. Unknown predicate/endpoint strings retain their frozen unresolved status; the manual format failure is separate.


Missing complete facts: [{"subject": "Jun", "relation": "located_in", "object": "River Room", "evidence_ids": ["S5"], "valid_from": 10, "valid_until": 14}, {"subject": "Soren", "relation": "located_in", "object": "Hill Court", "evidence_ids": ["S6"], "valid_from": 14, "valid_until": 18}, {"subject": "Vera", "relation": "located_in", "object": "South Loft", "evidence_ids": ["S7"], "valid_from": 11, "valid_until": 16}, {"subject": "Jade Drum", "relation": "located_in", "object": "Elm Dock", "evidence_ids": ["S8"], "valid_from": 12, "valid_until": 15}]

B contextual:

Manual source-output diagnosis (not rescoring): The location answer contains eleven records. All four target bare relationships are present, but only the final Jade Drum/located_in/Elm Dock record retains [12,15). Its earlier unqualified copy is retained as an error, not removed. Ownership, carrying and a later [18,20) location appear despite the time/location question; most source bounds are absent.

- Jade Drum — located_in → Elm Dock; evidence=["S8"] — qualification_or_evidence_error; missing, unsupported or wrong explicit temporal qualification
- Jun — owns → Jade Drum; evidence=["S13"] — supported_but_irrelevant
- Vera — owns → Copper Book; evidence=["S1"] — supported_but_irrelevant
- Soren — owns → Ivory Vase; evidence=["S2"] — supported_but_irrelevant
- Jun — carries → Copper Book; evidence=["S3"] — qualification_or_evidence_error; missing, unsupported or wrong explicit temporal qualification
- Vera — carries → Ivory Vase; evidence=["S4"] — qualification_or_evidence_error; missing, unsupported or wrong explicit temporal qualification
- Vera — is_in → South Loft; evidence=["S7"] — qualification_or_evidence_error; missing, unsupported or wrong explicit temporal qualification
- Jun — is_in → South Loft; evidence=["S14"] — qualification_or_evidence_error; missing, unsupported or wrong explicit temporal qualification
- Soren — is_in → Hill Court; evidence=["S6"] — qualification_or_evidence_error; missing, unsupported or wrong explicit temporal qualification
- Jun — is_in → River Room; evidence=["S5"] — qualification_or_evidence_error; missing, unsupported or wrong explicit temporal qualification
- Jade Drum — located_in → Elm Dock [12, 15); evidence=["S8"] — correct_complete_fact

Missing complete facts: [{"subject": "Jun", "relation": "located_in", "object": "River Room", "evidence_ids": ["S5"], "valid_from": 10, "valid_until": 14}, {"subject": "Soren", "relation": "located_in", "object": "Hill Court", "evidence_ids": ["S6"], "valid_from": 14, "valid_until": 18}, {"subject": "Vera", "relation": "located_in", "object": "South Loft", "evidence_ids": ["S7"], "valid_from": 11, "valid_until": 16}]

#### belief: Who explicitly believes what, and what reality is separately narrated about those same objects? Return each underlying believed relationship with its attribution and each matching narrated reality without attribution. Omit unrelated facts.

Reference answers:

- Copper Book — located_in → South Loft; holder=Soren, attitude=believes; evidence=["S9"]
- Copper Book — located_in → River Room; evidence=["S10"]
- Ivory Vase — located_in → Hill Court; holder=Jun, attitude=believes; evidence=["S11"]
- Ivory Vase — located_in → Elm Dock; evidence=["S12"]

A pre-extract/select:

Manual source-output diagnosis (not rescoring): All 14 sentences receive records; twelve bare relationships and five complete facts match. Seven explicit intervals are absent. Belief records embed entire propositions in object strings and omit holder/attitude. The frozen filter therefore cannot identify any attributed records and returns an empty belief answer. Unknown predicate/endpoint strings retain their frozen unresolved status; the manual format failure is separate.


Missing complete facts: [{"subject": "Copper Book", "relation": "located_in", "object": "South Loft", "evidence_ids": ["S9"], "holder": "Soren", "attitude": "believes"}, {"subject": "Copper Book", "relation": "located_in", "object": "River Room", "evidence_ids": ["S10"]}, {"subject": "Ivory Vase", "relation": "located_in", "object": "Hill Court", "evidence_ids": ["S11"], "holder": "Jun", "attitude": "believes"}, {"subject": "Ivory Vase", "relation": "located_in", "object": "Elm Dock", "evidence_ids": ["S12"]}]

B contextual:

Manual source-output diagnosis (not rescoring): The answer repeats fourteen facts rather than only belief/reality. Belief records include holder/attitude but still place the entire proposition in object, leaving the required underlying relationship undecomposed. Two narrated reality targets match. The first generated fact says Jun owns Copper Book while S1 explicitly says Vera owns Copper Book: this is a demonstrated unsupported endpoint substitution, independently of belief-format matching limits.

- Jun — owns → Copper Book; evidence=["S1"] — unsupported_explicit_relationship
- Soren — owns → Ivory Vase; evidence=["S2"] — supported_but_irrelevant
- Jun — carries → Copper Book; evidence=["S3"] — qualification_or_evidence_error; missing, unsupported or wrong explicit temporal qualification
- Vera — carries → Ivory Vase; evidence=["S4"] — qualification_or_evidence_error; missing, unsupported or wrong explicit temporal qualification
- Jun — is_in → River Room; evidence=["S5"] — qualification_or_evidence_error; missing, unsupported or wrong explicit temporal qualification
- Soren — is_in → Hill Court; evidence=["S6"] — qualification_or_evidence_error; missing, unsupported or wrong explicit temporal qualification
- Vera — is_in → South Loft; evidence=["S7"] — qualification_or_evidence_error; missing, unsupported or wrong explicit temporal qualification
- Jade Drum — is_in → Elm Dock; evidence=["S8"] — qualification_or_evidence_error; missing, unsupported or wrong explicit temporal qualification
- Soren — believes → Copper Book is in South Loft; holder=Soren, attitude=believes; evidence=["S9"] — unresolved_wording
- Copper Book — is_in → River Room; evidence=["S10"] — correct_complete_fact
- Jun — believes → Ivory Vase is in Hill Court; holder=Jun, attitude=believes; evidence=["S11"] — unresolved_wording
- Ivory Vase — is_in → Elm Dock; evidence=["S12"] — correct_complete_fact
- Jun — owns → Jade Drum; evidence=["S13"] — supported_but_irrelevant
- Jun — is_in → South Loft; evidence=["S14"] — qualification_or_evidence_error; missing, unsupported or wrong explicit temporal qualification

Missing complete facts: [{"subject": "Copper Book", "relation": "located_in", "object": "South Loft", "evidence_ids": ["S9"], "holder": "Soren", "attitude": "believes"}, {"subject": "Ivory Vase", "relation": "located_in", "object": "Hill Court", "evidence_ids": ["S11"], "holder": "Jun", "attitude": "believes"}]

Query-blind source collection (actual, before CPU selection):
```json
{"facts": [{"subject":"Vera","relation":"owns","object":"Copper Book","evidence_ids":["S1"]},{"subject":"Soren","relation":"owns","object":"Ivory Vase","evidence_ids":["S2"]},{"subject":"Jun","relation":"carries","object":"Copper Book","evidence_ids":["S3"]},{"subject":"Vera","relation":"carries","object":"Ivory Vase","evidence_ids":["S4"]},{"subject":"Jun","relation":"is_in","object":"River Room","evidence_ids":["S5"]},{"subject":"Soren","relation":"is_in","object":"Hill Court","evidence_ids":["S6"]},{"subject":"Vera","relation":"is_in","object":"South Loft","evidence_ids":["S7"]},{"subject":"Jade Drum","relation":"is_in","object":"Elm Dock","evidence_ids":["S8"]},{"subject":"Soren","relation":"believes","object":"Copper Book is in South Loft","evidence_ids":["S9"]},{"subject":"Copper Book","relation":"is_in","object":"River Room","evidence_ids":["S10"]},{"subject":"Jun","relation":"believes","object":"Ivory Vase is in Hill Court","evidence_ids":["S11"]},{"subject":"Ivory Vase","relation":"is_in","object":"Elm Dock","evidence_ids":["S12"]},{"subject":"Jun","relation":"owns","object":"Jade Drum","evidence_ids":["S13"]},{"subject":"Jun","relation":"is_in","object":"South Loft","evidence_ids":["S14"]}]}
```

## Unique model calls

| Call | Story/task | Completed / parseable | Finish | Tokens in/out | Request seconds |
|---|---|---|---|---|---|
| 1 | harbor/all | True / True | stop | 493/364 | 8.576 |
| 2 | orchard/all | True / True | stop | 504/353 | 6.436 |
| 3 | harbor/possession | True / False | stop | 484/243 | 4.890 |
| 4 | harbor/locations | True / True | stop | 511/130 | 3.136 |
| 5 | harbor/belief | True / True | stop | 500/351 | 6.790 |
| 6 | orchard/possession | True / False | stop | 495/243 | 4.911 |
| 7 | orchard/locations | True / True | stop | 524/279 | 5.868 |
| 8 | orchard/belief | True / True | stop | 511/371 | 6.835 |

Query-blind extraction scores before selection:

- Call 1: bare 12/14; complete qualified 5/14; valid citations 14/14.
- Call 2: bare 12/14; complete qualified 5/14; valid citations 14/14.

Complete qualified facts meet the exploratory 80% precision-and-recall criterion in 0/12 question/approach rows. Parseability, bare extraction and qualification are distinct outcomes; this small sample does not establish dependable temporal/epistemic formatting or contextual selection.

Sampled resource peaks (bytes): `{"gpu_vram_bytes": 22598909952, "process_ram_bytes": 7074017280, "project_storage_bytes": 17266616320}`.

## Execution and boundaries

Executed calls: 8/8, one attempt each; no repairs. Unique-call input/output tokens: 4022/2334. Baseline request costs shown on each of its three selections are shared, not three additional calls.

Additional allocated service time: 192.872472 seconds; cumulative: 10263.092129 seconds. Open allocation/service journals: 0/0. Stop reason: completed. Exact retained values are in the JSON table.

One 900-second envelope including loading and shutdown; all historical allocation retained. Same pinned Qwen3-8B-AWQ revision 4da05a8edb55c6046cce958586c33b61da07bb79, nonthinking, 12,288 context, seed 1988649846, temperature .7/top-p .8/top-k 20. Unconstrained JSON and a common 2,048-output allowance. Exact rendered packing and raw call timings are in the JSON table.

Selection uses relation category, explicit interval overlap without clipping, and attributed-subject matching for belief/reality. It does not fix wrong facts, infer aliases, invent qualifications or deduplicate. Bare matching ignores qualifications/citations; qualified matching requires their source-supported values. Required missing content remains a recall failure. An absent time/holder field is not a literal null or an invented unknown value.

Two fresh but deliberately clear, similarly structured stories cannot establish general reliability. No office continuity, event construction, identity merging, causal reasoning or registered ontology capability was tested. Original ladders, C0 scores, primary metrics, resource ceilings and held-out human-review gate are unchanged.

Readable side-by-side graphs: [HTML](figures/compact_story_comparison.html). Exact tables: [CSV](tables/compact_story_results.csv), [JSON](tables/compact_story_results.json).

CPU reproduction: `python -m scripts.report_compact_story --run <restricted-run-directory> --output reports`.
