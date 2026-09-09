# Simple synthetic ladder v2 — format and fresh-counterpart diagnostics

Exploratory development only. No registered ontology acceptance, held-out efficacy, model reliability or production-timing claim.

Eight requests and all references/rules were frozen before inference; one attempt per case, no adaptive repair. Unconstrained JSON, pinned Qwen3-8B-AWQ and nonthinking sampling are unchanged. V2 requires evidence_ids as supplied string lists, decomposed belief relationships with holder/attitude, and both requested interval bounds. Only predeclared endpoint aliases and original relation synonyms are permitted. Original v1 outputs and scores are unchanged.

All predictions remain in the applicable denominators, including errors and duplicates. Underlying relationship scoring ignores citations/qualifications; complete-fact scoring requires them. Required-field compliance includes conditional requirements when the underlying relationship is resolved. Unresolved wording is not accepted or automatically called false.

Parseable responses: 8/8. “Shape” checks field names and value types; “Required” also checks supported conditional fields, with unresolved cases not counted as passes. All actual outputs remain visible.

| Case | Cohort / task | Shape | Required | Citations | Underlying F1 | Complete P | Complete R | Complete F1 | Tokens in/out | Seconds |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | regression / direct | 3/3 | 3/3 | 3/3 | 1.000 | 1.000 | 1.000 | 1.000 | 262/73 | 2.610 |
| 2 | regression / temporal | 2/2 | 2/2 | 2/2 | 1.000 | 1.000 | 1.000 | 1.000 | 278/68 | 1.804 |
| 3 | original-revised / epistemic | 2/2 | 2/2 | 2/2 | 1.000 | 1.000 | 1.000 | 1.000 | 269/57 | 1.749 |
| 4 | fresh / epistemic | 2/2 | 2/2 | 2/2 | 1.000 | 1.000 | 1.000 | 1.000 | 271/60 | 1.656 |
| 5 | original-revised / office-person | 4/4 | 0/4 | 4/4 | 0.500 | 0.000 | 0.000 | 0.000 | 377/167 | 3.180 |
| 6 | original-revised / office-continuity | 2/4 | 0/4 | 4/4 | 0.000 | 0.000 | 0.000 | 0.000 | 379/179 | 3.693 |
| 7 | fresh / office-person | 4/4 | 2/4 | 4/4 | 1.000 | 0.500 | 0.500 | 0.500 | 378/242 | 4.403 |
| 8 | fresh / office-continuity | 2/4 | 2/4 | 4/4 | 0.500 | 0.500 | 0.500 | 0.500 | 380/151 | 3.356 |

Highest tested coverage meeting exploratory P/R ≥.8: belief format on original and fresh examples. This is not general reliability.

## Office comparison

Both person/office reference sets permit the same office-mediated organization. Therefore the pairs are insufficient to require an ontology-construction contrast; high factual scores alone are not construction success.

- original: frozen mechanical verdict **unresolved**.

  Manual interpretation (not rescoring): The automatic comparison is unresolved because the continuity answer uses an unlisted office-name suffix and inverse predicate. Visually both answers retain one shared office linked to occupants and responsibilities. No semantic-organization change is established.

- fresh: frozen mechanical verdict **different fact selection within same organization**.

  Manual interpretation (not rescoring): The frozen mechanical comparator labels changed directed bindings as different fact selection within the same office-mediated organization. That label is not evidence of different selected meanings: the changed edges use held_by in the inverse direction, outside the frozen equivalence rules. The evidence supports only a predicate/direction difference with organization still office-mediated; selection equivalence remains formally unresolved. No construction contrast is established.

## Original versus revised and fresh

Original frozen scores are shown without re-evaluation. V2 changes formatting and prospective normalization, so score differences are not an isolated model-capability effect.

| Task | Original v1 F1 | New revised response F1 | Fresh counterpart F1 |
|---|---|---|---|
| belief | 0.000 | 1.0 | 1.0 |
| person-office | 0.000 | 0.0 | 0.5 |
| office-continuity | 0.500 | 0.0 | 0.5 |

## Evidence, generated facts and errors

### Case 1: regression / direct

- S1: Mira carries a lantern.
- S2: Tomas owns the lantern.
- S3: Mira is in the courtyard.

Extract every explicitly stated relationship and no inferred relationships.

Actual facts:

- `{"evidence_ids": ["S1"], "object": "lantern", "relation": "carries", "subject": "Mira"}` — correct_complete_fact
- `{"evidence_ids": ["S2"], "object": "lantern", "relation": "owns", "subject": "Tomas"}` — correct_complete_fact
- `{"evidence_ids": ["S3"], "object": "courtyard", "relation": "located_in", "subject": "Mira"}` — correct_complete_fact

Qualification measurements: `{"epistemic": {"applicable_predictions": 0, "assessed_absence": 3, "correct": 0, "correct_absence": 3, "precision": null, "target_count": 0, "target_recall": null}, "temporal": {"applicable_predictions": 0, "assessed_absence": 3, "correct": 0, "correct_absence": 3, "precision": null, "target_count": 0, "target_recall": null}}`.

Missing complete facts: `[]`.

### Case 2: regression / temporal

- S1: Ada is in the atrium from day 0 up to but not including day 2.
- S2: Ada is in the library from day 2 up to but not including day 5.

State both location relationships with their explicitly given intervals.

Actual facts:

- `{"evidence_ids": ["S1"], "object": "atrium", "relation": "located_in", "subject": "Ada", "valid_from": 0, "valid_until": 2}` — correct_complete_fact
- `{"evidence_ids": ["S2"], "object": "library", "relation": "located_in", "subject": "Ada", "valid_from": 2, "valid_until": 5}` — correct_complete_fact

Qualification measurements: `{"epistemic": {"applicable_predictions": 0, "assessed_absence": 2, "correct": 0, "correct_absence": 2, "precision": null, "target_count": 0, "target_recall": null}, "temporal": {"applicable_predictions": 2, "assessed_absence": 0, "correct": 2, "correct_absence": 0, "precision": 1.0, "target_count": 2, "target_recall": 1.0}}`.

Missing complete facts: `[]`.

### Case 3: original-revised / epistemic

- S1: Rina believes that the coin is in the drawer.
- S2: In reality, the coin is in the chest.

State the believed location and the narrated real location, keeping belief separate from reality.

Actual facts:

- `{"attitude": "believes", "evidence_ids": ["S1"], "holder": "Rina", "object": "drawer", "relation": "located_in", "subject": "coin"}` — correct_complete_fact
- `{"evidence_ids": ["S2"], "object": "chest", "relation": "located_in", "subject": "coin"}` — correct_complete_fact

Qualification measurements: `{"epistemic": {"applicable_predictions": 1, "assessed_absence": 1, "correct": 1, "correct_absence": 1, "precision": 1.0, "target_count": 1, "target_recall": 1.0}, "temporal": {"applicable_predictions": 0, "assessed_absence": 2, "correct": 0, "correct_absence": 2, "precision": null, "target_count": 0, "target_recall": null}}`.

Missing complete facts: `[]`.

### Case 4: fresh / epistemic

- S1: Selin believes that the map is in the cabinet.
- S2: In reality, the map is in the satchel.

State the believed location and the narrated real location, keeping belief separate from reality.

Actual facts:

- `{"attitude": "believes", "evidence_ids": ["S1"], "holder": "Selin", "object": "cabinet", "relation": "located_in", "subject": "map"}` — correct_complete_fact
- `{"evidence_ids": ["S2"], "object": "satchel", "relation": "located_in", "subject": "map"}` — correct_complete_fact

Qualification measurements: `{"epistemic": {"applicable_predictions": 1, "assessed_absence": 1, "correct": 1, "correct_absence": 1, "precision": 1.0, "target_count": 1, "target_recall": 1.0}, "temporal": {"applicable_predictions": 0, "assessed_absence": 2, "correct": 0, "correct_absence": 2, "precision": null, "target_count": 0, "target_recall": null}}`.

Missing complete facts: `[]`.

### Case 5: original-revised / office-person

- S1: Ada holds the office of Harbor Warden from day 0 up to but not including day 3.
- S2: Bram holds the office of Harbor Warden from day 3 up to but not including day 6.
- S3: The office of Harbor Warden is responsible for inspecting boats.
- S4: The office of Harbor Warden is responsible for maintaining beacons.

Person-centered question: who held the office when, and what responsibilities did each person have through it?

Actual facts:

Manual interpretation (not rescoring): The model uses held_office, outside the frozen relation synonym list. Its literal holder intervals agree with the evidence, but those assertions do not receive a positive automated temporal score because their relationship matching is unresolved. All four assertions add believes attributions absent from the direct narration. Unsupported attribution is a real evidence-grounding failure, independent of the relation-wording limitation. The depicted responsibilities remain attached to the shared office.

- `{"attitude": "believes", "evidence_ids": ["S1"], "holder": "Ada", "object": "Harbor Warden", "relation": "held_office", "subject": "Ada", "valid_from": 0, "valid_until": 3}` — unresolved_wording_or_unsupported_endpoint
- `{"attitude": "believes", "evidence_ids": ["S2"], "holder": "Bram", "object": "Harbor Warden", "relation": "held_office", "subject": "Bram", "valid_from": 3, "valid_until": 6}` — unresolved_wording_or_unsupported_endpoint
- `{"attitude": "believes", "evidence_ids": ["S3"], "holder": "Ada", "object": "inspecting boats", "relation": "responsible_for", "subject": "Harbor Warden"}` — unmatched
  unsupported epistemic fields on this relationship; epistemic holder/attitude missing, changed or unsupported
- `{"attitude": "believes", "evidence_ids": ["S4"], "holder": "Ada", "object": "maintaining beacons", "relation": "responsible_for", "subject": "Harbor Warden"}` — unmatched
  unsupported epistemic fields on this relationship; epistemic holder/attitude missing, changed or unsupported

Qualification measurements: `{"epistemic": {"applicable_predictions": 4, "assessed_absence": 0, "correct": 0, "correct_absence": 0, "precision": 0.0, "target_count": 0, "target_recall": null}, "temporal": {"applicable_predictions": 2, "assessed_absence": 2, "correct": 0, "correct_absence": 2, "precision": 0.0, "target_count": 2, "target_recall": 0.0}}`.

Missing complete facts: `[{"evidence_ids": ["S1"], "object": "Harbor Warden", "relation": "holds_office", "subject": "Ada", "valid_from": 0, "valid_until": 3}, {"evidence_ids": ["S2"], "object": "Harbor Warden", "relation": "holds_office", "subject": "Bram", "valid_from": 3, "valid_until": 6}, {"evidence_ids": ["S3"], "object": "inspecting boats", "relation": "responsible_for", "subject": "Harbor Warden"}, {"evidence_ids": ["S4"], "object": "maintaining beacons", "relation": "responsible_for", "subject": "Harbor Warden"}]`.

### Case 6: original-revised / office-continuity

- S1: Ada holds the office of Harbor Warden from day 0 up to but not including day 3.
- S2: Bram holds the office of Harbor Warden from day 3 up to but not including day 6.
- S3: The office of Harbor Warden is responsible for inspecting boats.
- S4: The office of Harbor Warden is responsible for maintaining beacons.

Office-continuity question: represent the continuing office, its responsibilities, and the successive holders with their intervals.

Actual facts:

Manual interpretation (not rescoring): Harbor Warden office is not among the prospectively allowed office-name aliases; held_by is an inverse expression outside the frozen predicate matching rules. Neither is automatically classified as a false officeholding claim. The numerical intervals agree with the supplied text, but unresolved graph mapping prevents formal qualification credit. The model additionally invents believes attribution for officeholders and uses the invalid epistemic attitude has responsibility with the office as holder. Those epistemic fields are not supported by the narrated facts.

- `{"attitude": "believes", "evidence_ids": ["S1"], "holder": "Ada", "object": "Ada", "relation": "held_by", "subject": "Harbor Warden office", "valid_from": 0, "valid_until": 3}` — unresolved_wording_or_unsupported_endpoint
- `{"attitude": "believes", "evidence_ids": ["S2"], "holder": "Bram", "object": "Bram", "relation": "held_by", "subject": "Harbor Warden office", "valid_from": 3, "valid_until": 6}` — unresolved_wording_or_unsupported_endpoint
- `{"attitude": "has responsibility", "evidence_ids": ["S3"], "holder": "Harbor Warden office", "object": "inspecting boats", "relation": "responsible_for", "subject": "Harbor Warden office"}` — unresolved_wording_or_unsupported_endpoint
  this diagnostic's attributed attitude is believes
- `{"attitude": "has responsibility", "evidence_ids": ["S4"], "holder": "Harbor Warden office", "object": "maintaining beacons", "relation": "responsible_for", "subject": "Harbor Warden office"}` — unresolved_wording_or_unsupported_endpoint
  this diagnostic's attributed attitude is believes

Qualification measurements: `{"epistemic": {"applicable_predictions": 4, "assessed_absence": 0, "correct": 0, "correct_absence": 0, "precision": 0.0, "target_count": 0, "target_recall": null}, "temporal": {"applicable_predictions": 2, "assessed_absence": 0, "correct": 0, "correct_absence": 0, "precision": 0.0, "target_count": 2, "target_recall": 0.0}}`.

Missing complete facts: `[{"evidence_ids": ["S1"], "object": "Harbor Warden", "relation": "holds_office", "subject": "Ada", "valid_from": 0, "valid_until": 3}, {"evidence_ids": ["S2"], "object": "Harbor Warden", "relation": "holds_office", "subject": "Bram", "valid_from": 3, "valid_until": 6}, {"evidence_ids": ["S3"], "object": "inspecting boats", "relation": "responsible_for", "subject": "Harbor Warden"}, {"evidence_ids": ["S4"], "object": "maintaining beacons", "relation": "responsible_for", "subject": "Harbor Warden"}]`.

### Case 7: fresh / office-person

- S1: Celia holds the office of Orchard Steward from day 2 up to but not including day 5.
- S2: Dax holds the office of Orchard Steward from day 5 up to but not including day 9.
- S3: The office of Orchard Steward is responsible for checking irrigation.
- S4: The office of Orchard Steward is responsible for recording harvests.

Person-centered question: who held the office when, and what responsibilities did each person have through it?

Actual facts:

Manual interpretation (not rescoring): All underlying bindings, supplied citation IDs and explicit holder intervals match the fresh evidence under the frozen evaluator. Both officeholding assertions nevertheless add unsupported believes attribution. The two direct office-responsibility assertions omit epistemic fields correctly and are complete matches. This is an epistemic-scope failure, not a wrong-date failure.

- `{"attitude": "believes", "evidence_ids": ["S1"], "holder": "Celia", "object": "Orchard Steward", "relation": "holds", "subject": "Celia", "valid_from": 2, "valid_until": 5}` — unmatched
  unsupported epistemic fields on this relationship; epistemic holder/attitude missing, changed or unsupported
- `{"attitude": "believes", "evidence_ids": ["S2"], "holder": "Dax", "object": "Orchard Steward", "relation": "holds", "subject": "Dax", "valid_from": 5, "valid_until": 9}` — unmatched
  unsupported epistemic fields on this relationship; epistemic holder/attitude missing, changed or unsupported
- `{"evidence_ids": ["S3"], "object": "checking irrigation", "relation": "responsible_for", "subject": "Orchard Steward"}` — correct_complete_fact
- `{"evidence_ids": ["S4"], "object": "recording harvests", "relation": "responsible_for", "subject": "Orchard Steward"}` — correct_complete_fact

Qualification measurements: `{"epistemic": {"applicable_predictions": 2, "assessed_absence": 2, "correct": 0, "correct_absence": 2, "precision": 0.0, "target_count": 0, "target_recall": null}, "temporal": {"applicable_predictions": 2, "assessed_absence": 2, "correct": 2, "correct_absence": 2, "precision": 1.0, "target_count": 2, "target_recall": 1.0}}`.

Missing complete facts: `[{"evidence_ids": ["S1"], "object": "Orchard Steward", "relation": "holds_office", "subject": "Celia", "valid_from": 2, "valid_until": 5}, {"evidence_ids": ["S2"], "object": "Orchard Steward", "relation": "holds_office", "subject": "Dax", "valid_from": 5, "valid_until": 9}]`.

### Case 8: fresh / office-continuity

- S1: Celia holds the office of Orchard Steward from day 2 up to but not including day 5.
- S2: Dax holds the office of Orchard Steward from day 5 up to but not including day 9.
- S3: The office of Orchard Steward is responsible for checking irrigation.
- S4: The office of Orchard Steward is responsible for recording harvests.

Office-continuity question: represent the continuing office, its responsibilities, and the successive holders with their intervals.

Actual facts:

Manual interpretation (not rescoring): Officeholder assertions use inverse held_by wording, which remains unresolved by the frozen matcher; their written dates match the evidence. They also populate holder and attitude=holds, using epistemic fields for office occupancy rather than supported attribution. The two office-responsibility facts are complete matches. No inverse translation was added after seeing this answer.

- `{"attitude": "holds", "evidence_ids": ["S1"], "holder": "Celia", "object": "Celia", "relation": "held_by", "subject": "Orchard Steward", "valid_from": 2, "valid_until": 5}` — unresolved_wording_or_unsupported_endpoint
  this diagnostic's attributed attitude is believes
- `{"attitude": "holds", "evidence_ids": ["S2"], "holder": "Dax", "object": "Dax", "relation": "held_by", "subject": "Orchard Steward", "valid_from": 5, "valid_until": 9}` — unresolved_wording_or_unsupported_endpoint
  this diagnostic's attributed attitude is believes
- `{"evidence_ids": ["S3"], "object": "checking irrigation", "relation": "responsible_for", "subject": "Orchard Steward"}` — correct_complete_fact
- `{"evidence_ids": ["S4"], "object": "recording harvests", "relation": "responsible_for", "subject": "Orchard Steward"}` — correct_complete_fact

Qualification measurements: `{"epistemic": {"applicable_predictions": 2, "assessed_absence": 2, "correct": 0, "correct_absence": 2, "precision": 0.0, "target_count": 0, "target_recall": null}, "temporal": {"applicable_predictions": 2, "assessed_absence": 2, "correct": 0, "correct_absence": 2, "precision": 0.0, "target_count": 2, "target_recall": 0.0}}`.

Missing complete facts: `[{"evidence_ids": ["S1"], "object": "Orchard Steward", "relation": "holds_office", "subject": "Celia", "valid_from": 2, "valid_until": 5}, {"evidence_ids": ["S2"], "object": "Orchard Steward", "relation": "holds_office", "subject": "Dax", "valid_from": 5, "valid_until": 9}]`.

## Allocation and limits

New allocated GPU time **171.371713 s**; cumulative **10070.219657 s**. Open allocation/service journals: 0/0. Stop: `completed`. Unexecuted cases: none.

All service time is charged; request timings are not production p95. No benchmark/registered-acceptance change, C0 change or held-out execution. Full raw outputs, references, errors and qualification details appear in the HTML and JSON.

[Readable graphs and complete records](figures/simple_synthetic_v2_comparison.html).

Specific remaining limitation: The clarified belief format works on the revised original and fresh belief passages, but it does not prevent misuse of epistemic fields in office narratives. Some office wording is outside the frozen matcher's coverage. The office question/reference pairs also permit identical organization, so this batch cannot establish the intended construction contrast.
