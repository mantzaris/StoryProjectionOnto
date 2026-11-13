# Author review: focused decisions before submission

This sheet requests your judgment on existing Codex-authored interpretations. It is **not completed human review**, does not change frozen labels or scores, and does not block the editorial manuscript. Priorities reflect the claims emphasized in the paper, not new output selection or evaluation.

Read the supplied evidence only; do not use later plot knowledge to resolve a participant. Evidence below preserves wording and punctuation with whitespace reflow only. JSON records preserve all field values, including the misspelled key. Slash paths are zero-based JSON locations in the linked retained artifact; the original raw responses and full question-specific reference alternatives remain there.

For each priority, record **retain / propose revision / unresolved**, a short rationale tied to evidence, and your name/date. Any approved annotation change must receive a new version and separately derived scores; the current report, figures, parser failures, and original annotations must remain identifiable.

## Priority 1 — Alice: preserved thought, uncertain consideration, wrong reader

Exact supplied evidence (the complete excerpt; [readable results](../reports/REAL_TEXT_PROOF_OF_CONCEPT.md#L406)):

<!-- retained-evidence /stories/alice/evidence/S1 -->

S1:

> Alice was beginning to get very tired of sitting by her sister on the bank, and of having nothing to do: once or twice she had peeped into the book her sister was reading, but it had no pictures or conversations in it, “and what is the use of a book,” thought Alice “without pictures or conversations?”

<!-- retained-evidence /stories/alice/evidence/S2 -->

S2:

> So she was considering in her own mind (as well as she could, for the hot day made her feel very sleepy and stupid), whether the pleasure of making a daisy-chain would be worth the trouble of getting up and picking the daisies, when suddenly a White Rabbit with pink eyes ran close by her.

### Rhetorical thought

Retained record `/calls/6/parsed/facts/1` ([retained responses and references](../reports/tables/real_text_proof_of_concept.json)):

<!-- retained-fact /calls/6/parsed/facts/1 -->

```json
{"subject":"Alice", "relation":"thinks",
 "object":"use of a book without pictures or conversations is questionable", "evidence_ids":["S1"],
 "valid_from":null, "valid_until":null,
 "holder":null, "attitude":null}
```

Current assessment: **supported**; qualifications **correct**.

<!-- retained-assessment /calls/6/parsed/facts/1 -->

> The subject Alice, mental predicate thinks, and object questioning the use of a pictureless/conversationless book preserve her rhetorical thought from S1. It is not asserted as objective uselessness. This semantically supported mental-act representation remains a strict wording mismatch.

Reference location: `/references/7/0/2` in the [retained responses and references](../reports/tables/real_text_proof_of_concept.json), expressing Alice questioning the usefulness of a book without pictures or conversations.

Decision requested: confirm whether the literal mental-act record preserves the rhetorical thought despite the strict wording/decomposition mismatch. The uncertainty is about equivalence of representation, not whether the text states the thought. The frozen semantic assessment calls its qualifications correct; the manuscript's format criticism must not be read as an additional attribution-error score.

### Daisy-chain statement

Retained record `/calls/6/parsed/facts/2` ([retained responses and references](../reports/tables/real_text_proof_of_concept.json)):

<!-- retained-fact /calls/6/parsed/facts/2 -->

```json
{"subject":"making a daisy-chain", "relation":"considered",
 "object":"worth the trouble of getting up and picking the daisies", "evidence_ids":["S2"],
 "valid_from":null, "valid_until":null,
 "holder":"Alice", "attitude":null}
```

Current assessment: **unresolved**; qualifications **incorrect**.

<!-- retained-assessment /calls/6/parsed/facts/2 -->

> S2 says Alice is considering WHETHER making the chain would be worth the trouble. 'considered worth' may instead mean judged worthwhile. The missing attitude alongside holder Alice also violates the compact pairing rule. Source support for a settled positive judgment cannot be established; do not award the consideration target by guessing.

Reference location: `/references/7/0/3`, Alice considers making a daisy-chain.

Decision requested: retain unresolved meaning, or justify another interpretation from S2. Does “considered … worth” preserve an open question, or assert a positive judgment? Assess this separately from the established null-attitude/non-null-holder defect.

### Participant-error contrast

Retained record `/calls/5/parsed/facts/3` ([retained responses and references](../reports/tables/real_text_proof_of_concept.json)):

<!-- retained-fact /calls/5/parsed/facts/3 -->

```json
{"subject":"Alice", "relation":"reading",
 "object":"the book", "evidence_ids":["S1"],
 "valid_from":null, "valid_until":null,
 "holder":null, "attitude":null}
```

Current assessment: **unsupported**; qualifications **correct**.

<!-- retained-assessment /calls/5/parsed/facts/3 -->

> S1 explicitly assigns reading to the sister and peeping to Alice. It does not support Alice reading the book. This is an unsupported participant/action assignment, not proof that Alice never read anything.

Reference location: `/references/6/0/2`, the sister reads; `/references/6/0/3`, Alice peeps.

Decision requested: confirm that the paper correctly distinguishes this unsupported participant assignment from the thought's supported meaning with imperfect representation. No claim that Alice never reads is intended.

Author decision / rationale / name / date: **Pending.**

## Priority 2 — Holmes: local referents versus attribution

Exact supplied evidence (the complete excerpt; [readable results](../reports/REAL_TEXT_PROOF_OF_CONCEPT.md#L646)):

<!-- retained-evidence /stories/holmes/evidence/S1 -->

S1:

> I had called upon my friend, Mr. Sherlock Holmes, one day in the autumn of last year and found him in deep conversation with a very stout, florid-faced, elderly gentleman with fiery red hair.

<!-- retained-evidence /stories/holmes/evidence/S2 -->

S2:

> With an apology for my intrusion, I was about to withdraw when Holmes pulled me abruptly into the room and closed the door behind me.

<!-- retained-evidence /stories/holmes/evidence/S3 -->

S3:

> “You could not possibly have come at a better time, my dear Watson,” he said cordially.

<!-- retained-evidence /stories/holmes/evidence/S4 -->

S4:

> “I was afraid that you were engaged.”

<!-- retained-evidence /stories/holmes/evidence/S5 -->

S5:

> “So I am. Very much so.”

<!-- retained-evidence /stories/holmes/evidence/S6 -->

S6:

> “Then I can wait in the next room.”

<!-- retained-evidence /stories/holmes/evidence/S7 -->

S7:

> “Not at all. This gentleman, Mr. Wilson, has been my partner and helper in many of my most successful cases, and I have no doubt that he will be of the utmost use to me in yours also.”

<!-- retained-evidence /stories/holmes/evidence/S8 -->

S8:

> The stout gentleman half rose from his chair and gave a bob of greeting, with a quick little questioning glance from his small fat-encircled eyes.

### Vocative and participant references

Retained record `/calls/8/parsed/facts/1` ([retained responses and references](../reports/tables/real_text_proof_of_concept.json)):

<!-- retained-fact /calls/8/parsed/facts/1 -->

```json
{"subject":"Holmes", "relation":"has_past_partnership",
 "object":"with Mr. Wilson", "evidence_ids":["S7"],
 "valid_from":null, "valid_until":null,
 "holder":null, "attitude":null}
```

Current assessment: **unresolved**; qualifications **incorrect**.

<!-- retained-assessment /calls/8/parsed/facts/1 -->

> The frozen contextual reading of S7 is Holmes addressing Wilson while describing Watson as his past partner. This answer instead names Wilson. The comma/vocative wording is locally ambiguous, so identity is unresolved rather than declared a proven contradiction. Independently, it presents the reported partnership as a direct fact with null attribution.

Retained record `/calls/8/parsed/facts/2` ([retained responses and references](../reports/tables/real_text_proof_of_concept.json)):

<!-- retained-fact /calls/8/parsed/facts/2 -->

```json
{"subject":"Holmes", "relation":"expects_help_from",
 "object":"Mr. Wilson", "evidence_ids":["S7"],
 "valid_from":null, "valid_until":null,
 "holder":null, "attitude":null}
```

Current assessment: **unresolved**; qualifications **correct**.

<!-- retained-assessment /calls/8/parsed/facts/2 -->

> S7 describes expected usefulness to Holmes; the frozen reading assigns the expected helper to Watson, with Wilson addressed. The answer names Wilson instead. Local vocative ambiguity prevents positive participant credit. 'Holmes expects' does retain an explicitly holder-relative mental act, not accomplished help, despite null holder fields. No target coverage is awarded for the uncertain referent.

Reference locations: `/references/9/0/2` through `/references/9/0/4`: the frozen contextual reading treats Wilson as addressee and Watson as the described partner/helper and expected aid.

Decision requested: does the excerpt sufficiently establish that reading, or should referent identity remain unresolved? In particular, distinguish an appositive reading of “This gentleman, Mr. Wilson” from a vocative. Review the partnership's missing reported scope separately from the expectation's retained mental-act meaning; the latter currently has correct semantic qualifications despite unresolved identity.

### Added participant and speaker assignment

Retained record `/calls/8/parsed/facts/0` ([retained responses and references](../reports/tables/real_text_proof_of_concept.json)):

<!-- retained-fact /calls/8/parsed/facts/0 -->

```json
{"subject":"Holmes", "relation":"is_engaged",
 "object":"with Watson", "evidence_ids":["S4","S5"],
 "valid_from":null, "valid_until":null,
 "holder":null, "attitude":null}
```

Current assessment: **partial**; qualifications **incorrect**.

<!-- retained-assessment /calls/8/parsed/facts/0 -->

> S4-S5 contain Watson's concern and Holmes's confirmation of being engaged. They do not say engaged with Watson. That added participant is unsupported, and null holder/attitude loses the dialogue attribution. The busy-state component is recognizable but no complete scoped target is preserved.

Reference locations: `/references/9/0/0` and `/references/9/0/1` distinguish Watson's concern from Holmes's reply.

The following are **literal fragments of an unparseable query-blind response**, not recovered graph records. The complete call ends with an extra brace and remains a strict-parser failure. Source: `/calls/2/raw_text`; existing diagnosis: `/manual_diagnosis/cases/3`.

<!-- retained-raw /calls/2/raw_text -->

```text
{"subject":"I","relation":"said","object":"So I am. Very much so.","evidence_ids":["S5"],"valid_from":null,"valid_until":null,"holder":null,"attitude":null}
```

<!-- retained-raw /calls/2/raw_text -->

```text
{"subject":"Mr. Sherlock Holmes","relation":"said","object":"Then I can wait in the next room","evidence_ids":["S6"],"valid_from":null,"valid_until":null,"holder":null,"attitude":null}
```

Existing manual diagnosis, not graph scoring:

> The raw 'I said So I am. Very much so.' reverses the speaker of the S5 reply: S3-S6 dialogue assigns that reply to Holmes. Conversely the offer to wait from S6 is labelled as Holmes speaking. These are manual speaker errors visible before the syntax failure, not full graph evaluation.

Decision requested: confirm the local dialogue turn assignment and whether the current partial-credit description of the busy-state record is appropriate while retaining its unsupported “with Watson” addition and lost attribution. Do not conflate the stronger speaker-error finding with the unresolved S7 vocative.

Author decision / rationale / name / date: **Pending.**

## Priority 3 — Fable: conditional plea and partial action meaning

Exact supplied evidence relevant to these records, including the subsequent release for contrast ([complete fable and results](../reports/REAL_TEXT_PROOF_OF_CONCEPT.md#L74)):

<!-- retained-evidence /stories/fable/evidence/S1 -->

S1:

> A LION was awakened from sleep by a Mouse running over his face.

<!-- retained-evidence /stories/fable/evidence/S2 -->

S2:

> Rising up angrily, he caught him and was about to kill him, when the Mouse piteously entreated, saying: “If you would only spare my life, I would be sure to repay your kindness.”

<!-- retained-evidence /stories/fable/evidence/S3 -->

S3:

> The Lion laughed and let him go.

### Conditional plea

Retained record `/calls/0/parsed/facts/3` ([retained responses and references](../reports/tables/real_text_proof_of_concept.json)):

<!-- retained-fact /calls/0/parsed/facts/3 -->

```json
{"subject":"A LION", "relation":"spare",
 "object":"a Mouse", "evidence_ids":["S2"],
 "valid_from":null, "valid until":null,
 "holder":null, "attitude":null}
```

Current assessment: **partial**; qualifications **incorrect**.

<!-- retained-assessment /calls/0/parsed/facts/3 -->

> S2 contains a conditional request to spare, not the accomplished release (which is in S3). The answer asserts spare without the request/conditional scope and cites S2. It also spells valid_until as 'valid until'. No repair or inferred bound is applied.

Reference locations: `/references/5/0/2` (Mouse's conditional promise) and `/references/4/0/3` (the narrated release at S3). The displayed A action is selected from this actual query-blind record; it is not a new generation.

Decision requested: confirm whether “partial” is an appropriate semantic description of the unqualified spare claim, or propose a separately justified label. The uncertainty concerns how much meaning survives the loss of request/conditional scope. S3 must not be substituted for the model's S2 citation, and the misspelled field must not be silently fixed.

### Coarser action endpoint

Retained record `/calls/3/parsed/facts/0` ([retained responses and references](../reports/tables/real_text_proof_of_concept.json)):

<!-- retained-fact /calls/3/parsed/facts/0 -->

```json
{"subject":"Mouse", "relation":"running_over",
 "object":"Lion", "evidence_ids":["S1"],
 "valid_from":null, "valid_until":null,
 "holder":null, "attitude":null}
```

Current assessment: **partial**; qualifications **correct**.

<!-- retained-assessment /calls/3/parsed/facts/0 -->

> S1 says running over the Lion's face. Running over Lion retains the actor and broad action but loses the explicit body-part endpoint. This receives partial support, not full coverage of the frozen face-specific target.

Reference location: `/references/4/0/0`, running over the Lion's face.

Decision requested: confirm partial broad-action preservation with no full face-specific target credit, or explain a different reading. This is an omitted specificity judgment, distinct from the conditional-scope loss above.

Author decision / rationale / name / date: **Pending.**

## Publication details — single unresolved checklist

- Confirm the publication name and affiliation/contact information. The only established identity used is the repository's `a.v.mantzaris` copyright name; no affiliation or coauthor is inferred.
- Supply funding and conflict-of-interest declarations, and approve the manuscript's AI-assistance disclosure.
- Choose a publication venue when ready; confirm its excerpt-attribution/rights and submission requirements for the retained Gutenberg editions. No venue is currently assumed.

All publication and semantic decisions remain pending; none has been invented or applied by this editorial revision. This editable sheet is an input to the manuscript manifest and is **not overwritten by regeneration**.
