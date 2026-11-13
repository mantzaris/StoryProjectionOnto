# Published narrative prose: compact extraction proof of concept

Exploratory extraction, not factual verification of real events, registered C1/C2 acceptance, or ontology-construction efficacy. A = query-blind extraction then fixed CPU selection; B = direct contextual extraction. References and manual assessments are Codex-authored, not independent human review.

Executed 9/9 calls: 7 usable responses, 7 strict JSON successes, 0 syntax recoveries. Total request tokens: 6802 input / 2927 output. All received outputs and failed raw responses are retained below.

## Actual results

| Passage / question | Method | Strict qualified P/R/F1 | Supported / partial / unresolved records | Semantic target coverage |
|---|---|---|---|---|
| alice/actions | A | 0.500/0.200/0.286 | 2/0/0 of 2 | 3/5 |
| alice/actions | B | 0.400/0.400/0.400 | 3/1/0 of 5 | 3/5 |
| alice/claims | A | 0.000/0.000/0.000 | 2/2/0 of 4 | 1/4 |
| alice/claims | B | 0.000/0.000/0.000 | 2/0/1 of 3 | 3/4 |
| fable/actions | A | 0.625/0.625/0.625 | 6/1/1 of 8 | 6/8 |
| fable/actions | B | 0.400/0.250/0.308 | 4/1/0 of 5 | 4/8 |
| fable/claims | A | 0.000/0.000/0.000 | 1/0/0 of 1 | 2/5 |
| fable/claims | B | 0.000/0.000/0.000 | 1/3/0 of 4 | 0/5 |
| holmes/actions | A | not measurable | 0/0/0 of unavailable | unavailable |
| holmes/actions | B | not measurable | 0/0/0 of unavailable | unavailable |
| holmes/claims | A | not measurable | 0/0/0 of unavailable | unavailable |
| holmes/claims | B | 0.000/0.000/0.000 | 0/1/2 of 3 | 0/5 |

Useful graphs were obtained, but contextual extraction did not consistently improve the comparison. On the fable, fixed selection retained more of the physical-action targets; the contextual answer omitted the Lion's capture and release of the Mouse. Passive capture/release wording was semantically sound in several records even though the frozen strict matcher did not recognize it.

Alice shows the clearest useful contextual result: the answer preserved the book's absent pictures and conversations and Alice's rhetorical thought. A combined object preserved two reference concepts, so semantic coverage is higher than strict tuple agreement. Its daisy-chain answer remained ambiguous between considering a possibility and judging it worthwhile, and supplied a holder without an attitude. The action answer assigned reading to Alice instead of the sister identified in the text. These errors remain visible.

The Holmes comparison was substantially weaker. Both the query-blind collection and the contextual action response contain an extra terminal closing brace; neither is repaired under the trailing-comma-only rule. The usable dialogue graph added an unsupported 'with Watson' qualifier to being engaged and did not reliably preserve speaker attribution. Its Wilson-versus-Watson reading conflicts with the frozen contextual interpretation of the vocative dialogue, but that local ambiguity is reported as unresolved rather than a demonstrated contradiction. No full target meaning was credited by guessing the intended participant.

Fixed selection has its own visible weaknesses: the frozen fable subject filter rejects 'A LION' because it requires 'Lion', thereby losing existing capture/release facts. Alice's useful sitting/location content is embedded inside a feeling record that the action filter excludes. These unchanged filters are an exploratory baseline, not a credible stand-in for registered C1 or evidence that contextual ontology construction is superior.

The strongest supported conclusion is limited: this configuration can produce inspectable physical-action and book-content information from these published excerpts. It does not reliably preserve dialogue roles, conditional promises, compact attribution fields, or JSON syntax on all passages. Scope-limited strict scores and separately labelled Codex semantic assessments are both necessary to understand these outputs. Familiar texts, a tiny purposive sample, and non-independent references preclude general reliability claims.



Strict scores measure agreement with the frozen scoped annotation, not a rate of factual truth. Every prediction remains in its applicable denominator. Semantic review can recognize explicit meaning in another representation without changing strict scores; partial support and ambiguous wording are not full correctness. Coverage counts unique explicitly preserved target concepts, not a new reference set.
Failed parsing is a missing result, not a zero-precision empty graph. Table metrics are unavailable in that case; immutable raw call evaluations retain their original empty-input accounting. No malformed output is repaired beyond the frozen trailing-comma rule. Null bounds mean unsupported/unstated duration, not timelessness or falsehood.

## Format, citations and errors

| Passage/question | A/B | Strict / usable JSON | Format / citations / predictions | Bare relationship P/R/F1 | Unsupported / contradicted / qualification errors | Supported irrelevant |
|---|---|---|---|---|---|---|
| alice/actions | A | True/True | 2/2/2 | 0.500/0.200/0.286 | 0/0/0 | 0 |
| alice/actions | B | True/True | 5/5/5 | 0.400/0.400/0.400 | 1/0/0 | 0 |
| alice/claims | A | True/True | 4/4/4 | 0.000/0.000/0.000 | 0/0/0 | 0 |
| alice/claims | B | True/True | 2/3/3 | 0.000/0.000/0.000 | 0/0/1 | 0 |
| fable/actions | A | True/True | 7/8/8 | 0.625/0.625/0.625 | 0/0/1 | 0 |
| fable/actions | B | True/True | 5/5/5 | 0.400/0.250/0.308 | 0/0/0 | 0 |
| fable/claims | A | True/True | 1/1/1 | 0.000/0.000/0.000 | 0/0/0 | 0 |
| fable/claims | B | True/True | 4/4/4 | 0.000/0.000/0.000 | 0/0/2 | 0 |
| holmes/actions | A | False/False | 0/0/None | not measurable | 0/0/0 | 0 |
| holmes/actions | B | False/False | 0/0/None | not measurable | 0/0/0 | 0 |
| holmes/claims | A | False/False | 0/0/None | not measurable | 0/0/0 | 0 |
| holmes/claims | B | True/True | 3/3/3 | 0.000/0.000/0.000 | 0/0/2 | 0 |

### Failed-response manual diagnosis — call 3

Finish stop; output ends ]}}. The parser reports extra data at char 2531. Removing a closing brace is outside the authorized trailing-comma recovery; no records are imported into the evaluated graph.

- Visible raw fields contain the visit, pulling and door-closing actions, but are not promoted to a parsed result.
- The raw 'I said So I am. Very much so.' reverses the speaker of the S5 reply: S3-S6 dialogue assigns that reply to Holmes. Conversely the offer to wait from S6 is labelled as Holmes speaking. These are manual speaker errors visible before the syntax failure, not full graph evaluation.
- One visible field is 'valid until' instead of valid_until. The raw half-rose relation combines an action and a greeting in an object string, rather than explicit participant endpoints. The full generated text remains available for inspection.


### Failed-response manual diagnosis — call 8

Finish stop; output ends ]}}. The parser reports extra data at char 1682. This is not a trailing comma, a token-cap truncation or an authorized recovery candidate.

- The visible pulling and closing records have the narrated Holmes-to-Watson and Holmes-to-door roles. They are displayed as raw text, not counted as a completed graph.
- S8 describes the stout gentleman making a greeting; it does not support the generated Holmes greets elderly_gentleman. S3 describes Holmes addressing Watson, not the generated Watson greets Holmes. The gentleman's greeting recipient itself is contextually ambiguous, so elderly_gentleman greets Holmes is not labelled a proven contradiction solely because the frozen reference uses Watson.
- Several raw said-to records fall outside the narrowly listed visit/conversation/pulling/closing/rising/greeting action targets. No records were selected from this parser failure to improve its score.


## Three hundred Aesop’s fables — The Lion And The Mouse

Aesop; translated by George Fyler Townsend. [Source](https://www.gutenberg.org/ebooks/21); retrieved 2026-09-09T13:42:39.348162+00:00. Excerpt SHA-256 `340385bf6743191af33313e4998e9e16987d7f92d658417861516ce220b25fb0`; 133 words.

Exact contiguous excerpt (original line breaks retained):
```text
A LION was awakened from sleep by a Mouse running over his face. Rising
up angrily, he caught him and was about to kill him, when the Mouse
piteously entreated, saying: “If you would only spare my life, I would
be sure to repay your kindness.” The Lion laughed and let him go. It
happened shortly after this that the Lion was caught by some hunters,
who bound him by strong ropes to the ground. The Mouse, recognizing his
roar, came and gnawed the rope with his teeth, and set him free,
exclaiming:

“You ridiculed the idea of my ever being able to help you, not
expecting to receive from me any repayment of your favour; now you know
that it is possible for even a Mouse to confer benefits on a Lion.”

```

[Source attribution and full supplied license notices](../data/published_prose/pg21_notices.txt). Project Gutenberg lists this edition as public domain in the USA; original attribution and complete supplied license notices retained. Check local law outside the USA.

Evidence spans (numbers are added labels; concatenation reproduces the exact excerpt):

**S1**

```text
A LION was awakened from sleep by a Mouse running over his face.
```

**S2**

```text
 Rising
up angrily, he caught him and was about to kill him, when the Mouse
piteously entreated, saying: “If you would only spare my life, I would
be sure to repay your kindness.”
```

**S3**

```text
 The Lion laughed and let him go.
```

**S4**

```text
 It
happened shortly after this that the Lion was caught by some hunters,
who bound him by strong ropes to the ground.
```

**S5**

```text
 The Mouse, recognizing his
roar, came and gnawed the rope with his teeth, and set him free,
exclaiming:
```

**S6**

```text


“You ridiculed the idea of my ever being able to help you, not
expecting to receive from me any repayment of your favour; now you know
that it is possible for even a Mouse to confer benefits on a Lion.”

```

### Which physical running/waking, capture, binding, rope-gnawing and release relationships involve the Lion, Mouse, hunters and ropes? Exclude dialogue, intentions, laughter and the moral.

Codex-authored reference facts (alternatives in canonical JSON frozen rules):
```json
[
  {
    "subject": "Mouse",
    "relation": "runs_over",
    "object": "Lion's face",
    "evidence_ids": [
      "S1"
    ],
    "valid_from": null,
    "valid_until": null,
    "holder": null,
    "attitude": null
  },
  {
    "subject": "Mouse",
    "relation": "awakens",
    "object": "Lion",
    "evidence_ids": [
      "S1"
    ],
    "valid_from": null,
    "valid_until": null,
    "holder": null,
    "attitude": null
  },
  {
    "subject": "Lion",
    "relation": "catches",
    "object": "Mouse",
    "evidence_ids": [
      "S2"
    ],
    "valid_from": null,
    "valid_until": null,
    "holder": null,
    "attitude": null
  },
  {
    "subject": "Lion",
    "relation": "releases",
    "object": "Mouse",
    "evidence_ids": [
      "S3"
    ],
    "valid_from": null,
    "valid_until": null,
    "holder": null,
    "attitude": null
  },
  {
    "subject": "hunters",
    "relation": "catches",
    "object": "Lion",
    "evidence_ids": [
      "S4"
    ],
    "valid_from": null,
    "valid_until": null,
    "holder": null,
    "attitude": null
  },
  {
    "subject": "Lion",
    "relation": "bound_by",
    "object": "ropes",
    "evidence_ids": [
      "S4"
    ],
    "valid_from": null,
    "valid_until": null,
    "holder": null,
    "attitude": null
  },
  {
    "subject": "Mouse",
    "relation": "gnaws",
    "object": "ropes",
    "evidence_ids": [
      "S5"
    ],
    "valid_from": null,
    "valid_until": null,
    "holder": null,
    "attitude": null
  },
  {
    "subject": "Mouse",
    "relation": "releases",
    "object": "Lion",
    "evidence_ids": [
      "S5"
    ],
    "valid_from": null,
    "valid_until": null,
    "holder": null,
    "attitude": null
  }
]
```

#### A pre-extract/select

Actual model-authored records and source-based assessment:

1. `{"attitude": null, "evidence_ids": ["S1"], "holder": null, "object": "a Mouse", "relation": "awakened_from_sleep", "subject": "A LION", "valid_from": null, "valid_until": null}` — unresolved; qualifiers unresolved; relevant True. S1 supports the Mouse waking the Lion. The predicate awakened_from_sleep lacks 'by', leaving the Mouse object's role ambiguous. Do not guess active versus passive direction; no full relationship credit.

2. `{"attitude": null, "evidence_ids": ["S2"], "holder": null, "object": "a Mouse", "relation": "caught", "subject": "A LION", "valid_from": null, "valid_until": null}` — supported; qualifiers correct; relevant True. Frozen reference match, not independent verification.

3. `{"attitude": null, "evidence_ids": ["S2"], "holder": null, "object": "a Mouse", "relation": "spare", "subject": "A LION", "valid until": null, "valid_from": null}` — partial; qualifiers incorrect; relevant True. S2 contains a conditional request to spare, not the accomplished release (which is in S3). The answer asserts spare without the request/conditional scope and cites S2. It also spells valid_until as 'valid until'. No repair or inferred bound is applied.

4. `{"attitude": null, "evidence_ids": ["S3"], "holder": null, "object": "a Mouse", "relation": "let_go", "subject": "A LION", "valid_from": null, "valid_until": null}` — supported; qualifiers correct; relevant True. Frozen reference match, not independent verification.

5. `{"attitude": null, "evidence_ids": ["S4"], "holder": null, "object": "some hunters", "relation": "caught_by", "subject": "A LION", "valid_from": null, "valid_until": null}` — supported; qualifiers correct; relevant True. S4 explicitly says the Lion was caught by hunters. This passive relation preserves participants and direction; frozen strict synonyms do not include this inverse form. No graph rewriting was performed.

6. `{"attitude": null, "evidence_ids": ["S4"], "holder": null, "object": "strong ropes", "relation": "bound_by", "subject": "A LION", "valid_from": null, "valid_until": null}` — supported; qualifiers correct; relevant True. Frozen reference match, not independent verification.

7. `{"attitude": null, "evidence_ids": ["S5"], "holder": null, "object": "the rope", "relation": "gnawed", "subject": "a Mouse", "valid_from": null, "valid_until": null}` — supported; qualifiers correct; relevant True. Frozen reference match, not independent verification.

8. `{"attitude": null, "evidence_ids": ["S5"], "holder": null, "object": "A LION", "relation": "set_free", "subject": "a Mouse", "valid_from": null, "valid_until": null}` — supported; qualifiers correct; relevant True. Frozen reference match, not independent verification.

Missing strict targets: `[{"subject": "Mouse", "relation": "runs_over", "object": "Lion's face", "evidence_ids": ["S1"], "valid_from": null, "valid_until": null, "holder": null, "attitude": null}, {"subject": "Mouse", "relation": "awakens", "object": "Lion", "evidence_ids": ["S1"], "valid_from": null, "valid_until": null, "holder": null, "attitude": null}, {"subject": "hunters", "relation": "catches", "object": "Lion", "evidence_ids": ["S4"], "valid_from": null, "valid_until": null, "holder": null, "attitude": null}]`.

#### B contextual

Actual model-authored records and source-based assessment:

1. `{"attitude": null, "evidence_ids": ["S1"], "holder": null, "object": "Lion", "relation": "running_over", "subject": "Mouse", "valid_from": null, "valid_until": null}` — partial; qualifiers correct; relevant True. S1 says running over the Lion's face. Running over Lion retains the actor and broad action but loses the explicit body-part endpoint. This receives partial support, not full coverage of the frozen face-specific target.

2. `{"attitude": null, "evidence_ids": ["S4"], "holder": null, "object": "hunters", "relation": "caught_by", "subject": "Lion", "valid_from": null, "valid_until": null}` — supported; qualifiers correct; relevant True. S4 supports this passive capture relation exactly in meaning. The frozen strict matcher lacks the inverse caught_by alternative; its strict failure is not a falsehood judgment.

3. `{"attitude": null, "evidence_ids": ["S4"], "holder": null, "object": "ropes", "relation": "bound_by", "subject": "Lion", "valid_from": null, "valid_until": null}` — supported; qualifiers correct; relevant True. Frozen reference match, not independent verification.

4. `{"attitude": null, "evidence_ids": ["S5"], "holder": null, "object": "ropes", "relation": "gnawed", "subject": "Mouse", "valid_from": null, "valid_until": null}` — supported; qualifiers correct; relevant True. Frozen reference match, not independent verification.

5. `{"attitude": null, "evidence_ids": ["S5"], "holder": null, "object": "Mouse", "relation": "freed_by", "subject": "Lion", "valid_from": null, "valid_until": null}` — supported; qualifiers correct; relevant True. S5 explicitly says the Mouse set the Lion free. Lion freed_by Mouse is the same directed meaning in passive form, outside the frozen strict synonym set.

Missing strict targets: `[{"subject": "Mouse", "relation": "runs_over", "object": "Lion's face", "evidence_ids": ["S1"], "valid_from": null, "valid_until": null, "holder": null, "attitude": null}, {"subject": "Mouse", "relation": "awakens", "object": "Lion", "evidence_ids": ["S1"], "valid_from": null, "valid_until": null, "holder": null, "attitude": null}, {"subject": "Lion", "relation": "catches", "object": "Mouse", "evidence_ids": ["S2"], "valid_from": null, "valid_until": null, "holder": null, "attitude": null}, {"subject": "Lion", "relation": "releases", "object": "Mouse", "evidence_ids": ["S3"], "valid_from": null, "valid_until": null, "holder": null, "attitude": null}, {"subject": "hunters", "relation": "catches", "object": "Lion", "evidence_ids": ["S4"], "valid_from": null, "valid_until": null, "holder": null, "attitude": null}, {"subject": "Mouse", "relation": "releases", "object": "Lion", "evidence_ids": ["S5"], "valid_from": null, "valid_until": null, "holder": null, "attitude": null}]`.

### How does the Lion capture and release the Mouse, and what does the Mouse explicitly promise or claim about repayment, ability to help, and the Lion's earlier ridicule? Exclude hunters, other physical actions and unspoken intentions.

Codex-authored reference facts (alternatives in canonical JSON frozen rules):
```json
[
  {
    "subject": "Lion",
    "relation": "catches",
    "object": "Mouse",
    "evidence_ids": [
      "S2"
    ],
    "valid_from": null,
    "valid_until": null,
    "holder": null,
    "attitude": null
  },
  {
    "subject": "Lion",
    "relation": "releases",
    "object": "Mouse",
    "evidence_ids": [
      "S3"
    ],
    "valid_from": null,
    "valid_until": null,
    "holder": null,
    "attitude": null
  },
  {
    "subject": "Mouse",
    "relation": "would_repay_if_spared",
    "object": "Lion",
    "evidence_ids": [
      "S2"
    ],
    "valid_from": null,
    "valid_until": null,
    "holder": "Mouse",
    "attitude": "promises"
  },
  {
    "subject": "Mouse",
    "relation": "can_help",
    "object": "Lion",
    "evidence_ids": [
      "S6"
    ],
    "valid_from": null,
    "valid_until": null,
    "holder": "Mouse",
    "attitude": "reported"
  },
  {
    "subject": "Lion",
    "relation": "ridicules_help_from",
    "object": "Mouse",
    "evidence_ids": [
      "S6"
    ],
    "valid_from": null,
    "valid_until": null,
    "holder": "Mouse",
    "attitude": "reported"
  }
]
```

#### A pre-extract/select

Actual model-authored records and source-based assessment:

1. `{"attitude": null, "evidence_ids": ["S6"], "holder": null, "object": "You ridiculed the idea of my ever being able to help you, not expecting to receive from me any repayment of your favour; now you know that it is possible for even a Mouse to confer benefits on a Lion.", "relation": "exclaiming", "subject": "a Mouse", "valid_from": null, "valid_until": null}` — supported; qualifiers correct; relevant True. The object reproduces S6 as the Mouse's exclamation. Speaker and reported content remain explicit in the subject plus speech predicate, preserving the ridicule/help claims rather than asserting them as narrator truth. This is not the requested decomposed holder/attitude fact structure; strict scores remain zero. The quote refers to possible benefits, not an observed numerical validity interval.

Missing strict targets: `[{"subject": "Lion", "relation": "catches", "object": "Mouse", "evidence_ids": ["S2"], "valid_from": null, "valid_until": null, "holder": null, "attitude": null}, {"subject": "Lion", "relation": "releases", "object": "Mouse", "evidence_ids": ["S3"], "valid_from": null, "valid_until": null, "holder": null, "attitude": null}, {"subject": "Mouse", "relation": "would_repay_if_spared", "object": "Lion", "evidence_ids": ["S2"], "valid_from": null, "valid_until": null, "holder": "Mouse", "attitude": "promises"}, {"subject": "Mouse", "relation": "can_help", "object": "Lion", "evidence_ids": ["S6"], "valid_from": null, "valid_until": null, "holder": "Mouse", "attitude": "reported"}, {"subject": "Lion", "relation": "ridicules_help_from", "object": "Mouse", "evidence_ids": ["S6"], "valid_from": null, "valid_until": null, "holder": "Mouse", "attitude": "reported"}]`.

#### B contextual

Actual model-authored records and source-based assessment:

1. `{"attitude": null, "evidence_ids": ["S2"], "holder": null, "object": "Lion", "relation": "entreated", "subject": "Mouse", "valid_from": null, "valid_until": null}` — supported; qualifiers correct; relevant True. S2 explicitly narrates the Mouse entreating the Lion. This is relevant to the requested promise/release interaction but absent from the frozen target list: a reference-coverage limitation, not an unsupported fact. No new target is added.

2. `{"attitude": null, "evidence_ids": ["S2"], "holder": null, "object": "repayment", "relation": "promise", "subject": "Mouse", "valid_from": null, "valid_until": null}` — partial; qualifiers incorrect; relevant True. S2 supports a repayment promise. The answer omits its if-spared condition and the beneficiary; it does not preserve the complete promised relationship. The word promise avoids claiming actual repayment, but cannot supply the missing condition.

3. `{"attitude": null, "evidence_ids": ["S2"], "holder": null, "object": "ability_to_help", "relation": "promise", "subject": "Mouse", "valid_from": null, "valid_until": null}` — partial; qualifiers incorrect; relevant True. S2 supports a conditional repayment promise, not a fully specified promise of ability_to_help. The answer blends that promise with the S6 ability claim, omits the beneficiary and condition, and cites only S2. No full target credit.

4. `{"attitude": null, "evidence_ids": ["S6"], "holder": null, "object": "Lion_earlier_ridicule", "relation": "claim", "subject": "Mouse", "valid_from": null, "valid_until": null}` — partial; qualifiers correct; relevant True. S6 supports a Mouse claim about the Lion's earlier ridicule. The speech-act predicate preserves attribution, but Lion_earlier_ridicule omits what was ridiculed: the Mouse's ability to help. This is not the complete frozen proposition.

Missing strict targets: `[{"subject": "Lion", "relation": "catches", "object": "Mouse", "evidence_ids": ["S2"], "valid_from": null, "valid_until": null, "holder": null, "attitude": null}, {"subject": "Lion", "relation": "releases", "object": "Mouse", "evidence_ids": ["S3"], "valid_from": null, "valid_until": null, "holder": null, "attitude": null}, {"subject": "Mouse", "relation": "would_repay_if_spared", "object": "Lion", "evidence_ids": ["S2"], "valid_from": null, "valid_until": null, "holder": "Mouse", "attitude": "promises"}, {"subject": "Mouse", "relation": "can_help", "object": "Lion", "evidence_ids": ["S6"], "valid_from": null, "valid_until": null, "holder": "Mouse", "attitude": "reported"}, {"subject": "Lion", "relation": "ridicules_help_from", "object": "Mouse", "evidence_ids": ["S6"], "valid_from": null, "valid_until": null, "holder": "Mouse", "attitude": "reported"}]`.

### Complete query-blind collection (before filtering)

1. `{"attitude": null, "evidence_ids": ["S1"], "holder": null, "object": "a Mouse", "relation": "awakened_from_sleep", "subject": "A LION", "valid_from": null, "valid_until": null}` — unresolved; S1 supports the Mouse waking the Lion. The predicate awakened_from_sleep lacks 'by', leaving the Mouse object's role ambiguous. Do not guess active versus passive direction; no full relationship credit.

2. `{"attitude": null, "evidence_ids": ["S2"], "holder": null, "object": "a Mouse", "relation": "caught", "subject": "A LION", "valid_from": null, "valid_until": null}` — supported; Frozen reference match, not independent verification.

3. `{"attitude": null, "evidence_ids": ["S2"], "holder": null, "object": "a Mouse", "relation": "about_to_kill", "subject": "A LION", "valid_from": null, "valid_until": null}` — supported; S2 explicitly says the Lion was about to kill the Mouse. The predicate retains prospective rather than completed action. Both questions exclude this intention; supported but irrelevant.

4. `{"attitude": null, "evidence_ids": ["S2"], "holder": null, "object": "a Mouse", "relation": "spare", "subject": "A LION", "valid until": null, "valid_from": null}` — partial; S2 contains a conditional request to spare, not the accomplished release (which is in S3). The answer asserts spare without the request/conditional scope and cites S2. It also spells valid_until as 'valid until'. No repair or inferred bound is applied.

5. `{"attitude": null, "evidence_ids": ["S3"], "holder": null, "object": "a Mouse", "relation": "laughed", "subject": "A LION", "valid_from": null, "valid_until": null}` — unresolved; S3 says the Lion laughed; it does not give laughter a binary Mouse object. Laughing at the Mouse is plausible, but the role of the object is underdefined. Laughter is excluded from both questions.

6. `{"attitude": null, "evidence_ids": ["S3"], "holder": null, "object": "a Mouse", "relation": "let_go", "subject": "A LION", "valid_from": null, "valid_until": null}` — supported; Frozen reference match, not independent verification.

7. `{"attitude": null, "evidence_ids": ["S4"], "holder": null, "object": "some hunters", "relation": "caught_by", "subject": "A LION", "valid_from": null, "valid_until": null}` — supported; S4 explicitly says the Lion was caught by hunters. This passive relation preserves participants and direction; frozen strict synonyms do not include this inverse form. No graph rewriting was performed.

8. `{"attitude": null, "evidence_ids": ["S4"], "holder": null, "object": "strong ropes", "relation": "bound_by", "subject": "A LION", "valid_from": null, "valid_until": null}` — supported; Frozen reference match, not independent verification.

9. `{"attitude": null, "evidence_ids": ["S5"], "holder": null, "object": "a Mouse", "relation": "recognized_by", "subject": "A LION", "valid_from": null, "valid_until": null}` — partial; S5 says the Mouse recognized the Lion's roar, not a direct binary recognition of the Lion himself. The referent is connected, but the perceived object has been generalized. Recognition is outside both question scopes.

10. `{"attitude": null, "evidence_ids": ["S5"], "holder": null, "object": "the rope", "relation": "gnawed", "subject": "a Mouse", "valid_from": null, "valid_until": null}` — supported; Frozen reference match, not independent verification.

11. `{"attitude": null, "evidence_ids": ["S5"], "holder": null, "object": "A LION", "relation": "set_free", "subject": "a Mouse", "valid_from": null, "valid_until": null}` — supported; Frozen reference match, not independent verification.

12. `{"attitude": null, "evidence_ids": ["S6"], "holder": null, "object": "You ridiculed the idea of my ever being able to help you, not expecting to receive from me any repayment of your favour; now you know that it is possible for even a Mouse to confer benefits on a Lion.", "relation": "exclaiming", "subject": "a Mouse", "valid_from": null, "valid_until": null}` — supported; The object reproduces S6 as the Mouse's exclamation. Speaker and reported content remain explicit in the subject plus speech predicate, preserving the ridicule/help claims rather than asserting them as narrator truth. This is not the requested decomposed holder/attitude fact structure; strict scores remain zero. The quote refers to possible benefits, not an observed numerical validity interval.


## Alice’s Adventures in Wonderland — Chapter I: Down the Rabbit-Hole, opening two paragraphs

Lewis Carroll. [Source](https://www.gutenberg.org/ebooks/11); retrieved 2026-09-09T13:42:39.616173+00:00. Excerpt SHA-256 `63456811be1ee205a9908656ee756338064caa23145f1f8829dcb0ec52f1f77c`; 112 words.

Exact contiguous excerpt (original line breaks retained):
```text
Alice was beginning to get very tired of sitting by her sister on the
bank, and of having nothing to do: once or twice she had peeped into
the book her sister was reading, but it had no pictures or
conversations in it, “and what is the use of a book,” thought Alice
“without pictures or conversations?”

So she was considering in her own mind (as well as she could, for the
hot day made her feel very sleepy and stupid), whether the pleasure of
making a daisy-chain would be worth the trouble of getting up and
picking the daisies, when suddenly a White Rabbit with pink eyes ran
close by her.
```

[Source attribution and full supplied license notices](../data/published_prose/pg11_notices.txt). Project Gutenberg lists this edition as public domain in the USA; original attribution and complete supplied license notices retained. Check local law outside the USA.

Evidence spans (numbers are added labels; concatenation reproduces the exact excerpt):

**S1**

```text
Alice was beginning to get very tired of sitting by her sister on the
bank, and of having nothing to do: once or twice she had peeped into
the book her sister was reading, but it had no pictures or
conversations in it, “and what is the use of a book,” thought Alice
“without pictures or conversations?”
```

**S2**

```text


So she was considering in her own mind (as well as she could, for the
hot day made her feel very sleepy and stupid), whether the pleasure of
making a daisy-chain would be worth the trouble of getting up and
picking the daisies, when suddenly a White Rabbit with pink eyes ran
close by her.
```

### Which sitting/location, reading/inspection and running relationships involve Alice, her sister, the book, bank and Rabbit? Exclude thoughts, feelings, appearance and book-content properties.

Codex-authored reference facts (alternatives in canonical JSON frozen rules):
```json
[
  {
    "subject": "Alice",
    "relation": "sits_by",
    "object": "Alice's sister",
    "evidence_ids": [
      "S1"
    ],
    "valid_from": null,
    "valid_until": null,
    "holder": null,
    "attitude": null
  },
  {
    "subject": "Alice",
    "relation": "located_on",
    "object": "bank",
    "evidence_ids": [
      "S1"
    ],
    "valid_from": null,
    "valid_until": null,
    "holder": null,
    "attitude": null
  },
  {
    "subject": "Alice's sister",
    "relation": "reads",
    "object": "book",
    "evidence_ids": [
      "S1"
    ],
    "valid_from": null,
    "valid_until": null,
    "holder": null,
    "attitude": null
  },
  {
    "subject": "Alice",
    "relation": "peeps_into",
    "object": "book",
    "evidence_ids": [
      "S1"
    ],
    "valid_from": null,
    "valid_until": null,
    "holder": null,
    "attitude": null
  },
  {
    "subject": "White Rabbit",
    "relation": "runs_past",
    "object": "Alice",
    "evidence_ids": [
      "S2"
    ],
    "valid_from": null,
    "valid_until": null,
    "holder": null,
    "attitude": null
  }
]
```

#### A pre-extract/select

Actual model-authored records and source-based assessment:

1. `{"attitude": null, "evidence_ids": ["S1"], "holder": null, "object": "the book her sister was reading", "relation": "had_peeped_into", "subject": "Alice", "valid_from": null, "valid_until": null}` — supported; qualifiers correct; relevant True. S1 supports Alice peeping into the book; the model-authored object also explicitly says her sister was reading that book. Both meanings are present, though one is embedded in a long node label rather than a separate edge. This one record receives at most one supported-record credit and two distinct target-coverage credits.

2. `{"attitude": null, "evidence_ids": ["S2"], "holder": null, "object": "Alice", "relation": "ran close by", "subject": "White Rabbit", "valid_from": null, "valid_until": null}` — supported; qualifiers correct; relevant True. Frozen reference match, not independent verification.

Missing strict targets: `[{"subject": "Alice", "relation": "sits_by", "object": "Alice's sister", "evidence_ids": ["S1"], "valid_from": null, "valid_until": null, "holder": null, "attitude": null}, {"subject": "Alice", "relation": "located_on", "object": "bank", "evidence_ids": ["S1"], "valid_from": null, "valid_until": null, "holder": null, "attitude": null}, {"subject": "Alice's sister", "relation": "reads", "object": "book", "evidence_ids": ["S1"], "valid_from": null, "valid_until": null, "holder": null, "attitude": null}, {"subject": "Alice", "relation": "peeps_into", "object": "book", "evidence_ids": ["S1"], "valid_from": null, "valid_until": null, "holder": null, "attitude": null}]`.

#### B contextual

Actual model-authored records and source-based assessment:

1. `{"attitude": null, "evidence_ids": ["S1"], "holder": null, "object": "her sister", "relation": "sitting_by", "subject": "Alice", "valid_from": null, "valid_until": null}` — supported; qualifiers correct; relevant True. Frozen reference match, not independent verification.

2. `{"attitude": null, "evidence_ids": ["S1"], "holder": null, "object": "bank", "relation": "sitting_by", "subject": "Alice", "valid_from": null, "valid_until": null}` — partial; qualifiers correct; relevant True. S1 places Alice on the bank. sitting_by bank instead expresses a coarser adjacent-location relationship, not the explicit on-bank target. Do not normalize the preposition away; partial spatial information only.

3. `{"attitude": null, "evidence_ids": ["S1"], "holder": null, "object": "the book", "relation": "peeped_into", "subject": "Alice", "valid_from": null, "valid_until": null}` — supported; qualifiers correct; relevant True. Frozen reference match, not independent verification.

4. `{"attitude": null, "evidence_ids": ["S1"], "holder": null, "object": "the book", "relation": "reading", "subject": "Alice", "valid_from": null, "valid_until": null}` — unsupported; qualifiers correct; relevant True. S1 explicitly assigns reading to the sister and peeping to Alice. It does not support Alice reading the book. This is an unsupported participant/action assignment, not proof that Alice never read anything.

5. `{"attitude": null, "evidence_ids": ["S2"], "holder": null, "object": "close by her", "relation": "ran", "subject": "White Rabbit", "valid_from": null, "valid_until": null}` — supported; qualifiers correct; relevant True. S2 says the White Rabbit ran close by her, with her unambiguously Alice in the supplied passage. The output retains that exact referential phrase in its object. Manual meaning credit does not merge graph endpoints or change the strict matcher.

Missing strict targets: `[{"subject": "Alice", "relation": "located_on", "object": "bank", "evidence_ids": ["S1"], "valid_from": null, "valid_until": null, "holder": null, "attitude": null}, {"subject": "Alice's sister", "relation": "reads", "object": "book", "evidence_ids": ["S1"], "valid_from": null, "valid_until": null, "holder": null, "attitude": null}, {"subject": "White Rabbit", "relation": "runs_past", "object": "Alice", "evidence_ids": ["S2"], "valid_from": null, "valid_until": null, "holder": null, "attitude": null}]`.

### What does the passage say about the book's absent content and Alice's thoughts about the usefulness of such books and making a daisy-chain? Exclude physical actions, locations, appearance and the Rabbit; preserve questions and consideration without treating them as decisions.

Codex-authored reference facts (alternatives in canonical JSON frozen rules):
```json
[
  {
    "subject": "book",
    "relation": "lacks",
    "object": "pictures",
    "evidence_ids": [
      "S1"
    ],
    "valid_from": null,
    "valid_until": null,
    "holder": null,
    "attitude": null
  },
  {
    "subject": "book",
    "relation": "lacks",
    "object": "conversations",
    "evidence_ids": [
      "S1"
    ],
    "valid_from": null,
    "valid_until": null,
    "holder": null,
    "attitude": null
  },
  {
    "subject": "Alice",
    "relation": "questions_usefulness_of",
    "object": "book without pictures or conversations",
    "evidence_ids": [
      "S1"
    ],
    "valid_from": null,
    "valid_until": null,
    "holder": null,
    "attitude": null
  },
  {
    "subject": "Alice",
    "relation": "considers_making",
    "object": "daisy-chain",
    "evidence_ids": [
      "S2"
    ],
    "valid_from": null,
    "valid_until": null,
    "holder": null,
    "attitude": null
  }
]
```

#### A pre-extract/select

Actual model-authored records and source-based assessment:

1. `{"attitude": null, "evidence_ids": ["S1"], "holder": null, "object": "what is the use of a book", "relation": "thought", "subject": "Alice", "valid_from": null, "valid_until": null}` — partial; qualifiers correct; relevant True. This is an actual fragment of Alice's thought in S1. It drops the without-pictures-or-conversations restriction, so it does not independently preserve the full target. Do not join it to the next record to invent a complete proposition.

2. `{"attitude": null, "evidence_ids": ["S1"], "holder": null, "object": "without pictures or conversations", "relation": "thought", "subject": "Alice", "valid_from": null, "valid_until": null}` — partial; qualifiers unresolved; relevant True. The quoted words occur in S1, but 'without pictures or conversations' alone is not a proposition with a book referent or a question. Alice as thinker is explicit; complete qualification and meaning remain unresolved.

3. `{"attitude": null, "evidence_ids": ["S2"], "holder": null, "object": "the pleasure of making a daisy-chain", "relation": "was_considering", "subject": "Alice", "valid_from": null, "valid_until": null}` — supported; qualifiers correct; relevant True. S2 explicitly describes Alice considering the pleasure of making a daisy-chain. Consideration is retained rather than presented as a decision or completed act. The finer worth-versus-trouble deliberation is omitted; the frozen broad considers-making target is nevertheless explicit.

4. `{"attitude": null, "evidence_ids": ["S2"], "holder": null, "object": "whether the pleasure of making a daisy-chain would be worth the trouble of getting up and picking the daisies", "relation": "was_considering", "subject": "Alice", "valid_from": null, "valid_until": null}` — supported; qualifiers correct; relevant True. S2 directly supports this fuller deliberation, including whether pleasure is worth the effort. It repeats the same broad target as the preceding record: semantic coverage counts that target once, and both predictions stay in strict denominators.

Missing strict targets: `[{"subject": "book", "relation": "lacks", "object": "pictures", "evidence_ids": ["S1"], "valid_from": null, "valid_until": null, "holder": null, "attitude": null}, {"subject": "book", "relation": "lacks", "object": "conversations", "evidence_ids": ["S1"], "valid_from": null, "valid_until": null, "holder": null, "attitude": null}, {"subject": "Alice", "relation": "questions_usefulness_of", "object": "book without pictures or conversations", "evidence_ids": ["S1"], "valid_from": null, "valid_until": null, "holder": null, "attitude": null}, {"subject": "Alice", "relation": "considers_making", "object": "daisy-chain", "evidence_ids": ["S2"], "valid_from": null, "valid_until": null, "holder": null, "attitude": null}]`.

#### B contextual

Actual model-authored records and source-based assessment:

1. `{"attitude": null, "evidence_ids": ["S1"], "holder": null, "object": "pictures or conversations", "relation": "lacks", "subject": "book", "valid_from": null, "valid_until": null}` — supported; qualifiers correct; relevant True. S1 explicitly says the book has no pictures or conversations. The combined negative object preserves both absent-content targets; one record receives one supported-record credit, and two unique semantic-coverage credits. No splitting or new model facts are performed.

2. `{"attitude": null, "evidence_ids": ["S1"], "holder": null, "object": "use of a book without pictures or conversations is questionable", "relation": "thinks", "subject": "Alice", "valid_from": null, "valid_until": null}` — supported; qualifiers correct; relevant True. The subject Alice, mental predicate thinks, and object questioning the use of a pictureless/conversationless book preserve her rhetorical thought from S1. It is not asserted as objective uselessness. This semantically supported mental-act representation remains a strict wording mismatch.

3. `{"attitude": null, "evidence_ids": ["S2"], "holder": "Alice", "object": "worth the trouble of getting up and picking the daisies", "relation": "considered", "subject": "making a daisy-chain", "valid_from": null, "valid_until": null}` — unresolved; qualifiers incorrect; relevant True. S2 says Alice is considering WHETHER making the chain would be worth the trouble. 'considered worth' may instead mean judged worthwhile. The missing attitude alongside holder Alice also violates the compact pairing rule. Source support for a settled positive judgment cannot be established; do not award the consideration target by guessing.

Missing strict targets: `[{"subject": "book", "relation": "lacks", "object": "pictures", "evidence_ids": ["S1"], "valid_from": null, "valid_until": null, "holder": null, "attitude": null}, {"subject": "book", "relation": "lacks", "object": "conversations", "evidence_ids": ["S1"], "valid_from": null, "valid_until": null, "holder": null, "attitude": null}, {"subject": "Alice", "relation": "questions_usefulness_of", "object": "book without pictures or conversations", "evidence_ids": ["S1"], "valid_from": null, "valid_until": null, "holder": null, "attitude": null}, {"subject": "Alice", "relation": "considers_making", "object": "daisy-chain", "evidence_ids": ["S2"], "valid_from": null, "valid_until": null, "holder": null, "attitude": null}]`.

### Complete query-blind collection (before filtering)

1. `{"attitude": null, "evidence_ids": ["S1"], "holder": null, "object": "sitting by her sister on the bank", "relation": "is_tired_of", "subject": "Alice", "valid_from": null, "valid_until": null}` — supported; S1 directly supports tiredness and the embedded sitting-by-sister/on-bank relationship. Those two embedded targets are explicit in the object, but the whole record principally asserts a feeling excluded by both questions; the frozen selector rejects it. Full pre-extraction is displayed without injecting this record into either selected graph.

2. `{"attitude": null, "evidence_ids": ["S1"], "holder": null, "object": "having nothing to do", "relation": "is_tired_of", "subject": "Alice", "valid_from": null, "valid_until": null}` — supported; S1 explicitly describes tiredness at having nothing to do. This is a supported feeling outside both question scopes.

3. `{"attitude": null, "evidence_ids": ["S1"], "holder": null, "object": "the book her sister was reading", "relation": "had_peeped_into", "subject": "Alice", "valid_from": null, "valid_until": null}` — supported; S1 supports Alice peeping into the book; the model-authored object also explicitly says her sister was reading that book. Both meanings are present, though one is embedded in a long node label rather than a separate edge. This one record receives at most one supported-record credit and two distinct target-coverage credits.

4. `{"attitude": null, "evidence_ids": ["S1"], "holder": null, "object": "what is the use of a book", "relation": "thought", "subject": "Alice", "valid_from": null, "valid_until": null}` — partial; This is an actual fragment of Alice's thought in S1. It drops the without-pictures-or-conversations restriction, so it does not independently preserve the full target. Do not join it to the next record to invent a complete proposition.

5. `{"attitude": null, "evidence_ids": ["S1"], "holder": null, "object": "without pictures or conversations", "relation": "thought", "subject": "Alice", "valid_from": null, "valid_until": null}` — partial; The quoted words occur in S1, but 'without pictures or conversations' alone is not a proposition with a book referent or a question. Alice as thinker is explicit; complete qualification and meaning remain unresolved.

6. `{"attitude": null, "evidence_ids": ["S2"], "holder": null, "object": "the pleasure of making a daisy-chain", "relation": "was_considering", "subject": "Alice", "valid_from": null, "valid_until": null}` — supported; S2 explicitly describes Alice considering the pleasure of making a daisy-chain. Consideration is retained rather than presented as a decision or completed act. The finer worth-versus-trouble deliberation is omitted; the frozen broad considers-making target is nevertheless explicit.

7. `{"attitude": null, "evidence_ids": ["S2"], "holder": null, "object": "whether the pleasure of making a daisy-chain would be worth the trouble of getting up and picking the daisies", "relation": "was_considering", "subject": "Alice", "valid_from": null, "valid_until": null}` — supported; S2 directly supports this fuller deliberation, including whether pleasure is worth the effort. It repeats the same broad target as the preceding record: semantic coverage counts that target once, and both predictions stay in strict denominators.

8. `{"attitude": null, "evidence_ids": ["S2"], "holder": null, "object": "Alice", "relation": "ran close by", "subject": "White Rabbit", "valid_from": null, "valid_until": null}` — supported; Frozen reference match, not independent verification.


## The Adventures of Sherlock Holmes — II. The Red-Headed League, opening visit through the gentleman’s greeting

Arthur Conan Doyle. [Source](https://www.gutenberg.org/ebooks/1661); retrieved 2026-09-09T13:42:39.964188+00:00. Excerpt SHA-256 `07b8b4e4daa61ba653e966da4f78816bd72f92eabdfc113ddffb56c33829e024`; 159 words.

Exact contiguous excerpt (original line breaks retained):
```text
 I had called upon my friend, Mr. Sherlock Holmes, one day in the
 autumn of last year and found him in deep conversation with a very
 stout, florid-faced, elderly gentleman with fiery red hair. With an
 apology for my intrusion, I was about to withdraw when Holmes pulled
 me abruptly into the room and closed the door behind me.

“You could not possibly have come at a better time, my dear Watson,” he
said cordially.

“I was afraid that you were engaged.”

“So I am. Very much so.”

“Then I can wait in the next room.”

“Not at all. This gentleman, Mr. Wilson, has been my partner and helper
in many of my most successful cases, and I have no doubt that he will
be of the utmost use to me in yours also.”

The stout gentleman half rose from his chair and gave a bob of
greeting, with a quick little questioning glance from his small
fat-encircled eyes.
```

[Source attribution and full supplied license notices](../data/published_prose/pg1661_notices.txt). Project Gutenberg lists this edition as public domain in the USA; original attribution and complete supplied license notices retained. Check local law outside the USA.

Evidence spans (numbers are added labels; concatenation reproduces the exact excerpt):

**S1**

```text
 I had called upon my friend, Mr. Sherlock Holmes, one day in the
 autumn of last year and found him in deep conversation with a very
 stout, florid-faced, elderly gentleman with fiery red hair.
```

**S2**

```text
 With an
 apology for my intrusion, I was about to withdraw when Holmes pulled
 me abruptly into the room and closed the door behind me.
```

**S3**

```text


“You could not possibly have come at a better time, my dear Watson,” he
said cordially.
```

**S4**

```text


“I was afraid that you were engaged.”
```

**S5**

```text


“So I am. Very much so.”
```

**S6**

```text


“Then I can wait in the next room.”
```

**S7**

```text


“Not at all. This gentleman, Mr. Wilson, has been my partner and helper
in many of my most successful cases, and I have no doubt that he will
be of the utmost use to me in yours also.”
```

**S8**

```text


The stout gentleman half rose from his chair and gave a bob of
greeting, with a quick little questioning glance from his small
fat-encircled eyes.
```

### Which visit, conversation, pulling, door-closing, rising and greeting actions involve Watson, Holmes and the elderly gentleman? Exclude appearance and the content of dialogue.

Codex-authored reference facts (alternatives in canonical JSON frozen rules):
```json
[
  {
    "subject": "Watson",
    "relation": "visits",
    "object": "Holmes",
    "evidence_ids": [
      "S1"
    ],
    "valid_from": null,
    "valid_until": null,
    "holder": null,
    "attitude": null
  },
  {
    "subject": "Holmes",
    "relation": "converses_with",
    "object": "Wilson",
    "evidence_ids": [
      "S1"
    ],
    "valid_from": null,
    "valid_until": null,
    "holder": null,
    "attitude": null
  },
  {
    "subject": "Holmes",
    "relation": "pulls",
    "object": "Watson",
    "evidence_ids": [
      "S2"
    ],
    "valid_from": null,
    "valid_until": null,
    "holder": null,
    "attitude": null
  },
  {
    "subject": "Holmes",
    "relation": "closes",
    "object": "door",
    "evidence_ids": [
      "S2"
    ],
    "valid_from": null,
    "valid_until": null,
    "holder": null,
    "attitude": null
  },
  {
    "subject": "Wilson",
    "relation": "rises_from",
    "object": "chair",
    "evidence_ids": [
      "S8"
    ],
    "valid_from": null,
    "valid_until": null,
    "holder": null,
    "attitude": null
  },
  {
    "subject": "Wilson",
    "relation": "greets",
    "object": "Watson",
    "evidence_ids": [
      "S8"
    ],
    "valid_from": null,
    "valid_until": null,
    "holder": null,
    "attitude": null
  }
]
```

#### A pre-extract/select

Actual model-authored records and source-based assessment:

Missing strict targets: `[{"subject": "Watson", "relation": "visits", "object": "Holmes", "evidence_ids": ["S1"], "valid_from": null, "valid_until": null, "holder": null, "attitude": null}, {"subject": "Holmes", "relation": "converses_with", "object": "Wilson", "evidence_ids": ["S1"], "valid_from": null, "valid_until": null, "holder": null, "attitude": null}, {"subject": "Holmes", "relation": "pulls", "object": "Watson", "evidence_ids": ["S2"], "valid_from": null, "valid_until": null, "holder": null, "attitude": null}, {"subject": "Holmes", "relation": "closes", "object": "door", "evidence_ids": ["S2"], "valid_from": null, "valid_until": null, "holder": null, "attitude": null}, {"subject": "Wilson", "relation": "rises_from", "object": "chair", "evidence_ids": ["S8"], "valid_from": null, "valid_until": null, "holder": null, "attitude": null}, {"subject": "Wilson", "relation": "greets", "object": "Watson", "evidence_ids": ["S8"], "valid_from": null, "valid_until": null, "holder": null, "attitude": null}]`.

Structural failure: Extra data: line 1 column 2532 (char 2531)

Unusable/unexecuted raw response (not completed by CPU):
```text
{"facts":[{"subject":"I","relation":"called_upon","object":"Mr. Sherlock Holmes","evidence_ids":["S1"],"valid_from":null,"valid_until":null,"holder":null,"attitude":null},{"subject":"I","relation":"found","object":"Mr. Sherlock Holmes","evidence_ids":["S1"],"valid_from":null,"valid_until":null,"holder":null,"attitude":null},{"subject":"Mr. Sherlock Holmes","relation":"in_deep_conversation_with","object":"a very stout, florid-faced, elderly gentleman with fiery red hair","evidence_ids":["S1"],"valid_from":null,"valid_until":null,"holder":null,"attitude":null},{"subject":"I","relation":"about_to_withdraw","object":"from the room","evidence_ids":["S2"],"valid_from":null,"valid_until":null,"holder":null,"attitude":null},{"subject":"Mr. Sherlock Holmes","relation":"pulled","object":"I","evidence_ids":["S2"],"valid_from":null,"valid until":null,"holder":null,"attitude":null},{"subject":"Mr. Sherlock Holmes","relation":"closed","object":"the door behind me","evidence_ids":["S2"],"valid_from":null,"valid_until":null,"holder":null,"attitude":null},{"subject":"Mr. Sherlock Holmes","relation":"said","object":"You could not possibly have come at a better time, my dear Watson","evidence_ids":["S3"],"valid_from":null,"valid_until":null,"holder":null,"attitude":null},{"subject":"I","relation":"was_afraid","object":"that you were engaged","evidence_ids":["S4"],"valid_from":null,"valid_until":null,"holder":null,"attitude":null},{"subject":"I","relation":"said","object":"So I am. Very much so.","evidence_ids":["S5"],"valid_from":null,"valid_until":null,"holder":null,"attitude":null},{"subject":"Mr. Sherlock Holmes","relation":"said","object":"Then I can wait in the next room","evidence_ids":["S6"],"valid_from":null,"valid_until":null,"holder":null,"attitude":null},{"subject":"Mr. Sherlock Holmes","relation":"said","object":"Not at all. This gentleman, Mr. Wilson, has been my partner and helper in many of my most successful cases, and I have no doubt that he will be of the utmost use to me in yours also","evidence_ids":["S7"],"valid_from":null,"valid_until":null,"holder":null,"attitude":null},{"subject":"The stout gentleman","relation":"half rose from his chair","object":"and gave a bob of greeting","evidence_ids":["S8"],"valid_from":null,"valid_until":null,"holder":null,"attitude":null},{"subject":"The stout gentleman","relation":"gave","object":"a quick little questioning glance from his small fat-encircled eyes","evidence_ids":["S8"],"valid_from":null,"valid_until":null,"holder":null,"attitude":null}]}}
```

#### B contextual

Actual model-authored records and source-based assessment:

Missing strict targets: `[{"subject": "Watson", "relation": "visits", "object": "Holmes", "evidence_ids": ["S1"], "valid_from": null, "valid_until": null, "holder": null, "attitude": null}, {"subject": "Holmes", "relation": "converses_with", "object": "Wilson", "evidence_ids": ["S1"], "valid_from": null, "valid_until": null, "holder": null, "attitude": null}, {"subject": "Holmes", "relation": "pulls", "object": "Watson", "evidence_ids": ["S2"], "valid_from": null, "valid_until": null, "holder": null, "attitude": null}, {"subject": "Holmes", "relation": "closes", "object": "door", "evidence_ids": ["S2"], "valid_from": null, "valid_until": null, "holder": null, "attitude": null}, {"subject": "Wilson", "relation": "rises_from", "object": "chair", "evidence_ids": ["S8"], "valid_from": null, "valid_until": null, "holder": null, "attitude": null}, {"subject": "Wilson", "relation": "greets", "object": "Watson", "evidence_ids": ["S8"], "valid_from": null, "valid_until": null, "holder": null, "attitude": null}]`.

Structural failure: Extra data: line 1 column 1683 (char 1682)

Unusable/unexecuted raw response (not completed by CPU):
```text
{"facts":[{"subject":"Holmes","relation":"in_deep_conversation_with","object":"elderly_gentleman","evidence_ids":["S1"],"valid_from":null,"valid_until":null,"holder":null,"attitude":null},{"subject":"Holmes","relation":"pulled","object":"Watson","evidence_ids":["S2"],"valid_from":null,"valid_until":null,"holder":null,"attitude":null},{"subject":"Holmes","relation":"closed","object":"door","evidence_ids":["S2"],"valid_from":null,"valid_until":null,"holder":null,"attitude":null},{"subject":"Holmes","relation":"greeted","object":"elderly_gentleman","evidence_ids":["S8"],"valid_from":null,"valid_until":null,"holder":null,"attitude":null},{"subject":"elderly_gentleman","relation":"greeted","object":"Holmes","evidence_ids":["S8"],"valid_from":null,"valid_until":null,"holder":null,"attitude":null},{"subject":"Watson","relation":"greeted","object":"Holmes","evidence_ids":["S3"],"valid_from":null,"valid_until":null,"holder":null,"attitude":null},{"subject":"Holmes","relation":"said","object":"Watson","evidence_ids":["S3"],"valid_from":null,"valid_until":null,"holder":null,"attitude":null},{"subject":"Watson","relation":"said","object":"Holmes","evidence_ids":["S4"],"valid_from":null,"valid_until":null,"holder":null,"attitude":null},{"subject":"Holmes","relation":"said","object":"Watson","evidence_ids":["S5"],"valid_from":null,"valid_until":null,"holder":null,"attitude":null},{"subject":"Watson","relation":"said","object":"Holmes","evidence_ids":["S6"],"valid_from":null,"valid_until":null,"holder":null,"attitude":null},{"subject":"Holmes","relation":"said","object":"Watson","evidence_ids":["S7"],"valid_from":null,"valid_until":null,"holder":null,"attitude":null}]}}
```

### What does the dialogue explicitly convey about being engaged, past partnership/help in cases and expected help in the present case? Keep each claim or concern attributed to its speaker. Exclude physical movements, appearance, offers to wait and politeness about the arrival time.

Codex-authored reference facts (alternatives in canonical JSON frozen rules):
```json
[
  {
    "subject": "Holmes",
    "relation": "is",
    "object": "engaged",
    "evidence_ids": [
      "S4"
    ],
    "valid_from": null,
    "valid_until": null,
    "holder": "Watson",
    "attitude": "believes"
  },
  {
    "subject": "Holmes",
    "relation": "is",
    "object": "engaged",
    "evidence_ids": [
      "S5"
    ],
    "valid_from": null,
    "valid_until": null,
    "holder": "Holmes",
    "attitude": "reported"
  },
  {
    "subject": "Watson",
    "relation": "partner_of",
    "object": "Holmes",
    "evidence_ids": [
      "S7"
    ],
    "valid_from": null,
    "valid_until": null,
    "holder": "Holmes",
    "attitude": "reported"
  },
  {
    "subject": "Watson",
    "relation": "helper_of",
    "object": "Holmes",
    "evidence_ids": [
      "S7"
    ],
    "valid_from": null,
    "valid_until": null,
    "holder": "Holmes",
    "attitude": "reported"
  },
  {
    "subject": "Watson",
    "relation": "expected_useful_in",
    "object": "Wilson's case",
    "evidence_ids": [
      "S7"
    ],
    "valid_from": null,
    "valid_until": null,
    "holder": "Holmes",
    "attitude": "believes"
  }
]
```

#### A pre-extract/select

Actual model-authored records and source-based assessment:

Missing strict targets: `[{"subject": "Holmes", "relation": "is", "object": "engaged", "evidence_ids": ["S4"], "valid_from": null, "valid_until": null, "holder": "Watson", "attitude": "believes"}, {"subject": "Holmes", "relation": "is", "object": "engaged", "evidence_ids": ["S5"], "valid_from": null, "valid_until": null, "holder": "Holmes", "attitude": "reported"}, {"subject": "Watson", "relation": "partner_of", "object": "Holmes", "evidence_ids": ["S7"], "valid_from": null, "valid_until": null, "holder": "Holmes", "attitude": "reported"}, {"subject": "Watson", "relation": "helper_of", "object": "Holmes", "evidence_ids": ["S7"], "valid_from": null, "valid_until": null, "holder": "Holmes", "attitude": "reported"}, {"subject": "Watson", "relation": "expected_useful_in", "object": "Wilson's case", "evidence_ids": ["S7"], "valid_from": null, "valid_until": null, "holder": "Holmes", "attitude": "believes"}]`.

Structural failure: Extra data: line 1 column 2532 (char 2531)

Unusable/unexecuted raw response (not completed by CPU):
```text
{"facts":[{"subject":"I","relation":"called_upon","object":"Mr. Sherlock Holmes","evidence_ids":["S1"],"valid_from":null,"valid_until":null,"holder":null,"attitude":null},{"subject":"I","relation":"found","object":"Mr. Sherlock Holmes","evidence_ids":["S1"],"valid_from":null,"valid_until":null,"holder":null,"attitude":null},{"subject":"Mr. Sherlock Holmes","relation":"in_deep_conversation_with","object":"a very stout, florid-faced, elderly gentleman with fiery red hair","evidence_ids":["S1"],"valid_from":null,"valid_until":null,"holder":null,"attitude":null},{"subject":"I","relation":"about_to_withdraw","object":"from the room","evidence_ids":["S2"],"valid_from":null,"valid_until":null,"holder":null,"attitude":null},{"subject":"Mr. Sherlock Holmes","relation":"pulled","object":"I","evidence_ids":["S2"],"valid_from":null,"valid until":null,"holder":null,"attitude":null},{"subject":"Mr. Sherlock Holmes","relation":"closed","object":"the door behind me","evidence_ids":["S2"],"valid_from":null,"valid_until":null,"holder":null,"attitude":null},{"subject":"Mr. Sherlock Holmes","relation":"said","object":"You could not possibly have come at a better time, my dear Watson","evidence_ids":["S3"],"valid_from":null,"valid_until":null,"holder":null,"attitude":null},{"subject":"I","relation":"was_afraid","object":"that you were engaged","evidence_ids":["S4"],"valid_from":null,"valid_until":null,"holder":null,"attitude":null},{"subject":"I","relation":"said","object":"So I am. Very much so.","evidence_ids":["S5"],"valid_from":null,"valid_until":null,"holder":null,"attitude":null},{"subject":"Mr. Sherlock Holmes","relation":"said","object":"Then I can wait in the next room","evidence_ids":["S6"],"valid_from":null,"valid_until":null,"holder":null,"attitude":null},{"subject":"Mr. Sherlock Holmes","relation":"said","object":"Not at all. This gentleman, Mr. Wilson, has been my partner and helper in many of my most successful cases, and I have no doubt that he will be of the utmost use to me in yours also","evidence_ids":["S7"],"valid_from":null,"valid_until":null,"holder":null,"attitude":null},{"subject":"The stout gentleman","relation":"half rose from his chair","object":"and gave a bob of greeting","evidence_ids":["S8"],"valid_from":null,"valid_until":null,"holder":null,"attitude":null},{"subject":"The stout gentleman","relation":"gave","object":"a quick little questioning glance from his small fat-encircled eyes","evidence_ids":["S8"],"valid_from":null,"valid_until":null,"holder":null,"attitude":null}]}}
```

#### B contextual

Actual model-authored records and source-based assessment:

1. `{"attitude": null, "evidence_ids": ["S4", "S5"], "holder": null, "object": "with Watson", "relation": "is_engaged", "subject": "Holmes", "valid_from": null, "valid_until": null}` — partial; qualifiers incorrect; relevant True. S4-S5 contain Watson's concern and Holmes's confirmation of being engaged. They do not say engaged with Watson. That added participant is unsupported, and null holder/attitude loses the dialogue attribution. The busy-state component is recognizable but no complete scoped target is preserved.

2. `{"attitude": null, "evidence_ids": ["S7"], "holder": null, "object": "with Mr. Wilson", "relation": "has_past_partnership", "subject": "Holmes", "valid_from": null, "valid_until": null}` — unresolved; qualifiers incorrect; relevant True. The frozen contextual reading of S7 is Holmes addressing Wilson while describing Watson as his past partner. This answer instead names Wilson. The comma/vocative wording is locally ambiguous, so identity is unresolved rather than declared a proven contradiction. Independently, it presents the reported partnership as a direct fact with null attribution.

3. `{"attitude": null, "evidence_ids": ["S7"], "holder": null, "object": "Mr. Wilson", "relation": "expects_help_from", "subject": "Holmes", "valid_from": null, "valid_until": null}` — unresolved; qualifiers correct; relevant True. S7 describes expected usefulness to Holmes; the frozen reading assigns the expected helper to Watson, with Wilson addressed. The answer names Wilson instead. Local vocative ambiguity prevents positive participant credit. 'Holmes expects' does retain an explicitly holder-relative mental act, not accomplished help, despite null holder fields. No target coverage is awarded for the uncertain referent.

Missing strict targets: `[{"subject": "Holmes", "relation": "is", "object": "engaged", "evidence_ids": ["S4"], "valid_from": null, "valid_until": null, "holder": "Watson", "attitude": "believes"}, {"subject": "Holmes", "relation": "is", "object": "engaged", "evidence_ids": ["S5"], "valid_from": null, "valid_until": null, "holder": "Holmes", "attitude": "reported"}, {"subject": "Watson", "relation": "partner_of", "object": "Holmes", "evidence_ids": ["S7"], "valid_from": null, "valid_until": null, "holder": "Holmes", "attitude": "reported"}, {"subject": "Watson", "relation": "helper_of", "object": "Holmes", "evidence_ids": ["S7"], "valid_from": null, "valid_until": null, "holder": "Holmes", "attitude": "reported"}, {"subject": "Watson", "relation": "expected_useful_in", "object": "Wilson's case", "evidence_ids": ["S7"], "valid_from": null, "valid_until": null, "holder": "Holmes", "attitude": "believes"}]`.

### Complete query-blind collection (before filtering)

## Costs and limits

| Call | Passage/task | Finish | Input/output tokens | Request seconds |
|---|---|---|---|---|
| 1 | fable/all | stop | 747/554 | 10.233251 |
| 2 | alice/all | stop | 701/381 | 7.268172 |
| 3 | holmes/all | stop | 785/651 | 12.695497 |
| 4 | fable/actions | stop | 761/209 | 3.882963 |
| 5 | fable/claims | stop | 767/170 | 3.428558 |
| 6 | alice/actions | stop | 711/208 | 3.837432 |
| 7 | alice/claims | stop | 726/148 | 3.673451 |
| 8 | holmes/actions | stop | 793/467 | 8.778309 |
| 9 | holmes/claims | stop | 811/139 | 3.441627 |

9/9 calls executed; unexecuted IDs: []. Allocation 205.365216 seconds new / 10717.909776 cumulative. Open allocation/service journals 0/0; stop reason completed.
Sampled peaks across 166 observations: GPU VRAM 21.129 GiB; process RAM 6.687 GiB; project storage 16.202 GiB; CPU workers 8. Observed resource violations: []. These are observed, not continuous peak guarantees.

One pinned Qwen3-8B-AWQ revision `4da05a8edb55c6046cce958586c33b61da07bb79`, unchanged nonthinking decoding, 12,288 context and common 2,048 output reserve. Query-blind costs are shared across two selections; no adaptive retry. Samples do not establish p95, held-out generalization or full-study feasibility. Familiar published books may have appeared in training.

Temporal limitation: the fable says “shortly after this”; Holmes says “one day in the autumn of last year” and discusses past cases and future help. Null numerical validity avoids fabrication but does not preserve this chronology or duration. Relative order, conditional promises, and detailed speech content exceed some uses of this compact representation.

Coverage caveats: references cover the declared questions, not every defensible detail. Pronoun/greeting targets, rhetorical questions and vocative dialogue introduce interpretation uncertainty. Codex-authored judgments are exploratory, not independent review. Original synthetic/C0 results and registered gates are unchanged.

[Graphs](figures/real_text_proof_of_concept.html) · [CSV](tables/real_text_proof_of_concept.csv).

CPU reproduction: `python -m scripts.report_real_text_poc --run <restricted-run> --output reports`.
