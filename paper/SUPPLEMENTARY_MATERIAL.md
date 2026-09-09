# Supplementary material: retained exploratory narrative graphs

This supplement preserves the eight original review plates and the full-output appendix without altering their PDF pages. The new main figures use labelled display subsets; all evaluated records and historical failures remain available. References and semantic assessments are Codex-authored, not independent human review.

## Contents and evaluation versions

The pages below provide detailed canonical tables and exact executed questions. The original eight-plate review document follows, then its full-output appendix, including unparseable Holmes outputs shown as raw text. Source passages, reference alternatives, and all calls are also available in the immutable repository reports linked in the manuscript manifest.

## S1. Compact stories v2: fresh outputs only

| Story | Question | A n | A P/R/F1 | B n | B P/R/F1 |
| --- | --- | --- | --- | --- | --- |
| Harbor | possession | 5 | 1.000/1.000/1.000 | 10 | 0.500/1.000/0.667 |
| Harbor | locations | 4 | 1.000/1.000/1.000 | 6 | 0.667/1.000/0.800 |
| Harbor | belief | 5 | 0.000/0.000/0.000 | 14 | 0.143/0.500/0.222 |
| Orchard | possession | 5 | 1.000/1.000/1.000 | 12 | 0.417/1.000/0.588 |
| Orchard | locations | 4 | 1.000/1.000/1.000 | 5 | 0.600/0.750/0.667 |
| Orchard | belief | 4 | 1.000/1.000/1.000 | 13 | 0.154/0.500/0.235 |
| Museum | possession | 5 | 1.000/1.000/1.000 | 13 | 0.385/1.000/0.556 |
| Museum | locations | 4 | 1.000/1.000/1.000 | 5 | 0.800/1.000/0.889 |
| Museum | belief | 4 | 0.000/0.000/0.000 | 14 | 0.143/0.500/0.222 |

## S2. Published prose: separate strict and semantic outcomes

S/P/X/U denotes supported, partial, unsupported and unresolved records. Coverage is unique explicit target concepts. NA is unavailable parsing, not an empty graph.

| Passage / question | Method | n | Strict P/R/F1 | S/P/X/U | Coverage |
| --- | --- | --- | --- | --- | --- |
| Alice / actions | A | 2 | 0.500/0.200/0.286 | 2/0/0/0 | 3/5 |
| Alice / actions | B | 5 | 0.400/0.400/0.400 | 3/1/1/0 | 3/5 |
| Alice / claims | A | 4 | 0.000/0.000/0.000 | 2/2/0/0 | 1/4 |
| Alice / claims | B | 3 | 0.000/0.000/0.000 | 2/0/0/1 | 3/4 |
| Fable / actions | A | 8 | 0.625/0.625/0.625 | 6/1/0/1 | 6/8 |
| Fable / actions | B | 5 | 0.400/0.250/0.308 | 4/1/0/0 | 4/8 |
| Fable / claims | A | 1 | 0.000/0.000/0.000 | 1/0/0/0 | 2/5 |
| Fable / claims | B | 4 | 0.000/0.000/0.000 | 1/3/0/0 | 0/5 |
| Holmes / actions | A | NA | NA/NA/NA | NA | NA |
| Holmes / actions | B | NA | NA/NA/NA | NA | NA |
| Holmes / claims | A | NA | NA/NA/NA | NA | NA |
| Holmes / claims | B | 3 | 0.000/0.000/0.000 | 0/1/0/2 | 0/5 |

## S3. Distinct batch costs

Service allocation includes loading, checks, inference and shutdown. Request times alone exclude those costs. Costs may be added; accuracy across incompatible versions may not.

| Batch (not pooled) | Calls | Input / output tokens | Request s | Allocated s |
| --- | --- | --- | --- | --- |
| Simple ladder | 8 | 1344 / 848 | 20.03 | 157.35 |
| Simple ladder v2 | 8 | 2594 / 997 | 22.45 | 171.37 |
| Compact stories v2 | 12 | 8088 / 5860 | 104.72 | 249.45 |
| Published prose | 9 | 6802 / 2927 | 57.24 | 205.37 |

## S4. ladder — unchanged development scores

| Case | Task | Precision | Recall | F1 |
| --- | --- | --- | --- | --- |
| 1 | direct | 1.000 | 1.000 | 1.000 |
| 2 | direct | 1.000 | 1.000 | 1.000 |
| 3 | selection | 1.000 | 1.000 | 1.000 |
| 4 | selection | 1.000 | 1.000 | 1.000 |
| 5 | temporal | 1.000 | 1.000 | 1.000 |
| 6 | epistemic | 0.000 | 0.000 | 0.000 |
| 7 | ontology_contrast | 0.000 | 0.000 | 0.000 |
| 8 | ontology_contrast | 0.500 | 0.500 | 0.500 |

## S4. ladder v2 — unchanged development scores

| Case | Task | Precision | Recall | F1 |
| --- | --- | --- | --- | --- |
| 1 | direct | 1.000 | 1.000 | 1.000 |
| 2 | temporal | 1.000 | 1.000 | 1.000 |
| 3 | epistemic | 1.000 | 1.000 | 1.000 |
| 4 | epistemic | 1.000 | 1.000 | 1.000 |
| 5 | office-person | 0.000 | 0.000 | 0.000 |
| 6 | office-continuity | 0.000 | 0.000 | 0.000 |
| 7 | office-person | 0.500 | 0.500 | 0.500 |
| 8 | office-continuity | 0.500 | 0.500 | 0.500 |

## S5. Historical syntax recovery is not new generation

Earlier Harbor/Orchard possession outputs failed strict parsing. Their separately retained comma-only derivations are below. Original strict failures are unchanged and excluded from the fresh table. Full edit records and original scores are reproduced unchanged in tables/historical_syntax_recovery.json.

| Story | Matches | Predictions | References | P/R/F1 |
| --- | --- | --- | --- | --- |
| harbor | 3 | 10 | 5 | 0.300/0.600/0.400 |
| orchard | 3 | 10 | 5 | 0.300/0.600/0.400 |

## S6. Exact requests and source attribution


### fable_A

Which physical running/waking, capture, binding, rope-gnawing and release relationships involve the Lion, Mouse, hunters and ropes? Exclude dialogue, intentions, laughter and the moral.

### harbor_A_locations

Which directly narrated location relationships have an explicitly stated interval overlapping day 2 up to but not including day 4? Return all of them with their full evidence-stated intervals, not clipped to the question window. Exclude undated locations and beliefs.

### alice_B_actions

Which sitting/location, reading/inspection and running relationships involve Alice, her sister, the book, bank and Rabbit? Exclude thoughts, feelings, appearance and book-content properties.

### alice_B_claims

What does the passage say about the book's absent content and Alice's thoughts about the usefulness of such books and making a daisy-chain? Exclude physical actions, locations, appearance and the Rabbit; preserve questions and consideration without treating them as decisions.

### Alice’s Adventures in Wonderland

Lewis Carroll. Chapter I: Down the Rabbit-Hole, opening two paragraphs. [Source](https://www.gutenberg.org/ebooks/11). Retrieval: 2026-09-09T13:42:39.616173+00:00. Excerpt SHA-256: 63456811be1ee205a9908656ee756338064caa23145f1f8829dcb0ec52f1f77c.

Project Gutenberg lists this edition as public domain in the USA; original attribution and complete supplied license notices retained. Check local law outside the USA. Notices: data/published_prose/pg11_notices.txt.

**S1** Alice was beginning to get very tired of sitting by her sister on the bank, and of having nothing to do: once or twice she had peeped into the book her sister was reading, but it had no pictures or conversations in it, “and what is the use of a book,” thought Alice “without pictures or conversations?”

**S2** So she was considering in her own mind (as well as she could, for the hot day made her feel very sleepy and stupid), whether the pleasure of making a daisy-chain would be worth the trouble of getting up and picking the daisies, when suddenly a White Rabbit with pink eyes ran close by her.



### Three hundred Aesop’s fables

Aesop; translated by George Fyler Townsend. The Lion And The Mouse. [Source](https://www.gutenberg.org/ebooks/21). Retrieval: 2026-09-09T13:42:39.348162+00:00. Excerpt SHA-256: 340385bf6743191af33313e4998e9e16987d7f92d658417861516ce220b25fb0.

Project Gutenberg lists this edition as public domain in the USA; original attribution and complete supplied license notices retained. Check local law outside the USA. Notices: data/published_prose/pg21_notices.txt.

**S1** A LION was awakened from sleep by a Mouse running over his face.

**S2** Rising up angrily, he caught him and was about to kill him, when the Mouse piteously entreated, saying: “If you would only spare my life, I would be sure to repay your kindness.”

**S3** The Lion laughed and let him go.

**S4** It happened shortly after this that the Lion was caught by some hunters, who bound him by strong ropes to the ground.

**S5** The Mouse, recognizing his roar, came and gnawed the rope with his teeth, and set him free, exclaiming:

**S6** “You ridiculed the idea of my ever being able to help you, not expecting to receive from me any repayment of your favour; now you know that it is possible for even a Mouse to confer benefits on a Lion.”



### The Adventures of Sherlock Holmes

Arthur Conan Doyle. II. The Red-Headed League, opening visit through the gentleman’s greeting. [Source](https://www.gutenberg.org/ebooks/1661). Retrieval: 2026-09-09T13:42:39.964188+00:00. Excerpt SHA-256: 07b8b4e4daa61ba653e966da4f78816bd72f92eabdfc113ddffb56c33829e024.

Project Gutenberg lists this edition as public domain in the USA; original attribution and complete supplied license notices retained. Check local law outside the USA. Notices: data/published_prose/pg1661_notices.txt.

**S1** I had called upon my friend, Mr. Sherlock Holmes, one day in the autumn of last year and found him in deep conversation with a very stout, florid-faced, elderly gentleman with fiery red hair.

**S2** With an apology for my intrusion, I was about to withdraw when Holmes pulled me abruptly into the room and closed the door behind me.

**S3** “You could not possibly have come at a better time, my dear Watson,” he said cordially.

**S4** “I was afraid that you were engaged.”

**S5** “So I am. Very much so.”

**S6** “Then I can wait in the next room.”

**S7** “Not at all. This gentleman, Mr. Wilson, has been my partner and helper in many of my most successful cases, and I have no doubt that he will be of the utmost use to me in yours also.”

**S8** The stout gentleman half rose from his chair and gave a bob of greeting, with a quick little questioning glance from his small fat-encircled eyes.



## S7. Reproducibility and author review

All model outputs, timestamps, request hashes and assessments originate in the retained result artifacts listed in manuscript_manifest.json. This build runs no inference or evaluator. Figure edge manifests preserve original record dictionaries and statuses. Reference details were verified against primary sources on 9 September 2026. The prior visual review pages follow unchanged. Full author name, affiliation, funding/conflict declarations and publication readiness of the AI-authored annotations require author confirmation.
