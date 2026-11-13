# Minimal synthetic ladder — development-only amendment

Eight pedagogical requests and a conditional replacement plain-language control
are frozen in `simple_ladder.py`; references and conservative scoring are separate
in `scorer_only/simple_ladder.py`. No expected answers enter model requests. This
is not the original synthetic benchmark, registered C1/C2, a canonical ontology,
or a change to registered validators or held-out acceptance.

The existing pinned Qwen3-8B-AWQ, tokenizer, 12,288 context and nonthinking sampling
are unchanged. All requests use unconstrained streaming text, requested as tiny
JSON except the conditional plain-language control. No guided schema, ontology
certificate, field guide, aliases or semantic adapter is used. Only an exact
surrounding Markdown fence may be removed; incomplete JSON is never completed.

Inputs are capped at 10,240 tokens but measured prompts must remain genuinely
small; output caps are 384 for facts/selection, 512 for time/belief, 1,024 for
office contrast and 256 for the control. Freeze all exact rendered payloads and
counts before allocation. Authored capacity checks are not model outcomes.

Progress beyond the two direct cases requires combined precision and recall at
least .8. Otherwise run only the prepared plain-language control, then stop.
All predictions, including duplicates/unresolved wording, remain in denominators.
Complete temporal/epistemic facts and underlying relationships are scored
separately. The office cases permit predeclared office-mediated or combined
representations; the person-centered case also permits direct responsibility
links with holder intervals and joint citations. These are diagnostic alternatives,
not a post-output matching amendment. Selection success is not construction proof.

One additional service start, at most eight requests and at most **574.607992 s**
remain inside the SAME development allocation. Historical **9,741.500089 s**,
eight starts and seventeen reservations are retained. Maximum cumulative actual
allocation is **10,316.108081 s**; global scheduled 33,660 / strict actual <36,000
remain unchanged. The diagnostic complete-forecast exception adds no time.
Startup cap 300 s; live checks 15 s; each generation 45 s + 5 s exception drain;
validation/bookkeeping 10 s; shutdown reserve 60 s plus 5 s guard. The whole-session
deadline takes precedence. No new request starts without 120 s remaining before
the guarded whole deadline. The same existing guardian supervises the workload.

Stop after the ladder/control, unsafe service state, or remaining-time exhaustion.
No automatic return to full ontology generation. Toy sentence parsing is expressly
not C0. Existing C0 competence, contextual scores and human-review gate are unchanged.
