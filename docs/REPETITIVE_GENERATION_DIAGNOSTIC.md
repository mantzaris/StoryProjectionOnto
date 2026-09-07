# Repetitive-generation diagnostic extension

This is the user's cumulative extension of `output-capacity-recovery-v1`, not a
new block: three starts, five attempts, 1,800 allocated seconds including shutdown.
At approval two starts, two attempts and 636.437522 seconds were consumed. The
pre-block baseline remains 3,227.826324 seconds; total actual is 3,864.263846.
The remaining authority is one start, three attempts and 1,163.562478 seconds.
The 33,660 scheduled ceiling and strict actual stop before 36,000 remain unchanged.
The original nine-hour target was not met. Only complete-forecast admission for
this diagnostic block is excepted; no ordinary execution is admitted by it.

## CPU inspection and scoped repair

Both historical responses completed HTTP 200 but exhausted 6,144 tokens without
complete JSON. In the first, after six BudgetAccounting integers the grammar
allowed arbitrary whitespace before the trailing optional-field object. The
second used the working global whitespace restriction and entered an unrestricted
first assertion identifier string; its repeated `0, ` was permitted inside that
string. Its preceding interpretation string was already nonsensical. Neither is
evidence of useful scientific throughput or simply a too-small graph allowance.

The exact model-facing prompt contained required tuple **names**, but no types,
enumerations or optional-field definitions. The decoder schema was not included
as model-visible text. Thus the model had no explicit description of such fields
as subject/object endpoints, holder qualification, or temporal optional values.
The new legend derives every required position, optional field, type, enum and
default from the same schema the codec/decoder use. Tuple order is unchanged.
Readable complete compact JSON is tried first for all conditions. The previous
reversible input tables are used only if the complete readable input exceeds its
existing cap. No evidence, qualification or graph is truncated. Output caps remain
6,144 for constructive conditions and 3,072 for sealed selection.

Only identifier strings receive the activated length/alphabet constraint, including
legitimate slash-qualified IDs. Closed enums remain exact; descriptions are not
pattern-constrained. The supported global whitespace restriction remains enabled.

An initially suspected provenance-hash regex mismatch was **not active in the
historical production requests**: the existing compatibility transform had removed
that regex. The general codec now admits exact bound handles even when a canonical
pattern survives, but this latent fix is not an explanation of the observed failures.

Installed vLLM 0.10.2 V1 source passes `guided_json` to XGrammar and uses the global
`disable_any_whitespace` flag in `compile_json_schema`. The previous service log
confirms xgrammar/no-fallback/whitespace-restricted configuration. The pinned
tokenizer/template is unchanged; non-thinking rendering ends with the assistant
prefix and closed empty think block. EOS/turn-end is 151645, tokenizer BOS is not
added, and the launcher uses `--generation-config vllm` with explicit sampling
settings (temperature .7, top-p .8, top-k 20, disabled penalties/min-p, seed 0).
No reasoning text was present in either failed response.

## Execution decision frozen before the third start

The concrete prompt gap justifies testing the repaired full C1 first. Next is the
independent full C2 diagnostic, a discriminating contextual construction test even
if C1 fails. The third call is FixedSelect over the genuine accepted and immediately
sealed C1, if available; otherwise it is the prepared one-record evidence-only C1
diagnostic to distinguish basic codec/generation failure from full complexity.
There are no identical retries. A small diagnostic cannot satisfy production
acceptance. Transport/HTTP failure stops use of a potentially unhealthy service.

The authoritative implementation plan §§3–5 and its condition boundary require an
empty C2 pre-query ontology, not an accepted C1 before C2. This dependency was only
in the diagnostic controller. C2 query preparation follows completion of C1 (and
its seal if accepted); it receives no C1 objects. FixedSelect's dependency remains.
All calls still use the same scientific validators. The small test uses the same
production structural validator with only its supplied evidence admissible and
the existing restricted development semantic audit; no answer is supplied.

Full acceptance additionally requires the second C2 seed, applicable repair,
packing/resource gates and registered restart/resume verification. None is waived.
The failed-output capacity-ratio forecast is still an unmeasured conservative
sensitivity, not a measured p95. Remaining work before this start is 39,828.344344
seconds plus a 333.668869 acceptance/resume service envelope after shutdown:
40,162.013213 total. No reserve slot is reclassified and no required call is removed.

Raw requests, rendered chats, hashes, HTTP fragments, parsing/validation outcomes,
resource samples and allocation records stay in ignored restricted storage. The
CPU preparation check took 39.660042 seconds with zero GPU allocation. CPU checks
do not prove model comprehension, reliable completion, or successful throughput.
