# Small v2 format-and-counterpart diagnostic (development only)

Eight fixed, single-attempt requests: direct and explicit-interval regression
controls; revised original belief case plus a fresh belief counterpart; revised
original person/office pair plus a fresh matched pair. Original v1 evidence,
questions and references are retained for revised cases. Fresh cases change
names, objects, office responsibilities and interval values before any output.
Paired office evidence is identical. No adaptive later requests or retries.

The compact pathway, pinned model/revision/tokenizer, nonthinking settings and
12,288 context are unchanged. No guided schema or canonical ontology contract.
The only added conventions are a consistent `evidence_ids` list of supplied
strings, decomposed belief facts with holder/attitude, narration without invented
holders, both explicit requested interval bounds, and consistent endpoint names.
One unrelated formatting example is explicitly not evidence. Output allowances
remain 384 (direct), 512 (time/belief), 1,024 (office); exact complete requests and
authored capacity checks are frozen before allocation.

V2 separately measures parseability, required-field compliance, valid citations,
underlying triples, temporal and epistemic qualifications, and complete-fact P/R/F1.
All emitted facts stay in each applicable denominator; duplicate predictions are
penalized. Underlying scores are citation-independent, unlike v1. Full scores
require correct citations and qualification presence. Qualification target recall
includes omissions; applicable-prediction precision includes invented qualifications.
Unmatched wording is unresolved, not positive grounding or automatically false.

`scorer_only/simple_ladder_v2.py` freezes references, relation synonyms and explicit
unambiguous aliases. Case/whitespace normalization and listed article/office-prefix
aliases are prospective v2 policy only. No original model output is re-scored.
Any observed improvement confounds changed prompts/format with this normalization;
it is not an isolated model-capability effect. The .8 P/R criterion is exploratory.

Both office questions still permit the same office-mediated organization. These
pairs therefore cannot require a construction contrast. Report identical graph,
different fact selection within an organization, different semantic organization,
or unresolved assessment, using actual normalized bindings (not cosmetic order).
No high-scoring answer alone establishes registered ontology-construction efficacy.

One new service start and at most eight requests within **417.260137 s**. Historical
actual **9,898.847944 s**, nine development starts and twenty-five reservations
remain charged. Maximum cumulative **10,316.108081 s**. The development-only
complete-forecast exception adds no time; scheduled 33,660 / strict <36,000 remain.
Startup cap 240 s; live checks 15 s; each request 35 s plus 5 s exception drain;
validation 10 s; shutdown 60 s plus 5 s whole-session guard. The whole deadline
dominates. Do not start a request without 110 s remaining before the guarded whole
deadline. Unexecuted cases remain explicit. No extra startup, repeated batch,
ordinary study execution or change to C0, registered validators or held-out review.
