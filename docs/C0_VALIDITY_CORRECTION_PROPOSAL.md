# C0 validity: development-only finding and proposed correction

No gold, matching definition, scientific validator or competence threshold was
changed. Failed calibration and all original comparisons remain preserved.
This is a benchmark-semantic correction proposal, not a passing competence record.
No held-out gold was opened for this investigation.

The 30 already identified validity-only comparisons were traced through evidence,
neutral temporal clues, source `WorldFactSpec`, query scope and gold compilation.
Restricted row-level classification is in
`artifacts/restricted/c0-validity-classification-v1.json`; three readable exact
evidence/prediction/reference examples remain in
`artifacts/restricted/C0_VALIDITY_ONLY_EXAMPLES.md` and
`artifacts/restricted/c0-validity-only-examples-v1.json`. These scorer materials
are deliberately not copied into tracked/public documentation.

26 of the 30 reference intervals are clipped to the query window. Eight have
finite underlying ends that are not stated in the supplied evidence or temporal
clues. Counts overlap. None of these 30 exact reference intervals is fully
explicit in the cited evidence; no authoritative rule supplies those exact
intrinsic bounds. These are counts of the diagnosed development comparisons,
not independent samples or a benchmark-wide prevalence estimate.

Three representative patterns are:

| Evidence support | C0 validity | Reference validity | Finding |
|---|---|---|---|
| Membership observed at story step 1 | start 1, end unknown | [3,9] | Query clipping; evidence alone also does not prove onset at 1 or indefinite persistence |
| Coordination observed at step 5 | not applicable | [5,8] | Finite end 8 exists in latent world data, not the cited evidence/clues |
| Action at a place observed at step 4 | not applicable | [4,5] | Latent end 5 is not stated; story point does not establish this validity duration |

The exact restricted examples identify each entity, evidence ID, assertion ID,
normalized relation and all compared temporal fields. Endpoints/roles, normalized
predicate, direction, story time and applicable epistemic fields already align
in these 30 pairs: differing surface labels are not the cause of these failures.
This does not establish that every C0 validity choice is correct. In particular,
`not_applicable` is not automatically equivalent to unknown validity, and a
state's observed time need not be its onset.

## Authoritative rule and implementation mismatch

Methodological plan §4 (`C0 ClassicalPre`, line 120) forbids creating temporal or
epistemic qualifications after query reveal. §7 (lines 194–203), together with the
assertion representation at line 94, distinguishes story/event time from validity
and requires explicit unknown/underdetermined values. §12 (line 298) requires
matching essential story/validity scope, as well as aligned semantic fields and
supporting evidence. Implementation plan §13.2 (line 543) requires C0 precision
0.85, recall 0.70 on **directly stated development qualified assertions**, with
100% valid references. Neither plan licenses treating a user's display/query
window as a newly evidenced onset or cessation, or reveals hidden world bounds
to C0/C1 after their preontologies are sealed.

In `synthetic_benchmark._temporal_scope`, gold validity starts at the fact's story
position, uses its latent `validity_end`, then intersects those bounds with the
contextual story scope. Every classified row reproduces that intersection.
This is an implementation derivation, not an authoritative semantic inference.
An intersection can legitimately express **visibility within the query**; the
problem is requiring it as exact **intrinsic assertion validity**, especially
when intrinsic endpoints are unsupported. `GoldAlternativeSet`/essential-scope
matching must not silently substitute string equality for this distinction.

## Concrete correction proposed for approval

Separate evidence-supported intrinsic assertion validity from query-window
visibility/restriction. Preserve intrinsic qualification in sealed C0/C1 atoms;
apply the common query restriction in selection/rendering and in a declared,
condition-blind essential-scope comparison, without mutating those atoms.
Do not simply drop validity from strict matching.

Audit the latent finite intervals intended as known-answer targets. Where exact
duration is scientifically required, verbalize it in the shared query-blind
development evidence; otherwise use evidence-supported points/unknown bounds,
not an unspoken latent end. Apply this general rule consistently before any
held-out freeze, not case-by-case to benefit C0. Preserve the old benchmark and
failed records, version the correction, rerun mutation/firewall/evidence and
competence tests, and have the required independent review assess the corrected
interpretations. Thresholds and mandatory temporal comparisons stay unchanged.

This needs an explicit methodological/benchmark-semantic amendment before gold
or scoring changes. C0 remains unqualified for held-out execution until both the
reference inconsistency and any genuine extraction/validity bugs are resolved
and the unchanged competence gate is actually passed.
