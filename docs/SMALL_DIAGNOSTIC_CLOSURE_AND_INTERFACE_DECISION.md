# Small-task series closure and interface decision

2026-09-08. CPU-only follow-up to `64182de`; no new model output or allocation.
The current pinned Qwen3-8B-AWQ configuration failed scientific acceptance on both
latest development repairs despite complete, below-cap, schema/canonical/reference-
valid responses. This is a configuration-and-task result, not universal model
incapability. Original outputs, parents, frozen checker and failure records stand.

## What the repairs did—and did not target

Source: the two exact `small-retry-path-cpu-v4/*/request.json` conversations and
`feedback.json`, versus the immutable before/after graphs in
`artifacts/restricted/parent-repair-session-backup.vaJysn/BEFORE_AFTER_DIAGNOSIS.md`.
This table summarizes those existing findings; it is not a new acceptance review.

| Defect/mechanism | Was it specifically in repair feedback? | Observed outcome |
|---|---|---|
| Missing proposition declarations; nonnull scope combined with world commitment | Yes: `undeclared_reference` and `scope_commitment_conflict`, with exact paths | Both repaired structurally: all referenced contents declared and commitments changed to holder-attributed. This did not establish source support. |
| Unsupported known/reported holder scopes | No instance-specific support diagnostic; the general repair instruction did require positive evidence | First retains knowledge attribution on two assertions; second retains all reports. Supplying declarations preserved unsupported scopes rather than justifying them. |
| Unsupported numerical story/intrinsic/holder time | Not targeted by structural feedback; general time instructions remained present | Old unsupported point qualifications persist. New content records repeat them. |
| Wrong participants/types, predicate direction/upper mapping; six-node excess and revelation leakage in second graph | Not targeted | Second bindings/types/budget error persist; one first-graph carrying binding improves. First direct arrival endpoints improve but its upper mapping is still rejected. |
| Newly introduced first-graph defects | Not pre-existing targets | Attribution changes from the courier to the gate on nA3; arrival becomes disconnected; `schema_relation` now creates only proposition records. Its own object-at-place definition is reversed by gate→seal (manual diagnosis; frozen endpoint recognizer unresolved). |
| Prose/proposition matching limits | No targeted semantic feedback | Frozen combined-clause/proposition checks include unresolved assessments. Some prose is a faithful source paraphrase manually, but graph-support mappings fail independently. Unresolved is neither false nor accepted. |

The feedback could anchor generation on fixing declarations: it named missing
records and commitment consistency, included the full previous answer, and the
first repair's decision rationale explicitly described creating those records.
However, it did **not** require preserving unsupported attribution. Exact general
instructions included “A scope/holder needs positive evidence” and called the prior
answer “untrusted model output, not evidence”; preservation applied to
**evidence-supported** attribution. Anchoring is a plausible contributing mechanism,
not an established causal diagnosis. The model cannot be said to have ignored
instance-specific semantic feedback that was never transmitted.

## Completed CPU candidate: nested content, explicit identity

Implementation: `nested_semantic_candidate.py`; general instruction:
`prompts/diagnostics/nested_content_candidate_v1.md`. Neither is activated by a
condition or controller. Existing typed IDs, canonical/reference checks, primary
metrics, scientific checker, gold and C0 are unchanged.

Complete restricted review package:
`artifacts/restricted/nested-semantic-candidate-cpu-v1/`.
`model-facing/READABLE_REQUEST.json` contains every actual message, complete
development evidence, effective schema and decoding settings. `request.json` is
the exact proposed HTTP body; `generation.schema.json` and `rendered-chat.json`
are separately inspectable. Example: the existing ev-03 mechanic/courier passage,
with no expected graph or previous response in the request.

The candidate keeps all ordinary qualified-draft scientific fields. Binary/n-ary
bindings and proposition time/evidence occur inside an assertion's `content`.
There is no model-authored nP ID or duplicate binding declaration. Attribution is
explicitly optional; direct narrative assertions create no proposition record.
Confidence, relevance and source-confidence judgments are **not** administrative.

| Runtime-derived field | Exact model/source input and unambiguous rule | Model choice / rejection boundary |
|---|---|---|
| Canonical proposition ID and declaration | A nonnull authored scope plus its complete nested content; allocate nP in declaration order | Model supplies predicate, binary endpoints/n-ary roles, temporal content and citations. Missing content/bindings reject. No attribution is invented. |
| Shared versus separate proposition record | `content_identity` explicitly says `distinct`, or `shared` with a key; unasserted records have explicit keys | Equal text alone never merges. Shared keys require identical complete bodies; conflicts reject. Different keys/distinct occurrences stay separate. Identity is not inferred. |
| Assertion binding fields | Copy the single nested predicate/endpoints/roles exactly | Model still connects the graph and chooses direction. Canonical method already requires assertion/content bindings to agree; no endpoint selection or reversal is performed. |
| Assertion/scope proposition references | Both refer to that assertion's lifted content | No dangling reference to an undeclared model nP is possible. Scope, attitude and holder remain model-authored and require grounding. |
| Assertion temporal scope | Copy content time **only** for the explicit literal `"content"`; otherwise preserve the separately authored scope | Different proposition versus assertion times remain expressible. Direct content with unequal time/evidence rejects rather than losing fields. No observation-to-onset inference. |
| Content targets in decisions and temporal anchors | Explicit `content_of_assertion` or `unasserted_content` address | Missing, duplicate or non-attributed destinations reject. Runtime does not decide an operation's scientific meaning or supply targets. |
| Execution decision time | Observed generation completion; marked as execution observation | Not a claim about an internal model decision instant. Model supplies substantive operation, evidence, rationale and targets. |
| Provenance administrative ID/method/locator/source hash | Payload hash + citation position; adapter revision; exact frozen cited-source provenance | Citation and confidence are model-authored. Unknown citations and altered source metadata reject. |
| Versions, content hashes, budgets and token usage | Canonical serialization/version; exact record counts and observed execution usage | No scientific confidence or qualification is derived. CPU examples label synthetic execution placeholders explicitly. |
| Inactive canonical fields and wire tags | Canonical null/empty defaults for the unchosen temporal/binding form; retain representation receipt | No new semantic value. Receipt records identity/address syntax and hashes, not invented content. |

Independent unasserted content remains possible. Supported beliefs, reports,
denials, uncertainty and knowledge can share an explicitly chosen proposition
without global commitment. Story time, intrinsic validity, discourse, revelation,
holder time, spoiler horizon, unknown/not-applicable/open/relative/partial forms,
provenance, descriptions, merge/split/event/type/relation/abstraction decisions
remain explicit. C2 would still construct from evidence and context, not a hidden
graph—but this candidate has **not** been adopted or tested as C2 or FixedSelect.

CPU demonstration: exact candidate→expanded semantic records→candidate equality,
followed by canonical reconstruction, with a representation-only receipt.
Authored controls are exclusively under `authored-controls/`, never transmitted.
They cover binary/n-ary content, all five attitudes, shared/distinct identity,
independent content, addressed targets and temporal alternatives. Unsupported time,
invented attribution and reversed bindings remain failures under the frozen checker.
This does not establish model behavior or backend grammar acceptance.

Verification: **43 focused tests passed**, with the six affected timing/report
tests rerun successfully after the final reporting adjustment. Candidate package
regeneration preserved its immutable hashes. Ruff and diff checks passed. New
decision/prompt text passed the public-text scan; restricted records remain ignored.

Pinned tokenizer/revision: `4da05a8edb55c6046cce958586c33b61da07bb79`;
template SHA-256 `a55ee1b1660128b7098723e0abcd92caa0788061051c62d51cbe87d9cf1974d8`.
Every message plus generation prompt/nonthinking special tokens: **6,102 input**;
reserved output **6,144**; total **12,246 / 12,288**, only **42** input tokens of
slack. Initial verbose wording needed 6,560 input and was rejected, not truncated;
redundant prose was condensed without removing fields. No token-policy change was
activated. Authored standard-spaced output capacities: **857** tokens for the
ev-03 direct example, **2,031 / 2,205** for direct/attributed four-node development
stress fixtures. The attribution stress fixture is a codec/capacity control, not
supported attribution in its direct-narration evidence. These narrow packing
checks are not robust full-size capacity or generation reliability evidence.

## Ledger clarification and eligibility

Nine mismatches are three representation comparisons, the first named-semantic
call, the typed-ID call, two reconciled baselines and two latest repairs. Each has
`gpu_events.succeeded=1`, `model_calls.successful=0`, and an immutable failure.
The diagnostic writer used validation success for a column that the ordinary
fallback path explicitly defines as transport/model-execution completion.

`LEDGER_STATUS_CLARIFICATION.json` adds a read-only derived status for every case,
bound to the ledger and individual call/event/failure hashes. It reports execution
completed, science rejected, and no scientific-acceptance/production-timing/accepted-
result eligibility. No historical bit, failure or microsecond is overwritten.
The strict historical verifier's nine mismatches are **not waived**. Future
diagnostic writes use execution meaning consistently; immutable outcome/failure
records continue to carry scientific meaning separately.

Focused eligibility checks flip either success bit and still reject all nine.
An absent, structural-only, wrong-artifact or diagnostic-only assessment cannot
certify production science. Final accounting now labels execution outcome scope
and scientific status separately, with failure evidence taking precedence.
Ordinary metric execution still requires its projection/validation/lineage path;
failed intended outputs remain in ITT failure accounting, not accepted results.

A demonstrated fallback timing bug was also repaired: invalid base and invalid
repair durations formerly entered successful-output observations after transport
completion. They now remain in actual allocation only. Resume cannot reinsert an
invalid parent's duration. A valid repair contributes its own repair-class timing,
not fabricated base-class throughput. A missing class measurement fails admission;
no replacement inference slot or mandatory-comparison removal was introduced.

## One recommendation

| Route | Evidence for/against it |
|---|---|
| Existing interface with genuinely semantic feedback | Would test feedback not previously supplied. It remains untested, heavily developed on these examples, and cannot presently justify acceptance or the complete-run forecast. |
| Nested representation | Removes an objectively unnecessary declaration/cross-reference burden losslessly. But the latest outputs already resolved that burden; nesting alone does not explain or correct their residual semantic errors. Backend/model behavior and full-size packing remain unproven. |
| Close this configuration as unsuccessful; request a method/model feasibility decision | Matches observed 0/2 semantic acceptance, preserves structural progress and failures, and respects the still-failing resource forecast. **Recommended.** |

Do not initiate another small-task startup or adopt the candidate as production.
Record the current pinned serving/prompt/repair configuration as **not accepted**.
The next decision is whether to authorize a separately preregistered model/protocol
feasibility amendment beyond the exhausted pinned primary/fallback path—not another
identical retry. Such an amendment must identify a fixed model/protocol and revised
acceptance/packing/timing gates, preserve symmetric conditions and all historical
allocation, and obtain resource feasibility before ordinary execution. No model
search, replacement model, budget change or live comparison is proposed as already
authorized. This recommendation does not assert that genuinely semantic feedback
or this CPU candidate could never work.

Actual allocation remains **6,716.108081 s**. Scheduled/hard ceilings remain
**33,660 / strictly before 36,000 s**. The qualified complete-run proxy remains
**46,878.121294 s**, not a production p95 estimate; no new speed credit is taken.
C0 extraction remains passed (104/122, 104/114, 5/5 families, 100% valid evidence);
contextual 54/275 precision, 54/91 recall, F1 .295082 stays separate and unchanged.
Human review, full acceptance, full C1/C2/intact-C1 FixedSelect capacity, model
reliability and complete-study feasibility remain unproven/pending. No new PDF.
