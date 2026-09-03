# Diagnostic-only bounded repair prompt (v1.0.0)

You are performing the single permitted repair of one ontology draft. Thinking mode
is disabled. Return exactly one complete JSON object conforming to the same supplied
output schema. Do not emit analysis, Markdown, comments, a patch, or fields absent
from the schema.

The runner supplies: the original semantic request, the original raw draft, and a
fact-free list of validator diagnostics containing only codes, field paths, messages,
and related IDs. Diagnostics identify defects; they never prescribe replacement facts.
Use the original evidence and the original condition's capability allowlist to repair
only those diagnosed defects.

Do not introduce a fact, entity, event, identity decision, predicate, temporal value,
epistemic commitment, evidence citation, or factual description that is not supported
by the original request. Do not broaden the query, read outside the original packet,
change the seed/configuration, relax a horizon, or use scorer information. For
`A-FixedSelect`, selection-only prohibitions remain in force. If no supported repair
exists, abstain or omit the defective content rather than inventing semantics.

Preserve all valid unaffected content. Keep repair lineage direct from the base
attempt. This is repair attempt 1 of 1; no second repair is permitted. The runner will
validate the entire returned draft again and will retain both the invalid base output
and this attempt.
