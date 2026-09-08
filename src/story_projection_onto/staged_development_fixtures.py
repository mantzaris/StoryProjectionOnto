"""CPU-only capacity specimen, never a source for live graph construction.

The fixed authored specimen uses independent courier test semantics. Reference
aliases are mechanically replaced with development-index vocabulary ONLY to
exercise grammar/packing; this does not assert evidence support or score it.
"""

from .development_demo import EvidenceRecord, aliases, read, reference_translation, sources


def capacity_fixture(root):
    value = read(root / "tests/fixtures/staged_capacity.json")
    _, _, neutral = sources(root)
    evidence = tuple(EvidenceRecord.model_validate(e) for e in neutral["evidence"])
    vocabulary = aliases(evidence)
    by_prefix = {p: [v for v in vocabulary.values() if v.startswith(p)] for p in "emvrt"}
    references = {}

    def visit(v, key=""):
        if isinstance(v, dict):
            for k, x in v.items():
                visit(x, k)
        elif isinstance(v, list):
            for x in v:
                visit(x, key)
        elif isinstance(v, str) and key.endswith(("_id", "_ids")) and not v.startswith("n"):
            p = (
                "e"
                if v.startswith("ev-")
                else "m"
                if v.startswith("m-")
                else "v"
                if v.startswith("event-")
                else "r"
                if v.startswith("relation-")
                else None
            )
            if p:
                references.setdefault(
                    v,
                    by_prefix[p][
                        len([x for x in references.values() if x.startswith(p)]) % len(by_prefix[p])
                    ],
                )

    visit(value)
    return reference_translation(value, references)
