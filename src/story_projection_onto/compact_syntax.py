"""Bounded JSON syntax recovery; never fills content or resolves duplicate keys."""

import hashlib
import json


def _pairs(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key!r}")
        result[key] = value
    return result


def strict_parse(text):
    def reject_constant(value):
        raise ValueError("non-JSON constant: " + value)

    try:
        value = json.loads(text, object_pairs_hook=_pairs, parse_constant=reject_constant)
        if not isinstance(value, dict) or not isinstance(value.get("facts"), list):
            raise ValueError("JSON must contain a facts list")
        return value, None
    except (TypeError, ValueError) as exc:
        return None, str(exc)


def recover(text):
    """Delete only a comma after a value and before ]/}, outside strings.

    Edits are applied only if the complete edited document passes strict JSON
    and duplicate-key checks. UTF-8 offsets address the unchanged original text.
    """
    strict, strict_error = strict_parse(text)

    def digest(value):
        return hashlib.sha256(value.encode("utf-8")).hexdigest()

    record = {
        "strict_parseable": strict is not None,
        "strict_error": strict_error,
        "raw_text_sha256": digest(text),
        "edits": [],
        "applied": False,
        "recovered_text": None,
        "recovered_parseable": strict is not None,
        "recovery_error": None,
    }
    if strict is not None:
        return strict, record
    quoted = escaped = False
    candidate_offsets = []
    for i, char in enumerate(text):
        if quoted:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                quoted = False
            continue
        if char == '"':
            quoted = True
        elif char == ",":
            after, before = i + 1, i - 1
            while after < len(text) and text[after] in " \t\r\n":
                after += 1
            while before >= 0 and text[before] in " \t\r\n":
                before -= 1
            if (
                after < len(text)
                and text[after] in "]}"
                and before >= 0
                and text[before] in '"}]0123456789el'
            ):
                candidate_offsets.append(i)
    remove = set(candidate_offsets)
    candidate = "".join(char for i, char in enumerate(text) if i not in remove)
    parsed, error = strict_parse(candidate)
    if not candidate_offsets or parsed is None:
        record["recovery_error"] = error or strict_error
        return None, record
    record.update(
        applied=True,
        recovered_parseable=True,
        recovered_text=candidate,
        recovered_text_sha256=digest(candidate),
        edits=[
            {
                "byte_offset": len(text[:i].encode("utf-8")),
                "character_offset": i,
                "removed": ",",
                "replacement": "",
            }
            for i in candidate_offsets
        ],
    )
    return parsed, record
