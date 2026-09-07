"""Bounded parser for pinned vLLM OpenAI chat SSE, never an alternate model backend."""

from __future__ import annotations

import json
import re

from story_projection_onto.http_diagnostics import MAX_RESPONSE_BYTES


class StreamProtocolError(ValueError):
    pass


class ChatSSE:
    """Parse only after the caller has durably saved each received byte fragment."""

    def __init__(self, journal=None):
        self.journal = journal
        self.pending = b""
        self.content = []
        self.reasoning = []
        self.usage = None
        self.finish_reason = None
        self.response_id = None
        self.done = False
        self.event_count = 0
        self.first_event_seconds = None
        self.first_content_seconds = None
        self.received = 0

    def event(self, name, **fields):
        if self.journal is not None:
            self.journal.event(name, **fields)

    def feed(self, fragment: bytes, elapsed: float):
        self.received += len(fragment)
        if self.received > MAX_RESPONSE_BYTES:
            raise StreamProtocolError("stream exceeds bounded response bytes")
        self.pending += fragment
        while match := re.search(rb"\r?\n\r?\n", self.pending):
            block, self.pending = self.pending[: match.start()], self.pending[match.end() :]
            lines = [
                line[5:].lstrip(b" ") for line in block.splitlines() if line.startswith(b"data:")
            ]
            if not lines:
                continue
            if self.done:
                raise StreamProtocolError("data after SSE DONE")
            self.event_count += 1
            if self.first_event_seconds is None:
                self.first_event_seconds = elapsed
                self.event("stream_first_event", elapsed_seconds=elapsed)
            try:
                data = b"\n".join(lines).decode("utf-8")
            except UnicodeDecodeError as error:
                raise StreamProtocolError("invalid UTF-8 in SSE event") from error
            if data == "[DONE]":
                self.done = True
                self.event("stream_done", elapsed_seconds=elapsed, event_count=self.event_count)
                continue
            try:
                record = json.loads(data)
            except ValueError as error:
                raise StreamProtocolError("malformed SSE JSON event") from error
            if not isinstance(record, dict) or "error" in record:
                raise StreamProtocolError("server emitted a streaming error event")
            if record.get("id") is not None:
                if self.response_id is not None and self.response_id != record["id"]:
                    raise StreamProtocolError("response ID changed during stream")
                self.response_id = record["id"]
            choices = record.get("choices")
            if not isinstance(choices, list) or len(choices) > 1:
                raise StreamProtocolError("stream must contain one choice or usage-only event")
            if record.get("usage") is not None:
                self.usage = record["usage"]
                self.event("stream_usage", usage=self.usage, elapsed_seconds=elapsed)
            for choice in choices:
                if (
                    not isinstance(choice, dict)
                    or choice.get("index") != 0
                    or not isinstance(choice.get("delta"), dict)
                ):
                    raise StreamProtocolError("invalid streamed choice index/delta")
                delta = choice["delta"]
                if delta.get("tool_calls"):
                    raise StreamProtocolError("tool calls are outside ontology JSON response")
                for field, target in (
                    ("content", self.content),
                    ("reasoning_content", self.reasoning),
                ):
                    value = delta.get(field)
                    if value is not None:
                        if not isinstance(value, str):
                            raise StreamProtocolError("streamed text delta must be a string")
                        if value and field == "content" and self.first_content_seconds is None:
                            self.first_content_seconds = elapsed
                            self.event("stream_first_content", elapsed_seconds=elapsed)
                        target.append(value)
                reason = choice.get("finish_reason")
                if reason is not None:
                    if not isinstance(reason, str) or self.finish_reason is not None:
                        raise StreamProtocolError("invalid or repeated finish reason")
                    self.finish_reason = reason
                    self.event("stream_finish", finish_reason=reason, elapsed_seconds=elapsed)

    def envelope(self):
        if (
            self.pending.strip()
            or not self.done
            or self.finish_reason is None
            or self.usage is None
        ):
            raise StreamProtocolError(
                "incomplete SSE response: require final choice, usage and DONE"
            )
        if self.reasoning and "".join(self.reasoning):
            raise StreamProtocolError("unexpected reasoning in non-thinking diagnostic")
        return {
            "id": self.response_id,
            "choices": [
                {"message": {"content": "".join(self.content)}, "finish_reason": self.finish_reason}
            ],
            "usage": self.usage,
        }
