"""CPU-only socket tests: fixture responses, never a model service or inference."""

from __future__ import annotations

import json
import socket
import threading
import time
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

from story_projection_onto.fallback_acceptance import (
    _failure_kind,
    build_fallback_acceptance_request,
    fallback_pilot_calls,
)
from story_projection_onto.gpu_runtime import (
    RuntimeTransportError,
    RuntimeWatchdogTimeout,
    VLLMGuidedJSONClient,
)
from story_projection_onto.http_diagnostics import (
    ResponseEvidenceLimitError,
    RestrictedResponseJournal,
)
from story_projection_onto.model_gate import FallbackModelPolicy
from story_projection_onto.phase1_acceptance import validate_acceptance_generation
from story_projection_onto.phase1_legacy_provenance import Phase1LegacyEvidenceProvenanceBridge
from story_projection_onto.store import ArtifactRecord, FailureKind, ReleaseClass
from tests.unit.test_fallback_acceptance import FakeTokenizer, fallback_tokenizer_manifest

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture
def representative():
    call = fallback_pilot_calls(
        FallbackModelPolicy.load(ROOT / "configs/study/fallback_model.json")
    )[0]
    bridge = Phase1LegacyEvidenceProvenanceBridge.load(ROOT)
    request = build_fallback_acceptance_request(
        root=ROOT,
        call=call,
        tokenizer=FakeTokenizer(),
        tokenizer_manifest=fallback_tokenizer_manifest(),
        legacy_provenance_bridge=bridge,
    )
    draft = json.loads((ROOT / "tests/fixtures/phase1/c1_pre_output.json").read_bytes())
    source_hashes = {
        item["evidence_id"]: item["provenance"]["source_artifact_hash"]
        for item in json.loads(request.messages[1].content)["legacy_evidence_provenance"]["records"]
    }
    for assertion in draft["instance_graph"]["assertions"]:
        for provenance in assertion["provenance"]:
            provenance["source_artifact_hash"] = source_hashes[provenance["evidence_id"]]
    envelope = {
        "id": "cpu-fixture-response",
        "choices": [{"message": {"content": json.dumps(draft)}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": request.rendered_input_token_count, "completion_tokens": 50},
    }
    return call, bridge, request, envelope


@contextmanager
def response_server(body: bytes, *, status=200, mode="normal"):
    requests = []
    finished = threading.Event()

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_):
            pass

        def do_POST(self):
            requests.append(
                (
                    self.path,
                    dict(self.headers),
                    self.rfile.read(int(self.headers["Content-Length"])),
                )
            )
            try:
                if mode == "before_headers_timeout":
                    time.sleep(0.3)
                    return
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("X-Request-ID", "cpu-test")
                self.send_header("Authorization", "Bearer header-secret")
                self.send_header("Set-Cookie", "session=cookie-secret")
                if mode == "redirect":
                    self.send_header("Location", "http://127.0.0.1:9/must-not-follow")
                if mode == "chunked_incomplete":
                    self.send_header("Transfer-Encoding", "chunked")
                else:
                    self.send_header(
                        "Content-Length", str(len(body) + (100 if mode == "incomplete" else 0))
                    )
                self.end_headers()
                if mode == "chunked_incomplete":
                    self.wfile.write(b"100\r\n" + body)
                elif mode == "body_timeout":
                    self.wfile.write(body[:7])
                    self.wfile.flush()
                    time.sleep(0.3)
                    return
                else:
                    self.wfile.write(body)
                self.wfile.flush()
                self.close_connection = True
            except (BrokenPipeError, ConnectionResetError):
                pass
            finally:
                finished.set()

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(
        target=server.serve_forever, kwargs={"poll_interval": 0.01}, daemon=True
    )
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}", requests
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=1)
        finished.wait(1)


def recovered(root):
    paths = list(root.glob("*/events.jsonl"))
    assert len(paths) == 1
    events = [json.loads(line) for line in paths[0].read_text().splitlines()]
    import zstandard

    fragments = []
    for event in events:
        if event["event"] == "response_fragment":
            record = ArtifactRecord(**event["artifact"])
            assert record.release_class == ReleaseClass.RESTRICTED
            raw = zstandard.ZstdDecompressor().decompress(
                (paths[0].parent / "fragments" / record.relative_path).read_bytes()
            )
            fragments.append(raw)
    assert b"header-secret" not in paths[0].read_bytes()
    assert b"cookie-secret" not in paths[0].read_bytes()
    return events, b"".join(fragments)


@pytest.mark.parametrize(
    "mutation,expected_stage",
    [("none", None), ("schema", "schema_validation"), ("science", "scientific_validation")],
)
def test_real_fallback_http_path_reaches_unchanged_validators(
    tmp_path, representative, mutation, expected_stage
):
    call, bridge, request, envelope = representative
    draft = json.loads(envelope["choices"][0]["message"]["content"])
    if mutation == "schema":
        del draft["instance_graph"]
    elif mutation == "science":
        draft["instance_graph"]["assertions"][0]["temporal_scope"]["story_time"] = {
            "kind": "point",
            "point": 4,
            "label": "passage four",
        }
    envelope["choices"][0]["message"]["content"] = json.dumps(draft)
    body = json.dumps(envelope).encode()
    diagnostic_root = tmp_path / "restricted" / "responses"
    with response_server(body) as (url, requests):
        generated = VLLMGuidedJSONClient(url, diagnostic_root=diagnostic_root).generate(
            request, watchdog_seconds=5
        )
        before, raw = recovered(diagnostic_root)
        assert raw == body
        assert any(e["event"] == "response_complete" for e in before)
        assert not any(e["event"] == "validation_started" for e in before)
        assert requests[0][0] == "/v1/chat/completions"
        assert json.loads(requests[0][2]) == request.wire_payload()
        assert "Authorization" not in requests[0][1]

        def validate():
            return validate_acceptance_generation(
                root=ROOT,
                call=call.acceptance_call(),
                parsed_object=generated.parsed_object,
                authoritative_prompt_tokens=generated.prompt_tokens,
                authoritative_completion_tokens=generated.completion_tokens,
                legacy_provenance_bridge=bridge,
                diagnostic_journal=generated.diagnostic_journal,
            )

        if expected_stage:
            with pytest.raises(ValueError):
                validate()
        else:
            assert validate()["grounding_complete"] is True
    after, raw = recovered(diagnostic_root)
    failures = [e for e in after if e["event"] == "failure"]
    assert [e["stage"] for e in failures] == ([] if expected_stage is None else [expected_stage])
    assert raw == body
    assert "diagnostic" not in json.dumps(generated.public_manifest())


@pytest.mark.parametrize(
    "case,status,mode,stage",
    [
        ("outer_json", 200, "normal", "decoding"),
        ("invalid_utf8", 200, "normal", "decoding"),
        ("inner_json", 200, "normal", "decoding"),
        ("two_choices", 200, "normal", "decoding"),
        ("bad_usage", 200, "normal", "decoding"),
        ("array", 200, "normal", "decoding"),
        ("normal", 200, "incomplete", "transport"),
        ("outer_json", 200, "chunked_incomplete", "transport"),
        ("normal", 400, "normal", "http"),
        ("normal", 500, "normal", "http"),
        ("outer_json", 503, "normal", "http"),
        ("normal", 302, "redirect", "http"),
    ],
)
def test_real_http_failures_retain_exact_bytes_and_failure_chain(
    tmp_path, representative, case, status, mode, stage
):
    _, _, request, envelope = representative
    if case == "inner_json":
        envelope["choices"][0]["message"]["content"] = '{"unfinished":'
        envelope["choices"][0]["finish_reason"] = "length"
    elif case == "two_choices":
        envelope["choices"] *= 2
    elif case == "bad_usage":
        envelope["usage"]["completion_tokens"] = -1
    elif case == "array":
        envelope["choices"][0]["message"]["content"] = "[]"
    body = (
        b'{"unfinished":'
        if case == "outer_json"
        else b"\xff"
        if case == "invalid_utf8"
        else json.dumps(envelope).encode()
    )
    root = tmp_path / "restricted" / "responses"
    with (
        response_server(body, status=status, mode=mode) as (url, requests),
        pytest.raises(RuntimeTransportError) as caught,
    ):
        VLLMGuidedJSONClient(url, diagnostic_root=root).generate(request, watchdog_seconds=5)
    events, raw = recovered(root)
    assert raw == body
    failure = next(e for e in events if e["event"] == "failure")
    assert failure["stage"] == stage
    assert caught.value.failure_stage == stage
    assert _failure_kind(caught.value) is (
        FailureKind.INVALID_OUTPUT if stage == "decoding" else FailureKind.SERVICE
    )
    assert failure["exception_chain"]["frames"]
    assert next(e for e in events if e["event"] == "response_headers")["http_status"] == status
    if case in {"outer_json", "inner_json", "invalid_utf8"} and stage == "decoding":
        assert failure["exception_chain"]["cause"] is not None
    if case == "inner_json":
        assert (
            next(e for e in events if e["event"] == "completion_metadata")["finish_reason"]
            == "length"
        )
    assert len(requests) == 1


@pytest.mark.parametrize("mode", ["before_headers_timeout", "body_timeout"])
def test_actual_http_timeout_keeps_fragments_already_received(tmp_path, representative, mode):
    _, _, request, envelope = representative
    body = json.dumps(envelope).encode()
    root = tmp_path / "restricted" / "responses"
    with response_server(body, mode=mode) as (url, _), pytest.raises(RuntimeWatchdogTimeout):
        VLLMGuidedJSONClient(url, diagnostic_root=root).generate(request, watchdog_seconds=0.15)
    events, raw = recovered(root)
    assert raw == (body[:7] if mode == "body_timeout" else b"")
    assert any(e["event"] == "failure" and e["stage"] == "transport" for e in events)
    assert not any(e["event"] == "response_complete" for e in events)


def test_refused_connection_is_distinct_from_http_failure(tmp_path, representative):
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    root = tmp_path / "restricted" / "responses"
    with pytest.raises(RuntimeTransportError):
        VLLMGuidedJSONClient(f"http://127.0.0.1:{port}", diagnostic_root=root).generate(
            representative[2], watchdog_seconds=2
        )
    events, raw = recovered(root)
    assert raw == b""
    assert not any(e["event"] == "response_headers" for e in events)
    assert next(e for e in events if e["event"] == "failure")["exception_chain"]["cause"]


def test_journal_bound_secrets_and_full_exception_chain(tmp_path):
    root = tmp_path / "restricted" / "responses"
    journal = RestrictedResponseJournal(root, request_hash="a" * 64, maximum_bytes=4)
    journal.headers(
        200, {"Authorization": "Bearer header-secret", "Set-Cookie": "session=cookie-secret"}
    )
    with pytest.raises(ResponseEvidenceLimitError):
        journal.fragment(b"12345")
    try:
        try:
            raise ValueError("Authorization: Bearer header-secret")
        except ValueError as error:
            raise RuntimeError("Cookie: session=cookie-secret") from error
    except RuntimeError as error:
        journal.failure("decoding", error)
    events, raw = recovered(root)
    assert raw == b"1234"
    assert (
        next(e for e in events if e["event"] == "failure")["exception_chain"]["cause"]["type"]
        == "builtins.ValueError"
    )
    with pytest.raises(ValueError, match="restricted"):
        RestrictedResponseJournal(tmp_path / "public", request_hash="a" * 64)


def test_response_is_durable_before_the_actual_decoder_runs(tmp_path, representative, monkeypatch):
    root = tmp_path / "restricted" / "responses"
    body = json.dumps(representative[3]).encode()
    original = VLLMGuidedJSONClient._decode_generation_response
    reached = []

    def decode(request, status, received, headers, **kwargs):
        events, durable = recovered(root)
        assert durable == received == body
        assert events[-1]["event"] == "response_complete"
        reached.append(True)
        return original(request, status, received, headers, **kwargs)

    monkeypatch.setattr(VLLMGuidedJSONClient, "_decode_generation_response", staticmethod(decode))
    with response_server(body) as (url, _):
        VLLMGuidedJSONClient(url, diagnostic_root=root).generate(
            representative[2], watchdog_seconds=5
        )
    assert reached == [True]


def test_restricted_location_cannot_escape_through_symlink(tmp_path):
    (tmp_path / "public").mkdir()
    (tmp_path / "restricted").symlink_to(tmp_path / "public", target_is_directory=True)
    with pytest.raises(ValueError, match="non-symlink"):
        RestrictedResponseJournal(tmp_path / "restricted" / "responses", request_hash="a" * 64)


def test_fragment_bound_prevents_unbounded_tiny_fragment_metadata(tmp_path, monkeypatch):
    import story_projection_onto.http_diagnostics as diagnostics

    monkeypatch.setattr(diagnostics, "MAX_RESPONSE_FRAGMENTS", 2)
    root = tmp_path / "restricted" / "responses"
    journal = RestrictedResponseJournal(root, request_hash="a" * 64)
    journal.fragment(b"a")
    journal.fragment(b"b")
    with pytest.raises(ResponseEvidenceLimitError, match="fragment limit"):
        journal.fragment(b"c")
    assert recovered(root)[1] == b"ab"
