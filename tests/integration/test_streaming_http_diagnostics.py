"""CPU fixture servers exercise actual urllib + production codec, not model inference."""

import json
import threading
import time
from contextlib import contextmanager
from dataclasses import replace
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from story_projection_onto.gpu_runtime import (
    RuntimeTransportError,
    RuntimeWatchdogTimeout,
    VLLMGuidedJSONClient,
)
from story_projection_onto.output_wire import (
    RecordTupleCodec,
    pack_capacity_candidate,
    translate_references,
)
from story_projection_onto.phase1_acceptance import validate_acceptance_generation
from story_projection_onto.streaming_chat import ChatSSE, StreamProtocolError
from tests.integration.test_http_response_diagnostics import ROOT, recovered
from tests.integration.test_http_response_diagnostics import representative as _representative
from tests.unit.test_fallback_acceptance import FakeTokenizer


@pytest.fixture
def representative():
    return _representative.__wrapped__()


def event(value):
    return (
        b"data: "
        + (
            value.encode()
            if isinstance(value, str)
            else json.dumps(value, ensure_ascii=False).encode()
        )
        + b"\n\n"
    )


def delta(text, reason=None):
    return event(
        {
            "id": "cpu-sse",
            "choices": [{"index": 0, "delta": {"content": text}, "finish_reason": reason}],
        }
    )


def frames(request, envelope, mutation=None):
    codec = RecordTupleCodec(
        request.canonical_output_schema,
        request.sealed_record_copies,
        request.opaque_reference_aliases,
    )
    canonical = json.loads(envelope["choices"][0]["message"]["content"])
    if mutation == "science":
        canonical["instance_graph"]["assertions"][0]["temporal_scope"]["story_time"] = {
            "kind": "point",
            "point": 4,
            "label": "passage four",
        }
    wire = codec.encode(
        translate_references(canonical, request.opaque_reference_aliases, decode=False)
    )
    if mutation == "codec":
        wire["draft"].pop()
    text = json.dumps(wire, separators=(",", ":"))
    return [
        delta(""),
        delta(text[:30]),
        delta(text[30:]),
        delta("", "stop"),
        event(
            {
                "id": "cpu-sse",
                "choices": [],
                "usage": {
                    "prompt_tokens": request.rendered_input_token_count,
                    "completion_tokens": 50,
                },
            }
        ),
        event("[DONE]"),
    ]


@contextmanager
def sse_server(parts, *, pause=None, status=200):
    requests = []
    sent_prefix = threading.Event()
    disconnected = threading.Event()

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_):
            pass

        def do_POST(self):
            requests.append(json.loads(self.rfile.read(int(self.headers["Content-Length"]))))
            self.send_response(status)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Authorization", "Bearer header-secret")
            self.send_header("Transfer-Encoding", "chunked")
            self.end_headers()
            try:
                for index, part in enumerate(parts):
                    self.wfile.write(f"{len(part):x}\r\n".encode() + part + b"\r\n")
                    self.wfile.flush()
                    if index == 1 and pause is not None:
                        sent_prefix.set()
                        pause.wait(3)
                        # On timeout the client explicitly shuts down this socket.
                        self.connection.settimeout(0.05)
                        try:
                            if self.connection.recv(1) == b"":
                                disconnected.set()
                                return
                        except TimeoutError:
                            pass
                self.wfile.write(b"0\r\n\r\n")
                self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError):
                disconnected.set()
            self.close_connection = True

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(
        target=server.serve_forever, kwargs={"poll_interval": 0.01}, daemon=True
    )
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}", requests, sent_prefix, disconnected
    finally:
        if pause is not None:
            pause.set()
        server.shutdown()
        server.server_close()
        thread.join(1)


@pytest.mark.parametrize("mutation", [None, "science", "codec"])
def test_actual_stream_reaches_unchanged_validation(tmp_path, representative, mutation):
    tmp_path = tmp_path / "restricted"
    call, bridge, old, envelope = representative
    request = replace(pack_capacity_candidate(old, FakeTokenizer()), stream_response=True)
    parts = frames(request, envelope, mutation)
    with sse_server(parts) as (url, requests, *_):
        client = VLLMGuidedJSONClient(url, diagnostic_root=tmp_path)
        if mutation == "codec":
            with pytest.raises(RuntimeTransportError):
                client.generate(request, watchdog_seconds=5)
        else:
            result = client.generate(request, watchdog_seconds=5)

            def validate():
                return validate_acceptance_generation(
                    root=ROOT,
                    call=call.acceptance_call(),
                    parsed_object=result.parsed_object,
                    authoritative_prompt_tokens=result.prompt_tokens,
                    authoritative_completion_tokens=result.completion_tokens,
                    legacy_provenance_bridge=bridge,
                    diagnostic_journal=result.diagnostic_journal,
                )

            if mutation == "science":
                with pytest.raises(ValueError):
                    validate()
            else:
                assert validate()["grounding_complete"]
        assert requests == [request.wire_payload()]
        assert requests[0]["stream_options"] == {
            "include_usage": True,
            "continuous_usage_stats": False,
        }
    rows, raw = recovered(tmp_path)
    assert raw == b"".join(parts)
    assert any(r["event"] == "stream_done" for r in rows)
    failures = [r["stage"] for r in rows if r["event"] == "failure"]
    assert failures == (
        []
        if mutation is None
        else ["scientific_validation" if mutation == "science" else "decoding"]
    )


@pytest.mark.parametrize("timeout", [False, True])
def test_prefix_is_durable_before_parsing_and_survives_timeout(tmp_path, representative, timeout):
    tmp_path = tmp_path / "restricted"
    _, _, old, envelope = representative
    request = replace(pack_capacity_candidate(old, FakeTokenizer()), stream_response=True)
    parts = frames(request, envelope)
    release = threading.Event()
    results, errors = [], []
    with sse_server(parts, pause=release) as (url, _, prefix, disconnected):

        def run():
            try:
                results.append(
                    VLLMGuidedJSONClient(url, diagnostic_root=tmp_path).generate(
                        request, watchdog_seconds=0.8 if timeout else 5
                    )
                )
            except BaseException as error:
                errors.append(error)

        worker = threading.Thread(target=run)
        worker.start()
        assert prefix.wait(2)
        deadline = time.monotonic() + 0.5
        while True:
            rows, raw = recovered(tmp_path)
            if any(r["event"] == "stream_first_content" for r in rows):
                break
            assert time.monotonic() < deadline
            time.sleep(0.01)
        assert raw == b"".join(parts[:2])
        assert not any(
            r["event"]
            in {
                "response_complete",
                "model_content_json_complete",
                "canonical_reconstruction_passed",
            }
            for r in rows
        )
        if timeout:
            worker.join(2)
            assert len(errors) == 1 and isinstance(errors[0], RuntimeWatchdogTimeout)
            release.set()
            assert disconnected.wait(1)
            rows, raw = recovered(tmp_path)
            assert raw == b"".join(parts[:2])
            assert any(r["event"] == "cancellation_socket_shutdown" for r in rows)
        else:
            release.set()
            worker.join(3)
            assert len(results) == 1 and not errors


@pytest.mark.parametrize("mode", ["malformed", "incomplete", "error", "http500", "utf8"])
def test_failed_stream_preserves_evidence_and_stage(tmp_path, representative, mode):
    tmp_path = tmp_path / "restricted"
    _, _, old, envelope = representative
    request = replace(pack_capacity_candidate(old, FakeTokenizer()), stream_response=True)
    parts = frames(request, envelope)
    parts = {
        "malformed": [event("{oops")],
        "incomplete": parts[:2],
        "error": [event({"error": {"message": "fixture grammar failure"}})],
        "http500": [b"server failure"],
        "utf8": [b"data: \xff\n\n"],
    }[mode]
    with (
        sse_server(parts, status=500 if mode == "http500" else 200) as (url, *_),
        pytest.raises((StreamProtocolError, RuntimeTransportError)),
    ):
        VLLMGuidedJSONClient(url, diagnostic_root=tmp_path).generate(request, watchdog_seconds=5)
    rows, raw = recovered(tmp_path)
    assert raw == b"".join(parts)
    assert {r["stage"] for r in rows if r["event"] == "failure"} == {
        "http" if mode == "http500" else "decoding"
    }


def test_utf8_and_crlf_frames_may_cross_arbitrary_fragment_boundaries():
    parser = ChatSSE()
    raw = b"".join(
        [delta("café"), delta("", "stop"), event({"choices": [], "usage": {}}), event("[DONE]")]
    ).replace(b"\n", b"\r\n")
    for byte in raw:
        parser.feed(bytes([byte]), 1)
    assert parser.envelope()["choices"][0]["message"]["content"] == "café"
