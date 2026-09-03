from __future__ import annotations

import hashlib
import json
import signal
import subprocess
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace
from typing import cast

import pytest

from story_projection_onto.contracts import ConditionName, canonical_sha256
from story_projection_onto.experiment import AllocatedGPUMeter, ResourceLimits
from story_projection_onto.gpu_runtime import (
    PINNED_MODEL_REVISION,
    SERVED_MODEL_NAME,
    ChatMessage,
    GenerationResult,
    GuidedJSONRequest,
    ProcessTreeUsage,
    ResourceSampler,
    ResourceWatchdog,
    RuntimeConfigurationError,
    RuntimeResourceLimitExceeded,
    RuntimeTransportError,
    RuntimeWatchdogTimeout,
    ServiceState,
    TokenizerManifest,
    VLLMGuidedJSONClient,
    VLLMLaunchConfiguration,
    VLLMService,
    capture_runtime_stack,
    capture_tokenizer_manifest,
    sample_gpu_vram,
    sample_process_tree,
)
from story_projection_onto.llm import DecodingManifest, PackingReport, PackingSection
from story_projection_onto.phase1_acceptance import (
    EXPECTED_ACCEPTANCE_COUNTS,
    AcceptanceRunner,
    acceptance_plan_manifest,
    build_acceptance_request,
    phase1_acceptance_calls,
    validate_acceptance_calls,
    validate_acceptance_generation,
    validate_verified_model_manifest,
)
from story_projection_onto.phase1_acceptance import (
    main as acceptance_main,
)
from story_projection_onto.store import (
    ArtifactStore,
    BlobStore,
    Compression,
    GpuBudgetExceeded,
    GpuEventKind,
    Ledger,
    StorageBudget,
    StoragePreflight,
)

HASH_A = "a" * 64
HASH_B = "b" * 64
ROOT = Path(__file__).parents[2]


def limits(**changes: object) -> ResourceLimits:
    values: dict[str, object] = {
        "schema_version": "1.0.0",
        "maximum_cpu_workers": 8,
        "maximum_process_ram_bytes": 25_000_000_000,
        "maximum_peak_vram_bytes": 23 * 1024**3,
        "maximum_project_occupied_bytes": 25_000_000_000,
        "minimum_storage_headroom_bytes": 5_000_000_000,
        "maximum_project_allocation_bytes": 30_000_000_000,
        "scheduled_gpu_seconds": 32_400,
        "preferred_forecast_gpu_seconds": 29_700,
        "hard_gpu_seconds": 36_000,
        "model_cpu_offload_allowed": False,
        "generation_concurrency": 1,
    }
    values.update(changes)
    return ResourceLimits.from_mapping(values)


@pytest.fixture
def launch_configuration(tmp_path: Path) -> VLLMLaunchConfiguration:
    cache = tmp_path / "cache"
    snapshot = cache / "hub" / "model" / "snapshots" / PINNED_MODEL_REVISION
    snapshot.mkdir(parents=True)
    return VLLMLaunchConfiguration(snapshot_path=snapshot, shared_cache=cache)


def test_launcher_is_pinned_local_concurrency_one_and_no_offload(
    launch_configuration: VLLMLaunchConfiguration,
) -> None:
    command = launch_configuration.command("/controlled/python")
    environment = launch_configuration.environment({"PATH": "/bin"})

    assert command[:3] == (
        "/controlled/python",
        "-m",
        "vllm.entrypoints.openai.api_server",
    )
    for flag, value in {
        "--tensor-parallel-size": "1",
        "--max-model-len": "12288",
        "--gpu-memory-utilization": "0.88",
        "--cpu-offload-gb": "0",
        "--max-num-seqs": "1",
        "--dtype": "half",
    }.items():
        assert command[command.index(flag) + 1] == value
    assert "--no-enable-prefix-caching" in command
    assert "--no-enable-log-requests" in command
    assert "--disable-log-requests" not in command
    assert not any("speculative" in argument for argument in command)
    assert environment["HF_HUB_OFFLINE"] == "1"
    assert environment["TRANSFORMERS_OFFLINE"] == "1"
    assert {
        environment[name]
        for name in ("HF_HOME", "HF_HUB_CACHE", "TRANSFORMERS_CACHE", "VLLM_CACHE_ROOT")
    } == {str(launch_configuration.shared_cache)}
    assert environment["OMP_NUM_THREADS"] == "8"
    public = launch_configuration.public_manifest()
    assert not any(
        str(launch_configuration.snapshot_path) in str(value) for value in public.values()
    )
    assert public["cpu_offload_gb"] == 0
    assert public["prefix_caching"] is False
    assert public["speculative_decoding"] is False

    with pytest.raises(RuntimeConfigurationError, match="cpu_offload_gb"):
        VLLMLaunchConfiguration(
            snapshot_path=launch_configuration.snapshot_path,
            shared_cache=launch_configuration.shared_cache,
            cpu_offload_gb=1,
        )


class FakeTokenizer:
    eos_token_id = 7
    unk_token_id = 0
    chat_template = "{{ messages }}<|im_end|>{% if enable_thinking %}think{% endif %}"

    def convert_tokens_to_ids(self, token: str) -> int:
        return {"<|im_end|>": 8}.get(token, 0)

    def encode(self, text: str, **kwargs: object) -> list[int]:
        return list(range(len(text.split())))

    def apply_chat_template(self, conversation: object, **kwargs: object) -> str | list[int]:
        assert kwargs["enable_thinking"] is False
        assert isinstance(conversation, (list, tuple))
        contents = [str(message["content"]) for message in conversation]
        rendered = " chat_turn ".join(("chat_start", *contents, "assistant_start"))
        if kwargs["tokenize"]:
            return list(range(len(rendered.split())))
        return rendered


def test_tokenizer_capture_is_local_exact_nonthinking_and_public_safe(tmp_path: Path) -> None:
    snapshot = tmp_path / PINNED_MODEL_REVISION
    snapshot.mkdir()
    (snapshot / "tokenizer.json").write_text("{}")
    (snapshot / "tokenizer_config.json").write_text("{}")
    observed: dict[str, object] = {}

    def loader(path: str, **kwargs: object) -> FakeTokenizer:
        observed.update({"path": path, **kwargs})
        return FakeTokenizer()

    manifest = capture_tokenizer_manifest(snapshot, tokenizer_loader=loader)

    assert observed["local_files_only"] is True
    assert observed["trust_remote_code"] is False
    assert observed["revision"] == PINNED_MODEL_REVISION
    assert manifest.eos_token_id == 7
    assert manifest.end_of_turn_token_ids == (8,)
    assert manifest.stop_token_ids == (7, 8)
    assert [name for name, _ in manifest.tokenizer_file_sha256] == [
        "tokenizer.json",
        "tokenizer_config.json",
    ]
    assert manifest.enable_thinking is False
    assert (
        manifest.chat_template_sha256
        == hashlib.sha256(FakeTokenizer.chat_template.encode()).hexdigest()
    )
    public = manifest.public_manifest()
    assert "chat_template" not in public
    assert "nonthinking_probe" not in public
    assert public["manifest_sha256"] == manifest.manifest_sha256


def test_runtime_stack_capture_enforces_exact_installed_versions() -> None:
    versions = {"vllm": "0.10.2", "transformers": "4.55.2"}
    torch = SimpleNamespace(
        __version__="2.8.0+cu128",
        version=SimpleNamespace(cuda="12.8"),
    )
    manifest = capture_runtime_stack(
        version_reader=versions.__getitem__,
        module_importer=lambda name: torch,
    )
    assert manifest.vllm_version == "0.10.2"
    assert manifest.torch_cuda_version == "12.8"

    versions["vllm"] = "0.10.3"
    with pytest.raises(RuntimeConfigurationError, match="vllm_version"):
        capture_runtime_stack(
            version_reader=versions.__getitem__,
            module_importer=lambda name: torch,
        )


def _tokenizer_manifest() -> TokenizerManifest:
    return TokenizerManifest(
        schema_version="1.0.0",
        repository="Qwen/Qwen3-14B-AWQ",
        revision=PINNED_MODEL_REVISION,
        tokenizer_class="tests.FakeTokenizer",
        tokenizer_revision=PINNED_MODEL_REVISION,
        tokenizer_file_sha256=(("tokenizer.json", HASH_A),),
        eos_token_id=7,
        end_of_turn_token_ids=(8,),
        stop_token_ids=(7, 8),
        chat_template_sha256=HASH_A,
        nonthinking_probe_sha256=HASH_B,
        nonthinking_probe_token_count=10,
        local_files_only=True,
        trust_remote_code=False,
        enable_thinking=False,
    )


def _guided_request(*, repair: bool = False) -> GuidedJSONRequest:
    schema = {"type": "object", "properties": {"ok": {"type": "boolean"}}}
    constructor = DecodingManifest.repair if repair else DecodingManifest.first_pass
    decoding = constructor(
        seed=11,
        eos_token_id=7,
        end_of_turn_token_ids=(8,),
        chat_template_hash=HASH_A,
        output_schema_hash=canonical_sha256(schema),
        structured_decoder="vllm-0.10.2-guided_json",
        tokenizer_revision=PINNED_MODEL_REVISION,
    )
    required = ("system_prompt", "output_schema", "upper_ontology", "evidence_snapshot")
    sections = tuple(
        PackingSection(name=name, section_content_hash=HASH_A, token_count=2) for name in required
    )
    packing = PackingReport.build(
        condition=ConditionName.C1_LLM_PRE,
        tokenizer_revision=PINNED_MODEL_REVISION,
        maximum_model_tokens=12_288,
        maximum_input_tokens=decoding.maximum_input_tokens,
        reserved_output_tokens=decoding.maximum_output_tokens,
        sections=sections,
        required_section_names=required,
        complete_evidence_snapshot=True,
    )
    return GuidedJSONRequest(
        request_id="request-1",
        model_name=SERVED_MODEL_NAME,
        condition=ConditionName.C1_LLM_PRE,
        messages=(
            ChatMessage(role="system", content="system"),
            ChatMessage(role="user", content="user"),
        ),
        output_schema=schema,
        decoding=decoding,
        packing=packing,
        rendered_input_token_count=8,
    )


def test_guided_json_client_sends_complete_decoding_and_parses_one_object() -> None:
    observed: dict[str, object] = {}

    def transport(
        url: str, payload: bytes | None, timeout: float
    ) -> tuple[int, bytes, dict[str, str]]:
        observed.update(url=url, payload=payload, timeout=timeout)
        body = {
            "id": "server-private-id",
            "choices": [{"message": {"content": '{"ok":true}'}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 12, "completion_tokens": 4},
        }
        return 200, json.dumps(body).encode(), {}

    client = VLLMGuidedJSONClient("http://127.0.0.1:8000", transport=transport)
    result = client.generate(_guided_request(), watchdog_seconds=90)
    wire = json.loads(observed["payload"])

    assert observed["url"] == "http://127.0.0.1:8000/v1/chat/completions"
    assert wire["guided_json"]["type"] == "object"
    assert wire["chat_template_kwargs"] == {"enable_thinking": False}
    assert wire["temperature"] == 0.7
    assert wire["top_p"] == 0.8
    assert wire["top_k"] == 20
    assert wire["min_p"] == 0.0
    assert wire["repetition_penalty"] == 1.0
    assert wire["stop_token_ids"] == [7, 8]
    assert result.parsed_object == {"ok": True}
    assert result.prompt_tokens == 12
    assert "server-private-id" not in str(result.public_manifest())

    with pytest.raises(RuntimeConfigurationError, match="loopback"):
        VLLMGuidedJSONClient("https://example.test:8000", transport=transport)


def test_client_requires_usage_and_checks_exact_served_model() -> None:
    def transport(
        url: str, payload: bytes | None, timeout: float
    ) -> tuple[int, bytes, dict[str, str]]:
        del payload, timeout
        if url.endswith("/health"):
            return 200, b"", {}
        if url.endswith("/v1/models"):
            return 200, json.dumps({"data": [{"id": SERVED_MODEL_NAME}]}).encode(), {}
        return 200, b'{"choices":[{"message":{"content":"{\\"ok\\":true}"}}]}', {}

    client = VLLMGuidedJSONClient("http://127.0.0.1:8000", transport=transport)
    assert client.ready()
    with pytest.raises(RuntimeTransportError, match="guided JSON"):
        client.generate(_guided_request(), watchdog_seconds=90)

    wrong_model = VLLMGuidedJSONClient(
        "http://127.0.0.1:8000",
        transport=lambda url, *_: (
            (200, b"", {})
            if url.endswith("/health")
            else (200, b'{"data":[{"id":"not-the-pinned-model"}]}', {})
        ),
    )
    assert not wrong_model.ready()


def test_client_reports_timeout_and_invalid_guided_response() -> None:
    def timeout_transport(
        url: str, payload: bytes | None, timeout: float
    ) -> tuple[int, bytes, dict[str, str]]:
        raise RuntimeWatchdogTimeout("expired")

    client = VLLMGuidedJSONClient("http://localhost:8000", transport=timeout_transport)
    with pytest.raises(RuntimeWatchdogTimeout):
        client.generate(_guided_request(), watchdog_seconds=90)

    invalid = VLLMGuidedJSONClient(
        "http://localhost:8000",
        transport=lambda *_: (200, b'{"choices":[]}', {}),
    )
    with pytest.raises(RuntimeTransportError, match="guided JSON"):
        invalid.generate(_guided_request(), watchdog_seconds=90)


def _write_process(proc_root: Path, pid: int, rss_kib: int, children: str = "") -> None:
    task = proc_root / str(pid) / "task" / str(pid)
    task.mkdir(parents=True)
    (proc_root / str(pid) / "status").write_text(f"Name:\ttest\nVmRSS:\t{rss_kib} kB\n")
    (task / "children").write_text(children)


def test_process_tree_and_gpu_vram_sampling_are_scoped_to_controlled_pids(
    tmp_path: Path,
) -> None:
    proc = tmp_path / "proc"
    _write_process(proc, 10, 100, "11 12")
    _write_process(proc, 11, 200)
    _write_process(proc, 12, 300)

    usage = sample_process_tree(10, proc_root=proc)
    assert usage.pids == {10, 11, 12}
    assert usage.rss_bytes == 600 * 1024

    completed = subprocess.CompletedProcess(
        args=[],
        returncode=0,
        stdout="10, 100\n12, 300\n99, 900\n",
        stderr="",
    )
    assert sample_gpu_vram(usage.pids, runner=lambda *args, **kwargs: completed) == 400 * 1024**2


def test_resource_sampler_records_then_raises_on_hard_limit(tmp_path: Path) -> None:
    storage = StoragePreflight(
        tmp_path,
        budget=StorageBudget(
            total_allocation_bytes=30_000_000_000,
            max_occupied_bytes=25_000_000_000,
            min_headroom_bytes=5_000_000_000,
        ),
    )
    with Ledger(tmp_path / "ledger.sqlite3") as ledger:
        sampler = ResourceSampler(
            limits=limits(maximum_process_ram_bytes=100),
            storage=storage,
            ledger=ledger,
            process_sampler=lambda pid: ProcessTreeUsage(frozenset({pid}), 100),
            system_ram_sampler=lambda: 50_000_000_000,
            gpu_sampler=lambda pids: 1,
            filesystem_free_sampler=lambda: 10_000_000_000,
            wall_clock=lambda: datetime(2026, 9, 3, tzinfo=UTC),
        )
        with pytest.raises(RuntimeResourceLimitExceeded) as captured:
            sampler.sample(sample_id="at-limit", root_pid=42)

        assert captured.value.snapshot.violations == ("process_ram_not_below_limit",)
        assert sampler.samples == (captured.value.snapshot,)
        assert ledger.count_rows("resource_samples") == 1


def test_resource_watchdog_surfaces_first_violation() -> None:
    @dataclass
    class StubSampler:
        calls: int = 0

        def sample(self, **kwargs: object) -> object:
            self.calls += 1
            snapshot = object.__new__(type("Snapshot", (), {}))
            if self.calls == 1:
                return snapshot
            error_snapshot = type("Snapshot", (), {"violations": ("vram",)})()
            raise RuntimeResourceLimitExceeded(error_snapshot)

    sampler = StubSampler()
    seen: list[object] = []
    with (
        pytest.raises(RuntimeResourceLimitExceeded),
        ResourceWatchdog(
            sampler=cast(ResourceSampler, sampler),
            root_pid=12,
            sample_prefix="watch",
            interval_seconds=0.001,
            on_limit=seen.append,
        ) as watchdog,
    ):
        while watchdog.sample_count < 2:
            pass
    assert len(seen) == 1


@dataclass
class FakeClock:
    monotonic_value: float = 100.0
    wall_value: datetime = datetime(2026, 9, 3, tzinfo=UTC)

    def monotonic(self) -> float:
        return self.monotonic_value

    def wall(self) -> datetime:
        return self.wall_value

    def advance(self, seconds: float) -> None:
        self.monotonic_value += seconds
        self.wall_value += timedelta(seconds=seconds)


@dataclass
class FakeProcess:
    pid: int
    running: bool = True

    def poll(self) -> int | None:
        return None if self.running else 0

    def wait(self, timeout: float | None = None) -> int:
        if self.running:
            raise subprocess.TimeoutExpired("fake", timeout)
        return 0

    def terminate(self) -> None:
        self.running = False

    def kill(self) -> None:
        self.running = False


class FakeServiceClient:
    def __init__(self, clock: FakeClock) -> None:
        self.clock = clock
        self.fail_with: BaseException | None = None

    def health(self, timeout_seconds: float = 2.0) -> bool:
        return True

    def ready(self, timeout_seconds: float = 2.0) -> bool:
        return self.health(timeout_seconds)

    def generate(self, request: GuidedJSONRequest, *, watchdog_seconds: float) -> GenerationResult:
        self.clock.advance(2)
        if self.fail_with is not None:
            raise self.fail_with
        raw = b'{"choices":[{"message":{"content":"{\\"ok\\":true}"}}]}'
        return GenerationResult(
            request_id=request.request_id,
            request_hash=request.request_hash,
            response_sha256="c" * 64,
            parsed_object={"ok": True},
            raw_response=raw,
        )


def test_service_meters_mutually_exclusive_lifecycle_and_safe_shutdown(
    tmp_path: Path,
    launch_configuration: VLLMLaunchConfiguration,
) -> None:
    clock = FakeClock()
    processes: list[FakeProcess] = []

    def popen(*args: object, **kwargs: object) -> FakeProcess:
        process = FakeProcess(pid=1000 + len(processes))
        processes.append(process)
        assert kwargs["start_new_session"] is True
        return process

    def signal_group(pid: int, signal_number: int) -> None:
        process = next(item for item in processes if item.pid == pid)
        assert signal_number in {signal.SIGTERM, signal.SIGKILL}
        process.running = False

    with Ledger(tmp_path / "ledger.sqlite3") as ledger:
        meter = AllocatedGPUMeter(
            ledger,
            monotonic_clock=clock.monotonic,
            wall_clock=clock.wall,
        )
        client = FakeServiceClient(clock)
        service = VLLMService(
            configuration=launch_configuration,
            client=cast(VLLMGuidedJSONClient, client),
            meter=meter,
            popen_factory=popen,
            monotonic_clock=clock.monotonic,
            wall_clock=clock.wall,
            sleep=lambda seconds: clock.advance(seconds),
            process_group_signaler=signal_group,
            available_cpu_sampler=lambda: set(range(16)),
            affinity_setter=lambda pid, cpus: None,
        )
        service.start(session_id="pilot", event_id="load", watchdog_seconds=180)
        service.run_warmup(lambda: clock.advance(1), event_id="warm", watchdog_seconds=10)
        service.run_schema_probe(lambda: clock.advance(1), event_id="schema", watchdog_seconds=10)
        result = service.generate(_guided_request(), event_id="inference", watchdog_seconds=90)
        service.restart(event_id="restart", watchdog_seconds=30)
        uptime = service.shutdown()

        assert result.parsed_object == {"ok": True}
        assert service.state is ServiceState.STOPPED
        assert all(not process.running for process in processes)
        assert uptime is not None
        assert uptime.service_seconds == pytest.approx(4)
        summary = ledger.gpu_summary()
        assert summary.event_count == 5
        assert {kind for kind, _ in summary.by_kind_microseconds} == {
            GpuEventKind.GPU_SESSION_START,
            GpuEventKind.WARM_UP,
            GpuEventKind.SCHEMA_PROBE,
            GpuEventKind.INFERENCE,
            GpuEventKind.RESTART,
        }


def test_service_timeout_is_recorded_once_as_timeout(
    tmp_path: Path,
    launch_configuration: VLLMLaunchConfiguration,
) -> None:
    clock = FakeClock()
    process = FakeProcess(2000)
    client = FakeServiceClient(clock)
    client.fail_with = RuntimeWatchdogTimeout("expired")
    with Ledger(tmp_path / "ledger.sqlite3") as ledger:
        service = VLLMService(
            configuration=launch_configuration,
            client=cast(VLLMGuidedJSONClient, client),
            meter=AllocatedGPUMeter(
                ledger,
                monotonic_clock=clock.monotonic,
                wall_clock=clock.wall,
            ),
            popen_factory=lambda *args, **kwargs: process,
            monotonic_clock=clock.monotonic,
            wall_clock=clock.wall,
            process_group_signaler=lambda pid, sig: setattr(process, "running", False),
            available_cpu_sampler=lambda: set(range(16)),
            affinity_setter=lambda pid, cpus: None,
        )
        service.start(session_id="pilot", event_id="load", watchdog_seconds=180)
        with pytest.raises(RuntimeWatchdogTimeout):
            service.generate(_guided_request(), event_id="failed-call", watchdog_seconds=90)
        service.shutdown()
        summary = ledger.gpu_summary()
        assert summary.event_count == 2
        assert dict(summary.by_kind_microseconds)[GpuEventKind.TIMEOUT] == 2_000_000


def test_service_resume_checks_pid_start_and_command_without_duplicate_load(
    tmp_path: Path,
    launch_configuration: VLLMLaunchConfiguration,
) -> None:
    clock = FakeClock()
    process = FakeProcess(3456)
    proc = tmp_path / "proc" / str(process.pid)
    proc.mkdir(parents=True)
    # Fields 3..22; index 19 is the start tick used by the runtime.
    fields = ["S", *(["0"] * 18), "98765"]
    (proc / "stat").write_text(f"{process.pid} (vllm worker) {' '.join(fields)}\n")
    command = b"\0".join(
        (
            b"python",
            b"-m",
            b"vllm.entrypoints.openai.api_server",
            b"--model",
            str(launch_configuration.snapshot_path).encode(),
            b"",
        )
    )
    (proc / "cmdline").write_bytes(command)
    checkpoint = tmp_path / "service.json"

    with Ledger(tmp_path / "ledger.sqlite3") as ledger:
        meter = AllocatedGPUMeter(
            ledger,
            monotonic_clock=clock.monotonic,
            wall_clock=clock.wall,
        )
        first = VLLMService(
            configuration=launch_configuration,
            client=cast(VLLMGuidedJSONClient, FakeServiceClient(clock)),
            meter=meter,
            popen_factory=lambda *args, **kwargs: process,
            monotonic_clock=clock.monotonic,
            wall_clock=clock.wall,
            process_group_signaler=lambda pid, sig: None,
            available_cpu_sampler=lambda: set(range(16)),
            affinity_setter=lambda pid, cpus: None,
        )
        first.start(session_id="pilot", event_id="load", watchdog_seconds=180)
        clock.advance(3)
        first.write_resume_checkpoint(checkpoint, proc_root=tmp_path / "proc")
        clock.advance(5)

        adopted = FakeProcess(process.pid)
        second = VLLMService(
            configuration=launch_configuration,
            client=cast(VLLMGuidedJSONClient, FakeServiceClient(clock)),
            meter=meter,
            monotonic_clock=clock.monotonic,
            wall_clock=clock.wall,
            process_group_signaler=lambda pid, sig: setattr(adopted, "running", False),
            available_cpu_sampler=lambda: set(range(16)),
            affinity_setter=lambda pid, cpus: None,
        )
        assert second.resume_from_checkpoint(
            checkpoint,
            proc_root=tmp_path / "proc",
            adopted_factory=lambda pid: adopted,
        )
        assert ledger.gpu_summary().event_count == 1
        uptime = second.shutdown()
        assert uptime is not None
        assert uptime.service_seconds == pytest.approx(8)
        assert ledger.gpu_summary().total_allocated_seconds == pytest.approx(8)
        assert ledger.gpu_summary().service_session_count == 1


def test_service_overhead_is_durable_nonoverlapping_and_carries_hard_budget(
    tmp_path: Path,
    launch_configuration: VLLMLaunchConfiguration,
) -> None:
    clock = FakeClock()
    with Ledger(tmp_path / "durable-service-time.sqlite3") as ledger:
        for ordinal, idle_seconds in ((1, 30), (2, 20)):
            process = FakeProcess(5000 + ordinal)
            meter = AllocatedGPUMeter(
                ledger,
                scheduled_limit_seconds=90,
                hard_limit_seconds=100,
                monotonic_clock=clock.monotonic,
                wall_clock=clock.wall,
            )
            service = VLLMService(
                configuration=launch_configuration,
                client=cast(VLLMGuidedJSONClient, FakeServiceClient(clock)),
                meter=meter,
                popen_factory=lambda *args, _process=process, **kwargs: _process,
                monotonic_clock=clock.monotonic,
                wall_clock=clock.wall,
                process_group_signaler=lambda pid, sig, _process=process: setattr(
                    _process, "running", False
                ),
                available_cpu_sampler=lambda: set(range(16)),
                affinity_setter=lambda pid, cpus: None,
            )
            service.start(
                session_id=f"session-{ordinal}",
                event_id=f"load-{ordinal}",
                watchdog_seconds=10,
            )
            clock.advance(idle_seconds)
            uptime = service.shutdown()
            assert uptime is not None
            assert uptime.allocated_event_seconds == 0
            assert uptime.unclassified_service_seconds == idle_seconds

        summary = ledger.gpu_summary()
        assert summary.event_count == 2
        assert summary.service_session_count == 2
        assert summary.total_allocated_seconds == 50
        assert summary.seconds_for(GpuEventKind.SERVICE_OVERHEAD) == 50
        assert sum(value for _, value in summary.by_kind_microseconds) == 50_000_000
        assert ledger.count_rows("gpu_service_sessions") == 2

        resumed_meter = AllocatedGPUMeter(
            ledger,
            scheduled_limit_seconds=90,
            hard_limit_seconds=100,
        )
        assert resumed_meter.actual_allocated_gpu_seconds == 50
        with pytest.raises(ValueError, match="session-derived"):
            resumed_meter.allocation(
                event_id="forbidden-overhead",
                event_kind=GpuEventKind.SERVICE_OVERHEAD,
                maximum_seconds=1,
            )
        with pytest.raises(GpuBudgetExceeded, match="reach/cross hard limit"):
            resumed_meter.require_capacity(50)


def test_spawn_failure_stops_child_and_preserves_restricted_log(
    tmp_path: Path,
    launch_configuration: VLLMLaunchConfiguration,
) -> None:
    process = FakeProcess(4567)
    log_path = tmp_path / "runtime" / "vllm.log"
    with Ledger(tmp_path / "ledger.sqlite3") as ledger:
        service = VLLMService(
            configuration=launch_configuration,
            client=cast(VLLMGuidedJSONClient, FakeServiceClient(FakeClock())),
            meter=AllocatedGPUMeter(ledger),
            log_path=log_path,
            popen_factory=lambda *args, **kwargs: process,
            process_group_signaler=lambda pid, sig: setattr(process, "running", False),
            available_cpu_sampler=lambda: set(range(8)),
            affinity_setter=lambda pid, cpus: (_ for _ in ()).throw(OSError("affinity")),
        )
        with pytest.raises(OSError, match="affinity"):
            service.start(session_id="pilot", event_id="load", watchdog_seconds=180)

    assert not process.running
    assert service.state is ServiceState.STOPPED
    assert log_path.is_file()
    assert log_path.stat().st_mode & 0o777 == 0o600


def test_service_capacity_reserves_shutdown_time(
    tmp_path: Path,
    launch_configuration: VLLMLaunchConfiguration,
) -> None:
    with Ledger(tmp_path / "ledger.sqlite3") as ledger:
        service = VLLMService(
            configuration=launch_configuration,
            client=cast(VLLMGuidedJSONClient, FakeServiceClient(FakeClock())),
            meter=AllocatedGPUMeter(
                ledger,
                scheduled_limit_seconds=90,
                hard_limit_seconds=100,
            ),
        )
        with pytest.raises(GpuBudgetExceeded, match="hard GPU-service limit"):
            service.require_service_capacity(40)


def test_exact_eight_call_plan_and_dry_cli_never_start_model(tmp_path: Path) -> None:
    calls = phase1_acceptance_calls()
    validate_acceptance_calls(calls)
    counts = {
        name: sum(call.call_class == name for call in calls) for name in EXPECTED_ACCEPTANCE_COUNTS
    }
    assert counts == EXPECTED_ACCEPTANCE_COUNTS
    assert [call.watchdog_seconds for call in calls] == [180, 180, 120, 120, 120, 90, 90, 90]
    assert sum(call.parent_call_id is not None for call in calls) == 1

    plan = acceptance_plan_manifest(ROOT)
    assert plan["call_count"] == 8
    assert plan["executes_gpu"] is False
    output = tmp_path / "plan.json"
    assert acceptance_main(["--project-root", str(ROOT), "--output", str(output)]) == 0
    assert json.loads(output.read_text())["manifest_sha256"] == plan["manifest_sha256"]


def test_verified_model_manifest_rehashes_exact_snapshot(tmp_path: Path) -> None:
    cache = tmp_path / "cache"
    snapshot = cache / "models--Qwen--Qwen3-14B-AWQ" / "snapshots" / PINNED_MODEL_REVISION
    snapshot.mkdir(parents=True)
    model_file = snapshot / "config.json"
    model_file.write_text('{"model":"fixture"}\n')
    model_configuration = tmp_path / "model.json"
    model_configuration.write_text('{"runtime":"fixture"}\n')
    payload: dict[str, object] = {
        "schema_version": "1.0.0",
        "repository": "Qwen/Qwen3-14B-AWQ",
        "revision": PINNED_MODEL_REVISION,
        "license": "Apache-2.0",
        "quantization": "awq",
        "single_repository_in_shared_cache": True,
        "single_snapshot_in_shared_cache": True,
        "incomplete_file_count": 0,
        "file_count": 1,
        "total_bytes": model_file.stat().st_size,
        "model_configuration_sha256": hashlib.sha256(model_configuration.read_bytes()).hexdigest(),
        "files": [
            {
                "path": "config.json",
                "size_bytes": model_file.stat().st_size,
                "sha256": hashlib.sha256(model_file.read_bytes()).hexdigest(),
            }
        ],
    }
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({**payload, "manifest_sha256": canonical_sha256(payload)}))

    validated = validate_verified_model_manifest(
        manifest,
        snapshot_path=snapshot,
        shared_cache=cache,
        model_configuration_path=model_configuration,
    )
    assert validated["file_count"] == 1

    model_file.write_text("changed\n")
    with pytest.raises(ValueError, match="size changed"):
        validate_verified_model_manifest(
            manifest,
            snapshot_path=snapshot,
            shared_cache=cache,
            model_configuration_path=model_configuration,
        )


@pytest.mark.parametrize("call_index", [0, 2, 5, 7])
def test_acceptance_fixture_packing_is_complete_and_uses_captured_tokenizer(
    call_index: int,
) -> None:
    call = phase1_acceptance_calls()[call_index]
    request = build_acceptance_request(
        root=ROOT,
        call=call,
        tokenizer=FakeTokenizer(),
        tokenizer_manifest=_tokenizer_manifest(),
    )

    assert request.packing.truncation_applied is False
    assert request.packing.omitted_section_names == ()
    assert request.packing.input_token_count == request.rendered_input_token_count
    assert request.rendered_input_token_count <= request.decoding.maximum_input_tokens
    assert request.wire_payload()["chat_template_kwargs"] == {"enable_thinking": False}
    assert request.wire_payload()["guided_json"] == request.output_schema
    user_payload = json.loads(request.messages[1].content)
    assert "output_schema" not in user_payload
    schema_section = next(
        section for section in request.packing.sections if section.name == "output_schema"
    )
    assert schema_section.section_content_hash == canonical_sha256(request.output_schema)
    assert schema_section.token_count == 0
    assert request.decoding.maximum_output_tokens == (1_536 if call_index == 7 else 2_048)
    if call.condition is ConditionName.A_FIXED_SELECT:
        assert request.packing.complete_sealed_ontology is True
    if call_index == 7:
        original = json.loads((ROOT / "tests/fixtures/phase1/c2_query_request.json").read_text())
        repair = json.loads((ROOT / "tests/fixtures/phase1/invalid_repair_case.json").read_text())
        assert user_payload["upper_ontology"] == original["upper_ontology"]
        assert user_payload["query_context"] == original["context"]
        assert user_payload["evidence_packet"] == original["packet"]
        assert user_payload["semantic_controls"] == {
            "condition": original["condition"],
            "budgets": original["budgets"],
            "capabilities": original["capabilities"],
        }
        assert user_payload["invalid_draft"] == repair["base_draft"]
        assert user_payload["validation_diagnostics"] == repair["validation_report"]
        assert "runtime" not in user_payload
        assert "request_id" not in user_payload


def test_fixed_select_guided_schema_forbids_construction_and_novel_primary_ids() -> None:
    request = build_acceptance_request(
        root=ROOT,
        call=phase1_acceptance_calls()[5],
        tokenizer=FakeTokenizer(),
        tokenizer_manifest=_tokenizer_manifest(),
    )
    definitions = request.output_schema["$defs"]
    assert definitions["ConstructionOperator"]["enum"] == [
        "compression",
        "selection",
        "supported_description",
    ]
    decision = definitions["OntologyDecision"]["properties"]
    assert decision["created_object_ids"]["maxItems"] == 0
    assert decision["removed_object_ids"]["maxItems"] == 0
    allowed_entities = definitions["Entity"]["properties"]["entity_id"]["enum"]
    assert "ent-c1-lio" in allowed_entities
    assert "novel-entity" not in allowed_entities


@pytest.mark.parametrize(
    ("call_index", "output_name"),
    [
        (0, "c1_pre_output.json"),
        (2, "c2_query_output.json"),
        (5, "fixed_select_output.json"),
    ],
)
def test_mechanical_acceptance_audit_validates_condition_fixtures(
    call_index: int,
    output_name: str,
) -> None:
    output = json.loads((ROOT / "tests/fixtures/phase1" / output_name).read_text())
    audit = validate_acceptance_generation(
        root=ROOT,
        call=phase1_acceptance_calls()[call_index],
        parsed_object=output,
    )
    assert audit["schema_valid"] is True
    assert audit["evidence_ids_valid"] is True
    assert audit["capability_valid"] is True
    assert audit["grounding_complete"] is True
    assert audit["horizon_leak_count"] == 0


class FixtureAcceptanceClient:
    def __init__(self, clock: FakeClock) -> None:
        self.clock = clock

    def health(self, timeout_seconds: float = 2.0) -> bool:
        return True

    def generate(
        self,
        request: GuidedJSONRequest,
        *,
        watchdog_seconds: float,
    ) -> GenerationResult:
        self.clock.advance(1)
        output_name = (
            "c1_pre_output.json"
            if request.request_id.startswith("c1-")
            else (
                "fixed_select_output.json"
                if request.request_id.startswith("fixed-")
                else "c2_query_output.json"
            )
        )
        parsed = json.loads((ROOT / "tests/fixtures/phase1" / output_name).read_text())
        envelope = {
            "id": f"private-{request.request_id}",
            "choices": [{"message": {"content": json.dumps(parsed)}, "finish_reason": "stop"}],
            "usage": {
                "prompt_tokens": request.rendered_input_token_count,
                "completion_tokens": 50,
            },
        }
        raw = json.dumps(envelope).encode()
        return GenerationResult(
            request_id=request.request_id,
            request_hash=request.request_hash,
            response_sha256=hashlib.sha256(raw).hexdigest(),
            parsed_object=parsed,
            raw_response=raw,
            prompt_tokens=request.rendered_input_token_count,
            completion_tokens=50,
            finish_reason="stop",
            service_request_id=f"private-{request.request_id}",
        )


@dataclass
class FakeAcceptanceService:
    configuration: VLLMLaunchConfiguration
    client: FixtureAcceptanceClient
    meter: AllocatedGPUMeter
    state: ServiceState = ServiceState.STOPPED
    pid: int = 4321

    def start(self, *, event_id: str, watchdog_seconds: float, **kwargs: object) -> None:
        with self.meter.session_start(
            event_id=event_id,
            maximum_seconds=watchdog_seconds,
        ):
            self.state = ServiceState.READY

    def restart(self, *, event_id: str, watchdog_seconds: float, **kwargs: object) -> None:
        with self.meter.restart(event_id=event_id, maximum_seconds=watchdog_seconds):
            self.state = ServiceState.READY

    def handoff_resume(self, path: Path) -> FakeAcceptanceService:
        path.write_text('{"resumed":true}\n')
        return self

    def _operation(
        self,
        operation: Callable[[], object],
        *,
        event_kind: GpuEventKind,
        event_id: str,
        watchdog_seconds: float,
        job_id: str | None = None,
        attempt_id: str | None = None,
        remaining_required_seconds: float = 0,
    ) -> object:
        context = self.meter.allocation(
            event_id=event_id,
            event_kind=event_kind,
            maximum_seconds=watchdog_seconds,
            job_id=job_id,
            attempt_id=attempt_id,
        )
        with context:
            return operation()

    def run_warmup(self, operation: Callable[[], object], **kwargs: object) -> object:
        return self._operation(
            operation,
            event_kind=GpuEventKind.WARM_UP,
            **kwargs,
        )

    def run_schema_probe(self, operation: Callable[[], object], **kwargs: object) -> object:
        return self._operation(
            operation,
            event_kind=GpuEventKind.SCHEMA_PROBE,
            **kwargs,
        )

    def generate(
        self,
        request: GuidedJSONRequest,
        *,
        event_id: str,
        watchdog_seconds: float,
        repair: bool,
        job_id: str | None = None,
        attempt_id: str | None = None,
        **kwargs: object,
    ) -> GenerationResult:
        return cast(
            GenerationResult,
            self._operation(
                lambda: self.client.generate(request, watchdog_seconds=watchdog_seconds),
                event_kind=GpuEventKind.REPAIR if repair else GpuEventKind.INFERENCE,
                event_id=event_id,
                watchdog_seconds=watchdog_seconds,
                job_id=job_id,
                attempt_id=attempt_id,
            ),
        )

    def emergency_stop(self) -> None:
        self.state = ServiceState.STOPPED

    def shutdown(self) -> None:
        self.state = ServiceState.STOPPED


def test_acceptance_runner_executes_exact_calls_lifecycle_forecast_and_gates(
    tmp_path: Path,
    launch_configuration: VLLMLaunchConfiguration,
) -> None:
    clock = FakeClock()
    with Ledger(tmp_path / "ledger.sqlite3") as ledger:
        meter = AllocatedGPUMeter(
            ledger,
            monotonic_clock=clock.monotonic,
            wall_clock=clock.wall,
        )
        service = FakeAcceptanceService(
            configuration=launch_configuration,
            client=FixtureAcceptanceClient(clock),
            meter=meter,
        )
        storage = StoragePreflight(tmp_path)
        sampler = ResourceSampler(
            limits=limits(),
            storage=storage,
            ledger=ledger,
            process_sampler=lambda pid: ProcessTreeUsage(frozenset({pid}), 10_000),
            system_ram_sampler=lambda: 100_000_000_000,
            gpu_sampler=lambda pids: 1_000,
            filesystem_free_sampler=lambda: 100_000_000_000,
            wall_clock=clock.wall,
        )
        result = AcceptanceRunner(
            root=ROOT,
            run_id="unit-pilot",
            service=cast(VLLMService, service),
            ledger=ledger,
            artifacts=ArtifactStore(
                BlobStore(tmp_path / "blobs", compression=Compression.GZIP),
                ledger,
            ),
            resource_sampler=sampler,
            tokenizer=FakeTokenizer(),
            tokenizer_manifest=_tokenizer_manifest(),
            checkpoint_path=tmp_path / "checkpoint.json",
        ).run()

        assert result["gate_passed"] is True, result
        assert result["completed_call_count"] == 8
        assert len(result["calls"]) == 8
        assert result["operator_coverage_gate"]["c1_complete"] is True
        assert result["operator_coverage_gate"]["c2_complete"] is True
        assert result["grounding_horizon_gate"] is True
        assert result["full_manifest_forecast"]["admitted"] is True
        timings = {row["call_class"]: row for row in result["timing_by_call_class"]}
        assert timings["gpu_session_start"]["sample_count"] == 2
        assert timings["acceptance_c1"]["sample_count"] == 2
        assert timings["acceptance_c1"]["p95_seconds"] == 1
        assert result["timing_gate"]["all_classes_observed"] is True
        assert result["calls"][0]["packing_report"]["truncation_applied"] is False
        assert result["calls"][0]["decoding_manifest"]["thinking_mode"] is False
        kinds = {kind for kind, _ in ledger.gpu_summary().by_kind_microseconds}
        assert GpuEventKind.WARM_UP in kinds
        assert GpuEventKind.SCHEMA_PROBE in kinds
        assert GpuEventKind.RESTART in kinds
        repair = result["calls"][-1]
        assert repair["repair_lineage"]["independent_from_acceptance_c2_calls"] is True
        assert ledger.count_rows("model_calls") == 9
