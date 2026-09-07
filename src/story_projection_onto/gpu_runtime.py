"""Controlled local vLLM runtime for the Phase-1 acceptance pilot.

The module deliberately keeps process control, HTTP transport, resource sampling,
and public manifests separate from the semantic LLM contracts in :mod:`llm`.
Nothing here downloads a model: the launcher accepts only the verified, pinned
snapshot already present below the one shared project cache.
"""

from __future__ import annotations

import copy
import csv
import fcntl
import hashlib
import http.client
import importlib
import importlib.metadata
import json
import math
import os
import select
import signal
import stat
import subprocess
import sys
import tempfile
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Callable, Mapping, Sequence
from contextlib import suppress
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from types import TracebackType
from typing import BinaryIO, Protocol, Self, cast

from story_projection_onto.contracts import ConditionName, canonical_json, canonical_sha256
from story_projection_onto.experiment import (
    AllocatedGPUMeter,
    ForecastAdmissionError,
    ResourceLimits,
)
from story_projection_onto.http_diagnostics import (
    MAX_RESPONSE_BYTES,
    FailureStage,
    ResponseEvidenceLimitError,
    RestrictedResponseJournal,
)
from story_projection_onto.llm import DecodingManifest, DecodingPass, PackingReport
from story_projection_onto.store import (
    GpuBudgetExceeded,
    GpuServiceJournalState,
    GpuServiceSession,
    Ledger,
    StoragePreflight,
)

SCHEMA_VERSION = "1.0.0"
PINNED_MODEL_REPOSITORY = "Qwen/Qwen3-14B-AWQ"
PINNED_MODEL_REVISION = "1a6fe1ecf891437a270cce11ad54d796c4f56ce0"
PINNED_RUNTIME_VERSION = "0.10.2"
SERVED_MODEL_NAME = "qwen3-14b-awq-pinned"
FALLBACK_MODEL_REPOSITORY = "Qwen/Qwen3-8B-AWQ"
FALLBACK_MODEL_REVISION = "4da05a8edb55c6046cce958586c33b61da07bb79"
FALLBACK_SERVED_MODEL_NAME = "qwen3-8b-awq-fallback"
MAXIMUM_MODEL_LENGTH = 12_288
MAXIMUM_CPU_WORKERS = 8
GPU_MEMORY_UTILIZATION = 0.88
DEFAULT_SERVICE_START_WATCHDOG_SECONDS = 180
DEFAULT_SHUTDOWN_SECONDS = 30
DEFAULT_RESOURCE_SAMPLE_COMPLETION_SECONDS = 120.0
RESOURCE_AWARE_HARD_STOP_RESERVE_SECONDS = (
    DEFAULT_RESOURCE_SAMPLE_COMPLETION_SECONDS + 2 * DEFAULT_SHUTDOWN_SECONDS
)
DURABLE_EXEC_GATE_PROTOCOL = "pipe-eof-before-exec-v1"
DURABLE_EXEC_GATE_WATCHDOG_SECONDS = 5.0
SERVICE_INSTANCE_ENVIRONMENT_KEY = "STORY_PROJECTION_ONTO_SERVICE_INSTANCE"
_VLLM_ENGINE_CORE_PROCESS_TITLE = b"VLLM::EngineCore"
MEBIBYTE = 1024 * 1024
PROC_ROOT = Path("/proc")
SERVICE_LOCK_FILENAME = ".story-projection-onto-vllm.lock"
SERVICE_LEASE_SNAPSHOT_FILENAME = ".story-projection-onto-vllm.lease.json"
GUIDED_DECODING_BACKEND = "xgrammar"
RUNTIME_ENVIRONMENT_PASSTHROUGH = (
    "PATH",
    "LD_LIBRARY_PATH",
    "CUDA_HOME",
    "CUDA_PATH",
    "CUDA_VISIBLE_DEVICES",
    "NVIDIA_VISIBLE_DEVICES",
    "NVIDIA_DRIVER_CAPABILITIES",
)
MODEL_CANDIDATES = (
    ("primary", PINNED_MODEL_REPOSITORY, PINNED_MODEL_REVISION, SERVED_MODEL_NAME),
    (
        "fallback",
        FALLBACK_MODEL_REPOSITORY,
        FALLBACK_MODEL_REVISION,
        FALLBACK_SERVED_MODEL_NAME,
    ),
)

# A failed controller must never replace durable kill/accounting identity with
# ``null`` merely because its in-memory adoption did not finish.  These fields
# are copied forward only for unresolved terminalization states; normal live and
# terminal writes continue to be complete snapshots of the current controller.
_UNRESOLVED_LEASE_IDENTITY_FIELDS = (
    "session_id",
    "accounting_session_id",
    "service_pid",
    "process_start_ticks",
    "process_command_sha256",
    "process_group_id",
    "process_session_id",
    "service_instance_token_sha256",
    "launch_protocol",
    "launch_gate_token_sha256",
    "launch_supervisor_command_sha256",
    "service_started_at",
    "service_ended_at",
    "ledger_allocated_seconds_before_session",
    "observed_service_seconds",
)


_DURABLE_EXEC_GATE_PROGRAM = """\
import json
import os
import sys

descriptor = int(sys.argv[1])
expected = bytes.fromhex(sys.argv[2])
command = json.loads(sys.argv[3])
received = bytearray()
while len(received) < len(expected):
    chunk = os.read(descriptor, len(expected) - len(received))
    if not chunk:
        break
    received.extend(chunk)
os.close(descriptor)
if bytes(received) != expected:
    raise SystemExit(125)
os.execvpe(command[0], command, os.environ)
"""


class RuntimeConfigurationError(ValueError):
    """The launcher or request would depart from the frozen Phase-1 protocol."""


class RuntimeTransportError(RuntimeError):
    """The local vLLM HTTP service returned an invalid response."""

    def __init__(
        self,
        message: str,
        *,
        restricted_diagnostics: Mapping[str, object] | None = None,
    ) -> None:
        super().__init__(message)
        self._restricted_diagnostics = dict(restricted_diagnostics or {})
        self.failure_stage: FailureStage = "transport"

    @property
    def restricted_diagnostics(self) -> dict[str, object]:
        """Return a copy of bounded details that must not enter public results."""

        return copy.deepcopy(self._restricted_diagnostics)


class RuntimeWatchdogTimeout(TimeoutError):
    """A local service operation exceeded its admitted watchdog."""

    def __init__(self, message: str) -> None:
        super().__init__(message)
        self._restricted_diagnostics: dict[str, object] = {}

    @property
    def restricted_diagnostics(self) -> dict[str, object]:
        return copy.deepcopy(self._restricted_diagnostics)


class RuntimeResourceLimitExceeded(RuntimeError):
    """A sampled hard resource limit was exceeded."""

    def __init__(self, snapshot: ResourceSnapshot) -> None:
        self.snapshot = snapshot
        super().__init__("; ".join(snapshot.violations))


_MAXIMUM_TRANSPORT_DIAGNOSTIC_TEXT = 2_048


def _bounded_transport_diagnostic_text(value: object) -> str | None:
    if not isinstance(value, str):
        return None
    # Keep the restricted diagnostic single-line and bounded.  The raw response
    # is deliberately not retained here; its digest binds the exact bytes.
    printable = "".join(character if character.isprintable() else " " for character in value)
    normalized = " ".join(printable.split())
    if not normalized:
        return None
    return normalized[:_MAXIMUM_TRANSPORT_DIAGNOSTIC_TEXT]


def _non_200_transport_diagnostics(status: int, body: bytes) -> dict[str, object]:
    """Extract a bounded, restricted diagnostic from a non-success response."""

    diagnostics: dict[str, object] = {
        "http_status": status,
        "response_sha256": _sha256_bytes(body),
        "response_size_bytes": len(body),
    }
    try:
        payload = json.loads(body)
    except (UnicodeDecodeError, json.JSONDecodeError):
        diagnostics["response_json_object"] = False
        return diagnostics
    if not isinstance(payload, Mapping):
        diagnostics["response_json_object"] = False
        return diagnostics
    diagnostics["response_json_object"] = True
    error = payload.get("error", payload)
    if not isinstance(error, Mapping):
        return diagnostics
    error_type = _bounded_transport_diagnostic_text(error.get("type"))
    error_message = _bounded_transport_diagnostic_text(error.get("message"))
    error_code = error.get("code")
    if error_type is not None:
        diagnostics["error_type"] = error_type
    if error_message is not None:
        diagnostics["error_message"] = error_message
    if isinstance(error_code, int) and not isinstance(error_code, bool):
        diagnostics["error_code"] = error_code
    else:
        error_code_text = _bounded_transport_diagnostic_text(error_code)
        if error_code_text is not None:
            diagnostics["error_code"] = error_code_text
    return diagnostics


def restricted_transport_failure_details(error: BaseException) -> dict[str, object] | None:
    """Return non-public transport details suitable for restricted lineage."""

    if not isinstance(error, (RuntimeTransportError, RuntimeWatchdogTimeout)):
        return None
    diagnostics = error.restricted_diagnostics
    return diagnostics or None


def _model_candidate_name(repository: str, revision: str, served_model_name: str) -> str:
    for (
        candidate_name,
        candidate_repository,
        candidate_revision,
        candidate_alias,
    ) in MODEL_CANDIDATES:
        if (repository, revision, served_model_name) == (
            candidate_repository,
            candidate_revision,
            candidate_alias,
        ):
            return candidate_name
    raise RuntimeConfigurationError(
        "model repository, revision, and served alias must match one allowlisted candidate"
    )


def _model_candidate_identity(candidate_name: str) -> tuple[str, str, str]:
    _require_plain_identifier("model_candidate", candidate_name)
    for name, repository, revision, served_model_name in MODEL_CANDIDATES:
        if name == candidate_name:
            return repository, revision, served_model_name
    raise RuntimeConfigurationError("model_candidate must be 'primary' or 'fallback'")


def _model_candidate_for_tokenizer(repository: str, revision: str) -> str:
    for candidate_name, candidate_repository, candidate_revision, _ in MODEL_CANDIDATES:
        if (repository, revision) == (candidate_repository, candidate_revision):
            return candidate_name
    raise RuntimeConfigurationError(
        "tokenizer repository and revision must match one allowlisted model candidate"
    )


def _is_allowlisted_model_alias(model_name: str) -> bool:
    return any(model_name == candidate_alias for _, _, _, candidate_alias in MODEL_CANDIDATES)


def _model_revision_for_alias(model_name: str) -> str:
    for _, _, revision, candidate_alias in MODEL_CANDIDATES:
        if model_name == candidate_alias:
            return revision
    raise RuntimeConfigurationError("model alias is not an allowlisted candidate")


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _is_canonical_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _path_identity(path: Path) -> str:
    """Return a stable public identifier without publishing an absolute path."""

    return _sha256_bytes(str(path.resolve()).encode("utf-8"))


def _parse_aware_datetime(name: str, value: str) -> datetime:
    try:
        parsed = datetime.fromisoformat(
            value.removesuffix("Z") + ("+00:00" if value.endswith("Z") else "")
        )
    except (TypeError, ValueError) as exc:
        raise RuntimeConfigurationError(f"{name} must be an ISO-8601 timestamp") from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise RuntimeConfigurationError(f"{name} must include an offset")
    return parsed


def _require_plain_identifier(name: str, value: str) -> str:
    if (
        not isinstance(value, str)
        or not value
        or value.strip() != value
        or any(char in value for char in "\r\n\0")
    ):
        raise RuntimeConfigurationError(f"{name} must be a nonempty plain identifier")
    return value


def atomic_write_public_json(path: Path, value: Mapping[str, object]) -> None:
    """Atomically write a manifest after rejecting obvious private path fields."""

    for key in value:
        lowered = key.casefold()
        if lowered.endswith("_path") or lowered.endswith("_directory"):
            raise RuntimeConfigurationError(
                f"public manifest may not contain filesystem field {key!r}"
            )
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            # ``model_gate`` deliberately returns immutable ``MappingProxyType``
            # certificates.  The stdlib JSON encoder only recognizes concrete
            # dicts as mappings, so materialize the public top-level object while
            # preserving its already-canonical contents.
            json.dump(dict(value), stream, ensure_ascii=False, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


@dataclass(frozen=True, slots=True)
class VLLMLaunchConfiguration:
    """Pinned, local-only vLLM 0.10.2 service configuration."""

    snapshot_path: Path
    shared_cache: Path
    repository: str = PINNED_MODEL_REPOSITORY
    revision: str = PINNED_MODEL_REVISION
    runtime_version: str = PINNED_RUNTIME_VERSION
    served_model_name: str = SERVED_MODEL_NAME
    host: str = "127.0.0.1"
    port: int = 8000
    tensor_parallel_size: int = 1
    maximum_model_length: int = MAXIMUM_MODEL_LENGTH
    gpu_memory_utilization: float = GPU_MEMORY_UTILIZATION
    cpu_offload_gb: int = 0
    maximum_sequences: int = 1
    cpu_workers: int = MAXIMUM_CPU_WORKERS
    dtype: str = "half"
    prefix_caching: bool = False
    speculative_decoding: bool = False
    enforce_eager: bool = True
    verified_snapshot_manifest_sha256: str | None = None
    guided_decoding_disable_any_whitespace: bool = False

    def __post_init__(self) -> None:
        if type(self.guided_decoding_disable_any_whitespace) is not bool:
            raise RuntimeConfigurationError("decoder whitespace policy must be an exact boolean")
        snapshot = self.snapshot_path.resolve(strict=True)
        cache = self.shared_cache.resolve(strict=True)
        try:
            snapshot.relative_to(cache)
        except ValueError as exc:
            raise RuntimeConfigurationError(
                "the model snapshot must be inside the one shared cache"
            ) from exc
        expected = {
            "runtime_version": PINNED_RUNTIME_VERSION,
            "host": "127.0.0.1",
            "tensor_parallel_size": 1,
            "maximum_model_length": MAXIMUM_MODEL_LENGTH,
            "gpu_memory_utilization": GPU_MEMORY_UTILIZATION,
            "cpu_offload_gb": 0,
            "maximum_sequences": 1,
            "cpu_workers": MAXIMUM_CPU_WORKERS,
            "dtype": "half",
            "prefix_caching": False,
            "speculative_decoding": False,
            "enforce_eager": True,
        }
        for name, required in expected.items():
            if getattr(self, name) != required:
                raise RuntimeConfigurationError(f"{name} must remain {required!r}")
        _model_candidate_name(self.repository, self.revision, self.served_model_name)
        expected_repository_directory = "models--" + self.repository.replace("/", "--")
        if (
            snapshot.name != self.revision
            or snapshot.parent.name != "snapshots"
            or snapshot.parent.parent.name != expected_repository_directory
        ):
            raise RuntimeConfigurationError(
                "snapshot path must identify the selected repository and revision"
            )
        if self.verified_snapshot_manifest_sha256 is not None and (
            len(self.verified_snapshot_manifest_sha256) != 64
            or any(
                character not in "0123456789abcdef"
                for character in self.verified_snapshot_manifest_sha256
            )
        ):
            raise RuntimeConfigurationError(
                "verified snapshot manifest hash must be lowercase SHA-256"
            )
        if isinstance(self.port, bool) or not 1 <= self.port <= 65_535:
            raise RuntimeConfigurationError("port must be an integer from 1 through 65535")
        object.__setattr__(self, "snapshot_path", snapshot)
        object.__setattr__(self, "shared_cache", cache)

    @classmethod
    def from_model_configuration(
        cls,
        *,
        snapshot_path: Path,
        shared_cache: Path,
        model_configuration_path: Path,
        model_candidate: str = "primary",
        verified_snapshot_manifest_sha256: str | None = None,
        port: int = 8000,
    ) -> Self:
        raw = json.loads(model_configuration_path.read_text(encoding="utf-8"))
        if not isinstance(raw, Mapping):
            raise RuntimeConfigurationError("model configuration root must be an object")
        if (raw.get("repository"), raw.get("revision")) != (
            PINNED_MODEL_REPOSITORY,
            PINNED_MODEL_REVISION,
        ):
            raise RuntimeConfigurationError(
                "model configuration must retain the frozen primary candidate identity"
            )
        repository, revision, served_model_name = _model_candidate_identity(model_candidate)
        return cls(
            snapshot_path=snapshot_path,
            shared_cache=shared_cache,
            repository=repository,
            revision=revision,
            runtime_version=cast(str, raw.get("runtime_version")),
            served_model_name=served_model_name,
            tensor_parallel_size=cast(int, raw.get("tensor_parallel_size")),
            maximum_model_length=cast(int, raw.get("max_model_len")),
            gpu_memory_utilization=cast(float, raw.get("gpu_memory_utilization_pilot")),
            cpu_offload_gb=cast(int, raw.get("cpu_offload_gb")),
            maximum_sequences=cast(int, raw.get("request_concurrency")),
            prefix_caching=cast(bool, raw.get("prefix_decoding")),
            speculative_decoding=cast(bool, raw.get("speculative_decoding")),
            enforce_eager=cast(bool, raw.get("enforce_eager")),
            verified_snapshot_manifest_sha256=verified_snapshot_manifest_sha256,
            port=port,
        )

    @property
    def base_url(self) -> str:
        return f"http://{self.host}:{self.port}"

    @property
    def model_candidate(self) -> str:
        """Return the immutable allowlisted candidate kind selected by this config."""

        return _model_candidate_name(
            self.repository,
            self.revision,
            self.served_model_name,
        )

    @property
    def configuration_hash(self) -> str:
        return canonical_sha256(self.public_manifest())

    def public_manifest(self) -> dict[str, object]:
        payload: dict[str, object] = {
            "schema_version": SCHEMA_VERSION,
            "repository": self.repository,
            "revision": self.revision,
            "runtime": "vllm",
            "runtime_version": self.runtime_version,
            "model_candidate": self.model_candidate,
            "served_model_name": self.served_model_name,
            "snapshot_location_sha256": _path_identity(self.snapshot_path),
            "shared_cache_location_sha256": _path_identity(self.shared_cache),
            "local_snapshot_only": True,
            "host_scope": "loopback",
            "tensor_parallel_size": self.tensor_parallel_size,
            "maximum_model_length": self.maximum_model_length,
            "gpu_memory_utilization": self.gpu_memory_utilization,
            "cpu_offload_gb": self.cpu_offload_gb,
            "maximum_sequences": self.maximum_sequences,
            "cpu_workers": self.cpu_workers,
            "dtype": "float16",
            "prefix_caching": self.prefix_caching,
            "speculative_decoding": self.speculative_decoding,
            "enforce_eager": self.enforce_eager,
            "guided_decoding_backend": GUIDED_DECODING_BACKEND,
            "guided_decoding_fallback": False,
            "runtime_environment_policy": "explicit_os_cuda_allowlist",
            "runtime_environment_passthrough": list(RUNTIME_ENVIRONMENT_PASSTHROUGH),
            "verified_snapshot_manifest_sha256": self.verified_snapshot_manifest_sha256,
        }
        if self.guided_decoding_disable_any_whitespace:
            payload["guided_decoding_disable_any_whitespace"] = True
        return {**payload, "manifest_sha256": canonical_sha256(payload)}

    def command(self, python_executable: str = sys.executable) -> tuple[str, ...]:
        """Build the explicit vLLM 0.10.2 OpenAI-server command."""

        return (
            python_executable,
            "-m",
            "vllm.entrypoints.openai.api_server",
            "--model",
            str(self.snapshot_path),
            "--tokenizer",
            str(self.snapshot_path),
            "--served-model-name",
            self.served_model_name,
            "--host",
            self.host,
            "--port",
            str(self.port),
            "--tensor-parallel-size",
            "1",
            "--max-model-len",
            "12288",
            "--gpu-memory-utilization",
            "0.88",
            "--cpu-offload-gb",
            "0",
            "--max-num-seqs",
            "1",
            "--dtype",
            "half",
            "--generation-config",
            "vllm",
            "--guided-decoding-backend",
            GUIDED_DECODING_BACKEND,
            "--guided-decoding-disable-fallback",
            "--enforce-eager",
            "--no-enable-prefix-caching",
            "--no-enable-log-requests",
            "--uvicorn-log-level",
            "warning",
        ) + (
            ("--guided-decoding-disable-any-whitespace",)
            if self.guided_decoding_disable_any_whitespace
            else ()
        )

    def environment(self, base: Mapping[str, str] | None = None) -> dict[str, str]:
        inherited = os.environ if base is None else base
        environment = {
            name: inherited[name] for name in RUNTIME_ENVIRONMENT_PASSTHROUGH if name in inherited
        }
        cache = str(self.shared_cache)
        environment.update(
            {
                "HF_HOME": cache,
                "HF_HUB_CACHE": cache,
                "HUGGINGFACE_HUB_CACHE": cache,
                "TRANSFORMERS_CACHE": cache,
                "VLLM_CACHE_ROOT": cache,
                "XDG_CACHE_HOME": cache,
                "TORCH_HOME": cache,
                "TORCHINDUCTOR_CACHE_DIR": str(self.shared_cache / "torchinductor"),
                "TRITON_CACHE_DIR": str(self.shared_cache / "triton"),
                "CUDA_CACHE_PATH": str(self.shared_cache / "cuda"),
                "HF_HUB_OFFLINE": "1",
                "TRANSFORMERS_OFFLINE": "1",
                "HF_DATASETS_OFFLINE": "1",
                "HF_HUB_DISABLE_TELEMETRY": "1",
                "DO_NOT_TRACK": "1",
                "VLLM_NO_USAGE_STATS": "1",
                "TOKENIZERS_PARALLELISM": "false",
                "PYTHONHASHSEED": "0",
                "PYTHONNOUSERSITE": "1",
                "PYTHONSAFEPATH": "1",
                "OMP_NUM_THREADS": "8",
                "MKL_NUM_THREADS": "8",
                "OPENBLAS_NUM_THREADS": "8",
                "NUMEXPR_NUM_THREADS": "8",
                "TMPDIR": cache,
            }
        )
        return environment


class TokenizerLike(Protocol):
    eos_token_id: int | Sequence[int] | None
    unk_token_id: int | None
    chat_template: str | None

    def apply_chat_template(self, conversation: object, **kwargs: object) -> object: ...

    def convert_tokens_to_ids(self, tokens: str) -> int | Sequence[int] | None: ...


@dataclass(frozen=True, slots=True)
class RuntimeStackManifest:
    """Observed immutable runtime versions, captured without touching CUDA memory."""

    python_version: str
    vllm_version: str
    transformers_version: str
    torch_version: str
    torch_cuda_version: str

    def public_manifest(self) -> dict[str, object]:
        payload = {"schema_version": SCHEMA_VERSION, **asdict(self)}
        return {**payload, "manifest_sha256": canonical_sha256(payload)}


@dataclass(frozen=True, slots=True)
class GPUHardwareIdentity:
    """Exact live GPU identity linked to the tracked environment manifest."""

    name: str
    memory_mib: int
    driver_version: str
    pci_bus_id: str
    environment_manifest_sha256: str

    def public_manifest(self) -> dict[str, object]:
        payload = {"schema_version": SCHEMA_VERSION, "count": 1, **asdict(self)}
        return {**payload, "manifest_sha256": canonical_sha256(payload)}


def capture_gpu_hardware_identity(
    environment_manifest_path: Path,
    *,
    runner: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
) -> GPUHardwareIdentity:
    """Capture one GPU via ``nvidia-smi`` and require the tracked exact identity."""

    manifest_path = environment_manifest_path.resolve(strict=True)
    manifest_bytes = manifest_path.read_bytes()
    try:
        manifest = json.loads(manifest_bytes)
        expected = manifest["gpu"]
    except (KeyError, TypeError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RuntimeConfigurationError("environment manifest has no valid GPU identity") from exc
    if not isinstance(expected, Mapping):
        raise RuntimeConfigurationError("environment manifest GPU identity must be an object")
    expected_count = expected.get("count")
    expected_name = expected.get("name")
    expected_memory = expected.get("memory_mib")
    expected_driver = expected.get("driver_version")
    expected_bus = expected.get("pci_bus_id")
    if (
        isinstance(expected_count, bool)
        or not isinstance(expected_count, int)
        or expected_count != 1
        or not isinstance(expected_name, str)
        or not expected_name
        or isinstance(expected_memory, bool)
        or not isinstance(expected_memory, int)
        or expected_memory <= 0
        or not isinstance(expected_driver, str)
        or not expected_driver
        or not isinstance(expected_bus, str)
        or not expected_bus
    ):
        raise RuntimeConfigurationError(
            "environment manifest must freeze exactly one complete GPU identity"
        )
    try:
        completed = runner(
            [
                "nvidia-smi",
                "--query-gpu=name,memory.total,driver_version,pci.bus_id",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            capture_output=True,
            stdin=subprocess.DEVNULL,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise RuntimeConfigurationError("cannot capture live GPU hardware identity") from exc
    if completed.returncode != 0 or not isinstance(completed.stdout, str):
        raise RuntimeConfigurationError("cannot capture live GPU hardware identity")
    rows = [
        tuple(field.strip() for field in row)
        for row in csv.reader(completed.stdout.splitlines())
        if any(field.strip() for field in row)
    ]
    if len(rows) != 1 or len(rows[0]) != 4:
        raise RuntimeConfigurationError("live environment must expose exactly one GPU")
    name, memory_text, driver_version, pci_bus_id = rows[0]
    try:
        memory_mib = int(memory_text)
    except ValueError as exc:
        raise RuntimeConfigurationError("live GPU memory must be an integer MiB value") from exc
    observed = {
        "name": name,
        "memory_mib": memory_mib,
        "driver_version": driver_version,
        "pci_bus_id": pci_bus_id,
    }
    frozen = {
        "name": expected_name,
        "memory_mib": expected_memory,
        "driver_version": expected_driver,
        "pci_bus_id": expected_bus,
    }
    mismatches = [key for key in frozen if observed[key] != frozen[key]]
    if mismatches:
        raise RuntimeConfigurationError(
            "live GPU differs from environment manifest: " + ", ".join(mismatches)
        )
    return GPUHardwareIdentity(
        name=name,
        memory_mib=memory_mib,
        driver_version=driver_version,
        pci_bus_id=pci_bus_id,
        environment_manifest_sha256=_sha256_bytes(manifest_bytes),
    )


def capture_runtime_stack(
    *,
    version_reader: Callable[[str], str] = importlib.metadata.version,
    module_importer: Callable[[str], object] = importlib.import_module,
) -> RuntimeStackManifest:
    """Capture and enforce the pod's pinned vLLM/Transformers/Torch stack."""

    torch_module = module_importer("torch")
    torch_version = getattr(torch_module, "__version__", None)
    cuda_namespace = getattr(torch_module, "version", None)
    cuda_version = getattr(cuda_namespace, "cuda", None)
    if not isinstance(torch_version, str) or not isinstance(cuda_version, str):
        raise RuntimeConfigurationError("installed torch does not expose version/CUDA strings")
    observed = RuntimeStackManifest(
        python_version=".".join(str(value) for value in sys.version_info[:3]),
        vllm_version=version_reader("vllm"),
        transformers_version=version_reader("transformers"),
        torch_version=torch_version,
        torch_cuda_version=cuda_version,
    )
    expected = {
        "vllm_version": PINNED_RUNTIME_VERSION,
        "transformers_version": "4.55.2",
        "torch_version": "2.8.0+cu128",
        "torch_cuda_version": "12.8",
    }
    mismatches = [name for name, value in expected.items() if getattr(observed, name) != value]
    if mismatches:
        raise RuntimeConfigurationError(
            "installed runtime differs from frozen stack at: " + ", ".join(mismatches)
        )
    return observed


@dataclass(frozen=True, slots=True)
class TokenizerManifest:
    """Public-safe exact tokenizer and non-thinking chat-template capture."""

    schema_version: str
    repository: str
    revision: str
    tokenizer_class: str
    tokenizer_revision: str
    tokenizer_file_sha256: tuple[tuple[str, str], ...]
    eos_token_id: int
    end_of_turn_token_ids: tuple[int, ...]
    stop_token_ids: tuple[int, ...]
    chat_template_sha256: str
    nonthinking_probe_sha256: str
    nonthinking_probe_token_count: int
    local_files_only: bool
    trust_remote_code: bool
    enable_thinking: bool

    @property
    def manifest_sha256(self) -> str:
        return canonical_sha256(asdict(self))

    def public_manifest(self) -> dict[str, object]:
        payload = asdict(self)
        return {**payload, "manifest_sha256": self.manifest_sha256}


def _one_token_id(tokenizer: TokenizerLike, token: str) -> int:
    value = tokenizer.convert_tokens_to_ids(token)
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise RuntimeConfigurationError(f"tokenizer does not define exact token {token!r}")
    if tokenizer.unk_token_id is not None and value == tokenizer.unk_token_id:
        raise RuntimeConfigurationError(f"tokenizer mapped {token!r} to its unknown token")
    return value


def capture_tokenizer_manifest(
    snapshot_path: Path,
    *,
    repository: str = PINNED_MODEL_REPOSITORY,
    revision: str = PINNED_MODEL_REVISION,
    tokenizer_loader: Callable[..., TokenizerLike] | None = None,
    end_of_turn_tokens: Sequence[str] = ("<|im_end|>",),
) -> TokenizerManifest:
    """Load only the local pinned tokenizer and hash its exact non-thinking behavior."""

    _model_candidate_for_tokenizer(repository, revision)
    snapshot = snapshot_path.resolve(strict=True)
    if snapshot.name != revision:
        raise RuntimeConfigurationError("tokenizer snapshot directory must equal pinned revision")
    if tokenizer_loader is None:
        from transformers import AutoTokenizer

        tokenizer_loader = cast(Callable[..., TokenizerLike], AutoTokenizer.from_pretrained)
    required_tokenizer_files = ("tokenizer.json", "tokenizer_config.json")
    missing = [name for name in required_tokenizer_files if not (snapshot / name).is_file()]
    if missing:
        raise RuntimeConfigurationError(
            "pinned snapshot lacks tokenizer files: " + ", ".join(missing)
        )
    optional_tokenizer_files = ("special_tokens_map.json", "added_tokens.json")
    tokenizer_files = tuple(
        name
        for name in (*required_tokenizer_files, *optional_tokenizer_files)
        if (snapshot / name).is_file()
    )
    tokenizer = tokenizer_loader(
        str(snapshot),
        local_files_only=True,
        trust_remote_code=False,
        revision=revision,
    )
    template = tokenizer.chat_template
    if not isinstance(template, str) or not template:
        raise RuntimeConfigurationError("pinned tokenizer has no raw chat template")
    eos_value = tokenizer.eos_token_id
    if isinstance(eos_value, bool) or not isinstance(eos_value, int) or eos_value < 0:
        raise RuntimeConfigurationError("pinned tokenizer must expose one exact EOS token ID")
    if not end_of_turn_tokens:
        raise RuntimeConfigurationError("at least one end-of-turn token must be captured")
    end_ids = tuple(dict.fromkeys(_one_token_id(tokenizer, token) for token in end_of_turn_tokens))
    if not all(token in template for token in end_of_turn_tokens):
        raise RuntimeConfigurationError("declared end-of-turn token is absent from chat template")
    stop_ids = tuple(dict.fromkeys((eos_value, *end_ids)))
    probe = (
        {"role": "system", "content": "Return one JSON object."},
        {"role": "user", "content": "Acknowledge readiness."},
    )
    rendered = tokenizer.apply_chat_template(
        probe,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    encoded = tokenizer.apply_chat_template(
        probe,
        tokenize=True,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    if not isinstance(rendered, str):
        raise RuntimeConfigurationError("non-thinking tokenizer probe did not render text")
    if not isinstance(encoded, Sequence) or isinstance(encoded, (str, bytes, bytearray)):
        raise RuntimeConfigurationError("non-thinking tokenizer probe did not return token IDs")
    tokenizer_class = f"{type(tokenizer).__module__}.{type(tokenizer).__qualname__}"
    return TokenizerManifest(
        schema_version=SCHEMA_VERSION,
        repository=repository,
        revision=revision,
        tokenizer_class=tokenizer_class,
        tokenizer_revision=revision,
        tokenizer_file_sha256=tuple(
            (name, _sha256_bytes((snapshot / name).read_bytes())) for name in tokenizer_files
        ),
        eos_token_id=eos_value,
        end_of_turn_token_ids=end_ids,
        stop_token_ids=stop_ids,
        chat_template_sha256=_sha256_bytes(template.encode("utf-8")),
        nonthinking_probe_sha256=_sha256_bytes(rendered.encode("utf-8")),
        nonthinking_probe_token_count=len(encoded),
        local_files_only=True,
        trust_remote_code=False,
        enable_thinking=False,
    )


@dataclass(frozen=True, slots=True)
class ChatMessage:
    role: str
    content: str

    def __post_init__(self) -> None:
        if self.role not in {"system", "user", "assistant"}:
            raise RuntimeConfigurationError("unsupported chat role")
        if not self.content:
            raise RuntimeConfigurationError("chat message content must be nonempty")


@dataclass(frozen=True, slots=True)
class GuidedJSONRequest:
    """One fully packed request admitted to the loopback vLLM endpoint."""

    request_id: str
    model_name: str
    condition: ConditionName
    messages: tuple[ChatMessage, ...]
    output_schema: Mapping[str, object]
    decoding: DecodingManifest
    packing: PackingReport
    rendered_input_token_count: int
    canonical_output_schema: Mapping[str, object] | None = None
    opaque_reference_aliases: Mapping[str, str] | None = None
    sealed_record_copies: Mapping[str, object] | None = None
    stream_response: bool = False
    unconstrained_diagnostic: bool = False

    def __post_init__(self) -> None:
        if type(self.stream_response) is not bool:
            raise RuntimeConfigurationError("stream_response must be an explicit boolean")
        if type(self.unconstrained_diagnostic) is not bool or (
            self.unconstrained_diagnostic
            and (
                not self.request_id.startswith("representation-diagnostic-")
                or self.canonical_output_schema is not None
            )
        ):
            raise RuntimeConfigurationError("unconstrained output is named-field diagnostic only")
        if self.canonical_output_schema is not None:
            from story_projection_onto.output_wire import RecordTupleCodec

            expected = RecordTupleCodec(
                self.canonical_output_schema,
                self.sealed_record_copies,
                self.opaque_reference_aliases,
            ).wire_schema()
            if canonical_sha256(expected) != canonical_sha256(self.output_schema):
                raise RuntimeConfigurationError("tuple wire schema differs from canonical binding")
        if self.opaque_reference_aliases is not None and (
            self.canonical_output_schema is None
            or not self.messages
            or (
                "Opaque reference binding SHA256=" + canonical_sha256(self.opaque_reference_aliases)
                not in self.messages[0].content
            )
        ):
            raise RuntimeConfigurationError("opaque reference map is not request-bound")
        if self.sealed_record_copies and (
            self.condition is not ConditionName.A_FIXED_SELECT
            or "Sealed copy binding SHA256=" + canonical_sha256(self.sealed_record_copies)
            not in self.messages[0].content
        ):
            raise RuntimeConfigurationError("sealed copies require bound FixedSelect input")
        _require_plain_identifier("request_id", self.request_id)
        _require_plain_identifier("model_name", self.model_name)
        if not _is_allowlisted_model_alias(self.model_name):
            raise RuntimeConfigurationError(
                "request must target one allowlisted served-model alias"
            )
        if not self.messages:
            raise RuntimeConfigurationError("request must include messages")
        if self.condition is not self.packing.condition:
            raise RuntimeConfigurationError("request and packing conditions differ")
        if self.packing.tokenizer_revision != self.decoding.tokenizer_revision:
            raise RuntimeConfigurationError("packing and decoding tokenizer revisions differ")
        if self.decoding.tokenizer_revision != _model_revision_for_alias(self.model_name):
            raise RuntimeConfigurationError(
                "request tokenizer revision differs from its served-model candidate"
            )
        if self.decoding.output_schema_hash != canonical_sha256(self.output_schema):
            raise RuntimeConfigurationError("guided_json schema differs from decoding manifest")
        if self.packing.maximum_model_tokens != self.decoding.maximum_model_tokens:
            raise RuntimeConfigurationError("packing and decoding model-token caps differ")
        if self.packing.maximum_input_tokens != self.decoding.maximum_input_tokens:
            raise RuntimeConfigurationError("packing and decoding input caps differ")
        if self.packing.reserved_output_tokens != self.decoding.maximum_output_tokens:
            raise RuntimeConfigurationError("packing and decoding output caps differ")
        if self.rendered_input_token_count < 0:
            raise RuntimeConfigurationError("rendered_input_token_count must be nonnegative")
        if self.rendered_input_token_count > self.decoding.maximum_input_tokens:
            raise RuntimeConfigurationError("complete rendered prompt exceeds its input cap")
        if self.packing.input_token_count != self.rendered_input_token_count:
            raise RuntimeConfigurationError(
                "packing report token count differs from the exact rendered prompt"
            )
        if self.packing.truncation_applied or self.packing.omitted_section_names:
            raise RuntimeConfigurationError("runtime refuses truncated or omitted packing")

    @property
    def prompt_hash(self) -> str:
        return canonical_sha256([asdict(message) for message in self.messages])

    @property
    def request_hash(self) -> str:
        return canonical_sha256(self.wire_payload())

    def wire_payload(self) -> dict[str, object]:
        decoding = self.decoding
        return {
            "model": self.model_name,
            "messages": [asdict(message) for message in self.messages],
            **({} if self.unconstrained_diagnostic else {"guided_json": self.output_schema}),
            "chat_template_kwargs": {"enable_thinking": False},
            "temperature": decoding.temperature,
            "top_p": decoding.top_p,
            "top_k": decoding.top_k,
            "min_p": decoding.min_p,
            "presence_penalty": decoding.presence_penalty,
            "frequency_penalty": decoding.frequency_penalty,
            "repetition_penalty": decoding.repetition_penalty,
            "n": decoding.n,
            "best_of": decoding.best_of,
            "use_beam_search": decoding.beam_search,
            "ignore_eos": decoding.ignore_eos,
            "max_tokens": decoding.maximum_output_tokens,
            "seed": decoding.seed,
            "stop_token_ids": list(decoding.stop_token_ids),
            "stream": self.stream_response,
            **(
                {"stream_options": {"include_usage": True, "continuous_usage_stats": False}}
                if self.stream_response
                else {}
            ),
        }


@dataclass(frozen=True, slots=True)
class GenerationResult:
    request_id: str
    request_hash: str
    response_sha256: str
    parsed_object: Mapping[str, object]
    raw_response: bytes = field(repr=False)
    prompt_tokens: int = 0
    completion_tokens: int = 0
    finish_reason: str | None = None
    service_request_id: str | None = None
    diagnostic_journal: RestrictedResponseJournal | None = field(
        default=None,
        repr=False,
        compare=False,
    )

    def public_manifest(self) -> dict[str, object]:
        return {
            "schema_version": SCHEMA_VERSION,
            "request_id": self.request_id,
            "request_hash": self.request_hash,
            "response_sha256": self.response_sha256,
            "parsed_object_sha256": canonical_sha256(self.parsed_object),
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "finish_reason": self.finish_reason,
            "service_request_id_sha256": (
                None
                if self.service_request_id is None
                else _sha256_bytes(self.service_request_id.encode("utf-8"))
            ),
        }


Transport = Callable[[str, bytes | None, float], tuple[int, bytes, Mapping[str, str]]]


def _loopback_base_url(base_url: str) -> str:
    parsed = urllib.parse.urlsplit(base_url)
    if (
        parsed.scheme != "http"
        or parsed.hostname not in {"127.0.0.1", "localhost", "::1"}
        or parsed.username is not None
        or parsed.password is not None
        or parsed.query
        or parsed.fragment
        or parsed.path not in {"", "/"}
    ):
        raise RuntimeConfigurationError("vLLM endpoint must be unauthenticated loopback HTTP")
    return base_url.rstrip("/")


def _urllib_transport(
    url: str,
    payload: bytes | None,
    timeout_seconds: float,
    *,
    diagnostic_journal: RestrictedResponseJournal | None = None,
    stream_parser=None,
    abort_response=None,
    started_monotonic: float | None = None,
) -> tuple[int, bytes, Mapping[str, str]]:
    from story_projection_onto.streaming_chat import StreamProtocolError

    request = urllib.request.Request(
        url,
        data=payload,
        headers={"Content-Type": "application/json"} if payload is not None else {},
        method="POST" if payload is not None else "GET",
    )

    class NoRedirect(urllib.request.HTTPRedirectHandler):
        def redirect_request(self, req, fp, code, msg, headers, newurl):  # type: ignore[no-untyped-def]
            return None

    def receive(response: object) -> tuple[int, bytes, Mapping[str, str]]:
        stream = cast(http.client.HTTPResponse, response)
        status = stream.status
        headers = dict(stream.headers.items())
        if abort_response is not None:
            abort_response.append(stream)
        if diagnostic_journal is not None:
            diagnostic_journal.headers(status, headers)
        body = bytearray()
        while True:
            try:
                fragment = stream.read1(64 * 1024)
            except http.client.IncompleteRead as error:
                if diagnostic_journal is not None:
                    diagnostic_journal.fragment(error.partial)
                raise
            if not fragment:
                break
            received_elapsed = time.monotonic() - (started_monotonic or time.monotonic())
            if diagnostic_journal is not None:
                diagnostic_journal.fragment(fragment)
            body.extend(fragment)
            if stream_parser is not None and status == 200:
                stream_parser.feed(fragment, received_elapsed)
            if len(body) > MAX_RESPONSE_BYTES:
                raise ResponseEvidenceLimitError("HTTP response exceeded diagnostic byte limit")
        expected = stream.headers.get("content-length")
        if expected is not None and int(expected) != len(body):
            raise http.client.IncompleteRead(b"", int(expected) - len(body))
        if diagnostic_journal is not None:
            diagnostic_journal.complete()
        return status, bytes(body), headers

    try:
        # The endpoint is loopback-only and unauthenticated. Ignore proxy
        # environment variables; never forward a redirect or credentials.
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}), NoRedirect())
        try:
            response = opener.open(request, timeout=timeout_seconds)
        except urllib.error.HTTPError as error:
            response = error
        with response:
            return receive(response)
    except StreamProtocolError:
        raise
    except TimeoutError as exc:
        raise RuntimeWatchdogTimeout(f"local vLLM request exceeded {timeout_seconds}s") from exc
    except urllib.error.URLError as exc:
        if isinstance(exc.reason, TimeoutError):
            raise RuntimeWatchdogTimeout(f"local vLLM request exceeded {timeout_seconds}s") from exc
        raise RuntimeTransportError(f"cannot reach local vLLM service: {exc.reason}") from exc
    except (http.client.HTTPException, OSError, ValueError, ResponseEvidenceLimitError) as exc:
        raise RuntimeTransportError(
            "local vLLM response transport was incomplete or invalid"
        ) from exc


class VLLMGuidedJSONClient:
    """Concurrency-one guided-JSON client restricted to a local vLLM service."""

    def __init__(
        self,
        base_url: str,
        *,
        transport: Transport = _urllib_transport,
        diagnostic_root: Path | None = None,
    ) -> None:
        self.base_url = _loopback_base_url(base_url)
        self._transport = transport
        self._generation_lock = threading.Lock()
        self.diagnostic_root = diagnostic_root

    def _transport_with_wall_deadline(
        self,
        url: str,
        payload: bytes | None,
        timeout_seconds: float,
    ) -> tuple[int, bytes, Mapping[str, str]]:
        """Bound the whole transport call, including a trickling response body."""

        if not math.isfinite(timeout_seconds) or timeout_seconds <= 0:
            raise RuntimeConfigurationError("timeout_seconds must be positive and finite")
        completed = threading.Event()
        result: list[tuple[int, bytes, Mapping[str, str]]] = []
        failure: list[BaseException] = []

        def run_transport() -> None:
            try:
                result.append(self._transport(url, payload, timeout_seconds))
            except BaseException as exc:
                failure.append(exc)
            finally:
                completed.set()

        worker = threading.Thread(
            target=run_transport,
            name="vllm-http-transport",
            daemon=True,
        )
        worker.start()
        if not completed.wait(timeout_seconds):
            raise RuntimeWatchdogTimeout(
                f"local vLLM request exceeded {timeout_seconds}s total wall time"
            )
        if failure:
            raise failure[0]
        if len(result) != 1:
            raise RuntimeTransportError("local vLLM transport completed without one response")
        return result[0]

    def health(self, timeout_seconds: float = 2.0) -> bool:
        try:
            status, _, _ = self._transport_with_wall_deadline(
                f"{self.base_url}/health",
                None,
                timeout_seconds,
            )
        except (RuntimeTransportError, RuntimeWatchdogTimeout):
            return False
        return status == 200

    def serves_model(self, model_name: str, timeout_seconds: float = 2.0) -> bool:
        """Require the loopback endpoint to expose the frozen served-model alias."""

        _require_plain_identifier("model_name", model_name)
        try:
            status, body, _ = self._transport_with_wall_deadline(
                f"{self.base_url}/v1/models",
                None,
                timeout_seconds,
            )
            payload = json.loads(body)
            rows = payload["data"]
        except (
            IndexError,
            KeyError,
            TypeError,
            json.JSONDecodeError,
            RuntimeTransportError,
            RuntimeWatchdogTimeout,
        ):
            return False
        if status != 200 or not isinstance(rows, list):
            return False
        return any(isinstance(row, Mapping) and row.get("id") == model_name for row in rows)

    def endpoint_live(self, timeout_seconds: float = 0.25) -> bool:
        """Conservatively distinguish an absent endpoint from an unresponsive one."""

        try:
            self._transport_with_wall_deadline(
                f"{self.base_url}/health",
                None,
                timeout_seconds,
            )
        except RuntimeWatchdogTimeout:
            # A timeout is not proof that the controlled listener disappeared.
            return True
        except RuntimeTransportError:
            return False
        return True

    def ready(
        self,
        timeout_seconds: float = 2.0,
        *,
        model_name: str = SERVED_MODEL_NAME,
    ) -> bool:
        """Check both generic health and the exact model identity."""

        _require_plain_identifier("model_name", model_name)
        if not _is_allowlisted_model_alias(model_name):
            raise RuntimeConfigurationError(
                "readiness must target one allowlisted served-model alias"
            )
        return self.health(timeout_seconds) and self.serves_model(
            model_name,
            timeout_seconds,
        )

    def generate(
        self,
        request: GuidedJSONRequest,
        *,
        watchdog_seconds: float,
    ) -> GenerationResult:
        if not math.isfinite(watchdog_seconds) or watchdog_seconds <= 0:
            raise RuntimeConfigurationError("watchdog_seconds must be positive and finite")
        payload = canonical_json(request.wire_payload()).encode("utf-8")
        if not self._generation_lock.acquire(blocking=False):
            raise RuntimeConfigurationError("concurrent vLLM generation is mechanically forbidden")

        try:
            from story_projection_onto.streaming_chat import ChatSSE

            journal = (
                None
                if self.diagnostic_root is None
                else RestrictedResponseJournal(
                    self.diagnostic_root,
                    request_hash=request.request_hash,
                    maximum_fragments=8192 if request.stream_response else 1024,
                )
            )
            stream_parser = ChatSSE(journal) if request.stream_response else None
            if request.opaque_reference_aliases is not None:
                if journal is None:
                    raise RuntimeConfigurationError(
                        "opaque references require restricted diagnostics"
                    )
                journal.event(
                    "lossless_wire_binding",
                    canonical_output_schema=request.canonical_output_schema,
                    opaque_reference_aliases=request.opaque_reference_aliases,
                    sealed_record_copies=request.sealed_record_copies,
                )
        except BaseException:
            self._generation_lock.release()
            raise

        completed = threading.Event()
        result: list[GenerationResult] = []
        failure: list[BaseException] = []
        abort_response = []
        started_monotonic = time.monotonic()

        def run_generation() -> None:
            stage: FailureStage = "transport"
            try:
                if self._transport is _urllib_transport:
                    status, body, headers = _urllib_transport(
                        f"{self.base_url}/v1/chat/completions",
                        payload,
                        watchdog_seconds,
                        diagnostic_journal=journal,
                        stream_parser=stream_parser,
                        abort_response=abort_response,
                        started_monotonic=started_monotonic,
                    )
                else:
                    status, body, headers = self._transport(
                        f"{self.base_url}/v1/chat/completions",
                        payload,
                        watchdog_seconds,
                    )
                    if journal is not None:
                        journal.headers(status, headers)
                        journal.fragment(body)
                        journal.complete()
                    if stream_parser is not None and status == 200:
                        stream_parser.feed(body, time.monotonic() - started_monotonic)
                stage = "http" if status != 200 else "decoding"
                result.append(
                    self._decode_generation_response(
                        request,
                        status,
                        body,
                        headers,
                        diagnostic_journal=journal,
                        stream_parser=stream_parser,
                    )
                )
            except BaseException as exc:
                from story_projection_onto.streaming_chat import StreamProtocolError

                if isinstance(exc, StreamProtocolError):
                    stage = "decoding"
                failure.append(exc)
                if isinstance(exc, RuntimeTransportError):
                    exc.failure_stage = stage
                if journal is not None:
                    try:
                        journal.failure(stage, exc)
                        if isinstance(exc, (RuntimeTransportError, RuntimeWatchdogTimeout)):
                            exc._restricted_diagnostics.update(
                                {
                                    "failure_stage": stage,
                                    "diagnostic_journal": str(journal.path),
                                }
                            )
                    except BaseException as diagnostic_error:
                        # Evidence-write failures must not turn into a success
                        # or hide the original exception on the calling thread.
                        failure[0] = BaseExceptionGroup(
                            "response and diagnostic persistence failed",
                            [exc, diagnostic_error],
                        )
            finally:
                self._generation_lock.release()
                completed.set()

        worker = threading.Thread(
            target=run_generation,
            name=f"vllm-generation-{request.request_id}",
            daemon=True,
        )
        try:
            worker.start()
        except BaseException:
            self._generation_lock.release()
            raise
        if not completed.wait(watchdog_seconds):
            error = RuntimeWatchdogTimeout(
                f"local vLLM generation exceeded {watchdog_seconds}s total wall time"
            )
            if journal is not None:
                journal.failure("transport", error, cancel=True)
                error._restricted_diagnostics.update(
                    {
                        "failure_stage": "transport",
                        "diagnostic_journal": str(journal.path),
                    }
                )
            # Cancel the exact loopback response socket; smaller non-streaming
            # reads cannot expose model tokens. Shutdown unblocks read1 and
            # makes disconnect observable to the pinned server's generator.
            import socket

            for response in abort_response:
                try:
                    response.fp.raw._sock.shutdown(socket.SHUT_RDWR)
                    if journal is not None:
                        journal.event("cancellation_socket_shutdown")
                except (AttributeError, OSError):
                    if journal is not None:
                        journal.event("cancellation_socket_unavailable")
            raise error
        if failure:
            raise failure[0]
        if len(result) != 1:
            raise RuntimeTransportError("local vLLM generation completed without one response")
        return result[0]

    @staticmethod
    def _decode_generation_response(
        request: GuidedJSONRequest,
        status: int,
        body: bytes,
        headers: Mapping[str, str],
        *,
        diagnostic_journal: RestrictedResponseJournal | None = None,
        stream_parser=None,
    ) -> GenerationResult:
        if status != 200:
            raise RuntimeTransportError(
                f"vLLM returned HTTP {status}",
                restricted_diagnostics=_non_200_transport_diagnostics(status, body),
            )
        try:
            if request.stream_response:
                from story_projection_onto.streaming_chat import ChatSSE

                if not any(
                    k.lower() == "content-type" and v.startswith("text/event-stream")
                    for k, v in headers.items()
                ):
                    raise ValueError("streaming request did not receive text/event-stream")
                if stream_parser is None:
                    stream_parser = ChatSSE(diagnostic_journal)
                    stream_parser.feed(body, 0)
                response = stream_parser.envelope()
            else:
                response = json.loads(body)
            if not isinstance(response, Mapping) or len(response.get("choices", [])) != 1:
                raise ValueError("response must contain exactly one choice")
            choice = response["choices"][0]
            if diagnostic_journal is not None:
                diagnostic_journal.event(
                    "completion_metadata",
                    finish_reason=choice.get("finish_reason"),
                    usage=response.get("usage"),
                )
            content = choice["message"]["content"]
            parsed_object = json.loads(content)
            if diagnostic_journal is not None:
                diagnostic_journal.event("model_content_json_complete")
            if request.canonical_output_schema is not None:
                from story_projection_onto.output_wire import RecordTupleCodec

                codec = RecordTupleCodec(
                    request.canonical_output_schema,
                    request.sealed_record_copies,
                    request.opaque_reference_aliases,
                )
                parsed_object = codec.decode(parsed_object)
                if request.opaque_reference_aliases is not None:
                    from story_projection_onto.output_wire import translate_references

                    parsed_object = translate_references(
                        parsed_object,
                        request.opaque_reference_aliases,
                        decode=True,
                    )
                if diagnostic_journal is not None:
                    diagnostic_journal.event("canonical_reconstruction_passed")
            usage = response["usage"]
        except (IndexError, KeyError, TypeError, ValueError, UnicodeDecodeError) as exc:
            raise RuntimeTransportError("vLLM response is not one guided JSON choice") from exc
        if not isinstance(parsed_object, Mapping):
            raise RuntimeTransportError("guided_json response must decode to an object")
        if not isinstance(usage, Mapping):
            raise RuntimeTransportError("vLLM response usage must be an object")
        prompt_tokens = usage.get("prompt_tokens")
        completion_tokens = usage.get("completion_tokens")
        if (
            isinstance(prompt_tokens, bool)
            or not isinstance(prompt_tokens, int)
            or prompt_tokens < 0
            or isinstance(completion_tokens, bool)
            or not isinstance(completion_tokens, int)
            or completion_tokens < 0
        ):
            raise RuntimeTransportError("vLLM usage must contain nonnegative integer token counts")
        service_request_id = response.get("id") or headers.get("x-request-id")
        if service_request_id is not None and not isinstance(service_request_id, str):
            raise RuntimeTransportError("vLLM response request ID must be text")
        finish_reason = choice.get("finish_reason")
        if finish_reason is not None and not isinstance(finish_reason, str):
            raise RuntimeTransportError("vLLM finish reason must be text")
        return GenerationResult(
            request_id=request.request_id,
            request_hash=request.request_hash,
            response_sha256=_sha256_bytes(body),
            parsed_object=dict(parsed_object),
            raw_response=body,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            finish_reason=finish_reason,
            service_request_id=service_request_id,
            diagnostic_journal=diagnostic_journal,
        )


@dataclass(frozen=True, slots=True)
class ProcessTreeUsage:
    pids: frozenset[int]
    rss_bytes: int


def _proc_children(pid: int, proc_root: Path) -> tuple[int, ...]:
    children: set[int] = set()
    task_root = proc_root / str(pid) / "task"
    try:
        tasks = tuple(task_root.iterdir())
    except (FileNotFoundError, PermissionError):
        return ()
    for task in tasks:
        try:
            text = (task / "children").read_text(encoding="ascii").strip()
        except (FileNotFoundError, PermissionError, ProcessLookupError):
            continue
        children.update(int(value) for value in text.split() if value.isdecimal())
    return tuple(sorted(children))


def sample_process_tree(root_pid: int, *, proc_root: Path = PROC_ROOT) -> ProcessTreeUsage:
    """Conservatively sum resident memory for a Linux process and descendants."""

    if root_pid <= 0:
        raise ValueError("root_pid must be positive")
    pending = [root_pid]
    seen: set[int] = set()
    rss_bytes = 0
    while pending:
        pid = pending.pop()
        if pid in seen:
            continue
        seen.add(pid)
        status_path = proc_root / str(pid) / "status"
        try:
            status = status_path.read_text(encoding="ascii")
        except (FileNotFoundError, PermissionError, ProcessLookupError):
            continue
        for line in status.splitlines():
            if line.startswith("VmRSS:"):
                fields = line.split()
                if len(fields) < 2 or not fields[1].isdecimal():
                    raise RuntimeError(f"invalid VmRSS for process {pid}")
                rss_bytes += int(fields[1]) * 1024
                break
        pending.extend(_proc_children(pid, proc_root))
    return ProcessTreeUsage(pids=frozenset(seen), rss_bytes=rss_bytes)


def sample_system_available_ram(*, proc_root: Path = PROC_ROOT) -> int:
    for line in (proc_root / "meminfo").read_text(encoding="ascii").splitlines():
        if line.startswith("MemAvailable:"):
            fields = line.split()
            if len(fields) >= 2 and fields[1].isdecimal():
                return int(fields[1]) * 1024
    raise RuntimeError("/proc/meminfo does not expose MemAvailable")


CommandRunner = Callable[..., subprocess.CompletedProcess[str]]


def sample_gpu_vram(pids: frozenset[int], *, runner: CommandRunner = subprocess.run) -> int:
    """Sum compute-process VRAM for only the controlled process tree."""

    result = runner(
        [
            "nvidia-smi",
            "--query-compute-apps=pid,used_memory",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=2,
    )
    total_mib = 0
    for line in result.stdout.splitlines():
        if not line.strip():
            continue
        fields = tuple(part.strip() for part in line.split(","))
        if len(fields) != 2 or not fields[0].isdecimal() or not fields[1].isdecimal():
            raise RuntimeError("unexpected nvidia-smi compute-process row")
        if int(fields[0]) in pids:
            total_mib += int(fields[1])
    return total_mib * MEBIBYTE


@dataclass(frozen=True, slots=True)
class ResourceSnapshot:
    sample_id: str
    sampled_at: datetime
    root_pid: int
    process_ids: tuple[int, ...]
    process_ram_bytes: int
    system_available_ram_bytes: int
    gpu_vram_bytes: int
    project_storage_bytes: int
    filesystem_free_bytes: int
    cpu_worker_count: int
    violations: tuple[str, ...] = ()
    measurement_details: Mapping[str, object] = field(default_factory=dict)

    def public_manifest(self) -> dict[str, object]:
        return {
            "schema_version": SCHEMA_VERSION,
            "sample_id": self.sample_id,
            "sampled_at": self.sampled_at.isoformat(),
            "process_count": len(self.process_ids),
            "process_ram_bytes": self.process_ram_bytes,
            "system_available_ram_bytes": self.system_available_ram_bytes,
            "gpu_vram_bytes": self.gpu_vram_bytes,
            "project_storage_bytes": self.project_storage_bytes,
            "filesystem_free_bytes": self.filesystem_free_bytes,
            "cpu_worker_count": self.cpu_worker_count,
            "violations": list(self.violations),
            "measurement_details": dict(self.measurement_details),
        }


class _ResourceProbe:
    """Cancellable observer process; only its owning controller writes SQLite."""

    def __init__(self, mode: str, root: Path):
        directory = root / "artifacts/restricted/resource-monitor"
        directory.mkdir(parents=True, exist_ok=True, mode=0o700)
        fd, name = tempfile.mkstemp(prefix=mode + "-", suffix=".jsonl", dir=directory)
        os.close(fd)
        self.journal = Path(name)
        self._lock = threading.Lock()
        self._cancel_lock = threading.Lock()
        self._process = subprocess.Popen(
            [
                sys.executable,
                "-B",
                str(Path(__file__).with_name("resource_probe.py")),
                mode,
                str(os.getpid()),
                name,
            ],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
            bufsize=0,
        )
        self._birth_ticks = _process_start_ticks(self._process.pid, PROC_ROOT)

    def request(self, value: dict, timeout: float) -> dict:
        deadline = time.monotonic() + timeout
        if not self._lock.acquire(timeout=timeout):
            raise RuntimeWatchdogTimeout("resource probe synchronization deadline")
        try:
            process = self._process
            if process.poll() is not None:
                raise RuntimeError("resource probe is not live")
            os.write(process.stdin.fileno(), (json.dumps(value) + "\n").encode())
            data = b""
            while b"\n" not in data:
                remaining = deadline - time.monotonic()
                if remaining <= 0 or not select.select([process.stdout], [], [], remaining)[0]:
                    raise RuntimeWatchdogTimeout("resource probe observation deadline")
                fragment = os.read(process.stdout.fileno(), 65536)
                if not fragment:
                    raise RuntimeError("resource probe exited without an observation")
                data += fragment
                if len(data) > 1024 * 1024:
                    raise RuntimeError("resource probe response bound exceeded")
            record = json.loads(data)
            if not record["ok"]:
                raise RuntimeError("resource probe: " + record["error"])
            return record["value"]
        except BaseException:
            self.cancel()
            raise
        finally:
            self._lock.release()

    def cancel(self) -> None:
        # No RPC/sample/ledger lock is acquired. Popen retains exact child
        # ownership; a live child leads this deliberately separate session.
        with self._cancel_lock:
            process = self._process
            if process.poll() is None:
                with suppress(ProcessLookupError, FileNotFoundError):
                    if (
                        os.getpgid(process.pid) != process.pid
                        or _process_start_ticks(process.pid, PROC_ROOT) != self._birth_ticks
                    ):
                        raise RuntimeError("resource probe process-group identity changed")
                    os.killpg(process.pid, signal.SIGKILL)
            process.wait(timeout=1)

    def close(self) -> None:
        self.cancel()
        for stream in (self._process.stdin, self._process.stdout):
            if stream is not None:
                stream.close()


class ResourceSampler:
    """Hard-limit sampler that appends every observation before raising."""

    def __init__(
        self,
        *,
        limits: ResourceLimits,
        storage: StoragePreflight,
        ledger: Ledger | None = None,
        cpu_worker_count: int = MAXIMUM_CPU_WORKERS,
        process_sampler: Callable[[int], ProcessTreeUsage] = sample_process_tree,
        system_ram_sampler: Callable[[], int] = sample_system_available_ram,
        gpu_sampler: Callable[[frozenset[int]], int] = sample_gpu_vram,
        filesystem_free_sampler: Callable[[], int] | None = None,
        wall_clock: Callable[[], datetime] = lambda: datetime.now(UTC),
    ) -> None:
        if cpu_worker_count > limits.maximum_cpu_workers or cpu_worker_count > MAXIMUM_CPU_WORKERS:
            raise RuntimeConfigurationError("CPU worker count exceeds the hard eight-worker cap")
        if cpu_worker_count <= 0:
            raise RuntimeConfigurationError("CPU worker count must be positive")
        self.limits = limits
        self.storage = storage
        self.ledger = ledger
        self.cpu_worker_count = cpu_worker_count
        self._process_sampler = process_sampler
        self._system_ram_sampler = system_ram_sampler
        self._gpu_sampler = gpu_sampler
        self._filesystem_free_sampler = filesystem_free_sampler or (
            lambda: (
                os.statvfs(storage.quota_root).f_bavail * os.statvfs(storage.quota_root).f_frsize
            )
        )
        self._wall_clock = wall_clock
        self._samples: list[ResourceSnapshot] = []
        self._pending_samples: list[tuple[ResourceSnapshot, str | None, str | None]] = []
        self._native_probes = (
            process_sampler is sample_process_tree
            and gpu_sampler is sample_gpu_vram
            and filesystem_free_sampler is None
        )
        self._fast_probe: _ResourceProbe | None = None
        self._storage_probe: _ResourceProbe | None = None
        self._storage_observation: dict | None = None
        self._root_identities: dict[int, int] = {}
        self._flush_lock = threading.Lock()

    def prepare(self, *, force: bool = False, timeout_seconds: float = 120) -> None:
        """Complete exact storage census BEFORE allocation; never in a fast sample.

        A new service/checkpoint starts a new census, not a silently stale cache.
        Event loss or stale live observations fail closed until another checkpoint.
        """
        if not self._native_probes:
            return
        if self._storage_probe is not None and not force:
            return
        if self._storage_probe is not None:
            self._storage_probe.close()
        probe = _ResourceProbe("storage", self.storage.quota_root)
        self._storage_probe = probe
        try:
            value = probe.request({"root": str(self.storage.quota_root)}, timeout_seconds)
            self.storage.require(
                current_occupied_bytes=value["occupied_bytes"],
                filesystem_free_bytes=value["filesystem_free_bytes"],
            )
            self._storage_observation = value
        except BaseException:
            self.close_probes()
            raise

    def cancel_pending(self) -> None:
        """Interrupt the bounded fast observer without stopping storage tracking."""
        if self._fast_probe is not None:
            self._fast_probe.cancel()

    def close_probes(self) -> None:
        for name in ("_fast_probe", "_storage_probe"):
            probe = getattr(self, name)
            if probe is not None:
                probe.close()
                setattr(self, name, None)

    def flush(self) -> None:
        """The controlling process commits collected rows, never probe workers."""
        if not self._flush_lock.acquire(timeout=2):
            raise RuntimeWatchdogTimeout("resource ledger checkpoint synchronization deadline")
        try:
            self._flush_pending()
        finally:
            self._flush_lock.release()

    def _flush_pending(self) -> None:
        while self._pending_samples:
            snapshot, job_id, gpu_event_id = self._pending_samples[0]
            if self.ledger is not None:
                self.ledger.record_storage_sample(
                    self.storage.check(
                        current_occupied_bytes=snapshot.project_storage_bytes,
                        filesystem_free_bytes=snapshot.filesystem_free_bytes,
                    ),
                    phase=f"resource_sample:{snapshot.sample_id}",
                    sampled_at=snapshot.sampled_at,
                )
                self.ledger.record_resource_sample(
                    sample_id=snapshot.sample_id,
                    job_id=job_id,
                    gpu_event_id=gpu_event_id,
                    process_ram_bytes=snapshot.process_ram_bytes,
                    system_available_ram_bytes=snapshot.system_available_ram_bytes,
                    gpu_vram_bytes=snapshot.gpu_vram_bytes,
                    project_storage_bytes=snapshot.project_storage_bytes,
                    cpu_worker_count=snapshot.cpu_worker_count,
                    sampled_at=snapshot.sampled_at,
                )
            self._pending_samples.pop(0)

    @property
    def samples(self) -> tuple[ResourceSnapshot, ...]:
        """Return all observations, including the sample that tripped a gate."""

        return tuple(self._samples)

    def sample(
        self,
        *,
        sample_id: str,
        root_pid: int,
        job_id: str | None = None,
        gpu_event_id: str | None = None,
        persist: bool = True,
    ) -> ResourceSnapshot:
        _require_plain_identifier("sample_id", sample_id)
        details = {}
        if self._native_probes:
            if self._storage_probe is None:
                if root_pid != os.getpid():
                    raise RuntimeConfigurationError(
                        "full storage preparation required before sampling"
                    )
                # Existing pre-allocation controller checkpoints remain usable.
                self.prepare()
            if self._fast_probe is None or self._fast_probe._process.poll() is not None:
                if self._fast_probe is not None:
                    self._fast_probe.close()
                self._fast_probe = _ResourceProbe("fast", self.storage.quota_root)
            measured = self._fast_probe.request(
                {"pid": root_pid, "expected_start": self._root_identities.get(root_pid)}, 3
            )
            self._root_identities[root_pid] = measured["root_start_ticks"]
            # Only drain/update the separate event index; never do a traversal here.
            storage_value = self._storage_probe.request({}, 1)
            if time.monotonic() - measured["timing"]["began"]["monotonic"] > 5:
                raise RuntimeWatchdogTimeout("resource observation exceeded freshness deadline")
            self._storage_observation = storage_value
            process = ProcessTreeUsage(
                frozenset(measured["process_ids"]), measured["process_ram_bytes"]
            )
            project_storage = storage_value["occupied_bytes"]
            filesystem_free = storage_value["filesystem_free_bytes"]
            values = measured
            details = {
                "fast": measured["timing"],
                "process_identities": measured["identities"],
                "storage": storage_value,
                "maximum_observation_age_seconds": 5,
            }
        else:
            # Dependency-injected CPU fixtures: storage first, then fresh process/GPU.
            project_storage = self.storage.measure_occupied_bytes()
            filesystem_free = self._filesystem_free_sampler()
            process = self._process_sampler(root_pid)
            values = {
                "process_ram_bytes": process.rss_bytes,
                "system_available_ram_bytes": self._system_ram_sampler(),
                "gpu_vram_bytes": self._gpu_sampler(process.pids),
            }
        violations: list[str] = []
        if values["process_ram_bytes"] >= self.limits.maximum_process_ram_bytes:
            violations.append("process_ram_not_below_limit")
        if values["gpu_vram_bytes"] >= self.limits.maximum_peak_vram_bytes:
            violations.append("gpu_vram_not_below_limit")
        if project_storage > self.limits.maximum_project_occupied_bytes:
            violations.append("project_storage_exceeds_limit")
        if filesystem_free < self.limits.minimum_storage_headroom_bytes:
            violations.append("filesystem_headroom_below_minimum")
        if self.cpu_worker_count > self.limits.maximum_cpu_workers:
            violations.append("cpu_worker_count_exceeds_limit")
        sampled_at = self._wall_clock()
        if sampled_at.tzinfo is None or sampled_at.utcoffset() is None:
            raise RuntimeConfigurationError("resource sampler clock must be timezone-aware")
        snapshot = ResourceSnapshot(
            sample_id=sample_id,
            sampled_at=sampled_at,
            root_pid=root_pid,
            process_ids=tuple(sorted(process.pids)),
            process_ram_bytes=values["process_ram_bytes"],
            system_available_ram_bytes=values["system_available_ram_bytes"],
            gpu_vram_bytes=values["gpu_vram_bytes"],
            project_storage_bytes=project_storage,
            filesystem_free_bytes=filesystem_free,
            cpu_worker_count=self.cpu_worker_count,
            violations=tuple(violations),
            measurement_details=details,
        )
        self._samples.append(snapshot)
        self._pending_samples.append((snapshot, job_id, gpu_event_id))
        if persist:
            self.flush()
        if len(self._pending_samples) > 4096:
            raise RuntimeError("resource observations require a bounded ledger checkpoint")
        if violations:
            raise RuntimeResourceLimitExceeded(snapshot)
        return snapshot


@dataclass(slots=True)
class ResourceWatchdog:
    """Periodically sample a live service and surface the first hard violation."""

    sampler: ResourceSampler
    root_pid: int
    sample_prefix: str
    interval_seconds: float = 1.0
    sample_completion_timeout_seconds: float = DEFAULT_RESOURCE_SAMPLE_COMPLETION_SECONDS
    job_id: str | None = None
    gpu_event_id: str | None = None
    allocation_guard: Callable[[], None] | None = None
    on_limit: Callable[[ResourceSnapshot], None] | None = None
    on_failure: Callable[[BaseException], None] | None = None
    _stop: threading.Event = field(default_factory=threading.Event, init=False, repr=False)
    _thread: threading.Thread | None = field(default=None, init=False, repr=False)
    _failure: BaseException | None = field(default=None, init=False, repr=False)
    _sample_count: int = field(default=0, init=False)
    _sample_in_flight: threading.Event = field(
        default_factory=threading.Event,
        init=False,
        repr=False,
    )
    _failure_lock: threading.Lock = field(
        default_factory=threading.Lock,
        init=False,
        repr=False,
    )
    _drain_timeout_failure: BaseException | None = field(
        default=None,
        init=False,
        repr=False,
    )
    _completion_deadline_monotonic: float | None = field(
        default=None,
        init=False,
        repr=False,
    )
    _failure_callback_dispatched: bool = field(default=False, init=False, repr=False)

    def __post_init__(self) -> None:
        _require_plain_identifier("sample_prefix", self.sample_prefix)
        if self.root_pid <= 0:
            raise RuntimeConfigurationError("resource watchdog root PID must be positive")
        if self.interval_seconds <= 0:
            raise RuntimeConfigurationError("resource watchdog interval must be positive")
        if (
            not math.isfinite(self.sample_completion_timeout_seconds)
            or self.sample_completion_timeout_seconds <= 0
        ):
            raise RuntimeConfigurationError(
                "resource watchdog sample-completion timeout must be positive and finite"
            )

    @property
    def sample_count(self) -> int:
        return self._sample_count

    @property
    def failure(self) -> BaseException | None:
        with self._failure_lock:
            # A sampler/resource failure is scientifically authoritative and
            # supersedes the controller's earlier drain-timeout diagnostic.
            return self._failure or self._drain_timeout_failure

    @property
    def drain_timeout_failure(self) -> BaseException | None:
        """Return a bounded-drain diagnostic without masking sample failure."""

        with self._failure_lock:
            return self._drain_timeout_failure

    @property
    def sample_in_flight(self) -> bool:
        """Return whether this watchdog still owns a sampler invocation."""

        return self._sample_in_flight.is_set()

    @property
    def running(self) -> bool:
        """Return whether the owned watchdog thread has not yet terminated."""

        return self._thread is not None and self._thread.is_alive()

    def _record_failure(self, failure: BaseException) -> bool:
        """Retain the first failure raised by sampling or its allocation guard."""

        with self._failure_lock:
            if self._failure is not None:
                return False
            self._failure = failure
            return True

    def _record_drain_timeout(self, failure: BaseException) -> None:
        with self._failure_lock:
            if self._drain_timeout_failure is None:
                self._drain_timeout_failure = failure

    def _dispatch_sample_failure(
        self,
        failure: BaseException,
        *,
        limit_snapshot: ResourceSnapshot | None,
    ) -> None:
        """Dispatch callbacks once even if a prior controller drain timed out."""

        with self._failure_lock:
            if self._failure_callback_dispatched:
                return
            self._failure_callback_dispatched = True
        if limit_snapshot is not None and self.on_limit is not None:
            # The sampled limit remains the primary failure.  The general
            # failure callback must still receive it exactly once.
            with suppress(BaseException):
                self.on_limit(limit_snapshot)
        if self.on_failure is not None:
            # Shutdown coordination records its own durable failure state;
            # never replace the resource failure with a callback exception.
            with suppress(BaseException):
                self.on_failure(failure)

    def _run(self) -> None:
        while not self._stop.is_set():
            self._sample_count += 1
            self._sample_in_flight.set()
            sample_failure: BaseException | None = None
            limit_snapshot: ResourceSnapshot | None = None
            try:
                if self.allocation_guard is not None:
                    self.allocation_guard()
                self.sampler.sample(
                    sample_id=f"{self.sample_prefix}-{self._sample_count:06d}",
                    root_pid=self.root_pid,
                    job_id=self.job_id,
                    gpu_event_id=self.gpu_event_id,
                    **({"persist": False} if isinstance(self.sampler, ResourceSampler) else {}),
                )
            except RuntimeResourceLimitExceeded as exc:
                self._record_failure(exc)
                sample_failure = exc
                limit_snapshot = exc.snapshot
            except BaseException as exc:
                if self._stop.is_set() and isinstance(self.sampler, ResourceSampler):
                    return  # Requested cancellation, not an invented resource violation.
                self._record_failure(exc)
                sample_failure = exc
            finally:
                self._sample_in_flight.clear()
            if sample_failure is not None:
                self._dispatch_sample_failure(
                    sample_failure,
                    limit_snapshot=limit_snapshot,
                )
                self._stop.set()
                return
            self._stop.wait(self.interval_seconds)

    def __enter__(self) -> Self:
        return self.start()

    def start(self) -> Self:
        if self._thread is not None:
            raise RuntimeError("resource watchdog cannot be started twice")
        self._thread = threading.Thread(
            target=self._run,
            name=f"resource-watchdog-{self.sample_prefix}",
            daemon=True,
        )
        self._thread.start()
        return self

    def stop(
        self,
        *,
        raise_failure: bool = True,
        completion_timeout_seconds: float | None = None,
    ) -> bool:
        """Cancel bounded collection; completion never gates service signaling."""

        self._stop.set()
        cancel = getattr(self.sampler, "cancel_pending", None)
        if callable(cancel):
            cancel()
        if self._thread is None:
            return True
        completion_timeout = self.sample_completion_timeout_seconds
        if completion_timeout_seconds is not None:
            if not math.isfinite(completion_timeout_seconds) or completion_timeout_seconds < 0:
                raise RuntimeConfigurationError(
                    "resource watchdog stop timeout must be nonnegative and finite"
                )
            completion_timeout = min(completion_timeout, completion_timeout_seconds)
        with self._failure_lock:
            # Each cleanup invocation has its own bounded join. Physical
            # shutdown is already independent; an expired earlier drain must
            # not prevent reaping a now-cancelled observer.
            self._completion_deadline_monotonic = time.monotonic() + completion_timeout
            completion_deadline = self._completion_deadline_monotonic
        if self._thread is threading.current_thread():
            # A callback runs only after the sample releases ledger ownership,
            # but the watchdog thread still owns its lifecycle.  A distinct
            # coordinator must join it before unregistering, not before signaling.
            return False
        self._thread.join(timeout=max(0.0, completion_deadline - time.monotonic()))
        if self._thread.is_alive():
            timeout_failure = RuntimeError(
                "resource watchdog in-flight sample did not complete before its "
                "bounded shutdown deadline"
            )
            self._record_drain_timeout(timeout_failure)
            if raise_failure:
                raise timeout_failure
            return False
        failure = self.failure
        if failure is not None and raise_failure:
            raise failure
        return True

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> bool:
        self.stop(raise_failure=exc is None)
        return False


class ServiceState(StrEnum):
    STOPPED = "stopped"
    STARTING = "starting"
    READY = "ready"
    FAILED = "failed"


@dataclass(frozen=True, slots=True)
class ServiceUptime:
    session_id: str
    started_at: datetime
    ended_at: datetime
    service_seconds: float
    allocated_event_seconds: float

    @property
    def unclassified_service_seconds(self) -> float:
        return max(0.0, self.service_seconds - self.allocated_event_seconds)

    @property
    def allocated_service_seconds(self) -> float:
        """Scientific allocation including transparent between-event service time."""

        return max(self.service_seconds, self.allocated_event_seconds)

    def public_manifest(self) -> dict[str, object]:
        return {
            "schema_version": SCHEMA_VERSION,
            "session_id": self.session_id,
            "started_at": self.started_at.isoformat(),
            "ended_at": self.ended_at.isoformat(),
            "service_seconds": self.service_seconds,
            "allocated_event_seconds": self.allocated_event_seconds,
            "unclassified_service_seconds": self.unclassified_service_seconds,
            "allocated_service_seconds": self.allocated_service_seconds,
            "unclassified_time_persisted_as_service_overhead": True,
        }


@dataclass(frozen=True, slots=True)
class RecoveredServiceProcessIdentity:
    """Path-free process identity retained after terminal stale recovery."""

    configuration_hash: str
    session_id: str
    accounting_session_id: str
    pid: int
    process_start_ticks: int
    process_command_sha256: str
    service_started_at: datetime


class ProcessHandle(Protocol):
    pid: int

    def poll(self) -> int | None: ...

    def wait(self, timeout: float | None = None) -> int: ...

    def terminate(self) -> None: ...

    def kill(self) -> None: ...


PopenFactory = Callable[..., ProcessHandle]


def _signal_controlled_process_group(process_group: int, signal_number: int) -> None:
    """Signal only a live child that leads its launcher-created session."""

    observed_group = os.getpgid(process_group)
    if observed_group != process_group:
        raise RuntimeError("refusing to signal a process outside its controlled vLLM group")
    os.killpg(process_group, signal_number)


def _process_group_alive(process_group: int) -> bool:
    """Return whether any non-zombie process remains in the controlled group."""

    saw_group_member = False
    with suppress(OSError):
        for entry in PROC_ROOT.iterdir():
            if not entry.name.isdecimal():
                continue
            try:
                state, observed_group, _session_id = _process_stat_identity(int(entry.name))
            except (FileNotFoundError, ProcessLookupError, PermissionError, RuntimeError):
                continue
            if observed_group != process_group:
                continue
            saw_group_member = True
            if state != "Z":
                return True
    if saw_group_member:
        return False

    try:
        os.killpg(process_group, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        # Lack of permission is positive evidence that the group still exists.
        return True
    return True


def _process_alive(pid: int) -> bool:
    """Return whether a non-zombie PID exists, treating permission denial as live."""

    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    with suppress(OSError, RuntimeError):
        state, _process_group, _session_id = _process_stat_identity(pid)
        return state != "Z"
    return True


def _available_cpu_affinity() -> set[int]:
    """Return this controller's affinity set using the required current-PID sentinel."""

    return os.sched_getaffinity(0)


def _process_start_ticks(pid: int, proc_root: Path = PROC_ROOT) -> int:
    stat = (proc_root / str(pid) / "stat").read_text(encoding="ascii")
    _, separator, suffix = stat.rpartition(")")
    if not separator:
        raise RuntimeError("cannot parse controlled process identity")
    fields = suffix.split()
    # suffix starts at procfs field 3; starttime is field 22.
    if len(fields) <= 19 or not fields[19].isdecimal():
        raise RuntimeError("controlled process has no valid start time")
    return int(fields[19])


def _process_stat_identity(
    pid: int,
    proc_root: Path = PROC_ROOT,
) -> tuple[str, int, int]:
    """Return process state, process group, and session from one procfs record."""

    stat_value = (proc_root / str(pid) / "stat").read_text(encoding="ascii")
    _, separator, suffix = stat_value.rpartition(")")
    if not separator:
        raise RuntimeError("cannot parse controlled process group identity")
    fields = suffix.split()
    # suffix starts at procfs field 3: state, ppid, pgrp, session.
    if (
        len(fields) <= 3
        or len(fields[0]) != 1
        or not fields[2].isdecimal()
        or not fields[3].isdecimal()
    ):
        raise RuntimeError("controlled process has no valid group identity")
    return fields[0], int(fields[2]), int(fields[3])


def _process_environment_instance_sha256(
    pid: int,
    proc_root: Path = PROC_ROOT,
) -> str | None:
    """Hash the private inherited service-instance token in one process."""

    prefix = f"{SERVICE_INSTANCE_ENVIRONMENT_KEY}=".encode("ascii")
    entries = (proc_root / str(pid) / "environ").read_bytes().split(b"\0")
    values = [entry[len(prefix) :] for entry in entries if entry.startswith(prefix)]
    if not values:
        return None
    if len(values) != 1 or not values[0]:
        raise RuntimeError("controlled process has an ambiguous service-instance token")
    return hashlib.sha256(values[0]).hexdigest()


def _process_lineage_identity(
    pid: int,
    proc_root: Path = PROC_ROOT,
) -> tuple[str, int, int, int, int]:
    """Return state, parent, group, session, and start ticks from one stat read."""

    stat_value = (proc_root / str(pid) / "stat").read_text(encoding="ascii")
    _, separator, suffix = stat_value.rpartition(")")
    if not separator:
        raise RuntimeError("cannot parse controlled process lineage identity")
    fields = suffix.split()
    # suffix starts at procfs field 3: state, ppid, pgrp, session; starttime is 22.
    if (
        len(fields) <= 19
        or len(fields[0]) != 1
        or any(not fields[index].isdecimal() for index in (1, 2, 3, 19))
    ):
        raise RuntimeError("controlled process has no valid lineage identity")
    return (
        fields[0],
        int(fields[1]),
        int(fields[2]),
        int(fields[3]),
        int(fields[19]),
    )


def _is_exact_tokenless_engine_core_child(
    pid: int,
    process_group: int,
    session_id: int,
    instance_token_sha256: str,
    *,
    proc_root: Path,
) -> bool:
    """Verify vLLM's one narrow setproctitle exception without trusting its PID.

    vLLM 0.10.2's EngineCore overwrites both argv and the inherited environment
    region.  The exception is therefore limited to an immediate child of the
    still-live, token-bound session leader, with the exact padded process title.
    Identity, title, and token reads are repeated so an observed race fails
    closed instead of widening process-group ownership.
    """

    title_path = proc_root / str(pid) / "cmdline"
    expected_title_prefix = _VLLM_ENGINE_CORE_PROCESS_TITLE + b"\0"

    def exact_title() -> bytes:
        raw = title_path.read_bytes()
        if not raw.startswith(expected_title_prefix):
            return b""
        if raw.rstrip(b"\0") != _VLLM_ENGINE_CORE_PROCESS_TITLE:
            return b""
        return raw

    try:
        member_before = _process_lineage_identity(pid, proc_root)
        member_title_before = exact_title()
        member_token_before = _process_environment_instance_sha256(pid, proc_root)
        leader_before = _process_lineage_identity(process_group, proc_root)
        leader_token_before = _process_environment_instance_sha256(
            process_group,
            proc_root,
        )

        member_after = _process_lineage_identity(pid, proc_root)
        member_title_after = exact_title()
        member_token_after = _process_environment_instance_sha256(pid, proc_root)
        leader_after = _process_lineage_identity(process_group, proc_root)
        leader_token_after = _process_environment_instance_sha256(
            process_group,
            proc_root,
        )
    except (OSError, RuntimeError) as exc:
        raise RuntimeConfigurationError(
            "cannot verify a tokenless vLLM EngineCore process"
        ) from exc

    if member_before != member_after or leader_before != leader_after:
        raise RuntimeConfigurationError(
            "tokenless vLLM EngineCore process identity changed during verification"
        )
    if not member_title_before or member_title_before != member_title_after:
        return False
    if member_token_before is not None or member_token_after is not None:
        return False

    member_state, parent_pid, member_group, member_session, _ = member_before
    leader_state, _, leader_group, leader_session, _ = leader_before
    return (
        pid != process_group
        and member_state != "Z"
        and parent_pid == process_group
        and member_group == process_group
        and member_session == session_id
        and leader_state != "Z"
        and leader_group == process_group
        and leader_session == session_id
        and leader_token_before == instance_token_sha256
        and leader_token_after == instance_token_sha256
    )


def _bound_process_group_members(
    process_group: int,
    session_id: int,
    instance_token_sha256: str,
    *,
    proc_root: Path = PROC_ROOT,
) -> tuple[int, ...]:
    """Return exact live members of one token-bound launcher session.

    The inherited random token closes the otherwise unavoidable ambiguity after
    the original session leader has exited: a later unrelated session can reuse
    the numeric PID/PGID, but it cannot reproduce the persisted instance token.
    Every non-zombie member must retain both the original SID and token, except
    for the exact vLLM 0.10.2 EngineCore child verified by the narrow helper.
    """

    if (
        process_group <= 0
        or session_id != process_group
        or not _is_canonical_sha256(instance_token_sha256)
    ):
        raise RuntimeConfigurationError("service process-group binding is invalid")
    members: list[int] = []
    for entry in proc_root.iterdir():
        if not entry.name.isdecimal():
            continue
        pid = int(entry.name)
        try:
            state, observed_group, observed_session = _process_stat_identity(
                pid,
                proc_root,
            )
        except (FileNotFoundError, ProcessLookupError, PermissionError, RuntimeError):
            continue
        if observed_group != process_group or state == "Z":
            continue
        if observed_session != session_id:
            raise RuntimeConfigurationError(
                "service process group contains a process from another session"
            )
        try:
            observed_token_sha256 = _process_environment_instance_sha256(
                pid,
                proc_root,
            )
        except (FileNotFoundError, ProcessLookupError, PermissionError, RuntimeError) as exc:
            raise RuntimeConfigurationError("cannot verify a service process-group member") from exc
        if observed_token_sha256 == instance_token_sha256:
            members.append(pid)
            continue
        if observed_token_sha256 is not None:
            raise RuntimeConfigurationError("service process group contains an unbound process")
        if not _is_exact_tokenless_engine_core_child(
            pid,
            process_group,
            session_id,
            instance_token_sha256,
            proc_root=proc_root,
        ):
            raise RuntimeConfigurationError("service process group contains an unbound process")
        members.append(pid)
    return tuple(sorted(members))


def _process_command_sha256(pid: int, proc_root: Path = PROC_ROOT) -> str:
    """Hash the exact argv of a controlled process without persisting its paths."""

    raw = (proc_root / str(pid) / "cmdline").read_bytes()
    arguments = raw.split(b"\0")
    if arguments and arguments[-1] == b"":
        arguments.pop()
    if not arguments or any(not argument for argument in arguments):
        raise RuntimeError("controlled process has no valid command line")
    try:
        decoded = [argument.decode("utf-8") for argument in arguments]
    except UnicodeDecodeError as exc:
        raise RuntimeError("controlled process command line is not UTF-8") from exc
    return canonical_sha256(decoded)


@dataclass(slots=True)
class _AdoptedProcess:
    pid: int
    monotonic_clock: Callable[[], float] = time.monotonic
    sleep: Callable[[float], None] = time.sleep

    def poll(self) -> int | None:
        return None if _process_alive(self.pid) else 0

    def wait(self, timeout: float | None = None) -> int:
        deadline = None if timeout is None else self.monotonic_clock() + timeout
        while self.poll() is None:
            if deadline is not None and self.monotonic_clock() >= deadline:
                raise subprocess.TimeoutExpired("adopted-vllm", timeout)
            self.sleep(0.05)
        return 0

    def terminate(self) -> None:
        os.kill(self.pid, signal.SIGTERM)

    def kill(self) -> None:
        os.kill(self.pid, signal.SIGKILL)


@dataclass(slots=True)
class _AdoptedProcessGroup:
    """Process-handle facade for a live group whose original leader exited."""

    pid: int
    liveness_check: Callable[[int], bool] = _process_group_alive
    monotonic_clock: Callable[[], float] = time.monotonic
    sleep: Callable[[float], None] = time.sleep

    def poll(self) -> int | None:
        return None if self.liveness_check(self.pid) else 0

    def wait(self, timeout: float | None = None) -> int:
        deadline = None if timeout is None else self.monotonic_clock() + timeout
        while self.poll() is None:
            if deadline is not None and self.monotonic_clock() >= deadline:
                raise subprocess.TimeoutExpired("adopted-vllm-group", timeout)
            self.sleep(0.05)
        return 0

    def terminate(self) -> None:
        os.killpg(self.pid, signal.SIGTERM)

    def kill(self) -> None:
        os.killpg(self.pid, signal.SIGKILL)


@dataclass(slots=True)
class VLLMService:
    """Safe process lifecycle, metered without overlapping allocation intervals."""

    configuration: VLLMLaunchConfiguration
    client: VLLMGuidedJSONClient
    meter: AllocatedGPUMeter
    log_path: Path | None = None
    service_lock_path: Path | None = None
    startup_resource_sampler: ResourceSampler | None = None
    startup_sample_interval_seconds: float = 1.0
    service_heartbeat_interval_seconds: float = 5.0
    preflight_endpoint_check: Callable[[], bool] | None = None
    readiness_check: Callable[[], bool] | None = None
    popen_factory: PopenFactory = subprocess.Popen
    monotonic_clock: Callable[[], float] = time.monotonic
    wall_clock: Callable[[], datetime] = lambda: datetime.now(UTC)
    sleep: Callable[[float], None] = time.sleep
    process_group_signaler: Callable[[int, int], None] = _signal_controlled_process_group
    process_liveness_check: Callable[[int], bool] = _process_alive
    process_group_liveness_check: Callable[[int], bool] = _process_group_alive
    available_cpu_sampler: Callable[[], set[int]] = _available_cpu_affinity
    affinity_setter: Callable[[int, set[int]], None] = os.sched_setaffinity
    state: ServiceState = field(default=ServiceState.STOPPED, init=False)
    _process: ProcessHandle | None = field(default=None, init=False, repr=False)
    _last_service_pid: int | None = field(default=None, init=False, repr=False)
    _last_process_start_ticks: int | None = field(default=None, init=False, repr=False)
    _last_process_command_sha256: str | None = field(
        default=None,
        init=False,
        repr=False,
    )
    _last_process_group_id: int | None = field(default=None, init=False, repr=False)
    _last_process_session_id: int | None = field(default=None, init=False, repr=False)
    _service_instance_token: str | None = field(default=None, init=False, repr=False)
    _service_instance_token_sha256: str | None = field(
        default=None,
        init=False,
        repr=False,
    )
    _process_identity_proc_root: Path = field(
        default=PROC_ROOT,
        init=False,
        repr=False,
    )
    _launch_gate_token: bytes | None = field(default=None, init=False, repr=False)
    _launch_gate_token_sha256: str | None = field(default=None, init=False, repr=False)
    _launch_supervisor_command_sha256: str | None = field(
        default=None,
        init=False,
        repr=False,
    )
    _session_id: str | None = field(default=None, init=False, repr=False)
    _accounting_session_id: str | None = field(default=None, init=False, repr=False)
    _started_monotonic: float | None = field(default=None, init=False, repr=False)
    _started_at: datetime | None = field(default=None, init=False, repr=False)
    _allocated_at_start: float | None = field(default=None, init=False, repr=False)
    _carried_service_seconds: float = field(default=0.0, init=False, repr=False)
    _log_stream: BinaryIO | None = field(default=None, init=False, repr=False)
    _service_lock_stream: BinaryIO | None = field(default=None, init=False, repr=False)
    _prior_service_lease: dict[str, object] | None = field(
        default=None,
        init=False,
        repr=False,
    )
    _service_journal_opened: bool = field(default=False, init=False, repr=False)
    _service_heartbeat_stop: threading.Event = field(
        default_factory=threading.Event,
        init=False,
        repr=False,
    )
    _service_heartbeat_thread: threading.Thread | None = field(
        default=None,
        init=False,
        repr=False,
    )
    _service_heartbeat_failure: BaseException | None = field(
        default=None,
        init=False,
        repr=False,
    )
    _physically_stopped_at: datetime | None = field(default=None, init=False, repr=False)
    _physically_stopped_seconds: float | None = field(default=None, init=False, repr=False)
    _process_stop_journaled: bool = field(default=False, init=False, repr=False)
    _prior_process_stop_contradicted: bool = field(
        default=False,
        init=False,
        repr=False,
    )
    _last_recovered_process_identity: RecoveredServiceProcessIdentity | None = field(
        default=None,
        init=False,
        repr=False,
    )
    _process_control_lock: threading.RLock = field(
        default_factory=threading.RLock,
        init=False,
        repr=False,
    )
    _startup_resource_watchdog: ResourceWatchdog | None = field(
        default=None,
        init=False,
        repr=False,
    )
    _periodic_resource_watchdog: ResourceWatchdog | None = field(
        default=None,
        init=False,
        repr=False,
    )
    _emergency_stop_thread: threading.Thread | None = field(
        default=None,
        init=False,
        repr=False,
    )
    _emergency_stop_failure: BaseException | None = field(
        default=None,
        init=False,
        repr=False,
    )
    _emergency_stop_lock: threading.Lock = field(
        default_factory=threading.Lock,
        init=False,
        repr=False,
    )

    def __post_init__(self) -> None:
        if self.log_path is not None:
            self.log_path = self.log_path.resolve()
            if self.log_path.exists() and not self.log_path.is_file():
                raise RuntimeConfigurationError("vLLM log path must be a regular file")
        if self.startup_sample_interval_seconds <= 0:
            raise RuntimeConfigurationError("startup sample interval must be positive")
        if (
            not math.isfinite(self.service_heartbeat_interval_seconds)
            or self.service_heartbeat_interval_seconds <= 0
            or self.service_heartbeat_interval_seconds > 5
        ):
            raise RuntimeConfigurationError(
                "service heartbeat interval must be positive and no greater than five seconds"
            )
        lock_path = self.service_lock_path
        if lock_path is None:
            # The frozen configuration's one shared cache is project-scoped and
            # common to every ledger/checkpoint for this model execution.
            lock_path = self.configuration.shared_cache / SERVICE_LOCK_FILENAME
        lock_path = Path(lock_path)
        if lock_path.is_symlink():
            raise RuntimeConfigurationError("vLLM service lock path must not be a symlink")
        expected_lock_path = (self.configuration.shared_cache / SERVICE_LOCK_FILENAME).resolve()
        self.service_lock_path = lock_path.resolve()
        if self.service_lock_path != expected_lock_path:
            raise RuntimeConfigurationError(
                "vLLM service lock must use the project shared-cache lock path"
            )
        if self.service_lock_path.exists() and not self.service_lock_path.is_file():
            raise RuntimeConfigurationError("vLLM service lock path must be a regular file")

        snapshot_path = self._service_lease_snapshot_path()
        if snapshot_path.is_symlink():
            raise RuntimeConfigurationError("vLLM service lease snapshot must not be a symlink")
        if snapshot_path.exists() and not snapshot_path.is_file():
            raise RuntimeConfigurationError(
                "vLLM service lease snapshot path must be a regular file"
            )

    def _service_lease_snapshot_path(self) -> Path:
        return cast(Path, self.service_lock_path).with_name(SERVICE_LEASE_SNAPSHOT_FILENAME)

    def _read_service_lease_snapshot(self) -> dict[str, object] | None:
        snapshot_path = self._service_lease_snapshot_path()
        flags = os.O_RDONLY
        if hasattr(os, "O_CLOEXEC"):
            flags |= os.O_CLOEXEC
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        try:
            descriptor = os.open(snapshot_path, flags)
        except FileNotFoundError:
            return None
        try:
            metadata = os.fstat(descriptor)
            if not stat.S_ISREG(metadata.st_mode):
                raise RuntimeConfigurationError(
                    "vLLM service lease snapshot must be a regular file"
                )
            if metadata.st_size <= 0 or metadata.st_size > 65_536:
                raise RuntimeConfigurationError("vLLM service lease snapshot size is invalid")
            encoded = os.read(descriptor, metadata.st_size + 1)
        finally:
            os.close(descriptor)
        if len(encoded) != metadata.st_size:
            raise RuntimeConfigurationError("vLLM service lease snapshot changed while being read")
        try:
            decoded = json.loads(encoded)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise RuntimeConfigurationError("vLLM service lease snapshot is invalid") from exc
        if not isinstance(decoded, Mapping):
            raise RuntimeConfigurationError("vLLM service lease snapshot must be an object")
        payload = dict(decoded)
        recorded_hash = payload.pop("lease_manifest_sha256", None)
        if not isinstance(recorded_hash, str) or recorded_hash != canonical_sha256(payload):
            raise RuntimeConfigurationError("vLLM service lease snapshot hash is invalid")
        return payload

    def _write_service_lock_metadata(
        self,
        *,
        lease_state: str,
        service_pid: int | None = None,
        ended_at: datetime | None = None,
    ) -> None:
        stream = self._service_lock_stream
        if stream is None:
            raise RuntimeError("vLLM service lock is not held")
        _require_plain_identifier("lease_state", lease_state)
        updated_at = self.wall_clock()
        if updated_at.tzinfo is None or updated_at.utcoffset() is None:
            raise RuntimeConfigurationError("service lease clock must be timezone-aware")
        if ended_at is not None and (ended_at.tzinfo is None or ended_at.utcoffset() is None):
            raise RuntimeConfigurationError("service lease end must be timezone-aware")
        if self._started_at is not None and (
            self._started_at.tzinfo is None or self._started_at.utcoffset() is None
        ):
            raise RuntimeConfigurationError("service lease start must be timezone-aware")
        payload = {
            "schema_version": SCHEMA_VERSION,
            "configuration_hash": self.configuration.configuration_hash,
            "controller_pid": os.getpid(),
            "lease_state": lease_state,
            "session_id": self._session_id,
            "accounting_session_id": self._accounting_session_id,
            "service_pid": service_pid,
            "process_start_ticks": self._last_process_start_ticks,
            "process_command_sha256": self._last_process_command_sha256,
            "process_group_id": self._last_process_group_id,
            "process_session_id": self._last_process_session_id,
            "service_instance_token_sha256": (self._service_instance_token_sha256),
            "launch_protocol": (
                DURABLE_EXEC_GATE_PROTOCOL if self._launch_gate_token_sha256 is not None else None
            ),
            "launch_gate_token_sha256": self._launch_gate_token_sha256,
            "launch_supervisor_command_sha256": (self._launch_supervisor_command_sha256),
            "service_started_at": (
                None if self._started_at is None else self._started_at.isoformat()
            ),
            "service_ended_at": None if ended_at is None else ended_at.isoformat(),
            "ledger_allocated_seconds_before_session": self._allocated_at_start,
            "observed_service_seconds": (
                None if self._started_at is None else self._elapsed_service_seconds()
            ),
            "updated_at": updated_at.isoformat(),
        }
        prior_lease = self._prior_service_lease
        if (
            lease_state in {"accounting_pending", "shutdown_unverified"}
            and prior_lease is not None
            and prior_lease.get("lease_state") != "stopped_verified"
        ):
            # An adoption or emergency-stop error can reach this writer before
            # every in-memory field has been installed.  The authoritative
            # snapshot already holds stronger kill/accounting identity, and an
            # unresolved rewrite must never weaken that identity to null.
            prior_configuration_hash = prior_lease.get("configuration_hash")
            configuration_changed = (
                isinstance(prior_configuration_hash, str)
                and prior_configuration_hash != payload["configuration_hash"]
            )
            if configuration_changed:
                # A controller constructed with the wrong frozen model/config
                # may still be asked to make a best-effort emergency stop.  Its
                # failure record must remain wholly attributable to the prior
                # exact lease rather than hybridizing old process identity with
                # the new configuration hash.
                payload["configuration_hash"] = prior_configuration_hash
            for field_name in _UNRESOLVED_LEASE_IDENTITY_FIELDS:
                if prior_lease.get(field_name) is not None and (
                    configuration_changed or payload[field_name] is None
                ):
                    payload[field_name] = prior_lease[field_name]
        snapshot_payload = {
            **payload,
            "lease_manifest_sha256": canonical_sha256(payload),
        }
        # The flock inode provides exclusion only.  Persist the authoritative
        # metadata through an atomic sidecar replacement before mirroring it to
        # that inode.  If the controller dies during the in-place compatibility
        # write, recovery still has the last complete exact PID/argv identity.
        _atomic_write_private_json(
            self._service_lease_snapshot_path(),
            snapshot_payload,
        )
        metadata = canonical_json(payload).encode("utf-8")
        stream.seek(0)
        stream.truncate()
        stream.write(metadata + b"\n")
        stream.flush()
        os.fsync(stream.fileno())
        self._prior_service_lease = payload

    def _acquire_service_lock(self) -> bool:
        """Acquire the nonblocking project lock; return whether this call acquired it."""

        if self._service_lock_stream is not None:
            return False
        lock_path = cast(Path, self.service_lock_path)
        lock_path.parent.mkdir(parents=True, exist_ok=True)
        flags = os.O_CREAT | os.O_RDWR
        if hasattr(os, "O_CLOEXEC"):
            flags |= os.O_CLOEXEC
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        descriptor = os.open(lock_path, flags, 0o600)
        try:
            lock_stat = os.fstat(descriptor)
            if not stat.S_ISREG(lock_stat.st_mode):
                raise RuntimeConfigurationError("vLLM service lock path must be a regular file")
            os.fchmod(descriptor, 0o600)
            try:
                fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as exc:
                raise RuntimeConfigurationError(
                    "another project controller holds the exclusive vLLM service lock"
                ) from exc
            prior_lease = self._read_service_lease_snapshot()
            if prior_lease is None:
                lock_stat = os.fstat(descriptor)
                if lock_stat.st_size > 65_536:
                    raise RuntimeConfigurationError("vLLM service lease metadata is too large")
                os.lseek(descriptor, 0, os.SEEK_SET)
                prior_bytes = os.read(descriptor, lock_stat.st_size)
                if prior_bytes:
                    try:
                        prior_value = json.loads(prior_bytes)
                    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                        raise RuntimeConfigurationError(
                            "vLLM service lease metadata is invalid"
                        ) from exc
                    if not isinstance(prior_value, Mapping):
                        raise RuntimeConfigurationError(
                            "vLLM service lease metadata must be an object"
                        )
                    prior_lease = dict(prior_value)
            stream = os.fdopen(descriptor, "r+b", buffering=0)
        except BaseException:
            os.close(descriptor)
            raise
        try:
            directory_descriptor = os.open(lock_path.parent, os.O_RDONLY)
        except OSError:
            directory_descriptor = None
        if directory_descriptor is not None:
            try:
                os.fsync(directory_descriptor)
            except OSError:
                pass
            finally:
                os.close(directory_descriptor)
        self._service_lock_stream = stream
        self._prior_service_lease = prior_lease
        return True

    def _release_service_lock(self) -> None:
        stream = self._service_lock_stream
        self._service_lock_stream = None
        self._prior_service_lease = None
        if stream is None:
            return
        try:
            fcntl.flock(stream.fileno(), fcntl.LOCK_UN)
        finally:
            stream.close()

    def _open_service_journal(self) -> None:
        if self._service_journal_opened:
            return
        if (
            self._accounting_session_id is None
            or self._session_id is None
            or self._started_at is None
            or self._allocated_at_start is None
        ):
            raise RuntimeError("vLLM service accounting identity is incomplete")
        self.meter.open_service_journal(
            service_session_id=self._accounting_session_id,
            session_id=self._session_id,
            configuration_hash=self.configuration.configuration_hash,
            started_at=self._started_at,
            ledger_allocated_seconds_before_session=self._allocated_at_start,
            details={"model_candidate": self.configuration.model_candidate},
        )
        self._service_journal_opened = True

    def _observe_service_journal(
        self,
        *,
        process_stopped: bool = False,
        observed_at: datetime | None = None,
        elapsed_seconds: float | None = None,
        details: Mapping[str, object] | None = None,
    ) -> None:
        if not self._service_journal_opened:
            return
        if self._accounting_session_id is None:
            raise RuntimeError("vLLM service journal has no accounting identity")
        self.meter.observe_service_journal(
            service_session_id=self._accounting_session_id,
            elapsed_seconds=(
                self._elapsed_service_seconds() if elapsed_seconds is None else elapsed_seconds
            ),
            observed_at=self.wall_clock() if observed_at is None else observed_at,
            process_stopped=process_stopped,
            details=details,
        )

    def _adopt_service_journal(self) -> None:
        if (
            self._accounting_session_id is None
            or self._session_id is None
            or self._started_at is None
            or self._allocated_at_start is None
        ):
            raise RuntimeError("resumed vLLM service accounting identity is incomplete")
        latest = self.meter.ledger.latest_gpu_service_journal(self._accounting_session_id)
        if latest is None or latest.state.value not in {"opened", "heartbeat"}:
            raise RuntimeConfigurationError(
                "resumed vLLM service has no matching open service journal"
            )
        journal_started_at = _parse_aware_datetime(
            "service journal start",
            latest.service_started_at,
        )
        if (
            latest.session_id != self._session_id
            or latest.configuration_hash != self.configuration.configuration_hash
            or journal_started_at != self._started_at
            or abs(
                latest.ledger_allocated_microseconds_before_session / 1_000_000
                - self._allocated_at_start
            )
            > 1e-6
        ):
            raise RuntimeConfigurationError(
                "resumed vLLM service journal identity differs from its checkpoint"
            )
        self._carried_service_seconds = max(
            self._carried_service_seconds,
            latest.elapsed_microseconds / 1_000_000,
        )
        self._service_journal_opened = True

    def _validate_resume_lease(
        self,
        service_pid: int,
        *,
        controller_restart_handoff: bool = False,
    ) -> None:
        lease = self._prior_service_lease
        if lease is None:
            raise RuntimeConfigurationError(
                "resumed vLLM service has no matching durable service lease"
            )
        baseline = lease.get("ledger_allocated_seconds_before_session")
        if isinstance(baseline, bool) or not isinstance(baseline, int | float):
            raise RuntimeConfigurationError("resumed vLLM service lease baseline is invalid")
        expected = {
            "configuration_hash": self.configuration.configuration_hash,
            "lease_state": ("controller_restart_handoff" if controller_restart_handoff else "live"),
            "session_id": self._session_id,
            "accounting_session_id": self._accounting_session_id,
            "service_pid": service_pid,
            "process_start_ticks": self._last_process_start_ticks,
            "process_command_sha256": self._last_process_command_sha256,
            "process_group_id": self._last_process_group_id,
            "process_session_id": self._last_process_session_id,
            "service_instance_token_sha256": (self._service_instance_token_sha256),
            "service_started_at": cast(datetime, self._started_at).isoformat(),
        }
        mismatches = [name for name, value in expected.items() if lease.get(name) != value]
        if (
            self._allocated_at_start is None
            or abs(float(baseline) - self._allocated_at_start) > 1e-6
        ):
            mismatches.append("ledger_allocated_seconds_before_session")
        if mismatches:
            raise RuntimeConfigurationError(
                "resumed vLLM service lease identity differs from its checkpoint: "
                + ", ".join(mismatches)
            )

    def _journal_verified_process_stop(self) -> None:
        if self._process_stop_journaled or not self._service_journal_opened:
            return
        if (
            self._accounting_session_id is None
            or self._physically_stopped_at is None
            or self._physically_stopped_seconds is None
        ):
            raise RuntimeError("verified vLLM process stop identity is incomplete")
        try:
            self._observe_service_journal(
                process_stopped=True,
                observed_at=self._physically_stopped_at,
                elapsed_seconds=self._physically_stopped_seconds,
                details={
                    "process_and_endpoint_absence_verified": True,
                    "prior_stop_contradicted_by_exact_live_identity": (
                        self._prior_process_stop_contradicted
                    ),
                },
            )
        except BaseException:
            # The meter raises a strict hard-limit exception *after* persisting
            # the observation.  Preserve that durable transition and let final
            # reconciliation surface the same hard-boundary failure.
            latest = self.meter.ledger.latest_gpu_service_journal(self._accounting_session_id)
            if (
                latest is None
                or latest.state is not GpuServiceJournalState.PROCESS_STOPPED
                or _parse_aware_datetime("service journal stop", latest.observed_at)
                != self._physically_stopped_at
            ):
                raise
        self._process_stop_journaled = True

    def _service_heartbeat(self) -> None:
        while not self._service_heartbeat_stop.wait(self.service_heartbeat_interval_seconds):
            try:
                self.require_hard_stop_margin()
                self._observe_service_journal(
                    details={
                        "service_pid": None if self._process is None else self._process.pid,
                    }
                )
            except BaseException as exc:
                self._service_heartbeat_failure = exc
                self._service_heartbeat_stop.set()
                # Never signal directly from the heartbeat thread.  A periodic
                # resource sample may still own a ledger transaction, and the
                # ordinary stop path would also try to join this thread.  The
                # coordinator drains both owners before using the one bound
                # process-control path.
                self.request_emergency_stop()
                return

    def _start_service_heartbeat(self) -> None:
        if not self._service_journal_opened:
            return
        if self._service_heartbeat_thread is not None and self._service_heartbeat_thread.is_alive():
            raise RuntimeError("vLLM service heartbeat is already running")
        if self._service_heartbeat_failure is not None:
            raise RuntimeError("vLLM service heartbeat previously failed") from (
                self._service_heartbeat_failure
            )
        self._service_heartbeat_stop.clear()
        self._service_heartbeat_thread = threading.Thread(
            target=self._service_heartbeat,
            name=f"vllm-service-heartbeat-{self._accounting_session_id}",
            daemon=True,
        )
        self._service_heartbeat_thread.start()

    def _stop_service_heartbeat(self) -> None:
        self._service_heartbeat_stop.set()
        thread = self._service_heartbeat_thread
        if thread is None:
            return
        thread.join(timeout=self.service_heartbeat_interval_seconds + 1.0)
        if thread.is_alive():
            raise RuntimeError("vLLM service heartbeat did not stop")
        self._service_heartbeat_thread = None

    def _raise_service_heartbeat_failure(self) -> None:
        if self._service_heartbeat_failure is not None:
            raise RuntimeError("vLLM service heartbeat failed") from (
                self._service_heartbeat_failure
            )

    @property
    def pid(self) -> int:
        if self._process is None:
            raise RuntimeError("vLLM service has no controlled process")
        return self._process.pid

    @property
    def last_recovered_process_identity(
        self,
    ) -> RecoveredServiceProcessIdentity | None:
        """Return exact lease identity only after successful terminal recovery."""

        return self._last_recovered_process_identity

    def read_authoritative_service_lease(self) -> dict[str, object] | None:
        """Read the atomic lease snapshot under the exclusive flock.

        Old leases created before the atomic sidecar existed are read from the
        flock payload by :meth:`_acquire_service_lock`.  Callers therefore get
        one verified authoritative view without independently parsing the
        crash-vulnerable compatibility mirror.
        """

        acquired_here = self._acquire_service_lock()
        try:
            return copy.deepcopy(self._prior_service_lease)
        finally:
            if acquired_here:
                self._release_service_lock()

    def _open_log_stream(self) -> BinaryIO | int:
        if self.log_path is None:
            return subprocess.DEVNULL
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        flags = os.O_APPEND | os.O_CREAT | os.O_WRONLY
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        descriptor = os.open(self.log_path, flags, 0o600)
        os.chmod(self.log_path, 0o600)
        self._log_stream = os.fdopen(descriptor, "ab", buffering=0)
        return self._log_stream

    def _close_log_stream(self) -> None:
        if self._log_stream is not None:
            self._log_stream.close()
            self._log_stream = None

    def _prepare_service_instance_token(self) -> None:
        token = os.urandom(32).hex()
        self._service_instance_token = token
        self._service_instance_token_sha256 = None
        self._last_process_group_id = None
        self._last_process_session_id = None
        self._process_identity_proc_root = PROC_ROOT

    def _service_environment(self) -> dict[str, str]:
        environment = self.configuration.environment()
        token = self._service_instance_token
        if token is not None:
            environment[SERVICE_INSTANCE_ENVIRONMENT_KEY] = token
        return environment

    def _capture_process_group_identity(self, pid: int, *, required: bool) -> None:
        token = self._service_instance_token
        try:
            _state, process_group, session_id = _process_stat_identity(pid)
            observed_token_sha256 = _process_environment_instance_sha256(pid)
        except (
            FileNotFoundError,
            ProcessLookupError,
            PermissionError,
            OSError,
            RuntimeError,
        ) as exc:
            if required:
                raise RuntimeConfigurationError(
                    "cannot persist the controlled vLLM process-group identity"
                ) from exc
            self._last_process_group_id = None
            self._last_process_session_id = None
            self._service_instance_token_sha256 = None
            return
        expected_token_sha256 = (
            None if token is None else hashlib.sha256(token.encode("ascii")).hexdigest()
        )
        if (
            process_group != pid
            or session_id != pid
            or observed_token_sha256 != expected_token_sha256
            or expected_token_sha256 is None
        ):
            if required:
                raise RuntimeConfigurationError(
                    "vLLM process is not the exact launcher-created token-bound session leader"
                )
            self._last_process_group_id = None
            self._last_process_session_id = None
            self._service_instance_token_sha256 = None
            return
        self._last_process_group_id = process_group
        self._last_process_session_id = session_id
        self._service_instance_token_sha256 = expected_token_sha256

    def _validate_bound_service_process_group(self, process_group: int) -> None:
        persisted_group = self._last_process_group_id
        persisted_session = self._last_process_session_id
        token_sha256 = self._service_instance_token_sha256
        if persisted_group is None and persisted_session is None and token_sha256 is None:
            # Compatibility path for a live-leader v3 lease and test doubles.
            return
        if (
            persisted_group != process_group
            or persisted_session != process_group
            or not isinstance(token_sha256, str)
        ):
            raise RuntimeConfigurationError("service process-group signal identity is incomplete")
        members = _bound_process_group_members(
            process_group,
            persisted_session,
            token_sha256,
            proc_root=self._process_identity_proc_root,
        )
        if not members:
            raise ProcessLookupError(process_group)

    def _signal_bound_service_process_group(
        self,
        process_group: int,
        signal_number: int,
    ) -> None:
        self._validate_bound_service_process_group(process_group)
        if (
            self._service_instance_token_sha256 is not None
            and self.process_group_signaler is _signal_controlled_process_group
        ):
            # The exact inherited token authorizes the persisted PGID even when
            # the original leader has already exited.
            os.killpg(process_group, signal_number)
        else:
            self.process_group_signaler(process_group, signal_number)

    def _prepare_durable_exec_gate(self) -> bool:
        """Persist a pre-spawn intent whose child cannot exec after owner death.

        ``subprocess.Popen`` returns only after its own fork/exec handshake, which
        leaves an otherwise unavoidable interval before the launched PID can be
        written to the service lease.  The production launcher therefore starts a
        tiny, session-leading supervisor that blocks on an inherited pipe.  The
        target argv is released only after PID/start-ticks/argv intent are durable.
        EOF (including abrupt controller death) makes the supervisor exit without
        executing vLLM.
        """

        self._last_service_pid = None
        self._last_process_start_ticks = None
        self._prepare_service_instance_token()
        if self.popen_factory is not subprocess.Popen:
            self._last_process_command_sha256 = None
            self._launch_gate_token = None
            self._launch_gate_token_sha256 = None
            self._launch_supervisor_command_sha256 = None
            return False
        token = os.urandom(32)
        self._launch_gate_token = token
        self._launch_gate_token_sha256 = hashlib.sha256(token).hexdigest()
        self._launch_supervisor_command_sha256 = None
        self._last_process_command_sha256 = canonical_sha256(list(self.configuration.command()))
        return True

    @staticmethod
    def _write_exec_gate_token(descriptor: int, token: bytes) -> None:
        offset = 0
        while offset < len(token):
            try:
                written = os.write(descriptor, token[offset:])
            except InterruptedError:
                continue
            if written <= 0:
                raise RuntimeError("durable vLLM exec gate accepted no bytes")
            offset += written

    def _wait_for_durable_exec(
        self,
        pid: int,
        expected_start_ticks: int,
        *,
        startup_deadline: float,
    ) -> None:
        expected_command_sha256 = self._last_process_command_sha256
        if expected_command_sha256 is None:
            raise RuntimeError("durable vLLM exec gate lacks its target command hash")
        exec_gate_deadline = self.monotonic_clock() + DURABLE_EXEC_GATE_WATCHDOG_SECONDS
        deadline = min(exec_gate_deadline, startup_deadline)
        while True:
            process = self._process
            if process is None or process.poll() is not None:
                raise RuntimeConfigurationError(
                    "durable vLLM launch supervisor exited before target exec"
                )
            try:
                observed_start_ticks = _process_start_ticks(pid)
                observed_command_sha256 = _process_command_sha256(pid)
            except (FileNotFoundError, ProcessLookupError, RuntimeError, OSError):
                observed_start_ticks = None
                observed_command_sha256 = None
            if observed_start_ticks is not None and observed_start_ticks != expected_start_ticks:
                raise RuntimeConfigurationError(
                    "durable vLLM launch supervisor PID changed before target exec"
                )
            if observed_command_sha256 == expected_command_sha256:
                return
            if self.monotonic_clock() >= deadline:
                if deadline == startup_deadline:
                    raise RuntimeWatchdogTimeout(
                        "durable vLLM exec exhausted the service-start wall deadline"
                    )
                raise RuntimeConfigurationError(
                    "durable vLLM launch supervisor did not exec the frozen target argv"
                )
            self.sleep(0.01)

    def _spawn(self, *, startup_deadline: float) -> None:
        if self.monotonic_clock() >= startup_deadline:
            raise RuntimeWatchdogTimeout(
                "vLLM service startup exhausted its wall deadline before process spawn"
            )
        output = self._open_log_stream()
        command = self.configuration.command()
        if self.popen_factory is not subprocess.Popen:
            self._process = self.popen_factory(
                command,
                env=self._service_environment(),
                stdin=subprocess.DEVNULL,
                stdout=output,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            self._last_service_pid = self._process.pid
            self._last_process_command_sha256 = canonical_sha256(list(command))
            try:
                self._last_process_start_ticks = _process_start_ticks(self._process.pid)
                observed_command_sha256 = _process_command_sha256(self._process.pid)
            except (OSError, RuntimeError):
                # Unit-test process handles do not have procfs entries.
                self._last_process_start_ticks = None
                self._last_process_command_sha256 = None
            else:
                if observed_command_sha256 != self._last_process_command_sha256:
                    raise RuntimeConfigurationError(
                        "launched vLLM process command differs from its frozen argv"
                    )
            self._capture_process_group_identity(self._process.pid, required=False)
            self._write_service_lock_metadata(
                lease_state="starting",
                service_pid=self._process.pid,
            )
            allowed_cpus = set(
                sorted(self.available_cpu_sampler())[: self.configuration.cpu_workers]
            )
            if not allowed_cpus:
                raise RuntimeConfigurationError("vLLM process has no permitted CPU affinity")
            self.affinity_setter(self._process.pid, allowed_cpus)
            return

        token = self._launch_gate_token
        if token is None or hashlib.sha256(token).hexdigest() != self._launch_gate_token_sha256:
            raise RuntimeError("production vLLM launch lacks its durable exec-gate identity")
        read_descriptor, write_descriptor = os.pipe()
        supervisor_command = (
            sys.executable,
            "-I",
            "-c",
            _DURABLE_EXEC_GATE_PROGRAM,
            str(read_descriptor),
            token.hex(),
            canonical_json(list(command)),
        )
        try:
            self._process = subprocess.Popen(
                supervisor_command,
                env=self._service_environment(),
                stdin=subprocess.DEVNULL,
                stdout=output,
                stderr=subprocess.STDOUT,
                start_new_session=True,
                pass_fds=(read_descriptor,),
            )
            os.close(read_descriptor)
            read_descriptor = -1
            self._last_service_pid = self._process.pid
            try:
                self._last_process_start_ticks = _process_start_ticks(self._process.pid)
                observed_supervisor_hash = _process_command_sha256(self._process.pid)
            except (OSError, RuntimeError) as exc:
                raise RuntimeConfigurationError(
                    "cannot persist the durable vLLM launch-supervisor identity"
                ) from exc
            expected_supervisor_hash = canonical_sha256(list(supervisor_command))
            if observed_supervisor_hash != expected_supervisor_hash:
                raise RuntimeConfigurationError(
                    "durable vLLM launch supervisor differs from its frozen argv"
                )
            self._capture_process_group_identity(self._process.pid, required=True)
            self._launch_supervisor_command_sha256 = observed_supervisor_hash
            self._write_service_lock_metadata(
                lease_state="launch_gate_pending",
                service_pid=self._process.pid,
            )
            allowed_cpus = set(
                sorted(self.available_cpu_sampler())[: self.configuration.cpu_workers]
            )
            if not allowed_cpus:
                raise RuntimeConfigurationError("vLLM process has no permitted CPU affinity")
            self.affinity_setter(self._process.pid, allowed_cpus)
            self._write_exec_gate_token(write_descriptor, token)
            os.close(write_descriptor)
            write_descriptor = -1
            self._launch_gate_token = None
            self._wait_for_durable_exec(
                self._process.pid,
                cast(int, self._last_process_start_ticks),
                startup_deadline=startup_deadline,
            )
            self._write_service_lock_metadata(
                lease_state="starting",
                service_pid=self._process.pid,
            )
        finally:
            if read_descriptor >= 0:
                os.close(read_descriptor)
            if write_descriptor >= 0:
                os.close(write_descriptor)

    def _ready(self, timeout_seconds: float = 4.0) -> bool:
        if not math.isfinite(timeout_seconds) or timeout_seconds <= 0:
            raise RuntimeConfigurationError("readiness timeout must be positive and finite")
        if self.readiness_check is None:
            # ``client.ready`` performs two sequential HTTP operations.  Give
            # each half of the total readiness deadline.
            return self.client.ready(
                max(0.001, timeout_seconds / 2),
                model_name=self.configuration.served_model_name,
            )
        completed = threading.Event()
        result: list[bool] = []
        failure: list[BaseException] = []

        def check() -> None:
            try:
                result.append(bool(cast(Callable[[], bool], self.readiness_check)()))
            except BaseException as exc:
                failure.append(exc)
            finally:
                completed.set()

        worker = threading.Thread(
            target=check,
            name="vllm-readiness-check",
            daemon=True,
        )
        worker.start()
        if not completed.wait(timeout_seconds):
            raise RuntimeWatchdogTimeout("vLLM readiness check exceeded its wall deadline")
        if failure:
            raise failure[0]
        if len(result) != 1:
            raise RuntimeTransportError("vLLM readiness check returned no result")
        return result[0]

    def _wait_until_healthy(
        self,
        watchdog_seconds: float,
        *,
        absolute_deadline: float | None = None,
    ) -> None:
        deadline = self.monotonic_clock() + watchdog_seconds
        if absolute_deadline is not None:
            deadline = min(deadline, absolute_deadline)
        while self.monotonic_clock() < deadline:
            self._raise_service_heartbeat_failure()
            if self._process is None or self._process.poll() is not None:
                self.state = ServiceState.FAILED
                raise RuntimeTransportError("vLLM process exited before becoming healthy")
            remaining = max(0.001, deadline - self.monotonic_clock())
            if self._ready(remaining):
                self.state = ServiceState.READY
                return
            self.sleep(min(0.25, max(0.0, deadline - self.monotonic_clock())))
        self.state = ServiceState.FAILED
        raise RuntimeWatchdogTimeout("vLLM service did not become healthy before watchdog")

    def _start_unmetered(
        self,
        *,
        session_id: str,
        event_id: str,
        watchdog_seconds: float,
    ) -> None:
        if self.state is not ServiceState.STOPPED:
            raise RuntimeError("vLLM service can start only from stopped state")
        if not math.isfinite(watchdog_seconds) or watchdog_seconds <= 0:
            raise RuntimeConfigurationError(
                "vLLM service-start watchdog must be positive and finite"
            )
        startup_deadline = self.monotonic_clock() + watchdog_seconds
        acquired_here = self._acquire_service_lock()
        prior_state = (
            None
            if self._prior_service_lease is None
            else self._prior_service_lease.get("lease_state")
        )
        if acquired_here and prior_state not in {None, "stopped_verified"}:
            self._release_service_lock()
            raise RuntimeConfigurationError(
                "an unresolved prior vLLM service lease forbids a new launch"
            )
        try:
            remaining_preflight_seconds = startup_deadline - self.monotonic_clock()
            if remaining_preflight_seconds <= 0:
                raise RuntimeWatchdogTimeout(
                    "vLLM service startup exhausted its wall deadline before endpoint preflight"
                )
            endpoint_already_live = self._endpoint_live(
                max(0.001, min(0.25, remaining_preflight_seconds))
            )
            if self.preflight_endpoint_check is not None:
                endpoint_already_live = (
                    bool(self.preflight_endpoint_check()) or endpoint_already_live
                )
        except BaseException:
            self._release_service_lock()
            raise
        if endpoint_already_live:
            self._release_service_lock()
            raise RuntimeConfigurationError(
                "configured loopback port already exposes a service; refusing a duplicate launch"
            )
        if self.monotonic_clock() >= startup_deadline:
            self._release_service_lock()
            raise RuntimeWatchdogTimeout(
                "vLLM service startup exhausted its wall deadline during endpoint preflight"
            )
        startup_watchdog: ResourceWatchdog | None = None
        try:
            self.state = ServiceState.STARTING
            self._session_id = session_id
            if self._accounting_session_id is None:
                self._accounting_session_id = event_id
            if self._started_monotonic is None:
                self._started_monotonic = self.monotonic_clock()
            if self._started_at is None:
                started_at = self.wall_clock()
                if started_at.tzinfo is None or started_at.utcoffset() is None:
                    raise RuntimeConfigurationError("service start clock must be timezone-aware")
                self._started_at = started_at
            if self._allocated_at_start is None:
                self._allocated_at_start = self.meter.actual_allocated_gpu_seconds
            durable_exec_gate = self._prepare_durable_exec_gate()
            self._write_service_lock_metadata(
                lease_state=("launch_supervisor_pending" if durable_exec_gate else "starting")
            )
            self._open_service_journal()
            self._start_service_heartbeat()
            self._spawn(startup_deadline=startup_deadline)
            self._raise_service_heartbeat_failure()
            self._write_service_lock_metadata(lease_state="live", service_pid=self.pid)
            remaining_startup_seconds = startup_deadline - self.monotonic_clock()
            if remaining_startup_seconds <= 0:
                raise RuntimeWatchdogTimeout(
                    "vLLM service startup exhausted its wall deadline before readiness"
                )
            if self.startup_resource_sampler is not None:
                startup_watchdog = ResourceWatchdog(
                    sampler=self.startup_resource_sampler,
                    root_pid=self.pid,
                    sample_prefix=f"{event_id}-startup",
                    interval_seconds=self.startup_sample_interval_seconds,
                    sample_completion_timeout_seconds=max(
                        0.001,
                        min(
                            DEFAULT_RESOURCE_SAMPLE_COMPLETION_SECONDS,
                            remaining_startup_seconds,
                        ),
                    ),
                    allocation_guard=self.require_hard_stop_margin,
                    on_failure=lambda _: self.request_emergency_stop(),
                )
                self._startup_resource_watchdog = startup_watchdog
                startup_watchdog.start()
            remaining_startup_seconds = startup_deadline - self.monotonic_clock()
            if remaining_startup_seconds <= 0:
                raise RuntimeWatchdogTimeout(
                    "vLLM service startup exhausted its wall deadline before readiness"
                )
            self._wait_until_healthy(
                remaining_startup_seconds,
                absolute_deadline=startup_deadline,
            )
            if startup_watchdog is not None:
                startup_watchdog.stop(
                    completion_timeout_seconds=max(
                        0.0,
                        startup_deadline - self.monotonic_clock(),
                    )
                )
                self._startup_resource_watchdog = None
        except BaseException:
            # Signal the verified service before draining observers. A slow
            # sample must never extend allocation or suppress physical cleanup.
            self._stop_process(release_lock=False)
            raise

    def start(
        self,
        *,
        session_id: str,
        event_id: str,
        watchdog_seconds: float = DEFAULT_SERVICE_START_WATCHDOG_SECONDS,
        remaining_required_seconds: float = 0,
        admission_forecast_seconds: float | None = None,
        contingency_unlocked: bool = False,
        essential_recovery: bool = False,
    ) -> None:
        _require_plain_identifier("session_id", session_id)
        if (
            isinstance(watchdog_seconds, bool)
            or not isinstance(watchdog_seconds, int | float)
            or not math.isfinite(watchdog_seconds)
            or watchdog_seconds <= 0
        ):
            raise RuntimeConfigurationError("service-start watchdog must be positive and finite")
        if type(contingency_unlocked) is not bool or type(essential_recovery) is not bool:
            raise RuntimeConfigurationError(
                "service-start contingency authority flags must be exact booleans"
            )
        if contingency_unlocked != essential_recovery:
            raise RuntimeConfigurationError(
                "service-start contingency requires paired unlock and essential-recovery authority"
            )
        effective_admission_forecast_seconds = (
            watchdog_seconds if admission_forecast_seconds is None else admission_forecast_seconds
        )
        if (
            isinstance(effective_admission_forecast_seconds, bool)
            or not isinstance(effective_admission_forecast_seconds, int | float)
            or not math.isfinite(effective_admission_forecast_seconds)
            or effective_admission_forecast_seconds <= 0
        ):
            raise RuntimeConfigurationError(
                "service-start admission forecast must be positive and finite"
            )
        if effective_admission_forecast_seconds < watchdog_seconds:
            raise RuntimeConfigurationError(
                "service-start admission forecast cannot be below its watchdog"
            )
        if (
            self._service_lock_stream is not None
            or self._started_at is not None
            or self._accounting_session_id is not None
        ):
            raise RuntimeError("a prior vLLM service session must be reconciled before a new start")
        with self._emergency_stop_lock:
            emergency_thread = self._emergency_stop_thread
            if emergency_thread is not None and emergency_thread.is_alive():
                raise RuntimeError("an emergency-stop coordinator is still active")
            # A terminally reconciled prior session may reuse this controller;
            # its completed coordinator must not contaminate the new lifecycle.
            self._emergency_stop_thread = None
            self._emergency_stop_failure = None
        if isinstance(self.startup_resource_sampler, ResourceSampler):
            # Invariant full census occurs outside the allocated session. Exact
            # live identity/resource checks still run after spawn and before calls.
            self.startup_resource_sampler.prepare()
        try:
            with self.meter.session_start(
                event_id=event_id,
                maximum_seconds=watchdog_seconds,
                admission_forecast_seconds=effective_admission_forecast_seconds,
                remaining_required_seconds=remaining_required_seconds,
                contingency_unlocked=contingency_unlocked,
                essential_recovery=essential_recovery,
                details={
                    "session_id": session_id,
                    "configuration_hash": self.configuration.configuration_hash,
                    "service_start_watchdog_seconds": watchdog_seconds,
                    "admission_forecast_seconds": effective_admission_forecast_seconds,
                    "contingency_unlocked": contingency_unlocked,
                    "essential_recovery": essential_recovery,
                },
            ):
                self._start_unmetered(
                    session_id=session_id,
                    event_id=event_id,
                    watchdog_seconds=watchdog_seconds,
                )
        except BaseException:
            if self._process is None and isinstance(self.startup_resource_sampler, ResourceSampler):
                self.startup_resource_sampler.close_probes()
            raise

    def restart(
        self,
        *,
        event_id: str,
        watchdog_seconds: float,
        remaining_required_seconds: float = 0,
    ) -> None:
        if self._session_id is None:
            raise RuntimeError("cannot restart a service without a session identity")
        self.require_ready()
        self.require_service_capacity(
            watchdog_seconds,
            remaining_required_seconds=remaining_required_seconds,
        )
        session_id = self._session_id
        with self.meter.restart(
            event_id=event_id,
            maximum_seconds=watchdog_seconds,
            remaining_required_seconds=remaining_required_seconds,
            details={
                "session_id": session_id,
                "configuration_hash": self.configuration.configuration_hash,
            },
        ):
            self._stop_process(
                release_lock=False,
                continuing_service=True,
            )
            self._start_unmetered(
                session_id=session_id,
                event_id=event_id,
                watchdog_seconds=watchdog_seconds,
            )

    def write_resume_checkpoint(
        self,
        path: Path,
        *,
        proc_root: Path = PROC_ROOT,
        controller_restart_handoff: bool = False,
    ) -> None:
        """Persist a PID-reuse-safe operational checkpoint without a model path."""

        self.require_ready()
        observed_start_ticks = _process_start_ticks(self.pid, proc_root)
        observed_command_sha256 = _process_command_sha256(self.pid, proc_root)
        self._last_process_start_ticks = observed_start_ticks
        self._last_process_command_sha256 = observed_command_sha256
        # Refresh the authoritative atomic lease before publishing a checkpoint
        # so neither recovery source can describe a weaker process identity.
        self._write_service_lock_metadata(
            lease_state=("controller_restart_handoff" if controller_restart_handoff else "live"),
            service_pid=self.pid,
        )
        payload = {
            "schema_version": SCHEMA_VERSION,
            "configuration_hash": self.configuration.configuration_hash,
            "controller_pid": os.getpid(),
            "controller_restart_handoff": controller_restart_handoff,
            "session_id": self._session_id,
            "accounting_session_id": self._accounting_session_id,
            "pid": self.pid,
            "process_start_ticks": observed_start_ticks,
            "process_command_sha256": observed_command_sha256,
            "process_group_id": self._last_process_group_id,
            "process_session_id": self._last_process_session_id,
            "service_instance_token_sha256": (self._service_instance_token_sha256),
            "ledger_allocated_seconds_before_session": self._allocated_at_start,
            "session_service_seconds": self._elapsed_service_seconds(),
            "session_started_at": cast(datetime, self._started_at).isoformat(),
            "recorded_at": self.wall_clock().isoformat(),
        }
        _atomic_write_private_json(path, payload)

    def detach_for_controller_restart(
        self,
        path: Path,
        *,
        proc_root: Path = PROC_ROOT,
    ) -> None:
        """Leave one verified vLLM process live for a new controller invocation.

        This is intentionally narrower than a model restart: the service PID and
        loaded weights remain unchanged.  The exact PID/start-ticks/configuration
        checkpoint lets a later controller adopt it without a second load.  Any
        exception before ownership is released leaves this controller capable of
        performing the normal fail-safe shutdown.
        """

        self.require_ready()
        if self._periodic_resource_watchdog is not None:
            raise RuntimeError("periodic resource watchdog must stop before controller detach")
        with self._emergency_stop_lock:
            emergency_thread = self._emergency_stop_thread
            if emergency_thread is not None:
                raise RuntimeError("emergency-stop coordinator forbids controller detach")
        self.write_resume_checkpoint(
            path,
            proc_root=proc_root,
            controller_restart_handoff=True,
        )
        self._observe_service_journal(
            details={
                "controller_restart_handoff": True,
                "controller_pid": os.getpid(),
                "service_pid": self.pid,
            }
        )
        self._stop_service_heartbeat()
        process = self._process
        assert process is not None
        self._write_service_lock_metadata(
            lease_state="controller_restart_handoff",
            service_pid=process.pid,
        )
        self._close_log_stream()
        self._release_service_lock()
        self._process = None
        self._last_service_pid = None
        self._last_process_start_ticks = None
        self._last_process_command_sha256 = None
        self._last_process_group_id = None
        self._last_process_session_id = None
        self._service_instance_token = None
        self._service_instance_token_sha256 = None
        self._process_identity_proc_root = PROC_ROOT
        self._session_id = None
        self._accounting_session_id = None
        self._started_at = None
        self._started_monotonic = None
        self._allocated_at_start = None
        self._carried_service_seconds = 0.0
        self._service_journal_opened = False
        self.state = ServiceState.STOPPED

    def resume_live_service_lease(
        self,
        *,
        expected_session_id: str,
        expected_event_id: str,
        watchdog_seconds: float = DEFAULT_SERVICE_START_WATCHDOG_SECONDS,
        proc_root: Path = PROC_ROOT,
        adopted_factory: Callable[[int], ProcessHandle] = _AdoptedProcess,
        cleanup_only: bool = False,
    ) -> bool:
        """Adopt one exact live lease when its controller died before checkpointing.

        This path never launches a process and never opens a second GPU event.  The
        exclusive project lock proves the prior controller released ownership; the
        lease and open service journal then have to agree on configuration, logical
        session, accounting event, start time, and baseline.  PID start ticks and an
        exact argv hash close the PID-reuse and lookalike-process holes.  ``False``
        means the validated process is absent and is deliberately distinct from
        terminal stale-journal reconciliation.  ``cleanup_only`` additionally
        permits an exact launch-gate or terminalizing lease to be adopted solely
        so that :meth:`shutdown` can kill it; it never performs readiness or starts
        a heartbeat.
        """

        _require_plain_identifier("expected_session_id", expected_session_id)
        _require_plain_identifier("expected_event_id", expected_event_id)
        if not math.isfinite(watchdog_seconds) or watchdog_seconds <= 0:
            raise RuntimeConfigurationError(
                "live-lease recovery watchdog must be positive and finite"
            )
        if (
            self.state is not ServiceState.STOPPED
            or self._process is not None
            or self._service_lock_stream is not None
            or self._started_at is not None
            or self._accounting_session_id is not None
        ):
            raise RuntimeError("live-lease recovery requires a fresh stopped controller")
        self._last_recovered_process_identity = None
        acquired_here = self._acquire_service_lock()
        candidate_pid: int | None = None
        candidate_process_group: int | None = None
        try:
            lease = self._prior_service_lease
            if lease is None or lease.get("lease_state") == "stopped_verified":
                return False
            lease_state = lease.get("lease_state")
            if lease_state not in {
                "launch_supervisor_pending",
                "launch_gate_pending",
                "starting",
                "live",
                "controller_restart_handoff",
                "accounting_pending",
                "shutdown_unverified",
            }:
                raise RuntimeConfigurationError("live vLLM service lease state is invalid")
            pid = lease.get("service_pid")
            start_ticks = lease.get("process_start_ticks")
            command_sha256 = lease.get("process_command_sha256")
            process_group_id = lease.get("process_group_id")
            process_session_id = lease.get("process_session_id")
            instance_token_sha256 = lease.get("service_instance_token_sha256")
            session_id = lease.get("session_id")
            accounting_session_id = lease.get("accounting_session_id")
            started_at_text = lease.get("service_started_at")
            baseline = lease.get("ledger_allocated_seconds_before_session")
            launch_protocol = lease.get("launch_protocol")
            launch_token_sha256 = lease.get("launch_gate_token_sha256")
            launch_supervisor_sha256 = lease.get("launch_supervisor_command_sha256")
            launch_pending = lease_state in {
                "launch_supervisor_pending",
                "launch_gate_pending",
            }
            launch_identity_present = any(
                value is not None
                for value in (
                    launch_protocol,
                    launch_token_sha256,
                    launch_supervisor_sha256,
                )
            )
            if (launch_pending or launch_identity_present) and (
                launch_protocol != DURABLE_EXEC_GATE_PROTOCOL
                or not _is_canonical_sha256(launch_token_sha256)
                or (
                    lease_state != "launch_supervisor_pending"
                    and not _is_canonical_sha256(launch_supervisor_sha256)
                )
            ):
                raise RuntimeConfigurationError(
                    "pending vLLM launch lease lacks its durable exec-gate identity"
                )
            if lease_state == "launch_supervisor_pending" and pid is None:
                if not cleanup_only:
                    raise RuntimeConfigurationError(
                        "pre-exec vLLM launch lease can only be recovered for cleanup"
                    )
                if self._endpoint_live(0.25):
                    raise RuntimeConfigurationError(
                        "PID-less pre-exec vLLM lease unexpectedly exposes an endpoint"
                    )
                return False
            if (
                isinstance(pid, bool)
                or not isinstance(pid, int)
                or pid <= 0
                or isinstance(start_ticks, bool)
                or not isinstance(start_ticks, int)
                or start_ticks <= 0
                or not isinstance(command_sha256, str)
                or len(command_sha256) != 64
                or any(character not in "0123456789abcdef" for character in command_sha256)
                or not isinstance(session_id, str)
                or not isinstance(accounting_session_id, str)
                or not isinstance(started_at_text, str)
                or isinstance(baseline, bool)
                or not isinstance(baseline, int | float)
                or not math.isfinite(float(baseline))
                or baseline < 0
            ):
                raise RuntimeConfigurationError(
                    "live vLLM lease lacks an exact recoverable process identity"
                )
            candidate_pid = pid
            process_group_identity_present = any(
                value is not None
                for value in (
                    process_group_id,
                    process_session_id,
                    instance_token_sha256,
                )
            )
            if process_group_identity_present and (
                isinstance(process_group_id, bool)
                or not isinstance(process_group_id, int)
                or process_group_id != pid
                or isinstance(process_session_id, bool)
                or not isinstance(process_session_id, int)
                or process_session_id != pid
                or not _is_canonical_sha256(instance_token_sha256)
            ):
                raise RuntimeConfigurationError(
                    "live vLLM lease process-group identity is incomplete"
                )
            bound_process_group = (
                cast(int, process_group_id) if process_group_identity_present else pid
            )
            candidate_process_group = bound_process_group
            expected_command_sha256 = canonical_sha256(list(self.configuration.command()))
            if (
                lease.get("configuration_hash") != self.configuration.configuration_hash
                or session_id != expected_session_id
                or accounting_session_id != expected_event_id
                or command_sha256 != expected_command_sha256
            ):
                raise RuntimeConfigurationError(
                    "live vLLM lease differs from the expected activation identity"
                )
            started_at = _parse_aware_datetime(
                "live service lease start",
                started_at_text,
            )
            latest = self.meter.ledger.latest_gpu_service_journal(expected_event_id)
            allowed_journal_states = {
                GpuServiceJournalState.OPENED,
                GpuServiceJournalState.HEARTBEAT,
            }
            if cleanup_only:
                allowed_journal_states.add(GpuServiceJournalState.PROCESS_STOPPED)
            if latest is None or latest.state not in allowed_journal_states:
                raise RuntimeConfigurationError(
                    "live vLLM lease has no matching open service journal"
                )
            if (
                latest.session_id != expected_session_id
                or latest.configuration_hash != self.configuration.configuration_hash
                or _parse_aware_datetime(
                    "live service journal start",
                    latest.service_started_at,
                )
                != started_at
                or abs(
                    latest.ledger_allocated_microseconds_before_session / 1_000_000
                    - float(baseline)
                )
                > 1e-6
            ):
                raise RuntimeConfigurationError(
                    "live vLLM lease and service journal identities differ"
                )
            if cleanup_only and process_group_identity_present:
                # Install the already-validated durable kill identity before
                # procfs/liveness inspection.  If one of those fallible checks
                # fails, emergency cleanup can still address only the exact
                # token/start-tick/PGID/SID-bound group, and any unresolved
                # lease rewrite retains these fields.
                self._last_service_pid = pid
                self._last_process_start_ticks = start_ticks
                self._last_process_command_sha256 = command_sha256
                self._last_process_group_id = bound_process_group
                self._last_process_session_id = cast(int, process_session_id)
                self._service_instance_token = None
                self._service_instance_token_sha256 = cast(
                    str,
                    instance_token_sha256,
                )
                self._process_identity_proc_root = proc_root
                self._launch_gate_token_sha256 = (
                    cast(str, launch_token_sha256) if isinstance(launch_token_sha256, str) else None
                )
                self._launch_supervisor_command_sha256 = (
                    cast(str, launch_supervisor_sha256)
                    if isinstance(launch_supervisor_sha256, str)
                    else None
                )
            try:
                pid_live = self.process_liveness_check(pid)
                process_group_live = self.process_group_liveness_check(bound_process_group)
            except BaseException as exc:
                raise RuntimeConfigurationError(
                    "cannot verify the leased vLLM process identity"
                ) from exc
            if not pid_live and not process_group_live:
                if self._endpoint_live(0.25):
                    raise RuntimeConfigurationError(
                        "leased vLLM process is absent but its endpoint remains live"
                    )
                return False
            if (
                lease_state
                in {
                    "launch_supervisor_pending",
                    "launch_gate_pending",
                    "accounting_pending",
                    "shutdown_unverified",
                }
                and not cleanup_only
            ):
                raise RuntimeConfigurationError(
                    "non-ready vLLM lease can only be adopted for cleanup"
                )
            resumed_at = self.wall_clock()
            if resumed_at.tzinfo is None or resumed_at.utcoffset() is None:
                raise RuntimeConfigurationError("live-lease recovery clock must be aware")
            wall_service_seconds = (resumed_at - started_at).total_seconds()
            if wall_service_seconds < 0:
                raise RuntimeConfigurationError("live-lease recovery predates the service session")
            if cleanup_only:
                # Once liveness is positive, carry the matching journal identity
                # too.  A later argv/adopter failure can then kill and account
                # the service rather than detaching an unowned live allocation.
                self._last_service_pid = pid
                self._last_process_start_ticks = start_ticks
                self._last_process_command_sha256 = command_sha256
                self._last_process_group_id = (
                    bound_process_group if process_group_identity_present else None
                )
                self._last_process_session_id = (
                    cast(int, process_session_id) if process_group_identity_present else None
                )
                self._service_instance_token = None
                self._service_instance_token_sha256 = (
                    cast(str, instance_token_sha256) if process_group_identity_present else None
                )
                self._process_identity_proc_root = proc_root
                self._session_id = expected_session_id
                self._accounting_session_id = expected_event_id
                self._started_at = started_at
                self._started_monotonic = self.monotonic_clock()
                self._allocated_at_start = float(baseline)
                self._carried_service_seconds = wall_service_seconds
                self._service_journal_opened = True
                if latest.state is GpuServiceJournalState.PROCESS_STOPPED:
                    self._carried_service_seconds = max(
                        self._carried_service_seconds,
                        latest.elapsed_microseconds / 1_000_000,
                    )
                    self._prior_process_stop_contradicted = True
            group_only_adoption = not pid_live and process_group_live
            if group_only_adoption:
                if not cleanup_only or not process_group_identity_present:
                    raise RuntimeConfigurationError(
                        "an orphaned vLLM process group requires its exact v4 cleanup identity"
                    )
                members = _bound_process_group_members(
                    bound_process_group,
                    cast(int, process_session_id),
                    cast(str, instance_token_sha256),
                    proc_root=proc_root,
                )
                if not members:
                    if self._endpoint_live(0.25):
                        raise RuntimeConfigurationError(
                            "leased vLLM group disappeared while its endpoint remained live"
                        )
                    return False
                adopted: ProcessHandle = _AdoptedProcessGroup(
                    pid=bound_process_group,
                    liveness_check=self.process_group_liveness_check,
                    monotonic_clock=self.monotonic_clock,
                    sleep=self.sleep,
                )
            else:
                if not pid_live or not process_group_live:
                    raise RuntimeConfigurationError(
                        "leased vLLM PID and process-group liveness disagree"
                    )
                try:
                    observed_start_ticks = _process_start_ticks(pid, proc_root)
                except (OSError, RuntimeError) as exc:
                    raise RuntimeConfigurationError(
                        "cannot inspect the leased vLLM process start identity"
                    ) from exc
                if observed_start_ticks != start_ticks:
                    raise RuntimeConfigurationError("live vLLM lease PID was reused")
                if process_group_identity_present:
                    members = _bound_process_group_members(
                        bound_process_group,
                        cast(int, process_session_id),
                        cast(str, instance_token_sha256),
                        proc_root=proc_root,
                    )
                    if pid not in members:
                        raise RuntimeConfigurationError(
                            "live vLLM leader is absent from its bound process group"
                        )
                try:
                    observed_command_sha256 = _process_command_sha256(pid, proc_root)
                except (OSError, RuntimeError) as exc:
                    if not (cleanup_only and process_group_identity_present):
                        raise RuntimeConfigurationError(
                            "cannot inspect the leased vLLM process command line"
                        ) from exc
                    # ``setproctitle`` can intentionally make argv unparsable.
                    # Cleanup authority instead comes from the independently
                    # checked start ticks and token-bound PGID/SID membership.
                    observed_command_sha256 = None
                allowed_command_hashes = {command_sha256}
                if lease_state == "launch_gate_pending":
                    allowed_command_hashes.add(cast(str, launch_supervisor_sha256))
                if observed_command_sha256 not in allowed_command_hashes and not (
                    cleanup_only and process_group_identity_present
                ):
                    raise RuntimeConfigurationError("live vLLM lease command line changed")
                adopted = adopted_factory(pid)
            if adopted.poll() is not None:
                return False
            self._process = adopted
            self._last_service_pid = pid
            self._last_process_start_ticks = start_ticks
            self._last_process_command_sha256 = command_sha256
            self._last_process_group_id = (
                bound_process_group if process_group_identity_present else None
            )
            self._last_process_session_id = (
                cast(int, process_session_id) if process_group_identity_present else None
            )
            self._service_instance_token = None
            self._service_instance_token_sha256 = (
                cast(str, instance_token_sha256) if process_group_identity_present else None
            )
            self._process_identity_proc_root = proc_root
            self._launch_gate_token_sha256 = (
                cast(str, launch_token_sha256) if isinstance(launch_token_sha256, str) else None
            )
            self._launch_supervisor_command_sha256 = (
                cast(str, launch_supervisor_sha256)
                if isinstance(launch_supervisor_sha256, str)
                else None
            )
            self._session_id = expected_session_id
            self._accounting_session_id = expected_event_id
            self._started_at = started_at
            self._started_monotonic = self.monotonic_clock()
            self._allocated_at_start = float(baseline)
            self._carried_service_seconds = wall_service_seconds
            if latest.state is GpuServiceJournalState.PROCESS_STOPPED:
                # A physically live exact PID contradicts an earlier stop
                # observation.  Kill ownership may still transfer, but normal
                # close cannot safely reuse the earlier end timestamp.
                self._carried_service_seconds = max(
                    self._carried_service_seconds,
                    latest.elapsed_microseconds / 1_000_000,
                )
                self._service_journal_opened = True
                self._prior_process_stop_contradicted = True
            else:
                self._adopt_service_journal()
            if cleanup_only:
                self.state = ServiceState.FAILED
                self._write_service_lock_metadata(
                    lease_state=cast(str, lease_state),
                    service_pid=pid,
                )
                return True
            remaining = watchdog_seconds - wall_service_seconds
            if remaining <= 0:
                self.state = ServiceState.FAILED
                raise RuntimeWatchdogTimeout(
                    "live vLLM lease exceeded its original startup watchdog"
                )
            self.state = ServiceState.STARTING
            self._wait_until_healthy(remaining)
            self._write_service_lock_metadata(lease_state="live", service_pid=pid)
            self._observe_service_journal(
                details={
                    "recovered_without_checkpoint": True,
                    "service_pid": pid,
                }
            )
            self._start_service_heartbeat()
            return True
        finally:
            if acquired_here and self._process is None:
                endpoint_absent = False
                process_absent = candidate_pid is None
                with suppress(Exception):
                    endpoint_absent = not self._endpoint_live(0.25)
                if candidate_pid is not None:
                    with suppress(Exception):
                        process_absent = not self.process_liveness_check(
                            candidate_pid
                        ) and not self.process_group_liveness_check(
                            candidate_process_group or candidate_pid
                        )
                if endpoint_absent and process_absent:
                    self._release_service_lock()
                else:
                    self.state = ServiceState.FAILED

    def resume_from_checkpoint(
        self,
        path: Path,
        *,
        proc_root: Path = PROC_ROOT,
        adopted_factory: Callable[[int], ProcessHandle] = _AdoptedProcess,
        allow_same_controller_cleanup: bool = False,
    ) -> bool:
        """Adopt the exact still-live service; never start or meter a duplicate load."""

        if self.state is not ServiceState.STOPPED or self._process is not None:
            raise RuntimeError("resume requires a stopped controller")
        if not path.exists():
            return False
        acquired_here = self._acquire_service_lock()
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(value, Mapping):
                raise RuntimeConfigurationError("service checkpoint root must be an object")
            if value.get("configuration_hash") != self.configuration.configuration_hash:
                raise RuntimeConfigurationError("service checkpoint configuration changed")
            pid = value.get("pid")
            start_ticks = value.get("process_start_ticks")
            command_sha256 = value.get("process_command_sha256")
            process_group_id = value.get("process_group_id")
            process_session_id = value.get("process_session_id")
            instance_token_sha256 = value.get("service_instance_token_sha256")
            carried_seconds = value.get("session_service_seconds")
            ledger_before_session = value.get("ledger_allocated_seconds_before_session")
            started_at_text = value.get("session_started_at")
            session_id = value.get("session_id")
            accounting_session_id = value.get("accounting_session_id")
            checkpoint_controller_pid = value.get("controller_pid")
            controller_restart_handoff = value.get(
                "controller_restart_handoff",
                False,
            )
            if (
                isinstance(pid, bool)
                or not isinstance(pid, int)
                or pid <= 0
                or isinstance(start_ticks, bool)
                or not isinstance(start_ticks, int)
                or (command_sha256 is not None and not _is_canonical_sha256(command_sha256))
                or isinstance(carried_seconds, bool)
                or not isinstance(carried_seconds, int | float)
                or carried_seconds < 0
                or isinstance(ledger_before_session, bool)
                or not isinstance(ledger_before_session, int | float)
                or ledger_before_session < 0
                or not isinstance(started_at_text, str)
                or not isinstance(session_id, str)
                or not isinstance(accounting_session_id, str)
                or not accounting_session_id
                or isinstance(checkpoint_controller_pid, bool)
                or not isinstance(checkpoint_controller_pid, int)
                or checkpoint_controller_pid <= 0
                or not isinstance(controller_restart_handoff, bool)
            ):
                raise RuntimeConfigurationError("service checkpoint has invalid process identity")
            process_group_identity_present = any(
                item is not None
                for item in (
                    process_group_id,
                    process_session_id,
                    instance_token_sha256,
                )
            )
            if process_group_identity_present and (
                isinstance(process_group_id, bool)
                or not isinstance(process_group_id, int)
                or process_group_id != pid
                or isinstance(process_session_id, bool)
                or not isinstance(process_session_id, int)
                or process_session_id != pid
                or not _is_canonical_sha256(instance_token_sha256)
            ):
                raise RuntimeConfigurationError(
                    "service checkpoint process-group identity is incomplete"
                )
            if (
                controller_restart_handoff
                and checkpoint_controller_pid == os.getpid()
                and not allow_same_controller_cleanup
            ):
                raise RuntimeConfigurationError(
                    "controller-restart handoff requires a different controller PID"
                )
            try:
                started_at = datetime.fromisoformat(started_at_text)
            except ValueError as exc:
                raise RuntimeConfigurationError("checkpoint session start is not ISO-8601") from exc
            if started_at.tzinfo is None or started_at.utcoffset() is None:
                raise RuntimeConfigurationError("checkpoint session start must include an offset")
            try:
                observed_start_ticks = _process_start_ticks(pid, proc_root)
            except (FileNotFoundError, ProcessLookupError):
                return False
            if observed_start_ticks != start_ticks:
                raise RuntimeConfigurationError("service checkpoint PID was reused")
            observed_command_sha256 = _process_command_sha256(pid, proc_root)
            if command_sha256 is not None and observed_command_sha256 != command_sha256:
                raise RuntimeConfigurationError("service checkpoint process command changed")
            try:
                command_line = (proc_root / str(pid) / "cmdline").read_bytes().split(b"\0")
            except OSError as exc:
                raise RuntimeConfigurationError("cannot inspect checkpointed service") from exc
            required_parts = {
                b"vllm.entrypoints.openai.api_server",
                str(self.configuration.snapshot_path).encode("utf-8"),
            }
            if not required_parts.issubset(set(command_line)):
                raise RuntimeConfigurationError(
                    "checkpoint process is not the configured vLLM service"
                )
            if process_group_identity_present:
                members = _bound_process_group_members(
                    cast(int, process_group_id),
                    cast(int, process_session_id),
                    cast(str, instance_token_sha256),
                    proc_root=proc_root,
                )
                if pid not in members:
                    raise RuntimeConfigurationError(
                        "checkpoint service leader is absent from its bound process group"
                    )
            adopted = adopted_factory(pid)
            if adopted.poll() is not None:
                return False
            resumed_at = self.wall_clock()
            if resumed_at.tzinfo is None or resumed_at.utcoffset() is None:
                raise RuntimeConfigurationError("resume clock must be timezone-aware")
            wall_service_seconds = (resumed_at - started_at).total_seconds()
            if wall_service_seconds < 0:
                raise RuntimeConfigurationError("resume clock predates the checkpointed session")
            self._process = adopted
            self._last_service_pid = pid
            self._last_process_start_ticks = observed_start_ticks
            self._last_process_command_sha256 = observed_command_sha256
            self._last_process_group_id = (
                cast(int, process_group_id) if process_group_identity_present else None
            )
            self._last_process_session_id = (
                cast(int, process_session_id) if process_group_identity_present else None
            )
            self._service_instance_token = None
            self._service_instance_token_sha256 = (
                cast(str, instance_token_sha256) if process_group_identity_present else None
            )
            self._process_identity_proc_root = proc_root
            self._session_id = session_id
            self._accounting_session_id = accounting_session_id
            self._started_at = started_at
            self._started_monotonic = self.monotonic_clock()
            self._allocated_at_start = float(ledger_before_session)
            # A service that remained live consumed allocation between checkpoint
            # write and controller recovery. Charge the larger persisted/wall gap.
            self._carried_service_seconds = max(float(carried_seconds), wall_service_seconds)
            self._validate_resume_lease(
                pid,
                controller_restart_handoff=controller_restart_handoff,
            )
            self._adopt_service_journal()
            self._write_service_lock_metadata(lease_state="live", service_pid=pid)
            try:
                ready = self._ready()
            except BaseException as readiness_error:
                # Ownership has already transferred to this controller.  A
                # failed probe must not leave the adopted process detached: it
                # is now safe (and mandatory) to stop it and reconcile the
                # already-adopted service journal before reporting the probe
                # error to the caller.
                self.state = ServiceState.FAILED
                try:
                    self.shutdown()
                except BaseException as cleanup_error:
                    raise RuntimeConfigurationError(
                        "checkpoint adoption failed and the adopted service "
                        "could not be terminally reconciled"
                    ) from cleanup_error
                raise readiness_error
            if not ready:
                self.state = ServiceState.FAILED
                try:
                    self.shutdown()
                except BaseException as cleanup_error:
                    raise RuntimeConfigurationError(
                        "unready adopted service could not be terminally reconciled"
                    ) from cleanup_error
                return False
            self.state = ServiceState.READY
            self._observe_service_journal(details={"resumed_controller": True, "service_pid": pid})
            self._start_service_heartbeat()
            return True
        finally:
            if acquired_here and self._process is None:
                # A malformed/stale checkpoint is not proof that a detached
                # service disappeared. Retain the lease unless endpoint absence
                # can be positively established, so a subsequent start cannot
                # create an unowned duplicate.
                endpoint_absent = False
                with suppress(Exception):
                    endpoint_absent = not self._endpoint_live(0.25)
                if endpoint_absent:
                    self._release_service_lock()
                else:
                    self.state = ServiceState.FAILED

    def handoff_resume(
        self,
        path: Path,
        *,
        proc_root: Path = PROC_ROOT,
    ) -> VLLMService:
        """Exercise checkpoint/resume by transferring the live controlled handle."""

        if self._periodic_resource_watchdog is not None:
            raise RuntimeError("periodic resource watchdog must stop before controller handoff")
        with self._emergency_stop_lock:
            emergency_thread = self._emergency_stop_thread
            if emergency_thread is not None:
                raise RuntimeError("emergency-stop coordinator forbids controller handoff")
        self.write_resume_checkpoint(path, proc_root=proc_root)
        self._stop_service_heartbeat()
        process = self._process
        assert process is not None
        resumed = VLLMService(
            configuration=self.configuration,
            client=self.client,
            meter=self.meter,
            log_path=self.log_path,
            service_lock_path=self.service_lock_path,
            startup_resource_sampler=self.startup_resource_sampler,
            startup_sample_interval_seconds=self.startup_sample_interval_seconds,
            service_heartbeat_interval_seconds=self.service_heartbeat_interval_seconds,
            preflight_endpoint_check=self.preflight_endpoint_check,
            readiness_check=self.readiness_check,
            popen_factory=self.popen_factory,
            monotonic_clock=self.monotonic_clock,
            wall_clock=self.wall_clock,
            sleep=self.sleep,
            process_group_signaler=self.process_group_signaler,
            process_liveness_check=self.process_liveness_check,
            process_group_liveness_check=self.process_group_liveness_check,
            available_cpu_sampler=self.available_cpu_sampler,
            affinity_setter=self.affinity_setter,
        )
        # Keep one uninterrupted exclusive lock while controller ownership moves.
        resumed._service_lock_stream = self._service_lock_stream
        resumed._prior_service_lease = self._prior_service_lease
        try:
            if not resumed.resume_from_checkpoint(
                path,
                proc_root=proc_root,
                adopted_factory=lambda pid: process,
            ):
                raise RuntimeError("live vLLM service failed its resume health check")
        except BaseException:
            resumed._service_lock_stream = None
            resumed._process = None
            resumed.state = ServiceState.STOPPED
            self._start_service_heartbeat()
            raise
        # Ownership transfers only after every PID/config/health check succeeds.
        self._service_lock_stream = None
        self._prior_service_lease = None
        self._service_journal_opened = False
        resumed._log_stream = self._log_stream
        self._log_stream = None
        self._process = None
        self._last_service_pid = None
        self._last_process_start_ticks = None
        self._last_process_command_sha256 = None
        self._last_process_group_id = None
        self._last_process_session_id = None
        self._service_instance_token = None
        self._service_instance_token_sha256 = None
        self._process_identity_proc_root = PROC_ROOT
        self._accounting_session_id = None
        self._started_at = None
        self._started_monotonic = None
        self._allocated_at_start = None
        self._carried_service_seconds = 0.0
        self.state = ServiceState.STOPPED
        return resumed

    def run_warmup(
        self,
        operation: Callable[[], object],
        *,
        event_id: str,
        watchdog_seconds: float,
        job_id: str | None = None,
        attempt_id: str | None = None,
        remaining_required_seconds: float = 0,
    ) -> object:
        self.require_ready()
        self.require_service_capacity(
            watchdog_seconds,
            remaining_required_seconds=remaining_required_seconds,
        )
        with self.meter.warmup(
            event_id=event_id,
            maximum_seconds=watchdog_seconds,
            remaining_required_seconds=remaining_required_seconds,
            job_id=job_id,
            attempt_id=attempt_id,
        ):
            try:
                return operation()
            except RuntimeWatchdogTimeout:
                self.emergency_stop()
                raise

    def run_schema_probe(
        self,
        operation: Callable[[], object],
        *,
        event_id: str,
        watchdog_seconds: float,
        job_id: str | None = None,
        attempt_id: str | None = None,
        remaining_required_seconds: float = 0,
    ) -> object:
        self.require_ready()
        self.require_service_capacity(
            watchdog_seconds,
            remaining_required_seconds=remaining_required_seconds,
        )
        with self.meter.schema_probe(
            event_id=event_id,
            maximum_seconds=watchdog_seconds,
            remaining_required_seconds=remaining_required_seconds,
            job_id=job_id,
            attempt_id=attempt_id,
        ):
            try:
                return operation()
            except RuntimeWatchdogTimeout:
                self.emergency_stop()
                raise

    def generate(
        self,
        request: GuidedJSONRequest,
        *,
        event_id: str,
        watchdog_seconds: float,
        repair: bool = False,
        job_id: str | None = None,
        attempt_id: str | None = None,
        remaining_required_seconds: float = 0,
        accounting_details: Mapping[str, object] | None = None,
    ) -> GenerationResult:
        self.require_ready()
        if request.model_name != self.configuration.served_model_name:
            raise RuntimeConfigurationError(
                "request served-model alias differs from the controlled service"
            )
        self.require_service_capacity(
            watchdog_seconds,
            remaining_required_seconds=remaining_required_seconds,
        )
        if repair != (request.decoding.decoding_pass is DecodingPass.REPAIR):
            raise RuntimeConfigurationError("repair event kind differs from decoding pass")
        context = self.meter.repair if repair else self.meter.inference
        details = {"request_id": request.request_id, "request_hash": request.request_hash}
        for name, value in (accounting_details or {}).items():
            if name in details:
                raise RuntimeConfigurationError(
                    "additional accounting details cannot replace request identity"
                )
            details[name] = value
        with context(
            event_id=event_id,
            maximum_seconds=watchdog_seconds,
            remaining_required_seconds=remaining_required_seconds,
            job_id=job_id,
            attempt_id=attempt_id,
            details=details,
        ):
            try:
                return self.client.generate(request, watchdog_seconds=watchdog_seconds)
            except RuntimeWatchdogTimeout:
                self.emergency_stop()
                raise

    def run_fallback_test(
        self,
        request: GuidedJSONRequest,
        *,
        event_id: str,
        watchdog_seconds: float,
        reserve_call_class: str,
        reserve_reservation_id: str,
        job_id: str | None = None,
        attempt_id: str | None = None,
        remaining_required_seconds: float = 0,
    ) -> GenerationResult:
        """Run one first-pass fallback probe charged to its exact reserve tier."""

        expected_watchdog = {
            "reserve_long": 240,
            "reserve_standard": 150,
            "reserve_short": 90,
        }.get(reserve_call_class)
        if expected_watchdog is None or watchdog_seconds != expected_watchdog:
            raise RuntimeConfigurationError(
                "fallback watchdog must match its registered reserve call class"
            )
        _require_plain_identifier("reserve_reservation_id", reserve_reservation_id)
        if request.decoding.decoding_pass is not DecodingPass.FIRST_PASS:
            raise RuntimeConfigurationError("fallback base probe must use first-pass decoding")
        self.require_ready()
        if request.model_name != self.configuration.served_model_name:
            raise RuntimeConfigurationError(
                "request served-model alias differs from the controlled service"
            )
        if self.configuration.model_candidate != "fallback":
            raise RuntimeConfigurationError("fallback probe requires the pinned fallback model")
        self.require_service_capacity(
            watchdog_seconds,
            remaining_required_seconds=remaining_required_seconds,
        )
        with self.meter.fallback_test(
            event_id=event_id,
            maximum_seconds=watchdog_seconds,
            remaining_required_seconds=remaining_required_seconds,
            job_id=job_id,
            attempt_id=attempt_id,
            details={
                "request_id": request.request_id,
                "request_hash": request.request_hash,
                "reserve_call_class": reserve_call_class,
                "reserve_reservation_id": reserve_reservation_id,
            },
        ):
            try:
                return self.client.generate(request, watchdog_seconds=watchdog_seconds)
            except RuntimeWatchdogTimeout:
                self.emergency_stop()
                raise

    def require_ready(self) -> None:
        self._raise_service_heartbeat_failure()
        if (
            self.state is not ServiceState.READY
            or self._process is None
            or self._process.poll() is not None
        ):
            raise RuntimeError("vLLM service is not ready")

    @property
    def actual_allocated_service_seconds(self) -> float:
        """Resume-monotonic allocation including time between classified events."""

        ledger_seconds = self.meter.actual_allocated_gpu_seconds
        if self._started_monotonic is None or self._allocated_at_start is None:
            return ledger_seconds
        live_seconds = self._elapsed_service_seconds()
        return max(
            ledger_seconds,
            self._allocated_at_start + live_seconds,
        )

    def _elapsed_service_seconds(self) -> float:
        if self._physically_stopped_seconds is not None:
            return self._physically_stopped_seconds
        if self._started_monotonic is None:
            return self._carried_service_seconds
        elapsed = self.monotonic_clock() - self._started_monotonic
        if elapsed < 0:
            raise RuntimeError("monotonic service clock moved backwards")
        return self._carried_service_seconds + elapsed

    def require_service_capacity(
        self,
        next_maximum_seconds: float,
        *,
        remaining_required_seconds: float = 0,
    ) -> None:
        """Apply scheduled/hard gates to actual service allocation, not token time."""

        if next_maximum_seconds <= 0 or remaining_required_seconds < 0:
            raise RuntimeConfigurationError("GPU capacity values must be positive/nonnegative")
        actual = self.actual_allocated_service_seconds
        shutdown_reserve_seconds = RESOURCE_AWARE_HARD_STOP_RESERVE_SECONDS
        if (
            actual + next_maximum_seconds + shutdown_reserve_seconds
            >= self.meter.hard_limit_seconds
        ):
            raise GpuBudgetExceeded("next request would reach the hard GPU-service limit")
        if (
            actual + next_maximum_seconds + remaining_required_seconds
            > self.meter.scheduled_limit_seconds
        ):
            raise ForecastAdmissionError(
                "next request plus remaining work exceeds scheduled GPU-service allocation"
            )

    def require_hard_stop_margin(self) -> None:
        """Leave enough allocation to terminate and, if needed, kill the service."""

        shutdown_reserve_seconds = RESOURCE_AWARE_HARD_STOP_RESERVE_SECONDS
        # Allocation supervision of a live, concurrency-one owned service is
        # pure monotonic arithmetic. It must not wait for a SQLite writer lock.
        actual = (
            self._allocated_at_start + self._elapsed_service_seconds()
            if self._allocated_at_start is not None and self._started_monotonic is not None
            else self.meter.actual_allocated_gpu_seconds
        )
        if actual + shutdown_reserve_seconds >= self.meter.hard_limit_seconds:
            raise GpuBudgetExceeded("vLLM service reached its protected hard-stop margin")

    def _endpoint_live(self, timeout_seconds: float) -> bool:
        probe = getattr(self.client, "endpoint_live", None)
        if not callable(probe):
            raise RuntimeError("vLLM client cannot verify endpoint shutdown")
        return bool(probe(timeout_seconds))

    def _drain_startup_resource_watchdog(
        self,
        *,
        completion_timeout_seconds: float | None = None,
    ) -> None:
        """Finish owned sampler ledger work before any process-tree signal."""

        watchdog = self._startup_resource_watchdog
        if watchdog is None:
            return
        if not watchdog.stop(
            raise_failure=False,
            completion_timeout_seconds=completion_timeout_seconds,
        ):
            self.state = ServiceState.FAILED
            raise RuntimeError(
                "cannot terminate vLLM while its owned resource sample remains in flight"
            )
        self._startup_resource_watchdog = None

    def start_periodic_resource_watchdog(
        self,
        watchdog: ResourceWatchdog,
    ) -> ResourceWatchdog:
        """Start and own the one periodic sampler for this live service.

        Registration and thread start occur under the process-control lock, so
        shutdown cannot pass its sampler barrier while a runner concurrently
        starts an unowned watchdog.
        """

        with self._process_control_lock:
            if (
                self.state is not ServiceState.READY
                or self._process is None
                or self._process.poll() is not None
            ):
                raise RuntimeError("periodic resource sampling requires a ready service")
            if watchdog.root_pid != self._process.pid:
                raise RuntimeConfigurationError(
                    "periodic resource watchdog PID differs from the controlled service"
                )
            if self._periodic_resource_watchdog is not None:
                raise RuntimeError("vLLM service already owns a periodic resource watchdog")
            self._periodic_resource_watchdog = watchdog
            try:
                watchdog.start()
            except BaseException:
                self._periodic_resource_watchdog = None
                raise
            return watchdog

    def stop_periodic_resource_watchdog(
        self,
        watchdog: ResourceWatchdog,
        *,
        raise_failure: bool = True,
        completion_timeout_seconds: float | None = None,
    ) -> bool:
        """Boundedly drain and release an exactly owned periodic sampler."""

        with self._process_control_lock:
            owned = self._periodic_resource_watchdog
            if owned is None:
                if watchdog.running or watchdog.sample_in_flight:
                    raise RuntimeError(
                        "cannot release a running resource watchdog not owned by this service"
                    )
                return True
            if owned is not watchdog:
                raise RuntimeError("resource watchdog is not owned by this service")
            drained = watchdog.stop(
                raise_failure=raise_failure,
                completion_timeout_seconds=completion_timeout_seconds,
            )
            if not drained:
                return False
            if watchdog.running or watchdog.sample_in_flight:
                raise RuntimeError(
                    "resource watchdog reported completion while retaining sampler ownership"
                )
            self._periodic_resource_watchdog = None
            return True

    def _drain_periodic_resource_watchdog(self) -> None:
        """Enforce the periodic sampler barrier before process-tree signaling."""

        watchdog = self._periodic_resource_watchdog
        if watchdog is None:
            return
        if not self.stop_periodic_resource_watchdog(
            watchdog, raise_failure=False, completion_timeout_seconds=1
        ):
            self.state = ServiceState.FAILED
            raise RuntimeError(
                "cannot terminate vLLM while its periodic resource sample remains in flight"
            )

    def request_emergency_stop(self) -> None:
        """Request sampler-aware emergency cleanup from a distinct coordinator."""

        with self._emergency_stop_lock:
            thread = self._emergency_stop_thread
            if thread is not None and thread.is_alive():
                return

            def coordinate() -> None:
                try:
                    self.emergency_stop()
                except BaseException as exc:
                    # Physical cleanup precedes observer draining. Retain the
                    # exact lease if verification or accounting still fails.
                    self._emergency_stop_failure = exc

            self._emergency_stop_failure = None
            thread = threading.Thread(
                target=coordinate,
                name=f"vllm-emergency-stop-{self._accounting_session_id or 'unmetered'}",
                daemon=True,
            )
            # Publish only after ``start`` succeeds.  Every production reader
            # takes this same lock, so none can observe a Thread in Python's
            # pre-start state and mistakenly join/approve handoff against it.
            thread.start()
            self._emergency_stop_thread = thread

    def _shutdown_status(
        self,
        process: ProcessHandle | None,
        process_group: int | None,
        *,
        deadline: float,
    ) -> tuple[bool, bool, bool]:
        leader_live = process is not None and process.poll() is None
        group_live = process_group is not None and self.process_group_liveness_check(process_group)
        remaining = max(0.0, deadline - self.monotonic_clock())
        endpoint_live = self._endpoint_live(max(0.001, min(0.25, remaining)))
        return leader_live, group_live, endpoint_live

    def _stop_process(
        self,
        shutdown_seconds: float = DEFAULT_SHUTDOWN_SECONDS,
        *,
        release_lock: bool = True,
        continuing_service: bool = False,
        journal_process_stopped: bool = False,
    ) -> None:
        with self._process_control_lock:
            self._stop_process_locked(
                shutdown_seconds,
                release_lock=release_lock,
                continuing_service=continuing_service,
                journal_process_stopped=journal_process_stopped,
            )

    def _stop_process_locked(
        self,
        shutdown_seconds: float,
        *,
        release_lock: bool,
        continuing_service: bool,
        journal_process_stopped: bool,
    ) -> None:
        if not math.isfinite(shutdown_seconds) or shutdown_seconds <= 0:
            raise RuntimeConfigurationError("shutdown_seconds must be positive and finite")
        if continuing_service and journal_process_stopped:
            raise RuntimeConfigurationError(
                "an intermediate restart cannot terminalize the service journal"
            )
        verification_deadline = self.monotonic_clock() + shutdown_seconds
        heartbeat_stop_error: BaseException | None = None
        # Stop scheduling heartbeat writes, but do not join a writer before
        # the exact owned process has received its termination signal.
        self._service_heartbeat_stop.set()
        process = self._process
        if process is None and self._service_lock_stream is None:
            if isinstance(self.startup_resource_sampler, ResourceSampler):
                self.startup_resource_sampler.close_probes()
            self._close_log_stream()
            self.state = (
                ServiceState.STOPPED if heartbeat_stop_error is None else ServiceState.FAILED
            )
            if heartbeat_stop_error is not None:
                raise RuntimeError("vLLM service monitoring cleanup incomplete") from (
                    heartbeat_stop_error
                )
            return
        process_group = self._last_process_group_id or (
            self._last_service_pid if process is None else process.pid
        )
        physical_absence_verified = False
        verified_at: datetime | None = None
        try:
            if process is not None and process.poll() is None:
                with suppress(ProcessLookupError):
                    self._signal_bound_service_process_group(
                        cast(int, process_group),
                        signal.SIGTERM,
                    )
                with suppress(subprocess.TimeoutExpired):
                    process.wait(timeout=min(5.0, shutdown_seconds / 3))

            leader_live, group_live, endpoint_live = self._shutdown_status(
                process,
                process_group,
                deadline=verification_deadline,
            )
            if (leader_live or group_live) and process_group is not None:
                with suppress(ProcessLookupError):
                    self._signal_bound_service_process_group(
                        process_group,
                        signal.SIGKILL,
                    )
                if process is not None and process.poll() is None:
                    with suppress(subprocess.TimeoutExpired):
                        process.wait(
                            timeout=max(0.0, verification_deadline - self.monotonic_clock())
                        )
                leader_live, group_live, endpoint_live = self._shutdown_status(
                    process,
                    process_group,
                    deadline=verification_deadline,
                )

            while leader_live or group_live or endpoint_live:
                remaining = verification_deadline - self.monotonic_clock()
                if remaining <= 0:
                    self.state = ServiceState.FAILED
                    live_parts = [
                        name
                        for name, live in (
                            ("leader", leader_live),
                            ("process_group", group_live),
                            ("endpoint", endpoint_live),
                        )
                        if live
                    ]
                    raise RuntimeError(
                        "failed to verify vLLM shutdown; still live: " + ", ".join(live_parts)
                    )
                self.sleep(min(0.05, remaining))
                leader_live, group_live, endpoint_live = self._shutdown_status(
                    process,
                    process_group,
                    deadline=verification_deadline,
                )

            physical_absence_verified = True
            verified_at = self.wall_clock()
            if verified_at.tzinfo is None or verified_at.utcoffset() is None:
                raise RuntimeConfigurationError("service stop clock must be timezone-aware")
            if (
                not continuing_service
                and heartbeat_stop_error is None
                and self._physically_stopped_at is None
            ):
                service_seconds = self._elapsed_service_seconds()
                if service_seconds < 0:
                    raise RuntimeError("monotonic service clock moved backwards")
                self._physically_stopped_at = verified_at
                self._physically_stopped_seconds = service_seconds

            # Collection workers have private diagnostics and never SQLite
            # ownership. Kill/reap them even if observation or fsync is stuck.
            samplers = []
            for watchdog in (self._startup_resource_watchdog, self._periodic_resource_watchdog):
                if watchdog is not None:
                    watchdog._stop.set()
                if (
                    watchdog is not None
                    and isinstance(watchdog.sampler, ResourceSampler)
                    and watchdog.sampler not in samplers
                ):
                    samplers.append(watchdog.sampler)
            if (
                isinstance(self.startup_resource_sampler, ResourceSampler)
                and self.startup_resource_sampler not in samplers
            ):
                samplers.append(self.startup_resource_sampler)
            for sampler in samplers:
                if continuing_service:
                    # A registered same-controller restart keeps the storage
                    # event stream continuous while the model process reloads.
                    sampler.cancel_pending()
                else:
                    sampler.close_probes()
            try:
                self._drain_startup_resource_watchdog(completion_timeout_seconds=1)
                self._drain_periodic_resource_watchdog()
                self._stop_service_heartbeat()
                # The caller is now the only monitor-ledger writer. If a
                # commit fails, the lease remains pending; physical stop stands.
                for sampler in samplers:
                    sampler.flush()
            except BaseException as exc:
                heartbeat_stop_error = exc

            if self._service_lock_stream is not None:
                prior_state = (
                    None
                    if self._prior_service_lease is None
                    else self._prior_service_lease.get("lease_state")
                )
                current_session_needs_accounting = (
                    self._started_at is not None
                    or self._started_monotonic is not None
                    or self._session_id is not None
                )
                unresolved_prior_session = prior_state not in {
                    None,
                    "stopped_verified",
                }
                if current_session_needs_accounting:
                    self._write_service_lock_metadata(
                        lease_state="accounting_pending",
                        service_pid=process_group,
                        ended_at=(
                            verified_at if continuing_service else self._physically_stopped_at
                        ),
                    )
                elif not unresolved_prior_session:
                    self._write_service_lock_metadata(
                        lease_state="stopped_verified",
                        service_pid=process_group,
                        ended_at=verified_at,
                    )

            if not continuing_service and journal_process_stopped and heartbeat_stop_error is None:
                self._journal_verified_process_stop()
            self._process = None
            self.state = (
                ServiceState.STOPPED if heartbeat_stop_error is None else ServiceState.FAILED
            )
            if heartbeat_stop_error is not None:
                raise RuntimeError("vLLM service heartbeat could not be stopped") from (
                    heartbeat_stop_error
                )
            if release_lock:
                self._release_service_lock()
        except BaseException:
            self.state = ServiceState.FAILED
            if self._service_lock_stream is not None:
                with suppress(Exception):
                    self._write_service_lock_metadata(
                        lease_state=(
                            "accounting_pending"
                            if physical_absence_verified and self._session_id is not None
                            else "shutdown_unverified"
                        ),
                        service_pid=process_group,
                        ended_at=(
                            self._physically_stopped_at
                            if self._physically_stopped_at is not None
                            else verified_at
                        ),
                    )
            raise
        finally:
            self._close_log_stream()

    def emergency_stop(self) -> None:
        """Stop the controlled process tree while retaining uptime for final accounting."""

        self._stop_process(
            shutdown_seconds=min(5.0, DEFAULT_SHUTDOWN_SECONDS),
            release_lock=False,
            journal_process_stopped=False,
        )

    def restore_terminal_service_lease_from_identity(
        self,
        *,
        expected_session_id: str,
        expected_event_id: str,
        service_pid: int,
        process_start_ticks: int,
        observed_process_command_sha256: str,
        process_group_id: int,
        process_session_id: int,
        service_instance_token_sha256: str,
    ) -> GpuServiceSession:
        """Repair only the known null-identity/terminal-accounting failure state.

        This is not general stale-lease recovery.  It accepts exactly a
        ``shutdown_unverified`` lease whose durable service and process fields
        were all weakened to null, and only after the named journal and service
        row are already terminal.  The caller supplies the process identity
        captured before that overwrite, including the observed command hash
        even when process-title rewriting made it differ from the frozen launch
        argv.  No accounting row is created and no live process is ever adopted
        or signalled here.
        """

        _require_plain_identifier("expected_session_id", expected_session_id)
        _require_plain_identifier("expected_event_id", expected_event_id)
        if (
            isinstance(service_pid, bool)
            or not isinstance(service_pid, int)
            or service_pid <= 0
            or isinstance(process_start_ticks, bool)
            or not isinstance(process_start_ticks, int)
            or process_start_ticks <= 0
            or isinstance(process_group_id, bool)
            or not isinstance(process_group_id, int)
            or process_group_id != service_pid
            or isinstance(process_session_id, bool)
            or not isinstance(process_session_id, int)
            or process_session_id != service_pid
            or not _is_canonical_sha256(observed_process_command_sha256)
            or not _is_canonical_sha256(service_instance_token_sha256)
        ):
            raise RuntimeConfigurationError(
                "terminal lease restoration process identity is invalid"
            )
        if (
            self.state is not ServiceState.STOPPED
            or self._process is not None
            or self._service_lock_stream is not None
            or self._started_at is not None
            or self._accounting_session_id is not None
        ):
            raise RuntimeError("terminal lease restoration requires a fresh stopped controller")

        self._last_recovered_process_identity = None
        self._acquire_service_lock()
        restored = False
        try:
            lease = self._prior_service_lease
            if lease is None or lease.get("lease_state") != "shutdown_unverified":
                raise RuntimeConfigurationError(
                    "terminal lease restoration requires shutdown_unverified state"
                )
            null_damage_fields = _UNRESOLVED_LEASE_IDENTITY_FIELDS
            if any(
                field_name not in lease or lease[field_name] is not None
                for field_name in null_damage_fields
            ):
                raise RuntimeConfigurationError(
                    "terminal lease restoration requires the exact null-identity failure pattern"
                )
            if (
                lease.get("schema_version") != SCHEMA_VERSION
                or lease.get("configuration_hash") != self.configuration.configuration_hash
            ):
                raise RuntimeConfigurationError(
                    "terminal lease restoration configuration identity differs"
                )
            _parse_aware_datetime(
                "damaged terminal lease update",
                lease.get("updated_at"),
            )

            ledger = self.meter.ledger
            if ledger.unresolved_gpu_allocations():
                raise RuntimeConfigurationError(
                    "terminal lease restoration forbids unresolved GPU allocations"
                )
            if ledger.unresolved_gpu_service_journals():
                raise RuntimeConfigurationError(
                    "terminal lease restoration forbids unresolved service accounting"
                )
            latest = ledger.latest_gpu_service_journal(expected_event_id)
            record = ledger.get_gpu_service_session(expected_event_id)
            journal_records = tuple(
                row
                for row in ledger.gpu_service_journal_records()
                if row.service_session_id == expected_event_id
            )
            if (
                latest is None
                or record is None
                or not journal_records
                or latest.state
                not in {GpuServiceJournalState.CLOSED, GpuServiceJournalState.RECOVERED}
            ):
                raise RuntimeConfigurationError(
                    "terminal lease restoration requires a terminal journal and service row"
                )
            first = journal_records[0]
            stable_journal_identity = all(
                row.session_id == expected_session_id
                and row.configuration_hash == self.configuration.configuration_hash
                and row.service_started_at == first.service_started_at
                and row.ledger_allocated_microseconds_before_session
                == first.ledger_allocated_microseconds_before_session
                and row.hard_limit_microseconds == first.hard_limit_microseconds
                for row in journal_records
            )
            if (
                first.sequence != 0
                or first.state is not GpuServiceJournalState.OPENED
                or first.elapsed_microseconds != 0
                or first.observed_at != first.service_started_at
                or latest != journal_records[-1]
                or not stable_journal_identity
                or record.service_session_id != expected_event_id
                or record.session_id != expected_session_id
                or record.started_at != first.service_started_at
                or record.service_microseconds != latest.elapsed_microseconds
            ):
                raise RuntimeConfigurationError(
                    "terminal lease restoration journal identity is inconsistent"
                )
            started_at = _parse_aware_datetime(
                "terminal restoration service start",
                record.started_at,
            )
            ended_at = _parse_aware_datetime(
                "terminal restoration service end",
                record.ended_at,
            )
            latest_observed_at = _parse_aware_datetime(
                "terminal restoration journal end",
                latest.observed_at,
            )
            if (
                ended_at < started_at
                or latest_observed_at < ended_at
                or (
                    latest.state is GpuServiceJournalState.CLOSED and latest_observed_at != ended_at
                )
            ):
                raise RuntimeConfigurationError(
                    "terminal lease restoration timestamps are inconsistent"
                )
            baseline_microseconds = first.ledger_allocated_microseconds_before_session
            total_microseconds = ledger.gpu_summary().total_allocated_microseconds
            expected_hard_microseconds = round(self.meter.hard_limit_seconds * 1_000_000)
            if (
                baseline_microseconds < 0
                or baseline_microseconds + record.service_microseconds > total_microseconds
                or first.hard_limit_microseconds != expected_hard_microseconds
            ):
                raise RuntimeConfigurationError(
                    "terminal lease restoration accounting baseline is inconsistent"
                )
            try:
                pid_live = self.process_liveness_check(service_pid)
                process_group_live = self.process_group_liveness_check(process_group_id)
                endpoint_live = self._endpoint_live(0.25)
            except BaseException as exc:
                raise RuntimeConfigurationError(
                    "cannot prove terminal restoration process and endpoint absence"
                ) from exc
            if pid_live or process_group_live or endpoint_live:
                live_parts = [
                    name
                    for name, live in (
                        ("pid", pid_live),
                        ("process_group", process_group_live),
                        ("endpoint", endpoint_live),
                    )
                    if live
                ]
                raise RuntimeConfigurationError(
                    "terminal lease restoration found live service state: " + ", ".join(live_parts)
                )

            self._session_id = expected_session_id
            self._accounting_session_id = expected_event_id
            self._started_at = started_at
            self._allocated_at_start = baseline_microseconds / 1_000_000
            self._carried_service_seconds = record.service_seconds
            self._physically_stopped_at = ended_at
            self._physically_stopped_seconds = record.service_seconds
            self._last_service_pid = service_pid
            self._last_process_start_ticks = process_start_ticks
            # The captured observed hash may reflect a legitimate setproctitle
            # rewrite.  Preserve the frozen launch-command hash in the standard
            # lease field so terminal replay remains configuration-bound; the
            # caller's immutable incident record retains the observed hash.
            expected_process_command_sha256 = canonical_sha256(list(self.configuration.command()))
            self._last_process_command_sha256 = expected_process_command_sha256
            self._last_process_group_id = process_group_id
            self._last_process_session_id = process_session_id
            self._service_instance_token = None
            self._service_instance_token_sha256 = service_instance_token_sha256
            self._write_service_lock_metadata(
                lease_state="stopped_verified",
                service_pid=service_pid,
                ended_at=ended_at,
            )
            self._last_recovered_process_identity = RecoveredServiceProcessIdentity(
                configuration_hash=self.configuration.configuration_hash,
                session_id=expected_session_id,
                accounting_session_id=expected_event_id,
                pid=service_pid,
                process_start_ticks=process_start_ticks,
                process_command_sha256=expected_process_command_sha256,
                service_started_at=started_at,
            )
            restored = True
            return record
        finally:
            self.state = ServiceState.STOPPED if restored else ServiceState.FAILED
            self._session_id = None
            self._accounting_session_id = None
            self._started_at = None
            self._started_monotonic = None
            self._allocated_at_start = None
            self._carried_service_seconds = 0.0
            self._physically_stopped_at = None
            self._physically_stopped_seconds = None
            self._last_service_pid = None
            self._last_process_start_ticks = None
            self._last_process_command_sha256 = None
            self._last_process_group_id = None
            self._last_process_session_id = None
            self._service_instance_token = None
            self._service_instance_token_sha256 = None
            self._release_service_lock()

    def shutdown(
        self,
        *,
        shutdown_seconds: float = DEFAULT_SHUTDOWN_SECONDS,
    ) -> ServiceUptime | None:
        if self._started_at is None or self._started_monotonic is None or self._session_id is None:
            self._stop_process(shutdown_seconds)
            return None
        terminalized = False
        try:
            self._stop_process(
                shutdown_seconds,
                release_lock=False,
                journal_process_stopped=True,
            )
            ended_at = self._physically_stopped_at
            service_seconds = self._physically_stopped_seconds
            if ended_at is None or service_seconds is None:
                raise RuntimeError("vLLM service has no verified physical stop")
            if self._allocated_at_start is None:
                raise RuntimeError("live vLLM service has no durable allocation baseline")
            allocated_start = self._allocated_at_start
            # A guardian can SIGKILL a wedged controller while one of that
            # controller's allocation contexts is open.  The guardian's meter
            # predates that context, so startup recovery cannot close it.  Use
            # the already-verified physical stop time for both the allocation
            # recovery and enclosing service reconciliation.  Classified work
            # is durable before the service complement is calculated, avoiding
            # both an undercounted terminal receipt and later double counting.
            recovered_allocations = self.meter.recover_unclosed_allocations(recovered_at=ended_at)
            allocated_seconds = self.meter.actual_allocated_gpu_seconds - allocated_start
            if allocated_seconds < -1e-9:
                raise RuntimeError("GPU ledger predates the service allocation baseline")
            allocated_seconds = max(0.0, allocated_seconds)
            uptime = ServiceUptime(
                session_id=self._session_id,
                started_at=self._started_at,
                ended_at=ended_at,
                service_seconds=service_seconds,
                allocated_event_seconds=allocated_seconds,
            )
            accounting_session_id = self._accounting_session_id
            if accounting_session_id is None:
                raise RuntimeError("live vLLM service has no accounting session identity")
            if not self._service_journal_opened:
                raise RuntimeError(
                    "vLLM service cannot reconcile without its durable service journal"
                )
            try:
                self.meter.reconcile_service_session(
                    service_session_id=accounting_session_id,
                    session_id=self._session_id,
                    service_seconds=uptime.allocated_service_seconds,
                    classified_event_seconds=allocated_seconds,
                    started_at=self._started_at,
                    ended_at=ended_at,
                    details={
                        "accounting_method": "service_total_minus_classified_events",
                        "configuration_hash": self.configuration.configuration_hash,
                        "controller_lost_allocation_event_ids": [
                            event.event_id for event in recovered_allocations
                        ],
                        "shared_terminal_recovery_at": ended_at.isoformat(),
                    },
                )
            except BaseException:
                latest = self.meter.ledger.latest_gpu_service_journal(accounting_session_id)
                if (
                    latest is not None
                    and latest.state
                    in {GpuServiceJournalState.CLOSED, GpuServiceJournalState.RECOVERED}
                    and self.meter.ledger.get_gpu_service_session(accounting_session_id) is not None
                ):
                    self._write_service_lock_metadata(
                        lease_state="stopped_verified",
                        service_pid=self._last_service_pid,
                        ended_at=ended_at,
                    )
                    terminalized = True
                raise
            self._write_service_lock_metadata(
                lease_state="stopped_verified",
                service_pid=self._last_service_pid,
                ended_at=ended_at,
            )
            terminalized = True
            heartbeat_failure = self._service_heartbeat_failure
            if heartbeat_failure is not None:
                raise RuntimeError("vLLM service heartbeat failed") from heartbeat_failure
            return uptime
        finally:
            if terminalized:
                self._session_id = None
                self._started_at = None
                self._started_monotonic = None
                self._allocated_at_start = None
                self._accounting_session_id = None
                self._last_service_pid = None
                self._last_process_start_ticks = None
                self._last_process_command_sha256 = None
                self._last_process_group_id = None
                self._last_process_session_id = None
                self._service_instance_token = None
                self._service_instance_token_sha256 = None
                self._process_identity_proc_root = PROC_ROOT
                self._carried_service_seconds = 0.0
                self._service_journal_opened = False
                self._service_heartbeat_failure = None
                self._physically_stopped_at = None
                self._physically_stopped_seconds = None
                self._process_stop_journaled = False
                self._prior_process_stop_contradicted = False
                self.state = ServiceState.STOPPED
                self._release_service_lock()
            else:
                self.state = ServiceState.FAILED

    def recover_stale_service_lease(
        self,
        *,
        expected_current_lease_manifest_sha256: str | None = None,
    ) -> GpuServiceSession | None:
        """Recover one stale allocation only after proving all service absence.

        An incident-specific caller may pin the exact current lease payload by
        supplying its manifest hash.  The comparison occurs only after this
        controller holds the exclusive service lock, closing the gap between an
        external evidence check and the terminal lease transition.
        """

        if (
            self.state is not ServiceState.STOPPED
            or self._process is not None
            or self._service_lock_stream is not None
            or self._started_at is not None
            or self._accounting_session_id is not None
        ):
            raise RuntimeError("stale service recovery requires a fresh stopped controller")
        self._last_recovered_process_identity = None
        if expected_current_lease_manifest_sha256 is not None and not _is_canonical_sha256(
            expected_current_lease_manifest_sha256
        ):
            raise RuntimeConfigurationError(
                "expected current service lease manifest SHA-256 is invalid"
            )
        self._acquire_service_lock()
        terminalized = False
        try:
            lease = self._prior_service_lease
            if lease is None:
                if expected_current_lease_manifest_sha256 is not None:
                    raise RuntimeConfigurationError("expected current service lease is absent")
                return None
            if (
                expected_current_lease_manifest_sha256 is not None
                and canonical_sha256(lease) != expected_current_lease_manifest_sha256
            ):
                raise RuntimeConfigurationError(
                    "current service lease manifest SHA-256 differs from expectation"
                )
            lease_state = lease.get("lease_state")
            if lease_state not in {
                "launch_supervisor_pending",
                "launch_gate_pending",
                "starting",
                "live",
                "controller_restart_handoff",
                "accounting_pending",
                "shutdown_unverified",
                "stopped_verified",
            }:
                raise RuntimeConfigurationError("stale vLLM service lease state is invalid")
            service_session_id = lease.get("accounting_session_id")
            session_id = lease.get("session_id")
            service_pid = lease.get("service_pid")
            process_start_ticks = lease.get("process_start_ticks")
            process_command_sha256 = lease.get("process_command_sha256")
            process_group_id = lease.get("process_group_id")
            process_session_id = lease.get("process_session_id")
            instance_token_sha256 = lease.get("service_instance_token_sha256")
            started_at_text = lease.get("service_started_at")
            baseline = lease.get("ledger_allocated_seconds_before_session")
            launch_protocol = lease.get("launch_protocol")
            launch_token_sha256 = lease.get("launch_gate_token_sha256")
            launch_supervisor_sha256 = lease.get("launch_supervisor_command_sha256")
            is_launch_protocol_lease = launch_protocol == DURABLE_EXEC_GATE_PROTOCOL
            launch_pending = lease_state in {
                "launch_supervisor_pending",
                "launch_gate_pending",
            }
            launch_identity_present = any(
                value is not None
                for value in (
                    launch_protocol,
                    launch_token_sha256,
                    launch_supervisor_sha256,
                )
            )
            supervisor_identity_optional = lease_state == "launch_supervisor_pending" or (
                lease_state == "stopped_verified" and service_pid is None
            )
            if (launch_pending or launch_identity_present) and (
                not is_launch_protocol_lease
                or not _is_canonical_sha256(launch_token_sha256)
                or process_command_sha256 != canonical_sha256(list(self.configuration.command()))
                or (
                    not supervisor_identity_optional
                    and not _is_canonical_sha256(launch_supervisor_sha256)
                )
            ):
                raise RuntimeConfigurationError(
                    "stale pending launch lacks its durable exec-gate identity"
                )
            pidless_preexec = (
                service_pid is None
                and is_launch_protocol_lease
                and lease_state in {"launch_supervisor_pending", "stopped_verified"}
            )
            if pidless_preexec:
                if (
                    not isinstance(service_session_id, str)
                    or not service_session_id
                    or not isinstance(session_id, str)
                    or not session_id
                    or not isinstance(started_at_text, str)
                    or isinstance(baseline, bool)
                    or not isinstance(baseline, int | float)
                    or not math.isfinite(float(baseline))
                    or baseline < 0
                    or process_start_ticks is not None
                    or launch_supervisor_sha256 is not None
                    or not isinstance(launch_token_sha256, str)
                    or process_command_sha256
                    != canonical_sha256(list(self.configuration.command()))
                    or lease.get("configuration_hash") != self.configuration.configuration_hash
                ):
                    raise RuntimeConfigurationError("PID-less pre-exec lease identity is invalid")
                if self._endpoint_live(0.25):
                    raise RuntimeConfigurationError(
                        "PID-less pre-exec lease unexpectedly exposes a live endpoint"
                    )
                recovered_at = self.wall_clock()
                if recovered_at.tzinfo is None or recovered_at.utcoffset() is None:
                    raise RuntimeConfigurationError("service recovery clock must be timezone-aware")
                recovered_allocations = self.meter.recover_unclosed_allocations(
                    recovered_at=recovered_at
                )
                latest = self.meter.ledger.latest_gpu_service_journal(service_session_id)
                self._session_id = session_id
                self._accounting_session_id = service_session_id
                self._started_at = _parse_aware_datetime(
                    "PID-less pre-exec lease start",
                    started_at_text,
                )
                self._allocated_at_start = float(baseline)
                self._last_process_command_sha256 = process_command_sha256
                self._launch_gate_token_sha256 = launch_token_sha256
                if latest is None:
                    if lease_state != "stopped_verified":
                        self._write_service_lock_metadata(
                            lease_state="stopped_verified",
                            service_pid=None,
                            ended_at=recovered_at,
                        )
                    terminalized = True
                    return None
                if (
                    latest.session_id != session_id
                    or latest.configuration_hash != self.configuration.configuration_hash
                    or _parse_aware_datetime(
                        "PID-less pre-exec journal start",
                        latest.service_started_at,
                    )
                    != self._started_at
                    or abs(
                        latest.ledger_allocated_microseconds_before_session / 1_000_000
                        - float(baseline)
                    )
                    > 1e-6
                ):
                    raise RuntimeConfigurationError(
                        "PID-less pre-exec lease and journal identities differ"
                    )
                record = self.meter.recover_service_journal(
                    service_session_id=service_session_id,
                    recovered_at=recovered_at,
                    details={
                        "durable_exec_gate_prevented_target_exec": True,
                        "endpoint_absence_verified": True,
                        "pid_was_never_released_to_target": True,
                        "controller_lost_allocation_event_ids": [
                            event.event_id for event in recovered_allocations
                        ],
                        "shared_terminal_recovery_at": recovered_at.isoformat(),
                    },
                )
                self._carried_service_seconds = record.service_seconds
                self._physically_stopped_seconds = record.service_seconds
                self._physically_stopped_at = _parse_aware_datetime(
                    "PID-less pre-exec recovery end",
                    record.ended_at,
                )
                self._write_service_lock_metadata(
                    lease_state="stopped_verified",
                    service_pid=None,
                    ended_at=self._physically_stopped_at,
                )
                terminalized = True
                return record
            if (
                not isinstance(service_session_id, str)
                or not service_session_id
                or not isinstance(session_id, str)
                or not session_id
                or isinstance(service_pid, bool)
                or not isinstance(service_pid, int)
                or service_pid <= 0
                or not isinstance(started_at_text, str)
                or isinstance(baseline, bool)
                or not isinstance(baseline, int | float)
                or not math.isfinite(float(baseline))
                or baseline < 0
            ):
                raise RuntimeConfigurationError(
                    "stale vLLM lease lacks an exact recoverable process identity"
                )
            process_group_identity_present = any(
                value is not None
                for value in (
                    process_group_id,
                    process_session_id,
                    instance_token_sha256,
                )
            )
            if process_group_identity_present and (
                isinstance(process_group_id, bool)
                or not isinstance(process_group_id, int)
                or process_group_id != service_pid
                or isinstance(process_session_id, bool)
                or not isinstance(process_session_id, int)
                or process_session_id != service_pid
                or not _is_canonical_sha256(instance_token_sha256)
            ):
                raise RuntimeConfigurationError(
                    "stale vLLM lease process-group identity is incomplete"
                )
            bound_process_group = (
                cast(int, process_group_id) if process_group_identity_present else service_pid
            )
            latest = self.meter.ledger.latest_gpu_service_journal(service_session_id)
            if latest is None:
                raise RuntimeConfigurationError("stale vLLM lease has no durable service journal")
            lease_started_at = _parse_aware_datetime(
                "stale service lease start",
                started_at_text,
            )
            recovered_process_identity: RecoveredServiceProcessIdentity | None = None
            if process_start_ticks is not None or process_command_sha256 is not None:
                if (
                    isinstance(process_start_ticks, bool)
                    or not isinstance(process_start_ticks, int)
                    or process_start_ticks <= 0
                    or not isinstance(process_command_sha256, str)
                    or len(process_command_sha256) != 64
                    or any(
                        character not in "0123456789abcdef" for character in process_command_sha256
                    )
                    or process_command_sha256
                    != canonical_sha256(list(self.configuration.command()))
                ):
                    raise RuntimeConfigurationError("stale vLLM lease process identity is invalid")
                if lease_state != "launch_gate_pending":
                    recovered_process_identity = RecoveredServiceProcessIdentity(
                        configuration_hash=self.configuration.configuration_hash,
                        session_id=session_id,
                        accounting_session_id=service_session_id,
                        pid=service_pid,
                        process_start_ticks=process_start_ticks,
                        process_command_sha256=process_command_sha256,
                        service_started_at=lease_started_at,
                    )
            journal_started_at = _parse_aware_datetime(
                "stale service journal start",
                latest.service_started_at,
            )
            if (
                lease.get("configuration_hash") != self.configuration.configuration_hash
                or latest.configuration_hash != self.configuration.configuration_hash
                or latest.session_id != session_id
                or lease_started_at != journal_started_at
                or abs(
                    latest.ledger_allocated_microseconds_before_session / 1_000_000
                    - float(baseline)
                )
                > 1e-6
            ):
                raise RuntimeConfigurationError(
                    "stale vLLM lease and service journal identities differ"
                )
            if lease_state == "stopped_verified":
                if self.meter.ledger.unresolved_gpu_allocations():
                    raise RuntimeConfigurationError(
                        "terminal vLLM lease still has unresolved allocation accounting"
                    )
                record = self.meter.ledger.get_gpu_service_session(service_session_id)
                ended_at_text = lease.get("service_ended_at")
                if (
                    latest.state
                    not in {GpuServiceJournalState.CLOSED, GpuServiceJournalState.RECOVERED}
                    or record is None
                    or not isinstance(ended_at_text, str)
                    or record.service_session_id != service_session_id
                    or record.session_id != session_id
                    or _parse_aware_datetime("terminal service start", record.started_at)
                    != lease_started_at
                    or _parse_aware_datetime("terminal service end", record.ended_at)
                    != _parse_aware_datetime("terminal lease end", ended_at_text)
                ):
                    raise RuntimeConfigurationError(
                        "terminal vLLM lease has no matching service accounting record"
                    )
                self._last_recovered_process_identity = recovered_process_identity
                terminalized = True
                return record
            try:
                pid_live = self.process_liveness_check(service_pid)
                process_group_live = self.process_group_liveness_check(bound_process_group)
                endpoint_live = self._endpoint_live(0.25)
            except BaseException as exc:
                raise RuntimeConfigurationError(
                    "cannot prove stale vLLM process and endpoint absence"
                ) from exc
            if pid_live or process_group_live or endpoint_live:
                live_parts = [
                    name
                    for name, live in (
                        ("pid", pid_live),
                        ("process_group", process_group_live),
                        ("endpoint", endpoint_live),
                    )
                    if live
                ]
                raise RuntimeConfigurationError(
                    "stale vLLM service is still live: " + ", ".join(live_parts)
                )
            recovered_at = self.wall_clock()
            if recovered_at.tzinfo is None or recovered_at.utcoffset() is None:
                raise RuntimeConfigurationError("service recovery clock must be timezone-aware")
            recovered_allocations = self.meter.recover_unclosed_allocations(
                recovered_at=recovered_at
            )

            def finalize_lease(record: GpuServiceSession) -> None:
                nonlocal terminalized
                self._session_id = record.session_id
                self._accounting_session_id = record.service_session_id
                self._started_at = _parse_aware_datetime(
                    "recovered service start",
                    record.started_at,
                )
                self._allocated_at_start = float(baseline)
                self._carried_service_seconds = record.service_seconds
                self._physically_stopped_at = _parse_aware_datetime(
                    "recovered service end",
                    record.ended_at,
                )
                self._physically_stopped_seconds = record.service_seconds
                self._last_service_pid = service_pid
                self._last_process_start_ticks = process_start_ticks
                self._last_process_command_sha256 = process_command_sha256
                self._last_process_group_id = (
                    bound_process_group if process_group_identity_present else None
                )
                self._last_process_session_id = (
                    cast(int, process_session_id) if process_group_identity_present else None
                )
                self._service_instance_token_sha256 = (
                    cast(str, instance_token_sha256) if process_group_identity_present else None
                )
                self._launch_gate_token_sha256 = (
                    cast(str, launch_token_sha256) if isinstance(launch_token_sha256, str) else None
                )
                self._launch_supervisor_command_sha256 = (
                    cast(str, launch_supervisor_sha256)
                    if isinstance(launch_supervisor_sha256, str)
                    else None
                )
                self._write_service_lock_metadata(
                    lease_state="stopped_verified",
                    service_pid=service_pid,
                    ended_at=self._physically_stopped_at,
                )
                self._last_recovered_process_identity = recovered_process_identity
                terminalized = True

            try:
                record = self.meter.recover_service_journal(
                    service_session_id=service_session_id,
                    recovered_at=recovered_at,
                    details={
                        "pid_absence_verified": True,
                        "process_group_absence_verified": True,
                        "endpoint_absence_verified": True,
                        "controller_lost_allocation_event_ids": [
                            event.event_id for event in recovered_allocations
                        ],
                        "shared_terminal_recovery_at": recovered_at.isoformat(),
                    },
                )
            except BaseException:
                latest = self.meter.ledger.latest_gpu_service_journal(service_session_id)
                record = self.meter.ledger.get_gpu_service_session(service_session_id)
                if (
                    latest is not None
                    and latest.state
                    in {GpuServiceJournalState.CLOSED, GpuServiceJournalState.RECOVERED}
                    and record is not None
                ):
                    finalize_lease(record)
                raise
            finalize_lease(record)
            return record
        finally:
            if terminalized:
                self.state = ServiceState.STOPPED
            self._session_id = None
            self._accounting_session_id = None
            self._started_at = None
            self._started_monotonic = None
            self._allocated_at_start = None
            self._carried_service_seconds = 0.0
            self._physically_stopped_at = None
            self._physically_stopped_seconds = None
            self._process_stop_journaled = False
            self._prior_process_stop_contradicted = False
            self._service_journal_opened = False
            self._service_heartbeat_failure = None
            self._release_service_lock()

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> bool:
        self.shutdown()
        return False


def _atomic_write_private_json(path: Path, value: Mapping[str, object]) -> None:
    """Atomic operational state; unlike public manifests this may contain a PID."""

    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(value, stream, ensure_ascii=False, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(path)
        try:
            directory_descriptor = os.open(path.parent, os.O_RDONLY)
        except OSError:
            directory_descriptor = None
        if directory_descriptor is not None:
            try:
                os.fsync(directory_descriptor)
            finally:
                os.close(directory_descriptor)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def public_runtime_manifest(
    *,
    launcher: VLLMLaunchConfiguration,
    tokenizer: TokenizerManifest,
    runtime_stack: RuntimeStackManifest | None = None,
    gpu_hardware: GPUHardwareIdentity | None = None,
    resource_samples: Sequence[ResourceSnapshot] = (),
    uptime: ServiceUptime | None = None,
    ledger: Ledger | None = None,
) -> dict[str, object]:
    """Compose hashes and accounting only; prompts, outputs, and paths are excluded."""

    if (
        tokenizer.repository != launcher.repository
        or tokenizer.revision != launcher.revision
        or tokenizer.tokenizer_revision != launcher.revision
    ):
        raise RuntimeConfigurationError(
            "runtime tokenizer identity differs from the selected model candidate"
        )
    payload: dict[str, object] = {
        "schema_version": SCHEMA_VERSION,
        "launcher": launcher.public_manifest(),
        "tokenizer": tokenizer.public_manifest(),
        "runtime_stack": None if runtime_stack is None else runtime_stack.public_manifest(),
        "gpu_hardware": None if gpu_hardware is None else gpu_hardware.public_manifest(),
        "resource_samples": [sample.public_manifest() for sample in resource_samples],
        "service_uptime": None if uptime is None else uptime.public_manifest(),
    }
    if ledger is not None:
        summary = ledger.gpu_summary()
        payload["gpu_accounting"] = {
            "event_count": summary.event_count,
            "service_session_count": summary.service_session_count,
            "total_allocated_microseconds": summary.total_allocated_microseconds,
            "by_kind_microseconds": {
                kind.value: microseconds for kind, microseconds in summary.by_kind_microseconds
            },
        }
    return {**payload, "manifest_sha256": canonical_sha256(payload)}


__all__ = [
    "DEFAULT_RESOURCE_SAMPLE_COMPLETION_SECONDS",
    "DEFAULT_SERVICE_START_WATCHDOG_SECONDS",
    "DEFAULT_SHUTDOWN_SECONDS",
    "DURABLE_EXEC_GATE_PROTOCOL",
    "FALLBACK_MODEL_REPOSITORY",
    "FALLBACK_MODEL_REVISION",
    "FALLBACK_SERVED_MODEL_NAME",
    "GPU_MEMORY_UTILIZATION",
    "GUIDED_DECODING_BACKEND",
    "MAXIMUM_CPU_WORKERS",
    "MAXIMUM_MODEL_LENGTH",
    "MODEL_CANDIDATES",
    "PINNED_MODEL_REPOSITORY",
    "PINNED_MODEL_REVISION",
    "PINNED_RUNTIME_VERSION",
    "PROC_ROOT",
    "RESOURCE_AWARE_HARD_STOP_RESERVE_SECONDS",
    "SCHEMA_VERSION",
    "SERVED_MODEL_NAME",
    "ChatMessage",
    "GPUHardwareIdentity",
    "GenerationResult",
    "GuidedJSONRequest",
    "ProcessTreeUsage",
    "ResourceSampler",
    "ResourceSnapshot",
    "ResourceWatchdog",
    "RuntimeConfigurationError",
    "RuntimeResourceLimitExceeded",
    "RuntimeStackManifest",
    "RuntimeTransportError",
    "RuntimeWatchdogTimeout",
    "ServiceState",
    "ServiceUptime",
    "TokenizerManifest",
    "VLLMGuidedJSONClient",
    "VLLMLaunchConfiguration",
    "VLLMService",
    "atomic_write_public_json",
    "capture_gpu_hardware_identity",
    "capture_runtime_stack",
    "capture_tokenizer_manifest",
    "public_runtime_manifest",
    "restricted_transport_failure_details",
    "sample_gpu_vram",
    "sample_process_tree",
    "sample_system_available_ram",
]
