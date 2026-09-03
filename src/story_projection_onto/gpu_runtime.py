"""Controlled local vLLM runtime for the Phase-1 acceptance pilot.

The module deliberately keeps process control, HTTP transport, resource sampling,
and public manifests separate from the semantic LLM contracts in :mod:`llm`.
Nothing here downloads a model: the launcher accepts only the verified, pinned
snapshot already present below the one shared project cache.
"""

from __future__ import annotations

import hashlib
import importlib
import importlib.metadata
import json
import os
import signal
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
from story_projection_onto.llm import DecodingManifest, DecodingPass, PackingReport
from story_projection_onto.store import GpuBudgetExceeded, Ledger, StoragePreflight

SCHEMA_VERSION = "1.0.0"
PINNED_MODEL_REPOSITORY = "Qwen/Qwen3-14B-AWQ"
PINNED_MODEL_REVISION = "1a6fe1ecf891437a270cce11ad54d796c4f56ce0"
PINNED_RUNTIME_VERSION = "0.10.2"
SERVED_MODEL_NAME = "qwen3-14b-awq-pinned"
MAXIMUM_MODEL_LENGTH = 12_288
MAXIMUM_CPU_WORKERS = 8
GPU_MEMORY_UTILIZATION = 0.88
DEFAULT_SERVICE_START_WATCHDOG_SECONDS = 180
DEFAULT_SHUTDOWN_SECONDS = 30
MEBIBYTE = 1024 * 1024
PROC_ROOT = Path("/proc")


class RuntimeConfigurationError(ValueError):
    """The launcher or request would depart from the frozen Phase-1 protocol."""


class RuntimeTransportError(RuntimeError):
    """The local vLLM HTTP service returned an invalid response."""


class RuntimeWatchdogTimeout(TimeoutError):
    """A local service operation exceeded its admitted watchdog."""


class RuntimeResourceLimitExceeded(RuntimeError):
    """A sampled hard resource limit was exceeded."""

    def __init__(self, snapshot: ResourceSnapshot) -> None:
        self.snapshot = snapshot
        super().__init__("; ".join(snapshot.violations))


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _path_identity(path: Path) -> str:
    """Return a stable public identifier without publishing an absolute path."""

    return _sha256_bytes(str(path.resolve()).encode("utf-8"))


def _require_plain_identifier(name: str, value: str) -> str:
    if not value or value.strip() != value or any(char in value for char in "\r\n\0"):
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
            json.dump(value, stream, ensure_ascii=False, indent=2, sort_keys=True)
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
    verified_snapshot_manifest_sha256: str | None = None

    def __post_init__(self) -> None:
        snapshot = self.snapshot_path.resolve(strict=True)
        cache = self.shared_cache.resolve(strict=True)
        try:
            snapshot.relative_to(cache)
        except ValueError as exc:
            raise RuntimeConfigurationError(
                "the model snapshot must be inside the one shared cache"
            ) from exc
        expected = {
            "repository": PINNED_MODEL_REPOSITORY,
            "revision": PINNED_MODEL_REVISION,
            "runtime_version": PINNED_RUNTIME_VERSION,
            "served_model_name": SERVED_MODEL_NAME,
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
        }
        for name, required in expected.items():
            if getattr(self, name) != required:
                raise RuntimeConfigurationError(f"{name} must remain {required!r}")
        if snapshot.name != self.revision:
            raise RuntimeConfigurationError("snapshot directory name must equal pinned revision")
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
        verified_snapshot_manifest_sha256: str | None = None,
        port: int = 8000,
    ) -> Self:
        raw = json.loads(model_configuration_path.read_text(encoding="utf-8"))
        if not isinstance(raw, Mapping):
            raise RuntimeConfigurationError("model configuration root must be an object")
        return cls(
            snapshot_path=snapshot_path,
            shared_cache=shared_cache,
            repository=cast(str, raw.get("repository")),
            revision=cast(str, raw.get("revision")),
            runtime_version=cast(str, raw.get("runtime_version")),
            tensor_parallel_size=cast(int, raw.get("tensor_parallel_size")),
            maximum_model_length=cast(int, raw.get("max_model_len")),
            gpu_memory_utilization=cast(float, raw.get("gpu_memory_utilization_pilot")),
            cpu_offload_gb=cast(int, raw.get("cpu_offload_gb")),
            maximum_sequences=cast(int, raw.get("request_concurrency")),
            prefix_caching=cast(bool, raw.get("prefix_decoding")),
            speculative_decoding=cast(bool, raw.get("speculative_decoding")),
            verified_snapshot_manifest_sha256=verified_snapshot_manifest_sha256,
            port=port,
        )

    @property
    def base_url(self) -> str:
        return f"http://{self.host}:{self.port}"

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
            "verified_snapshot_manifest_sha256": self.verified_snapshot_manifest_sha256,
        }
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
            "--no-enable-prefix-caching",
            "--no-enable-log-requests",
            "--uvicorn-log-level",
            "warning",
        )

    def environment(self, base: Mapping[str, str] | None = None) -> dict[str, str]:
        environment = dict(os.environ if base is None else base)
        cache = str(self.shared_cache)
        environment.update(
            {
                "HF_HOME": cache,
                "HF_HUB_CACHE": cache,
                "HUGGINGFACE_HUB_CACHE": cache,
                "TRANSFORMERS_CACHE": cache,
                "VLLM_CACHE_ROOT": cache,
                "HF_HUB_OFFLINE": "1",
                "TRANSFORMERS_OFFLINE": "1",
                "HF_DATASETS_OFFLINE": "1",
                "TOKENIZERS_PARALLELISM": "false",
                "OMP_NUM_THREADS": "8",
                "MKL_NUM_THREADS": "8",
                "OPENBLAS_NUM_THREADS": "8",
                "NUMEXPR_NUM_THREADS": "8",
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

    def __post_init__(self) -> None:
        _require_plain_identifier("request_id", self.request_id)
        if self.model_name != SERVED_MODEL_NAME:
            raise RuntimeConfigurationError("request must target the pinned served-model alias")
        if not self.messages:
            raise RuntimeConfigurationError("request must include messages")
        if self.condition is not self.packing.condition:
            raise RuntimeConfigurationError("request and packing conditions differ")
        if self.packing.tokenizer_revision != self.decoding.tokenizer_revision:
            raise RuntimeConfigurationError("packing and decoding tokenizer revisions differ")
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
            "guided_json": self.output_schema,
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
            "stream": False,
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
) -> tuple[int, bytes, Mapping[str, str]]:
    request = urllib.request.Request(
        url,
        data=payload,
        headers={"Content-Type": "application/json"} if payload is not None else {},
        method="POST" if payload is not None else "GET",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
            return response.status, response.read(), dict(response.headers.items())
    except urllib.error.HTTPError as exc:
        # Preserve the actual service response.  HTTPError is also a URLError,
        # so it must be handled first or a schema/request rejection is
        # misleadingly reported as a connection failure.
        return exc.code, exc.read(), dict(exc.headers.items())
    except TimeoutError as exc:
        raise RuntimeWatchdogTimeout(f"local vLLM request exceeded {timeout_seconds}s") from exc
    except urllib.error.URLError as exc:
        if isinstance(exc.reason, TimeoutError):
            raise RuntimeWatchdogTimeout(f"local vLLM request exceeded {timeout_seconds}s") from exc
        raise RuntimeTransportError(f"cannot reach local vLLM service: {exc.reason}") from exc


class VLLMGuidedJSONClient:
    """Concurrency-one guided-JSON client restricted to a local vLLM service."""

    def __init__(self, base_url: str, *, transport: Transport = _urllib_transport) -> None:
        self.base_url = _loopback_base_url(base_url)
        self._transport = transport
        self._generation_lock = threading.Lock()

    def health(self, timeout_seconds: float = 2.0) -> bool:
        try:
            status, _, _ = self._transport(
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
            status, body, _ = self._transport(
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

    def ready(self, timeout_seconds: float = 2.0) -> bool:
        """Check both generic health and the exact model identity."""

        return self.health(timeout_seconds) and self.serves_model(
            SERVED_MODEL_NAME,
            timeout_seconds,
        )

    def generate(
        self,
        request: GuidedJSONRequest,
        *,
        watchdog_seconds: float,
    ) -> GenerationResult:
        if watchdog_seconds <= 0:
            raise RuntimeConfigurationError("watchdog_seconds must be positive")
        payload = canonical_json(request.wire_payload()).encode("utf-8")
        if not self._generation_lock.acquire(blocking=False):
            raise RuntimeConfigurationError("concurrent vLLM generation is mechanically forbidden")
        try:
            status, body, headers = self._transport(
                f"{self.base_url}/v1/chat/completions",
                payload,
                watchdog_seconds,
            )
        finally:
            self._generation_lock.release()
        if status != 200:
            raise RuntimeTransportError(f"vLLM returned HTTP {status}")
        try:
            response = json.loads(body)
            choice = response["choices"][0]
            content = choice["message"]["content"]
            parsed_object = json.loads(content)
            usage = response["usage"]
        except (IndexError, KeyError, TypeError, json.JSONDecodeError) as exc:
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
        }


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
    ) -> ResourceSnapshot:
        _require_plain_identifier("sample_id", sample_id)
        process = self._process_sampler(root_pid)
        project_storage = self.storage.measure_occupied_bytes()
        filesystem_free = self._filesystem_free_sampler()
        values = {
            "process_ram_bytes": process.rss_bytes,
            "system_available_ram_bytes": self._system_ram_sampler(),
            "gpu_vram_bytes": self._gpu_sampler(process.pids),
            "project_storage_bytes": project_storage,
            "filesystem_free_bytes": filesystem_free,
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
        )
        self._samples.append(snapshot)
        if self.ledger is not None:
            self.ledger.record_resource_sample(
                sample_id=sample_id,
                job_id=job_id,
                gpu_event_id=gpu_event_id,
                process_ram_bytes=snapshot.process_ram_bytes,
                system_available_ram_bytes=snapshot.system_available_ram_bytes,
                gpu_vram_bytes=snapshot.gpu_vram_bytes,
                project_storage_bytes=snapshot.project_storage_bytes,
                cpu_worker_count=snapshot.cpu_worker_count,
                sampled_at=sampled_at,
            )
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
    job_id: str | None = None
    gpu_event_id: str | None = None
    allocation_guard: Callable[[], None] | None = None
    on_limit: Callable[[ResourceSnapshot], None] | None = None
    on_failure: Callable[[BaseException], None] | None = None
    _stop: threading.Event = field(default_factory=threading.Event, init=False, repr=False)
    _thread: threading.Thread | None = field(default=None, init=False, repr=False)
    _failure: BaseException | None = field(default=None, init=False, repr=False)
    _sample_count: int = field(default=0, init=False)

    def __post_init__(self) -> None:
        _require_plain_identifier("sample_prefix", self.sample_prefix)
        if self.root_pid <= 0:
            raise RuntimeConfigurationError("resource watchdog root PID must be positive")
        if self.interval_seconds <= 0:
            raise RuntimeConfigurationError("resource watchdog interval must be positive")

    @property
    def sample_count(self) -> int:
        return self._sample_count

    @property
    def failure(self) -> BaseException | None:
        return self._failure

    def _run(self) -> None:
        while not self._stop.is_set():
            self._sample_count += 1
            try:
                if self.allocation_guard is not None:
                    self.allocation_guard()
                self.sampler.sample(
                    sample_id=f"{self.sample_prefix}-{self._sample_count:06d}",
                    root_pid=self.root_pid,
                    job_id=self.job_id,
                    gpu_event_id=self.gpu_event_id,
                )
            except RuntimeResourceLimitExceeded as exc:
                self._failure = exc
                if self.on_limit is not None:
                    self.on_limit(exc.snapshot)
                if self.on_failure is not None:
                    self.on_failure(exc)
                self._stop.set()
                return
            except BaseException as exc:
                self._failure = exc
                if self.on_failure is not None:
                    self.on_failure(exc)
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

    def stop(self, *, raise_failure: bool = True) -> None:
        self._stop.set()
        if self._thread is None:
            return
        self._thread.join(timeout=max(1.0, self.interval_seconds * 2))
        if self._thread.is_alive() and raise_failure:
            raise RuntimeError("resource watchdog did not stop")
        if self._failure is not None and raise_failure:
            raise self._failure

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


class ProcessHandle(Protocol):
    pid: int

    def poll(self) -> int | None: ...

    def wait(self, timeout: float | None = None) -> int: ...

    def terminate(self) -> None: ...

    def kill(self) -> None: ...


PopenFactory = Callable[..., ProcessHandle]


def _signal_controlled_process_group(pid: int, signal_number: int) -> None:
    """Signal only a child that is the leader of its launcher-created session."""

    process_group = os.getpgid(pid)
    if process_group != pid:
        raise RuntimeError("refusing to signal a process outside its controlled vLLM group")
    os.killpg(process_group, signal_number)


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


@dataclass(slots=True)
class _AdoptedProcess:
    pid: int
    monotonic_clock: Callable[[], float] = time.monotonic
    sleep: Callable[[float], None] = time.sleep

    def poll(self) -> int | None:
        try:
            os.kill(self.pid, 0)
        except ProcessLookupError:
            return 0
        return None

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
class VLLMService:
    """Safe process lifecycle, metered without overlapping allocation intervals."""

    configuration: VLLMLaunchConfiguration
    client: VLLMGuidedJSONClient
    meter: AllocatedGPUMeter
    log_path: Path | None = None
    startup_resource_sampler: ResourceSampler | None = None
    startup_sample_interval_seconds: float = 1.0
    preflight_endpoint_check: Callable[[], bool] | None = None
    readiness_check: Callable[[], bool] | None = None
    popen_factory: PopenFactory = subprocess.Popen
    monotonic_clock: Callable[[], float] = time.monotonic
    wall_clock: Callable[[], datetime] = lambda: datetime.now(UTC)
    sleep: Callable[[float], None] = time.sleep
    process_group_signaler: Callable[[int, int], None] = _signal_controlled_process_group
    available_cpu_sampler: Callable[[], set[int]] = os.sched_getaffinity
    affinity_setter: Callable[[int, set[int]], None] = os.sched_setaffinity
    state: ServiceState = field(default=ServiceState.STOPPED, init=False)
    _process: ProcessHandle | None = field(default=None, init=False, repr=False)
    _session_id: str | None = field(default=None, init=False, repr=False)
    _accounting_session_id: str | None = field(default=None, init=False, repr=False)
    _started_monotonic: float | None = field(default=None, init=False, repr=False)
    _started_at: datetime | None = field(default=None, init=False, repr=False)
    _allocated_at_start: float | None = field(default=None, init=False, repr=False)
    _carried_service_seconds: float = field(default=0.0, init=False, repr=False)
    _log_stream: BinaryIO | None = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        if self.log_path is not None:
            self.log_path = self.log_path.resolve()
            if self.log_path.exists() and not self.log_path.is_file():
                raise RuntimeConfigurationError("vLLM log path must be a regular file")
        if self.startup_sample_interval_seconds <= 0:
            raise RuntimeConfigurationError("startup sample interval must be positive")

    @property
    def pid(self) -> int:
        if self._process is None:
            raise RuntimeError("vLLM service has no controlled process")
        return self._process.pid

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

    def _spawn(self) -> None:
        output = self._open_log_stream()
        self._process = self.popen_factory(
            self.configuration.command(),
            env=self.configuration.environment(),
            stdin=subprocess.DEVNULL,
            stdout=output,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        allowed_cpus = set(sorted(self.available_cpu_sampler())[: self.configuration.cpu_workers])
        if not allowed_cpus:
            raise RuntimeConfigurationError("vLLM process has no permitted CPU affinity")
        self.affinity_setter(self._process.pid, allowed_cpus)

    def _ready(self) -> bool:
        if self.readiness_check is not None:
            return self.readiness_check()
        return self.client.ready()

    def _wait_until_healthy(self, watchdog_seconds: float) -> None:
        deadline = self.monotonic_clock() + watchdog_seconds
        while self.monotonic_clock() < deadline:
            if self._process is None or self._process.poll() is not None:
                self.state = ServiceState.FAILED
                raise RuntimeTransportError("vLLM process exited before becoming healthy")
            if self._ready():
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
        if self.preflight_endpoint_check is not None and self.preflight_endpoint_check():
            raise RuntimeConfigurationError(
                "configured loopback port already exposes a service; refusing a duplicate launch"
            )
        self.state = ServiceState.STARTING
        self._session_id = session_id
        if self._accounting_session_id is None:
            self._accounting_session_id = event_id
        if self._started_monotonic is None:
            self._started_monotonic = self.monotonic_clock()
        if self._started_at is None:
            self._started_at = self.wall_clock()
        if self._allocated_at_start is None:
            self._allocated_at_start = self.meter.actual_allocated_gpu_seconds
        startup_watchdog: ResourceWatchdog | None = None
        try:
            self._spawn()
            if self.startup_resource_sampler is not None:
                startup_watchdog = ResourceWatchdog(
                    sampler=self.startup_resource_sampler,
                    root_pid=self.pid,
                    sample_prefix=f"{event_id}-startup",
                    interval_seconds=self.startup_sample_interval_seconds,
                    allocation_guard=self.require_hard_stop_margin,
                    on_failure=lambda _: self.emergency_stop(),
                )
                startup_watchdog.start()
            self._wait_until_healthy(watchdog_seconds)
            if startup_watchdog is not None:
                startup_watchdog.stop()
        except BaseException:
            if startup_watchdog is not None:
                startup_watchdog.stop(raise_failure=False)
            self._stop_process()
            raise

    def start(
        self,
        *,
        session_id: str,
        event_id: str,
        watchdog_seconds: float = DEFAULT_SERVICE_START_WATCHDOG_SECONDS,
        remaining_required_seconds: float = 0,
    ) -> None:
        _require_plain_identifier("session_id", session_id)
        with self.meter.session_start(
            event_id=event_id,
            maximum_seconds=watchdog_seconds,
            remaining_required_seconds=remaining_required_seconds,
            details={
                "session_id": session_id,
                "configuration_hash": self.configuration.configuration_hash,
            },
        ):
            self._start_unmetered(
                session_id=session_id,
                event_id=event_id,
                watchdog_seconds=watchdog_seconds,
            )

    def restart(
        self,
        *,
        event_id: str,
        watchdog_seconds: float,
        remaining_required_seconds: float = 0,
    ) -> None:
        if self._session_id is None:
            raise RuntimeError("cannot restart a service without a session identity")
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
            self._stop_process()
            self._start_unmetered(
                session_id=session_id,
                event_id=event_id,
                watchdog_seconds=watchdog_seconds,
            )

    def write_resume_checkpoint(self, path: Path, *, proc_root: Path = PROC_ROOT) -> None:
        """Persist a PID-reuse-safe operational checkpoint without a model path."""

        self.require_ready()
        payload = {
            "schema_version": SCHEMA_VERSION,
            "configuration_hash": self.configuration.configuration_hash,
            "session_id": self._session_id,
            "accounting_session_id": self._accounting_session_id,
            "pid": self.pid,
            "process_start_ticks": _process_start_ticks(self.pid, proc_root),
            "ledger_allocated_seconds_before_session": self._allocated_at_start,
            "session_service_seconds": self._elapsed_service_seconds(),
            "session_started_at": cast(datetime, self._started_at).isoformat(),
            "recorded_at": self.wall_clock().isoformat(),
        }
        _atomic_write_private_json(path, payload)

    def resume_from_checkpoint(
        self,
        path: Path,
        *,
        proc_root: Path = PROC_ROOT,
        adopted_factory: Callable[[int], ProcessHandle] = _AdoptedProcess,
    ) -> bool:
        """Adopt the exact still-live service; never start or meter a duplicate load."""

        if self.state is not ServiceState.STOPPED or self._process is not None:
            raise RuntimeError("resume requires a stopped controller")
        if not path.exists():
            return False
        value = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(value, Mapping):
            raise RuntimeConfigurationError("service checkpoint root must be an object")
        if value.get("configuration_hash") != self.configuration.configuration_hash:
            raise RuntimeConfigurationError("service checkpoint configuration changed")
        pid = value.get("pid")
        start_ticks = value.get("process_start_ticks")
        carried_seconds = value.get("session_service_seconds")
        ledger_before_session = value.get("ledger_allocated_seconds_before_session")
        started_at_text = value.get("session_started_at")
        session_id = value.get("session_id")
        accounting_session_id = value.get("accounting_session_id")
        if (
            isinstance(pid, bool)
            or not isinstance(pid, int)
            or pid <= 0
            or isinstance(start_ticks, bool)
            or not isinstance(start_ticks, int)
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
        ):
            raise RuntimeConfigurationError("service checkpoint has invalid process identity")
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
        try:
            command_line = (proc_root / str(pid) / "cmdline").read_bytes().split(b"\0")
        except OSError as exc:
            raise RuntimeConfigurationError("cannot inspect checkpointed service") from exc
        required_parts = {
            b"vllm.entrypoints.openai.api_server",
            str(self.configuration.snapshot_path).encode("utf-8"),
        }
        if not required_parts.issubset(set(command_line)):
            raise RuntimeConfigurationError("checkpoint process is not the configured vLLM service")
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
        self._session_id = session_id
        self._accounting_session_id = accounting_session_id
        self._started_at = started_at
        self._started_monotonic = self.monotonic_clock()
        self._allocated_at_start = float(ledger_before_session)
        # A service that remained live consumed allocation between checkpoint
        # write and controller recovery.  Charge the larger of the persisted
        # monotonic duration and wall-clock duration instead of losing that gap.
        self._carried_service_seconds = max(float(carried_seconds), wall_service_seconds)
        if not self._ready():
            self.state = ServiceState.FAILED
            return False
        self.state = ServiceState.READY
        return True

    def handoff_resume(
        self,
        path: Path,
        *,
        proc_root: Path = PROC_ROOT,
    ) -> VLLMService:
        """Exercise checkpoint/resume by transferring the live controlled handle."""

        self.write_resume_checkpoint(path, proc_root=proc_root)
        process = self._process
        assert process is not None
        resumed = VLLMService(
            configuration=self.configuration,
            client=self.client,
            meter=self.meter,
            log_path=self.log_path,
            startup_resource_sampler=self.startup_resource_sampler,
            startup_sample_interval_seconds=self.startup_sample_interval_seconds,
            preflight_endpoint_check=self.preflight_endpoint_check,
            readiness_check=self.readiness_check,
            popen_factory=self.popen_factory,
            monotonic_clock=self.monotonic_clock,
            wall_clock=self.wall_clock,
            sleep=self.sleep,
            process_group_signaler=self.process_group_signaler,
            available_cpu_sampler=self.available_cpu_sampler,
            affinity_setter=self.affinity_setter,
        )
        if not resumed.resume_from_checkpoint(
            path,
            proc_root=proc_root,
            adopted_factory=lambda pid: process,
        ):
            raise RuntimeError("live vLLM service failed its resume health check")
        # Ownership transfers only after every PID/config/health check succeeds.
        resumed._log_stream = self._log_stream
        self._log_stream = None
        self._process = None
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
            return operation()

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
            return operation()

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
    ) -> GenerationResult:
        self.require_ready()
        self.require_service_capacity(
            watchdog_seconds,
            remaining_required_seconds=remaining_required_seconds,
        )
        if repair != (request.decoding.decoding_pass is DecodingPass.REPAIR):
            raise RuntimeConfigurationError("repair event kind differs from decoding pass")
        context = self.meter.repair if repair else self.meter.inference
        with context(
            event_id=event_id,
            maximum_seconds=watchdog_seconds,
            remaining_required_seconds=remaining_required_seconds,
            job_id=job_id,
            attempt_id=attempt_id,
            details={"request_id": request.request_id, "request_hash": request.request_hash},
        ):
            return self.client.generate(request, watchdog_seconds=watchdog_seconds)

    def require_ready(self) -> None:
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
        shutdown_reserve_seconds = 2 * DEFAULT_SHUTDOWN_SECONDS
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

        shutdown_reserve_seconds = 2 * DEFAULT_SHUTDOWN_SECONDS
        if (
            self.actual_allocated_service_seconds + shutdown_reserve_seconds
            >= self.meter.hard_limit_seconds
        ):
            raise GpuBudgetExceeded("vLLM service reached its protected hard-stop margin")

    def _stop_process(self, shutdown_seconds: float = DEFAULT_SHUTDOWN_SECONDS) -> None:
        process = self._process
        self._process = None
        if process is None:
            self._close_log_stream()
            self.state = ServiceState.STOPPED
            return
        try:
            if process.poll() is None:
                with suppress(ProcessLookupError):
                    self.process_group_signaler(process.pid, signal.SIGTERM)
                try:
                    process.wait(timeout=shutdown_seconds)
                except subprocess.TimeoutExpired:
                    with suppress(ProcessLookupError):
                        self.process_group_signaler(process.pid, signal.SIGKILL)
                    process.wait(timeout=shutdown_seconds)
        finally:
            self._close_log_stream()
            self.state = ServiceState.STOPPED

    def emergency_stop(self) -> None:
        """Stop the controlled process tree while retaining uptime for final accounting."""

        self._stop_process(shutdown_seconds=min(5.0, DEFAULT_SHUTDOWN_SECONDS))

    def shutdown(
        self,
        *,
        shutdown_seconds: float = DEFAULT_SHUTDOWN_SECONDS,
    ) -> ServiceUptime | None:
        if self._started_at is None or self._started_monotonic is None or self._session_id is None:
            self._stop_process(shutdown_seconds)
            return None
        self._stop_process(shutdown_seconds)
        ended_at = self.wall_clock()
        service_seconds = self._elapsed_service_seconds()
        if service_seconds < 0:
            raise RuntimeError("monotonic service clock moved backwards")
        allocated_start = self._allocated_at_start or 0.0
        allocated_seconds = self.meter.actual_allocated_gpu_seconds - allocated_start
        uptime = ServiceUptime(
            session_id=self._session_id,
            started_at=self._started_at,
            ended_at=ended_at,
            service_seconds=service_seconds,
            allocated_event_seconds=allocated_seconds,
        )
        accounting_session_id = self._accounting_session_id
        try:
            if accounting_session_id is None:
                raise RuntimeError("live vLLM service has no accounting session identity")
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
                },
            )
        finally:
            self._started_at = None
            self._started_monotonic = None
            self._allocated_at_start = None
            self._accounting_session_id = None
            self._carried_service_seconds = 0.0
        return uptime

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
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def public_runtime_manifest(
    *,
    launcher: VLLMLaunchConfiguration,
    tokenizer: TokenizerManifest,
    runtime_stack: RuntimeStackManifest | None = None,
    resource_samples: Sequence[ResourceSnapshot] = (),
    uptime: ServiceUptime | None = None,
    ledger: Ledger | None = None,
) -> dict[str, object]:
    """Compose hashes and accounting only; prompts, outputs, and paths are excluded."""

    payload: dict[str, object] = {
        "schema_version": SCHEMA_VERSION,
        "launcher": launcher.public_manifest(),
        "tokenizer": tokenizer.public_manifest(),
        "runtime_stack": None if runtime_stack is None else runtime_stack.public_manifest(),
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
    "DEFAULT_SERVICE_START_WATCHDOG_SECONDS",
    "DEFAULT_SHUTDOWN_SECONDS",
    "GPU_MEMORY_UTILIZATION",
    "MAXIMUM_CPU_WORKERS",
    "MAXIMUM_MODEL_LENGTH",
    "PINNED_MODEL_REPOSITORY",
    "PINNED_MODEL_REVISION",
    "PINNED_RUNTIME_VERSION",
    "PROC_ROOT",
    "SCHEMA_VERSION",
    "SERVED_MODEL_NAME",
    "ChatMessage",
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
    "capture_runtime_stack",
    "capture_tokenizer_manifest",
    "public_runtime_manifest",
    "sample_gpu_vram",
    "sample_process_tree",
    "sample_system_available_ram",
]
