"""Typed public record for the terminal fallback-v8 runtime incident.

V8 loaded the pinned fallback model and completed the controller handoff, but
the restarted controller could not adopt the service because vLLM's EngineCore
process did not retain a matching private service-instance token.  The strict
process-group guard failed closed before inference.  This record authenticates
the public failures, private control records, exact operator stop, and
conservative terminal accounting without publishing commands, paths, tokens,
or log text.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import sqlite3
import stat
import tempfile
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal, Self

from pydantic import AwareDatetime, BaseModel, ConfigDict, Field, ValidationError, model_validator

from story_projection_onto.contracts import GitRevision, Sha256Digest, canonical_json
from story_projection_onto.contracts import canonical_sha256 as _canonical_sha256
from story_projection_onto.fallback_control_plane_incident import (
    OpaqueFileBinding,
    SafeBasename,
    SelfHashedFileBinding,
)
from story_projection_onto.store import GpuEventKind, ReadOnlyLedger

FALLBACK_V8_RUN_ID = "fallback-qwen3-8b-awq-development-v8"
FALLBACK_V8_SOURCE_REVISION = "fallback-second-recovery-v8"
FALLBACK_V8_RUNTIME_INCIDENT_KIND = "fallback_gpu_acceptance_v8_runtime_incident"
FALLBACK_V8_SERVICE_EVENT_ID = f"{FALLBACK_V8_RUN_ID}-service-start-001"
FALLBACK_V8_SAFE_ERROR_CLASS = (
    "engine_core_service_instance_token_nonmatch_blocked_exact_group_adoption"
)
FALLBACK_V8_ROOT_CAUSE = (
    "vLLM EngineCore did not retain a matching private service-instance token, "
    "so exact process-group adoption and automated shutdown failed closed"
)

EXPECTED_BASELINE_MICROSECONDS = 1_507_850_965
EXPECTED_PRE_RECOVERY_MICROSECONDS = 1_735_537_551
EXPECTED_CLASSIFIED_MICROSECONDS = 227_686_586
EXPECTED_OVERHEAD_MICROSECONDS = 845_730_152
EXPECTED_SERVICE_MICROSECONDS = 1_073_416_738
EXPECTED_TERMINAL_MICROSECONDS = 2_581_267_703
EXPECTED_SCHEMA_SHA256 = "d9e134f8d608258c28e633bad2c8126aa88d3a37f016080ab8315c8e807ece0a"
EXPECTED_PROTECTED_CONTENT_SHA256 = (
    "1923c9e9e45e611761032543ca5a22f8ff3a00e0bb59e513a742639a4f259151"
)
EXPECTED_AUDITED_AT = "2026-09-05T21:07:57.812706+00:00"
EXPECTED_SOURCE_GIT_COMMIT = "4b5b0cc768844f7661fc7ca316bd7e63cc15a2a6"
EXPECTED_SOURCE_TREE_SHA256 = (
    "3ac890be9fa0b13d2d4db5f28795287f2c1f62faed3db892fa33fc4ec7a7b194"
)
EXPECTED_RUN_ROOT_INVENTORY_SHA256 = (
    "2f714c01596522e47265dfff121dbfb43187ecd6de7ce100803b9ac090caa5e9"
)
EXPECTED_EXECUTION_ARGUMENTS_SHA256 = (
    "0aa53a69c752e09ba7c5f06f2af491e10ef1acec98ac502fed688f3f66af8d25"
)
EXPECTED_EXECUTION_SHA256 = (
    "0ae547bc8dfd8823206cd9b141196275f2839636f2b14cdb7c3558af3c6e094c"
)
EXPECTED_PRE_STOP_LEASE_FILE_SHA256 = (
    "50e1354f2734ac00ef06f639b9c58c43fd3cde86df9c73ea2f1f26cfbca00a62"
)
EXPECTED_PRE_STOP_LEASE_MANIFEST_SHA256 = (
    "0d3a9f8d57cf52568bc9016fb4990d5809d869f82b3d01d22b1a5febe90ec098"
)

# Each tuple is basename, byte size, file SHA-256, kind, and canonical manifest SHA-256.
# ``None`` means that the preserved evidence is intentionally opaque rather than
# self-hashed JSON.  These constants make this an incident-specific contract, not
# a generic shape that can authenticate a substituted V8 history.
EXPECTED_SOURCE_BINDINGS = {
    "source_association": (
        "source_tree_fallback_second_recovery_v8.association.json",
        1106,
        "5dd314003f802717feed6c72bf801d89adf4aca848351f8686409554bc46bc75",
        "local_remote_source_tree_association",
        "28e002f1b5eaa76fe0b1b753c0ebcee6f3b186ba23bca355b1a40035ce8dbd11",
    ),
    "authorization_overlay": (
        "fallback-second-recovery-v8.authorized.json",
        37829,
        "3b215d02c73066546257da53e067ba4d47ab3f0f986640cede5d08c14b67c775",
        "phase1_fallback_second_recovery_overlay",
        "c9a0e50862d7c1301c2bae4fc73911014ab6ab44aebbbac1f0b6710ef159ad4e",
    ),
    "execution_preflight": (
        "fallback_gpu_acceptance_development_v8.preflight.json",
        3295,
        "2f59e51263f352dafd212d575d492bec21528b823b5bb83e75c5758074bba1c2",
        "phase1_fallback_execution_preflight",
        "070da9bf50161c43cfec94d7880b51f9e550bb614140746786d4857ee58a3305",
    ),
}
EXPECTED_PUBLIC_BINDINGS = {
    "controller_handoff": (
        "fallback_gpu_acceptance_development_v8.json.controller-handoff.json",
        3454,
        "c5ca07fe734898766c053d08a8c1bac14511a2f1b17370016cad50ed6350e003",
        "phase1_fallback_controller_restart_handoff",
        "3721072a6def595d62f37a458df413c6ce90503af25a2901dda90cb7f0892f2b",
    ),
    "run_result": (
        "fallback_gpu_acceptance_development_v8.json",
        9217,
        "c023b29626bae21421f5bc26e307bf47e6973cd7164cda38ab9d9a7813ef112e",
        "phase1_fallback_micro_pilot_result",
        "efb1aa29a02ab51a75ef02cb895582eb6b12412200df8224bf5fda89cef05fc2",
    ),
    "orphan_cleanup": (
        "fallback_gpu_acceptance_development_v8.json.orphan-cleanup.json",
        9234,
        "d0738f06692a94833ab1cf12177308b64a6c9cfde3d6ff5be2e3cc668d3a290a",
        "phase1_fallback_micro_pilot_result",
        "17f6da3408ea51bffcca449a40dc24b1c3297243fb8194c024b8eee4320fc5c4",
    ),
}
EXPECTED_RESTRICTED_BINDINGS = {
    "orchestration_invocation": (
        "checkpoint.json.orchestrator-invocation.json",
        1245,
        "84cf24d2f3f3166ccfcf66fa1988d295d9def755a498c8525a0f98f9c16042b4",
        "fallback_controller_orchestration_invocation",
        "b085244d3a3ca1dcc77a3bed11399d42a62b63696fa5ee9226e9dd4a4fb5c1d3",
    ),
    "guardian_ticket": (
        "checkpoint.json.guardian-ticket.json",
        755,
        "0e683057e181a7c1b2821a1745aa2fad6dd4180a504f4c4b1282391f19210869",
        "fallback_service_guardian_ticket",
        "b7b29fe54b3aba8f6a269d5a05b64b0fcb2dc39a3b8c3215e583c71db0436746",
    ),
    "guardian_ready": (
        "checkpoint.json.guardian-ready-000001.json",
        753,
        "8f15318628fe6015d17bfb083985b4fad7ed0ca6d7b48b4751fe22151b728c03",
        "fallback_service_guardian_ready",
        "bf0bfce246a640f8cda74b694ecf999969f2c956ccac82a274dd150641f49575",
    ),
    "guardian_terminal_request": (
        "checkpoint.json.guardian-terminal-request.json",
        574,
        "b0c29a02a3a043483a81e26d09a26f1503b28ccd94ce1c35413c5e9456f91ed1",
        "fallback_guardian_terminal_request",
        "ab09c3a55e9e6804656fd47264ecf116603b6de0e6cd28479735446764f4eb8a",
    ),
    "orchestrator_guard": (
        "checkpoint.json.orchestrator-guard-000001.json",
        836,
        "f6a34edb56c0671f5ef54803ac3963f86b378ba0e824588e278c7986c35978cb",
        "fallback_controller_orchestrator_guard",
        "7ff06ba6cc8fac9f6e469c5232db61de54df0246e321c6781ff3c153aa9b7378",
    ),
    "closed_orchestrator_guard": (
        ".checkpoint.json.orchestrator-guard-000001.json.closed.json",
        484,
        "4c2a35f9c9371263efb0f2c0da077597a5b3b0df596aba602f9cf424f02ab80b",
        "fallback_controller_orchestrator_closed",
        "3bd37431f6b9cb26109217ad386a01b3ef30eaeb4338685c927c3495f2fbadf1",
    ),
    "prepare_controller_receipt": (
        "checkpoint.json.internal-controller-000001.json",
        1001,
        "809f3f992dfa33dbd842dbf9af737dd5e7299edec86420f4301d218012a3781e",
        "fallback_internal_controller_receipt",
        "e42b78737643c34f3bd911e3e9cc78d380e202119b8627aad5bdfcb1b5b172f2",
    ),
    "run_controller_receipt": (
        "checkpoint.json.internal-controller-000002.json",
        1035,
        "cca2517216bcb7d2038064ddedacbe272f318f7ae1b65bd5c59f0bdb6301cadd",
        "fallback_internal_controller_receipt",
        "e68bd194e52da8786588f2bd259c65506385b7f706388144e20ce3a516043389",
    ),
    "cleanup_controller_receipt": (
        "checkpoint.json.internal-controller-000003.json",
        1059,
        "4121913392c3f409ca75a8bc9a9ee3f86ab0798b3e4e20b3bb49c4eeec932662",
        "fallback_internal_controller_receipt",
        "cceae4462a8689add1a070396121d9e17f12916aae9a62bb381a3168dd6d6c5f",
    ),
    "checkpoint": (
        "checkpoint.json",
        1608,
        "73b8104297e70d22e7adb41dd772d114335ea6e3502f725f1943976cae5171d1",
        None,
        None,
    ),
    "service_checkpoint": (
        "checkpoint.json.service",
        829,
        "77d22690972d1f53d69fdfb9ec2756203f4b4488aa27b1fbd8a4bf37e1b4661b",
        None,
        None,
    ),
    "guardian_log": (
        "checkpoint.json.guardian.log",
        3446,
        "4ab40f879bbdd46d4cc0526f96de99c1adc160e00b5293a277c24942a7bad098",
        None,
        None,
    ),
    "orchestrator_log": (
        "orchestrator.20260905T201854Z.log",
        4268,
        "4347945d5ab6de32e0f4121ae01f50444ea719ab11fd203a234ffd042076f948",
        None,
        None,
    ),
    "vllm_log": (
        "fallback-qwen3-8b-awq-development-v8.vllm.log",
        13190,
        "8fddf7e3df2eb1fce1e4e936dfe9428e97b9a956290586897e979aa87643d6f7",
        None,
        None,
    ),
}
EXPECTED_MANUAL_BINDINGS = {
    "stop_intent": (
        "manual_stop_intent.json",
        1621,
        "0001e21c099a6e825d1a8ea046a40d89795d35b2a2de381f91ed7a802e9db559",
        "fallback_v8_exact_orphan_stop_intent",
        "9198b14ee69375409f278e59622cf4cd4358b3ce6eee2d1bdcbbe0bfe019637d",
    ),
    "stop_outcome": (
        "manual_stop_outcome.json",
        561,
        "50bb7f864737f47aabdb5e5159a6f4f46eb852eade451acb52d421f039c6ba1b",
        "fallback_v8_exact_orphan_stop_outcome",
        "c110120ed6ae8dd512f3f245861d273f7cc8f59581c9b04b3287e04f516b44c5",
    ),
    "accounting_receipt": (
        "terminal_accounting_receipt.json",
        1664,
        "0cd173be577f5c66717dc6a9365075b3471a860e8dd3432601f0f098ef87e420",
        "fallback_v8_terminal_accounting_recovery",
        "8588ad6cda3531a8d7a53052248c910d977b6127a3a4daa1a2e9dbd1c05d3041",
    ),
    "stop_utility": (
        "manual_stop_v8.py",
        12035,
        "11756d0b47c90d4f74b95705c8cc3bf49c960a34f78fe7cc887ee8c2b4e88197",
        None,
        None,
    ),
    "accounting_utility": (
        "recover_v8_ledger.py",
        13636,
        "42d7de801b7bc67fdd338ff92e33c49f0f3ba2af5f5d5085af090452fb30da93",
        None,
        None,
    ),
}
EXPECTED_LEDGER_IDENTITIES = {
    "before_v8": (
        "phase1_acceptance.before_v8.local.sqlite",
        602112,
        "742adfdf6c9ffe97da4ba898848e1a642b7808d0e22f119d20cb924efcc60f22",
    ),
    "before_manual_recovery": (
        "phase1_acceptance.before_recovery.sqlite",
        647168,
        "55b51b58d2cac2013b2b505cd7ea44777748edef38553bfeb156871a8e25b4c4",
    ),
    "terminal": (
        "phase1_acceptance.after_recovery.sqlite",
        647168,
        "33b18e16478b8c73951ff0269479ae697874ca1772669d87539ff2d7ac8884e2",
    ),
}
EXPECTED_LEDGER_SUMMARIES = {
    "before_v8": {
        "total_allocated_microseconds": 1_507_850_965,
        "event_count": 6,
        "service_session_count": 5,
        "attempt_count": 1,
        "model_call_count": 1,
        "artifact_count": 17,
        "storage_sample_count": 123,
        "resource_sample_count": 109,
        "allocation_journal_count": 182,
        "service_journal_count": 192,
        "unresolved_gpu_allocation_count": 0,
        "unresolved_gpu_service_count": 0,
        "by_kind_microseconds": {
            "failure": 225_183_297,
            "gpu_session_start": 214_034_494,
            "service_overhead": 645_767_518,
            "timeout": 422_865_656,
        },
    },
    "before_manual_recovery": {
        "total_allocated_microseconds": 1_735_537_551,
        "event_count": 7,
        "service_session_count": 5,
        "attempt_count": 1,
        "model_call_count": 1,
        "artifact_count": 17,
        "storage_sample_count": 133,
        "resource_sample_count": 115,
        "allocation_journal_count": 229,
        "service_journal_count": 247,
        "unresolved_gpu_allocation_count": 0,
        "unresolved_gpu_service_count": 1,
        "by_kind_microseconds": {
            "failure": 225_183_297,
            "gpu_session_start": 441_721_080,
            "service_overhead": 645_767_518,
            "timeout": 422_865_656,
        },
    },
    "terminal": {
        "total_allocated_microseconds": 2_581_267_703,
        "event_count": 7,
        "service_session_count": 6,
        "attempt_count": 1,
        "model_call_count": 1,
        "artifact_count": 17,
        "storage_sample_count": 133,
        "resource_sample_count": 115,
        "allocation_journal_count": 229,
        "service_journal_count": 248,
        "unresolved_gpu_allocation_count": 0,
        "unresolved_gpu_service_count": 0,
        "by_kind_microseconds": {
            "failure": 225_183_297,
            "gpu_session_start": 441_721_080,
            "service_overhead": 1_491_497_670,
            "timeout": 422_865_656,
        },
    },
}
EXPECTED_EXECUTION_IDENTITY = {
    "activation_certificate_sha256": (
        "224181b0929fd3f7800d9ad99e31e39ed6ed3ace42f2afdd81106efae989f657"
    ),
    "cache_replacement_receipt_sha256": (
        "6965fd3f9e2f557ec3360093748a8e7a093e98cdca9b37b753123dac94643558"
    ),
    "development_adopter_registration_hash": (
        "de2553e6bf6a96efb0685f2ec38bd701fa470ceae8a90b1af4626cb2d1ae3905"
    ),
    "fallback_plan_manifest_sha256": (
        "ce002b7063b7ca48c63e8351570e56e4e2ec36fb2a36fc6f7e34a36dbe9a88bd"
    ),
    "fallback_service_retry_amendment_sha256": (
        "9ef82782c03d9915c081e89cf554a531ef3d1fba42836c522d6ebf97cdce0f28"
    ),
    "gpu_hardware_manifest_sha256": (
        "ac2f28ba23e17a2233241d9cddc82bcfb6709533b64456aa57d71e45436f28b1"
    ),
    "launcher_configuration_sha256": (
        "7c2c97c7fd579653c72bf44cf45d11f9aa7d1fc602a8937338b7635894008576"
    ),
    "pre_fallback_gpu_accounting_sha256": (
        "1af5d9de726c9de3e14b871336991c60812a16327508b7167a75b1a107e8843c"
    ),
    "prior_fallback_failure_sha256": (
        "d08ce5ba02e866768a715eee9758be7ef13e5ed3c8e9844457e8d38b20240c7c"
    ),
    "prior_v6_control_plane_incident_sha256": (
        "d052713bd262745afc2060e0a75a3b565cf91db93365832b1a8b27dac09a81e9"
    ),
    "prior_v7_runtime_incident_sha256": (
        "4a550f74c1627e196d07db475acf9200fda2127f59516a59ad14b40e4867aa15"
    ),
    "runtime_stack_manifest_sha256": (
        "e43fa0c114ef63edebf617ab573f1e12b5562c11eb95572a0685a92388791bb1"
    ),
    "second_fallback_recovery_overlay_sha256": (
        "c9a0e50862d7c1301c2bae4fc73911014ab6ab44aebbbac1f0b6710ef159ad4e"
    ),
    "second_recovery_v3_incident_sha256": (
        "2e33bcca745dfd5e85b02e5f0f1039444cb082cd7bb88ed252677414f83e249a"
    ),
    "second_recovery_v3_result_sha256": (
        "6a76fae9bb970fdf11a9ae37cf785910cf6c270146ca7b752a6aedb678de3efd"
    ),
    "service_start_watchdog_seconds": 300,
    "snapshot_manifest_sha256": (
        "ea7a19bebd6cfd02d6f71846e0e46f2672ffff5e0ced607cc29e4ca1ce867f69"
    ),
    "source_association_manifest_sha256": (
        "28e002f1b5eaa76fe0b1b753c0ebcee6f3b186ba23bca355b1a40035ce8dbd11"
    ),
    "source_tree_sha256": EXPECTED_SOURCE_TREE_SHA256,
    "tokenizer_manifest_sha256": (
        "7717ed428d9b1d1791532c4714d022eff8c3cce10921bd0297f0a6c68c8992fc"
    ),
}
EXPECTED_PROCESS_MEMBERS = (
    {
        "comm": "python",
        "pid": 885_933,
        "ppid": 1,
        "process_group": 885_933,
        "raw_command_sha256": (
            "4e542a61b2d2882a2eb68d069862a0d405871761ffd36d0d0a7564d125ac82fe"
        ),
        "session_id": 885_933,
        "start_ticks": 557_670_858,
        "state": "S",
        "token_matches": True,
    },
    {
        "comm": "python",
        "pid": 887_673,
        "ppid": 885_933,
        "process_group": 885_933,
        "raw_command_sha256": (
            "321452557983e919c63d49cb97aba620d13105e050270390801e8cefbb6c06f4"
        ),
        "session_id": 885_933,
        "start_ticks": 557_683_180,
        "state": "S",
        "token_matches": True,
    },
    {
        "comm": "VLLM::EngineCor",
        "pid": 887_674,
        "ppid": 885_933,
        "process_group": 885_933,
        "raw_command_sha256": (
            "9731ae977a4f6e6eab9d8dd519c64fd8a5227b61b554f9adcc66b11716df93b4"
        ),
        "session_id": 885_933,
        "start_ticks": 557_683_181,
        "state": "S",
        "token_matches": False,
    },
)

EXPECTED_RUN_ROOT_BASENAMES = (
    ".checkpoint.json.orchestrator-guard-000001.json.closed.json",
    "checkpoint.json",
    "checkpoint.json.guardian-ready-000001.json",
    "checkpoint.json.guardian-terminal-request.json",
    "checkpoint.json.guardian-ticket.json",
    "checkpoint.json.guardian.lock",
    "checkpoint.json.guardian.log",
    "checkpoint.json.internal-controller-000001.json",
    "checkpoint.json.internal-controller-000002.json",
    "checkpoint.json.internal-controller-000003.json",
    "checkpoint.json.orchestrator-guard-000001.json",
    "checkpoint.json.orchestrator-invocation.json",
    "checkpoint.json.service",
    "fallback-qwen3-8b-awq-development-v8.vllm.log",
    "orchestrator.20260905T201854Z.log",
)
EXPECTED_ABSENT_BASENAMES = (
    "checkpoint.json.guardian-controller-takeover.json",
    "checkpoint.json.guardian-result.json",
)

_OPERATIONAL_LEDGER_TABLES = frozenset(
    {
        "gpu_allocation_journal",
        "gpu_events",
        "gpu_service_journal",
        "gpu_service_sessions",
        "resource_samples",
        "storage_samples",
    }
)


def _binding_matches(
    binding: SelfHashedFileBinding | OpaqueFileBinding,
    expected: tuple[str, int, str, str | None, str | None],
) -> bool:
    basename, size_bytes, file_sha256, kind, manifest_sha256 = expected
    if (
        binding.basename != basename
        or binding.size_bytes != size_bytes
        or binding.file_sha256 != file_sha256
    ):
        return False
    if isinstance(binding, SelfHashedFileBinding):
        return binding.kind == kind and binding.manifest_sha256 == manifest_sha256
    return kind is None and manifest_sha256 is None


class _StrictIncidentRecord(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        str_strip_whitespace=True,
        validate_default=True,
    )


class FallbackV8SourceAndAuthorization(_StrictIncidentRecord):
    source_revision: Literal["fallback-second-recovery-v8"]
    source_git_commit: GitRevision
    source_tree_sha256: Sha256Digest
    source_association: SelfHashedFileBinding
    authorization_overlay: SelfHashedFileBinding
    execution_preflight: SelfHashedFileBinding

    @model_validator(mode="after")
    def require_exact_artifact_identities(self) -> Self:
        if (
            self.source_git_commit != EXPECTED_SOURCE_GIT_COMMIT
            or self.source_tree_sha256 != EXPECTED_SOURCE_TREE_SHA256
            or any(
                not _binding_matches(getattr(self, name), expected)
                for name, expected in EXPECTED_SOURCE_BINDINGS.items()
            )
        ):
            raise ValueError("v8 source, authorization, or preflight identity changed")
        return self


class FallbackV8PublicEvidence(_StrictIncidentRecord):
    controller_handoff: SelfHashedFileBinding
    run_result: SelfHashedFileBinding
    orphan_cleanup: SelfHashedFileBinding

    @model_validator(mode="after")
    def require_exact_artifact_identities(self) -> Self:
        if any(
            not _binding_matches(getattr(self, name), expected)
            for name, expected in EXPECTED_PUBLIC_BINDINGS.items()
        ):
            raise ValueError("v8 public result identity changed")
        return self


class FallbackV8RestrictedEvidence(_StrictIncidentRecord):
    orchestration_invocation: SelfHashedFileBinding
    guardian_ticket: SelfHashedFileBinding
    guardian_ready: SelfHashedFileBinding
    guardian_terminal_request: SelfHashedFileBinding
    orchestrator_guard: SelfHashedFileBinding
    closed_orchestrator_guard: SelfHashedFileBinding
    prepare_controller_receipt: SelfHashedFileBinding
    run_controller_receipt: SelfHashedFileBinding
    cleanup_controller_receipt: SelfHashedFileBinding
    checkpoint: OpaqueFileBinding
    service_checkpoint: OpaqueFileBinding
    guardian_log: OpaqueFileBinding
    orchestrator_log: OpaqueFileBinding
    vllm_log: OpaqueFileBinding
    run_root_inventory_sha256: Literal[
        "2f714c01596522e47265dfff121dbfb43187ecd6de7ce100803b9ac090caa5e9"
    ]

    @model_validator(mode="after")
    def require_exact_restricted_bindings(self) -> Self:
        if any(
            not _binding_matches(getattr(self, name), expected)
            for name, expected in EXPECTED_RESTRICTED_BINDINGS.items()
        ):
            raise ValueError("v8 restricted evidence identity changed")
        return self


class FallbackV8ServiceLeaderIdentity(_StrictIncidentRecord):
    role: Literal["service_leader"]
    pid: Literal[885933]
    parent_pid: Literal[1]
    process_group_id: Literal[885933]
    session_id: Literal[885933]
    start_ticks: Literal[557670858]
    process_name: Literal["python"]
    process_state: Literal["S"]
    argv_sha256: Literal[
        "4e542a61b2d2882a2eb68d069862a0d405871761ffd36d0d0a7564d125ac82fe"
    ]


class FallbackV8WorkerIdentity(_StrictIncidentRecord):
    role: Literal["service_worker"]
    pid: Literal[887673]
    parent_pid: Literal[885933]
    process_group_id: Literal[885933]
    session_id: Literal[885933]
    start_ticks: Literal[557683180]
    process_name: Literal["python"]
    process_state: Literal["S"]
    argv_sha256: Literal[
        "321452557983e919c63d49cb97aba620d13105e050270390801e8cefbb6c06f4"
    ]


class FallbackV8EngineCoreIdentity(_StrictIncidentRecord):
    role: Literal["engine_core"]
    pid: Literal[887674]
    parent_pid: Literal[885933]
    process_group_id: Literal[885933]
    session_id: Literal[885933]
    start_ticks: Literal[557683181]
    process_name: Literal["VLLM::EngineCor"]
    process_state: Literal["S"]
    argv_sha256: Literal[
        "9731ae977a4f6e6eab9d8dd519c64fd8a5227b61b554f9adcc66b11716df93b4"
    ]


class FallbackV8ManualRecoveryEvidence(_StrictIncidentRecord):
    stop_intent: SelfHashedFileBinding
    stop_outcome: SelfHashedFileBinding
    accounting_receipt: SelfHashedFileBinding
    stop_utility: OpaqueFileBinding
    accounting_utility: OpaqueFileBinding
    pre_stop_lease_file_sha256: Literal[
        "50e1354f2734ac00ef06f639b9c58c43fd3cde86df9c73ea2f1f26cfbca00a62"
    ]
    pre_stop_lease_manifest_sha256: Literal[
        "0d3a9f8d57cf52568bc9016fb4990d5809d869f82b3d01d22b1a5febe90ec098"
    ]
    process_group_id: Literal[885933]
    process_members: tuple[
        FallbackV8ServiceLeaderIdentity,
        FallbackV8WorkerIdentity,
        FallbackV8EngineCoreIdentity,
    ]
    member_count: Literal[3]
    engine_core_token_matched: Literal[False]
    sole_gpu_process_memory_mib: Literal[21482]
    sole_port_owner_verified: Literal[True]
    signals_sent: tuple[Literal["SIGTERM"]]
    stopped_at: AwareDatetime
    physical_absence_verified: Literal[True]

    @model_validator(mode="after")
    def require_exact_recovery_artifacts(self) -> Self:
        if any(
            not _binding_matches(getattr(self, name), expected)
            for name, expected in EXPECTED_MANUAL_BINDINGS.items()
        ):
            raise ValueError("v8 manual recovery identity changed")
        if (
            self.pre_stop_lease_file_sha256 != EXPECTED_PRE_STOP_LEASE_FILE_SHA256
            or self.pre_stop_lease_manifest_sha256
            != EXPECTED_PRE_STOP_LEASE_MANIFEST_SHA256
            or tuple(member.pid for member in self.process_members)
            != (885_933, 887_673, 887_674)
            or any(
                member.process_group_id != self.process_group_id
                for member in self.process_members
            )
            or any(member.session_id != self.process_group_id for member in self.process_members)
            or self.signals_sent != ("SIGTERM",)
            or self.stopped_at
            != datetime.fromisoformat("2026-09-05T20:40:09.324832+00:00")
        ):
            raise ValueError("v8 manual recovery lease, process, or signal changed")
        return self


class FallbackV8LedgerSummary(_StrictIncidentRecord):
    total_allocated_microseconds: int = Field(ge=0, strict=True)
    event_count: int = Field(ge=0, strict=True)
    service_session_count: int = Field(ge=0, strict=True)
    attempt_count: int = Field(ge=0, strict=True)
    model_call_count: int = Field(ge=0, strict=True)
    artifact_count: int = Field(ge=0, strict=True)
    storage_sample_count: int = Field(ge=0, strict=True)
    resource_sample_count: int = Field(ge=0, strict=True)
    allocation_journal_count: int = Field(ge=0, strict=True)
    service_journal_count: int = Field(ge=0, strict=True)
    unresolved_gpu_allocation_count: int = Field(ge=0, strict=True)
    unresolved_gpu_service_count: int = Field(ge=0, strict=True)
    by_kind_microseconds: dict[str, int]

    @model_validator(mode="after")
    def reconcile_total(self) -> Self:
        if any(
            not key or isinstance(value, bool) or not isinstance(value, int) or value < 0
            for key, value in self.by_kind_microseconds.items()
        ):
            raise ValueError("GPU accounting kinds must be nonnegative integers")
        if self.total_allocated_microseconds != sum(self.by_kind_microseconds.values()):
            raise ValueError("GPU accounting kinds do not reconcile")
        return self


class FallbackV8LedgerBinding(_StrictIncidentRecord):
    basename: SafeBasename
    size_bytes: int = Field(gt=0, strict=True)
    file_sha256: Sha256Digest
    schema_sha256: Sha256Digest
    protected_content_sha256: Sha256Digest
    protected_table_count: Literal[20]
    summary: FallbackV8LedgerSummary


class FallbackV8StartupEvent(_StrictIncidentRecord):
    event_id: Literal["fallback-qwen3-8b-awq-development-v8-service-start-001"]
    event_kind: Literal["gpu_session_start"]
    allocated_microseconds: Literal[227686586]
    started_at: AwareDatetime
    ended_at: AwareDatetime
    succeeded: Literal[True]
    intended_event_kind: Literal["gpu_session_start"]
    admitted_maximum_microseconds: Literal[300000000]

    @model_validator(mode="after")
    def require_time_order(self) -> Self:
        if (
            self.started_at
            != datetime.fromisoformat("2026-09-05T20:22:15.867763+00:00")
            or self.ended_at
            != datetime.fromisoformat("2026-09-05T20:26:03.554356+00:00")
            or self.ended_at <= self.started_at
        ):
            raise ValueError("v8 startup event time order changed")
        return self


class FallbackV8ServiceAccounting(_StrictIncidentRecord):
    service_session_id: Literal[
        "fallback-qwen3-8b-awq-development-v8-service-start-001"
    ]
    session_id: Literal["fallback-qwen3-8b-awq-development-v8"]
    service_microseconds: Literal[1073416738]
    classified_event_microseconds: Literal[227686586]
    overhead_microseconds: Literal[845730152]
    started_at: AwareDatetime
    ended_at: AwareDatetime
    accounting_method: Literal["conservative_service_journal_recovery"]
    recovery_journal_sequence: Literal[55]
    recovery_journal_state: Literal["recovered"]

    @model_validator(mode="after")
    def reconcile_service(self) -> Self:
        if (
            self.service_microseconds
            != self.classified_event_microseconds + self.overhead_microseconds
            or self.started_at
            != datetime.fromisoformat("2026-09-05T20:22:15.908094+00:00")
            or self.ended_at
            != datetime.fromisoformat("2026-09-05T20:40:09.324832+00:00")
            or self.ended_at <= self.started_at
        ):
            raise ValueError("v8 service accounting does not reconcile")
        return self


class FallbackV8AccountingDelta(_StrictIncidentRecord):
    allocated_gpu_microseconds: Literal[1073416738]
    classified_startup_microseconds: Literal[227686586]
    conservative_service_overhead_microseconds: Literal[845730152]
    gpu_events: Literal[1]
    service_sessions: Literal[1]
    attempts: Literal[0]
    inference_model_calls: Literal[0]
    artifacts: Literal[0]
    storage_samples: Literal[10]
    resource_samples: Literal[6]
    allocation_journal_rows: Literal[47]
    service_journal_rows: Literal[56]
    accepted_outputs: Literal[0]


class FallbackV8Accounting(_StrictIncidentRecord):
    before_v8: FallbackV8LedgerBinding
    before_manual_recovery: FallbackV8LedgerBinding
    terminal: FallbackV8LedgerBinding
    startup_event: FallbackV8StartupEvent
    service: FallbackV8ServiceAccounting
    delta: FallbackV8AccountingDelta

    @model_validator(mode="after")
    def require_exact_three_state_reconciliation(self) -> Self:
        before = self.before_v8.summary
        failed = self.before_manual_recovery.summary
        terminal = self.terminal.summary
        bindings = (self.before_v8, self.before_manual_recovery, self.terminal)
        for name, binding in (
            ("before_v8", self.before_v8),
            ("before_manual_recovery", self.before_manual_recovery),
            ("terminal", self.terminal),
        ):
            expected_basename, expected_size, expected_file_sha256 = (
                EXPECTED_LEDGER_IDENTITIES[name]
            )
            if (
                binding.basename != expected_basename
                or binding.size_bytes != expected_size
                or binding.file_sha256 != expected_file_sha256
                or binding.summary.model_dump(mode="python")
                != EXPECTED_LEDGER_SUMMARIES[name]
            ):
                raise ValueError("v8 exact ledger binding or summary changed")
        if (
            before.total_allocated_microseconds != EXPECTED_BASELINE_MICROSECONDS
            or failed.total_allocated_microseconds != EXPECTED_PRE_RECOVERY_MICROSECONDS
            or terminal.total_allocated_microseconds != EXPECTED_TERMINAL_MICROSECONDS
            or terminal.total_allocated_microseconds - before.total_allocated_microseconds
            != EXPECTED_SERVICE_MICROSECONDS
            or any(binding.schema_sha256 != EXPECTED_SCHEMA_SHA256 for binding in bindings)
            or any(
                binding.protected_content_sha256 != EXPECTED_PROTECTED_CONTENT_SHA256
                for binding in bindings
            )
        ):
            raise ValueError("v8 ledger identities or cumulative accounting changed")
        for name in ("attempt_count", "model_call_count", "artifact_count"):
            if len({getattr(item, name) for item in (before, failed, terminal)}) != 1:
                raise ValueError("v8 changed an inference or artifact count")
        if (
            before.attempt_count != 1
            or before.model_call_count != 1
            or before.artifact_count != 17
            or failed.event_count - before.event_count != 1
            or terminal.event_count != failed.event_count
            or failed.service_session_count != before.service_session_count
            or terminal.service_session_count - failed.service_session_count != 1
            or terminal.storage_sample_count - before.storage_sample_count != 10
            or terminal.resource_sample_count - before.resource_sample_count != 6
            or terminal.allocation_journal_count - before.allocation_journal_count != 47
            or terminal.service_journal_count - before.service_journal_count != 56
            or failed.service_journal_count - before.service_journal_count != 55
            or terminal.service_journal_count - failed.service_journal_count != 1
            or any(
                item.unresolved_gpu_allocation_count != 0 for item in (before, failed, terminal)
            )
            or before.unresolved_gpu_service_count != 0
            or failed.unresolved_gpu_service_count != 1
            or terminal.unresolved_gpu_service_count != 0
        ):
            raise ValueError("v8 exact ledger delta changed")
        return self


class FallbackV8ResourceMaxima(_StrictIncidentRecord):
    sample_count: Literal[6]
    process_ram_bytes: Literal[2972696576]
    gpu_vram_bytes: Literal[22525509632]
    project_storage_bytes: Literal[10344769024]
    cpu_worker_count: Literal[8]
    all_registered_limits_satisfied: Literal[True]


class FallbackV8FailureSequence(_StrictIncidentRecord):
    safe_error_class: Literal[
        "engine_core_service_instance_token_nonmatch_blocked_exact_group_adoption"
    ]
    root_cause: Literal[
        "vLLM EngineCore did not retain a matching private service-instance token, "
        "so exact process-group adoption and automated shutdown failed closed"
    ]
    root_cause_basis: Literal["source_bound_guardian_trace_and_exact_pre_stop_inventory"]
    failure_stage: Literal["post_handoff_service_adoption_before_micro_pilot_inference"]
    endpoint_healthy_before_failure: Literal[True]
    endpoint_healthy_at: AwareDatetime
    controller_handoff_completed: Literal[True]
    prepare_return_code: Literal[0]
    run_return_code: Literal[2]
    cleanup_return_code: Literal[2]
    guardian_outer_failure_type: Literal["RuntimeError"]
    adoption_inner_failure_type: Literal["RuntimeConfigurationError"]
    scientific_inference_reached: Literal[False]
    command_line_public: Literal[False]
    full_log_text_public: Literal[False]
    remote_absolute_paths_public: Literal[False]

    @model_validator(mode="after")
    def require_exact_endpoint_time(self) -> Self:
        if self.endpoint_healthy_at != datetime.fromisoformat(
            "2026-09-05T20:25:52+00:00"
        ):
            raise ValueError("v8 endpoint time changed")
        return self


class FallbackV8AbsenceInventory(_StrictIncidentRecord):
    observed_run_root_basenames: tuple[SafeBasename, ...]
    absent_basenames: tuple[SafeBasename, ...]
    all_required_paths_absent: Literal[True]

    @model_validator(mode="after")
    def require_exact_inventory(self) -> Self:
        if (
            self.observed_run_root_basenames != EXPECTED_RUN_ROOT_BASENAMES
            or self.absent_basenames != EXPECTED_ABSENT_BASENAMES
        ):
            raise ValueError("v8 run-root inventory changed")
        return self


class FallbackV8TerminalState(_StrictIncidentRecord):
    public_controller_handoff_present: Literal[True]
    public_failed_result_present: Literal[True]
    public_cleanup_result_present: Literal[True]
    accepted_output_count: Literal[0]
    inference_attempt_count_delta: Literal[0]
    inference_model_call_count_delta: Literal[0]
    phase1_gate_passed: Literal[False]
    service_checkpoint_present: Literal[True]
    guardian_result_present: Literal[False]
    automated_cleanup_succeeded: Literal[False]
    manual_physical_stop_verified: Literal[True]
    endpoint_live: Literal[False]
    gpu_process_live: Literal[False]
    exact_service_process_live: Literal[False]
    unresolved_gpu_allocation_count: Literal[0]
    unresolved_gpu_service_count: Literal[0]
    terminal_ledger_verified: Literal[True]
    resume_allowed: Literal[False]
    source_bound_v8_resume_permitted: Literal[False]
    fresh_repaired_source_required: Literal[True]


class FallbackV8RuntimeIncident(_StrictIncidentRecord):
    """Self-hashed public description of the exact fallback-v8 incident."""

    schema_version: Literal["1.0.0"] = "1.0.0"
    kind: Literal["fallback_gpu_acceptance_v8_runtime_incident"]
    run_id: Literal["fallback-qwen3-8b-awq-development-v8"]
    audited_at: AwareDatetime
    source_and_authorization: FallbackV8SourceAndAuthorization
    public_evidence: FallbackV8PublicEvidence
    restricted_evidence: FallbackV8RestrictedEvidence
    manual_recovery: FallbackV8ManualRecoveryEvidence
    accounting: FallbackV8Accounting
    resource_maxima: FallbackV8ResourceMaxima
    failure_sequence: FallbackV8FailureSequence
    absence_inventory: FallbackV8AbsenceInventory
    terminal_state: FallbackV8TerminalState
    manifest_sha256: Sha256Digest

    @model_validator(mode="after")
    def verify_manifest_and_terminal_time(self) -> Self:
        if (
            self.audited_at != datetime.fromisoformat(EXPECTED_AUDITED_AT)
            or self.audited_at < self.manual_recovery.stopped_at
            or self.manual_recovery.stopped_at != self.accounting.service.ended_at
            or self.failure_sequence.endpoint_healthy_at >= self.accounting.service.ended_at
            or self.manual_recovery.process_group_id != 885_933
            or self.restricted_evidence.service_checkpoint.file_sha256
            != EXPECTED_RESTRICTED_BINDINGS["service_checkpoint"][2]
        ):
            raise ValueError("v8 exact audit, process, or terminal chronology changed")
        immutable = self.model_dump(mode="python", exclude={"manifest_sha256"})
        if self.manifest_sha256 != _canonical_sha256(immutable):
            raise ValueError("fallback-v8 incident manifest hash changed")
        return self


def _sha256_file(path: Path, *, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(chunk_size):
            digest.update(block)
    return digest.hexdigest()


def _require_no_symlink_ancestry(path: Path, *, label: str) -> None:
    supplied = Path(path).absolute()
    for component in (supplied, *supplied.parents):
        if component.is_symlink():
            raise ValueError(f"{label} cannot have symlink ancestry")


def _require_regular_file(path: Path, *, label: str) -> Path:
    supplied = Path(path).absolute()
    _require_no_symlink_ancestry(supplied, label=label)
    try:
        resolved = supplied.resolve(strict=True)
    except OSError as exc:
        raise ValueError(f"{label} does not exist") from exc
    if not stat.S_ISREG(resolved.stat().st_mode):
        raise ValueError(f"{label} must be a regular file")
    return resolved


def _load_json(path: Path, *, label: str) -> tuple[Path, dict[str, Any]]:
    resolved = _require_regular_file(path, label=label)
    try:
        value = json.loads(resolved.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{label} is not valid UTF-8 JSON") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{label} must contain one JSON object")
    return resolved, value


def _load_self_hashed_json(
    path: Path, *, label: str, expected_kind: str
) -> tuple[Path, dict[str, Any]]:
    resolved, value = _load_json(path, label=label)
    supplied = value.get("manifest_sha256")
    immutable = {key: item for key, item in value.items() if key != "manifest_sha256"}
    if (
        value.get("kind") != expected_kind
        or not isinstance(supplied, str)
        or supplied != _canonical_sha256(immutable)
    ):
        raise ValueError(f"{label} has an invalid kind or canonical self-hash")
    return resolved, value


def _self_hashed_binding(path: Path, value: Mapping[str, Any]) -> SelfHashedFileBinding:
    return SelfHashedFileBinding(
        basename=path.name,
        size_bytes=path.stat().st_size,
        file_sha256=_sha256_file(path),
        kind=value["kind"],
        manifest_sha256=value["manifest_sha256"],
    )


def _opaque_binding(path: Path, *, label: str) -> OpaqueFileBinding:
    resolved = _require_regular_file(path, label=label)
    if resolved.stat().st_size <= 0:
        raise ValueError(f"{label} cannot be empty")
    return OpaqueFileBinding(
        basename=resolved.name,
        size_bytes=resolved.stat().st_size,
        file_sha256=_sha256_file(resolved),
    )


def _typed_sql_value(value: object) -> dict[str, object]:
    if value is None:
        return {"type": "null", "value": None}
    if isinstance(value, bytes):
        return {"type": "blob", "value": value.hex()}
    if isinstance(value, str):
        return {"type": "text", "value": value}
    if isinstance(value, int):
        return {"type": "integer", "value": value}
    if isinstance(value, float) and math.isfinite(value):
        return {"type": "real", "value": value.hex()}
    raise ValueError("ledger contains an unsupported SQLite value")


def _quoted_identifier(value: str) -> str:
    return '"' + value.replace('"', '""') + '"'


def _protected_ledger_fingerprints(path: Path) -> tuple[str, str, int]:
    connection = sqlite3.connect(f"{path.as_uri()}?mode=ro&immutable=1", uri=True)
    try:
        objects = connection.execute(
            "SELECT type, name, tbl_name, sql FROM sqlite_master "
            "WHERE name NOT LIKE 'sqlite_%' ORDER BY type, name"
        ).fetchall()
        schema_payload = [
            {"type": row[0], "name": row[1], "table": row[2], "sql": row[3]}
            for row in objects
        ]
        tables = sorted(
            row[1]
            for row in objects
            if row[0] == "table" and row[1] not in _OPERATIONAL_LEDGER_TABLES
        )
        protected_payload: list[dict[str, object]] = []
        for table in tables:
            columns = [
                row[1]
                for row in connection.execute(
                    f"PRAGMA table_info({_quoted_identifier(table)})"
                ).fetchall()
            ]
            rows = [
                [_typed_sql_value(value) for value in row]
                for row in connection.execute(
                    f"SELECT * FROM {_quoted_identifier(table)}"
                ).fetchall()
            ]
            rows.sort(key=canonical_json)
            protected_payload.append({"table": table, "columns": columns, "rows": rows})
        return (
            _canonical_sha256(schema_payload),
            _canonical_sha256(protected_payload),
            len(protected_payload),
        )
    finally:
        connection.close()


def _ledger_binding(path: Path, *, label: str) -> FallbackV8LedgerBinding:
    resolved = _require_regular_file(path, label=label)
    sidecars = tuple(Path(f"{resolved}{suffix}") for suffix in ("-wal", "-shm", "-journal"))
    sidecar_state: tuple[tuple[int, str] | None, ...] = tuple(
        None
        if not item.exists()
        else (item.stat().st_size, _sha256_file(_require_regular_file(item, label=label)))
        for item in sidecars
    )
    if any(
        state is not None and state[0] > 0
        for state, suffix in zip(sidecar_state, ("-wal", "-shm", "-journal"), strict=True)
        if suffix != "-shm"
    ):
        raise ValueError(f"{label} must be closed and checkpointed")
    size_before = resolved.stat().st_size
    hash_before = _sha256_file(resolved)
    with ReadOnlyLedger(resolved) as ledger:
        gpu = ledger.gpu_summary()
        summary = FallbackV8LedgerSummary(
            total_allocated_microseconds=gpu.total_allocated_microseconds,
            event_count=gpu.event_count,
            service_session_count=gpu.service_session_count,
            attempt_count=ledger.count_rows("attempts"),
            model_call_count=ledger.count_rows("model_calls"),
            artifact_count=ledger.count_rows("artifacts"),
            storage_sample_count=ledger.count_rows("storage_samples"),
            resource_sample_count=ledger.count_rows("resource_samples"),
            allocation_journal_count=ledger.count_rows("gpu_allocation_journal"),
            service_journal_count=ledger.count_rows("gpu_service_journal"),
            unresolved_gpu_allocation_count=len(ledger.unresolved_gpu_allocations()),
            unresolved_gpu_service_count=len(ledger.unresolved_gpu_service_journals()),
            by_kind_microseconds={
                kind.value: value for kind, value in gpu.by_kind_microseconds
            },
        )
    schema_hash, protected_hash, protected_count = _protected_ledger_fingerprints(resolved)
    if (
        resolved.stat().st_size != size_before
        or _sha256_file(resolved) != hash_before
        or tuple(
            None
            if not item.exists()
            else (item.stat().st_size, _sha256_file(_require_regular_file(item, label=label)))
            for item in sidecars
        )
        != sidecar_state
    ):
        raise ValueError(f"{label} changed while it was read")
    return FallbackV8LedgerBinding(
        basename=resolved.name,
        size_bytes=size_before,
        file_sha256=hash_before,
        schema_sha256=schema_hash,
        protected_content_sha256=protected_hash,
        protected_table_count=protected_count,
        summary=summary,
    )


def _microseconds(value: object, *, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or value < 0:
        raise ValueError(f"{label} must be nonnegative")
    result = round(float(value) * 1_000_000)
    if abs(float(value) - result / 1_000_000) > 1e-9:
        raise ValueError(f"{label} is not exact to microsecond precision")
    return result


def _mapping(value: object, *, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be an object")
    return value


def _aware_datetime(value: object, *, label: str) -> datetime:
    if not isinstance(value, str):
        raise ValueError(f"{label} must be an ISO-8601 timestamp")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError(f"{label} must be an ISO-8601 timestamp") from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError(f"{label} must be timezone-aware")
    return parsed


def _require_source_chain(
    source_path: Path,
    source: Mapping[str, Any],
    overlay: Mapping[str, Any],
    preflight: Mapping[str, Any],
    before: FallbackV8LedgerBinding,
) -> None:
    source_hash = source.get("manifest_sha256")
    overlay_hash = overlay.get("manifest_sha256")
    overlay_source = _mapping(overlay.get("source"), label="v8 overlay source")
    authorization = _mapping(overlay.get("authorization"), label="v8 authorization")
    accounting = {
        "total_allocated_microseconds": before.summary.total_allocated_microseconds,
        "event_count": before.summary.event_count,
        "service_session_count": before.summary.service_session_count,
        "by_kind_microseconds": before.summary.by_kind_microseconds,
    }
    if (
        source.get("revision_label") != FALLBACK_V8_SOURCE_REVISION
        or source.get("git_commit") != EXPECTED_SOURCE_GIT_COMMIT
        or source.get("local_tree_sha256") != source.get("remote_tree_sha256")
        or source.get("local_tree_sha256") != EXPECTED_SOURCE_TREE_SHA256
        or source.get("local_manifest_file_sha256")
        != "806b237d29b78809f6a796c9c9671ec0c0b18e91a2571d289313fe02faae5150"
        or source.get("remote_manifest_file_sha256")
        != "806b237d29b78809f6a796c9c9671ec0c0b18e91a2571d289313fe02faae5150"
        or overlay.get("schema_version") != "1.6.0"
        or overlay.get("authorized_recovery_run_id") != FALLBACK_V8_RUN_ID
        or authorization.get("status") != "authorized"
        or overlay_source.get("current_association_manifest_sha256") != source_hash
        or overlay_source.get("current_association_file_sha256") != _sha256_file(source_path)
        or overlay_source.get("current_tree_sha256") != source.get("local_tree_sha256")
        or overlay.get("cumulative_gpu_accounting") != accounting
        or preflight.get("run_id") != FALLBACK_V8_RUN_ID
        or preflight.get("source_association_sha256") != source_hash
        or preflight.get("source_tree_sha256") != source.get("local_tree_sha256")
        or preflight.get("second_recovery_overlay_sha256") != overlay_hash
        or preflight.get("gpu_accounting_before_start") != accounting
        or preflight.get("execution_authorized") is not True
        or preflight.get("passed") is not True
        or preflight.get("gpu_allocation_performed") is not False
        or preflight.get("model_process_started") is not False
        or preflight.get("checkpoint_absent") is not True
    ):
        raise ValueError("v8 source, authorization, preflight, or baseline changed")


def _require_control_chain(
    *,
    invocation: Mapping[str, Any],
    ticket: Mapping[str, Any],
    ready: Mapping[str, Any],
    terminal_request: Mapping[str, Any],
    guard: Mapping[str, Any],
    closed: Mapping[str, Any],
    receipts: tuple[Mapping[str, Any], ...],
    checkpoint: Mapping[str, Any],
    service_checkpoint: Mapping[str, Any],
    handoff: Mapping[str, Any],
    result: Mapping[str, Any],
    cleanup: Mapping[str, Any],
    result_path: Path,
    before: FallbackV8LedgerBinding,
    startup_event: FallbackV8StartupEvent,
    service: FallbackV8ServiceAccounting,
) -> None:
    invocation_hash = invocation.get("manifest_sha256")
    ticket_hash = ticket.get("manifest_sha256")
    guard_hash = guard.get("manifest_sha256")
    arguments_hash = invocation.get("execution_arguments_sha256")
    invocation_at = _aware_datetime(invocation.get("created_at"), label="v8 invocation time")
    guard_at = _aware_datetime(guard.get("started_at"), label="v8 guard time")
    ready_at = _aware_datetime(ready.get("ready_at"), label="v8 guardian ready time")
    if (
        invocation.get("run_id") != FALLBACK_V8_RUN_ID
        or invocation.get("service_session_id") != FALLBACK_V8_RUN_ID
        or invocation.get("service_event_id") != FALLBACK_V8_SERVICE_EVENT_ID
        or invocation_hash
        != "b085244d3a3ca1dcc77a3bed11399d42a62b63696fa5ee9226e9dd4a4fb5c1d3"
        or arguments_hash != EXPECTED_EXECUTION_ARGUMENTS_SHA256
        or invocation_at
        != datetime.fromisoformat("2026-09-05T20:18:57.192077+00:00")
        or Path(str(invocation.get("handoff_output"))).name
        != EXPECTED_PUBLIC_BINDINGS["controller_handoff"][0]
        or Path(str(invocation.get("result_output"))).name
        != EXPECTED_PUBLIC_BINDINGS["run_result"][0]
        or Path(str(invocation.get("cleanup_output"))).name
        != EXPECTED_PUBLIC_BINDINGS["orphan_cleanup"][0]
        or result_path.name != EXPECTED_PUBLIC_BINDINGS["run_result"][0]
        or _microseconds(
            invocation.get("gpu_seconds_before_invocation"), label="v8 invocation accounting"
        )
        != before.summary.total_allocated_microseconds
        or ticket.get("run_id") != FALLBACK_V8_RUN_ID
        or ticket.get("orchestration_invocation_sha256") != invocation_hash
        or ticket.get("execution_arguments_sha256") != arguments_hash
        or ticket.get("service_session_id") != FALLBACK_V8_RUN_ID
        or ticket.get("service_event_id") != FALLBACK_V8_SERVICE_EVENT_ID
        or ready.get("run_id") != FALLBACK_V8_RUN_ID
        or ready.get("guardian_ticket_sha256") != ticket_hash
        or ready.get("orchestration_invocation_sha256") != invocation_hash
        or ready.get("guardian_pid") != 884_711
        or ready.get("guardian_process_group_id") != 884_711
        or ready.get("guardian_session_id") != 884_711
        or ready.get("guardian_start_ticks") != 557_650_990
        or ready.get("guardian_command_sha256")
        != "2baf5fc55075385bac0a6abc51ee6f1c70d78a8195a4d3bb05c3638b74a771e5"
        or ready.get("guardian_process_command_sha256")
        != "72a6521a8b17bfdf45db5d4cec07d457449f2caf3e4e9ac3db72ac05572891ee"
        or ready_at
        != datetime.fromisoformat("2026-09-05T20:20:38.647461+00:00")
        or guard.get("run_id") != FALLBACK_V8_RUN_ID
        or guard.get("sequence") != 1
        or guard.get("state") != "active"
        or guard.get("previous_guard_sha256") is not None
        or guard.get("guardian_ticket_sha256") != ticket_hash
        or guard.get("orchestration_invocation_sha256") != invocation_hash
        or guard.get("execution_arguments_sha256") != arguments_hash
        or guard.get("orchestrator_pid") != 884_693
        or guard.get("orchestrator_process_group_id") != 884_693
        or guard.get("orchestrator_session_id") != 884_693
        or guard.get("orchestrator_start_ticks") != 557_650_688
        or guard.get("orchestrator_command_sha256")
        != "fce738411d53b116d367e9e7dd9e44b4a89c99465cf83e9601f85f9735c21259"
        or guard_at
        != datetime.fromisoformat("2026-09-05T20:18:57.280081+00:00")
        or not invocation_at < guard_at < ready_at
    ):
        raise ValueError("v8 invocation, guardian, or guard chain changed")
    expected_controllers = (
        {
            "stage": "prepare",
            "pid": 885_318,
            "start_ticks": 557_661_133,
            "registered_at": "2026-09-05T20:20:40.554407+00:00",
            "output": EXPECTED_PUBLIC_BINDINGS["controller_handoff"][0],
            "command_sha256": (
                "12290419d44602bbc0675df5df611e8d1184dd6b05b58d7d599d9d78113b1445"
            ),
            "manifest_sha256": (
                "e42b78737643c34f3bd911e3e9cc78d380e202119b8627aad5bdfcb1b5b172f2"
            ),
        },
        {
            "stage": "run",
            "pid": 889_386,
            "start_ticks": 557_697_951,
            "registered_at": "2026-09-05T20:26:48.629761+00:00",
            "output": EXPECTED_PUBLIC_BINDINGS["run_result"][0],
            "command_sha256": (
                "e5350dfdacdfcc59da3daf8f5da3494496b61d00b499be6f56aad698f8d2690e"
            ),
            "manifest_sha256": (
                "e68bd194e52da8786588f2bd259c65506385b7f706388144e20ce3a516043389"
            ),
        },
        {
            "stage": "cleanup",
            "pid": 891_513,
            "start_ticks": 557_712_871,
            "registered_at": "2026-09-05T20:29:18.394466+00:00",
            "output": EXPECTED_PUBLIC_BINDINGS["orphan_cleanup"][0],
            "command_sha256": (
                "d1a758003fab9be2adb5d113ee3dace57946bc3db5ee717129bda90fbcdeae51"
            ),
            "manifest_sha256": (
                "cceae4462a8689add1a070396121d9e17f12916aae9a62bb381a3168dd6d6c5f"
            ),
        },
    )
    controller_times: list[datetime] = []
    previous: object = None
    for sequence, (receipt, expected) in enumerate(
        zip(receipts, expected_controllers, strict=True), start=1
    ):
        registered_at = _aware_datetime(
            receipt.get("registered_at"), label=f"v8 controller {sequence} time"
        )
        controller_times.append(registered_at)
        if (
            receipt.get("run_id") != FALLBACK_V8_RUN_ID
            or receipt.get("sequence") != sequence
            or receipt.get("controller_stage") != expected["stage"]
            or receipt.get("previous_controller_receipt_sha256") != previous
            or receipt.get("execution_arguments_sha256") != arguments_hash
            or receipt.get("orchestration_invocation_sha256") != invocation_hash
            or receipt.get("orchestrator_guard_sha256") != guard_hash
            or receipt.get("controller_pid") != expected["pid"]
            or receipt.get("controller_process_group_id") != 884_693
            or receipt.get("controller_session_id") != 884_693
            or receipt.get("controller_start_ticks") != expected["start_ticks"]
            or receipt.get("controller_command_sha256") != expected["command_sha256"]
            or Path(str(receipt.get("controller_output"))).name != expected["output"]
            or receipt.get("manifest_sha256") != expected["manifest_sha256"]
            or registered_at != datetime.fromisoformat(str(expected["registered_at"]))
        ):
            raise ValueError("v8 controller receipt chain changed")
        previous = receipt.get("manifest_sha256")
    if len({item.get("controller_pid") for item in receipts}) != 3:
        raise ValueError("v8 controller PIDs are not distinct")
    terminal_request_at = _aware_datetime(
        terminal_request.get("requested_at"), label="v8 terminal request time"
    )
    closed_at = _aware_datetime(closed.get("closed_at"), label="v8 closed guard time")
    if (
        terminal_request.get("run_id") != FALLBACK_V8_RUN_ID
        or terminal_request.get("guardian_ticket_sha256") != ticket_hash
        or terminal_request.get("orchestration_invocation_sha256") != invocation_hash
        or terminal_request.get("orchestrator_guard_sha256") != guard_hash
        or terminal_request.get("controller_result_sha256") is not None
        or terminal_request_at
        != datetime.fromisoformat("2026-09-05T20:31:04.579099+00:00")
        or closed.get("run_id") != FALLBACK_V8_RUN_ID
        or closed.get("orchestrator_guard_sha256") != guard_hash
        or closed.get("prepare_return_code") != 0
        or closed.get("run_return_code") != 2
        or closed.get("cleanup_return_code") != 2
        or closed.get("guardian_result_sha256") is not None
        or closed.get("physical_shutdown_verified") is not False
        or closed_at
        != datetime.fromisoformat("2026-09-05T20:32:34.613186+00:00")
        or not (
            ready_at
            < controller_times[0]
            < startup_event.started_at
            < service.started_at
            < startup_event.ended_at
            < controller_times[1]
            < controller_times[2]
            < terminal_request_at
            < closed_at
            < service.ended_at
        )
    ):
        raise ValueError("v8 terminal orchestration chain changed")
    exact_checkpoint = {
        "run_id": FALLBACK_V8_RUN_ID,
        "service_start_attempted": True,
        "controller_handoff_complete": True,
        "active_attempt": None,
        "active_call_id": None,
        "completed_call_ids": [],
        "accepted_outputs": {},
        "orphan_cleanup_completed": False,
        "development_continuation_completed": False,
        "accepted_micro_pilot_result": None,
        "micro_pilot_acceptance_receipt": None,
        "execution_hash": EXPECTED_EXECUTION_SHA256,
        "handoff_service_pid": 885_933,
        "stage_one_controller_pid": 885_318,
        "stage_two_controller_pid": None,
        "development_adopter_registration_hash": (
            EXPECTED_EXECUTION_IDENTITY["development_adopter_registration_hash"]
        ),
    }
    if any(checkpoint.get(key) != value for key, value in exact_checkpoint.items()):
        raise ValueError("v8 terminal checkpoint changed")
    if (
        service_checkpoint.get("session_id") != FALLBACK_V8_RUN_ID
        or service_checkpoint.get("accounting_session_id") != FALLBACK_V8_SERVICE_EVENT_ID
        or service_checkpoint.get("controller_restart_handoff") is not True
        or service_checkpoint.get("pid") != 885_933
        or service_checkpoint.get("pid") != checkpoint.get("handoff_service_pid")
        or service_checkpoint.get("process_group_id") != 885_933
        or service_checkpoint.get("process_session_id") != 885_933
        or service_checkpoint.get("process_start_ticks") != 557_670_858
        or service_checkpoint.get("controller_pid") != 885_318
        or service_checkpoint.get("controller_pid")
        != receipts[0].get("controller_pid")
        or service_checkpoint.get("configuration_hash")
        != EXPECTED_EXECUTION_IDENTITY["launcher_configuration_sha256"]
        or service_checkpoint.get("process_command_sha256")
        != "122fef78fbd6c9bf3d8f602a5360ed6109ac98c2fcdda536ab87ef62e191af48"
        or _aware_datetime(
            service_checkpoint.get("recorded_at"), label="v8 service checkpoint time"
        )
        != datetime.fromisoformat("2026-09-05T20:26:45.981521+00:00")
        or _aware_datetime(
            service_checkpoint.get("session_started_at"),
            label="v8 service checkpoint start time",
        )
        != service.started_at
    ):
        raise ValueError("v8 service checkpoint identity changed")
    for value, unbound_sha256 in zip(
        (handoff, result, cleanup),
        (
            "574387053da78102689fcad3517b2c431de52bdc509d1d12aa0ddb3e5561b8e4",
            "e983708e19256aa4017fd3288b5dd01ac00a8e3ec3b3170e81cc7b2f9b097e14",
            "77fbfe87c86cd0586b4e8c819f806f13a78d3374ae03f2001b49d328ca4be18f",
        ),
        strict=True,
    ):
        if (
            value.get("execution_identity") != EXPECTED_EXECUTION_IDENTITY
            or value.get("execution_hash") != EXPECTED_EXECUTION_SHA256
            or value.get("execution_arguments_sha256")
            != EXPECTED_EXECUTION_ARGUMENTS_SHA256
            or value.get("orchestration_invocation_sha256") != invocation_hash
            or value.get("orchestrator_guard_sha256") != guard_hash
            or value.get("unbound_controller_result_sha256") != unbound_sha256
        ):
            raise ValueError("v8 public execution identity changed")
    if (
        handoff.get("run_id") != FALLBACK_V8_RUN_ID
        or handoff.get("orchestration_controller_stage") != "prepare"
        or handoff.get("controller_stage") != "prepare_complete"
        or handoff.get("next_required_stage") != "run_under_a_different_controller_pid"
        or handoff.get("model_service_left_live_for_controller_restart") is not True
        or handoff.get("physical_service_live") is not True
        or handoff.get("controller_process_receipt_sha256")
        != receipts[0].get("manifest_sha256")
    ):
        raise ValueError("v8 controller handoff changed")
    for value, stage, receipt in zip(
        (result, cleanup), ("run", "cleanup"), receipts[1:], strict=True
    ):
        if (
            value.get("run_id") != FALLBACK_V8_RUN_ID
            or value.get("orchestration_controller_stage") != stage
            or value.get("failure_type") != "RuntimeError"
            or value.get("failure_stage") != "fallback_micro_pilot"
            or value.get("micro_pilot_passed") is not False
            or value.get("phase1_gate_passed") is not False
            or value.get("gate_passed") is not False
            or value.get("normal_acceptance_block_executed") is not False
            or value.get("completed_base_call_count") != 0
            or value.get("completed_call_ids") != []
            or value.get("accepted_micro_pilot_result") is not None
            or value.get("physical_service_state_unverified") is not True
            or value.get("vllm_service_stopped") is not False
            or value.get("controller_process_receipt_sha256")
            != receipt.get("manifest_sha256")
            or value.get("execution_arguments_sha256") != arguments_hash
            or value.get("orchestration_invocation_sha256") != invocation_hash
            or value.get("orchestrator_guard_sha256") != guard_hash
        ):
            raise ValueError(f"v8 {stage} failure result changed")


def _extract_accounting(
    ledger_path: Path,
) -> tuple[FallbackV8StartupEvent, FallbackV8ServiceAccounting, FallbackV8ResourceMaxima]:
    with ReadOnlyLedger(ledger_path) as ledger:
        events = [
            item for item in ledger.gpu_events() if item.event_id == FALLBACK_V8_SERVICE_EVENT_ID
        ]
        services = [
            item
            for item in ledger.gpu_service_sessions()
            if item.service_session_id == FALLBACK_V8_SERVICE_EVENT_ID
        ]
        journals = [
            item
            for item in ledger.gpu_service_journal_records()
            if item.service_session_id == FALLBACK_V8_SERVICE_EVENT_ID
        ]
    connection = sqlite3.connect(
        f"{ledger_path.as_uri()}?mode=ro&immutable=1",
        uri=True,
    )
    try:
        samples = connection.execute(
            "SELECT sample_id, process_ram_bytes, gpu_vram_bytes, "
            "project_storage_bytes, cpu_worker_count FROM resource_samples "
            "WHERE sample_id LIKE ? ORDER BY sampled_at, sample_id",
            (f"{FALLBACK_V8_RUN_ID}%",),
        ).fetchall()
    finally:
        connection.close()
    if len(events) != 1 or len(services) != 1 or not journals or len(samples) != 6:
        raise ValueError("v8 terminal ledger lacks its exact runtime rows")
    event = events[0]
    service = services[0]
    journal = journals[-1]
    try:
        event_details = json.loads(event.details_json)
        service_details = json.loads(service.details_json)
    except json.JSONDecodeError as exc:
        raise ValueError("v8 accounting details are invalid JSON") from exc
    if (
        event.event_kind is not GpuEventKind.GPU_SESSION_START
        or event.allocated_microseconds != EXPECTED_CLASSIFIED_MICROSECONDS
        or event.succeeded is not True
        or event_details.get("intended_event_kind") != "gpu_session_start"
        or _microseconds(
            event_details.get("admitted_maximum_seconds"), label="v8 startup watchdog"
        )
        != 300_000_000
        or service.service_microseconds != EXPECTED_SERVICE_MICROSECONDS
        or service.classified_event_microseconds != EXPECTED_CLASSIFIED_MICROSECONDS
        or service.overhead_microseconds != EXPECTED_OVERHEAD_MICROSECONDS
        or service_details.get("accounting_method")
        != "conservative_service_journal_recovery"
        or journal.sequence != 55
        or journal.state.value != "recovered"
        or journal.elapsed_microseconds != EXPECTED_SERVICE_MICROSECONDS
    ):
        raise ValueError("v8 startup event or terminal service accounting changed")
    event_record = FallbackV8StartupEvent(
        event_id=event.event_id,
        event_kind=event.event_kind.value,
        allocated_microseconds=event.allocated_microseconds,
        started_at=event.started_at,
        ended_at=event.ended_at,
        succeeded=event.succeeded,
        intended_event_kind=event_details["intended_event_kind"],
        admitted_maximum_microseconds=300_000_000,
    )
    service_record = FallbackV8ServiceAccounting(
        service_session_id=service.service_session_id,
        session_id=service.session_id,
        service_microseconds=service.service_microseconds,
        classified_event_microseconds=service.classified_event_microseconds,
        overhead_microseconds=service.overhead_microseconds,
        started_at=service.started_at,
        ended_at=service.ended_at,
        accounting_method=service_details["accounting_method"],
        recovery_journal_sequence=journal.sequence,
        recovery_journal_state=journal.state.value,
    )
    maxima = FallbackV8ResourceMaxima(
        sample_count=len(samples),
        process_ram_bytes=max(item[1] for item in samples),
        gpu_vram_bytes=max(item[2] for item in samples),
        project_storage_bytes=max(item[3] for item in samples),
        cpu_worker_count=max(item[4] for item in samples),
        all_registered_limits_satisfied=True,
    )
    return event_record, service_record, maxima


def _require_manual_recovery_chain(
    *,
    intent_path: Path,
    intent: Mapping[str, Any],
    outcome_path: Path,
    outcome: Mapping[str, Any],
    receipt: Mapping[str, Any],
    stop_utility_path: Path,
    accounting_utility_path: Path,
    before_recovery: FallbackV8LedgerBinding,
    terminal: FallbackV8LedgerBinding,
    checkpoint: Mapping[str, Any],
    service_checkpoint: Mapping[str, Any],
    service: FallbackV8ServiceAccounting,
) -> tuple[int, datetime]:
    members = intent.get("member_identities")
    group = intent.get("process_group_id")
    if not isinstance(members, list) or len(members) != 3 or not isinstance(group, int):
        raise ValueError("v8 exact pre-stop process inventory changed")
    engine = [
        item
        for item in members
        # Linux ``comm`` is limited to 15 visible bytes; the vLLM log binds
        # this truncated kernel name to the EngineCore process.
        if isinstance(item, Mapping) and item.get("comm") == "VLLM::EngineCor"
    ]
    token_matches = [item.get("token_matches") for item in members if isinstance(item, Mapping)]
    intent_at = _aware_datetime(intent.get("recorded_at"), label="v8 stop intent time")
    if (
        intent.get("run_id") != FALLBACK_V8_RUN_ID
        or intent.get("service_event_id") != FALLBACK_V8_SERVICE_EVENT_ID
        or intent.get("process_group_id") != 885_933
        or tuple(members) != EXPECTED_PROCESS_MEMBERS
        or intent.get("lease_file_sha256") != EXPECTED_PRE_STOP_LEASE_FILE_SHA256
        or intent.get("lease_manifest_sha256")
        != EXPECTED_PRE_STOP_LEASE_MANIFEST_SHA256
        or intent_at != datetime.fromisoformat("2026-09-05T20:40:08.002465+00:00")
        or intent.get("sole_port_8000_owner_verified") is not True
        or len(engine) != 1
        or engine[0].get("token_matches") is not False
        or token_matches.count(False) != 1
        or token_matches.count(True) != 2
        or any(
            item.get("process_group") != group or item.get("session_id") != group
            for item in members
            if isinstance(item, Mapping)
        )
        or _mapping(intent.get("sole_gpu_process"), label="v8 sole GPU process").get("pid")
        != engine[0].get("pid")
        or _mapping(intent.get("sole_gpu_process"), label="v8 sole GPU process").get(
            "used_gpu_memory_mib"
        )
        != 21482
        or intent.get("utility_file_sha256") != _sha256_file(stop_utility_path)
        or group != service_checkpoint.get("pid")
        or group != service_checkpoint.get("process_group_id")
        or group != service_checkpoint.get("process_session_id")
        or group != checkpoint.get("handoff_service_pid")
        or members[0].get("pid") != service_checkpoint.get("pid")
        or members[0].get("start_ticks")
        != service_checkpoint.get("process_start_ticks")
    ):
        raise ValueError("v8 stop intent identity changed")
    if (
        outcome.get("run_id") != FALLBACK_V8_RUN_ID
        or outcome.get("service_event_id") != FALLBACK_V8_SERVICE_EVENT_ID
        or outcome.get("intent_file_sha256") != _sha256_file(intent_path)
        or outcome.get("signals_sent") != ["SIGTERM"]
        or outcome.get("remaining_group_members") != []
        or outcome.get("gpu_processes_after") != []
        or outcome.get("port_8000_listener_absent") is not True
        or outcome.get("physical_absence_verified") is not True
    ):
        raise ValueError("v8 exact stop outcome changed")
    if (
        receipt.get("run_id") != FALLBACK_V8_RUN_ID
        or receipt.get("service_event_id") != FALLBACK_V8_SERVICE_EVENT_ID
        or receipt.get("manual_stop_outcome_file_sha256") != _sha256_file(outcome_path)
        or receipt.get("manual_stop_outcome_manifest_sha256")
        != outcome.get("manifest_sha256")
        or receipt.get("physical_absence_revalidated") is not True
        or receipt.get("ledger_file_sha256_before") != before_recovery.file_sha256
        or receipt.get("ledger_file_sha256_after") != terminal.file_sha256
        or receipt.get("unresolved_gpu_allocation_count_after") != 0
        or receipt.get("unresolved_gpu_service_count_after") != 0
        or receipt.get("utility_file_sha256") != _sha256_file(accounting_utility_path)
    ):
        raise ValueError("v8 terminal accounting receipt changed")
    stopped = outcome.get("stopped_at")
    if not isinstance(stopped, str):
        raise ValueError("v8 stop outcome lacks a timestamp")
    stopped_at = _aware_datetime(stopped, label="v8 stop timestamp")
    expected_before_receipt_summary = {
        key: before_recovery.summary.model_dump(mode="python")[key]
        for key in (
            "total_allocated_microseconds",
            "event_count",
            "service_session_count",
            "by_kind_microseconds",
        )
    }
    expected_after_receipt_summary = {
        key: terminal.summary.model_dump(mode="python")[key]
        for key in (
            "total_allocated_microseconds",
            "event_count",
            "service_session_count",
            "by_kind_microseconds",
        )
    }
    expected_receipt_service = {
        "started_at": service.started_at.isoformat().replace("+00:00", "Z"),
        "ended_at": service.ended_at.isoformat().replace("+00:00", "Z"),
        "classified_event_microseconds": service.classified_event_microseconds,
        "overhead_microseconds": service.overhead_microseconds,
        "service_microseconds": service.service_microseconds,
    }
    if (
        stopped_at != datetime.fromisoformat("2026-09-05T20:40:09.324832+00:00")
        or intent_at >= stopped_at
        or receipt.get("accounted_through") != stopped
        or receipt.get("ledger_size_bytes_before") != before_recovery.size_bytes
        or receipt.get("ledger_size_bytes_after") != terminal.size_bytes
        or receipt.get("before_summary") != expected_before_receipt_summary
        or receipt.get("after_summary") != expected_after_receipt_summary
        or receipt.get("service") != expected_receipt_service
        or stopped_at != service.ended_at
    ):
        raise ValueError("v8 terminal accounting receipt summary or chronology changed")
    return group, stopped_at


def build_fallback_v8_runtime_incident(
    *,
    run_id: str,
    source_association_path: Path,
    authorization_overlay_path: Path,
    preflight_path: Path,
    controller_handoff_path: Path,
    run_result_path: Path,
    orphan_cleanup_path: Path,
    restricted_run_root: Path,
    manual_recovery_root: Path,
    ledger_before_v8_path: Path,
    ledger_before_manual_recovery_path: Path,
    terminal_ledger_path: Path,
    audited_at: datetime,
) -> FallbackV8RuntimeIncident:
    """Build V8's exact public terminal record from immutable evidence."""

    if run_id != FALLBACK_V8_RUN_ID:
        raise ValueError("v8 incident builder is restricted to the exact v8 run")
    if audited_at.tzinfo is None or audited_at.utcoffset() is None:
        raise ValueError("incident audit timestamp must be timezone-aware")
    run_root = Path(restricted_run_root).absolute()
    _require_no_symlink_ancestry(run_root, label="restricted v8 run root")
    if not run_root.resolve(strict=True).is_dir():
        raise ValueError("restricted v8 run root must be a directory")
    run_root = run_root.resolve(strict=True)
    observed_names = tuple(sorted(item.name for item in run_root.iterdir()))
    if observed_names != EXPECTED_RUN_ROOT_BASENAMES:
        raise ValueError("restricted v8 run root is not the exact preserved artifact set")
    run_root_files = {
        name: _require_regular_file(run_root / name, label=f"v8 run-root entry {name}")
        for name in observed_names
    }

    source_path, source = _load_self_hashed_json(
        source_association_path,
        label="v8 source association",
        expected_kind="local_remote_source_tree_association",
    )
    overlay_path, overlay = _load_self_hashed_json(
        authorization_overlay_path,
        label="v8 authorization overlay",
        expected_kind="phase1_fallback_second_recovery_overlay",
    )
    preflight_resolved, preflight = _load_self_hashed_json(
        preflight_path,
        label="v8 preflight",
        expected_kind="phase1_fallback_execution_preflight",
    )
    handoff_path, handoff = _load_self_hashed_json(
        controller_handoff_path,
        label="v8 controller handoff",
        expected_kind="phase1_fallback_controller_restart_handoff",
    )
    result_path, result = _load_self_hashed_json(
        run_result_path,
        label="v8 run result",
        expected_kind="phase1_fallback_micro_pilot_result",
    )
    cleanup_path, cleanup = _load_self_hashed_json(
        orphan_cleanup_path,
        label="v8 orphan cleanup",
        expected_kind="phase1_fallback_micro_pilot_result",
    )

    specs = {
        "invocation": (
            "checkpoint.json.orchestrator-invocation.json",
            "fallback_controller_orchestration_invocation",
        ),
        "ticket": (
            "checkpoint.json.guardian-ticket.json",
            "fallback_service_guardian_ticket",
        ),
        "ready": (
            "checkpoint.json.guardian-ready-000001.json",
            "fallback_service_guardian_ready",
        ),
        "terminal_request": (
            "checkpoint.json.guardian-terminal-request.json",
            "fallback_guardian_terminal_request",
        ),
        "guard": (
            "checkpoint.json.orchestrator-guard-000001.json",
            "fallback_controller_orchestrator_guard",
        ),
        "closed": (
            ".checkpoint.json.orchestrator-guard-000001.json.closed.json",
            "fallback_controller_orchestrator_closed",
        ),
        "prepare": (
            "checkpoint.json.internal-controller-000001.json",
            "fallback_internal_controller_receipt",
        ),
        "run": (
            "checkpoint.json.internal-controller-000002.json",
            "fallback_internal_controller_receipt",
        ),
        "cleanup": (
            "checkpoint.json.internal-controller-000003.json",
            "fallback_internal_controller_receipt",
        ),
    }
    loaded = {
        name: _load_self_hashed_json(
            run_root / basename, label=f"v8 {name}", expected_kind=kind
        )
        for name, (basename, kind) in specs.items()
    }
    checkpoint_path, checkpoint = _load_json(run_root / "checkpoint.json", label="v8 checkpoint")
    service_path, service_checkpoint = _load_json(
        run_root / "checkpoint.json.service", label="v8 service checkpoint"
    )
    guardian_log_path = _require_regular_file(
        run_root / "checkpoint.json.guardian.log", label="v8 guardian log"
    )
    orchestrator_log_path = _require_regular_file(
        run_root / "orchestrator.20260905T201854Z.log", label="v8 orchestrator log"
    )
    vllm_log_path = _require_regular_file(
        run_root / f"{run_id}.vllm.log", label="v8 vLLM log"
    )
    guardian_log = guardian_log_path.read_text(encoding="utf-8")
    orchestrator_log = orchestrator_log_path.read_text(encoding="utf-8")
    vllm_log = vllm_log_path.read_text(encoding="utf-8")
    if (
        guardian_log.count("service process group contains an unbound process") < 2
        or "fallback service adoption failed and physical shutdown was not verified"
        not in guardian_log
        or "run result does not certify physical shutdown" not in orchestrator_log
        or "Starting vLLM API server 0 on http://127.0.0.1:8000" not in vllm_log
        or "Shutting down FastAPI HTTP server" not in vllm_log
    ):
        raise ValueError("v8 diagnostic evidence markers changed")

    before = _ledger_binding(ledger_before_v8_path, label="pre-v8 ledger")
    before_recovery = _ledger_binding(
        ledger_before_manual_recovery_path, label="pre-recovery v8 ledger"
    )
    terminal_path = _require_regular_file(terminal_ledger_path, label="terminal v8 ledger")
    terminal = _ledger_binding(terminal_path, label="terminal v8 ledger")
    startup_event, service, resource_maxima = _extract_accounting(terminal_path)
    accounting = FallbackV8Accounting(
        before_v8=before,
        before_manual_recovery=before_recovery,
        terminal=terminal,
        startup_event=startup_event,
        service=service,
        delta=FallbackV8AccountingDelta(
            allocated_gpu_microseconds=EXPECTED_SERVICE_MICROSECONDS,
            classified_startup_microseconds=EXPECTED_CLASSIFIED_MICROSECONDS,
            conservative_service_overhead_microseconds=EXPECTED_OVERHEAD_MICROSECONDS,
            gpu_events=1,
            service_sessions=1,
            attempts=0,
            inference_model_calls=0,
            artifacts=0,
            storage_samples=10,
            resource_samples=6,
            allocation_journal_rows=47,
            service_journal_rows=56,
            accepted_outputs=0,
        ),
    )
    _require_source_chain(source_path, source, overlay, preflight, before)
    values = {key: value for key, (_path, value) in loaded.items()}
    _require_control_chain(
        invocation=values["invocation"],
        ticket=values["ticket"],
        ready=values["ready"],
        terminal_request=values["terminal_request"],
        guard=values["guard"],
        closed=values["closed"],
        receipts=(values["prepare"], values["run"], values["cleanup"]),
        checkpoint=checkpoint,
        service_checkpoint=service_checkpoint,
        handoff=handoff,
        result=result,
        cleanup=cleanup,
        result_path=result_path,
        before=before,
        startup_event=startup_event,
        service=service,
    )

    recovery_root = Path(manual_recovery_root).absolute().resolve(strict=True)
    intent_path, intent = _load_self_hashed_json(
        recovery_root / "manual_stop_intent.json",
        label="v8 stop intent",
        expected_kind="fallback_v8_exact_orphan_stop_intent",
    )
    outcome_path, outcome = _load_self_hashed_json(
        recovery_root / "manual_stop_outcome.json",
        label="v8 stop outcome",
        expected_kind="fallback_v8_exact_orphan_stop_outcome",
    )
    receipt_path, receipt = _load_self_hashed_json(
        recovery_root / "terminal_accounting_receipt.json",
        label="v8 accounting receipt",
        expected_kind="fallback_v8_terminal_accounting_recovery",
    )
    stop_utility_path = _require_regular_file(
        recovery_root / "manual_stop_v8.py", label="v8 stop utility"
    )
    accounting_utility_path = _require_regular_file(
        recovery_root / "recover_v8_ledger.py", label="v8 accounting utility"
    )
    group, stopped_at = _require_manual_recovery_chain(
        intent_path=intent_path,
        intent=intent,
        outcome_path=outcome_path,
        outcome=outcome,
        receipt=receipt,
        stop_utility_path=stop_utility_path,
        accounting_utility_path=accounting_utility_path,
        before_recovery=before_recovery,
        terminal=terminal,
        checkpoint=checkpoint,
        service_checkpoint=service_checkpoint,
        service=service,
    )
    if stopped_at != service.ended_at:
        raise ValueError("v8 stop and service timestamps changed")

    absent_paths = tuple(run_root / name for name in EXPECTED_ABSENT_BASENAMES)
    if any(path.exists() or path.is_symlink() for path in absent_paths):
        raise ValueError("a required v8 terminal-state path exists")
    inventory = {
        "files": [
            {
                "basename": name,
                "size_bytes": run_root_files[name].stat().st_size,
                "file_sha256": _sha256_file(run_root_files[name]),
            }
            for name in observed_names
        ]
    }
    endpoint_matches = re.findall(
        r"INFO 09-05 (\d{2}):(\d{2}):(\d{2}) .*Starting vLLM API server 0",
        vllm_log,
    )
    if endpoint_matches != [("20", "25", "52")]:
        raise ValueError("v8 healthy endpoint timestamp changed")
    endpoint_at = datetime(2026, 9, 5, 20, 25, 52, tzinfo=UTC)

    def binding(name: str) -> SelfHashedFileBinding:
        path, value = loaded[name]
        return _self_hashed_binding(path, value)

    payload: dict[str, object] = {
        "schema_version": "1.0.0",
        "kind": FALLBACK_V8_RUNTIME_INCIDENT_KIND,
        "run_id": run_id,
        "audited_at": audited_at,
        "source_and_authorization": FallbackV8SourceAndAuthorization(
            source_revision=FALLBACK_V8_SOURCE_REVISION,
            source_git_commit=source["git_commit"],
            source_tree_sha256=source["local_tree_sha256"],
            source_association=_self_hashed_binding(source_path, source),
            authorization_overlay=_self_hashed_binding(overlay_path, overlay),
            execution_preflight=_self_hashed_binding(preflight_resolved, preflight),
        ),
        "public_evidence": FallbackV8PublicEvidence(
            controller_handoff=_self_hashed_binding(handoff_path, handoff),
            run_result=_self_hashed_binding(result_path, result),
            orphan_cleanup=_self_hashed_binding(cleanup_path, cleanup),
        ),
        "restricted_evidence": FallbackV8RestrictedEvidence(
            orchestration_invocation=binding("invocation"),
            guardian_ticket=binding("ticket"),
            guardian_ready=binding("ready"),
            guardian_terminal_request=binding("terminal_request"),
            orchestrator_guard=binding("guard"),
            closed_orchestrator_guard=binding("closed"),
            prepare_controller_receipt=binding("prepare"),
            run_controller_receipt=binding("run"),
            cleanup_controller_receipt=binding("cleanup"),
            checkpoint=_opaque_binding(checkpoint_path, label="v8 checkpoint"),
            service_checkpoint=_opaque_binding(service_path, label="v8 service checkpoint"),
            guardian_log=_opaque_binding(guardian_log_path, label="v8 guardian log"),
            orchestrator_log=_opaque_binding(orchestrator_log_path, label="v8 orchestrator log"),
            vllm_log=_opaque_binding(vllm_log_path, label="v8 vLLM log"),
            run_root_inventory_sha256=_canonical_sha256(inventory),
        ),
        "manual_recovery": FallbackV8ManualRecoveryEvidence(
            stop_intent=_self_hashed_binding(intent_path, intent),
            stop_outcome=_self_hashed_binding(outcome_path, outcome),
            accounting_receipt=_self_hashed_binding(receipt_path, receipt),
            stop_utility=_opaque_binding(stop_utility_path, label="v8 stop utility"),
            accounting_utility=_opaque_binding(
                accounting_utility_path, label="v8 accounting utility"
            ),
            pre_stop_lease_file_sha256=intent["lease_file_sha256"],
            pre_stop_lease_manifest_sha256=intent["lease_manifest_sha256"],
            process_group_id=group,
            process_members=(
                FallbackV8ServiceLeaderIdentity(
                    role="service_leader",
                    pid=EXPECTED_PROCESS_MEMBERS[0]["pid"],
                    parent_pid=EXPECTED_PROCESS_MEMBERS[0]["ppid"],
                    process_group_id=EXPECTED_PROCESS_MEMBERS[0]["process_group"],
                    session_id=EXPECTED_PROCESS_MEMBERS[0]["session_id"],
                    start_ticks=EXPECTED_PROCESS_MEMBERS[0]["start_ticks"],
                    process_name=EXPECTED_PROCESS_MEMBERS[0]["comm"],
                    process_state=EXPECTED_PROCESS_MEMBERS[0]["state"],
                    argv_sha256=EXPECTED_PROCESS_MEMBERS[0]["raw_command_sha256"],
                ),
                FallbackV8WorkerIdentity(
                    role="service_worker",
                    pid=EXPECTED_PROCESS_MEMBERS[1]["pid"],
                    parent_pid=EXPECTED_PROCESS_MEMBERS[1]["ppid"],
                    process_group_id=EXPECTED_PROCESS_MEMBERS[1]["process_group"],
                    session_id=EXPECTED_PROCESS_MEMBERS[1]["session_id"],
                    start_ticks=EXPECTED_PROCESS_MEMBERS[1]["start_ticks"],
                    process_name=EXPECTED_PROCESS_MEMBERS[1]["comm"],
                    process_state=EXPECTED_PROCESS_MEMBERS[1]["state"],
                    argv_sha256=EXPECTED_PROCESS_MEMBERS[1]["raw_command_sha256"],
                ),
                FallbackV8EngineCoreIdentity(
                    role="engine_core",
                    pid=EXPECTED_PROCESS_MEMBERS[2]["pid"],
                    parent_pid=EXPECTED_PROCESS_MEMBERS[2]["ppid"],
                    process_group_id=EXPECTED_PROCESS_MEMBERS[2]["process_group"],
                    session_id=EXPECTED_PROCESS_MEMBERS[2]["session_id"],
                    start_ticks=EXPECTED_PROCESS_MEMBERS[2]["start_ticks"],
                    process_name=EXPECTED_PROCESS_MEMBERS[2]["comm"],
                    process_state=EXPECTED_PROCESS_MEMBERS[2]["state"],
                    argv_sha256=EXPECTED_PROCESS_MEMBERS[2]["raw_command_sha256"],
                ),
            ),
            member_count=3,
            engine_core_token_matched=False,
            sole_gpu_process_memory_mib=21482,
            sole_port_owner_verified=True,
            signals_sent=("SIGTERM",),
            stopped_at=stopped_at,
            physical_absence_verified=True,
        ),
        "accounting": accounting,
        "resource_maxima": resource_maxima,
        "failure_sequence": FallbackV8FailureSequence(
            safe_error_class=FALLBACK_V8_SAFE_ERROR_CLASS,
            root_cause=FALLBACK_V8_ROOT_CAUSE,
            root_cause_basis="source_bound_guardian_trace_and_exact_pre_stop_inventory",
            failure_stage="post_handoff_service_adoption_before_micro_pilot_inference",
            endpoint_healthy_before_failure=True,
            endpoint_healthy_at=endpoint_at,
            controller_handoff_completed=True,
            prepare_return_code=0,
            run_return_code=2,
            cleanup_return_code=2,
            guardian_outer_failure_type="RuntimeError",
            adoption_inner_failure_type="RuntimeConfigurationError",
            scientific_inference_reached=False,
            command_line_public=False,
            full_log_text_public=False,
            remote_absolute_paths_public=False,
        ),
        "absence_inventory": FallbackV8AbsenceInventory(
            observed_run_root_basenames=observed_names,
            absent_basenames=EXPECTED_ABSENT_BASENAMES,
            all_required_paths_absent=True,
        ),
        "terminal_state": FallbackV8TerminalState(
            public_controller_handoff_present=True,
            public_failed_result_present=True,
            public_cleanup_result_present=True,
            accepted_output_count=0,
            inference_attempt_count_delta=0,
            inference_model_call_count_delta=0,
            phase1_gate_passed=False,
            service_checkpoint_present=True,
            guardian_result_present=False,
            automated_cleanup_succeeded=False,
            manual_physical_stop_verified=True,
            endpoint_live=False,
            gpu_process_live=False,
            exact_service_process_live=False,
            unresolved_gpu_allocation_count=0,
            unresolved_gpu_service_count=0,
            terminal_ledger_verified=True,
            resume_allowed=False,
            source_bound_v8_resume_permitted=False,
            fresh_repaired_source_required=True,
        ),
    }
    return FallbackV8RuntimeIncident.model_validate(
        {**payload, "manifest_sha256": _canonical_sha256(payload)}
    )


def load_fallback_v8_runtime_incident(path: Path) -> FallbackV8RuntimeIncident:
    resolved = _require_regular_file(path, label="fallback-v8 runtime incident")
    try:
        value = json.loads(resolved.read_text(encoding="utf-8"))
        return FallbackV8RuntimeIncident.model_validate(value)
    except (OSError, UnicodeError, json.JSONDecodeError, ValidationError) as exc:
        raise ValueError("fallback-v8 incident contract is invalid") from exc


def validate_fallback_v8_runtime_incident(
    path: Path,
    *,
    expected_source_association_manifest_sha256: str,
    expected_overlay_manifest_sha256: str,
    expected_preflight_manifest_sha256: str,
    expected_terminal_ledger_file_sha256: str,
) -> FallbackV8RuntimeIncident:
    incident = load_fallback_v8_runtime_incident(path)
    source = incident.source_and_authorization
    if (
        source.source_association.manifest_sha256
        != expected_source_association_manifest_sha256
        or source.authorization_overlay.manifest_sha256
        != expected_overlay_manifest_sha256
        or source.execution_preflight.manifest_sha256
        != expected_preflight_manifest_sha256
        or incident.accounting.terminal.file_sha256
        != expected_terminal_ledger_file_sha256
    ):
        raise ValueError("fallback-v8 incident differs from its expected identity")
    return incident


def write_fallback_v8_runtime_incident(
    path: Path, incident: FallbackV8RuntimeIncident
) -> None:
    destination = Path(path).absolute()
    _require_no_symlink_ancestry(destination, label="v8 incident output")
    destination.parent.mkdir(parents=True, exist_ok=True)
    _require_no_symlink_ancestry(destination, label="v8 incident output")
    payload = (canonical_json(incident) + "\n").encode("utf-8")
    if destination.exists():
        if (
            destination.is_file()
            and not destination.is_symlink()
            and destination.read_bytes() == payload
        ):
            return
        raise FileExistsError("append-only v8 incident output already differs")
    descriptor, temporary_name = tempfile.mkstemp(
        dir=destination.parent, prefix=f".{destination.name}.", suffix=".tmp"
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            descriptor = -1
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, destination)
        directory = os.open(destination.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        temporary.unlink(missing_ok=True)


__all__ = [
    "FALLBACK_V8_ROOT_CAUSE",
    "FALLBACK_V8_RUNTIME_INCIDENT_KIND",
    "FALLBACK_V8_RUN_ID",
    "FallbackV8RuntimeIncident",
    "build_fallback_v8_runtime_incident",
    "load_fallback_v8_runtime_incident",
    "validate_fallback_v8_runtime_incident",
    "write_fallback_v8_runtime_incident",
]
