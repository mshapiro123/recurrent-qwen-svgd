"""Separate, fail-closed runtime identity for WEFT-1 G-TOK GPU training.

The corpus P-A lock intentionally contains no PyTorch and therefore cannot
authorize training.  This module defines a distinct training-runtime binding,
an executable hash-pinned venv builder, and an attestation that covers the
installed RECORD trees, interpreter, PyTorch/CUDA/cuDNN build, driver, and A100
device.  Under A2-R7, one environment that satisfies the fixed policy is bound
mechanically; a diagnostic request can also be emitted without granting run
authority.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import base64
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import re
import site
import subprocess
import sys
import sysconfig
from typing import Any, Mapping

from training.weft1_gtok_contract import canonical_json_bytes, canonical_sha256
from training.weft1_strict_io import assert_no_symlink_ancestors


RUNTIME_BINDING_SCHEMA_V2 = "weft1_gtok_training_runtime_binding_v2"
RUNTIME_RECEIPT_SCHEMA_V2 = "weft1_gtok_training_runtime_receipt_v2"
RUNTIME_REQUEST_SCHEMA_V2 = "weft1_gtok_training_runtime_binding_request_v2"
PYTHON_VERSION_V2 = "3.11.9"
TORCH_VERSION_V2 = "2.11.0+cu128"
TORCH_CUDA_VERSION_V2 = "12.8"
TORCH_CUDNN_VERSION_V2 = 91_900
TRAINING_LOCK_SHA256_V2 = "7dd3360c23517c70b9853496135f766a2bc710c59e4fce95e8454b3d4dc52e72"
_CLOSED_ENVIRONMENT = {
    "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
    "LANG": "C.UTF-8",
    "LC_ALL": "C.UTF-8",
    "PYTHONHASHSEED": "0",
    "TOKENIZERS_PARALLELISM": "false",
    "TZ": "UTC",
    "PIP_CONFIG_FILE": os.devnull,
    "PIP_DISABLE_PIP_VERSION_CHECK": "1",
    "PIP_NO_INDEX": "1",
    "PIP_REQUIRE_VIRTUALENV": "1",
}
_REMOVED_ENVIRONMENT = (
    "LD_LIBRARY_PATH",
    "PYTHONHOME",
    "PYTHONPATH",
    "PYTHONUSERBASE",
    "PIP_CERT",
    "PIP_CLIENT_CERT",
    "PIP_EXTRA_INDEX_URL",
    "PIP_FIND_LINKS",
    "PIP_INDEX_URL",
    "PIP_PROXY",
    "PIP_TRUSTED_HOST",
    "REQUESTS_CA_BUNDLE",
    "UV_EXTRA_INDEX_URL",
    "UV_INDEX",
    "UV_INDEX_URL",
    "UV_NATIVE_TLS",
    "UV_NO_CONFIG",
    "UV_PYTHON",
    "UV_PYTHON_DOWNLOADS",
    "UV_SYSTEM_PYTHON",
)
_HEX = frozenset("0123456789abcdef")


class GTokRuntimeV2Error(RuntimeError):
    """A training-runtime identity or build invariant failed."""


class GTokRuntimeBindingRequiredV2(GTokRuntimeV2Error):
    """The observed GPU stack has no exact literal binding."""


def closed_training_environment_v2(
    source: Mapping[str, str] | None = None,
) -> dict[str, str]:
    """Return the deterministic process environment used by builder launches."""

    environment = dict(os.environ if source is None else source)
    for name in _REMOVED_ENVIRONMENT:
        environment.pop(name, None)
    environment.update(_CLOSED_ENVIRONMENT)
    environment["PYTHONNOUSERSITE"] = "1"
    return environment


def assert_closed_training_environment_v2(
    source: Mapping[str, str] | None = None,
) -> None:
    environment = os.environ if source is None else source
    if any(environment.get(name) != value for name, value in _CLOSED_ENVIRONMENT.items()):
        raise GTokRuntimeV2Error("G-TOK process environment is not C.UTF-8/UTC/hash-seed closed")
    if environment.get("PYTHONNOUSERSITE") != "1":
        raise GTokRuntimeV2Error("G-TOK process must disable the user site")
    if any(environment.get(name) not in (None, "") for name in _REMOVED_ENVIRONMENT):
        raise GTokRuntimeV2Error("G-TOK process inherits a forbidden Python/linker path")
    if source is None and (not sys.flags.isolated or not sys.dont_write_bytecode):
        raise GTokRuntimeV2Error("G-TOK interpreter must be launched with exact -I -B flags")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _load_pa_runtime_build_receipt_v2(
    path: Path,
) -> tuple[Mapping[str, Any], str, str]:
    """Verify the exact authoritative P-A interpreter receipt used as base."""

    resolved = assert_no_symlink_ancestors(path).resolve(strict=True)
    raw = resolved.read_bytes()
    try:
        value = json.loads(raw.decode("utf-8", errors="strict"))
        from scripts.build_weft1_pa_runtime import verify_build_receipt_payload

        identity = verify_build_receipt_payload(value)
    except (UnicodeDecodeError, json.JSONDecodeError, ImportError, RuntimeError) as error:
        raise GTokRuntimeV2Error("P-A runtime build receipt is not authoritative") from error
    if raw != canonical_json_bytes(value):
        raise GTokRuntimeV2Error("P-A runtime build receipt is not canonical JSON")
    return value, identity, hashlib.sha256(raw).hexdigest()


def _normalized_distribution_name(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def installed_record_tree_inventory_v2() -> tuple[dict[str, Any], ...]:
    """Hash every file named by every installed distribution's RECORD tree."""

    rows: list[dict[str, Any]] = []
    prefix = Path(sys.prefix).resolve(strict=True)
    for distribution in importlib.metadata.distributions():
        name = distribution.metadata.get("Name")
        if not isinstance(name, str) or not name:
            raise GTokRuntimeV2Error("installed distribution lacks a canonical name")
        files = distribution.files
        if files is None:
            raise GTokRuntimeV2Error(f"installed distribution {name!r} has no RECORD tree")
        file_rows: list[dict[str, Any]] = []
        by_name = {str(item): item for item in files}
        for relative in sorted(by_name):
            package_path = by_name[relative]
            located = Path(distribution.locate_file(package_path))
            if not located.is_file():
                raise GTokRuntimeV2Error(
                    f"installed RECORD path is missing for {name!r}: {relative}"
                )
            if located.is_symlink():
                raise GTokRuntimeV2Error(f"installed RECORD path is a symlink: {located}")
            resolved = located.resolve(strict=True)
            if resolved != prefix and prefix not in resolved.parents:
                raise GTokRuntimeV2Error(f"installed RECORD path escapes venv prefix: {resolved}")
            size = located.stat().st_size
            if package_path.size is not None and size != package_path.size:
                raise GTokRuntimeV2Error(f"installed RECORD size drifted: {relative}")
            digest = _sha256_file(located)
            if package_path.hash is None:
                if not relative.replace("\\", "/").endswith(".dist-info/RECORD"):
                    raise GTokRuntimeV2Error(f"installed RECORD row has no hash: {relative}")
            elif package_path.hash.mode != "sha256" or package_path.hash.value != (
                base64.urlsafe_b64encode(bytes.fromhex(digest)).rstrip(b"=").decode("ascii")
            ):
                raise GTokRuntimeV2Error(f"installed RECORD hash drifted: {relative}")
            file_rows.append(
                {
                    "bytes": size,
                    "relative_path": relative.replace("\\", "/"),
                    "sha256": digest,
                }
            )
        core = {
            "name": _normalized_distribution_name(name),
            "record_files": tuple(file_rows),
            "version": distribution.version,
        }
        rows.append(
            {
                "name": core["name"],
                "record_file_count": len(file_rows),
                "record_tree_sha256": canonical_sha256(core),
                "version": distribution.version,
            }
        )
    ordered = tuple(sorted(rows, key=lambda row: (row["name"], row["version"])))
    names = tuple(row["name"] for row in ordered)
    if len(set(names)) != len(names):
        raise GTokRuntimeV2Error("training environment contains duplicate distributions")
    return ordered


def assert_no_unowned_site_files_v2() -> str:
    """Reject files in site-packages that are absent from every wheel RECORD."""

    owned: set[Path] = set()
    for distribution in importlib.metadata.distributions():
        if distribution.files is None:
            raise GTokRuntimeV2Error("installed distribution has no RECORD ownership tree")
        for relative in distribution.files:
            located = Path(distribution.locate_file(relative))
            if located.is_file():
                owned.add(located.resolve(strict=True))
    observed: list[dict[str, Any]] = []
    prefix = Path(sys.prefix).resolve(strict=True)
    roots = tuple(
        root
        for root in (Path(value).resolve(strict=True) for value in site.getsitepackages())
        if root == prefix or prefix in root.parents
    )
    if not roots:
        raise GTokRuntimeV2Error("no venv-owned site-packages root was found")
    for root in roots:
        for path in sorted(root.rglob("*")):
            if not path.is_file() or path.suffix == ".pyc" or "__pycache__" in path.parts:
                continue
            resolved = path.resolve(strict=True)
            if resolved not in owned:
                raise GTokRuntimeV2Error(f"unowned training-runtime file: {resolved}")
            observed.append(
                {
                    "path": resolved.relative_to(root).as_posix(),
                    "root": str(root),
                    "sha256": _sha256_file(resolved),
                }
            )
    if not observed:
        raise GTokRuntimeV2Error("training-runtime site-packages inventory is empty")
    return canonical_sha256(tuple(observed))


def _python_identity_v2() -> dict[str, Any]:
    executable = Path(sys.executable).resolve(strict=True)
    return {
        "abi_flags": getattr(sys, "abiflags", ""),
        "base_prefix": str(Path(sys.base_prefix).resolve(strict=True)),
        "executable_sha256": _sha256_file(executable),
        "implementation": platform.python_implementation(),
        "platform": sysconfig.get_platform(),
        "prefix": str(Path(sys.prefix).resolve(strict=True)),
        "python_version": platform.python_version(),
        "soabi": sysconfig.get_config_var("SOABI"),
    }


def _loaded_cuda_library_identity_v2() -> tuple[dict[str, Any], ...]:
    maps = Path("/proc/self/maps")
    if not maps.is_file():
        raise GTokRuntimeV2Error("production runtime cannot attest loaded CUDA libraries")
    loaded_paths: set[tuple[str, Path]] = set()
    for line in maps.read_text(encoding="utf-8", errors="strict").splitlines():
        candidate = line.rsplit(maxsplit=1)[-1]
        library_name = Path(candidate).name.lower()
        kind = (
            "cudart"
            if library_name.startswith("libcudart.")
            else "cudnn"
            if library_name.startswith("libcudnn.")
            else "cuda_driver"
            if library_name.startswith("libcuda.")
            else None
        )
        if candidate.startswith("/") and kind is not None:
            path = Path(candidate)
            if path.is_file():
                loaded_paths.add((kind, path.resolve(strict=True)))
    kinds = {kind for kind, _path in loaded_paths}
    if kinds != {"cuda_driver", "cudart", "cudnn"}:
        raise GTokRuntimeV2Error("loaded CUDA/cuDNN library closure is incomplete")
    prefix = Path(sys.prefix).resolve(strict=True)
    for kind, path in loaded_paths:
        inside_venv = path == prefix or prefix in path.parents
        if kind in {"cudart", "cudnn"} and not inside_venv:
            raise GTokRuntimeV2Error(
                f"loaded {kind} library is not owned by the exact training venv"
            )
        if kind == "cuda_driver" and inside_venv:
            raise GTokRuntimeV2Error("CUDA driver library must come from the host driver")
    return tuple(
        {
            "bytes": path.stat().st_size,
            "kind": kind,
            "path": str(path),
            "sha256": _sha256_file(path),
        }
        for kind, path in sorted(loaded_paths, key=lambda row: (row[0], str(row[1])))
    )


def _torch_cuda_identity_v2(*, device_index: int) -> dict[str, Any]:
    try:
        import torch
    except ImportError as error:
        raise GTokRuntimeV2Error("training runtime has no PyTorch distribution") from error
    if not torch.cuda.is_available():
        raise GTokRuntimeV2Error("training runtime has no CUDA device")
    if device_index < 0 or device_index >= torch.cuda.device_count():
        raise GTokRuntimeV2Error("selected CUDA device index is unavailable")
    device = torch.device(f"cuda:{device_index}")
    properties = torch.cuda.get_device_properties(device)
    device_name = torch.cuda.get_device_name(device)
    if "A100" not in device_name.upper() or tuple(torch.cuda.get_device_capability(device)) != (8, 0):
        raise GTokRuntimeV2Error("production G-TOK requires an NVIDIA A100 (sm80)")
    driver_getter = getattr(torch._C, "_cuda_getDriverVersion", None)
    if driver_getter is not None:
        driver_version: int | str = int(driver_getter())
    else:
        try:
            driver_version = subprocess.run(
                [
                    "nvidia-smi",
                    f"--id={device_index}",
                    "--query-gpu=driver_version",
                    "--format=csv,noheader,nounits",
                ],
                check=True,
                capture_output=True,
                text=True,
                env=closed_training_environment_v2(),
            ).stdout.strip()
        except (OSError, subprocess.CalledProcessError) as error:
            raise GTokRuntimeV2Error("CUDA driver version is unavailable") from error
    if not driver_version or torch.backends.cudnn.version() is None:
        raise GTokRuntimeV2Error("CUDA driver/cuDNN runtime identity is unavailable")
    # Force CUDA runtime and cuDNN linkage before inspecting the process map.
    sample = torch.zeros((1, 1, 3, 3), device=device)
    kernel = torch.ones((1, 1, 1, 1), device=device)
    torch.nn.functional.conv2d(sample, kernel)
    torch.cuda.synchronize(device)
    loaded_libraries = _loaded_cuda_library_identity_v2()
    return {
        "cuda": {
            "compiled_version": torch.version.cuda,
            "cudnn_version": torch.backends.cudnn.version(),
            "driver_version": driver_version,
            "loaded_libraries": loaded_libraries,
        },
        "device": {
            "capability": tuple(torch.cuda.get_device_capability(device)),
            "name": device_name,
            "total_memory_bytes": int(properties.total_memory),
        },
        "torch": {
            "debug_build": bool(torch.version.debug),
            "git_version": torch.version.git_version,
            "version": torch.__version__,
        },
    }


def gpu_uuid_provenance_v2(*, device_index: int) -> str:
    """Record physical A100 provenance without making it runtime equality identity."""

    if type(device_index) is not int or device_index < 0:
        raise GTokRuntimeV2Error("GPU UUID device index must be non-negative")
    try:
        value = subprocess.run(
            [
                "nvidia-smi",
                f"--id={device_index}",
                "--query-gpu=uuid",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            capture_output=True,
            text=True,
            env=closed_training_environment_v2(),
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError) as error:
        raise GTokRuntimeV2Error("GPU UUID provenance is unavailable") from error
    if re.fullmatch(r"GPU-[0-9A-Fa-f-]{16,}", value) is None:
        raise GTokRuntimeV2Error("GPU UUID provenance is malformed")
    return value


_CPU_RUNTIME_IDENTITY_KEYS_V2 = (
    "closed_environment",
    "installed_distributions",
    "owned_site_tree_sha256",
    "python",
    "requirements_lock_bytes",
    "requirements_lock_sha256",
    "pa_runtime_build_receipt_identity_sha256",
    "pa_runtime_build_receipt_physical_sha256",
    "runtime_build_receipt_sha256",
)


def cpu_runtime_identity_sha256_from_payload_v2(payload: Mapping[str, Any]) -> str:
    if not isinstance(payload, Mapping) or any(
        name not in payload for name in _CPU_RUNTIME_IDENTITY_KEYS_V2
    ):
        raise GTokRuntimeV2Error("CPU runtime identity payload is incomplete")
    return canonical_sha256(
        {name: payload[name] for name in _CPU_RUNTIME_IDENTITY_KEYS_V2}
    )


def observed_training_cpu_runtime_payload_v2(
    *,
    requirements_lock: Path,
    runtime_build_receipt: Path,
    pa_runtime_build_receipt: Path,
) -> dict[str, Any]:
    """Attest the exact training venv without touching CUDA or selecting a GPU."""

    # Reporting the desired closed mapping is not evidence that the live CPU
    # precompute process actually used it.  Fail before opening any build,
    # lock, corpus, or tokenizer evidence when the interpreter/environment is
    # not the exact production ``python -I -B`` posture.
    assert_closed_training_environment_v2()
    lock = assert_no_symlink_ancestors(requirements_lock).resolve(strict=True)
    lock_bytes, lock_rows = _parse_hash_pinned_lock_v2(lock)
    if hashlib.sha256(lock_bytes).hexdigest() != TRAINING_LOCK_SHA256_V2:
        raise GTokRuntimeV2Error("CPU precompute lock differs from fixed A2 policy")
    build_receipt = assert_no_symlink_ancestors(runtime_build_receipt).resolve(strict=True)
    build_raw = build_receipt.read_bytes()
    try:
        build = json.loads(build_raw.decode("utf-8", errors="strict"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise GTokRuntimeV2Error("CPU precompute build receipt is invalid JSON") from error
    if (
        not isinstance(build, dict)
        or build_raw != canonical_json_bytes(build) + b"\n"
        or build.get("schema") != "weft1_gtok_training_runtime_build_v2"
    ):
        raise GTokRuntimeV2Error("CPU precompute build receipt is noncanonical")
    inventory = installed_record_tree_inventory_v2()
    expected_versions = {(row["name"], row["version"]) for row in lock_rows}
    observed_versions = {(row["name"], row["version"]) for row in inventory}
    if observed_versions != expected_versions:
        raise GTokRuntimeV2Error("CPU precompute distribution closure differs from lock")
    owned_tree = assert_no_unowned_site_files_v2()
    python_identity = _python_identity_v2()
    if python_identity.get("python_version") != PYTHON_VERSION_V2:
        raise GTokRuntimeV2Error("CPU precompute requires exact Python 3.11.9")
    if (
        build.get("requirements_lock_sha256") != hashlib.sha256(lock_bytes).hexdigest()
        or build.get("requirements_lock_bytes") != len(lock_bytes)
        or canonical_json_bytes(build.get("lock_rows")) != canonical_json_bytes(lock_rows)
        or canonical_json_bytes(build.get("installed_distributions"))
        != canonical_json_bytes(inventory)
        or build.get("owned_site_tree_sha256") != owned_tree
        or build.get("built_python_executable_sha256")
        != python_identity["executable_sha256"]
        or Path(str(build.get("built_python_executable"))).resolve(strict=True)
        != Path(sys.executable).resolve(strict=True)
        or Path(str(build.get("venv_prefix"))).resolve(strict=True)
        != Path(sys.prefix).resolve(strict=True)
        or Path(str(build.get("base_python_prefix"))).resolve(strict=True)
        != Path(sys.base_prefix).resolve(strict=True)
    ):
        raise GTokRuntimeV2Error("CPU precompute runtime differs from live training venv")
    pa_value, pa_identity, pa_physical_sha256 = _load_pa_runtime_build_receipt_v2(
        pa_runtime_build_receipt
    )
    pa_evidence = pa_value.get("evidence")
    pa_artifacts = pa_evidence.get("artifacts") if isinstance(pa_evidence, Mapping) else None
    if (
        not isinstance(pa_evidence, Mapping)
        or not isinstance(pa_artifacts, Mapping)
        or build.get("pa_runtime_build_receipt_identity_sha256") != pa_identity
        or build.get("pa_runtime_build_receipt_physical_sha256") != pa_physical_sha256
        or Path(str(pa_evidence.get("prefix"))).resolve(strict=True)
        != Path(sys.base_prefix).resolve(strict=True)
        or pa_artifacts.get("cpython_executable_sha256")
        != build.get("base_python_executable_sha256")
    ):
        raise GTokRuntimeV2Error("CPU precompute venv differs from exact P-A ancestor")
    return {
        "closed_environment": dict(_CLOSED_ENVIRONMENT, PYTHONNOUSERSITE="1"),
        "installed_distributions": inventory,
        "owned_site_tree_sha256": owned_tree,
        "python": python_identity,
        "requirements_lock_bytes": len(lock_bytes),
        "requirements_lock_sha256": hashlib.sha256(lock_bytes).hexdigest(),
        "pa_runtime_build_receipt_identity_sha256": pa_identity,
        "pa_runtime_build_receipt_physical_sha256": pa_physical_sha256,
        "runtime_build_receipt_sha256": _sha256_file(build_receipt),
    }


def observed_training_runtime_payload_v2(
    *,
    requirements_lock: Path,
    runtime_build_receipt: Path,
    pa_runtime_build_receipt: Path,
    device_index: int,
) -> dict[str, Any]:
    lock = assert_no_symlink_ancestors(requirements_lock).resolve(strict=True)
    lock_bytes, lock_rows = _parse_hash_pinned_lock_v2(lock)
    if hashlib.sha256(lock_bytes).hexdigest() != TRAINING_LOCK_SHA256_V2:
        raise GTokRuntimeV2Error("training requirements lock differs from fixed A2 policy")
    build_receipt = assert_no_symlink_ancestors(runtime_build_receipt).resolve(strict=True)
    build_raw = build_receipt.read_bytes()
    try:
        build = json.loads(build_raw.decode("utf-8", errors="strict"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise GTokRuntimeV2Error("training runtime build receipt is invalid JSON") from error
    if (
        not isinstance(build, dict)
        or build_raw != canonical_json_bytes(build) + b"\n"
        or build.get("schema") != "weft1_gtok_training_runtime_build_v2"
    ):
        raise GTokRuntimeV2Error("training runtime build receipt is noncanonical or wrong schema")
    inventory = installed_record_tree_inventory_v2()
    expected_versions = {(row["name"], row["version"]) for row in lock_rows}
    observed_versions = {(row["name"], row["version"]) for row in inventory}
    if observed_versions != expected_versions:
        raise GTokRuntimeV2Error(
            "installed distribution closure differs from the fixed training lock"
        )
    owned_tree = assert_no_unowned_site_files_v2()
    python_identity = _python_identity_v2()
    if (
        build.get("requirements_lock_sha256") != hashlib.sha256(lock_bytes).hexdigest()
        or build.get("requirements_lock_bytes") != len(lock_bytes)
        or canonical_json_bytes(build.get("lock_rows"))
        != canonical_json_bytes(lock_rows)
        or canonical_json_bytes(build.get("installed_distributions"))
        != canonical_json_bytes(inventory)
        or build.get("owned_site_tree_sha256") != owned_tree
        or build.get("built_python_executable_sha256")
        != python_identity["executable_sha256"]
        or Path(str(build.get("built_python_executable"))).resolve(strict=True)
        != Path(sys.executable).resolve(strict=True)
        or Path(str(build.get("venv_prefix"))).resolve(strict=True)
        != Path(sys.prefix).resolve(strict=True)
        or Path(str(build.get("base_python_prefix"))).resolve(strict=True)
        != Path(sys.base_prefix).resolve(strict=True)
    ):
        raise GTokRuntimeV2Error("training runtime build receipt differs from live venv")
    pa_value, pa_identity, pa_physical_sha256 = _load_pa_runtime_build_receipt_v2(
        pa_runtime_build_receipt
    )
    pa_evidence = pa_value.get("evidence")
    pa_artifacts = (
        pa_evidence.get("artifacts") if isinstance(pa_evidence, Mapping) else None
    )
    if (
        not isinstance(pa_evidence, Mapping)
        or not isinstance(pa_artifacts, Mapping)
        or build.get("pa_runtime_build_receipt_identity_sha256") != pa_identity
        or build.get("pa_runtime_build_receipt_physical_sha256")
        != pa_physical_sha256
        or Path(str(pa_evidence.get("prefix"))).resolve(strict=True)
        != Path(sys.base_prefix).resolve(strict=True)
        or pa_artifacts.get("cpython_executable_sha256")
        != build.get("base_python_executable_sha256")
    ):
        raise GTokRuntimeV2Error(
            "training runtime does not descend from the exact supplied P-A runtime"
        )
    payload = {
        "closed_environment": dict(_CLOSED_ENVIRONMENT, PYTHONNOUSERSITE="1"),
        "installed_distributions": inventory,
        "owned_site_tree_sha256": owned_tree,
        "python": python_identity,
        "requirements_lock_bytes": len(lock_bytes),
        "requirements_lock_sha256": hashlib.sha256(lock_bytes).hexdigest(),
        "pa_runtime_build_receipt_identity_sha256": pa_identity,
        "pa_runtime_build_receipt_physical_sha256": pa_physical_sha256,
        "runtime_build_receipt_sha256": _sha256_file(build_receipt),
        **_torch_cuda_identity_v2(device_index=device_index),
    }
    return payload


def _enforce_fixed_runtime_policy_v2(payload: Mapping[str, Any]) -> None:
    python = payload.get("python")
    torch_row = payload.get("torch")
    cuda = payload.get("cuda")
    device = payload.get("device")
    expected_closed_environment = dict(_CLOSED_ENVIRONMENT, PYTHONNOUSERSITE="1")
    if canonical_json_bytes(payload.get("closed_environment")) != canonical_json_bytes(
        expected_closed_environment
    ):
        raise GTokRuntimeV2Error("runtime policy requires the exact closed environment")
    if not isinstance(python, Mapping) or python.get("python_version") != PYTHON_VERSION_V2:
        raise GTokRuntimeV2Error("runtime policy requires exact Python 3.11.9")
    if not isinstance(torch_row, Mapping) or torch_row.get("version") != TORCH_VERSION_V2:
        raise GTokRuntimeV2Error("runtime policy requires exact torch 2.11.0+cu128")
    if (
        not isinstance(cuda, Mapping)
        or cuda.get("compiled_version") != TORCH_CUDA_VERSION_V2
        or cuda.get("cudnn_version") != TORCH_CUDNN_VERSION_V2
    ):
        raise GTokRuntimeV2Error("runtime policy requires CUDA 12.8 and cuDNN 9.19")
    loaded = cuda.get("loaded_libraries") if isinstance(cuda, Mapping) else None
    if not isinstance(loaded, (list, tuple)) or not loaded:
        raise GTokRuntimeV2Error("runtime policy requires hashed loaded CUDA libraries")
    loaded_kinds: set[str] = set()
    for row in loaded:
        if (
            not isinstance(row, Mapping)
            or row.get("kind") not in {"cuda_driver", "cudart", "cudnn"}
            or type(row.get("bytes")) is not int
            or row["bytes"] < 1
            or not isinstance(row.get("path"), str)
            or not row["path"].startswith("/")
            or not isinstance(row.get("sha256"), str)
            or len(row["sha256"]) != 64
            or any(character not in _HEX for character in row["sha256"])
        ):
            raise GTokRuntimeV2Error("runtime policy has an invalid loaded-library row")
        loaded_kinds.add(str(row["kind"]))
    if loaded_kinds != {"cuda_driver", "cudart", "cudnn"}:
        raise GTokRuntimeV2Error("runtime policy requires driver, CUDA, and cuDNN libraries")
    if not cuda.get("driver_version"):
        raise GTokRuntimeV2Error("runtime policy requires a non-null CUDA driver")
    if (
        not isinstance(device, Mapping)
        or "A100" not in str(device.get("name", "")).upper()
        or tuple(device.get("capability", ())) != (8, 0)
    ):
        raise GTokRuntimeV2Error("runtime policy requires an NVIDIA A100 (sm80)")
    if payload.get("requirements_lock_sha256") != TRAINING_LOCK_SHA256_V2:
        raise GTokRuntimeV2Error("runtime policy requires the checked-in training lock")
    for field in (
        "pa_runtime_build_receipt_identity_sha256",
        "pa_runtime_build_receipt_physical_sha256",
        "runtime_build_receipt_sha256",
    ):
        value = payload.get(field)
        if (
            not isinstance(value, str)
            or len(value) != 64
            or any(character not in _HEX for character in value)
        ):
            raise GTokRuntimeV2Error(f"runtime policy requires exact {field}")


@dataclass(frozen=True)
class GTokTrainingRuntimeReceiptV2:
    binding_sha256: str
    environment_payload: Mapping[str, Any]
    environment_payload_sha256: str
    schema: str = RUNTIME_RECEIPT_SCHEMA_V2

    def __post_init__(self) -> None:
        for value in (self.binding_sha256, self.environment_payload_sha256):
            if len(value) != 64 or any(character not in _HEX for character in value):
                raise ValueError("training-runtime receipt hashes must be SHA-256")
        if self.environment_payload_sha256 != canonical_sha256(self.environment_payload):
            raise ValueError("training-runtime payload identity drifted")

    @property
    def receipt_sha256(self) -> str:
        return canonical_sha256(asdict(self))


def _load_binding(path: Path) -> tuple[dict[str, Any], str]:
    resolved = assert_no_symlink_ancestors(path).resolve(strict=True)
    raw = resolved.read_bytes()
    try:
        value = json.loads(raw.decode("utf-8", errors="strict"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise GTokRuntimeV2Error("training-runtime binding is not strict JSON") from error
    if not isinstance(value, dict) or set(value) != {"environment_payload", "schema"}:
        raise GTokRuntimeV2Error("training-runtime binding envelope drifted")
    if value.get("schema") != RUNTIME_BINDING_SCHEMA_V2:
        raise GTokRuntimeV2Error("training-runtime binding schema drifted")
    if raw != canonical_json_bytes(value) + b"\n":
        raise GTokRuntimeV2Error("training-runtime binding is not canonical JSON")
    return value, hashlib.sha256(raw).hexdigest()


def attest_gtok_training_runtime_v2(
    *,
    binding_path: Path,
    requirements_lock: Path,
    runtime_build_receipt: Path,
    pa_runtime_build_receipt: Path,
    device_index: int,
) -> GTokTrainingRuntimeReceiptV2:
    """Require byte-identical equality with the reviewed GPU runtime binding."""

    assert_closed_training_environment_v2()
    binding, binding_sha = _load_binding(binding_path)
    observed = observed_training_runtime_payload_v2(
        requirements_lock=requirements_lock,
        runtime_build_receipt=runtime_build_receipt,
        pa_runtime_build_receipt=pa_runtime_build_receipt,
        device_index=device_index,
    )
    _enforce_fixed_runtime_policy_v2(observed)
    expected = binding["environment_payload"]
    if canonical_json_bytes(observed) != canonical_json_bytes(expected):
        raise GTokRuntimeV2Error("observed G-TOK Python/torch/CUDA runtime differs from binding")
    return GTokTrainingRuntimeReceiptV2(
        binding_sha256=binding_sha,
        environment_payload=observed,
        environment_payload_sha256=canonical_sha256(observed),
    )


def write_runtime_binding_request_v2(
    *,
    output_path: Path,
    requirements_lock: Path,
    runtime_build_receipt: Path,
    pa_runtime_build_receipt: Path,
    device_index: int,
) -> str:
    """Write observed candidate facts without granting them execution authority."""

    assert_closed_training_environment_v2()
    observed = observed_training_runtime_payload_v2(
            requirements_lock=requirements_lock,
            runtime_build_receipt=runtime_build_receipt,
            pa_runtime_build_receipt=pa_runtime_build_receipt,
            device_index=device_index,
        )
    _enforce_fixed_runtime_policy_v2(observed)
    payload = {
        "environment_payload": observed,
        "requires_local_literal_binding": True,
        "schema": RUNTIME_REQUEST_SCHEMA_V2,
        "status": "LOCAL_BINDING_REQUIRED_NO_EXECUTION_AUTHORITY",
    }
    raw = canonical_json_bytes(payload) + b"\n"
    descriptor = os.open(output_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "wb") as handle:
        handle.write(raw)
        handle.flush()
        os.fsync(handle.fileno())
    return hashlib.sha256(raw).hexdigest()


def mint_gtok_training_runtime_binding_v2(
    *,
    output_path: Path,
    requirements_lock: Path,
    runtime_build_receipt: Path,
    pa_runtime_build_receipt: Path,
    device_index: int,
) -> str:
    """Mechanically bind one observed A100 stack under A2-R7 literal authority."""

    assert_closed_training_environment_v2()
    observed = observed_training_runtime_payload_v2(
            requirements_lock=requirements_lock,
            runtime_build_receipt=runtime_build_receipt,
            pa_runtime_build_receipt=pa_runtime_build_receipt,
            device_index=device_index,
        )
    _enforce_fixed_runtime_policy_v2(observed)
    payload = {
        "environment_payload": observed,
        "schema": RUNTIME_BINDING_SCHEMA_V2,
    }
    raw = canonical_json_bytes(payload) + b"\n"
    descriptor = os.open(output_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "wb") as handle:
        handle.write(raw)
        handle.flush()
        os.fsync(handle.fileno())
    return hashlib.sha256(raw).hexdigest()


def _parse_hash_pinned_lock_v2(path: Path) -> tuple[bytes, tuple[dict[str, Any], ...]]:
    raw = assert_no_symlink_ancestors(path).resolve(strict=True).read_bytes()
    text = raw.decode("utf-8", errors="strict")
    logical: list[str] = []
    pending = ""
    for source in text.splitlines():
        stripped = source.strip()
        if not stripped or stripped.startswith("#"):
            continue
        pending += (" " if pending else "") + stripped.removesuffix("\\").strip()
        if stripped.endswith("\\"):
            continue
        logical.append(pending)
        pending = ""
    if pending:
        raise GTokRuntimeV2Error("training requirements lock ends in a continuation")
    if any(row.startswith("--") for row in logical):
        raise GTokRuntimeV2Error("training lock may not contain option rows")
    requirements = tuple(logical)
    rows: list[dict[str, Any]] = []
    for requirement in requirements:
        head = requirement.split("--hash=sha256:", 1)[0].strip()
        match = re.fullmatch(r"([A-Za-z0-9_.-]+)==([^\s;]+)", head)
        hashes = tuple(sorted(set(re.findall(r"--hash=sha256:([0-9a-f]{64})", requirement))))
        if match is None or len(hashes) != 1:
            raise GTokRuntimeV2Error(
                "every training requirement must use exact name==version and one SHA-256 hash"
            )
        rows.append(
            {
                "artifact_sha256": hashes,
                "name": _normalized_distribution_name(match.group(1)),
                "version": match.group(2),
            }
        )
    ordered = tuple(sorted(rows, key=lambda row: row["name"]))
    if not ordered or len({row["name"] for row in ordered}) != len(ordered):
        raise GTokRuntimeV2Error("training lock closure is empty or repeats a distribution")
    return raw, ordered


def _validate_hash_pinned_lock(path: Path) -> bytes:
    """Compatibility test surface returning the validated lock bytes."""

    return _parse_hash_pinned_lock_v2(path)[0]


def _offline_wheelhouse_receipt_v2(
    wheelhouse: Path,
    lock_rows: tuple[dict[str, Any], ...],
) -> tuple[tuple[dict[str, Any], ...], Path]:
    root = assert_no_symlink_ancestors(wheelhouse).resolve(strict=True)
    declared = {
        digest
        for row in lock_rows
        for digest in row["artifact_sha256"]
    }
    files: list[dict[str, Any]] = []
    for path in sorted(root.iterdir()):
        if not path.is_file() or path.suffix != ".whl":
            raise GTokRuntimeV2Error("offline wheelhouse may contain only wheel artifacts")
        digest = _sha256_file(path)
        if digest not in declared:
            raise GTokRuntimeV2Error("offline wheelhouse contains an undeclared artifact")
        files.append(
            {
                "bytes": path.stat().st_size,
                "filename": path.name,
                "sha256": digest,
            }
        )
    if not files:
        raise GTokRuntimeV2Error("offline wheelhouse is empty")
    observed = {row["sha256"] for row in files}
    if any(not (set(row["artifact_sha256"]) & observed) for row in lock_rows):
        raise GTokRuntimeV2Error("offline wheelhouse does not close every locked distribution")
    if len(files) != len(lock_rows) or len(observed) != len(files):
        raise GTokRuntimeV2Error("offline wheelhouse requires exactly one wheel per distribution")
    pip_row = next((row for row in lock_rows if row["name"] == "pip"), None)
    pip_candidates = tuple(
        root / row["filename"]
        for row in files
        if row["filename"].lower().startswith("pip-")
        and pip_row is not None
        and row["sha256"] in set(pip_row["artifact_sha256"])
    )
    if len(pip_candidates) != 1:
        raise GTokRuntimeV2Error("offline installer chain requires one exact locked pip wheel")
    return tuple(files), pip_candidates[0]


def build_gtok_training_venv_v2(
    *,
    python_executable: Path,
    requirements_lock: Path,
    wheelhouse: Path,
    wheelhouse_receipt: Path,
    pa_runtime_build_receipt: Path,
    venv_root: Path,
    binding_path: Path | None,
) -> Path:
    """Build an isolated hash-pinned venv; authority still requires a binding."""

    lock_bytes, lock_rows = _parse_hash_pinned_lock_v2(requirements_lock)
    if hashlib.sha256(lock_bytes).hexdigest() != TRAINING_LOCK_SHA256_V2:
        raise GTokRuntimeV2Error("venv builder requires the fixed training lock")
    wheel_rows, pip_wheel = _offline_wheelhouse_receipt_v2(wheelhouse, lock_rows)
    wheelhouse_receipt_path = assert_no_symlink_ancestors(wheelhouse_receipt).resolve(
        strict=True
    )
    wheelhouse_receipt_raw = wheelhouse_receipt_path.read_bytes()
    try:
        wheelhouse_value = json.loads(
            wheelhouse_receipt_raw.decode("utf-8", errors="strict")
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise GTokRuntimeV2Error("wheelhouse receipt is invalid JSON") from error
    if (
        not isinstance(wheelhouse_value, dict)
        or wheelhouse_receipt_raw != canonical_json_bytes(wheelhouse_value) + b"\n"
        or wheelhouse_value.get("schema") != "weft1_gtok_training_wheelhouse_v2"
        or wheelhouse_value.get("lock_sha256") != hashlib.sha256(lock_bytes).hexdigest()
        or canonical_json_bytes(wheelhouse_value.get("wheels"))
        != canonical_json_bytes(wheel_rows)
    ):
        raise GTokRuntimeV2Error("wheelhouse receipt differs from exact offline closure")
    python = assert_no_symlink_ancestors(python_executable).resolve(strict=True)
    if venv_root.exists() or venv_root.is_symlink():
        raise FileExistsError("G-TOK training venv root must be new")
    environment = closed_training_environment_v2()
    base_probe = json.loads(subprocess.run(
        [
            str(python),
            "-I",
            "-B",
            "-c",
            "import json,platform,sys;print(json.dumps({'version':platform.python_version(),'prefix':sys.prefix,'base_prefix':sys.base_prefix}))",
        ],
        check=True,
        capture_output=True,
        text=True,
        env=environment,
    ).stdout)
    if base_probe["version"] != PYTHON_VERSION_V2:
        raise GTokRuntimeV2Error("venv builder requires exact Python 3.11.9")
    pa_value, pa_identity, pa_physical_sha256 = _load_pa_runtime_build_receipt_v2(
        pa_runtime_build_receipt
    )
    pa_evidence = pa_value.get("evidence")
    pa_artifacts = (
        pa_evidence.get("artifacts") if isinstance(pa_evidence, Mapping) else None
    )
    if (
        not isinstance(pa_evidence, Mapping)
        or not isinstance(pa_artifacts, Mapping)
        or Path(str(pa_evidence.get("prefix"))).resolve(strict=True)
        != Path(base_probe["prefix"]).resolve(strict=True)
        or pa_artifacts.get("cpython_executable_sha256")
        != _sha256_file(python)
    ):
        raise GTokRuntimeV2Error("builder Python differs from exact P-A runtime receipt")
    subprocess.run(
        [
            str(python),
            "-I",
            "-B",
            "-m",
            "venv",
            "--copies",
            "--without-pip",
            str(venv_root),
        ],
        check=True,
        env=environment,
    )
    built_python = venv_root / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
    if not built_python.is_file() or built_python.is_symlink():
        raise GTokRuntimeV2Error("venv interpreter must be a physical --copies executable")
    bootstrap = (
        "import runpy,sys;"
        f"sys.path.insert(0,{str(pip_wheel)!r});"
        "sys.argv=['pip']+sys.argv[1:];"
        "runpy.run_module('pip',run_name='__main__')"
    )
    subprocess.run(
        [
            str(built_python),
            "-I",
            "-B",
            "-c",
            bootstrap,
            "install",
            "--require-hashes",
            "--no-deps",
            "--no-index",
            "--no-compile",
            "--disable-pip-version-check",
            "--find-links",
            str(wheelhouse.resolve(strict=True)),
            "-r",
            str(requirements_lock.resolve(strict=True)),
        ],
        check=True,
        env=environment,
    )
    repository_root = Path(__file__).resolve().parents[1]
    probe_code = (
        "import json,sys;"
        f"sys.path.insert(0,{str(repository_root)!r});"
        "from training.weft1_gtok_runtime_v2 import "
        "installed_record_tree_inventory_v2,assert_no_unowned_site_files_v2;"
        "print(json.dumps({'base_prefix':sys.base_prefix,'executable':sys.executable,'prefix':sys.prefix,"
        "'inventory':installed_record_tree_inventory_v2(),"
        "'owned_site_tree_sha256':assert_no_unowned_site_files_v2()},"
        "sort_keys=True,separators=(',',':')))"
    )
    probe = json.loads(
        subprocess.run(
            [str(built_python), "-I", "-B", "-c", probe_code],
            check=True,
            capture_output=True,
            text=True,
            env=environment,
        ).stdout
    )
    if Path(probe["prefix"]) != venv_root.resolve(strict=True):
        raise GTokRuntimeV2Error("built interpreter prefix escaped the lexical venv")
    if Path(probe["base_prefix"]).resolve(strict=True) != Path(
        base_probe["prefix"]
    ).resolve(strict=True):
        raise GTokRuntimeV2Error("built interpreter is not linked to the exact builder base")
    expected_versions = {(row["name"], row["version"]) for row in lock_rows}
    observed_versions = {
        (row["name"], row["version"]) for row in probe["inventory"]
    }
    if observed_versions != expected_versions:
        raise GTokRuntimeV2Error("installed distribution closure differs from exact lock")
    build_payload = {
        "base_python_executable_sha256": _sha256_file(python),
        "base_python_prefix": str(Path(base_probe["prefix"]).resolve(strict=True)),
        "built_python_executable": str(built_python.absolute()),
        "built_python_executable_sha256": _sha256_file(built_python),
        "installed_distributions": probe["inventory"],
        "installer_chain": {
            "bootstrap_pip_wheel": pip_wheel.name,
            "bootstrap_pip_wheel_sha256": _sha256_file(pip_wheel),
            "offline_no_deps_require_hashes": True,
        },
        "lock_rows": lock_rows,
        "owned_site_tree_sha256": probe["owned_site_tree_sha256"],
        "pa_runtime_build_receipt_identity_sha256": pa_identity,
        "pa_runtime_build_receipt_physical_sha256": pa_physical_sha256,
        "requirements_lock_bytes": len(lock_bytes),
        "requirements_lock_sha256": hashlib.sha256(lock_bytes).hexdigest(),
        "schema": "weft1_gtok_training_runtime_build_v2",
        "venv_prefix": str(venv_root.resolve(strict=True)),
        "wheelhouse": wheel_rows,
        "wheelhouse_receipt_sha256": hashlib.sha256(wheelhouse_receipt_raw).hexdigest(),
    }
    build_path = venv_root / "runtime-build-receipt.json"
    build_raw = canonical_json_bytes(build_payload) + b"\n"
    descriptor = os.open(build_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "wb") as handle:
        handle.write(build_raw)
        handle.flush()
        os.fsync(handle.fileno())
    # Physical GPU attestation occurs inside this interpreter.  Building alone
    # never grants campaign authority; A2-R7 can mechanically bind it later.
    if binding_path is not None:
        _load_binding(binding_path)
    return built_python.absolute()


__all__ = [
    "GTokRuntimeBindingRequiredV2",
    "GTokRuntimeV2Error",
    "GTokTrainingRuntimeReceiptV2",
    "PYTHON_VERSION_V2",
    "RUNTIME_BINDING_SCHEMA_V2",
    "attest_gtok_training_runtime_v2",
    "assert_closed_training_environment_v2",
    "build_gtok_training_venv_v2",
    "closed_training_environment_v2",
    "cpu_runtime_identity_sha256_from_payload_v2",
    "gpu_uuid_provenance_v2",
    "installed_record_tree_inventory_v2",
    "mint_gtok_training_runtime_binding_v2",
    "observed_training_runtime_payload_v2",
    "observed_training_cpu_runtime_payload_v2",
    "write_runtime_binding_request_v2",
]
