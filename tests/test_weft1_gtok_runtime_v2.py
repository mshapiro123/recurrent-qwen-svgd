from __future__ import annotations

import hashlib
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

from training.weft1_gtok_contract import canonical_json_bytes
import training.weft1_gtok_runtime_v2 as runtime


def _payload(torch_version: str = "2.11.0+cu128") -> dict:
    return {
        "closed_environment": dict(
            runtime._CLOSED_ENVIRONMENT, PYTHONNOUSERSITE="1"
        ),
        "installed_distributions": (
            {
                "name": "torch",
                "record_file_count": 1,
                "record_tree_sha256": "1" * 64,
                "version": torch_version,
            },
        ),
        "owned_site_tree_sha256": "b" * 64,
        "python": {
            "abi_flags": "",
            "base_prefix": "/python",
            "executable_sha256": "2" * 64,
            "implementation": "CPython",
            "platform": "manylinux",
            "prefix": "/venv",
            "python_version": "3.11.9",
            "soabi": "cpython-311-x86_64-linux-gnu",
        },
        "requirements_lock_bytes": 10,
        "requirements_lock_sha256": runtime.TRAINING_LOCK_SHA256_V2,
        "runtime_build_receipt_sha256": "5" * 64,
        "pa_runtime_build_receipt_identity_sha256": "6" * 64,
        "pa_runtime_build_receipt_physical_sha256": "7" * 64,
        "cuda": {
            "compiled_version": "12.8",
            "cudnn_version": 91_900,
            "driver_version": 12080,
            "loaded_libraries": (
                {
                    "bytes": 1,
                    "kind": "cuda_driver",
                    "path": "/usr/lib/libcuda.so.1",
                    "sha256": "8" * 64,
                },
                {
                    "bytes": 1,
                    "kind": "cudart",
                    "path": "/venv/lib/libcudart.so.12",
                    "sha256": "9" * 64,
                },
                {
                    "bytes": 1,
                    "kind": "cudnn",
                    "path": "/venv/lib/libcudnn.so.9",
                    "sha256": "a" * 64,
                },
            ),
        },
        "device": {
            "capability": (8, 0),
            "name": "NVIDIA A100-SXM4-80GB",
            "total_memory_bytes": 85_000_000_000,
        },
        "torch": {
            "debug_build": False,
            "git_version": "4" * 40,
            "version": torch_version,
        },
    }


def _binding(tmp_path: Path, payload: dict) -> Path:
    path = tmp_path / "runtime-binding.json"
    path.write_bytes(
        canonical_json_bytes(
            {
                "environment_payload": payload,
                "schema": runtime.RUNTIME_BINDING_SCHEMA_V2,
            }
        )
        + b"\n"
    )
    return path


def test_closed_environment_rejects_ambient_pythonpath_and_wrong_locale() -> None:
    good = runtime.closed_training_environment_v2(
        {
            "PYTHONPATH": "/ambient",
            "PYTHONHOME": "/wrong",
            "LANG": "wrong",
        }
    )
    assert "PYTHONPATH" not in good
    assert "PYTHONHOME" not in good
    assert good["LANG"] == "C.UTF-8"
    runtime.assert_closed_training_environment_v2(good)
    with pytest.raises(runtime.GTokRuntimeV2Error, match="forbidden"):
        runtime.assert_closed_training_environment_v2(dict(good, PYTHONPATH="/ambient"))


def test_cpu_precompute_attestation_rejects_live_ambient_env_before_files(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    for name, value in runtime._CLOSED_ENVIRONMENT.items():
        monkeypatch.setenv(name, value)
    monkeypatch.setenv("PYTHONNOUSERSITE", "1")
    for name in runtime._REMOVED_ENVIRONMENT:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("PYTHONPATH", "/ambient/hostile")
    absent = tmp_path / "must-not-be-opened"
    with pytest.raises(runtime.GTokRuntimeV2Error, match="forbidden"):
        runtime.observed_training_cpu_runtime_payload_v2(
            requirements_lock=absent,
            runtime_build_receipt=absent,
            pa_runtime_build_receipt=absent,
        )


def test_cpu_precompute_identity_binds_interpreter_and_record_tree_not_gpu() -> None:
    first = _payload()
    second = _payload()
    second["device"] = dict(second["device"], name="NVIDIA A100-PCIE-40GB")
    second["cuda"] = dict(second["cuda"], driver_version=12090)
    assert (
        runtime.cpu_runtime_identity_sha256_from_payload_v2(first)
        == runtime.cpu_runtime_identity_sha256_from_payload_v2(second)
    )
    second["installed_distributions"] = (
        dict(second["installed_distributions"][0], record_tree_sha256="c" * 64),
    )
    assert (
        runtime.cpu_runtime_identity_sha256_from_payload_v2(first)
        != runtime.cpu_runtime_identity_sha256_from_payload_v2(second)
    )


def test_exact_runtime_attestation_rejects_wrong_torch_before_return(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    expected = _payload()
    binding = _binding(tmp_path, expected)
    lock = tmp_path / "lock.txt"
    lock.write_bytes(b"0123456789")
    monkeypatch.setattr(runtime, "assert_closed_training_environment_v2", lambda: None)
    monkeypatch.setattr(
        runtime,
        "observed_training_runtime_payload_v2",
        lambda **_: _payload("2.13.0+cu128"),
    )
    with pytest.raises(runtime.GTokRuntimeV2Error, match="exact torch"):
        runtime.attest_gtok_training_runtime_v2(
            binding_path=binding,
            requirements_lock=lock,
            runtime_build_receipt=lock,
            pa_runtime_build_receipt=lock,
            device_index=0,
        )


def test_exact_runtime_attestation_mints_one_identity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    payload = _payload()
    binding = _binding(tmp_path, payload)
    lock = tmp_path / "lock.txt"
    lock.write_bytes(b"0123456789")
    monkeypatch.setattr(runtime, "assert_closed_training_environment_v2", lambda: None)
    monkeypatch.setattr(
        runtime,
        "observed_training_runtime_payload_v2",
        lambda **_: payload,
    )
    receipt = runtime.attest_gtok_training_runtime_v2(
        binding_path=binding,
        requirements_lock=lock,
        runtime_build_receipt=lock,
        pa_runtime_build_receipt=lock,
        device_index=0,
    )
    assert receipt.binding_sha256 == hashlib.sha256(binding.read_bytes()).hexdigest()
    assert receipt.environment_payload_sha256 == runtime.canonical_sha256(payload)


def test_cuda_ordinal_is_operational_but_a100_stack_identity_remains_exact(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    payload = _payload()
    binding = _binding(tmp_path, payload)
    placeholder = tmp_path / "placeholder"
    placeholder.write_bytes(b"x")
    observed_indices: list[int] = []

    def observed(**kwargs):
        observed_indices.append(kwargs["device_index"])
        return _payload()

    monkeypatch.setattr(runtime, "assert_closed_training_environment_v2", lambda: None)
    monkeypatch.setattr(runtime, "observed_training_runtime_payload_v2", observed)
    first = runtime.attest_gtok_training_runtime_v2(
        binding_path=binding,
        requirements_lock=placeholder,
        runtime_build_receipt=placeholder,
        pa_runtime_build_receipt=placeholder,
        device_index=0,
    )
    second = runtime.attest_gtok_training_runtime_v2(
        binding_path=binding,
        requirements_lock=placeholder,
        runtime_build_receipt=placeholder,
        pa_runtime_build_receipt=placeholder,
        device_index=1,
    )
    assert observed_indices == [0, 1]
    assert first == second
    drifted = _payload()
    drifted["device"] = dict(
        drifted["device"],
        name="NVIDIA A100-PCIE-40GB",
        total_memory_bytes=42_000_000_000,
    )
    monkeypatch.setattr(
        runtime,
        "observed_training_runtime_payload_v2",
        lambda **_: drifted,
    )
    with pytest.raises(runtime.GTokRuntimeV2Error, match="differs from binding"):
        runtime.attest_gtok_training_runtime_v2(
            binding_path=binding,
            requirements_lock=placeholder,
            runtime_build_receipt=placeholder,
            pa_runtime_build_receipt=placeholder,
            device_index=1,
        )


def test_live_shaped_cuda_probe_preserves_a100_name_through_library_scan(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cuda = SimpleNamespace(
        is_available=lambda: True,
        device_count=lambda: 2,
        get_device_properties=lambda _device: SimpleNamespace(
            total_memory=85_000_000_000
        ),
        get_device_name=lambda _device: "NVIDIA A100-SXM4-80GB",
        get_device_capability=lambda _device: (8, 0),
        synchronize=lambda _device: None,
    )
    fake_torch = SimpleNamespace(
        __version__=runtime.TORCH_VERSION_V2,
        _C=SimpleNamespace(_cuda_getDriverVersion=lambda: 12080),
        backends=SimpleNamespace(cudnn=SimpleNamespace(version=lambda: 91_900)),
        cuda=cuda,
        device=lambda value: value,
        nn=SimpleNamespace(functional=SimpleNamespace(conv2d=lambda *_: None)),
        ones=lambda *_args, **_kwargs: object(),
        zeros=lambda *_args, **_kwargs: object(),
        version=SimpleNamespace(
            cuda=runtime.TORCH_CUDA_VERSION_V2,
            debug=False,
            git_version="4" * 40,
        ),
    )
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    libraries = _payload()["cuda"]["loaded_libraries"]
    monkeypatch.setattr(
        runtime, "_loaded_cuda_library_identity_v2", lambda: libraries
    )
    identity = runtime._torch_cuda_identity_v2(device_index=1)
    assert identity["device"] == {
        "capability": (8, 0),
        "name": "NVIDIA A100-SXM4-80GB",
        "total_memory_bytes": 85_000_000_000,
    }
    assert "index" not in identity["device"]
    payload = _payload()
    payload.update(identity)
    runtime._enforce_fixed_runtime_policy_v2(payload)


def test_hash_lock_rejects_unhashed_requirement(tmp_path: Path) -> None:
    lock = tmp_path / "bad-lock.txt"
    lock.write_text("torch==2.8.0\n", encoding="utf-8")
    with pytest.raises(runtime.GTokRuntimeV2Error, match="one SHA-256 hash"):
        runtime._validate_hash_pinned_lock(lock)


def test_runtime_policy_rejects_missing_loaded_driver_library() -> None:
    payload = _payload()
    payload["cuda"]["loaded_libraries"] = tuple(
        row
        for row in payload["cuda"]["loaded_libraries"]
        if row["kind"] != "cuda_driver"
    )
    with pytest.raises(runtime.GTokRuntimeV2Error, match="driver, CUDA, and cuDNN"):
        runtime._enforce_fixed_runtime_policy_v2(payload)


def test_live_runtime_rejects_pa_receipt_substitution(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    lock = tmp_path / "lock.txt"
    lock.write_text(
        f"torch=={runtime.TORCH_VERSION_V2} --hash=sha256:{'0' * 64}\n",
        encoding="utf-8",
        newline="\n",
    )
    lock_sha = hashlib.sha256(lock.read_bytes()).hexdigest()
    monkeypatch.setattr(runtime, "TRAINING_LOCK_SHA256_V2", lock_sha)
    _lock_bytes, lock_rows = runtime._parse_hash_pinned_lock_v2(lock)
    python_identity = runtime._python_identity_v2()
    inventory = (
        {
            "name": "torch",
            "record_file_count": 1,
            "record_tree_sha256": "1" * 64,
            "version": runtime.TORCH_VERSION_V2,
        },
    )
    build = {
        "base_python_executable_sha256": "f" * 64,
        "base_python_prefix": python_identity["base_prefix"],
        "built_python_executable": str(Path(sys.executable).resolve()),
        "built_python_executable_sha256": python_identity["executable_sha256"],
        "installed_distributions": inventory,
        "lock_rows": lock_rows,
        "owned_site_tree_sha256": "2" * 64,
        "pa_runtime_build_receipt_identity_sha256": "3" * 64,
        "pa_runtime_build_receipt_physical_sha256": "4" * 64,
        "requirements_lock_bytes": len(lock.read_bytes()),
        "requirements_lock_sha256": lock_sha,
        "schema": "weft1_gtok_training_runtime_build_v2",
        "venv_prefix": python_identity["prefix"],
    }
    build_path = tmp_path / "runtime-build.json"
    build_path.write_bytes(canonical_json_bytes(build) + b"\n")
    monkeypatch.setattr(runtime, "installed_record_tree_inventory_v2", lambda: inventory)
    monkeypatch.setattr(runtime, "assert_no_unowned_site_files_v2", lambda: "2" * 64)
    monkeypatch.setattr(runtime, "_python_identity_v2", lambda: python_identity)
    monkeypatch.setattr(
        runtime,
        "_load_pa_runtime_build_receipt_v2",
        lambda _path: (
            {"evidence": {"artifacts": {"cpython_executable_sha256": "f" * 64}, "prefix": python_identity["base_prefix"]}},
            "3" * 64,
            "5" * 64,
        ),
    )
    with pytest.raises(runtime.GTokRuntimeV2Error, match="exact supplied P-A"):
        runtime.observed_training_runtime_payload_v2(
            requirements_lock=lock,
            runtime_build_receipt=build_path,
            pa_runtime_build_receipt=tmp_path,
            device_index=0,
        )


def test_live_runtime_rejects_ambient_extra_distribution_before_gpu_probe(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    lock = tmp_path / "lock.txt"
    lock.write_text(
        f"torch=={runtime.TORCH_VERSION_V2} --hash=sha256:{'0' * 64}\n",
        encoding="utf-8",
        newline="\n",
    )
    monkeypatch.setattr(
        runtime, "TRAINING_LOCK_SHA256_V2", hashlib.sha256(lock.read_bytes()).hexdigest()
    )
    monkeypatch.setattr(
        runtime,
        "installed_record_tree_inventory_v2",
        lambda: (
            {
                "name": "torch",
                "record_file_count": 1,
                "record_tree_sha256": "1" * 64,
                "version": runtime.TORCH_VERSION_V2,
            },
            {
                "name": "ambient-extra",
                "record_file_count": 1,
                "record_tree_sha256": "2" * 64,
                "version": "1.0",
            },
        ),
    )
    monkeypatch.setattr(
        runtime,
        "_torch_cuda_identity_v2",
        lambda **_: (_ for _ in ()).throw(AssertionError("GPU probe must not run")),
    )
    build = tmp_path / "runtime-build.json"
    build.write_bytes(
        canonical_json_bytes(
            {"schema": "weft1_gtok_training_runtime_build_v2"}
        )
        + b"\n"
    )
    with pytest.raises(runtime.GTokRuntimeV2Error, match="installed distribution closure"):
        runtime.observed_training_runtime_payload_v2(
            requirements_lock=lock,
            runtime_build_receipt=build,
            pa_runtime_build_receipt=tmp_path,
            device_index=0,
        )
