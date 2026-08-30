from __future__ import annotations

from dataclasses import asdict
import errno
import hashlib
import os
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

from training.weft1_gtok_contract import canonical_json_bytes
import training.weft1_gtok_offline_v2 as offline
from scripts import launch_weft1_gtok_offline_v2 as launcher


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _receipt(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[Path, str]:
    unshare = tmp_path / "unshare"
    unshare.write_bytes(b"unshare-binary")
    monkeypatch.setattr(offline, "LINUX_UNSHARE_PATH_V1", unshare.resolve())
    monkeypatch.setattr(offline, "LINUX_UNSHARE_SHA256_V1", _sha(unshare))
    python = Path(sys.executable).resolve()
    script = Path(sys.argv[0]).resolve()
    receipt = offline.OfflineParentLaunchReceiptV2(
        parent_network_namespace="net:[100]",
        unshare_executable=str(unshare.resolve()),
        unshare_executable_sha256=_sha(unshare),
        python_executable=str(python),
        python_executable_sha256=_sha(python),
        campaign_script=str(script),
        campaign_script_sha256=_sha(script),
    )
    path = tmp_path / "offline.json"
    raw = canonical_json_bytes(asdict(receipt)) + b"\n"
    path.write_bytes(raw)
    return path, hashlib.sha256(raw).hexdigest()


class _UnreachableSocket:
    def settimeout(self, _timeout: float) -> None:
        return None

    def connect(self, _address: object) -> None:
        raise OSError(errno.ENETUNREACH, "Network is unreachable")

    def close(self) -> None:
        return None


def test_offline_child_requires_parent_receipt_namespace_and_errno101(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path, physical_sha = _receipt(tmp_path, monkeypatch)
    monkeypatch.setenv(offline.OFFLINE_RECEIPT_ENV_V2, physical_sha)
    monkeypatch.setattr(offline, "linux_network_namespace_v2", lambda: "net:[101]")
    monkeypatch.setattr(offline.socket, "socket", lambda *_: _UnreachableSocket())
    assert offline.assert_offline_campaign_child_v2(path) == physical_sha


def test_offline_child_rejects_direct_same_namespace_before_network_probe(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path, physical_sha = _receipt(tmp_path, monkeypatch)
    monkeypatch.setenv(offline.OFFLINE_RECEIPT_ENV_V2, physical_sha)
    monkeypatch.setattr(offline, "linux_network_namespace_v2", lambda: "net:[100]")
    monkeypatch.setattr(
        offline.socket,
        "socket",
        lambda *_: (_ for _ in ()).throw(AssertionError("probe must not run")),
    )
    with pytest.raises(offline.GTokOfflineV2Error, match="parent network namespace"):
        offline.assert_offline_campaign_child_v2(path)


def test_offline_child_rejects_forged_or_mutated_parent_receipt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path, physical_sha = _receipt(tmp_path, monkeypatch)
    monkeypatch.setenv(offline.OFFLINE_RECEIPT_ENV_V2, physical_sha)
    path.write_bytes(path.read_bytes().replace(b"net:[100]", b"net:[999]"))
    with pytest.raises(offline.GTokOfflineV2Error, match="not bound"):
        offline.assert_offline_campaign_child_v2(path)


def test_offline_launcher_strips_exact_separator_and_appends_receipt_last(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    unshare = tmp_path / "unshare"
    unshare.write_bytes(b"unshare")
    script = (Path(launcher.__file__).resolve().parent / "run_weft1_gtok_v2.py").resolve(
        strict=True
    )
    python = Path(sys.executable).resolve()
    receipt = tmp_path / "offline-parent.json"
    unshare_sha256 = _sha(unshare)
    monkeypatch.setattr(launcher, "LINUX_UNSHARE_PATH_V1", unshare.resolve())
    monkeypatch.setattr(offline, "LINUX_UNSHARE_PATH_V1", unshare.resolve())
    monkeypatch.setattr(offline, "LINUX_UNSHARE_SHA256_V1", unshare_sha256)
    monkeypatch.setattr(
        launcher,
        "_resolve_unshare_executable",
        lambda _path: unshare.resolve(),
    )
    monkeypatch.setattr(
        launcher,
        "_verify_unshare_network_isolation",
        lambda **_: None,
    )
    monkeypatch.setattr(launcher, "linux_network_namespace_v2", lambda: "net:[7]")
    monkeypatch.setattr(
        launcher,
        "closed_training_environment_v2",
        lambda: {"CUBLAS_WORKSPACE_CONFIG": ":4096:8"},
    )
    seen: dict[str, object] = {}

    def run(command, **kwargs):
        seen["command"] = command
        seen["kwargs"] = kwargs
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(launcher.subprocess, "run", run)
    assert launcher.main(
        [
            "--python-executable",
            str(python),
            "--campaign-script",
            str(script),
            "--offline-receipt",
            str(receipt),
            "--",
            *(
                child_arguments := [
                    "--corpus-root",
                    "/durable/corpus",
                    "--freeze-receipt",
                    "/durable/freeze.json",
                    "--gate-bundle",
                    "/durable/gates.json",
                    "--c2-evidence",
                    "/durable/c2.json",
                    "--decon-receipt",
                    "/durable/decon.json",
                    "--dependency-lock",
                    "/durable/requirements.lock",
                    "--worker-executable",
                    str(python),
                    "fit-all",
                    "--output-root",
                    "/durable/tokenizers",
                ]
            ),
        ]
    ) == 0
    command = seen["command"]
    assert command[:5] == [str(unshare.resolve()), "--net", "--", str(python), "-I"]
    assert command[5:7] == ["-B", str(script.resolve())]
    assert command[7:] == [
        *child_arguments,
        "--offline-network-receipt",
        str(receipt.resolve()),
    ]
    assert seen["kwargs"]["env"]["CUBLAS_WORKSPACE_CONFIG"] == ":4096:8"


@pytest.mark.parametrize("tail", [[], ["--"], ["--", "--", "--child"]])
def test_offline_launcher_rejects_missing_empty_or_double_separator(
    tmp_path: Path, tail: list[str]
) -> None:
    with pytest.raises(ValueError, match="separator|nonempty"):
        launcher.main(
            [
                "--python-executable",
                str(Path(sys.executable)),
                "--campaign-script",
                str(tmp_path / "unused.py"),
                "--offline-receipt",
                str(tmp_path / "unused.json"),
                *tail,
            ]
        )
