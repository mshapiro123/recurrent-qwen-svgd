"""Parent-probed Linux network isolation for authoritative G-TOK execution."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import errno
import hashlib
import json
import os
from pathlib import Path
import socket
import sys
from typing import Any, Mapping

from training.weft1_corpus_replay_a2 import (
    LINUX_UNSHARE_PATH_V1,
    LINUX_UNSHARE_SHA256_V1,
)
from training.weft1_gtok_contract import canonical_json_bytes, canonical_sha256
from training.weft1_strict_io import assert_no_symlink_ancestors


OFFLINE_RECEIPT_SCHEMA_V2 = "weft1_gtok_offline_parent_launch_v2"
OFFLINE_RECEIPT_ENV_V2 = "WEFT1_GTOK_OFFLINE_PARENT_RECEIPT_SHA256"
_HEX = frozenset("0123456789abcdef")


class GTokOfflineV2Error(RuntimeError):
    """The authoritative P-C process is not inside the bound network namespace."""


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def linux_network_namespace_v2() -> str:
    if os.name != "posix" or sys.platform != "linux":
        raise GTokOfflineV2Error("authoritative P-C requires Linux network namespaces")
    try:
        identity = os.readlink("/proc/self/ns/net")
    except OSError as error:
        raise GTokOfflineV2Error("current Linux network namespace is unavailable") from error
    if not identity.startswith("net:[") or not identity.endswith("]"):
        raise GTokOfflineV2Error("current Linux network namespace identity is malformed")
    return identity


@dataclass(frozen=True)
class OfflineParentLaunchReceiptV2:
    parent_network_namespace: str
    unshare_executable: str
    unshare_executable_sha256: str
    python_executable: str
    python_executable_sha256: str
    campaign_script: str
    campaign_script_sha256: str
    status: str = "PARENT_PROBED_UNSHARE_NET_READY"
    schema: str = OFFLINE_RECEIPT_SCHEMA_V2

    def __post_init__(self) -> None:
        if not self.parent_network_namespace.startswith("net:["):
            raise ValueError("offline receipt lacks the parent network namespace")
        if self.unshare_executable != str(LINUX_UNSHARE_PATH_V1):
            raise ValueError("offline receipt uses a different unshare executable")
        if self.unshare_executable_sha256 != LINUX_UNSHARE_SHA256_V1:
            raise ValueError("offline receipt uses a different A2 unshare identity")
        for name in (
            "python_executable_sha256",
            "campaign_script_sha256",
        ):
            value = getattr(self, name)
            if len(value) != 64 or any(character not in _HEX for character in value):
                raise ValueError(f"{name} must be SHA-256")
        if self.status != "PARENT_PROBED_UNSHARE_NET_READY":
            raise ValueError("offline parent probe did not pass")

    @property
    def receipt_sha256(self) -> str:
        return canonical_sha256(self)

    @property
    def policy_sha256(self) -> str:
        """Stable isolation policy identity, excluding the launch netns inode."""

        return canonical_sha256(
            {
                "campaign_script": self.campaign_script,
                "campaign_script_sha256": self.campaign_script_sha256,
                "python_executable": self.python_executable,
                "python_executable_sha256": self.python_executable_sha256,
                "schema": self.schema,
                "status": self.status,
                "unshare_executable": self.unshare_executable,
                "unshare_executable_sha256": self.unshare_executable_sha256,
            }
        )


def load_offline_parent_receipt_v2(path: Path) -> tuple[OfflineParentLaunchReceiptV2, str]:
    resolved = assert_no_symlink_ancestors(path).resolve(strict=True)
    raw = resolved.read_bytes()
    try:
        value = json.loads(raw.decode("utf-8", errors="strict"))
        receipt = OfflineParentLaunchReceiptV2(**value)
    except (UnicodeDecodeError, json.JSONDecodeError, TypeError, ValueError) as error:
        raise GTokOfflineV2Error("offline parent receipt is invalid") from error
    if raw != canonical_json_bytes(asdict(receipt)) + b"\n":
        raise GTokOfflineV2Error("offline parent receipt is not canonical JSON")
    return receipt, hashlib.sha256(raw).hexdigest()


def _assert_offline_descendant_v2(
    receipt: OfflineParentLaunchReceiptV2,
    *,
    physical_sha256: str,
) -> None:
    """Verify inherited network isolation without assuming ``sys.argv[0]``.

    Tokenizer fit workers are fresh ``python -I -B -c`` descendants of the
    parent-verified tokenizer command.  They inherit its network namespace, but
    their ``sys.argv[0]`` is necessarily ``-c`` rather than the campaign script.
    This shared verifier therefore binds the physical parent receipt,
    interpreter, exact ``unshare`` binary, namespace transition, and a live
    network-unreachable probe.  The top-level campaign verifier adds the script
    identity check below.
    """

    if os.environ.get(OFFLINE_RECEIPT_ENV_V2) != physical_sha256:
        raise GTokOfflineV2Error("offline receipt was not bound by the parent launcher")
    unshare = assert_no_symlink_ancestors(Path(receipt.unshare_executable)).resolve(
        strict=True
    )
    if unshare != LINUX_UNSHARE_PATH_V1 or _sha256_file(unshare) != LINUX_UNSHARE_SHA256_V1:
        raise GTokOfflineV2Error("live unshare executable differs from the A2 binding")
    executable = Path(sys.executable).resolve(strict=True)
    if (
        str(executable) != receipt.python_executable
        or _sha256_file(executable) != receipt.python_executable_sha256
    ):
        raise GTokOfflineV2Error("offline child interpreter differs from its parent receipt")
    if linux_network_namespace_v2() == receipt.parent_network_namespace:
        raise GTokOfflineV2Error("campaign remained in the parent network namespace")
    probe = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    probe.settimeout(1.0)
    try:
        probe.connect(("1.1.1.1", 53))
    except OSError as error:
        if error.errno not in (errno.ENETUNREACH, errno.EHOSTUNREACH):
            raise GTokOfflineV2Error(
                "isolated child failed for a reason other than network unreachability"
            ) from error
    else:
        raise GTokOfflineV2Error("authoritative P-C child can still reach the network")
    finally:
        probe.close()


def assert_offline_descendant_v2(path: Path) -> str:
    """Fail unless a subprocess inherited the exact verified offline launch."""

    receipt, physical_sha256 = load_offline_parent_receipt_v2(path)
    _assert_offline_descendant_v2(receipt, physical_sha256=physical_sha256)
    return physical_sha256


def assert_offline_campaign_child_v2(path: Path) -> str:
    """Fail unless this process is the exact parent-launched unshare child."""

    receipt, physical_sha256 = load_offline_parent_receipt_v2(path)
    script = Path(sys.argv[0]).resolve(strict=True)
    if (
        str(script) != receipt.campaign_script
        or _sha256_file(script) != receipt.campaign_script_sha256
    ):
        raise GTokOfflineV2Error("offline child campaign source differs from its parent receipt")
    _assert_offline_descendant_v2(receipt, physical_sha256=physical_sha256)
    return physical_sha256


__all__ = [
    "GTokOfflineV2Error",
    "OFFLINE_RECEIPT_ENV_V2",
    "OFFLINE_RECEIPT_SCHEMA_V2",
    "OfflineParentLaunchReceiptV2",
    "assert_offline_campaign_child_v2",
    "assert_offline_descendant_v2",
    "linux_network_namespace_v2",
    "load_offline_parent_receipt_v2",
]
