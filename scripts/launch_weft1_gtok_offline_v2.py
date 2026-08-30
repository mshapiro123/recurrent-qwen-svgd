#!/usr/bin/env python3
"""Parent-probe and launch the authoritative full G-TOK CLI under unshare --net."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import os
from pathlib import Path
import subprocess
import sys


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from training.weft1_corpus_replay_a2 import (  # noqa: E402
    LINUX_UNSHARE_PATH_V1,
    _resolve_unshare_executable,
    _verify_unshare_network_isolation,
)
from training.weft1_gtok_contract import canonical_json_bytes  # noqa: E402
from training.weft1_gtok_offline_v2 import (  # noqa: E402
    OFFLINE_RECEIPT_ENV_V2,
    OfflineParentLaunchReceiptV2,
    linux_network_namespace_v2,
)
from training.weft1_gtok_runtime_v2 import closed_training_environment_v2  # noqa: E402
from training.weft1_strict_io import assert_no_symlink_ancestors  # noqa: E402


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _write_or_validate(path: Path, payload: bytes) -> tuple[Path, str]:
    """Persist every launch receipt without overwriting an older VM's netns."""

    physical_sha256 = hashlib.sha256(payload).hexdigest()
    if path.exists() or path.is_symlink():
        if path.is_symlink() or not path.is_file():
            raise RuntimeError("offline parent-launch receipt path is not a regular file")
        if path.read_bytes() == payload:
            return path.resolve(strict=True), physical_sha256
        path = path.with_name(
            f"{path.stem}.{physical_sha256}{path.suffix or '.json'}"
        )
        if path.exists() or path.is_symlink():
            if path.is_symlink() or not path.is_file() or path.read_bytes() != payload:
                raise RuntimeError("offline parent-launch hash path collided")
            return path.resolve(strict=True), physical_sha256
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "wb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    return path.resolve(strict=True), physical_sha256


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--python-executable", type=Path, required=True)
    parser.add_argument("--campaign-script", type=Path, required=True)
    parser.add_argument("--offline-receipt", type=Path, required=True)
    parser.add_argument("campaign_arguments", nargs=argparse.REMAINDER)
    arguments = parser.parse_args(argv)
    remainder = list(arguments.campaign_arguments)
    if not remainder or remainder[0] != "--":
        raise ValueError("offline launcher requires exactly one -- child separator")
    remainder = remainder[1:]
    if not remainder or remainder[0] == "--" or "--offline-network-receipt" in remainder:
        raise ValueError(
            "offline launcher requires one nonempty child argv and owns its receipt argument"
        )
    python = assert_no_symlink_ancestors(arguments.python_executable).resolve(strict=True)
    script = assert_no_symlink_ancestors(arguments.campaign_script).resolve(strict=True)
    if not python.is_file() or not script.is_file():
        raise FileNotFoundError("offline launcher inputs must be regular files")
    unshare = _resolve_unshare_executable(LINUX_UNSHARE_PATH_V1)
    _verify_unshare_network_isolation(
        unshare_executable=unshare,
        python_executable=python,
    )
    receipt = OfflineParentLaunchReceiptV2(
        parent_network_namespace=linux_network_namespace_v2(),
        unshare_executable=str(unshare),
        unshare_executable_sha256=_sha256_file(unshare),
        python_executable=str(python),
        python_executable_sha256=_sha256_file(python),
        campaign_script=str(script),
        campaign_script_sha256=_sha256_file(script),
    )
    raw = canonical_json_bytes(asdict(receipt)) + b"\n"
    actual_receipt_path, physical_sha256 = _write_or_validate(
        arguments.offline_receipt,
        raw,
    )
    environment = closed_training_environment_v2()
    environment[OFFLINE_RECEIPT_ENV_V2] = physical_sha256
    command = [
        str(unshare),
        "--net",
        "--",
        str(python),
        "-I",
        "-B",
        str(script),
        *remainder,
        "--offline-network-receipt",
        str(actual_receipt_path),
    ]
    completed = subprocess.run(
        command,
        check=False,
        env=environment,
        stdin=subprocess.DEVNULL,
        shell=False,
    )
    return int(completed.returncode)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
