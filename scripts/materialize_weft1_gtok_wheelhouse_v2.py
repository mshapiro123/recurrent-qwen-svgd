#!/usr/bin/env python3
"""Materialize the bounded offline WEFT-1 G-TOK Linux wheelhouse."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from training.weft1_gtok_contract import canonical_json_bytes  # noqa: E402
from training.weft1_gtok_runtime_v2 import (  # noqa: E402
    _offline_wheelhouse_receipt_v2,
    _parse_hash_pinned_lock_v2,
    closed_training_environment_v2,
)


PYPI_INDEX = "https://pypi.org/simple"
PYTORCH_CU128_INDEX = "https://download.pytorch.org/whl/cu128"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--downloader-python", type=Path, required=True)
    parser.add_argument("--requirements-lock", type=Path, required=True)
    parser.add_argument("--output-wheelhouse", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    arguments = parser.parse_args(argv)
    if arguments.output_wheelhouse.exists() or arguments.output_wheelhouse.is_symlink():
        raise FileExistsError("wheelhouse output must be new")
    if arguments.receipt.exists() or arguments.receipt.is_symlink():
        raise FileExistsError("wheelhouse receipt must be new")
    lock_bytes, lock_rows = _parse_hash_pinned_lock_v2(arguments.requirements_lock)
    arguments.output_wheelhouse.mkdir(parents=True, exist_ok=False)
    environment = closed_training_environment_v2()
    environment["PIP_NO_INDEX"] = "0"
    subprocess.run(
        [
            str(arguments.downloader_python.resolve(strict=True)),
            "-I",
            "-B",
            "-m",
            "pip",
            "download",
            "--require-hashes",
            "--only-binary=:all:",
            "--platform=manylinux_2_28_x86_64",
            "--platform=manylinux2014_x86_64",
            "--platform=manylinux_2_17_x86_64",
            "--python-version=3.11",
            "--implementation=cp",
            "--abi=cp311",
            "--index-url",
            PYPI_INDEX,
            "--extra-index-url",
            PYTORCH_CU128_INDEX,
            "--dest",
            str(arguments.output_wheelhouse),
            "-r",
            str(arguments.requirements_lock.resolve(strict=True)),
        ],
        check=True,
        env=environment,
    )
    wheels, _pip = _offline_wheelhouse_receipt_v2(
        arguments.output_wheelhouse,
        lock_rows,
    )
    payload = {
        "indexes": (PYPI_INDEX, PYTORCH_CU128_INDEX),
        "lock_bytes": len(lock_bytes),
        "lock_sha256": hashlib.sha256(lock_bytes).hexdigest(),
        "platform": "CPython-3.11.9-manylinux_2_28_x86_64-with-compatible-manylinux2014",
        "schema": "weft1_gtok_training_wheelhouse_v2",
        "wheels": wheels,
    }
    raw = canonical_json_bytes(payload) + b"\n"
    descriptor = os.open(arguments.receipt, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "wb") as handle:
        handle.write(raw)
        handle.flush()
        os.fsync(handle.fileno())
    print(json.dumps({"receipt_sha256": hashlib.sha256(raw).hexdigest()}))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
