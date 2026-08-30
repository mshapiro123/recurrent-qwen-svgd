#!/usr/bin/env python3
"""Build the separate hash-pinned WEFT-1 G-TOK Python 3.11.9 runtime."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from training.weft1_gtok_runtime_v2 import (  # noqa: E402
    build_gtok_training_venv_v2,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--python-3-11-9", type=Path, required=True)
    parser.add_argument("--training-requirements-lock", type=Path, required=True)
    parser.add_argument("--offline-wheelhouse", type=Path, required=True)
    parser.add_argument("--wheelhouse-receipt", type=Path, required=True)
    parser.add_argument("--pa-runtime-build-receipt", type=Path, required=True)
    parser.add_argument("--venv-root", type=Path, required=True)
    parser.add_argument("--runtime-binding", type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    built = build_gtok_training_venv_v2(
        python_executable=arguments.python_3_11_9,
        requirements_lock=arguments.training_requirements_lock,
        wheelhouse=arguments.offline_wheelhouse,
        wheelhouse_receipt=arguments.wheelhouse_receipt,
        pa_runtime_build_receipt=arguments.pa_runtime_build_receipt,
        venv_root=arguments.venv_root,
        binding_path=arguments.runtime_binding,
    )
    print(built)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
