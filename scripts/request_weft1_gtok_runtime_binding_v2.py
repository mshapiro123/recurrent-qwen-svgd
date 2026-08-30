#!/usr/bin/env python3
"""Emit a non-authorizing G-TOK runtime binding request on the selected A100."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from training.weft1_gtok_runtime_v2 import write_runtime_binding_request_v2  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training-requirements-lock", type=Path, required=True)
    parser.add_argument("--runtime-build-receipt", type=Path, required=True)
    parser.add_argument("--pa-runtime-build-receipt", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cuda-device-index", type=int, default=0)
    arguments = parser.parse_args(argv)
    digest = write_runtime_binding_request_v2(
        output_path=arguments.output,
        requirements_lock=arguments.training_requirements_lock,
        runtime_build_receipt=arguments.runtime_build_receipt,
        pa_runtime_build_receipt=arguments.pa_runtime_build_receipt,
        device_index=arguments.cuda_device_index,
    )
    print(digest)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
