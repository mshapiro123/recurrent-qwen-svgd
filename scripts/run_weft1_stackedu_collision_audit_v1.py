"""Run the offline, hash-only StackEdu native-ID collision diagnostic."""

from __future__ import annotations

import argparse
from dataclasses import asdict
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from training.weft1_gtok_contract import canonical_json_bytes  # noqa: E402
from training.weft1_stackedu_collision_audit_v1 import (  # noqa: E402
    run_stackedu_collision_audit_v1,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-cache-manifest", required=True, type=Path)
    parser.add_argument("--source-cache", required=True, type=Path)
    parser.add_argument("--work-root", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    arguments = build_parser().parse_args(argv)
    try:
        receipt = run_stackedu_collision_audit_v1(
            source_manifest_path=arguments.source_cache_manifest,
            cache_root=arguments.source_cache,
            work_root=arguments.work_root,
            output_root=arguments.output_root,
        )
    except (
        OSError,
        RuntimeError,
        TypeError,
        ValueError,
    ) as error:
        payload = {
            "error": type(error).__name__,
            "message": str(error),
            "schema": "weft1_stackedu_collision_audit_failure_v1",
        }
        sys.stderr.buffer.write(canonical_json_bytes(payload) + b"\n")
        return 2
    payload = {
        "receipt": asdict(receipt),
        "receipt_sha256": receipt.receipt_sha256,
    }
    sys.stdout.buffer.write(canonical_json_bytes(payload) + b"\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
