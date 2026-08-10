"""Seal P3.1 CONFIRM membership before any reference model is loaded."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from training.paper2_phase3_p31_completion import seal_confirm_membership, sha256_file


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source_summary", type=Path, required=True)
    parser.add_argument("--source_rows", type=Path, required=True)
    parser.add_argument("--source_manifest", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    args = parser.parse_args()
    summary = json.loads(args.source_summary.read_text(encoding="utf-8"))
    result = seal_confirm_membership(
        summary["ledger"],
        output_dir=args.output_dir,
        source_rows_sha256=sha256_file(args.source_rows),
        source_manifest_sha256=sha256_file(args.source_manifest),
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
