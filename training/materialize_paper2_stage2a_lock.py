"""Bind Stage 2A score-blind receipts into a still-disabled lock candidate."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from training.paper2_stage2a_lock import load_stage2a_lock, materialize_stage2a_lock
from training.paper2_phase3_p31_completion import sha256_file


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--draft", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    summary = json.loads(args.summary.read_text(encoding="utf-8"))
    lock = load_stage2a_lock(args.draft) if args.draft else load_stage2a_lock()
    materialized = materialize_stage2a_lock(
        lock, summary, summary_sha256=sha256_file(args.summary)
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(
        json.dumps(materialized, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(args.output)
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
