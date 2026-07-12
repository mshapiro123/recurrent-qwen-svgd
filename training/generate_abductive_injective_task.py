"""Generate the paired Phase G-alpha abductive/injective task gate."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from training.abductive_injective_task import (
    AbductiveInjectiveConfig,
    build_rows,
    validate_rows,
    write_jsonl,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--n_symbols", type=int, default=20)
    parser.add_argument("--max_depth", type=int, default=8)
    parser.add_argument("--train_rows_per_depth", type=int, default=256)
    parser.add_argument("--test_rows_per_depth", type=int, default=128)
    parser.add_argument("--seed", type=int, default=1_104_729)
    parser.add_argument("--min_solutions", type=int, default=2)
    parser.add_argument("--max_solutions", type=int, default=4)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    summary: dict[str, object] = {
        "kind": "abductive_injective_task_gate",
        "status": "started",
        "config": vars(args),
        "datasets": {},
    }
    all_ids: set[str] = set()
    for split, rows_per_depth in (("train", args.train_rows_per_depth), ("test", args.test_rows_per_depth)):
        for mode in ("injective", "abductive"):
            config = AbductiveInjectiveConfig(
                n_symbols=args.n_symbols,
                max_depth=args.max_depth,
                rows_per_depth=rows_per_depth,
                seed=args.seed,
                min_solutions=args.min_solutions,
                max_solutions=args.max_solutions,
            )
            rows = build_rows(config, split=split, mode=mode)
            validation = validate_rows(rows, expected_mode=mode)
            if validation["status"] != "passed":
                raise RuntimeError(f"{split}/{mode} validation failed: {validation['errors'][:5]}")
            overlap = all_ids.intersection(str(row["id"]) for row in rows)
            if overlap:
                raise RuntimeError(f"row ids overlap across datasets: {sorted(overlap)[:5]}")
            all_ids.update(str(row["id"]) for row in rows)
            path = output_dir / f"{split}_{mode}.jsonl"
            write_jsonl(path, rows)
            summary["datasets"][f"{split}_{mode}"] = {  # type: ignore[index]
                "path": str(path),
                **validation,
            }

    summary["status"] = "passed"
    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
