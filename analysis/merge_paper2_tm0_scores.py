"""Merge transport-partitioned TM-0 scores under exact overlap checks."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from training.paper2_tm0 import atomic_json, read_jsonl, sha256_file


def merge_scores(
    panel: list[dict[str, Any]], sources: list[Path]
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    expected = [str(row["item_id"]) for row in panel]
    expected_set = set(expected)
    if len(expected) != len(expected_set):
        raise RuntimeError("TM-0 merge panel has duplicate item ids")
    merged: dict[str, dict[str, Any]] = {}
    source_receipts = []
    overlap_rows = 0
    for source in sources:
        rows = read_jsonl(source)
        source_receipts.append(
            {"path": str(source), "rows": len(rows), "sha256": sha256_file(source)}
        )
        for row in rows:
            item_id = str(row["item_id"])
            if item_id not in expected_set:
                raise RuntimeError(f"TM-0 merge source row is outside panel: {item_id}")
            if item_id in merged:
                overlap_rows += 1
                if row != merged[item_id]:
                    raise RuntimeError(f"TM-0 overlapping score bytes disagree: {item_id}")
            else:
                merged[item_id] = row
    missing = [item_id for item_id in expected if item_id not in merged]
    if missing:
        raise RuntimeError(f"TM-0 merged scores miss {len(missing)} panel rows")
    ordered = [merged[item_id] for item_id in expected]
    receipt = {
        "kind": "paper2_tm0_transport_partition_merge_v1",
        "panel_rows": len(expected),
        "panel_item_id_order_sha256": __import__("hashlib").sha256(
            "\n".join(expected).encode("utf-8")
        ).hexdigest(),
        "sources": source_receipts,
        "overlap_rows_exact": overlap_rows,
        "output_rows": len(ordered),
        "optimizer_steps": 0,
        "confirm_scored": False,
        "eval_e_scored": False,
    }
    return ordered, receipt


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--panel", type=Path, required=True)
    parser.add_argument("--source", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    args = parser.parse_args()
    rows, receipt = merge_scores(read_jsonl(args.panel), args.source)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    temporary.replace(args.output)
    receipt["output_path"] = str(args.output)
    receipt["output_sha256"] = sha256_file(args.output)
    atomic_json(args.receipt, receipt)
    print(json.dumps(receipt, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
