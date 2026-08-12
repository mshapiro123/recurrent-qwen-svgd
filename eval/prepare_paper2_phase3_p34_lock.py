"""Freeze the ratified P3.4 task panel without model scoring."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable, Mapping

from training.paper2_phase3_p34_lock import build_task_panel


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(dict(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(dict(row), sort_keys=True, ensure_ascii=True) + "\n")
    temporary.replace(path)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source_rows", type=Path, required=True)
    parser.add_argument("--panel_jsonl", type=Path, required=True)
    parser.add_argument("--receipt_json", type=Path, required=True)
    parser.add_argument("--base_scores", type=Path)
    parser.add_argument("--panel_base_scores", type=Path)
    args = parser.parse_args()
    if (args.base_scores is None) != (args.panel_base_scores is None):
        raise ValueError("P3.4 panel base-score inputs must be supplied together")
    panel, receipt = build_task_panel(read_jsonl(args.source_rows))
    write_jsonl(args.panel_jsonl, panel)
    if args.base_scores is not None:
        base_rows = read_jsonl(args.base_scores)
        base_lookup = {str(row["item_id"]): row for row in base_rows}
        item_ids = [str(row["item_id"]) for row in panel]
        if len(base_lookup) != len(base_rows) or any(
            item_id not in base_lookup for item_id in item_ids
        ):
            raise RuntimeError("P3.4 panel base-score coverage is incomplete or duplicated")
        selected_base = [base_lookup[item_id] for item_id in item_ids]
        write_jsonl(args.panel_base_scores, selected_base)
        receipt["panel_base_score_rows"] = len(selected_base)
        receipt["panel_base_score_source"] = str(args.base_scores)
    write_json(args.receipt_json, receipt)
    print(json.dumps(receipt, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
