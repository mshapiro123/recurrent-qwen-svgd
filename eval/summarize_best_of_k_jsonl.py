"""Summarize structured rows emitted by eval_best_of_k_jsonl.py."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any


def setting_key(row: dict[str, Any]) -> tuple[Any, ...]:
    return (
        row.get("mode"),
        row.get("temperature"),
        row.get("particle_init_noise"),
        row.get("particle_noise_steps"),
        row.get("svgd_eps"),
        row.get("svgd_repulsion_scale"),
        row.get("svgd_repulsion_max_norm"),
    )


def format_key(key: tuple[Any, ...]) -> str:
    names = (
        "mode",
        "temp",
        "noise",
        "noise_steps",
        "eps",
        "repulsion",
        "max_norm",
    )
    return " ".join(f"{name}={value}" for name, value in zip(names, key))


def read_rows(path: Path) -> list[dict[str, Any]]:
    rows = []
    for line in path.read_text(encoding="utf-8-sig").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("jsonl", type=Path)
    parser.add_argument("--show_tasks", action="store_true")
    parser.add_argument("--show_diagnostics", action="store_true")
    args = parser.parse_args()

    rows = read_rows(args.jsonl)
    if not rows:
        raise SystemExit(f"No rows found in {args.jsonl}")

    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[setting_key(row)].append(row)

    print(f"rows={len(rows)}")
    for key, setting_rows in grouped.items():
        task_groups: dict[tuple[Any, Any], list[dict[str, Any]]] = defaultdict(list)
        for row in setting_rows:
            task_groups[(row.get("seed"), row.get("task"))].append(row)

        best_hits = sum(any(candidate.get("hit") for candidate in candidates) for candidates in task_groups.values())
        candidate_hits = sum(bool(row.get("hit")) for row in setting_rows)
        candidate_count = len(setting_rows)
        task_count = len(task_groups)
        print()
        print(format_key(key))
        print(f"best_hits={best_hits}/{task_count}")
        print(f"candidate_hits={candidate_hits}/{candidate_count}")

        if args.show_tasks:
            by_task: dict[str, list[dict[str, Any]]] = defaultdict(list)
            for row in setting_rows:
                by_task[str(row.get("task"))].append(row)
            print("tasks:")
            for task, task_rows in sorted(by_task.items()):
                task_seed_groups: dict[Any, list[dict[str, Any]]] = defaultdict(list)
                for row in task_rows:
                    task_seed_groups[row.get("seed")].append(row)
                task_best = sum(
                    any(candidate.get("hit") for candidate in candidates)
                    for candidates in task_seed_groups.values()
                )
                task_candidate_hits = sum(bool(row.get("hit")) for row in task_rows)
                print(f"  {task}: best={task_best}/{len(task_seed_groups)} candidates={task_candidate_hits}/{len(task_rows)}")

        if args.show_diagnostics:
            diagnostics: dict[str, list[float]] = defaultdict(list)
            for row in setting_rows:
                for name, value in row.get("diagnostics", {}).items():
                    if isinstance(value, (int, float)):
                        diagnostics[name].append(float(value))
            if diagnostics:
                print("diagnostics:")
                for name, values in sorted(diagnostics.items()):
                    print(f"  {name}={sum(values) / len(values):.6g}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
