"""Analyze exact coverage of the small symbolic ARC-AGI candidate generator."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from eval.arc_agi_symbolic import exact_symbolic_candidate, symbolic_candidates  # noqa: E402
from eval.arc_agi_utils import ArcAgiExample, load_arc_agi_examples  # noqa: E402


def analyze_examples(examples: list[ArcAgiExample]) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    exact_by_source: Counter[str] = Counter()
    candidate_by_source: Counter[str] = Counter()
    target_count = 0
    exact_count = 0
    for example in examples:
        candidates = symbolic_candidates(example)
        for candidate in candidates:
            candidate_by_source[candidate.name] += 1
        exact = exact_symbolic_candidate(example)
        has_target = example.test_output is not None
        target_count += int(has_target)
        exact_count += int(exact is not None)
        if exact is not None:
            exact_by_source[exact.name] += 1
        rows.append(
            {
                "task_id": example.task_id,
                "test_index": example.test_index,
                "has_target": has_target,
                "num_symbolic_candidates": len(candidates),
                "exact_symbolic": exact is not None,
                "exact_source": exact.name if exact is not None else None,
                "candidate_sources": [candidate.name for candidate in candidates],
            }
        )

    task_ids = sorted({example.task_id for example in examples if example.test_output is not None})
    solved_tasks = 0
    for task_id in task_ids:
        task_rows = [row for row in rows if row["task_id"] == task_id and row["has_target"]]
        if task_rows and all(row["exact_symbolic"] for row in task_rows):
            solved_tasks += 1

    return {
        "summary": {
            "examples": len(examples),
            "examples_with_targets": target_count,
            "exact_symbolic": exact_count,
            "exact_symbolic_rate": exact_count / max(target_count, 1),
            "tasks_with_targets": len(task_ids),
            "tasks_solved_symbolic": solved_tasks,
            "task_solve_rate_symbolic": solved_tasks / max(len(task_ids), 1),
            "mean_candidates": sum(row["num_symbolic_candidates"] for row in rows) / max(len(rows), 1),
            "exact_by_source": dict(sorted(exact_by_source.items())),
            "candidate_by_source": dict(sorted(candidate_by_source.items())),
        },
        "examples": rows,
    }


def write_json(path: str | Path | None, payload: dict[str, Any]) -> None:
    if not path:
        return
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def write_md(path: str | Path | None, payload: dict[str, Any]) -> None:
    if not path:
        return
    summary = payload["summary"]
    lines = [
        "# ARC-AGI Symbolic Coverage",
        "",
        f"- Examples with targets: `{summary['examples_with_targets']}`",
        f"- Exact symbolic examples: `{summary['exact_symbolic']}` = `{summary['exact_symbolic_rate']}`",
        f"- Tasks solved symbolically: `{summary['tasks_solved_symbolic']}` / `{summary['tasks_with_targets']}` = "
        f"`{summary['task_solve_rate_symbolic']}`",
        f"- Mean symbolic candidates: `{summary['mean_candidates']}`",
        "",
        "## Exact By Source",
    ]
    if summary["exact_by_source"]:
        for source, count in summary["exact_by_source"].items():
            lines.append(f"- `{source}`: `{count}`")
    else:
        lines.append("- None")
    lines += ["", "## Candidate By Source"]
    if summary["candidate_by_source"]:
        for source, count in summary["candidate_by_source"].items():
            lines.append(f"- `{source}`: `{count}`")
    else:
        lines.append("- None")
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tasks_path", required=True)
    parser.add_argument("--solutions_path")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--summary_json")
    parser.add_argument("--summary_md")
    args = parser.parse_args()

    examples = load_arc_agi_examples(args.tasks_path, solutions_path=args.solutions_path, limit=args.limit)
    payload = {
        "tasks_path": args.tasks_path,
        "solutions_path": args.solutions_path,
        "limit": args.limit,
        **analyze_examples(examples),
    }
    write_json(args.summary_json, payload)
    write_md(args.summary_md, payload)
    print(json.dumps(payload["summary"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
