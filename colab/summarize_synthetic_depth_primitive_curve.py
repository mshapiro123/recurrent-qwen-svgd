"""Summarize Phase 1 synthetic-depth primitive curve runs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def read_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _cell_accuracy(matrix: dict[str, Any], *, depth: int, loop: int) -> float:
    return float(matrix["matrix"][str(depth)][str(loop)]["accuracy"])


def _cell_correct(matrix: dict[str, Any], *, depth: int, loop: int) -> int:
    return int(matrix["matrix"][str(depth)][str(loop)]["correct"])


def _cell_total(matrix: dict[str, Any], *, depth: int, loop: int) -> int:
    return int(matrix["matrix"][str(depth)][str(loop)]["total"])


def summarize_run(summary_path: str | Path, *, primitive_bar: float, strong_bar: float) -> dict[str, Any]:
    summary_path = Path(summary_path)
    summary = read_json(summary_path)
    # The run summary stores dataset_summary as a repo-relative path. Prefer
    # that path if it exists; otherwise fall back to the standard run layout.
    repo_relative_dataset = Path(summary["dataset_summary"])
    if repo_relative_dataset.exists():
        dataset_summary = read_json(repo_relative_dataset)
    else:
        dataset_summary = read_json(summary_path.parent / "data" / "summary.json")
    config = dataset_summary["config"]
    matrix = summary["matrix"]
    base_matrix = summary.get("base_matrix") or {}
    recurrent_accuracy = _cell_accuracy(matrix, depth=1, loop=1)
    base_accuracy = _cell_accuracy(base_matrix, depth=1, loop=0) if base_matrix else None
    return {
        "run_id": summary["run_id"],
        "summary_path": str(summary_path).replace("\\", "/"),
        "n_symbols": int(config["n_symbols"]),
        "rows_per_depth": int(config["rows_per_depth"]),
        "max_steps": int(read_json(summary_path).get("max_steps", 0) or 0),
        "train_format": summary.get("train_format"),
        "base_accuracy": base_accuracy,
        "base_correct": _cell_correct(base_matrix, depth=1, loop=0) if base_matrix else None,
        "base_total": _cell_total(base_matrix, depth=1, loop=0) if base_matrix else None,
        "recurrent_accuracy": recurrent_accuracy,
        "recurrent_correct": _cell_correct(matrix, depth=1, loop=1),
        "recurrent_total": _cell_total(matrix, depth=1, loop=1),
        "clears_primitive_bar": recurrent_accuracy >= primitive_bar,
        "clears_strong_bar": recurrent_accuracy >= strong_bar,
        "checkpoint": summary.get("checkpoint"),
    }


def summarize_curve(
    summary_paths: list[str | Path],
    *,
    primitive_bar: float = 0.71,
    strong_bar: float = 0.9,
) -> dict[str, Any]:
    runs = sorted(
        [summarize_run(path, primitive_bar=primitive_bar, strong_bar=strong_bar) for path in summary_paths],
        key=lambda row: row["n_symbols"],
    )
    clearing = [row["n_symbols"] for row in runs if row["clears_primitive_bar"]]
    strong = [row["n_symbols"] for row in runs if row["clears_strong_bar"]]
    recommended_n = max(clearing) if clearing else None
    return {
        "kind": "stage5_synthetic_depth_primitive_curve",
        "primitive_accuracy_bar": primitive_bar,
        "strong_accuracy_bar": strong_bar,
        "runs": runs,
        "largest_n_clearing_primitive_bar": recommended_n,
        "largest_n_clearing_strong_bar": max(strong) if strong else None,
        "all_runs_clear_primitive_bar": len(clearing) == len(runs) and bool(runs),
        "recommended_phase2_n_symbols": recommended_n,
        "decision_rule": (
            "Proceed to staged-depth forced-loop staircase at the largest N whose "
            "depth-1 primitive accuracy clears the primitive bar; prefer the "
            "largest N clearing the strong bar if available."
        ),
    }


def write_markdown(summary: dict[str, Any], path: str | Path) -> None:
    lines = [
        "# Synthetic Depth Primitive Curve",
        "",
        f"- Primitive bar: `{summary['primitive_accuracy_bar']}`",
        f"- Strong bar: `{summary['strong_accuracy_bar']}`",
        f"- Largest N clearing primitive bar: `{summary['largest_n_clearing_primitive_bar']}`",
        f"- Largest N clearing strong bar: `{summary['largest_n_clearing_strong_bar']}`",
        f"- Recommended Phase 2 N: `{summary['recommended_phase2_n_symbols']}`",
        "",
        "| N | Base acc | Recurrent acc | Clears 0.71 | Clears 0.90 |",
        "|---:|---:|---:|:---:|:---:|",
    ]
    for row in summary["runs"]:
        base = "n/a" if row["base_accuracy"] is None else f"{row['base_accuracy']:.3f}"
        lines.append(
            "| "
            f"{row['n_symbols']} | {base} | {row['recurrent_accuracy']:.3f} | "
            f"{'yes' if row['clears_primitive_bar'] else 'no'} | "
            f"{'yes' if row['clears_strong_bar'] else 'no'} |"
        )
    lines.append("")
    lines.append(summary["decision_rule"])
    Path(path).write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary_paths", nargs="+", required=True)
    parser.add_argument("--output_json", required=True)
    parser.add_argument("--output_md", required=True)
    parser.add_argument("--primitive_bar", type=float, default=0.71)
    parser.add_argument("--strong_bar", type=float, default=0.9)
    args = parser.parse_args()

    summary = summarize_curve(
        args.summary_paths,
        primitive_bar=args.primitive_bar,
        strong_bar=args.strong_bar,
    )
    Path(args.output_json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output_json).write_text(json.dumps(summary, indent=2), encoding="utf-8")
    write_markdown(summary, args.output_md)
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
