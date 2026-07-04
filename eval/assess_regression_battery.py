"""Assess loop-1 regression evidence against paired base outputs.

The input is a ``colab/run_stage5_benchmark_suite.py`` summary. That runner
already stores per-item paired counts for base Qwen versus a recurrent
checkpoint; this script adds the standing non-inferiority policy for regression
checks before narrow synthetic/depth training arms.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Any


Z_95 = 1.959963984540054


def path_for_cli(path: Path, *, root: Path | None = None) -> str:
    if root is not None:
        try:
            return path.relative_to(root).as_posix()
        except ValueError:
            pass
    return str(path).replace("\\", "/")


def read_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def parse_target_specs(value: str) -> list[tuple[str, str]]:
    specs: list[tuple[str, str]] = []
    for raw in value.split(","):
        item = raw.strip()
        if not item:
            continue
        if ":" in item:
            target, aggregate = [part.strip() for part in item.split(":", 1)]
        else:
            target, aggregate = item, "mean"
        specs.append((target, aggregate))
    return specs


def mcnemar_delta(*, paired_examples: int, wins: int, losses: int) -> dict[str, Any]:
    """Return paired delta and asymptotic confidence interval.

    ``wins`` are items the recurrent checkpoint gets right when the base gets
    them wrong; ``losses`` are the reverse. This matches the benchmark-suite
    field names.
    """

    n = int(paired_examples)
    if n <= 0:
        return {
            "paired_examples": n,
            "wins": int(wins),
            "losses": int(losses),
            "delta": None,
            "ci95": None,
            "standard_error": None,
        }
    b = int(losses)
    c = int(wins)
    delta = (c - b) / n
    variance_numerator = max((b + c) - (((b - c) ** 2) / n), 1e-12)
    se = math.sqrt(variance_numerator) / n
    return {
        "paired_examples": n,
        "wins": c,
        "losses": b,
        "delta": delta,
        "ci95": [delta - Z_95 * se, delta + Z_95 * se],
        "standard_error": se,
    }


def verdict(delta: float | None, ci95: list[float] | None, *, margin: float, yellow_margin: float) -> str:
    if delta is None or ci95 is None:
        return "grey_underpowered"
    lo, hi = ci95
    if hi < -margin:
        return "red_regression_established"
    if delta < -yellow_margin:
        return "yellow_drift_watch"
    if lo > -margin:
        return "green_noninferior"
    return "grey_underpowered"


def paired_row(payload: dict[str, Any], benchmark: str, score_target: str, aggregate: str) -> dict[str, Any] | None:
    row = (
        (payload.get("paired_comparisons") or {})
        .get(benchmark, {})
        .get(score_target, {})
        .get(aggregate)
    )
    return row if isinstance(row, dict) else None


def assess_row(
    *,
    benchmark: str,
    score_target: str,
    aggregate: str,
    row: dict[str, Any] | None,
    margin: float,
    yellow_margin: float,
) -> dict[str, Any]:
    if row is None:
        return {
            "benchmark": benchmark,
            "score_target": score_target,
            "aggregate": aggregate,
            "present": False,
            "paired_examples": 0,
            "verdict": "missing",
        }
    stats = mcnemar_delta(
        paired_examples=int(row.get("paired_examples", 0) or 0),
        wins=int(row.get("wins", 0) or 0),
        losses=int(row.get("losses", 0) or 0),
    )
    row_verdict = verdict(
        stats["delta"],
        stats["ci95"],
        margin=margin,
        yellow_margin=yellow_margin,
    )
    return {
        "benchmark": benchmark,
        "score_target": score_target,
        "aggregate": aggregate,
        "present": True,
        "base_correct": int(row.get("base_correct", 0) or 0),
        "recurrent_correct": int(row.get("recurrent_correct", 0) or 0),
        "base_accuracy": float(row.get("base_accuracy", 0.0) or 0.0),
        "recurrent_accuracy": float(row.get("recurrent_accuracy", 0.0) or 0.0),
        "accuracy_delta_recurrent_vs_base": float(
            row.get("accuracy_delta_recurrent_vs_base", stats["delta"] or 0.0) or 0.0
        ),
        "ties": int(row.get("ties", 0) or 0),
        **stats,
        "noninferiority_margin": margin,
        "yellow_margin": yellow_margin,
        "verdict": row_verdict,
    }


def pooled_rows(rows: list[dict[str, Any]], *, margin: float, yellow_margin: float) -> dict[str, Any]:
    present = [row for row in rows if row.get("present")]
    n = sum(int(row.get("paired_examples", 0) or 0) for row in present)
    wins = sum(int(row.get("wins", 0) or 0) for row in present)
    losses = sum(int(row.get("losses", 0) or 0) for row in present)
    base_correct = sum(int(row.get("base_correct", 0) or 0) for row in present)
    recurrent_correct = sum(int(row.get("recurrent_correct", 0) or 0) for row in present)
    stats = mcnemar_delta(paired_examples=n, wins=wins, losses=losses)
    return {
        "paired_examples": n,
        "base_correct": base_correct,
        "recurrent_correct": recurrent_correct,
        "base_accuracy": base_correct / max(n, 1),
        "recurrent_accuracy": recurrent_correct / max(n, 1),
        "accuracy_delta_recurrent_vs_base": (recurrent_correct - base_correct) / max(n, 1),
        **stats,
        "noninferiority_margin": margin,
        "yellow_margin": yellow_margin,
        "verdict": verdict(stats["delta"], stats["ci95"], margin=margin, yellow_margin=yellow_margin),
    }


def overall_status(rows: list[dict[str, Any]], pooled: dict[str, Any]) -> str:
    verdicts = [str(row.get("verdict")) for row in rows]
    pooled_verdicts = [str(row.get("verdict")) for row in pooled.values()]
    all_verdicts = verdicts + pooled_verdicts
    if any(item.startswith("red") for item in all_verdicts):
        return "red_regression_established"
    if any(item == "missing" for item in all_verdicts):
        return "incomplete_missing_rows"
    if any(item.startswith("yellow") for item in all_verdicts):
        return "yellow_drift_watch"
    if any(item.startswith("grey") for item in all_verdicts):
        return "grey_underpowered"
    return "green_noninferior"


def build_assessment(
    *,
    suite_summary: Path,
    suite_payload: dict[str, Any],
    required_benchmarks: list[str],
    target_specs: list[tuple[str, str]],
    margin: float,
    yellow_margin: float,
    run_id: str,
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for benchmark in required_benchmarks:
        for score_target, aggregate in target_specs:
            rows.append(
                assess_row(
                    benchmark=benchmark,
                    score_target=score_target,
                    aggregate=aggregate,
                    row=paired_row(suite_payload, benchmark, score_target, aggregate),
                    margin=margin,
                    yellow_margin=yellow_margin,
                )
            )
    pooled: dict[str, Any] = {}
    for score_target, aggregate in target_specs:
        key = f"{score_target}:{aggregate}"
        pooled[key] = pooled_rows(
            [
                row
                for row in rows
                if row.get("score_target") == score_target and row.get("aggregate") == aggregate
            ],
            margin=margin,
            yellow_margin=yellow_margin,
        )
    return {
        "kind": "stage5_regression_battery_assessment",
        "run_id": run_id,
        "status": overall_status(rows, pooled),
        "source_benchmark_suite": path_for_cli(suite_summary),
        "suite_status": suite_payload.get("status"),
        "source_summary": suite_payload.get("source_summary"),
        "checkpoint": suite_payload.get("checkpoint"),
        "primary_loop": suite_payload.get("recurrent_forced_loop_count") or suite_payload.get("recurrent_max_loops"),
        "required_benchmarks": required_benchmarks,
        "target_specs": [{"score_target": target, "aggregate": aggregate} for target, aggregate in target_specs],
        "noninferiority_margin_accuracy": margin,
        "yellow_margin_accuracy": yellow_margin,
        "rows": rows,
        "pooled": pooled,
        "pending_extensions": {
            "tier1_natural_text_nll_canary": "not yet wired in this repository",
            "hellaswag_500": "not yet supported by colab/run_stage5_benchmark_suite.py",
            "winogrande_500": "not yet supported by colab/run_stage5_benchmark_suite.py",
            "lambada_500": "not yet supported by colab/run_stage5_benchmark_suite.py",
        },
    }


def write_markdown(payload: dict[str, Any], path: Path) -> None:
    def fmt_delta(value: Any) -> str:
        return "n/a" if value is None else f"{float(value):+.4f}"

    lines = [
        f"# Regression Battery: {payload['run_id']}",
        "",
        f"- Status: `{payload['status']}`",
        f"- Source suite: `{payload['source_benchmark_suite']}`",
        f"- Checkpoint: `{payload.get('checkpoint')}`",
        f"- Loop: `{payload.get('primary_loop')}`",
        f"- Accuracy non-inferiority margin: `{payload['noninferiority_margin_accuracy']:.3f}`",
        "",
        "## Pooled",
    ]
    for key, row in payload["pooled"].items():
        ci = row.get("ci95")
        ci_text = "n/a" if ci is None else f"[{ci[0]:+.4f}, {ci[1]:+.4f}]"
        lines.append(
            f"- `{key}`: verdict=`{row['verdict']}`, delta={fmt_delta(row.get('delta'))}, "
            f"ci95={ci_text}, n={row['paired_examples']}, "
            f"base={row['base_correct']}, recurrent={row['recurrent_correct']}"
        )
    lines.extend(["", "## Rows"])
    for row in payload["rows"]:
        if not row.get("present"):
            lines.append(
                f"- `{row['benchmark']}` `{row['score_target']}:{row['aggregate']}`: missing"
            )
            continue
        ci = row.get("ci95")
        ci_text = "n/a" if ci is None else f"[{ci[0]:+.4f}, {ci[1]:+.4f}]"
        lines.append(
            f"- `{row['benchmark']}` `{row['score_target']}:{row['aggregate']}`: "
            f"verdict=`{row['verdict']}`, delta={fmt_delta(row.get('delta'))}, "
            f"ci95={ci_text}, n={row['paired_examples']}, "
            f"wins={row['wins']}, losses={row['losses']}"
        )
    lines.extend(
        [
            "",
            "## Pending Extensions",
            "- Tier 1 natural-text NLL canary is not wired yet.",
            "- HellaSwag, Winogrande, and LAMBADA are recorded as requested extensions, not run by this ARC gate.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite_summary", required=True)
    parser.add_argument("--output_json", required=True)
    parser.add_argument("--output_md", required=True)
    parser.add_argument("--run_id", default="")
    parser.add_argument("--required_benchmarks", default="arc_easy,arc_challenge")
    parser.add_argument(
        "--target_specs",
        default="content_question_only:mean,cyclic_label_aggregated:permutation_mean",
    )
    parser.add_argument("--margin", type=float, default=0.03)
    parser.add_argument("--yellow_margin", type=float, default=0.015)
    args = parser.parse_args()

    suite_summary = Path(args.suite_summary)
    payload = build_assessment(
        suite_summary=suite_summary,
        suite_payload=read_json(suite_summary),
        required_benchmarks=[item.strip() for item in args.required_benchmarks.split(",") if item.strip()],
        target_specs=parse_target_specs(args.target_specs),
        margin=args.margin,
        yellow_margin=args.yellow_margin,
        run_id=args.run_id or time.strftime("stage5_regression_battery_assessment_%Y%m%d_%H%M%S"),
    )
    output_json = Path(args.output_json)
    output_md = Path(args.output_md)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_md.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    write_markdown(payload, output_md)
    print(json.dumps({"status": payload["status"], "output_json": str(output_json)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
