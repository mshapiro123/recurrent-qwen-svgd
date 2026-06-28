"""Evaluate whether deeper-loop rescue is detectable before selector training.

This is the cheap gate after a forced-depth sweep. It uses only selector-safe
loop-1 telemetry, fits regularized low-dimensional rescue directions on random
subsamples, then compares pairwise direction agreement against a label
permutation null. If real-label agreement clears the null, the rescue signal is
stable enough to justify held-out selector transfer work.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from eval.evaluate_rescue_selector_transfer import (  # noqa: E402
    DEFAULT_PROBE_SHRINKAGES,
    SELECTOR_FEATURES,
    category_counts,
    diverse_probe_detectability,
    examples_for_sweep,
    rescue_discrimination,
    train_supervised_probes,
)


def parse_floats(value: str) -> list[float]:
    return [float(item.strip()) for item in value.split(",") if item.strip()]


def path_for_cli(path: Path) -> str:
    try:
        return path.relative_to(ROOT).as_posix()
    except ValueError:
        return str(path)


def resolve_path(path: str | Path) -> Path:
    candidate = Path(str(path).replace("\\", "/"))
    return candidate if candidate.is_absolute() else ROOT / candidate


def best_detectability_row(rows: list[dict[str, Any]]) -> dict[str, Any] | None:
    available = [row for row in rows if row.get("available")]
    if not available:
        return None
    return sorted(
        available,
        key=lambda row: (
            bool(row.get("clears_null_p95")),
            float(row.get("observed_minus_null_p95") or -1e9),
            float(row.get("observed_alignment") or 0.0),
        ),
        reverse=True,
    )[0]


def analyze_detectability(
    *,
    sweep_summary: Path,
    benchmark: str,
    score_target: str,
    aggregate: str,
    shrinkages: list[float],
    repeats: int,
    permutations: int,
    sample_fraction: float,
    seed: int,
    run_id: str | None = None,
) -> dict[str, Any]:
    loops, examples = examples_for_sweep(
        sweep_summary,
        benchmark=benchmark,
        score_target=score_target,
        aggregate=aggregate,
    )
    detectability_rows = []
    for index, shrinkage in enumerate(shrinkages):
        row = diverse_probe_detectability(
            examples,
            features=SELECTOR_FEATURES,
            shrinkage=shrinkage,
            repeats=repeats,
            permutations=permutations,
            sample_fraction=sample_fraction,
            seed=seed + 1009 * index,
        )
        observed = row.get("observed_alignment")
        null_p95 = row.get("null_p95_alignment")
        row["observed_minus_null_p95"] = (
            None if observed is None or null_p95 is None else float(observed) - float(null_p95)
        )
        detectability_rows.append(row)

    supervised_probes = train_supervised_probes(
        examples,
        loops,
        features=SELECTOR_FEATURES,
        shrinkages=shrinkages,
    )
    best_row = best_detectability_row(detectability_rows)
    gate_status = "passed" if best_row and best_row.get("clears_null_p95") else "needs_review"
    return {
        "kind": "stage5_rescue_detectability_gate",
        "run_id": run_id or "stage5_rescue_detectability_" + time.strftime("%Y%m%d_%H%M%S"),
        "status": gate_status,
        "source_sweep_summary": path_for_cli(sweep_summary),
        "benchmark": benchmark,
        "score_target": score_target,
        "aggregate": aggregate,
        "loops": loops,
        "category_counts": category_counts(examples),
        "rescue_discrimination": rescue_discrimination(examples),
        "detectability_by_shrinkage": detectability_rows,
        "best_detectability": best_row,
        "supervised_probe_discovery": [
            {
                key: value
                for key, value in probe.items()
                if key not in {"stats", "direction", "discovery_curve"}
            }
            for probe in supervised_probes
        ],
    }


def write_summary_md(payload: dict[str, Any], path: Path) -> None:
    best = payload.get("best_detectability") or {}
    lines = [
        f"# Rescue Detectability Gate - {payload['run_id']}",
        "",
        f"- Source sweep: `{payload['source_sweep_summary']}`",
        f"- Benchmark: `{payload['benchmark']}`",
        f"- Score target: `{payload['score_target']}` / `{payload['aggregate']}`",
        f"- Loops: `{payload['loops']}`",
        f"- Status: `{payload['status']}`",
        f"- Category counts: `{payload['category_counts']}`",
        "",
        "## Direction Agreement Gate",
        "",
    ]
    if best:
        lines.extend(
            [
                f"- Best shrinkage: `{best.get('shrinkage')}`",
                f"- Observed agreement: `{best.get('observed_alignment')}`",
                f"- Null mean agreement: `{best.get('null_mean_alignment')}`",
                f"- Null p95 agreement: `{best.get('null_p95_alignment')}`",
                f"- Observed minus null p95: `{best.get('observed_minus_null_p95')}`",
                f"- Clears null p95: `{best.get('clears_null_p95')}`",
                "",
            ]
        )
    for row in payload.get("detectability_by_shrinkage", []):
        lines.append(
            "- shrinkage "
            f"`{row.get('shrinkage')}`: observed `{row.get('observed_alignment')}`, "
            f"null_p95 `{row.get('null_p95_alignment')}`, "
            f"margin `{row.get('observed_minus_null_p95')}`, "
            f"clears `{row.get('clears_null_p95')}`"
        )
    lines.extend(["", "## Supervised Probe Discovery Curves", ""])
    for probe in payload.get("supervised_probe_discovery", []):
        summary = probe.get("discovery_curve_summary", {})
        lines.append(f"### {probe.get('feature')}")
        for label in ["zero_harm", "harm_budget_1", "harm_budget_2", "max_net"]:
            row = summary.get(label)
            if row:
                lines.append(
                    f"- {label}: correct `{row.get('correct')}`, delta `{row.get('delta_vs_loop1')}`, "
                    f"rescue `{row.get('rescue_captured')}`, harm `{row.get('harm_triggered')}`, "
                    f"routed `{row.get('routed_deep')}`"
                )
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sweep_summary", required=True)
    parser.add_argument("--benchmark", default="arc_challenge")
    parser.add_argument("--score_target", default="content_question_only")
    parser.add_argument("--aggregate", default="mean")
    parser.add_argument("--shrinkages", default=",".join(str(v) for v in DEFAULT_PROBE_SHRINKAGES))
    parser.add_argument("--repeats", type=int, default=64)
    parser.add_argument("--permutations", type=int, default=128)
    parser.add_argument("--sample_fraction", type=float, default=0.7)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--run_id", default="")
    parser.add_argument("--output_dir", default="")
    args = parser.parse_args()

    payload = analyze_detectability(
        sweep_summary=resolve_path(args.sweep_summary),
        benchmark=args.benchmark,
        score_target=args.score_target,
        aggregate=args.aggregate,
        shrinkages=parse_floats(args.shrinkages),
        repeats=args.repeats,
        permutations=args.permutations,
        sample_fraction=args.sample_fraction,
        seed=args.seed,
        run_id=args.run_id or None,
    )
    out_dir = resolve_path(args.output_dir) if args.output_dir else (
        ROOT / "outputs" / "stage5" / str(payload["run_id"])
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "summary.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    write_summary_md(payload, out_dir / "summary.md")
    print((out_dir / "summary.md").read_text(encoding="utf-8"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
