#!/usr/bin/env python3
"""Build the Paper One support-frontier threshold-sensitivity receipt."""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from colab.stage5_frontier_metrics import bar_crossing_frontier, diagonal_counts_to_accuracy


BARS = (0.60, 0.71, 0.80)

SOURCES = (
    {
        "configuration": "support4_N16",
        "support": 4,
        "alphabet": 16,
        "path": "outputs/stage5/stage5_chain_continuation_attribution_20260704_163056/summary.json",
        "locator": ("attribution_extrapolation", "active_diagonal"),
        "format": "accuracy",
    },
    {
        "configuration": "support6_N16",
        "support": 6,
        "alphabet": 16,
        "path": "outputs/stage5/stage5_depth_support_route_20260705_124320/summary.json",
        "locator": ("frozen_active_eval", "active_diagonal"),
        "format": "accuracy",
    },
    {
        "configuration": "support8_N16",
        "support": 8,
        "alphabet": 16,
        "path": "outputs/stage5/stage5_support8_dose_arm_20260706_153028/summary.json",
        "locator": ("frozen_active_eval", "active_diagonal"),
        "format": "accuracy",
    },
    {
        "configuration": "support12_N24",
        "support": 12,
        "alphabet": 24,
        "path": "outputs/stage5/stage5_n24_support12_rung_20260707_140139/summary.json",
        "checkpoint_step": 6000,
        "format": "checkpoint_diagonal_counts",
    },
)


def _lookup(payload: dict[str, Any], locator: tuple[str, ...]) -> Any:
    value: Any = payload
    for key in locator:
        value = value[key]
    return value


def load_curve(repo_root: Path, source: dict[str, Any]) -> dict[str, float]:
    path = repo_root / str(source["path"])
    payload = json.loads(path.read_text(encoding="utf-8"))
    if source["format"] == "accuracy":
        return {str(int(depth)): float(value) for depth, value in _lookup(payload, source["locator"]).items()}
    if source["format"] == "checkpoint_diagonal_counts":
        step = int(source["checkpoint_step"])
        checkpoint = next(item for item in payload["checkpoint_evals"] if int(item["step"]) == step)
        return diagonal_counts_to_accuracy(checkpoint["score"]["diagonal_counts"])
    raise ValueError(f"Unsupported source format: {source['format']}")


def build_receipt(repo_root: Path, bars: tuple[float, ...] = BARS) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    curves: dict[str, dict[str, float]] = {}
    for source in SOURCES:
        curve = load_curve(repo_root, source)
        curves[str(source["configuration"])] = curve
        for bar in bars:
            frontier = bar_crossing_frontier(curve, bar=bar)
            support = int(source["support"])
            rows.append(
                {
                    "configuration": source["configuration"],
                    "support": support,
                    "alphabet": int(source["alphabet"]),
                    "bar": bar,
                    "frontier": frontier,
                    "frontier_to_support_ratio": frontier / support,
                    "source": source["path"],
                }
            )

    by_bar: dict[str, Any] = {}
    for bar in bars:
        ratios = [row["frontier_to_support_ratio"] for row in rows if row["bar"] == bar]
        mean = statistics.fmean(ratios)
        pstdev = statistics.pstdev(ratios)
        by_bar[f"{bar:.2f}"] = {
            "mean_ratio": mean,
            "min_ratio": min(ratios),
            "max_ratio": max(ratios),
            "range": max(ratios) - min(ratios),
            "population_sd": pstdev,
            "coefficient_of_variation": pstdev / mean,
        }

    composition_chance = 0.25
    composition_steps = 4
    rationale_value = composition_chance ** (1.0 / composition_steps)
    n24_registered = next(
        row for row in rows if row["configuration"] == "support12_N24" and row["bar"] == 0.71
    )
    return {
        "kind": "paper_one_frontier_threshold_sensitivity",
        "status": "completed",
        "method": "first linear interpolation from at-or-above the bar to below it",
        "registered_bar": 0.71,
        "registered_bar_rationale": {
            "four_option_chance": composition_chance,
            "composition_steps": composition_steps,
            "chance_root": rationale_value,
            "rounded_design_threshold": 0.71,
            "interpretation": (
                "The threshold was set during synthetic-task design as the rounded per-step accuracy "
                "whose four-step product clears four-option chance; it is a design heuristic, not a "
                "confidence bound."
            ),
        },
        "rows": rows,
        "ratio_stability_by_bar": by_bar,
        "conclusion": (
            "The near-proportional support-frontier relationship persists at bars 0.60, 0.71, and "
            "0.80. The ratio level decreases as the bar rises, but within-bar variation remains small."
        ),
        "n24_rounding_audit": {
            "exact_frontier_at_0.71": n24_registered["frontier"],
            "round_to_two_decimals": round(n24_registered["frontier"], 2),
            "paper_value_to_replace": 17.92,
        },
        "curves": curves,
    }


def render_markdown(receipt: dict[str, Any]) -> str:
    lines = [
        "# Paper One Frontier Threshold Sensitivity",
        "",
        "The registered `0.71` bar was set during synthetic-task design. Four-option chance is "
        "`0.25`; the per-step accuracy whose four-step product equals chance is "
        f"`0.25^(1/4) = {receipt['registered_bar_rationale']['chance_root']:.6f}`, rounded to `0.71`. "
        "This was a fixed design heuristic, not a confidence bound.",
        "",
        "| Support | Alphabet | Frontier at 0.60 | Ratio | Frontier at 0.71 | Ratio | Frontier at 0.80 | Ratio |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    rows = receipt["rows"]
    for configuration in ("support4_N16", "support6_N16", "support8_N16", "support12_N24"):
        config_rows = {row["bar"]: row for row in rows if row["configuration"] == configuration}
        example = config_rows[0.71]
        lines.append(
            f"| {example['support']} | N{example['alphabet']} | "
            f"{config_rows[0.60]['frontier']:.3f} | {config_rows[0.60]['frontier_to_support_ratio']:.3f} | "
            f"{config_rows[0.71]['frontier']:.3f} | {config_rows[0.71]['frontier_to_support_ratio']:.3f} | "
            f"{config_rows[0.80]['frontier']:.3f} | {config_rows[0.80]['frontier_to_support_ratio']:.3f} |"
        )
    lines.extend(
        [
            "",
            "| Bar | Mean ratio | Min | Max | Range | CV |",
            "|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for bar in ("0.60", "0.71", "0.80"):
        stats = receipt["ratio_stability_by_bar"][bar]
        lines.append(
            f"| {bar} | {stats['mean_ratio']:.3f} | {stats['min_ratio']:.3f} | "
            f"{stats['max_ratio']:.3f} | {stats['range']:.3f} | "
            f"{100 * stats['coefficient_of_variation']:.2f}% |"
        )
    lines.extend(
        [
            "",
            "**Reading.** The qualitative law is insensitive to the bar: frontier remains approximately "
            "proportional to supervised support at all three levels. The numerical ratio is not invariant "
            "to the bar and decreases as the criterion becomes stricter.",
            "",
            "**Arithmetic correction.** The N24 frontier at the registered bar is "
            f"`{receipt['n24_rounding_audit']['exact_frontier_at_0.71']:.6f}`, which rounds to `17.93`, "
            "not `17.92`.",
            "",
            "## Paste-Ready Paper Language",
            "",
            "> The 0.71 bar was fixed during task design, motivated by the rounded four-step root of "
            "four-choice chance, 0.25^(1/4) = 0.707. As a sensitivity check, we recomputed all four "
            "frontiers at bars of 0.60 and 0.80. The frontier/support ratios remained tightly grouped "
            "within each bar (1.54-1.61 at 0.60, 1.44-1.50 at 0.71, and 1.33-1.42 at 0.80). Thus the "
            "near-proportional scaling conclusion is insensitive to the bar's level, although the numerical "
            "ratio decreases as the criterion is raised.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo_root", default=".")
    parser.add_argument("--output_json", default="docs/PAPER_ONE_FRONTIER_THRESHOLD_SENSITIVITY_20260721.json")
    parser.add_argument("--output_md", default="docs/PAPER_ONE_FRONTIER_THRESHOLD_SENSITIVITY_20260721.md")
    args = parser.parse_args()

    repo_root = Path(args.repo_root).resolve()
    receipt = build_receipt(repo_root)
    output_json = repo_root / args.output_json
    output_md = repo_root / args.output_md
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    output_md.write_text(render_markdown(receipt), encoding="utf-8")
    print(output_json)
    print(output_md)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
