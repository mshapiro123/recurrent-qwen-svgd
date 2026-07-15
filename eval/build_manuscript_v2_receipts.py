#!/usr/bin/env python3
"""Build reproducible manuscript-v2 receipts from frozen Part 1 artifacts."""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean, median
from typing import Any, Iterable


DEFAULT_ROWS = Path(
    "outputs/stage5/stage5_part1_closeout_pivot_20260715/branching_screen/"
    "natural_step2000_N20_verbal/rows.jsonl"
)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def softmax(values: Iterable[float]) -> list[float]:
    values = list(values)
    maximum = max(values)
    exps = [math.exp(value - maximum) for value in values]
    total = sum(exps)
    return [value / total for value in exps]


def entropy_nats(probabilities: Iterable[float]) -> float:
    return -sum(p * math.log(p) for p in probabilities if p > 0.0)


def quantiles(values: list[float]) -> dict[str, float]:
    ordered = sorted(values)

    def nearest(fraction: float) -> float:
        return ordered[round((len(ordered) - 1) * fraction)]

    return {
        "mean": mean(ordered),
        "median": median(ordered),
        "p25": nearest(0.25),
        "p75": nearest(0.75),
        "min": ordered[0],
        "max": ordered[-1],
    }


def summarize_group(rows: list[dict[str, Any]]) -> dict[str, Any]:
    valid = sum(bool(row["valid"]) for row in rows)
    symbols = sorted(rows[0]["scores"])
    score_entropies: list[float] = []
    normalized_entropies: list[float] = []
    top1_probabilities: list[float] = []
    prediction_counts: Counter[str] = Counter()

    for row in rows:
        probabilities = softmax(float(row["scores"][symbol]) for symbol in symbols)
        entropy = entropy_nats(probabilities)
        score_entropies.append(entropy)
        normalized_entropies.append(entropy / math.log(len(symbols)))
        top1_probabilities.append(max(probabilities))
        prediction_counts[str(row["prediction"])] += 1

    empirical = [count / len(rows) for count in prediction_counts.values()]
    modal_symbol, modal_count = prediction_counts.most_common(1)[0]
    return {
        "rows": len(rows),
        "valid": valid,
        "invalid": len(rows) - valid,
        "validity": valid / len(rows),
        "mean_reachable_set_size": mean(int(row["reachable_set_size"]) for row in rows),
        "score_entropy_nats": quantiles(score_entropies),
        "score_entropy_normalized": quantiles(normalized_entropies),
        "top1_softmax_probability": quantiles(top1_probabilities),
        "prediction_distribution": {
            "unique_symbols": len(prediction_counts),
            "modal_symbol": modal_symbol,
            "modal_count": modal_count,
            "modal_rate": modal_count / len(rows),
            "empirical_entropy_nats": entropy_nats(empirical),
            "counts": dict(sorted(prediction_counts.items())),
        },
    }


def build_receipt(rows: list[dict[str, Any]], source: Path) -> dict[str, Any]:
    by_depth: dict[int, list[dict[str, Any]]] = defaultdict(list)
    by_stratum: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_depth_stratum: dict[tuple[int, str], list[dict[str, Any]]] = defaultdict(list)
    size_counts: Counter[int] = Counter()

    for row in rows:
        depth = int(row["depth"])
        stratum = str(row["reachable_set_stratum"])
        by_depth[depth].append(row)
        by_stratum[stratum].append(row)
        by_depth_stratum[(depth, stratum)].append(row)
        size_counts[int(row["reachable_set_size"])] += 1

    return {
        "schema_version": 1,
        "kind": "manuscript_v2_margin_lock_inputs",
        "source_rows": source.as_posix(),
        "overall": summarize_group(rows),
        "by_depth": {str(key): summarize_group(value) for key, value in sorted(by_depth.items())},
        "by_stratum": {key: summarize_group(value) for key, value in sorted(by_stratum.items())},
        "by_depth_and_stratum": {
            f"depth_{depth}__stratum_{stratum}": summarize_group(value)
            for (depth, stratum), value in sorted(by_depth_stratum.items())
        },
        "reachable_set_size_distribution": {
            str(size): {"rows": count, "fraction": count / len(rows)}
            for size, count in sorted(size_counts.items())
        },
    }


def render_markdown(receipt: dict[str, Any]) -> str:
    lines = [
        "# Phase G-alpha Margin-Lock Inputs",
        "",
        f"Source: `{receipt['source_rows']}`",
        "",
        "## Overall",
        "",
        "| Rows | Valid | Invalid | Validity | Mean reachable set | Mean score entropy (nats) | Mean top-1 probability | Modal prediction rate |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    overall = receipt["overall"]
    lines.append(
        f"| {overall['rows']} | {overall['valid']} | {overall['invalid']} | "
        f"{overall['validity']:.4f} | {overall['mean_reachable_set_size']:.3f} | "
        f"{overall['score_entropy_nats']['mean']:.4f} | "
        f"{overall['top1_softmax_probability']['mean']:.4f} | "
        f"{overall['prediction_distribution']['modal_rate']:.4f} |"
    )

    lines.extend(["", "## Validity by depth and stratum", "", "| Depth | Stratum | Valid | Rows | Validity |", "|---:|:---|---:|---:|---:|"])
    for key, value in receipt["by_depth_and_stratum"].items():
        depth, stratum = key.removeprefix("depth_").split("__stratum_")
        lines.append(f"| {depth} | {stratum} | {value['valid']} | {value['rows']} | {value['validity']:.4f} |")

    lines.extend(["", "## Reachable-set-size distribution", "", "| Set size | Rows | Fraction |", "|---:|---:|---:|"])
    for size, value in receipt["reachable_set_size_distribution"].items():
        lines.append(f"| {size} | {value['rows']} | {value['fraction']:.4f} |")

    lines.extend(["", "## Deterministic collapse profile by stratum", "", "| Stratum | Rows | Score entropy (nats) | Normalized entropy | Top-1 probability | Modal symbol | Modal rate | Empirical answer entropy (nats) |", "|:---|---:|---:|---:|---:|:---|---:|---:|"])
    for stratum, value in receipt["by_stratum"].items():
        distribution = value["prediction_distribution"]
        lines.append(
            f"| {stratum} | {value['rows']} | {value['score_entropy_nats']['mean']:.4f} | "
            f"{value['score_entropy_normalized']['mean']:.4f} | "
            f"{value['top1_softmax_probability']['mean']:.4f} | {distribution['modal_symbol']} | "
            f"{distribution['modal_rate']:.4f} | {distribution['empirical_entropy_nats']:.4f} |"
        )

    lines.extend([
        "",
        "## Interpretation boundary",
        "",
        "These statistics characterize the frozen deterministic keeper and set the entropy-matching target for Phase G-alpha comparators. They do not measure stochastic coverage and do not authorize a G-alpha launch without a separately locked powered margin.",
        "",
    ])
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", type=Path, default=DEFAULT_ROWS)
    parser.add_argument("--output_json", type=Path, required=True)
    parser.add_argument("--output_md", type=Path, required=True)
    args = parser.parse_args()

    rows = read_jsonl(args.rows)
    if len(rows) != 512:
        raise ValueError(f"Expected 512 frozen rows, found {len(rows)}")
    receipt = build_receipt(rows, args.rows)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    args.output_md.write_text(render_markdown(receipt), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
