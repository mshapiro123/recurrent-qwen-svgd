"""Analyze whether deeper-loop rescue is predictable from existing MCQ traces.

This is the cheap precursor after the fixed-damper recovery readout. It uses
already-collected forced-depth MCQ rows, labels each example as loop-1 correct,
rescued by a deeper loop, or harmed by a deeper loop, then asks whether simple
selector-available telemetry can separate rescued examples from the rest.

No model loading is required. This is not the final selector; it is the
go/no-go instrument for whether building a selector can pay on the current
substrate.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from eval.analyze_depth_sweep import (  # noqa: E402
    joined_examples,
    load_loop_payloads,
    path_for_cli,
    resolve_path,
    sign_test_p_value,
)


EPS = 1e-12


def finite_float(value: Any) -> float | None:
    if isinstance(value, (int, float)) and math.isfinite(float(value)):
        return float(value)
    return None


def score_entropy(scores: dict[int | str, float] | dict[str, float]) -> float | None:
    values = [float(v) for v in scores.values() if math.isfinite(float(v))]
    if not values:
        return None
    high = max(values)
    exps = [math.exp(value - high) for value in values]
    total = sum(exps)
    if total <= 0.0:
        return None
    probs = [value / total for value in exps]
    return -sum(p * math.log(max(p, EPS)) for p in probs)


def mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def variance(values: list[float]) -> float | None:
    if len(values) < 2:
        return 0.0 if values else None
    return statistics.pvariance(values)


def average_ranks(values: list[float]) -> list[float]:
    order = sorted(range(len(values)), key=lambda idx: values[idx])
    ranks = [0.0] * len(values)
    idx = 0
    while idx < len(order):
        end = idx + 1
        while end < len(order) and values[order[end]] == values[order[idx]]:
            end += 1
        # Ranks are one-based; use the average rank for ties.
        avg_rank = (idx + 1 + end) / 2.0
        for pos in range(idx, end):
            ranks[order[pos]] = avg_rank
        idx = end
    return ranks


def auc_for_binary(scores: list[float], labels: list[bool]) -> float | None:
    n_pos = sum(1 for label in labels if label)
    n_neg = len(labels) - n_pos
    if n_pos == 0 or n_neg == 0:
        return None
    ranks = average_ranks(scores)
    rank_sum_pos = sum(rank for rank, label in zip(ranks, labels, strict=True) if label)
    u_stat = rank_sum_pos - (n_pos * (n_pos + 1) / 2.0)
    return u_stat / (n_pos * n_neg)


def fisher_ratio(pos: list[float], neg: list[float]) -> float | None:
    if not pos or not neg:
        return None
    numerator = (mean(pos) - mean(neg)) ** 2  # type: ignore[operator]
    denominator = (variance(pos) or 0.0) + (variance(neg) or 0.0) + EPS
    return numerator / denominator


def safe_metric(value: float | None) -> float | None:
    if value is None or not math.isfinite(value):
        return None
    return float(value)


def feature_value(example: dict[str, Any], feature: str) -> float | None:
    loop1 = min(example["loop_hits"])
    loop1_diag = example.get("loop_diagnostics", {}).get(loop1, {}) or {}
    base_margin = finite_float(example.get("base_predicted_margin"))
    loop1_margin = finite_float(example.get("loop_predicted_margins", {}).get(loop1))

    if feature == "base_predicted_margin":
        return base_margin
    if feature == "loop1_predicted_margin":
        return loop1_margin
    if feature == "loop1_margin_minus_base_margin":
        if base_margin is None or loop1_margin is None:
            return None
        return loop1_margin - base_margin
    if feature == "loop1_score_entropy":
        return score_entropy(example.get("loop_scores", {}).get(loop1, {}) or {})
    if feature == "loop1_mean_expected_loops":
        return finite_float(loop1_diag.get("mean_expected_loops"))
    if feature == "loop1_mean_halt_entropy":
        return finite_float(loop1_diag.get("mean_halt_entropy"))
    if feature == "loop1_prediction_expected_loops":
        return finite_float(loop1_diag.get("prediction_expected_loops"))
    if feature == "loop1_prediction_halt_entropy":
        return finite_float(loop1_diag.get("prediction_halt_entropy"))
    # Gold-leaking diagnostics, useful only as an upper-bound sanity check.
    if feature == "loop1_answer_margin_gold":
        return finite_float(example.get("loop_answer_margins", {}).get(loop1))
    if feature == "loop1_answer_expected_loops_gold":
        return finite_float(loop1_diag.get("answer_expected_loops"))
    if feature == "loop1_answer_halt_entropy_gold":
        return finite_float(loop1_diag.get("answer_halt_entropy"))
    raise KeyError(feature)


SELECTOR_FEATURES = [
    "base_predicted_margin",
    "loop1_predicted_margin",
    "loop1_margin_minus_base_margin",
    "loop1_score_entropy",
    "loop1_mean_expected_loops",
    "loop1_mean_halt_entropy",
    "loop1_prediction_expected_loops",
    "loop1_prediction_halt_entropy",
]

GOLD_DIAGNOSTIC_FEATURES = [
    "loop1_answer_margin_gold",
    "loop1_answer_expected_loops_gold",
    "loop1_answer_halt_entropy_gold",
]


def label_examples(examples: list[dict[str, Any]], loops: list[int]) -> list[dict[str, Any]]:
    loop1 = min(loops)
    deeper = [loop for loop in loops if loop != loop1]
    labelled: list[dict[str, Any]] = []
    for example in examples:
        loop1_hit = bool(example["loop_hits"][loop1])
        deeper_hits = {loop: bool(example["loop_hits"][loop]) for loop in deeper}
        any_deeper_hit = any(deeper_hits.values())
        any_deeper_miss = any(not hit for hit in deeper_hits.values())
        rescuable = (not loop1_hit) and any_deeper_hit
        harmable = loop1_hit and any_deeper_miss
        if rescuable:
            category = "rescuable"
        elif harmable:
            category = "harmable"
        elif loop1_hit:
            category = "stable_correct"
        else:
            category = "stable_wrong"
        labelled.append(
            {
                **example,
                "loop1_hit": loop1_hit,
                "any_deeper_hit": any_deeper_hit,
                "rescuable": rescuable,
                "harmable": harmable,
                "category": category,
            }
        )
    return labelled


def discrimination_for_feature(
    examples: list[dict[str, Any]],
    feature: str,
    *,
    positive_label: str,
) -> dict[str, Any]:
    pairs: list[tuple[float, bool]] = []
    for example in examples:
        value = feature_value(example, feature)
        if value is None:
            continue
        label = bool(example[positive_label])
        pairs.append((value, label))
    if not pairs:
        return {"feature": feature, "n": 0}
    scores = [pair[0] for pair in pairs]
    labels = [pair[1] for pair in pairs]
    pos = [score for score, label in pairs if label]
    neg = [score for score, label in pairs if not label]
    auc = auc_for_binary(scores, labels)
    oriented_auc = None if auc is None else max(auc, 1.0 - auc)
    direction = None
    if auc is not None:
        direction = "high_predicts_positive" if auc >= 0.5 else "low_predicts_positive"
    return {
        "feature": feature,
        "n": len(pairs),
        "positive": len(pos),
        "negative": len(neg),
        "auc": safe_metric(auc),
        "oriented_auc": safe_metric(oriented_auc),
        "direction": direction,
        "fisher_ratio": safe_metric(fisher_ratio(pos, neg)),
        "positive_mean": safe_metric(mean(pos)),
        "negative_mean": safe_metric(mean(neg)),
    }


def quantile_thresholds(values: list[float]) -> list[float]:
    if not values:
        return []
    sorted_values = sorted(values)
    thresholds: list[float] = []
    for q in [0.05, 0.1, 0.15, 0.2, 0.25, 0.33, 0.4, 0.5, 0.6, 0.67, 0.75, 0.8, 0.85, 0.9, 0.95]:
        idx = min(len(sorted_values) - 1, max(0, round(q * (len(sorted_values) - 1))))
        value = sorted_values[idx]
        if value not in thresholds:
            thresholds.append(value)
    return thresholds


def evaluate_binary_gate(
    examples: list[dict[str, Any]],
    *,
    feature: str,
    threshold: float,
    direction: str,
    fallback_loop: int,
) -> dict[str, Any]:
    loop1 = min(examples[0]["loop_hits"]) if examples else 1
    loop1_correct = sum(1 for example in examples if example["loop_hits"][loop1])
    oracle_correct = sum(
        1 for example in examples if example["loop_hits"][loop1] or example["loop_hits"].get(fallback_loop, False)
    )
    correct = 0
    routed_deep = 0
    wins = 0
    losses = 0
    rescue_captured = 0
    harm_triggered = 0
    eligible = 0
    for example in examples:
        value = feature_value(example, feature)
        if value is None:
            use_deep = False
        else:
            eligible += 1
            use_deep = value >= threshold if direction == "high" else value <= threshold
        chosen_loop = fallback_loop if use_deep else loop1
        routed_deep += int(use_deep)
        hit = bool(example["loop_hits"][chosen_loop])
        correct += int(hit)
        loop1_hit = bool(example["loop_hits"][loop1])
        if use_deep and example["rescuable"] and hit:
            rescue_captured += 1
        if use_deep and example["harmable"] and loop1_hit and not hit:
            harm_triggered += 1
        if hit and not loop1_hit:
            wins += 1
        elif loop1_hit and not hit:
            losses += 1
    gap = max(0, oracle_correct - loop1_correct)
    return {
        "feature": feature,
        "threshold": threshold,
        "direction": direction,
        "fallback_loop": fallback_loop,
        "eligible": eligible,
        "total": len(examples),
        "correct": correct,
        "loop1_correct": loop1_correct,
        "oracle_correct": oracle_correct,
        "delta_vs_loop1": correct - loop1_correct,
        "oracle_gap_capture": None if gap == 0 else (correct - loop1_correct) / gap,
        "routed_deep": routed_deep,
        "wins_vs_loop1": wins,
        "losses_vs_loop1": losses,
        "sign_test_p": sign_test_p_value(wins, losses),
        "rescue_captured": rescue_captured,
        "harm_triggered": harm_triggered,
    }


def binary_gate_sweep(examples: list[dict[str, Any]], loops: list[int], features: list[str]) -> list[dict[str, Any]]:
    deeper = [loop for loop in loops if loop != min(loops)]
    rows: list[dict[str, Any]] = []
    for fallback_loop in deeper:
        for feature in features:
            values = [value for example in examples if (value := feature_value(example, feature)) is not None]
            for threshold in quantile_thresholds(values):
                for direction in ("low", "high"):
                    rows.append(
                        evaluate_binary_gate(
                            examples,
                            feature=feature,
                            threshold=threshold,
                            direction=direction,
                            fallback_loop=fallback_loop,
                        )
                    )
    return sorted(rows, key=lambda row: (row["delta_vs_loop1"], row["correct"], -row["losses_vs_loop1"]), reverse=True)


def category_counts(examples: list[dict[str, Any]]) -> dict[str, int]:
    return dict(Counter(str(example["category"]) for example in examples))


def benchmark_analysis(
    examples: list[dict[str, Any]],
    *,
    loops: list[int],
    benchmark: str,
) -> dict[str, Any]:
    labelled = label_examples(examples, loops)
    loop1 = min(loops)
    any_deeper_correct = sum(1 for ex in labelled if ex["any_deeper_hit"])
    loop1_correct = sum(1 for ex in labelled if ex["loop1_hit"])
    oracle_correct = sum(1 for ex in labelled if ex["loop1_hit"] or ex["any_deeper_hit"])
    selector_features = SELECTOR_FEATURES
    all_features = selector_features + GOLD_DIAGNOSTIC_FEATURES
    rescue_discrimination = [
        discrimination_for_feature(labelled, feature, positive_label="rescuable")
        for feature in all_features
    ]
    harm_discrimination = [
        discrimination_for_feature(labelled, feature, positive_label="harmable")
        for feature in all_features
    ]
    gates = binary_gate_sweep(labelled, loops, selector_features)
    return {
        "benchmark": benchmark,
        "total": len(labelled),
        "loops": loops,
        "loop1_correct": loop1_correct,
        "any_deeper_correct": any_deeper_correct,
        "oracle_correct": oracle_correct,
        "oracle_gain_vs_loop1": oracle_correct - loop1_correct,
        "category_counts": category_counts(labelled),
        "rescue_discrimination": sorted(
            rescue_discrimination,
            key=lambda row: (row.get("oriented_auc") or 0.0, row.get("fisher_ratio") or 0.0),
            reverse=True,
        ),
        "harm_discrimination": sorted(
            harm_discrimination,
            key=lambda row: (row.get("oriented_auc") or 0.0, row.get("fisher_ratio") or 0.0),
            reverse=True,
        ),
        "binary_gate_top10": gates[:10],
        "baseline_policy": {
            "default_loop": loop1,
            "binary_fallback_loop": 2 if 2 in loops else (loops[1] if len(loops) > 1 else loop1),
            "conservative_default": "loop1",
        },
    }


def analyze(sweep_summary: Path, *, score_target: str, aggregate: str) -> dict[str, Any]:
    sweep, loop_payloads = load_loop_payloads(sweep_summary)
    loops = sorted(loop_payloads)
    benchmarks = list(loop_payloads[loops[0]].get("benchmarks", []))
    payload = {
        "kind": "stage5_rescue_predictability_analysis",
        "run_id": "stage5_rescue_predictability_" + time.strftime("%Y%m%d_%H%M%S"),
        "source_sweep_summary": path_for_cli(sweep_summary),
        "source_sweep_run_id": sweep.get("run_id"),
        "score_target": score_target,
        "aggregate": aggregate,
        "loops": loops,
        "feature_note": (
            "Selector features use existing MCQ telemetry. *_gold features are diagnostic-only "
            "and must not be used for deployment selectors."
        ),
        "benchmarks": {},
    }
    for benchmark in benchmarks:
        examples = joined_examples(loop_payloads, benchmark, score_target, aggregate)
        payload["benchmarks"][benchmark] = benchmark_analysis(examples, loops=loops, benchmark=benchmark)
    return payload


def write_markdown(payload: dict[str, Any], path: Path) -> None:
    lines = [
        f"# Rescue Predictability - {payload['source_sweep_run_id']}",
        "",
        f"- Source: `{payload['source_sweep_summary']}`",
        f"- Score target: `{payload['score_target']}`",
        f"- Aggregate: `{payload['aggregate']}`",
        f"- Loops: `{payload['loops']}`",
        "",
        "This is the cheap precursor: it asks whether deeper-loop rescue is predictable enough "
        "to justify a selector before spending GPU on more recurrence or particle work.",
        "",
    ]
    for benchmark, result in payload["benchmarks"].items():
        lines.extend(
            [
                f"## {benchmark}",
                "",
                f"- Loop-1 correct: `{result['loop1_correct']}/{result['total']}`",
                f"- Oracle correct over loop-1/deeper: `{result['oracle_correct']}/{result['total']}`",
                f"- Oracle gain vs loop-1: `{result['oracle_gain_vs_loop1']}`",
                f"- Categories: `{result['category_counts']}`",
                "",
                "### Rescue Predictability",
                "",
            ]
        )
        for row in result["rescue_discrimination"][:8]:
            lines.append(
                f"- `{row['feature']}`: oriented AUC `{row.get('oriented_auc')}`, "
                f"raw AUC `{row.get('auc')}`, direction `{row.get('direction')}`, "
                f"Fisher `{row.get('fisher_ratio')}`"
            )
        lines.extend(["", "### Conservative Binary Gates", ""])
        for row in result["binary_gate_top10"]:
            lines.append(
                f"- `{row['feature']}` {row['direction']} threshold `{row['threshold']}` -> loop "
                f"`{row['fallback_loop']}`: correct `{row['correct']}/{row['total']}`, "
                f"delta `{row['delta_vs_loop1']}`, gap capture `{row['oracle_gap_capture']}`, "
                f"routed `{row['routed_deep']}`, W/L `{row['wins_vs_loop1']}/{row['losses_vs_loop1']}`, "
                f"rescued/harmed `{row['rescue_captured']}/{row['harm_triggered']}`"
            )
        lines.extend(["", "### Harm Predictability", ""])
        for row in result["harm_discrimination"][:5]:
            lines.append(
                f"- `{row['feature']}`: oriented AUC `{row.get('oriented_auc')}`, "
                f"direction `{row.get('direction')}`, Fisher `{row.get('fisher_ratio')}`"
            )
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sweep_summary", required=True)
    parser.add_argument("--score_target", default="content_question_only")
    parser.add_argument("--aggregate", default="mean")
    parser.add_argument("--output_dir", default="")
    args = parser.parse_args()

    sweep_summary = resolve_path(args.sweep_summary)
    payload = analyze(sweep_summary, score_target=args.score_target, aggregate=args.aggregate)
    output_dir = resolve_path(args.output_dir) if args.output_dir else sweep_summary.parent / payload["run_id"]
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "summary.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    write_markdown(payload, output_dir / "summary.md")
    print((output_dir / "summary.md").read_text(encoding="utf-8"), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
