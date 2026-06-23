"""Train/test split analysis for recurrent loop-depth selectors.

`analyze_depth_sweep.py` answers the in-sample question: "is there a selector
family that can recover some deeper-loop signal?"  This script answers the next
question: "does a selector chosen on one slice survive on a held-out slice?"

The selectors are intentionally simple and inspectable. This is not meant to be
the final router; it is a cheap gate before spending GPU time on depth SFT.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from eval.analyze_depth_sweep import (  # noqa: E402
    aggregate_scores,
    analyze,
    evaluate_score_selector,
    evaluate_threshold_router,
    joined_examples,
    load_loop_payloads,
    path_for_cli,
    resolve_path,
    sign_test_p_value,
)


THRESHOLDS = [0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0]
WEIGHTS = [0.1, 0.25, 0.5, 0.75, 1.0]


def parse_int_csv(value: str) -> list[int]:
    seeds = [int(part.strip()) for part in value.split(",") if part.strip()]
    if not seeds:
        raise argparse.ArgumentTypeError("expected at least one integer")
    return seeds


def mean(values: list[float]) -> float | None:
    if not values:
        return None
    return sum(values) / len(values)


def stable_split_key(example: dict[str, Any], *, benchmark: str, seed: int) -> int:
    raw = f"{benchmark}\0{example['id']}\0{seed}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(raw).digest()[:8], "big")


def split_examples(
    examples: list[dict[str, Any]],
    *,
    benchmark: str,
    seed: int,
    train_fraction: float,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if not 0.0 < train_fraction < 1.0:
        raise ValueError("--train_fraction must be between 0 and 1")
    ordered = sorted(examples, key=lambda ex: stable_split_key(ex, benchmark=benchmark, seed=seed))
    split = max(1, min(len(ordered) - 1, int(round(len(ordered) * train_fraction))))
    return ordered[:split], ordered[split:]


def threshold_candidates(loops: list[int]) -> list[dict[str, Any]]:
    return [
        {
            "family": "threshold_router",
            "source": source,
            "threshold": threshold,
            "fallback_loop": fallback_loop,
        }
        for source in ["base", "loop1"]
        for fallback_loop in loops
        if fallback_loop != 1
        for threshold in THRESHOLDS
    ]


def score_candidates(loops: list[int]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for end_idx in range(1, len(loops) + 1):
        subset = loops[:end_idx]
        rows.append({"family": "score_selector", "subset": subset, "method": "mean", "weight": None})
        rows.append({"family": "score_selector", "subset": subset, "method": "max", "weight": None})
        if 1 in subset and len(subset) > 1:
            for weight in WEIGHTS:
                rows.append(
                    {
                        "family": "score_selector",
                        "subset": subset,
                        "method": "loop1_plus_weighted_deeper",
                        "weight": weight,
                    }
                )
    return rows


def candidate_spec(candidate: dict[str, Any]) -> dict[str, Any]:
    if candidate["family"] == "threshold_router":
        return {
            "family": "threshold_router",
            "source": candidate["source"],
            "threshold": candidate["threshold"],
            "fallback_loop": candidate["fallback_loop"],
        }
    if candidate["family"] == "score_selector":
        return {
            "family": "score_selector",
            "subset": list(candidate["subset"]),
            "method": candidate["method"],
            "weight": candidate.get("weight"),
        }
    raise ValueError(candidate["family"])


def evaluate_candidate(examples: list[dict[str, Any]], candidate: dict[str, Any]) -> dict[str, Any]:
    spec = candidate_spec(candidate)
    if candidate["family"] == "threshold_router":
        result = evaluate_threshold_router(
            examples,
            source=str(candidate["source"]),
            threshold=float(candidate["threshold"]),
            fallback_loop=int(candidate["fallback_loop"]),
        )
        return {**spec, **result}
    elif candidate["family"] == "score_selector":
        result = evaluate_score_selector(
            examples,
            subset=tuple(int(loop) for loop in candidate["subset"]),
            method=str(candidate["method"]),
            weight=candidate.get("weight"),
        )
        return {**spec, **result, "method": spec["method"], "display_method": result["method"]}
    else:
        raise ValueError(candidate["family"])


def loop1_baseline(examples: list[dict[str, Any]]) -> dict[str, Any]:
    correct = sum(1 for ex in examples if ex["loop_hits"][1])
    base_correct = sum(1 for ex in examples if ex["base_hit"])
    wins_vs_base = sum(1 for ex in examples if ex["loop_hits"][1] and not ex["base_hit"])
    losses_vs_base = sum(1 for ex in examples if ex["base_hit"] and not ex["loop_hits"][1])
    return {
        "total": len(examples),
        "base_correct": base_correct,
        "loop1_correct": correct,
        "loop1_delta_vs_base": correct - base_correct,
        "loop1_wins_vs_base": wins_vs_base,
        "loop1_losses_vs_base": losses_vs_base,
        "loop1_sign_test_p_vs_base": sign_test_p_value(wins_vs_base, losses_vs_base),
    }


def oracle_summary(examples: list[dict[str, Any]], loops: list[int]) -> dict[str, Any]:
    loop1_correct = sum(1 for ex in examples if ex["loop_hits"][1])
    any_correct = sum(1 for ex in examples if any(ex["loop_hits"][loop] for loop in loops))
    base_or_any = sum(1 for ex in examples if ex["base_hit"] or any(ex["loop_hits"][loop] for loop in loops))
    return {
        "any_depth_correct": any_correct,
        "any_depth_gain_vs_loop1": any_correct - loop1_correct,
        "base_or_any_depth_correct": base_or_any,
    }


def select_best(candidates: list[dict[str, Any]]) -> dict[str, Any]:
    return max(
        candidates,
        key=lambda row: (
            int(row["delta_vs_loop1"]),
            int(row["wins_vs_loop1"]) - int(row["losses_vs_loop1"]),
            int(row["correct"]),
            -int(row.get("routed_deep", 0)),
        ),
    )


def summarize_prediction_counts(row: dict[str, Any]) -> dict[str, Any]:
    counts = row.get("prediction_counts")
    if not isinstance(counts, dict):
        return {}
    total = sum(int(v) for v in counts.values())
    if total <= 0:
        return {}
    top_label, top_count = max(counts.items(), key=lambda item: int(item[1]))
    return {
        "top_prediction": top_label,
        "top_prediction_fraction": int(top_count) / total,
    }


def analyze_split(
    sweep_summary: Path,
    *,
    seed: int,
    train_fraction: float,
    include_reference_analysis: bool = True,
) -> dict[str, Any]:
    sweep, loop_payloads = load_loop_payloads(sweep_summary)
    loops = sorted(loop_payloads)
    benchmarks = list(loop_payloads[loops[0]].get("benchmarks", []))
    payload: dict[str, Any] = {
        "kind": "stage5_depth_selector_split",
        "source_sweep_summary": path_for_cli(sweep_summary),
        "source_sweep_run_id": sweep.get("run_id"),
        "seed": seed,
        "train_fraction": train_fraction,
        "loops": loops,
        "benchmarks": {},
    }

    for benchmark in benchmarks:
        examples = joined_examples(loop_payloads, benchmark)
        train, test = split_examples(
            examples,
            benchmark=benchmark,
            seed=seed,
            train_fraction=train_fraction,
        )
        candidates = threshold_candidates(loops) + score_candidates(loops)
        train_rows = [evaluate_candidate(train, candidate) for candidate in candidates]
        best = select_best(train_rows)
        test_row = evaluate_candidate(test, best)
        payload["benchmarks"][benchmark] = {
            "train": {
                "baseline": loop1_baseline(train),
                "oracle": oracle_summary(train, loops),
                "best_selector": best,
            },
            "test": {
                "baseline": loop1_baseline(test),
                "oracle": oracle_summary(test, loops),
                "selected_selector": test_row,
                "prediction_bias": summarize_prediction_counts(test_row),
            },
            "all_train_candidates_top10": sorted(
                train_rows,
                key=lambda row: (row["delta_vs_loop1"], row["correct"]),
                reverse=True,
            )[:10],
        }

    if include_reference_analysis:
        payload["reference_in_sample_analysis"] = analyze(sweep_summary)
    return payload


def selector_signature(row: dict[str, Any]) -> str:
    if row["family"] == "threshold_router":
        return (
            f"threshold:{row['source']}<"
            f"{row['threshold']}->loop{row['fallback_loop']}"
        )
    method = row.get("display_method", row.get("method"))
    subset = ",".join(str(loop) for loop in row.get("subset", []))
    return f"score:{method}[{subset}]"


def summarize_many_splits(splits: list[dict[str, Any]]) -> dict[str, Any]:
    if not splits:
        raise ValueError("No split payloads to summarize")
    benchmarks = sorted(splits[0]["benchmarks"])
    summary: dict[str, Any] = {
        "kind": "stage5_depth_selector_split_stability",
        "source_sweep_summary": splits[0]["source_sweep_summary"],
        "source_sweep_run_id": splits[0]["source_sweep_run_id"],
        "seeds": [row["seed"] for row in splits],
        "train_fraction": splits[0]["train_fraction"],
        "loops": splits[0]["loops"],
        "benchmarks": {},
    }
    for benchmark in benchmarks:
        rows = [split["benchmarks"][benchmark] for split in splits]
        deltas = [int(row["test"]["selected_selector"]["delta_vs_loop1"]) for row in rows]
        correct = [int(row["test"]["selected_selector"]["correct"]) for row in rows]
        loop1 = [int(row["test"]["baseline"]["loop1_correct"]) for row in rows]
        base = [int(row["test"]["baseline"]["base_correct"]) for row in rows]
        oracle_gains = [int(row["test"]["oracle"]["any_depth_gain_vs_loop1"]) for row in rows]
        wins = [int(row["test"]["selected_selector"]["wins_vs_loop1"]) for row in rows]
        losses = [int(row["test"]["selected_selector"]["losses_vs_loop1"]) for row in rows]
        signatures = Counter(selector_signature(row["train"]["best_selector"]) for row in rows)
        summary["benchmarks"][benchmark] = {
            "num_splits": len(rows),
            "test_total_per_split": rows[0]["test"]["baseline"]["total"],
            "mean_test_selected_correct": mean([float(v) for v in correct]),
            "mean_test_loop1_correct": mean([float(v) for v in loop1]),
            "mean_test_base_correct": mean([float(v) for v in base]),
            "mean_test_delta_vs_loop1": mean([float(v) for v in deltas]),
            "min_test_delta_vs_loop1": min(deltas),
            "max_test_delta_vs_loop1": max(deltas),
            "positive_delta_splits": sum(1 for value in deltas if value > 0),
            "negative_delta_splits": sum(1 for value in deltas if value < 0),
            "zero_delta_splits": sum(1 for value in deltas if value == 0),
            "mean_test_oracle_gain_vs_loop1": mean([float(v) for v in oracle_gains]),
            "mean_test_wins_vs_loop1": mean([float(v) for v in wins]),
            "mean_test_losses_vs_loop1": mean([float(v) for v in losses]),
            "selector_signatures": dict(signatures),
            "per_seed": [
                {
                    "seed": split["seed"],
                    "selected_correct": row["test"]["selected_selector"]["correct"],
                    "loop1_correct": row["test"]["baseline"]["loop1_correct"],
                    "base_correct": row["test"]["baseline"]["base_correct"],
                    "delta_vs_loop1": row["test"]["selected_selector"]["delta_vs_loop1"],
                    "wins_vs_loop1": row["test"]["selected_selector"]["wins_vs_loop1"],
                    "losses_vs_loop1": row["test"]["selected_selector"]["losses_vs_loop1"],
                    "oracle_gain_vs_loop1": row["test"]["oracle"]["any_depth_gain_vs_loop1"],
                    "selector": selector_signature(row["train"]["best_selector"]),
                }
                for split, row in zip(splits, rows)
            ],
        }
    return summary


def write_markdown(payload: dict[str, Any], path: Path) -> None:
    lines = [
        f"# Depth Selector Split - {payload['source_sweep_run_id']}",
        "",
        f"- Source: `{payload['source_sweep_summary']}`",
        f"- Seed: `{payload['seed']}`",
        f"- Train fraction: `{payload['train_fraction']}`",
        f"- Loops: `{payload['loops']}`",
        "",
    ]
    for benchmark, result in payload["benchmarks"].items():
        train_base = result["train"]["baseline"]
        test_base = result["test"]["baseline"]
        train_sel = result["train"]["best_selector"]
        test_sel = result["test"]["selected_selector"]
        train_oracle = result["train"]["oracle"]
        test_oracle = result["test"]["oracle"]
        lines.extend(
            [
                f"## {benchmark}",
                "",
                "### Baselines",
                "",
                f"- train loop1: `{train_base['loop1_correct']}/{train_base['total']}` "
                f"(base `{train_base['base_correct']}/{train_base['total']}`, "
                f"delta `{train_base['loop1_delta_vs_base']}`)",
                f"- test loop1: `{test_base['loop1_correct']}/{test_base['total']}` "
                f"(base `{test_base['base_correct']}/{test_base['total']}`, "
                f"delta `{test_base['loop1_delta_vs_base']}`)",
                f"- train any-depth oracle: `{train_oracle['any_depth_correct']}/{train_base['total']}` "
                f"(gain vs loop1 `{train_oracle['any_depth_gain_vs_loop1']}`)",
                f"- test any-depth oracle: `{test_oracle['any_depth_correct']}/{test_base['total']}` "
                f"(gain vs loop1 `{test_oracle['any_depth_gain_vs_loop1']}`)",
                "",
                "### Selector Chosen On Train",
                "",
                f"- family: `{train_sel['family']}`",
                f"- train correct: `{train_sel['correct']}/{train_sel['total']}` "
                f"(delta vs loop1 `{train_sel['delta_vs_loop1']}`, "
                f"W/L `{train_sel['wins_vs_loop1']}/{train_sel['losses_vs_loop1']}`, "
                f"p `{train_sel['sign_test_p']}`)",
                f"- test correct: `{test_sel['correct']}/{test_sel['total']}` "
                f"(delta vs loop1 `{test_sel['delta_vs_loop1']}`, "
                f"W/L `{test_sel['wins_vs_loop1']}/{test_sel['losses_vs_loop1']}`, "
                f"p `{test_sel['sign_test_p']}`)",
                "",
            ]
        )
        if train_sel["family"] == "threshold_router":
            lines.extend(
                [
                    f"- source: `{train_sel['source']}`",
                    f"- threshold: `{train_sel['threshold']}`",
                    f"- fallback loop: `{train_sel['fallback_loop']}`",
                    f"- test routed deep: `{test_sel['routed_deep']}/{test_sel['total']}`",
                    "",
                ]
            )
        else:
            display_method = train_sel.get("display_method", train_sel["method"])
            lines.extend(
                [
                    f"- subset: `{train_sel['subset']}`",
                    f"- method: `{display_method}`",
                    "",
                ]
            )
        bias = result["test"].get("prediction_bias") or {}
        if bias:
            lines.extend(
                [
                    "### Test Prediction Bias",
                    "",
                    f"- top prediction: `{bias['top_prediction']}`",
                    f"- top prediction fraction: `{bias['top_prediction_fraction']:.3f}`",
                    "",
                ]
            )
    path.write_text("\n".join(lines), encoding="utf-8")


def write_stability_markdown(payload: dict[str, Any], path: Path) -> None:
    lines = [
        f"# Depth Selector Split Stability - {payload['source_sweep_run_id']}",
        "",
        f"- Source: `{payload['source_sweep_summary']}`",
        f"- Seeds: `{payload['seeds']}`",
        f"- Train fraction: `{payload['train_fraction']}`",
        f"- Loops: `{payload['loops']}`",
        "",
    ]
    for benchmark, result in payload["benchmarks"].items():
        lines.extend(
            [
                f"## {benchmark}",
                "",
                f"- splits: `{result['num_splits']}`",
                f"- test total per split: `{result['test_total_per_split']}`",
                f"- mean base correct: `{result['mean_test_base_correct']:.3f}`",
                f"- mean loop1 correct: `{result['mean_test_loop1_correct']:.3f}`",
                f"- mean selected correct: `{result['mean_test_selected_correct']:.3f}`",
                f"- mean selected delta vs loop1: `{result['mean_test_delta_vs_loop1']:.3f}` "
                f"(min `{result['min_test_delta_vs_loop1']}`, max `{result['max_test_delta_vs_loop1']}`)",
                f"- split signs: +`{result['positive_delta_splits']}` / "
                f"0`{result['zero_delta_splits']}` / -`{result['negative_delta_splits']}`",
                f"- mean any-depth oracle gain vs loop1: `{result['mean_test_oracle_gain_vs_loop1']:.3f}`",
                f"- mean W/L vs loop1: `{result['mean_test_wins_vs_loop1']:.3f}`/"
                f"`{result['mean_test_losses_vs_loop1']:.3f}`",
                "",
                "### Selector Choices",
                "",
            ]
        )
        for signature, count in sorted(result["selector_signatures"].items(), key=lambda item: (-item[1], item[0])):
            lines.append(f"- `{signature}`: `{count}`")
        lines.extend(["", "### Per Seed", ""])
        for row in result["per_seed"]:
            lines.append(
                f"- seed `{row['seed']}`: selected `{row['selected_correct']}`, "
                f"loop1 `{row['loop1_correct']}`, base `{row['base_correct']}`, "
                f"delta `{row['delta_vs_loop1']}`, W/L "
                f"`{row['wins_vs_loop1']}/{row['losses_vs_loop1']}`, "
                f"oracle gain `{row['oracle_gain_vs_loop1']}`, selector `{row['selector']}`"
            )
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sweep_summary", required=True)
    parser.add_argument("--output_dir", default="")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--seeds", type=parse_int_csv, default=None)
    parser.add_argument("--train_fraction", type=float, default=0.5)
    args = parser.parse_args()

    sweep_summary = resolve_path(args.sweep_summary)
    if args.seeds is not None:
        splits = [
            analyze_split(
                sweep_summary,
                seed=seed,
                train_fraction=args.train_fraction,
                include_reference_analysis=False,
            )
            for seed in args.seeds
        ]
        payload = summarize_many_splits(splits)
        output_dir = (
            resolve_path(args.output_dir)
            if args.output_dir
            else sweep_summary.parent / "selector_split_stability"
        )
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "summary.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
        write_stability_markdown(payload, output_dir / "summary.md")
        print((output_dir / "summary.md").read_text(encoding="utf-8"))
        return 0

    payload = analyze_split(sweep_summary, seed=args.seed, train_fraction=args.train_fraction)
    output_dir = (
        resolve_path(args.output_dir)
        if args.output_dir
        else sweep_summary.parent / f"selector_split_seed{args.seed}"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "summary.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    write_markdown(payload, output_dir / "summary.md")
    print((output_dir / "summary.md").read_text(encoding="utf-8"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
