"""K-fold validation for conservative recurrent-depth rescue selectors.

This is the final deterministic-recurrence gate. It assumes a forced-depth
MCQ sweep already exists, labels loop-1 misses rescued by deeper loops, and
tests whether a low-dimensional regularized selector can route to a moderate
deeper loop out of sample without giving back the gains as harm.

The selector uses only loop-1/base telemetry features from
``analyze_rescue_predictability.SELECTOR_FEATURES``. It does not inspect raw
hidden states or answer labels at inference time.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from eval.analyze_rescue_predictability import SELECTOR_FEATURES, sign_test_p_value  # noqa: E402
from eval.evaluate_rescue_selector_transfer import (  # noqa: E402
    DEFAULT_PROBE_SHRINKAGES,
    best_result,
    category_counts,
    evaluate_score_gate,
    examples_for_sweep,
    fit_feature_stats,
    fit_whitened_direction,
    rescue_discrimination,
    score_gate_sweep,
    transform_features,
)
from eval.analyze_depth_sweep import path_for_cli, resolve_path  # noqa: E402


POLICY_SPECS = [
    ("zero_harm", 0),
    ("harm_budget_1", 1),
    ("harm_budget_2", 2),
    ("max_net", None),
]


def parse_csv(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def parse_floats(value: str) -> list[float]:
    return [float(item.strip()) for item in value.split(",") if item.strip()]


def stable_fold(example: dict[str, Any], *, folds: int, seed: int) -> int:
    raw = f"{example['benchmark']}\0{example['id']}\0{seed}".encode("utf-8")
    value = int.from_bytes(hashlib.sha256(raw).digest()[:8], "big")
    return value % folds


def load_pooled_examples(
    sweep_summary: Path,
    *,
    benchmarks: list[str],
    score_target: str,
    aggregate: str,
) -> tuple[list[int], list[dict[str, Any]], dict[str, dict[str, Any]]]:
    pooled: list[dict[str, Any]] = []
    loops: list[int] | None = None
    per_benchmark: dict[str, dict[str, Any]] = {}
    for benchmark in benchmarks:
        bench_loops, examples = examples_for_sweep(
            sweep_summary,
            benchmark=benchmark,
            score_target=score_target,
            aggregate=aggregate,
        )
        if loops is None:
            loops = bench_loops
        elif loops != bench_loops:
            raise ValueError(f"Loop mismatch for {benchmark}: {bench_loops} != {loops}")
        tagged = [{**example, "benchmark": benchmark} for example in examples]
        pooled.extend(tagged)
        per_benchmark[benchmark] = {
            "total": len(tagged),
            "category_counts": category_counts(tagged),
            "rescue_discrimination": rescue_discrimination(tagged),
        }
    if loops is None:
        raise ValueError("No benchmarks loaded")
    return loops, pooled, per_benchmark


def loop1_correct(examples: list[dict[str, Any]]) -> int:
    return sum(1 for example in examples if example["loop_hits"][min(example["loop_hits"])])


def any_depth_correct(examples: list[dict[str, Any]], loops: list[int]) -> int:
    return sum(1 for example in examples if any(example["loop_hits"].get(loop, False) for loop in loops))


def train_probe_curve(
    train_examples: list[dict[str, Any]],
    loops: list[int],
    *,
    shrinkage: float,
) -> dict[str, Any] | None:
    stats = fit_feature_stats(train_examples, SELECTOR_FEATURES)
    x = transform_features(train_examples, stats)
    labels = torch.tensor([example["category"] == "rescuable" for example in train_examples], dtype=torch.bool)
    direction = fit_whitened_direction(x, labels, shrinkage=shrinkage)
    if direction is None:
        return None
    train_scores = [float(score) for score in (x @ direction).tolist()]
    curve = score_gate_sweep(
        train_examples,
        train_scores,
        loops,
        feature=f"whitened_rescue_score_shrinkage_{shrinkage:g}",
    )
    return {
        "shrinkage": shrinkage,
        "stats": stats,
        "direction": [float(value) for value in direction.tolist()],
        "curve": curve,
        "train_curve_summary": {
            label: best_result(curve, label=label, harm_budget=harm_budget)
            for label, harm_budget in POLICY_SPECS
        },
    }


def apply_fixed_policy(
    test_examples: list[dict[str, Any]],
    probe: dict[str, Any],
    policy: dict[str, Any],
) -> dict[str, Any]:
    x = transform_features(test_examples, probe["stats"])
    direction = torch.tensor(probe["direction"], dtype=torch.float64)
    scores = [float(score) for score in (x @ direction).tolist()]
    return evaluate_score_gate(
        test_examples,
        scores,
        feature=str(policy["feature"]),
        threshold=float(policy["threshold"]),
        direction=str(policy["direction"]),
        fallback_loop=int(policy["fallback_loop"]),
    )


def zero_result(total: int, loop1: int, oracle: int, *, policy_label: str, shrinkage: float) -> dict[str, Any]:
    return {
        "policy_label": policy_label,
        "shrinkage": shrinkage,
        "folds": 0,
        "total": total,
        "loop1_correct": loop1,
        "oracle_correct": oracle,
        "correct": loop1,
        "routed_deep": 0,
        "wins_vs_loop1": 0,
        "losses_vs_loop1": 0,
        "rescue_captured": 0,
        "harm_triggered": 0,
        "train_policy_unavailable_folds": 0,
    }


def aggregate_fold_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {}
    first = rows[0]
    out = {
        "policy_label": first["policy_label"],
        "shrinkage": first["shrinkage"],
        "folds": len(rows),
        "total": sum(int(row["total"]) for row in rows),
        "loop1_correct": sum(int(row["loop1_correct"]) for row in rows),
        "oracle_correct": sum(int(row["oracle_correct"]) for row in rows),
        "correct": sum(int(row["correct"]) for row in rows),
        "routed_deep": sum(int(row["routed_deep"]) for row in rows),
        "wins_vs_loop1": sum(int(row["wins_vs_loop1"]) for row in rows),
        "losses_vs_loop1": sum(int(row["losses_vs_loop1"]) for row in rows),
        "rescue_captured": sum(int(row["rescue_captured"]) for row in rows),
        "harm_triggered": sum(int(row["harm_triggered"]) for row in rows),
        "train_policy_unavailable_folds": sum(int(row.get("train_policy_unavailable", 0)) for row in rows),
    }
    out["delta_vs_loop1"] = out["correct"] - out["loop1_correct"]
    gap = max(0, out["oracle_correct"] - out["loop1_correct"])
    out["oracle_gap_capture"] = None if gap == 0 else out["delta_vs_loop1"] / gap
    out["sign_test_p"] = sign_test_p_value(out["wins_vs_loop1"], out["losses_vs_loop1"])
    out["accuracy"] = out["correct"] / out["total"] if out["total"] else None
    out["loop1_accuracy"] = out["loop1_correct"] / out["total"] if out["total"] else None
    out["harm_rate"] = out["harm_triggered"] / out["total"] if out["total"] else None
    out["rescue_rate"] = out["rescue_captured"] / out["total"] if out["total"] else None
    return out


def conservative_primary(
    aggregate_rows: list[dict[str, Any]],
    *,
    primary_shrinkage: float | None,
) -> dict[str, Any] | None:
    candidates = [
        row
        for row in aggregate_rows
        if row.get("policy_label") in {"zero_harm", "harm_budget_1"}
        and (primary_shrinkage is None or math.isclose(float(row["shrinkage"]), primary_shrinkage))
    ]
    if not candidates:
        candidates = [row for row in aggregate_rows if row.get("policy_label") in {"zero_harm", "harm_budget_1"}]
    if not candidates:
        return None
    return sorted(
        candidates,
        key=lambda row: (
            int(row.get("delta_vs_loop1", 0)),
            int(row.get("rescue_captured", 0)),
            -int(row.get("harm_triggered", 0)),
            -int(row.get("routed_deep", 0)),
        ),
        reverse=True,
    )[0]


def analyze_kfold(
    *,
    sweep_summary: Path,
    benchmarks: list[str],
    score_target: str,
    aggregate: str,
    folds: int,
    seed: int,
    shrinkages: list[float],
    primary_shrinkage: float | None,
    run_id: str | None = None,
) -> dict[str, Any]:
    loops, examples, per_benchmark = load_pooled_examples(
        sweep_summary,
        benchmarks=benchmarks,
        score_target=score_target,
        aggregate=aggregate,
    )
    if folds < 2:
        raise ValueError("--folds must be >= 2")
    fold_rows: list[dict[str, Any]] = []
    fold_details: list[dict[str, Any]] = []
    for fold in range(folds):
        train = [example for example in examples if stable_fold(example, folds=folds, seed=seed) != fold]
        test = [example for example in examples if stable_fold(example, folds=folds, seed=seed) == fold]
        fold_detail: dict[str, Any] = {
            "fold": fold,
            "train_total": len(train),
            "test_total": len(test),
            "train_categories": category_counts(train),
            "test_categories": category_counts(test),
            "selected": [],
        }
        for shrinkage in shrinkages:
            probe = train_probe_curve(train, loops, shrinkage=shrinkage)
            for label, _harm_budget in POLICY_SPECS:
                policy = (probe or {}).get("train_curve_summary", {}).get(label)
                if not probe or not policy:
                    row = zero_result(
                        len(test),
                        loop1_correct(test),
                        any_depth_correct(test, loops),
                        policy_label=label,
                        shrinkage=shrinkage,
                    )
                    row["fold"] = fold
                    row["train_policy_unavailable"] = 1
                else:
                    applied = apply_fixed_policy(test, probe, policy)
                    row = {
                        **applied,
                        "fold": fold,
                        "policy_label": label,
                        "shrinkage": shrinkage,
                        "train_policy_unavailable": 0,
                        "train_policy": {
                            key: policy.get(key)
                            for key in [
                                "feature",
                                "threshold",
                                "direction",
                                "fallback_loop",
                                "delta_vs_loop1",
                                "rescue_captured",
                                "harm_triggered",
                                "routed_deep",
                            ]
                        },
                    }
                fold_rows.append(row)
                if label in {"zero_harm", "harm_budget_1"}:
                    fold_detail["selected"].append(
                        {
                            "policy_label": label,
                            "shrinkage": shrinkage,
                            "correct": row["correct"],
                            "delta_vs_loop1": row.get("delta_vs_loop1"),
                            "rescue_captured": row["rescue_captured"],
                            "harm_triggered": row["harm_triggered"],
                            "routed_deep": row["routed_deep"],
                        }
                    )
        fold_details.append(fold_detail)

    grouped: dict[tuple[str, float], list[dict[str, Any]]] = defaultdict(list)
    for row in fold_rows:
        grouped[(str(row["policy_label"]), float(row["shrinkage"]))].append(row)
    aggregate_rows = [
        aggregate_fold_rows(rows)
        for _key, rows in sorted(grouped.items(), key=lambda item: (item[0][0], item[0][1]))
    ]
    aggregate_rows = sorted(
        aggregate_rows,
        key=lambda row: (
            int(row.get("delta_vs_loop1", 0)),
            int(row.get("rescue_captured", 0)),
            -int(row.get("harm_triggered", 0)),
            -int(row.get("routed_deep", 0)),
        ),
        reverse=True,
    )
    primary = conservative_primary(aggregate_rows, primary_shrinkage=primary_shrinkage)
    status = "selector_transfer_failed"
    if primary and int(primary.get("delta_vs_loop1", 0)) > 0 and int(primary.get("harm_triggered", 0)) <= folds:
        status = "selector_transfer_passed"
    elif primary and int(primary.get("rescue_captured", 0)) > int(primary.get("harm_triggered", 0)):
        status = "selector_transfer_needs_review"
    return {
        "kind": "stage5_rescue_selector_kfold",
        "run_id": run_id or "stage5_rescue_selector_kfold_" + time.strftime("%Y%m%d_%H%M%S"),
        "status": status,
        "source_sweep_summary": path_for_cli(sweep_summary),
        "benchmarks": benchmarks,
        "score_target": score_target,
        "aggregate": aggregate,
        "loops": loops,
        "folds": folds,
        "seed": seed,
        "shrinkages": shrinkages,
        "primary_shrinkage": primary_shrinkage,
        "pooled": {
            "total": len(examples),
            "loop1_correct": loop1_correct(examples),
            "any_depth_correct": any_depth_correct(examples, loops),
            "oracle_gain_vs_loop1": any_depth_correct(examples, loops) - loop1_correct(examples),
            "category_counts": category_counts(examples),
            "rescue_discrimination": rescue_discrimination(examples),
        },
        "per_benchmark": per_benchmark,
        "primary_conservative_result": primary,
        "aggregate_policy_results": aggregate_rows,
        "fold_details": fold_details,
    }


def write_markdown(payload: dict[str, Any], path: Path) -> None:
    pooled = payload["pooled"]
    primary = payload.get("primary_conservative_result") or {}
    lines = [
        f"# Rescue Selector K-Fold - {payload['run_id']}",
        "",
        f"- Status: `{payload['status']}`",
        f"- Source sweep: `{payload['source_sweep_summary']}`",
        f"- Benchmarks: `{payload['benchmarks']}`",
        f"- Score target / aggregate: `{payload['score_target']}` / `{payload['aggregate']}`",
        f"- Loops: `{payload['loops']}`",
        f"- Folds: `{payload['folds']}`",
        f"- Shrinkages: `{payload['shrinkages']}`",
        f"- Primary shrinkage: `{payload['primary_shrinkage']}`",
        "",
        "## Pooled Oracle",
        "",
        f"- Total: `{pooled['total']}`",
        f"- Loop 1 correct: `{pooled['loop1_correct']}`",
        f"- Any-depth correct: `{pooled['any_depth_correct']}`",
        f"- Oracle gain vs loop 1: `{pooled['oracle_gain_vs_loop1']}`",
        f"- Categories: `{pooled['category_counts']}`",
        "",
        "## Primary Conservative Result",
        "",
    ]
    if primary:
        lines.extend(
            [
                f"- Policy: `{primary.get('policy_label')}` shrinkage `{primary.get('shrinkage')}`",
                f"- Correct: `{primary.get('correct')}/{primary.get('total')}`",
                f"- Loop 1 correct: `{primary.get('loop1_correct')}/{primary.get('total')}`",
                f"- Delta vs loop 1: `{primary.get('delta_vs_loop1')}`",
                f"- Oracle gap capture: `{primary.get('oracle_gap_capture')}`",
                f"- Routed deep: `{primary.get('routed_deep')}`",
                f"- W/L vs loop 1: `{primary.get('wins_vs_loop1')}/{primary.get('losses_vs_loop1')}`",
                f"- Rescue/harm: `{primary.get('rescue_captured')}/{primary.get('harm_triggered')}`",
                "",
            ]
        )
    lines.extend(["## Aggregate Policy Results", ""])
    for row in payload["aggregate_policy_results"][:16]:
        lines.append(
            f"- `{row['policy_label']}` shrinkage `{row['shrinkage']}`: "
            f"correct `{row['correct']}/{row['total']}`, delta `{row['delta_vs_loop1']}`, "
            f"gap `{row['oracle_gap_capture']}`, routed `{row['routed_deep']}`, "
            f"W/L `{row['wins_vs_loop1']}/{row['losses_vs_loop1']}`, "
            f"rescue/harm `{row['rescue_captured']}/{row['harm_triggered']}`"
        )
    lines.extend(["", "## Per Benchmark", ""])
    for benchmark, row in payload["per_benchmark"].items():
        lines.append(f"### {benchmark}")
        lines.append(f"- Total: `{row['total']}`")
        lines.append(f"- Categories: `{row['category_counts']}`")
        aucs = row.get("rescue_discrimination", [])[:4]
        if aucs:
            lines.append(
                "- Top rescue AUCs: "
                + ", ".join(f"`{auc['feature']}` {auc.get('oriented_auc')}" for auc in aucs)
            )
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sweep_summary", required=True)
    parser.add_argument("--benchmarks", default="arc_easy,arc_challenge,open_hard_arc_challenge")
    parser.add_argument("--score_target", default="cyclic_label_aggregated")
    parser.add_argument("--aggregate", default="permutation_mean")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--shrinkages", default=",".join(str(value) for value in DEFAULT_PROBE_SHRINKAGES))
    parser.add_argument("--primary_shrinkage", type=float, default=None)
    parser.add_argument("--run_id", default="")
    parser.add_argument("--output_dir", default="")
    args = parser.parse_args()

    payload = analyze_kfold(
        sweep_summary=resolve_path(args.sweep_summary),
        benchmarks=parse_csv(args.benchmarks),
        score_target=args.score_target,
        aggregate=args.aggregate,
        folds=args.folds,
        seed=args.seed,
        shrinkages=parse_floats(args.shrinkages),
        primary_shrinkage=args.primary_shrinkage,
        run_id=args.run_id or None,
    )
    output_dir = resolve_path(args.output_dir) if args.output_dir else ROOT / "outputs" / "stage5" / payload["run_id"]
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "summary.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    write_markdown(payload, output_dir / "summary.md")
    print((output_dir / "summary.md").read_text(encoding="utf-8"), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
