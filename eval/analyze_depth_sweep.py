"""Analyze loop-depth sweeps for recurrent MCQ evaluation outputs.

The benchmark suite compares base Qwen against one recurrent depth at a time.
This script joins the repeated loop runs into one per-example table and asks
the routing questions that matter after the direct-preservation result:

* Does any deeper loop solve examples that loop 1 misses?
* How much damage does each deeper loop do relative to loop 1?
* What is the oracle best-of-depth ceiling?
* Can simple confidence thresholds route from loop 1 to deeper loops safely?
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def path_for_cli(path: Path) -> str:
    try:
        return path.relative_to(ROOT).as_posix()
    except ValueError:
        return str(path)


def resolve_path(path: str | Path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else ROOT / p


def loop_from_run_id(run_id: str) -> int:
    match = re.search(r"_loop(\d+)$", run_id)
    if not match:
        raise ValueError(f"Could not parse loop depth from run_id={run_id!r}")
    return int(match.group(1))


def rows_by_id(path: Path) -> dict[str, dict[str, Any]]:
    return {str(row["id"]): row for row in read_jsonl(path)}


def score_dict(row: dict[str, Any]) -> dict[str, float]:
    return {str(label): float(score) for label, score in (row.get("scores") or {}).items()}


def predicted_margin(row: dict[str, Any]) -> float | None:
    scores = sorted(score_dict(row).values(), reverse=True)
    if len(scores) < 2:
        return None
    return scores[0] - scores[1]


def answer_margin(row: dict[str, Any]) -> float | None:
    scores = score_dict(row)
    answer = str(row.get("answer"))
    if answer not in scores:
        return None
    others = [score for label, score in scores.items() if label != answer]
    if not others:
        return None
    return scores[answer] - max(others)


def finite(value: Any) -> bool:
    return isinstance(value, (int, float)) and math.isfinite(float(value))


def mean(values: list[float | None]) -> float | None:
    kept = [float(v) for v in values if finite(v)]
    if not kept:
        return None
    return sum(kept) / len(kept)


def sign_test_p_value(wins: int, losses: int) -> float | None:
    total = wins + losses
    if total <= 0:
        return None
    smaller = min(wins, losses)
    # Two-sided exact binomial under p=0.5.
    cumulative = sum(math.comb(total, k) for k in range(smaller + 1)) / (2**total)
    return min(1.0, 2.0 * cumulative)


def load_loop_payloads(sweep_summary: Path) -> tuple[dict[str, Any], dict[int, dict[str, Any]]]:
    sweep = read_json(sweep_summary)
    loop_payloads: dict[int, dict[str, Any]] = {}
    for run_id in sweep.get("loop_run_ids", []):
        loop = loop_from_run_id(str(run_id))
        summary_path = sweep_summary.parent.parent / str(run_id) / "summary.json"
        if not summary_path.exists():
            summary_path = ROOT / "outputs" / "stage5" / str(run_id) / "summary.json"
        if not summary_path.exists():
            raise FileNotFoundError(summary_path)
        loop_payloads[loop] = read_json(summary_path)
    return sweep, dict(sorted(loop_payloads.items()))


def row_file_for(loop_payload: dict[str, Any], benchmark: str, arm: str, score_target: str) -> Path:
    results = loop_payload.get("results", [])
    if isinstance(results, list):
        for row in results:
            if row.get("benchmark") == benchmark and row.get("arm") == arm and row.get("score_target") == score_target:
                value = row.get("output_jsonl")
                if value:
                    return resolve_path(value)
    run_id = loop_payload.get("run_id")
    if run_id:
        inferred = ROOT / "outputs" / "stage5" / str(run_id) / f"{benchmark}_{arm}_{score_target}.jsonl"
        if inferred.exists():
            return inferred
    raise FileNotFoundError(f"No {score_target} output for benchmark={benchmark} arm={arm}")


def joined_examples(
    loop_payloads: dict[int, dict[str, Any]],
    benchmark: str,
    score_target: str,
    aggregate: str,
) -> list[dict[str, Any]]:
    loops = sorted(loop_payloads)
    base_rows = rows_by_id_for_aggregate(row_file_for(loop_payloads[loops[0]], benchmark, "base", score_target), aggregate)
    recurrent_by_loop = {
        loop: rows_by_id_for_aggregate(row_file_for(payload, benchmark, "recurrent", score_target), aggregate)
        for loop, payload in loop_payloads.items()
    }
    ids = sorted(set(base_rows).intersection(*(set(rows) for rows in recurrent_by_loop.values())))
    examples: list[dict[str, Any]] = []
    for row_id in ids:
        base = base_rows[row_id]
        rec = {loop: recurrent_by_loop[loop][row_id] for loop in loops}
        hits = {loop: bool(row.get("hit")) for loop, row in rec.items()}
        examples.append(
            {
                "id": row_id,
                "answer": base.get("answer"),
                "base_hit": bool(base.get("hit")),
                "base_prediction": base.get("prediction"),
                "base_predicted_margin": predicted_margin(base),
                "base_answer_margin": answer_margin(base),
                "loop_hits": hits,
                "loop_predictions": {loop: row.get("prediction") for loop, row in rec.items()},
                "loop_scores": {loop: score_dict(row) for loop, row in rec.items()},
                "loop_predicted_margins": {loop: predicted_margin(row) for loop, row in rec.items()},
                "loop_answer_margins": {loop: answer_margin(row) for loop, row in rec.items()},
            }
        )
    return examples


def rows_by_id_for_aggregate(path: Path, aggregate: str) -> dict[str, dict[str, Any]]:
    rows = read_jsonl(path)
    selected = [row for row in rows if str(row.get("aggregate") or "mean") == aggregate]
    return {str(row["id"]): row for row in selected}


def bucket_by_margin(margin: float | None) -> str:
    if margin is None:
        return "unknown"
    if margin >= 2.0:
        return "very_confident"
    if margin >= 1.0:
        return "confident"
    if margin >= 0.5:
        return "thin"
    return "low"


def loop_summary(examples: list[dict[str, Any]], loop: int) -> dict[str, Any]:
    base_correct = sum(1 for ex in examples if ex["base_hit"])
    loop_correct = sum(1 for ex in examples if ex["loop_hits"][loop])
    wins = sum(1 for ex in examples if ex["loop_hits"][loop] and not ex["base_hit"])
    losses = sum(1 for ex in examples if ex["base_hit"] and not ex["loop_hits"][loop])
    ties = len(examples) - wins - losses
    return {
        "loop": loop,
        "total": len(examples),
        "base_correct": base_correct,
        "loop_correct": loop_correct,
        "delta_vs_base": loop_correct - base_correct,
        "wins_vs_base": wins,
        "losses_vs_base": losses,
        "ties_vs_base": ties,
        "sign_test_p": sign_test_p_value(wins, losses),
        "mean_predicted_margin": mean([ex["loop_predicted_margins"][loop] for ex in examples]),
        "mean_answer_margin": mean([ex["loop_answer_margins"][loop] for ex in examples]),
    }


def depth_interaction_summary(examples: list[dict[str, Any]], loops: list[int]) -> dict[str, Any]:
    loop1 = min(loops)
    base_correct = sum(1 for ex in examples if ex["base_hit"])
    loop1_correct = sum(1 for ex in examples if ex["loop_hits"][loop1])
    any_recurrent_correct = sum(1 for ex in examples if any(ex["loop_hits"].values()))
    any_base_or_recurrent_correct = sum(1 for ex in examples if ex["base_hit"] or any(ex["loop_hits"].values()))
    deeper = [loop for loop in loops if loop != loop1]
    deeper_unique_over_loop1 = sum(
        1 for ex in examples if not ex["loop_hits"][loop1] and any(ex["loop_hits"][loop] for loop in deeper)
    )
    deeper_unique_over_base_and_loop1 = sum(
        1
        for ex in examples
        if not ex["base_hit"] and not ex["loop_hits"][loop1] and any(ex["loop_hits"][loop] for loop in deeper)
    )
    loop1_harmed_by_any_deeper = sum(
        1 for ex in examples if ex["loop_hits"][loop1] and any(not ex["loop_hits"][loop] for loop in deeper)
    )
    all_depths_correct = sum(1 for ex in examples if all(ex["loop_hits"][loop] for loop in loops))
    all_depths_wrong = sum(1 for ex in examples if not ex["base_hit"] and not any(ex["loop_hits"].values()))
    pattern_counts = Counter(
        "".join("1" if ex["loop_hits"][loop] else "0" for loop in loops)
        for ex in examples
    )
    return {
        "total": len(examples),
        "base_correct": base_correct,
        "loop1_correct": loop1_correct,
        "any_recurrent_correct": any_recurrent_correct,
        "any_base_or_recurrent_correct": any_base_or_recurrent_correct,
        "oracle_recurrent_gain_vs_loop1": any_recurrent_correct - loop1_correct,
        "oracle_base_or_recurrent_gain_vs_base": any_base_or_recurrent_correct - base_correct,
        "deeper_unique_over_loop1": deeper_unique_over_loop1,
        "deeper_unique_over_base_and_loop1": deeper_unique_over_base_and_loop1,
        "loop1_harmed_by_any_deeper": loop1_harmed_by_any_deeper,
        "all_depths_correct": all_depths_correct,
        "all_depths_wrong": all_depths_wrong,
        "depth_hit_patterns": dict(pattern_counts),
    }


def evaluate_threshold_router(
    examples: list[dict[str, Any]],
    *,
    source: str,
    threshold: float,
    fallback_loop: int,
    default_loop: int = 1,
) -> dict[str, Any]:
    correct = 0
    routed_deep = 0
    wins_vs_loop1 = 0
    losses_vs_loop1 = 0
    for ex in examples:
        if source == "base":
            margin = ex["base_predicted_margin"]
        elif source == "loop1":
            margin = ex["loop_predicted_margins"][default_loop]
        else:
            raise ValueError(source)
        use_deep = finite(margin) and float(margin) < threshold
        chosen = fallback_loop if use_deep else default_loop
        routed_deep += int(use_deep)
        hit = bool(ex["loop_hits"][chosen])
        correct += int(hit)
        loop1_hit = bool(ex["loop_hits"][default_loop])
        if hit and not loop1_hit:
            wins_vs_loop1 += 1
        elif loop1_hit and not hit:
            losses_vs_loop1 += 1
    loop1_correct = sum(1 for ex in examples if ex["loop_hits"][default_loop])
    return {
        "source": source,
        "threshold": threshold,
        "fallback_loop": fallback_loop,
        "total": len(examples),
        "correct": correct,
        "delta_vs_loop1": correct - loop1_correct,
        "routed_deep": routed_deep,
        "wins_vs_loop1": wins_vs_loop1,
        "losses_vs_loop1": losses_vs_loop1,
        "sign_test_p": sign_test_p_value(wins_vs_loop1, losses_vs_loop1),
    }


def threshold_router_sweep(examples: list[dict[str, Any]], loops: list[int]) -> list[dict[str, Any]]:
    thresholds = [0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0]
    rows = []
    for source in ["base", "loop1"]:
        for fallback_loop in [loop for loop in loops if loop != 1]:
            for threshold in thresholds:
                rows.append(
                    evaluate_threshold_router(
                        examples,
                        source=source,
                        threshold=threshold,
                        fallback_loop=fallback_loop,
                    )
                )
    return sorted(rows, key=lambda row: (row["delta_vs_loop1"], row["correct"]), reverse=True)


def aggregate_scores(ex: dict[str, Any], subset: tuple[int, ...], method: str, weight: float | None = None) -> dict[str, float]:
    labels = sorted({label for loop in subset for label in ex["loop_scores"][loop]})
    if method == "mean":
        return {
            label: mean([ex["loop_scores"][loop].get(label) for loop in subset]) or float("-inf")
            for label in labels
        }
    if method == "max":
        return {
            label: max(
                [ex["loop_scores"][loop].get(label, float("-inf")) for loop in subset]
            )
            for label in labels
        }
    if method == "loop1_plus_weighted_deeper":
        assert weight is not None
        deeper = [loop for loop in subset if loop != 1]
        return {
            label: ex["loop_scores"][1].get(label, float("-inf"))
            + weight
            * (
                sum(ex["loop_scores"][loop].get(label, 0.0) for loop in deeper) / max(1, len(deeper))
            )
            for label in labels
        }
    raise ValueError(method)


def evaluate_score_selector(
    examples: list[dict[str, Any]],
    *,
    subset: tuple[int, ...],
    method: str,
    weight: float | None = None,
) -> dict[str, Any]:
    correct = 0
    wins_vs_loop1 = 0
    losses_vs_loop1 = 0
    prediction_counts: Counter[str] = Counter()
    for ex in examples:
        scores = aggregate_scores(ex, subset, method, weight)
        prediction = max(scores, key=scores.get)
        hit = prediction == str(ex["answer"])
        prediction_counts[prediction] += 1
        correct += int(hit)
        loop1_hit = bool(ex["loop_hits"][1])
        if hit and not loop1_hit:
            wins_vs_loop1 += 1
        elif loop1_hit and not hit:
            losses_vs_loop1 += 1
    loop1_correct = sum(1 for ex in examples if ex["loop_hits"][1])
    return {
        "subset": list(subset),
        "method": method if weight is None else f"{method}:{weight}",
        "total": len(examples),
        "correct": correct,
        "delta_vs_loop1": correct - loop1_correct,
        "wins_vs_loop1": wins_vs_loop1,
        "losses_vs_loop1": losses_vs_loop1,
        "sign_test_p": sign_test_p_value(wins_vs_loop1, losses_vs_loop1),
        "prediction_counts": dict(prediction_counts),
    }


def score_selector_sweep(examples: list[dict[str, Any]], loops: list[int]) -> list[dict[str, Any]]:
    rows = []
    for end_idx in range(1, len(loops) + 1):
        subset = tuple(loops[:end_idx])
        rows.append(evaluate_score_selector(examples, subset=subset, method="mean"))
        rows.append(evaluate_score_selector(examples, subset=subset, method="max"))
        if len(subset) > 1 and 1 in subset:
            for weight in [0.1, 0.25, 0.5, 0.75, 1.0]:
                rows.append(
                    evaluate_score_selector(
                        examples,
                        subset=subset,
                        method="loop1_plus_weighted_deeper",
                        weight=weight,
                    )
                )
    return sorted(rows, key=lambda row: (row["delta_vs_loop1"], row["correct"]), reverse=True)


def margin_bucket_summary(examples: list[dict[str, Any]], loops: list[int]) -> dict[str, Any]:
    buckets: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for ex in examples:
        buckets[bucket_by_margin(ex["base_predicted_margin"])].append(ex)
    out = {}
    for bucket, rows in sorted(buckets.items()):
        out[bucket] = {
            "n": len(rows),
            "base_correct": sum(1 for ex in rows if ex["base_hit"]),
            "by_loop": {loop: sum(1 for ex in rows if ex["loop_hits"][loop]) for loop in loops},
            "any_recurrent_correct": sum(1 for ex in rows if any(ex["loop_hits"].values())),
        }
    return out


def analyze(sweep_summary: Path, *, score_target: str = "label", aggregate: str = "mean") -> dict[str, Any]:
    sweep, loop_payloads = load_loop_payloads(sweep_summary)
    loops = sorted(loop_payloads)
    benchmarks = list(loop_payloads[loops[0]].get("benchmarks", []))
    payload: dict[str, Any] = {
        "kind": "stage5_depth_sweep_analysis",
        "source_sweep_summary": path_for_cli(sweep_summary),
        "source_sweep_run_id": sweep.get("run_id"),
        "score_target": score_target,
        "aggregate": aggregate,
        "loops": loops,
        "benchmarks": {},
    }
    for benchmark in benchmarks:
        examples = joined_examples(loop_payloads, benchmark, score_target, aggregate)
        payload["benchmarks"][benchmark] = {
            "loop_summaries": [loop_summary(examples, loop) for loop in loops],
            "depth_interactions": depth_interaction_summary(examples, loops),
            "margin_buckets": margin_bucket_summary(examples, loops),
            "threshold_router_top10": threshold_router_sweep(examples, loops)[:10],
            "score_selector_top10": score_selector_sweep(examples, loops)[:10],
        }
    return payload


def write_markdown(payload: dict[str, Any], path: Path) -> None:
    lines = [
        f"# Depth Sweep Analysis - {payload['source_sweep_run_id']}",
        "",
        f"- Source: `{payload['source_sweep_summary']}`",
        f"- Score target: `{payload.get('score_target')}`",
        f"- Aggregate: `{payload.get('aggregate')}`",
        f"- Loops: `{payload['loops']}`",
        "",
    ]
    for benchmark, result in payload["benchmarks"].items():
        lines.extend([f"## {benchmark}", "", "### Loop Summaries", ""])
        for row in result["loop_summaries"]:
            lines.append(
                f"- loop `{row['loop']}`: recurrent `{row['loop_correct']}/{row['total']}`, "
                f"base `{row['base_correct']}/{row['total']}`, delta `{row['delta_vs_base']}`, "
                f"W/L/T `{row['wins_vs_base']}/{row['losses_vs_base']}/{row['ties_vs_base']}`, "
                f"p `{row['sign_test_p']}`"
            )
        d = result["depth_interactions"]
        lines.extend(
            [
                "",
                "### Depth Interaction",
                "",
                f"- loop1 correct: `{d['loop1_correct']}/{d['total']}`",
                f"- any recurrent depth correct: `{d['any_recurrent_correct']}/{d['total']}` "
                f"(oracle gain vs loop1 `{d['oracle_recurrent_gain_vs_loop1']}`)",
                f"- base or any recurrent correct: `{d['any_base_or_recurrent_correct']}/{d['total']}` "
                f"(oracle gain vs base `{d['oracle_base_or_recurrent_gain_vs_base']}`)",
                f"- deeper unique over loop1: `{d['deeper_unique_over_loop1']}`",
                f"- deeper unique over base+loop1: `{d['deeper_unique_over_base_and_loop1']}`",
                f"- loop1 harmed by at least one deeper loop: `{d['loop1_harmed_by_any_deeper']}`",
                f"- depth hit patterns: `{d['depth_hit_patterns']}`",
                "",
                "### Best Simple Threshold Routers",
                "",
            ]
        )
        for row in result["threshold_router_top10"]:
            lines.append(
                f"- `{row['source']}` margin < `{row['threshold']}` -> loop `{row['fallback_loop']}`: "
                f"correct `{row['correct']}/{row['total']}`, delta vs loop1 `{row['delta_vs_loop1']}`, "
                f"routed deep `{row['routed_deep']}`, W/L `{row['wins_vs_loop1']}/{row['losses_vs_loop1']}`"
            )
        lines.extend(["", "### Best Score Selectors", ""])
        for row in result["score_selector_top10"]:
            lines.append(
                f"- subset `{row['subset']}` `{row['method']}`: correct `{row['correct']}/{row['total']}`, "
                f"delta vs loop1 `{row['delta_vs_loop1']}`, W/L "
                f"`{row['wins_vs_loop1']}/{row['losses_vs_loop1']}`, p `{row['sign_test_p']}`"
            )
        lines.extend(["", "### Base Confidence Buckets", ""])
        for bucket, row in result["margin_buckets"].items():
            lines.append(
                f"- `{bucket}` n `{row['n']}`: base `{row['base_correct']}`, "
                f"by_loop `{row['by_loop']}`, any recurrent `{row['any_recurrent_correct']}`"
            )
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sweep_summary", required=True)
    parser.add_argument("--score_target", default="label")
    parser.add_argument("--aggregate", default="mean")
    parser.add_argument("--output_dir", default="")
    args = parser.parse_args()

    sweep_summary = resolve_path(args.sweep_summary)
    payload = analyze(sweep_summary, score_target=args.score_target, aggregate=args.aggregate)
    output_dir = resolve_path(args.output_dir) if args.output_dir else sweep_summary.parent / "depth_analysis"
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "summary.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    write_markdown(payload, output_dir / "summary.md")
    print((output_dir / "summary.md").read_text(encoding="utf-8"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
