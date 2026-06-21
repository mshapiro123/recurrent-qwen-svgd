"""Compare two ARC-AGI eval summaries with paired uncertainty estimates."""

from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path
from typing import Any

from eval.eval_arc_agi import task_family


Metric = str
ExampleKey = tuple[str, int]
METRICS: tuple[Metric, ...] = ("selected_exact", "best_of_k_exact", "first_exact")


def read_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def example_key(example: dict[str, Any]) -> ExampleKey:
    return str(example["task_id"]), int(example.get("test_index", 0))


def examples_by_key(payload: dict[str, Any]) -> dict[ExampleKey, dict[str, Any]]:
    rows: dict[ExampleKey, dict[str, Any]] = {}
    for example in payload.get("examples", []):
        if example.get("has_target"):
            rows[example_key(example)] = example
    return rows


def bool_score(example: dict[str, Any], metric: Metric) -> int | None:
    value = example.get(metric)
    if value is None:
        return None
    return int(bool(value))


def example_difficulty_bucket(*examples: dict[str, Any]) -> str:
    for example in examples:
        value = example.get("difficulty_bucket") or example.get("difficulty")
        if value:
            return str(value)
    return "unknown"


def paired_rows(
    reference: dict[str, Any],
    candidate: dict[str, Any],
    metric: Metric,
) -> list[dict[str, Any]]:
    ref_examples = examples_by_key(reference)
    cand_examples = examples_by_key(candidate)
    rows: list[dict[str, Any]] = []
    for key in sorted(set(ref_examples) & set(cand_examples)):
        ref_score = bool_score(ref_examples[key], metric)
        cand_score = bool_score(cand_examples[key], metric)
        if ref_score is None or cand_score is None:
            continue
        rows.append(
            {
                "task_id": key[0],
                "test_index": key[1],
                "family": task_family(key[0]),
                "difficulty_bucket": example_difficulty_bucket(cand_examples[key], ref_examples[key]),
                "reference": ref_score,
                "candidate": cand_score,
                "delta": cand_score - ref_score,
            }
        )
    return rows


def percentile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * q
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    frac = position - lower
    return ordered[lower] * (1.0 - frac) + ordered[upper] * frac


def bootstrap_delta_ci(
    rows: list[dict[str, Any]],
    *,
    samples: int,
    seed: int,
) -> dict[str, float | None]:
    if not rows or samples <= 0:
        return {"low": None, "high": None}
    rng = random.Random(seed)
    n = len(rows)
    deltas: list[float] = []
    for _ in range(samples):
        total = sum(rows[rng.randrange(n)]["delta"] for _idx in range(n))
        deltas.append(total / n)
    return {
        "low": percentile(deltas, 0.025),
        "high": percentile(deltas, 0.975),
    }


def two_sided_sign_p_value(wins: int, losses: int) -> float | None:
    trials = wins + losses
    if trials == 0:
        return None
    observed = min(wins, losses)
    probability = sum(math.comb(trials, k) for k in range(observed + 1)) / (2**trials)
    return min(1.0, 2.0 * probability)


def metric_summary(
    rows: list[dict[str, Any]],
    *,
    bootstrap_samples: int,
    seed: int,
) -> dict[str, Any]:
    n = len(rows)
    ref = sum(row["reference"] for row in rows)
    cand = sum(row["candidate"] for row in rows)
    wins = sum(1 for row in rows if row["delta"] > 0)
    losses = sum(1 for row in rows if row["delta"] < 0)
    ties = n - wins - losses
    delta = cand - ref
    return {
        "paired_examples": n,
        "reference_exact": ref,
        "candidate_exact": cand,
        "delta_exact": delta,
        "reference_accuracy": ref / max(n, 1),
        "candidate_accuracy": cand / max(n, 1),
        "delta_accuracy": delta / max(n, 1),
        "wins": wins,
        "losses": losses,
        "ties": ties,
        "sign_test_p_value": two_sided_sign_p_value(wins, losses),
        "bootstrap_delta_accuracy_ci95": bootstrap_delta_ci(rows, samples=bootstrap_samples, seed=seed),
    }


def family_summaries(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return stratified_summaries(rows, "family")


def difficulty_summaries(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return stratified_summaries(rows, "difficulty_bucket")


def stratified_summaries(rows: list[dict[str, Any]], field: str) -> dict[str, Any]:
    by_group: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_group.setdefault(str(row.get(field) or "unknown"), []).append(row)
    return {
        group: metric_summary(items, bootstrap_samples=0, seed=0)
        for group, items in sorted(by_group.items())
    }


def compare_payloads(
    reference: dict[str, Any],
    candidate: dict[str, Any],
    *,
    reference_label: str,
    candidate_label: str,
    bootstrap_samples: int = 1000,
    seed: int = 0,
) -> dict[str, Any]:
    metrics: dict[str, Any] = {}
    families: dict[str, Any] = {}
    difficulties: dict[str, Any] = {}
    for metric in METRICS:
        rows = paired_rows(reference, candidate, metric)
        metrics[metric] = metric_summary(rows, bootstrap_samples=bootstrap_samples, seed=seed)
        families[metric] = family_summaries(rows)
        difficulties[metric] = difficulty_summaries(rows)
    return {
        "reference_label": reference_label,
        "candidate_label": candidate_label,
        "reference_summary": reference.get("summary", {}),
        "candidate_summary": candidate.get("summary", {}),
        "common_examples": metrics["selected_exact"]["paired_examples"],
        "metrics": metrics,
        "task_family_metrics": families,
        "difficulty_metrics": difficulties,
    }


def write_json(path: str | Path | None, payload: dict[str, Any]) -> None:
    if not path:
        return
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def write_markdown(path: str | Path | None, payload: dict[str, Any]) -> None:
    if not path:
        return
    lines = [
        f"# ARC-AGI Paired Comparison: {payload['candidate_label']} vs {payload['reference_label']}",
        "",
        f"- Common examples: `{payload['common_examples']}`",
        "",
        "## Metrics",
        "",
        "| Metric | Reference | Candidate | Delta | Win/Loss/Tie | Sign p | CI95 delta acc |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for metric, stats in payload["metrics"].items():
        ci = stats["bootstrap_delta_accuracy_ci95"]
        lines.append(
            f"| `{metric}` | {stats['reference_exact']}/{stats['paired_examples']} | "
            f"{stats['candidate_exact']}/{stats['paired_examples']} | "
            f"{stats['delta_exact']} ({stats['delta_accuracy']:.4f}) | "
            f"{stats['wins']}/{stats['losses']}/{stats['ties']} | "
            f"{stats['sign_test_p_value']} | "
            f"[{ci['low']}, {ci['high']}] |"
        )
    lines += ["", "## Task Family Deltas", ""]
    for metric, family_rows in payload["task_family_metrics"].items():
        lines.append(f"### {metric}")
        for family, stats in family_rows.items():
            lines.append(
                f"- `{family}`: candidate `{stats['candidate_exact']}` / `{stats['paired_examples']}`, "
                f"reference `{stats['reference_exact']}` / `{stats['paired_examples']}`, "
                f"delta `{stats['delta_exact']}`"
            )
        lines.append("")
    lines += ["", "## Difficulty Bucket Deltas", ""]
    for metric, difficulty_rows in payload.get("difficulty_metrics", {}).items():
        lines.append(f"### {metric}")
        for bucket, stats in difficulty_rows.items():
            lines.append(
                f"- `{bucket}`: candidate `{stats['candidate_exact']}` / `{stats['paired_examples']}`, "
                f"reference `{stats['reference_exact']}` / `{stats['paired_examples']}`, "
                f"delta `{stats['delta_exact']}`"
            )
        lines.append("")
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference_summary_json", required=True)
    parser.add_argument("--candidate_summary_json", required=True)
    parser.add_argument("--reference_label", default="reference")
    parser.add_argument("--candidate_label", default="candidate")
    parser.add_argument("--output_json")
    parser.add_argument("--output_md")
    parser.add_argument("--bootstrap_samples", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    payload = compare_payloads(
        read_json(args.reference_summary_json),
        read_json(args.candidate_summary_json),
        reference_label=args.reference_label,
        candidate_label=args.candidate_label,
        bootstrap_samples=args.bootstrap_samples,
        seed=args.seed,
    )
    write_json(args.output_json, payload)
    write_markdown(args.output_md, payload)
    print(json.dumps(payload["metrics"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
