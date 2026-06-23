"""Assess same-recipe dense-vs-recurrent MCQ evidence.

This is the architecture-control gate for the ARC MCQ trace-SFT work. It
compares a dense Qwen LoRA trained on the same traced curriculum against a
matched recurrent checkpoint on the same ARC-Easy/ARC-Challenge MCQ rows and
scoring surfaces.

The important read is not just aggregate accuracy. The current thesis-relevant
signal is hard-tail lift, especially ARC-Challenge, while ARC-Easy content loss
can be a surface/order-sensitivity repair problem. This assessor keeps those
facts separate so the next Colab run produces interpretable evidence.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
RUN_ID = os.environ.get("STAGE5_MCQ_RECIPE_CONTROL_RUN_ID") or time.strftime(
    "stage5_mcq_recipe_control_%Y%m%d_%H%M%S"
)

DEFAULT_PRIMARY_BENCHMARK = "arc_challenge"
DEFAULT_PRIMARY_SCORE_TARGET = "cyclic_label_aggregated"
DEFAULT_PRIMARY_AGGREGATE = "permutation_mean"


def resolve_path(value: str | Path) -> Path:
    path = Path(str(value).replace("\\", "/"))
    return path if path.is_absolute() else ROOT / path


def path_for_cli(path: Path) -> str:
    try:
        return path.relative_to(ROOT).as_posix()
    except ValueError:
        return str(path).replace("\\", "/")


def read_json(path: str | Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} is not a JSON object")
    return payload


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def two_sided_sign_p_value(wins: int, losses: int) -> float | None:
    trials = wins + losses
    if trials == 0:
        return None
    observed = min(wins, losses)
    return min(1.0, 2.0 * sum(math.comb(trials, k) for k in range(observed + 1)) / (2**trials))


def summary_is_benchmark_suite(payload: dict[str, Any]) -> bool:
    return payload.get("kind") == "stage5_benchmark_suite"


def resolve_recurrent_benchmark_summary(path: Path) -> tuple[Path, dict[str, Any]]:
    payload = read_json(path)
    if summary_is_benchmark_suite(payload):
        return path, payload

    for key in ("benchmark_summary", "recurrent_benchmark_summary", "source_summary"):
        value = payload.get(key)
        if not value:
            continue
        candidate = resolve_path(str(value))
        if candidate.exists():
            candidate_payload = read_json(candidate)
            if summary_is_benchmark_suite(candidate_payload):
                return candidate, candidate_payload

    raise ValueError(f"{path} is not a stage5_benchmark_suite summary and does not point to one")


def dense_artifact_paths(dense_summary_path: Path, dense_summary: dict[str, Any]) -> dict[tuple[str, str], Path]:
    artifacts = dense_summary.get("artifacts")
    if not isinstance(artifacts, dict):
        raise KeyError(f"{dense_summary_path} does not contain dense-control artifacts")

    result: dict[tuple[str, str], Path] = {}
    for benchmark, benchmark_artifacts in artifacts.items():
        if not isinstance(benchmark_artifacts, dict):
            continue
        for score_target, arm_artifacts in benchmark_artifacts.items():
            if score_target == "data_jsonl" or not isinstance(arm_artifacts, dict):
                continue
            dense_path = arm_artifacts.get("dense")
            if dense_path:
                result[(str(benchmark), str(score_target))] = resolve_path(str(dense_path))
    return result


def recurrent_artifact_path(
    recurrent_summary_path: Path,
    recurrent_summary: dict[str, Any],
    benchmark: str,
    score_target: str,
) -> Path:
    artifacts = recurrent_summary.get("artifacts")
    if isinstance(artifacts, dict):
        benchmark_artifacts = artifacts.get(benchmark)
        if isinstance(benchmark_artifacts, dict):
            score_artifacts = benchmark_artifacts.get(score_target)
            if isinstance(score_artifacts, dict) and score_artifacts.get("recurrent"):
                return resolve_path(str(score_artifacts["recurrent"]))

    candidate = recurrent_summary_path.parent / f"{benchmark}_recurrent_{score_target}.jsonl"
    if candidate.exists():
        return candidate
    raise FileNotFoundError(candidate)


def rows_by_aggregate_and_id(path: Path) -> dict[str, dict[str, dict[str, Any]]]:
    grouped: dict[str, dict[str, dict[str, Any]]] = {}
    for row in read_jsonl(path):
        row_id = row.get("id")
        if row_id is None:
            continue
        aggregate = str(row.get("aggregate") or "mean")
        grouped.setdefault(aggregate, {})[str(row_id)] = row
    return grouped


def compare_hit_rows(reference_path: Path, candidate_path: Path) -> dict[str, dict[str, Any]]:
    reference = rows_by_aggregate_and_id(reference_path)
    candidate = rows_by_aggregate_and_id(candidate_path)
    aggregates = sorted(set(reference) & set(candidate))
    result: dict[str, dict[str, Any]] = {}
    for aggregate in aggregates:
        reference_by_id = reference[aggregate]
        candidate_by_id = candidate[aggregate]
        common = sorted(set(reference_by_id) & set(candidate_by_id))
        wins: list[str] = []
        losses: list[str] = []
        ties = 0
        reference_correct = 0
        candidate_correct = 0
        for row_id in common:
            reference_hit = bool(reference_by_id[row_id].get("hit"))
            candidate_hit = bool(candidate_by_id[row_id].get("hit"))
            reference_correct += int(reference_hit)
            candidate_correct += int(candidate_hit)
            if candidate_hit and not reference_hit:
                wins.append(row_id)
            elif reference_hit and not candidate_hit:
                losses.append(row_id)
            else:
                ties += 1
        total = len(common)
        delta = candidate_correct - reference_correct
        result[aggregate] = {
            "paired_examples": total,
            "dense_correct": reference_correct,
            "recurrent_correct": candidate_correct,
            "correct_delta_recurrent_vs_dense": delta,
            "dense_accuracy": reference_correct / max(total, 1),
            "recurrent_accuracy": candidate_correct / max(total, 1),
            "accuracy_delta_recurrent_vs_dense": delta / max(total, 1),
            "wins": len(wins),
            "losses": len(losses),
            "ties": ties,
            "sign_test_p_value": two_sided_sign_p_value(len(wins), len(losses)),
            "win_ids": wins,
            "loss_ids": losses,
        }
    return result


def recurrent_vs_base_surface_notes(recurrent_summary: dict[str, Any]) -> dict[str, Any]:
    paired = recurrent_summary.get("paired_comparisons")
    if not isinstance(paired, dict):
        return {"available": False}

    easy = paired.get("arc_easy")
    challenge = paired.get("arc_challenge")
    notes: dict[str, Any] = {"available": True}
    for benchmark_name, benchmark_payload in (("arc_easy", easy), ("arc_challenge", challenge)):
        if not isinstance(benchmark_payload, dict):
            continue
        content = (
            benchmark_payload.get("content_question_only", {})
            if isinstance(benchmark_payload.get("content_question_only"), dict)
            else {}
        )
        cyclic = (
            benchmark_payload.get("cyclic_label_aggregated", {})
            if isinstance(benchmark_payload.get("cyclic_label_aggregated"), dict)
            else {}
        )
        content_stats = content.get("mean") if isinstance(content.get("mean"), dict) else {}
        cyclic_stats = cyclic.get("permutation_mean") if isinstance(cyclic.get("permutation_mean"), dict) else {}
        content_delta = int(content_stats.get("correct_delta_recurrent_vs_base", 0) or 0)
        cyclic_delta = int(cyclic_stats.get("correct_delta_recurrent_vs_base", 0) or 0)
        if content_stats or cyclic_stats:
            notes[benchmark_name] = {
                "content_delta_recurrent_vs_base": content_delta,
                "cyclic_delta_recurrent_vs_base": cyclic_delta,
                "pattern": (
                    "content_down_cyclic_up"
                    if content_delta < 0 < cyclic_delta
                    else "both_up"
                    if content_delta > 0 and cyclic_delta > 0
                    else "both_down"
                    if content_delta < 0 and cyclic_delta < 0
                    else "mixed_or_flat"
                ),
            }
    return notes


def metadata_differences(dense_summary: dict[str, Any], recurrent_summary: dict[str, Any]) -> dict[str, Any]:
    dense_config = dense_summary.get("config") if isinstance(dense_summary.get("config"), dict) else {}
    differences: dict[str, Any] = {}
    comparisons = {
        "source_summary": (dense_summary.get("source_summary"), recurrent_summary.get("source_summary")),
        "model_name": (dense_config.get("model_name"), recurrent_summary.get("model_name")),
        "benchmarks": (dense_config.get("benchmarks"), recurrent_summary.get("benchmarks")),
        "score_targets": (dense_config.get("score_targets"), recurrent_summary.get("score_targets")),
        "aggregates": (dense_config.get("aggregates"), recurrent_summary.get("aggregates")),
    }
    for key, (dense_value, recurrent_value) in comparisons.items():
        if dense_value is not None and recurrent_value is not None and dense_value != recurrent_value:
            differences[key] = {"dense": dense_value, "recurrent": recurrent_value}
    return differences


def lookup(
    comparisons: dict[str, dict[str, dict[str, dict[str, Any]]]],
    benchmark: str,
    score_target: str,
    aggregate: str,
) -> dict[str, Any]:
    return comparisons.get(benchmark, {}).get(score_target, {}).get(aggregate, {})


def delta(stats: dict[str, Any]) -> int:
    return int(stats.get("correct_delta_recurrent_vs_dense", 0) or 0)


def evidence_fragment(stats: dict[str, Any]) -> str:
    if not stats:
        return "missing"
    return (
        f"delta {stats.get('correct_delta_recurrent_vs_dense', 0)} "
        f"({stats.get('wins', 0)}/{stats.get('losses', 0)}/{stats.get('ties', 0)} W/L/T, "
        f"n={stats.get('paired_examples', 0)}, p={stats.get('sign_test_p_value')})"
    )


def decide(
    comparisons: dict[str, dict[str, dict[str, dict[str, Any]]]],
    *,
    primary_benchmark: str,
    primary_score_target: str,
    primary_aggregate: str,
    min_primary_examples: int,
) -> tuple[str, bool, str, str, dict[str, Any]]:
    primary = lookup(comparisons, primary_benchmark, primary_score_target, primary_aggregate)
    challenge_content = lookup(comparisons, "arc_challenge", "content_question_only", "mean")
    challenge_cyclic = lookup(comparisons, "arc_challenge", "cyclic_label_aggregated", "permutation_mean")
    easy_content = lookup(comparisons, "arc_easy", "content_question_only", "mean")
    easy_cyclic = lookup(comparisons, "arc_easy", "cyclic_label_aggregated", "permutation_mean")
    decision_evidence = {
        "primary": primary,
        "arc_challenge_content": challenge_content,
        "arc_challenge_cyclic": challenge_cyclic,
        "arc_easy_content": easy_content,
        "arc_easy_cyclic": easy_cyclic,
        "primary_evidence": evidence_fragment(primary),
        "arc_challenge_content_evidence": evidence_fragment(challenge_content),
        "arc_challenge_cyclic_evidence": evidence_fragment(challenge_cyclic),
        "arc_easy_content_evidence": evidence_fragment(easy_content),
        "arc_easy_cyclic_evidence": evidence_fragment(easy_cyclic),
    }

    if not primary or int(primary.get("paired_examples", 0) or 0) < min_primary_examples:
        return (
            "needs_more_evidence",
            False,
            "Primary hard-tail surface has too few paired examples.",
            "Run the same dense and recurrent arms on the intended ARC slice before interpreting architecture lift.",
            decision_evidence,
        )

    challenge_positive = delta(challenge_content) > 0 or delta(challenge_cyclic) > 0
    challenge_both_nonnegative = bool(challenge_content) and bool(challenge_cyclic) and delta(challenge_content) >= 0 and delta(challenge_cyclic) >= 0
    primary_positive = delta(primary) > 0 and int(primary.get("wins", 0) or 0) > int(primary.get("losses", 0) or 0)
    easy_mixed_surface = bool(easy_content) and bool(easy_cyclic) and delta(easy_content) < 0 < delta(easy_cyclic)

    if primary_positive and challenge_both_nonnegative:
        return (
            "hard_tail_lift_vs_dense",
            True,
            "Recurrent same-recipe arm beats dense control on the primary hard-tail surface and is nonnegative on both ARC-Challenge surfaces.",
            "Replicate on a larger held-out slice; if it survives, treat recurrence as contributing beyond trace data.",
            decision_evidence,
        )
    if challenge_positive:
        return (
            "mixed_hard_tail_signal_vs_dense",
            False,
            "At least one ARC-Challenge surface favors recurrent over dense, but the hard-tail evidence is not clean across surfaces.",
            "Inspect the disagreeing Challenge surface and rerun after the surface-alignment repair.",
            decision_evidence,
        )
    if easy_mixed_surface:
        return (
            "easy_surface_invariance_issue",
            False,
            "Dense control beats recurrent on the hard-tail surface, while Easy content/cyclic movement shows a surface-invariance pattern.",
            "Prioritize conditional-invariance repair before deciding the architecture failed.",
            decision_evidence,
        )
    return (
        "no_architecture_lift_vs_dense",
        False,
        "Recurrent same-recipe arm does not beat the dense same-recipe control on the hard-tail surfaces.",
        "Improve recurrent recovery or depth-routing targets before scaling architecture-specific work.",
        decision_evidence,
    )


def assess_mcq_recipe_control(
    *,
    dense_summary_path: Path,
    recurrent_summary_path: Path,
    primary_benchmark: str = DEFAULT_PRIMARY_BENCHMARK,
    primary_score_target: str = DEFAULT_PRIMARY_SCORE_TARGET,
    primary_aggregate: str = DEFAULT_PRIMARY_AGGREGATE,
    min_primary_examples: int = 20,
) -> dict[str, Any]:
    dense_summary = read_json(dense_summary_path)
    recurrent_summary_path, recurrent_summary = resolve_recurrent_benchmark_summary(recurrent_summary_path)
    dense_paths = dense_artifact_paths(dense_summary_path, dense_summary)
    comparisons: dict[str, dict[str, dict[str, dict[str, Any]]]] = {}
    artifacts: dict[str, Any] = {}

    for (benchmark, score_target), dense_path in sorted(dense_paths.items()):
        recurrent_path = recurrent_artifact_path(recurrent_summary_path, recurrent_summary, benchmark, score_target)
        if not dense_path.exists():
            raise FileNotFoundError(dense_path)
        if not recurrent_path.exists():
            raise FileNotFoundError(recurrent_path)
        comparisons.setdefault(benchmark, {})[score_target] = compare_hit_rows(dense_path, recurrent_path)
        artifacts.setdefault(benchmark, {})[score_target] = {
            "dense": path_for_cli(dense_path),
            "recurrent": path_for_cli(recurrent_path),
        }

    status, passed, reason, next_step, decision_evidence = decide(
        comparisons,
        primary_benchmark=primary_benchmark,
        primary_score_target=primary_score_target,
        primary_aggregate=primary_aggregate,
        min_primary_examples=min_primary_examples,
    )
    metadata_diff = metadata_differences(dense_summary, recurrent_summary)
    if metadata_diff:
        status = "needs_review"
        passed = False
        reason = "Dense and recurrent MCQ summaries differ on matching metadata."
        next_step = "Rerun the mismatched arm with matched source, benchmarks, score targets, and aggregates before judging architecture lift."

    return {
        "run_id": RUN_ID,
        "kind": "stage5_mcq_recipe_control_assessment",
        "gate": "stage5_same_recipe_mcq_architecture",
        "status": status,
        "passed": passed,
        "reason": reason,
        "next_step": next_step,
        "dense_summary": path_for_cli(dense_summary_path),
        "recurrent_summary": path_for_cli(recurrent_summary_path),
        "primary": {
            "benchmark": primary_benchmark,
            "score_target": primary_score_target,
            "aggregate": primary_aggregate,
            "min_examples": min_primary_examples,
        },
        "metadata_differences": metadata_diff,
        "surface_notes_recurrent_vs_base": recurrent_vs_base_surface_notes(recurrent_summary),
        "decision_evidence": decision_evidence,
        "paired_comparisons": comparisons,
        "artifacts": artifacts,
    }


def write_report(payload: dict[str, Any], *, output_json: Path, output_md: Path) -> None:
    write_json(output_json, payload)
    evidence = payload["decision_evidence"]
    surface = payload.get("surface_notes_recurrent_vs_base") or {}
    lines = [
        f"# Stage 5 MCQ Same-Recipe Architecture Assessment - {payload['run_id']}",
        "",
        f"- Status: `{payload['status']}`",
        f"- Passed: `{payload['passed']}`",
        f"- Dense summary: `{payload['dense_summary']}`",
        f"- Recurrent summary: `{payload['recurrent_summary']}`",
        f"- Reason: {payload['reason']}",
        f"- Next step: {payload['next_step']}",
        "",
        "## Decision Evidence",
        "",
        f"- Primary recurrent-vs-dense: {evidence['primary_evidence']}",
        f"- ARC-Challenge content: {evidence['arc_challenge_content_evidence']}",
        f"- ARC-Challenge cyclic: {evidence['arc_challenge_cyclic_evidence']}",
        f"- ARC-Easy content: {evidence['arc_easy_content_evidence']}",
        f"- ARC-Easy cyclic: {evidence['arc_easy_cyclic_evidence']}",
        f"- Metadata differences: `{payload['metadata_differences']}`",
        "",
        "## Surface Notes",
        "",
        f"- Recurrent-vs-base surface notes: `{surface}`",
    ]
    output_md.parent.mkdir(parents=True, exist_ok=True)
    output_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(output_md.read_text(encoding="utf-8"), flush=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dense_summary_json", required=True)
    parser.add_argument("--recurrent_summary_json", required=True)
    parser.add_argument("--output_json")
    parser.add_argument("--output_md")
    parser.add_argument("--primary_benchmark", default=DEFAULT_PRIMARY_BENCHMARK)
    parser.add_argument("--primary_score_target", default=DEFAULT_PRIMARY_SCORE_TARGET)
    parser.add_argument("--primary_aggregate", default=DEFAULT_PRIMARY_AGGREGATE)
    parser.add_argument("--min_primary_examples", type=int, default=20)
    args = parser.parse_args()

    output_dir = ROOT / "outputs" / "stage5" / RUN_ID
    output_json = resolve_path(args.output_json) if args.output_json else output_dir / "summary.json"
    output_md = resolve_path(args.output_md) if args.output_md else output_dir / "summary.md"
    payload = assess_mcq_recipe_control(
        dense_summary_path=resolve_path(args.dense_summary_json),
        recurrent_summary_path=resolve_path(args.recurrent_summary_json),
        primary_benchmark=args.primary_benchmark,
        primary_score_target=args.primary_score_target,
        primary_aggregate=args.primary_aggregate,
        min_primary_examples=args.min_primary_examples,
    )
    write_report(payload, output_json=output_json, output_md=output_md)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
