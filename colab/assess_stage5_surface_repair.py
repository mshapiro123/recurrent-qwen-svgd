"""Assess whether a surface-alignment repair helped without hard-tail harm.

The generic benchmark gate asks whether a checkpoint is nonnegative versus
base. This assessor answers the narrower repair question:

* Did the repaired recurrent checkpoint improve ARC-Easy content scoring versus
  the source recurrent checkpoint?
* Did the repaired checkpoint preserve ARC-Challenge content/cyclic behavior?
* Where does the repaired checkpoint stand versus base on the same surfaces?
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
RUN_ID = os.environ.get("STAGE5_SURFACE_REPAIR_ASSESS_RUN_ID") or time.strftime(
    "stage5_surface_repair_assessment_%Y%m%d_%H%M%S"
)


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


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def two_sided_sign_p_value(wins: int, losses: int) -> float | None:
    trials = wins + losses
    if trials == 0:
        return None
    observed = min(wins, losses)
    return min(1.0, 2.0 * sum(math.comb(trials, k) for k in range(observed + 1)) / (2**trials))


def rows_by_id(path: Path, aggregate: str) -> dict[str, dict[str, Any]]:
    return {
        str(row["id"]): row
        for row in read_jsonl(path)
        if row.get("id") is not None and str(row.get("aggregate") or "mean") == aggregate
    }


def compare_hit_rows(reference_path: Path, candidate_path: Path, *, aggregate: str) -> dict[str, Any]:
    reference = rows_by_id(reference_path, aggregate)
    candidate = rows_by_id(candidate_path, aggregate)
    common = sorted(set(reference) & set(candidate))
    wins: list[str] = []
    losses: list[str] = []
    ties = 0
    reference_correct = 0
    candidate_correct = 0
    for row_id in common:
        reference_hit = bool(reference[row_id].get("hit"))
        candidate_hit = bool(candidate[row_id].get("hit"))
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
    return {
        "aggregate": aggregate,
        "paired_examples": total,
        "reference_correct": reference_correct,
        "candidate_correct": candidate_correct,
        "correct_delta_candidate_vs_reference": delta,
        "reference_accuracy": reference_correct / max(total, 1),
        "candidate_accuracy": candidate_correct / max(total, 1),
        "accuracy_delta_candidate_vs_reference": delta / max(total, 1),
        "wins": len(wins),
        "losses": len(losses),
        "ties": ties,
        "sign_test_p_value": two_sided_sign_p_value(len(wins), len(losses)),
        "win_ids": wins,
        "loss_ids": losses,
    }


def recurrent_path(summary_path: Path, benchmark: str, score_target: str) -> Path:
    candidate = summary_path.parent / f"{benchmark}_recurrent_{score_target}.jsonl"
    if not candidate.exists():
        raise FileNotFoundError(candidate)
    return candidate


def paired_vs_base(summary: dict[str, Any], benchmark: str, score_target: str, aggregate: str) -> dict[str, Any]:
    row = (
        (summary.get("paired_comparisons") or {})
        .get(benchmark, {})
        .get(score_target, {})
        .get(aggregate)
    )
    return row if isinstance(row, dict) else {}


def surface_comparison(
    *,
    source_summary_path: Path,
    repaired_summary_path: Path,
    benchmark: str,
    score_target: str,
    aggregate: str,
) -> dict[str, Any]:
    source_payload = read_json(source_summary_path)
    repaired_payload = read_json(repaired_summary_path)
    before_after = compare_hit_rows(
        recurrent_path(source_summary_path, benchmark, score_target),
        recurrent_path(repaired_summary_path, benchmark, score_target),
        aggregate=aggregate,
    )
    return {
        "benchmark": benchmark,
        "score_target": score_target,
        "aggregate": aggregate,
        "source_recurrent_vs_repaired_recurrent": before_after,
        "source_recurrent_vs_base": paired_vs_base(source_payload, benchmark, score_target, aggregate),
        "repaired_recurrent_vs_base": paired_vs_base(repaired_payload, benchmark, score_target, aggregate),
    }


def base_delta(row: dict[str, Any]) -> int:
    return int(row.get("correct_delta_recurrent_vs_base", 0) or 0)


def repair_delta(row: dict[str, Any]) -> int:
    before_after = row.get("source_recurrent_vs_repaired_recurrent") or {}
    return int(before_after.get("correct_delta_candidate_vs_reference", 0) or 0)


def evidence_fragment(row: dict[str, Any]) -> str:
    before_after = row.get("source_recurrent_vs_repaired_recurrent") or {}
    repaired_base = row.get("repaired_recurrent_vs_base") or {}
    return (
        f"repair_delta {before_after.get('correct_delta_candidate_vs_reference', 0)} "
        f"({before_after.get('wins', 0)}/{before_after.get('losses', 0)}/{before_after.get('ties', 0)} W/L/T, "
        f"n={before_after.get('paired_examples', 0)}, p={before_after.get('sign_test_p_value')}), "
        f"repaired_vs_base_delta {repaired_base.get('correct_delta_recurrent_vs_base', 0)}"
    )


def decide(comparisons: dict[str, dict[str, Any]], *, allowed_challenge_regression: int) -> tuple[str, bool, str, str]:
    easy_content = comparisons["arc_easy_content"]
    easy_cyclic = comparisons["arc_easy_cyclic"]
    challenge_content = comparisons["arc_challenge_content"]
    challenge_cyclic = comparisons["arc_challenge_cyclic"]

    easy_content_improved = repair_delta(easy_content) > 0
    easy_content_base_delta = base_delta(easy_content.get("repaired_recurrent_vs_base") or {})
    easy_cyclic_preserved = repair_delta(easy_cyclic) >= -allowed_challenge_regression
    challenge_content_preserved = repair_delta(challenge_content) >= -allowed_challenge_regression
    challenge_cyclic_preserved = repair_delta(challenge_cyclic) >= -allowed_challenge_regression
    hard_preserved = challenge_content_preserved and challenge_cyclic_preserved

    if easy_content_improved and easy_content_base_delta >= 0 and hard_preserved and easy_cyclic_preserved:
        return (
            "surface_repair_passed",
            True,
            "Repair improved ARC-Easy content, restored nonnegative Easy content versus base, and preserved Challenge surfaces.",
            "Run dense same-curriculum control, then compare recurrent-vs-dense architecture lift.",
        )
    if easy_content_improved and hard_preserved:
        return (
            "surface_repair_partial",
            False,
            "Repair improved ARC-Easy content and preserved Challenge surfaces, but did not fully restore Easy content versus base.",
            "Use this as a diagnostic improvement, but do not claim recovered base parity yet.",
        )
    if easy_content_improved and not hard_preserved:
        return (
            "surface_repair_tradeoff",
            False,
            "Repair improved ARC-Easy content but regressed at least one ARC-Challenge surface.",
            "Tighten direct-path-only or invariance weighting before scaling this repair.",
        )
    return (
        "surface_repair_no_easy_content_lift",
        False,
        "Repair did not improve ARC-Easy content versus the source recurrent checkpoint.",
        "Inspect training rows and consider score-level alignment rather than more SFT on the same shard.",
    )


def assess_surface_repair(
    *,
    source_benchmark_summary: Path,
    repaired_benchmark_summary: Path,
    allowed_challenge_regression: int = 0,
) -> dict[str, Any]:
    source_payload = read_json(source_benchmark_summary)
    repaired_payload = read_json(repaired_benchmark_summary)
    if source_payload.get("kind") != "stage5_benchmark_suite":
        raise ValueError(f"Not a benchmark suite summary: {source_benchmark_summary}")
    if repaired_payload.get("kind") != "stage5_benchmark_suite":
        raise ValueError(f"Not a benchmark suite summary: {repaired_benchmark_summary}")

    comparisons = {
        "arc_easy_content": surface_comparison(
            source_summary_path=source_benchmark_summary,
            repaired_summary_path=repaired_benchmark_summary,
            benchmark="arc_easy",
            score_target="content_question_only",
            aggregate="mean",
        ),
        "arc_easy_cyclic": surface_comparison(
            source_summary_path=source_benchmark_summary,
            repaired_summary_path=repaired_benchmark_summary,
            benchmark="arc_easy",
            score_target="cyclic_label_aggregated",
            aggregate="permutation_mean",
        ),
        "arc_challenge_content": surface_comparison(
            source_summary_path=source_benchmark_summary,
            repaired_summary_path=repaired_benchmark_summary,
            benchmark="arc_challenge",
            score_target="content_question_only",
            aggregate="mean",
        ),
        "arc_challenge_cyclic": surface_comparison(
            source_summary_path=source_benchmark_summary,
            repaired_summary_path=repaired_benchmark_summary,
            benchmark="arc_challenge",
            score_target="cyclic_label_aggregated",
            aggregate="permutation_mean",
        ),
    }
    status, passed, reason, next_step = decide(
        comparisons,
        allowed_challenge_regression=allowed_challenge_regression,
    )
    return {
        "run_id": RUN_ID,
        "kind": "stage5_surface_repair_assessment",
        "status": status,
        "passed": passed,
        "reason": reason,
        "next_step": next_step,
        "source_benchmark_summary": path_for_cli(source_benchmark_summary),
        "repaired_benchmark_summary": path_for_cli(repaired_benchmark_summary),
        "allowed_challenge_regression": allowed_challenge_regression,
        "comparisons": comparisons,
        "decision_evidence": {key: evidence_fragment(value) for key, value in comparisons.items()},
    }


def write_report(payload: dict[str, Any], *, output_json: Path, output_md: Path) -> None:
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    lines = [
        f"# Stage 5 Surface Repair Assessment - {payload['run_id']}",
        "",
        f"- Status: `{payload['status']}`",
        f"- Passed: `{payload['passed']}`",
        f"- Source benchmark: `{payload['source_benchmark_summary']}`",
        f"- Repaired benchmark: `{payload['repaired_benchmark_summary']}`",
        f"- Reason: {payload['reason']}",
        f"- Next step: {payload['next_step']}",
        "",
        "## Decision Evidence",
        "",
    ]
    for key, evidence in payload["decision_evidence"].items():
        lines.append(f"- `{key}`: {evidence}")
    output_md.parent.mkdir(parents=True, exist_ok=True)
    output_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(output_md.read_text(encoding="utf-8"), flush=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source_benchmark_summary", required=True)
    parser.add_argument("--repaired_benchmark_summary", required=True)
    parser.add_argument("--output_json")
    parser.add_argument("--output_md")
    parser.add_argument("--allowed_challenge_regression", type=int, default=0)
    args = parser.parse_args()

    output_dir = ROOT / "outputs" / "stage5" / RUN_ID
    output_json = resolve_path(args.output_json) if args.output_json else output_dir / "summary.json"
    output_md = resolve_path(args.output_md) if args.output_md else output_json.with_suffix(".md")
    payload = assess_surface_repair(
        source_benchmark_summary=resolve_path(args.source_benchmark_summary),
        repaired_benchmark_summary=resolve_path(args.repaired_benchmark_summary),
        allowed_challenge_regression=args.allowed_challenge_regression,
    )
    write_report(payload, output_json=output_json, output_md=output_md)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
