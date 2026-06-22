"""Review a finished Stage 5 ARC-mix proxy and state the next spend decision.

This is a no-GPU helper for the low-credit workflow. It reads a
``stage5_balanced_arc_mix_gate`` summary, runs the normal Stage 5 planner on
that summary, and prints a compact report answering the only question that
matters after the proxy finishes:

    Does this result justify one full balanced ARC confirmation run?
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any

try:
    from colab.plan_stage5_next_run import path_for_cli, plan_next_actions, read_json
except ModuleNotFoundError:  # pragma: no cover - direct ``python colab/script.py`` execution
    from plan_stage5_next_run import path_for_cli, plan_next_actions, read_json


ROOT = Path(__file__).resolve().parents[1]
RUN_ID = os.environ.get("STAGE5_ARC_MIX_REVIEW_RUN_ID") or time.strftime(
    "stage5_arc_mix_review_%Y%m%d_%H%M%S"
)
RUN_DIR = ROOT / "outputs" / "stage5" / RUN_ID


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--summary",
        default=None,
        help="ARC-mix summary JSON. Defaults to the latest stage5_balanced_arc_mix_gate summary.",
    )
    parser.add_argument(
        "--no-write",
        action="store_true",
        help="Print the review but do not write outputs/stage5/<run_id>/summary.{json,md}.",
    )
    return parser.parse_args(argv)


def latest_arc_mix_summary() -> Path:
    candidates: list[Path] = []
    for path in ROOT.glob("outputs/stage5/*/summary.json"):
        try:
            payload = read_json(path)
        except Exception:
            continue
        if payload.get("kind") == "stage5_balanced_arc_mix_gate":
            candidates.append(path)
    if not candidates:
        raise FileNotFoundError("No stage5_balanced_arc_mix_gate summary found.")
    return sorted(candidates, key=lambda item: item.stat().st_mtime, reverse=True)[0]


def resolve_summary(value: str | None) -> Path:
    if not value:
        return latest_arc_mix_summary()
    path = Path(value.replace("\\", "/"))
    return path if path.is_absolute() else ROOT / path


def nested_int(payload: dict[str, Any], keys: tuple[str, ...]) -> int | None:
    current: Any = payload
    for key in keys:
        if not isinstance(current, dict):
            return None
        current = current.get(key)
    if current is None:
        return None
    return int(current)


def best_arm_metrics(best_arm: dict[str, Any]) -> dict[str, Any]:
    best_checkpoint = best_arm.get("best_checkpoint") if isinstance(best_arm, dict) else {}
    phase1_start = best_arm.get("phase1_start") if isinstance(best_arm, dict) else {}
    base_arc = best_arm.get("base_arc") if isinstance(best_arm, dict) else {}
    best_mean = ((best_checkpoint or {}).get("arc") or {}).get("mean") or {}
    start_mean = ((phase1_start or {}).get("arc") or {}).get("mean") or {}
    base_mean = ((base_arc or {}).get("mean") or {})
    comparison = (best_checkpoint or {}).get("comparison_to_base") or {}

    best_correct = int(best_mean.get("correct", 0))
    start_correct = int(start_mean.get("correct", 0))
    base_correct = int(base_mean.get("correct", 0))
    return {
        "arm": best_arm.get("arm") if isinstance(best_arm, dict) else None,
        "checkpoint": (best_checkpoint or {}).get("checkpoint"),
        "best_correct": best_correct,
        "best_total": int(best_mean.get("total", 0)),
        "start_correct": start_correct,
        "start_total": int(start_mean.get("total", 0)),
        "base_correct": base_correct,
        "base_total": int(base_mean.get("total", 0)),
        "lift_vs_start": best_correct - start_correct,
        "gap_vs_base": best_correct - base_correct,
        "mean_margin_delta": comparison.get("mean_margin_delta"),
        "max_abs_prediction_count_delta": comparison.get("max_abs_prediction_count_delta"),
        "calibration_ok": comparison.get("calibration_ok"),
    }


def build_review(payload: dict[str, Any], *, source_summary: Path) -> dict[str, Any]:
    actions = plan_next_actions(payload, source_summary=source_summary)
    action = actions[0] if actions else None
    decision = str(payload.get("decision") or "")
    action_command = str((action or {}).get("command") or "")
    legacy_full_assessment = not decision and "run_stage5_recovery_full_assessment.py" in action_command
    full_assessment_justified = decision == "run_full_balanced_assessment" or legacy_full_assessment
    return {
        "run_id": RUN_ID,
        "kind": "stage5_arc_mix_result_review",
        "source_summary": path_for_cli(source_summary),
        "source_status": payload.get("status"),
        "source_passed": payload.get("passed"),
        "decision": decision,
        "blocked_reason": payload.get("blocked_reason"),
        "next_step": payload.get("next_step"),
        "full_assessment_justified": full_assessment_justified,
        "decision_basis": "explicit_decision" if decision else "legacy_planner_action",
        "best_arm": best_arm_metrics(payload.get("best_arm") or {}),
        "recommended_action": action,
        "all_actions": actions,
    }


def render_markdown(review: dict[str, Any]) -> str:
    best = review["best_arm"]
    action = review.get("recommended_action") or {}
    if review["full_assessment_justified"]:
        next_spend = "YES: run exactly one full balanced ARC confirmation."
        if review.get("decision_basis") == "legacy_planner_action":
            next_spend += " Legacy summary lacks explicit decision/calibration fields."
    else:
        next_spend = "NO: stop A100 work and repair locally."
    margin = best.get("mean_margin_delta")
    lines = [
        f"# Stage 5 ARC-Mix Result Review - {review['run_id']}",
        "",
        f"- Source summary: `{review['source_summary']}`",
        f"- Source status: `{review['source_status']}`",
        f"- Source passed: `{review['source_passed']}`",
        f"- Decision: `{review['decision']}`",
        f"- Decision basis: `{review['decision_basis']}`",
        f"- Blocked reason: {review['blocked_reason'] or 'none'}",
        f"- Next A100 spend: **{next_spend}**",
        "",
        "## Best Arm",
        "",
        f"- Arm: `{best.get('arm')}`",
        f"- Checkpoint: `{best.get('checkpoint')}`",
        f"- Proxy score: `{best['best_correct']}/{best['best_total']}`",
        f"- Start score: `{best['start_correct']}/{best['start_total']}`",
        f"- Base score: `{best['base_correct']}/{best['base_total']}`",
        f"- Lift vs start: `{best['lift_vs_start']}`",
        f"- Gap vs base: `{best['gap_vs_base']}`",
        f"- Mean margin delta vs base: `{'n/a' if margin is None else f'{float(margin):.6f}'}`",
        f"- Max prediction-count shift: `{best.get('max_abs_prediction_count_delta')}`",
        f"- Calibration OK: `{best.get('calibration_ok')}`",
        "",
        "## Planner Action",
        "",
        f"- Name: `{action.get('name')}`",
        f"- Priority: `{action.get('priority')}`",
        f"- Command: `{action.get('command')}`",
        "",
    ]
    return "\n".join(lines)


def write_review(review: dict[str, Any], markdown: str) -> None:
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    (RUN_DIR / "summary.json").write_text(json.dumps(review, indent=2), encoding="utf-8")
    (RUN_DIR / "summary.md").write_text(markdown + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    summary = resolve_summary(args.summary)
    payload = read_json(summary)
    if payload.get("kind") != "stage5_balanced_arc_mix_gate":
        raise ValueError(f"Expected stage5_balanced_arc_mix_gate summary, got {payload.get('kind')!r}")
    review = build_review(payload, source_summary=summary)
    markdown = render_markdown(review)
    print(markdown)
    if not args.no_write:
        write_review(review, markdown)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
