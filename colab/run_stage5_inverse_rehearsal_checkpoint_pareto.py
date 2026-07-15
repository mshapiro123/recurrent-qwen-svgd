"""Evaluate saved cap-3 rehearsal checkpoints against the locked joint gates.

This is a diagnostic checkpoint sweep, not a post-hoc authorization mechanism.
Any apparent winner requires an independent confirmation canary before it can
be used as a successor keeper.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from colab.run_stage5_inverse_composition_staircase import (
    _guardrail_receipt,
    _prepare_guardrail_data,
    _publish,
    _run_diagonal,
    _run_staircase_eval,
    build_matched_data,
    write_json,
)
from colab.run_stage5_natural_surface_transfer import restore_checkpoint, sha256_file
from colab.stage5_n24_rung import tier1_canary_verdict
from colab.stage5_chain_consolidation_utils import path_for_cli


ROOT = Path(os.environ.get("STAGE5_ROOT", REPO_ROOT))
RUN_ID = "stage5_inverse_table_cap3_rehearsal_20260714"
DEFAULT_STEPS = (100, 200, 300, 334)
TASK_REQUIRED_CORRECT = 46
SYNTHETIC_FLOOR = 0.93
DRIVE_TRAIN_ROOT = Path(
    "/content/drive/MyDrive/recurrent-qwen-svgd-artifacts/"
    f"stage5/{RUN_ID}/train/checkpoints"
)


def read_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def parse_steps(value: str | None) -> tuple[int, ...]:
    if not value:
        return DEFAULT_STEPS
    steps = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    if not steps or any(step <= 0 for step in steps):
        raise ValueError("STAGE5_REHEARSAL_PARETO_STEPS must contain positive integer steps")
    return steps


def assess_checkpoint_pareto(
    *,
    task_correct: int,
    task_total: int,
    synthetic_min: float,
    natural_accuracy: float,
    natural_baseline_accuracy: float,
) -> dict[str, Any]:
    task_passed = int(task_total) == 64 and int(task_correct) >= TASK_REQUIRED_CORRECT
    synthetic_passed = float(synthetic_min) >= SYNTHETIC_FLOOR
    natural_delta = float(natural_accuracy) - float(natural_baseline_accuracy)
    natural_verdict = tier1_canary_verdict(accuracy_delta=natural_delta, ppl_relative_delta=None)
    natural_passed = natural_verdict["status"] != "red_hard_stop"
    all_passed = task_passed and synthetic_passed and natural_passed
    return {
        "task": {
            "correct": int(task_correct),
            "total": int(task_total),
            "required_correct": TASK_REQUIRED_CORRECT,
            "passed": task_passed,
        },
        "synthetic": {
            "active_diagonal_min": float(synthetic_min),
            "floor": SYNTHETIC_FLOOR,
            "passed": synthetic_passed,
        },
        "natural": {
            "baseline_accuracy": float(natural_baseline_accuracy),
            "candidate_accuracy": float(natural_accuracy),
            "accuracy_delta": natural_delta,
            "verdict": natural_verdict,
            "passed": natural_passed,
        },
        "all_current_gates_passed": all_passed,
        "selection_status": "candidate_requires_fresh_confirmation" if all_passed else "not_a_joint_gate_candidate",
    }


def checkpoint_candidates(run_dir: Path, step: int) -> list[Path]:
    checkpoint_name = f"unfrozen_recurrent_step_{int(step)}.pt"
    candidates = [
        run_dir / "train" / "cap3_rehearsal" / checkpoint_name,
        DRIVE_TRAIN_ROOT / checkpoint_name,
    ]
    if int(step) == 334:
        candidates.append(
            Path(
                "/content/drive/MyDrive/recurrent-qwen-svgd-checkpoints/"
                f"{RUN_ID}/cap3_rehearsal_selected/{checkpoint_name}"
            )
        )
    return candidates


def main() -> int:
    run_dir = ROOT / "outputs" / "stage5" / RUN_ID
    rehearsal_summary = read_json(run_dir / "summary.json")
    if rehearsal_summary.get("kind") != "stage5_inverse_table_cap3_rehearsal":
        raise RuntimeError("Expected the cap-3 rehearsal summary")
    steps = parse_steps(os.environ.get("STAGE5_REHEARSAL_PARETO_STEPS"))
    data_dir = run_dir / "checkpoint_pareto" / "data"
    build_matched_data(data_dir)
    guardrail_paths = _prepare_guardrail_data(run_dir / "checkpoint_pareto")
    task_test = data_dir / "test_C_inverse_table.jsonl"
    task_train = data_dir / "train_C_inverse_table.jsonl"
    locked_baseline = read_json(
        ROOT
        / "outputs/stage5/stage5_inverse_composition_staircase_20260713/"
        "guardrails/tier1_natural_surface_keeper/summary.json"
    )
    conditions: list[dict[str, Any]] = []
    for step in steps:
        checkpoint, restore_receipt = restore_checkpoint(
            checkpoint_candidates(run_dir, step),
            run_dir / "checkpoint_pareto" / "restored" / f"step_{step}.pt",
            label=f"rehearsal_pareto_step_{step}",
        )
        checkpoint_sha = sha256_file(checkpoint)
        condition_dir = run_dir / "checkpoint_pareto" / f"step_{step}"
        task_eval = _run_staircase_eval(
            condition_dir,
            label="task",
            checkpoint=checkpoint,
            train_jsonl=task_train,
            test_jsonl=task_test,
            cap=3,
            probes=False,
        )
        task_row = task_eval["test"]["diagonal_by_depth"]["3"]
        synthetic = _guardrail_receipt(
            condition_dir,
            label="synthetic",
            checkpoint=checkpoint,
            data_jsonl=guardrail_paths["synthetic"],
        )
        natural = _run_diagonal(
            condition_dir,
            label="natural",
            checkpoint=checkpoint,
            data_jsonl=guardrail_paths["natural"],
            max_depth=8,
            value_prefix="name:",
        )
        assessment = assess_checkpoint_pareto(
            task_correct=int(task_row["correct"]),
            task_total=int(task_row["total"]),
            synthetic_min=float(synthetic["active_diagonal_min"]),
            natural_accuracy=float(natural["accuracy"]),
            natural_baseline_accuracy=float(locked_baseline["accuracy"]),
        )
        condition = {
            "step": int(step),
            "checkpoint": path_for_cli(checkpoint),
            "checkpoint_sha256": checkpoint_sha,
            "restore_receipt": restore_receipt,
            "task_summary": path_for_cli(condition_dir / "eval" / "task" / "summary.json"),
            "synthetic_summary": synthetic["summary"],
            "natural_summary": path_for_cli(condition_dir / "guardrails" / "natural" / "summary.json"),
            "assessment": assessment,
        }
        write_json(condition_dir / "condition.json", condition)
        conditions.append(condition)
    candidates = [condition for condition in conditions if condition["assessment"]["all_current_gates_passed"]]
    payload = {
        "kind": "stage5_inverse_rehearsal_checkpoint_pareto",
        "run_id": RUN_ID,
        "steps": list(steps),
        "conditions": conditions,
        "joint_gate_candidates": [condition["step"] for condition in candidates],
        "status": "confirmation_required" if candidates else "no_joint_gate_candidate",
        "cap4_authorized": False,
        "note": "This post-training sweep is diagnostic only; any candidate requires a fresh confirmation canary.",
    }
    pareto_dir = run_dir / "checkpoint_pareto"
    write_json(pareto_dir / "summary.json", payload)
    lines = [
        "# Cap-3 Rehearsal Checkpoint Pareto Sweep",
        "",
        "| Step | Task d3 | Synthetic min | Natural accuracy | Natural delta | Current gates |",
        "|---:|---:|---:|---:|---:|---|",
    ]
    for condition in conditions:
        assessment = condition["assessment"]
        lines.append(
            f"| {condition['step']} | {assessment['task']['correct']}/{assessment['task']['total']} | "
            f"{assessment['synthetic']['active_diagonal_min']:.5f} | "
            f"{assessment['natural']['candidate_accuracy']:.5f} | "
            f"{assessment['natural']['accuracy_delta']:+.5f} | "
            f"{assessment['all_current_gates_passed']} |"
        )
    lines.extend(["", "No checkpoint is authorized by this post-training sweep."])
    (pareto_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True), flush=True)
    _publish(run_dir, f"Record cap-3 rehearsal checkpoint Pareto sweep {RUN_ID} [skip ci]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
