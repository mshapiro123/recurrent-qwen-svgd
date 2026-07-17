"""Locked decision rules for the corrected-loop PEFT and PonderNet closure."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any


CANONICAL_CORRECT = 46
CANONICAL_TOTAL = 64
CANONICAL_THRESHOLD = 0.71
FULL_BLOCK_REFERENCE = {"1": 1.0, "2": 0.984375, "3": 0.984375, "4": 0.921875}
CANARY_ROWS = 64
CANARY_MIN_BASELINE_ACCURACY = 0.5


@dataclass(frozen=True)
class PeftArm:
    name: str
    rank: int
    alpha: int
    estimated_block_trainable: int
    full_block_fraction: float


ARMS = (
    PeftArm("R16", 16, 32, 4_400_000, 0.025),
    PeftArm("R64", 64, 128, 17_600_000, 0.098),
    PeftArm("R256", 256, 512, 70_400_000, 0.393),
)


def locked_spec() -> dict[str, Any]:
    return {
        "arms": [asdict(arm) for arm in ARMS],
        "task": "N16 synthetic forward chain, depths 1-4",
        "optimizer": "adamw",
        "bridge_projection_mode": "split",
        "bridge_prelude_lr_multiplier": 10.0,
        "bridge_prelude_grad_multiplier": 1.0,
        "identity_max_abs_diff": 1e-3,
        "canary": {
            "kind": "frozen_base_capability_arithmetic",
            "rows": CANARY_ROWS,
            "minimum_baseline_accuracy": CANARY_MIN_BASELINE_ACCURACY,
            "hard_stop_accuracy_delta": -0.03,
        },
        "p1_steps": 6000,
        "p1_interval": 1000,
        "p1_gate": {
            "correct_per_depth": CANONICAL_CORRECT,
            "total_per_depth": CANONICAL_TOTAL,
            "threshold_label": CANONICAL_THRESHOLD,
        },
        "r256_rider_total_steps": 12000,
        "p2_steps": 2000,
        "p2_gate": {
            "mean_depth_exclusive_min": 1.5,
            "mean_depth_exclusive_max": 3.5,
            "learned_accuracy_max_gap": 0.03,
            "requires_loss_decrease": True,
            "requires_kl_stabilization": True,
        },
        "full_block_reference": dict(FULL_BLOCK_REFERENCE),
    }


def _canary_row(
    *,
    family: str,
    index: int,
    question: str,
    answer: int,
    target_index: int,
) -> dict[str, Any]:
    distractors = [answer - 2, answer - 1, answer + 1]
    candidates = [str(value) for value in distractors]
    candidates.insert(target_index, str(answer))
    return {
        "id": f"base_capability_{family}_{index:02d}",
        "question": f"{question} Answer with only the number.",
        "target": target_index,
        "answer_text": str(answer),
        "depth": 1,
        "synthetic_depth": 1,
        "start": 0,
        "orbit": [0, target_index],
        "mapping": {str(value): str(value) for value in range(4)},
        "symbol_names": candidates,
        "n_symbols": 4,
        "score_target": "full_symbols",
        "prompt_style": "question_only",
        "intermediate_chain_supervision": True,
        "chain_symbol_by_loop": {"1": str(answer)},
        "chain_answer_by_loop": {"1": str(answer)},
        "loop_completions": [f" {answer}"],
        "completion": f" {answer}",
        "target_loop_count": 1,
        "synthetic_task": "base_capability_canary",
        "canary_family": family,
    }


def build_base_capability_canary_rows() -> list[dict[str, Any]]:
    """Return the frozen, base-solvable loop-1 preservation canary."""

    rows: list[dict[str, Any]] = []
    for index in range(16):
        left = 13 + index
        right = 7 + (3 * index) % 17
        rows.append(
            _canary_row(
                family="addition",
                index=index,
                question=f"What is {left} + {right}?",
                answer=left + right,
                target_index=index % 4,
            )
        )

        minuend = 40 + 2 * index
        subtrahend = 5 + index % 11
        rows.append(
            _canary_row(
                family="subtraction",
                index=index,
                question=f"What is {minuend} - {subtrahend}?",
                answer=minuend - subtrahend,
                target_index=(index + 1) % 4,
            )
        )

        factor_a = 2 + index % 9
        factor_b = 3 + (2 * index) % 8
        rows.append(
            _canary_row(
                family="multiplication",
                index=index,
                question=f"What is {factor_a} times {factor_b}?",
                answer=factor_a * factor_b,
                target_index=(index + 2) % 4,
            )
        )

        divisor = 2 + index % 7
        quotient = 3 + (3 * index) % 11
        dividend = divisor * quotient
        rows.append(
            _canary_row(
                family="division",
                index=index,
                question=f"What is {dividend} divided by {divisor}?",
                answer=quotient,
                target_index=(index + 3) % 4,
            )
        )
    if len(rows) != CANARY_ROWS:
        raise AssertionError(f"Expected {CANARY_ROWS} canary rows, got {len(rows)}")
    return rows


def canary_baseline_gate(
    active_summary: dict[str, Any],
    *,
    minimum_accuracy: float = CANARY_MIN_BASELINE_ACCURACY,
) -> dict[str, Any]:
    total_row = active_summary.get("active_total") or {}
    correct = int(total_row.get("correct", 0))
    total = int(total_row.get("total", 0))
    accuracy = correct / total if total else 0.0
    passed = total == CANARY_ROWS and accuracy >= minimum_accuracy
    return {
        "passed": passed,
        "correct": correct,
        "total": total,
        "accuracy": accuracy,
        "minimum_accuracy": minimum_accuracy,
        "reason": (
            "nonvacuous_baseline_confirmed"
            if passed
            else "baseline_below_nonvacuous_floor"
        ),
    }


def depth_counts(active_summary: dict[str, Any]) -> dict[str, dict[str, int | float]]:
    matrix = active_summary.get("active_matrix") or {}
    out: dict[str, dict[str, int | float]] = {}
    for depth in ("1", "2", "3", "4"):
        cell = (matrix.get(depth) or {}).get(depth) or {}
        correct = int(cell.get("correct", 0))
        total = int(cell.get("total", 0))
        out[depth] = {
            "correct": correct,
            "total": total,
            "accuracy": correct / total if total else 0.0,
        }
    return out


def p1_gate(active_summary: dict[str, Any]) -> dict[str, Any]:
    counts = depth_counts(active_summary)
    passed = all(
        int(cell["correct"]) >= CANONICAL_CORRECT
        and int(cell["total"]) == CANONICAL_TOTAL
        for cell in counts.values()
    )
    return {
        "passed": passed,
        "counts": counts,
        "required_correct": CANONICAL_CORRECT,
        "required_total": CANONICAL_TOTAL,
    }


def next_p1_action(results: list[dict[str, Any]]) -> str:
    """Return the next preregistered ladder action."""

    if not results:
        return "run_R16"
    last = results[-1]
    if bool((last.get("gate") or {}).get("passed")):
        return "run_P2_on_first_pass"
    arm = str(last["arm"])
    total_steps = int(last.get("total_steps", 6000))
    if arm == "R16":
        return "run_R64"
    if arm == "R64":
        return "run_R256"
    if arm == "R256" and total_steps < 12000:
        return "continue_R256_to_12000"
    return "close_P1_bounded_refutation"


def wilson_interval(correct: int, total: int, z: float = 1.959963984540054) -> tuple[float, float]:
    if total <= 0:
        return (0.0, 0.0)
    p = correct / total
    denominator = 1.0 + z * z / total
    center = (p + z * z / (2.0 * total)) / denominator
    radius = z * math.sqrt((p * (1.0 - p) + z * z / (4.0 * total)) / total) / denominator
    return (max(0.0, center - radius), min(1.0, center + radius))


def full_block_comparison(active_summary: dict[str, Any]) -> dict[str, Any]:
    counts = depth_counts(active_summary)
    return {
        depth: {
            **cell,
            "reference_accuracy": FULL_BLOCK_REFERENCE[depth],
            "delta": float(cell["accuracy"]) - FULL_BLOCK_REFERENCE[depth],
            "wilson_95": list(wilson_interval(int(cell["correct"]), int(cell["total"]))),
        }
        for depth, cell in counts.items()
    }


def training_trace_gate(trace: list[dict[str, Any]]) -> dict[str, Any]:
    metrics = [row.get("metrics") or {} for row in trace]
    losses = [float(row["loss"]) for row in metrics if row.get("loss") is not None]
    kls = [float(row["halting_kl"]) for row in metrics if row.get("halting_kl") is not None]
    loss_decreased = len(losses) >= 2 and losses[-1] < losses[0]
    kl_stable = False
    if len(kls) >= 4 and all(math.isfinite(value) for value in kls):
        tail = kls[-max(4, len(kls) // 4) :]
        scale = max(abs(sum(tail) / len(tail)), 1e-8)
        kl_stable = (max(tail) - min(tail)) / scale <= 0.5
    return {
        "loss_decreased": loss_decreased,
        "kl_stable": kl_stable,
        "loss_first": losses[0] if losses else None,
        "loss_last": losses[-1] if losses else None,
        "kl_tail": kls[-max(4, len(kls) // 4) :] if kls else [],
    }


def p2_gate(training_summary: dict[str, Any], eval_summary: dict[str, Any]) -> dict[str, Any]:
    trace = training_trace_gate(list(training_summary.get("curriculum_trace") or []))
    mean_depth = float(eval_summary.get("mean_expected_loops", 0.0))
    learned_accuracy = float(eval_summary.get("learned_depth_accuracy", 0.0))
    forced_accuracy = float(eval_summary.get("forced_depth_accuracy", 0.0))
    accuracy_gap = forced_accuracy - learned_accuracy
    depth_nontrivial = 1.5 < mean_depth < 3.5
    accuracy_preserved = accuracy_gap <= 0.03
    passed = (
        trace["loss_decreased"]
        and trace["kl_stable"]
        and depth_nontrivial
        and accuracy_preserved
    )
    return {
        "passed": passed,
        **trace,
        "mean_expected_loops": mean_depth,
        "depth_nontrivial": depth_nontrivial,
        "learned_depth_accuracy": learned_accuracy,
        "forced_depth_accuracy": forced_accuracy,
        "accuracy_gap": accuracy_gap,
        "accuracy_preserved": accuracy_preserved,
    }


def historical_archive_receipt() -> dict[str, Any]:
    return {
        "repaired_loop_peft_arm_found": False,
        "inadmissible_runs": [
            {
                "ranks": [32, 64, 128],
                "date": "2026-06-27",
                "steps": 100,
                "reason": (
                    "Capacity-localization arms predated the corrected input re-injection "
                    "and corrected split-bridge true-LR training path."
                ),
                "summary": "outputs/stage5/stage5_capacity_localization_20260627_210858/summary.json",
            }
        ],
    }
