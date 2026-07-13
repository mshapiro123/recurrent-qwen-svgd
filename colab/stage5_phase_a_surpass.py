"""Pre-registered scoring helpers for the Phase-A surpass comparison."""

from __future__ import annotations

import math
from typing import Any


PHASE_A_SUPPORT_DEPTH = 8
PHASE_A_EVAL_MAX_DEPTH = 14
PHASE_A_ROWS_PER_DEPTH = 128
PHASE_A_DENSE_STEPS = 4000


def _log_choose(n: int, k: int) -> float:
    if k < 0 or k > n:
        return float("-inf")
    return math.lgamma(n + 1) - math.lgamma(k + 1) - math.lgamma(n - k + 1)


def fisher_one_sided_greater(a_correct: int, b_correct: int, n: int) -> float:
    """One-sided Fisher p-value for A having higher accuracy than B.

    This treats the two arms as independent binomial samples with fixed margins.
    It is deliberately conservative for the preregistered public table; paired
    row-level tests can be added when both arms share complete row IDs.
    """

    a = int(a_correct)
    b = int(b_correct)
    total_correct = a + b
    n = int(n)
    observed = a
    max_a = min(n, total_correct)
    denominator = _log_choose(2 * n, total_correct)
    probs = [
        math.exp(_log_choose(n, x) + _log_choose(n, total_correct - x) - denominator)
        for x in range(observed, max_a + 1)
    ]
    return min(1.0, sum(probs))


def consecutive_depths_where_a_beats_b(
    a_counts: dict[str, int],
    b_counts: dict[str, int],
    *,
    rows_per_depth: int = PHASE_A_ROWS_PER_DEPTH,
    alpha: float = 0.05,
) -> list[int]:
    passing: list[int] = []
    for depth in sorted(set(a_counts) & set(b_counts), key=int):
        p_value = fisher_one_sided_greater(a_counts[depth], b_counts[depth], rows_per_depth)
        if int(a_counts[depth]) > int(b_counts[depth]) and p_value < alpha:
            passing.append(int(depth))
    best_run: list[int] = []
    current: list[int] = []
    for depth in passing:
        if not current or depth == current[-1] + 1:
            current.append(depth)
        else:
            if len(current) > len(best_run):
                best_run = current
            current = [depth]
    if len(current) > len(best_run):
        best_run = current
    return best_run


def surpass_gate(
    a_counts: dict[str, int],
    b_counts: dict[str, int],
    *,
    rows_per_depth: int = PHASE_A_ROWS_PER_DEPTH,
    alpha: float = 0.05,
    min_consecutive_depths: int = 3,
) -> dict[str, Any]:
    run = consecutive_depths_where_a_beats_b(
        a_counts,
        b_counts,
        rows_per_depth=rows_per_depth,
        alpha=alpha,
    )
    per_depth = {
        str(depth): {
            "a_correct": int(a_counts[str(depth)]),
            "b_correct": int(b_counts[str(depth)]),
            "p_one_sided_fisher": fisher_one_sided_greater(
                int(a_counts[str(depth)]),
                int(b_counts[str(depth)]),
                rows_per_depth,
            ),
            "a_beats_b": int(a_counts[str(depth)]) > int(b_counts[str(depth)]),
        }
        for depth in sorted(set(a_counts) & set(b_counts), key=int)
    }
    return {
        "rows_per_depth": rows_per_depth,
        "alpha": alpha,
        "min_consecutive_depths": min_consecutive_depths,
        "passing_consecutive_depths": run,
        "pass": len(run) >= min_consecutive_depths,
        "per_depth": per_depth,
    }


def phase_a_preregistration() -> dict[str, Any]:
    return {
        "kind": "stage5_phase_a_surpass_preregistration",
        "arms": {
            "A_looped": "support-8 dose-arm checkpoint, same-reader final-symbol metric",
            "B_dense_direct": "full-model Qwen2.5-0.5B AdamW direct final-symbol SFT, 4000 steps",
            "C_dense_scratchpad": "full-model Qwen2.5-0.5B AdamW serialized-orbit scratchpad SFT, 4000 steps",
            "D_dense_1p5b_direct_optional": "full-model Qwen2.5-1.5B AdamW direct-answer exchange-rate arm",
        },
        "train_distribution": {
            "n_symbols": 16,
            "support_depths": "1..8",
            "rows_per_depth": 256,
            "eval_frozen_set": "stage5_synthetic_depth_frozen_eval_v2_depth14",
        },
        "surpass_gate": {
            "primary": "A beats B at >=3 consecutive depths with one-sided Fisher p<0.05 per depth",
            "pooled_secondary": "pooled three-depth test may be reported, but does not replace row-level shape",
            "rows_per_depth": PHASE_A_ROWS_PER_DEPTH,
            "per_depth_mde_note": "about 14-15 percentage points at 80% power",
            "three_depth_mde_note": "about 8-9 points pooled",
        },
        "compute_ledger": {
            "looped_arm_context_growth": "zero text context growth across latent loops",
            "scratchpad_arm_context_growth": "linear serialized-orbit context growth",
            "flop_claim_policy": "do not claim raw FLOP advantage; report one parallel pass per latent step and sequence/context tradeoff",
            "optimizer_protocol": "all dense arms use full-model AdamW with FP32 parameters/moments and BF16 compute",
        },
    }
