"""Read-only DC1 follow-up scoring and provisional Stage A contracts."""

from __future__ import annotations

from typing import Any, Iterable

import torch


PRE_D0_CHECKPOINT_SHA256 = (
    "93d2e5f9a941bbe79a0b2fc3f9bf43d582bf054990c14b1a93ff67024140062d"
)
POST_D0_CHECKPOINT_SHA256 = (
    "8245cabfe7639dcd442c19e03496623b9d59eec31b39e253e08fd6f78b1086cf"
)
SCALE_ABOVE_RAW = (1.5, 2.0)
STAGE_A_NET_CI_FLOOR_FRACTION = -0.0025
STAGE_A_HURT_REDUCTION_FRACTION = 0.50
STAGE_A_RESOURCE_PROPOSAL = {
    "status": "proposal_for_preregistration_not_training_authority",
    "hardware": "NVIDIA A100-SXM4-80GB",
    "precision": "full_fp32_model_feedback_boundary_gradients_and_optimizer",
    "step_ceiling": 2000,
    "microbatch_rows": 1,
    "gradient_accumulation_steps": 1,
    "effective_batch_rows": 1,
    "maximum_sequence_length": 512,
    "optimizer": "AdamW",
    "learning_rate": 1e-4,
    "weight_decay": 0.0,
    "gradient_clip_norm": 0.5,
    "passive_checkpoint_steps": [500, 1000, 1500, 2000],
    "wall_clock_estimate_hours": [2.0, 4.0],
    "estimate_basis": (
        "single-row full-recompute backward through the frozen 0.5B graph in fp32; "
        "only the 802816-parameter horizontal delta matrix is optimized"
    ),
}


def transition_ledger(
    predictions: torch.Tensor,
    teacher: torch.Tensor,
    *,
    before_depth: int,
    after_depth: int,
    split_mask: torch.Tensor | None = None,
) -> dict[str, Any]:
    """Score one forced-depth transition with an explicit population split."""

    if predictions.ndim != 2 or teacher.ndim != 1 or len(predictions) != len(teacher):
        raise ValueError("predictions [positions, depths] and teacher [positions] must align")
    before_index = int(before_depth) - 1
    after_index = int(after_depth) - 1
    if before_index < 0 or after_index < 0 or max(before_index, after_index) >= predictions.shape[1]:
        raise ValueError("requested depths are outside the prediction grid")
    before = predictions[:, before_index].eq(teacher)
    after = predictions[:, after_index].eq(teacher)
    if split_mask is None:
        split_mask = predictions[:, 0].eq(teacher)
    if split_mask.shape != teacher.shape or split_mask.dtype != torch.bool:
        raise ValueError("split_mask must be an aligned boolean tensor")

    def score(mask: torch.Tensor) -> dict[str, Any]:
        count = int(mask.sum())
        helps = int((mask & ~before & after).sum())
        hurts = int((mask & before & ~after).sum())
        before_correct = int((mask & before).sum())
        after_correct = int((mask & after).sum())
        return {
            "positions": count,
            "helps": helps,
            "hurts": hurts,
            "neutral": count - helps - hurts,
            "net_correct_delta": helps - hurts,
            "before_correct": before_correct,
            "after_correct": after_correct,
            "before_accuracy": before_correct / count if count else None,
            "after_accuracy": after_correct / count if count else None,
            "harm_to_help_ratio": hurts / helps if helps else None,
        }

    return {
        "transition": f"{before_depth}_to_{after_depth}",
        "split_definition": "depth-1 prediction agrees with the cached 7B greedy token",
        "all_positions": score(torch.ones_like(split_mask)),
        "depth1_accepted": score(split_mask),
        "depth1_rejected": score(~split_mask),
    }


def floor_payload_has_all_positions(payload: dict[str, Any], *, rejected_positions: int) -> bool:
    rows = payload.get("all_position_rows")
    if not isinstance(rows, list) or not rows:
        return False
    return len(rows) > int(rejected_positions) and all(
        isinstance(row, dict)
        and isinstance(row.get("predictions"), list)
        and len(row["predictions"]) >= 2
        for row in rows
    )


def summarize_values(values: torch.Tensor) -> dict[str, Any]:
    work = values.detach().float().reshape(-1)
    if not work.numel():
        return {"count": 0, "mean": None, "median": None, "q25": None, "q75": None}
    quantiles = torch.quantile(work, torch.tensor([0.25, 0.5, 0.75]))
    return {
        "count": int(work.numel()),
        "mean": float(work.mean()),
        "median": float(quantiles[1]),
        "q25": float(quantiles[0]),
        "q75": float(quantiles[2]),
    }


def scale_response_reading(rows: Iterable[dict[str, Any]]) -> dict[str, Any]:
    ordered = list(rows)
    if len(ordered) < 3:
        raise ValueError("scale-response reading needs at least three scales")
    trough = min(ordered, key=lambda row: float(row["transition"]["after_accuracy"]))
    crossover = min(
        ordered,
        key=lambda row: abs(
            float(row["cosine_to_fed"]["mean"])
            - float(row["cosine_to_k0"]["mean"])
        ),
    )
    fed_means = [float(row["cosine_to_fed"]["mean"]) for row in ordered]
    rising_pairs = sum(right >= left for left, right in zip(fed_means, fed_means[1:]))
    return {
        "accuracy_trough_label": str(trough["label"]),
        "cosine_crossover_label": str(crossover["label"]),
        "trough_coincides_with_nearest_measured_crossover": (
            str(trough["label"]) == str(crossover["label"])
        ),
        "fed_cosine_non_decreasing_pairs": rising_pairs,
        "fed_cosine_adjacent_pairs": len(fed_means) - 1,
        "status": "descriptive_non_gating",
    }


def score_stage_a_verdict(
    *,
    trained_helps: int,
    trained_hurts: int,
    untrained_helps: int,
    untrained_hurts: int,
    positions: int,
    row_cluster_bootstrap_net_ci95_lower: float,
) -> dict[str, Any]:
    """Encode the strategy-locked Stage A bands before EVAL-C is read."""

    if min(trained_helps, trained_hurts, untrained_helps, untrained_hurts, positions) < 0:
        raise ValueError("Stage A counts must be nonnegative")
    if positions < 1:
        raise ValueError("Stage A requires at least one scored position")
    net = int(trained_helps) - int(trained_hurts)
    lower_fraction = float(row_cluster_bootstrap_net_ci95_lower) / int(positions)
    qualifies = net >= 0 and lower_fraction >= STAGE_A_NET_CI_FLOOR_FRACTION
    partial = (
        not qualifies
        and net < 0
        and int(trained_hurts) <= STAGE_A_HURT_REDUCTION_FRACTION * int(untrained_hurts)
        and int(trained_helps) >= int(untrained_helps)
    )
    verdict = (
        "qualifies"
        if qualifies
        else "partial_domestication"
        if partial
        else "no_material_improvement"
    )
    return {
        "verdict": verdict,
        "trained_net_correct_delta": net,
        "bootstrap_lower_fraction": lower_fraction,
        "qualification_floor_fraction": STAGE_A_NET_CI_FLOOR_FRACTION,
        "hurt_reduction_required_fraction": STAGE_A_HURT_REDUCTION_FRACTION,
    }
