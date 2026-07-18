"""Locked specification and paired scoring for adapter-budget Arm E."""

from __future__ import annotations

import math
from typing import Any, Iterable


ARM_A_COUNTS = {
    1: 128,
    2: 127,
    3: 126,
    4: 125,
    5: 127,
    6: 126,
    7: 124,
    8: 122,
    9: 116,
    10: 113,
    11: 97,
    12: 87,
    13: 57,
    14: 31,
}
ROWS_PER_DEPTH = 128
TOTAL_ROWS = 14 * ROWS_PER_DEPTH
ARM_A_POOLED_ACCURACY = sum(ARM_A_COUNTS.values()) / TOTAL_ROWS
ARM_C_POOLED_ACCURACY = 0.531
PARITY_POOLED_MARGIN = 0.03
PARITY_MAX_DEPTH_DEFICIT = 8


def locked_spec() -> dict[str, Any]:
    """Return the immutable Arm-E protocol reconstructed from Arm A."""

    common = {
        "batch_size": 1,
        "gradient_accumulation_steps": 1,
        "weight_decay": 0.0,
        "max_grad_norm": 0.5,
        "optimizer": "adamw",
    }
    stages = [
        {
            "name": "primitive_depth1",
            "max_loops": 1,
            "max_steps": 500,
            "learning_rate": 2e-5,
            "bridge_prelude_lr_multiplier": 1.0,
            "training_seed": 0,
            "dataset_seed": 20260701,
            "rows_per_depth": 256,
            "supervision": "mcq_option_text",
            **common,
        },
        {
            "name": "chain_depth_le2",
            "max_loops": 2,
            "max_steps": 2000,
            "learning_rate": 1e-5,
            "bridge_prelude_lr_multiplier": 10.0,
            "training_seed": 0,
            "dataset_seed": 20260702,
            "rows_per_depth": 256,
            "supervision": "per_loop_labels",
            **common,
        },
        {
            "name": "chain_depth_le4",
            "max_loops": 4,
            "max_steps": 4000,
            "learning_rate": 1e-5,
            "bridge_prelude_lr_multiplier": 10.0,
            "training_seed": 0,
            "dataset_seed": 20260702,
            "rows_per_depth": 256,
            "supervision": "per_loop_labels",
            **common,
        },
        {
            "name": "chain_depth_le8",
            "max_loops": 8,
            "max_steps": 2000,
            "learning_rate": 1e-5,
            "bridge_prelude_lr_multiplier": 10.0,
            "training_seed": 0,
            "dataset_seed": 20260704,
            "rows_per_depth": 256,
            "supervision": "per_loop_labels",
            **common,
        },
        {
            "name": "chain_depth_le8_dose",
            "max_loops": 8,
            "max_steps": 2000,
            "learning_rate": 1e-5,
            "bridge_prelude_lr_multiplier": 10.0,
            "training_seed": 0,
            "dataset_seed": 20260704,
            "rows_per_depth": 256,
            "supervision": "per_loop_labels",
            **common,
        },
    ]
    return {
        "kind": "adapter_budget_arm_e_preregistration",
        "arm": {"name": "E", "rank": 16, "alpha": 32},
        "initialization": "fresh_base_qwen_surgery",
        "model": "Qwen/Qwen2.5-0.5B-Instruct",
        "layer_split": "6,18",
        "bridge_projection_mode": "split",
        "optimizer": "adamw",
        "trainable_set": "rank16_lora_all_recurrent_projections_plus_repaired_bridge",
        "frozen_set": "all_pretrained_qwen_parameters",
        "compute_policy": (
            "fixed_stage_loop_cap_with_row_capped_targets_matching_historical_Arm_A; "
            "row_specific_forward_compute_would_introduce_a_second_variable"
        ),
        "stages": stages,
        "total_optimizer_steps": sum(stage["max_steps"] for stage in stages),
        "interval": {
            "checkpoint_every": 1000,
            "phase_a_smoke_rows": 128,
            "tier1_canary_every": 1000,
        },
        "final_eval": {
            "rows": TOTAL_ROWS,
            "rows_per_depth": ROWS_PER_DEPTH,
            "depths": list(range(1, 15)),
            "reader": "same_reader_full_symbol_at_loop_equal_depth",
        },
        "gates": {
            "parity_pooled_absolute_margin": PARITY_POOLED_MARGIN,
            "parity_max_depth_correct_deficit": PARITY_MAX_DEPTH_DEFICIT,
            "catastrophic_pooled_floor": ARM_C_POOLED_ACCURACY,
            "paired_test": "exact_paired_sign_mcnemar",
        },
        "reference": {
            "arm_a_correct": sum(ARM_A_COUNTS.values()),
            "arm_a_total": TOTAL_ROWS,
            "arm_a_accuracy": ARM_A_POOLED_ACCURACY,
            "arm_a_by_depth": dict(ARM_A_COUNTS),
            "arm_a_tail_correct_depth11_14": sum(ARM_A_COUNTS[d] for d in range(11, 15)),
        },
        "claim_boundaries": [
            "no_budget_independent_claim_without_parity",
            "no_capacity_limited_claim_without_paired_depth_evidence",
            "no_rank_sweep",
            "not_a_keeper_lineage_run",
        ],
    }


def _binomial_upper_tail(successes: int, trials: int) -> float:
    if trials <= 0:
        return 1.0
    return sum(math.comb(trials, value) for value in range(successes, trials + 1)) / (2**trials)


def paired_binary_test(left: Iterable[bool], right: Iterable[bool]) -> dict[str, Any]:
    pairs = [(bool(a), bool(b)) for a, b in zip(left, right, strict=True)]
    helped = sum(a and not b for a, b in pairs)
    hurt = sum(b and not a for a, b in pairs)
    tied = len(pairs) - helped - hurt
    discordant = helped + hurt
    smaller = min(helped, hurt)
    lower = (
        sum(math.comb(discordant, value) for value in range(smaller + 1)) / (2**discordant)
        if discordant
        else 0.5
    )
    return {
        "helped": helped,
        "hurt": hurt,
        "tied": tied,
        "discordant": discordant,
        "net_correct": helped - hurt,
        "one_sided_p": _binomial_upper_tail(helped, discordant),
        "two_sided_p": 1.0 if not discordant else min(1.0, 2.0 * lower),
        "test": "exact_paired_sign_mcnemar",
    }


def _hit(row: dict[str, Any]) -> bool:
    for key in ("same_reader_final_hit", "hit", "correct"):
        if key in row:
            return bool(row[key])
    raise ValueError(f"Row has no recognized correctness field: {row.get('id')}")


def _deficit_shape(depth_deltas: dict[str, int]) -> str:
    early_loss = sum(max(0, -depth_deltas[str(depth)]) for depth in range(1, 9))
    tail_loss = sum(max(0, -depth_deltas[str(depth)]) for depth in range(9, 15))
    if not early_loss and not tail_loss:
        return "none"
    if tail_loss and (not early_loss or tail_loss >= 2 * early_loss):
        return "tail_concentrated"
    if early_loss and tail_loss:
        return "distributed"
    return "early_concentrated"


def score_adapter_budget_arm(
    arm_a_rows: list[dict[str, Any]],
    arm_e_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    """Pair Arm E against Arm A on the immutable 1,792 Phase-A rows."""

    left = {str(row["id"]): row for row in arm_a_rows}
    right = {str(row["id"]): row for row in arm_e_rows}
    if len(left) != len(arm_a_rows) or len(right) != len(arm_e_rows):
        raise ValueError("Arm A and Arm E rows must each have unique IDs")
    if set(left) != set(right):
        raise ValueError("Arm A and Arm E must contain identical row IDs")
    if len(left) != TOTAL_ROWS:
        raise ValueError(f"Expected {TOTAL_ROWS} frozen Phase-A rows, got {len(left)}")

    ordered_ids = sorted(left, key=lambda row_id: (int(left[row_id]["depth"]), row_id))
    depths = {depth: [] for depth in range(1, 15)}
    merged: list[dict[str, Any]] = []
    for row_id in ordered_ids:
        a_row = left[row_id]
        e_row = right[row_id]
        depth = int(a_row["depth"])
        if depth != int(e_row["depth"]):
            raise ValueError(f"Depth mismatch for {row_id}")
        if depth not in depths:
            raise ValueError(f"Unexpected depth {depth} for {row_id}")
        row = {"id": row_id, "depth": depth, "A": _hit(a_row), "E": _hit(e_row)}
        depths[depth].append(row)
        merged.append(row)
    if any(len(rows) != ROWS_PER_DEPTH for rows in depths.values()):
        raise ValueError(
            "Every Phase-A depth must contain 128 rows: "
            f"{ {depth: len(rows) for depth, rows in depths.items()} }"
        )

    a_counts = {str(depth): sum(row["A"] for row in rows) for depth, rows in depths.items()}
    e_counts = {str(depth): sum(row["E"] for row in rows) for depth, rows in depths.items()}
    if {int(depth): count for depth, count in a_counts.items()} != ARM_A_COUNTS:
        raise ValueError(
            "Arm A rows do not match the locked 1506/1792 receipt: "
            f"observed={a_counts}"
        )
    depth_deltas = {
        str(depth): e_counts[str(depth)] - a_counts[str(depth)]
        for depth in range(1, 15)
    }
    paired_by_depth = {
        str(depth): paired_binary_test(
            [row["E"] for row in rows],
            [row["A"] for row in rows],
        )
        for depth, rows in depths.items()
    }
    a_correct = sum(a_counts.values())
    e_correct = sum(e_counts.values())
    a_accuracy = a_correct / TOTAL_ROWS
    e_accuracy = e_correct / TOTAL_ROWS
    pooled_within = abs(e_accuracy - a_accuracy) <= PARITY_POOLED_MARGIN
    pooled_meets_floor = e_accuracy >= a_accuracy - PARITY_POOLED_MARGIN
    depth_floor = all(delta >= -PARITY_MAX_DEPTH_DEFICIT for delta in depth_deltas.values())
    above_arm_c = e_accuracy >= ARM_C_POOLED_ACCURACY

    if not above_arm_c:
        verdict = "catastrophic_training_recipe_alarm"
    elif pooled_meets_floor and depth_floor:
        verdict = "parity"
    else:
        verdict = "deficit"
    return {
        "kind": "adapter_budget_depth_profile",
        "rows": TOTAL_ROWS,
        "row_ids_match": True,
        "arm_a": {"correct": a_correct, "total": TOTAL_ROWS, "accuracy": a_accuracy},
        "arm_e": {"correct": e_correct, "total": TOTAL_ROWS, "accuracy": e_accuracy},
        "counts_by_depth": {"A": a_counts, "E": e_counts},
        "delta_correct_by_depth": depth_deltas,
        "paired_pooled": paired_binary_test(
            [row["E"] for row in merged],
            [row["A"] for row in merged],
        ),
        "paired_by_depth": paired_by_depth,
        "gates": {
            "pooled_within_three_points": pooled_within,
            "pooled_meets_or_exceeds_parity_floor": pooled_meets_floor,
            "pooled_absolute_delta": abs(e_accuracy - a_accuracy),
            "pooled_margin": PARITY_POOLED_MARGIN,
            "no_depth_worse_by_more_than_8": depth_floor,
            "max_depth_deficit": PARITY_MAX_DEPTH_DEFICIT,
            "above_arm_c_floor": above_arm_c,
            "arm_c_floor": ARM_C_POOLED_ACCURACY,
        },
        "deficit_shape": _deficit_shape(depth_deltas),
        "verdict": verdict,
        "required_followup": (
            [
                "verify_step_zero_identity",
                "verify_base_hash_unchanged",
                "verify_weighted_label_dose",
            ]
            if verdict == "catastrophic_training_recipe_alarm"
            else []
        ),
    }
