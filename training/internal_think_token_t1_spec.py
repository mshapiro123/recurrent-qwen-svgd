"""Machine-readable draft for Phase T1 token-pathway halting."""

from __future__ import annotations

from typing import Any


TRAINED_DEPTHS = tuple(range(1, 9))
REHEARSAL_FRACTION = 0.30
CHAIN_MARGIN_POINTS = 0.03
SELF_HALTED_MARGIN_POINTS = 0.03
CONTROL_ACCURACY_FLOOR = 0.90


def phase_t1_draft() -> dict[str, Any]:
    return {
        "kind": "paper2_internal_think_token_phase_t1_preregistration",
        "status": "draft_not_locked",
        "training_authorized": False,
        "fresh_base_lineages": {
            "full_block": {
                "trainable": [
                    "recurrent_block",
                    "repaired_split_bridge",
                    "three_new_control_token_rows_only",
                ],
                "nonhalting_reference": {
                    "receipt": "outputs/stage5/stage5_phase_a_surpass_receipt_20260714/summary.json",
                    "arm": "A",
                    "checkpoint_sha256": "dc00f7b694ce32427eb13b0b85d365bc15e0c0317130bd22d4bbc3568544f71b",
                    "trained_depths_correct": 1005,
                    "trained_depths_total": 1024,
                },
            },
            "r16_adapter_bridge": {
                "trainable": [
                    "rank16_recurrent_block_lora",
                    "repaired_split_bridge",
                    "three_new_control_token_rows_only",
                ],
                "base_qwen_parameters_frozen": True,
                "nonhalting_reference": {
                    "receipt": "outputs/stage5/stage5_adapter_budget_arm_e_20260718/summary.json",
                    "arm": "E",
                    "checkpoint_sha256": "bffa8c4277ce82ae9f662db3243a21a50a08c4c041820c9d7506d8f250e82839",
                    "trained_depths_correct": 1021,
                    "trained_depths_total": 1024,
                },
            },
        },
        "data": {
            "family": "controlled synthetic transition family",
            "trained_depths": list(TRAINED_DEPTHS),
            "target_rule": "continue before exact row depth; stop at exact row depth",
            "control_position": "reserved per-loop readout position",
            "visible_control_tokens": False,
            "rehearsal_fraction": REHEARSAL_FRACTION,
            "rehearsal_definition": (
                "original per-loop chain supervision sampled independently of the "
                "70 percent control-target stream"
            ),
        },
        "proposed_training_budget": {
            "status": "requires_mark_lock_before_training",
            "curriculum": [
                {"stage": "primitive_depth1", "support": [1], "steps": 500, "lr": 2e-5},
                {"stage": "chain_depth_le2", "support": [1, 2], "steps": 2000, "lr": 1e-5},
                {"stage": "chain_depth_le4", "support": [1, 2, 3, 4], "steps": 4000, "lr": 1e-5},
                {"stage": "chain_depth_le8", "support": list(TRAINED_DEPTHS), "steps": 2000, "lr": 1e-5},
                {"stage": "chain_depth_le8_dose", "support": list(TRAINED_DEPTHS), "steps": 2000, "lr": 1e-5},
            ],
            "total_steps_per_lineage": 10500,
            "training_seeds": [0],
            "single_seed_limitation_required": True,
        },
        "evaluation": {
            "paired_forced_and_self_halted": True,
            "same_reader": "full-symbol, question-only, first completed response",
            "forced_depth": "row depth",
            "frozen_rows": "Phase A row IDs, 128 per depth",
            "frozen_row_id_sha256": "14482ca4d1b539172e4ccced6d870818c8658314b7f9680d0fb6e685b0317500",
            "reported_by_depth": True,
        },
        "gates": {
            "chain_accuracy": {
                "rule": "forced-depth chain accuracy within three points of lineage reference",
                "absolute_margin": CHAIN_MARGIN_POINTS,
            },
            "self_halted_accuracy": {
                "rule": "self-halted accuracy within three points of paired forced depth",
                "absolute_margin": SELF_HALTED_MARGIN_POINTS,
            },
            "control_selection": {
                "rule": "continue/stop selection accuracy at every trained depth",
                "minimum_each_depth": CONTROL_ACCURACY_FLOOR,
            },
            "causal_override": {
                "required": True,
                "stop_override": "forced stop at a model-continue transition terminates there",
                "continue_override": (
                    "forced continue at a model-stop transition executes at least one extra "
                    "loop within the registered maximum"
                ),
                "no_answer_only_proxy": True,
            },
            "all_four_required_for_positive": True,
        },
        "integrity": {
            "phase_t0_receipt_required": True,
            "step_zero_one_loop_identity_max_abs_diff": 1e-3,
            "base_hash_assertion": True,
            "tier1_canary_hard_stop": True,
            "visible_generation_mask_hard_assertion": True,
            "requested_executed_selected_loop_logging": True,
        },
        "do_not_claim": [
            "natural content-determined depth selection",
            "general adaptive computation",
            "efficiency improvement without measured compute",
            "token-pathway success if any gate fails",
            "multi-seed robustness",
        ],
        "phase_t2_authorized": False,
        "width_reopened": False,
    }


def validate_locked_phase_t1(payload: dict[str, Any]) -> None:
    if payload.get("status") != "locked_before_training":
        raise AssertionError("Phase T1 preregistration is not locked")
    if payload.get("training_authorized") is not True:
        raise AssertionError("Phase T1 training is not authorized")
    budget = payload.get("proposed_training_budget", {})
    if budget.get("status") != "locked_before_training":
        raise AssertionError("Phase T1 training budget is not locked")
    if int(budget.get("total_steps_per_lineage", 0)) <= 0:
        raise AssertionError("Phase T1 requires a positive per-lineage step budget")
    for lineage in ("full_block", "r16_adapter_bridge"):
        reference = payload["fresh_base_lineages"][lineage]["nonhalting_reference"]
        if not reference.get("receipt"):
            raise AssertionError(f"Phase T1 {lineage} lacks a non-halting reference")
    if payload["gates"]["all_four_required_for_positive"] is not True:
        raise AssertionError("Phase T1 must require all four registered gates")
