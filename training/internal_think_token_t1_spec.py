"""Machine-readable draft for Phase T1 token-pathway halting."""

from __future__ import annotations

from typing import Any

from training.internal_think_token_t1 import (
    PILOT_STEPS,
    aggregate_control_label_counts,
    class_weights_from_ratio,
    pilot_grid,
)


TRAINED_DEPTHS = tuple(range(1, 9))
REHEARSAL_FRACTION = 0.30
CHAIN_MARGIN_POINTS = 0.03
SELF_HALTED_MARGIN_POINTS = 0.03
CONTROL_ACCURACY_FLOOR = 0.90
FULL_BLOCK_GATE1_FLOOR = 975
ADAPTER_GATE1_FLOOR = 991


def phase_t1_draft() -> dict[str, Any]:
    label_counts = aggregate_control_label_counts(TRAINED_DEPTHS)
    default_weights = class_weights_from_ratio(
        stop_to_continue_ratio=3.5,
        continue_count=label_counts["continue"],
        stop_count=label_counts["stop"],
    )
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
        "pilot_p0": {
            "status": "authorized_uncitable_prelock_pilot",
            "authorized_before_lock": True,
            "registered_t1_training": False,
            "lineage": "r16_adapter_bridge",
            "seed": 9999,
            "steps_per_cell": 1500,
            "evaluation_steps": list(PILOT_STEPS),
            "pilot_rows": 256,
            "pilot_rows_per_depth": 32,
            "cells": [cell.to_dict() for cell in pilot_grid()],
            "default_label_counts_uniform_depths": label_counts,
            "default_normalized_class_weights": {
                "continue": default_weights[0],
                "stop": default_weights[1],
            },
            "selection": {
                "minimum_stop_recall": 0.60,
                "minimum_continue_recall": 0.60,
                "objective": "smallest_answer_accuracy_drop_against_lambda_zero",
                "tie_break": "toward_lambda_1_then_ratio_3p5",
                "no_qualifying_cell": "reassess_openly_before_lock_no_silent_extension",
            },
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
            "optimizer": "adamw",
            "batch_size": 1,
            "gradient_accumulation_steps": 1,
            "effective_batch_size": 1,
            "weight_decay": 0.0,
            "max_grad_norm": 0.5,
            "precision": {"base": "bfloat16", "trainable_adapters": "float32"},
            "bridge_prelude_lr_multiplier": {
                "primitive_depth1": 1.0,
                "all_chain_stages": 10.0,
            },
            "recipe_receipt": (
                "outputs/stage5/stage5_adapter_budget_arm_e_20260718/"
                "preregistration.json"
            ),
            "recipe_handoff": "docs/ARM_E_ADAPTER_BUDGET_PUBLICATION_HANDOFF_20260718.md",
            "training_seeds": [0],
            "single_seed_limitation_required": True,
        },
        "evaluation": {
            "paired_forced_and_self_halted": True,
            "same_reader": "full-symbol, question-only, first completed response",
            "forced_depth": "row depth",
            "gated": {
                "depths": list(TRAINED_DEPTHS),
                "rows_per_depth": 128,
                "rows": 1024,
                "frozen_row_id_sha256": "14482ca4d1b539172e4ccced6d870818c8658314b7f9680d0fb6e685b0317500",
            },
            "extrapolation": {
                "depths": list(range(9, 15)),
                "rows_per_depth": 128,
                "rows": 768,
                "gated": False,
            },
            "calibration": {
                "depths": list(TRAINED_DEPTHS),
                "rows_per_depth": 64,
                "rows": 512,
                "use": "fit_training_free_baseline_thresholds_only",
            },
            "self_halt_max_loops": {"gated": 12, "extrapolation": 16},
            "exhaustion_scored_as_selection_failure": True,
            "reported_by_depth": True,
        },
        "gates": {
            "chain_accuracy": {
                "rule": "forced-depth chain accuracy within three points of lineage reference",
                "absolute_margin": CHAIN_MARGIN_POINTS,
                "full_block_minimum_correct": FULL_BLOCK_GATE1_FLOOR,
                "adapter_minimum_correct": ADAPTER_GATE1_FLOOR,
                "total": 1024,
            },
            "self_halted_accuracy": {
                "rule": "self-halted accuracy within three points of paired forced depth",
                "absolute_margin": SELF_HALTED_MARGIN_POINTS,
            },
            "control_selection": {
                "metric": "row_level_exact_selected_depth",
                "rule": "selected stop loop equals stated required depth",
                "minimum_each_depth": CONTROL_ACCURACY_FLOOR,
                "minimum_correct_each_depth": 115,
                "rows_each_depth": 128,
                "minimum_correct_pooled": 922,
                "rows_pooled": 1024,
                "transition_micro_accuracy_is_gate": False,
                "always_continue_transition_micro_accuracy": 28 / 36,
            },
            "causal_override": {
                "required": True,
                "intervention_level": "control_logits_not_loop_counter_or_max_loops",
                "stop_override": "force stop at every k from 1 through row depth",
                "continue_override": "force continue at required depth and execute depth plus one",
                "forced_stop_executions": 4608,
                "forced_continue_executions": 1024,
                "required_exact_agreement": 5632,
                "implementation_failure_policy": "fix actuator and rerun gate4_only",
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
