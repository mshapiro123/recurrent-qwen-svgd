"""Machine-readable draft for Phase T1 token-pathway halting."""

from __future__ import annotations

from copy import deepcopy
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
SELECTED_CONTROL_LOSS_LAMBDA = 0.5
SELECTED_STOP_TO_CONTINUE_RATIO = 1.0
LOCKED_DATE = "2026-07-24"


def phase_t1_draft() -> dict[str, Any]:
    label_counts = aggregate_control_label_counts(TRAINED_DEPTHS)
    default_weights = class_weights_from_ratio(
        stop_to_continue_ratio=3.5,
        continue_count=label_counts["continue"],
        stop_count=label_counts["stop"],
    )
    return {
        "kind": "paper2_internal_think_token_phase_t1_preregistration",
        "program_mode": "t1_lite_full_block_actuator_qualification",
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
        },
        "descoped_lineage": {
            "lineage": "r16_adapter_bridge",
            "reason": "T1-lite qualifies the actuator for D0, whose substrate trains the full recurrent block",
            "capacity_contrast_forfeited": True,
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
            "registered_t1_lineage": "full_block",
            "role_after_draft3_pivot": "loss_feasibility_and_hyperparameter_calibration_only",
            "matched_lineage_evidence": False,
            "transfer_rule": (
                "the selected lambda and ratio may be locked for T1-lite, but the "
                "full-block run must independently clear all four gates"
            ),
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
            "lineages": ["full_block"],
            "total_steps": 10500,
            "optimizer": "adamw",
            "batch_size": 1,
            "gradient_accumulation_steps": 1,
            "effective_batch_size": 1,
            "weight_decay": 0.0,
            "max_grad_norm": 0.5,
            "precision": {
                "base_and_trainable_recurrent_block": "bfloat16",
                "bridge_and_control_rows": "float32",
            },
            "bridge_prelude_lr_multiplier": {
                "primitive_depth1": 1.0,
                "all_chain_stages": 10.0,
            },
            "recipe_receipt": "outputs/stage5/stage5_support8_dose_arm_20260706_153028/summary.json",
            "recipe_config": (
                "outputs/stage5/stage5_support8_dose_arm_20260706_153028/"
                "chain_continuation_train_config.yaml"
            ),
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
        "d0_status": "preregistration_drafting_only",
        "paper_packaging": "deferred_until_post_d0_pilot",
        "width_reopened": False,
    }


def phase_t1_locked() -> dict[str, Any]:
    """Return the ratified Draft 4 contract used by registered T1-lite."""

    payload = deepcopy(phase_t1_draft())
    payload.update(
        {
            "status": "locked_before_training",
            "training_authorized": True,
            "locked_date": LOCKED_DATE,
            "governing_document": "docs/PHASE_T1_LITE_PREREGISTRATION_DRAFT4_20260724.md",
            "strategy_ratification": (
                "docs/PAPER2_T1_P0_CALIBRATION_STRATEGY_HANDOFF_20260724.md"
            ),
        }
    )
    payload["pilot_p0"].update(
        {
            "status": "complete_uncitable_prelock_pilot",
            "receipt": (
                "outputs/stage5/stage5_paper2_internal_token_t1_p0_letter_v2_20260724/"
                "summary.json"
            ),
            "selected_cell_id": "lambda0p5_ratio1",
            "selected_control_loss_lambda": SELECTED_CONTROL_LOSS_LAMBDA,
            "selected_stop_to_continue_ratio": SELECTED_STOP_TO_CONTINUE_RATIO,
            "selected_step_1500": {
                "stop_recall": 0.69140625,
                "continue_recall": 0.9877232142857143,
                "exact_selected_depth_accuracy": 0.6484375,
                "answer_accuracy": 0.58984375,
                "lambda_zero_answer_accuracy": 0.53125,
            },
            "citable": False,
        }
    )
    payload["loss"] = {
        "mechanism_answer_loss": "per_loop_chain_cross_entropy",
        "control_loss": "two_class_continue_stop_cross_entropy",
        "control_loss_lambda": SELECTED_CONTROL_LOSS_LAMBDA,
        "stop_to_continue_ratio": SELECTED_STOP_TO_CONTINUE_RATIO,
        "normalized_class_weights": {"continue": 1.0, "stop": 1.0},
        "transferred_from_unmatched_adapter_p0": True,
        "full_block_must_independently_clear_all_gates": True,
    }
    budget = payload["proposed_training_budget"]
    budget.update(
        {
            "status": "locked_before_training",
            "learning_rate_source": {
                "receipt": (
                    "outputs/stage5/stage5_support8_dose_arm_20260706_153028/summary.json"
                ),
                "config": (
                    "outputs/stage5/stage5_support8_dose_arm_20260706_153028/"
                    "chain_continuation_train_config.yaml"
                ),
                "primitive_depth1": 2e-5,
                "chain_stages": 1e-5,
            },
        }
    )
    payload["stage_boundary_liveness"] = {
        "descriptive_only": True,
        "may_change_registered_constants": False,
        "may_change_curriculum": False,
        "may_change_gates": False,
        "pilot_slice": payload["pilot_p0"]["receipt"],
        "restrict_recall_to_trained_depths": True,
        "boundaries": [
            {"step": 500, "completed_stage": "primitive_depth1", "trained_depths": [1]},
            {"step": 2500, "completed_stage": "chain_depth_le2", "trained_depths": [1, 2]},
            {
                "step": 6500,
                "completed_stage": "chain_depth_le4",
                "trained_depths": [1, 2, 3, 4],
            },
            {
                "step": 8500,
                "completed_stage": "chain_depth_le8",
                "trained_depths": list(TRAINED_DEPTHS),
            },
        ],
        "flat_control_loss_definition": {
            "statistic": "ordinary_least_squares_slope_over_all_stage_log_points",
            "flat_if_slope_greater_than_or_equal_to": -1e-5,
            "units": "control_loss_per_training_step",
        },
        "abort_rule": {
            "all_conditions_required": True,
            "conditions": [
                "control_loss_flat_over_the_completed_stage",
                "stop_recall_exactly_zero_on_pilot_rows_at_trained_depths",
            ],
            "action": "abort_for_diagnosis_write_receipts_attempt_not_consumed",
        },
    }
    frozen_source = (
        "outputs/stage5/stage5_synthetic_depth_frozen_eval_v2_depth14/"
        "data/test_chain_mcq.jsonl"
    )
    payload["evaluation"]["gated"].update(
        {
            "source": frozen_source,
            "depth_filter": [1, 8],
            "row_id_sha256": (
                "7aa673d046803c691226dd0a9950972ca141b4aaa89fcc118cc049b7e71fdcbe"
            ),
            "row_sha256": (
                "cacaf2ba6cf39424dc29c22f91f20f9edcedeeefe6200b59471898118c216faf"
            ),
        }
    )
    payload["evaluation"]["gated"].pop("frozen_row_id_sha256", None)
    payload["evaluation"]["extrapolation"].update(
        {
            "source": frozen_source,
            "depth_filter": [9, 14],
            "row_id_sha256": (
                "74c56235a033cc783963bc71584e2203b0b6936ba3996cf174616da3d1414b48"
            ),
            "row_sha256": (
                "82e4d687d65fb4901847b49ab7212888faea2707363251bd9789286216799575"
            ),
        }
    )
    payload["evaluation"]["calibration"].update(
        {
            "generator": "training.synthetic_depth_task.write_synthetic_depth_dataset",
            "seed": 2026072401,
            "n_symbols": 16,
            "id_prefix": "t1_calibration_",
            "split": "test_chain_mcq",
            "row_id_sha256": (
                "3175178e33b56406d9b7147cd4af5a76f3e47027a414b67a62d804991c7715c7"
            ),
            "row_sha256": (
                "9c4e7dacd30c720ed8b2ffba2770c39482e838544426219acce4760ee96e07ab"
            ),
            "disjoint_from_gated_by_generation_seed_and_id_prefix": True,
        }
    )
    payload["replication"] = {
        "primary_seed": 0,
        "seed_1_trigger": (
            "full_pass_or_near_threshold_as_defined_in_governing_document_or_"
            "strong_negative_boundary"
        ),
        "positive_headline_requires_seed_1_pass": True,
        "strong_negative_boundary_requires_seed_1_confirmation": True,
        "seed_1_runs_in_parallel_with_subsequent_authorized_phase": True,
    }
    return payload


def validate_locked_phase_t1(payload: dict[str, Any]) -> None:
    if payload.get("status") != "locked_before_training":
        raise AssertionError("Phase T1 preregistration is not locked")
    if payload.get("training_authorized") is not True:
        raise AssertionError("Phase T1 training is not authorized")
    budget = payload.get("proposed_training_budget", {})
    if budget.get("status") != "locked_before_training":
        raise AssertionError("Phase T1 training budget is not locked")
    if int(budget.get("total_steps", 0)) <= 0:
        raise AssertionError("Phase T1-lite requires a positive step budget")
    if budget.get("lineages") != ["full_block"]:
        raise AssertionError("Phase T1-lite authorizes only the full-block lineage")
    for lineage in ("full_block",):
        reference = payload["fresh_base_lineages"][lineage]["nonhalting_reference"]
        if not reference.get("receipt"):
            raise AssertionError(f"Phase T1 {lineage} lacks a non-halting reference")
    if payload["gates"]["all_four_required_for_positive"] is not True:
        raise AssertionError("Phase T1 must require all four registered gates")
    loss = payload.get("loss", {})
    if float(loss.get("control_loss_lambda", -1)) != SELECTED_CONTROL_LOSS_LAMBDA:
        raise AssertionError("Phase T1 control-loss lambda differs from the P0 lock")
    if float(loss.get("stop_to_continue_ratio", -1)) != SELECTED_STOP_TO_CONTINUE_RATIO:
        raise AssertionError("Phase T1 class-weight ratio differs from the P0 lock")
    liveness = payload.get("stage_boundary_liveness", {})
    if [row.get("step") for row in liveness.get("boundaries", [])] != [500, 2500, 6500, 8500]:
        raise AssertionError("Phase T1 stage-boundary liveness schedule drifted")
    if liveness.get("may_change_registered_constants") is not False:
        raise AssertionError("Phase T1 liveness readouts cannot tune registered constants")
    for split in ("gated", "calibration", "extrapolation"):
        manifest = payload.get("evaluation", {}).get(split, {})
        if not manifest.get("row_id_sha256") or not manifest.get("row_sha256"):
            raise AssertionError(f"Phase T1 {split} manifest is not hash-locked")
