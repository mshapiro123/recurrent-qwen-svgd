"""Fail-closed preregistration scaffold for speculative-depth pilot D0."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch


UNRESOLVED = "RESOLVE_BEFORE_LOCK"


def d0_draft() -> dict[str, Any]:
    return {
        "kind": "paper2_speculative_decoding_depth_d0_preregistration",
        "status": "draft_not_locked",
        "training_authorized": False,
        "launch_target_exists": False,
        "build_validation_target_exists": True,
        "substrate_family": "Qwen",
        "dependency": {
            "t1_lite_verdict_required": True,
            "d0_lock_required": True,
            "automatic_launch_from_t1": False,
        },
        "teacher_ladder": {
            "teacher_checkpoints": UNRESOLVED,
            "same_tokenizer_assertion": True,
            "checkpoint_sha256": UNRESOLVED,
        },
        "labeling": {
            "corpus_and_license": UNRESOLVED,
            "split_and_manifest_sha256": UNRESOLVED,
            "candidate_generation": UNRESOLVED,
            "answer_verification": UNRESOLVED,
            "acceptance_rule": UNRESOLVED,
            "disagreement_to_depth_mapping": UNRESOLVED,
            "mapping_locked_before_labels_are_scored": True,
        },
        "distillation": {
            "objective": UNRESOLVED,
            "trainable_parameter_set": UNRESOLVED,
            "training_budget": UNRESOLVED,
            "seeds": UNRESOLVED,
        },
        "evaluation": {
            "depth_recoverable_fraction": UNRESOLVED,
            "acceptance_rate_uplift": UNRESOLVED,
            "matched_baseline": UNRESOLVED,
            "natural_surface_non_degradation_guardrail": UNRESOLVED,
            "frozen_sets_and_hashes": UNRESOLVED,
        },
        "decision": {
            "success_thresholds": UNRESOLVED,
            "stopping_rules": UNRESOLVED,
            "failure_interpretations": UNRESOLVED,
            "paper_packaging_deferred_until_result": True,
        },
        "do_not_claim": [
            "speculative disagreement yields useful depth labels",
            "depth is recoverable on natural data",
            "acceptance rate improves",
            "natural reasoning improves",
        ],
    }


@dataclass(frozen=True)
class D0ExecutionPolicy:
    """Current authorization boundary: code construction only."""

    labeling_gpu_authorized: bool = False
    training_authorized: bool = False

    def assert_allowed(self, *, labeling: bool, training: bool) -> None:
        if labeling and not self.labeling_gpu_authorized:
            raise RuntimeError("D0 labeling is not authorized")
        if training and not self.training_authorized:
            raise RuntimeError("D0 training is not authorized")


def build_only_contract() -> dict[str, Any]:
    """Return the implementable D0 contract without resolving preregistration choices."""

    return {
        "kind": "paper2_speculative_depth_d0_build_contract",
        "status": "build_only_no_labeling_no_training",
        "labeling_gpu_authorized": False,
        "training_authorized": False,
        "preregistration_locked": False,
        "allowed": [
            "schema_and_manifest_code",
            "exact_match_and_signal_scorers",
            "depth_mapping_logic",
            "teacher_added_token_masking",
            "synthetic_unit_fixtures",
            "cpu_dry_run_receipt",
        ],
        "forbidden": [
            "teacher_forward_labeling",
            "corpus_download_or_selection_from_probe_results",
            "gpu_labeling",
            "optimizer_construction",
            "training_step",
            "checkpoint_write",
        ],
        "models": {
            "drafter": "T1-lite raw endpoint selected only after T1-lite-R review",
            "teacher": "unresolved at markup: Qwen2.5-7B, Qwen2.5-14B, or calibration-only dual teacher",
            "ema_endpoints_excluded": True,
        },
        "corpus": {
            "strata": ["post_cutoff_fineweb_edu", "stack_v2_permissive_license"],
            "default_mix": {"general": 0.5, "code": 0.5},
            "partitions": ["label_train", "calibration", "evaluation"],
            "document_disjoint": True,
            "full_preimage_or_answer_truth_claim": False,
        },
        "labels": {
            "primary": "drafter_greedy_token_equals_teacher_greedy_token",
            "context": "teacher_forced_true_prefix",
            "signals": [
                "drafter_token_rank_under_teacher",
                "drafter_token_probability_under_teacher",
                "teacher_to_drafter_kl",
                "teacher_entropy",
                "rejection_run_length",
            ],
        },
        "calibration": {
            "forced_depths": [1, 2, 3, 4],
            "severity_bins": "teacher_to_drafter_kl_quartiles",
            "graded_if_depth4_minus_depth1_at_least": 0.02,
            "graded_minimum_bins": 2,
            "plateau_within_depth4": 0.01,
            "flat_fallback": "first_loop_matching_teacher_else_depth4",
            "run_length_tail_excluded_above": 8,
            "primary_fit": "monotone_isotonic",
            "comparison_fits": ["linear_run_length", "linear_log_kl", "saturating"],
        },
        "readouts": {
            "depth_recoverable_fraction": "self_halted_match_rate_minus_loop1_match_rate_on_rejections",
            "acceptance_rate_uplift": True,
            "expected_loops_per_token": True,
            "compute_per_accepted_token": True,
            "per_stratum": True,
        },
        "interpretation_bands": {
            "minimal_below": 0.02,
            "partial_from": 0.02,
            "partial_below": 0.10,
            "strong_from": 0.10,
        },
    }


def dynamic_depth_target(loop_matches_teacher: list[bool], *, max_depth: int = 4) -> int:
    if max_depth < 1:
        raise ValueError("max_depth must be positive")
    for index, matched in enumerate(loop_matches_teacher[:max_depth], start=1):
        if bool(matched):
            return index
    return int(max_depth)


def calibrated_depth_targets(
    bin_curves: dict[str, list[float]],
    *,
    minimum_gain: float = 0.02,
    plateau_tolerance: float = 0.01,
    minimum_graded_bins: int = 2,
) -> dict[str, Any]:
    if not bin_curves or any(len(values) != 4 for values in bin_curves.values()):
        raise ValueError("D0 calibration requires four forced-depth values per severity bin")
    graded_bins = [
        name
        for name, values in bin_curves.items()
        if float(values[3]) - float(values[0]) + 1e-12 >= float(minimum_gain)
    ]
    if len(graded_bins) < int(minimum_graded_bins):
        return {
            "branch": "flat_floor_dynamic_targets",
            "graded_bins": graded_bins,
            "targets": None,
        }
    targets: dict[str, int] = {}
    for name, values in bin_curves.items():
        if name not in graded_bins:
            targets[name] = 1
            continue
        depth4 = float(values[3])
        targets[name] = next(
            depth
            for depth, value in enumerate(values, start=1)
            if depth4 - float(value) <= float(plateau_tolerance) + 1e-12
        )
    return {
        "branch": "graded_floor_curve",
        "graded_bins": graded_bins,
        "targets": targets,
    }


def mask_teacher_added_token_probabilities(
    probabilities: torch.Tensor,
    *,
    added_token_ids: list[int] | tuple[int, ...],
) -> torch.Tensor:
    masked = probabilities.clone()
    masked[..., list(added_token_ids)] = 0
    denominator = masked.sum(dim=-1, keepdim=True)
    if bool((denominator <= 0).any()):
        raise ValueError("teacher added-token mask removed all probability mass")
    return masked / denominator


def depth_recoverable_fraction(
    *,
    loop1_matches: int,
    self_halted_matches: int,
    rejected_positions: int,
) -> dict[str, Any]:
    total = int(rejected_positions)
    if total <= 0:
        raise ValueError("rejected_positions must be positive")
    if not 0 <= int(loop1_matches) <= total or not 0 <= int(self_halted_matches) <= total:
        raise ValueError("match counts must lie within rejected_positions")
    loop1_rate = int(loop1_matches) / total
    self_halted_rate = int(self_halted_matches) / total
    return {
        "rejected_positions": total,
        "loop1_matches": int(loop1_matches),
        "loop1_match_rate": loop1_rate,
        "self_halted_matches": int(self_halted_matches),
        "self_halted_match_rate": self_halted_rate,
        "depth_recoverable_fraction": self_halted_rate - loop1_rate,
    }


def unresolved_paths(payload: Any, prefix: str = "") -> list[str]:
    paths: list[str] = []
    if isinstance(payload, dict):
        for key, value in payload.items():
            path = f"{prefix}.{key}" if prefix else str(key)
            paths.extend(unresolved_paths(value, path))
    elif isinstance(payload, list):
        for index, value in enumerate(payload):
            paths.extend(unresolved_paths(value, f"{prefix}[{index}]"))
    elif payload == UNRESOLVED:
        paths.append(prefix)
    return paths


def validate_locked_d0(payload: dict[str, Any]) -> None:
    if payload.get("status") != "locked_before_training":
        raise AssertionError("D0 preregistration is not locked")
    if payload.get("training_authorized") is not True:
        raise AssertionError("D0 training is not authorized")
    unresolved = unresolved_paths(payload)
    if unresolved:
        raise AssertionError(f"D0 preregistration has unresolved fields: {unresolved}")
    dependency = payload.get("dependency", {})
    if dependency.get("t1_lite_verdict_required") is not True:
        raise AssertionError("D0 lock must retain the T1-lite verdict dependency")
    if dependency.get("d0_lock_required") is not True:
        raise AssertionError("D0 lock dependency is missing")
