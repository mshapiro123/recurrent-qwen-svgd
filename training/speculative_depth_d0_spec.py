"""Fail-closed preregistration scaffold for speculative-depth pilot D0."""

from __future__ import annotations

from dataclasses import dataclass
from copy import deepcopy
from typing import Any

import torch


UNRESOLVED = "RESOLVE_BEFORE_LOCK"
GOVERNING_DOCUMENT = "docs/PHASE_D0_PREREGISTRATION_DRAFT7_20260726.md"
GOVERNING_DOCUMENT_SHA256 = "5606f193902b9faae88891cfb7309f8242e4704dd916fb37b8f23b7a9e9cea22"
GOVERNING_DOCUMENT_HANDOFF_SHA256 = "c5318c23e078038cf9ef5cf63d042848391b0dfb2df717c4b748768cb26845bf"
DRAFTER_CHECKPOINT_SHA256 = "93d2e5f9a941bbe79a0b2fc3f9bf43d582bf054990c14b1a93ff67024140062d"
FINEWEB_DATASET = "HuggingFaceFW/fineweb-edu"
FINEWEB_REVISION = "87f09149ef4734204d70ed1d046ddc9ca3f2b8f9"
FINEWEB_DUMP = "CC-MAIN-2025-26"
FINEWEB_IN_ERA_DUMP = "CC-MAIN-2021-49"
STACK_DATASET = "bigcode/the-stack-smol"
STACK_REVISION = "4a6938ce94446f324c6629e7de00ac591710044b"
STACK_LANGUAGE_DIRECTORIES = {
    "c": "C",
    "c-sharp": "C#",
    "c++": "C++",
    "go": "Go",
    "java": "Java",
    "javascript": "JavaScript",
    "python": "Python",
    "rust": "Rust",
    "shell": "Shell",
    "typescript": "TypeScript",
}
STACK_LANGUAGES = list(STACK_LANGUAGE_DIRECTORIES.values())
PARTITION_SEED = 20260725
PILOT_SEED = 20260726
DRAFTER_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
DRAFTER_MODEL_REVISION = "7ae557604adf67be50417f59c2c2f167def9a775"
TEACHER_7B = "Qwen/Qwen2.5-7B-Instruct"
TEACHER_7B_REVISION = "a09a35458c702b33eeacc393d103063234e8bc28"
TEACHER_14B = "Qwen/Qwen2.5-14B-Instruct"
TEACHER_14B_REVISION = "cf98f3b3bbb457ad9e2bb7baf9a0125b6b88caa8"


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
    """Authorization boundary separated at the preregistration lock."""

    density_probe_authorized: bool = False
    labeling_gpu_authorized: bool = False
    training_authorized: bool = False

    def assert_allowed(self, *, density_probe: bool = False, labeling: bool, training: bool) -> None:
        if density_probe and not self.density_probe_authorized:
            raise RuntimeError("D0 pre-lock density probe is not authorized")
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
            "strata": ["post_cutoff_fineweb_edu", "stack_v1_smol_permissive_license"],
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
            "forced_measurement_depths": [1, 2, 3, 4, 5, 6],
            "maximum_training_target_depth": 4,
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


def prelock_contract() -> dict[str, Any]:
    """Return the only GPU work authorized before the D0 lock commit."""

    return {
        "kind": "paper2_speculative_depth_d0_prelock_contract",
        "status": "density_probe_and_partition_freeze_only",
        "governing_document": GOVERNING_DOCUMENT,
        "governing_document_sha256": GOVERNING_DOCUMENT_SHA256,
        "governing_document_handoff_sha256": GOVERNING_DOCUMENT_HANDOFF_SHA256,
        "density_probe_authorized": True,
        "labeling_gpu_authorized": False,
        "training_authorized": False,
        "teachers": {
            "density_probe": {"model": TEACHER_7B, "revision": TEACHER_7B_REVISION},
            "labeling_primary": {"model": TEACHER_7B, "revision": TEACHER_7B_REVISION},
            "calibration_secondary": {"model": TEACHER_14B, "revision": TEACHER_14B_REVISION},
        },
        "drafter_checkpoint_sha256": DRAFTER_CHECKPOINT_SHA256,
        "corpus": {
            "fineweb": {
                "dataset": FINEWEB_DATASET,
                "revision": FINEWEB_REVISION,
                "dump": FINEWEB_DUMP,
                "in_era_dump": FINEWEB_IN_ERA_DUMP,
            },
            "stack_smol": {
                "dataset": STACK_DATASET,
                "revision": STACK_REVISION,
                "languages": list(STACK_LANGUAGES),
                "language_directories": deepcopy(STACK_LANGUAGE_DIRECTORIES),
                "lineage": "Stack_v1",
                "provenance_period": "in_pretraining_era",
                "license_filter": "The_Stack_permissive_source_with_nonempty_per_file_license_metadata",
                "content_store": "huggingface_direct_text",
                "aws_credentials_required": False,
                "software_heritage_raw_api_forbidden": True,
                "silent_substitution_forbidden": True,
            },
            "partition_seed": PARTITION_SEED,
            "pilot_seed": PILOT_SEED,
            "density_tokens_per_stratum": 100_000,
            "total_labeled_tokens": 2_000_000,
            "partition_fractions": {
                "label_train": 0.8,
                "calibration": 0.1,
                "evaluation": 0.1,
            },
            "mix_rule": {
                "balanced_when_minimum_density_ratio_at_least": 0.5,
                "balanced": {"general": 0.5, "code": 0.5},
                "density_skewed": {"higher_density": 0.6, "lower_density": 0.4},
                "domain_floor": 0.4,
            },
            "in_era_contrast_tokens": 100_000,
            "pilot_rows": 256,
        },
        "forbidden": [
            "labeling_proper",
            "fourteen_b_teacher_forward",
            "optimizer_construction",
            "training_step",
            "trained_checkpoint_write",
        ],
    }


def locked_d0_from_manifest(manifest: dict[str, Any]) -> dict[str, Any]:
    """Materialize Draft 7 after the pre-lock corpus job freezes its outputs."""

    required_hashes = {
        "label_train",
        "calibration",
        "evaluation",
        "density_general",
        "density_code",
        "in_era_contrast",
        "pilot_256",
        "general_label_train",
        "general_calibration",
        "general_evaluation",
        "code_label_train",
        "code_calibration",
        "code_evaluation",
    }
    observed = set((manifest.get("artifacts") or {}).keys())
    missing = sorted(required_hashes - observed)
    if missing:
        raise ValueError(f"D0 manifest is missing locked artifacts: {missing}")
    if manifest.get("document_disjoint") is not True:
        raise ValueError("D0 corpus partitions are not document-disjoint")
    contract = prelock_contract()
    payload = {
        "kind": "paper2_speculative_decoding_depth_d0_preregistration",
        "status": "locked_before_training",
        "locked_before_training": True,
        "training_authorized": True,
        "labeling_gpu_authorized": True,
        "launch_target_exists": False,
        "governing_document": GOVERNING_DOCUMENT,
        "governing_document_sha256": GOVERNING_DOCUMENT_SHA256,
        "dependency": {
            "t1_lite_verdict_required": True,
            "t1_lite_r_receipt": "outputs/stage5/stage5_paper2_t1_lite_r_20260725/summary.json",
            "d0_lock_required": True,
            "automatic_launch_from_t1": False,
        },
        "models": {
            "drafter": {
                "model": DRAFTER_MODEL,
                "model_revision": DRAFTER_MODEL_REVISION,
                "checkpoint_role": "t1_lite_r_seed1_raw_final_step",
                "checkpoint_sha256": DRAFTER_CHECKPOINT_SHA256,
                "ema_excluded": True,
            },
            "teacher_primary": {"model": TEACHER_7B, "revision": TEACHER_7B_REVISION},
            "teacher_calibration_secondary": {"model": TEACHER_14B, "revision": TEACHER_14B_REVISION},
            "teacher_dtype": "bfloat16",
            "single_pass_cache_required": True,
        },
        "corpus": deepcopy(contract["corpus"]),
        "corpus_manifest": manifest,
        "labeling": {
            "context": "teacher_forced_true_prefix",
            "primary_rule": "drafter_greedy_equals_teacher_greedy",
            "recorded_signals": [
                "teacher_greedy_token_id",
                "drafter_token_logprob_under_teacher",
                "drafter_token_rank_under_teacher",
                "teacher_entropy",
                "teacher_to_plain_drafter_kl",
                "rejection_run_length",
            ],
            "fourteen_b_partitions": ["calibration"],
            "teacher_reload_after_cache": False,
        },
        "calibration": {
            "forced_measurement_depths": [1, 2, 3, 4, 5, 6],
            "training_target_cap": 4,
            "severity_bins": "teacher_to_plain_drafter_kl_quartiles",
            "graded_minimum_gain": 0.02,
            "graded_minimum_bins": 2,
            "plateau_tolerance": 0.01,
            "flat_fallback": "first_loop_matching_teacher_else_depth4",
            "primary_fit": "monotone_isotonic",
            "comparison_fits": ["linear_run_length", "linear_log_kl", "saturating"],
            "rejection_run_length_cap": 8,
        },
        "training": {
            "steps": 4000,
            "batch_size": 1,
            "optimizer": "adamw_t1_lite_settings",
            "seed": 0,
            "ema_decay": 0.999,
            "primary_endpoint": "final_step_ema",
            "control_lambda": 0.5,
            "control_class_weights": "equal",
            "mechanism_rehearsal_fraction": 0.30,
            "guardrail_steps": [1000, 2000, 3000],
            "liveness_abort": "flat_control_loss_and_zero_stop_recall",
        },
        "evaluation": {
            "interpretation_only_no_pass_fail_gate": True,
            "gamma": [2, 4, 8],
            "forced_teacher_shift_depths": [1, 2, 3, 4, 5, 6],
            "arc_allocation_probe_descriptive_only": True,
            "hard_guardrails": {
                "accepted_loop1_drop_max": 0.01,
                "t1_mechanism_drop_max": 0.03,
            },
            "interpretation_bands": {
                "minimal_below": 0.02,
                "partial_from": 0.02,
                "partial_below": 0.10,
                "strong_from": 0.10,
            },
        },
        "do_not_claim": [
            "recovered teacher agreement is reasoning",
            "general efficiency from simulation-grade latency",
            "generalization beyond the corpus or on-policy drafting",
            "anything about GRAM or stochastic width",
            "seed or scale robustness",
            "unrecovered_at_depth4_means_knowledge_limited",
            "question_answering_capability_from_arc_allocation",
            "the_stack_v2_was_used",
        ],
    }
    return payload


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
    if payload.get("governing_document_sha256") != GOVERNING_DOCUMENT_SHA256:
        raise AssertionError("D0 governing document hash drifted")
    if payload.get("models", {}).get("drafter", {}).get("checkpoint_sha256") != DRAFTER_CHECKPOINT_SHA256:
        raise AssertionError("D0 drafter checkpoint hash drifted")
    corpus = payload.get("corpus", {})
    if corpus.get("fineweb", {}).get("dump") != FINEWEB_DUMP:
        raise AssertionError("D0 FineWeb-Edu dump drifted")
    if corpus.get("stack_smol", {}).get("revision") != STACK_REVISION:
        raise AssertionError("D0 the-stack-smol revision drifted")
    if corpus.get("stack_smol", {}).get("lineage") != "Stack_v1":
        raise AssertionError("D0 code-corpus lineage drifted")
    if payload.get("corpus_manifest", {}).get("document_disjoint") is not True:
        raise AssertionError("D0 corpus manifest is not document-disjoint")
