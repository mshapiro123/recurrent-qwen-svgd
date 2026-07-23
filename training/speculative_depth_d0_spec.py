"""Fail-closed preregistration scaffold for speculative-depth pilot D0."""

from __future__ import annotations

from typing import Any


UNRESOLVED = "RESOLVE_BEFORE_LOCK"


def d0_draft() -> dict[str, Any]:
    return {
        "kind": "paper2_speculative_decoding_depth_d0_preregistration",
        "status": "draft_not_locked",
        "training_authorized": False,
        "launch_target_exists": False,
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
