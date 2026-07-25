"""Locked amendment contract for the registered T1-lite seed-1 replication."""

from __future__ import annotations

from copy import deepcopy
from typing import Any

from training.internal_think_token_t1_spec import phase_t1_locked


LOCKED_DATE = "2026-07-25"
ORIGINAL_T1_LOCK = (
    "outputs/stage5/stage5_paper2_t1_lite_preregistration_20260724/"
    "preregistration.json"
)
ORIGINAL_T1_LOCK_SHA256 = (
    "4e55e946a8019d2c0c278bfaff2e76cd97b3efb7822b954b2cb74a539c037cba"
)
STAGE_CHECKPOINT_STEPS = (500, 2500, 6500, 8500, 10500)


def phase_t1_lite_r_locked() -> dict[str, Any]:
    """Return the locked seed-1 contract as an amendment to T1-lite."""

    payload = deepcopy(phase_t1_locked())
    payload.update(
        {
            "kind": "paper2_internal_think_token_phase_t1_lite_r_preregistration",
            "program_mode": "t1_lite_full_block_seed1_replication",
            "status": "locked_before_training",
            "training_authorized": True,
            "locked_date": LOCKED_DATE,
            "governing_document": (
                "docs/PHASE_T1_LITE_R_PREREGISTRATION_20260725.md"
            ),
            "amendment": {
                "base_contract": ORIGINAL_T1_LOCK,
                "original_lock_sha256": ORIGINAL_T1_LOCK_SHA256,
                "registered_attempt": 2,
                "single_amended_factor": "endpoint_policy",
                "seed_change_required_by_replication_rule": True,
                "seed0_verdict_unchanged": True,
            },
            "endpoint_policy": {
                "primary": "raw_final_step",
                "intermediate_checkpoint_selection": False,
                "continuous_ema": {
                    "decay": 0.999,
                    "role": "passive_descriptive_shadow",
                    "may_change_updates": False,
                    "may_be_primary": False,
                },
                "stage_reset_ema": {
                    "decay": 0.999,
                    "reset_from_raw_at_stage_start": True,
                    "role": "passive_descriptive_shadow",
                    "may_change_updates": False,
                    "may_be_primary": False,
                },
            },
            "stage_checkpoint_manifest": {
                "required_steps": list(STAGE_CHECKPOINT_STEPS),
                "states": ["raw", "continuous_ema", "stage_reset_ema"],
                "atomic_write": True,
                "sha256_each_state": True,
                "backup_to_drive": True,
                "end_of_run_availability_required": True,
                "missing_artifact_policy": "run_incomplete_not_scored",
            },
            "authorized_seed0_read_only_audits": {
                "layer_group_swap": True,
                "per_depth_interpolation_breakdown": True,
                "may_change_seed0_verdict": False,
                "may_change_seed1_gates": False,
            },
        }
    )
    payload["replication"] = deepcopy(payload["replication"])
    payload["replication"].update(
        {
            "primary_seed": 1,
            "trigger_receipt": (
                "outputs/stage5/stage5_paper2_t1_lite_20260724/summary.json"
            ),
            "trigger_reason": "seed0_raw_near_threshold",
            "seed0_registered_negative_is_final": True,
        }
    )
    payload["proposed_training_budget"] = deepcopy(payload["proposed_training_budget"])
    payload["proposed_training_budget"]["training_seeds"] = [1]
    payload["proposed_training_budget"]["single_seed_limitation_required"] = False
    payload["do_not_claim"] = list(payload["do_not_claim"]) + [
        "that the EMA implementation was mathematically wrong",
        "which seed-0 curriculum stage first produced EMA lag",
        "any result from a passive EMA shadow as a registered endpoint",
        "extrapolation beyond trained support",
    ]
    return payload


def validate_phase_t1_lite_r_locked(payload: dict[str, Any]) -> None:
    """Fail closed on any change outside the registered replication amendment."""

    base = phase_t1_locked()
    if payload.get("status") != "locked_before_training" or payload.get("training_authorized") is not True:
        raise AssertionError("T1-lite-R is not locked before training")
    if payload.get("replication", {}).get("primary_seed") != 1:
        raise AssertionError("T1-lite-R must use registered seed 1")
    amendment = payload.get("amendment", {})
    if amendment.get("original_lock_sha256") != ORIGINAL_T1_LOCK_SHA256:
        raise AssertionError("T1-lite-R original lock hash drifted")
    endpoint = payload.get("endpoint_policy", {})
    if endpoint.get("primary") != "raw_final_step":
        raise AssertionError("T1-lite-R raw final-step endpoint is not primary")
    for shadow in ("continuous_ema", "stage_reset_ema"):
        if endpoint.get(shadow, {}).get("role") != "passive_descriptive_shadow":
            raise AssertionError(f"T1-lite-R {shadow} is not passive")
    manifest = payload.get("stage_checkpoint_manifest", {})
    if manifest.get("required_steps") != list(STAGE_CHECKPOINT_STEPS):
        raise AssertionError("T1-lite-R boundary manifest steps drifted")
    if manifest.get("states") != ["raw", "continuous_ema", "stage_reset_ema"]:
        raise AssertionError("T1-lite-R boundary manifest state set drifted")
    if not manifest.get("atomic_write") or not manifest.get("end_of_run_availability_required"):
        raise AssertionError("T1-lite-R boundary artifacts are not fail closed")

    unchanged_keys = (
        "data",
        "evaluation",
        "gates",
        "integrity",
        "loss",
        "stage_boundary_liveness",
        "fresh_base_lineages",
    )
    drift = [key for key in unchanged_keys if payload.get(key) != base.get(key)]
    base_budget = deepcopy(base["proposed_training_budget"])
    observed_budget = deepcopy(payload["proposed_training_budget"])
    base_budget["training_seeds"] = [1]
    base_budget["single_seed_limitation_required"] = False
    if observed_budget != base_budget:
        drift.append("proposed_training_budget")
    if drift:
        raise AssertionError(f"T1-lite-R drifted from base T1-lite contract: {drift}")

