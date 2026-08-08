"""Pure contracts for the Phase-2 E1 read-once confirmation pass."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping


ROOT = Path(__file__).resolve().parents[1]
DRAFT_REGISTRATION = (
    ROOT / "training/paper2_phase2_e1_confirmation_preregistration.draft.json"
)
RULE_INVENTORY = ROOT / "training/paper2_phase2_e1_confirmation_rule_inventory.json"
OPTION_B_SUMMARY = (
    ROOT / "outputs/stage5/stage5_paper2_phase2_option_b_20260807/summary.json"
)
LEGACY_EVAL_DE_SUMMARY = (
    ROOT
    / "outputs/stage5/stage5_paper2_phase2_prewindow_20260731/eval_de/summary.json"
)

OPTION_B_CACHE_KIND = "paper2_phase2_matched_alpha_cache_v1"
E1_EVAL_D_FREEZE_KIND = "paper2_phase2_e1_eval_d_freeze_v1"
REQUIRED_CACHE_FIELDS = {
    "documents",
    "strata",
    "positions",
    "student_hidden",
    "target_centered_raw",
    "candidate_ids",
    "candidate_mask",
    "base_log_probs",
    "base_tail",
    "teacher_log_probs",
    "teacher_tail",
    "teacher_topk_ids",
    "teacher_topk_log_probs",
    "whiten_basis",
    "whiten_eigenvalues",
    "decoder_weight_alpha_0p5",
    "decoder_bias",
}


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_json_sha256(payload: Any) -> str:
    encoded = json.dumps(
        payload, ensure_ascii=True, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def endpoint_hashes_from_option_b(summary: Mapping[str, Any]) -> dict[str, str]:
    result: dict[str, str] = {}
    for row in summary.get("arms", []):
        seed = int(row["seed"])
        arm = str(row["arm"])
        result[f"seed_{seed}_{arm}"] = str(row["checkpoint"]["sha256"])
    return result


def expected_endpoint_hashes(registration: Mapping[str, Any]) -> dict[str, str]:
    return {
        str(name): str(value["sha256"])
        for name, value in registration["checkpoints"].items()
    }


def assess_readiness(
    *,
    registration: Mapping[str, Any],
    rule_inventory: Mapping[str, Any],
    option_b_summary: Mapping[str, Any] | None,
    eval_d_freeze: Mapping[str, Any] | None,
) -> dict[str, Any]:
    blockers: list[str] = []
    observations: dict[str, Any] = {}

    if registration.get("status") != "draft_awaiting_option_b_compatible_eval_d_not_locked":
        blockers.append("registration_draft_status_unexpected")
    if registration.get("locked_before_e1_scoring") is not False:
        blockers.append("draft_must_not_claim_locked")
    if registration.get("e1_evaluation_authorized") is not False:
        blockers.append("draft_must_not_authorize_evaluation")
    if rule_inventory.get("kind") != (
        "paper2_phase2_e1_confirmation_evaluation_rule_inventory"
    ):
        blockers.append("rule_inventory_kind_mismatch")
    if rule_inventory.get("continuous_shapers") != []:
        blockers.append("evaluation_rule_inventory_must_not_contain_shapers")

    expected = expected_endpoint_hashes(registration)
    if option_b_summary is None:
        blockers.append("option_b_public_summary_missing")
    else:
        observed = endpoint_hashes_from_option_b(option_b_summary)
        observations["option_b_endpoint_hashes"] = observed
        if observed != expected:
            blockers.append("option_b_endpoint_hashes_do_not_match_registration")
        incomplete = [
            f"seed_{row.get('seed')}_{row.get('arm')}"
            for row in option_b_summary.get("arms", [])
            if int(row.get("step", -1)) != 20_000
            or row.get("status") != "complete"
            or row.get("abort_reason") is not None
        ]
        if incomplete:
            blockers.append("option_b_endpoint_not_complete_or_aborted")
            observations["incomplete_option_b_arms"] = incomplete

    if eval_d_freeze is None:
        blockers.append("eval_d_freeze_receipt_missing")
    else:
        observations["eval_d_freeze_kind"] = eval_d_freeze.get("kind")
        if eval_d_freeze.get("kind") == "paper2_phase2_eval_de_freeze":
            blockers.append("legacy_eval_d_schema_is_7b_only_not_option_b_compatible")
        elif eval_d_freeze.get("kind") != E1_EVAL_D_FREEZE_KIND:
            blockers.append("eval_d_freeze_kind_mismatch")
        else:
            if eval_d_freeze.get("status") != "complete_frozen_unscored":
                blockers.append("eval_d_not_complete_frozen_unscored")
            if eval_d_freeze.get("partition") != "eval_d":
                blockers.append("wrong_confirmation_partition")
            if eval_d_freeze.get("scores_exposed") is not False:
                blockers.append("eval_d_scores_already_exposed")
            if eval_d_freeze.get("read_once_scoring_spent") is not False:
                blockers.append("eval_d_read_once_already_spent")
            if eval_d_freeze.get("training_started") is not False:
                blockers.append("eval_d_freeze_must_not_train")
            if int(eval_d_freeze.get("optimizer_steps", -1)) != 0:
                blockers.append("eval_d_freeze_optimizer_steps_nonzero")
            if eval_d_freeze.get("cross_partition_document_overlap") not in ([], None):
                blockers.append("eval_d_document_overlap")

            cache = eval_d_freeze.get("option_b_cache", {})
            if cache.get("kind") != OPTION_B_CACHE_KIND:
                blockers.append("eval_d_option_b_cache_kind_mismatch")
            fields = set(cache.get("fields", []))
            missing_fields = sorted(REQUIRED_CACHE_FIELDS - fields)
            if missing_fields:
                blockers.append("eval_d_option_b_cache_fields_missing")
                observations["missing_option_b_cache_fields"] = missing_fields
            for field in (
                "anchor_count",
                "document_count",
                "data_sha256",
                "position_key_sha256",
                "private_cache_sha256",
                "canonicalizer_sha256",
            ):
                value = cache.get(field)
                if value in (None, "", 0):
                    blockers.append(f"eval_d_{field}_missing")

    unique_blockers = list(dict.fromkeys(blockers))
    return {
        "kind": "paper2_phase2_e1_readiness",
        "version": "e1_readiness_v1_20260808",
        "status": "ready_to_lock" if not unique_blockers else "blocked_before_lock",
        "ready_to_lock": not unique_blockers,
        "e1_scoring_authorized": False,
        "read_once_scoring_spent": False,
        "blockers": unique_blockers,
        "observations": observations,
        "required_option_b_cache_kind": OPTION_B_CACHE_KIND,
        "required_option_b_cache_fields": sorted(REQUIRED_CACHE_FIELDS),
        "note": (
            "Readiness authorizes completion of the preregistration lock only; "
            "the eventual locked registration must separately authorize E1 scoring."
        ),
    }
