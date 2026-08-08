"""Pure contracts for the Phase-2 E1 read-once confirmation pass."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping


ROOT = Path(__file__).resolve().parents[1]
LOCKED_REGISTRATION = (
    ROOT / "training/paper2_phase2_e1_confirmation_preregistration.json"
)
# Kept for the score-blind readiness CLI; after lock, callers must pass an
# explicit historical draft if they intend to rerun the pre-lock checker.
DRAFT_REGISTRATION = LOCKED_REGISTRATION
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
E1_SPARSE_SUPPORT_QC_KIND = "paper2_phase2_e1_sparse_support_qc_v1"
REQUIRED_CACHE_FIELDS = {
    "anchor_keys",
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


def git_lf_sha256_file(path: str | Path) -> str:
    """Hash repository text using the LF bytes materialized on Colab/Linux."""
    payload = Path(path).read_bytes().replace(b"\r\n", b"\n")
    return hashlib.sha256(payload).hexdigest()


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
    sparse_support_qc: Mapping[str, Any] | None,
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
            for field in (
                "endpoint_checkpoints_loaded",
                "model_quality_scores_computed",
                "eal_computed",
                "retention_computed",
                "acceptance_computed",
                "student_teacher_quality_aggregates_emitted",
            ):
                if eval_d_freeze.get(field) is not False:
                    blockers.append(f"eval_d_score_blind_contract_failed_{field}")

            selection = eval_d_freeze.get("selection", {})
            population = registration["evaluation"]["population"]
            if int(selection.get("seed", -1)) != int(population["selection_seed"]):
                blockers.append("eval_d_selection_seed_mismatch")
            if selection.get("rule") != population["selection_rule"]:
                blockers.append("eval_d_selection_rule_mismatch")
            if selection.get("anchors_per_stratum") != population["anchors_per_stratum"]:
                blockers.append("eval_d_stratum_population_mismatch")

            estimators = eval_d_freeze.get("estimators", {})
            primary_weights = estimators.get("primary", {}).get("weights")
            if primary_weights != {"general": 0.5, "code": 0.5}:
                blockers.append("eval_d_balanced_primary_weights_mismatch")
            dev_weights = estimators.get("dev_mixture_reweighted_secondary", {}).get(
                "weights"
            )
            if not isinstance(dev_weights, dict) or set(dev_weights) != {
                "general",
                "code",
            }:
                blockers.append("eval_d_dev_mixture_weights_missing")
            elif abs(sum(float(value) for value in dev_weights.values()) - 1.0) > 1e-9:
                blockers.append("eval_d_dev_mixture_weights_do_not_sum_to_one")

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
            if int(cache.get("anchor_count", -1)) != int(population["anchor_count"]):
                blockers.append("eval_d_anchor_count_mismatch")
            if cache.get("anchors_per_stratum") != population["anchors_per_stratum"]:
                blockers.append("eval_d_cache_stratum_counts_mismatch")

    if sparse_support_qc is None:
        blockers.append("eval_d_sparse_support_qc_missing")
    else:
        observations["eval_d_sparse_support_qc_kind"] = sparse_support_qc.get("kind")
        if sparse_support_qc.get("kind") != E1_SPARSE_SUPPORT_QC_KIND:
            blockers.append("eval_d_sparse_support_qc_kind_mismatch")
        if sparse_support_qc.get("status") != "complete_score_blind_integrity_only":
            blockers.append("eval_d_sparse_support_qc_incomplete")
        if sparse_support_qc.get("all_emitted_metrics_finite") is not True:
            blockers.append("eval_d_sparse_support_qc_nonfinite")
        if sparse_support_qc.get("ready_for_lock_transcription") is not True:
            blockers.append("eval_d_sparse_support_qc_not_ready")
        if sparse_support_qc.get("score_blind") is not True:
            blockers.append("eval_d_sparse_support_qc_not_score_blind")
        for field in (
            "endpoint_checkpoints_loaded",
            "outcome_scores_computed",
            "model_quality_scores_computed",
            "read_once_scoring_spent",
            "training_started",
        ):
            if sparse_support_qc.get(field) is not False:
                blockers.append(f"eval_d_sparse_support_qc_contract_failed_{field}")
        if int(sparse_support_qc.get("optimizer_steps", -1)) != 0:
            blockers.append("eval_d_sparse_support_qc_optimizer_steps_nonzero")
        models = sparse_support_qc.get("models", {})
        if set(models) != {"student_0p5b", "teacher_7b", "teacher_14b", "teacher_32b"}:
            blockers.append("eval_d_sparse_support_qc_model_set_mismatch")
        else:
            for model_key, model in models.items():
                if int(model.get("counts", {}).get("audit_rows", -1)) != 320:
                    blockers.append(
                        f"eval_d_sparse_support_qc_audit_rows_mismatch_{model_key}"
                    )

    unique_blockers = list(dict.fromkeys(blockers))
    return {
        "kind": "paper2_phase2_e1_readiness",
        "version": "e1_readiness_v2_20260808",
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
