from __future__ import annotations

import json
from pathlib import Path

from training.paper2_phase2_e1_confirmation import (
    E1_EVAL_D_FREEZE_KIND,
    REQUIRED_CACHE_FIELDS,
    assess_readiness,
)


ROOT = Path(__file__).resolve().parents[1]
REGISTRATION = ROOT / "training/paper2_phase2_e1_confirmation_preregistration.draft.json"
INVENTORY = ROOT / "training/paper2_phase2_e1_confirmation_rule_inventory.json"
OPTION_B = ROOT / "outputs/stage5/stage5_paper2_phase2_option_b_20260807/summary.json"


def loaded(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def compatible_freeze() -> dict:
    return {
        "kind": E1_EVAL_D_FREEZE_KIND,
        "status": "complete_frozen_unscored",
        "partition": "eval_d",
        "scores_exposed": False,
        "read_once_scoring_spent": False,
        "training_started": False,
        "optimizer_steps": 0,
        "endpoint_checkpoints_loaded": False,
        "model_quality_scores_computed": False,
        "eal_computed": False,
        "retention_computed": False,
        "acceptance_computed": False,
        "student_teacher_quality_aggregates_emitted": False,
        "cross_partition_document_overlap": [],
        "selection": {
            "seed": 20260808,
            "rule": (
                "stable_sha256_rank_within_stratum_then_greedy_nonoverlapping_"
                "four_position_spans_then_row_major_execution_order"
            ),
            "anchors_per_stratum": {"general": 4000, "code": 4000},
        },
        "estimators": {
            "primary": {"weights": {"general": 0.5, "code": 0.5}},
            "dev_mixture_reweighted_secondary": {
                "weights": {"general": 0.51, "code": 0.49}
            },
        },
        "option_b_cache": {
            "kind": "paper2_phase2_matched_alpha_cache_v1",
            "fields": sorted(REQUIRED_CACHE_FIELDS),
            "anchor_count": 8000,
            "anchors_per_stratum": {"general": 4000, "code": 4000},
            "document_count": 85,
            "data_sha256": "a" * 64,
            "position_key_sha256": "b" * 64,
            "private_cache_sha256": "c" * 64,
            "canonicalizer_sha256": "d" * 64,
        },
    }


def test_draft_is_explicitly_unlocked_with_ratified_endpoints() -> None:
    registration = loaded(REGISTRATION)
    assert registration["locked_before_e1_scoring"] is False
    assert registration["e1_evaluation_authorized"] is False
    assert registration["quality"]["point_retention_minimum"] == 0.995
    assert registration["quality"]["wilson_95_lower_minimum"] == 0.990
    assert registration["primary"]["required_seeds"] == [0, 1]
    assert len(registration["checkpoints"]) == 4


def test_rule_inventory_contains_only_evaluation_tripwires() -> None:
    inventory = loaded(INVENTORY)
    assert inventory["continuous_shapers"] == []
    assert inventory["endpoint_thresholds_are_process_aborts"] is False
    assert all(row["class"] == "tripwire" for row in inventory["rules"])


def test_legacy_eval_d_receipt_is_rejected_as_schema_incompatible() -> None:
    result = assess_readiness(
        registration=loaded(REGISTRATION),
        rule_inventory=loaded(INVENTORY),
        option_b_summary=loaded(OPTION_B),
        eval_d_freeze={
            "kind": "paper2_phase2_eval_de_freeze",
            "status": "complete_frozen_unscored",
        },
    )
    assert result["ready_to_lock"] is False
    assert "legacy_eval_d_schema_is_7b_only_not_option_b_compatible" in result["blockers"]


def test_compatible_unscored_freeze_clears_readiness() -> None:
    result = assess_readiness(
        registration=loaded(REGISTRATION),
        rule_inventory=loaded(INVENTORY),
        option_b_summary=loaded(OPTION_B),
        eval_d_freeze=compatible_freeze(),
    )
    assert result["ready_to_lock"] is True
    assert result["status"] == "ready_to_lock"
    assert result["e1_scoring_authorized"] is False


def test_score_exposure_or_missing_cache_field_blocks_lock() -> None:
    freeze = compatible_freeze()
    freeze["scores_exposed"] = True
    freeze["option_b_cache"]["fields"].remove("teacher_topk_ids")
    result = assess_readiness(
        registration=loaded(REGISTRATION),
        rule_inventory=loaded(INVENTORY),
        option_b_summary=loaded(OPTION_B),
        eval_d_freeze=freeze,
    )
    assert result["ready_to_lock"] is False
    assert "eval_d_scores_already_exposed" in result["blockers"]
    assert "eval_d_option_b_cache_fields_missing" in result["blockers"]
    assert result["observations"]["missing_option_b_cache_fields"] == [
        "teacher_topk_ids"
    ]
