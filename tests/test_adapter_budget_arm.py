from __future__ import annotations

from training.adapter_budget_arm import (
    ARM_A_COUNTS,
    ARM_A_POOLED_ACCURACY,
    ARM_C_POOLED_ACCURACY,
    locked_spec,
    score_adapter_budget_arm,
)


def _rows(counts: dict[int, int], *, field: str) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for depth in range(1, 15):
        correct = counts[depth]
        for index in range(128):
            rows.append(
                {
                    "id": f"test_d{depth:02d}_{index:05d}",
                    "depth": depth,
                    field: index < correct,
                }
            )
    return rows


def test_locked_spec_replays_complete_arm_a_lineage_from_base() -> None:
    spec = locked_spec()

    assert spec["arm"] == {"name": "E", "rank": 16, "alpha": 32}
    assert spec["initialization"] == "fresh_base_qwen_surgery"
    assert spec["optimizer"] == "adamw"
    assert "matching_historical_Arm_A" in spec["compute_policy"]
    assert "second_variable" in spec["compute_policy"]
    assert spec["total_optimizer_steps"] == 10_500
    assert [stage["max_steps"] for stage in spec["stages"]] == [500, 2000, 4000, 2000, 2000]
    assert [stage["max_loops"] for stage in spec["stages"]] == [1, 2, 4, 8, 8]
    assert all(stage["batch_size"] == 1 for stage in spec["stages"])
    assert spec["final_eval"]["rows"] == 1792
    assert spec["final_eval"]["rows_per_depth"] == 128


def test_arm_a_reference_is_frozen_to_receipt_counts() -> None:
    assert sum(ARM_A_COUNTS.values()) == 1506
    assert ARM_A_POOLED_ACCURACY == 1506 / 1792
    assert ARM_C_POOLED_ACCURACY == 0.531


def test_parity_requires_pooled_band_and_every_depth_floor() -> None:
    arm_a = _rows(ARM_A_COUNTS, field="same_reader_final_hit")
    parity_counts = dict(ARM_A_COUNTS)
    parity_counts[14] -= 8
    arm_e = _rows(parity_counts, field="same_reader_final_hit")

    result = score_adapter_budget_arm(arm_a, arm_e)

    assert result["verdict"] == "parity"
    assert result["gates"]["pooled_within_three_points"] is True
    assert result["gates"]["no_depth_worse_by_more_than_8"] is True
    assert result["paired_by_depth"]["14"]["net_correct"] == -8


def test_ninth_loss_at_one_depth_is_a_deficit_not_parity() -> None:
    arm_a = _rows(ARM_A_COUNTS, field="same_reader_final_hit")
    counts = dict(ARM_A_COUNTS)
    counts[14] -= 9
    arm_e = _rows(counts, field="same_reader_final_hit")

    result = score_adapter_budget_arm(arm_a, arm_e)

    assert result["verdict"] == "deficit"
    assert result["gates"]["pooled_within_three_points"] is True
    assert result["gates"]["no_depth_worse_by_more_than_8"] is False
    assert result["deficit_shape"] == "tail_concentrated"


def test_clear_improvement_is_not_mislabeled_as_capacity_deficit() -> None:
    arm_a = _rows(ARM_A_COUNTS, field="same_reader_final_hit")
    counts = dict(ARM_A_COUNTS)
    counts[12] += 20
    counts[13] += 20
    counts[14] += 20
    arm_e = _rows(counts, field="same_reader_final_hit")

    result = score_adapter_budget_arm(arm_a, arm_e)

    assert result["gates"]["pooled_within_three_points"] is False
    assert result["gates"]["pooled_meets_or_exceeds_parity_floor"] is True
    assert result["verdict"] == "parity"


def test_catastrophic_floor_overrides_depth_shape() -> None:
    arm_a = _rows(ARM_A_COUNTS, field="same_reader_final_hit")
    arm_e = _rows({depth: 60 for depth in range(1, 15)}, field="same_reader_final_hit")

    result = score_adapter_budget_arm(arm_a, arm_e)

    assert result["verdict"] == "catastrophic_training_recipe_alarm"
    assert result["gates"]["above_arm_c_floor"] is False
    assert result["required_followup"] == [
        "verify_step_zero_identity",
        "verify_base_hash_unchanged",
        "verify_weighted_label_dose",
    ]


def test_row_ids_must_match_before_pairing() -> None:
    arm_a = _rows(ARM_A_COUNTS, field="same_reader_final_hit")
    arm_e = _rows(ARM_A_COUNTS, field="same_reader_final_hit")
    arm_e[-1]["id"] = "different"

    try:
        score_adapter_budget_arm(arm_a, arm_e)
    except ValueError as exc:
        assert "identical row IDs" in str(exc)
    else:
        raise AssertionError("mismatched rows must fail")
