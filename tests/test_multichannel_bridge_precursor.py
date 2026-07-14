from __future__ import annotations

from pathlib import Path

import torch

from eval.eval_multichannel_bridge_precursor import (
    classify_injection_heterogeneity,
    classify_retrieval_heads,
    classify_subspace_drift,
    output_projection_head_subspaces,
    random_orthogonal_partitions,
    table_character_span,
)
from colab.run_stage5_multichannel_bridge_precursor import (
    aggregate_battery,
    resolve_staircase_reading,
)


ROOT = Path(__file__).resolve().parents[1]


def test_output_projection_identity_yields_axis_aligned_head_subspaces() -> None:
    weight = torch.eye(8)
    bases = output_projection_head_subspaces(weight, num_heads=2)

    assert bases.shape == (2, 8, 4)
    assert torch.allclose(bases[0].T @ bases[0], torch.eye(4), atol=1e-6)
    assert torch.allclose(bases[1].T @ bases[1], torch.eye(4), atol=1e-6)
    assert torch.allclose(bases[0][4:], torch.zeros(4, 4), atol=1e-6)


def test_random_partitions_are_orthonormal_and_reproducible() -> None:
    left = random_orthogonal_partitions(hidden_size=8, num_heads=2, draws=3, seed=17)
    right = random_orthogonal_partitions(hidden_size=8, num_heads=2, draws=3, seed=17)

    assert left.shape == (3, 2, 8, 4)
    assert torch.equal(left, right)
    for draw in left:
        full = torch.cat([draw[index] for index in range(2)], dim=1)
        assert torch.allclose(full.T @ full, torch.eye(8), atol=1e-5)


def test_m1_requires_consistent_two_x_advantage() -> None:
    summary = classify_subspace_drift(
        head_top3_by_loop={"6": 0.72, "7": 0.70, "8": 0.68, "9": 0.69},
        random_top3_mean_by_loop={"6": 0.30, "7": 0.31, "8": 0.32, "9": 0.31},
        minimum_loop=6,
        consistency_fraction=0.75,
    )
    assert summary["confirmed"] is True
    assert summary["qualifying_loops"] == 4


def test_m2_requires_two_stable_three_x_heads_and_random_null_win() -> None:
    summary = classify_retrieval_heads(
        head_ratio_by_id={"L7H2": 4.0, "L9H11": 3.2, "L8H4": 1.8},
        stable_fraction_by_id={"L7H2": 0.75, "L9H11": 0.60, "L8H4": 0.90},
        actual_concentration=0.41,
        random_concentrations=[0.20, 0.24, 0.25, 0.28, 0.30],
        minimum_stable_fraction=0.50,
    )
    assert summary["confirmed"] is True
    assert summary["qualifying_heads"] == ["L7H2", "L9H11"]


def test_m3_requires_five_x_damage_and_random_null_win() -> None:
    summary = classify_injection_heterogeneity(
        head_damage=[0.01, 0.02, 0.03, 0.20],
        random_damage=[0.01, 0.02, 0.025, 0.03, 0.04],
    )
    assert summary["confirmed"] is True
    assert summary["max_to_median_ratio"] >= 5.0


def test_table_span_supports_forward_and_handoff_renderings() -> None:
    forward = "Intro\nFunction table:\nA -> B\nB -> C\n\nStart value: A"
    handoff = "Ada always passes the key to Ben.\nBen always passes the key to Sam.\n\nAfter two handoffs"

    forward_start, forward_end = table_character_span(forward)
    handoff_start, handoff_end = table_character_span(handoff)

    assert forward[forward_start:forward_end] == "A -> B\nB -> C"
    assert handoff[handoff_start:handoff_end].startswith("Ada always passes")


def test_master_decision_requires_cross_task_replication_and_staircase_gate() -> None:
    positive = {"classification": {"confirmed": True}}
    conditions = {
        "n24_step6000": {"measurements": {"m1": positive, "m2": positive, "m3": positive}},
        "backward_recovery": {"measurements": {"m1": positive, "m2": positive}},
    }

    banked = aggregate_battery(conditions, reading_one=False)
    eligible = aggregate_battery(conditions, reading_one=True)

    assert banked["battery_specialization_confirmed"] is True
    assert banked["architecture_activation_eligible"] is False
    assert banked["decision"] == "remain_banked"
    assert eligible["architecture_activation_eligible"] is True


def test_master_decision_does_not_count_single_task_m1_m2() -> None:
    positive = {"classification": {"confirmed": True}}
    conditions = {
        "n24_step6000": {"measurements": {"m1": positive, "m2": positive, "m3": positive}},
    }

    result = aggregate_battery(conditions, reading_one=True)

    assert result["measurement_votes"] == {"m1": False, "m2": False, "m3": True}
    assert result["battery_specialization_confirmed"] is False


def test_staircase_reading_resolves_nested_cap_schema_and_post_run_correction() -> None:
    result = resolve_staircase_reading(
        {
            "matched_arm_reading": {
                "2": {
                    "experiment_weighted_labels_to_bar": None,
                    "control_weighted_labels_to_bar": 1598.4,
                    "ratio": None,
                    "reading": "non_native_position_cost",
                }
            }
        }
    )

    assert result["cap"] == "2"
    assert result["reported_reading"] == "non_native_position_cost"
    assert result["reading"] == "experiment_stalled_at_matched_dose"
    assert result["correction"] == "post_run_clarification_no_experiment_dose_to_bar"
    assert result["reading_one"] is False


def test_staircase_reading_accepts_scalar_and_direct_entry_schemas() -> None:
    scalar = resolve_staircase_reading({"matched_arm_reading": "reading_one"})
    direct = resolve_staircase_reading(
        {"matched_arm_reading": {"reading": "per_position_install_cost_confirmed"}}
    )

    assert scalar["reading"] == "reading_one"
    assert scalar["reading_one"] is True
    assert direct["reading"] == "per_position_install_cost_confirmed"
    assert direct["reading_one"] is True


def test_staircase_reading_uses_highest_numeric_cap_and_handles_missing_data() -> None:
    nested = resolve_staircase_reading(
        {
            "matched_arm_reading": {
                "2": {"reading": "composition_hard_both"},
                "10": {"reading": "reading_one"},
                "not_a_cap": {"reading": "ignored"},
            }
        }
    )
    missing = resolve_staircase_reading({"matched_arm_reading": {"2": {"reading": None}}})

    assert nested["cap"] == "10"
    assert nested["reading"] == "reading_one"
    assert missing["reading"] is None
    assert missing["reading_one"] is False


def test_colab_target_is_wired_with_eval_only_safety_markers() -> None:
    bootstrap = (ROOT / "colab/CURRENT_A100_BOOTSTRAP_CELL.py").read_text(encoding="utf-8")
    cell = (ROOT / "colab/STAGE5_MULTICHANNEL_BRIDGE_PRECURSOR_CELL.py").read_text(encoding="utf-8")

    assert '"multichannel_bridge_precursor"' in bootstrap
    assert "STAGE5_MULTICHANNEL_BRIDGE_PRECURSOR_CELL_VERSION" in bootstrap
    assert "eval/eval_multichannel_bridge_precursor.py" in bootstrap
    assert "tests/test_multichannel_bridge_precursor.py" in cell
    assert "run_stage5_multichannel_bridge_precursor.py" in cell
    assert "train" not in (ROOT / "eval/eval_multichannel_bridge_precursor.py").read_text(encoding="utf-8").lower().splitlines()[0]
