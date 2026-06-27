from __future__ import annotations

import math

import torch

import eval.evaluate_tail_convergence_selector as tail


def test_tail_convergence_features_deceleration_is_positive_when_motion_shrinks() -> None:
    features = tail.tail_convergence_features(
        {
            1: torch.tensor([1.0, 0.0]),
            2: torch.tensor([2.0, 0.0]),
            3: torch.tensor([2.25, 0.0]),
        }
    )

    assert features["tail_rel_disp_12"] > features["tail_rel_disp_23"]
    assert features["tail_deceleration_12_minus_23"] > 0
    assert features["tail_disp_ratio_23_over_12"] < 1
    assert math.isclose(features["tail_cos_12"], 1.0)
    assert math.isclose(features["tail_cos_23"], 1.0)


def test_principal_subspace_rotation_reports_unit_overlap_for_same_cloud() -> None:
    cloud = [torch.tensor([float(i), float(i % 2), 0.0]) for i in range(8)]
    result = tail.principal_subspace_rotation({1: cloud, 2: cloud, 3: cloud}, rank=2)

    assert result["1_to_2"]["rank"] == 2
    assert result["1_to_2"]["mean_squared_cosine"] > 0.999
    assert result["2_to_3"]["min_cosine"] > 0.999


def test_feature_policy_scores_rescue_when_high_deceleration_marks_rescuable() -> None:
    examples = [
        {
            "id": "rescue",
            "loop_hits": {1: False, 2: True, 3: True},
            "loop1_hit": False,
            "rescuable": True,
            "harmable": False,
            "category": "rescuable",
            "tail_convergence": {"tail_deceleration_12_minus_23": 3.0},
        },
        {
            "id": "harm",
            "loop_hits": {1: True, 2: False, 3: False},
            "loop1_hit": True,
            "rescuable": False,
            "harmable": True,
            "category": "harmable",
            "tail_convergence": {"tail_deceleration_12_minus_23": -1.0},
        },
        {
            "id": "stable_correct",
            "loop_hits": {1: True, 2: True, 3: True},
            "loop1_hit": True,
            "rescuable": False,
            "harmable": False,
            "category": "stable_correct",
            "tail_convergence": {"tail_deceleration_12_minus_23": 0.2},
        },
    ]

    policies = tail.policies_from_feature_curve(examples, [1, 2, 3], "tail_deceleration_12_minus_23")
    rows = tail.apply_feature_policies(examples, [1, 2, 3], "tail_deceleration_12_minus_23", policies)
    best = tail.curve_summary(rows)["max_net"]

    assert best["delta_vs_loop1"] >= 1
    assert best["rescue_captured"] >= 1
    assert best["harm_triggered"] == 0
