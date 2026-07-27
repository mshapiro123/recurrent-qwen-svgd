from __future__ import annotations

import inspect

import torch

from eval.eval_speculative_depth_router_probe import (
    analyze_feature_cache,
    cluster_bootstrap_budget_lower_bounds,
    fit_ridge_probe,
    fixed_projection,
    select_sequential_frontier,
    simulate_sequential_policy,
)


def test_feature_extraction_uses_the_floor_equivalent_forward_path() -> None:
    from eval.eval_speculative_depth_router_probe import extract_feature_cache

    source = inspect.getsource(extract_feature_cache)
    assert "recurrent_application_sink=recurrent_states" in source
    assert "output_hidden_states=True" not in source
    assert "return_loop_recurrent_states=True" not in source


def test_fixed_projection_is_reproducible_and_orthonormal() -> None:
    first = fixed_projection(16, 4, 20260727)
    second = fixed_projection(16, 4, 20260727)
    assert torch.equal(first, second)
    assert torch.allclose(first.T @ first, torch.eye(4), atol=1e-5)


def test_ridge_probe_recovers_a_separable_signal() -> None:
    negative = torch.stack([torch.linspace(-3.0, -0.2, 30), torch.zeros(30)], dim=1)
    positive = torch.stack([torch.linspace(0.2, 3.0, 30), torch.zeros(30)], dim=1)
    features = torch.cat([negative, positive], dim=0)
    labels = torch.tensor([False] * 30 + [True] * 30)
    result = fit_ridge_probe(
        features,
        labels,
        torch.tensor(list(range(0, 20)) + list(range(30, 50))),
        torch.tensor(list(range(20, 25)) + list(range(50, 55))),
        torch.tensor(list(range(25, 30)) + list(range(55, 60))),
    )
    assert result["validation_auroc"] == 1.0
    assert result["test_auroc"] == 1.0


def test_cluster_bootstrap_preserves_source_row_clusters() -> None:
    rows = []
    for row_index in range(20):
        for position in range(3):
            benefit = row_index < 10
            rows.append(
                {
                    "row_index": row_index,
                    "stratum": "general",
                    "loop1_correct": False,
                    "loop2_correct": benefit,
                    "score": 1.0 if benefit else 0.0,
                    "position": position,
                }
            )
    lower = cluster_bootstrap_budget_lower_bounds(
        rows,
        fractions=(0.5,),
        draws=100,
        seed=7,
    )
    assert lower[0.5] > 0.0


def test_sequential_policy_stops_or_continues_from_each_loop() -> None:
    scores = torch.tensor(
        [
            [-1.0, -1.0, -1.0, -1.0, -1.0],
            [1.0, -1.0, -1.0, -1.0, -1.0],
            [1.0, 1.0, 1.0, 1.0, 1.0],
        ]
    )
    matches = torch.tensor(
        [
            [True, False, False, False, False, False],
            [False, True, False, False, False, False],
            [False, False, False, False, False, True],
        ]
    )
    result = simulate_sequential_policy(scores, matches, threshold=0.0)
    assert result["correct"] == 3
    assert result["selected_depth_counts"] == {
        "1": 1,
        "2": 1,
        "3": 0,
        "4": 0,
        "5": 0,
        "6": 1,
    }
    assert result["mean_loops"] == 3.0


def test_sequential_frontier_uses_validation_threshold_without_test_labels() -> None:
    validation_scores = torch.tensor(
        [
            [-1.0] * 5,
            [1.0, -1.0, -1.0, -1.0, -1.0],
            [1.0] * 5,
            [-1.0] * 5,
        ]
    )
    validation_matches = torch.tensor(
        [
            [True, False, False, False, False, False],
            [False, True, False, False, False, False],
            [False, False, False, False, False, True],
            [True, False, False, False, False, False],
        ]
    )
    output = select_sequential_frontier(
        validation_scores,
        validation_matches,
        validation_scores.clone(),
        validation_matches.clone(),
        budgets=(3.0,),
    )
    assert output[0]["test"]["correct"] == 4
    assert output[0]["test"]["mean_loops"] == 2.5


def test_feature_cache_analysis_emits_preloop_and_sequential_receipts() -> None:
    count = 140
    metadata = [
        {
            "row_index": index,
            "local_position": 3,
            "sequence_length": 32,
            "stratum": "code" if index % 2 else "general",
        }
        for index in range(count)
    ]
    first_depths = [1 + index % 7 for index in range(count)]
    matches = torch.zeros(count, 6, dtype=torch.bool)
    for index, depth in enumerate(first_depths):
        if depth <= 6:
            matches[index, depth - 1] = True
    prelude = torch.zeros(count, 4, dtype=torch.float32)
    prelude[:, 0] = torch.tensor(first_depths, dtype=torch.float32)
    states = prelude[:, None, :].repeat(1, 6, 1)
    states[:, :, 1] = torch.arange(1, 7).float()[None, :]
    scalars = torch.zeros(count, 6, 8, dtype=torch.float32)
    scalars[:, :, 0] = states[:, :, 1]
    result = analyze_feature_cache(
        {
            "metadata": metadata,
            "prelude_projection": prelude,
            "state_projection": states,
            "scalars": scalars,
            "matches": matches,
        },
        seed=20260727,
    )
    assert set(result["preloop"]) >= {
        "any_extra_depth",
        "loop2_decision",
        "loop2_budget_points",
        "verdict",
    }
    assert set(result["sequential"]["per_loop"]) == {"1", "2", "3", "4", "5"}
    assert len(result["sequential"]["frontier"]) == 4
    assert result["teacher_features_used"] is False
    assert result["evaluation_partition_touched"] is False
