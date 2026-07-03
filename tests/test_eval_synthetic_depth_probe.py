from __future__ import annotations

import torch

from eval.eval_synthetic_depth_probe import (
    deterministic_split,
    loop_index_deflation_curve,
    probe_grid,
    permutation_p95,
    ridge_multiclass_accuracy,
    target_for_step,
)


def test_target_for_step_continues_serialized_mapping() -> None:
    row = {
        "id": "row",
        "start": "A",
        "mapping": {"A": "B", "B": "C", "C": "D", "D": "E"},
    }

    assert target_for_step(row, 0, value_prefix="letter:") == 0
    assert target_for_step(row, 1, value_prefix="letter:") == 1
    assert target_for_step(row, 3, value_prefix="letter:") == 3


def test_ridge_multiclass_accuracy_decodes_planted_signal() -> None:
    train_x = torch.tensor(
        [
            [1.0, 0.0],
            [0.9, 0.1],
            [0.0, 1.0],
            [0.1, 0.9],
        ]
    )
    train_y = torch.tensor([0, 0, 1, 1])
    test_x = torch.tensor([[0.8, 0.2], [0.2, 0.8]])
    test_y = torch.tensor([0, 1])

    assert ridge_multiclass_accuracy(train_x, train_y, test_x, test_y, n_classes=2, l2=1e-3) == 1.0


def test_permutation_null_stays_below_planted_signal() -> None:
    generator = torch.Generator().manual_seed(0)
    centers = torch.eye(4)
    train_x = centers.repeat_interleave(12, dim=0) + 0.03 * torch.randn(48, 4, generator=generator)
    train_y = torch.arange(4).repeat_interleave(12)
    test_x = centers.repeat_interleave(8, dim=0) + 0.03 * torch.randn(32, 4, generator=generator)
    test_y = torch.arange(4).repeat_interleave(8)

    observed = ridge_multiclass_accuracy(train_x, train_y, test_x, test_y, n_classes=4, l2=1e-3)
    null = permutation_p95(
        train_x,
        train_y,
        test_x,
        test_y,
        n_classes=4,
        l2=1e-3,
        permutations=16,
        seed=0,
    )

    assert observed == 1.0
    assert null < observed


def test_deterministic_split_is_reproducible() -> None:
    first = deterministic_split(10, train_frac=0.7, seed=123)
    second = deterministic_split(10, train_frac=0.7, seed=123)

    assert first == second
    assert len(first[0]) == 7
    assert len(first[1]) == 3


def test_probe_grid_reports_depth_stratified_diagonal() -> None:
    records = []
    for row_idx in range(12):
        depth = 1 + (row_idx % 2)
        for loop in [1, 2]:
            label = loop % 4
            feature = torch.zeros(4)
            feature[label] = 1.0
            records.append(
                {
                    "row_index": row_idx,
                    "depth": depth,
                    "loop": loop,
                    "feature": feature,
                    "targets": {"0": 0, "1": 1, "2": 2},
                }
            )

    class Args:
        train_frac = 0.75
        seed = 0
        loop_counts = "1,2"
        target_steps = "0,1,2"
        ridge_l2 = 1e-3
        permutations = 0
        deflation_max_rank = 2

    summary = probe_grid(records, Args(), n_symbols=4)

    assert set(summary["depth_stratified_diagonal"]) == {"1", "2"}
    assert summary["depth_stratified_diagonal"]["1"]["1"]["test_rows"] > 0
    assert summary["depth_stratified_diagonal"]["2"]["2"]["test_rows"] > 0
    assert "loop_index_deflation_curve" in summary


def test_loop_index_deflation_curve_removes_low_rank_clock() -> None:
    records = []
    for row_idx in range(30):
        for loop in [1, 2, 3]:
            feature = torch.randn(8, generator=torch.Generator().manual_seed(row_idx * 10 + loop)) * 0.001
            feature[loop - 1] = 1.0
            records.append(
                {
                    "row_index": row_idx,
                    "depth": 3,
                    "loop": loop,
                    "feature": feature,
                    "targets": {"0": 0, "1": 1, "2": 2, "3": 3},
                }
            )

    train_rows, test_rows = deterministic_split(30, train_frac=0.7, seed=1)

    class Args:
        ridge_l2 = 1e-3
        permutations = 0
        seed = 0
        deflation_max_rank = 2

    curve = loop_index_deflation_curve(
        records,
        Args(),
        loops=[1, 2, 3],
        train_set=set(train_rows),
        test_set=set(test_rows),
    )

    assert curve[0]["accuracy"] > 0.9
    assert curve[2]["accuracy"] < curve[0]["accuracy"]
