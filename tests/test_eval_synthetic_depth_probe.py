from __future__ import annotations

import torch

from eval.eval_synthetic_depth_probe import (
    deterministic_split,
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
    train_x = torch.eye(4).repeat_interleave(4, dim=0)
    train_y = torch.arange(4).repeat_interleave(4)
    test_x = torch.eye(4)
    test_y = torch.arange(4)

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
