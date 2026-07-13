from __future__ import annotations

import torch

from eval.eval_abductive_staircase import (
    centered_linear_cka,
    conditional_transition_success,
    target_decodability_probe,
)


def test_conditional_transition_success_uses_only_rows_with_previous_loop_correct() -> None:
    rows = [
        {"id": "a", "loop": 1, "active": True, "hit": True},
        {"id": "a", "loop": 2, "active": True, "hit": True},
        {"id": "b", "loop": 1, "active": True, "hit": True},
        {"id": "b", "loop": 2, "active": True, "hit": False},
        {"id": "c", "loop": 1, "active": True, "hit": False},
        {"id": "c", "loop": 2, "active": True, "hit": True},
    ]

    result = conditional_transition_success(rows, loop=2)

    assert result == {"correct": 1, "total": 2, "accuracy": 0.5}


def test_target_decodability_probe_separates_a_linear_signal_from_permutation_null() -> None:
    features = []
    targets = []
    for repeat in range(20):
        for label in range(3):
            feature = torch.zeros(4)
            feature[label] = 10.0
            feature[-1] = repeat / 100.0
            features.append(feature)
            targets.append(label)

    result = target_decodability_probe(
        torch.stack(features),
        torch.tensor(targets),
        ids=[f"row_{index:03d}" for index in range(len(features))],
        n_classes=3,
        permutations=20,
        seed=17,
    )

    assert result["accuracy"] == 1.0
    assert result["accuracy"] > result["permutation_p95"]
    assert result["train_rows"] + result["test_rows"] == 60


def test_centered_linear_cka_is_one_for_identical_features() -> None:
    features = torch.tensor([[1.0, 0.0], [0.0, 1.0], [2.0, 1.0], [1.0, 2.0]])
    assert centered_linear_cka(features, features) == 1.0

