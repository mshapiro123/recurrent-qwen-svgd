from __future__ import annotations

import math

import pytest
import torch

from training.staircase_curriculum import (
    LoopDoseLedger,
    equalized_loop_weights,
    exposure_fractions,
    optimizer_steps_for_weighted_budget,
)


def depth_balanced_rows(cap: int, rows_per_depth: int = 8) -> list[dict[str, int]]:
    return [
        {"depth": depth}
        for depth in range(1, cap + 1)
        for _ in range(rows_per_depth)
    ]


def test_equalized_weights_cancel_depth_exposure_and_double_newest_loop() -> None:
    cap = 4
    exposure = exposure_fractions(depth_balanced_rows(cap), cap=cap)
    weights = equalized_loop_weights(exposure, cap=cap, newest_multiplier=2.0)
    expected_mass = [exposure[index] * weights[index] for index in range(cap)]

    assert math.isclose(sum(weights), cap)
    assert expected_mass[-1] == pytest.approx(2.0 * expected_mass[0])
    assert expected_mass[0] == pytest.approx(expected_mass[1])
    assert expected_mass[1] == pytest.approx(expected_mass[2])


def test_dose_ledger_tracks_raw_and_weighted_active_rows_without_token_counting() -> None:
    weights = [0.5, 1.5, 0.0]
    ledger = LoopDoseLedger(weights=weights, newest_loop=2, newest_multiplier=2.0)
    labels = torch.full((2, 3, 4), -100, dtype=torch.long)
    labels[0, 0, -1] = 3
    labels[0, 1, -1] = 4
    labels[1, 0, -1] = 5

    ledger.update(labels)
    payload = ledger.as_dict()

    assert payload["raw_active_labels"] == [2, 1, 0]
    assert payload["weighted_active_labels"] == [1.0, 1.5, 0.0]
    assert payload["equalization_mass_excluding_newest_multiplier"] == [1.0, 0.75, 0.0]


def test_realized_equalization_assertion_accepts_expected_mass_and_rejects_starvation() -> None:
    ledger = LoopDoseLedger(weights=[0.4, 1.6], newest_loop=2, newest_multiplier=2.0)
    labels = torch.full((4, 2, 2), -100, dtype=torch.long)
    labels[:, 0, -1] = 1
    labels[:2, 1, -1] = 2
    ledger.update(labels)

    ledger.assert_equalized(min_ratio=0.8, max_ratio=1.25)

    starved = LoopDoseLedger(weights=[0.4, 1.6], newest_loop=2, newest_multiplier=2.0)
    labels[:1, 1, -1] = 2
    labels[1:, 1, -1] = -100
    starved.update(labels)
    with pytest.raises(RuntimeError, match="weighted loop-mass equalization failed"):
        starved.assert_equalized(min_ratio=0.8, max_ratio=1.25)


def test_weighted_budget_uses_last_checkpoint_not_above_hard_cap() -> None:
    steps = optimizer_steps_for_weighted_budget(
        weighted_label_budget=1500.0,
        newest_exposure=1.0 / 3.0,
        newest_weight=2.1176470588235294,
        effective_batch_size=8,
        eval_every=250,
    )

    assert steps == 250


def test_weighted_budget_allows_one_unavoidable_checkpoint_interval() -> None:
    steps = optimizer_steps_for_weighted_budget(
        weighted_label_budget=100.0,
        newest_exposure=0.5,
        newest_weight=1.6,
        effective_batch_size=8,
        eval_every=250,
    )

    assert steps == 250
