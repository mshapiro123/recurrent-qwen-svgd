from __future__ import annotations

import torch

from training.paper2_bicameral_w1 import (
    bootstrap_mean_ci,
    deterministic_permutation,
    project_cost_hours,
    resolve_phase_a,
    scale_external_write,
)


def test_external_write_is_exactly_gamma_rms_on_active_rows() -> None:
    hidden = torch.randn(3, 5, 8)
    direction = torch.randn(3, 8)
    mask = torch.tensor([[1, 1, 1, 0, 0], [1, 1, 1, 1, 1], [1, 0, 0, 0, 0]])
    deployed, receipt = scale_external_write(hidden, direction, mask)
    assert deployed.shape == hidden.shape
    assert torch.allclose(receipt["write_ratio"], torch.full((3,), 0.05), atol=1e-6)
    assert torch.equal(deployed[0, 3:], hidden[0, 3:])


def test_frozen_interface_can_be_promoted_to_a_gradient_leaf() -> None:
    hidden = torch.randn(2, 3, 4)
    assert hidden.requires_grad is False
    hidden.requires_grad_(True)
    hidden.square().mean().backward()
    assert hidden.grad is not None
    assert torch.count_nonzero(hidden.grad) > 0


def test_shuffle_is_deterministic_derangement() -> None:
    left = deterministic_permutation(64, family="l0a")
    right = deterministic_permutation(64, family="l0a")
    assert left == right
    assert sorted(left) == list(range(64))
    assert all(index != value for index, value in enumerate(left))


def test_cost_gate_and_bootstrap_are_deterministic() -> None:
    first = bootstrap_mean_ci([1.0, 2.0, 3.0], draws=100)
    second = bootstrap_mean_ci([1.0, 2.0, 3.0], draws=100)
    assert first == second
    projection = project_cost_hours(
        target_seconds_per_row={"a": 0.1},
        margin_seconds_per_row=0.01,
        rows=2048,
        seeds=2,
        phase_a_cells_per_seed=11,
        phase_b_cells_per_seed=4,
    )
    assert projection["within_cap"] is True


def test_winner_requires_both_seeds_and_prefers_l0d_on_tie() -> None:
    cells = []
    for seed in (0, 1):
        for arm, mean, low in (
            ("l0a", 0.2, 0.1),
            ("l0d", 0.2, 0.1),
            ("l5_a", 0.0, -0.1),
            ("l5_d", 0.0, -0.1),
            ("l4", 0.0, -0.1),
        ):
            cells.append({"arm": arm, "seed": seed, "mean": mean, "ci_low": low})
    result = resolve_phase_a(cells)
    assert result["winner"] == "l0d"
