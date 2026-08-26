from __future__ import annotations

import torch

from models.ablation_lm.geometry import (
    SplitCliffordState,
    lanes_to_split_clifford,
    split_clifford_product,
)
from models.ablation_lm.jets import (
    estimate_jacobian_spectral_norm,
    first_order_jet,
    trajectory_jet_metrics,
)
from models.ablation_lm.scratch import TwoLaneBirkhoffMixer


def test_split_clifford_coordinates_round_trip_exactly() -> None:
    lanes = torch.randn(2, 3, 2, 5)
    coordinates = lanes_to_split_clifford(lanes)

    torch.testing.assert_close(coordinates.lanes(), lanes, rtol=1e-6, atol=1e-7)
    lane_energy = lanes.float().square().sum(dim=(-2, -1))
    coordinate_energy = 2 * (
        coordinates.mu.float().square().sum(dim=-1)
        + coordinates.delta.float().square().sum(dim=-1)
    )
    torch.testing.assert_close(lane_energy, coordinate_energy)


def test_split_clifford_product_uses_e_squared_equals_one() -> None:
    left = SplitCliffordState(torch.tensor(2.0), torch.tensor(3.0))
    right = SplitCliffordState(torch.tensor(5.0), torch.tensor(7.0))
    product = split_clifford_product(left, right)

    assert product.mu.item() == 31.0
    assert product.delta.item() == 29.0


def test_two_lane_birkhoff_mixer_preserves_mu_and_damps_delta() -> None:
    mixer = TwoLaneBirkhoffMixer(rho_init=0.125)
    lanes = torch.randn(2, 4, 2, 8)
    before = lanes_to_split_clifford(lanes)
    after = lanes_to_split_clifford(mixer(lanes))
    rho = mixer.rho().detach()
    matrix = mixer.matrix().detach()

    torch.testing.assert_close(after.mu, before.mu)
    torch.testing.assert_close(after.delta, (1 - 2 * rho) * before.delta)
    torch.testing.assert_close(matrix.sum(dim=0), torch.ones(2))
    torch.testing.assert_close(matrix.sum(dim=1), torch.ones(2))
    assert bool(matrix.ge(0).all())
    assert torch.linalg.svdvals(matrix).max().item() <= 1.0 + 1e-6

    repeated = lanes
    for _ in range(3):
        repeated = mixer(repeated)
    repeated_coordinates = lanes_to_split_clifford(repeated)
    torch.testing.assert_close(repeated_coordinates.mu, before.mu)
    torch.testing.assert_close(
        repeated_coordinates.delta,
        (1 - 2 * rho) ** 3 * before.delta,
    )
    product = torch.linalg.matrix_power(matrix, 3)
    torch.testing.assert_close(product.sum(dim=0), torch.ones(2))
    torch.testing.assert_close(product.sum(dim=1), torch.ones(2))


def test_first_order_jets_match_linear_map_and_recover_spectral_norm() -> None:
    matrix = torch.tensor([[3.0, 0.0], [0.0, 1.0]])
    function = lambda values: values @ matrix.T
    primal = torch.tensor([[0.5, -2.0]])
    tangent = torch.tensor([[2.0, 4.0]])

    jet = first_order_jet(function, primal, tangent)
    estimate = estimate_jacobian_spectral_norm(function, primal, iterations=12, seed=11)

    torch.testing.assert_close(jet.primal, function(primal))
    torch.testing.assert_close(jet.tangent, function(tangent))
    torch.testing.assert_close(estimate, torch.tensor(3.0), rtol=1e-4, atol=1e-4)


def test_trajectory_jet_wedge_is_zero_for_straight_motion_and_positive_for_turning() -> None:
    straight = torch.tensor([[[0.0, 0.0]], [[1.0, 0.0]], [[2.0, 0.0]]])
    turning = torch.tensor([[[0.0, 0.0]], [[1.0, 0.0]], [[1.0, 1.0]]])

    straight_metrics = trajectory_jet_metrics(straight)
    turning_metrics = trajectory_jet_metrics(turning)

    assert straight_metrics.wedge_gram.item() == 0.0
    assert turning_metrics.wedge_gram.item() > 0.0
