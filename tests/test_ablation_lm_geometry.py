from __future__ import annotations

import pytest
import torch

from models.ablation_lm.geometry import (
    Cl20Rotor,
    cl20_rotate,
    lanes_to_modes,
)
from models.ablation_lm.jets import (
    LoopGradientProbe,
    estimate_jacobian_spectral_norm,
    first_order_jet,
    loop_gradient_metrics,
    plane_probe_features,
    trajectory_jet_metrics,
)
from models.ablation_lm.scratch import TwoLaneBirkhoffMixer


def test_lane_mode_coordinates_round_trip_exactly() -> None:
    lanes = torch.randn(2, 3, 2, 5, dtype=torch.float64)
    coordinates = lanes_to_modes(lanes)

    torch.testing.assert_close(coordinates.lanes(), lanes, rtol=1e-6, atol=1e-7)
    lane_energy = lanes.float().square().sum(dim=(-2, -1))
    coordinate_energy = 2 * (
        coordinates.mu.float().square().sum(dim=-1)
        + coordinates.delta.float().square().sum(dim=-1)
    )
    torch.testing.assert_close(lane_energy, coordinate_energy)


def test_t5_euclidean_cl20_rotor_preserves_norm() -> None:
    vector = torch.randn(64, 2, dtype=torch.float64)
    rotated = cl20_rotate(vector, torch.tensor(0.73, dtype=torch.float64))
    relative_error = (rotated.norm() / vector.norm() - 1).abs()

    assert relative_error.item() < 1e-6
    rotor = Cl20Rotor.from_angle(torch.tensor(0.73, dtype=torch.float64))
    torch.testing.assert_close(
        rotor.scalar.square() + rotor.bivector.square(),
        torch.tensor(1.0, dtype=torch.float64),
    )
    with torch.no_grad():
        rotor.angle.add_(1.0)
    mutated_rotation = rotor.rotate(vector)
    torch.testing.assert_close(mutated_rotation.norm(), vector.norm())
    for dtype in (torch.float16, torch.bfloat16):
        low_precision = Cl20Rotor.from_angle(torch.tensor(-9.796875, dtype=dtype))
        assert low_precision.angle.dtype is torch.float32
        assert torch.isfinite(low_precision.scalar)

    angles = torch.linspace(-1.0, 1.0, 8, dtype=torch.float64)
    vectors = torch.randn(8, 2, dtype=torch.float64)
    vmapped = torch.vmap(cl20_rotate)(vectors, angles)
    torch.testing.assert_close(vmapped.norm(dim=-1), vectors.norm(dim=-1))
    compiled = torch.compile(cl20_rotate, backend="eager", fullgraph=True)
    torch.testing.assert_close(compiled(vectors, angles), cl20_rotate(vectors, angles))

    batched_vectors = torch.randn(2, 3, 2, dtype=torch.float64)
    batched_angles = torch.randn(2, 3, dtype=torch.float64)
    batched_rotated = cl20_rotate(batched_vectors, batched_angles)
    assert batched_rotated.shape == batched_vectors.shape
    torch.testing.assert_close(
        batched_rotated.norm(dim=-1),
        batched_vectors.norm(dim=-1),
    )
    with pytest.raises(ValueError, match="exactly match"):
        cl20_rotate(batched_vectors, torch.randn(3, dtype=torch.float64))
    with pytest.raises(ValueError, match="exactly match"):
        cl20_rotate(torch.randn(2, 2), torch.randn(2, 1))


def test_two_lane_birkhoff_mixer_preserves_mu_and_damps_delta() -> None:
    mixer = TwoLaneBirkhoffMixer(rho_init=0.005, max_steps=4, retention_floor=0.9)
    lanes = torch.randn(2, 4, 2, 8)
    before = lanes_to_modes(lanes)
    after = lanes_to_modes(mixer(lanes))
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
    repeated_coordinates = lanes_to_modes(repeated)
    torch.testing.assert_close(repeated_coordinates.mu, before.mu)
    torch.testing.assert_close(
        repeated_coordinates.delta,
        (1 - 2 * rho) ** 3 * before.delta,
    )
    product = torch.linalg.matrix_power(matrix, 3)
    torch.testing.assert_close(product.sum(dim=0), torch.ones(2))
    torch.testing.assert_close(product.sum(dim=1), torch.ones(2))
    assert mixer.minimum_retention(4).item() >= 0.9
    assert (1 - 2 * mixer.rho()).item() == pytest.approx(
        torch.linalg.svdvals(matrix).min().item()
    )


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
    colinear = torch.tensor([[[0.0, 0.0]], [[1.0, 0.0]], [[3.0, 0.0]]])
    stationary = torch.zeros(3, 1, 2)
    turning = torch.tensor([[[0.0, 0.0]], [[1.0, 0.0]], [[1.0, 1.0]]])

    straight_metrics = trajectory_jet_metrics(straight)
    colinear_metrics = trajectory_jet_metrics(colinear)
    stationary_metrics = trajectory_jet_metrics(stationary)
    turning_metrics = trajectory_jet_metrics(turning)

    assert straight_metrics.wedge_gram.item() == 0.0
    assert turning_metrics.wedge_gram.item() > 0.0
    assert straight_metrics.curvature.item() == 0.0
    assert turning_metrics.curvature.item() > 0.0
    assert colinear_metrics.gram_eigenvalue_ratio.item() == 0.0
    assert stationary_metrics.gram_eigenvalue_ratio.item() == 0.0
    assert turning_metrics.gram_eigenvalue_ratio.item() > 0.0


def test_loop_gradient_cosines_distinguish_aligned_and_opposed_visits() -> None:
    metrics = loop_gradient_metrics(
        (
            torch.tensor([1.0, 0.0]),
            torch.tensor([2.0, 0.0]),
            torch.tensor([-1.0, 0.0]),
        )
    )

    torch.testing.assert_close(metrics.adjacent_cosines, torch.tensor([1.0, -1.0]))
    assert metrics.pairwise_cosines.shape == (3, 3)
    assert bool(metrics.gradient_rms.gt(0).all())

    first = torch.zeros(1, 2, 2, requires_grad=True)
    second = torch.zeros(1, 2, 2, requires_grad=True)
    first.grad = torch.tensor([[[1.0, 0.0], [1_000.0, 0.0]]])
    second.grad = torch.tensor([[[2.0, 0.0], [-1_000.0, 0.0]]])
    masked = LoopGradientProbe(
        (first, second),
        torch.tensor([[True, False]]),
    ).metrics()
    torch.testing.assert_close(masked.adjacent_cosines, torch.tensor([1.0]))


def test_plane_probes_use_velocity_and_acceleration_together() -> None:
    velocity = torch.tensor([[1.0, 0.0, 0.0]])
    acceleration = torch.tensor([[0.0, 1.0, 0.0]])
    p = torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    q = torch.tensor([[0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])

    features = plane_probe_features(velocity, acceleration, p, q)

    assert features.shape == (1, 2)
    assert features.abs().sum().item() > 0
