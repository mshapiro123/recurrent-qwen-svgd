from __future__ import annotations

import torch

from training.paper2_bicameral_w1 import (
    ORACLE_TARGET_ASSISTED,
    POPULATION_TARGET,
    bootstrap_mean_ci,
    build_crossfitted_residual_directions,
    build_phase_b_granularity_targets,
    deterministic_permutation,
    extend_frozen_centroids,
    project_cost_hours,
    orient_residual_directions,
    resolve_phase_a,
    scale_external_write,
    validate_cluster_extension,
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


def test_phase_b_granularity_targets_use_frozen_k2_assignments() -> None:
    targets = torch.tensor([[1.0, 0.0], [3.0, 2.0], [9.0, 4.0]])
    assignments = torch.tensor([0, 0, 1])
    result = build_phase_b_granularity_targets(targets, assignments)

    assert torch.equal(result["cluster_means"], torch.tensor([[2.0, 1.0], [9.0, 4.0]]))
    assert torch.equal(result["l1"], torch.tensor([[2.0, 1.0], [2.0, 1.0], [9.0, 4.0]]))
    assert torch.equal(result["l2"], torch.tensor([[13 / 3, 2.0]]).expand(3, -1))
    assert torch.equal(result["l3"], torch.tensor([[9.0, 4.0], [9.0, 4.0], [2.0, 1.0]]))


def test_phase_b_granularity_targets_reject_non_k2_labels() -> None:
    with torch.no_grad():
        try:
            build_phase_b_granularity_targets(
                torch.ones(3, 2), torch.tensor([0, 1, 2])
            )
        except ValueError as error:
            assert "k=2" in str(error)
        else:
            raise AssertionError("non-k2 Phase B assignments were accepted")


def test_frozen_centroid_extension_gate_and_target_tags() -> None:
    receipt = validate_cluster_extension(torch.tensor([0] * 100 + [1] * 1900))
    assert receipt["passed"] is True
    assert receipt["fractions"] == [0.05, 0.95]
    assert ORACLE_TARGET_ASSISTED == "oracle-target-assisted"
    assert POPULATION_TARGET == "population-target"

    try:
        validate_cluster_extension(torch.tensor([0] * 99 + [1] * 1901))
    except RuntimeError as error:
        assert "extension gate failed" in str(error)
    else:
        raise AssertionError("degenerate frozen-centroid extension was accepted")


def test_l6_orientation_is_nonnegative_against_correction_mean() -> None:
    directions = torch.tensor([[1.0, 0.0], [-1.0, 1.0], [0.0, 1.0]])
    oriented, receipt = orient_residual_directions(
        directions, torch.tensor([1.0, 0.0])
    )
    assert torch.equal(oriented, torch.tensor([[1.0, 0.0], [1.0, -1.0], [0.0, 1.0]]))
    assert receipt["orientation_signs"] == [1, -1, 1]
    assert all(value >= 0 for value in receipt["oriented_inner_products"])


def test_frozen_centroid_extension_uses_cosine_without_refitting() -> None:
    features = torch.tensor([[2.0, 0.0], [0.0, 4.0], [3.0, 1.0], [1.0, 3.0]])
    centroids = torch.eye(2)
    assignments, receipt = extend_frozen_centroids(features, centroids)
    assert torch.equal(assignments, torch.tensor([0, 1, 0, 1]))
    assert receipt["counts"] == [2, 2]
    assert receipt["assignment"] == "nearest_frozen_stage0_centroid_no_refit"


def test_crossfitted_residual_directions_are_deterministic_and_oriented() -> None:
    generator = torch.Generator().manual_seed(7)
    corrections = torch.randn(40, 12, generator=generator)
    corrections[:20, 0] += 2.0
    corrections[20:, 1] += 2.0
    labels = torch.tensor([0] * 20 + [1] * 20)
    first, first_receipt = build_crossfitted_residual_directions(
        corrections,
        labels,
        directions=3,
        splits=3,
        nuisance_rank=2,
    )
    second, second_receipt = build_crossfitted_residual_directions(
        corrections,
        labels,
        directions=3,
        splits=3,
        nuisance_rank=2,
    )
    assert torch.equal(first, second)
    assert first_receipt == second_receipt
    assert first.shape == (3, 12)
    assert torch.allclose(first @ first.T, torch.eye(3), atol=1e-5)
    assert all(value >= 0 for value in first_receipt["orientation"]["oriented_inner_products"])
