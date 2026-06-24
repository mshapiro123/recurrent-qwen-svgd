import math

import pytest
import torch

from eval.pathway_diversity import (
    effective_pathways,
    effective_pathways_from_similarity,
    gaussian_similarity,
    median_nearest_neighbor_sigma,
)


def test_similarity_sensitive_diversity_identity_counts_all_pathways():
    similarity = torch.eye(5)

    out = effective_pathways_from_similarity(similarity, qs=(0.0, 1.0, 2.0, math.inf))

    assert out == pytest.approx({"0": 5.0, "1": 5.0, "2": 5.0, "inf": 5.0})


def test_similarity_sensitive_diversity_all_similar_collapses_to_one():
    similarity = torch.ones(7, 7)

    out = effective_pathways_from_similarity(similarity, qs=(0.0, 1.0, 2.0, math.inf))

    assert out == pytest.approx({"0": 1.0, "1": 1.0, "2": 1.0, "inf": 1.0})


def test_effective_pathways_recovers_separated_basins():
    generator = torch.Generator().manual_seed(123)
    centers = torch.tensor(
        [
            [-8.0, 0.0],
            [-2.0, 0.0],
            [2.0, 0.0],
            [8.0, 0.0],
        ]
    )
    states = torch.cat([center + 0.03 * torch.randn(6, 2, generator=generator) for center in centers])

    out, diagnostics = effective_pathways(states)

    assert out["2"] > 3.0
    assert out["1"] > 3.0
    assert diagnostics["median_nearest_neighbor"] > 0.0


def test_effective_pathways_discounts_near_duplicates():
    generator = torch.Generator().manual_seed(456)
    regular_centers = torch.tensor(
        [
            [0.0, 0.0],
            [4.0, 0.0],
            [8.0, 0.0],
            [12.0, 0.0],
        ]
    )
    duplicate_centers = torch.tensor(
        [
            [0.0, 0.0],
            [0.04, 0.0],
            [8.0, 0.0],
            [12.0, 0.0],
        ]
    )
    regular = torch.cat([center + 0.02 * torch.randn(5, 2, generator=generator) for center in regular_centers])
    duplicate = torch.cat([center + 0.02 * torch.randn(5, 2, generator=generator) for center in duplicate_centers])

    regular_out, _ = effective_pathways(regular)
    duplicate_out, _ = effective_pathways(duplicate)

    assert regular_out["2"] > duplicate_out["2"]


def test_gaussian_similarity_rejects_bad_shapes():
    with pytest.raises(ValueError, match="at least one pathway"):
        gaussian_similarity(torch.empty(0, 4))

    with pytest.raises(ValueError, match="rank-2"):
        median_nearest_neighbor_sigma(torch.zeros(2, 3, 4))
