from __future__ import annotations

import pytest
import torch

from eval.eval_paper2_phase3_p34_ar import (
    leading_covariance_basis,
    numerical_column_basis,
    projected_energy,
)


def test_ar_exactly_prices_known_readout_span() -> None:
    projection = torch.tensor([[1.0, 0.0], [0.0, 1.0], [0.0, 0.0]])
    basis, receipt = numerical_column_basis(projection)
    directions = torch.tensor([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    energy = projected_energy(directions, basis)
    assert receipt["rank"] == 2
    assert energy["aggregate_energy_fraction"] == pytest.approx(0.5)
    assert energy["mean_row_energy_fraction"] == pytest.approx(0.5)


def test_matched_covariance_basis_uses_requested_rank() -> None:
    states = torch.tensor(
        [[-3.0, 0.0, 0.0], [-1.0, 0.1, 0.0], [1.0, -0.1, 0.0], [3.0, 0.0, 0.0]]
    )
    basis, receipt = leading_covariance_basis(states, rank=1)
    assert basis.shape == (3, 1)
    assert receipt["rank"] == 1
    assert receipt["variance_fraction"] > 0.99
