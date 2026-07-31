from __future__ import annotations

import pytest
import torch

from eval.eval_paper2_phase2_v1_v2 import (
    bound_compatible_fractions,
    finite_difference_directional_gain,
    quantile_summary,
)


def test_centered_finite_difference_gain_recovers_linear_direction() -> None:
    matrix = torch.tensor([[3.0, 0.0], [0.0, 2.0]])
    state = torch.tensor([1.0, -1.0])
    direction = torch.tensor([1.0, 0.0])

    gain = finite_difference_directional_gain(
        lambda value: matrix @ value,
        state,
        direction,
        epsilon=1e-3,
    )

    assert gain == pytest.approx(3.0, rel=1e-4)


def test_bound_compatible_fraction_uses_position_matched_scale_and_gain() -> None:
    result = bound_compatible_fractions(
        margins=[0.2, 1.0],
        sampled_max_gains=[2.0, 1.0],
        state_rms=[1.0, 2.0],
        hidden_size=4,
        c_values=[0.01, 0.05],
        gamma=0.5,
        rho=0.5,
    )

    # Bound = gain * gamma * c * RMS * sqrt(d) / (1-rho).
    assert result["0.01"]["compatible"] == 0
    assert result["0.05"]["compatible"] == 1
    assert result["0.05"]["fraction"] == 0.5


def test_quantile_summary_is_stable_for_small_samples() -> None:
    summary = quantile_summary([1.0, 2.0, 3.0, 4.0])
    assert summary["count"] == 4
    assert summary["min"] == 1.0
    assert summary["median"] == pytest.approx(2.5)
    assert summary["max"] == 4.0
