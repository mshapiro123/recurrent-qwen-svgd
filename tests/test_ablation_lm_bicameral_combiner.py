from __future__ import annotations

import pytest
import torch

from models.ablation_lm.bicameral_combiner import PerBandUnitCircleCombiner


def test_s2_initializes_at_consensus_and_angle_gradient_reads_disagreement() -> None:
    combiner = PerBandUnitCircleCombiner(16, num_bands=4)
    h_a = torch.randn(2, 5, 16)
    h_b = torch.randn(2, 5, 16)
    expected_consensus = (h_a + h_b) / 2.0

    output = combiner(h_a, h_b)
    torch.testing.assert_close(output, expected_consensus, rtol=2e-6, atol=2e-6)
    output.square().mean().backward()
    assert combiner.theta.grad is not None
    assert bool(torch.isfinite(combiner.theta.grad).all())
    assert bool(combiner.theta.grad.ne(0).any())


def test_s2_unit_circle_coefficients_and_joint_state_nonexpansiveness() -> None:
    combiner = PerBandUnitCircleCombiner(32, num_bands=8)
    with torch.no_grad():
        combiner.theta.copy_(torch.linspace(-2.0, 2.0, 8))
    cosine, sine = combiner.unit_circle_coefficients()
    torch.testing.assert_close(cosine.square() + sine.square(), torch.ones_like(cosine))

    h_a = torch.randn(3, 7, 32)
    h_b = torch.randn(3, 7, 32)
    consensus = (h_a + h_b) / 2.0
    disagreement = (h_a - h_b) / 2.0
    output = combiner(h_a, h_b)
    joint_norm = torch.sqrt(consensus.square().sum() + disagreement.square().sum())
    assert output.norm() <= joint_norm * (1.0 + 2e-6)


def test_s2_preserves_dense_identity_when_hemispheres_coincide() -> None:
    combiner = PerBandUnitCircleCombiner(16, num_bands=4)
    hidden = torch.randn(2, 4, 16)
    output = combiner(hidden, hidden)
    torch.testing.assert_close(output, hidden, rtol=2e-6, atol=2e-6)


def test_s2_keeps_theta_fp32_across_module_dtype_conversion() -> None:
    combiner = PerBandUnitCircleCombiner(16, num_bands=4).to(dtype=torch.bfloat16)
    assert combiner.theta.dtype is torch.float32
    hidden = torch.randn(1, 3, 16, dtype=torch.bfloat16)
    assert combiner(hidden, hidden).dtype is torch.bfloat16


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"d_model": 12}, "positive power of two"),
        ({"d_model": 16, "num_bands": 3}, "positive power of two"),
        ({"d_model": 16, "num_bands": 32}, "must divide"),
    ],
)
def test_s2_configuration_fails_closed(kwargs: dict[str, int], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        PerBandUnitCircleCombiner(**kwargs)


def test_s2_input_contract_fails_closed() -> None:
    combiner = PerBandUnitCircleCombiner(16, num_bands=4)
    hidden = torch.randn(1, 3, 16)
    with pytest.raises(ValueError, match="identical shapes"):
        combiner(hidden, hidden[:, :2])
    with pytest.raises(ValueError, match="final width"):
        combiner(hidden[..., :8], hidden[..., :8])
    with pytest.raises(ValueError, match="finite"):
        invalid = hidden.clone()
        invalid[0, 0, 0] = float("nan")
        combiner(invalid, hidden)
