from __future__ import annotations

import torch

from models.ablation_lm.scratch import PositionAlignedScratch


def _scratch() -> PositionAlignedScratch:
    return PositionAlignedScratch(
        16,
        lane_width=4,
        max_steps=8,
        layer_scale=1e-3,
        rho_init=0.005,
        retention_floor=0.9,
        norm_eps=1e-5,
        use_carrier=False,
    )


def test_bicameral_lane_update_reduces_exactly_to_shared_update() -> None:
    torch.manual_seed(11)
    scratch = _scratch().eval()
    hidden = torch.randn(2, 5, 16)
    lanes = scratch.initialize(hidden)

    shared = scratch.step(lanes, hidden, step_index=2, residual_scale=0.25)
    paired = scratch.step_bicameral(
        lanes,
        hidden,
        hidden,
        step_index=2,
        residual_scale=0.25,
    )

    assert torch.equal(shared, paired)


def test_bicameral_lane_update_has_no_cross_hemisphere_or_future_read() -> None:
    torch.manual_seed(17)
    scratch = _scratch().eval()
    anchor = torch.randn(1, 5, 16)
    lanes = scratch.initialize(anchor).detach()
    h_a = torch.randn(1, 5, 16, requires_grad=True)
    h_b = torch.randn(1, 5, 16, requires_grad=True)

    updated = scratch.step_bicameral(
        lanes,
        h_a,
        h_b,
        step_index=1,
        residual_scale=0.5,
    )
    grad_a, grad_b = torch.autograd.grad(updated[0, 2, 0].sum(), (h_a, h_b))

    assert torch.count_nonzero(grad_a[0, 2]) > 0
    assert torch.count_nonzero(grad_a[0, :2]) == 0
    assert torch.count_nonzero(grad_a[0, 3:]) == 0
    assert torch.count_nonzero(grad_b) == 0
