from __future__ import annotations

import math

import pytest
import torch
from torch import nn

from models.paper2_dc2_student import (
    AnchoredBridge,
    ControlState,
    Phase2StudentModules,
    ResidualDraftHead,
    ScratchpadInitializer,
    SharedResidualFlow,
    masked_effective_rank,
    masked_huber_loss,
)


def test_scratchpad_initializer_is_anchor_dominated_and_shape_stable() -> None:
    torch.manual_seed(7)
    module = ScratchpadInitializer(hidden_size=16, latent_dim=8, n_slots=8)
    hidden = torch.randn(3, 11, 16)
    scratch = module(hidden)

    assert scratch.shape == (3, 8, 8)
    anchor = module.anchors.unsqueeze(0).expand_as(scratch)
    assert float((scratch - anchor).abs().max().detach()) < 0.02


def test_residual_flow_uses_scalar_softplus_magnitude_without_state_projection() -> None:
    torch.manual_seed(11)
    module = SharedResidualFlow(latent_dim=8, context_dim=16, n_slots=8, max_steps=4)
    state = torch.randn(2, 8, 8) * 3.0
    context = torch.randn(2, 16)

    output = module(state, context, steps=2)

    assert output.state.shape == state.shape
    assert output.magnitudes.shape == (2, 2)
    assert torch.all(output.magnitudes > 0)
    assert torch.allclose(
        output.magnitudes[:, 0],
        torch.full((2,), torch.nn.functional.softplus(torch.tensor(-4.0))),
        atol=1e-6,
    )
    assert torch.allclose(module.direction_norm.weight, torch.ones(8))
    assert not torch.allclose(
        output.state.square().mean(dim=-1).sqrt(),
        torch.ones(2, 8),
        atol=1e-3,
    )
    assert output.trust_penalty.ndim == 0
    assert output.initial_update_ratio.shape == (2,)


def test_residual_flow_enforces_registered_loop_cap() -> None:
    module = SharedResidualFlow(latent_dim=8, context_dim=16, n_slots=8, max_steps=4)
    with pytest.raises(ValueError, match="loop cap"):
        module(torch.randn(1, 8, 8), torch.randn(1, 16), steps=5)


def test_anchored_bridge_has_exact_inactive_identity_and_closes_position_zero() -> None:
    torch.manual_seed(13)
    module = AnchoredBridge(hidden_size=16, latent_dim=8, max_steps=4, rms_cap=0.55)
    h0 = torch.randn(2, 6, 16)
    previous = torch.randn(2, 6, 16)
    scratch = torch.randn(2, 8, 8)

    inactive = module(h0=h0, previous=previous, scratch=scratch, loop_index=0, active=False)
    active = module(h0=h0, previous=previous, scratch=scratch, loop_index=0, active=True)

    assert torch.equal(inactive.hidden, previous)
    assert torch.equal(active.hidden[:, 0], h0[:, 0] + 0.95 * (previous[:, 0] - h0[:, 0]))
    assert active.position_zero_gate_closed
    assert module.output_projection.weight.detach().abs().sum() > 0


def test_masked_slots_are_excluded_from_loss_and_effective_rank() -> None:
    prediction = torch.zeros(2, 8, 4)
    target = torch.zeros_like(prediction)
    mask = torch.zeros(2, 8, dtype=torch.bool)
    mask[:, :4] = True
    prediction[:, 4:] = 1_000.0

    assert float(masked_huber_loss(prediction, target, mask)) == 0.0

    populated = torch.zeros(2, 8, 4)
    populated[:, 0, 0] = 1
    populated[:, 1, 1] = 1
    populated[:, 2, 2] = 1
    populated[:, 3, 3] = 1
    populated[:, 4:] = 1000 * torch.randn(2, 4, 4)
    rank = masked_effective_rank(populated, mask)
    expected = math.exp(-4 * 0.25 * math.log(0.25))
    assert float(rank) == pytest.approx(expected, rel=1e-5)


def test_control_state_and_draft_head_are_stage_c_ready() -> None:
    torch.manual_seed(17)
    embedding = nn.Embedding(31, 16)
    control = ControlState(latent_dim=8, control_dim=6)
    head = ResidualDraftHead(
        tied_embedding=embedding,
        latent_dim=8,
        control_dim=6,
        hidden_size=16,
        rank=4,
        horizons=4,
    )
    scratch = torch.randn(2, 8, 8)
    state = control(
        scratch=scratch,
        previous=None,
        innovation_norm=torch.tensor([0.2, 0.4]),
        student_entropy=torch.tensor([1.0, 1.2]),
        top2_margin=torch.tensor([0.1, 0.3]),
        position_bucket=torch.tensor([0, 3]),
    )
    base_logits = torch.randn(2, 4, 31)
    output = head(previous_logits=base_logits, scratch=scratch, control_state=state)

    assert state.shape == (2, 6)
    assert output.logits.shape == base_logits.shape
    assert output.write_gates.shape == (2, 4)
    assert torch.allclose(
        output.write_gates,
        torch.full((2, 4), torch.sigmoid(torch.tensor(-3.5))),
        atol=1e-6,
    )


def test_complete_student_build_has_no_loss_and_exposes_control_read() -> None:
    torch.manual_seed(19)
    embedding = nn.Embedding(37, 16)
    module = Phase2StudentModules(
        tied_embedding=embedding,
        hidden_size=16,
        latent_dim=8,
        n_slots=8,
        control_dim=6,
        draft_rank=4,
        max_steps=4,
        rms_cap=0.55,
    )
    hidden = torch.randn(2, 9, 16)
    base_logits = torch.randn(2, 4, 37)
    output = module(hidden=hidden, previous_logits=base_logits, steps=1)

    assert output.loss is None
    assert output.control_state.shape == (2, 6)
    assert output.control_read.shape == (2, 6)
    assert output.scratch.shape == (2, 8, 8)
    assert output.hidden.shape == hidden.shape
    assert output.logits.shape == base_logits.shape
