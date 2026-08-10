from __future__ import annotations

import torch
from torch import nn

from models.paper2_dc2_student import (
    Phase2StudentModules,
    Phase3PerPositionAnchoredBridge,
    Phase3StudentModules,
)
from training.paper2_phase3_migration import (
    PHASE3_NEW_GATE_PARAMETERS,
    migrate_phase2_trainable_state,
    phase3_trainable_parameter_count,
    trainable_state,
)


def test_position_gate_starts_as_exact_scalar_migration() -> None:
    torch.manual_seed(20260809)
    bridge = Phase3PerPositionAnchoredBridge(
        hidden_size=16,
        latent_dim=8,
        control_dim=6,
        max_steps=4,
        rms_cap=0.55,
    )
    with torch.no_grad():
        bridge.gate_logits.copy_(torch.tensor([-3.0, -2.0, -1.0, 0.0]))
    h0 = torch.randn(2, 7, 16)
    previous = torch.randn_like(h0)
    scratch = torch.randn(2, 8, 8)
    control = torch.randn(2, 6)

    output = bridge(
        h0=h0,
        previous=previous,
        scratch=scratch,
        control_state=control,
        loop_index=1,
    )
    scalar = torch.sigmoid(bridge.gate_logits[1])
    expected_gate = scalar.expand_as(output.position_gate[:, 1:])
    gate_mask = torch.ones_like(output.delta[..., :1])
    gate_mask[:, 0] = 0
    expected_hidden = h0 + output.rho * (previous - h0) + scalar * gate_mask * output.delta

    assert torch.equal(output.position_gate[:, 0], torch.zeros_like(output.position_gate[:, 0]))
    assert torch.equal(output.position_gate[:, 1:], expected_gate)
    assert torch.equal(output.gate, scalar)
    assert torch.equal(output.hidden, expected_hidden)


def test_position_gate_can_depend_on_position_scratch_and_control() -> None:
    torch.manual_seed(20260810)
    bridge = Phase3PerPositionAnchoredBridge(
        hidden_size=16, latent_dim=8, control_dim=6, max_steps=4
    )
    h0 = torch.randn(2, 7, 16)
    previous = torch.randn_like(h0)
    scratch = torch.randn(2, 8, 8)
    control = torch.randn(2, 6)
    baseline = bridge(
        h0=h0,
        previous=previous,
        scratch=scratch,
        control_state=control,
        loop_index=0,
    )
    with torch.no_grad():
        bridge.gate_hidden.weight.fill_(0.01)
        bridge.gate_scratch.weight.fill_(0.02)
        bridge.gate_control.weight.fill_(0.03)
    changed = bridge(
        h0=h0,
        previous=previous,
        scratch=scratch,
        control_state=control,
        loop_index=0,
    )

    assert not torch.equal(changed.position_gate[:, 1:], baseline.position_gate[:, 1:])
    assert torch.unique(changed.position_gate[:, 1:]).numel() > 1


def test_phase2_scalar_gate_migration_is_one_way_and_counted() -> None:
    torch.manual_seed(20260811)
    embedding = nn.Embedding(257, 896)
    embedding.requires_grad_(False)
    phase2 = Phase2StudentModules(tied_embedding=embedding, hidden_size=896)
    module = Phase3StudentModules(tied_embedding=embedding, hidden_size=896)
    source = trainable_state(module)
    for name in PHASE3_NEW_GATE_PARAMETERS:
        source.pop(name)
    source["bridge.gate_logits"] = torch.tensor([-4.25, -3.75, -3.25, -2.75])

    receipt = migrate_phase2_trainable_state(module, source)

    assert torch.equal(module.bridge.gate_logits.detach().cpu(), source["bridge.gate_logits"])
    assert receipt["new_parameters_are_zero"]
    assert receipt["phase2_trainable_parameter_count"] == 1_184_917
    assert phase3_trainable_parameter_count(phase2) == 1_184_917
    assert receipt["phase3_trainable_parameter_count"] == 1_185_973
    assert phase3_trainable_parameter_count(module) == 1_185_973
