from __future__ import annotations

import torch
from torch import nn

from models.paper2_dc2_student import Phase3StudentModules
from training.paper2_phase3_p33_i1 import (
    P33_I1_AIM_SHARE_FLOOR,
    P33_I1_GATE_NAMES,
    P33_I1_PRESERVATION_SHARE_CEILING,
    build_p33_i1_adamw_groups,
    i1_result_band,
    p33_i1_total,
    postclip_gradient_shares,
    set_p33_i1_trainable,
)


def test_i1_freezes_complete_selector_and_exposes_only_direction() -> None:
    embedding = nn.Embedding(257, 896)
    embedding.requires_grad_(False)
    module = Phase3StudentModules(tied_embedding=embedding, hidden_size=896)
    trainable = set_p33_i1_trainable(module)
    assert sum(parameter.numel() for parameter in trainable.values()) == 114_688
    assert set(trainable) == {"bridge.output_projection.weight"}
    groups = build_p33_i1_adamw_groups(trainable, weight_decay=0.01)
    assert groups[0]["weight_decay"] == 0.01
    assert all(name.startswith("bridge.") for name in trainable)
    assert all(not dict(module.named_parameters())[name].requires_grad for name in P33_I1_GATE_NAMES)
    assert all(
        not parameter.requires_grad
        for name, parameter in module.named_parameters()
        if name.startswith("control.")
    )


def test_i1_loss_has_no_gate_term_and_share_is_postclip() -> None:
    embedding = nn.Embedding(257, 896)
    embedding.requires_grad_(False)
    module = Phase3StudentModules(tied_embedding=embedding, hidden_size=896)
    trainable = list(set_p33_i1_trainable(module).values())
    left = trainable[0].float().square().mean()
    losses = {"aim": left, "preserve": 0.1 * left}
    total = p33_i1_total(losses, aim_weight=2.0)
    total.backward(retain_graph=True)
    assert torch.isfinite(total)
    audit = postclip_gradient_shares(
        losses=losses,
        module=module,
        parameters=trainable,
        aim_weight=2.0,
    )
    assert set(audit["shares"]) == {"aim", "preserve"}
    assert abs(sum(audit["shares"].values()) - 1.0) < 1e-8
    assert P33_I1_AIM_SHARE_FLOOR == 0.70
    assert P33_I1_PRESERVATION_SHARE_CEILING == 0.25


def test_i1_trainable_projection_cannot_change_gate_outputs() -> None:
    torch.manual_seed(11)
    embedding = nn.Embedding(257, 896)
    embedding.requires_grad_(False)
    module = Phase3StudentModules(tied_embedding=embedding, hidden_size=896)
    set_p33_i1_trainable(module)
    hidden = torch.randn(2, 5, 896)
    previous = torch.randn(2, 4, 7)
    attention = torch.ones(2, 5, dtype=torch.bool)
    attention[:, 0] = False
    candidates = torch.randint(0, 257, (2, 4, 7))
    positions = torch.zeros(2, dtype=torch.long)
    before = module(
        hidden=hidden,
        previous_logits=previous,
        steps=1,
        attention_mask=attention,
        position_bucket=positions,
        candidate_ids=candidates,
    ).bridge.position_gate_unclamped
    with torch.no_grad():
        module.bridge.output_projection.weight.add_(0.1 * torch.randn_like(module.bridge.output_projection.weight))
    after = module(
        hidden=hidden,
        previous_logits=previous,
        steps=1,
        attention_mask=attention,
        position_bucket=positions,
        candidate_ids=candidates,
    ).bridge.position_gate_unclamped
    assert torch.equal(before, after)


def test_i1_result_bands_are_locked() -> None:
    assert i1_result_band(0.25) == "p34_charter_funded"
    assert i1_result_band(0.14901) == "middle_band_strategy_review"
    assert i1_result_band(0.04999) == "boundary_diagnostic"
