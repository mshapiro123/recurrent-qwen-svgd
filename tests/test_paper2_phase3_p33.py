from __future__ import annotations

import torch
from torch import nn

from models.paper2_dc2_student import Phase3StudentModules
from training.paper2_phase3_p32 import GateLabel
from training.paper2_phase3_p33 import (
    P33_LOOK_INTERVAL,
    activate_operating_clamp,
    gate_classification,
    learning_rate_at_step,
    look_steps,
    p33_forward_losses,
    set_p33_trainable,
    weighted_total,
)


def _batch() -> dict[str, torch.Tensor]:
    torch.manual_seed(3)
    batch, horizons, candidates = 4, 4, 7
    ids = torch.randint(0, 257, (batch, horizons, candidates))
    mask = torch.ones_like(ids, dtype=torch.bool)
    base = torch.log_softmax(torch.randn(batch, horizons, candidates), dim=-1)
    labels = torch.tensor(
        [
            [1, 0, -1, 0],
            [0, 1, 0, -1],
            [1, 0, 0, -1],
            [0, -1, 1, 0],
        ],
        dtype=torch.long,
    )
    return {
        "hidden4": torch.randn(batch, horizons, 896),
        "candidate_ids": ids,
        "candidate_mask": mask,
        "base_candidates": base,
        "base_tail": torch.full((batch, horizons), -20.0),
        "gate_labels": labels,
        "oracle_directions": torch.randn(batch, horizons, 896),
        "position_bucket": torch.zeros(batch, dtype=torch.long),
    }


def test_registered_schedule_has_exactly_twenty_looks() -> None:
    assert look_steps() == list(range(P33_LOOK_INTERVAL, 1001, P33_LOOK_INTERVAL))
    assert learning_rate_at_step(50) == 1.5e-4
    assert learning_rate_at_step(100) == 3e-4
    assert learning_rate_at_step(1000) == 3e-4


def test_loss_path_trains_only_bridge_and_control_with_unclamped_gate_bce() -> None:
    embedding = nn.Embedding(257, 896)
    embedding.requires_grad_(False)
    module = Phase3StudentModules(tied_embedding=embedding, hidden_size=896)
    trainable = set_p33_trainable(module)
    assert trainable
    assert all(name.startswith(("bridge.", "control.")) for name in trainable)
    assert all(not name.startswith(("flow.", "draft.", "initializer.")) for name in trainable)
    clamp = activate_operating_clamp(module)
    assert clamp["ceiling"] == 0.02
    losses, metrics = p33_forward_losses(module=module, tied_embedding=embedding, **_batch())
    total = weighted_total(losses)
    assert torch.isfinite(total)
    total.backward()
    assert float(metrics["gate_probability_deployed"].detach().max()) <= 0.02
    assert module.bridge.gate_logits.grad is not None
    assert bool(torch.isfinite(module.bridge.gate_logits.grad).all())


def test_gate_metrics_use_unclamped_half_probability_threshold() -> None:
    probabilities = torch.tensor([0.9, 0.4, 0.7, 0.1])
    labels = torch.tensor(
        [GateLabel.POSITIVE, GateLabel.POSITIVE, GateLabel.NEGATIVE, GateLabel.NEGATIVE]
    )
    result = gate_classification(probabilities, labels)
    assert result["recall"] == 0.5
    assert result["precision"] == 0.5
    assert result["false_positive_rate"] == 0.5
