from __future__ import annotations

import json
from pathlib import Path

import torch
from torch import nn

from models.paper2_dc2_student import Phase2StudentModules
from training.run_paper2_phase2_a2_calibration import (
    LOSSES,
    _a2_losses,
    _distribution,
    _set_a2_trainable,
)


ROOT = Path(__file__).resolve().parents[1]


def _tiny_batch(*, batch_size: int = 2, hidden_size: int = 16, vocab: int = 31):
    horizons, candidates = 4, 5
    candidate_ids = torch.randint(0, vocab, (batch_size, horizons, candidates))
    candidate_mask = torch.ones_like(candidate_ids, dtype=torch.bool)
    return {
        "hidden": torch.randn(batch_size, horizons, hidden_size),
        "candidate_ids": candidate_ids,
        "candidate_mask": candidate_mask,
        "base_candidates": torch.log_softmax(
            torch.randn(batch_size, horizons, candidates), dim=-1
        ),
        "base_tail": torch.full((batch_size, horizons), -4.0),
        "teacher_candidates": torch.log_softmax(
            torch.randn(batch_size, horizons, candidates), dim=-1
        ),
        "teacher_tail": torch.full((batch_size, horizons), -4.0),
        "target_scratch": torch.randn(batch_size, 8, 128),
        "position_bucket": torch.zeros(batch_size, dtype=torch.long),
    }


def test_a2_trainable_surface_freezes_flow_initializer_and_tied_embedding() -> None:
    torch.manual_seed(12)
    embedding = nn.Embedding(31, 16)
    module = Phase2StudentModules(tied_embedding=embedding, hidden_size=16)
    _set_a2_trainable(module)
    trainable = {name for name, value in module.named_parameters() if value.requires_grad}
    assert trainable
    assert all(
        name.startswith(("bridge.", "control.", "draft.down.", "draft.up.", "draft.write_gate."))
        for name in trainable
    )
    assert not any(name.startswith(("flow.", "initializer.", "draft.tied_embedding.")) for name in trainable)


def test_a2_loss_graph_reaches_each_trainable_module_without_reaching_flow() -> None:
    torch.manual_seed(13)
    embedding = nn.Embedding(31, 16)
    module = Phase2StudentModules(tied_embedding=embedding, hidden_size=16)
    _set_a2_trainable(module)
    losses = _a2_losses(module=module, batch=_tiny_batch(), embedding=embedding)
    assert tuple(losses) == LOSSES
    assert all(torch.isfinite(value) for value in losses.values())
    sum(losses.values()).backward()
    for prefix in ("bridge.", "control.", "draft."):
        assert any(
            value.grad is not None and bool(torch.isfinite(value.grad).all())
            for name, value in module.named_parameters()
            if name.startswith(prefix) and value.requires_grad
        )
    assert all(value.grad is None for value in module.flow.parameters())
    assert all(value.grad is None for value in module.initializer.parameters())


def test_distribution_is_complete_and_deterministic() -> None:
    result = _distribution([1.0, 2.0, 3.0, 4.0])
    assert result["count"] == 4
    assert result["minimum"] == 1.0
    assert result["median"] == 2.5
    assert result["maximum"] == 4.0


def test_registration_authorizes_calibration_but_not_a2_training() -> None:
    registration = json.loads(
        (ROOT / "training/paper2_phase2_staged_repilot_preregistration.json").read_text(
            encoding="utf-8"
        )
    )
    assert registration["a1_strategy_bank_20260805"]["status"] == "banked_a1_pass"
    authorization = registration["a2_calibration_authorization_20260805"]
    assert authorization["status"] == "authorized_zero_update_prelock"
    assert authorization["optimizer_updates"] == 0
    assert authorization["a2_training_authorized"] is False
    assert authorization["strategy_review_required_before_a2_training"] is True


def test_runner_contains_no_optimizer_or_training_launch() -> None:
    source = (ROOT / "training/run_paper2_phase2_a2_calibration.py").read_text(
        encoding="utf-8"
    )
    assert "torch.optim" not in source
    assert '"optimizer_updates": 0' in source
    assert '"a2_training_launched": False' in source
    assert "strategy_review_required_before_a2_training" in source
    assert "draft_head_only_control_launched" in source
