from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch
from torch import nn

from models.paper2_dc2_student import Phase2StudentModules
from training.paper2_phase2_staged_repilot import (
    PROTOCOL_LOCK_COMMIT,
    a1_gate,
    a1_should_extend,
    drift_alarm,
    realized_gradient_shares,
    shares_within_absolute_tolerance,
    solve_static_weights,
    trust_tripwire,
)
from training.run_paper2_phase2_staged_a1 import _a1_losses, _set_a1_trainable


ROOT = Path(__file__).resolve().parents[1]


def test_static_weights_hit_requested_independent_gradient_shares() -> None:
    norms = {"flow": 12.0, "functional_probe_kl": 3.0, "counterfactual_preserve_kl": 6.0}
    target = {"flow": 0.60, "functional_probe_kl": 0.20, "counterfactual_preserve_kl": 0.20}
    weights = solve_static_weights(norms, target, anchor="flow")
    assert weights["flow"] == pytest.approx(1.0)
    assert realized_gradient_shares(norms, weights) == pytest.approx(target)


def test_static_weight_contract_fails_closed_on_missing_or_zero_gradients() -> None:
    with pytest.raises(ValueError, match="same losses"):
        solve_static_weights({"flow": 1.0}, {"flow": 0.6, "probe": 0.4}, anchor="flow")
    with pytest.raises(ValueError, match="unusable"):
        solve_static_weights(
            {"flow": 1.0, "probe": 0.0},
            {"flow": 0.6, "probe": 0.4},
            anchor="flow",
        )


def test_share_tolerance_drift_and_trust_tripwire_boundaries() -> None:
    target = {"flow": 0.6, "probe": 0.2, "preserve": 0.2}
    assert shares_within_absolute_tolerance(
        {"flow": 0.7, "probe": 0.15, "preserve": 0.15}, target, tolerance=0.10
    )
    assert not shares_within_absolute_tolerance(
        {"flow": 0.71, "probe": 0.145, "preserve": 0.145}, target, tolerance=0.10
    )
    assert drift_alarm({"flow": 0.29, "probe": 0.3, "preserve": 0.41}, target, ratio=2.0)
    assert not trust_tripwire([True] * 50 + [False] * 50)
    assert trust_tripwire([True] * 51 + [False] * 49)


def test_a1_gate_and_single_extension_rule() -> None:
    assert a1_gate(
        initial_probe_kl=0.5,
        final_probe_kl=0.4,
        initial_flow_mse=2.0,
        final_flow_mse=1.9,
    )
    assert not a1_gate(
        initial_probe_kl=0.5,
        final_probe_kl=0.401,
        initial_flow_mse=2.0,
        final_flow_mse=1.9,
    )
    assert a1_should_extend(gate_passed=False, step_900_flow_loss=1.0, step_1000_flow_loss=0.4)
    assert a1_should_extend(gate_passed=True, step_900_flow_loss=1.0, step_1000_flow_loss=0.99)
    assert not a1_should_extend(gate_passed=True, step_900_flow_loss=1.0, step_1000_flow_loss=0.996)


def test_locked_registration_and_a1_runner_encode_stage_boundary() -> None:
    registration = json.loads(
        (ROOT / "training/paper2_phase2_staged_repilot_preregistration.json").read_text(
            encoding="utf-8"
        )
    )
    source = (ROOT / "training/run_paper2_phase2_staged_a1.py").read_text(encoding="utf-8")
    assert registration["status"] == "locked_before_training"
    assert registration["a1"]["trainable_modules"] == ["flow"]
    assert registration["a1"]["execution_gates_closed"] is True
    assert f'PROTOCOL_LOCK_COMMIT = "{PROTOCOL_LOCK_COMMIT}"' in (
        ROOT / "training/paper2_phase2_staged_repilot.py"
    ).read_text(encoding="utf-8")
    assert "module.flow(" in source
    assert "counterfactual_bridge = module.bridge(" in source
    assert "module(" not in source.split("def _a1_losses", 1)[1].split("def _loss_gradients", 1)[0]
    assert '"a2_launched": False' in source
    assert "run_stage2" not in source.lower()
    assert '"optimizer_updates": 0' in source


def test_a1_loss_path_trains_flow_and_isolates_execution_gates() -> None:
    torch.manual_seed(9)
    batch_size, positions, candidates, vocab, hidden_size = 2, 4, 3, 23, 16
    embedding = nn.Embedding(vocab, hidden_size)
    teacher_embedding = nn.Embedding(vocab, hidden_size)
    module = Phase2StudentModules(tied_embedding=embedding, hidden_size=hidden_size)
    _set_a1_trainable(module)
    candidate_ids = torch.randint(0, vocab, (batch_size, positions, candidates))
    candidate_mask = torch.ones_like(candidate_ids, dtype=torch.bool)
    base_candidates = torch.log_softmax(
        torch.randn(batch_size, positions, candidates), dim=-1
    )
    teacher_candidates = torch.log_softmax(
        torch.randn(batch_size, positions, candidates), dim=-1
    )
    batch = {
        "hidden": torch.randn(batch_size, positions, hidden_size),
        "candidate_ids": candidate_ids,
        "candidate_mask": candidate_mask,
        "base_candidates": base_candidates,
        "base_tail": torch.full((batch_size, positions), -3.0),
        "teacher_candidates": teacher_candidates,
        "teacher_tail": torch.full((batch_size, positions), -3.0),
        "teacher_topk_ids": torch.randint(0, vocab, (batch_size, positions, candidates)),
        "teacher_topk_log_probs": torch.log_softmax(
            torch.randn(batch_size, positions, candidates), dim=-1
        ),
        "target_scratch": torch.randn(batch_size, 8, 128),
        "position_bucket": torch.zeros(batch_size, dtype=torch.long),
    }
    losses, metrics = _a1_losses(
        module=module,
        batch=batch,
        embedding=embedding,
        teacher_embedding=teacher_embedding,
        decoder=torch.randn(128, hidden_size),
        decoder_bias=torch.randn(hidden_size),
        huber_delta=0.5,
    )
    total = sum(losses.values())
    assert torch.isfinite(total)
    total.backward()
    assert any(parameter.grad is not None for parameter in module.flow.parameters())
    assert all(
        parameter.grad is None
        for name, parameter in module.named_parameters()
        if not name.startswith("flow.")
    )
    assert float(metrics["executed_bridge_gate"]) == 0.0
    assert float(metrics["executed_draft_gate"]) == 0.0
    assert float(metrics["counterfactual_bridge_gate"]) > 0.0
