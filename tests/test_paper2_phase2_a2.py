from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch
from torch import nn

from models.paper2_dc2_student import Phase2StudentModules
from training.paper2_phase2_a2 import (
    classify_directional_shares,
    paired_verdict,
    repeated_marginal_bounds,
    should_extend,
)
from training.run_paper2_phase2_a2 import _forward, _set_trainable


ROOT = Path(__file__).resolve().parents[1]


def _tiny_batch(batch_size: int = 2, hidden_size: int = 16, vocab: int = 31):
    horizons, candidates = 4, 5
    candidate_ids = torch.randint(0, vocab, (batch_size, horizons, candidates))
    return {
        "hidden": torch.randn(batch_size, horizons, hidden_size),
        "candidate_ids": candidate_ids,
        "candidate_mask": torch.ones_like(candidate_ids, dtype=torch.bool),
        "base_candidates": torch.log_softmax(torch.randn(batch_size, horizons, candidates), -1),
        "base_tail": torch.full((batch_size, horizons), -4.0),
        "teacher_candidates": torch.log_softmax(torch.randn(batch_size, horizons, candidates), -1),
        "teacher_tail": torch.full((batch_size, horizons), -4.0),
        "target_scratch": torch.randn(batch_size, 8, 128),
        "position_bucket": torch.zeros(batch_size, dtype=torch.long),
    }


def test_directional_severity_contract_uses_aggregate_primary_share() -> None:
    passed = classify_directional_shares(
        {"cumulative_kl": 0.35, "local_ce": 0.16, "final_ce": 0.24, "preserve_kl": 0.25}
    )
    assert passed["classification"] == "pass"
    assert passed["primary_share"] == pytest.approx(0.51)
    marginal = classify_directional_shares(
        {"cumulative_kl": 0.35, "local_ce": 0.10, "final_ce": 0.30, "preserve_kl": 0.25}
    )
    assert marginal["classification"] == "marginal"
    assert set(marginal["marginal_bounds"]) == {
        "primary:below_0p50",
        "final_ce:above_0p25",
    }
    gross = classify_directional_shares(
        {"cumulative_kl": 0.20, "local_ce": 0.15, "final_ce": 0.36, "preserve_kl": 0.29}
    )
    assert gross["classification"] == "gross"
    assert "primary:below_0p40" in gross["gross_bounds"]
    assert "final_ce:above_0p35" in gross["gross_bounds"]


def test_two_consecutive_marginal_misses_require_same_bound() -> None:
    assert repeated_marginal_bounds(["primary:below_0p50"], ["final_ce:above_0p25"]) == []
    assert repeated_marginal_bounds(
        ["primary:below_0p50", "final_ce:above_0p25"],
        ["primary:below_0p50"],
    ) == ["primary:below_0p50"]


def test_extension_and_pair_verdict_contracts() -> None:
    assert should_extend(relative_headroom=0.019, accepted_length_slope=0.0)
    assert should_extend(relative_headroom=0.03, accepted_length_slope=0.0021)
    assert not should_extend(relative_headroom=0.02, accepted_length_slope=0.002)
    assert paired_verdict(
        relative_headroom=0.02, full_mean=2.2, control_mean=2.1, quality_noninferior=True
    )["verdict"] == "positive"


def test_draft_only_control_has_no_bridge_or_flow_gradient_and_unchanged_path() -> None:
    torch.manual_seed(9)
    embedding = nn.Embedding(31, 16)
    module = Phase2StudentModules(tied_embedding=embedding, hidden_size=16)
    _set_trainable(module, arm="draft_only_control")
    losses, metrics = _forward(
        module=module, batch=_tiny_batch(), embedding=embedding, arm="draft_only_control"
    )
    assert torch.equal(metrics["bridge_log"], metrics["base_log"])
    (losses["cumulative_kl"] + losses["local_ce"]).backward()
    assert all(parameter.grad is None for parameter in module.flow.parameters())
    assert all(parameter.grad is None for parameter in module.bridge.parameters())
    assert any(parameter.grad is not None for parameter in module.control.parameters())
    assert any(parameter.grad is not None for parameter in module.draft.parameters())


def test_a2_lock_authorizes_exact_four_run_matrix() -> None:
    registration = json.loads(
        (ROOT / "training/paper2_phase2_staged_repilot_preregistration.json").read_text(
            encoding="utf-8"
        )
    )
    lock = registration["a2_lock_amendment_20260805"]
    assert lock["status"] == "locked_before_a2_training"
    assert lock["directional_audit"]["primary_share_aggregation"] == "sum_before_classification"
    assert [(row["seed"], row["arm"]) for row in lock["run_matrix"]] == [
        (0, "full_a2"),
        (0, "draft_only_control"),
        (1, "full_a2"),
        (1, "draft_only_control"),
    ]
    assert lock["control"]["flow_calls"] == 0
    assert lock["control"]["bridge_writeback_calls"] == 0
