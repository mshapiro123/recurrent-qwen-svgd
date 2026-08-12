from __future__ import annotations

import torch

from training.paper2_phase3_p33_verification import (
    P33_CANONICAL_READER_NAME,
    canonical_logits,
    canonical_top1,
    fixed_pair_margin,
    margin_delta_summary,
    verification_verdict,
)


def test_canonical_reader_is_explicit_bfloat16_matmul() -> None:
    states = torch.tensor([[1.001, -0.75], [0.25, 2.0]], dtype=torch.float32)
    embedding = torch.tensor([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.5]])
    expected = states.to(torch.bfloat16) @ embedding.to(torch.bfloat16).T
    assert P33_CANONICAL_READER_NAME == "bf16_serving_matmul_v1"
    assert torch.equal(canonical_logits(states, embedding), expected)
    assert torch.equal(canonical_top1(states, embedding), expected.argmax(dim=-1))


def test_fixed_pair_margin_keeps_the_base_pair() -> None:
    logits = torch.tensor([[4.0, 3.0, 9.0], [1.0, -2.0, 0.0]])
    margin = fixed_pair_margin(logits, torch.tensor([2, 0]), torch.tensor([0, 2]))
    assert torch.equal(margin, torch.tensor([5.0, 1.0]))


def _row(delta: float, *, opened: bool, forced: bool = False) -> dict[str, object]:
    return {
        "margin_delta": delta,
        "hidden_delta_rms": 0.001,
        "gate_predicted_open": opened,
        "forced_open_collateral_change": forced,
        "runner_up_control_change": forced,
        "trained_direction_change_by_radius": {
            "0.3": forced,
            "0.6": forced,
            "1.0": forced,
        },
    }


def test_nonzero_check_requires_at_least_one_open_row_to_move() -> None:
    summary = margin_delta_summary([_row(-0.5, opened=True), _row(0.5, opened=True)])
    assert summary["gate_predicted_open_rows"] == 2
    assert summary["exact_zero_margin_delta_on_open_rows"] == 0
    assert summary["passed_nonzero_delta_check"] is True
    dead = margin_delta_summary([_row(0.0, opened=True), _row(0.0, opened=True)])
    assert dead["passed_nonzero_delta_check"] is False


def test_verification_requires_all_three_positive_controls() -> None:
    negative = [_row(0.25, opened=True, forced=True)]
    retention = [_row(-0.25, opened=True)]
    passed = verification_verdict(
        negative_rows=negative, retention_rows=retention, positive_deployed_flips=3
    )
    assert passed["all_passed"] is True
    failed = verification_verdict(
        negative_rows=negative, retention_rows=retention, positive_deployed_flips=0
    )
    assert failed["all_passed"] is False
