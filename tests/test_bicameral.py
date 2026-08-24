from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.bicameral import (
    OPERATING_GATE_VALUE,
    SEQUENTIAL_EXECUTION_SCHEDULE,
    BicameralCore,
    MuDeltaCombiner,
    WHTFrame,
)


class FrozenMiddle(nn.Module):
    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        self.layers = nn.ModuleList(
            [
                nn.Sequential(
                    nn.LayerNorm(hidden_size),
                    nn.Linear(hidden_size, hidden_size),
                    nn.Tanh(),
                )
                for _ in range(3)
            ]
        )
        for parameter in self.parameters():
            parameter.requires_grad_(False)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            hidden = hidden + layer(hidden)
        return hidden


def test_t1_all_gates_zero_is_bit_exact() -> None:
    torch.manual_seed(3)
    core = BicameralCore(128)
    middle = FrozenMiddle(128)
    hidden = torch.randn((4, 6, 128))
    output, branch_a, branch_b = core(hidden, middle)
    base = middle(hidden)
    assert torch.equal(branch_a, base)
    assert torch.equal(branch_b, base)
    assert torch.equal(output, base)


def test_middle_execution_is_sequential_and_schedule_is_declared() -> None:
    torch.manual_seed(4)
    core = BicameralCore(128)
    hidden = torch.randn((3, 5, 128))
    calls: list[tuple[int, ...]] = []

    def middle(value: torch.Tensor) -> torch.Tensor:
        calls.append(tuple(value.shape))
        return value

    core(hidden, middle)
    assert calls == [(3, 5, 128), (3, 5, 128)]
    assert core.execution_schedule == SEQUENTIAL_EXECUTION_SCHEDULE


def test_t2_cold_start_gradient_contract() -> None:
    torch.manual_seed(5)
    core = BicameralCore(128)
    middle = FrozenMiddle(128)
    hidden = torch.randn((4, 6, 128))
    target = torch.randn_like(hidden)
    output, _branch_a, _branch_b = core(hidden, middle)
    F.mse_loss(output, target).backward()
    alive = (
        core.callosum.gate_a.grad,
        core.callosum.gate_b.grad,
        core.bank_a.gate.grad,
        core.bank_b.gate.grad,
        core.combiner.mu.grad,
    )
    assert all(value is not None and torch.isfinite(value).all() for value in alive)
    assert all(value.abs().max() > 0 for value in alive)
    assert torch.count_nonzero(core.combiner.delta.grad) == 0
    assert torch.count_nonzero(core.bank_a.gains.grad) == 0
    assert torch.count_nonzero(core.bank_b.gains.grad) == 0


def test_combiner_closed_form_fit_reduces_state_error() -> None:
    torch.manual_seed(7)
    combiner = MuDeltaCombiner(128, rms_cap=1e9)
    branch_a = torch.randn((3, 5, 128))
    branch_b = torch.randn((3, 5, 128))
    target_mu = 1.0 + 0.2 * torch.randn(128)
    target_delta = 0.3 * torch.randn(128)
    frame = WHTFrame(128)
    target = frame.inverse_transform(
        target_mu * (frame.forward_transform(branch_a) + frame.forward_transform(branch_b)) / 2
        + target_delta * (frame.forward_transform(branch_a) - frame.forward_transform(branch_b)) / 2
    )
    before = F.mse_loss(combiner(branch_a, branch_b), target)
    combiner.fit_state_matching(branch_a, branch_b, target)
    after = F.mse_loss(combiner(branch_a, branch_b), target)
    assert after < before * 1e-4


def test_step1_trainable_set_is_combiner_only() -> None:
    core = BicameralCore(128)
    assert sum(parameter.numel() for parameter in core.parameters()) == 2308
    names = core.configure_step1_trainable()
    assert names == ["combiner.mu", "combiner.delta"]
    assert sum(parameter.numel() for parameter in core.parameters() if parameter.requires_grad) == 256


def test_operating_gates_require_measurement_receipt() -> None:
    core = BicameralCore(128)
    try:
        core.set_conditioning_gates(
            callosum_a=1.0,
            callosum_b=1.0,
            bank_a=1.0,
            bank_b=1.0,
            source_receipt_sha256="",
        )
    except ValueError:
        pass
    else:
        raise AssertionError("unreceipted operating gates were accepted")


def test_strategy_operating_gates_are_bound_without_search() -> None:
    core = BicameralCore(128)
    core.bind_strategy_operating_gates(source_receipt_sha256="a" * 64)
    assert float(core.callosum.gate_a.detach()) == OPERATING_GATE_VALUE
    assert float(core.callosum.gate_b.detach()) == OPERATING_GATE_VALUE
    assert float(core.bank_a.gate.detach()) == OPERATING_GATE_VALUE
    assert float(core.bank_b.gate.detach()) == OPERATING_GATE_VALUE
    assert core.conditioning_receipt_sha256 == "a" * 64
