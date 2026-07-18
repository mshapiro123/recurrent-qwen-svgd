from __future__ import annotations

import pytest
import torch

from models.oracle_reentry_conditioner import OracleReentryConditioner


@pytest.mark.parametrize("mode", ["additive", "film"])
def test_oracle_conditioner_is_exact_identity_at_installation(mode: str) -> None:
    torch.manual_seed(11)
    module = OracleReentryConditioner(hidden_dim=8, bottleneck_dim=4)
    states = torch.randn(2, 5, 8)
    commands = torch.randn(2, 8)
    mask = torch.ones(2, 5, dtype=torch.long)

    output = module(states, mask, commands, mode=mode)

    assert torch.equal(output.states, states)
    assert output.residual_rms_ratio.item() == 0.0


def test_oracle_conditioner_modes_have_identical_parameter_budgets() -> None:
    additive = OracleReentryConditioner(hidden_dim=8, bottleneck_dim=4)
    film = OracleReentryConditioner(hidden_dim=8, bottleneck_dim=4)

    assert sum(parameter.numel() for parameter in additive.parameters()) == sum(
        parameter.numel() for parameter in film.parameters()
    )


@pytest.mark.parametrize("mode", ["additive", "film"])
def test_oracle_conditioner_output_layers_are_gradient_live(mode: str) -> None:
    torch.manual_seed(13)
    module = OracleReentryConditioner(hidden_dim=8, bottleneck_dim=4)
    states = torch.randn(2, 5, 8)
    commands = torch.randn(2, 8)

    output = module(states, None, commands, mode=mode)
    output.states.square().mean().backward()

    for branch in (module.branch_a, module.branch_b):
        assert branch.net[-1].weight.grad is not None
        assert branch.net[-1].weight.grad.count_nonzero().item() > 0


@pytest.mark.parametrize("mode", ["additive", "film"])
def test_force_identity_bypasses_a_trained_conditioner_exactly(mode: str) -> None:
    torch.manual_seed(17)
    module = OracleReentryConditioner(hidden_dim=8, bottleneck_dim=4)
    with torch.no_grad():
        module.branch_a.net[-1].weight.normal_()
        module.branch_b.net[-1].weight.normal_()
    states = torch.randn(2, 5, 8)
    commands = torch.randn(2, 8)

    active = module(states, None, commands, mode=mode)
    bypassed = module(
        states,
        None,
        commands,
        mode=mode,
        force_identity=True,
    )

    assert not torch.equal(active.states, states)
    assert torch.equal(bypassed.states, states)
