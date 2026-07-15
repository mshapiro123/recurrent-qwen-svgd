from __future__ import annotations

import torch

from training.phase_g_alpha_spec import (
    assert_frozen_gradients_zero,
    assert_frozen_parameter_contract,
    preregistration_payload,
)


class ParameterSet(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.recurrent_block = torch.nn.Linear(2, 2)
        self.phase_g_prior_head = torch.nn.Linear(2, 2)
        self.phase_g_posterior_head = torch.nn.Linear(2, 2)
        self.phase_g_injection_scale = torch.nn.Parameter(torch.zeros(()))


def freeze_substrate(module: ParameterSet) -> None:
    for name, parameter in module.named_parameters():
        parameter.requires_grad = name.startswith("phase_g_")


def test_frozen_parameter_contract_accepts_only_three_phase_g_groups() -> None:
    module = ParameterSet()
    freeze_substrate(module)

    ledger = assert_frozen_parameter_contract(module.named_parameters())

    assert ledger["unexpected_trainable"] == []
    assert any(name.startswith("phase_g_prior_head.") for name in ledger["allowed_trainable"])


def test_frozen_parameter_contract_rejects_trainable_recurrent_block() -> None:
    module = ParameterSet()
    freeze_substrate(module)
    module.recurrent_block.weight.requires_grad = True

    try:
        assert_frozen_parameter_contract(module.named_parameters())
    except AssertionError as exc:
        assert "recurrent_block.weight" in str(exc)
    else:
        raise AssertionError("Expected the trainable recurrent block to be rejected")


def test_frozen_gradient_assertion_is_exact() -> None:
    module = ParameterSet()
    freeze_substrate(module)
    module.recurrent_block.weight.grad = torch.zeros_like(module.recurrent_block.weight)
    assert_frozen_gradients_zero(module.named_parameters())

    module.recurrent_block.weight.grad[0, 0] = 1e-12
    try:
        assert_frozen_gradients_zero(module.named_parameters())
    except AssertionError as exc:
        assert "recurrent_block.weight" in str(exc)
    else:
        raise AssertionError("Any nonzero frozen gradient must fail")


def test_preregistration_leaves_only_powered_margin_blank() -> None:
    payload = preregistration_payload()

    assert payload["status"] == "forms_locked_numeric_margins_pending_power_calculation"
    assert "branching_relations" in payload["substrate_gate"]["required"]
    assert payload["frozen_evaluation"]["task_family"] == "multi_valued_forward_relations"
    assert payload["frozen_evaluation"]["reachable_set_size_bins"] == ["2", "3-4", "5-8", "9-16"]
    assert payload["primary_gate_form"]["alpha"] == 0.05
    assert payload["power_calculation_todo"]["only_remaining_preregistration_blank"] is True
    assert payload["deferred_until_G_alpha_win"] == ["LPRM", "per_trajectory_halting", "SVGD"]
