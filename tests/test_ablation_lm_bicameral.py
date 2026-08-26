from __future__ import annotations

import copy
import math

import pytest
import torch
import torch.nn.functional as F
from torch import nn

from models.ablation_lm.bicameral import SwapLinear
from models.ablation_lm.optim import (
    OptimizerTarget,
    ParameterRole,
    RANK_ONLY_MUON_PROHIBITED_ATTR,
    partition_optimizer_parameters,
)
from training.muon import split_muon_and_adamw_params


def test_swap_linear_matches_explicit_dense_hemisphere_maps() -> None:
    torch.manual_seed(4)
    layer = SwapLinear(7, 5, rank=3, sigma_delta0=0.03, seed=17)
    inputs = torch.randn(2, 4, 7)
    disagreement = layer.dU @ layer.dV.T

    expected_a = F.linear(inputs, layer.mu + disagreement)
    expected_b = F.linear(inputs, layer.mu - disagreement)
    actual_a = layer(inputs, +1)
    actual_b = layer(inputs, -1)

    torch.testing.assert_close(actual_a, expected_a)
    torch.testing.assert_close(actual_b, expected_b)
    torch.testing.assert_close(
        (actual_a + actual_b) / 2,
        F.linear(inputs, layer.mu),
    )
    assert set(dict(layer.named_parameters())) == {"mu", "dU", "dV"}
    assert set(layer.state_dict()) == {"mu", "dU", "dV"}
    assert not hasattr(layer, "weight_a")
    assert not hasattr(layer, "weight_b")


def test_deterministic_initialization_breaks_hemisphere_symmetry() -> None:
    torch.manual_seed(1)
    first = SwapLinear(8, 6, rank=2, sigma_delta0=0.02, seed=29)
    torch.manual_seed(9_999)
    second = SwapLinear(8, 6, rank=2, sigma_delta0=0.02, seed=29)
    different_seed = SwapLinear(8, 6, rank=2, sigma_delta0=0.02, seed=30)

    for name in ("mu", "dU", "dV"):
        torch.testing.assert_close(getattr(first, name), getattr(second, name))
    assert not torch.equal(first.mu, different_seed.mu)
    assert bool(first.dU.ne(0).any())
    assert bool(first.dV.ne(0).any())
    assert first.dU.norm().item() > 0.0
    assert first.dV.norm().item() > 0.0

    inputs = torch.randn(3, 8)
    assert not torch.equal(first(inputs, +1), first(inputs, -1))


def test_all_stored_modes_have_live_gradients() -> None:
    torch.manual_seed(12)
    layer = SwapLinear(6, 4, rank=2, seed=7)
    inputs = torch.randn(5, 6, requires_grad=True)
    output_a = layer(inputs, +1)
    output_b = layer(inputs, -1)
    loss = output_a.square().mean() + 0.37 * output_b.square().mean()

    loss.backward()

    assert inputs.grad is not None and bool(inputs.grad.ne(0).any())
    for parameter in (layer.mu, layer.dU, layer.dV):
        assert parameter.grad is not None
        assert bool(torch.isfinite(parameter.grad).all())
        assert bool(parameter.grad.ne(0).any())


def test_coupled_optimizer_provenance_is_adamw_and_rank_splitter_prohibited() -> None:
    layer = SwapLinear(8, 8, rank=2, seed=5)
    partition = partition_optimizer_parameters(layer)

    assert layer.optimizer_provenance() == {
        "mu": "auxiliary_adamw",
        "dU": "auxiliary_adamw",
        "dV": "auxiliary_adamw",
    }
    for name in ("mu", "dU", "dV"):
        assignment = partition.assignment_for(name)
        assert assignment.role is ParameterRole.COUPLED_MODE
        assert assignment.target is OptimizerTarget.AUXILIARY_ADAMW
        assert bool(getattr(getattr(layer, name), RANK_ONLY_MUON_PROHIBITED_ATTR))
    with pytest.raises(RuntimeError, match="rank-only Muon splitting is prohibited"):
        split_muon_and_adamw_params(layer.named_parameters())


def test_weight_decay_families_are_separate_complete_and_unpriced() -> None:
    layer = SwapLinear(8, 6, rank=3, seed=17)

    families = layer.adamw_weight_decay_families()

    assert tuple(families) == ("lambda_mu", "lambda_delta")
    assert families["lambda_mu"] == (layer.mu,)
    assert families["lambda_delta"] == (layer.dU, layer.dV)
    flattened = tuple(parameter for values in families.values() for parameter in values)
    assert {id(parameter) for parameter in flattened} == {
        id(parameter) for parameter in layer.parameters()
    }
    assert len({id(parameter) for parameter in flattened}) == len(flattened)


def test_state_dtype_assign_load_and_deepcopy_preserve_safety() -> None:
    torch.manual_seed(42)
    source = SwapLinear(8, 6, rank=2, seed=13)
    inputs = torch.randn(2, 8)
    expected = source(inputs, +1)
    checkpoint = {
        name: value.detach().clone() for name, value in source.state_dict().items()
    }

    restored = SwapLinear(8, 6, rank=2, seed=99)
    restored.load_state_dict(checkpoint)
    torch.testing.assert_close(restored(inputs, +1), expected)

    assigned = SwapLinear(8, 6, rank=2, seed=100)
    assigned.load_state_dict(checkpoint, assign=True)
    torch.testing.assert_close(assigned(inputs, +1), expected)

    cloned = copy.deepcopy(source)
    torch.testing.assert_close(cloned(inputs, +1), expected)
    for name in ("mu", "dU", "dV"):
        assert getattr(cloned, name) is not getattr(source, name)
        assert bool(getattr(getattr(cloned, name), RANK_ONLY_MUON_PROHIBITED_ATTR))
    assert cloned.optimizer_provenance() == source.optimizer_provenance()

    low_precision = copy.deepcopy(source).to(dtype=torch.bfloat16)
    low_input = inputs.to(dtype=torch.bfloat16)
    low_output = low_precision(low_input, -1)
    assert low_output.dtype is torch.bfloat16
    assert bool(torch.isfinite(low_output).all())
    for parameter in low_precision.parameters():
        assert parameter.dtype is torch.bfloat16
        assert bool(getattr(parameter, RANK_ONLY_MUON_PROHIBITED_ATTR))


def test_parent_assign_load_restores_recursive_optimizer_safety() -> None:
    source = nn.ModuleDict({"paired": SwapLinear(8, 6, rank=2, seed=13)})
    restored = nn.ModuleDict({"paired": SwapLinear(8, 6, rank=2, seed=99)})

    restored.load_state_dict(source.state_dict(), assign=True)

    paired = restored["paired"]
    assert isinstance(paired, SwapLinear)
    assert paired.optimizer_provenance() == {
        "mu": "auxiliary_adamw",
        "dU": "auxiliary_adamw",
        "dV": "auxiliary_adamw",
    }
    for parameter in paired.parameters():
        assert bool(getattr(parameter, RANK_ONLY_MUON_PROHIBITED_ATTR))
    with pytest.raises(RuntimeError, match="rank-only Muon splitting is prohibited"):
        split_muon_and_adamw_params(restored.named_parameters())


@pytest.mark.parametrize(
    ("args", "kwargs", "error", "match"),
    [
        ((0, 4), {}, ValueError, "d_in"),
        ((4, 0), {}, ValueError, "d_out"),
        ((True, 4), {}, ValueError, "d_in"),
        ((4, 4), {"rank": 0}, ValueError, "rank"),
        ((4, 4), {"rank": 5}, ValueError, "rank"),
        (
            (4, 4),
            {"rank": 2, "sigma_delta0": 0.0},
            ValueError,
            "sigma_delta0",
        ),
        (
            (4, 4),
            {"rank": 2, "sigma_delta0": math.inf},
            ValueError,
            "sigma_delta0",
        ),
        (
            (4, 4),
            {"rank": 2, "sigma_delta0": True},
            TypeError,
            "sigma_delta0",
        ),
        ((4, 4), {"rank": 2, "seed": -1}, ValueError, "seed"),
        ((4, 4), {"rank": 2, "seed": True}, ValueError, "seed"),
        (
            (4, 4),
            {"rank": 2, "dtype": torch.int64},
            TypeError,
            "floating dtype",
        ),
    ],
)
def test_constructor_rejects_invalid_contracts(
    args: tuple[object, ...],
    kwargs: dict[str, object],
    error: type[Exception],
    match: str,
) -> None:
    with pytest.raises(error, match=match):
        SwapLinear(*args, **kwargs)


def test_forward_rejects_invalid_shapes_dtypes_values_and_hemispheres() -> None:
    layer = SwapLinear(8, 4, rank=2, seed=3)
    valid = torch.randn(2, 8)

    for invalid_hemi in (0, 2, 1.0, -1.0, True, torch.tensor(1)):
        with pytest.raises(ValueError, match="exact integer"):
            layer(valid, invalid_hemi)  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="tensor"):
        layer([0.0] * 8, +1)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="final width"):
        layer(torch.tensor(1.0), +1)
    with pytest.raises(ValueError, match="final width"):
        layer(torch.randn(2, 7), +1)
    with pytest.raises(TypeError, match="floating-point"):
        layer(torch.ones(2, 8, dtype=torch.long), +1)
    with pytest.raises(TypeError, match="share a dtype"):
        layer(valid.double(), +1)
    non_finite = valid.clone()
    non_finite[0, 0] = torch.nan
    with pytest.raises(ValueError, match="finite"):
        layer(non_finite, +1)
