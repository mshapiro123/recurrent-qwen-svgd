from __future__ import annotations

import pytest
import torch
from torch import nn

from training.muon import Muon, split_muon_and_adamw_params, zeropower_via_newtonschulz5


def test_zeropower_returns_finite_matrix_with_matching_shape() -> None:
    torch.manual_seed(0)
    update = torch.randn(6, 4)

    out = zeropower_via_newtonschulz5(update, steps=2)

    assert out.shape == update.shape
    assert torch.isfinite(out).all()


def test_muon_steps_matrix_parameter() -> None:
    torch.manual_seed(0)
    param = nn.Parameter(torch.randn(4, 4))
    before = param.detach().clone()
    param.grad = torch.randn_like(param)
    optimizer = Muon([param], lr=1e-3, ns_steps=2)

    optimizer.step()

    assert not torch.allclose(param.detach(), before)
    assert torch.isfinite(param).all()


def test_muon_rejects_vector_parameter() -> None:
    param = nn.Parameter(torch.randn(4))
    param.grad = torch.randn_like(param)
    optimizer = Muon([param], lr=1e-3)

    with pytest.raises(ValueError, match="ndim >= 2"):
        optimizer.step()


def test_split_muon_and_adamw_params_routes_by_rank() -> None:
    model = nn.Sequential(nn.Linear(3, 4), nn.LayerNorm(4))

    muon_params, adamw_params = split_muon_and_adamw_params(model.named_parameters())

    assert any(param.ndim >= 2 for param in muon_params)
    assert all(param.ndim >= 2 for param in muon_params)
    assert any(param.ndim < 2 for param in adamw_params)
    assert all(param.ndim < 2 for param in adamw_params)
