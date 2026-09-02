from __future__ import annotations

import copy
from dataclasses import replace
import math
from unittest.mock import patch

import pytest
import torch

from models.ablation_lm.bicameral import SwapLinear
from models.ablation_lm.bicameral_core import (
    BicameralProjectedKeyValue,
    BicameralTransformerBlock,
)
from models.ablation_lm.config import MUP_D_HEAD_BASE, AblationLMConfig
from models.ablation_lm.layers import RMSNorm, TransformerBlock


def _tiny_block(**updates: object) -> BicameralTransformerBlock:
    kwargs: dict[str, object] = {
        "d_model": 16,
        "n_heads": 4,
        "n_kv_heads": 2,
        "d_ff": 32,
        "max_sequence_length": 32,
        "rank": 4,
        "initialization_seed": 812,
        "module_path": "model.bicameral_core.test",
    }
    kwargs.update(updates)
    return BicameralTransformerBlock(**kwargs)


def _tiny_dense_config() -> AblationLMConfig:
    return AblationLMConfig(
        vocab_size=64,
        d_model=16,
        n_heads=4,
        n_kv_heads=2,
        d_ff=32,
        n_prelude_layers=1,
        n_core_blocks=1,
        n_coda_layers=1,
        max_sequence_length=32,
        scratch_width=8,
    )


def _realized_weight(layer: SwapLinear, hemi: int) -> torch.Tensor:
    return layer.mu + hemi * (layer.dU @ layer.dV.T)


def _dense_hemisphere(
    paired: BicameralTransformerBlock,
    *,
    hemi: int,
) -> TransformerBlock:
    dense = TransformerBlock(
        _tiny_dense_config(),
        module_path=f"model.dense_reference.{hemi + 1}",
    )
    with torch.no_grad():
        dense.attention_norm.weight.copy_(paired.attention_norm.weight)
        dense.ffn_norm.weight.copy_(paired.ffn_norm.weight)
        dense.attention.query_norm.weight.copy_(paired.query_norm.weight)
        dense.attention.key_norm.weight.copy_(paired.key_norm.weight)
        for target, source in (
            (dense.attention.q_proj, paired.q_proj),
            (dense.attention.k_proj, paired.k_proj),
            (dense.attention.v_proj, paired.v_proj),
            (dense.attention.output_proj, paired.o_proj),
            (dense.feed_forward.gate_proj, paired.gate_proj),
            (dense.feed_forward.up_proj, paired.up_proj),
            (dense.feed_forward.down_proj, paired.down_proj),
        ):
            target.weight.copy_(_realized_weight(source, hemi))
    return dense.eval()


def test_structure_initialization_and_exact_d512_parameter_delta() -> None:
    torch.manual_seed(91)
    ambient_state = torch.random.get_rng_state().clone()
    block = BicameralTransformerBlock(
        512,
        n_heads=8,
        n_kv_heads=4,
        d_ff=1_408,
        max_sequence_length=2_048,
        module_path="model.bicameral_core.0",
    )

    assert block.head_dim == MUP_D_HEAD_BASE == 64
    assert (
        1.0 / math.sqrt(block.head_dim)
        == math.sqrt(MUP_D_HEAD_BASE) / block.head_dim
        == 0.125
    )
    assert torch.equal(torch.random.get_rng_state(), ambient_state)
    assert len(block.swap_linears) == 7
    assert all(isinstance(layer, SwapLinear) for layer in block.swap_linears)
    assert len({seed for _name, seed in block.projection_initialization_seeds}) == 7
    assert all(bool(layer.dU.ne(0).any()) for layer in block.swap_linears)
    assert all(bool(layer.dV.ne(0).any()) for layer in block.swap_linears)
    assert all(isinstance(norm, RMSNorm) for norm in (
        block.attention_norm,
        block.ffn_norm,
        block.query_norm,
        block.key_norm,
    ))
    assert not any(
        name.endswith("weight_a") or name.endswith("weight_b")
        for name, _ in block.named_parameters()
    )

    dense = TransformerBlock(
        AblationLMConfig(),
        module_path="model.dense_parameter_reference",
    )
    paired_parameters = sum(parameter.numel() for parameter in block.parameters())
    dense_parameters = sum(parameter.numel() for parameter in dense.parameters())
    assert block.disagreement_parameter_count == 299_008
    assert paired_parameters - dense_parameters == 299_008


def test_parameter_initialization_is_replica_invariant_and_path_namespaced() -> None:
    torch.manual_seed(1)
    first = _tiny_block(module_path="model.bicameral_core.identity")
    torch.manual_seed(9_999_999)
    second = _tiny_block(module_path="model.bicameral_core.identity")

    assert first.projection_initialization_seeds == second.projection_initialization_seeds
    for (first_name, first_parameter), (second_name, second_parameter) in zip(
        first.named_parameters(),
        second.named_parameters(),
        strict=True,
    ):
        assert first_name == second_name
        torch.testing.assert_close(first_parameter, second_parameter, rtol=0, atol=0)
    assert not hasattr(first, "rng_replica")
    with pytest.raises(TypeError, match="unexpected keyword argument"):
        _tiny_block(rng_replica=1)

    other_path = _tiny_block(module_path="model.bicameral_core.other")
    assert first.projection_initialization_seeds != other_path.projection_initialization_seeds
    first_seeds = dict(first.projection_initialization_seeds)
    other_seeds = dict(other_path.projection_initialization_seeds)
    assert all(first_seeds[name] != other_seeds[name] for name in first_seeds)
    assert any(
        not torch.equal(first_parameter, other_parameter)
        for first_parameter, other_parameter in zip(
            first.parameters(),
            other_path.parameters(),
            strict=True,
        )
    )


def test_each_hemisphere_matches_its_realized_dense_static_kv_block() -> None:
    torch.manual_seed(17)
    paired = _tiny_block().eval()
    dense_a = _dense_hemisphere(paired, hemi=+1)
    dense_b = _dense_hemisphere(paired, hemi=-1)
    h0 = torch.randn(2, 6, 16)
    h_a = torch.randn(2, 6, 16)
    h_b = torch.randn(2, 6, 16)
    positions = torch.arange(6).view(1, -1).expand(2, -1)
    cache = paired.project_kv(h0, position_ids=positions)

    actual_a, actual_b = paired(
        h_a,
        h_b,
        projected_kv=cache,
        position_ids=positions,
        force_math_attention=True,
    )
    expected_a = dense_a(
        h_a,
        projected_kv=dense_a.project_kv(h0, position_ids=positions),
        position_ids=positions,
        force_math_attention=True,
    )
    expected_b = dense_b(
        h_b,
        projected_kv=dense_b.project_kv(h0, position_ids=positions),
        position_ids=positions,
        force_math_attention=True,
    )

    torch.testing.assert_close(actual_a, expected_a, rtol=2e-5, atol=2e-6)
    torch.testing.assert_close(actual_b, expected_b, rtol=2e-5, atol=2e-6)


def test_one_paired_cache_removes_the_recurrent_visit_multiplier() -> None:
    torch.manual_seed(29)
    block = _tiny_block().eval()
    h0 = torch.randn(1, 7, 16)
    h_a = h0.clone()
    h_b = h0.clone()
    positions = torch.arange(7).view(1, -1)

    with (
        patch.object(block.k_proj, "forward", wraps=block.k_proj.forward) as project_k,
        patch.object(block.v_proj, "forward", wraps=block.v_proj.forward) as project_v,
        patch.object(block.q_proj, "forward", wraps=block.q_proj.forward) as project_q,
    ):
        cache = block.project_kv(h0, position_ids=positions)
        for _visit in range(4):
            h_a, h_b = block(
                h_a,
                h_b,
                projected_kv=cache,
                position_ids=positions,
            )

    assert project_k.call_count == 2
    assert project_v.call_count == 2
    assert project_q.call_count == 8
    assert cache.position_ids.data_ptr() != positions.data_ptr()


def test_all_seven_mu_and_disagreement_factors_have_live_gradients() -> None:
    torch.manual_seed(41)
    block = _tiny_block().train()
    h0 = torch.randn(2, 5, 16, requires_grad=True)
    h_a = torch.randn(2, 5, 16, requires_grad=True)
    h_b = torch.randn(2, 5, 16, requires_grad=True)
    cache = block.project_kv(h0)

    output_a, output_b = block(h_a, h_b, projected_kv=cache)
    loss = output_a.square().mean() + 0.37 * output_b.square().mean()
    loss.backward()

    for name in (
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ):
        layer = getattr(block, name)
        for parameter_name in ("mu", "dU", "dV"):
            gradient = getattr(layer, parameter_name).grad
            assert gradient is not None, f"{name}.{parameter_name} has no gradient"
            assert bool(torch.isfinite(gradient).all())
            assert bool(gradient.ne(0).any()), f"{name}.{parameter_name} gradient is zero"


def test_paired_static_kv_is_exactly_causal_across_packing_and_padding() -> None:
    torch.manual_seed(53)
    block = _tiny_block().eval()
    h0 = torch.randn(1, 6, 16, requires_grad=True)
    h_a = torch.randn(1, 6, 16, requires_grad=True)
    h_b = torch.randn(1, 6, 16, requires_grad=True)
    attention_mask = torch.tensor([[1, 1, 1, 1, 1, 0]])
    document_ids = torch.tensor([[0, 0, 1, 1, 1, -1]])
    position_ids = torch.tensor([[0, 1, 0, 1, 2, 0]])
    cache = block.project_kv(h0, position_ids=position_ids)

    output_a, output_b = block(
        h_a,
        h_b,
        projected_kv=cache,
        attention_mask=attention_mask,
        document_ids=document_ids,
        position_ids=position_ids,
        force_math_attention=True,
    )
    gradients = torch.autograd.grad(
        output_a[0, 3].sum() + 0.31 * output_b[0, 3].sum(),
        (h0, h_a, h_b),
    )

    anchor_gradient, query_a_gradient, query_b_gradient = gradients
    assert torch.count_nonzero(anchor_gradient[0, :2]) == 0
    assert torch.count_nonzero(anchor_gradient[0, 2:4]) > 0
    assert torch.count_nonzero(anchor_gradient[0, 4:]) == 0
    for query_gradient in (query_a_gradient, query_b_gradient):
        assert torch.count_nonzero(query_gradient[0, :3]) == 0
        assert torch.count_nonzero(query_gradient[0, 3]) > 0
        assert torch.count_nonzero(query_gradient[0, 4:]) == 0


def test_cache_owner_positions_dtype_and_device_fail_closed() -> None:
    first = _tiny_block(module_path="model.bicameral_core.first").eval()
    second = _tiny_block(module_path="model.bicameral_core.second").eval()
    hidden = torch.randn(1, 4, 16)
    positions = torch.arange(4).view(1, -1)
    cache = first.project_kv(hidden, position_ids=positions)
    original_positions = positions.clone()
    positions.add_(1)
    torch.testing.assert_close(cache.position_ids, original_positions)

    with pytest.raises(ValueError, match="different bicameral block"):
        second(hidden, hidden, projected_kv=cache, position_ids=original_positions)
    with pytest.raises(ValueError, match="different position IDs"):
        first(hidden, hidden, projected_kv=cache, position_ids=positions)

    stale_dtype = copy.deepcopy(first)
    stale_cache = stale_dtype.project_kv(hidden, position_ids=original_positions)
    stale_dtype.to(dtype=torch.bfloat16)
    with pytest.raises(TypeError, match="query dtype"):
        stale_dtype(
            hidden.to(torch.bfloat16),
            hidden.to(torch.bfloat16),
            projected_kv=stale_cache,
            position_ids=original_positions,
        )
    meta_cache = replace(
        cache,
        key_a=cache.key_a.to(device="meta"),
    )
    with pytest.raises(ValueError, match="query device"):
        first(hidden, hidden, projected_kv=meta_cache, position_ids=original_positions)


def test_zero_dropout_and_full_width_contracts_are_structural() -> None:
    with pytest.raises(ValueError, match="2:1"):
        _tiny_block(n_kv_heads=1)
    with pytest.raises(ValueError, match="structurally fixed at zero"):
        _tiny_block(attention_dropout=0.1)
    with pytest.raises(TypeError, match="exact real scalar"):
        _tiny_block(attention_dropout=True)

    block = _tiny_block()
    hidden = torch.randn(1, 4, 16)
    cache = block.project_kv(hidden)
    assert isinstance(cache, BicameralProjectedKeyValue)
    assert cache.key_a.shape == (1, 2, 4, 4)
    assert cache.key_b.shape == cache.key_a.shape
    assert cache.value_a.shape == cache.key_a.shape
    assert cache.value_b.shape == cache.key_a.shape
    with pytest.raises(TypeError, match="BicameralProjectedKeyValue"):
        block(hidden, hidden, projected_kv=None)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="identical shapes"):
        block(hidden, hidden[:, :3], projected_kv=cache)
    with pytest.raises(ValueError, match="attention_mask"):
        block(
            hidden,
            hidden,
            projected_kv=cache,
            attention_mask=torch.ones(2, 4, dtype=torch.long),
        )


@pytest.mark.parametrize(
    ("field", "value", "error"),
    [
        ("rope_theta", True, TypeError),
        ("rope_theta", "500000", TypeError),
        ("rope_theta", 1.0, ValueError),
        ("rope_theta", float("inf"), ValueError),
        ("rope_theta", float("nan"), ValueError),
        ("norm_eps", False, TypeError),
        ("norm_eps", "1e-5", TypeError),
        ("norm_eps", 0.0, ValueError),
        ("norm_eps", -1e-5, ValueError),
        ("norm_eps", float("nan"), ValueError),
    ],
)
def test_rope_and_norm_scalars_fail_closed(
    field: str,
    value: object,
    error: type[Exception],
) -> None:
    with pytest.raises(error):
        _tiny_block(**{field: value})
