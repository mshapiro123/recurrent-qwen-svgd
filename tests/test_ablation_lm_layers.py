from __future__ import annotations

import copy
from dataclasses import replace
import math

import pytest
import torch

from models.ablation_lm.config import MUP_D_HEAD_BASE, AblationLMConfig
from models.ablation_lm.hadamard import sequency_permutation, wht
from models.ablation_lm.layers import (
    GroupedQueryAttention,
    ModifiedHadamardExpertBank,
    ProjectedKeyValue,
    RotaryEmbedding,
)
from models.sidecar_v2 import fast_wht


@pytest.mark.parametrize("width", (8, 512, 1_024))
def test_t3a_unnormalized_wht_round_trip_uses_one_binary_scale(width: int) -> None:
    generator = torch.Generator().manual_seed(20_260_826)
    values = torch.randn(7, width, generator=generator, dtype=torch.float32)

    stable = wht(wht(values)) * (2.0 ** -(width.bit_length() - 1))
    stable_error = (stable - values).abs().max() / values.norm()

    assert stable_error < 1e-5


@pytest.mark.parametrize("width", (8, 512, 1_024))
def test_t3b_wht_satisfies_parseval(width: int) -> None:
    generator = torch.Generator().manual_seed(31)
    values = torch.randn(5, width, generator=generator, dtype=torch.float32)

    transformed = wht(values)

    torch.testing.assert_close(
        transformed.norm(dim=-1) / math.sqrt(width),
        values.norm(dim=-1),
        rtol=2e-6,
        atol=2e-6,
    )


def test_t3c_wht_is_linear_in_fp32() -> None:
    generator = torch.Generator().manual_seed(32)
    left = torch.randn(4, 64, generator=generator)
    right = torch.randn(4, 64, generator=generator)
    alpha, beta = 0.375, -1.25

    combined = wht(alpha * left + beta * right)
    separate = alpha * wht(left) + beta * wht(right)

    torch.testing.assert_close(combined, separate, rtol=2e-6, atol=2e-6)


@pytest.mark.parametrize("width", (64, 512, 1_024))
def test_t3d_sampled_wht_basis_is_orthogonal(width: int) -> None:
    basis_indices = torch.tensor((0, 1, 3, 7, 11, 19, 31))
    basis = torch.eye(width, dtype=torch.float32).index_select(0, basis_indices)
    transformed = wht(basis)

    gram = transformed @ transformed.T / width

    torch.testing.assert_close(gram, torch.eye(len(basis_indices)), rtol=0, atol=0)


@pytest.mark.parametrize("width", (8, 512, 1_024))
def test_t3e_binary_inverse_scaling_constant_is_exact(width: int) -> None:
    exponent = width.bit_length() - 1

    assert (2.0**-exponent) * (2.0**exponent) == 1.0


def test_t3f_wht_is_deterministic_and_forces_fp32_under_autocast() -> None:
    values = torch.randn(3, 64, dtype=torch.bfloat16)

    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        first = wht(values)
        second = wht(values)

    assert first.dtype is torch.float32
    assert torch.equal(first, second)


def test_t4_bitrev_gray_permutation_has_exact_sequency() -> None:
    assert sequency_permutation(8).tolist() == [0, 4, 6, 2, 3, 7, 5, 1]
    for width in (1, 2, 4, 8, 16, 32, 512, 1_024):
        matrix = wht(torch.eye(width, dtype=torch.float32))
        ordered = matrix[sequency_permutation(width)]
        sign_changes = ordered[:, 1:].ne(ordered[:, :-1]).sum(dim=-1)

        assert torch.equal(sign_changes, torch.arange(width))


def _attention_config() -> AblationLMConfig:
    return AblationLMConfig(
        vocab_size=32,
        d_model=8,
        n_heads=2,
        n_kv_heads=1,
        d_ff=16,
        n_prelude_layers=1,
        n_core_blocks=1,
        n_coda_layers=0,
        recurrent_steps=1,
        max_recurrent_steps=2,
        max_sequence_length=16,
        scratch_width=4,
        long_term_memory_width=4,
    )


def _base_head_attention_config() -> AblationLMConfig:
    return replace(
        _attention_config(),
        d_model=128,
        d_ff=256,
        scratch_width=32,
        long_term_memory_width=32,
    )


def test_rotary_embedding_is_orthogonal_and_has_no_loop_coordinate() -> None:
    rope = RotaryEmbedding(4, theta=10_000.0, max_sequence_length=16)
    query = torch.randn(2, 3, 5, 4)
    key = torch.randn(2, 1, 5, 4)
    positions = torch.arange(5).view(1, -1).expand(2, -1)

    rotated_query, rotated_key = rope(query, key, positions)

    torch.testing.assert_close(rotated_query.square().sum(dim=-1), query.square().sum(dim=-1))
    torch.testing.assert_close(rotated_key.square().sum(dim=-1), key.square().sum(dim=-1))
    assert set(rope.forward.__annotations__) == {"query", "key", "position_ids", "return"}


def test_gqa_sdpa_matches_explicit_base_shape_mup_causal_attention() -> None:
    torch.manual_seed(7)
    config = _base_head_attention_config()
    attention = GroupedQueryAttention(config, module_path="model.test.sdpa").eval()
    hidden = torch.randn(2, 4, config.d_model)
    positions = torch.arange(4).view(1, -1).expand(2, -1)

    actual = attention(hidden, position_ids=positions)
    actual_math = attention(
        hidden,
        position_ids=positions,
        force_math_attention=True,
    )
    query = attention.query_norm(attention._split_heads(attention.q_proj(hidden), config.n_heads))
    key = attention.key_norm(attention._split_heads(attention.k_proj(hidden), config.n_kv_heads))
    value = attention._split_heads(attention.v_proj(hidden), config.n_kv_heads)
    query, key = attention.rope(query, key, positions)
    key = key.repeat_interleave(config.n_heads // config.n_kv_heads, dim=1)
    value = value.repeat_interleave(config.n_heads // config.n_kv_heads, dim=1)
    assert MUP_D_HEAD_BASE == 64
    assert config.head_dim == MUP_D_HEAD_BASE
    expected_scale = math.sqrt(MUP_D_HEAD_BASE) / config.head_dim
    assert expected_scale == 1.0 / math.sqrt(config.head_dim) == 0.125
    scores = query @ key.transpose(-1, -2) * expected_scale
    scores.masked_fill_(torch.ones(4, 4, dtype=torch.bool).triu(1), float("-inf"))
    reference = torch.softmax(scores.float(), dim=-1) @ value.float()
    reference = reference.transpose(1, 2).contiguous().view(2, 4, config.d_model)
    reference = attention.output_proj(reference)

    torch.testing.assert_close(actual, reference, rtol=2e-5, atol=2e-6)
    torch.testing.assert_close(actual_math, reference, rtol=2e-5, atol=2e-6)


def test_attention_dropout_is_fail_closed_until_a_generator_aware_fused_kernel() -> None:
    with pytest.raises(ValueError, match="generator-aware fused attention"):
        replace(_attention_config(), attention_dropout=0.25)

    config = _attention_config()
    attention = GroupedQueryAttention(
        config,
        module_path="model.test_attention",
    ).train()
    hidden = torch.randn(2, 5, config.d_model)
    positions = torch.arange(5).view(1, -1).expand(2, -1)

    attention(hidden, position_ids=positions, rng_coordinate=1)
    assert attention.dropout_rng.draw_indices == (0,) * config.max_recurrent_steps
    attention.dropout = 0.25
    with pytest.raises(RuntimeError, match="generator-aware fused kernel"):
        attention(hidden, position_ids=positions)


def test_t15_projected_kv_matches_same_source_recomputation_in_fp32() -> None:
    torch.manual_seed(8)
    config = _attention_config()
    attention = GroupedQueryAttention(config, module_path="model.test.t15_forward").eval()
    hidden = torch.randn(2, 5, config.d_model, dtype=torch.float32)
    positions = torch.arange(5).view(1, -1).expand(2, -1)

    recomputed = attention(hidden, position_ids=positions)
    projected_kv = attention.project_kv(hidden, position_ids=positions)
    cached = attention(
        hidden,
        projected_kv=projected_kv,
        position_ids=positions,
    )

    torch.testing.assert_close(cached, recomputed, rtol=0, atol=0)


def test_t15_projected_kv_matches_recomputation_backward_in_fp32() -> None:
    torch.manual_seed(81)
    config = _attention_config()
    cached_attention = GroupedQueryAttention(
        config,
        module_path="model.test.t15_backward",
    ).eval()
    recomputed_attention = copy.deepcopy(cached_attention)
    cached_hidden = torch.randn(
        2,
        5,
        config.d_model,
        dtype=torch.float32,
        requires_grad=True,
    )
    recomputed_hidden = cached_hidden.detach().clone().requires_grad_(True)
    positions = torch.arange(5).view(1, -1).expand(2, -1)

    projected_kv = cached_attention.project_kv(
        cached_hidden,
        position_ids=positions,
    )
    cached = cached_attention(
        cached_hidden,
        projected_kv=projected_kv,
        position_ids=positions,
    )
    recomputed = recomputed_attention(
        recomputed_hidden,
        position_ids=positions,
    )
    weights = torch.linspace(0.25, 1.25, cached.numel()).reshape_as(cached)
    (cached * weights).sum().backward()
    (recomputed * weights).sum().backward()

    torch.testing.assert_close(cached, recomputed, rtol=0, atol=0)
    torch.testing.assert_close(cached_hidden.grad, recomputed_hidden.grad)
    for (cached_name, cached_parameter), (reference_name, reference_parameter) in zip(
        cached_attention.named_parameters(),
        recomputed_attention.named_parameters(),
        strict=True,
    ):
        assert cached_name == reference_name
        assert cached_parameter.grad is not None
        assert reference_parameter.grad is not None
        torch.testing.assert_close(cached_parameter.grad, reference_parameter.grad)


def test_projected_kv_rejects_wrong_grouped_head_shape() -> None:
    config = _attention_config()
    attention = GroupedQueryAttention(config, module_path="model.test.bad_shape").eval()
    hidden = torch.randn(2, 5, config.d_model)
    wrong = ProjectedKeyValue(
        key=torch.randn(2, config.n_heads, 5, config.head_dim),
        value=torch.randn(2, config.n_heads, 5, config.head_dim),
        position_ids=torch.arange(5).view(1, -1).expand(2, -1),
        owner_id=id(attention),
    )

    with pytest.raises(ValueError, match="projected key"):
        attention(hidden, projected_kv=wrong)


def test_projected_kv_rejects_position_id_mismatch() -> None:
    config = _attention_config()
    attention = GroupedQueryAttention(
        config,
        module_path="model.test.position_mismatch",
    ).eval()
    hidden = torch.randn(2, 5, config.d_model)
    positions = torch.arange(5).view(1, -1).expand(2, -1)
    projected_kv = attention.project_kv(hidden, position_ids=positions)
    shifted = positions + 1

    with pytest.raises(ValueError, match="different position IDs"):
        attention(
            hidden,
            projected_kv=projected_kv,
            position_ids=shifted,
        )


def test_projected_kv_rejects_wrong_owner_and_position_metadata_types() -> None:
    config = _attention_config()
    first = GroupedQueryAttention(config, module_path="model.test.owner.first").eval()
    second = GroupedQueryAttention(config, module_path="model.test.owner.second").eval()
    hidden = torch.randn(2, 5, config.d_model)
    positions = torch.arange(5).view(1, -1).expand(2, -1)
    projected_kv = first.project_kv(hidden, position_ids=positions)

    with pytest.raises(ValueError, match="different attention block"):
        second(hidden, projected_kv=projected_kv, position_ids=positions)
    with pytest.raises(TypeError, match="integer dtype"):
        first.project_kv(hidden, position_ids=positions.float())
    with pytest.raises(ValueError, match="hidden-state device"):
        first.project_kv(hidden, position_ids=positions.to(device="meta"))
    with pytest.raises(TypeError, match="integer dtype"):
        first(hidden, position_ids=positions.float())


def test_projected_kv_rejects_stale_dtype_after_module_conversion() -> None:
    config = _attention_config()
    attention = GroupedQueryAttention(config, module_path="model.test.stale_dtype").eval()
    hidden = torch.randn(2, 5, config.d_model)
    projected_kv = attention.project_kv(hidden)
    attention.to(dtype=torch.bfloat16)

    with pytest.raises(ValueError, match="share the query dtype"):
        attention(hidden.to(torch.bfloat16), projected_kv=projected_kv)


def test_vectorized_hadamard_experts_match_explicit_expert_loop_forward_and_backward() -> None:
    torch.manual_seed(17)
    bank = ModifiedHadamardExpertBank(
        8,
        experts=3,
        layer_scale=1e-3,
        norm_eps=1e-5,
        seed=23,
    )
    reference_bank = copy.deepcopy(bank)
    hidden = torch.randn(2, 4, 8, requires_grad=True)
    reference_hidden = hidden.detach().clone().requires_grad_(True)

    actual = bank(hidden)
    normalized = reference_bank.norm(reference_hidden)
    routing = torch.softmax(reference_bank.router(normalized).float(), dim=-1)
    coefficients = fast_wht(normalized)
    expert_updates = []
    for expert in range(reference_bank.experts):
        filtered = (
            coefficients[..., reference_bank.permutations[expert]]
            * reference_bank.signs[expert]
            * reference_bank.expert_gains[expert]
        )
        expert_updates.append(fast_wht(filtered))
    loop_update = (
        routing.unsqueeze(-1) * torch.stack(expert_updates, dim=-2)
    ).sum(dim=-2)
    expected = reference_hidden + reference_bank.layer_scale * loop_update

    torch.testing.assert_close(actual, expected)
    weights = torch.linspace(0.5, 1.5, actual.numel()).reshape_as(actual)
    (actual * weights).sum().backward()
    (expected * weights).sum().backward()
    torch.testing.assert_close(hidden.grad, reference_hidden.grad)
    torch.testing.assert_close(bank.router.weight.grad, reference_bank.router.weight.grad)
    torch.testing.assert_close(bank.expert_gains.grad, reference_bank.expert_gains.grad)
