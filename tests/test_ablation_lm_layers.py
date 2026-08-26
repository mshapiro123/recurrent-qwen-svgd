from __future__ import annotations

import copy
from dataclasses import replace
import math

import torch

from models.ablation_lm.config import AblationLMConfig
from models.ablation_lm.hadamard import sequency_permutation, wht
from models.ablation_lm.layers import (
    GroupedQueryAttention,
    ModifiedHadamardExpertBank,
    RotaryEmbedding,
)
from models.sidecar_v2 import fast_wht


def test_t3_unnormalized_wht_round_trip_uses_one_binary_scale() -> None:
    generator = torch.Generator().manual_seed(20_260_826)
    values = torch.randn(7, 512, generator=generator, dtype=torch.float32)

    stable = wht(wht(values)) * (2.0**-9)
    stable_error = (stable - values).abs().max() / values.norm()

    assert stable_error < 1e-5


def test_t4_sequency_row_k_has_exactly_k_sign_changes() -> None:
    for width in (1, 2, 4, 8, 16, 32):
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


def test_rotary_embedding_is_orthogonal_and_has_no_loop_coordinate() -> None:
    rope = RotaryEmbedding(4, theta=10_000.0, max_sequence_length=16)
    query = torch.randn(2, 3, 5, 4)
    key = torch.randn(2, 1, 5, 4)
    positions = torch.arange(5).view(1, -1).expand(2, -1)

    rotated_query, rotated_key = rope(query, key, positions)

    torch.testing.assert_close(rotated_query.square().sum(dim=-1), query.square().sum(dim=-1))
    torch.testing.assert_close(rotated_key.square().sum(dim=-1), key.square().sum(dim=-1))
    assert set(rope.forward.__annotations__) == {"query", "key", "position_ids", "return"}


def test_gqa_sdpa_matches_explicit_float32_causal_attention() -> None:
    torch.manual_seed(7)
    config = _attention_config()
    attention = GroupedQueryAttention(config).eval()
    hidden = torch.randn(2, 4, config.d_model)
    positions = torch.arange(4).view(1, -1).expand(2, -1)

    actual = attention(hidden, position_ids=positions)
    query = attention.query_norm(attention._split_heads(attention.q_proj(hidden), config.n_heads))
    key = attention.key_norm(attention._split_heads(attention.k_proj(hidden), config.n_kv_heads))
    value = attention._split_heads(attention.v_proj(hidden), config.n_kv_heads)
    query, key = attention.rope(query, key, positions)
    key = key.repeat_interleave(config.n_heads // config.n_kv_heads, dim=1)
    value = value.repeat_interleave(config.n_heads // config.n_kv_heads, dim=1)
    scores = query @ key.transpose(-1, -2) / math.sqrt(config.head_dim)
    scores.masked_fill_(torch.ones(4, 4, dtype=torch.bool).triu(1), float("-inf"))
    reference = torch.softmax(scores.float(), dim=-1) @ value.float()
    reference = reference.transpose(1, 2).contiguous().view(2, 4, config.d_model)
    reference = attention.output_proj(reference)

    torch.testing.assert_close(actual, reference, rtol=2e-5, atol=2e-6)


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
