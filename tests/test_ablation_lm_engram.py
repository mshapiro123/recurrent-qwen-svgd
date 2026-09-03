from __future__ import annotations

import math

import pytest
import torch

from models.ablation_lm.engram import CausalTokenEngram, TokenEngramConfig


def _tiny_engram(*, seed: int = 17) -> CausalTokenEngram:
    return CausalTokenEngram(
        TokenEngramConfig(
            hidden_dim=8,
            num_slots=31,
            ngram_orders=(2, 3),
            num_hash_heads=2,
            head_dim=3,
            initial_scale=1.0e-3,
            max_scale=0.1,
            seed=seed,
        )
    )


def test_disabled_engram_is_exact_structural_bypass() -> None:
    module = _tiny_engram()
    hidden = torch.randn(2, 5, 8, requires_grad=True)
    tokens = torch.tensor([[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]])

    output, audit = module(hidden, tokens, enabled=False)

    assert output is hidden
    assert not bool(audit["enabled"])
    assert set(audit) == {"enabled"}


def test_suffix_keys_are_causal_and_future_changes_do_not_propagate_left() -> None:
    module = _tiny_engram()
    generator = torch.Generator().manual_seed(23)
    hidden = torch.randn(1, 7, 8, generator=generator)
    tokens = torch.tensor([[4, 8, 15, 16, 23, 42, 99]])
    changed_hidden = hidden.clone()
    changed_hidden[:, 4:] = torch.randn(1, 3, 8, generator=generator)
    changed_tokens = tokens.clone()
    changed_tokens[:, 4:] = torch.tensor([[3, 2, 1]])

    output, audit = module(hidden, tokens)
    changed_output, changed_audit = module(changed_hidden, changed_tokens)

    torch.testing.assert_close(output[:, :4], changed_output[:, :4], rtol=0, atol=0)
    for name in ("indices_n2_h0", "indices_n2_h1", "indices_n3_h0", "indices_n3_h1"):
        torch.testing.assert_close(
            audit[name][:, :4], changed_audit[name][:, :4], rtol=0, atol=0
        )


def test_document_boundaries_invalidate_cross_document_suffixes() -> None:
    module = _tiny_engram()
    hidden = torch.randn(1, 6, 8, generator=torch.Generator().manual_seed(29))
    tokens = torch.tensor([[10, 11, 20, 21, 22, 30]])
    documents = torch.tensor([[0, 0, 1, 1, 1, -1]])

    output, audit = module(hidden, tokens, document_ids=documents)

    assert audit["valid_n2"].tolist() == [[False, True, False, True, True, False]]
    assert audit["valid_n3"].tolist() == [[False, False, False, False, True, False]]
    for head in range(2):
        assert audit[f"indices_n2_h{head}"][0, 2].item() == -1
        assert audit[f"indices_n3_h{head}"][0, 3].item() == -1
        assert audit[f"indices_n2_h{head}"][0, 5].item() == -1
    torch.testing.assert_close(output[:, 2], hidden[:, 2], rtol=0, atol=0)
    torch.testing.assert_close(output[:, 5], hidden[:, 5], rtol=0, atol=0)

    changed_tokens = tokens.clone()
    changed_tokens[:, :2] = torch.tensor([[1, 2]])
    changed_output, _ = module(hidden, changed_tokens, document_ids=documents)
    torch.testing.assert_close(output[:, 2:], changed_output[:, 2:], rtol=0, atol=0)


def test_active_arm_has_small_nonzero_scale_and_live_gradients() -> None:
    module = _tiny_engram()
    hidden = torch.randn(
        2, 6, 8, generator=torch.Generator().manual_seed(31), requires_grad=True
    )
    tokens = torch.tensor(
        [[1, 2, 3, 4, 5, 6], [7, 8, 9, 10, 11, 12]], dtype=torch.long
    )
    documents = torch.zeros_like(tokens)

    output, audit = module(hidden, tokens, document_ids=documents)
    weights = torch.linspace(0.5, 1.5, output.numel()).reshape_as(output)
    loss = (output * weights).sum()
    loss.backward()

    torch.testing.assert_close(
        audit["residual_scale"], torch.tensor(1.0e-3), rtol=1.0e-5, atol=1.0e-8
    )
    assert not torch.equal(output.detach(), hidden.detach())
    assert hidden.grad is not None and torch.count_nonzero(hidden.grad)
    assert module.raw_residual_scale.grad is not None
    assert module.raw_residual_scale.grad.abs().item() > 0
    assert module.gate_bias.grad is not None and module.gate_bias.grad.abs().item() > 0
    for projection in (module.query_proj, module.value_proj):
        assert projection.weight.grad is not None
        assert projection.weight.grad.abs().sum().item() > 0

    for name, table in module.tables.items():
        assert table.weight.grad is not None
        hit_rows = audit[f"indices_{name}"]
        hit_rows = hit_rows[hit_rows.ge(0)].unique()
        assert hit_rows.numel() > 0
        assert table.weight.grad[hit_rows].abs().sum().item() > 0


def test_seed_controls_all_table_and_projection_initialization() -> None:
    first = _tiny_engram(seed=41)
    second = _tiny_engram(seed=41)

    for (first_name, first_parameter), (second_name, second_parameter) in zip(
        first.named_parameters(), second.named_parameters(), strict=True
    ):
        assert first_name == second_name
        torch.testing.assert_close(first_parameter, second_parameter, rtol=0, atol=0)


def test_gate_is_exact_memory_space_dot_with_memory_width_divisor() -> None:
    module = _tiny_engram()
    hidden = torch.randn(1, 5, 8, generator=torch.Generator().manual_seed(43))
    tokens = torch.tensor([[2, 3, 5, 7, 11]])

    _, audit = module(hidden, tokens)

    assert module.memory_dim == 12
    assert module.query_proj.weight.shape == (module.memory_dim, module.hidden_dim)
    assert not hasattr(module, "key_proj")
    retrieved = []
    for order in module.ngram_orders:
        for head in range(module.num_hash_heads):
            name = module._table_name(order, head)
            indices = audit[f"indices_{name}"]
            valid = indices.ge(0)
            rows = module.tables[name](indices.masked_fill(~valid, 0))
            retrieved.append(rows * valid.unsqueeze(-1).to(rows.dtype))
    normalized_memory = module.memory_norm(torch.cat(retrieved, dim=-1))
    query = module.query_norm(module.query_proj(hidden))
    expected_gate = torch.sigmoid(
        (query.float() * normalized_memory.float()).sum(dim=-1, keepdim=True)
        / math.sqrt(module.memory_dim)
        + module.gate_bias.float()
    )
    expected_gate = expected_gate * audit["has_memory"].to(expected_gate.dtype)
    torch.testing.assert_close(audit["gate"], expected_gate, rtol=0, atol=0)


@pytest.mark.parametrize(
    "updates",
    (
        {"num_slots": 31.9},
        {"ngram_orders": (2.5, 3)},
        {"seed": 17.9},
        {"max_scale": float("inf")},
    ),
)
def test_public_engram_config_rejects_silent_coercions_and_nonfinite_scales(
    updates: dict[str, object],
) -> None:
    config = TokenEngramConfig(hidden_dim=8, **updates)

    with pytest.raises(ValueError):
        CausalTokenEngram(config)
