import math

import pytest
import torch

from models.sidecar_v2 import LiteralNGramMemory, fast_wht


def dense_hadamard(width: int, *, dtype: torch.dtype = torch.float64) -> torch.Tensor:
    matrix = torch.ones((1, 1), dtype=dtype)
    while matrix.shape[0] < width:
        matrix = torch.cat(
            (
                torch.cat((matrix, matrix), dim=1),
                torch.cat((matrix, -matrix), dim=1),
            ),
            dim=0,
        )
    return matrix / math.sqrt(width)


@pytest.mark.parametrize("width", [1, 2, 8, 256])
def test_fast_wht_matches_dense_right_multiplication(width: int) -> None:
    generator = torch.Generator().manual_seed(20260815 + width)
    values = torch.randn((3, 5, width), generator=generator, dtype=torch.float64)
    expected = values @ dense_hadamard(width).T
    torch.testing.assert_close(fast_wht(values), expected, rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize("width", [2, 8, 64])
def test_fast_wht_matches_dense_left_multiplication(width: int) -> None:
    generator = torch.Generator().manual_seed(20260816 + width)
    columns = torch.randn((width, 7), generator=generator, dtype=torch.float64)
    expected = dense_hadamard(width) @ columns
    actual = fast_wht(columns.T).T
    torch.testing.assert_close(actual, expected, rtol=1e-12, atol=1e-12)


def test_fast_wht_preserves_norm_and_gradient() -> None:
    values = torch.randn((4, 16), dtype=torch.float64, requires_grad=True)
    transformed = fast_wht(values)
    torch.testing.assert_close(
        transformed.square().sum(-1), values.square().sum(-1), rtol=1e-12, atol=1e-12
    )
    transformed.sum().backward()
    assert values.grad is not None
    assert torch.isfinite(values.grad).all()


@pytest.mark.parametrize("width", [0, 3, 12])
def test_fast_wht_rejects_invalid_width(width: int) -> None:
    with pytest.raises(ValueError, match="power-of-two"):
        fast_wht(torch.empty((2, width)))


def test_literal_ngram_memory_is_causal_and_deterministic() -> None:
    memory = LiteralNGramMemory(value_dim=6, num_slots=257, seed=19)
    tokens = torch.tensor([[4, 8, 15, 16, 23, 42], [4, 8, 15, 99, 23, 42]])
    values, audit = memory(tokens)
    repeat, repeat_audit = memory(tokens.clone())
    torch.testing.assert_close(values, repeat)
    assert audit.keys() == repeat_audit.keys()
    for key in audit:
        torch.testing.assert_close(audit[key], repeat_audit[key])
    # Changing token 4 cannot affect any retrieval ending before token 4.
    torch.testing.assert_close(values[0, :3], values[1, :3])
    assert values.shape == (2, 6, 6)
    assert set(audit) == {"n2_h0", "n2_h1", "n3_h0", "n3_h1"}


def test_literal_ngram_memory_has_zero_substrate_contact() -> None:
    memory = LiteralNGramMemory(value_dim=4, num_slots=31, ngram_sizes=(1, 2))
    tokens = torch.tensor([[1, 2, 3]], dtype=torch.long)
    before = tokens.clone()
    values, _ = memory(tokens)
    values.sum().backward()
    torch.testing.assert_close(tokens, before)
    assert all(parameter.grad is not None for parameter in memory.parameters())

