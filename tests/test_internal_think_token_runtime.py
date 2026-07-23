from __future__ import annotations

from types import SimpleNamespace

import pytest
torch = pytest.importorskip("torch")
from torch import nn

from training.internal_think_token_runtime import (
    forced_loop_accounting,
    install_internal_control_tokens,
    mask_internal_control_logits,
)
from training.internal_think_token_spec import INTERNAL_CONTROL_TOKENS


class TinyTokenizer:
    def __init__(self) -> None:
        self.vocab = {"A": 0, "B": 1, "C": 2, "D": 3, "E": 4}

    def get_vocab(self) -> dict[str, int]:
        return dict(self.vocab)

    def add_special_tokens(self, payload: dict) -> int:
        added = 0
        for token in payload["additional_special_tokens"]:
            if token not in self.vocab:
                self.vocab[token] = len(self.vocab)
                added += 1
        return added

    def convert_tokens_to_ids(self, token: str) -> int:
        return self.vocab[token]

    def __len__(self) -> int:
        return len(self.vocab)


class TinyLM(nn.Module):
    def __init__(self, *, tied: bool) -> None:
        super().__init__()
        self.config = SimpleNamespace(tie_word_embeddings=tied, vocab_size=5)
        self.input = nn.Embedding(5, 4)
        self.output = nn.Linear(4, 5, bias=False)
        if tied:
            self.output.weight = self.input.weight

    def get_input_embeddings(self) -> nn.Module:
        return self.input

    def get_output_embeddings(self) -> nn.Module:
        return self.output

    def resize_token_embeddings(self, size: int, **_: object) -> None:
        old_input = self.input.weight.detach().clone()
        old_output = self.output.weight.detach().clone()
        tied = self.config.tie_word_embeddings
        self.input = nn.Embedding(size, old_input.shape[1])
        self.output = nn.Linear(old_output.shape[1], size, bias=False)
        with torch.no_grad():
            self.input.weight[: old_input.shape[0]].copy_(old_input)
            self.output.weight[: old_output.shape[0]].copy_(old_output)
        if tied:
            self.output.weight = self.input.weight
        self.config.vocab_size = size


@pytest.mark.parametrize("tied", [False, True])
def test_control_token_resize_preserves_policy_rows_and_pair_initialization(tied: bool) -> None:
    tokenizer = TinyTokenizer()
    model = TinyLM(tied=tied)

    receipt = install_internal_control_tokens(tokenizer, model)

    assert receipt.added_token_count == 3
    assert receipt.resized_vocab_size == 8
    assert receipt.embedding_lm_head_tied_after is tied
    assert receipt.new_row_pair_max_abs_diff == 0.0
    assert receipt.old_input_rows_max_abs_diff == 0.0
    assert receipt.old_output_rows_max_abs_diff == 0.0
    assert receipt.added_parameter_count == 12 * (1 if tied else 2)
    assert tuple(tokenizer.vocab[token] for token in INTERNAL_CONTROL_TOKENS) == (
        5,
        6,
        7,
    )


def test_visible_generation_mask_blocks_adversarially_high_control_logits() -> None:
    logits = torch.zeros(2, 8)
    logits[:, [5, 6, 7]] = 1000
    logits[:, 3] = 2

    masked = mask_internal_control_logits(logits, (5, 6, 7))

    assert masked.argmax(dim=-1).tolist() == [3, 3]
    assert torch.all(masked[:, [5, 6, 7]] == torch.finfo(masked.dtype).min)
    assert torch.equal(logits[:, [5, 6, 7]], torch.full((2, 3), 1000.0))


def test_forced_loop_accounting_requires_all_three_counts_to_agree() -> None:
    assert forced_loop_accounting(
        requested_loops=4,
        executed_loops=4,
        selected_loop=4,
    )["forced_counts_agree"]

    with pytest.raises(AssertionError, match="mismatch"):
        forced_loop_accounting(
            requested_loops=4,
            executed_loops=3,
            selected_loop=4,
        )
