"""Executable Phase T0 control-token preparation contracts."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Iterable

import torch

from training.internal_think_token_spec import (
    INTERNAL_CONTROL_TOKENS,
    validate_tokenizer_preflight,
)


@dataclass(frozen=True)
class ControlTokenResizeReceipt:
    original_vocab_size: int
    resized_vocab_size: int
    added_token_count: int
    control_token_ids: tuple[int, int, int]
    tie_word_embeddings_before: bool
    tie_word_embeddings_after: bool
    embedding_lm_head_tied_before: bool
    embedding_lm_head_tied_after: bool
    added_parameter_count: int
    old_input_rows_max_abs_diff: float
    old_output_rows_max_abs_diff: float
    new_row_pair_max_abs_diff: float

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["control_token_ids"] = list(self.control_token_ids)
        return payload


def _tied(input_embedding: Any, output_embedding: Any) -> bool:
    return input_embedding.weight.data_ptr() == output_embedding.weight.data_ptr()


def _unique_embedding_parameter_count(model: Any) -> int:
    parameters = (
        model.get_input_embeddings().weight,
        model.get_output_embeddings().weight,
    )
    seen: set[int] = set()
    total = 0
    for parameter in parameters:
        pointer = parameter.data_ptr()
        if pointer not in seen:
            total += parameter.numel()
            seen.add(pointer)
    return total


def install_internal_control_tokens(tokenizer: Any, model: Any) -> ControlTokenResizeReceipt:
    """Add exactly three internal tokens while preserving old rows and tie policy."""

    original_vocab = dict(tokenizer.get_vocab())
    validate_tokenizer_preflight(
        existing_vocabulary=original_vocab,
        added_token_count=len(INTERNAL_CONTROL_TOKENS),
    )
    input_before = model.get_input_embeddings()
    output_before = model.get_output_embeddings()
    old_input = input_before.weight.detach().clone()
    old_output = output_before.weight.detach().clone()
    original_size = int(old_input.shape[0])
    policy_before = bool(getattr(model.config, "tie_word_embeddings", False))
    tied_before = _tied(input_before, output_before)
    if tied_before != policy_before:
        raise AssertionError(
            "Base embedding pointer sharing disagrees with tie_word_embeddings policy"
        )
    parameters_before = _unique_embedding_parameter_count(model)

    added = int(
        tokenizer.add_special_tokens(
            {"additional_special_tokens": list(INTERNAL_CONTROL_TOKENS)}
        )
    )
    if added != len(INTERNAL_CONTROL_TOKENS):
        raise AssertionError(f"Expected exactly three added control tokens, observed {added}")
    try:
        model.resize_token_embeddings(len(tokenizer), mean_resizing=False)
    except TypeError:  # Older Transformers releases do not expose mean_resizing.
        model.resize_token_embeddings(len(tokenizer))

    input_after = model.get_input_embeddings()
    output_after = model.get_output_embeddings()
    if bool(getattr(model.config, "tie_word_embeddings", False)) != policy_before:
        raise AssertionError("Vocabulary resize changed tie_word_embeddings policy")
    tied_after = _tied(input_after, output_after)
    if tied_after != tied_before:
        raise AssertionError("Vocabulary resize changed embedding/LM-head pointer sharing")
    if int(input_after.weight.shape[0]) != original_size + 3:
        raise AssertionError("Input embedding did not gain exactly three rows")
    if int(output_after.weight.shape[0]) != original_size + 3:
        raise AssertionError("LM head did not gain exactly three rows")

    control_ids = tuple(int(tokenizer.convert_tokens_to_ids(token)) for token in INTERNAL_CONTROL_TOKENS)
    if len(set(control_ids)) != 3 or min(control_ids) < original_size:
        raise AssertionError(f"Control token IDs are not three new rows: {control_ids}")
    initial_row = old_input.float().mean(dim=0).to(
        device=input_after.weight.device,
        dtype=input_after.weight.dtype,
    )
    with torch.no_grad():
        for token_id in control_ids:
            input_after.weight[token_id].copy_(initial_row)
            output_after.weight[token_id].copy_(
                initial_row.to(
                    device=output_after.weight.device,
                    dtype=output_after.weight.dtype,
                )
            )

    input_diff = float(
        (input_after.weight[:original_size].float() - old_input.float()).abs().max().item()
    )
    output_diff = float(
        (output_after.weight[:original_size].float() - old_output.float()).abs().max().item()
    )
    pair_diff = max(
        float(
            (
                input_after.weight[token_id].float()
                - output_after.weight[token_id].float()
            )
            .abs()
            .max()
            .item()
        )
        for token_id in control_ids
    )
    if input_diff != 0.0 or output_diff != 0.0:
        raise AssertionError(
            f"Vocabulary resize changed old rows: input={input_diff}, output={output_diff}"
        )
    if pair_diff != 0.0:
        raise AssertionError(f"New input/output row pairs differ by {pair_diff}")

    added_parameters = _unique_embedding_parameter_count(model) - parameters_before
    expected_parameters = 3 * int(input_after.weight.shape[1]) * (1 if tied_after else 2)
    if added_parameters != expected_parameters:
        raise AssertionError(
            f"Added parameter count {added_parameters} != expected {expected_parameters}"
        )
    return ControlTokenResizeReceipt(
        original_vocab_size=original_size,
        resized_vocab_size=original_size + 3,
        added_token_count=added,
        control_token_ids=control_ids,
        tie_word_embeddings_before=policy_before,
        tie_word_embeddings_after=policy_before,
        embedding_lm_head_tied_before=tied_before,
        embedding_lm_head_tied_after=tied_after,
        added_parameter_count=added_parameters,
        old_input_rows_max_abs_diff=input_diff,
        old_output_rows_max_abs_diff=output_diff,
        new_row_pair_max_abs_diff=pair_diff,
    )


def mask_internal_control_logits(
    logits: torch.Tensor,
    control_token_ids: Iterable[int],
) -> torch.Tensor:
    """Return visible-generation logits with all internal controls impossible."""

    token_ids = tuple(int(value) for value in control_token_ids)
    if not token_ids:
        raise ValueError("At least one internal control-token ID is required")
    masked = logits.clone()
    masked[..., list(token_ids)] = torch.finfo(masked.dtype).min
    return masked


def forced_loop_accounting(
    *,
    requested_loops: int,
    executed_loops: int,
    selected_loop: int,
) -> dict[str, Any]:
    receipt = {
        "requested_loops": int(requested_loops),
        "executed_loops": int(executed_loops),
        "selected_loop": int(selected_loop),
    }
    receipt["forced_counts_agree"] = len(set(receipt.values())) == 1
    if not receipt["forced_counts_agree"]:
        raise AssertionError(f"Forced loop accounting mismatch: {receipt}")
    return receipt
