from __future__ import annotations

import torch

from eval.eval_paper2_phase2_a2_tripwire_audit import (
    STOP_ATTEMPT,
    WINDOW_END,
    WINDOW_START,
    batch_hash,
    matched_reference_batches,
    reconstruct_scheduled_batches,
)


def test_schedule_reconstruction_is_deterministic_and_windowed() -> None:
    train_indices = torch.arange(1000)
    first = reconstruct_scheduled_batches(
        train_indices,
        seed=20260805,
        first_attempt=WINDOW_START,
        last_attempt=WINDOW_END,
    )
    second = reconstruct_scheduled_batches(
        train_indices,
        seed=20260805,
        first_attempt=WINDOW_START,
        last_attempt=WINDOW_END,
    )
    assert list(first) == list(range(WINDOW_START, WINDOW_END + 1))
    assert STOP_ATTEMPT in first
    assert batch_hash(first[STOP_ATTEMPT]) == batch_hash(second[STOP_ATTEMPT])
    assert first[STOP_ATTEMPT].numel() == 128


def test_matched_reference_reproduces_registered_batch_count() -> None:
    train_indices = torch.arange(1000)
    batches = matched_reference_batches(train_indices, seed=0)
    assert len(batches) == 51
    assert all(batch.numel() == 128 for batch in batches)
    assert batch_hash(batches[0]) != batch_hash(batches[-1])

