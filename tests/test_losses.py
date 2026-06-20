from __future__ import annotations

import pytest
import torch

from training.losses import causal_kl_distillation_loss


def test_causal_kl_distillation_zero_for_matching_logits() -> None:
    logits = torch.randn(2, 4, 7)
    mask = torch.ones(2, 4, dtype=torch.bool)
    loss = causal_kl_distillation_loss(logits, logits.clone(), mask)
    assert loss.item() == pytest.approx(0.0, abs=1e-6)


def test_causal_kl_distillation_respects_mask() -> None:
    student = torch.zeros(1, 3, 2)
    teacher = torch.zeros(1, 3, 2)
    teacher[:, 0, 0] = 5.0
    teacher[:, 1, 1] = 5.0
    mask_first_shifted_token_only = torch.tensor([[False, True, False]])
    mask_no_tokens = torch.zeros(1, 3, dtype=torch.bool)
    active_loss = causal_kl_distillation_loss(student, teacher, mask_first_shifted_token_only)
    empty_loss = causal_kl_distillation_loss(student, teacher, mask_no_tokens)
    assert active_loss.item() > 0
    assert empty_loss.item() == pytest.approx(0.0, abs=1e-6)


def test_causal_kl_distillation_rejects_bad_shapes() -> None:
    with pytest.raises(ValueError):
        causal_kl_distillation_loss(torch.zeros(1, 2, 3), torch.zeros(1, 2, 4), torch.ones(1, 2))
