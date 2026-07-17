from __future__ import annotations

import torch

from training.phase_g_training import (
    PhaseGEMA,
    first_active_loop_token_ids,
    posterior_target_embeddings,
)


class TinyBase(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.embedding = torch.nn.Embedding(16, 4)

    def get_input_embeddings(self):
        return self.embedding


def test_phase_g_targets_use_first_active_symbol_and_detach_keeper() -> None:
    labels = torch.tensor(
        [
            [
                [-100, -100, 3, 4],
                [-100, 7, 8, 9],
            ]
        ]
    )
    base = TinyBase()

    assert first_active_loop_token_ids(labels).tolist() == [[3, 7]]
    targets = posterior_target_embeddings(base, labels)
    assert targets.shape == (1, 2, 4)
    assert targets.requires_grad is False


def test_phase_g_targets_reject_unsupervised_requested_loop() -> None:
    labels = torch.full((1, 2, 3), -100)
    try:
        first_active_loop_token_ids(labels)
    except ValueError as exc:
        assert "needs a gold symbol" in str(exc)
    else:
        raise AssertionError("Missing posterior targets must fail")


def test_phase_g_ema_updates_copies_and_restores() -> None:
    module = torch.nn.Linear(2, 1)
    ema = PhaseGEMA(module.named_parameters(), decay=0.5)
    original = {name: value.detach().clone() for name, value in module.named_parameters()}
    with torch.no_grad():
        for parameter in module.parameters():
            parameter.add_(2.0)
    ema.update(module.named_parameters())
    trained = {name: value.detach().clone() for name, value in module.named_parameters()}

    backup = ema.copy_to(module.named_parameters())
    for name, parameter in module.named_parameters():
        assert torch.allclose(parameter, original[name] + 1.0)
    PhaseGEMA.restore(module.named_parameters(), backup)
    for name, parameter in module.named_parameters():
        assert torch.equal(parameter, trained[name])
