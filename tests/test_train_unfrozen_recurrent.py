from __future__ import annotations

import torch
from torch import nn

from models.recurrent_wrapper import LayerSplit
from training.train_unfrozen_recurrent import (
    configure_trainable_modules,
    curriculum_target_counts,
    scheduled_loop_count,
    trainable_parameter_summary,
)


class TinyWrapper(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.layer_split = LayerSplit(prelude_end=1, recurrent_end=3)
        self.base_model = nn.Module()
        self.qwen = nn.Module()
        self.qwen.layers = nn.ModuleList([nn.Linear(2, 2) for _ in range(4)])
        self.base_model.layers_alias = self.qwen.layers
        self.bridge = nn.Linear(2, 2)
        self.halt_predictor = nn.Linear(2, 2)
        self.reentry_adapter = nn.Linear(2, 2)
        self.latent_trajectory = nn.Linear(2, 2)


def test_scheduled_loop_count_linear_reaches_bounds() -> None:
    assert scheduled_loop_count(0, 5, start=1, end=8) == 1
    assert scheduled_loop_count(4, 5, start=1, end=8) == 8
    assert 1 < scheduled_loop_count(2, 5, start=1, end=8) < 8


def test_scheduled_loop_count_one_minus_sqrt_reaches_end() -> None:
    assert scheduled_loop_count(0, 10, start=1, end=8, schedule="one_minus_sqrt") == 1
    assert scheduled_loop_count(9, 10, start=1, end=8, schedule="one_minus_sqrt") == 8


def test_curriculum_target_counts_modes() -> None:
    row_targets = torch.tensor([1, 3, 8])

    assert curriculum_target_counts(row_targets, 4, mode="schedule").tolist() == [4, 4, 4]
    assert curriculum_target_counts(row_targets, 4, mode="row_capped").tolist() == [1, 3, 4]
    assert curriculum_target_counts(row_targets, 4, mode="row_or_schedule_max").tolist() == [4, 4, 8]


def test_configure_trainable_modules_unfreezes_only_recurrent_block_by_default() -> None:
    wrapper = TinyWrapper()

    configure_trainable_modules(wrapper, {"train_auxiliary": {"bridge": True, "halting": True}})

    assert not any(param.requires_grad for param in wrapper.qwen.layers[0].parameters())
    assert all(param.requires_grad for param in wrapper.qwen.layers[1].parameters())
    assert all(param.requires_grad for param in wrapper.qwen.layers[2].parameters())
    assert not any(param.requires_grad for param in wrapper.qwen.layers[3].parameters())
    assert all(param.requires_grad for param in wrapper.bridge.parameters())
    assert all(param.requires_grad for param in wrapper.halt_predictor.parameters())
    assert not any(param.requires_grad for param in wrapper.reentry_adapter.parameters())
    summary = trainable_parameter_summary(wrapper)  # type: ignore[arg-type]
    assert summary["recurrent_block"] > 0
    assert summary["total"] >= summary["recurrent_block"]
