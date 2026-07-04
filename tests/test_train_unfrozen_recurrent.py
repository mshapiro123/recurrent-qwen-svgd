from __future__ import annotations

import torch
from torch import nn

from models.bridge import IdentityGatedBridge
from models.recurrent_wrapper import LayerSplit
from training.train_unfrozen_recurrent import (
    apply_bridge_prelude_grad_multiplier,
    bridge_prelude_optimizer_parameters,
    bridge_prelude_optimizer_setup,
    bridge_prelude_grad_stats,
    bridge_prelude_weight_stats,
    bridge_uses_split_projection,
    build_optimizer,
    chain_label_weight,
    configure_trainable_modules,
    cosine_with_previous,
    curriculum_target_counts,
    resolve_resume_lora_config,
    scheduled_loop_count,
    trainable_parameter_norm_stats,
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


def test_chain_label_weight_ramps_then_holds_zero() -> None:
    assert chain_label_weight(0, 10, hold_frac=0.5) == 1.0
    assert chain_label_weight(2, 10, hold_frac=0.5) == 0.6
    assert chain_label_weight(5, 10, hold_frac=0.5) == 0.0
    assert chain_label_weight(9, 10, hold_frac=0.5) == 0.0
    assert chain_label_weight(0, 10, hold_frac=1.0) == 0.0


def test_resolve_resume_lora_config_reads_checkpoint_config(tmp_path) -> None:
    checkpoint = tmp_path / "phase1.pt"
    torch.save({"config": {"lora": {"rank": 64, "alpha": 128}}}, checkpoint)

    resolved = resolve_resume_lora_config(
        {
            "resume_from": str(checkpoint),
            "resume_lora": {"enabled": True, "rank": "auto", "alpha": "auto", "dropout": 0.0},
        }
    )

    assert resolved["rank"] == 64
    assert resolved["alpha"] == 128.0


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


def test_trainable_parameter_norm_stats_groups_recurrent_and_bridge_params() -> None:
    wrapper = TinyWrapper()
    configure_trainable_modules(wrapper, {"train_auxiliary": {"bridge": True, "halting": False}})

    stats = trainable_parameter_norm_stats(wrapper)  # type: ignore[arg-type]

    assert stats["trainable_total_param_numel"] > 0
    assert stats["recurrent_block_param_numel"] > 0
    assert stats["bridge_param_numel"] > 0
    assert stats["halting_param_numel"] == 0.0
    assert stats["trainable_total_param_rms"] > 0.0
    assert stats["recurrent_block_param_l2"] > 0.0


class TinyBridgeWrapper(nn.Module):
    def __init__(self, *, split: bool = False) -> None:
        super().__init__()
        self.bridge = IdentityGatedBridge(2, projection_mode="split" if split else "concat")


def test_bridge_prelude_grad_multiplier_scales_only_prelude_half() -> None:
    wrapper = TinyBridgeWrapper()
    wrapper.bridge.proj.weight.grad = torch.ones_like(wrapper.bridge.proj.weight)

    stats = apply_bridge_prelude_grad_multiplier(wrapper, 10.0)  # type: ignore[arg-type]

    grad = wrapper.bridge.proj.weight.grad
    assert grad is not None
    assert torch.allclose(grad[:, :2], torch.full_like(grad[:, :2], 10.0))
    assert torch.allclose(grad[:, 2:], torch.ones_like(grad[:, 2:]))
    assert stats["bridge_prelude_grad_multiplier"] == 10.0
    assert stats["bridge_prelude_grad_rms_after_multiplier"] > stats["bridge_prelude_grad_rms"]


def test_bridge_prelude_weight_stats_reports_warm_start() -> None:
    wrapper = TinyBridgeWrapper()

    stats = bridge_prelude_weight_stats(wrapper)  # type: ignore[arg-type]

    assert stats["bridge_prelude_weight_rms"] == 0.0
    assert stats["bridge_prelude_weight_max_abs"] == 0.0
    assert stats["bridge_state_identity_max_abs_diff"] == 0.0


def test_split_bridge_prelude_weight_stats_reports_warm_start() -> None:
    wrapper = TinyBridgeWrapper(split=True)

    stats = bridge_prelude_weight_stats(wrapper)  # type: ignore[arg-type]

    assert bridge_uses_split_projection(wrapper) is True  # type: ignore[arg-type]
    assert stats["bridge_prelude_weight_rms"] == 0.0
    assert stats["bridge_prelude_weight_max_abs"] == 0.0
    assert stats["bridge_state_identity_max_abs_diff"] == 0.0


def test_bridge_prelude_grad_stats_handles_missing_grad() -> None:
    wrapper = TinyBridgeWrapper()

    stats = bridge_prelude_grad_stats(wrapper)  # type: ignore[arg-type]

    assert stats == {"bridge_prelude_grad_rms": 0.0, "bridge_state_grad_rms": 0.0}


def test_adamw_split_bridge_builds_true_prelude_lr_group() -> None:
    wrapper = TinyBridgeWrapper(split=True)
    cfg = {
        "optimizer": "adamw",
        "learning_rate": 1e-4,
        "weight_decay": 0.01,
        "bridge_prelude_lr_multiplier": 10.0,
        "bridge_prelude_weight_decay": 0.0,
    }

    optimizer = build_optimizer(wrapper, cfg)  # type: ignore[arg-type]
    prelude_params = bridge_prelude_optimizer_parameters(wrapper, cfg)  # type: ignore[arg-type]
    setup = bridge_prelude_optimizer_setup(optimizer, prelude_params, expected_lr=1e-3)

    assert len(optimizer.param_groups) == 2
    assert setup["bridge_prelude_optimizer_group_ok"] is True
    assert setup["bridge_prelude_optimizer_group_lr"] == 1e-3
    assert setup["bridge_prelude_optimizer_group_weight_decay"] == 0.0


def test_prelude_lr_multiplier_requires_split_bridge() -> None:
    wrapper = TinyBridgeWrapper()

    try:
        build_optimizer(
            wrapper,  # type: ignore[arg-type]
            {
                "optimizer": "adamw",
                "learning_rate": 1e-4,
                "bridge_prelude_lr_multiplier": 10.0,
            },
        )
    except ValueError as exc:
        assert "bridge_projection_mode='split'" in str(exc)
    else:
        raise AssertionError("Expected true prelude LR to require split bridge")


def test_cosine_with_previous_reports_gradient_persistence() -> None:
    first = torch.tensor([1.0, 0.0])
    second = torch.tensor([0.0, 1.0])

    cosine, previous = cosine_with_previous(first, None)
    assert cosine == 0.0
    assert torch.allclose(previous, first)

    cosine, previous = cosine_with_previous(first, previous)
    assert cosine == 1.0
    assert torch.allclose(previous, first)

    cosine, previous = cosine_with_previous(second, previous)
    assert cosine == 0.0
    assert torch.allclose(previous, second)
