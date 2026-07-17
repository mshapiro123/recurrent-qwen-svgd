from __future__ import annotations

import torch
from torch import nn

from eval.eval_ponder_depth import summarize
from models.lora import LoRALinear
from models.recurrent_wrapper import LayerSplit
from training.checkpointing import trainable_state_dict
from training.peft_ponder_closure import (
    full_block_comparison,
    historical_archive_receipt,
    locked_spec,
    next_p1_action,
    p1_gate,
    p2_gate,
)
from training.train_unfrozen_recurrent import (
    assert_pretrained_base_frozen,
    configure_trainable_modules,
    hash_pretrained_base_parameters,
)


def active_summary(correct: list[int]) -> dict:
    return {
        "active_matrix": {
            str(depth): {
                str(depth): {"correct": value, "total": 64, "accuracy": value / 64}
            }
            for depth, value in enumerate(correct, start=1)
        }
    }


def test_locked_spec_matches_preregistered_optimizer_and_ladder() -> None:
    spec = locked_spec()
    assert [arm["rank"] for arm in spec["arms"]] == [16, 64, 256]
    assert spec["optimizer"] == "adamw"
    assert spec["bridge_projection_mode"] == "split"
    assert spec["bridge_prelude_lr_multiplier"] == 10.0
    assert spec["bridge_prelude_grad_multiplier"] == 1.0
    assert spec["p1_gate"]["correct_per_depth"] == 46
    assert spec["p1_steps"] == 6000
    assert spec["r256_rider_total_steps"] == 12000
    assert spec["p2_steps"] == 2000


def test_p1_gate_requires_46_of_64_at_every_depth() -> None:
    assert p1_gate(active_summary([46, 46, 46, 46]))["passed"] is True
    assert p1_gate(active_summary([64, 64, 45, 64]))["passed"] is False


def test_ladder_stops_on_first_pass_and_allows_one_r256_rider() -> None:
    assert next_p1_action([]) == "run_R16"
    assert next_p1_action([{"arm": "R16", "total_steps": 6000, "gate": {"passed": False}}]) == "run_R64"
    assert next_p1_action([{"arm": "R64", "total_steps": 6000, "gate": {"passed": False}}]) == "run_R256"
    assert next_p1_action([{"arm": "R256", "total_steps": 6000, "gate": {"passed": False}}]) == (
        "continue_R256_to_12000"
    )
    assert next_p1_action([{"arm": "R256", "total_steps": 12000, "gate": {"passed": False}}]) == (
        "close_P1_bounded_refutation"
    )
    assert next_p1_action([{"arm": "R64", "total_steps": 6000, "gate": {"passed": True}}]) == (
        "run_P2_on_first_pass"
    )


def test_full_block_comparison_reports_delta_and_interval() -> None:
    comparison = full_block_comparison(active_summary([64, 63, 63, 59]))
    assert comparison["1"]["delta"] == 0.0
    assert comparison["4"]["reference_accuracy"] == 0.921875
    assert len(comparison["4"]["wilson_95"]) == 2


def test_p2_gate_requires_all_four_readings() -> None:
    training = {
        "curriculum_trace": [
            {"metrics": {"loss": 2.0, "halting_kl": 0.20}},
            {"metrics": {"loss": 1.5, "halting_kl": 0.11}},
            {"metrics": {"loss": 1.2, "halting_kl": 0.10}},
            {"metrics": {"loss": 1.0, "halting_kl": 0.10}},
            {"metrics": {"loss": 0.9, "halting_kl": 0.10}},
        ]
    }
    evaluation = {
        "mean_expected_loops": 2.5,
        "learned_depth_accuracy": 0.90,
        "forced_depth_accuracy": 0.92,
    }
    assert p2_gate(training, evaluation)["passed"] is True
    evaluation["mean_expected_loops"] = 1.0
    assert p2_gate(training, evaluation)["passed"] is False


class TinyWrapper(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.layer_split = LayerSplit(1, 2)
        self.base_model = nn.Module()
        self.base_model.layers = nn.ModuleList([nn.Linear(3, 3), nn.Linear(3, 3)])
        self.qwen = self.base_model
        self.qwen.layers[1] = LoRALinear(self.qwen.layers[1], rank=2, alpha=4)
        self.bridge = nn.Linear(3, 3)
        self.halt_predictor = nn.Linear(3, 1)
        self.reentry_adapter = nn.Linear(3, 3)
        self.latent_trajectory = nn.Linear(3, 3)


def test_frozen_lora_mode_trains_only_lora_and_bridge() -> None:
    wrapper = TinyWrapper()
    before = hash_pretrained_base_parameters(wrapper)  # type: ignore[arg-type]
    configure_trainable_modules(
        wrapper,  # type: ignore[arg-type]
        {
            "training_mode": "frozen_lora",
            "train_auxiliary": {"bridge": True, "halting": False},
        },
    )
    assert_pretrained_base_frozen(wrapper)  # type: ignore[arg-type]
    trainable = [name for name, parameter in wrapper.named_parameters() if parameter.requires_grad]
    assert any(".lora_a." in name for name in trainable)
    assert any(".lora_b." in name for name in trainable)
    assert any(name.startswith("bridge.") for name in trainable)
    assert not any(name.startswith("halt_predictor.") for name in trainable)
    assert hash_pretrained_base_parameters(wrapper) == before  # type: ignore[arg-type]


def test_controller_only_mode_has_no_mechanism_gradients() -> None:
    wrapper = TinyWrapper()
    configure_trainable_modules(
        wrapper,  # type: ignore[arg-type]
        {
            "training_mode": "controller_only",
            "train_auxiliary": {"bridge": False, "halting": True},
        },
    )
    trainable = [name for name, parameter in wrapper.named_parameters() if parameter.requires_grad]
    assert trainable
    assert all(name.startswith("halt_predictor.") for name in trainable)


def test_p2_checkpoint_can_include_frozen_mechanism() -> None:
    wrapper = TinyWrapper()
    configure_trainable_modules(
        wrapper,  # type: ignore[arg-type]
        {
            "training_mode": "controller_only",
            "train_auxiliary": {"bridge": False, "halting": True},
        },
    )
    state = trainable_state_dict(
        wrapper,
        include_frozen_prefixes=("bridge.",),
        include_frozen_lora=True,
    )
    assert any(name.startswith("bridge.") for name in state)
    assert any(".lora_a." in name for name in state)
    assert any(name.startswith("halt_predictor.") for name in state)


def test_ponder_eval_summary_uses_forced_and_selected_depths() -> None:
    payload = summarize(
        [
            {"forced_hit": True, "learned_hit": True, "selected_loop": 2, "expected_loops": 2.2},
            {"forced_hit": True, "learned_hit": False, "selected_loop": 1, "expected_loops": 1.8},
        ]
    )
    assert payload["forced_depth_accuracy"] == 1.0
    assert payload["learned_depth_accuracy"] == 0.5
    assert payload["mean_expected_loops"] == 2.0


def test_historical_peft_arms_are_classified_as_pre_repair() -> None:
    receipt = historical_archive_receipt()
    assert receipt["repaired_loop_peft_arm_found"] is False
    assert receipt["inadmissible_runs"][0]["ranks"] == [32, 64, 128]
