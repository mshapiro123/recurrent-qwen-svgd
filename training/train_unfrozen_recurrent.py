"""Train the recurrent block itself with a loop-count curriculum.

This is the bounded unfreeze+Muon branch after the rank-only LoRA capacity
arm. It keeps the existing split and bridge path fixed, merges any recovered
LoRA initialization into the recurrent block, then trains the recurrent block's
own parameters plus lightweight loop controls.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import shutil
import sys
from functools import partial
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import torch
import yaml
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from eval.eval_identity import model_load_kwargs, parse_split, resolve_dtype
from models.lora import apply_lora_to_recurrent_block, mark_only_lora_trainable, merge_lora_adapters
from models.recurrent_wrapper import RecurrentQwenForCausalLM
from training.checkpointing import load_trainable_checkpoint, save_trainable_checkpoint
from training.dataset import JsonlCausalDataset, collate_causal_batch
from training.muon import Muon, OptimizerBundle, split_muon_and_adamw_params
from training.staircase_curriculum import LoopDoseLedger, assert_mass_equalized, exposure_fractions
from training.stability import (
    assert_finite_trainable_gradients,
    assert_finite_trainable_parameters,
    assert_finite_training_state,
)


def load_config(path: str | Path) -> dict[str, Any]:
    return yaml.safe_load(Path(path).read_text(encoding="utf-8"))


def write_training_progress(path: str | Path, payload: dict[str, Any]) -> None:
    """Atomically persist a compact liveness receipt for long Colab training jobs."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(destination)


def backup_checkpoint(checkpoint: Path, backup_dir: str | Path | None) -> Path | None:
    if not backup_dir:
        return None
    destination = Path(backup_dir) / checkpoint.name
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(checkpoint, destination)
    return destination


def cfg_float(cfg: dict[str, Any], key: str, default: float) -> float:
    value = cfg.get(key, default)
    return default if value is None else float(value)


def cfg_int(cfg: dict[str, Any], key: str, default: int) -> int:
    value = cfg.get(key, default)
    return default if value is None else int(value)


def resolve_canary_specs(cfg: dict[str, Any]) -> list[dict[str, Any]]:
    """Normalize legacy or named canary configuration into one validated list."""

    raw_specs = cfg.get("canary_specs")
    if raw_specs is None:
        data_jsonl = cfg.get("canary_jsonl")
        if not data_jsonl:
            return []
        if cfg.get("canary_baseline_accuracy") is None:
            raise ValueError("canary_jsonl requires canary_baseline_accuracy")
        raw_specs = [
            {
                "name": "legacy",
                "data_jsonl": data_jsonl,
                "baseline_accuracy": cfg["canary_baseline_accuracy"],
                "hard_stop_delta": cfg.get("canary_hard_stop_delta", -0.03),
                "value_prefix": cfg.get("canary_value_prefix", "name:"),
                "mode": "loop1",
                "max_depth": 1,
            }
        ]
    if not isinstance(raw_specs, list) or not raw_specs:
        raise ValueError("canary_specs must be a non-empty list")

    normalized: list[dict[str, Any]] = []
    names: set[str] = set()
    for index, raw in enumerate(raw_specs):
        if not isinstance(raw, dict):
            raise ValueError(f"canary_specs[{index}] must be a mapping")
        name = str(raw.get("name") or "").strip()
        if not name:
            raise ValueError(f"canary_specs[{index}] requires a name")
        if name in names:
            raise ValueError("canary_specs names must be unique")
        names.add(name)
        data_jsonl = str(raw.get("data_jsonl") or "").strip()
        if not data_jsonl:
            raise ValueError(f"canary_specs[{index}] requires data_jsonl")
        if raw.get("baseline_accuracy") is None:
            raise ValueError(f"canary_specs[{index}] requires baseline_accuracy")
        mode = str(raw.get("mode", "loop1")).lower()
        if mode not in {"loop1", "diagonal"}:
            raise ValueError("canary mode must be one of: loop1, diagonal")
        max_depth = int(raw.get("max_depth", 1))
        if max_depth < 1:
            raise ValueError("canary max_depth must be positive")
        normalized.append(
            {
                "name": name,
                "data_jsonl": data_jsonl,
                "baseline_accuracy": float(raw["baseline_accuracy"]),
                "hard_stop_delta": float(raw.get("hard_stop_delta", -0.03)),
                "value_prefix": str(raw.get("value_prefix", "name:")),
                "mode": mode,
                "max_depth": max_depth,
            }
        )
    return normalized


def seed_training_rng(seed: int) -> torch.Generator:
    """Seed model stochasticity and return a seeded CPU DataLoader generator."""

    seed = int(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    return torch.Generator(device="cpu").manual_seed(seed)


def supervision_counts(item: dict[str, torch.Tensor]) -> dict[str, int]:
    loop_labels = item.get("loop_labels")
    return {
        "outcome_tokens": int(item["labels"].ne(-100).sum().item()),
        "loop_tokens": int(loop_labels.ne(-100).sum().item()) if loop_labels is not None else 0,
        "active_loops": (
            int(loop_labels.ne(-100).any(dim=-1).sum().item()) if loop_labels is not None else 0
        ),
    }


def assert_active_supervision(
    item: dict[str, torch.Tensor],
    *,
    loop_loss_mode: str,
) -> dict[str, int]:
    counts = supervision_counts(item)
    if counts["outcome_tokens"] <= 0:
        raise RuntimeError(
            "Training row has zero active outcome tokens. Check prompt/completion tokenization boundary."
        )
    if loop_loss_mode in {
        "per_loop_labels",
        "weighted_per_loop_labels",
        "annealed_chain_to_outcome",
    }:
        if counts["loop_tokens"] <= 0 or counts["active_loops"] <= 0:
            raise RuntimeError(
                "Training row has zero active loop-label tokens. Check prompt/completion tokenization boundary."
            )
    return counts


def assert_nonzero_trainable_gradient(module: torch.nn.Module) -> None:
    nonzero = 0
    for parameter in module.parameters():
        if parameter.requires_grad and parameter.grad is not None:
            nonzero += int(parameter.grad.detach().count_nonzero().item())
    if nonzero <= 0:
        raise RuntimeError(
            "Backward produced zero trainable gradients despite active supervision; refusing no-op training."
        )


def scheduled_loop_count(
    step: int,
    max_steps: int,
    *,
    start: int,
    end: int,
    schedule: str = "linear",
) -> int:
    if start < 1 or end < start:
        raise ValueError("loop curriculum requires 1 <= start <= end")
    if max_steps <= 1:
        return end
    progress = min(1.0, max(0.0, step / float(max_steps - 1)))
    if schedule == "linear":
        value = start + (end - start) * progress
    elif schedule in {"one_minus_sqrt", "1-sqrt"}:
        value = end - (end - start) * math.sqrt(1.0 - progress)
    else:
        raise ValueError("schedule must be one of: linear, one_minus_sqrt")
    return max(start, min(end, int(round(value))))


def curriculum_target_counts(
    row_targets: torch.Tensor,
    scheduled: int,
    *,
    mode: str,
) -> torch.Tensor:
    if mode == "schedule":
        return torch.full_like(row_targets, int(scheduled))
    if mode == "row_capped":
        return row_targets.clamp(1, int(scheduled))
    if mode == "row_or_schedule_max":
        scheduled_tensor = torch.full_like(row_targets, int(scheduled))
        return torch.maximum(row_targets, scheduled_tensor)
    raise ValueError("target_source must be one of: schedule, row_capped, row_or_schedule_max")


def chain_label_weight(
    step: int,
    total_steps: int,
    *,
    hold_frac: float = 0.5,
) -> float:
    """Anneal intermediate-chain supervision from 1 to 0, then hold at 0."""

    if total_steps <= 0:
        raise ValueError("total_steps must be positive")
    if hold_frac < 0.0 or hold_frac > 1.0:
        raise ValueError("hold_frac must be in [0, 1]")
    ramp_steps = int(total_steps * (1.0 - hold_frac))
    if ramp_steps <= 0:
        return 0.0
    if step >= ramp_steps:
        return 0.0
    return max(0.0, 1.0 - float(step) / float(ramp_steps))


def freeze_all_base_then_unfreeze_recurrent_block(wrapper: RecurrentQwenForCausalLM) -> None:
    for param in wrapper.base_model.parameters():
        param.requires_grad_(False)
    for layer_idx in range(wrapper.layer_split.prelude_end, wrapper.layer_split.recurrent_end):
        for param in wrapper.qwen.layers[layer_idx].parameters():
            param.requires_grad_(True)


def pretrained_base_parameter_items(
    wrapper: RecurrentQwenForCausalLM,
) -> list[tuple[str, torch.nn.Parameter]]:
    """Return pretrained Qwen tensors, excluding newly attached LoRA tensors."""

    return [
        (name, parameter)
        for name, parameter in wrapper.base_model.named_parameters()
        if ".lora_a." not in name and ".lora_b." not in name
    ]


def hash_pretrained_base_parameters(wrapper: RecurrentQwenForCausalLM) -> str:
    digest = hashlib.sha256()
    for name, parameter in pretrained_base_parameter_items(wrapper):
        tensor = parameter.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(tuple(tensor.shape)).encode("ascii"))
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(tensor.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def assert_pretrained_base_frozen(wrapper: RecurrentQwenForCausalLM) -> None:
    trainable = [
        name
        for name, parameter in pretrained_base_parameter_items(wrapper)
        if parameter.requires_grad
    ]
    if trainable:
        raise RuntimeError(
            "Pretrained base tensors must remain frozen; trainable examples="
            f"{trainable[:8]}"
        )


def evaluate_loop1_canary(
    wrapper: RecurrentQwenForCausalLM,
    tokenizer: Any,
    *,
    data_jsonl: str | Path,
    device: str,
    value_prefix: str,
) -> dict[str, float | int]:
    from eval.eval_synthetic_depth_active_labels import (
        active_target_for_loop,
        candidates_for_row,
        prompt_for_row,
        read_jsonl,
        score_candidates_all_loops,
    )

    rows = read_jsonl(data_jsonl)
    score_args = SimpleNamespace(
        device=device,
        force_slow_candidate_score=False,
        normalize_candidate_score=True,
    )
    correct = 0
    wrapper.eval()
    try:
        for row in rows:
            prompt = prompt_for_row(
                row,
                prediction_space="full_symbols",
                prompt_style="question_only",
            )
            candidates = candidates_for_row(
                row,
                prediction_space="full_symbols",
                value_prefix=value_prefix,
            )
            scores = score_candidates_all_loops(
                wrapper,
                tokenizer,
                prompt,
                candidates,
                score_args,
                loop_counts=[1],
            )[1]
            prediction = max(scores.items(), key=lambda item: item[1])[0]
            target = active_target_for_loop(
                row,
                1,
                prediction_space="full_symbols",
                value_prefix=value_prefix,
            )
            correct += int(prediction == target)
    finally:
        wrapper.train()
    total = len(rows)
    return {
        "correct": correct,
        "total": total,
        "accuracy": correct / total if total else 0.0,
    }


def evaluate_diagonal_canary(
    wrapper: RecurrentQwenForCausalLM,
    tokenizer: Any,
    *,
    data_jsonl: str | Path,
    device: str,
    value_prefix: str,
    max_depth: int,
) -> dict[str, float | int]:
    from eval.eval_synthetic_depth_active_labels import (
        active_target_for_loop,
        candidates_for_row,
        prompt_for_row,
        read_jsonl,
        score_candidates_all_loops,
    )

    rows = read_jsonl(data_jsonl)
    score_args = SimpleNamespace(
        device=device,
        force_slow_candidate_score=False,
        normalize_candidate_score=True,
    )
    correct = 0
    wrapper.eval()
    try:
        for row in rows:
            depth = int(row["depth"])
            if not 1 <= depth <= int(max_depth):
                raise ValueError(
                    f"Diagonal canary row depth {depth} is outside [1, {max_depth}]"
                )
            prompt = prompt_for_row(
                row,
                prediction_space="full_symbols",
                prompt_style="question_only",
            )
            candidates = candidates_for_row(
                row,
                prediction_space="full_symbols",
                value_prefix=value_prefix,
            )
            scores = score_candidates_all_loops(
                wrapper,
                tokenizer,
                prompt,
                candidates,
                score_args,
                loop_counts=[depth],
            )[depth]
            prediction = max(scores.items(), key=lambda item: item[1])[0]
            target = active_target_for_loop(
                row,
                depth,
                prediction_space="full_symbols",
                value_prefix=value_prefix,
            )
            correct += int(prediction == target)
    finally:
        wrapper.train()
    total = len(rows)
    return {
        "correct": correct,
        "total": total,
        "accuracy": correct / total if total else 0.0,
    }


def evaluate_canary_spec(
    wrapper: RecurrentQwenForCausalLM,
    tokenizer: Any,
    *,
    spec: dict[str, Any],
    device: str,
) -> dict[str, float | int]:
    common = {
        "data_jsonl": spec["data_jsonl"],
        "device": device,
        "value_prefix": spec["value_prefix"],
    }
    if spec["mode"] == "diagonal":
        return evaluate_diagonal_canary(
            wrapper,
            tokenizer,
            **common,
            max_depth=int(spec["max_depth"]),
        )
    return evaluate_loop1_canary(wrapper, tokenizer, **common)


def configure_trainable_modules(wrapper: RecurrentQwenForCausalLM, cfg: dict[str, Any]) -> None:
    training_mode = str(cfg.get("training_mode", "full_block")).lower()
    if training_mode == "full_block":
        freeze_all_base_then_unfreeze_recurrent_block(wrapper)
    elif training_mode == "frozen_lora":
        for parameter in wrapper.parameters():
            parameter.requires_grad_(False)
        mark_only_lora_trainable(wrapper.base_model)
    elif training_mode == "controller_only":
        for parameter in wrapper.parameters():
            parameter.requires_grad_(False)
    else:
        raise ValueError(
            "training_mode must be one of: full_block, frozen_lora, controller_only"
        )
    aux = cfg.get("train_auxiliary", {})
    bridge_enabled = bool(aux.get("bridge", True)) and training_mode != "controller_only"
    bridge_is_split = bool(getattr(wrapper.bridge, "split_projection", False))
    for name, param in wrapper.bridge.named_parameters():
        # Split mode retains the old concat projection only so historical
        # checkpoints can load. It is not in the forward graph and must not be
        # counted or passed to the optimizer in new runs.
        bypassed_legacy_concat = bridge_is_split and name.startswith("proj.")
        param.requires_grad_(bridge_enabled and not bypassed_legacy_concat)
    for param in wrapper.halt_predictor.parameters():
        param.requires_grad_(
            bool(aux.get("halting", True))
            or training_mode == "controller_only"
        )
    for param in wrapper.reentry_adapter.parameters():
        param.requires_grad_(bool(aux.get("reentry_adapter", False)))
    for param in wrapper.latent_trajectory.parameters():
        param.requires_grad_(bool(aux.get("latent", False)))
    if training_mode in {"frozen_lora", "controller_only"}:
        assert_pretrained_base_frozen(wrapper)
    if training_mode == "controller_only":
        unexpected = [
            name
            for name, parameter in wrapper.named_parameters()
            if parameter.requires_grad and not name.startswith("halt_predictor.")
        ]
        if unexpected:
            raise RuntimeError(
                "controller_only selected non-halting parameters: "
                f"{unexpected[:8]}"
            )


def trainable_parameter_summary(wrapper: RecurrentQwenForCausalLM) -> dict[str, int]:
    recurrent = 0
    for layer_idx in range(wrapper.layer_split.prelude_end, wrapper.layer_split.recurrent_end):
        recurrent += sum(param.numel() for param in wrapper.qwen.layers[layer_idx].parameters() if param.requires_grad)
    bridge = sum(param.numel() for param in wrapper.bridge.parameters() if param.requires_grad)
    halt = sum(param.numel() for param in wrapper.halt_predictor.parameters() if param.requires_grad)
    reentry = sum(param.numel() for param in wrapper.reentry_adapter.parameters() if param.requires_grad)
    latent = sum(param.numel() for param in wrapper.latent_trajectory.parameters() if param.requires_grad)
    total = sum(param.numel() for param in wrapper.parameters() if param.requires_grad)
    return {
        "total": total,
        "recurrent_block": recurrent,
        "bridge": bridge,
        "halting": halt,
        "reentry_adapter": reentry,
        "latent": latent,
    }


def _parameter_norm_summary(named_params: list[tuple[str, torch.nn.Parameter]], prefix: str) -> dict[str, float]:
    total_numel = 0
    sum_sq = 0.0
    max_abs = 0.0
    for _, param in named_params:
        if not param.requires_grad:
            continue
        tensor = param.detach().float()
        if tensor.numel() == 0:
            continue
        total_numel += int(tensor.numel())
        sum_sq += float(tensor.square().sum().item())
        max_abs = max(max_abs, float(tensor.abs().max().item()))
    l2 = math.sqrt(sum_sq) if sum_sq > 0.0 else 0.0
    rms = math.sqrt(sum_sq / total_numel) if total_numel > 0 and sum_sq > 0.0 else 0.0
    return {
        f"{prefix}_param_numel": float(total_numel),
        f"{prefix}_param_l2": l2,
        f"{prefix}_param_rms": rms,
        f"{prefix}_param_max_abs": max_abs,
    }


def trainable_parameter_norm_stats(wrapper: RecurrentQwenForCausalLM) -> dict[str, float]:
    recurrent_params: list[tuple[str, torch.nn.Parameter]] = []
    for layer_idx in range(wrapper.layer_split.prelude_end, wrapper.layer_split.recurrent_end):
        recurrent_params.extend(
            (f"recurrent_layer_{layer_idx}.{name}", param)
            for name, param in wrapper.qwen.layers[layer_idx].named_parameters()
        )
    groups = {
        "trainable_total": [(name, param) for name, param in wrapper.named_parameters()],
        "recurrent_block": recurrent_params,
        "bridge": [(f"bridge.{name}", param) for name, param in wrapper.bridge.named_parameters()],
        "halting": [(f"halt_predictor.{name}", param) for name, param in wrapper.halt_predictor.named_parameters()],
        "reentry_adapter": [
            (f"reentry_adapter.{name}", param) for name, param in wrapper.reentry_adapter.named_parameters()
        ],
        "latent": [(f"latent_trajectory.{name}", param) for name, param in wrapper.latent_trajectory.named_parameters()],
    }
    stats: dict[str, float] = {}
    for prefix, params in groups.items():
        stats.update(_parameter_norm_summary(params, prefix))
    return stats


def bridge_uses_split_projection(wrapper: RecurrentQwenForCausalLM) -> bool:
    bridge = wrapper.bridge
    return bool(
        getattr(bridge, "split_projection", False)
        and hasattr(bridge, "prelude_proj")
        and hasattr(bridge, "state_proj")
    )


def bridge_prelude_weight_stats(wrapper: RecurrentQwenForCausalLM) -> dict[str, float]:
    if bridge_uses_split_projection(wrapper):
        prelude_weight = wrapper.bridge.prelude_proj.weight.detach().float()
        state_weight = wrapper.bridge.state_proj.weight.detach().float()
        hidden_size = int(getattr(wrapper.bridge, "hidden_size", state_weight.shape[0]))
    else:
        weight = wrapper.bridge.proj.weight.detach().float()
        hidden_size = int(getattr(wrapper.bridge, "hidden_size", weight.shape[0]))
        if weight.dim() != 2 or weight.shape[1] != 2 * hidden_size:
            return {
                "bridge_prelude_weight_rms": 0.0,
                "bridge_prelude_weight_max_abs": 0.0,
                "bridge_state_identity_max_abs_diff": 0.0,
            }
        prelude_weight = weight[:, :hidden_size]
        state_weight = weight[:, hidden_size:]
    if state_weight.shape != (hidden_size, hidden_size):
        return {
            "bridge_prelude_weight_rms": 0.0,
            "bridge_prelude_weight_max_abs": 0.0,
            "bridge_state_identity_max_abs_diff": 0.0,
        }
    eye = torch.eye(hidden_size, device=state_weight.device, dtype=state_weight.dtype)
    return {
        "bridge_prelude_weight_rms": float(prelude_weight.square().mean().sqrt().item()),
        "bridge_prelude_weight_max_abs": float(prelude_weight.abs().max().item()),
        "bridge_state_identity_max_abs_diff": float((state_weight - eye).abs().max().item()),
    }


def bridge_prelude_grad_stats(wrapper: RecurrentQwenForCausalLM) -> dict[str, float]:
    if bridge_uses_split_projection(wrapper):
        prelude_grad = wrapper.bridge.prelude_proj.weight.grad
        state_grad = wrapper.bridge.state_proj.weight.grad
        return {
            "bridge_prelude_grad_rms": 0.0 if prelude_grad is None else float(
                prelude_grad.detach().float().square().mean().sqrt().item()
            ),
            "bridge_state_grad_rms": 0.0 if state_grad is None else float(
                state_grad.detach().float().square().mean().sqrt().item()
            ),
        }
    grad = wrapper.bridge.proj.weight.grad
    hidden_size = int(getattr(wrapper.bridge, "hidden_size", wrapper.bridge.proj.weight.shape[0]))
    if grad is None or grad.dim() != 2 or grad.shape[1] != 2 * hidden_size:
        return {
            "bridge_prelude_grad_rms": 0.0,
            "bridge_state_grad_rms": 0.0,
        }
    prelude_grad = grad[:, :hidden_size].detach().float()
    state_grad = grad[:, hidden_size:].detach().float()
    return {
        "bridge_prelude_grad_rms": float(prelude_grad.square().mean().sqrt().item()),
        "bridge_state_grad_rms": float(state_grad.square().mean().sqrt().item()),
    }


def bridge_prelude_grad_vector(wrapper: RecurrentQwenForCausalLM) -> torch.Tensor | None:
    if bridge_uses_split_projection(wrapper):
        grad = wrapper.bridge.prelude_proj.weight.grad
        if grad is None:
            return None
        return grad.detach().float().flatten()
    grad = wrapper.bridge.proj.weight.grad
    hidden_size = int(getattr(wrapper.bridge, "hidden_size", wrapper.bridge.proj.weight.shape[0]))
    if grad is None or grad.dim() != 2 or grad.shape[1] != 2 * hidden_size:
        return None
    return grad[:, :hidden_size].detach().float().flatten()


def cosine_with_previous(
    current: torch.Tensor | None,
    previous: torch.Tensor | None,
) -> tuple[float, torch.Tensor | None]:
    if current is None or current.numel() == 0:
        return 0.0, previous
    if previous is None or previous.shape != current.shape:
        return 0.0, current.detach().clone()
    denom = current.norm() * previous.norm()
    if float(denom.item()) == 0.0:
        return 0.0, current.detach().clone()
    return float(torch.dot(current, previous).div(denom).item()), current.detach().clone()


def apply_bridge_prelude_grad_multiplier(
    wrapper: RecurrentQwenForCausalLM,
    multiplier: float,
) -> dict[str, float]:
    """Scale the bridge prelude-half gradient as a slice-specific LR proxy.

    PyTorch optimizers cannot assign a separate parameter group to a slice of a
    single ``Parameter``. The corrected bridge keeps a single projection weight
    for checkpoint compatibility, so this multiplier is applied after global
    clipping and before the optimizer step.
    """

    before = bridge_prelude_grad_stats(wrapper)
    if multiplier == 1.0:
        return {**before, "bridge_prelude_grad_multiplier": 1.0}
    if bridge_uses_split_projection(wrapper):
        raise ValueError(
            "bridge_prelude_grad_multiplier is the old slice-gradient proxy. "
            "Use bridge_prelude_lr_multiplier with bridge_projection_mode='split' instead."
        )
    grad = wrapper.bridge.proj.weight.grad
    hidden_size = int(getattr(wrapper.bridge, "hidden_size", wrapper.bridge.proj.weight.shape[0]))
    if grad is not None and grad.dim() == 2 and grad.shape[1] == 2 * hidden_size:
        grad[:, :hidden_size].mul_(float(multiplier))
    after = bridge_prelude_grad_stats(wrapper)
    return {
        **before,
        "bridge_prelude_grad_multiplier": float(multiplier),
        "bridge_prelude_grad_rms_after_multiplier": after["bridge_prelude_grad_rms"],
        "bridge_state_grad_rms_after_multiplier": after["bridge_state_grad_rms"],
    }


def bridge_prelude_optimizer_parameters(
    wrapper: RecurrentQwenForCausalLM,
    cfg: dict[str, Any],
) -> list[torch.nn.Parameter]:
    if not bridge_uses_split_projection(wrapper):
        return []
    params = [param for param in wrapper.bridge.prelude_proj.parameters() if param.requires_grad]
    if bool(cfg.get("bridge_prelude_lr_include_norm", False)):
        params.extend(param for param in wrapper.bridge.prelude_norm.parameters() if param.requires_grad)
    return params


def bridge_prelude_optimizer_setup(
    optimizer: OptimizerBundle | torch.optim.Optimizer,
    prelude_params: list[torch.nn.Parameter],
    *,
    expected_lr: float,
) -> dict[str, Any]:
    if isinstance(optimizer, OptimizerBundle):
        groups = [
            group
            for opt in optimizer.optimizers
            for group in opt.param_groups
        ]
    else:
        groups = list(optimizer.param_groups)
    prelude_ids = {id(param) for param in prelude_params}
    matching = [
        group
        for group in groups
        if any(id(param) in prelude_ids for param in group.get("params", []))
    ]
    ok = len(matching) == 1 and abs(float(matching[0]["lr"]) - float(expected_lr)) < 1e-15
    if not ok:
        raise AssertionError(
            "bridge prelude optimizer group missing or has wrong LR: "
            f"expected_lr={expected_lr}, group_lrs={[float(group['lr']) for group in groups]}"
        )
    return {
        "bridge_prelude_optimizer_group_ok": True,
        "bridge_prelude_optimizer_group_lr": float(matching[0]["lr"]),
        "bridge_prelude_optimizer_group_weight_decay": float(matching[0].get("weight_decay", 0.0)),
        "bridge_prelude_optimizer_group_num_tensors": len(matching[0].get("params", [])),
    }


def build_optimizer(wrapper: RecurrentQwenForCausalLM, cfg: dict[str, Any]) -> OptimizerBundle | torch.optim.Optimizer:
    optimizer_name = str(cfg.get("optimizer", "muon")).lower()
    params = [param for param in wrapper.parameters() if param.requires_grad]
    if not params:
        raise ValueError("No trainable parameters selected")
    lr = cfg_float(cfg, "learning_rate", 5e-6)
    weight_decay = cfg_float(cfg, "weight_decay", 0.0)
    prelude_lr_multiplier = cfg_float(cfg, "bridge_prelude_lr_multiplier", 1.0)
    prelude_params = bridge_prelude_optimizer_parameters(wrapper, cfg)
    prelude_param_ids = {id(param) for param in prelude_params}
    if prelude_lr_multiplier != 1.0 and not prelude_params:
        raise ValueError(
            "bridge_prelude_lr_multiplier requires bridge_projection_mode='split' "
            "and train_auxiliary.bridge=true"
        )
    if optimizer_name == "adamw":
        if prelude_params:
            rest = [param for param in params if id(param) not in prelude_param_ids]
            return torch.optim.AdamW(
                [
                    {"params": rest, "lr": lr, "weight_decay": weight_decay},
                    {
                        "params": prelude_params,
                        "lr": prelude_lr_multiplier * lr,
                        "weight_decay": cfg_float(cfg, "bridge_prelude_weight_decay", 0.0),
                    },
                ]
            )
        return torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)
    if optimizer_name != "muon":
        raise ValueError("optimizer must be one of: muon, adamw")
    if prelude_lr_multiplier != 1.0:
        raise ValueError(
            "bridge_prelude_lr_multiplier is currently supported for optimizer='adamw' only. "
            "Use AdamW for the split-bridge micro-test."
        )

    muon_params, adamw_params = split_muon_and_adamw_params(wrapper.named_parameters())
    optimizers: list[torch.optim.Optimizer] = []
    if muon_params:
        optimizers.append(
            Muon(
                muon_params,
                lr=lr,
                momentum=cfg_float(cfg, "muon_momentum", 0.95),
                weight_decay=weight_decay,
                ns_steps=cfg_int(cfg, "muon_ns_steps", 5),
            )
        )
    if adamw_params:
        optimizers.append(
            torch.optim.AdamW(
                adamw_params,
                lr=cfg_float(cfg, "adamw_lr", lr),
                weight_decay=weight_decay,
            )
        )
    return OptimizerBundle(optimizers)


def maybe_enable_gradient_checkpointing(model: torch.nn.Module, cfg: dict[str, Any]) -> None:
    if not bool(cfg.get("gradient_checkpointing", True)):
        return
    enable = getattr(model, "gradient_checkpointing_enable", None)
    if callable(enable):
        enable()


def resolve_resume_lora_config(cfg: dict[str, Any]) -> dict[str, Any]:
    resume_lora = dict(cfg.get("resume_lora", {}))
    if not resume_lora.get("enabled", True):
        return {"enabled": False}
    rank = resume_lora.get("rank", "auto")
    alpha = resume_lora.get("alpha", "auto")
    needs_checkpoint = str(rank).lower() == "auto" or str(alpha).lower() == "auto"
    checkpoint_lora: dict[str, Any] = {}
    if needs_checkpoint:
        resume_from = cfg.get("resume_from")
        if not resume_from:
            raise ValueError("resume_lora rank/alpha auto requires resume_from")
        checkpoint = torch.load(resume_from, map_location="cpu")
        checkpoint_cfg = checkpoint.get("config", {})
        checkpoint_lora = dict(checkpoint_cfg.get("lora", {}))
    if str(rank).lower() == "auto":
        if "rank" not in checkpoint_lora:
            raise ValueError("Could not infer LoRA rank from checkpoint config")
        rank = checkpoint_lora["rank"]
    if str(alpha).lower() == "auto":
        alpha = checkpoint_lora.get("alpha", 2 * int(rank))
    return {
        "enabled": True,
        "rank": int(rank),
        "alpha": float(alpha),
        "dropout": float(resume_lora.get("dropout", 0.0)),
    }


def lora_key_counts(load_info: dict[str, Any]) -> dict[str, int]:
    loaded = load_info.get("loaded_keys", [])
    skipped = load_info.get("skipped", {})
    return {
        "loaded_lora_keys": sum(".lora_a." in key or ".lora_b." in key for key in loaded),
        "skipped_lora_keys": sum(".lora_a." in key or ".lora_b." in key for key in skipped),
    }


def prepare_wrapper(
    model: torch.nn.Module,
    cfg: dict[str, Any],
    *,
    device: str,
) -> tuple[RecurrentQwenForCausalLM, dict[str, Any]]:
    wrapper = RecurrentQwenForCausalLM(
        model,
        layer_split=parse_split(cfg.get("layer_split", "auto")),
        initial_halt_prob=cfg_float(cfg, "initial_halt_prob", 0.15),
    ).to(device)
    bridge_projection_mode = str(cfg.get("bridge_projection_mode", "concat")).lower()
    if bridge_projection_mode not in {"concat", "split"}:
        raise ValueError("bridge_projection_mode must be one of: concat, split")
    if bridge_projection_mode == "split":
        wrapper.bridge.convert_to_split_projection()
        print("bridge_projection_mode=split true_prelude_lr_groups_available=1")
    else:
        print("bridge_projection_mode=concat true_prelude_lr_groups_available=0")
    adapter_dtype = resolve_dtype(cfg.get("adapter_dtype", "float32"))
    resume_lora = resolve_resume_lora_config(cfg)
    lora_wrapped = 0
    if resume_lora.get("enabled", True):
        lora_wrapped = apply_lora_to_recurrent_block(
            wrapper,
            rank=int(resume_lora.get("rank", 32)),
            alpha=float(resume_lora.get("alpha", 64)),
            dropout=float(resume_lora.get("dropout", 0.0)),
            adapter_dtype=adapter_dtype,
        )
        print(f"resume_lora_recurrent_modules={lora_wrapped}")
    wrapper.set_trainable_modules_dtype(adapter_dtype)
    load_counts = {"loaded_lora_keys": 0, "skipped_lora_keys": 0}
    if cfg.get("resume_from"):
        info = load_trainable_checkpoint(wrapper, cfg["resume_from"])
        print(f"loaded_checkpoint={cfg['resume_from']} loaded_keys={len(info['loaded_keys'])}")
        load_counts = lora_key_counts(info)
        print(
            "checkpoint_lora_key_counts="
            + " ".join(f"{key}={value}" for key, value in load_counts.items())
        )
        if (
            bool(cfg.get("require_all_lora_loaded", False))
            and lora_wrapped
            and load_counts["loaded_lora_keys"] < 2 * lora_wrapped
        ):
            raise RuntimeError(
                "Checkpoint did not restore every LoRA A/B tensor: "
                f"wrapped_modules={lora_wrapped}, {load_counts}"
            )
        loaded_keys = list(info["loaded_keys"])
        for prefix in cfg.get("require_loaded_prefixes") or []:
            if not any(name.startswith(str(prefix)) for name in loaded_keys):
                raise RuntimeError(
                    f"Checkpoint restored no tensors under required prefix {prefix!r}"
                )
        if (
            bool(cfg.get("require_lora_loaded_before_merge", True))
            and bool(cfg.get("merge_lora_before_unfreeze", True))
            and lora_wrapped
            and load_counts["loaded_lora_keys"] < 2 * lora_wrapped
        ):
            raise RuntimeError(
                "Refusing to merge LoRA before unfreeze because not all LoRA keys loaded: "
                f"wrapped_modules={lora_wrapped}, {load_counts}. "
                "Set resume_lora.rank/alpha correctly or use auto inference."
            )
    merged_lora = 0
    if bool(cfg.get("merge_lora_before_unfreeze", True)):
        merged_lora = merge_lora_adapters(wrapper.base_model)
        print(f"merged_lora_modules={merged_lora}")
    configure_trainable_modules(wrapper, cfg)
    wrapper.set_trainable_modules_dtype(adapter_dtype)
    return wrapper, {
        "resume_lora_wrapped": lora_wrapped,
        "merged_lora_modules": merged_lora,
        "resume_lora": resume_lora,
        "checkpoint_lora_key_counts": load_counts,
        "trainable_parameters": trainable_parameter_summary(wrapper),
        "bridge_projection_mode": bridge_projection_mode,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--train_jsonl", required=True)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    cfg = load_config(args.config)
    if bool(cfg.get("reject_muon", False)) and str(cfg.get("optimizer", "")).lower() != "adamw":
        raise ValueError("This experiment rejects Muon; optimizer must be AdamW")
    training_seed = cfg_int(cfg, "seed", 0)
    loader_generator = seed_training_rng(training_seed)
    print(f"training_seed={training_seed}")

    tokenizer = AutoTokenizer.from_pretrained(cfg["model_name"])
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        cfg["model_name"],
        **model_load_kwargs(cfg.get("dtype", "auto"), cfg.get("attn_implementation", "default")),
    ).to(args.device)
    maybe_enable_gradient_checkpointing(model, cfg)
    wrapper, setup = prepare_wrapper(model, cfg, device=args.device)
    base_hash_start = None
    if bool(cfg.get("require_frozen_base_hash", False)):
        assert_pretrained_base_frozen(wrapper)
        base_hash_start = hash_pretrained_base_parameters(wrapper)
        print(f"[assert-ok] pretrained_base_sha256_start={base_hash_start}", flush=True)
    assert_finite_trainable_parameters(wrapper, step=0)
    wrapper.train()

    max_loops = cfg_int(cfg, "max_loops", 8)
    dataset = JsonlCausalDataset(
        args.train_jsonl,
        tokenizer=tokenizer,
        max_length=cfg_int(cfg, "max_length", 512),
        max_train_loops=max_loops,
        train_on_prompt=bool(cfg.get("train_on_prompt", False)),
    )
    loop_loss_mode = str(cfg.get("loop_loss_mode", "halting_weighted"))
    if bool(cfg.get("require_active_supervision", False)):
        preflight_counts = assert_active_supervision(dataset[0], loop_loss_mode=loop_loss_mode)
        print(f"[assert-ok] active_supervision={preflight_counts}")
    batch_size = cfg_int(cfg, "batch_size", 1)
    gradient_accumulation_steps = cfg_int(cfg, "gradient_accumulation_steps", 1)
    if gradient_accumulation_steps < 1:
        raise ValueError("gradient_accumulation_steps must be positive")
    effective_batch_size = batch_size * gradient_accumulation_steps
    minimum_effective_batch_size = cfg_int(cfg, "minimum_effective_batch_size", 1)
    if effective_batch_size < minimum_effective_batch_size:
        raise RuntimeError(
            "Effective batch is below the configured floor: "
            f"batch_size={batch_size}, gradient_accumulation_steps={gradient_accumulation_steps}, "
            f"effective_batch_size={effective_batch_size}, floor={minimum_effective_batch_size}"
        )
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        generator=loader_generator,
        collate_fn=partial(collate_causal_batch, pad_token_id=tokenizer.pad_token_id),
    )
    optimizer = build_optimizer(wrapper, cfg)
    optimizer_setup: dict[str, Any] = {}
    prelude_lr_multiplier = cfg_float(cfg, "bridge_prelude_lr_multiplier", 1.0)
    prelude_params = bridge_prelude_optimizer_parameters(wrapper, cfg)
    if prelude_params:
        optimizer_setup = bridge_prelude_optimizer_setup(
            optimizer,
            prelude_params,
            expected_lr=prelude_lr_multiplier * cfg_float(cfg, "learning_rate", 5e-6),
        )
        print(
            "[assert-ok] bridge_prelude_optimizer_group "
            f"lr={optimizer_setup['bridge_prelude_optimizer_group_lr']:.6e} "
            f"wd={optimizer_setup['bridge_prelude_optimizer_group_weight_decay']:.6e} "
            f"tensors={optimizer_setup['bridge_prelude_optimizer_group_num_tensors']}"
        )
    optimizer.zero_grad(set_to_none=True)

    curriculum = cfg.get("recurrence_curriculum", {})
    curriculum_enabled = bool(curriculum.get("enabled", True))
    start_loop = int(curriculum.get("start_loop", 1))
    end_loop = int(curriculum.get("end_loop", max_loops))
    schedule = str(curriculum.get("schedule", "linear"))
    target_source = str(curriculum.get("target_source", "schedule"))
    ramp_compute = bool(curriculum.get("ramp_compute", True))

    loop_label_weights: torch.Tensor | None = None
    dose_ledger: LoopDoseLedger | None = None
    row_specific_loop_weights = bool(dataset.rows) and all(
        "loop_label_weights" in row for row in dataset.rows
    )
    if any("loop_label_weights" in row for row in dataset.rows) and not row_specific_loop_weights:
        raise ValueError("Every row must provide loop_label_weights when row-specific weighting is enabled")
    row_specific_forward_loops = bool(cfg.get("row_specific_forward_loops", False))
    if row_specific_forward_loops and not all("forward_loop_count" in row for row in dataset.rows):
        raise ValueError("row_specific_forward_loops requires forward_loop_count on every row")
    dose_assert_every = cfg_int(cfg, "dose_assert_every", 200)
    dose_ratio_min = cfg_float(cfg, "dose_ratio_min", 0.8)
    dose_ratio_max = cfg_float(cfg, "dose_ratio_max", 1.25)
    newest_loop_multiplier = cfg_float(cfg, "newest_loop_multiplier", 2.0)
    dose_requires_equalization = False
    if loop_loss_mode == "weighted_per_loop_labels":
        if row_specific_loop_weights:
            for row in dataset.rows:
                configured = [float(value) for value in row["loop_label_weights"]]
                if len(configured) != max_loops:
                    raise ValueError(
                        "row-specific loop_label_weights requires one entry per max_loops; "
                        f"row={row.get('id')}, got {len(configured)} vs {max_loops}"
                    )
                if any(not math.isfinite(value) or value < 0.0 for value in configured):
                    raise ValueError(f"row {row.get('id')} has invalid loop_label_weights")
            print(
                "[assert-ok] row_specific_loop_label_weights "
                f"rows={len(dataset.rows)} max_loops={max_loops}",
                flush=True,
            )
        else:
            configured_weights = [float(value) for value in (cfg.get("loop_label_weights") or [])]
            if len(configured_weights) != max_loops:
                raise ValueError(
                    "weighted_per_loop_labels requires one loop_label_weights entry per max_loops; "
                    f"got {len(configured_weights)} for max_loops={max_loops}"
                )
            loop_label_weights = torch.tensor(configured_weights, device=args.device, dtype=torch.float32)
            dose_ledger = LoopDoseLedger(
                weights=configured_weights,
                newest_loop=max_loops,
                newest_multiplier=newest_loop_multiplier,
            )
            dose_requires_equalization = True
            theoretical_exposure = exposure_fractions(dataset.rows, cap=max_loops)
            theoretical_mass = [
                theoretical_exposure[index] * configured_weights[index]
                for index in range(max_loops)
            ]
            theoretical_receipt = assert_mass_equalized(
                theoretical_mass,
                newest_loop=max_loops,
                newest_multiplier=newest_loop_multiplier,
                min_ratio=dose_ratio_min,
                max_ratio=dose_ratio_max,
            )
            print(
                "[assert-ok] weighted_loop_mass_startup "
                f"exposure={theoretical_exposure} weights={configured_weights} "
                f"receipt={theoretical_receipt}",
                flush=True,
            )
    elif loop_loss_mode == "per_loop_labels" and bool(cfg.get("track_loop_dose", False)):
        # Arm-comparison curricula need an auditable count of supervised labels
        # at every loop, but equal mass is not expected when row depths vary.
        dose_ledger = LoopDoseLedger(
            weights=[1.0] * max_loops,
            newest_loop=max_loops,
            newest_multiplier=1.0,
        )
        print(
            "[assert-ok] per_loop_label_dose_tracking "
            f"max_loops={max_loops} equalization_required=0",
            flush=True,
        )

    summary = {
        "setup": setup,
        "optimizer_setup": optimizer_setup,
        "training_seed": training_seed,
        "batch_size": batch_size,
        "gradient_accumulation_steps": gradient_accumulation_steps,
        "effective_batch_size": effective_batch_size,
        "curriculum_trace": [],
        "interval_checkpoints": [],
        "dose_trace": [],
        "pretrained_base_sha256_start": base_hash_start,
        "canary_trace": [],
        "canary_traces": {},
    }
    max_steps = cfg_int(cfg, "max_steps", 25)
    save_every = cfg_int(cfg, "save_every", 0) if cfg.get("save_every", 0) else 0
    checkpoint_backup_every = (
        cfg_int(cfg, "checkpoint_backup_every", 0) if cfg.get("checkpoint_backup_every", 0) else 0
    )
    checkpoint_backup_dir = cfg.get("checkpoint_backup_dir")
    progress_backup_path = cfg.get("progress_backup_path")
    output_dir_config = cfg.get("output_dir")
    progress_path = Path(output_dir_config) / "train_unfrozen_recurrent_progress.json" if output_dir_config else None

    def persist_progress(
        *,
        status: str,
        completed_step: int,
        metrics: dict[str, float] | None = None,
        checkpoint: Path | None = None,
        checkpoint_backup: Path | None = None,
    ) -> None:
        if progress_path is None:
            return
        payload: dict[str, Any] = {
            "kind": "train_unfrozen_recurrent_progress",
            "status": status,
            "completed_step": int(completed_step),
            "max_steps": int(max_steps),
            "micro_step": int(micro_step),
            "checkpoint": str(checkpoint) if checkpoint else None,
            "checkpoint_backup": str(checkpoint_backup) if checkpoint_backup else None,
            "metrics": metrics or {},
        }
        write_training_progress(progress_path, payload)
        if progress_backup_path:
            backup_destination = Path(progress_backup_path)
            backup_destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(progress_path, backup_destination)
    save_steps = {
        int(item)
        for item in (cfg.get("save_steps") or [])
        if int(item) > 0
    }
    prelude_grad_multiplier = cfg_float(cfg, "bridge_prelude_grad_multiplier", 1.0)
    canary_every = cfg_int(cfg, "canary_every", 0) if cfg.get("canary_every", 0) else 0
    canary_specs = resolve_canary_specs(cfg)
    if canary_every and not canary_specs:
        raise ValueError("canary_every requires canary_jsonl or canary_specs")
    if canary_specs and not canary_every:
        raise ValueError("configured canaries require a positive canary_every")
    summary["canary_traces"] = {spec["name"]: [] for spec in canary_specs}
    chain_anneal_hold_frac = cfg_float(cfg, "chain_anneal_hold_frac", 0.5)
    chain_outcome_loss_weight = cfg_float(cfg, "chain_outcome_loss_weight", 1.0)
    if prelude_grad_multiplier != 1.0 and bridge_uses_split_projection(wrapper):
        raise ValueError(
            "Do not combine split bridge true prelude LR with bridge_prelude_grad_multiplier. "
            "Set bridge_prelude_grad_multiplier=1.0."
        )
    step = 0
    micro_step = 0
    accumulated_microbatches = 0
    previous_prelude_grad: torch.Tensor | None = None
    hard_stop = False
    persist_progress(status="started", completed_step=step)
    while step < max_steps:
        for batch in loader:
            batch = {key: value.to(args.device) for key, value in batch.items()}
            batch_loop_label_weights = batch.pop("loop_label_weights", None)
            batch_forward_loop_counts = batch.pop("forward_loop_counts", None)
            scheduled = (
                scheduled_loop_count(step, max_steps, start=start_loop, end=end_loop, schedule=schedule)
                if curriculum_enabled
                else max_loops
            )
            forward_loops = scheduled if ramp_compute else max_loops
            if row_specific_forward_loops:
                if batch_forward_loop_counts is None:
                    raise RuntimeError("forward_loop_counts missing from a row-specific compute batch")
                forward_loops = int(batch_forward_loop_counts.max().item())
                if not 1 <= forward_loops <= max_loops:
                    raise RuntimeError(f"Invalid row-specific forward loop count: {forward_loops}")
            batch["target_loop_counts"] = curriculum_target_counts(
                batch["target_loop_counts"],
                scheduled,
                mode=target_source,
            ).clamp(1, forward_loops)
            output = wrapper(
                **batch,
                max_loops=forward_loops,
                beta=cfg_float(cfg, "beta", 0.08),
                halt_target_nll_weight=cfg_float(cfg, "halt_target_nll_weight", 0.0),
                loop_loss_mode=loop_loss_mode,
                loop_label_weights=(
                    batch_loop_label_weights
                    if batch_loop_label_weights is not None
                    else loop_label_weights
                ),
                loop_label_loss_weight=(
                    chain_label_weight(step, max_steps, hold_frac=chain_anneal_hold_frac)
                    if loop_loss_mode == "annealed_chain_to_outcome"
                    else 1.0
                ),
                outcome_label_loss_weight=(
                    chain_outcome_loss_weight
                    if loop_loss_mode == "annealed_chain_to_outcome"
                    else 1.0
                ),
                reentry_rescale_mode="none",
                use_reentry_adapter=False,
                reentry_tail_damper_path=None,
                reentry_tail_damper_strength=0.0,
                use_cache=False,
                return_dict=True,
            )
            assert output.loss is not None
            assert_finite_training_state(wrapper, output.loss, output.metrics, step)
            (output.loss / gradient_accumulation_steps).backward()
            assert_finite_trainable_gradients(wrapper, step)
            if dose_ledger is not None:
                dose_ledger.update(batch["loop_labels"])
            micro_step += 1
            accumulated_microbatches += 1
            if accumulated_microbatches < gradient_accumulation_steps:
                continue
            if bool(cfg.get("require_nonzero_train_gradient", False)):
                assert_nonzero_trainable_gradient(wrapper)
            prelude_grad_cosine_prev, previous_prelude_grad = cosine_with_previous(
                bridge_prelude_grad_vector(wrapper),
                previous_prelude_grad,
            )
            pre_clip_bridge_grad = {
                f"pre_clip_{key}": value
                for key, value in bridge_prelude_grad_stats(wrapper).items()
            }
            pre_clip_bridge_grad["bridge_prelude_grad_cosine_prev"] = prelude_grad_cosine_prev
            torch.nn.utils.clip_grad_norm_(
                [param for param in wrapper.parameters() if param.requires_grad],
                cfg_float(cfg, "max_grad_norm", 0.5),
                error_if_nonfinite=True,
            )
            bridge_grad = apply_bridge_prelude_grad_multiplier(wrapper, prelude_grad_multiplier)
            assert_finite_trainable_gradients(wrapper, step)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            assert_finite_trainable_parameters(wrapper, step + 1)

            accumulated_microbatches = 0
            completed_step = step + 1
            if dose_ledger is not None and dose_assert_every and completed_step % dose_assert_every == 0:
                if dose_requires_equalization:
                    dose_ledger.assert_equalized(min_ratio=dose_ratio_min, max_ratio=dose_ratio_max)
                receipt = {"step": completed_step, **dose_ledger.as_dict()}
                summary["dose_trace"].append(receipt)
                label = "weighted_loop_mass" if dose_requires_equalization else "per_loop_label_dose"
                print(f"[assert-ok] {label} step={completed_step} receipt={receipt}", flush=True)

            if completed_step == 1 or completed_step % cfg_int(cfg, "log_every", 5) == 0:
                bridge_weight = bridge_prelude_weight_stats(wrapper)
                parameter_norms = trainable_parameter_norm_stats(wrapper)
                metric_values = {
                    **{key: float(value) for key, value in output.metrics.items()},
                    **pre_clip_bridge_grad,
                    **bridge_grad,
                    **bridge_weight,
                    **parameter_norms,
                }
                metrics = " ".join(f"{key}={float(value):.4f}" for key, value in metric_values.items())
                print(
                    f"step={completed_step} micro_step={micro_step} scheduled_loops={scheduled} "
                    f"forward_loops={forward_loops} {metrics}"
                )
                summary["curriculum_trace"].append(
                    {
                        "step": completed_step,
                        "micro_step": micro_step,
                        "scheduled_loops": scheduled,
                        "forward_loops": forward_loops,
                        "metrics": metric_values,
                    }
                )
                persist_progress(status="training", completed_step=completed_step, metrics=metric_values)
            step = completed_step
            if canary_every and step % canary_every == 0:
                canary_hard_stop = False
                for spec in canary_specs:
                    canary = evaluate_canary_spec(
                        wrapper,
                        tokenizer,
                        spec=spec,
                        device=args.device,
                    )
                    baseline_accuracy = float(spec["baseline_accuracy"])
                    hard_stop_delta = float(spec["hard_stop_delta"])
                    accuracy_delta = float(canary["accuracy"]) - baseline_accuracy
                    canary_receipt = {
                        "name": spec["name"],
                        "mode": spec["mode"],
                        "step": step,
                        **canary,
                        "baseline_accuracy": baseline_accuracy,
                        "accuracy_delta": accuracy_delta,
                        "hard_stop_delta": hard_stop_delta,
                        "status": (
                            "red_hard_stop"
                            if accuracy_delta < hard_stop_delta
                            else "green_continue"
                        ),
                    }
                    summary["canary_trace"].append(canary_receipt)
                    summary["canary_traces"][spec["name"]].append(canary_receipt)
                    print(f"[canary] {canary_receipt}", flush=True)
                    canary_hard_stop = canary_hard_stop or accuracy_delta < hard_stop_delta
                hard_stop = hard_stop or canary_hard_stop
            if output_dir := cfg.get("output_dir"):
                should_save_interval = (save_every and step % save_every == 0) or step in save_steps
                if should_save_interval and (step < max_steps or hard_stop):
                    checkpoint_path = save_trainable_checkpoint(wrapper, output_dir, "unfrozen_recurrent", step, cfg)
                    print(f"saved_checkpoint={checkpoint_path}")
                    summary["interval_checkpoints"].append(str(checkpoint_path))
                    checkpoint_backup = None
                    if checkpoint_backup_every and step % checkpoint_backup_every == 0:
                        checkpoint_backup = backup_checkpoint(checkpoint_path, checkpoint_backup_dir)
                        if checkpoint_backup is not None:
                            print(f"backed_up_checkpoint={checkpoint_backup}")
                    persist_progress(
                        status="checkpoint_saved",
                        completed_step=step,
                        checkpoint=checkpoint_path,
                        checkpoint_backup=checkpoint_backup,
                    )
                    if dose_ledger is not None:
                        summary["dose_trace"].append(
                            {"step": step, "checkpoint": str(checkpoint_path), **dose_ledger.as_dict()}
                        )
            if hard_stop:
                break
            if step >= max_steps:
                break
        if hard_stop:
            break

    output_dir = cfg.get("output_dir")
    if output_dir:
        checkpoint_path = save_trainable_checkpoint(wrapper, output_dir, "unfrozen_recurrent", step, cfg)
        print(f"saved_checkpoint={checkpoint_path}")
        checkpoint_backup = backup_checkpoint(checkpoint_path, checkpoint_backup_dir)
        if checkpoint_backup is not None:
            print(f"backed_up_checkpoint={checkpoint_backup}")
        summary["checkpoint"] = str(checkpoint_path)
        if dose_ledger is not None:
            summary["dose_trace"].append(
                {"step": step, "checkpoint": str(checkpoint_path), **dose_ledger.as_dict()}
            )
    summary["final_step"] = step
    summary["trainable_parameters"] = trainable_parameter_summary(wrapper)
    summary["bridge_prelude_weight_stats"] = bridge_prelude_weight_stats(wrapper)
    summary["parameter_norm_stats"] = trainable_parameter_norm_stats(wrapper)
    summary["bridge_prelude_grad_multiplier"] = prelude_grad_multiplier
    summary["micro_steps"] = micro_step
    summary["status"] = "hard_stopped_canary" if hard_stop else "finished"
    if base_hash_start is not None:
        assert_pretrained_base_frozen(wrapper)
        base_hash_end = hash_pretrained_base_parameters(wrapper)
        if base_hash_end != base_hash_start:
            raise RuntimeError(
                "Pretrained base hash changed during frozen-base training: "
                f"start={base_hash_start} end={base_hash_end}"
            )
        summary["pretrained_base_sha256_end"] = base_hash_end
        summary["pretrained_base_hash_unchanged"] = True
        print(f"[assert-ok] pretrained_base_sha256_end={base_hash_end}", flush=True)
    if dose_ledger is not None:
        summary["dose_ledger"] = dose_ledger.as_dict()
    if output_dir:
        summary_path = Path(output_dir) / "train_unfrozen_recurrent_summary.json"
        summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(f"training_summary={summary_path}")
    persist_progress(
        status="hard_stopped_canary" if hard_stop else "finished",
        completed_step=step,
        checkpoint=Path(summary["checkpoint"]) if summary.get("checkpoint") else None,
        checkpoint_backup=checkpoint_backup if output_dir else None,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
