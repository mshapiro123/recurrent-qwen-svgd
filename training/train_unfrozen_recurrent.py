"""Train the recurrent block itself with a loop-count curriculum.

This is the bounded unfreeze+Muon branch after the rank-only LoRA capacity
arm. It keeps the existing split and bridge path fixed, merges any recovered
LoRA initialization into the recurrent block, then trains the recurrent block's
own parameters plus lightweight loop controls.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from functools import partial
from pathlib import Path
from typing import Any

import torch
import yaml
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from eval.eval_identity import model_load_kwargs, parse_split, resolve_dtype
from models.lora import apply_lora_to_recurrent_block, merge_lora_adapters
from models.recurrent_wrapper import RecurrentQwenForCausalLM
from training.checkpointing import load_trainable_checkpoint, save_trainable_checkpoint
from training.dataset import JsonlCausalDataset, collate_causal_batch
from training.muon import Muon, OptimizerBundle, split_muon_and_adamw_params
from training.stability import (
    assert_finite_trainable_gradients,
    assert_finite_trainable_parameters,
    assert_finite_training_state,
)


def load_config(path: str | Path) -> dict[str, Any]:
    return yaml.safe_load(Path(path).read_text(encoding="utf-8"))


def cfg_float(cfg: dict[str, Any], key: str, default: float) -> float:
    value = cfg.get(key, default)
    return default if value is None else float(value)


def cfg_int(cfg: dict[str, Any], key: str, default: int) -> int:
    value = cfg.get(key, default)
    return default if value is None else int(value)


def seed_training_rng(seed: int) -> torch.Generator:
    """Seed model stochasticity and return a seeded CPU DataLoader generator."""

    seed = int(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    return torch.Generator(device="cpu").manual_seed(seed)


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


def configure_trainable_modules(wrapper: RecurrentQwenForCausalLM, cfg: dict[str, Any]) -> None:
    freeze_all_base_then_unfreeze_recurrent_block(wrapper)
    aux = cfg.get("train_auxiliary", {})
    for param in wrapper.bridge.parameters():
        param.requires_grad_(bool(aux.get("bridge", True)))
    for param in wrapper.halt_predictor.parameters():
        param.requires_grad_(bool(aux.get("halting", True)))
    for param in wrapper.reentry_adapter.parameters():
        param.requires_grad_(bool(aux.get("reentry_adapter", False)))
    for param in wrapper.latent_trajectory.parameters():
        param.requires_grad_(bool(aux.get("latent", False)))


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
    loader = DataLoader(
        dataset,
        batch_size=cfg_int(cfg, "batch_size", 1),
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

    summary = {
        "setup": setup,
        "optimizer_setup": optimizer_setup,
        "training_seed": training_seed,
        "curriculum_trace": [],
        "interval_checkpoints": [],
    }
    max_steps = cfg_int(cfg, "max_steps", 25)
    save_every = cfg_int(cfg, "save_every", 0) if cfg.get("save_every", 0) else 0
    save_steps = {
        int(item)
        for item in (cfg.get("save_steps") or [])
        if int(item) > 0
    }
    prelude_grad_multiplier = cfg_float(cfg, "bridge_prelude_grad_multiplier", 1.0)
    loop_loss_mode = str(cfg.get("loop_loss_mode", "halting_weighted"))
    chain_anneal_hold_frac = cfg_float(cfg, "chain_anneal_hold_frac", 0.5)
    chain_outcome_loss_weight = cfg_float(cfg, "chain_outcome_loss_weight", 1.0)
    if prelude_grad_multiplier != 1.0 and bridge_uses_split_projection(wrapper):
        raise ValueError(
            "Do not combine split bridge true prelude LR with bridge_prelude_grad_multiplier. "
            "Set bridge_prelude_grad_multiplier=1.0."
        )
    step = 0
    previous_prelude_grad: torch.Tensor | None = None
    while step < max_steps:
        for batch in loader:
            batch = {key: value.to(args.device) for key, value in batch.items()}
            scheduled = (
                scheduled_loop_count(step, max_steps, start=start_loop, end=end_loop, schedule=schedule)
                if curriculum_enabled
                else max_loops
            )
            forward_loops = scheduled if ramp_compute else max_loops
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
            output.loss.backward()
            assert_finite_trainable_gradients(wrapper, step)
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

            if step % cfg_int(cfg, "log_every", 5) == 0:
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
                print(f"step={step} scheduled_loops={scheduled} forward_loops={forward_loops} {metrics}")
                summary["curriculum_trace"].append(
                    {
                        "step": step,
                        "scheduled_loops": scheduled,
                        "forward_loops": forward_loops,
                        "metrics": metric_values,
                    }
                )
            step += 1
            if output_dir := cfg.get("output_dir"):
                should_save_interval = (save_every and step % save_every == 0) or step in save_steps
                if should_save_interval and step < max_steps:
                    checkpoint_path = save_trainable_checkpoint(wrapper, output_dir, "unfrozen_recurrent", step, cfg)
                    print(f"saved_checkpoint={checkpoint_path}")
                    summary["interval_checkpoints"].append(str(checkpoint_path))
            if step >= max_steps:
                break

    output_dir = cfg.get("output_dir")
    if output_dir:
        checkpoint_path = save_trainable_checkpoint(wrapper, output_dir, "unfrozen_recurrent", step, cfg)
        print(f"saved_checkpoint={checkpoint_path}")
        summary["checkpoint"] = str(checkpoint_path)
    summary["final_step"] = step
    summary["trainable_parameters"] = trainable_parameter_summary(wrapper)
    summary["bridge_prelude_weight_stats"] = bridge_prelude_weight_stats(wrapper)
    summary["parameter_norm_stats"] = trainable_parameter_norm_stats(wrapper)
    summary["bridge_prelude_grad_multiplier"] = prelude_grad_multiplier
    if output_dir:
        summary_path = Path(output_dir) / "train_unfrozen_recurrent_summary.json"
        summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(f"training_summary={summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
