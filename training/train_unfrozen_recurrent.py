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


def bridge_prelude_weight_stats(wrapper: RecurrentQwenForCausalLM) -> dict[str, float]:
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
    eye = torch.eye(hidden_size, device=weight.device, dtype=weight.dtype)
    return {
        "bridge_prelude_weight_rms": float(prelude_weight.square().mean().sqrt().item()),
        "bridge_prelude_weight_max_abs": float(prelude_weight.abs().max().item()),
        "bridge_state_identity_max_abs_diff": float((state_weight - eye).abs().max().item()),
    }


def bridge_prelude_grad_stats(wrapper: RecurrentQwenForCausalLM) -> dict[str, float]:
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


def build_optimizer(wrapper: RecurrentQwenForCausalLM, cfg: dict[str, Any]) -> OptimizerBundle | torch.optim.Optimizer:
    optimizer_name = str(cfg.get("optimizer", "muon")).lower()
    params = [param for param in wrapper.parameters() if param.requires_grad]
    if not params:
        raise ValueError("No trainable parameters selected")
    lr = cfg_float(cfg, "learning_rate", 5e-6)
    weight_decay = cfg_float(cfg, "weight_decay", 0.0)
    if optimizer_name == "adamw":
        return torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)
    if optimizer_name != "muon":
        raise ValueError("optimizer must be one of: muon, adamw")

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
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--train_jsonl", required=True)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    cfg = load_config(args.config)

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
        collate_fn=partial(collate_causal_batch, pad_token_id=tokenizer.pad_token_id),
    )
    optimizer = build_optimizer(wrapper, cfg)
    optimizer.zero_grad(set_to_none=True)

    curriculum = cfg.get("recurrence_curriculum", {})
    curriculum_enabled = bool(curriculum.get("enabled", True))
    start_loop = int(curriculum.get("start_loop", 1))
    end_loop = int(curriculum.get("end_loop", max_loops))
    schedule = str(curriculum.get("schedule", "linear"))
    target_source = str(curriculum.get("target_source", "schedule"))
    ramp_compute = bool(curriculum.get("ramp_compute", True))

    summary = {"setup": setup, "curriculum_trace": [], "interval_checkpoints": []}
    max_steps = cfg_int(cfg, "max_steps", 25)
    save_every = cfg_int(cfg, "save_every", 0) if cfg.get("save_every", 0) else 0
    prelude_grad_multiplier = cfg_float(cfg, "bridge_prelude_grad_multiplier", 1.0)
    step = 0
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
            pre_clip_bridge_grad = {
                f"pre_clip_{key}": value
                for key, value in bridge_prelude_grad_stats(wrapper).items()
            }
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
                metric_values = {
                    **{key: float(value) for key, value in output.metrics.items()},
                    **pre_clip_bridge_grad,
                    **bridge_grad,
                    **bridge_weight,
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
                if save_every and step % save_every == 0 and step < max_steps:
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
    summary["bridge_prelude_grad_multiplier"] = prelude_grad_multiplier
    if output_dir:
        summary_path = Path(output_dir) / "train_unfrozen_recurrent_summary.json"
        summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(f"training_summary={summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
