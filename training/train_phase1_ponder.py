"""Minimal Phase 1 deterministic PonderNet training loop."""

from __future__ import annotations

import argparse
import sys
from functools import partial
from pathlib import Path

import torch
import yaml
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from eval.eval_identity import model_load_kwargs, parse_split, resolve_dtype
from models.lora import apply_lora_to_recurrent_block
from models.recurrent_wrapper import RecurrentQwenForCausalLM
from training.checkpointing import save_trainable_checkpoint
from training.dataset import JsonlCausalDataset, collate_causal_batch
from training.losses import causal_kl_distillation_loss
from training.reentry_repair import apply_reentry_repair_controls
from training.stability import (
    assert_finite_trainable_gradients,
    assert_finite_trainable_parameters,
    assert_finite_training_state,
)


def load_config(path: str | Path) -> dict:
    return yaml.safe_load(Path(path).read_text(encoding="utf-8"))


def distillation_mask(batch: dict[str, torch.Tensor], mode: str) -> torch.Tensor:
    if mode == "response":
        return batch["labels"].ne(-100)
    if mode == "all":
        return batch["attention_mask"].ne(0)
    raise ValueError("distillation.on must be one of: response, all")


def optimizer_parameters(wrapper: RecurrentQwenForCausalLM, cfg: dict) -> list[torch.nn.Parameter]:
    """Select which trainable components the optimizer updates.

    ``requires_grad`` remains true for all lightweight adapter parameters so
    checkpoints still include the complete adapter state. This selector only
    controls which parameters receive optimizer steps.
    """

    modules = str(cfg.get("optimizer_modules", "all")).strip().lower()
    if modules in {"", "all"}:
        return wrapper.trainable_component_parameters()

    selected: list[torch.nn.Parameter] = []
    requested = {item.strip() for item in modules.split(",") if item.strip()}
    valid = {"lora", "bridge", "halt", "latent"}
    unknown = requested - valid
    if unknown:
        raise ValueError(f"Unknown optimizer_modules entries: {sorted(unknown)}")
    if "lora" in requested:
        selected.extend(param for param in wrapper.base_model.parameters() if param.requires_grad)
    if "bridge" in requested:
        selected.extend(param for param in wrapper.bridge.parameters() if param.requires_grad)
    if "halt" in requested:
        selected.extend(param for param in wrapper.halt_predictor.parameters() if param.requires_grad)
    if "latent" in requested:
        selected.extend(param for param in wrapper.latent_trajectory.parameters() if param.requires_grad)
    if not selected:
        raise ValueError(f"optimizer_modules={modules!r} selected no trainable parameters")
    return selected


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config/qwen_0_5b_phase1.yaml")
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
    distill_cfg = cfg.get("distillation", {})
    teacher = None
    if distill_cfg.get("enabled", False):
        teacher_name = distill_cfg.get("teacher_model_name", cfg["model_name"])
        teacher = AutoModelForCausalLM.from_pretrained(
            teacher_name,
            **model_load_kwargs(distill_cfg.get("dtype", cfg.get("dtype", "auto")), cfg.get("attn_implementation", "default")),
        ).to(args.device)
        teacher.eval()
        for param in teacher.parameters():
            param.requires_grad_(False)
        print(f"distillation_teacher={teacher_name}")

    wrapper = RecurrentQwenForCausalLM(
        model,
        layer_split=parse_split(cfg.get("layer_split", "auto")),
        initial_halt_prob=cfg.get("initial_halt_prob", 0.25),
    ).to(args.device)
    adapter_dtype = resolve_dtype(cfg.get("adapter_dtype", "float32"))
    lora_cfg = cfg.get("lora", {})
    if lora_cfg.get("enabled", True):
        replaced = apply_lora_to_recurrent_block(
            wrapper,
            rank=lora_cfg.get("rank", 8),
            alpha=lora_cfg.get("alpha", 16),
            dropout=lora_cfg.get("dropout", 0.0),
            adapter_dtype=adapter_dtype,
        )
        print(f"lora_recurrent_modules={replaced}")
    wrapper.freeze_base_model()
    wrapper.set_latent_trainable(False)
    wrapper.set_trainable_modules_dtype(adapter_dtype)
    if cfg.get("resume_from"):
        from training.checkpointing import load_trainable_checkpoint

        load_info = load_trainable_checkpoint(wrapper, cfg["resume_from"])
        print(f"loaded_checkpoint={cfg['resume_from']} loaded_keys={len(load_info['loaded_keys'])}")
    repair_info = apply_reentry_repair_controls(wrapper, cfg)
    if repair_info["applied"]:
        print("reentry_repair_controls=" + " ".join(f"{key}={value}" for key, value in repair_info.items()))
    assert_finite_trainable_parameters(wrapper, step=0)
    wrapper.train()

    dataset = JsonlCausalDataset(
        args.train_jsonl,
        tokenizer=tokenizer,
        max_length=cfg.get("max_length", 1024),
        max_train_loops=cfg.get("max_loops", 4),
        train_on_prompt=cfg.get("train_on_prompt", False),
    )
    loader = DataLoader(
        dataset,
        batch_size=cfg.get("batch_size", 1),
        shuffle=True,
        collate_fn=partial(collate_causal_batch, pad_token_id=tokenizer.pad_token_id),
    )
    optimizer_params = optimizer_parameters(wrapper, cfg)
    print(f"optimizer_parameter_tensors={len(optimizer_params)}")
    optimizer = torch.optim.AdamW(
        optimizer_params,
        lr=cfg.get("learning_rate", 1e-4),
        weight_decay=cfg.get("weight_decay", 0.0),
    )
    wrapper.zero_grad(set_to_none=True)

    max_steps = int(cfg.get("max_steps", 100))
    save_every = int(cfg.get("save_every", 0) or 0)
    step = 0
    while step < max_steps:
        for batch in loader:
            batch = {key: value.to(args.device) for key, value in batch.items()}
            if cfg.get("use_target_loop_control", False):
                batch["halt_control_loop_counts"] = batch["target_loop_counts"]
            output = wrapper(
                **batch,
                max_loops=cfg.get("max_loops", 4),
                beta=cfg.get("beta", 0.02),
                halt_target_nll_weight=cfg.get("halt_target_nll_weight", 0.0),
                use_learned_loop_control=cfg.get("use_learned_loop_control", False),
                loop_control_ce_weight=cfg.get("loop_control_ce_weight", 0.0),
                reentry_rescale_mode=cfg.get("reentry_rescale_mode", "none"),
                use_cache=False,
                return_dict=True,
            )
            if teacher is not None:
                with torch.no_grad():
                    teacher_out = teacher(
                        input_ids=batch["input_ids"],
                        attention_mask=batch["attention_mask"],
                        use_cache=False,
                        return_dict=True,
                    )
                distill_loss = causal_kl_distillation_loss(
                    output.logits,
                    teacher_out.logits,
                    distillation_mask(batch, distill_cfg.get("on", "response")),
                    temperature=float(distill_cfg.get("temperature", 1.0)),
                )
                output.loss = output.loss + float(distill_cfg.get("weight", 0.1)) * distill_loss
                output.metrics["base_distill_kl"] = distill_loss.detach()
                output.metrics["loss"] = output.loss.detach()
            assert_finite_training_state(wrapper, output.loss, output.metrics, step)
            output.loss.backward()
            assert_finite_trainable_gradients(wrapper, step)
            torch.nn.utils.clip_grad_norm_(
                optimizer_params,
                cfg.get("max_grad_norm", 1.0),
                error_if_nonfinite=True,
            )
            optimizer.step()
            wrapper.zero_grad(set_to_none=True)
            assert_finite_trainable_parameters(wrapper, step + 1)

            if step % cfg.get("log_every", 10) == 0:
                metrics = " ".join(f"{key}={float(value):.4f}" for key, value in output.metrics.items())
                print(f"step={step} {metrics}")
            step += 1
            if output_dir := cfg.get("output_dir"):
                if save_every and step % save_every == 0 and step < max_steps:
                    checkpoint_path = save_trainable_checkpoint(wrapper, output_dir, "phase1", step, cfg)
                    print(f"saved_checkpoint={checkpoint_path}")
            if step >= max_steps:
                break
    output_dir = cfg.get("output_dir")
    if output_dir:
        checkpoint_path = save_trainable_checkpoint(wrapper, output_dir, "phase1", step, cfg)
        print(f"saved_checkpoint={checkpoint_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
