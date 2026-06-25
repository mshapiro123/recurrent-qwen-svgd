"""Minimal dense Qwen LoRA SFT loop for standard-control experiments."""

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

from eval.eval_identity import model_load_kwargs, resolve_dtype  # noqa: E402
from models.lora import apply_lora_to_qwen_layers, mark_only_lora_trainable, set_lora_adapter_dtype  # noqa: E402
from training.checkpointing import load_trainable_checkpoint, save_trainable_checkpoint  # noqa: E402
from training.dataset import JsonlCausalDataset, collate_causal_batch  # noqa: E402
from training.losses import causal_kl_distillation_loss  # noqa: E402
from training.stability import (  # noqa: E402
    assert_finite_trainable_gradients,
    assert_finite_trainable_parameters,
    assert_finite_training_state,
)
from training.train_phase1_ponder import cfg_float, cfg_int  # noqa: E402


def load_config(path: str | Path) -> dict:
    return yaml.safe_load(Path(path).read_text(encoding="utf-8"))


def trainable_parameters(module: torch.nn.Module) -> list[torch.nn.Parameter]:
    return [param for param in module.parameters() if param.requires_grad]


def parse_layer_range(value: str | None, num_layers: int) -> tuple[int, int]:
    if value is None or value.lower() in {"", "auto", "all"}:
        return 0, num_layers
    left, right = value.split(",", maxsplit=1)
    start, end = int(left), int(right)
    if not 0 <= start < end <= num_layers:
        raise ValueError(f"Invalid dense LoRA layer range {value!r} for {num_layers} layers")
    return start, end


def distillation_mask(batch: dict[str, torch.Tensor], mode: str) -> torch.Tensor:
    if mode == "response":
        return batch["labels"].ne(-100)
    if mode == "all":
        return batch["attention_mask"].ne(0)
    raise ValueError("distillation.on must be one of: response, all")


def model_batch(batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {
        "input_ids": batch["input_ids"],
        "attention_mask": batch["attention_mask"],
        "labels": batch["labels"],
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
    for param in model.parameters():
        param.requires_grad_(False)

    adapter_dtype = resolve_dtype(cfg.get("adapter_dtype", "float32"))
    lora_cfg = cfg.get("lora", {})
    if lora_cfg.get("enabled", True):
        num_layers = len(model.model.layers)
        layer_range = lora_cfg.get("layer_range", cfg.get("layer_split", "6,18"))
        start, end = parse_layer_range(layer_range, num_layers)
        replaced = apply_lora_to_qwen_layers(
            model,
            start_layer=start,
            end_layer=end,
            rank=int(lora_cfg.get("rank", 8)),
            alpha=float(lora_cfg.get("alpha", 16)),
            dropout=float(lora_cfg.get("dropout", 0.0)),
            adapter_dtype=adapter_dtype,
        )
        print(f"dense_lora_modules={replaced} layer_range={start},{end}")
    mark_only_lora_trainable(model)
    set_lora_adapter_dtype(model, adapter_dtype)

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

    if cfg.get("resume_from"):
        load_info = load_trainable_checkpoint(model, cfg["resume_from"])
        print(f"loaded_checkpoint={cfg['resume_from']} loaded_keys={len(load_info['loaded_keys'])}")
    assert_finite_trainable_parameters(model, step=0)
    model.train()

    dataset = JsonlCausalDataset(
        args.train_jsonl,
        tokenizer=tokenizer,
        max_length=cfg_int(cfg, "max_length", 1024),
        max_train_loops=cfg_int(cfg, "max_loops", 4),
        train_on_prompt=cfg.get("train_on_prompt", False),
    )
    loader = DataLoader(
        dataset,
        batch_size=cfg_int(cfg, "batch_size", 1),
        shuffle=True,
        collate_fn=partial(collate_causal_batch, pad_token_id=tokenizer.pad_token_id),
    )
    optimizer = torch.optim.AdamW(
        trainable_parameters(model),
        lr=cfg_float(cfg, "learning_rate", 1e-5),
        weight_decay=cfg_float(cfg, "weight_decay", 0.0),
    )

    max_steps = cfg_int(cfg, "max_steps", 100)
    save_every = cfg_int(cfg, "save_every", 0) if cfg.get("save_every", 0) else 0
    step = 0
    while step < max_steps:
        for batch in loader:
            batch = {key: value.to(args.device) for key, value in batch.items()}
            output = model(**model_batch(batch), use_cache=False, return_dict=True)
            metrics: dict[str, torch.Tensor] = {"loss": output.loss.detach(), "ce": output.loss.detach()}
            loss = output.loss
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
                loss = loss + float(distill_cfg.get("weight", 0.1)) * distill_loss
                metrics["base_distill_kl"] = distill_loss.detach()
                metrics["loss"] = loss.detach()

            assert_finite_training_state(model, loss, metrics, step)
            loss.backward()
            assert_finite_trainable_gradients(model, step)
            torch.nn.utils.clip_grad_norm_(
                trainable_parameters(model),
                cfg_float(cfg, "max_grad_norm", 1.0),
                error_if_nonfinite=True,
            )
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            assert_finite_trainable_parameters(model, step + 1)

            if step % cfg_int(cfg, "log_every", 10) == 0:
                metric_text = " ".join(f"{key}={float(value):.4f}" for key, value in metrics.items())
                print(f"step={step} {metric_text}")
            step += 1
            if output_dir := cfg.get("output_dir"):
                if save_every and step % save_every == 0 and step < max_steps:
                    checkpoint_path = save_trainable_checkpoint(model, output_dir, "dense_lora", step, cfg)
                    print(f"saved_checkpoint={checkpoint_path}")
            if step >= max_steps:
                break

    output_dir = cfg.get("output_dir")
    if output_dir:
        checkpoint_path = save_trainable_checkpoint(model, output_dir, "dense_lora", step, cfg)
        print(f"saved_checkpoint={checkpoint_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
