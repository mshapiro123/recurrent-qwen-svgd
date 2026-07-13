"""Full-model dense SFT with FP32 AdamW state and BF16 compute."""

from __future__ import annotations

import argparse
import json
import os
import random
import shutil
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

from eval.eval_identity import model_load_kwargs, resolve_dtype  # noqa: E402
from training.dataset import JsonlCausalDataset, collate_causal_batch  # noqa: E402


def checkpoint_name(step: int) -> str:
    return f"dense_full_step_{int(step)}"


def validate_config(config: dict[str, Any]) -> dict[str, Any]:
    required = (
        "model_name",
        "revision",
        "optimizer",
        "parameter_dtype",
        "compute_dtype",
        "max_steps",
        "gradient_accumulation_steps",
    )
    missing = [key for key in required if key not in config]
    if missing:
        raise ValueError(f"Dense-full config missing keys: {missing}")
    if str(config["optimizer"]).lower() != "adamw":
        raise ValueError("Dense-full Phase A requires optimizer=adamw")
    if str(config["parameter_dtype"]).lower() not in {"float32", "fp32"}:
        raise ValueError("Dense-full Phase A requires parameter_dtype=float32 for FP32 AdamW state")
    if int(config["max_steps"]) <= 0 or int(config["gradient_accumulation_steps"]) <= 0:
        raise ValueError("max_steps and gradient_accumulation_steps must be positive")
    return dict(config)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _save_checkpoint(
    model: torch.nn.Module,
    tokenizer: Any,
    *,
    output_dir: Path,
    backup_root: Path | None,
    step: int,
    metadata: dict[str, Any],
) -> tuple[Path, Path | None]:
    destination = output_dir / checkpoint_name(step)
    if destination.exists():
        shutil.rmtree(destination)
    destination.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(destination, safe_serialization=True, max_shard_size="2GB")
    tokenizer.save_pretrained(destination)
    _write_json(destination / "stage5_dense_full_metadata.json", {**metadata, "step": int(step)})
    backup = None
    if backup_root is not None:
        backup = backup_root / checkpoint_name(step)
        temp = backup.with_name(backup.name + ".partial")
        if temp.exists():
            shutil.rmtree(temp)
        temp.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(destination, temp)
        if backup.exists():
            shutil.rmtree(backup)
        temp.rename(backup)
        print(f"checkpoint_drive_backup={backup}", flush=True)
    print(f"saved_checkpoint={destination}", flush=True)
    return destination, backup


def _optimizer_state_is_fp32(optimizer: torch.optim.Optimizer) -> bool:
    floating = [
        value
        for state in optimizer.state.values()
        for value in state.values()
        if isinstance(value, torch.Tensor) and value.is_floating_point() and value.numel() > 1
    ]
    return bool(floating) and all(value.dtype == torch.float32 for value in floating)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--train_jsonl", required=True)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    cfg = validate_config(yaml.safe_load(Path(args.config).read_text(encoding="utf-8")))
    device = torch.device(args.device)
    if device.type != "cuda":
        raise RuntimeError("Full-model Phase A training requires CUDA")

    seed = int(cfg.get("seed", 93_1337))
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    tokenizer = AutoTokenizer.from_pretrained(cfg["model_name"], revision=cfg["revision"])
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    model = AutoModelForCausalLM.from_pretrained(
        cfg["model_name"],
        revision=cfg["revision"],
        **model_load_kwargs(
            cfg["parameter_dtype"],
            cfg.get("attn_implementation", "default"),
            low_cpu_mem_usage=True,
        ),
    ).to(device)
    model.config.use_cache = False
    if bool(cfg.get("gradient_checkpointing", True)):
        model.gradient_checkpointing_enable()
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
    for parameter in model.parameters():
        parameter.requires_grad_(True)
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if not trainable or any(parameter.dtype != torch.float32 for parameter in trainable):
        raise RuntimeError("Dense-full parameters are not all trainable FP32 tensors")
    trainable_numel = sum(parameter.numel() for parameter in trainable)
    total_numel = sum(parameter.numel() for parameter in model.parameters())
    if trainable_numel != total_numel:
        raise RuntimeError("Dense-full training did not unfreeze the complete model")
    print(f"dense_full_trainable_parameters={trainable_numel}", flush=True)

    dataset = JsonlCausalDataset(
        args.train_jsonl,
        tokenizer=tokenizer,
        max_length=int(cfg.get("max_length", 512)),
        max_train_loops=1,
        train_on_prompt=False,
    )
    generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(
        dataset,
        batch_size=int(cfg.get("batch_size", 1)),
        shuffle=True,
        generator=generator,
        collate_fn=partial(collate_causal_batch, pad_token_id=tokenizer.pad_token_id),
    )
    optimizer_kwargs: dict[str, Any] = {
        "lr": float(cfg.get("learning_rate", 2e-6)),
        "weight_decay": float(cfg.get("weight_decay", 0.0)),
    }
    if device.type == "cuda":
        optimizer_kwargs["fused"] = True
    optimizer = torch.optim.AdamW(trainable, **optimizer_kwargs)
    accumulation = int(cfg["gradient_accumulation_steps"])
    max_steps = int(cfg["max_steps"])
    log_every = int(cfg.get("log_every", 25))
    save_every = int(cfg.get("save_every", 2000))
    compute_dtype = resolve_dtype(str(cfg["compute_dtype"]))
    output_dir = Path(cfg["output_dir"])
    backup_root = Path(cfg["backup_root"]) if cfg.get("backup_root") else None
    output_dir.mkdir(parents=True, exist_ok=True)
    optimizer.zero_grad(set_to_none=True)
    iterator = iter(loader)
    loss_trace: list[dict[str, float | int]] = []
    checkpoint_receipts: list[dict[str, Any]] = []
    optimizer_step = 0

    while optimizer_step < max_steps:
        accumulated_loss = 0.0
        for _ in range(accumulation):
            try:
                batch = next(iterator)
            except StopIteration:
                iterator = iter(loader)
                batch = next(iterator)
            batch = {key: value.to(device) for key, value in batch.items() if key in {"input_ids", "attention_mask", "labels"}}
            with torch.autocast(device_type="cuda", dtype=compute_dtype):
                output = model(**batch, use_cache=False, return_dict=True)
                raw_loss = output.loss
            if not bool(torch.isfinite(raw_loss)):
                raise FloatingPointError(f"Non-finite dense-full loss at optimizer step {optimizer_step}")
            (raw_loss / accumulation).backward()
            accumulated_loss += float(raw_loss.detach())

        grad_norm = torch.nn.utils.clip_grad_norm_(
            trainable,
            float(cfg.get("max_grad_norm", 1.0)),
            error_if_nonfinite=True,
        )
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        optimizer_step += 1
        if optimizer_step == 1 and not _optimizer_state_is_fp32(optimizer):
            raise RuntimeError("AdamW optimizer moments are not FP32")
        mean_loss = accumulated_loss / accumulation
        if optimizer_step == 1 or optimizer_step % log_every == 0:
            row = {"step": optimizer_step, "loss": mean_loss, "grad_norm": float(grad_norm)}
            loss_trace.append(row)
            print(f"step={optimizer_step}/{max_steps} loss={mean_loss:.6f} grad_norm={float(grad_norm):.6f}", flush=True)
        if save_every and optimizer_step % save_every == 0:
            checkpoint, backup = _save_checkpoint(
                model,
                tokenizer,
                output_dir=output_dir,
                backup_root=backup_root,
                step=optimizer_step,
                metadata={"config": cfg, "train_jsonl": args.train_jsonl},
            )
            checkpoint_receipts.append(
                {"step": optimizer_step, "checkpoint": str(checkpoint), "drive_backup": str(backup) if backup else None}
            )

    if not checkpoint_receipts or int(checkpoint_receipts[-1]["step"]) != max_steps:
        checkpoint, backup = _save_checkpoint(
            model,
            tokenizer,
            output_dir=output_dir,
            backup_root=backup_root,
            step=max_steps,
            metadata={"config": cfg, "train_jsonl": args.train_jsonl},
        )
        checkpoint_receipts.append(
            {"step": max_steps, "checkpoint": str(checkpoint), "drive_backup": str(backup) if backup else None}
        )
    summary = {
        "kind": "stage5_dense_full_training",
        "status": "finished",
        "optimizer": "adamw_full_fp32_state",
        "model_name": cfg["model_name"],
        "revision": cfg["revision"],
        "trainable_parameters": trainable_numel,
        "effective_batch_size": int(cfg.get("batch_size", 1)) * accumulation,
        "max_steps": max_steps,
        "loss_trace": loss_trace,
        "checkpoints": checkpoint_receipts,
        "final_checkpoint": checkpoint_receipts[-1]["checkpoint"],
        "final_checkpoint_drive_backup": checkpoint_receipts[-1]["drive_backup"],
    }
    _write_json(output_dir / "train_dense_full_summary.json", summary)
    print(json.dumps(summary, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
