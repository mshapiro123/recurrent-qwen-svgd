"""Minimal Phase 2 stochastic latent trajectory training loop."""

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
from training.checkpointing import load_trainable_checkpoint, save_trainable_checkpoint
from training.dataset import JsonlCausalDataset, collate_causal_batch
from training.losses import causal_kl_distillation_loss
from training.reentry_repair import apply_reentry_repair_controls
from training.stability import (
    assert_finite_trainable_gradients,
    assert_finite_trainable_parameters,
    assert_finite_training_state,
)


def distillation_mask(batch: dict[str, torch.Tensor], mode: str) -> torch.Tensor:
    if mode == "response":
        return batch["labels"].ne(-100)
    if mode == "all":
        return batch["attention_mask"].ne(0)
    raise ValueError("distillation.on must be one of: response, all")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config/qwen_0_5b_phase2.yaml")
    parser.add_argument("--train_jsonl", required=True)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))

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
            **model_load_kwargs(
                distill_cfg.get("dtype", cfg.get("dtype", "auto")),
                cfg.get("attn_implementation", "default"),
            ),
        ).to(args.device)
        teacher.eval()
        for param in teacher.parameters():
            param.requires_grad_(False)
        print(f"distillation_teacher={teacher_name}")

    wrapper = RecurrentQwenForCausalLM(
        model,
        layer_split=parse_split(cfg.get("layer_split", "auto")),
        latent_dim=cfg.get("latent_dim", 256),
        latent_scale_init=cfg.get("latent_scale_init", 0.01),
        latent_adapter_std=cfg.get("latent_adapter_std", 1e-4),
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
    if not cfg.get("sample_latents", True):
        wrapper.set_latent_trainable(False)
    wrapper.set_trainable_modules_dtype(adapter_dtype)
    if cfg.get("resume_from"):
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
    optimizer = torch.optim.AdamW(
        wrapper.trainable_component_parameters(),
        lr=cfg.get("learning_rate", 1e-4),
        weight_decay=cfg.get("weight_decay", 0.0),
    )

    max_steps = int(cfg.get("max_steps", 100))
    save_every = int(cfg.get("save_every", 0) or 0)
    step = 0
    while step < max_steps:
        for batch in loader:
            batch = {key: value.to(args.device) for key, value in batch.items()}
            output = wrapper(
                **batch,
                max_loops=cfg.get("max_loops", 4),
                num_trajectories=cfg.get("num_trajectories", 2),
                sample_latents=cfg.get("sample_latents", True),
                beta=cfg.get("beta", 0.02),
                halt_target_nll_weight=cfg.get("halt_target_nll_weight", 0.0),
                eta=cfg.get("eta", 1e-4),
                rho=cfg.get("rho", 1e-3),
                latent_injection_mode=cfg.get("latent_injection_mode", "pre"),
                particle_update_mode=cfg.get("particle_update_mode", "none"),
                particle_init_noise=cfg.get("particle_init_noise", 0.0),
                svgd_eps=cfg.get("svgd_eps", 1.0),
                svgd_repulsion_scale=cfg.get("svgd_repulsion_scale", 1.0),
                svgd_bandwidth=cfg.get("svgd_bandwidth", "median"),
                svgd_bandwidth_floor=cfg.get("svgd_bandwidth_floor", 1e-6),
                svgd_repulsion_max_norm=cfg.get("svgd_repulsion_max_norm"),
                svgd_kernel_projection_dim=cfg.get("svgd_kernel_projection_dim"),
                svgd_kernel_projection_path=cfg.get("svgd_kernel_projection_path"),
                svgd_kernel_geometry=cfg.get("svgd_kernel_geometry", "euclidean"),
                svgd_projection_seed=cfg.get("svgd_projection_seed", 0),
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
                mask = distillation_mask(batch, distill_cfg.get("on", "response"))
                student_logits = output.logits
                teacher_logits = teacher_out.logits
                if distill_cfg.get("target", "mean") == "trajectories":
                    if output.trajectory_logits is None:
                        raise RuntimeError("trajectory distillation requires num_trajectories > 1")
                    num_trajectories = output.trajectory_logits.shape[1]
                    student_logits = output.trajectory_logits.reshape(
                        output.trajectory_logits.shape[0] * num_trajectories,
                        *output.trajectory_logits.shape[2:],
                    )
                    teacher_logits = teacher_out.logits.repeat_interleave(num_trajectories, dim=0)
                    mask = mask.repeat_interleave(num_trajectories, dim=0)
                elif distill_cfg.get("target", "mean") != "mean":
                    raise ValueError("distillation.target must be one of: mean, trajectories")

                distill_loss = causal_kl_distillation_loss(
                    student_logits,
                    teacher_logits,
                    mask,
                    temperature=float(distill_cfg.get("temperature", 1.0)),
                )
                output.loss = output.loss + float(distill_cfg.get("weight", 0.1)) * distill_loss
                output.metrics["base_distill_kl"] = distill_loss.detach()
                output.metrics["loss"] = output.loss.detach()
            assert_finite_training_state(wrapper, output.loss, output.metrics, step)
            output.loss.backward()
            assert_finite_trainable_gradients(wrapper, step)
            torch.nn.utils.clip_grad_norm_(
                wrapper.trainable_component_parameters(),
                cfg.get("max_grad_norm", 1.0),
                error_if_nonfinite=True,
            )
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            assert_finite_trainable_parameters(wrapper, step + 1)

            if step % cfg.get("log_every", 10) == 0:
                metrics = " ".join(f"{key}={float(value):.4f}" for key, value in output.metrics.items())
                print(f"step={step} {metrics}")
            step += 1
            if output_dir := cfg.get("output_dir"):
                if save_every and step % save_every == 0 and step < max_steps:
                    checkpoint_path = save_trainable_checkpoint(wrapper, output_dir, "phase2", step, cfg)
                    print(f"saved_checkpoint={checkpoint_path}")
            if step >= max_steps:
                break
    output_dir = cfg.get("output_dir")
    if output_dir:
        checkpoint_path = save_trainable_checkpoint(wrapper, output_dir, "phase2", step, cfg)
        print(f"saved_checkpoint={checkpoint_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
