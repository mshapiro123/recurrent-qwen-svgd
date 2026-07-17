"""Train only the guided stochastic heads on a frozen recurrent keeper."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from functools import partial
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import torch
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from eval.eval_mcq import load_recurrent_wrapper  # noqa: E402
from training.checkpointing import save_trainable_checkpoint  # noqa: E402
from training.dataset import JsonlCausalDataset, collate_causal_batch  # noqa: E402
from training.phase_g_alpha_spec import (  # noqa: E402
    assert_frozen_gradients_zero,
    assert_frozen_parameter_contract,
    phase_g_active_lineage_hash,
)
from training.phase_g_training import PhaseGEMA, posterior_target_embeddings  # noqa: E402


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def jsonable_metrics(metrics: dict[str, torch.Tensor]) -> dict[str, float]:
    return {
        name: float(value.detach().float().cpu().item())
        for name, value in metrics.items()
        if value.numel() == 1
    }


def loader_args(args: argparse.Namespace) -> SimpleNamespace:
    return SimpleNamespace(
        model_name=args.model_name,
        checkpoint=args.keeper,
        split=args.split,
        bridge_projection_mode="split",
        dtype=args.dtype,
        attn_implementation=args.attn_implementation,
        device=args.device,
        lora_rank=0,
        lora_alpha=16,
        adapter_dtype="float32",
        base_lora_layer_range="all",
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model_name", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--train_jsonl", required=True)
    parser.add_argument("--keeper", required=True)
    parser.add_argument("--expected_keeper_sha256", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--kl_coefficient", type=float, required=True)
    parser.add_argument("--kl_balance", type=float, default=0.8)
    parser.add_argument("--ema_decay", type=float, default=0.999)
    parser.add_argument("--num_trajectories", type=int, default=4)
    parser.add_argument("--latent_dim", type=int, default=64)
    parser.add_argument("--projection_seed", type=int, default=20260717)
    parser.add_argument("--injection_scale_init", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=20260717)
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--split", default="6,18")
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--attn_implementation", default="default")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--log_every", type=int, default=25)
    args = parser.parse_args()

    if args.steps < 1 or args.num_trajectories < 1:
        raise ValueError("steps and num_trajectories must be positive")
    keeper_sha = sha256_file(args.keeper)
    if keeper_sha != args.expected_keeper_sha256:
        raise RuntimeError(
            f"Keeper SHA mismatch: expected {args.expected_keeper_sha256}, got {keeper_sha}"
        )

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    wrapper = load_recurrent_wrapper(loader_args(args), args.keeper)
    wrapper.enable_phase_g_guidance(
        latent_dim=args.latent_dim,
        projection_seed=args.projection_seed,
        injection_scale_init=args.injection_scale_init,
    )
    trainable_names = wrapper.configure_phase_g_trainable()
    contract = assert_frozen_parameter_contract(wrapper.named_parameters())
    if sorted(trainable_names) != sorted(contract["allowed_trainable"]):
        raise AssertionError("Phase G trainable-name enumeration disagrees with the contract")
    lineage_start = phase_g_active_lineage_hash(wrapper.named_parameters())
    wrapper.train()

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    dataset = JsonlCausalDataset(
        args.train_jsonl,
        tokenizer=tokenizer,
        max_length=args.max_length,
        max_train_loops=4,
        train_on_prompt=False,
    )
    if not dataset.rows or not all("loop_completions" in row for row in dataset.rows):
        raise RuntimeError("Phase G training requires gold loop_completions on every row")
    sampler = random.Random(args.seed)

    trainable = [parameter for parameter in wrapper.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(
        trainable,
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    ema = PhaseGEMA(wrapper.named_parameters(), decay=args.ema_decay)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    trace_path = output_dir / "training_trace.jsonl"
    seed_manifest_path = output_dir / "rng_manifest.jsonl"
    trace_handle = trace_path.open("w", encoding="utf-8")
    seed_handle = seed_manifest_path.open("w", encoding="utf-8")
    frozen_assertions = 0

    try:
        for step in range(1, args.steps + 1):
            row_index = sampler.randrange(len(dataset))
            row = dataset.rows[row_index]
            depth = int(row["depth"])
            item = dataset[row_index]
            batch = collate_causal_batch([item], pad_token_id=tokenizer.pad_token_id)
            batch = {name: value.to(args.device) for name, value in batch.items()}
            loop_labels = batch["loop_labels"][:, :depth, :]
            posterior_targets = posterior_target_embeddings(
                wrapper.base_model,
                loop_labels,
            )
            trajectory_seeds = [
                args.seed + step * 1_000_003 + trajectory
                for trajectory in range(args.num_trajectories)
            ]

            optimizer.zero_grad(set_to_none=True)
            output = wrapper(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                labels=batch["labels"],
                loop_labels=loop_labels,
                target_loop_counts=batch["target_loop_counts"].clamp_max(depth),
                max_loops=depth,
                num_trajectories=args.num_trajectories,
                loop_loss_mode="per_loop_labels",
                particle_update_mode="none",
                use_cache=False,
                return_dict=True,
                phase_g_enabled=True,
                phase_g_use_posterior=True,
                phase_g_posterior_targets=posterior_targets,
                phase_g_trajectory_seeds=trajectory_seeds,
                phase_g_kl_balance=args.kl_balance,
                phase_g_kl_coefficient=args.kl_coefficient,
            )
            if output.loss is None or not bool(torch.isfinite(output.loss)):
                raise FloatingPointError(f"Nonfinite Phase G loss at step {step}")
            output.loss.backward()
            assert_frozen_gradients_zero(wrapper.named_parameters())
            frozen_assertions += 1
            grad_norm = torch.nn.utils.clip_grad_norm_(trainable, max_norm=1.0)
            if not bool(torch.isfinite(grad_norm)):
                raise FloatingPointError(f"Nonfinite Phase G gradient at step {step}")
            optimizer.step()
            ema.update(wrapper.named_parameters())

            trace = {
                "step": step,
                "row_id": row["id"],
                "depth": depth,
                "reachable_set_stratum": row["reachable_set_stratum"],
                "loss": float(output.loss.detach().float().cpu().item()),
                "gradient_norm": float(grad_norm.detach().float().cpu().item()),
                "trajectory_seeds": trajectory_seeds,
                **jsonable_metrics(output.metrics),
            }
            trace_handle.write(json.dumps(trace, sort_keys=True) + "\n")
            seed_handle.write(
                json.dumps(
                    {
                        "step": step,
                        "row_id": row["id"],
                        "trajectory_seeds": trajectory_seeds,
                    },
                    sort_keys=True,
                )
                + "\n"
            )
            if step == 1 or step % args.log_every == 0 or step == args.steps:
                trace_handle.flush()
                seed_handle.flush()
                print(
                    "phase_g_train "
                    f"step={step}/{args.steps} depth={depth} "
                    f"loss={trace['loss']:.6f} kl={trace.get('phase_g_balanced_kl', 0.0):.6f} "
                    f"scale={trace.get('phase_g_injection_scale', 0.0):.6g}",
                    flush=True,
                )
    finally:
        trace_handle.close()
        seed_handle.close()

    lineage_end = phase_g_active_lineage_hash(wrapper.named_parameters())
    if lineage_end != lineage_start:
        raise AssertionError("Frozen Phase G substrate changed during training")

    config: dict[str, Any] = {
        **vars(args),
        "keeper_sha256": keeper_sha,
        "active_lineage_sha256_start": lineage_start,
        "active_lineage_sha256_end": lineage_end,
        "trainable_names": trainable_names,
        "frozen_gradient_assertions": frozen_assertions,
    }
    raw_path = save_trainable_checkpoint(
        wrapper,
        output_dir,
        "phase_g_raw",
        args.steps,
        config,
    )
    backup = ema.copy_to(wrapper.named_parameters())
    ema_path = save_trainable_checkpoint(
        wrapper,
        output_dir,
        "phase_g_ema",
        args.steps,
        config,
    )
    PhaseGEMA.restore(wrapper.named_parameters(), backup)
    torch.save(ema.state_dict(), output_dir / "ema_state.pt")

    summary = {
        "kind": "phase_g_alpha_training",
        "status": "finished",
        "config": config,
        "raw_checkpoint": str(raw_path),
        "ema_checkpoint": str(ema_path),
        "training_trace": str(trace_path),
        "rng_manifest": str(seed_manifest_path),
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
