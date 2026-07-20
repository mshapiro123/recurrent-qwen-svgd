"""Train one oracle-conditioned re-entry interface on a frozen recurrent keeper."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import shutil
import sys
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
from training.oracle_interface_probe_spec import (  # noqa: E402
    LOCKED_ROUTES,
    assert_oracle_frozen_gradients_zero,
    assert_oracle_frozen_parameter_contract,
)
from training.oracle_intrablock_control_spec import (  # noqa: E402
    LOCKED_ROUTE as LAYERWISE_ROUTE,
    assert_oracle_intrablock_frozen_gradients_zero,
    assert_oracle_intrablock_frozen_parameter_contract,
)
from training.phase_g_alpha_spec import phase_g_active_lineage_hash  # noqa: E402
from training.phase_g_sampling import (  # noqa: E402
    build_base_problem_groups,
    sample_phase_g_row_index,
)
from training.phase_g_training import (  # noqa: E402
    PhaseGEMA,
    load_phase_g_training_progress,
    posterior_target_embeddings,
    save_phase_g_training_progress,
)


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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


def atomic_copy(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".tmp")
    shutil.copy2(source, temporary)
    os.replace(temporary, destination)


def truncate_jsonl_after_step(path: Path, step: int) -> None:
    if not path.exists():
        return
    retained = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if int(row["step"]) <= int(step):
            retained.append(json.dumps(row, sort_keys=True))
    path.write_text(
        "".join(line + "\n" for line in retained),
        encoding="utf-8",
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model_name", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--train_jsonl", required=True)
    parser.add_argument("--keeper", required=True)
    parser.add_argument("--expected_keeper_sha256", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument(
        "--route",
        choices=(*LOCKED_ROUTES, LAYERWISE_ROUTE),
        required=True,
    )
    parser.add_argument("--steps", type=int, default=1500)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--ema_decay", type=float, default=0.999)
    parser.add_argument("--bottleneck_dim", type=int, default=256)
    parser.add_argument("--seed", type=int, default=20260718)
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--split", default="6,18")
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--attn_implementation", default="default")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--log_every", type=int, default=25)
    parser.add_argument("--checkpoint_every", type=int, default=100)
    parser.add_argument("--progress_checkpoint")
    parser.add_argument("--progress_backup_path")
    parser.add_argument("--progress_backup_dir")
    args = parser.parse_args()
    if args.steps < 1 or args.bottleneck_dim < 1 or args.checkpoint_every < 1:
        raise ValueError("steps, bottleneck_dim, and checkpoint_every must be positive")

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
    lineage_start = phase_g_active_lineage_hash(wrapper.named_parameters())
    layerwise = args.route == LAYERWISE_ROUTE
    if layerwise:
        wrapper.enable_oracle_intrablock_conditioner(
            bottleneck_dim=args.bottleneck_dim,
        )
        trainable_names = wrapper.configure_oracle_intrablock_trainable()
        contract_receipt = assert_oracle_intrablock_frozen_parameter_contract(
            wrapper.named_parameters()
        )
    else:
        wrapper.enable_oracle_reentry_conditioner(
            bottleneck_dim=args.bottleneck_dim,
        )
        trainable_names = wrapper.configure_oracle_reentry_trainable()
        contract_receipt = assert_oracle_frozen_parameter_contract(
            wrapper.named_parameters()
        )
    if sorted(trainable_names) != contract_receipt["allowed_trainable"]:
        raise AssertionError("Oracle trainable-name enumeration disagrees with contract")
    trainable = [
        parameter for parameter in wrapper.parameters() if parameter.requires_grad
    ]
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
        raise RuntimeError("Oracle training requires loop_completions on every row")
    groups = build_base_problem_groups(dataset.rows)
    sampler = random.Random(args.seed)

    identity_item = dataset[0]
    identity_depth = int(dataset.rows[0]["depth"])
    identity_batch = collate_causal_batch(
        [identity_item],
        pad_token_id=tokenizer.pad_token_id,
    )
    identity_batch = {
        name: value.to(args.device) for name, value in identity_batch.items()
    }
    identity_loop_labels = identity_batch["loop_labels"][:, :identity_depth, :]
    identity_commands = posterior_target_embeddings(
        wrapper.base_model,
        identity_loop_labels,
    )
    wrapper.eval()
    with torch.no_grad():
        identity_baseline = wrapper(
            input_ids=identity_batch["input_ids"],
            attention_mask=identity_batch["attention_mask"],
            max_loops=identity_depth,
            use_cache=False,
            return_dict=True,
            return_loop_logits=True,
            logits_to_keep=1,
        )
        identity_kwargs = (
            {
                "oracle_intrablock_enabled": True,
                "oracle_intrablock_targets": identity_commands,
            }
            if layerwise
            else {
                "oracle_reentry_enabled": True,
                "oracle_reentry_mode": args.route,
                "oracle_reentry_targets": identity_commands,
            }
        )
        identity_installed = wrapper(
            input_ids=identity_batch["input_ids"],
            attention_mask=identity_batch["attention_mask"],
            max_loops=identity_depth,
            use_cache=False,
            return_dict=True,
            return_loop_logits=True,
            logits_to_keep=1,
            **identity_kwargs,
        )
    if not torch.equal(
        identity_baseline.loop_logits,
        identity_installed.loop_logits,
    ):
        raise AssertionError(
            f"Step-zero {args.route} conditioner is not an exact keeper identity"
        )
    step_zero_identity_exact = True
    wrapper.train()

    optimizer = torch.optim.AdamW(
        trainable,
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    ema = PhaseGEMA(wrapper.named_parameters(), decay=args.ema_decay)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    trace_path = output_dir / "training_trace.jsonl"
    progress_path = (
        Path(args.progress_checkpoint)
        if args.progress_checkpoint
        else output_dir / "training_progress.pt"
    )
    progress_backup_path = (
        Path(args.progress_backup_path) if args.progress_backup_path else None
    )
    progress_backup_dir = (
        Path(args.progress_backup_dir) if args.progress_backup_dir else None
    )
    if (
        not progress_path.exists()
        and progress_backup_path is not None
        and progress_backup_path.exists()
    ):
        atomic_copy(progress_backup_path, progress_path)
    if (
        progress_backup_dir is not None
        and not trace_path.exists()
        and (progress_backup_dir / trace_path.name).exists()
    ):
        atomic_copy(progress_backup_dir / trace_path.name, trace_path)

    progress_contract: dict[str, Any] = {
        "kind": "oracle_interface_probe_training",
        "model_name": args.model_name,
        "keeper_sha256": keeper_sha,
        "train_jsonl_sha256": sha256_file(args.train_jsonl),
        "route": args.route,
        "steps": args.steps,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "ema_decay": args.ema_decay,
        "bottleneck_dim": args.bottleneck_dim,
        "seed": args.seed,
        "max_length": args.max_length,
        "split": args.split,
        "dtype": args.dtype,
        "trainable_names": trainable_names,
    }
    restored_step = 0
    if progress_path.exists():
        restored_step = load_phase_g_training_progress(
            progress_path,
            module=wrapper,
            optimizer=optimizer,
            ema=ema,
            sampler=sampler,
            expected_contract=progress_contract,
        )
        truncate_jsonl_after_step(trace_path, restored_step)
        print(
            f"oracle_training_resume route={args.route} step={restored_step}",
            flush=True,
        )
    trace_handle = trace_path.open("a" if restored_step else "w", encoding="utf-8")
    frozen_assertions = restored_step
    gradient_liveness_assertions = restored_step
    last_completed_step = restored_step

    def persist(step: int) -> None:
        trace_handle.flush()
        save_phase_g_training_progress(
            progress_path,
            module=wrapper,
            optimizer=optimizer,
            ema=ema,
            sampler=sampler,
            step=step,
            contract=progress_contract,
        )
        if progress_backup_path is not None:
            atomic_copy(progress_path, progress_backup_path)
        if progress_backup_dir is not None:
            atomic_copy(trace_path, progress_backup_dir / trace_path.name)
        print(
            f"oracle_progress_saved route={args.route} step={step} path={progress_path}",
            flush=True,
        )

    try:
        for step in range(restored_step + 1, args.steps + 1):
            row_index = sample_phase_g_row_index(
                sampler,
                rows=dataset.rows,
                policy="base_problem_uniform",
                groups=groups,
            )
            row = dataset.rows[row_index]
            depth = int(row["depth"])
            item = dataset[row_index]
            batch = collate_causal_batch(
                [item],
                pad_token_id=tokenizer.pad_token_id,
            )
            batch = {name: value.to(args.device) for name, value in batch.items()}
            loop_labels = batch["loop_labels"][:, :depth, :]
            command_targets = posterior_target_embeddings(
                wrapper.base_model,
                loop_labels,
            )

            optimizer.zero_grad(set_to_none=True)
            oracle_kwargs = (
                {
                    "oracle_intrablock_enabled": True,
                    "oracle_intrablock_targets": command_targets,
                }
                if layerwise
                else {
                    "oracle_reentry_enabled": True,
                    "oracle_reentry_mode": args.route,
                    "oracle_reentry_targets": command_targets,
                }
            )
            output = wrapper(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                labels=batch["labels"],
                loop_labels=loop_labels,
                target_loop_counts=batch["target_loop_counts"].clamp_max(depth),
                max_loops=depth,
                loop_loss_mode="per_loop_labels",
                particle_update_mode="none",
                use_cache=False,
                return_dict=True,
                **oracle_kwargs,
            )
            if output.loss is None or not bool(torch.isfinite(output.loss)):
                raise FloatingPointError(
                    f"Nonfinite oracle interface loss at step {step}"
                )
            output.loss.backward()
            if layerwise:
                assert_oracle_intrablock_frozen_gradients_zero(
                    wrapper.named_parameters()
                )
            else:
                assert_oracle_frozen_gradients_zero(wrapper.named_parameters())
            frozen_assertions += 1
            live = sum(
                int(parameter.grad.detach().count_nonzero().item())
                for parameter in trainable
                if parameter.grad is not None
            )
            if live < 1:
                raise AssertionError(
                    f"Oracle conditioner gradient is not live at step {step}"
                )
            gradient_liveness_assertions += 1
            grad_norm = torch.nn.utils.clip_grad_norm_(trainable, max_norm=1.0)
            if not bool(torch.isfinite(grad_norm)):
                raise FloatingPointError(
                    f"Nonfinite oracle interface gradient at step {step}"
                )
            optimizer.step()
            ema.update(wrapper.named_parameters())
            last_completed_step = step

            trace = {
                "step": step,
                "route": args.route,
                "row_id": str(row["id"]),
                "base_problem_id": str(row["base_problem_id"]),
                "target_variant_index": int(row["target_variant_index"]),
                "depth": depth,
                "loss": float(output.loss.detach().float().cpu().item()),
                "gradient_norm": float(grad_norm.detach().float().cpu().item()),
                **{
                    name: float(value.detach().float().cpu().item())
                    for name, value in output.metrics.items()
                    if name.startswith(
                        ("oracle_reentry_", "oracle_intrablock_")
                    )
                    and value.numel() == 1
                },
            }
            trace_handle.write(json.dumps(trace, sort_keys=True) + "\n")
            if step == 1 or step % args.log_every == 0 or step == args.steps:
                trace_handle.flush()
                print(
                    f"oracle_train route={args.route} step={step}/{args.steps} "
                    f"depth={depth} loss={trace['loss']:.6f} "
                    "residual="
                    f"{trace.get('oracle_reentry_residual_rms_ratio', trace.get('oracle_intrablock_residual_rms_ratio', 0.0)):.6g}",
                    flush=True,
                )
            if step % args.checkpoint_every == 0 or step == args.steps:
                persist(step)
    except BaseException:
        if last_completed_step > restored_step:
            optimizer.zero_grad(set_to_none=True)
            try:
                persist(last_completed_step)
            except Exception as progress_exc:
                print(
                    "oracle_emergency_progress_save_failed="
                    f"{type(progress_exc).__name__}: {progress_exc}",
                    flush=True,
                )
        raise
    finally:
        trace_handle.close()

    lineage_end = phase_g_active_lineage_hash(wrapper.named_parameters())
    if lineage_end != lineage_start:
        raise AssertionError("Frozen keeper changed during oracle interface training")
    config = {
        **vars(args),
        "keeper_sha256": keeper_sha,
        "active_lineage_sha256_start": lineage_start,
        "active_lineage_sha256_end": lineage_end,
        "trainable_names": trainable_names,
        "trainable_parameter_count": sum(
            parameter.numel() for parameter in trainable
        ),
        "frozen_gradient_assertions": frozen_assertions,
        "gradient_liveness_assertions": gradient_liveness_assertions,
        "step_zero_identity_exact": step_zero_identity_exact,
        "sampling_policy": "base_problem_uniform",
        "progress_contract": progress_contract,
        "resumed_from_step": restored_step,
    }
    raw_path = save_trainable_checkpoint(
        wrapper,
        output_dir,
        f"oracle_{args.route}_raw",
        args.steps,
        config,
    )
    backup = ema.copy_to(wrapper.named_parameters())
    ema_path = save_trainable_checkpoint(
        wrapper,
        output_dir,
        f"oracle_{args.route}_ema",
        args.steps,
        config,
    )
    PhaseGEMA.restore(wrapper.named_parameters(), backup)
    torch.save(ema.state_dict(), output_dir / "ema_state.pt")
    summary = {
        "kind": "phase_g_oracle_interface_training",
        "status": "finished",
        "route": args.route,
        "config": config,
        "raw_checkpoint": str(raw_path),
        "ema_checkpoint": str(ema_path),
        "training_trace": str(trace_path),
        "progress_checkpoint": str(progress_path),
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
