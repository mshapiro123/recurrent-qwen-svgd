"""Train only the DC1 horizontal bridge under the locked Stage A protocol."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import time
from pathlib import Path
from typing import Any

import torch

from eval.cache_speculative_depth_d0_teachers import load_drafter, read_jsonl
from eval.eval_speculative_depth_d0_floor import load_partition_cache
from models.coconut_composite import CoconutRecurrentQwen
from training.speculative_depth_d0_corpus import sha256_file


def write_json(path: str | Path, payload: Any) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(temporary, destination)


def names_sha256(names: list[str] | set[str]) -> str:
    payload = ("\n".join(sorted(names)) + "\n").encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def exact_frozen_fingerprint(model: torch.nn.Module, allowlist: set[str]) -> str:
    digest = hashlib.sha256()
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            if name in allowlist:
                continue
            value = parameter.detach().cpu().contiguous()
            digest.update(name.encode("utf-8"))
            digest.update(str(tuple(value.shape)).encode("ascii"))
            digest.update(str(value.dtype).encode("ascii"))
            digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def freeze_except_allowlist(
    model: torch.nn.Module, allowlist: set[str]
) -> dict[str, Any]:
    observed = {name for name, _parameter in model.named_parameters()}
    missing = sorted(allowlist - observed)
    if missing:
        raise RuntimeError(f"Stage A trainable allowlist names are missing: {missing}")
    for name, parameter in model.named_parameters():
        parameter.requires_grad_(name in allowlist)
    trainable = {name for name, parameter in model.named_parameters() if parameter.requires_grad}
    if trainable != allowlist:
        raise RuntimeError(
            f"Stage A trainable set mismatch: expected={sorted(allowlist)} observed={sorted(trainable)}"
        )
    return {
        "parameter_names": sorted(trainable),
        "parameter_names_sha256": names_sha256(trainable),
        "trainable_parameters": sum(
            parameter.numel()
            for name, parameter in model.named_parameters()
            if name in allowlist
        ),
    }


def build_stage_a_example(
    *,
    row: dict[str, Any],
    teacher_ids: torch.Tensor,
    local_position: int,
    latent_token_id: int,
    terminal_token_id: int,
    device: str,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    values = [int(value) for value in row["input_ids"]]
    position = int(local_position)
    if position < 0 or position >= len(values) - 1:
        raise ValueError("Stage A local position must predict a real next token")
    if position >= len(teacher_ids):
        raise ValueError("Stage A teacher cache does not cover the sampled position")
    prefix = values[: position + 1]
    sequence = [*prefix, int(latent_token_id), int(terminal_token_id)]
    input_ids = torch.tensor([sequence], dtype=torch.long, device=device)
    attention_mask = torch.ones_like(input_ids)
    labels = torch.full_like(input_ids, -100)
    labels[0, -1] = int(teacher_ids[position])
    return input_ids, attention_mask, labels


def assert_only_allowlist_gradients(
    model: torch.nn.Module, allowlist: set[str]
) -> None:
    for name, parameter in model.named_parameters():
        if name in allowlist:
            if parameter.grad is None or not torch.isfinite(parameter.grad).all():
                raise RuntimeError(f"Stage A trainable gradient missing or non-finite: {name}")
        elif parameter.grad is not None and torch.count_nonzero(parameter.grad).item() != 0:
            raise RuntimeError(f"Stage A frozen parameter received a nonzero gradient: {name}")


def save_checkpoint(
    *,
    path: Path,
    composite: CoconutRecurrentQwen,
    optimizer: torch.optim.Optimizer,
    step: int,
    rng: random.Random,
    prereg_sha256: str,
) -> None:
    payload = {
        "kind": "paper2_dc1_stage_a_bridge_checkpoint",
        "step": int(step),
        "horizontal_bridge": composite.horizontal_bridge.state_dict(),
        "optimizer": optimizer.state_dict(),
        "python_rng_state": rng.getstate(),
        "torch_rng_state": torch.random.get_rng_state(),
        "cuda_rng_state": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        "prereg_sha256": prereg_sha256,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prereg", required=True)
    parser.add_argument("--data_jsonl", required=True)
    parser.add_argument("--teacher_cache_summary", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--private_train_dir", required=True)
    parser.add_argument("--output_summary", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--attn_implementation", default="sdpa")
    args = parser.parse_args()

    prereg_path = Path(args.prereg)
    prereg = json.loads(prereg_path.read_text(encoding="utf-8"))
    if prereg.get("locked_before_training") is not True:
        raise RuntimeError("Stage A training requires a locked preregistration")
    if sha256_file(args.checkpoint) != prereg["init_checkpoint_sha256"]:
        raise RuntimeError("Stage A initialization checkpoint SHA-256 mismatch")
    if sha256_file(args.data_jsonl) != prereg["train_partition"]["jsonl_sha256"]:
        raise RuntimeError("Stage A DEV-C JSONL SHA-256 mismatch")
    prereg_sha = sha256_file(prereg_path)
    policy = prereg["optimization"]
    if policy["precision"] != "full_fp32":
        raise RuntimeError("Stage A requires full fp32")

    rows = read_jsonl(args.data_jsonl)
    teacher_summary = json.loads(
        Path(args.teacher_cache_summary).read_text(encoding="utf-8")
    )
    teacher_rows = load_partition_cache(teacher_summary, "teacher_7b", "dev_c")
    _tokenizer, wrapper, resize, _original_vocab = load_drafter(
        checkpoint=Path(args.checkpoint),
        device=args.device,
        dtype="float32",
        attn_implementation=args.attn_implementation,
    )
    composite = CoconutRecurrentQwen(
        wrapper, latent_token_id=int(resize.control_token_ids[2])
    ).to(device=args.device, dtype=torch.float32)
    allowlist = set(prereg["trainable"]["allowlist"])
    trainable_receipt = freeze_except_allowlist(composite, allowlist)
    if composite.horizontal_bridge.delta.weight.detach().abs().max().item() != 0:
        raise RuntimeError("Stage A bridge did not initialize at exact identity")
    composite.train()

    frozen_before = exact_frozen_fingerprint(composite, allowlist)
    probe = torch.tensor([rows[0]["input_ids"][:16]], device=args.device)
    composite.eval()
    with torch.no_grad():
        k0_before = composite(
            input_ids=probe,
            attention_mask=torch.ones_like(probe),
            horizontal_steps=0,
            max_loops=1,
        ).logits.detach().cpu()
    composite.train()

    optimizer = torch.optim.AdamW(
        [composite.horizontal_bridge.delta.weight],
        lr=float(policy["lr"]),
        weight_decay=float(policy["weight_decay"]),
    )
    rng = random.Random(int(policy["seed"]))
    torch.manual_seed(int(policy["seed"]))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(policy["seed"]))

    private_dir = Path(args.private_train_dir)
    private_dir.mkdir(parents=True, exist_ok=True)
    checkpoints = [int(value) for value in policy["checkpoints_passive"]]
    start_step = 0
    for step in reversed(checkpoints):
        candidate = private_dir / f"stage_a_step_{step}.pt"
        if not candidate.exists():
            continue
        payload = torch.load(candidate, map_location="cpu", weights_only=False)
        if payload.get("prereg_sha256") != prereg_sha:
            raise RuntimeError("Stage A resume checkpoint preregistration mismatch")
        composite.horizontal_bridge.load_state_dict(payload["horizontal_bridge"])
        optimizer.load_state_dict(payload["optimizer"])
        rng.setstate(payload["python_rng_state"])
        torch.random.set_rng_state(payload["torch_rng_state"])
        if torch.cuda.is_available() and payload.get("cuda_rng_state") is not None:
            torch.cuda.set_rng_state_all(payload["cuda_rng_state"])
        start_step = int(payload["step"])
        print(f"stage_a_resume step={start_step} checkpoint={candidate}", flush=True)
        break

    losses: list[float] = []
    records: list[dict[str, Any]] = []
    started = time.monotonic()
    maximum_steps = int(policy["steps"])
    for step in range(start_step + 1, maximum_steps + 1):
        row_index = rng.randrange(len(rows))
        row = rows[row_index]
        available_positions = min(len(row["input_ids"]) - 1, int(policy["seq_len"]) - 2)
        if available_positions <= 0:
            raise RuntimeError("Stage A sampled a row without a prediction position")
        local_position = rng.randrange(available_positions)
        teacher_ids = teacher_rows[row_index]["teacher_greedy_token_id"].long()
        input_ids, attention_mask, labels = build_stage_a_example(
            row=row,
            teacher_ids=teacher_ids,
            local_position=local_position,
            latent_token_id=int(resize.control_token_ids[2]),
            terminal_token_id=int(_tokenizer.eos_token_id),
            device=args.device,
        )
        optimizer.zero_grad(set_to_none=True)
        output = composite(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            horizontal_steps=1,
            max_loops=1,
            execution_mode="recompute",
            raw_feedback=False,
        )
        if output.loss is None or not torch.isfinite(output.loss):
            raise RuntimeError(f"Stage A loss is missing or non-finite at step {step}")
        output.loss.backward()
        assert_only_allowlist_gradients(composite, allowlist)
        grad_norm = float(
            torch.nn.utils.clip_grad_norm_(
                [composite.horizontal_bridge.delta.weight],
                max_norm=float(policy["grad_clip"]),
            ).detach().cpu()
        )
        optimizer.step()
        loss_value = float(output.loss.detach().cpu())
        losses.append(loss_value)
        if step == 1 or step % 25 == 0:
            record = {
                "step": step,
                "loss": loss_value,
                "mean_loss_last_25": sum(losses[-25:]) / len(losses[-25:]),
                "grad_norm_pre_clip": grad_norm,
                "row_index": row_index,
                "local_position": local_position,
                "sequence_length": int(input_ids.shape[1]),
                "bridge_delta_rms": float(
                    composite.horizontal_bridge.delta.weight.detach().square().mean().sqrt().cpu()
                ),
                "elapsed_seconds": time.monotonic() - started,
            }
            records.append(record)
            print("stage_a_progress", json.dumps(record, sort_keys=True), flush=True)
        if step == 20:
            elapsed = time.monotonic() - started
            projected = elapsed / 20.0 * maximum_steps
            print(
                f"stage_a_throughput_check elapsed_20={elapsed:.2f}s "
                f"projected_total_hours={projected / 3600:.2f} "
                f"cuda_peak_mib={torch.cuda.max_memory_allocated() / 2**20:.1f}",
                flush=True,
            )
        if step in checkpoints:
            save_checkpoint(
                path=private_dir / f"stage_a_step_{step}.pt",
                composite=composite,
                optimizer=optimizer,
                step=step,
                rng=rng,
                prereg_sha256=prereg_sha,
            )

    frozen_after = exact_frozen_fingerprint(composite, allowlist)
    if frozen_after != frozen_before:
        raise RuntimeError("Stage A frozen parameter hash changed")
    composite.eval()
    with torch.no_grad():
        k0_after = composite(
            input_ids=probe,
            attention_mask=torch.ones_like(probe),
            horizontal_steps=0,
            max_loops=1,
        ).logits.detach().cpu()
    k0_max_abs_difference = float((k0_before - k0_after).abs().max())
    if not torch.equal(k0_before, k0_after):
        raise RuntimeError(
            f"Stage A k=0 bit identity failed: max_abs_difference={k0_max_abs_difference}"
        )

    checkpoint_receipts = {}
    for step in checkpoints:
        path = private_dir / f"stage_a_step_{step}.pt"
        checkpoint_receipts[str(step)] = {
            "path": str(path),
            "sha256": sha256_file(path),
        }
    summary = {
        "kind": "paper2_dc1_stage_a_training",
        "status": "complete_ready_for_single_eval_c_pass",
        "prereg_sha256": prereg_sha,
        "init_checkpoint_sha256": prereg["init_checkpoint_sha256"],
        "train_partition_sha256": prereg["train_partition"]["jsonl_sha256"],
        "steps": maximum_steps,
        "start_step": start_step,
        "training_seed": int(policy["seed"]),
        "sampling": "one_uniform_valid_decision_position_from_one_uniform_dev_c_row_per_step",
        "precision": "full_fp32",
        "trainable": trainable_receipt,
        "frozen_parameter_sha256_before": frozen_before,
        "frozen_parameter_sha256_after": frozen_after,
        "frozen_parameter_hash_match": True,
        "k0_bit_identity": True,
        "k0_max_abs_difference": k0_max_abs_difference,
        "loss": {
            "first": losses[0] if losses else None,
            "last": losses[-1] if losses else None,
            "mean_last_100": sum(losses[-100:]) / len(losses[-100:]) if losses else None,
        },
        "records": records,
        "checkpoints": checkpoint_receipts,
        "primary_checkpoint": checkpoint_receipts[str(maximum_steps)],
        "evaluation_c_touched": False,
        "read_once_scoring_spent": False,
    }
    write_json(args.output_summary, summary)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
