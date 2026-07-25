"""Train the locked full-block Paper Two T1-lite lineage with resumable stages."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import shutil
import sys
from functools import partial
from pathlib import Path
from typing import Any, Iterable

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from eval.eval_identity import model_load_kwargs
from training.internal_think_token_runtime import (
    install_internal_control_tokens,
    mask_internal_control_logits,
    split_internal_control_token_rows,
)
from training.internal_think_token_spec import INTERNAL_CONTROL_TOKENS
from training.internal_think_token_t1 import (
    augment_control_row,
    build_candidate_trie_contract,
    gather_control_examples,
    locate_readout_positions,
    stage_boundary_liveness_verdict,
)
from training.internal_think_token_t1_spec import phase_t1_locked, validate_locked_phase_t1
from training.internal_think_token_t1_r_spec import (
    STAGE_CHECKPOINT_STEPS,
    phase_t1_lite_r_locked,
    validate_phase_t1_lite_r_locked,
)
from training.run_internal_think_token_p0_cell import (
    PilotDataset,
    assert_loop_completion_alignment,
    candidate_values_from_rows,
    collate_pilot,
    evaluate_pilot,
    read_jsonl,
    tensor_sha256,
)
from training.stability import assert_finite_trainable_gradients, assert_finite_trainable_parameters
from training.train_unfrozen_recurrent import (
    build_optimizer,
    evaluate_loop1_canary,
    prepare_wrapper,
    trainable_parameter_summary,
)


STAGES = (
    {"name": "primitive_depth1", "support": (1,), "steps": 500, "lr": 2e-5, "prelude_multiplier": 1.0},
    {"name": "chain_depth_le2", "support": (1, 2), "steps": 2000, "lr": 1e-5, "prelude_multiplier": 10.0},
    {"name": "chain_depth_le4", "support": (1, 2, 3, 4), "steps": 4000, "lr": 1e-5, "prelude_multiplier": 10.0},
    {"name": "chain_depth_le8", "support": tuple(range(1, 9)), "steps": 2000, "lr": 1e-5, "prelude_multiplier": 10.0},
    {"name": "chain_depth_le8_dose", "support": tuple(range(1, 9)), "steps": 2000, "lr": 1e-5, "prelude_multiplier": 10.0},
)
BOUNDARIES = {500, 2500, 6500, 8500}


class DeviceEMA:
    """EMA in each parameter's native device/dtype; avoids per-step CPU copies."""

    def __init__(self, named_parameters: Iterable[tuple[str, torch.nn.Parameter]], *, decay: float) -> None:
        self.decay = float(decay)
        if not 0.0 < self.decay < 1.0:
            raise ValueError("EMA decay must be in (0, 1)")
        self.shadow = {
            name: parameter.detach().clone()
            for name, parameter in named_parameters
            if parameter.requires_grad
        }
        if not self.shadow:
            raise ValueError("EMA requires trainable parameters")

    @torch.no_grad()
    def update(self, named_parameters: Iterable[tuple[str, torch.nn.Parameter]]) -> None:
        current = dict(named_parameters)
        for name, shadow in self.shadow.items():
            value = current[name].detach().to(device=shadow.device, dtype=shadow.dtype)
            shadow.mul_(self.decay).add_(value, alpha=1.0 - self.decay)

    @torch.no_grad()
    def copy_to(self, named_parameters: Iterable[tuple[str, torch.nn.Parameter]]) -> dict[str, torch.Tensor]:
        current = dict(named_parameters)
        backup = {name: current[name].detach().clone() for name in self.shadow}
        for name, shadow in self.shadow.items():
            current[name].copy_(shadow.to(device=current[name].device, dtype=current[name].dtype))
        return backup

    @staticmethod
    @torch.no_grad()
    def restore(named_parameters: Iterable[tuple[str, torch.nn.Parameter]], backup: dict[str, torch.Tensor]) -> None:
        current = dict(named_parameters)
        for name, value in backup.items():
            current[name].copy_(value.to(device=current[name].device, dtype=current[name].dtype))

    def state_dict(self) -> dict[str, Any]:
        return {
            "decay": self.decay,
            "shadow": {name: value.detach().cpu() for name, value in self.shadow.items()},
        }

    @torch.no_grad()
    def reset_from_parameters(
        self,
        named_parameters: Iterable[tuple[str, torch.nn.Parameter]],
    ) -> None:
        current = dict(named_parameters)
        if set(current).issuperset(self.shadow) is False:
            missing = sorted(set(self.shadow) - set(current))
            raise RuntimeError(f"EMA reset is missing parameters: {missing[:8]}")
        self.shadow = {
            name: current[name].detach().clone().to(device=value.device, dtype=value.dtype)
            for name, value in self.shadow.items()
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        if float(state["decay"]) != self.decay or set(state["shadow"]) != set(self.shadow):
            raise RuntimeError("EMA contract changed while resuming T1-lite")
        self.shadow = {
            name: value.to(device=self.shadow[name].device, dtype=self.shadow[name].dtype)
            for name, value in state["shadow"].items()
        }


def write_json(path: str | Path, payload: Any) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _cpu_state(state: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {name: value.detach().cpu().clone() for name, value in state.items()}


def _atomic_torch_save(payload: dict[str, Any], destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".tmp")
    if temporary.exists():
        temporary.unlink()
    torch.save(payload, temporary)
    os.replace(temporary, destination)


def _atomic_copy(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".tmp")
    if temporary.exists():
        temporary.unlink()
    shutil.copy2(source, temporary)
    os.replace(temporary, destination)


def write_stage_checkpoint_bundle(
    *,
    local_dir: Path,
    backup_dir: Path | None,
    step: int,
    raw_state: dict[str, torch.Tensor],
    continuous_ema_state: dict[str, torch.Tensor],
    stage_reset_ema_state: dict[str, torch.Tensor],
) -> dict[str, Any]:
    """Atomically persist and hash all registered state variants at one boundary."""

    states = {
        "raw": raw_state,
        "continuous_ema": continuous_ema_state,
        "stage_reset_ema": stage_reset_ema_state,
    }
    receipt: dict[str, Any] = {"step": int(step), "states": {}}
    for label, state in states.items():
        filename = f"t1_lite_r_step_{int(step)}_{label}.pt"
        local_path = local_dir / filename
        _atomic_torch_save(
            {
                "kind": "paper2_t1_lite_r_stage_state",
                "step": int(step),
                "state": label,
                "trainable_state_dict": _cpu_state(state),
            },
            local_path,
        )
        local_sha = sha256_file(local_path)
        state_receipt: dict[str, Any] = {
            "local_path": str(local_path),
            "local_sha256": local_sha,
        }
        if backup_dir is not None:
            backup_path = backup_dir / "stage_states" / filename
            _atomic_copy(local_path, backup_path)
            state_receipt.update(
                {
                    "backup_path": str(backup_path),
                    "backup_sha256": sha256_file(backup_path),
                }
            )
            if state_receipt["backup_sha256"] != local_sha:
                raise RuntimeError(f"stage-state backup hash mismatch for {filename}")
        receipt["states"][label] = state_receipt
    return receipt


def verify_stage_checkpoint_manifest(
    *,
    receipts: list[dict[str, Any]],
    required_steps: Iterable[int] = STAGE_CHECKPOINT_STEPS,
    require_backup: bool,
) -> dict[str, Any]:
    """Verify every raw and shadow state before the replication can be scored."""

    by_step = {int(receipt["step"]): receipt for receipt in receipts}
    missing: list[str] = []
    mismatched: list[str] = []
    expected_states = ("raw", "continuous_ema", "stage_reset_ema")
    for step in (int(value) for value in required_steps):
        receipt = by_step.get(step)
        if receipt is None:
            missing.append(f"step_{step}")
            continue
        for label in expected_states:
            state = receipt.get("states", {}).get(label)
            if state is None:
                missing.append(f"step_{step}:{label}")
                continue
            local = Path(state["local_path"])
            if not local.exists():
                missing.append(str(local))
            elif sha256_file(local) != state["local_sha256"]:
                mismatched.append(str(local))
            if require_backup:
                backup_value = state.get("backup_path")
                if not backup_value or not Path(backup_value).exists():
                    missing.append(str(backup_value or f"step_{step}:{label}:backup"))
                elif sha256_file(backup_value) != state.get("backup_sha256"):
                    mismatched.append(str(backup_value))
    complete = not missing and not mismatched
    manifest = {
        "kind": "paper2_t1_lite_r_stage_checkpoint_manifest",
        "required_steps": [int(value) for value in required_steps],
        "required_states": list(expected_states),
        "receipts": receipts,
        "missing": missing,
        "hash_mismatches": mismatched,
        "complete": complete,
    }
    if not complete:
        raise RuntimeError(f"T1-lite-R stage checkpoint manifest incomplete: {manifest}")
    return manifest


def seed_all(seed: int) -> None:
    random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def frozen_base_sha256(wrapper: Any) -> str:
    """Hash only frozen pretrained tensors; trained block and new controls are excluded."""

    digest = hashlib.sha256()
    for name, parameter in wrapper.base_model.named_parameters():
        if parameter.requires_grad or name.endswith("control_rows"):
            continue
        tensor = parameter.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(tuple(tensor.shape)).encode("ascii"))
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(tensor.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def stage_for_step(global_step: int) -> tuple[int, dict[str, Any], int, int]:
    start = 0
    for index, stage in enumerate(STAGES):
        end = start + int(stage["steps"])
        if start < int(global_step) <= end:
            return index, stage, int(global_step) - start, start
        start = end
    raise ValueError(f"global step {global_step} lies outside the 10,500-step curriculum")


def deterministic_stage_indices(length: int, *, seed: int, stage_index: int) -> list[int]:
    if length <= 0:
        raise ValueError("stage dataset is empty")
    values = list(range(length))
    random.Random(int(seed) + 1009 * int(stage_index)).shuffle(values)
    return values


def compact_trainable_state(wrapper: Any) -> dict[str, torch.Tensor]:
    return {
        name: parameter.detach().cpu()
        for name, parameter in wrapper.named_parameters()
        if parameter.requires_grad
    }


def restore_trainable_state(wrapper: Any, state: dict[str, torch.Tensor]) -> None:
    current = dict(wrapper.named_parameters())
    if set(state) != {name for name, value in current.items() if value.requires_grad}:
        missing = sorted({name for name, value in current.items() if value.requires_grad} - set(state))
        extra = sorted(set(state) - set(current))
        raise RuntimeError(f"T1 resume trainable-key mismatch missing={missing[:8]} extra={extra[:8]}")
    with torch.no_grad():
        for name, value in state.items():
            parameter = current[name]
            if tuple(value.shape) != tuple(parameter.shape):
                raise RuntimeError(f"T1 resume shape mismatch for {name}")
            parameter.copy_(value.to(device=parameter.device, dtype=parameter.dtype))


def save_progress(
    path: Path,
    *,
    wrapper: Any,
    optimizer: Any,
    ema: DeviceEMA,
    stage_reset_ema: DeviceEMA | None,
    global_step: int,
    trace: list[dict[str, Any]],
    control_history: list[dict[str, Any]],
    stage_receipts: list[dict[str, Any]],
    stage_state_receipts: list[dict[str, Any]],
    token_receipt: dict[str, Any],
    contract_name: str,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "kind": f"paper2_{contract_name}_progress",
            "global_step": int(global_step),
            "trainable_state_dict": compact_trainable_state(wrapper),
            "optimizer_state_dict": optimizer.state_dict(),
            "ema_state_dict": ema.state_dict(),
            "stage_reset_ema_state_dict": (
                stage_reset_ema.state_dict() if stage_reset_ema is not None else None
            ),
            "trace": trace,
            "control_history": control_history,
            "stage_receipts": stage_receipts,
            "stage_state_receipts": stage_state_receipts,
            "control_token_resize": token_receipt,
        },
        path,
    )


def latest_progress(local_dir: Path, backup_dir: Path | None) -> Path | None:
    candidates = list(local_dir.glob("t1_progress_step_*.pt"))
    if backup_dir is not None and backup_dir.exists():
        candidates.extend(backup_dir.glob("t1_progress_step_*.pt"))
    if not candidates:
        return None
    return max(candidates, key=lambda path: int(path.stem.rsplit("_", 1)[-1]))


def set_stage_learning_rates(optimizer: Any, *, lr: float, prelude_multiplier: float) -> None:
    groups = optimizer.param_groups
    if len(groups) != 2:
        raise RuntimeError("T1-lite requires AdamW rest and split-prelude parameter groups")
    groups[0]["lr"] = float(lr)
    groups[1]["lr"] = float(lr) * float(prelude_multiplier)


def trainable_contract(wrapper: Any, control_rows: torch.nn.Parameter) -> dict[str, Any]:
    unexpected = []
    allowed_prefixes = ("base_model.model.layers.", "bridge.")
    for name, parameter in wrapper.named_parameters():
        if not parameter.requires_grad:
            continue
        if parameter is control_rows:
            continue
        if not name.startswith(allowed_prefixes):
            unexpected.append(name)
    if unexpected:
        raise RuntimeError(f"T1-lite selected unregistered trainable tensors: {unexpected[:12]}")
    summary = trainable_parameter_summary(wrapper)
    summary["control_token_rows"] = int(control_rows.numel())
    summary["total_with_control_rows"] = int(summary["total"])
    return summary


@torch.no_grad()
def one_loop_identity(wrapper: Any, tokenizer: Any, device: str) -> dict[str, Any]:
    encoded = tokenizer("A -> B\nStart value: A\nAnswer:", return_tensors="pt")
    encoded = {key: value.to(device) for key, value in encoded.items()}
    base = wrapper(**encoded, max_loops=1, force_base_model=True, use_cache=False, return_dict=True).logits
    recurrent = wrapper(**encoded, max_loops=1, use_cache=False, return_dict=True).logits
    maximum = float((base.float() - recurrent.float()).abs().max().item())
    return {"max_abs_logit_difference": maximum, "threshold": 1e-3, "passed": maximum < 1e-3}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train_jsonl", required=True)
    parser.add_argument("--pilot_jsonl", required=True)
    parser.add_argument("--canary_jsonl", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--backup_dir")
    parser.add_argument("--model_name", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--attn_implementation", default="default")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--registered_contract",
        choices=("t1_lite", "t1_lite_r"),
        default="t1_lite",
    )
    args = parser.parse_args()

    if args.registered_contract == "t1_lite_r":
        prereg = phase_t1_lite_r_locked()
        validate_phase_t1_lite_r_locked(prereg)
    else:
        prereg = phase_t1_locked()
        validate_locked_phase_t1(prereg)
    if int(args.seed) != int(prereg["replication"]["primary_seed"]):
        raise ValueError(
            f"{args.registered_contract} requires registered seed "
            f"{prereg['replication']['primary_seed']}"
        )
    seed_all(args.seed)
    output_dir = Path(args.output_dir)
    backup_dir = Path(args.backup_dir) if args.backup_dir else None
    output_dir.mkdir(parents=True, exist_ok=True)
    if backup_dir is not None:
        backup_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        **model_load_kwargs(args.dtype, args.attn_implementation),
    ).to(args.device)
    resize = install_internal_control_tokens(tokenizer, model)
    split = split_internal_control_token_rows(model, original_vocab_size=resize.original_vocab_size)
    config = {
        "layer_split": "6,18",
        "initial_halt_prob": 0.15,
        "bridge_projection_mode": "split",
        "adapter_dtype": "float32",
        "training_mode": "full_block",
        "resume_lora": {"enabled": False},
        "merge_lora_before_unfreeze": False,
        "train_auxiliary": {"bridge": True, "halting": False, "reentry_adapter": False, "latent": False},
        "optimizer": "adamw",
        "learning_rate": 2e-5,
        "weight_decay": 0.0,
        "bridge_prelude_lr_multiplier": 1.0,
        "bridge_prelude_weight_decay": 0.0,
    }
    wrapper, setup = prepare_wrapper(model, config, device=args.device)
    control_rows = wrapper.base_model.get_input_embeddings().control_rows
    control_rows.requires_grad_(True)
    contract = trainable_contract(wrapper, control_rows)
    optimizer = build_optimizer(wrapper, config)
    ema = DeviceEMA(wrapper.named_parameters(), decay=0.999)
    stage_reset_ema = (
        DeviceEMA(wrapper.named_parameters(), decay=0.999)
        if args.registered_contract == "t1_lite_r"
        else None
    )
    assert_finite_trainable_parameters(wrapper, step=0)
    identity = one_loop_identity(wrapper.eval(), tokenizer, args.device)
    if not identity["passed"]:
        raise RuntimeError(f"T1-lite one-loop identity failed: {identity}")
    adversarial = torch.zeros(len(tokenizer), device=args.device)
    adversarial[list(resize.control_token_ids)] = 1e4
    if int(mask_internal_control_logits(adversarial, resize.control_token_ids).argmax()) in resize.control_token_ids:
        raise RuntimeError("visible-generation control-token masking failed")

    source_rows = read_jsonl(args.train_jsonl)
    pilot_rows = read_jsonl(args.pilot_jsonl)
    assert_loop_completion_alignment(source_rows, tokenizer, max_length=512)
    assert_loop_completion_alignment(pilot_rows, tokenizer, max_length=512)
    datasets: list[PilotDataset] = []
    stage_paths: list[Path] = []
    for stage in STAGES:
        path = output_dir / "data" / f"{stage['name']}.jsonl"
        rows = [row for row in source_rows if int(row["depth"]) in set(stage["support"])]
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("".join(json.dumps(row, ensure_ascii=True) + "\n" for row in rows), encoding="utf-8")
        stage_paths.append(path)
        datasets.append(PilotDataset(path, tokenizer, max_length=512, max_loops=8))
    pilot_dataset = PilotDataset(args.pilot_jsonl, tokenizer, max_length=512, max_loops=8)
    pilot_loaders: list[DataLoader] = []
    for stage in STAGES:
        pilot_path = output_dir / "data" / f"{stage['name']}_pilot.jsonl"
        stage_pilot_rows = [row for row in pilot_rows if int(row["depth"]) in set(stage["support"])]
        pilot_path.write_text(
            "".join(json.dumps(row, ensure_ascii=True) + "\n" for row in stage_pilot_rows),
            encoding="utf-8",
        )
        stage_pilot_dataset = PilotDataset(pilot_path, tokenizer, max_length=512, max_loops=8)
        pilot_loaders.append(
            DataLoader(
                stage_pilot_dataset,
                batch_size=8,
                shuffle=False,
                collate_fn=partial(collate_pilot, pad_token_id=tokenizer.pad_token_id),
            )
        )
    candidate_values = candidate_values_from_rows(pilot_dataset.base.rows)
    candidate_contract = build_candidate_trie_contract(
        tokenizer,
        prompt=str(pilot_dataset.base.rows[0]["prompt"]),
        candidate_values=candidate_values,
    )
    if any(len(tokens) != 1 for tokens in candidate_contract.candidate_token_ids):
        raise AssertionError("T1-lite registered letter candidates must each be one token")
    continue_id, stop_id, readout_id = (int(value) for value in resize.control_token_ids)
    frozen_hash_start = frozen_base_sha256(wrapper)
    old_embedding_hash_start = tensor_sha256(wrapper.base_model.get_input_embeddings().old_weight)

    trace: list[dict[str, Any]] = []
    control_history: list[dict[str, Any]] = []
    stage_receipts: list[dict[str, Any]] = []
    stage_state_receipts: list[dict[str, Any]] = []
    global_step = 0
    progress = latest_progress(output_dir / "checkpoints", backup_dir)
    if progress is not None:
        payload = torch.load(progress, map_location="cpu")
        restore_trainable_state(wrapper, payload["trainable_state_dict"])
        optimizer.load_state_dict(payload["optimizer_state_dict"])
        for state in optimizer.state.values():
            for key, value in state.items():
                if torch.is_tensor(value):
                    state[key] = value.to(args.device)
        ema.load_state_dict(payload["ema_state_dict"])
        if stage_reset_ema is not None:
            state = payload.get("stage_reset_ema_state_dict")
            if state is None:
                raise RuntimeError("T1-lite-R resume lacks stage-reset EMA state")
            stage_reset_ema.load_state_dict(state)
        global_step = int(payload["global_step"])
        trace = list(payload.get("trace") or [])
        control_history = list(payload.get("control_history") or [])
        stage_receipts = list(payload.get("stage_receipts") or [])
        stage_state_receipts = list(payload.get("stage_state_receipts") or [])
        print(f"resumed_t1_lite_progress={progress} global_step={global_step}", flush=True)

    total_steps = sum(int(stage["steps"]) for stage in STAGES)
    previous_stage_index = stage_for_step(global_step)[0] if global_step else None
    for next_step in range(global_step + 1, total_steps + 1):
        stage_index, stage, local_step, stage_start = stage_for_step(next_step)
        if stage_reset_ema is not None and stage_index != previous_stage_index:
            stage_reset_ema.reset_from_parameters(wrapper.named_parameters())
        previous_stage_index = stage_index
        set_stage_learning_rates(
            optimizer,
            lr=float(stage["lr"]),
            prelude_multiplier=float(stage["prelude_multiplier"]),
        )
        dataset = datasets[stage_index]
        ordering = deterministic_stage_indices(len(dataset), seed=args.seed, stage_index=stage_index)
        item_index = ordering[(local_step - 1) % len(ordering)]
        batch = collate_pilot([dataset[item_index]], pad_token_id=tokenizer.pad_token_id)
        batch = {key: value.to(args.device) for key, value in batch.items()}
        wrapper.train()
        output = wrapper(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            labels=batch["labels"],
            loop_labels=batch["loop_labels"],
            target_loop_counts=batch["target_loop_counts"],
            max_loops=max(stage["support"]),
            use_cache=False,
            return_dict=True,
            return_loop_logits=True,
            loop_loss_mode="per_loop_labels",
            beta=0.0,
            halt_target_nll_weight=0.0,
        )
        if output.loss is None or output.loop_logits is None:
            raise AssertionError("T1-lite requires mechanism loss and loop logits")
        positions = locate_readout_positions(
            batch["input_ids"], readout_token_id=readout_id, control_active=batch["control_active"]
        )
        if bool(batch["control_active"].any()):
            control_logits, targets, _, _ = gather_control_examples(
                output.loop_logits,
                readout_positions=positions,
                required_depths=batch["required_depth"],
                control_active=batch["control_active"],
                continue_token_id=continue_id,
                stop_token_id=stop_id,
            )
            control_loss = F.cross_entropy(control_logits.float(), targets)
        else:
            control_loss = output.loss.new_zeros(())
        total_loss = output.loss + 0.5 * control_loss
        if not bool(torch.isfinite(total_loss)):
            raise FloatingPointError(f"nonfinite T1-lite loss at step {next_step}")
        optimizer.zero_grad(set_to_none=True)
        total_loss.backward()
        assert_finite_trainable_gradients(wrapper, next_step)
        torch.nn.utils.clip_grad_norm_(
            [parameter for parameter in wrapper.parameters() if parameter.requires_grad], 0.5
        )
        optimizer.step()
        ema.update(wrapper.named_parameters())
        if stage_reset_ema is not None:
            stage_reset_ema.update(wrapper.named_parameters())
        if bool(batch["control_active"].item()):
            control_history.append(
                {
                    "global_step": next_step,
                    "stage": stage["name"],
                    "control_loss": float(control_loss.detach().cpu()),
                }
            )
        if next_step == 1 or next_step % 100 == 0:
            record = {
                "global_step": next_step,
                "stage": stage["name"],
                "stage_step": local_step,
                "mechanism_loss": float(output.loss.detach().cpu()),
                "control_loss": float(control_loss.detach().cpu()),
                "total_loss": float(total_loss.detach().cpu()),
                "control_active": bool(batch["control_active"].item()),
            }
            trace.append(record)
            print(json.dumps(record, sort_keys=True), flush=True)

        if next_step % 500 == 0 or next_step == total_steps:
            checkpoint = output_dir / "checkpoints" / f"t1_progress_step_{next_step}.pt"
            save_progress(
                checkpoint,
                wrapper=wrapper,
                optimizer=optimizer,
                ema=ema,
                stage_reset_ema=stage_reset_ema,
                global_step=next_step,
                trace=trace,
                control_history=control_history,
                stage_receipts=stage_receipts,
                stage_state_receipts=stage_state_receipts,
                token_receipt=resize.to_dict(),
                contract_name=args.registered_contract,
            )
            if backup_dir is not None:
                shutil.copy2(checkpoint, backup_dir / checkpoint.name)
            keep_steps = BOUNDARIES | {total_steps, next_step}
            for directory in (output_dir / "checkpoints", backup_dir):
                if directory is None or not directory.exists():
                    continue
                for prior in directory.glob("t1_progress_step_*.pt"):
                    prior_step = int(prior.stem.rsplit("_", 1)[-1])
                    if prior_step not in keep_steps:
                        prior.unlink()

        if next_step in BOUNDARIES:
            metrics = evaluate_pilot(
                wrapper,
                pilot_loaders[stage_index],
                readout_token_id=readout_id,
                continue_token_id=continue_id,
                stop_token_id=stop_id,
                candidate_contract=candidate_contract,
                pad_token_id=tokenizer.pad_token_id,
                device=args.device,
                max_loops=max(stage["support"]),
            )
            stage_points = [
                (int(row["global_step"]), float(row["control_loss"]))
                for row in control_history
                if row["stage"] == stage["name"]
            ]
            liveness = stage_boundary_liveness_verdict(
                stage_points,
                stop_correct=int(metrics["stop_correct"]),
                stop_total=int(metrics["stop_total"]),
            )
            canary = evaluate_loop1_canary(
                wrapper,
                tokenizer,
                data_jsonl=args.canary_jsonl,
                device=args.device,
                value_prefix="name:",
            )
            receipt = {
                "global_step": next_step,
                "stage": stage["name"],
                "trained_depths": list(stage["support"]),
                "pilot": metrics,
                "liveness": liveness,
                "tier1_canary": canary,
            }
            stage_receipts.append(receipt)
            write_json(output_dir / "stage_receipts" / f"step_{next_step}.json", receipt)
            print(json.dumps({"stage_boundary": receipt}, sort_keys=True), flush=True)
            if stage_reset_ema is not None:
                stage_state_receipts.append(
                    write_stage_checkpoint_bundle(
                        local_dir=output_dir / "stage_states",
                        backup_dir=backup_dir,
                        step=next_step,
                        raw_state=compact_trainable_state(wrapper),
                        continuous_ema_state=ema.state_dict()["shadow"],
                        stage_reset_ema_state=stage_reset_ema.state_dict()["shadow"],
                    )
                )
            # Persist the completed boundary readout into the resumable state. A
            # runtime loss after evaluation must not silently discard the gate.
            checkpoint = output_dir / "checkpoints" / f"t1_progress_step_{next_step}.pt"
            save_progress(
                checkpoint,
                wrapper=wrapper,
                optimizer=optimizer,
                ema=ema,
                stage_reset_ema=stage_reset_ema,
                global_step=next_step,
                trace=trace,
                control_history=control_history,
                stage_receipts=stage_receipts,
                stage_state_receipts=stage_state_receipts,
                token_receipt=resize.to_dict(),
                contract_name=args.registered_contract,
            )
            if backup_dir is not None:
                shutil.copy2(checkpoint, backup_dir / checkpoint.name)
            if int(canary["correct"]) < 60:
                write_json(output_dir / "summary.json", {"status": "hard_stopped_tier1_canary", **receipt})
                return 2
            if liveness["abort_for_diagnosis"]:
                write_json(
                    output_dir / "summary.json",
                    {"status": "aborted_liveness_attempt_not_consumed", **receipt},
                )
                return 2

    stage_manifest: dict[str, Any] | None = None
    if stage_reset_ema is not None:
        if not any(int(receipt["step"]) == total_steps for receipt in stage_state_receipts):
            stage_state_receipts.append(
                write_stage_checkpoint_bundle(
                    local_dir=output_dir / "stage_states",
                    backup_dir=backup_dir,
                    step=total_steps,
                    raw_state=compact_trainable_state(wrapper),
                    continuous_ema_state=ema.state_dict()["shadow"],
                    stage_reset_ema_state=stage_reset_ema.state_dict()["shadow"],
                )
            )
        stage_manifest = verify_stage_checkpoint_manifest(
            receipts=stage_state_receipts,
            required_steps=STAGE_CHECKPOINT_STEPS,
            require_backup=backup_dir is not None,
        )
        stage_manifest_path = output_dir / "stage_checkpoint_manifest.json"
        write_json(stage_manifest_path, stage_manifest)
        if backup_dir is not None:
            _atomic_copy(stage_manifest_path, backup_dir / stage_manifest_path.name)

    checkpoint_prefix = "t1_lite_r" if args.registered_contract == "t1_lite_r" else "t1_lite"
    raw_path = output_dir / f"{checkpoint_prefix}_raw_step_10500.pt"
    torch.save(
        {
            "kind": f"paper2_{checkpoint_prefix}_final_raw",
            "step": total_steps,
            "trainable_state_dict": compact_trainable_state(wrapper),
            "control_token_resize": resize.to_dict(),
            "split_control_rows": split.to_dict(),
            "setup": setup,
        },
        raw_path,
    )
    backup = ema.copy_to(wrapper.named_parameters())
    ema_path = output_dir / (
        "t1_lite_ema_step_10500.pt"
        if args.registered_contract == "t1_lite"
        else "t1_lite_r_continuous_ema_step_10500.pt"
    )
    torch.save(
        {
            "kind": (
                "paper2_t1_lite_final_ema"
                if args.registered_contract == "t1_lite"
                else "paper2_t1_lite_r_final_continuous_ema"
            ),
            "step": total_steps,
            "trainable_state_dict": compact_trainable_state(wrapper),
            "control_token_resize": resize.to_dict(),
            "split_control_rows": split.to_dict(),
            "setup": setup,
        },
        ema_path,
    )
    ema.restore(wrapper.named_parameters(), backup)
    stage_reset_ema_path: Path | None = None
    if stage_reset_ema is not None:
        stage_reset_backup = stage_reset_ema.copy_to(wrapper.named_parameters())
        stage_reset_ema_path = output_dir / f"{checkpoint_prefix}_stage_reset_ema_step_10500.pt"
        torch.save(
            {
                "kind": f"paper2_{checkpoint_prefix}_final_stage_reset_ema",
                "step": total_steps,
                "trainable_state_dict": compact_trainable_state(wrapper),
                "control_token_resize": resize.to_dict(),
                "split_control_rows": split.to_dict(),
                "setup": setup,
            },
            stage_reset_ema_path,
        )
        stage_reset_ema.restore(wrapper.named_parameters(), stage_reset_backup)
    if backup_dir is not None:
        shutil.copy2(raw_path, backup_dir / raw_path.name)
        shutil.copy2(ema_path, backup_dir / ema_path.name)
        if stage_reset_ema_path is not None:
            shutil.copy2(stage_reset_ema_path, backup_dir / stage_reset_ema_path.name)
    frozen_hash_end = frozen_base_sha256(wrapper)
    old_embedding_hash_end = tensor_sha256(wrapper.base_model.get_input_embeddings().old_weight)
    if frozen_hash_end != frozen_hash_start or old_embedding_hash_end != old_embedding_hash_start:
        raise RuntimeError("T1-lite changed a frozen pretrained tensor")
    summary = {
        "kind": f"paper2_{checkpoint_prefix}_training",
        "status": "training_finished",
        "seed": args.seed,
        "steps": total_steps,
        "preregistration": prereg,
        "identity": identity,
        "control_token_resize": resize.to_dict(),
        "split_control_rows": split.to_dict(),
        "trainable_contract": contract,
        "frozen_base_sha256_start": frozen_hash_start,
        "frozen_base_sha256_end": frozen_hash_end,
        "frozen_base_unchanged": True,
        "old_embedding_sha256_start": old_embedding_hash_start,
        "old_embedding_sha256_end": old_embedding_hash_end,
        "raw_checkpoint": str(raw_path),
        "raw_checkpoint_sha256": sha256_file(raw_path),
        "ema_checkpoint": str(ema_path),
        "ema_checkpoint_sha256": sha256_file(ema_path),
        "continuous_ema_checkpoint": str(ema_path),
        "continuous_ema_checkpoint_sha256": sha256_file(ema_path),
        "stage_reset_ema_checkpoint": (
            str(stage_reset_ema_path) if stage_reset_ema_path is not None else None
        ),
        "stage_reset_ema_checkpoint_sha256": (
            sha256_file(stage_reset_ema_path) if stage_reset_ema_path is not None else None
        ),
        "stage_checkpoint_manifest": stage_manifest,
        "stage_receipts": stage_receipts,
        "trace": trace,
        "control_history": control_history,
    }
    write_json(output_dir / "training_summary.json", summary)
    print(
        json.dumps(
            {
                key: summary[key]
                for key in (
                    "status",
                    "steps",
                    "raw_checkpoint",
                    "continuous_ema_checkpoint",
                    "stage_reset_ema_checkpoint",
                )
            },
            indent=2,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
