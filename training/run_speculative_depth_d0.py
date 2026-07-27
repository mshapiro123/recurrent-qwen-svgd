"""Train the locked 4,000-step D0 adaptive-depth continuation."""

from __future__ import annotations

import argparse
import bisect
import json
import os
import random
import shutil
import sys
from functools import partial
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from eval.eval_identity import model_load_kwargs
from eval.eval_internal_think_token_t1_lite import restore_checkpoint
from eval.eval_speculative_depth_d0_floor import load_partition_cache
from training.internal_think_token_runtime import install_internal_control_tokens, split_internal_control_token_rows
from training.internal_think_token_spec import INTERNAL_CONTROL_TOKENS
from training.internal_think_token_t1 import (
    build_candidate_trie_contract,
    gather_control_examples,
    locate_readout_positions,
    stage_boundary_liveness_verdict,
)
from training.run_internal_think_token_p0_cell import (
    PilotDataset,
    candidate_values_from_rows,
    collate_pilot,
    evaluate_pilot,
    read_jsonl,
)
from training.run_internal_think_token_t1_lite import (
    DeviceEMA,
    compact_trainable_state,
    frozen_base_sha256,
    restore_trainable_state,
)
from training.speculative_depth_d0_corpus import sha256_file
from training.speculative_depth_d0_postlock import (
    build_training_schedule,
    predict_isotonic,
    validate_cache_summary,
)
from training.speculative_depth_d0_spec import (
    D0ExecutionPolicy,
    DRAFTER_CHECKPOINT_SHA256,
    DRAFTER_MODEL,
    DRAFTER_MODEL_REVISION,
    dynamic_depth_target,
    validate_locked_d0,
)
from training.stability import assert_finite_trainable_gradients, assert_finite_trainable_parameters
from training.train_unfrozen_recurrent import build_optimizer, prepare_wrapper


def read_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path: str | Path, payload: Any) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, destination)


def atomic_torch_save(payload: dict[str, Any], destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, destination)


def verified_copy(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".partial")
    shutil.copy2(source, temporary)
    if sha256_file(source) != sha256_file(temporary):
        temporary.unlink(missing_ok=True)
        raise RuntimeError(f"D0 checkpoint backup hash mismatch: {destination}")
    os.replace(temporary, destination)


def locate_global_position(rows: list[dict[str, Any]], global_position: int) -> tuple[int, int]:
    offsets = [0]
    for row in rows:
        offsets.append(offsets[-1] + len(row["input_ids"]) - 1)
    row_index = bisect.bisect_right(offsets, int(global_position)) - 1
    if row_index < 0 or row_index >= len(rows):
        raise IndexError("D0 scheduled natural position is outside label-train")
    return row_index, int(global_position) - offsets[row_index]


def full_teacher_logits(row_cache: dict[str, Any], local_position: int) -> torch.Tensor:
    positions = row_cache["full_logit_local_positions"].tolist()
    if int(local_position) not in positions:
        raise RuntimeError("D0 scheduled position lacks its exact cached teacher logits")
    return row_cache["full_teacher_logits_bfloat16"][positions.index(int(local_position))]


def target_depth(
    *,
    floor: dict[str, Any],
    row_cache: dict[str, Any],
    local_position: int,
    answer_logits: torch.Tensor,
    teacher_token_id: int,
    original_vocab_size: int,
) -> int:
    branch = floor["calibration"]["branch"]
    if bool(row_cache["accepted"][local_position]):
        return 1
    if branch == "graded_floor_curve":
        return predict_isotonic(
            floor["calibration"]["disagreement_to_depth_mapping"]["primary_fit"],
            float(row_cache["teacher_to_plain_drafter_kl"][local_position]),
        )
    matches = (
        answer_logits[:, :original_vocab_size].detach().argmax(dim=-1).eq(int(teacher_token_id)).tolist()
    )
    return dynamic_depth_target(matches, max_depth=4)


def natural_loss(
    wrapper: Any,
    *,
    row: dict[str, Any],
    row_cache: dict[str, Any],
    local_position: int,
    floor: dict[str, Any],
    control_ids: tuple[int, int],
    original_vocab_size: int,
    device: str,
) -> tuple[torch.Tensor, dict[str, Any]]:
    prefix = row["input_ids"][: local_position + 1]
    input_ids = torch.tensor([prefix], dtype=torch.long, device=device)
    output = wrapper(
        input_ids=input_ids,
        attention_mask=torch.ones_like(input_ids),
        labels=None,
        max_loops=4,
        use_cache=False,
        return_dict=True,
        return_loop_logits=True,
    )
    if output.loop_logits is None:
        raise RuntimeError("D0 natural training requires loop logits")
    logits = output.loop_logits[0, 0, :, -1]
    teacher_id = int(row_cache["teacher_greedy_token_id"][local_position])
    selected_depth = target_depth(
        floor=floor,
        row_cache=row_cache,
        local_position=local_position,
        answer_logits=logits,
        teacher_token_id=teacher_id,
        original_vocab_size=original_vocab_size,
    )
    selected = logits[selected_depth - 1, :original_vocab_size].float()
    corpus_target = int(row_cache["target_token_id"][local_position])
    lm_loss = F.cross_entropy(selected.unsqueeze(0), torch.tensor([corpus_target], device=device))
    rejected = not bool(row_cache["accepted"][local_position])
    if rejected:
        teacher_logits = full_teacher_logits(row_cache, local_position).to(device=device).float()
        distill_loss = F.kl_div(
            torch.log_softmax(selected, dim=-1),
            torch.softmax(teacher_logits, dim=-1),
            reduction="sum",
        )
    else:
        distill_loss = lm_loss.new_zeros(())
    control_logits = logits[:selected_depth].index_select(
        -1, torch.tensor(control_ids, dtype=torch.long, device=device)
    ).float()
    controls = torch.tensor([0] * (selected_depth - 1) + [1], dtype=torch.long, device=device)
    control_loss = F.cross_entropy(control_logits, controls)
    total = lm_loss + distill_loss + 0.5 * control_loss
    return total, {
        "kind": "natural",
        "target_depth": selected_depth,
        "rejected": rejected,
        "lm_loss": float(lm_loss.detach().cpu()),
        "distill_loss": float(distill_loss.detach().cpu()),
        "control_loss": float(control_loss.detach().cpu()),
    }


def rehearsal_loss(
    wrapper: Any,
    *,
    batch: dict[str, torch.Tensor],
    continue_id: int,
    stop_id: int,
    readout_id: int,
) -> tuple[torch.Tensor, dict[str, Any]]:
    max_loops = int(batch["required_depth"].max().item())
    output = wrapper(
        input_ids=batch["input_ids"],
        attention_mask=batch["attention_mask"],
        labels=batch["labels"],
        loop_labels=batch["loop_labels"],
        target_loop_counts=batch["target_loop_counts"],
        max_loops=max_loops,
        use_cache=False,
        return_dict=True,
        return_loop_logits=True,
        loop_loss_mode="per_loop_labels",
        beta=0.0,
        halt_target_nll_weight=0.0,
    )
    if output.loss is None or output.loop_logits is None:
        raise RuntimeError("D0 rehearsal requires mechanism and control logits")
    positions = locate_readout_positions(
        batch["input_ids"], readout_token_id=readout_id, control_active=batch["control_active"]
    )
    control_logits, targets, _, _ = gather_control_examples(
        output.loop_logits,
        readout_positions=positions,
        required_depths=batch["required_depth"],
        control_active=batch["control_active"],
        continue_token_id=continue_id,
        stop_token_id=stop_id,
    )
    control_loss = F.cross_entropy(control_logits.float(), targets)
    total = output.loss + 0.5 * control_loss
    return total, {
        "kind": "rehearsal",
        "mechanism_loss": float(output.loss.detach().cpu()),
        "control_loss": float(control_loss.detach().cpu()),
    }


@torch.inference_mode()
def natural_stop_recall(
    wrapper: Any,
    *,
    calibration_rows: list[dict[str, Any]],
    floor_rows: list[dict[str, Any]],
    floor: dict[str, Any],
    continue_id: int,
    stop_id: int,
    original_vocab_size: int,
    device: str,
) -> tuple[int, int]:
    correct = 0
    sample = floor_rows[: min(128, len(floor_rows))]
    for receipt in sample:
        source = calibration_rows[int(receipt["row_index"])]
        local = int(receipt["local_position"])
        prefix = torch.tensor([source["input_ids"][: local + 1]], dtype=torch.long, device=device)
        output = wrapper(
            input_ids=prefix,
            attention_mask=torch.ones_like(prefix),
            labels=None,
            max_loops=4,
            use_cache=False,
            return_dict=True,
            return_loop_logits=True,
        )
        logits = output.loop_logits[0, 0, :, -1]
        if floor["calibration"]["branch"] == "graded_floor_curve":
            depth = predict_isotonic(
                floor["calibration"]["disagreement_to_depth_mapping"]["primary_fit"],
                float(receipt["kl"]),
            )
        else:
            depth = dynamic_depth_target(receipt["matches_teacher_7b"], max_depth=4)
        decision = logits[depth - 1, [continue_id, stop_id]].argmax().item()
        correct += int(decision == 1)
    return correct, len(sample)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preregistration", required=True)
    parser.add_argument("--label_train_jsonl", required=True)
    parser.add_argument("--calibration_jsonl", required=True)
    parser.add_argument("--teacher_cache_summary", required=True)
    parser.add_argument("--floor_summary", required=True)
    parser.add_argument("--floor_private_rows", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--rehearsal_jsonl", required=True)
    parser.add_argument("--rehearsal_pilot_jsonl", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--backup_dir", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--attn_implementation", default="sdpa")
    args = parser.parse_args()

    D0ExecutionPolicy(labeling_gpu_authorized=True, training_authorized=True).assert_allowed(
        labeling=False, training=True
    )
    prereg = read_json(args.preregistration)
    validate_locked_d0(prereg)
    if sha256_file(args.checkpoint) != DRAFTER_CHECKPOINT_SHA256:
        raise RuntimeError("D0 training received the wrong locked drafter checkpoint")
    cache_summary = read_json(args.teacher_cache_summary)
    validate_cache_summary(cache_summary)
    floor = read_json(args.floor_summary)
    if floor.get("status") != "complete":
        raise RuntimeError("D0 training requires a landed floor branch")
    floor_private = read_json(args.floor_private_rows)
    label_rows = read_jsonl(args.label_train_jsonl)
    calibration_rows = read_jsonl(args.calibration_jsonl)
    cache_rows = load_partition_cache(cache_summary, "teacher_7b", "label_train")
    natural_positions = sum(len(row["input_ids"]) - 1 for row in label_rows)
    schedule = build_training_schedule(total_steps=4000, natural_positions=natural_positions, seed=0)
    stored_schedule = read_json(Path(cache_summary["cache_root"]) / "registered_training_schedule.json")
    if stored_schedule["schedule"] != schedule:
        raise RuntimeError("D0 training schedule drifted after the teacher pass")

    random.seed(0)
    torch.manual_seed(0)
    tokenizer = AutoTokenizer.from_pretrained(DRAFTER_MODEL, revision=DRAFTER_MODEL_REVISION)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        DRAFTER_MODEL,
        revision=DRAFTER_MODEL_REVISION,
        **model_load_kwargs(args.dtype, args.attn_implementation),
    ).to(args.device)
    resize = install_internal_control_tokens(tokenizer, model)
    split_internal_control_token_rows(model, original_vocab_size=resize.original_vocab_size)
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
        "learning_rate": 1e-5,
        "weight_decay": 0.0,
        "bridge_prelude_lr_multiplier": 10.0,
        "bridge_prelude_weight_decay": 0.0,
    }
    wrapper, setup = prepare_wrapper(model, config, device=args.device)
    wrapper.base_model.get_input_embeddings().control_rows.requires_grad_(True)
    restore_checkpoint(wrapper, args.checkpoint)
    assert_finite_trainable_parameters(wrapper, step=0)
    optimizer = build_optimizer(wrapper, config)
    ema = DeviceEMA(wrapper.named_parameters(), decay=0.999)
    frozen_start = frozen_base_sha256(wrapper)
    continue_id, stop_id, readout_id = (int(value) for value in resize.control_token_ids)

    rehearsal_dataset = PilotDataset(args.rehearsal_jsonl, tokenizer, max_length=512, max_loops=8)
    rehearsal_pilot = PilotDataset(args.rehearsal_pilot_jsonl, tokenizer, max_length=512, max_loops=8)
    pilot_loader = DataLoader(
        rehearsal_pilot,
        batch_size=8,
        shuffle=False,
        collate_fn=partial(collate_pilot, pad_token_id=tokenizer.pad_token_id),
    )
    candidate_contract = build_candidate_trie_contract(
        tokenizer,
        prompt=str(rehearsal_pilot.base.rows[0]["prompt"]),
        candidate_values=candidate_values_from_rows(rehearsal_pilot.base.rows),
    )
    rehearsal_order = list(range(len(rehearsal_dataset)))
    random.Random(0).shuffle(rehearsal_order)

    output_dir = Path(args.output_dir)
    backup_dir = Path(args.backup_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    backup_dir.mkdir(parents=True, exist_ok=True)
    final_backup_summary = backup_dir / "final_training_summary.json"
    if final_backup_summary.exists():
        completed = read_json(final_backup_summary)
        raw_backup = backup_dir / "d0_raw_step_4000.pt"
        ema_backup = backup_dir / "d0_ema_step_4000.pt"
        if (
            completed.get("status") == "training_finished"
            and raw_backup.exists()
            and ema_backup.exists()
            and sha256_file(raw_backup) == completed.get("raw_checkpoint_sha256")
            and sha256_file(ema_backup) == completed.get("ema_checkpoint_sha256")
        ):
            raw_path = output_dir / raw_backup.name
            ema_path = output_dir / ema_backup.name
            verified_copy(raw_backup, raw_path)
            verified_copy(ema_backup, ema_path)
            completed["raw_checkpoint"] = str(raw_path)
            completed["ema_checkpoint"] = str(ema_path)
            write_json(output_dir / "summary.json", completed)
            print(f"restored_completed_d0_training={final_backup_summary}", flush=True)
            return 0
    progress_candidates = sorted(backup_dir.glob("d0_progress_step_*.pt"), key=lambda p: int(p.stem.rsplit("_", 1)[-1]))
    trace: list[dict[str, Any]] = []
    global_step = 0
    if progress_candidates:
        progress = progress_candidates[-1]
        payload = torch.load(progress, map_location="cpu", weights_only=False)
        restore_trainable_state(wrapper, payload["trainable_state_dict"])
        optimizer.load_state_dict(payload["optimizer_state_dict"])
        for state in optimizer.state.values():
            for key, value in state.items():
                if torch.is_tensor(value):
                    state[key] = value.to(args.device)
        ema.load_state_dict(payload["ema_state_dict"])
        trace = list(payload["trace"])
        global_step = int(payload["global_step"])
        print(f"resumed_d0_training={progress} global_step={global_step}", flush=True)

    interval_start = global_step - (global_step % 1000) + 1
    for step in range(global_step + 1, 4001):
        item = schedule[step - 1]
        wrapper.train()
        if item["kind"] == "natural":
            row_index, local_position = locate_global_position(label_rows, int(item["position_index"]))
            loss, receipt = natural_loss(
                wrapper,
                row=label_rows[row_index],
                row_cache=cache_rows[row_index],
                local_position=local_position,
                floor=floor,
                control_ids=(continue_id, stop_id),
                original_vocab_size=resize.original_vocab_size,
                device=args.device,
            )
        else:
            index = rehearsal_order[int(item["rehearsal_index"]) % len(rehearsal_order)]
            batch = collate_pilot([rehearsal_dataset[index]], pad_token_id=tokenizer.pad_token_id)
            batch = {name: value.to(args.device) for name, value in batch.items()}
            loss, receipt = rehearsal_loss(
                wrapper,
                batch=batch,
                continue_id=continue_id,
                stop_id=stop_id,
                readout_id=readout_id,
            )
        if not bool(torch.isfinite(loss)):
            raise FloatingPointError(f"nonfinite D0 loss at step {step}")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        assert_finite_trainable_gradients(wrapper, step)
        torch.nn.utils.clip_grad_norm_([p for p in wrapper.parameters() if p.requires_grad], 0.5)
        optimizer.step()
        ema.update(wrapper.named_parameters())
        record = {"step": step, "total_loss": float(loss.detach().cpu()), **receipt}
        trace.append(record)
        if step == 1 or step % 100 == 0:
            print(json.dumps(record, sort_keys=True), flush=True)

        if step in (1000, 2000, 3000):
            wrapper.eval()
            mechanism = evaluate_pilot(
                wrapper,
                pilot_loader,
                readout_token_id=readout_id,
                continue_token_id=continue_id,
                stop_token_id=stop_id,
                candidate_contract=candidate_contract,
                pad_token_id=tokenizer.pad_token_id,
                device=args.device,
                max_loops=8,
            )
            stop_correct, stop_total = natural_stop_recall(
                wrapper,
                calibration_rows=calibration_rows,
                floor_rows=floor_private["rows"],
                floor=floor,
                continue_id=continue_id,
                stop_id=stop_id,
                original_vocab_size=resize.original_vocab_size,
                device=args.device,
            )
            points = [
                (int(row["step"]), float(row["control_loss"]))
                for row in trace
                if interval_start <= int(row["step"]) <= step and "control_loss" in row
            ]
            liveness = stage_boundary_liveness_verdict(
                points, stop_correct=stop_correct, stop_total=stop_total
            )
            guardrail = {"step": step, "mechanism": mechanism, "natural_liveness": liveness}
            write_json(output_dir / "guardrails" / f"step_{step}.json", guardrail)
            if liveness["abort_for_diagnosis"]:
                write_json(output_dir / "summary.json", {"status": "aborted_liveness", **guardrail})
                return 2
            interval_start = step + 1

        if step % 500 == 0:
            checkpoint = output_dir / "checkpoints" / f"d0_progress_step_{step}.pt"
            payload = {
                "kind": "paper2_d0_training_progress",
                "global_step": step,
                "trainable_state_dict": compact_trainable_state(wrapper),
                "optimizer_state_dict": optimizer.state_dict(),
                "ema_state_dict": ema.state_dict(),
                "trace": trace,
            }
            atomic_torch_save(payload, checkpoint)
            backup_checkpoint = backup_dir / checkpoint.name
            verified_copy(checkpoint, backup_checkpoint)
            for stale in backup_dir.glob("d0_progress_step_*.pt"):
                if stale != backup_checkpoint:
                    stale.unlink(missing_ok=True)

    raw_path = output_dir / "d0_raw_step_4000.pt"
    atomic_torch_save(
        {
            "kind": "paper2_d0_final_raw",
            "step": 4000,
            "trainable_state_dict": compact_trainable_state(wrapper),
            "setup": setup,
            "control_token_resize": resize.to_dict(),
        },
        raw_path,
    )
    backup = ema.copy_to(wrapper.named_parameters())
    ema_path = output_dir / "d0_ema_step_4000.pt"
    atomic_torch_save(
        {
            "kind": "paper2_d0_final_ema",
            "step": 4000,
            "trainable_state_dict": compact_trainable_state(wrapper),
            "setup": setup,
            "control_token_resize": resize.to_dict(),
        },
        ema_path,
    )
    ema.restore(wrapper.named_parameters(), backup)
    verified_copy(raw_path, backup_dir / raw_path.name)
    verified_copy(ema_path, backup_dir / ema_path.name)
    frozen_end = frozen_base_sha256(wrapper)
    if frozen_end != frozen_start:
        raise RuntimeError("D0 training changed a frozen pretrained tensor")
    summary = {
        "kind": "paper2_d0_training",
        "status": "training_finished",
        "steps": 4000,
        "seed": 0,
        "optimizer": "adamw",
        "learning_rate": 1e-5,
        "bridge_prelude_lr_multiplier": 10.0,
        "natural_steps": 2800,
        "rehearsal_steps": 1200,
        "floor_branch": floor["calibration"]["branch"],
        "raw_checkpoint": str(raw_path),
        "raw_checkpoint_sha256": sha256_file(raw_path),
        "ema_checkpoint": str(ema_path),
        "ema_checkpoint_sha256": sha256_file(ema_path),
        "primary_endpoint": "final_step_ema",
        "frozen_base_sha256_start": frozen_start,
        "frozen_base_sha256_end": frozen_end,
        "trace": trace,
    }
    write_json(output_dir / "summary.json", summary)
    write_json(final_backup_summary, summary)
    for progress in backup_dir.glob("d0_progress_step_*.pt"):
        progress.unlink(missing_ok=True)
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
