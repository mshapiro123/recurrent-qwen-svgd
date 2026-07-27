"""Cache the registered D0 teacher signals in one resumable labeling pass."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from eval.eval_identity import model_load_kwargs
from eval.eval_internal_think_token_t1_lite import restore_checkpoint
from training.internal_think_token_runtime import install_internal_control_tokens, split_internal_control_token_rows
from training.internal_think_token_spec import INTERNAL_CONTROL_TOKENS
from training.speculative_depth_d0_corpus import sha256_file
from training.speculative_depth_d0_postlock import (
    D0_LOCK_COMMIT,
    build_training_schedule,
    cache_plan,
    rejection_run_lengths,
    score_teacher_signals,
    validate_cache_summary,
)
from training.speculative_depth_d0_spec import (
    D0ExecutionPolicy,
    DRAFTER_CHECKPOINT_SHA256,
    DRAFTER_MODEL,
    DRAFTER_MODEL_REVISION,
    TEACHER_14B,
    TEACHER_14B_REVISION,
    TEACHER_7B,
    TEACHER_7B_REVISION,
    validate_locked_d0,
)
from training.train_unfrozen_recurrent import prepare_wrapper


SHARD_ROWS = 8


def read_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_json(path: str | Path, payload: Any) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, destination)


def atomic_copy(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".partial")
    shutil.copy2(source, temporary)
    if sha256_file(source) != sha256_file(temporary):
        temporary.unlink(missing_ok=True)
        raise RuntimeError(f"D0 cache copy hash mismatch: {destination}")
    os.replace(temporary, destination)


def row_digest(row: dict[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(row, sort_keys=True, ensure_ascii=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def prediction_positions(rows: list[dict[str, Any]]) -> int:
    return sum(max(0, len(row["input_ids"]) - 1) for row in rows)


def selected_positions_by_row(
    rows: list[dict[str, Any]], selected_global_positions: set[int]
) -> dict[int, list[int]]:
    selected: dict[int, list[int]] = {}
    offset = 0
    for row_index, row in enumerate(rows):
        count = max(0, len(row["input_ids"]) - 1)
        local = [value - offset for value in selected_global_positions if offset <= value < offset + count]
        if local:
            selected[row_index] = sorted(local)
        offset += count
    return selected


def completed_shard(path: Path, *, teacher: str, partition: str, start: int, stop: int) -> bool:
    if not path.exists():
        return False
    try:
        payload = torch.load(path, map_location="cpu", weights_only=False)
    except Exception:
        return False
    return (
        payload.get("kind") == "paper2_d0_teacher_cache_shard"
        and payload.get("teacher") == teacher
        and payload.get("partition") == partition
        and int(payload.get("row_start", -1)) == start
        and int(payload.get("row_stop", -1)) == stop
        and len(payload.get("rows") or []) == stop - start
    )


def load_drafter(
    *, checkpoint: Path, device: str, dtype: str, attn_implementation: str
) -> tuple[Any, Any, Any]:
    tokenizer = AutoTokenizer.from_pretrained(DRAFTER_MODEL, revision=DRAFTER_MODEL_REVISION)
    model = AutoModelForCausalLM.from_pretrained(
        DRAFTER_MODEL,
        revision=DRAFTER_MODEL_REVISION,
        **model_load_kwargs(dtype, attn_implementation),
    ).to(device)
    resize = install_internal_control_tokens(tokenizer, model)
    split_internal_control_token_rows(model, original_vocab_size=resize.original_vocab_size)
    wrapper, _ = prepare_wrapper(
        model,
        {
            "layer_split": "6,18",
            "initial_halt_prob": 0.15,
            "bridge_projection_mode": "split",
            "adapter_dtype": "float32",
            "training_mode": "full_block",
            "resume_lora": {"enabled": False},
            "merge_lora_before_unfreeze": False,
            "train_auxiliary": {
                "bridge": True,
                "halting": False,
                "reentry_adapter": False,
                "latent": False,
            },
        },
        device=device,
    )
    wrapper.base_model.get_input_embeddings().control_rows.requires_grad_(True)
    restore_checkpoint(wrapper, checkpoint)
    wrapper.eval()
    return tokenizer, wrapper, resize


def load_teacher(
    *, model_name: str, revision: str, device: str, dtype: str, attn_implementation: str
) -> tuple[Any, Any]:
    tokenizer = AutoTokenizer.from_pretrained(model_name, revision=revision)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        revision=revision,
        **model_load_kwargs(dtype, attn_implementation),
    ).to(device)
    model.eval()
    return tokenizer, model


@torch.inference_mode()
def cache_partition(
    *,
    teacher_key: str,
    teacher_model: Any,
    teacher_tokenizer: Any,
    drafter_wrapper: Any,
    drafter_tokenizer: Any,
    rows: list[dict[str, Any]],
    partition: str,
    cache_root: Path,
    selected_full_logits: dict[int, list[int]],
    device: str,
) -> dict[str, Any]:
    drafter_original_vocab = {
        key: value
        for key, value in drafter_tokenizer.get_vocab().items()
        if key not in set(INTERNAL_CONTROL_TOKENS)
    }
    if teacher_tokenizer.get_vocab() != drafter_original_vocab:
        raise RuntimeError("D0 teacher and drafter original tokenizers are not exactly aligned")
    partition_dir = cache_root / teacher_key / partition
    partition_dir.mkdir(parents=True, exist_ok=True)
    shard_receipts: list[dict[str, Any]] = []
    processed_rows = 0
    for start in range(0, len(rows), SHARD_ROWS):
        stop = min(len(rows), start + SHARD_ROWS)
        destination = partition_dir / f"rows_{start:06d}_{stop:06d}.pt"
        if completed_shard(
            destination, teacher=teacher_key, partition=partition, start=start, stop=stop
        ):
            shard_receipts.append(
                {"path": str(destination), "sha256": sha256_file(destination), "rows": stop - start}
            )
            processed_rows += stop - start
            print(f"d0_cache_resume {teacher_key}/{partition} rows={processed_rows}/{len(rows)}", flush=True)
            continue
        payload_rows: list[dict[str, Any]] = []
        for row_index in range(start, stop):
            row = rows[row_index]
            values = torch.tensor([row["input_ids"]], dtype=torch.long, device=device)
            attention = torch.ones_like(values)
            draft_output = drafter_wrapper(
                input_ids=values,
                attention_mask=attention,
                labels=None,
                max_loops=1,
                use_cache=False,
                return_dict=True,
            )
            teacher_logits = teacher_model(
                input_ids=values, attention_mask=attention, use_cache=False
            ).logits[0, :-1]
            drafter_logits = draft_output.logits[0, :-1]
            targets = values[0, 1:]
            signals = score_teacher_signals(teacher_logits, drafter_logits, targets)
            rejected = (~signals["accepted"]).cpu().tolist()
            local_selected = selected_full_logits.get(row_index, [])
            payload_rows.append(
                {
                    "row_index": row_index,
                    "row_id": str(row.get("row_id") or row.get("id") or row_index),
                    "row_sha256": row_digest(row),
                    "positions": int(targets.numel()),
                    "teacher_greedy_token_id": signals["teacher_greedy_token_id"].cpu(),
                    "drafter_greedy_token_id": signals["drafter_greedy_token_id"].cpu(),
                    "target_token_id": signals["target_token_id"].cpu(),
                    "accepted": signals["accepted"].cpu(),
                    "drafter_token_logprob_under_teacher": signals[
                        "drafter_token_logprob_under_teacher"
                    ].to(torch.float32).cpu(),
                    "drafter_token_rank_under_teacher": signals[
                        "drafter_token_rank_under_teacher"
                    ].cpu(),
                    "teacher_entropy": signals["teacher_entropy"].to(torch.float32).cpu(),
                    "teacher_to_plain_drafter_kl": signals[
                        "teacher_to_plain_drafter_kl"
                    ].to(torch.float32).cpu(),
                    "rejection_run_length": torch.tensor(
                        rejection_run_lengths(rejected), dtype=torch.int16
                    ),
                    "full_logit_local_positions": torch.tensor(local_selected, dtype=torch.int32),
                    "full_teacher_logits_bfloat16": (
                        teacher_logits.index_select(
                            0, torch.tensor(local_selected, dtype=torch.long, device=device)
                        )
                        .to(torch.bfloat16)
                        .cpu()
                        if local_selected
                        else torch.empty((0, teacher_logits.shape[-1]), dtype=torch.bfloat16)
                    ),
                }
            )
            del teacher_logits, drafter_logits, draft_output, signals
        with tempfile.TemporaryDirectory() as temporary_dir:
            local = Path(temporary_dir) / destination.name
            torch.save(
                {
                    "kind": "paper2_d0_teacher_cache_shard",
                    "lock_commit": D0_LOCK_COMMIT,
                    "teacher": teacher_key,
                    "partition": partition,
                    "row_start": start,
                    "row_stop": stop,
                    "rows": payload_rows,
                },
                local,
            )
            atomic_copy(local, destination)
        shard_receipts.append(
            {"path": str(destination), "sha256": sha256_file(destination), "rows": stop - start}
        )
        processed_rows += stop - start
        print(f"d0_cache_progress {teacher_key}/{partition} rows={processed_rows}/{len(rows)}", flush=True)
    return {
        "status": "complete",
        "rows": len(rows),
        "prediction_positions": prediction_positions(rows),
        "shards": shard_receipts,
        "full_logit_rows": len(selected_full_logits),
        "full_logit_positions": sum(len(values) for values in selected_full_logits.values()),
    }


def summarize_cache(cache_root: Path, caches: dict[str, Any]) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    for teacher, partitions in caches.items():
        summary[teacher] = {}
        for partition, receipt in partitions.items():
            accepted = 0
            positions = 0
            for shard in receipt["shards"]:
                payload = torch.load(shard["path"], map_location="cpu", weights_only=False)
                for row in payload["rows"]:
                    accepted += int(row["accepted"].sum().item())
                    positions += int(row["positions"])
            summary[teacher][partition] = {
                **receipt,
                "accepted_positions": accepted,
                "rejected_positions": positions - accepted,
                "acceptance_rate": accepted / positions if positions else 0.0,
            }
    return summary


def completed_cache_receipt(
    *, cache_root: Path, teacher: str, partition: str, expected_rows: int
) -> dict[str, Any] | None:
    shards: list[dict[str, Any]] = []
    full_logit_rows = 0
    full_logit_positions = 0
    for start in range(0, expected_rows, SHARD_ROWS):
        stop = min(expected_rows, start + SHARD_ROWS)
        path = cache_root / teacher / partition / f"rows_{start:06d}_{stop:06d}.pt"
        if not completed_shard(path, teacher=teacher, partition=partition, start=start, stop=stop):
            return None
        payload = torch.load(path, map_location="cpu", weights_only=False)
        for row in payload["rows"]:
            count = int(row["full_logit_local_positions"].numel())
            full_logit_rows += int(count > 0)
            full_logit_positions += count
        shards.append({"path": str(path), "sha256": sha256_file(path), "rows": stop - start})
    return {
        "status": "complete",
        "rows": expected_rows,
        "prediction_positions": 0,
        "shards": shards,
        "full_logit_rows": full_logit_rows,
        "full_logit_positions": full_logit_positions,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preregistration", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--cache_root", required=True)
    parser.add_argument("--output_summary", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--attn_implementation", default="sdpa")
    args = parser.parse_args()

    D0ExecutionPolicy(labeling_gpu_authorized=True).assert_allowed(labeling=True, training=False)
    prereg = read_json(args.preregistration)
    validate_locked_d0(prereg)
    if prereg.get("labeling_gpu_authorized") is not True:
        raise RuntimeError("D0 lock does not authorize labeling")
    checkpoint = Path(args.checkpoint)
    if sha256_file(checkpoint) != DRAFTER_CHECKPOINT_SHA256:
        raise RuntimeError("D0 teacher cache received the wrong drafter checkpoint")
    manifest = read_json(args.manifest)
    rows_by_partition = {
        partition: read_jsonl(manifest["artifacts"][partition]["local_restore_path"])
        for partition in cache_plan()["teacher_7b"]["partitions"]
    }
    natural_positions = prediction_positions(rows_by_partition["label_train"])
    schedule = build_training_schedule(total_steps=4000, natural_positions=natural_positions, seed=0)
    selected_global = {
        int(row["position_index"]) for row in schedule if row["kind"] == "natural"
    }
    selected_by_row = selected_positions_by_row(rows_by_partition["label_train"], selected_global)
    cache_root = Path(args.cache_root)
    cache_root.mkdir(parents=True, exist_ok=True)
    schedule_path = cache_root / "registered_training_schedule.json"
    write_json(schedule_path, {"lock_commit": D0_LOCK_COMMIT, "schedule": schedule})

    completed: dict[str, Any] = {}
    all_complete = True
    for teacher, plan in cache_plan().items():
        completed[teacher] = {}
        for partition in plan["partitions"]:
            receipt = completed_cache_receipt(
                cache_root=cache_root,
                teacher=teacher,
                partition=partition,
                expected_rows=len(rows_by_partition[partition]),
            )
            if receipt is None:
                all_complete = False
            else:
                receipt["prediction_positions"] = prediction_positions(rows_by_partition[partition])
                completed[teacher][partition] = receipt
    if all_complete:
        summary = {
            "kind": "paper2_d0_teacher_cache",
            "status": "complete",
            "lock_commit": D0_LOCK_COMMIT,
            "teacher_reloaded_after_completed_cache": False,
            "restored_from_complete_shards_without_teacher_load": True,
            "registered_natural_training_positions": len(selected_global),
            "training_schedule_sha256": sha256_file(schedule_path),
            "cache_root": str(cache_root),
            "caches": summarize_cache(cache_root, completed),
            "optimizer_steps": 0,
            "training_started": False,
        }
        validate_cache_summary(summary)
        write_json(args.output_summary, summary)
        print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
        return 0

    drafter_tokenizer, drafter_wrapper, _ = load_drafter(
        checkpoint=checkpoint,
        device=args.device,
        dtype=args.dtype,
        attn_implementation=args.attn_implementation,
    )
    caches: dict[str, Any] = {}
    teachers = (
        ("teacher_7b", TEACHER_7B, TEACHER_7B_REVISION),
        ("teacher_14b", TEACHER_14B, TEACHER_14B_REVISION),
    )
    for teacher_key, model_name, revision in teachers:
        teacher_tokenizer, teacher_model = load_teacher(
            model_name=model_name,
            revision=revision,
            device=args.device,
            dtype=args.dtype,
            attn_implementation=args.attn_implementation,
        )
        caches[teacher_key] = {}
        for partition in cache_plan()[teacher_key]["partitions"]:
            caches[teacher_key][partition] = cache_partition(
                teacher_key=teacher_key,
                teacher_model=teacher_model,
                teacher_tokenizer=teacher_tokenizer,
                drafter_wrapper=drafter_wrapper,
                drafter_tokenizer=drafter_tokenizer,
                rows=rows_by_partition[partition],
                partition=partition,
                cache_root=cache_root,
                selected_full_logits=(
                    selected_by_row if teacher_key == "teacher_7b" and partition == "label_train" else {}
                ),
                device=args.device,
            )
        del teacher_model
        gc.collect()
        torch.cuda.empty_cache()
    summary = {
        "kind": "paper2_d0_teacher_cache",
        "status": "complete",
        "lock_commit": D0_LOCK_COMMIT,
        "teacher_reloaded_after_completed_cache": False,
        "registered_natural_training_positions": len(selected_global),
        "training_schedule_sha256": sha256_file(schedule_path),
        "cache_root": str(cache_root),
        "caches": summarize_cache(cache_root, caches),
        "optimizer_steps": 0,
        "training_started": False,
    }
    validate_cache_summary(summary)
    write_json(args.output_summary, summary)
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
