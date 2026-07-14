"""Causal cap-3 retention replay with additive forward-task rehearsal."""

from __future__ import annotations

import hashlib
import json
import math
import os
import random
import sys
import time
from pathlib import Path
from typing import Any

import yaml

from colab.run_stage5_inverse_composition_staircase import (
    _guardrail_receipt,
    _prepare_guardrail_data,
    _publish,
    _run_diagonal,
    _run_staircase_eval,
    build_matched_data,
)
from colab.run_stage5_natural_surface_transfer import restore_checkpoint, run, sha256_file
from colab.stage5_chain_consolidation_utils import backup_checkpoint_to_drive, path_for_cli
from colab.stage5_n24_rung import tier1_canary_verdict
from training.staircase_curriculum import equalized_loop_weights, exposure_fractions


ROOT = Path(os.environ.get("STAGE5_ROOT", Path(__file__).resolve().parents[1]))
SOURCE_SUMMARY = ROOT / "outputs/stage5/stage5_inverse_table_rebase_caps3_4_20260713/summary.json"
STAIRCASE_SUMMARY = ROOT / "outputs/stage5/stage5_inverse_composition_staircase_20260713/summary.json"
REHEARSAL_SOURCE = ROOT / "outputs/stage5/stage5_n24_support12_rung_20260707_140139/data/train_chain_mcq.jsonl"
SOURCE_CAP3_SHA256 = "83767ebff2c2a13a2f15fe8266f605fb8485985c3289c1f1720cd70c122a9ac5"
BASELINE_STEPS = 250
EFFECTIVE_BATCH_SIZE = 8
REHEARSAL_FRACTION = 0.25
MAX_LOOPS = 12


def read_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def write_json(path: str | Path, payload: dict[str, Any]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_jsonl(path: str | Path, rows: list[dict[str, Any]]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8")


def rehearsal_optimizer_steps(
    *,
    baseline_steps: int,
    effective_batch_size: int,
    rehearsal_fraction: float,
) -> int:
    if not 0.0 < rehearsal_fraction < 1.0:
        raise ValueError("rehearsal_fraction must be between zero and one")
    baseline_task_rows = int(baseline_steps) * int(effective_batch_size)
    total_rows = math.ceil(baseline_task_rows / (1.0 - float(rehearsal_fraction)))
    return math.ceil(total_rows / int(effective_batch_size))


def _cycled_sample(rows: list[dict[str, Any]], count: int, rng: random.Random) -> list[dict[str, Any]]:
    if not rows:
        raise ValueError("Cannot sample from an empty row set")
    output: list[dict[str, Any]] = []
    while len(output) < count:
        cycle = list(rows)
        rng.shuffle(cycle)
        output.extend(dict(row) for row in cycle[: count - len(output)])
    return output


def build_rehearsal_mix(
    task_rows: list[dict[str, Any]],
    rehearsal_rows: list[dict[str, Any]],
    *,
    optimizer_steps: int,
    effective_batch_size: int,
    rehearsal_fraction: float,
    seed: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    total_rows = int(optimizer_steps) * int(effective_batch_size)
    rehearsal_count = int(round(total_rows * float(rehearsal_fraction)))
    task_count = total_rows - rehearsal_count
    rng = random.Random(int(seed))
    task = _cycled_sample(task_rows, task_count, rng)
    rehearsal = _cycled_sample(rehearsal_rows, rehearsal_count, rng)
    for row in task:
        row["training_source"] = "inverse_table_task"
    for row in rehearsal:
        row["training_source"] = "forward_synthetic_rehearsal"
    mixed = task + rehearsal
    rng.shuffle(mixed)
    ids = "\n".join(f"{row['training_source']}|{row.get('id') or row.get('instance_id')}" for row in mixed)
    return mixed, {
        "rows": total_rows,
        "task_rows": task_count,
        "rehearsal_rows": rehearsal_count,
        "realized_rehearsal_fraction": rehearsal_count / total_rows,
        "schedule_sha256": hashlib.sha256(ids.encode("utf-8")).hexdigest(),
        "one_exact_epoch": True,
    }


def rehearsal_weight_profiles(
    *,
    task_weights: list[float],
    rehearsal_depths: list[int],
    max_loops: int,
) -> dict[str, list[float]]:
    if len(task_weights) > int(max_loops):
        raise ValueError("task weights exceed max_loops")
    task_profile = [float(value) for value in task_weights] + [0.0] * (int(max_loops) - len(task_weights))
    rehearsal_rows = [{"depth": int(depth)} for depth in rehearsal_depths]
    exposure = exposure_fractions(rehearsal_rows, cap=max_loops)
    rehearsal_profile = equalized_loop_weights(exposure, cap=max_loops, newest_multiplier=2.0)
    target_sum = sum(task_profile)
    scale = target_sum / sum(rehearsal_profile)
    rehearsal_profile = [value * scale for value in rehearsal_profile]
    return {"task": task_profile, "rehearsal": rehearsal_profile}


def fixed_schedule_dose(rows: list[dict[str, Any]], *, max_loops: int) -> dict[str, Any]:
    by_source: dict[str, dict[str, Any]] = {}
    for row in rows:
        source = str(row["training_source"])
        bucket = by_source.setdefault(
            source,
            {"rows": 0, "weighted_active_labels_by_loop": [0.0] * int(max_loops)},
        )
        bucket["rows"] += 1
        depth = int(row["depth"])
        weights = [float(value) for value in row["loop_label_weights"]]
        for loop_index in range(min(depth, int(max_loops))):
            bucket["weighted_active_labels_by_loop"][loop_index] += weights[loop_index]
    for bucket in by_source.values():
        bucket["weighted_active_labels_total"] = sum(bucket["weighted_active_labels_by_loop"])
    return by_source


def _source_cap3(source: dict[str, Any]) -> dict[str, Any]:
    matches = [stage for stage in source.get("stages", []) if int(stage.get("cap", -1)) == 3]
    if len(matches) != 1:
        raise RuntimeError(f"Expected exactly one C cap-3 stage, found {len(matches)}")
    stage = matches[0]
    if stage.get("checkpoint_sha256") != SOURCE_CAP3_SHA256:
        raise RuntimeError("C cap-3 source checkpoint SHA mismatch")
    return stage


def _write_config(
    path: Path,
    *,
    checkpoint: Path,
    output_dir: Path,
    max_steps: int,
    seed: int,
) -> None:
    cfg = {
        "model_name": "Qwen/Qwen2.5-0.5B-Instruct",
        "dtype": os.environ.get("STAGE5_REHEARSAL_DTYPE", "bfloat16"),
        "adapter_dtype": "float32",
        "attn_implementation": os.environ.get("ATTN_IMPLEMENTATION", "default"),
        "layer_split": "6,18",
        "max_length": 640,
        "max_loops": MAX_LOOPS,
        "loop_loss_mode": "weighted_per_loop_labels",
        "row_specific_forward_loops": True,
        "batch_size": 1,
        "gradient_accumulation_steps": EFFECTIVE_BATCH_SIZE,
        "minimum_effective_batch_size": EFFECTIVE_BATCH_SIZE,
        "seed": int(seed),
        "optimizer": "adamw",
        "learning_rate": 1e-5,
        "adamw_lr": 1e-5,
        "weight_decay": 0.0,
        "max_grad_norm": 0.5,
        "max_steps": int(max_steps),
        "save_every": int(max_steps),
        "log_every": 25,
        "bridge_projection_mode": "split",
        "bridge_prelude_lr_multiplier": 10.0,
        "bridge_prelude_weight_decay": 0.0,
        "bridge_prelude_lr_include_norm": False,
        "bridge_prelude_grad_multiplier": 1.0,
        "train_on_prompt": False,
        "gradient_checkpointing": True,
        "require_active_supervision": True,
        "require_nonzero_train_gradient": True,
        "output_dir": path_for_cli(output_dir),
        "resume_from": path_for_cli(checkpoint),
        "resume_lora": {"enabled": False},
        "merge_lora_before_unfreeze": False,
        "require_lora_loaded_before_merge": False,
        "train_auxiliary": {"bridge": True, "halting": False, "reentry_adapter": False, "latent": False},
        "recurrence_curriculum": {
            "enabled": False,
            "start_loop": MAX_LOOPS,
            "end_loop": MAX_LOOPS,
            "schedule": "linear",
            "target_source": "row_capped",
            "ramp_compute": False,
        },
        "synthetic_phase": "inverse_table_cap3_rehearsal",
        "synthetic_stage": "cap_3_retention_replay",
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")


def main() -> int:
    run_id = os.environ.get("STAGE5_REHEARSAL_RUN_ID") or time.strftime(
        "stage5_inverse_table_cap3_rehearsal_%Y%m%d_%H%M%S"
    )
    run_dir = ROOT / "outputs/stage5" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    source_path = Path(os.environ.get("STAGE5_REHEARSAL_SOURCE_SUMMARY", str(SOURCE_SUMMARY)))
    if not source_path.is_absolute():
        source_path = ROOT / source_path
    source = read_json(source_path)
    stage = _source_cap3(source)
    checkpoint, restore_receipt = restore_checkpoint(
        [stage.get("checkpoint_drive_backup"), stage.get("checkpoint")],
        run_dir / "restored" / "C_cap3.pt",
        label="rehearsal_C_cap3",
    )
    if restore_receipt["selected_checkpoint_sha256"] != SOURCE_CAP3_SHA256:
        raise RuntimeError("Restored C cap-3 checkpoint SHA mismatch")

    data_dir = run_dir / "data"
    build_matched_data(data_dir)
    task_rows = [row for row in read_jsonl(data_dir / "train_C_inverse_table.jsonl") if int(row["depth"]) <= 3]
    task_test = data_dir / "test_C_inverse_table.jsonl"
    rehearsal_source = Path(os.environ.get("STAGE5_REHEARSAL_FORWARD_DATA", str(REHEARSAL_SOURCE)))
    if not rehearsal_source.is_absolute():
        rehearsal_source = ROOT / rehearsal_source
    rehearsal_rows = read_jsonl(rehearsal_source)
    if sorted({int(row["depth"]) for row in rehearsal_rows}) != list(range(1, MAX_LOOPS + 1)):
        raise RuntimeError("Forward rehearsal source must cover depths 1-12")
    max_steps = rehearsal_optimizer_steps(
        baseline_steps=BASELINE_STEPS,
        effective_batch_size=EFFECTIVE_BATCH_SIZE,
        rehearsal_fraction=REHEARSAL_FRACTION,
    )
    mixed, mix_receipt = build_rehearsal_mix(
        task_rows,
        rehearsal_rows,
        optimizer_steps=max_steps,
        effective_batch_size=EFFECTIVE_BATCH_SIZE,
        rehearsal_fraction=REHEARSAL_FRACTION,
        seed=81_903,
    )
    task_weights = [float(value) for value in stage["plan"]["loop_label_weights"]]
    profiles = rehearsal_weight_profiles(
        task_weights=task_weights,
        rehearsal_depths=[
            int(row["depth"])
            for row in mixed
            if row["training_source"] == "forward_synthetic_rehearsal"
        ],
        max_loops=MAX_LOOPS,
    )
    for row in mixed:
        if row["training_source"] == "inverse_table_task":
            row["loop_label_weights"] = profiles["task"]
            row["forward_loop_count"] = 3
        else:
            row["loop_label_weights"] = profiles["rehearsal"]
            row["forward_loop_count"] = int(row["depth"])
    mix_receipt["dose"] = fixed_schedule_dose(mixed, max_loops=MAX_LOOPS)
    train_path = data_dir / "train_cap3_plus_rehearsal.jsonl"
    write_jsonl(train_path, mixed)
    config_path = run_dir / "config" / "cap3_rehearsal.yaml"
    train_dir = run_dir / "train" / "cap3_rehearsal"
    _write_config(config_path, checkpoint=checkpoint, output_dir=train_dir, max_steps=max_steps, seed=81_903)
    run(
        [
            sys.executable,
            "training/train_unfrozen_recurrent.py",
            "--config",
            path_for_cli(config_path),
            "--train_jsonl",
            path_for_cli(train_path),
            "--device",
            os.environ.get("DEVICE", "cuda"),
        ],
        cwd=ROOT,
    )
    trained = train_dir / f"unfrozen_recurrent_step_{max_steps}.pt"
    if not trained.exists():
        raise RuntimeError(f"Training did not produce {trained}")
    checkpoint_sha = sha256_file(trained)
    drive_backup = backup_checkpoint_to_drive(
        trained,
        run_id=run_id,
        stage_name="cap3_rehearsal_selected",
        enabled=True,
    )
    task_eval = _run_staircase_eval(
        run_dir,
        label="cap3_task",
        checkpoint=trained,
        train_jsonl=train_path,
        test_jsonl=task_test,
        cap=3,
        probes=False,
    )
    task_row = task_eval["test"]["diagonal_by_depth"]["3"]
    task_gate = {
        "correct": int(task_row["correct"]),
        "total": int(task_row["total"]),
        "required_correct": 46,
        "passed": int(task_row["total"]) == 64 and int(task_row["correct"]) >= 46,
    }
    staircase_source = read_json(STAIRCASE_SUMMARY)
    guardrail_paths = _prepare_guardrail_data(run_dir, staircase_source)
    synthetic = _guardrail_receipt(
        run_dir,
        label="cap3_rehearsal_synthetic",
        checkpoint=trained,
        data_jsonl=guardrail_paths["synthetic"],
    )
    natural = _run_diagonal(
        run_dir,
        label="cap3_rehearsal_natural",
        checkpoint=trained,
        data_jsonl=guardrail_paths["natural"],
        max_depth=8,
        value_prefix="name:",
    )
    baseline_accuracy = float(staircase_source["tier1_canary_baseline"]["accuracy"])
    natural_delta = float(natural["accuracy"]) - baseline_accuracy
    natural_verdict = tier1_canary_verdict(accuracy_delta=natural_delta, ppl_relative_delta=None)
    natural_gate = {
        "baseline_accuracy": baseline_accuracy,
        "candidate_accuracy": float(natural["accuracy"]),
        "accuracy_delta": natural_delta,
        "verdict": natural_verdict,
        "passed": natural_verdict["status"] != "red_hard_stop",
    }
    green = task_gate["passed"] and synthetic["passed"] and natural_gate["passed"]
    payload = {
        "kind": "stage5_inverse_table_cap3_rehearsal",
        "run_id": run_id,
        "status": "rehearsal_green_cap4_authorized" if green else "rehearsal_failed_review_required",
        "source_summary": path_for_cli(source_path),
        "source_checkpoint_sha256": SOURCE_CAP3_SHA256,
        "restore_receipt": restore_receipt,
        "checkpoint": path_for_cli(trained),
        "checkpoint_sha256": checkpoint_sha,
        "checkpoint_drive_backup": drive_backup,
        "optimizer_steps": max_steps,
        "mix": mix_receipt,
        "weight_profiles": profiles,
        "task_gate": task_gate,
        "synthetic_retention": synthetic,
        "natural_canary": natural_gate,
        "cap4_authorized": green,
    }
    write_json(run_dir / "summary.json", payload)
    (run_dir / "summary.md").write_text(
        "\n".join(
            [
                f"# Cap-3 Rehearsal Replay - {run_id}",
                "",
                f"- Status: `{payload['status']}`",
                f"- Task gate: `{task_gate['correct']}/{task_gate['total']}`",
                f"- Synthetic retention: `{synthetic['active_diagonal_min']}`",
                f"- Natural canary: `{natural_verdict['status']}`",
                f"- Cap 4 authorized: `{green}`",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    _publish(run_dir, f"Record cap-3 rehearsal replay {run_id} [skip ci]")
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if green else 2


if __name__ == "__main__":
    raise SystemExit(main())
