"""Replicate natural-surface rung zero with a fine checkpoint dose curve.

This trains on a fresh relay-verbal seed but evaluates every checkpoint on the
same frozen relay, pointer, and synthetic sets used by the landed rung-zero run.
That keeps the replication and early-stopping dose curve paired to the original
evidence rather than mixing in new eval-row sampling noise.
"""

from __future__ import annotations

import json
import os
import random
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(os.environ.get("STAGE5_ROOT", "/content/recurrent-qwen-svgd"))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from colab.run_stage5_natural_surface_transfer import (  # noqa: E402
    DEFAULT_DATA_SUMMARY,
    DEFAULT_INIT_SOURCE_SUMMARY,
    active_diag_min,
    backup_checkpoint_to_drive,
    data_paths,
    env_flag,
    eval_active_checkpoint,
    path_for_cli,
    publish_run,
    read_json,
    restore_summary_checkpoint,
    root_path,
    score_experiment,
    train_verbal_rung_zero,
    verify_expected_init_checkpoint,
    write_json,
    write_jsonl,
)
from training.natural_surface_transfer import (  # noqa: E402
    NaturalSurfaceConfig,
    build_synthetic_rehearsal_rows,
    build_verbal_rows,
    manifest_for_rows,
)


SOURCE_RUN_ID = "stage5_natural_surface_transfer_rung0_fixed_prompt_20260709_133812"
SOURCE_SUMMARY = f"outputs/stage5/{SOURCE_RUN_ID}/summary.json"
SOURCE_RECEIPT_SUMMARY = "outputs/stage5/stage5_natural_surface_receipts_20260709_210151/summary.json"
SYNTHETIC_FULL_WIDTH_DATA = (
    "outputs/stage5/stage5_synthetic_depth_frozen_eval_v3_depth22_n24/data/test_chain_mcq.jsonl"
)


def parse_steps(raw: str) -> list[int]:
    return sorted({int(item) for item in raw.split(",") if item.strip()})


def checkpoint_at(output_dir: Path, step: int) -> Path:
    path = output_dir / f"unfrozen_recurrent_step_{int(step)}.pt"
    if not path.exists():
        raise FileNotFoundError(path)
    return path


def write_training_mix(out_dir: Path, *, seed: int) -> dict[str, Any]:
    data_dir = out_dir / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    cfg = NaturalSurfaceConfig(
        n_symbols=20,
        train_max_depth=8,
        eval_max_depth=12,
        train_rows_per_depth=256,
        val_rows_per_depth=64,
        eval_rows_per_depth=128,
        seed=seed,
        max_target_loops=8,
    )
    relay = build_verbal_rows(
        family="relay",
        split="train",
        n_symbols=cfg.n_symbols,
        max_depth=cfg.train_max_depth,
        rows_per_depth=cfg.train_rows_per_depth,
        seed=cfg.seed,
        max_target_loops=cfg.train_max_depth,
    )
    synthetic = build_synthetic_rehearsal_rows(cfg)
    mix = [row | {"curriculum_family": "relay_verbal"} for row in relay] + synthetic
    random.Random(seed + 77_777).shuffle(mix)
    write_jsonl(data_dir / "train_relay_chain_symbol_sft.jsonl", relay)
    write_jsonl(data_dir / "synthetic_rehearsal_chain_symbol_sft.jsonl", synthetic)
    write_jsonl(data_dir / "rung0_train_mix_chain_symbol_sft.jsonl", mix)
    summary = {
        "seed": seed,
        "files": {
            "train_relay_chain_symbol_sft": path_for_cli(data_dir / "train_relay_chain_symbol_sft.jsonl"),
            "synthetic_rehearsal_chain_symbol_sft": path_for_cli(data_dir / "synthetic_rehearsal_chain_symbol_sft.jsonl"),
            "rung0_train_mix_chain_symbol_sft": path_for_cli(data_dir / "rung0_train_mix_chain_symbol_sft.jsonl"),
        },
        "manifests": {
            "train_relay_chain_symbol_sft": manifest_for_rows(relay),
            "synthetic_rehearsal_chain_symbol_sft": manifest_for_rows(synthetic),
            "rung0_train_mix_chain_symbol_sft": manifest_for_rows(mix),
        },
        "eval_policy": "All checkpoint evals use the original landed frozen relay/pointer/synthetic rows.",
    }
    write_json(out_dir / "data" / "summary.json", summary)
    return summary


def eval_checkpoint_step(
    out_dir: Path,
    *,
    checkpoint: Path,
    step: int,
    paths: dict[str, Path],
    frozen_baseline: dict[str, Any],
    frozen_full_width_min: float,
    dtype: str,
) -> dict[str, Any]:
    step_dir = out_dir / f"eval_step_{step}"
    step_dir.mkdir(parents=True, exist_ok=True)
    relay = eval_active_checkpoint(
        step_dir,
        name=f"step{step}_relay",
        checkpoint=checkpoint,
        data_jsonl=paths["relay_test_chain_mcq"],
        loop_counts="1,2,3,4,5,6,7,8,9,10,11,12",
        value_prefix="name:",
        dtype=dtype,
    )
    pointer = eval_active_checkpoint(
        step_dir,
        name=f"step{step}_pointer",
        checkpoint=checkpoint,
        data_jsonl=paths["pointer_test_chain_mcq"],
        loop_counts="1,2,3,4,5,6,7,8,9,10,11,12",
        value_prefix="name:",
        dtype=dtype,
    )
    synthetic = eval_active_checkpoint(
        step_dir,
        name=f"step{step}_synthetic_rehearsal",
        checkpoint=checkpoint,
        data_jsonl=paths["synthetic_rehearsal_chain_symbol_sft"],
        loop_counts="1,2,3,4,5,6,7,8",
        value_prefix="letter:",
        dtype=dtype,
    )
    synthetic_full_width = eval_active_checkpoint(
        step_dir,
        name=f"step{step}_synthetic_full_width",
        checkpoint=checkpoint,
        data_jsonl=root_path(SYNTHETIC_FULL_WIDTH_DATA),
        loop_counts="1,2,3,4,5,6,7,8,9,10,11,12",
        value_prefix="letter:",
        dtype=dtype,
    )
    decision = score_experiment(
        frozen=frozen_baseline,
        post={"relay": relay, "pointer": pointer, "synthetic_rehearsal": synthetic},
        train_depth_max=8,
        eval_depth_max=12,
    )["experiment_1"]
    full_width_min = active_diag_min(synthetic_full_width, depths=range(1, 13))
    decision["synthetic_full_width_min_1_12"] = full_width_min
    decision["synthetic_full_width_delta_vs_frozen"] = full_width_min - float(frozen_full_width_min)
    decision["synthetic_full_width_nonregression_pass"] = bool(
        full_width_min >= float(frozen_full_width_min) - 0.03
    )
    return {
        "step": step,
        "checkpoint": path_for_cli(checkpoint),
        "relay": relay,
        "pointer": pointer,
        "synthetic_rehearsal": synthetic,
        "synthetic_full_width": synthetic_full_width,
        "decision_read": decision,
    }


def checkpoint_eval_finished(row: dict[str, Any]) -> bool:
    required = ["relay", "pointer", "synthetic_rehearsal", "synthetic_full_width"]
    for name in required:
        raw = (row.get(name) or {}).get("active_summary")
        if not raw:
            return False
        summary = root_path(raw)
        if not summary.exists():
            return False
    return True


def frozen_full_width_min_from_receipt(receipt: dict[str, Any]) -> float:
    record = (
        receipt.get("evals", {})
        .get("frozen_n24", {})
        .get("synthetic_frozen_v3_d1_12", {})
    )
    diagonal = record.get("active_diagonal") or {}
    values = [float(diagonal[str(depth)]) for depth in range(1, 13) if str(depth) in diagonal]
    if len(values) != 12:
        raise RuntimeError("Frozen full-width synthetic receipt is incomplete for depths 1-12")
    return min(values)


def curve_shape(checkpoint_evals: list[dict[str, Any]]) -> dict[str, Any]:
    values = []
    for row in checkpoint_evals:
        relay = row["decision_read"]["relay_extrap_depth_min"]
        pointer = row["decision_read"]["pointer_extrap_depth_min"]
        values.append({"step": int(row["step"]), "pooled_tail_min": min(float(relay), float(pointer))})
    best = max(values, key=lambda item: item["pooled_tail_min"]) if values else None
    final = values[-1] if values else None
    return {
        "values": values,
        "best": best,
        "final": final,
        "tail_peaks_before_final": bool(best and final and best["step"] < final["step"]),
    }


def write_markdown(out_dir: Path, payload: dict[str, Any]) -> None:
    lines = [
        f"# Natural-Surface Replication Dose - {payload['run_id']}",
        "",
        f"- Status: `{payload['status']}`",
        f"- Source eval rows: `{payload['source_summary']}`",
        f"- Train seed: `{payload['train_seed']}`",
        f"- Save steps: `{payload['save_steps']}`",
        f"- Init checkpoint: `{payload.get('init_checkpoint_metadata')}`",
        "",
        "## Checkpoint Evals",
        "",
    ]
    for row in payload.get("checkpoint_evals", []):
        lines.append(f"- Step `{row['step']}`: `{row.get('decision_read')}`")
    lines.extend(["", "## Curve Shape", "", f"`{payload.get('curve_shape')}`", ""])
    (out_dir / "summary.md").write_text("\n".join(lines), encoding="utf-8")


def publish(out_dir: Path, payload: dict[str, Any], *, message: str) -> None:
    write_json(out_dir / "summary.json", payload)
    write_markdown(out_dir, payload)
    publish_run(out_dir, message=message, update_pointer=False)


def main() -> int:
    run_id = os.environ.get("STAGE5_NATURAL_REPLICATION_RUN_ID") or time.strftime(
        "stage5_natural_surface_replication_dose_%Y%m%d_%H%M%S"
    )
    out_dir = ROOT / "outputs" / "stage5" / run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    train_seed = int(os.environ.get("STAGE5_NATURAL_REPLICATION_SEED", "931337"))
    save_steps = parse_steps(os.environ.get("STAGE5_NATURAL_REPLICATION_SAVE_STEPS", "1000,1500,2000,2500,3000,4000,6000"))
    train_steps = int(os.environ.get("STAGE5_NATURAL_REPLICATION_TRAIN_STEPS", "6000"))
    if train_steps < max(save_steps):
        raise ValueError(f"train_steps={train_steps} is below final save step {max(save_steps)}")
    dtype = os.environ.get("STAGE5_NATURAL_REPLICATION_DTYPE", "bfloat16")
    source_summary = read_json(os.environ.get("STAGE5_NATURAL_REPLICATION_SOURCE_SUMMARY", SOURCE_SUMMARY))
    source_receipt = read_json(
        os.environ.get("STAGE5_NATURAL_REPLICATION_RECEIPT_SUMMARY", SOURCE_RECEIPT_SUMMARY)
    )
    frozen_full_width_min = frozen_full_width_min_from_receipt(source_receipt)
    paths = data_paths(source_summary.get("data_summary", DEFAULT_DATA_SUMMARY))
    existing_payload = read_json(out_dir / "summary.json") if (out_dir / "summary.json").exists() else None
    train_data = (existing_payload or {}).get("train_data") or write_training_mix(out_dir, seed=train_seed)
    checkpoint, init_meta = restore_summary_checkpoint(
        os.environ.get("STAGE5_NATURAL_REPLICATION_INIT_SOURCE_SUMMARY", DEFAULT_INIT_SOURCE_SUMMARY),
        out_dir / "restored" / "n24_support12_step6000.pt",
        label="n24_support12_step6000",
        preferred_step=6000,
    )
    verify_expected_init_checkpoint(init_meta)

    payload: dict[str, Any] = existing_payload or {
        "kind": "stage5_natural_surface_replication_dose",
        "run_id": run_id,
        "status": "started",
        "source_summary": os.environ.get("STAGE5_NATURAL_REPLICATION_SOURCE_SUMMARY", SOURCE_SUMMARY),
        "train_seed": train_seed,
        "save_steps": save_steps,
        "train_steps": train_steps,
        "dtype": dtype,
        "train_data": train_data,
        "init_checkpoint_metadata": init_meta,
        "checkpoint_evals": [],
        "checkpoint_backups": {},
        "frozen_full_width_min_1_12": frozen_full_width_min,
    }
    if existing_payload is None:
        publish(out_dir, payload, message=f"Record natural-surface replication start {run_id} [skip ci]")
    else:
        if int(payload.get("train_seed", -1)) != train_seed:
            raise RuntimeError("Refusing to resume replication with a different train seed")
        if list(payload.get("save_steps", [])) != save_steps:
            raise RuntimeError("Refusing to resume replication with different save steps")

    os.environ["STAGE5_NATURAL_TRANSFER_SAVE_STEPS"] = ",".join(str(step) for step in save_steps)
    os.environ["STAGE5_NATURAL_TRANSFER_SAVE_EVERY"] = "0"
    os.environ["STAGE5_NATURAL_TRANSFER_TRAIN_SEED"] = str(train_seed)
    output_dir = out_dir / "train" / "verbal_rung_zero"
    backup_enabled = env_flag("STAGE5_NATURAL_REPLICATION_BACKUP_CHECKPOINTS_TO_DRIVE", "1")
    backups = payload.setdefault("checkpoint_backups", {})
    backups_complete = all(
        bool(backups.get(str(step))) and Path(str(backups[str(step)])).is_file()
        for step in save_steps
    )
    if not backups_complete:
        stage = train_verbal_rung_zero(
            out_dir,
            train_jsonl=out_dir / "data" / "rung0_train_mix_chain_symbol_sft.jsonl",
            resume_from=checkpoint,
            max_steps=train_steps,
            max_loops=8,
            dtype=dtype,
        )
        payload["train_stage"] = stage
        payload["status"] = "training_finished"
        publish(out_dir, payload, message=f"Record natural-surface replication training {run_id} [skip ci]")
        for step in save_steps:
            ckpt = checkpoint_at(output_dir, step)
            backup = backup_checkpoint_to_drive(
                ckpt,
                run_id=run_id,
                stage_name="verbal_rung_zero",
                enabled=backup_enabled,
            )
            if not backup:
                backup = str(ckpt)
            backups[str(step)] = backup
        payload["status"] = "checkpoints_backed_up"
        publish(out_dir, payload, message=f"Back up natural-surface replication checkpoints {run_id} [skip ci]")
    else:
        print("replication_training_resume=using_backed_up_checkpoints", flush=True)

    for step in save_steps:
        existing_row = next((row for row in payload.get("checkpoint_evals", []) if int(row["step"]) == step), None)
        if existing_row and checkpoint_eval_finished(existing_row):
            print(f"replication_eval_skip_completed_step={step}", flush=True)
            continue
        local_ckpt = output_dir / f"unfrozen_recurrent_step_{step}.pt"
        ckpt = local_ckpt if local_ckpt.exists() else Path(backups[str(step)])
        if not ckpt.exists():
            raise FileNotFoundError(f"Missing checkpoint for step {step}: {ckpt}")
        row = eval_checkpoint_step(
            out_dir,
            checkpoint=ckpt,
            step=step,
            paths=paths,
            frozen_baseline=source_summary["frozen_baseline"],
            frozen_full_width_min=frozen_full_width_min,
            dtype=dtype,
        )
        row["checkpoint_drive_backup"] = backups[str(step)]
        payload["checkpoint_evals"] = [
            old for old in payload.get("checkpoint_evals", []) if int(old["step"]) != step
        ] + [row]
        payload["checkpoint_evals"].sort(key=lambda item: int(item["step"]))
        payload["curve_shape"] = curve_shape(payload["checkpoint_evals"])
        payload["status"] = f"evaluated_step_{step}"
        publish(out_dir, payload, message=f"Record natural-surface replication eval {run_id} step {step} [skip ci]")

    payload["status"] = "finished"
    payload["curve_shape"] = curve_shape(payload["checkpoint_evals"])
    publish(out_dir, payload, message=f"Record natural-surface replication dose final {run_id} [skip ci]")
    print(json.dumps({"run_id": run_id, "summary": path_for_cli(out_dir / "summary.json")}, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
