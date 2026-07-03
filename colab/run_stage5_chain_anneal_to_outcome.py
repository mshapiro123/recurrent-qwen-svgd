"""Fine-tune the positive synthetic chain checkpoint while annealing chain labels away."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from colab.stage5_chain_consolidation_utils import (
    ROOT,
    backup_checkpoint_to_drive,
    checkpoint_at_step,
    latest_checkpoint,
    path_for_cli,
    publish_run,
    read_json,
    read_jsonl,
    resolve_checkpoint_reference,
    write_json,
    write_jsonl,
)


def run(cmd: list[str | os.PathLike[str]], *, check: bool = True) -> subprocess.CompletedProcess[str]:
    print("$", " ".join(map(str, cmd)), flush=True)
    proc = subprocess.run(
        list(map(str, cmd)),
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    print(proc.stdout, flush=True)
    if check and proc.returncode:
        raise subprocess.CalledProcessError(proc.returncode, proc.args, output=proc.stdout)
    return proc


def limit_rows_per_depth(source: Path, dest: Path, *, rows_per_depth: int) -> dict[str, Any]:
    kept: list[dict[str, Any]] = []
    counts: dict[str, int] = {}
    for row in read_jsonl(source):
        key = str(int(row.get("depth", row.get("synthetic_depth", 0))))
        if counts.get(key, 0) >= rows_per_depth:
            continue
        counts[key] = counts.get(key, 0) + 1
        kept.append(row)
    write_jsonl(dest, kept)
    return {
        "source": path_for_cli(source),
        "path": path_for_cli(dest),
        "rows_per_depth": rows_per_depth,
        "rows": len(kept),
        "depth_counts": dict(sorted(counts.items(), key=lambda item: int(item[0]))),
    }


def active_diag(summary: dict[str, Any]) -> dict[str, float]:
    return {str(depth): float(value) for depth, value in summary.get("active_diagonal", {}).items()}


def active_diag_min(summary: dict[str, Any]) -> float:
    values = list(active_diag(summary).values())
    return min(values) if values else 0.0


def eval_active(
    run_dir: Path,
    *,
    name: str,
    checkpoint: Path,
    data_jsonl: Path,
    loop_counts: str,
    threshold: float,
    dtype: str,
    value_prefix: str,
) -> dict[str, Any]:
    rows_path = run_dir / "eval" / f"{name}_active_rows.jsonl"
    summary_path = run_dir / "eval" / f"{name}_active_summary.json"
    run(
        [
            sys.executable,
            "eval/eval_synthetic_depth_active_labels.py",
            "--data_jsonl",
            path_for_cli(data_jsonl),
            "--checkpoint",
            path_for_cli(checkpoint),
            "--output_jsonl",
            path_for_cli(rows_path),
            "--output_summary",
            path_for_cli(summary_path),
            "--loop_counts",
            loop_counts,
            "--threshold",
            str(threshold),
            "--prediction_space",
            "full_symbols",
            "--prompt_style",
            "question_only",
            "--value_prefix",
            value_prefix,
            "--bridge_projection_mode",
            "split",
            "--dtype",
            dtype,
            "--adapter_dtype",
            "float32",
            "--device",
            os.environ.get("DEVICE", "cuda"),
        ]
    )
    summary = read_json(summary_path)
    return {
        "name": name,
        "active_rows": path_for_cli(rows_path),
        "active_summary": path_for_cli(summary_path),
        "active_diagonal": active_diag(summary),
        "active_diagonal_min": active_diag_min(summary),
        "active_total": summary.get("active_total", {}),
        "above_diagonal": summary.get("above_diagonal", {}),
    }


def eval_final(
    run_dir: Path,
    *,
    name: str,
    checkpoint: Path,
    data_jsonl: Path,
    loop_counts: str,
    threshold: float,
    dtype: str,
) -> dict[str, Any]:
    rows_path = run_dir / "eval" / f"{name}_matrix_rows.jsonl"
    summary_path = run_dir / "eval" / f"{name}_matrix_summary.json"
    run(
        [
            sys.executable,
            "eval/eval_synthetic_depth_matrix.py",
            "--mode",
            "recurrent",
            "--data_jsonl",
            path_for_cli(data_jsonl),
            "--checkpoint",
            path_for_cli(checkpoint),
            "--output_jsonl",
            path_for_cli(rows_path),
            "--output_summary",
            path_for_cli(summary_path),
            "--loop_counts",
            loop_counts,
            "--threshold",
            str(threshold),
            "--score_target",
            "option_text",
            "--bridge_projection_mode",
            "split",
            "--dtype",
            dtype,
            "--adapter_dtype",
            "float32",
            "--device",
            os.environ.get("DEVICE", "cuda"),
        ]
    )
    summary = read_json(summary_path)
    diagonal = {
        str(depth): float(summary.get("matrix", {}).get(str(depth), {}).get(str(depth), {}).get("accuracy", 0.0))
        for depth in summary.get("depths", [])
    }
    return {
        "name": name,
        "matrix_rows": path_for_cli(rows_path),
        "matrix_summary": path_for_cli(summary_path),
        "final_answer_diagonal": diagonal,
        "frontier_by_loop": summary.get("frontier_by_loop", {}),
    }


def write_train_config(
    run_dir: Path,
    *,
    train_jsonl: Path,
    resume_from: Path,
    max_loops: int,
    total_steps: int,
    ramp_steps: int,
    dtype: str,
    prelude_lr_mult: float,
) -> Path:
    lr = float(os.environ.get("STAGE5_ANNEAL_LR", "1e-5"))
    cfg = {
        "model_name": os.environ.get("MODEL_NAME", "Qwen/Qwen2.5-0.5B-Instruct"),
        "dtype": dtype,
        "adapter_dtype": "float32",
        "attn_implementation": os.environ.get("ATTN_IMPLEMENTATION", "default"),
        "layer_split": os.environ.get("STAGE5_ANNEAL_LAYER_SPLIT", "6,18"),
        "max_length": int(os.environ.get("STAGE5_ANNEAL_MAX_LENGTH", "512")),
        "max_loops": int(max_loops),
        "loop_loss_mode": "annealed_chain_to_outcome",
        "chain_anneal_hold_frac": float(os.environ.get("STAGE5_ANNEAL_HOLD_FRAC", "0.5")),
        "chain_outcome_loss_weight": 1.0,
        "initial_halt_prob": 0.15,
        "beta": 0.0,
        "halt_target_nll_weight": 0.0,
        "loop_control_ce_weight": 0.0,
        "batch_size": int(os.environ.get("STAGE5_ANNEAL_BATCH_SIZE", "1")),
        "optimizer": "adamw",
        "learning_rate": lr,
        "adamw_lr": lr,
        "weight_decay": float(os.environ.get("STAGE5_ANNEAL_WEIGHT_DECAY", "0.0")),
        "max_grad_norm": float(os.environ.get("STAGE5_ANNEAL_MAX_GRAD_NORM", "0.5")),
        "max_steps": int(total_steps),
        "save_every": int(ramp_steps),
        "log_every": int(os.environ.get("STAGE5_ANNEAL_LOG_EVERY", "100")),
        "bridge_projection_mode": "split",
        "bridge_prelude_lr_multiplier": float(prelude_lr_mult),
        "bridge_prelude_weight_decay": 0.0,
        "train_on_prompt": False,
        "gradient_checkpointing": True,
        "output_dir": path_for_cli(run_dir / "train" / "anneal_to_outcome"),
        "resume_from": path_for_cli(resume_from),
        "resume_lora": {"enabled": False},
        "merge_lora_before_unfreeze": False,
        "require_lora_loaded_before_merge": False,
        "train_auxiliary": {
            "bridge": True,
            "halting": False,
            "reentry_adapter": False,
            "latent": False,
        },
        "recurrence_curriculum": {
            "enabled": False,
            "start_loop": int(max_loops),
            "end_loop": int(max_loops),
            "schedule": "linear",
            "target_source": "row_capped",
            "ramp_compute": False,
        },
        "synthetic_phase": "chain_anneal_to_outcome",
    }
    path = run_dir / "anneal_to_outcome_train_config.yaml"
    path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
    return path


def write_markdown(run_dir: Path, payload: dict[str, Any]) -> None:
    lines = [
        f"# Chain Anneal To Outcome - {payload['run_id']}",
        "",
        f"- Status: `{payload['status']}`",
        f"- Init checkpoint: `{payload['init_checkpoint']}`",
        f"- Ramp checkpoint: `{payload.get('ramp_checkpoint')}`",
        f"- Final checkpoint: `{payload.get('final_checkpoint')}`",
        f"- Ramp active diagonal: `{payload.get('ramp_eval', {}).get('active', {}).get('active_diagonal')}`",
        f"- Final active diagonal: `{payload.get('final_eval', {}).get('active', {}).get('active_diagonal')}`",
        f"- Final-answer diagonal: `{payload.get('final_eval', {}).get('final', {}).get('final_answer_diagonal')}`",
        "",
    ]
    (run_dir / "summary.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    run_id = os.environ.get("STAGE5_ANNEAL_RUN_ID") or time.strftime("stage5_chain_anneal_%Y%m%d_%H%M%S")
    run_dir = ROOT / "outputs" / "stage5" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    n_symbols = int(os.environ.get("STAGE5_ANNEAL_N_SYMBOLS", "16"))
    max_depth = int(os.environ.get("STAGE5_ANNEAL_MAX_DEPTH", "4"))
    rows_per_depth = int(os.environ.get("STAGE5_ANNEAL_ROWS_PER_DEPTH", "256"))
    heldout_rows = int(os.environ.get("STAGE5_ANNEAL_HELDOUT_ROWS_PER_DEPTH", "64"))
    total_steps = int(os.environ.get("STAGE5_ANNEAL_TOTAL_STEPS", "2000"))
    hold_frac = float(os.environ.get("STAGE5_ANNEAL_HOLD_FRAC", "0.5"))
    ramp_steps = int(total_steps * (1.0 - hold_frac))
    if ramp_steps <= 0:
        raise ValueError("Anneal run requires a positive ramp step count")
    dtype = os.environ.get("STAGE5_ANNEAL_DTYPE", "bfloat16")
    threshold = float(os.environ.get("STAGE5_ANNEAL_THRESHOLD", "0.71"))
    value_prefix = os.environ.get("STAGE5_ANNEAL_VALUE_PREFIX", "letter:")
    source_ref = os.environ.get(
        "STAGE5_ANNEAL_INIT_CHECKPOINT",
        "outputs/stage5/stage5_chain_scaled_corrected_20260702_182827/summary.json",
    )
    init_checkpoint, checkpoint_meta = resolve_checkpoint_reference(
        source_ref,
        run_dir / "restored" / "scaled_corrected_final.pt",
        label="scaled_corrected_final",
    )
    payload: dict[str, Any] = {
        "kind": "stage5_chain_anneal_to_outcome",
        "run_id": run_id,
        "status": "started",
        "n_symbols": n_symbols,
        "max_depth": max_depth,
        "rows_per_depth": rows_per_depth,
        "heldout_rows_per_depth": heldout_rows,
        "total_steps": total_steps,
        "ramp_steps": ramp_steps,
        "hold_steps": total_steps - ramp_steps,
        "threshold": threshold,
        "init_checkpoint": path_for_cli(init_checkpoint),
        "init_checkpoint_metadata": checkpoint_meta,
    }
    write_json(run_dir / "summary.json", payload)

    data_dir = run_dir / "data"
    run(
        [
            sys.executable,
            "training/generate_synthetic_depth_task.py",
            "--output_dir",
            path_for_cli(data_dir),
            "--n_symbols",
            str(n_symbols),
            "--max_depth",
            str(max_depth),
            "--rows_per_depth",
            str(rows_per_depth),
            "--seed",
            os.environ.get("STAGE5_ANNEAL_SEED", "20260705"),
            "--num_choices",
            "4",
            "--max_target_loops",
            str(max_depth),
            "--value_prefix",
            value_prefix,
        ]
    )
    heldout_chain = data_dir / f"test_chain_mcq_heldout{heldout_rows}.jsonl"
    heldout_final = data_dir / f"test_mcq_heldout{heldout_rows}.jsonl"
    filters = {
        "heldout_active_eval": limit_rows_per_depth(
            data_dir / "test_chain_mcq.jsonl",
            heldout_chain,
            rows_per_depth=heldout_rows,
        ),
        "heldout_final_eval": limit_rows_per_depth(
            data_dir / "test_mcq.jsonl",
            heldout_final,
            rows_per_depth=heldout_rows,
        ),
    }
    config_path = write_train_config(
        run_dir,
        train_jsonl=data_dir / "train_chain_symbol_sft.jsonl",
        resume_from=init_checkpoint,
        max_loops=max_depth,
        total_steps=total_steps,
        ramp_steps=ramp_steps,
        dtype=dtype,
        prelude_lr_mult=float(os.environ.get("STAGE5_ANNEAL_PRELUDE_LR_MULT", "10.0")),
    )
    payload.update(
        {
            "status": "dataset_ready",
            "data_summary": path_for_cli(data_dir / "summary.json"),
            "filters": filters,
            "train_config": path_for_cli(config_path),
        }
    )
    write_json(run_dir / "summary.json", payload)
    write_markdown(run_dir, payload)
    publish_run(run_dir, message=f"Record chain anneal dataset {run_id} [skip ci]")

    proc = run(
        [
            sys.executable,
            "training/train_unfrozen_recurrent.py",
            "--config",
            path_for_cli(config_path),
            "--train_jsonl",
            path_for_cli(data_dir / "train_chain_symbol_sft.jsonl"),
            "--device",
            os.environ.get("DEVICE", "cuda"),
        ]
    )
    train_log = run_dir / "train" / "anneal_to_outcome_train.log"
    train_log.parent.mkdir(parents=True, exist_ok=True)
    train_log.write_text(proc.stdout or "", encoding="utf-8")
    train_dir = run_dir / "train" / "anneal_to_outcome"
    ramp_checkpoint = checkpoint_at_step(train_dir, ramp_steps)
    final_checkpoint = latest_checkpoint(train_dir)
    ramp_backup = backup_checkpoint_to_drive(
        ramp_checkpoint,
        run_id=run_id,
        stage_name="anneal_to_outcome_ramp",
        enabled=os.environ.get("STAGE5_ANNEAL_BACKUP_CHECKPOINTS_TO_DRIVE", "1") != "0",
    )
    final_backup = backup_checkpoint_to_drive(
        final_checkpoint,
        run_id=run_id,
        stage_name="anneal_to_outcome_final",
        enabled=os.environ.get("STAGE5_ANNEAL_BACKUP_CHECKPOINTS_TO_DRIVE", "1") != "0",
    )
    loop_counts = ",".join(str(idx) for idx in range(1, max_depth + 1))
    payload.update(
        {
            "status": "training_finished",
            "train_log": path_for_cli(train_log),
            "training_summary": path_for_cli(train_dir / "train_unfrozen_recurrent_summary.json"),
            "ramp_checkpoint": path_for_cli(ramp_checkpoint),
            "ramp_checkpoint_drive_backup": ramp_backup,
            "final_checkpoint": path_for_cli(final_checkpoint),
            "final_checkpoint_drive_backup": final_backup,
            "prelude_trajectory_source": path_for_cli(train_dir / "train_unfrozen_recurrent_summary.json"),
        }
    )
    write_json(run_dir / "summary.json", payload)
    write_markdown(run_dir, payload)
    publish_run(run_dir, message=f"Record chain anneal training {run_id} [skip ci]")

    payload["ramp_eval"] = {
        "active": eval_active(
            run_dir,
            name="ramp_checkpoint_heldout",
            checkpoint=ramp_checkpoint,
            data_jsonl=heldout_chain,
            loop_counts=loop_counts,
            threshold=threshold,
            dtype=dtype,
            value_prefix=value_prefix,
        ),
        "final": eval_final(
            run_dir,
            name="ramp_checkpoint_heldout",
            checkpoint=ramp_checkpoint,
            data_jsonl=heldout_final,
            loop_counts=loop_counts,
            threshold=threshold,
            dtype=dtype,
        ),
    }
    payload["final_eval"] = {
        "active": eval_active(
            run_dir,
            name="final_checkpoint_heldout",
            checkpoint=final_checkpoint,
            data_jsonl=heldout_chain,
            loop_counts=loop_counts,
            threshold=threshold,
            dtype=dtype,
            value_prefix=value_prefix,
        ),
        "final": eval_final(
            run_dir,
            name="final_checkpoint_heldout",
            checkpoint=final_checkpoint,
            data_jsonl=heldout_final,
            loop_counts=loop_counts,
            threshold=threshold,
            dtype=dtype,
        ),
    }
    final_active_min = float(payload["final_eval"]["active"]["active_diagonal_min"])
    final_answer_values = list(payload["final_eval"]["final"]["final_answer_diagonal"].values())
    payload["decision_read"] = {
        "capability_component": {
            "metric": "heldout final-answer diagonal minimum",
            "min_accuracy": min(final_answer_values) if final_answer_values else 0.0,
            "strong_bar": 0.90,
        },
        "mechanism_component": {
            "metric": "heldout active-label diagonal minimum",
            "min_accuracy": final_active_min,
            "bar": threshold,
            "pass": final_active_min >= threshold,
        },
    }
    payload["status"] = "finished"
    write_json(run_dir / "summary.json", payload)
    write_markdown(run_dir, payload)
    publish_run(run_dir, message=f"Record Stage 5 chain anneal to outcome {run_id} [skip ci]", update_pointer=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
