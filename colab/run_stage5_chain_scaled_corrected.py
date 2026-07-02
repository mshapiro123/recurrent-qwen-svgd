"""Run the corrected scaled synthetic-depth chain experiment.

This is the experiment the split-bridge micro-test was supposed to set up:
train full-symbol intermediate chain labels on N=16, evaluate active labels
instead of final-answer off-diagonal cells, and publish intermediate summaries.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(os.environ.get("STAGE5_ROOT", "/content/recurrent-qwen-svgd"))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
DRIVE_CHECKPOINT_ROOT = Path(
    os.environ.get(
        "STAGE5_CHAIN_CORRECTED_DRIVE_ROOT",
        "/content/drive/MyDrive/recurrent-qwen-svgd-checkpoints",
    )
)


def redact(text: str) -> str:
    out = str(text)
    for name in ("GH_TOKEN", "GITHUB_TOKEN", "HF_TOKEN", "HUGGINGFACE_HUB_TOKEN"):
        token = os.environ.get(name)
        if token:
            out = out.replace(token, "****")
    return out


def run(
    cmd: list[str | os.PathLike[str]],
    *,
    cwd: Path = ROOT,
    env: dict[str, str] | None = None,
    check: bool = True,
) -> subprocess.CompletedProcess[str]:
    print("$", redact(" ".join(map(str, cmd))), flush=True)
    process = subprocess.Popen(
        list(map(str, cmd)),
        cwd=str(cwd),
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        bufsize=1,
    )
    chunks: list[str] = []
    assert process.stdout is not None
    for line in process.stdout:
        safe = redact(line)
        print(safe, end="", flush=True)
        chunks.append(safe)
    stdout = "".join(chunks)
    proc = subprocess.CompletedProcess(list(map(str, cmd)), process.wait(), stdout, None)
    if check and proc.returncode:
        print("FAILED_COMMAND_TAIL_START", flush=True)
        print("\n".join(stdout.splitlines()[-180:]), flush=True)
        print("FAILED_COMMAND_TAIL_END", flush=True)
        raise subprocess.CalledProcessError(proc.returncode, proc.args, output=stdout)
    return proc


def env_flag(name: str, default: str = "0") -> bool:
    return os.environ.get(name, default).strip().lower() in {"1", "true", "yes", "y", "on"}


def path_for_cli(path: str | Path) -> str:
    candidate = Path(path)
    try:
        return candidate.relative_to(ROOT).as_posix()
    except ValueError:
        return str(candidate).replace("\\", "/")


def root_path(path: str | Path) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else ROOT / candidate


def read_json(path: str | Path) -> dict[str, Any]:
    return json.loads(root_path(path).read_text(encoding="utf-8"))


def write_json(path: str | Path, payload: dict[str, Any]) -> None:
    out = root_path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def write_yaml(path: str | Path, payload: dict[str, Any]) -> None:
    out = root_path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")


def latest_checkpoint(output_dir: Path) -> Path:
    checkpoints = sorted(output_dir.glob("unfrozen_recurrent_step_*.pt"), key=lambda path: path.stat().st_mtime)
    if not checkpoints:
        raise FileNotFoundError(f"No checkpoint under {output_dir}")
    return checkpoints[-1]


def maybe_existing_path(raw: str | None) -> Path | None:
    if not raw:
        return None
    candidates = [Path(raw)]
    if not Path(raw).is_absolute():
        candidates.insert(0, ROOT / raw)
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def restore_checkpoint(candidates: list[str | None], dest: Path, *, label: str) -> Path:
    for raw in candidates:
        source = maybe_existing_path(raw)
        if source is None:
            continue
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, dest)
        print(f"restored_{label}={dest}", flush=True)
        return dest
    raise FileNotFoundError(
        f"Could not restore {label}. Tried: " + ", ".join(str(item) for item in candidates if item)
    )


def restore_primitive_checkpoint(run_dir: Path, *, n_symbols: int) -> Path:
    curve_summary_path = Path(
        os.environ.get(
            "STAGE5_CHAIN_CORRECTED_PRIMITIVE_CURVE_SUMMARY",
            "outputs/stage5/stage5_synthetic_depth_primitive_curve_20260701_161524/summary.json",
        )
    )
    curve = read_json(curve_summary_path)
    row = next((item for item in curve.get("runs", []) if int(item.get("n_symbols", -1)) == int(n_symbols)), None)
    if row is None:
        raise RuntimeError(f"No primitive curve row found for n_symbols={n_symbols} in {curve_summary_path}")
    run_summary = read_json(row["summary_path"])
    return restore_checkpoint(
        [run_summary.get("checkpoint_drive_backup"), run_summary.get("checkpoint"), row.get("checkpoint")],
        run_dir / "restored" / f"primitive_n{n_symbols}_checkpoint.pt",
        label=f"primitive_n{n_symbols}_checkpoint",
    )


def restore_microtest_checkpoint(run_dir: Path) -> tuple[Path, dict[str, Any]]:
    micro_summary_path = Path(
        os.environ.get(
            "STAGE5_CHAIN_CORRECTED_MICROTEST_SUMMARY",
            "outputs/stage5/stage5_split_bridge_microtest_20260702_154804/summary.json",
        )
    )
    summary = read_json(micro_summary_path)
    stages = summary.get("stages", [])
    if not stages:
        raise RuntimeError(f"No stages found in {micro_summary_path}")
    final_stage = stages[-1]
    checkpoint = restore_checkpoint(
        [final_stage.get("checkpoint_drive_backup"), final_stage.get("checkpoint")],
        run_dir / "restored" / "split_bridge_microtest_final.pt",
        label="split_bridge_microtest_final",
    )
    return checkpoint, summary


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in root_path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def write_jsonl(path: str | Path, rows: list[dict[str, Any]]) -> None:
    out = root_path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("".join(json.dumps(row, ensure_ascii=True) + "\n" for row in rows), encoding="utf-8")


def filter_depth_jsonl(source: Path, dest: Path, *, max_depth: int) -> dict[str, Any]:
    rows = read_jsonl(source)
    kept = [row for row in rows if int(row.get("depth", row.get("synthetic_depth", 0))) <= int(max_depth)]
    write_jsonl(dest, kept)
    counts: dict[str, int] = {}
    for row in kept:
        key = str(int(row.get("depth", row.get("synthetic_depth", 0))))
        counts[key] = counts.get(key, 0) + 1
    return {
        "source": path_for_cli(source),
        "path": path_for_cli(dest),
        "max_depth": max_depth,
        "rows": len(kept),
        "depth_counts": dict(sorted(counts.items(), key=lambda item: int(item[0]))),
    }


def limit_rows_per_depth(source: Path, dest: Path, *, rows_per_depth: int) -> dict[str, Any]:
    kept: list[dict[str, Any]] = []
    counts: dict[str, int] = {}
    for row in read_jsonl(source):
        key = str(int(row.get("depth", row.get("synthetic_depth", 0))))
        if counts.get(key, 0) >= rows_per_depth:
            continue
        kept.append(row)
        counts[key] = counts.get(key, 0) + 1
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


def matrix_total_hits(matrix: dict[str, Any]) -> dict[str, int]:
    correct = 0
    total = 0
    for by_loop in matrix.get("matrix", {}).values():
        for cell in by_loop.values():
            correct += int(cell.get("correct", 0))
            total += int(cell.get("total", 0))
    return {"correct": correct, "total": total}


def eval_active_checkpoint(
    run_dir: Path,
    *,
    name: str,
    checkpoint: Path,
    data_jsonl: Path,
    prediction_space: str,
    prompt_style: str,
    loop_counts: str,
    threshold: float,
    dtype: str,
) -> dict[str, Any]:
    rows_path = run_dir / "eval" / f"{name}_active_rows.jsonl"
    summary_path = run_dir / "eval" / f"{name}_active_summary.json"
    proc = run(
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
            prediction_space,
            "--prompt_style",
            prompt_style,
            "--split",
            os.environ.get("STAGE5_CHAIN_CORRECTED_LAYER_SPLIT", "6,18"),
            "--bridge_projection_mode",
            "split",
            "--lora_rank",
            "0",
            "--dtype",
            dtype,
            "--adapter_dtype",
            "float32",
            "--device",
            os.environ.get("DEVICE", "cuda"),
        ]
    )
    log_path = run_dir / "eval" / f"{name}_active_eval.log"
    log_path.write_text(proc.stdout or "", encoding="utf-8")
    summary = read_json(summary_path)
    return {
        "name": name,
        "data_jsonl": path_for_cli(data_jsonl),
        "active_rows": path_for_cli(rows_path),
        "active_summary": path_for_cli(summary_path),
        "eval_log": path_for_cli(log_path),
        "prediction_space": prediction_space,
        "prompt_style": prompt_style,
        "active_diagonal": active_diag(summary),
        "active_diagonal_min": active_diag_min(summary),
        "active_total": summary.get("active_total", {}),
        "above_diagonal": summary.get("above_diagonal", {}),
    }


def eval_matrix_checkpoint(
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
    proc = run(
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
            "--split",
            os.environ.get("STAGE5_CHAIN_CORRECTED_LAYER_SPLIT", "6,18"),
            "--bridge_projection_mode",
            "split",
            "--lora_rank",
            "0",
            "--dtype",
            dtype,
            "--adapter_dtype",
            "float32",
            "--device",
            os.environ.get("DEVICE", "cuda"),
        ]
    )
    log_path = run_dir / "eval" / f"{name}_matrix_eval.log"
    log_path.write_text(proc.stdout or "", encoding="utf-8")
    matrix = read_json(summary_path)
    return {
        "name": name,
        "data_jsonl": path_for_cli(data_jsonl),
        "matrix_rows": path_for_cli(rows_path),
        "matrix_summary": path_for_cli(summary_path),
        "eval_log": path_for_cli(log_path),
        "total_hits": matrix_total_hits(matrix),
        "frontier_by_loop": matrix.get("frontier_by_loop", {}),
    }


def write_training_config(
    run_dir: Path,
    *,
    stage_name: str,
    train_jsonl: Path,
    resume_from: Path,
    max_loops: int,
    max_steps: int,
    dtype: str,
) -> Path:
    lr = float(os.environ.get("STAGE5_CHAIN_CORRECTED_LR", "1e-5"))
    cfg = {
        "model_name": os.environ.get("MODEL_NAME", "Qwen/Qwen2.5-0.5B-Instruct"),
        "dtype": dtype,
        "adapter_dtype": "float32",
        "attn_implementation": os.environ.get("ATTN_IMPLEMENTATION", "default"),
        "layer_split": os.environ.get("STAGE5_CHAIN_CORRECTED_LAYER_SPLIT", "6,18"),
        "max_length": int(os.environ.get("STAGE5_CHAIN_CORRECTED_MAX_LENGTH", "512")),
        "max_loops": int(max_loops),
        "loop_loss_mode": "per_loop_labels",
        "initial_halt_prob": 0.15,
        "beta": 0.0,
        "halt_target_nll_weight": 0.0,
        "loop_control_ce_weight": 0.0,
        "batch_size": int(os.environ.get("STAGE5_CHAIN_CORRECTED_BATCH_SIZE", "1")),
        "optimizer": "adamw",
        "learning_rate": lr,
        "adamw_lr": lr,
        "weight_decay": float(os.environ.get("STAGE5_CHAIN_CORRECTED_WEIGHT_DECAY", "0.0")),
        "max_grad_norm": float(os.environ.get("STAGE5_CHAIN_CORRECTED_MAX_GRAD_NORM", "0.5")),
        "max_steps": int(max_steps),
        "save_every": 0,
        "log_every": int(os.environ.get("STAGE5_CHAIN_CORRECTED_LOG_EVERY", "100")),
        "bridge_projection_mode": "split",
        "bridge_prelude_lr_multiplier": float(
            os.environ.get("STAGE5_CHAIN_CORRECTED_PRELUDE_LR_MULTIPLIER", "10.0")
        ),
        "bridge_prelude_weight_decay": 0.0,
        "bridge_prelude_lr_include_norm": env_flag("STAGE5_CHAIN_CORRECTED_PRELUDE_LR_INCLUDE_NORM", "0"),
        "bridge_prelude_grad_multiplier": 1.0,
        "train_on_prompt": False,
        "gradient_checkpointing": True,
        "output_dir": path_for_cli(run_dir / "train" / stage_name),
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
        "synthetic_phase": "chain_scaled_corrected",
        "synthetic_stage": stage_name,
    }
    path = run_dir / f"{stage_name}_train_config.yaml"
    write_yaml(path, cfg)
    return path


def train_stage(
    run_dir: Path,
    *,
    stage_name: str,
    train_jsonl: Path,
    resume_from: Path,
    max_loops: int,
    max_steps: int,
    dtype: str,
) -> dict[str, Any]:
    config_path = write_training_config(
        run_dir,
        stage_name=stage_name,
        train_jsonl=train_jsonl,
        resume_from=resume_from,
        max_loops=max_loops,
        max_steps=max_steps,
        dtype=dtype,
    )
    proc = run(
        [
            sys.executable,
            "training/train_unfrozen_recurrent.py",
            "--config",
            path_for_cli(config_path),
            "--train_jsonl",
            path_for_cli(train_jsonl),
            "--device",
            os.environ.get("DEVICE", "cuda"),
        ]
    )
    log_path = run_dir / "train" / f"{stage_name}_train_unfrozen_recurrent.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text(proc.stdout or "", encoding="utf-8")
    checkpoint = latest_checkpoint(run_dir / "train" / stage_name)
    training_summary = read_json(run_dir / "train" / stage_name / "train_unfrozen_recurrent_summary.json")
    return {
        "stage_name": stage_name,
        "max_loops": max_loops,
        "max_steps": max_steps,
        "train_jsonl": path_for_cli(train_jsonl),
        "train_config": path_for_cli(config_path),
        "train_log": path_for_cli(log_path),
        "training_summary": path_for_cli(run_dir / "train" / stage_name / "train_unfrozen_recurrent_summary.json"),
        "optimizer_setup": training_summary.get("optimizer_setup", {}),
        "bridge_prelude_weight_stats": training_summary.get("bridge_prelude_weight_stats", {}),
        "checkpoint": path_for_cli(checkpoint),
        "checkpoint_abs": str(checkpoint),
    }


def maybe_backup_checkpoint_to_drive(checkpoint: Path, *, run_id: str, stage_name: str) -> str | None:
    if not env_flag("STAGE5_CHAIN_CORRECTED_BACKUP_CHECKPOINTS_TO_DRIVE", "1"):
        return None
    dest_dir = DRIVE_CHECKPOINT_ROOT / run_id / stage_name
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / checkpoint.name
    shutil.copy2(checkpoint, dest)
    print(f"checkpoint_drive_backup={dest}", flush=True)
    return str(dest)


def write_summary_markdown(run_dir: Path, payload: dict[str, Any]) -> None:
    lines = [
        f"# Corrected Scaled Synthetic-Depth Chain Run - {payload['run_id']}",
        "",
        f"- Status: `{payload['status']}`",
        f"- N symbols: `{payload['n_symbols']}`",
        f"- Rows/depth train: `{payload['rows_per_depth']}`",
        f"- Heldout rows/depth eval: `{payload['heldout_rows_per_depth']}`",
        f"- Primary threshold: `{payload['threshold']}`",
        "",
    ]
    if payload.get("microtest_active_readout"):
        micro = payload["microtest_active_readout"]
        lines.extend(
            [
                "## Existing Microtest Active-Label Readout",
                "",
                f"- Train active diagonal: `{micro.get('train', {}).get('active_diagonal')}`",
                f"- Test active diagonal: `{micro.get('test', {}).get('active_diagonal')}`",
                "",
            ]
        )
    for stage in payload.get("stages", []):
        lines.extend(
            [
                f"## {stage['stage_name']}",
                "",
                f"- Checkpoint: `{stage.get('checkpoint')}`",
                f"- Drive backup: `{stage.get('checkpoint_drive_backup')}`",
                f"- Train active diagonal: `{stage.get('train_active_eval', {}).get('active_diagonal')}`",
                f"- Heldout active diagonal: `{stage.get('heldout_active_eval', {}).get('active_diagonal')}`",
                f"- Heldout active min: `{stage.get('heldout_active_eval', {}).get('active_diagonal_min')}`",
                f"- Final-answer heldout hits: `{stage.get('heldout_final_eval', {}).get('total_hits')}`",
                f"- Bridge prelude stats: `{stage.get('bridge_prelude_weight_stats')}`",
                "",
            ]
        )
    (run_dir / "summary.md").write_text("\n".join(lines), encoding="utf-8")


def write_run_summary(run_dir: Path, payload: dict[str, Any]) -> None:
    write_json(run_dir / "summary.json", payload)
    write_summary_markdown(run_dir, payload)
    print((run_dir / "summary.md").read_text(encoding="utf-8"), flush=True)


def publish(run_dir: Path, *, message: str, update_pointer: bool = False) -> None:
    from colab.stage5_publish_utils import publishable_artifact_paths, update_current_source_summary

    paths = publishable_artifact_paths(run_dir)
    if update_pointer:
        paths.append(update_current_source_summary(ROOT, run_dir / "summary.json"))
    run(["git", "pull", "--rebase", "--autostash", "origin", "main"], check=False)
    for path in paths:
        if path.exists():
            run(["git", "add", "-f", path_for_cli(path)], check=False)
    diff = run(["git", "diff", "--cached", "--quiet"], check=False)
    if diff.returncode == 0:
        print("No corrected chain artifacts to publish.", flush=True)
        return
    run(["git", "commit", "-m", message])
    push = run(["git", "push", "origin", "main"], check=False)
    if push.returncode == 0:
        return
    run(["git", "pull", "--rebase", "--autostash", "origin", "main"])
    run(["git", "push", "origin", "main"])


def run_microtest_active_readout(run_dir: Path, *, loop_counts: str, threshold: float) -> dict[str, Any]:
    if env_flag("STAGE5_CHAIN_CORRECTED_SKIP_MICRO_READOUT", "0"):
        return {"status": "skipped"}
    checkpoint, micro = restore_microtest_checkpoint(run_dir)
    data_root = root_path(micro["data_summary"]).parent
    dtype = os.environ.get("STAGE5_CHAIN_CORRECTED_MICRO_DTYPE", "float32")
    return {
        "status": "finished",
        "source_summary": os.environ.get(
            "STAGE5_CHAIN_CORRECTED_MICROTEST_SUMMARY",
            "outputs/stage5/stage5_split_bridge_microtest_20260702_154804/summary.json",
        ),
        "checkpoint": path_for_cli(checkpoint),
        "train": eval_active_checkpoint(
            run_dir,
            name="microtest_train_choice_active",
            checkpoint=checkpoint,
            data_jsonl=data_root / "train_chain_mcq.jsonl",
            prediction_space="choice_labels",
            prompt_style="with_options",
            loop_counts=loop_counts,
            threshold=threshold,
            dtype=dtype,
        ),
        "test": eval_active_checkpoint(
            run_dir,
            name="microtest_test_choice_active",
            checkpoint=checkpoint,
            data_jsonl=data_root / "test_chain_mcq.jsonl",
            prediction_space="choice_labels",
            prompt_style="with_options",
            loop_counts=loop_counts,
            threshold=threshold,
            dtype=dtype,
        ),
    }


def main() -> int:
    run_id = os.environ.get("STAGE5_CHAIN_CORRECTED_RUN_ID") or time.strftime(
        "stage5_chain_scaled_corrected_%Y%m%d_%H%M%S"
    )
    run_dir = ROOT / "outputs" / "stage5" / run_id
    n_symbols = int(os.environ.get("STAGE5_CHAIN_CORRECTED_N_SYMBOLS", "16"))
    max_depth = int(os.environ.get("STAGE5_CHAIN_CORRECTED_MAX_DEPTH", "4"))
    rows_per_depth = int(os.environ.get("STAGE5_CHAIN_CORRECTED_ROWS_PER_DEPTH", "256"))
    heldout_rows_per_depth = int(os.environ.get("STAGE5_CHAIN_CORRECTED_HELDOUT_ROWS_PER_DEPTH", "64"))
    train_eval_rows_per_depth = int(os.environ.get("STAGE5_CHAIN_CORRECTED_TRAIN_EVAL_ROWS_PER_DEPTH", "64"))
    seed = int(os.environ.get("STAGE5_CHAIN_CORRECTED_SEED", "20260702"))
    threshold = float(os.environ.get("STAGE5_CHAIN_CORRECTED_THRESHOLD", "0.71"))
    loop_counts = os.environ.get("STAGE5_CHAIN_CORRECTED_EVAL_LOOPS", "1,2,3,4")
    dtype = os.environ.get("STAGE5_CHAIN_CORRECTED_DTYPE", "bfloat16")

    run_dir.mkdir(parents=True, exist_ok=True)
    summary: dict[str, Any] = {
        "kind": "stage5_chain_scaled_corrected",
        "run_id": run_id,
        "status": "started",
        "n_symbols": n_symbols,
        "max_depth": max_depth,
        "rows_per_depth": rows_per_depth,
        "heldout_rows_per_depth": heldout_rows_per_depth,
        "train_eval_rows_per_depth": train_eval_rows_per_depth,
        "seed": seed,
        "threshold": threshold,
        "eval_loops": loop_counts,
        "dtype": dtype,
        "prediction_space": "full_symbols",
        "prompt_style": "question_only",
        "bridge_projection_mode": "split",
        "prelude_lr_multiplier": float(os.environ.get("STAGE5_CHAIN_CORRECTED_PRELUDE_LR_MULTIPLIER", "10.0")),
        "microtest_active_readout": None,
        "stages": [],
    }
    write_run_summary(run_dir, summary)

    summary["microtest_active_readout"] = run_microtest_active_readout(
        run_dir,
        loop_counts=loop_counts,
        threshold=threshold,
    )
    summary["status"] = "microtest_active_readout_finished"
    write_run_summary(run_dir, summary)
    publish(run_dir, message=f"Record active-label microtest readout {run_id} [skip ci]")

    primitive_checkpoint = restore_primitive_checkpoint(run_dir, n_symbols=n_symbols)
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
            str(seed),
            "--num_choices",
            "4",
            "--max_target_loops",
            str(max_depth),
        ]
    )
    train_le2 = data_dir / "train_chain_symbol_depth_le2_sft.jsonl"
    train_le4 = data_dir / "train_chain_symbol_depth_le4_sft.jsonl"
    train_active_eval = data_dir / f"train_chain_mcq_eval{train_eval_rows_per_depth}.jsonl"
    heldout_active_eval = data_dir / f"test_chain_mcq_heldout{heldout_rows_per_depth}.jsonl"
    train_final_eval = data_dir / f"train_mcq_eval{train_eval_rows_per_depth}.jsonl"
    heldout_final_eval = data_dir / f"test_mcq_heldout{heldout_rows_per_depth}.jsonl"
    filters = {
        "train_depth_le2": filter_depth_jsonl(data_dir / "train_chain_symbol_sft.jsonl", train_le2, max_depth=2),
        "train_depth_le4": filter_depth_jsonl(data_dir / "train_chain_symbol_sft.jsonl", train_le4, max_depth=max_depth),
        "train_active_eval": limit_rows_per_depth(
            data_dir / "train_chain_mcq.jsonl",
            train_active_eval,
            rows_per_depth=train_eval_rows_per_depth,
        ),
        "heldout_active_eval": limit_rows_per_depth(
            data_dir / "test_chain_mcq.jsonl",
            heldout_active_eval,
            rows_per_depth=heldout_rows_per_depth,
        ),
        "train_final_eval": limit_rows_per_depth(
            data_dir / "train_mcq.jsonl",
            train_final_eval,
            rows_per_depth=train_eval_rows_per_depth,
        ),
        "heldout_final_eval": limit_rows_per_depth(
            data_dir / "test_mcq.jsonl",
            heldout_final_eval,
            rows_per_depth=heldout_rows_per_depth,
        ),
    }
    summary.update(
        {
            "status": "dataset_ready",
            "primitive_checkpoint": path_for_cli(primitive_checkpoint),
            "data_summary": path_for_cli(data_dir / "summary.json"),
            "filters": filters,
        }
    )
    write_run_summary(run_dir, summary)
    publish(run_dir, message=f"Record corrected chain dataset {run_id} [skip ci]")

    stage12 = train_stage(
        run_dir,
        stage_name="chain_scaled_corrected_depth_le2",
        train_jsonl=train_le2,
        resume_from=primitive_checkpoint,
        max_loops=2,
        max_steps=int(os.environ.get("STAGE5_CHAIN_CORRECTED_STAGE12_STEPS", "2000")),
        dtype=dtype,
    )
    stage12_checkpoint = Path(stage12["checkpoint_abs"])
    stage12["checkpoint_drive_backup"] = maybe_backup_checkpoint_to_drive(
        stage12_checkpoint,
        run_id=run_id,
        stage_name="chain_scaled_corrected_depth_le2",
    )
    stage12["train_active_eval"] = eval_active_checkpoint(
        run_dir,
        name="chain_scaled_corrected_depth_le2_train_full_symbol_active",
        checkpoint=stage12_checkpoint,
        data_jsonl=train_active_eval,
        prediction_space="full_symbols",
        prompt_style="question_only",
        loop_counts=loop_counts,
        threshold=threshold,
        dtype=dtype,
    )
    stage12["heldout_active_eval"] = eval_active_checkpoint(
        run_dir,
        name="chain_scaled_corrected_depth_le2_heldout_full_symbol_active",
        checkpoint=stage12_checkpoint,
        data_jsonl=heldout_active_eval,
        prediction_space="full_symbols",
        prompt_style="question_only",
        loop_counts=loop_counts,
        threshold=threshold,
        dtype=dtype,
    )
    stage12["heldout_final_eval"] = eval_matrix_checkpoint(
        run_dir,
        name="chain_scaled_corrected_depth_le2_heldout_final_answer",
        checkpoint=stage12_checkpoint,
        data_jsonl=heldout_final_eval,
        loop_counts=loop_counts,
        threshold=threshold,
        dtype=dtype,
    )
    stage12.pop("checkpoint_abs", None)
    summary.update({"status": "stage_depth_le2_finished", "stages": [stage12]})
    write_run_summary(run_dir, summary)
    publish(run_dir, message=f"Record corrected chain depth<=2 {run_id} [skip ci]")

    stage1234 = train_stage(
        run_dir,
        stage_name="chain_scaled_corrected_depth_le4",
        train_jsonl=train_le4,
        resume_from=stage12_checkpoint,
        max_loops=max_depth,
        max_steps=int(os.environ.get("STAGE5_CHAIN_CORRECTED_STAGE1234_STEPS", "4000")),
        dtype=dtype,
    )
    stage1234_checkpoint = Path(stage1234["checkpoint_abs"])
    stage1234["checkpoint_drive_backup"] = maybe_backup_checkpoint_to_drive(
        stage1234_checkpoint,
        run_id=run_id,
        stage_name="chain_scaled_corrected_depth_le4",
    )
    stage1234["train_active_eval"] = eval_active_checkpoint(
        run_dir,
        name="chain_scaled_corrected_depth_le4_train_full_symbol_active",
        checkpoint=stage1234_checkpoint,
        data_jsonl=train_active_eval,
        prediction_space="full_symbols",
        prompt_style="question_only",
        loop_counts=loop_counts,
        threshold=threshold,
        dtype=dtype,
    )
    stage1234["heldout_active_eval"] = eval_active_checkpoint(
        run_dir,
        name="chain_scaled_corrected_depth_le4_heldout_full_symbol_active",
        checkpoint=stage1234_checkpoint,
        data_jsonl=heldout_active_eval,
        prediction_space="full_symbols",
        prompt_style="question_only",
        loop_counts=loop_counts,
        threshold=threshold,
        dtype=dtype,
    )
    stage1234["heldout_final_eval"] = eval_matrix_checkpoint(
        run_dir,
        name="chain_scaled_corrected_depth_le4_heldout_final_answer",
        checkpoint=stage1234_checkpoint,
        data_jsonl=heldout_final_eval,
        loop_counts=loop_counts,
        threshold=threshold,
        dtype=dtype,
    )
    stage1234.pop("checkpoint_abs", None)
    heldout_min = float(stage1234["heldout_active_eval"]["active_diagonal_min"])
    summary.update(
        {
            "status": "finished",
            "stages": [stage12, stage1234],
            "decision_gate": {
                "primary_metric": "heldout active-label diagonal minimum through depth 4",
                "threshold": threshold,
                "heldout_active_diagonal_min": heldout_min,
                "pass": heldout_min >= threshold,
                "heldout_active_diagonal": stage1234["heldout_active_eval"]["active_diagonal"],
                "train_active_diagonal": stage1234["train_active_eval"]["active_diagonal"],
                "heldout_above_diagonal": stage1234["heldout_active_eval"].get("above_diagonal", {}),
            },
        }
    )
    write_run_summary(run_dir, summary)
    publish(run_dir, message=f"Record Stage 5 corrected scaled chain run {run_id} [skip ci]", update_pointer=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
