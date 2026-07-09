"""Run Stage 5 natural-surface transfer Experiments 0 and 1.

Experiment 0 is a frozen natural-surface baseline: evaluate the synthetic
mechanism checkpoint on relay and pointer verbal surfaces without training.

Experiment 1 is rung-zero verbal SFT: continue from the strongest synthetic
mechanism checkpoint on relay verbal rows plus symbolic rehearsal, then rerun
the same readouts.  The checkpoint is backed up to Drive, but only lightweight
summaries and sampled rows are published to GitHub.
"""

from __future__ import annotations

import json
import hashlib
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

from colab.stage5_chain_consolidation_utils import (
    DRIVE_CHECKPOINT_ROOT,
    backup_checkpoint_to_drive,
    latest_checkpoint,
    path_for_cli,
    publish_run,
    read_json,
    root_path,
    write_json,
)


DEFAULT_DATA_SUMMARY = "outputs/stage5/stage5_natural_surface_transfer_20260708_230229/summary.json"
DEFAULT_INIT_SOURCE_SUMMARY = "outputs/stage5/stage5_n24_support12_rung_20260707_140139/summary.json"
DEFAULT_SUPPORT6_SOURCE_SUMMARY = "outputs/stage5/stage5_support6_seed26_plateau_20260708_130654_dose2000/summary.json"


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


def write_yaml(path: str | Path, payload: dict[str, Any]) -> None:
    out = root_path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")


def read_jsonl(path: str | Path, *, limit: int | None = None) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in root_path(path).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        rows.append(json.loads(line))
        if limit is not None and len(rows) >= limit:
            break
    return rows


def write_jsonl(path: str | Path, rows: list[dict[str, Any]]) -> None:
    out = root_path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("".join(json.dumps(row, ensure_ascii=True) + "\n" for row in rows), encoding="utf-8")


def existing_path(raw: str | Path | None) -> Path | None:
    if not raw:
        return None
    candidate = Path(raw)
    candidates = [candidate]
    if not candidate.is_absolute():
        candidates.insert(0, ROOT / candidate)
    for item in candidates:
        if item.exists():
            return item
    return None


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def restore_checkpoint(candidates: list[str | Path | None], dest: Path, *, label: str) -> tuple[Path, dict[str, Any]]:
    for raw in candidates:
        source = existing_path(raw)
        if source is None:
            continue
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, dest)
        source_sha256 = sha256_file(source)
        restored_sha256 = sha256_file(dest)
        if source_sha256 != restored_sha256:
            raise RuntimeError(
                f"Restored checkpoint hash mismatch for {label}: source={source_sha256} restored={restored_sha256}"
            )
        print(f"restored_{label}={dest}", flush=True)
        print(f"restored_{label}_source={source}", flush=True)
        print(f"restored_{label}_sha256={restored_sha256}", flush=True)
        return dest, {
            "selected_checkpoint_reference": str(raw),
            "selected_checkpoint_source": str(source),
            "selected_checkpoint_sha256": restored_sha256,
        }
    tried = ", ".join(str(item) for item in candidates if item)
    raise FileNotFoundError(f"Could not restore {label}; tried: {tried}")


def checkpoint_candidates_from_summary(summary: dict[str, Any], *, preferred_step: int | None = None) -> list[str | None]:
    candidates: list[str | None] = []
    if preferred_step is not None:
        for row in summary.get("checkpoint_evals") or []:
            if int(row.get("step", -1)) == int(preferred_step):
                candidates.extend(
                    [
                        row.get("checkpoint_drive_backup"),
                        row.get("checkpoint"),
                    ]
                )
    candidates.extend(
        [
            summary.get("final_checkpoint_drive_backup"),
            summary.get("checkpoint_drive_backup"),
            summary.get("final_checkpoint"),
            summary.get("checkpoint"),
            summary.get("ramp_checkpoint_drive_backup"),
            summary.get("ramp_checkpoint"),
        ]
    )
    init = summary.get("init_checkpoint_metadata") or {}
    candidates.extend(
        [
            init.get("source_final_checkpoint_drive_backup"),
            init.get("source_checkpoint_drive_backup"),
            init.get("source_final_checkpoint"),
            init.get("source_checkpoint"),
        ]
    )
    return candidates


def restore_summary_checkpoint(
    source_summary: str | Path,
    dest: Path,
    *,
    label: str,
    preferred_step: int | None = None,
) -> tuple[Path, dict[str, Any]]:
    summary_path = root_path(source_summary)
    summary = read_json(summary_path)
    checkpoint, restore_metadata = restore_checkpoint(
        checkpoint_candidates_from_summary(summary, preferred_step=preferred_step),
        dest,
        label=label,
    )
    metadata = {
        "source_summary": path_for_cli(summary_path),
        "source_run_id": summary.get("run_id"),
        "source_kind": summary.get("kind"),
        "preferred_step": preferred_step,
        "restored_checkpoint": path_for_cli(checkpoint),
        "drive_checkpoint_root": str(DRIVE_CHECKPOINT_ROOT),
    }
    metadata.update(restore_metadata)
    return checkpoint, metadata


def verify_expected_init_checkpoint(metadata: dict[str, Any]) -> None:
    expected_run_id = os.environ.get(
        "STAGE5_NATURAL_TRANSFER_EXPECTED_INIT_RUN_ID",
        "stage5_n24_support12_rung_20260707_140139",
    )
    expected_step = int(os.environ.get("STAGE5_NATURAL_TRANSFER_EXPECTED_INIT_STEP", "6000"))
    expected_sha256 = os.environ.get("STAGE5_NATURAL_TRANSFER_EXPECTED_INIT_SHA256", "").strip()
    actual_run_id = metadata.get("source_run_id")
    actual_step = metadata.get("preferred_step")
    actual_sha256 = str(metadata.get("selected_checkpoint_sha256") or "")
    if actual_run_id != expected_run_id:
        raise RuntimeError(f"Wrong natural-transfer init run_id: got {actual_run_id!r}, expected {expected_run_id!r}")
    if int(actual_step or -1) != expected_step:
        raise RuntimeError(f"Wrong natural-transfer init step: got {actual_step!r}, expected {expected_step!r}")
    if expected_sha256 and actual_sha256 != expected_sha256:
        raise RuntimeError(
            "Wrong natural-transfer init checkpoint hash: "
            f"got {actual_sha256}, expected {expected_sha256}"
        )
    print(
        json.dumps(
            {
                "kind": "natural_transfer_init_checkpoint_verified",
                "expected_run_id": expected_run_id,
                "actual_run_id": actual_run_id,
                "expected_step": expected_step,
                "actual_step": actual_step,
                "selected_checkpoint_reference": metadata.get("selected_checkpoint_reference"),
                "selected_checkpoint_source": metadata.get("selected_checkpoint_source"),
                "selected_checkpoint_sha256": actual_sha256,
                "pinned_sha256_enforced": bool(expected_sha256),
            },
            indent=2,
        ),
        flush=True,
    )


def data_paths(data_summary_path: str | Path) -> dict[str, Path]:
    summary = read_json(data_summary_path)
    data_summary = read_json(summary["data_summary"])
    data_dir = root_path(summary["data_summary"]).parent
    files = data_summary.get("files") or summary.get("dataset", {}).get("files") or {}
    required = [
        "rung0_train_mix_chain_symbol_sft",
        "relay_test_chain_mcq",
        "pointer_test_chain_mcq",
        "synthetic_rehearsal_chain_symbol_sft",
    ]
    missing = [name for name in required if name not in files]
    if missing:
        raise KeyError(f"Natural-surface data summary missing files: {missing}")
    out = {name: data_dir / files[name] for name in required}
    out["data_summary"] = root_path(summary["data_summary"])
    return out


def compact_rows(rows_path: Path, *, sample_rows: int) -> dict[str, Any]:
    if not rows_path.exists():
        return {"status": "missing", "path": path_for_cli(rows_path)}
    lines = [line for line in rows_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    sample_path = rows_path.with_name(rows_path.stem + "_sample.jsonl")
    sample = []
    for line in lines[:sample_rows]:
        row = json.loads(line)
        row.pop("scores", None)
        sample.append(row)
    write_jsonl(sample_path, sample)
    keep_full = env_flag("STAGE5_NATURAL_TRANSFER_KEEP_FULL_ACTIVE_ROWS", "0")
    if not keep_full:
        rows_path.unlink()
    return {
        "status": "kept_full" if keep_full else "sampled_and_deleted_full",
        "full_rows_path": path_for_cli(rows_path),
        "full_rows_count": len(lines),
        "sample_path": path_for_cli(sample_path),
        "sample_rows": len(sample),
    }


def existing_row_manifest(rows_path: Path) -> dict[str, Any]:
    sample_path = rows_path.with_name(rows_path.stem + "_sample.jsonl")
    if sample_path.exists():
        return {
            "status": "reused_sample",
            "full_rows_path": path_for_cli(rows_path),
            "sample_path": path_for_cli(sample_path),
            "sample_rows": len(read_jsonl(sample_path)),
        }
    if rows_path.exists():
        return compact_rows(
            rows_path,
            sample_rows=int(os.environ.get("STAGE5_NATURAL_TRANSFER_SAMPLE_ACTIVE_ROWS", "256")),
        )
    return {
        "status": "summary_only_reused",
        "full_rows_path": path_for_cli(rows_path),
        "sample_path": path_for_cli(sample_path),
    }


def active_diag(summary: dict[str, Any]) -> dict[str, float]:
    return {str(depth): float(value) for depth, value in (summary.get("active_diagonal") or {}).items()}


def active_diag_min(summary: dict[str, Any], *, depths: range | None = None) -> float:
    diag = active_diag(summary)
    if depths is not None:
        values = [diag[str(depth)] for depth in depths if str(depth) in diag]
    else:
        values = list(diag.values())
    return min(values) if values else 0.0


def eval_active_checkpoint(
    run_dir: Path,
    *,
    name: str,
    checkpoint: Path,
    data_jsonl: Path,
    loop_counts: str,
    value_prefix: str,
    dtype: str,
) -> dict[str, Any]:
    rows_path = run_dir / "eval" / f"{name}_active_rows.jsonl"
    summary_path = run_dir / "eval" / f"{name}_active_summary.json"
    if env_flag("STAGE5_NATURAL_TRANSFER_REUSE_EXISTING", "1") and summary_path.exists():
        print(f"reusing_active_eval={summary_path}", flush=True)
        summary = read_json(summary_path)
        return {
            "name": name,
            "data_jsonl": path_for_cli(data_jsonl),
            "checkpoint": path_for_cli(checkpoint),
            "active_summary": path_for_cli(summary_path),
            "eval_log": path_for_cli(run_dir / "eval" / f"{name}_active_eval.log"),
            "row_manifest": existing_row_manifest(rows_path),
            "loop_counts": loop_counts,
            "value_prefix": value_prefix,
            "active_diagonal": active_diag(summary),
            "active_diagonal_min": active_diag_min(summary),
            "active_total": summary.get("active_total", {}),
            "above_diagonal": summary.get("above_diagonal", {}),
        }
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
            os.environ.get("STAGE5_NATURAL_TRANSFER_THRESHOLD", "0.71"),
            "--prediction_space",
            "full_symbols",
            "--prompt_style",
            "question_only",
            "--value_prefix",
            value_prefix,
            "--split",
            os.environ.get("STAGE5_NATURAL_TRANSFER_LAYER_SPLIT", "6,18"),
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
            "--progress_every",
            os.environ.get("STAGE5_NATURAL_TRANSFER_PROGRESS_EVERY", "64"),
        ]
    )
    log_path = run_dir / "eval" / f"{name}_active_eval.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text(proc.stdout or "", encoding="utf-8")
    summary = read_json(summary_path)
    row_manifest = compact_rows(
        rows_path,
        sample_rows=int(os.environ.get("STAGE5_NATURAL_TRANSFER_SAMPLE_ACTIVE_ROWS", "256")),
    )
    return {
        "name": name,
        "data_jsonl": path_for_cli(data_jsonl),
        "checkpoint": path_for_cli(checkpoint),
        "active_summary": path_for_cli(summary_path),
        "eval_log": path_for_cli(log_path),
        "row_manifest": row_manifest,
        "loop_counts": loop_counts,
        "value_prefix": value_prefix,
        "active_diagonal": active_diag(summary),
        "active_diagonal_min": active_diag_min(summary),
        "active_total": summary.get("active_total", {}),
        "above_diagonal": summary.get("above_diagonal", {}),
    }


def run_artifact_check(
    run_dir: Path,
    *,
    name: str,
    checkpoint: Path,
    data_jsonl: Path,
    max_loops: int,
    value_prefix: str,
    dtype: str,
) -> dict[str, Any]:
    out = run_dir / "eval" / f"{name}_artifact_check.json"
    if env_flag("STAGE5_NATURAL_TRANSFER_REUSE_EXISTING", "1") and out.exists():
        print(f"reusing_artifact_check={out}", flush=True)
        return read_json(out)
    run(
        [
            sys.executable,
            "eval/eval_synthetic_depth_artifact_check.py",
            "--data_jsonl",
            path_for_cli(data_jsonl),
            "--checkpoint",
            path_for_cli(checkpoint),
            "--output_summary",
            path_for_cli(out),
            "--max_loops",
            str(max_loops),
            "--prediction_space",
            "full_symbols",
            "--prompt_style",
            "question_only",
            "--value_prefix",
            value_prefix,
            "--split",
            os.environ.get("STAGE5_NATURAL_TRANSFER_LAYER_SPLIT", "6,18"),
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
    return read_json(out)


def write_training_config(
    run_dir: Path,
    *,
    train_jsonl: Path,
    resume_from: Path,
    max_steps: int,
    max_loops: int,
    dtype: str,
) -> Path:
    lr = float(os.environ.get("STAGE5_NATURAL_TRANSFER_LR", "1e-5"))
    cfg = {
        "model_name": os.environ.get("MODEL_NAME", "Qwen/Qwen2.5-0.5B-Instruct"),
        "dtype": dtype,
        "adapter_dtype": "float32",
        "attn_implementation": os.environ.get("ATTN_IMPLEMENTATION", "default"),
        "layer_split": os.environ.get("STAGE5_NATURAL_TRANSFER_LAYER_SPLIT", "6,18"),
        "max_length": int(os.environ.get("STAGE5_NATURAL_TRANSFER_MAX_LENGTH", "640")),
        "max_loops": int(max_loops),
        "loop_loss_mode": "per_loop_labels",
        "initial_halt_prob": 0.15,
        "beta": 0.0,
        "halt_target_nll_weight": 0.0,
        "loop_control_ce_weight": 0.0,
        "batch_size": int(os.environ.get("STAGE5_NATURAL_TRANSFER_BATCH_SIZE", "1")),
        "optimizer": "adamw",
        "learning_rate": lr,
        "adamw_lr": lr,
        "weight_decay": float(os.environ.get("STAGE5_NATURAL_TRANSFER_WEIGHT_DECAY", "0.0")),
        "max_grad_norm": float(os.environ.get("STAGE5_NATURAL_TRANSFER_MAX_GRAD_NORM", "0.5")),
        "max_steps": int(max_steps),
        "save_every": int(os.environ.get("STAGE5_NATURAL_TRANSFER_SAVE_EVERY", "2000")),
        "log_every": int(os.environ.get("STAGE5_NATURAL_TRANSFER_LOG_EVERY", "100")),
        "bridge_projection_mode": "split",
        "bridge_prelude_lr_multiplier": float(
            os.environ.get("STAGE5_NATURAL_TRANSFER_PRELUDE_LR_MULTIPLIER", "10.0")
        ),
        "bridge_prelude_weight_decay": 0.0,
        "bridge_prelude_lr_include_norm": env_flag("STAGE5_NATURAL_TRANSFER_PRELUDE_LR_INCLUDE_NORM", "0"),
        "bridge_prelude_grad_multiplier": 1.0,
        "train_on_prompt": False,
        "gradient_checkpointing": True,
        "output_dir": path_for_cli(run_dir / "train" / "verbal_rung_zero"),
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
        "synthetic_phase": "natural_surface_transfer",
        "synthetic_stage": "verbal_rung_zero",
    }
    path = run_dir / "verbal_rung_zero_train_config.yaml"
    write_yaml(path, cfg)
    return path


def train_verbal_rung_zero(
    run_dir: Path,
    *,
    train_jsonl: Path,
    resume_from: Path,
    max_steps: int,
    max_loops: int,
    dtype: str,
) -> dict[str, Any]:
    output_dir = run_dir / "train" / "verbal_rung_zero"
    summary_path = output_dir / "train_unfrozen_recurrent_summary.json"
    if env_flag("STAGE5_NATURAL_TRANSFER_REUSE_EXISTING", "1") and summary_path.exists():
        checkpoint = latest_checkpoint(output_dir)
        training_summary = read_json(summary_path)
        print(f"reusing_verbal_rung_zero_checkpoint={checkpoint}", flush=True)
        return {
            "stage_name": "verbal_rung_zero",
            "train_jsonl": path_for_cli(train_jsonl),
            "train_config": path_for_cli(run_dir / "verbal_rung_zero_train_config.yaml"),
            "train_log": path_for_cli(run_dir / "train" / "verbal_rung_zero_train.log"),
            "training_summary": path_for_cli(summary_path),
            "checkpoint": path_for_cli(checkpoint),
            "checkpoint_drive_backup": None,
            "optimizer_setup": training_summary.get("optimizer_setup", {}),
            "bridge_prelude_weight_stats": training_summary.get("bridge_prelude_weight_stats", {}),
            "interval_checkpoints": training_summary.get("interval_checkpoints", []),
            "reused_existing": True,
        }
    config_path = write_training_config(
        run_dir,
        train_jsonl=train_jsonl,
        resume_from=resume_from,
        max_steps=max_steps,
        max_loops=max_loops,
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
    log_path = run_dir / "train" / "verbal_rung_zero_train.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text(proc.stdout or "", encoding="utf-8")
    checkpoint = latest_checkpoint(output_dir)
    training_summary = read_json(output_dir / "train_unfrozen_recurrent_summary.json")
    drive_backup = backup_checkpoint_to_drive(
        checkpoint,
        run_id=run_dir.name,
        stage_name="verbal_rung_zero",
        enabled=env_flag("STAGE5_NATURAL_TRANSFER_BACKUP_CHECKPOINTS_TO_DRIVE", "1"),
    )
    return {
        "stage_name": "verbal_rung_zero",
        "train_jsonl": path_for_cli(train_jsonl),
        "train_config": path_for_cli(config_path),
        "train_log": path_for_cli(log_path),
        "training_summary": path_for_cli(output_dir / "train_unfrozen_recurrent_summary.json"),
        "checkpoint": path_for_cli(checkpoint),
        "checkpoint_drive_backup": drive_backup,
        "optimizer_setup": training_summary.get("optimizer_setup", {}),
        "bridge_prelude_weight_stats": training_summary.get("bridge_prelude_weight_stats", {}),
        "interval_checkpoints": training_summary.get("interval_checkpoints", []),
    }


def min_depths(eval_payload: dict[str, Any], start: int, end: int) -> float:
    return active_diag_min({"active_diagonal": eval_payload.get("active_diagonal", {})}, depths=range(start, end + 1))


def score_experiment(
    *,
    frozen: dict[str, Any],
    post: dict[str, Any] | None,
    train_depth_max: int,
    eval_depth_max: int,
) -> dict[str, Any]:
    relay_frozen = frozen["n24"]["relay"]
    pointer_frozen = frozen["n24"]["pointer"]
    decision = {
        "experiment_0": {
            "relay_depth_1_to_8_min": min_depths(relay_frozen, 1, min(8, eval_depth_max)),
            "relay_depth_9_to_12_min": min_depths(relay_frozen, 9, eval_depth_max),
            "pointer_depth_1_to_8_min": min_depths(pointer_frozen, 1, min(8, eval_depth_max)),
            "pointer_depth_9_to_12_min": min_depths(pointer_frozen, 9, eval_depth_max),
            "status": "frozen_baseline_finished",
        },
        "experiment_1": None,
    }
    if post:
        relay_post = post["relay"]
        pointer_post = post["pointer"]
        synthetic_post = post["synthetic_rehearsal"]
        synthetic_frozen = frozen["n24"]["synthetic_rehearsal"]
        synth_delta = {
            depth: float(synthetic_post["active_diagonal"].get(depth, 0.0))
            - float(synthetic_frozen["active_diagonal"].get(depth, 0.0))
            for depth in synthetic_frozen["active_diagonal"]
        }
        decision["experiment_1"] = {
            "relay_train_depth_min": min_depths(relay_post, 1, train_depth_max),
            "relay_extrap_depth_min": min_depths(relay_post, train_depth_max + 1, eval_depth_max),
            "pointer_train_depth_min": min_depths(pointer_post, 1, train_depth_max),
            "pointer_extrap_depth_min": min_depths(pointer_post, train_depth_max + 1, eval_depth_max),
            "synthetic_rehearsal_min": min_depths(synthetic_post, 1, train_depth_max),
            "synthetic_rehearsal_delta_by_depth": synth_delta,
            "synthetic_rehearsal_min_delta": min(synth_delta.values()) if synth_delta else 0.0,
            "status": "verbal_rung_zero_finished",
        }
    return decision


def write_summary_markdown(run_dir: Path, payload: dict[str, Any]) -> None:
    decision = payload.get("decision_read", {})
    lines = [
        f"# Natural-Surface Transfer Rung Zero - {payload['run_id']}",
        "",
        f"- Status: `{payload['status']}`",
        f"- Init source: `{payload['init_source_summary']}`",
        f"- Data summary: `{payload['data_summary']}`",
        f"- Train steps: `{payload['train_steps']}`",
        f"- Train enabled: `{payload['run_train']}`",
        "",
        "## Experiment 0 - Frozen Natural Surface",
        "",
        f"- Relay diagonal: `{payload.get('frozen_baseline', {}).get('n24', {}).get('relay', {}).get('active_diagonal')}`",
        f"- Pointer diagonal: `{payload.get('frozen_baseline', {}).get('n24', {}).get('pointer', {}).get('active_diagonal')}`",
        f"- Synthetic rehearsal diagonal: `{payload.get('frozen_baseline', {}).get('n24', {}).get('synthetic_rehearsal', {}).get('active_diagonal')}`",
        f"- Decision read: `{decision.get('experiment_0')}`",
        "",
    ]
    if payload.get("train_stage"):
        lines.extend(
            [
                "## Experiment 1 - Verbal Rung Zero",
                "",
                f"- Checkpoint: `{payload['train_stage'].get('checkpoint')}`",
                f"- Drive backup: `{payload['train_stage'].get('checkpoint_drive_backup')}`",
                f"- Relay diagonal: `{payload.get('post_train_evals', {}).get('relay', {}).get('active_diagonal')}`",
                f"- Pointer diagonal: `{payload.get('post_train_evals', {}).get('pointer', {}).get('active_diagonal')}`",
                f"- Synthetic rehearsal diagonal: `{payload.get('post_train_evals', {}).get('synthetic_rehearsal', {}).get('active_diagonal')}`",
                f"- Decision read: `{decision.get('experiment_1')}`",
                "",
            ]
        )
    (run_dir / "summary.md").write_text("\n".join(lines), encoding="utf-8")


def write_run_summary(run_dir: Path, payload: dict[str, Any]) -> None:
    write_json(run_dir / "summary.json", payload)
    write_summary_markdown(run_dir, payload)
    print((run_dir / "summary.md").read_text(encoding="utf-8"), flush=True)


def run_frozen_evals(
    run_dir: Path,
    *,
    checkpoint: Path,
    paths: dict[str, Path],
    dtype: str,
    train_depth_max: int,
    eval_depth_max: int,
    label: str,
) -> dict[str, Any]:
    loop_counts_eval = ",".join(str(idx) for idx in range(1, eval_depth_max + 1))
    loop_counts_train = ",".join(str(idx) for idx in range(1, train_depth_max + 1))
    return {
        "artifact_check": run_artifact_check(
            run_dir,
            name=f"{label}_relay",
            checkpoint=checkpoint,
            data_jsonl=paths["relay_test_chain_mcq"],
            max_loops=eval_depth_max,
            value_prefix="name:",
            dtype=dtype,
        ),
        "relay": eval_active_checkpoint(
            run_dir,
            name=f"{label}_relay",
            checkpoint=checkpoint,
            data_jsonl=paths["relay_test_chain_mcq"],
            loop_counts=loop_counts_eval,
            value_prefix="name:",
            dtype=dtype,
        ),
        "pointer": eval_active_checkpoint(
            run_dir,
            name=f"{label}_pointer",
            checkpoint=checkpoint,
            data_jsonl=paths["pointer_test_chain_mcq"],
            loop_counts=loop_counts_eval,
            value_prefix="name:",
            dtype=dtype,
        ),
        "synthetic_rehearsal": eval_active_checkpoint(
            run_dir,
            name=f"{label}_synthetic_rehearsal",
            checkpoint=checkpoint,
            data_jsonl=paths["synthetic_rehearsal_chain_symbol_sft"],
            loop_counts=loop_counts_train,
            value_prefix="letter:",
            dtype=dtype,
        ),
    }


def main() -> int:
    run_id = os.environ.get("STAGE5_NATURAL_TRANSFER_RUN_ID") or time.strftime(
        "stage5_natural_surface_transfer_rung0_%Y%m%d_%H%M%S"
    )
    run_dir = ROOT / "outputs" / "stage5" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    data_summary = os.environ.get("STAGE5_NATURAL_TRANSFER_DATA_SUMMARY", DEFAULT_DATA_SUMMARY)
    init_summary = os.environ.get("STAGE5_NATURAL_TRANSFER_INIT_SOURCE_SUMMARY", DEFAULT_INIT_SOURCE_SUMMARY)
    support6_summary = os.environ.get("STAGE5_NATURAL_TRANSFER_SUPPORT6_SOURCE_SUMMARY", DEFAULT_SUPPORT6_SOURCE_SUMMARY)
    train_steps = int(os.environ.get("STAGE5_NATURAL_TRANSFER_TRAIN_STEPS", "8000"))
    train_depth_max = int(os.environ.get("STAGE5_NATURAL_TRANSFER_TRAIN_MAX_DEPTH", "8"))
    eval_depth_max = int(os.environ.get("STAGE5_NATURAL_TRANSFER_EVAL_MAX_DEPTH", "12"))
    train_max_loops = int(os.environ.get("STAGE5_NATURAL_TRANSFER_TRAIN_MAX_LOOPS", str(train_depth_max)))
    dtype = os.environ.get("STAGE5_NATURAL_TRANSFER_DTYPE", "bfloat16")
    run_train = env_flag("STAGE5_NATURAL_TRANSFER_RUN_TRAIN", "1")

    paths = data_paths(data_summary)
    n24_checkpoint, n24_meta = restore_summary_checkpoint(
        init_summary,
        run_dir / "restored" / "n24_support12_step6000.pt",
        label="n24_support12_step6000",
        preferred_step=int(os.environ.get("STAGE5_NATURAL_TRANSFER_INIT_STEP", "6000")),
    )
    verify_expected_init_checkpoint(n24_meta)

    payload: dict[str, Any] = {
        "kind": "stage5_natural_surface_transfer_rung0",
        "run_id": run_id,
        "status": "started",
        "data_summary": data_summary,
        "init_source_summary": init_summary,
        "init_checkpoint_metadata": n24_meta,
        "support6_source_summary": support6_summary,
        "train_steps": train_steps,
        "train_depth_max": train_depth_max,
        "eval_depth_max": eval_depth_max,
        "train_max_loops": train_max_loops,
        "dtype": dtype,
        "threshold": float(os.environ.get("STAGE5_NATURAL_TRANSFER_THRESHOLD", "0.71")),
        "run_train": run_train,
        "data_paths": {key: path_for_cli(value) for key, value in paths.items()},
        "frozen_baseline": {},
        "train_stage": None,
        "post_train_evals": None,
        "decision_read": {},
    }
    write_run_summary(run_dir, payload)

    payload["frozen_baseline"]["n24"] = run_frozen_evals(
        run_dir,
        checkpoint=n24_checkpoint,
        paths=paths,
        dtype=dtype,
        train_depth_max=train_depth_max,
        eval_depth_max=eval_depth_max,
        label="frozen_n24",
    )
    if env_flag("STAGE5_NATURAL_TRANSFER_RUN_SUPPORT6_FROZEN", "0"):
        support6_checkpoint, support6_meta = restore_summary_checkpoint(
            support6_summary,
            run_dir / "restored" / "support6_seed26.pt",
            label="support6_seed26",
        )
        payload["frozen_baseline"]["support6_seed26"] = {
            "metadata": support6_meta,
            **run_frozen_evals(
                run_dir,
                checkpoint=support6_checkpoint,
                paths=paths,
                dtype=dtype,
                train_depth_max=train_depth_max,
                eval_depth_max=eval_depth_max,
                label="frozen_support6_seed26",
            ),
        }
    payload["status"] = "experiment_0_frozen_baseline_finished"
    payload["decision_read"] = score_experiment(
        frozen=payload["frozen_baseline"],
        post=None,
        train_depth_max=train_depth_max,
        eval_depth_max=eval_depth_max,
    )
    write_run_summary(run_dir, payload)
    publish_run(run_dir, message=f"Record natural-surface frozen baseline {run_id} [skip ci]", update_pointer=False)

    if not run_train:
        payload["status"] = "finished_frozen_only"
        write_run_summary(run_dir, payload)
        publish_run(run_dir, message=f"Record natural-surface frozen-only final {run_id} [skip ci]", update_pointer=False)
        return 0

    payload["train_stage"] = train_verbal_rung_zero(
        run_dir,
        train_jsonl=paths["rung0_train_mix_chain_symbol_sft"],
        resume_from=n24_checkpoint,
        max_steps=train_steps,
        max_loops=train_max_loops,
        dtype=dtype,
    )
    trained_checkpoint = root_path(payload["train_stage"]["checkpoint"])
    payload["post_train_evals"] = {
        "relay": eval_active_checkpoint(
            run_dir,
            name="post_rung0_relay",
            checkpoint=trained_checkpoint,
            data_jsonl=paths["relay_test_chain_mcq"],
            loop_counts=",".join(str(idx) for idx in range(1, eval_depth_max + 1)),
            value_prefix="name:",
            dtype=dtype,
        ),
        "pointer": eval_active_checkpoint(
            run_dir,
            name="post_rung0_pointer",
            checkpoint=trained_checkpoint,
            data_jsonl=paths["pointer_test_chain_mcq"],
            loop_counts=",".join(str(idx) for idx in range(1, eval_depth_max + 1)),
            value_prefix="name:",
            dtype=dtype,
        ),
        "synthetic_rehearsal": eval_active_checkpoint(
            run_dir,
            name="post_rung0_synthetic_rehearsal",
            checkpoint=trained_checkpoint,
            data_jsonl=paths["synthetic_rehearsal_chain_symbol_sft"],
            loop_counts=",".join(str(idx) for idx in range(1, train_depth_max + 1)),
            value_prefix="letter:",
            dtype=dtype,
        ),
    }
    payload["status"] = "finished"
    payload["decision_read"] = score_experiment(
        frozen=payload["frozen_baseline"],
        post=payload["post_train_evals"],
        train_depth_max=train_depth_max,
        eval_depth_max=eval_depth_max,
    )
    write_run_summary(run_dir, payload)
    publish_run(run_dir, message=f"Record natural-surface rung-zero transfer {run_id} [skip ci]", update_pointer=False)
    print(json.dumps({"run_id": run_id, "status": payload["status"], "summary": path_for_cli(run_dir / "summary.json")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
