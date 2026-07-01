"""Colab cell: staged synthetic-depth staircase from a proven primitive.

This is Phase 2 after the primitive curve. It resumes from the best depth-1
N=16 primitive checkpoint, trains target-loop CE on depths <=2, then trains
target-loop CE on depths <=4. The readout is an A(depth, forced_loop) matrix.
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
from google.colab import drive, runtime, userdata


STAGE5_SYNTHETIC_DEPTH_STAGED_STAIRCASE_CELL_VERSION = "synthetic_depth_staged_staircase_v1"
# Safety marker: Phase 2 resumes from primitive N=16 and uses loop_loss_mode=target.
REPO = "mshapiro123/recurrent-qwen-svgd"
ROOT = Path("/content/recurrent-qwen-svgd")
DRIVE_CHECKPOINT_ROOT = Path("/content/drive/MyDrive/recurrent-qwen-svgd-checkpoints")


def secret(*names: str) -> str | None:
    for name in names:
        value = os.environ.get(name)
        if value:
            return value
        try:
            value = userdata.get(name)
        except Exception:
            value = None
        if value:
            return value
    return None


GH_TOKEN = secret("GH_TOKEN", "GITHUB_TOKEN")
HF_TOKEN = secret("HF_TOKEN", "HUGGINGFACE_HUB_TOKEN")
assert GH_TOKEN, "Missing GH_TOKEN or GITHUB_TOKEN in Colab secrets."
if HF_TOKEN:
    os.environ["HF_TOKEN"] = HF_TOKEN
    os.environ["HUGGINGFACE_HUB_TOKEN"] = HF_TOKEN
    print("HF token loaded", flush=True)
else:
    print("HF token missing; model downloads will use anonymous Hub access.", flush=True)


def redact(text: str) -> str:
    out = str(text)
    for token in (GH_TOKEN, HF_TOKEN):
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


def sync_repo() -> None:
    clone_url = f"https://x-access-token:{GH_TOKEN}@github.com/{REPO}.git"
    if ROOT.exists():
        run(["git", "remote", "set-url", "origin", clone_url])
        run(["git", "fetch", "origin", "main"])
        run(["git", "checkout", "main"])
        run(["git", "reset", "--hard", "origin/main"])
    else:
        run(["git", "clone", clone_url, ROOT], cwd=Path("/content"))
    run(["git", "config", "user.email", "colab-runner@local"])
    run(["git", "config", "user.name", "Colab Runner"])
    run(["git", "log", "--oneline", "-5"], check=False)
    root_str = str(ROOT)
    if root_str not in sys.path:
        sys.path.insert(0, root_str)


def require_gpu_runtime() -> None:
    if shutil.which("nvidia-smi") is None:
        raise RuntimeError("Attach a GPU runtime first. L4/T4 is sufficient for this staged staircase.")
    run(["nvidia-smi"], cwd=Path("/content"), check=False)


def read_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def write_yaml(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")


def latest_checkpoint(output_dir: Path) -> Path:
    checkpoints = sorted(output_dir.glob("unfrozen_recurrent_step_*.pt"), key=lambda path: path.stat().st_mtime)
    if not checkpoints:
        raise FileNotFoundError(f"No checkpoint under {output_dir}")
    return checkpoints[-1]


def publish(paths: list[Path], *, message: str) -> None:
    run(["git", "pull", "--rebase", "--autostash", "origin", "main"], check=False)
    for path in paths:
        if path.exists():
            run(["git", "add", "-f", path_for_cli(path)], check=False)
    diff = run(["git", "diff", "--cached", "--quiet"], check=False)
    if diff.returncode == 0:
        print("No staged-staircase changes to publish.", flush=True)
        return
    run(["git", "commit", "-m", message])
    push = run(["git", "push", "origin", "main"], check=False)
    if push.returncode == 0:
        return
    run(["git", "pull", "--rebase", "--autostash", "origin", "main"])
    run(["git", "push", "origin", "main"])


def restore_primitive_checkpoint(run_dir: Path, *, n_symbols: int) -> Path:
    override = os.environ.get("STAGE5_SYNTH_STAIRCASE_PRIMITIVE_CHECKPOINT", "").strip()
    if override:
        candidate = Path(override)
        if not candidate.exists():
            raise FileNotFoundError(f"Primitive checkpoint override is missing: {candidate}")
        restored = run_dir / "restored" / "primitive_checkpoint.pt"
        restored.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(candidate, restored)
        return restored

    curve_summary_path = Path(
        os.environ.get(
            "STAGE5_SYNTH_STAIRCASE_PRIMITIVE_CURVE_SUMMARY",
            "outputs/stage5/stage5_synthetic_depth_primitive_curve_20260701_161524/summary.json",
        )
    )
    curve = read_json(curve_summary_path)
    chosen = None
    for row in curve.get("runs", []):
        if int(row.get("n_symbols", -1)) == int(n_symbols):
            chosen = row
            break
    if not chosen:
        raise RuntimeError(f"No primitive curve row found for n_symbols={n_symbols} in {curve_summary_path}")
    run_summary = read_json(chosen["summary_path"])
    candidates = [
        run_summary.get("checkpoint_drive_backup"),
        run_summary.get("checkpoint"),
        chosen.get("checkpoint"),
    ]
    for raw in candidates:
        if not raw:
            continue
        candidate = Path(raw)
        if candidate.exists():
            restored = run_dir / "restored" / "primitive_checkpoint.pt"
            restored.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(candidate, restored)
            return restored
    raise FileNotFoundError(
        "Could not restore primitive checkpoint. Tried: "
        + ", ".join(str(item) for item in candidates if item)
        + ". If this is a fresh runtime, approve Drive mount or set "
        "STAGE5_SYNTH_STAIRCASE_PRIMITIVE_CHECKPOINT."
    )


def filter_depth_jsonl(source: Path, dest: Path, *, max_depth: int) -> dict[str, Any]:
    rows = [json.loads(line) for line in source.read_text(encoding="utf-8").splitlines() if line.strip()]
    kept = [row for row in rows if int(row.get("depth", row.get("synthetic_depth", 0))) <= int(max_depth)]
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text("".join(json.dumps(row, ensure_ascii=True) + "\n" for row in kept), encoding="utf-8")
    counts: dict[str, int] = {}
    target_loop_counts: dict[str, int] = {}
    for row in kept:
        depth_key = str(int(row.get("depth", row.get("synthetic_depth", 0))))
        loop_key = str(int(row.get("target_loop_count", 0)))
        counts[depth_key] = counts.get(depth_key, 0) + 1
        target_loop_counts[loop_key] = target_loop_counts.get(loop_key, 0) + 1
    return {
        "source": path_for_cli(source),
        "path": path_for_cli(dest),
        "max_depth": max_depth,
        "rows": len(kept),
        "depth_counts": dict(sorted(counts.items(), key=lambda item: int(item[0]))),
        "target_loop_counts": dict(sorted(target_loop_counts.items(), key=lambda item: int(item[0]))),
    }


def write_training_config(
    run_dir: Path,
    *,
    stage_name: str,
    train_jsonl: Path,
    resume_from: Path,
    max_loops: int,
    max_steps: int,
    lr: float,
) -> Path:
    cfg = {
        "model_name": os.environ.get("MODEL_NAME", "Qwen/Qwen2.5-0.5B-Instruct"),
        "dtype": os.environ.get("DTYPE", "bfloat16"),
        "adapter_dtype": os.environ.get("ADAPTER_DTYPE", "float32"),
        "attn_implementation": os.environ.get("ATTN_IMPLEMENTATION", "default"),
        "layer_split": os.environ.get("STAGE5_SYNTH_STAIRCASE_LAYER_SPLIT", "6,18"),
        "max_length": int(os.environ.get("STAGE5_SYNTH_STAIRCASE_MAX_LENGTH", "512")),
        "max_loops": max_loops,
        "loop_loss_mode": "target",
        "initial_halt_prob": float(os.environ.get("STAGE5_SYNTH_STAIRCASE_INITIAL_HALT_PROB", "0.15")),
        "beta": 0.0,
        "halt_target_nll_weight": 0.0,
        "loop_control_ce_weight": 0.0,
        "batch_size": int(os.environ.get("STAGE5_SYNTH_STAIRCASE_BATCH_SIZE", "1")),
        "optimizer": os.environ.get("STAGE5_SYNTH_STAIRCASE_OPTIMIZER", "adamw"),
        "learning_rate": lr,
        "adamw_lr": lr,
        "weight_decay": float(os.environ.get("STAGE5_SYNTH_STAIRCASE_WEIGHT_DECAY", "0.0")),
        "max_grad_norm": float(os.environ.get("STAGE5_SYNTH_STAIRCASE_MAX_GRAD_NORM", "0.5")),
        "max_steps": max_steps,
        "save_every": int(os.environ.get("STAGE5_SYNTH_STAIRCASE_SAVE_EVERY", "0")),
        "log_every": int(os.environ.get("STAGE5_SYNTH_STAIRCASE_LOG_EVERY", "25")),
        "bridge_prelude_grad_multiplier": float(
            os.environ.get("STAGE5_SYNTH_STAIRCASE_BRIDGE_PRELUDE_GRAD_MULTIPLIER", "1.0")
        ),
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
            "start_loop": max_loops,
            "end_loop": max_loops,
            "schedule": "linear",
            "target_source": "row_capped",
            "ramp_compute": False,
        },
        "synthetic_train_jsonl": path_for_cli(train_jsonl),
        "synthetic_train_format": "mcq_option_text",
        "synthetic_phase": "staged_staircase",
        "synthetic_stage": stage_name,
    }
    config_path = run_dir / f"{stage_name}_train_config.yaml"
    write_yaml(config_path, cfg)
    return config_path


def run_training_stage(
    run_dir: Path,
    *,
    stage_name: str,
    train_jsonl: Path,
    resume_from: Path,
    max_loops: int,
    max_steps: int,
    lr: float,
) -> dict[str, Any]:
    config_path = write_training_config(
        run_dir,
        stage_name=stage_name,
        train_jsonl=train_jsonl,
        resume_from=resume_from,
        max_loops=max_loops,
        max_steps=max_steps,
        lr=lr,
    )
    train_proc = run(
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
    train_log = run_dir / "train" / f"{stage_name}_train_unfrozen_recurrent.log"
    train_log.parent.mkdir(parents=True, exist_ok=True)
    train_log.write_text(train_proc.stdout or "", encoding="utf-8")
    checkpoint = latest_checkpoint(run_dir / "train" / stage_name)
    training_summary = run_dir / "train" / stage_name / "train_unfrozen_recurrent_summary.json"
    return {
        "stage_name": stage_name,
        "max_loops": max_loops,
        "max_steps": max_steps,
        "learning_rate": lr,
        "train_jsonl": path_for_cli(train_jsonl),
        "train_config": path_for_cli(config_path),
        "train_log": path_for_cli(train_log),
        "training_summary": path_for_cli(training_summary),
        "checkpoint": path_for_cli(checkpoint),
        "checkpoint_abs": str(checkpoint),
    }


def eval_checkpoint(
    run_dir: Path,
    *,
    stage_name: str,
    checkpoint: Path,
    data_jsonl: Path,
    loop_counts: str,
    threshold: float,
) -> dict[str, Any]:
    matrix_jsonl = run_dir / "eval" / f"{stage_name}_matrix_rows.jsonl"
    matrix_summary = run_dir / "eval" / f"{stage_name}_matrix_summary.json"
    eval_proc = run(
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
            path_for_cli(matrix_jsonl),
            "--output_summary",
            path_for_cli(matrix_summary),
            "--loop_counts",
            loop_counts,
            "--threshold",
            str(threshold),
            "--score_target",
            "option_text",
            "--split",
            os.environ.get("STAGE5_SYNTH_STAIRCASE_LAYER_SPLIT", "6,18"),
            "--lora_rank",
            "0",
            "--dtype",
            os.environ.get("DTYPE", "bfloat16"),
            "--adapter_dtype",
            os.environ.get("ADAPTER_DTYPE", "float32"),
            "--device",
            os.environ.get("DEVICE", "cuda"),
        ]
    )
    eval_log = run_dir / "eval" / f"{stage_name}_eval_synthetic_depth_matrix.log"
    eval_log.write_text(eval_proc.stdout or "", encoding="utf-8")
    matrix = read_json(matrix_summary)
    return {
        "stage_name": stage_name,
        "matrix_rows": path_for_cli(matrix_jsonl),
        "matrix_summary": path_for_cli(matrix_summary),
        "eval_log": path_for_cli(eval_log),
        "matrix": matrix,
    }


def maybe_backup_checkpoint_to_drive(checkpoint: Path, *, run_id: str, stage_name: str) -> str | None:
    if not env_flag("STAGE5_SYNTH_STAIRCASE_BACKUP_CHECKPOINTS_TO_DRIVE", "0"):
        return None
    dest_dir = DRIVE_CHECKPOINT_ROOT / run_id / stage_name
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / checkpoint.name
    shutil.copy2(checkpoint, dest)
    print(f"checkpoint_drive_backup={dest}", flush=True)
    return str(dest)


def write_run_summary(run_dir: Path, payload: dict[str, Any]) -> None:
    write_json(run_dir / "summary.json", payload)
    lines = [
        f"# Synthetic Depth Staged Staircase - {payload['run_id']}",
        "",
        f"- Status: `{payload['status']}`",
        f"- N symbols: `{payload['n_symbols']}`",
        f"- Max depth: `{payload['max_depth']}`",
        f"- Primitive checkpoint: `{payload['primitive_checkpoint']}`",
        f"- Loop loss mode: `target`",
        "",
        "## Stage Matrices",
        "",
    ]
    for stage in payload.get("stages", []):
        matrix = stage.get("eval", {}).get("matrix", {})
        lines.extend(
            [
                f"### {stage['stage_name']}",
                "",
                f"- Checkpoint: `{stage.get('checkpoint')}`",
                f"- Drive backup: `{stage.get('checkpoint_drive_backup')}`",
                f"- Frontier by loop: `{matrix.get('frontier_by_loop')}`",
                f"- Strictly expands: `{matrix.get('frontier_strictly_expands')}`",
                "",
            ]
        )
    (run_dir / "summary.md").write_text("\n".join(lines), encoding="utf-8")
    print((run_dir / "summary.md").read_text(encoding="utf-8"), flush=True)


try:
    require_gpu_runtime()
    drive.mount("/content/drive", force_remount=False)
    sync_repo()
    os.chdir(ROOT)
    print(
        "STAGE5_SYNTHETIC_DEPTH_STAGED_STAIRCASE_CELL_VERSION="
        f"{STAGE5_SYNTHETIC_DEPTH_STAGED_STAIRCASE_CELL_VERSION}",
        flush=True,
    )
    run(["python", "-m", "pip", "install", "-q", "-r", "requirements.txt"])
    run(
        [
            sys.executable,
            "-m",
            "pytest",
            "-q",
            "tests/test_recurrent_wrapper_tiny.py::test_target_loop_loss_mode_uses_requested_loop_on_tiny_model",
            "tests/test_train_unfrozen_recurrent.py",
            "tests/test_synthetic_depth_task.py",
            "tests/test_eval_synthetic_depth_matrix.py",
        ]
    )

    run_id = os.environ.get("STAGE5_SYNTH_STAIRCASE_RUN_ID") or time.strftime(
        "stage5_synthetic_depth_staged_staircase_%Y%m%d_%H%M%S"
    )
    run_dir = ROOT / "outputs" / "stage5" / run_id
    n_symbols = int(os.environ.get("STAGE5_SYNTH_STAIRCASE_N_SYMBOLS", "16"))
    max_depth = int(os.environ.get("STAGE5_SYNTH_STAIRCASE_MAX_DEPTH", "4"))
    rows_per_depth = int(os.environ.get("STAGE5_SYNTH_STAIRCASE_ROWS_PER_DEPTH", "256"))
    seed = int(os.environ.get("STAGE5_SYNTH_STAIRCASE_SEED", "20260702"))
    threshold = float(os.environ.get("STAGE5_SYNTH_STAIRCASE_THRESHOLD", "0.71"))
    eval_loops = os.environ.get("STAGE5_SYNTH_STAIRCASE_EVAL_LOOPS", "1,2,3,4")

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
            os.environ.get("STAGE5_SYNTH_STAIRCASE_NUM_CHOICES", "4"),
            "--max_target_loops",
            str(max_depth),
        ]
    )
    source_train_jsonl = data_dir / "train_mcq_option_text_sft.jsonl"
    train_le2 = data_dir / "train_depth_le2_mcq_option_text_sft.jsonl"
    train_le4 = data_dir / "train_depth_le4_mcq_option_text_sft.jsonl"
    le2_summary = filter_depth_jsonl(source_train_jsonl, train_le2, max_depth=2)
    le4_summary = filter_depth_jsonl(source_train_jsonl, train_le4, max_depth=max_depth)

    dataset_payload = {
        "kind": "stage5_synthetic_depth_staged_staircase_dataset",
        "run_id": run_id,
        "n_symbols": n_symbols,
        "max_depth": max_depth,
        "rows_per_depth": rows_per_depth,
        "seed": seed,
        "primitive_checkpoint": path_for_cli(primitive_checkpoint),
        "data_summary": path_for_cli(data_dir / "summary.json"),
        "train_filters": {"depth_le2": le2_summary, "depth_le4": le4_summary},
        "loop_loss_mode": "target",
    }
    write_json(run_dir / "dataset_summary.json", dataset_payload)
    publish(
        [run_dir / "dataset_summary.json", data_dir / "summary.json", train_le2, train_le4],
        message=f"Record staged synthetic-depth dataset {run_id} [skip ci]",
    )

    stages: list[dict[str, Any]] = []
    stage1_steps = int(os.environ.get("STAGE5_SYNTH_STAIRCASE_STAGE12_STEPS", "500"))
    stage2_steps = int(os.environ.get("STAGE5_SYNTH_STAIRCASE_STAGE1234_STEPS", "1000"))
    lr = float(os.environ.get("STAGE5_SYNTH_STAIRCASE_LR", "1e-5"))

    stage12 = run_training_stage(
        run_dir,
        stage_name="depth_le2",
        train_jsonl=train_le2,
        resume_from=primitive_checkpoint,
        max_loops=2,
        max_steps=stage1_steps,
        lr=lr,
    )
    stage12_checkpoint = Path(stage12["checkpoint_abs"])
    stage12["checkpoint_drive_backup"] = maybe_backup_checkpoint_to_drive(
        stage12_checkpoint,
        run_id=run_id,
        stage_name="depth_le2",
    )
    stage12["eval"] = eval_checkpoint(
        run_dir,
        stage_name="depth_le2",
        checkpoint=stage12_checkpoint,
        data_jsonl=data_dir / "test_mcq.jsonl",
        loop_counts=eval_loops,
        threshold=threshold,
    )
    stage12.pop("checkpoint_abs", None)
    stages.append(stage12)
    partial_summary = {
        "kind": "stage5_synthetic_depth_staged_staircase",
        "run_id": run_id,
        "cell_version": STAGE5_SYNTHETIC_DEPTH_STAGED_STAIRCASE_CELL_VERSION,
        "status": "stage_depth_le2_finished",
        "n_symbols": n_symbols,
        "max_depth": max_depth,
        "primitive_checkpoint": path_for_cli(primitive_checkpoint),
        "dataset": dataset_payload,
        "stages": stages,
    }
    write_run_summary(run_dir, partial_summary)
    publish(
        [
            run_dir / "summary.json",
            run_dir / "summary.md",
            Path(stage12["train_config"]),
            Path(stage12["training_summary"]),
            Path(stage12["train_log"]),
            Path(stage12["eval"]["matrix_summary"]),
            Path(stage12["eval"]["matrix_rows"]),
            Path(stage12["eval"]["eval_log"]),
        ],
        message=f"Record staged synthetic-depth depth<=2 {run_id} [skip ci]",
    )

    stage1234 = run_training_stage(
        run_dir,
        stage_name="depth_le4",
        train_jsonl=train_le4,
        resume_from=stage12_checkpoint,
        max_loops=max_depth,
        max_steps=stage2_steps,
        lr=lr,
    )
    stage1234_checkpoint = Path(stage1234["checkpoint_abs"])
    stage1234["checkpoint_drive_backup"] = maybe_backup_checkpoint_to_drive(
        stage1234_checkpoint,
        run_id=run_id,
        stage_name="depth_le4",
    )
    stage1234["eval"] = eval_checkpoint(
        run_dir,
        stage_name="depth_le4",
        checkpoint=stage1234_checkpoint,
        data_jsonl=data_dir / "test_mcq.jsonl",
        loop_counts=eval_loops,
        threshold=threshold,
    )
    stage1234.pop("checkpoint_abs", None)
    stages.append(stage1234)

    final_summary = {
        "kind": "stage5_synthetic_depth_staged_staircase",
        "run_id": run_id,
        "cell_version": STAGE5_SYNTHETIC_DEPTH_STAGED_STAIRCASE_CELL_VERSION,
        "status": "finished",
        "n_symbols": n_symbols,
        "max_depth": max_depth,
        "rows_per_depth": rows_per_depth,
        "threshold": threshold,
        "eval_loops": eval_loops,
        "primitive_checkpoint": path_for_cli(primitive_checkpoint),
        "dataset": dataset_payload,
        "stages": stages,
        "interpretation_gate": {
            "success_signature": "depth_le4 matrix frontier rises with forced loop count after primitive warm start",
            "stage_depth_le4_frontier": stage1234["eval"]["matrix"].get("frontier_by_loop"),
            "stage_depth_le4_strictly_expands": stage1234["eval"]["matrix"].get("frontier_strictly_expands"),
        },
    }
    write_run_summary(run_dir, final_summary)
    publish(
        [
            run_dir / "summary.json",
            run_dir / "summary.md",
            run_dir / "dataset_summary.json",
            data_dir / "summary.json",
            train_le2,
            train_le4,
            Path(stage1234["train_config"]),
            Path(stage1234["training_summary"]),
            Path(stage1234["train_log"]),
            Path(stage1234["eval"]["matrix_summary"]),
            Path(stage1234["eval"]["matrix_rows"]),
            Path(stage1234["eval"]["eval_log"]),
        ],
        message=f"Record Stage 5 synthetic staged staircase {run_id} [skip ci]",
    )
    if env_flag("STAGE5_SYNTH_STAIRCASE_DISCONNECT", "0"):
        print("Disconnecting Colab runtime after staged staircase.", flush=True)
        runtime.unassign()
except Exception:
    print("Synthetic-depth staged staircase errored; leaving runtime connected.", flush=True)
    raise
