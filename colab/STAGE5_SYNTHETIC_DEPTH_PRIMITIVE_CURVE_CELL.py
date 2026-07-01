"""Colab cell: Phase 1 synthetic-depth primitive-generalization curve.

This target runs depth-1-only trainings at separate symbol-space sizes. It is
deliberately not a staircase run. The goal is to establish the single lookup
primitive before asking whether recurrence composes it into deeper reasoning.
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


STAGE5_SYNTHETIC_DEPTH_PRIMITIVE_CURVE_CELL_VERSION = "synthetic_depth_primitive_curve_v1"
# Safety marker: Phase 1 changes only N and keeps max_depth=1.
REPO = "mshapiro123/recurrent-qwen-svgd"
ROOT = Path("/content/recurrent-qwen-svgd")


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
    print("HF token missing; model downloads will be anonymous.", flush=True)


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
        print("\n".join(stdout.splitlines()[-160:]), flush=True)
        print("FAILED_COMMAND_TAIL_END", flush=True)
        raise subprocess.CalledProcessError(proc.returncode, proc.args, output=stdout)
    return proc


def env_flag(name: str, default: str = "0") -> bool:
    return os.environ.get(name, default).strip().lower() in {"1", "true", "yes", "y", "on"}


def parse_int_csv(value: str) -> list[int]:
    parsed = [int(item.strip()) for item in value.split(",") if item.strip()]
    if not parsed:
        raise ValueError("Expected at least one integer")
    return parsed


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
        raise RuntimeError("Attach a cheap GPU runtime first. L4/T4 is sufficient for the primitive curve.")
    run(["nvidia-smi"], cwd=Path("/content"), check=False)


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


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
        print("No primitive-curve changes to publish.", flush=True)
        return
    run(["git", "commit", "-m", message])
    push = run(["git", "push", "origin", "main"], check=False)
    if push.returncode == 0:
        return
    run(["git", "pull", "--rebase", "--autostash", "origin", "main"])
    run(["git", "push", "origin", "main"])


def maybe_backup_checkpoint_to_drive(checkpoint: Path, *, curve_id: str, run_id: str) -> str | None:
    if not env_flag("STAGE5_SYNTH_PRIMITIVE_BACKUP_CHECKPOINTS_TO_DRIVE", "0"):
        return None
    drive.mount("/content/drive", force_remount=False)
    dest_dir = Path("/content/drive/MyDrive/recurrent-qwen-svgd-checkpoints") / curve_id / run_id
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / checkpoint.name
    shutil.copy2(checkpoint, dest)
    print(f"checkpoint_drive_backup={dest}", flush=True)
    return str(dest)


def write_training_config(run_dir: Path, train_jsonl: Path, *, n_symbols: int, max_steps: int) -> Path:
    cfg = {
        "model_name": os.environ.get("MODEL_NAME", "Qwen/Qwen2.5-0.5B-Instruct"),
        "dtype": os.environ.get("DTYPE", "bfloat16"),
        "adapter_dtype": os.environ.get("ADAPTER_DTYPE", "float32"),
        "attn_implementation": os.environ.get("ATTN_IMPLEMENTATION", "default"),
        "layer_split": os.environ.get("STAGE5_SYNTH_PRIMITIVE_LAYER_SPLIT", "6,18"),
        "max_length": int(os.environ.get("STAGE5_SYNTH_PRIMITIVE_MAX_LENGTH", "512")),
        "max_loops": 1,
        "initial_halt_prob": float(os.environ.get("STAGE5_SYNTH_PRIMITIVE_INITIAL_HALT_PROB", "0.15")),
        "beta": 0.0,
        "halt_target_nll_weight": 0.0,
        "batch_size": int(os.environ.get("STAGE5_SYNTH_PRIMITIVE_BATCH_SIZE", "1")),
        "optimizer": os.environ.get("STAGE5_SYNTH_PRIMITIVE_OPTIMIZER", "adamw"),
        "learning_rate": float(os.environ.get("STAGE5_SYNTH_PRIMITIVE_LR", "2e-5")),
        "adamw_lr": float(os.environ.get("STAGE5_SYNTH_PRIMITIVE_ADAMW_LR", os.environ.get("STAGE5_SYNTH_PRIMITIVE_LR", "2e-5"))),
        "weight_decay": float(os.environ.get("STAGE5_SYNTH_PRIMITIVE_WEIGHT_DECAY", "0.0")),
        "max_grad_norm": float(os.environ.get("STAGE5_SYNTH_PRIMITIVE_MAX_GRAD_NORM", "0.5")),
        "max_steps": max_steps,
        "save_every": int(os.environ.get("STAGE5_SYNTH_PRIMITIVE_SAVE_EVERY", "0")),
        "log_every": int(os.environ.get("STAGE5_SYNTH_PRIMITIVE_LOG_EVERY", "25")),
        "bridge_prelude_grad_multiplier": float(os.environ.get("STAGE5_SYNTH_PRIMITIVE_BRIDGE_PRELUDE_GRAD_MULTIPLIER", "1.0")),
        "train_on_prompt": False,
        "gradient_checkpointing": True,
        "output_dir": path_for_cli(run_dir / "train" / "unfrozen"),
        "resume_from": None,
        "resume_lora": {"enabled": False},
        "merge_lora_before_unfreeze": False,
        "require_lora_loaded_before_merge": False,
        "train_auxiliary": {
            "bridge": True,
            "halting": True,
            "reentry_adapter": False,
            "latent": False,
        },
        "recurrence_curriculum": {
            "enabled": True,
            "start_loop": 1,
            "end_loop": 1,
            "schedule": "constant",
            "target_source": "row_capped",
            "ramp_compute": True,
        },
        "synthetic_train_jsonl": path_for_cli(train_jsonl),
        "synthetic_train_format": "mcq_option_text",
        "synthetic_phase": "primitive_curve",
        "synthetic_n_symbols": n_symbols,
    }
    config_path = run_dir / "synthetic_depth_train_config.yaml"
    write_yaml(config_path, cfg)
    return config_path


def run_one_primitive_setting(
    *,
    curve_id: str,
    n_symbols: int,
    rows_per_depth: int,
    max_steps: int,
    seed: int,
    primitive_bar: float,
    strong_bar: float,
) -> Path:
    run_id = f"{curve_id}_N{n_symbols}"
    run_dir = ROOT / "outputs" / "stage5" / run_id
    data_dir = run_dir / "data"
    print(f"\n===== primitive curve N={n_symbols} run_id={run_id} =====", flush=True)
    run(
        [
            sys.executable,
            "training/generate_synthetic_depth_task.py",
            "--output_dir",
            path_for_cli(data_dir),
            "--n_symbols",
            str(n_symbols),
            "--max_depth",
            "1",
            "--rows_per_depth",
            str(rows_per_depth),
            "--seed",
            str(seed),
            "--num_choices",
            os.environ.get("STAGE5_SYNTH_PRIMITIVE_NUM_CHOICES", "4"),
            "--max_target_loops",
            "1",
        ]
    )
    dataset_summary = data_dir / "summary.json"
    publish([dataset_summary], message=f"Record synthetic primitive dataset {run_id} [skip ci]")

    base_matrix_jsonl = run_dir / "eval" / "base_test_matrix_rows.jsonl"
    base_matrix_summary = run_dir / "eval" / "base_test_matrix_summary.json"
    base_eval_log = run_dir / "eval" / "base_eval_synthetic_depth_matrix.log"
    if env_flag("STAGE5_SYNTH_PRIMITIVE_RUN_BASE_EVAL", "1"):
        base_proc = run(
            [
                sys.executable,
                "eval/eval_synthetic_depth_matrix.py",
                "--mode",
                "base",
                "--data_jsonl",
                path_for_cli(data_dir / "test_mcq.jsonl"),
                "--output_jsonl",
                path_for_cli(base_matrix_jsonl),
                "--output_summary",
                path_for_cli(base_matrix_summary),
                "--threshold",
                str(primitive_bar),
                "--score_target",
                "option_text",
                "--dtype",
                os.environ.get("DTYPE", "bfloat16"),
                "--adapter_dtype",
                os.environ.get("ADAPTER_DTYPE", "float32"),
                "--device",
                os.environ.get("DEVICE", "cuda"),
            ]
        )
        base_eval_log.parent.mkdir(parents=True, exist_ok=True)
        base_eval_log.write_text(base_proc.stdout or "", encoding="utf-8")

    train_jsonl = data_dir / "train_mcq_option_text_sft.jsonl"
    config_path = write_training_config(run_dir, train_jsonl, n_symbols=n_symbols, max_steps=max_steps)
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
    train_log = run_dir / "train" / "train_unfrozen_recurrent.log"
    train_log.parent.mkdir(parents=True, exist_ok=True)
    train_log.write_text(train_proc.stdout or "", encoding="utf-8")
    checkpoint = latest_checkpoint(run_dir / "train" / "unfrozen")
    checkpoint_drive_backup = maybe_backup_checkpoint_to_drive(checkpoint, curve_id=curve_id, run_id=run_id)
    training_summary = run_dir / "train" / "unfrozen" / "train_unfrozen_recurrent_summary.json"

    matrix_jsonl = run_dir / "eval" / "test_matrix_rows.jsonl"
    matrix_summary = run_dir / "eval" / "test_matrix_summary.json"
    eval_proc = run(
        [
            sys.executable,
            "eval/eval_synthetic_depth_matrix.py",
            "--mode",
            "recurrent",
            "--data_jsonl",
            path_for_cli(data_dir / "test_mcq.jsonl"),
            "--checkpoint",
            path_for_cli(checkpoint),
            "--output_jsonl",
            path_for_cli(matrix_jsonl),
            "--output_summary",
            path_for_cli(matrix_summary),
            "--loop_counts",
            "1",
            "--threshold",
            str(primitive_bar),
            "--score_target",
            "option_text",
            "--split",
            os.environ.get("STAGE5_SYNTH_PRIMITIVE_LAYER_SPLIT", "6,18"),
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
    eval_log = run_dir / "eval" / "eval_synthetic_depth_matrix.log"
    eval_log.write_text(eval_proc.stdout or "", encoding="utf-8")

    matrix = read_json(matrix_summary)
    base_matrix = read_json(base_matrix_summary) if base_matrix_summary.exists() else None
    recurrent_acc = matrix["matrix"]["1"]["1"]["accuracy"]
    summary = {
        "kind": "stage5_synthetic_depth_primitive_curve_run",
        "run_id": run_id,
        "curve_id": curve_id,
        "cell_version": STAGE5_SYNTHETIC_DEPTH_PRIMITIVE_CURVE_CELL_VERSION,
        "status": "finished",
        "phase": "phase1_primitive_generalization",
        "n_symbols": n_symbols,
        "max_depth": 1,
        "max_loops": 1,
        "rows_per_depth": rows_per_depth,
        "max_steps": max_steps,
        "train_format": "mcq_option_text",
        "primitive_accuracy_bar": primitive_bar,
        "strong_accuracy_bar": strong_bar,
        "clears_primitive_bar": recurrent_acc >= primitive_bar,
        "clears_strong_bar": recurrent_acc >= strong_bar,
        "dataset_summary": path_for_cli(dataset_summary),
        "train_config": path_for_cli(config_path),
        "training_summary": path_for_cli(training_summary),
        "checkpoint": path_for_cli(checkpoint),
        "checkpoint_drive_backup": checkpoint_drive_backup,
        "base_matrix_rows": path_for_cli(base_matrix_jsonl) if base_matrix_jsonl.exists() else None,
        "base_matrix_summary": path_for_cli(base_matrix_summary) if base_matrix_summary.exists() else None,
        "base_matrix": base_matrix,
        "matrix_rows": path_for_cli(matrix_jsonl),
        "matrix_summary": path_for_cli(matrix_summary),
        "matrix": matrix,
    }
    write_json(run_dir / "summary.json", summary)
    lines = [
        f"# Synthetic Primitive Curve Run - N={n_symbols}",
        "",
        f"- Run: `{run_id}`",
        f"- Recurrent accuracy: `{recurrent_acc:.4f}`",
        f"- Clears primitive bar `{primitive_bar}`: `{summary['clears_primitive_bar']}`",
        f"- Clears strong bar `{strong_bar}`: `{summary['clears_strong_bar']}`",
        f"- Checkpoint Drive backup: `{checkpoint_drive_backup}`",
        "",
    ]
    (run_dir / "summary.md").write_text("\n".join(lines), encoding="utf-8")
    print((run_dir / "summary.md").read_text(encoding="utf-8"), flush=True)
    publish(
        [
            run_dir / "summary.json",
            run_dir / "summary.md",
            dataset_summary,
            config_path,
            training_summary,
            train_log,
            base_matrix_summary,
            base_matrix_jsonl,
            base_eval_log,
            matrix_summary,
            matrix_jsonl,
            eval_log,
        ],
        message=f"Record Stage 5 synthetic primitive curve N={n_symbols} {run_id} [skip ci]",
    )
    return run_dir / "summary.json"


try:
    require_gpu_runtime()
    sync_repo()
    os.chdir(ROOT)
    print(
        "STAGE5_SYNTHETIC_DEPTH_PRIMITIVE_CURVE_CELL_VERSION="
        f"{STAGE5_SYNTHETIC_DEPTH_PRIMITIVE_CURVE_CELL_VERSION}",
        flush=True,
    )
    run(["python", "-m", "pip", "install", "-q", "-r", "requirements.txt"])
    run(
        [
            sys.executable,
            "-m",
            "pytest",
            "-q",
            "tests/test_synthetic_depth_task.py",
            "tests/test_eval_synthetic_depth_matrix.py",
            "tests/test_synthetic_depth_primitive_curve.py",
        ]
    )

    curve_id = os.environ.get("STAGE5_SYNTH_PRIMITIVE_CURVE_ID") or time.strftime(
        "stage5_synthetic_depth_primitive_curve_%Y%m%d_%H%M%S"
    )
    n_values = parse_int_csv(os.environ.get("STAGE5_SYNTH_PRIMITIVE_N_VALUES", "8,12,16"))
    rows_per_depth = int(os.environ.get("STAGE5_SYNTH_PRIMITIVE_ROWS_PER_DEPTH", "256"))
    max_steps = int(os.environ.get("STAGE5_SYNTH_PRIMITIVE_MAX_STEPS", "500"))
    seed = int(os.environ.get("STAGE5_SYNTH_PRIMITIVE_SEED", "20260701"))
    primitive_bar = float(os.environ.get("STAGE5_SYNTH_PRIMITIVE_BAR", "0.71"))
    strong_bar = float(os.environ.get("STAGE5_SYNTH_PRIMITIVE_STRONG_BAR", "0.90"))

    summaries: list[Path] = []
    for n_symbols in n_values:
        summaries.append(
            run_one_primitive_setting(
                curve_id=curve_id,
                n_symbols=n_symbols,
                rows_per_depth=rows_per_depth,
                max_steps=max_steps,
                seed=seed,
                primitive_bar=primitive_bar,
                strong_bar=strong_bar,
            )
        )

    curve_dir = ROOT / "outputs" / "stage5" / curve_id
    curve_summary_json = curve_dir / "summary.json"
    curve_summary_md = curve_dir / "summary.md"
    run(
        [
            sys.executable,
            "colab/summarize_synthetic_depth_primitive_curve.py",
            "--summary_paths",
            *[path_for_cli(path) for path in summaries],
            "--output_json",
            path_for_cli(curve_summary_json),
            "--output_md",
            path_for_cli(curve_summary_md),
            "--primitive_bar",
            str(primitive_bar),
            "--strong_bar",
            str(strong_bar),
        ]
    )
    print(curve_summary_md.read_text(encoding="utf-8"), flush=True)
    publish(
        [curve_summary_json, curve_summary_md],
        message=f"Record Stage 5 synthetic primitive curve {curve_id} [skip ci]",
    )
    if env_flag("STAGE5_SYNTH_PRIMITIVE_DISCONNECT", "0"):
        print("Disconnecting Colab runtime after primitive curve.", flush=True)
        runtime.unassign()
except Exception:
    print("Synthetic primitive curve run errored; leaving runtime connected.", flush=True)
    raise
