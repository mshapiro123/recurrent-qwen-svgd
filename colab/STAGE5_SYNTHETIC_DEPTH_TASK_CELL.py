"""Colab cell: synthetic sequential-depth mechanism test.

This target asks a narrow question: after training on an iterated-function task
with a real dependent lookup chain, does forced recurrent loop count extend the
maximum solved depth?  The readout is the full A(depth, loop) matrix.
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
from google.colab import runtime, userdata


STAGE5_SYNTHETIC_DEPTH_TASK_CELL_VERSION = "synthetic_depth_task_v1"
# Safety marker: generated instances must preserve distinct_prefix_length_depth_plus_one.
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
        raise RuntimeError("Attach a cheap GPU runtime first. L4/T4 is sufficient for the synthetic-depth pilot.")
    run(["nvidia-smi"], check=False)


def write_yaml(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


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
        print("No synthetic-depth changes to publish.", flush=True)
        return
    run(["git", "commit", "-m", message])
    push = run(["git", "push", "origin", "main"], check=False)
    if push.returncode == 0:
        return
    run(["git", "pull", "--rebase", "--autostash", "origin", "main"])
    run(["git", "push", "origin", "main"])


def write_training_config(run_dir: Path, train_jsonl: Path) -> Path:
    max_loops = int(os.environ.get("STAGE5_SYNTH_DEPTH_MAX_LOOPS", "8"))
    cfg = {
        "model_name": os.environ.get("MODEL_NAME", "Qwen/Qwen2.5-0.5B-Instruct"),
        "dtype": os.environ.get("DTYPE", "bfloat16"),
        "adapter_dtype": os.environ.get("ADAPTER_DTYPE", "float32"),
        "attn_implementation": os.environ.get("ATTN_IMPLEMENTATION", "default"),
        "layer_split": os.environ.get("STAGE5_SYNTH_DEPTH_LAYER_SPLIT", "6,18"),
        "max_length": int(os.environ.get("STAGE5_SYNTH_DEPTH_MAX_LENGTH", "512")),
        "max_loops": max_loops,
        "initial_halt_prob": float(os.environ.get("STAGE5_SYNTH_DEPTH_INITIAL_HALT_PROB", "0.15")),
        "beta": float(os.environ.get("STAGE5_SYNTH_DEPTH_BETA", "0.02")),
        "halt_target_nll_weight": float(os.environ.get("STAGE5_SYNTH_DEPTH_HALT_TARGET_NLL_WEIGHT", "0.5")),
        "batch_size": int(os.environ.get("STAGE5_SYNTH_DEPTH_BATCH_SIZE", "1")),
        "optimizer": os.environ.get("STAGE5_SYNTH_DEPTH_OPTIMIZER", "adamw"),
        "learning_rate": float(os.environ.get("STAGE5_SYNTH_DEPTH_LR", "1e-5")),
        "adamw_lr": float(os.environ.get("STAGE5_SYNTH_DEPTH_ADAMW_LR", os.environ.get("STAGE5_SYNTH_DEPTH_LR", "1e-5"))),
        "weight_decay": float(os.environ.get("STAGE5_SYNTH_DEPTH_WEIGHT_DECAY", "0.0")),
        "max_grad_norm": float(os.environ.get("STAGE5_SYNTH_DEPTH_MAX_GRAD_NORM", "0.5")),
        "max_steps": int(os.environ.get("STAGE5_SYNTH_DEPTH_MAX_STEPS", "25")),
        "save_every": int(os.environ.get("STAGE5_SYNTH_DEPTH_SAVE_EVERY", "0")),
        "log_every": int(os.environ.get("STAGE5_SYNTH_DEPTH_LOG_EVERY", "5")),
        "bridge_prelude_grad_multiplier": float(os.environ.get("STAGE5_SYNTH_DEPTH_BRIDGE_PRELUDE_GRAD_MULTIPLIER", "2.0")),
        "train_on_prompt": False,
        "gradient_checkpointing": True,
        "output_dir": path_for_cli(run_dir / "train" / "unfrozen"),
        "resume_from": os.environ.get("STAGE5_SYNTH_DEPTH_RESUME_FROM", "").strip() or None,
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
            "start_loop": int(os.environ.get("STAGE5_SYNTH_DEPTH_START_LOOP", "1")),
            "end_loop": max_loops,
            "schedule": os.environ.get("STAGE5_SYNTH_DEPTH_LOOP_SCHEDULE", "linear"),
            "target_source": os.environ.get("STAGE5_SYNTH_DEPTH_TARGET_SOURCE", "row_capped"),
            "ramp_compute": True,
        },
        "synthetic_train_jsonl": path_for_cli(train_jsonl),
    }
    config_path = run_dir / "synthetic_depth_train_config.yaml"
    write_yaml(config_path, cfg)
    return config_path


def write_summary(run_dir: Path, payload: dict[str, Any]) -> None:
    (run_dir / "summary.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    lines = [
        f"# Synthetic Depth Mechanism Test - {payload['run_id']}",
        "",
        f"- Status: `{payload['status']}`",
        f"- Dataset: `{payload['dataset_summary']}`",
        f"- Checkpoint: `{payload.get('checkpoint')}`",
        f"- Matrix summary: `{payload.get('matrix_summary')}`",
        "",
    ]
    matrix = payload.get("matrix") or {}
    if matrix:
        lines.extend(
            [
                "## Matrix Readout",
                "",
                f"- Threshold: `{matrix.get('threshold')}`",
                f"- Frontier by loop: `{matrix.get('frontier_by_loop')}`",
                f"- Non-decreasing frontier: `{matrix.get('frontier_is_non_decreasing')}`",
                f"- Strictly expands: `{matrix.get('frontier_strictly_expands')}`",
                "",
            ]
        )
    (run_dir / "summary.md").write_text("\n".join(lines), encoding="utf-8")
    print((run_dir / "summary.md").read_text(encoding="utf-8"), flush=True)


try:
    require_gpu_runtime()
    sync_repo()
    os.chdir(ROOT)
    print(f"STAGE5_SYNTHETIC_DEPTH_TASK_CELL_VERSION={STAGE5_SYNTHETIC_DEPTH_TASK_CELL_VERSION}", flush=True)
    run(["python", "-m", "pip", "install", "-q", "-r", "requirements.txt"])
    run(
        [
            sys.executable,
            "-m",
            "pytest",
            "-q",
            "tests/test_synthetic_depth_task.py",
            "tests/test_eval_synthetic_depth_matrix.py",
        ]
    )

    run_id = os.environ.get("STAGE5_SYNTH_DEPTH_RUN_ID") or time.strftime("stage5_synthetic_depth_%Y%m%d_%H%M%S")
    run_dir = ROOT / "outputs" / "stage5" / run_id
    data_dir = run_dir / "data"
    max_depth = os.environ.get("STAGE5_SYNTH_DEPTH_MAX_DEPTH", "8")
    max_loops = os.environ.get("STAGE5_SYNTH_DEPTH_MAX_LOOPS", "8")
    rows_per_depth = os.environ.get("STAGE5_SYNTH_DEPTH_ROWS_PER_DEPTH", "24")
    run(
        [
            sys.executable,
            "training/generate_synthetic_depth_task.py",
            "--output_dir",
            path_for_cli(data_dir),
            "--n_symbols",
            os.environ.get("STAGE5_SYNTH_DEPTH_N_SYMBOLS", "16"),
            "--max_depth",
            max_depth,
            "--rows_per_depth",
            rows_per_depth,
            "--seed",
            os.environ.get("STAGE5_SYNTH_DEPTH_SEED", "20260701"),
            "--num_choices",
            os.environ.get("STAGE5_SYNTH_DEPTH_NUM_CHOICES", "4"),
            "--max_target_loops",
            max_loops,
        ]
    )
    dataset_summary = data_dir / "summary.json"
    publish(
        [dataset_summary],
        message=f"Record synthetic depth dataset {run_id} [skip ci]",
    )

    config_path = write_training_config(run_dir, data_dir / "train_sft.jsonl")
    train_proc = run(
        [
            sys.executable,
            "training/train_unfrozen_recurrent.py",
            "--config",
            path_for_cli(config_path),
            "--train_jsonl",
            path_for_cli(data_dir / "train_sft.jsonl"),
            "--device",
            os.environ.get("DEVICE", "cuda"),
        ]
    )
    (run_dir / "train" / "train_unfrozen_recurrent.log").write_text(train_proc.stdout or "", encoding="utf-8")
    checkpoint = latest_checkpoint(run_dir / "train" / "unfrozen")
    training_summary = run_dir / "train" / "unfrozen" / "train_unfrozen_recurrent_summary.json"
    matrix_jsonl = run_dir / "eval" / "test_matrix_rows.jsonl"
    matrix_summary = run_dir / "eval" / "test_matrix_summary.json"
    eval_proc = run(
        [
            sys.executable,
            "eval/eval_synthetic_depth_matrix.py",
            "--data_jsonl",
            path_for_cli(data_dir / "test_mcq.jsonl"),
            "--checkpoint",
            path_for_cli(checkpoint),
            "--output_jsonl",
            path_for_cli(matrix_jsonl),
            "--output_summary",
            path_for_cli(matrix_summary),
            "--loop_counts",
            os.environ.get("STAGE5_SYNTH_DEPTH_EVAL_LOOPS", "1,2,4,8"),
            "--threshold",
            os.environ.get("STAGE5_SYNTH_DEPTH_THRESHOLD", "0.75"),
            "--score_target",
            os.environ.get("STAGE5_SYNTH_DEPTH_SCORE_TARGET", "option_text"),
            "--split",
            os.environ.get("STAGE5_SYNTH_DEPTH_LAYER_SPLIT", "6,18"),
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
    (run_dir / "eval" / "eval_synthetic_depth_matrix.log").write_text(eval_proc.stdout or "", encoding="utf-8")
    matrix = read_json(matrix_summary)
    summary = {
        "kind": "stage5_synthetic_depth_task",
        "run_id": run_id,
        "cell_version": STAGE5_SYNTHETIC_DEPTH_TASK_CELL_VERSION,
        "status": "finished",
        "dataset_summary": path_for_cli(dataset_summary),
        "train_config": path_for_cli(config_path),
        "training_summary": path_for_cli(training_summary),
        "checkpoint": path_for_cli(checkpoint),
        "matrix_rows": path_for_cli(matrix_jsonl),
        "matrix_summary": path_for_cli(matrix_summary),
        "matrix": matrix,
        "interpretation_gate": {
            "success_signature": "frontier_strictly_expands with loop count after loop-1 ceiling is visible",
            "observed_frontier_strictly_expands": matrix.get("frontier_strictly_expands"),
            "observed_frontier_by_loop": matrix.get("frontier_by_loop"),
        },
    }
    write_summary(run_dir, summary)
    publish(
        [
            run_dir / "summary.json",
            run_dir / "summary.md",
            config_path,
            dataset_summary,
            training_summary,
            run_dir / "train" / "train_unfrozen_recurrent.log",
            matrix_summary,
            matrix_jsonl,
            run_dir / "eval" / "eval_synthetic_depth_matrix.log",
        ],
        message=f"Record Stage 5 synthetic depth task {run_id} [skip ci]",
    )
    if env_flag("STAGE5_SYNTH_DEPTH_DISCONNECT", "0"):
        print("Disconnecting Colab runtime after synthetic-depth run.", flush=True)
        runtime.unassign()
except Exception:
    print("Synthetic-depth task run errored; leaving runtime connected.", flush=True)
    raise
