"""Colab launcher: unfreeze recurrent block + Muon + loop curriculum.

This target tests the next clean hypothesis after rank-only LoRA capacity
localization: merge the recovered LoRA adapters into the recurrent block, then
train the recurrent block weights themselves under a bounded loop-count
curriculum. It deliberately disables re-entry adapters, tail dampers, SVGD, and
particles so the readout is about recurrence capacity rather than another
selection or geometry variant.

Benchmark follow-up for the produced checkpoint must set
``STAGE5_BENCHMARK_LORA_RANK=0`` because the LoRA initialization has been merged
into dense recurrent block weights before unfreezing.
"""

from __future__ import annotations

import json
import os
import random
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import yaml
from google.colab import drive, runtime, userdata


STAGE5_UNFREEZE_RECURRENT_CURRICULUM_CELL_VERSION = "unfreeze_recurrent_curriculum_v1"
REPO = "mshapiro123/recurrent-qwen-svgd"
ROOT = Path("/content/recurrent-qwen-svgd")
DRIVE_ARTIFACT_ROOT = Path("/content/drive/MyDrive/recurrent-qwen-svgd-artifacts")
LEGACY_DRIVE_ROOT = Path("/content/drive/MyDrive/recurrent-qwen-svgd")
DEFAULT_SOURCE_SUMMARY = "outputs/stage5/stage5_reentry_recovery_20260627_190155/summary.json"
DEFAULT_TRACE_JSONL = "data/curriculum/stage5_capability_ladder_trace_collection_20260623_194537/positive_sft.jsonl"


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
    print("HF token not found; downloads will use anonymous Hub access.", flush=True)


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
    printable = redact(" ".join(map(str, cmd)))
    print(f"$ {printable}", flush=True)
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


def resolve_repo_path(path: str | Path) -> Path:
    raw = Path(str(path).replace("\\", "/"))
    return raw if raw.is_absolute() else ROOT / str(raw).lstrip("/")


def read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} is not a JSON object")
    return payload


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def attached_gpu_names() -> list[str]:
    try:
        proc = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            check=False,
        )
    except FileNotFoundError:
        return []
    if proc.returncode != 0:
        return []
    return [line.strip() for line in proc.stdout.splitlines() if line.strip()]


def require_gpu_runtime() -> None:
    names = attached_gpu_names()
    if not names:
        raise RuntimeError("Attach an L4/T4/A100/H100 GPU runtime before running the unfreeze curriculum.")
    print("unfreeze_gpu_runtime=" + "; ".join(names), flush=True)


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
    run(["git", "log", "--oneline", "-5"])
    root_str = str(ROOT)
    if root_str not in sys.path:
        sys.path.insert(0, root_str)


def mount_drive_if_needed() -> None:
    if not Path("/content/drive/MyDrive").exists():
        drive.mount("/content/drive", force_remount=False)


def source_summary_path() -> Path:
    raw = os.environ.get("STAGE5_UNFREEZE_SOURCE_SUMMARY", DEFAULT_SOURCE_SUMMARY).strip()
    path = resolve_repo_path(raw)
    if not path.exists():
        pointer = ROOT / "config" / "stage5_current_source_summary.txt"
        if pointer.exists():
            pointer_value = pointer.read_text(encoding="utf-8").strip()
            if pointer_value:
                candidate = resolve_repo_path(pointer_value)
                if candidate.exists() and os.environ.get("STAGE5_UNFREEZE_ALLOW_POINTER_SOURCE", "0") == "1":
                    return candidate
    if not path.exists():
        raise FileNotFoundError(f"Missing source summary: {path_for_cli(path)}")
    return path


def checkpoint_from_summary(payload: dict[str, Any]) -> str:
    for key in ("phase1_checkpoint", "checkpoint"):
        value = payload.get(key)
        if value:
            return str(value)
    child = payload.get("child_summary")
    if child:
        child_path = resolve_repo_path(str(child))
        if child_path.exists():
            return checkpoint_from_summary(read_json(child_path))
    raise KeyError("source summary has no checkpoint or phase1_checkpoint")


def infer_run_id(path: Path) -> str | None:
    parts = [part for part in path.parts if part.startswith("stage5_")]
    return parts[-1] if parts else None


def unique_paths(paths: list[Path]) -> list[Path]:
    seen: set[str] = set()
    out: list[Path] = []
    for path in paths:
        key = path.as_posix()
        if key not in seen:
            out.append(path)
            seen.add(key)
    return out


def drive_checkpoint_candidates(candidate: Path) -> list[Path]:
    rel = Path(path_for_cli(candidate))
    run_id = infer_run_id(candidate)
    filename = candidate.name
    roots = [DRIVE_ARTIFACT_ROOT, LEGACY_DRIVE_ROOT]
    candidates: list[Path] = []
    for root in roots:
        candidates.append(root / rel)
        candidates.append(root / "outputs" / "stage5" / rel)
        if run_id:
            candidates.extend(root.glob(f"**/{run_id}*/**/{filename}"))
    return unique_paths(candidates)


def restore_checkpoint(candidate: Path) -> Path:
    if candidate.exists():
        return candidate
    mount_drive_if_needed()
    for drive_candidate in drive_checkpoint_candidates(candidate):
        if drive_candidate.exists():
            candidate.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(drive_candidate, candidate)
            print(f"restored_unfreeze_checkpoint={drive_candidate} -> {candidate}", flush=True)
            return candidate
    searched = "\n".join(str(path) for path in drive_checkpoint_candidates(candidate)[:40])
    raise FileNotFoundError(f"Could not restore checkpoint {path_for_cli(candidate)}. Searched:\n{searched}")


def split_trace_jsonl(source: Path, run_dir: Path) -> tuple[Path, Path, dict[str, Any]]:
    rows = [json.loads(line) for line in source.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not rows:
        raise ValueError(f"No rows in {source}")
    rng = random.Random(int(os.environ.get("STAGE5_UNFREEZE_SPLIT_SEED", "17")))
    indices = list(range(len(rows)))
    rng.shuffle(indices)
    val_min = int(os.environ.get("STAGE5_UNFREEZE_VAL_MIN_ROWS", "6"))
    val_fraction = float(os.environ.get("STAGE5_UNFREEZE_VAL_FRACTION", "0.1"))
    val_count = min(len(rows) - 1, max(val_min, int(round(len(rows) * val_fraction))))
    val_indices = set(indices[:val_count])
    train_rows = [row for idx, row in enumerate(rows) if idx not in val_indices]
    val_rows = [row for idx, row in enumerate(rows) if idx in val_indices]
    data_dir = run_dir / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    train_path = data_dir / "train.jsonl"
    val_path = data_dir / "val.jsonl"
    train_path.write_text("".join(json.dumps(row, ensure_ascii=True) + "\n" for row in train_rows), encoding="utf-8")
    val_path.write_text("".join(json.dumps(row, ensure_ascii=True) + "\n" for row in val_rows), encoding="utf-8")
    counts: dict[str, int] = {}
    for row in rows:
        key = str(row.get("target_loop_count", "missing"))
        counts[key] = counts.get(key, 0) + 1
    return train_path, val_path, {
        "source_jsonl": path_for_cli(source),
        "rows": len(rows),
        "train_rows": len(train_rows),
        "val_rows": len(val_rows),
        "target_loop_counts": counts,
    }


def parse_metric_stdout(stdout: str) -> dict[str, float]:
    metrics: dict[str, float] = {}
    for line in stdout.splitlines():
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()
        try:
            metrics[key] = float(value)
        except ValueError:
            continue
    return metrics


def latest_checkpoint(output_dir: Path) -> Path:
    checkpoints = sorted(output_dir.glob("unfrozen_recurrent_step_*.pt"), key=lambda path: path.stat().st_mtime)
    if not checkpoints:
        raise FileNotFoundError(f"No unfrozen recurrent checkpoint saved under {output_dir}")
    return checkpoints[-1]


def write_config(run_dir: Path, checkpoint: Path) -> Path:
    max_steps = int(os.environ.get("STAGE5_UNFREEZE_MAX_STEPS", "50"))
    end_loop = int(os.environ.get("STAGE5_UNFREEZE_END_LOOP", "8"))
    resume_lora_rank = os.environ.get("STAGE5_UNFREEZE_RESUME_LORA_RANK", "auto")
    resume_lora_alpha = os.environ.get("STAGE5_UNFREEZE_RESUME_LORA_ALPHA", "auto")
    cfg = {
        "model_name": os.environ.get("MODEL_NAME", "Qwen/Qwen2.5-0.5B-Instruct"),
        "dtype": os.environ.get("DTYPE", "bfloat16"),
        "adapter_dtype": os.environ.get("ADAPTER_DTYPE", "float32"),
        "attn_implementation": os.environ.get("ATTN_IMPLEMENTATION", "default"),
        "layer_split": os.environ.get("STAGE5_UNFREEZE_LAYER_SPLIT", "6,18"),
        "max_length": int(os.environ.get("STAGE5_UNFREEZE_MAX_LENGTH", "512")),
        "max_loops": end_loop,
        "initial_halt_prob": float(os.environ.get("STAGE5_UNFREEZE_INITIAL_HALT_PROB", "0.15")),
        "beta": float(os.environ.get("STAGE5_UNFREEZE_BETA", "0.1")),
        "halt_target_nll_weight": float(os.environ.get("STAGE5_UNFREEZE_HALT_TARGET_NLL_WEIGHT", "1.0")),
        "batch_size": int(os.environ.get("STAGE5_UNFREEZE_BATCH_SIZE", "1")),
        "optimizer": os.environ.get("STAGE5_UNFREEZE_OPTIMIZER", "muon"),
        "learning_rate": float(os.environ.get("STAGE5_UNFREEZE_LR", "5e-6")),
        "adamw_lr": float(os.environ.get("STAGE5_UNFREEZE_ADAMW_LR", os.environ.get("STAGE5_UNFREEZE_LR", "5e-6"))),
        "weight_decay": float(os.environ.get("STAGE5_UNFREEZE_WEIGHT_DECAY", "0.0")),
        "max_grad_norm": float(os.environ.get("STAGE5_UNFREEZE_MAX_GRAD_NORM", "0.5")),
        "max_steps": max_steps,
        "log_every": int(os.environ.get("STAGE5_UNFREEZE_LOG_EVERY", "5")),
        "train_on_prompt": False,
        "gradient_checkpointing": True,
        "output_dir": path_for_cli(run_dir / "unfrozen"),
        "resume_from": path_for_cli(checkpoint),
        "resume_lora": {
            "enabled": True,
            "rank": resume_lora_rank,
            "alpha": resume_lora_alpha,
            "dropout": 0.0,
        },
        "merge_lora_before_unfreeze": True,
        "require_lora_loaded_before_merge": True,
        "train_auxiliary": {
            "bridge": True,
            "halting": True,
            "reentry_adapter": False,
            "latent": False,
        },
        "recurrence_curriculum": {
            "enabled": True,
            "start_loop": int(os.environ.get("STAGE5_UNFREEZE_START_LOOP", "1")),
            "end_loop": end_loop,
            "schedule": os.environ.get("STAGE5_UNFREEZE_LOOP_SCHEDULE", "linear"),
            "target_source": os.environ.get("STAGE5_UNFREEZE_TARGET_SOURCE", "schedule"),
            "ramp_compute": True,
        },
    }
    config_path = run_dir / "unfrozen_recurrent_config.yaml"
    config_path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
    return config_path


def run_val_loop_eval(checkpoint: Path, val_jsonl: Path, run_dir: Path) -> dict[str, dict[str, float]]:
    loop_metrics: dict[str, dict[str, float]] = {}
    for loops in [1, 2, 4, 8]:
        if loops > int(os.environ.get("STAGE5_UNFREEZE_END_LOOP", "8")):
            continue
        proc = run(
            [
                sys.executable,
                "eval/eval_jsonl.py",
                "--model_name",
                os.environ.get("MODEL_NAME", "Qwen/Qwen2.5-0.5B-Instruct"),
                "--data_jsonl",
                path_for_cli(val_jsonl),
                "--checkpoint",
                path_for_cli(checkpoint),
                "--split",
                os.environ.get("STAGE5_UNFREEZE_LAYER_SPLIT", "6,18"),
                "--max_loops",
                str(loops),
                "--max_length",
                os.environ.get("STAGE5_UNFREEZE_MAX_LENGTH", "512"),
                "--beta",
                os.environ.get("STAGE5_UNFREEZE_BETA", "0.1"),
                "--dtype",
                os.environ.get("DTYPE", "bfloat16"),
                "--adapter_dtype",
                os.environ.get("ADAPTER_DTYPE", "float32"),
                "--lora_rank",
                "0",
                "--device",
                os.environ.get("DEVICE", "cuda"),
                "--group_by_field",
                "curriculum_mode",
                "--group_by_field",
                "target_loop_count",
            ],
            cwd=ROOT,
        )
        (run_dir / f"eval_val_loops{loops}.log").write_text(proc.stdout, encoding="utf-8")
        loop_metrics[str(loops)] = parse_metric_stdout(proc.stdout)
    return loop_metrics


def run_drift_probe(checkpoint: Path, run_dir: Path) -> Path:
    output_json = run_dir / "unfrozen_reentry_drift.json"
    run(
        [
            sys.executable,
            "eval/eval_reentry_drift.py",
            "--checkpoint",
            path_for_cli(checkpoint),
            "--prompts_jsonl",
            "eval/smoke_exact_tasks_v2.jsonl",
            "--limit",
            os.environ.get("STAGE5_UNFREEZE_DRIFT_LIMIT", "8"),
            "--max_loops",
            os.environ.get("STAGE5_UNFREEZE_END_LOOP", "8"),
            "--max_length",
            "256",
            "--dtype",
            os.environ.get("DTYPE", "bfloat16"),
            "--adapter_dtype",
            os.environ.get("ADAPTER_DTYPE", "float32"),
            "--lora_rank",
            "0",
            "--device",
            os.environ.get("DEVICE", "cuda"),
            "--output_json",
            path_for_cli(output_json),
            "--output_jsonl",
            path_for_cli(run_dir / "unfrozen_reentry_drift.jsonl"),
        ],
        cwd=ROOT,
    )
    return output_json


def backup_run_dir_to_drive(run_dir: Path) -> None:
    if not env_flag("STAGE5_UNFREEZE_BACKUP_TO_DRIVE", "1"):
        print("Drive backup disabled for unfreeze run.", flush=True)
        return
    mount_drive_if_needed()
    dst = DRIVE_ARTIFACT_ROOT / "outputs" / "stage5" / run_dir.name
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(run_dir, dst, dirs_exist_ok=True)
    print(f"backed_up_unfreeze_run={dst}", flush=True)


def write_markdown_summary(summary_path: Path, payload: dict[str, Any]) -> None:
    loop_eval = payload.get("loop_eval", {})
    lines = [
        f"# Stage 5 Unfreeze Recurrent Curriculum: {payload['run_id']}",
        "",
        f"- Status: `{payload['status']}`",
        f"- Source summary: `{payload['source_summary']}`",
        f"- Source checkpoint: `{payload['source_checkpoint']}`",
        f"- Checkpoint: `{payload['checkpoint']}`",
        f"- Training rows: `{payload['dataset']['train_rows']}`; validation rows: `{payload['dataset']['val_rows']}`",
        f"- Optimizer: `{payload['config']['optimizer']}`; max steps: `{payload['config']['max_steps']}`",
        f"- Merge LoRA before unfreeze: `{payload['config']['merge_lora_before_unfreeze']}`",
        "",
        "## Validation Loop Sweep",
        "",
    ]
    for loops, metrics in loop_eval.items():
        lines.append(
            f"- Loops `{loops}`: expected_ce `{metrics.get('expected_ce')}`, "
            f"loss `{metrics.get('loss')}`, mean_expected_loops `{metrics.get('mean_expected_loops')}`"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            f"- Any deeper loop beats loop 1 on validation CE: `{payload['decision']['any_deeper_loop_val_ce_beats_loop1']}`",
            f"- Best validation loop by CE: `{payload['decision']['best_val_ce_loop']}`",
            f"- Next step: `{payload['decision']['next_step']}`",
        ]
    )
    summary_path.with_suffix(".md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def publish(run_dir: Path, summary_path: Path) -> None:
    from colab.stage5_publish_utils import publishable_artifact_paths, update_current_source_summary

    pointer = update_current_source_summary(ROOT, summary_path)
    run(["git", "pull", "--rebase", "--autostash", "origin", "main"], cwd=ROOT, check=False)
    for path in publishable_artifact_paths(run_dir):
        run(["git", "add", "-f", path_for_cli(path)], cwd=ROOT, check=False)
    run(["git", "add", "-f", path_for_cli(pointer)], cwd=ROOT, check=False)
    status = run(["git", "diff", "--cached", "--quiet"], cwd=ROOT, check=False)
    if status.returncode == 0:
        print("No unfreeze curriculum outputs changed.", flush=True)
        return
    run(["git", "commit", "-m", f"Record Stage 5 unfreeze recurrent curriculum {run_dir.name} [skip ci]"], cwd=ROOT)
    push = run(["git", "push", "origin", "main"], cwd=ROOT, check=False)
    if push.returncode:
        run(["git", "pull", "--rebase", "--autostash", "origin", "main"], cwd=ROOT)
        run(["git", "push", "origin", "main"], cwd=ROOT)


def disconnect(reason: str) -> None:
    if not env_flag("STAGE5_UNFREEZE_DISCONNECT", "0"):
        print(f"Leaving Colab runtime connected: {reason}", flush=True)
        return
    try:
        print(f"Disconnecting Colab runtime: {reason}", flush=True)
        runtime.unassign()
    except Exception as exc:
        print(f"Runtime disconnect skipped: {exc}", flush=True)


try:
    require_gpu_runtime()
    sync_repo()
    os.chdir(ROOT)
    print(f"STAGE5_UNFREEZE_RECURRENT_CURRICULUM_CELL_VERSION={STAGE5_UNFREEZE_RECURRENT_CURRICULUM_CELL_VERSION}", flush=True)
    run(["git", "log", "--oneline", "-5"], cwd=ROOT, check=False)
    run([sys.executable, "-m", "pip", "install", "-q", "-r", "requirements.txt"], cwd=ROOT)
    run(
        [
            sys.executable,
            "-m",
            "pytest",
            "-q",
            "tests/test_lora.py",
            "tests/test_muon.py",
            "tests/test_train_unfrozen_recurrent.py",
            "tests/test_stage5_notebooks.py::test_current_bootstrap_exposes_unfreeze_recurrent_curriculum_target",
        ],
        cwd=ROOT,
    )

    source_summary = source_summary_path()
    source_payload = read_json(source_summary)
    checkpoint = restore_checkpoint(resolve_repo_path(checkpoint_from_summary(source_payload)))
    trace_jsonl = resolve_repo_path(
        os.environ.get(
            "STAGE5_UNFREEZE_TRAIN_JSONL",
            str(source_payload.get("dataset", {}).get("source_positive_sft") or DEFAULT_TRACE_JSONL),
        )
    )
    if not trace_jsonl.exists():
        raise FileNotFoundError(f"Missing trace JSONL: {path_for_cli(trace_jsonl)}")

    run_id = os.environ.get("STAGE5_UNFREEZE_RUN_ID") or time.strftime(
        "stage5_unfreeze_recurrent_curriculum_%Y%m%d_%H%M%S"
    )
    run_dir = ROOT / "outputs" / "stage5" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    train_jsonl, val_jsonl, dataset_summary = split_trace_jsonl(trace_jsonl, run_dir)
    config_path = write_config(run_dir, checkpoint)

    print("unfreeze_source_summary:", path_for_cli(source_summary), flush=True)
    print("unfreeze_source_checkpoint:", path_for_cli(checkpoint), flush=True)
    print("unfreeze_train_jsonl:", path_for_cli(train_jsonl), flush=True)
    print("unfreeze_val_jsonl:", path_for_cli(val_jsonl), flush=True)
    print("unfreeze_config:", path_for_cli(config_path), flush=True)

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
        ],
        cwd=ROOT,
    )
    (run_dir / "train_unfrozen_recurrent.log").write_text(train_proc.stdout, encoding="utf-8")

    checkpoint_out = latest_checkpoint(run_dir / "unfrozen")
    loop_eval = run_val_loop_eval(checkpoint_out, val_jsonl, run_dir)
    drift_json = run_drift_probe(checkpoint_out, run_dir)
    training_summary_path = run_dir / "unfrozen" / "train_unfrozen_recurrent_summary.json"
    training_summary = read_json(training_summary_path) if training_summary_path.exists() else {}

    loop1_ce = loop_eval.get("1", {}).get("expected_ce")
    deeper = {
        loop: metrics.get("expected_ce")
        for loop, metrics in loop_eval.items()
        if loop != "1" and metrics.get("expected_ce") is not None
    }
    best_loop = None
    best_ce = None
    for loop, metrics in loop_eval.items():
        ce = metrics.get("expected_ce")
        if ce is None:
            continue
        if best_ce is None or ce < best_ce:
            best_loop = loop
            best_ce = ce
    any_deeper_beats = bool(loop1_ce is not None and any(ce < loop1_ce for ce in deeper.values()))
    summary_path = run_dir / "summary.json"
    payload = {
        "kind": "stage5_unfreeze_recurrent_curriculum",
        "cell_version": STAGE5_UNFREEZE_RECURRENT_CURRICULUM_CELL_VERSION,
        "run_id": run_id,
        "status": "finished",
        "source_summary": path_for_cli(source_summary),
        "source_checkpoint": path_for_cli(checkpoint),
        "checkpoint": path_for_cli(checkpoint_out),
        "config_path": path_for_cli(config_path),
        "training_summary": path_for_cli(training_summary_path),
        "drift_summary": path_for_cli(drift_json),
        "dataset": dataset_summary,
        "config": yaml.safe_load(config_path.read_text(encoding="utf-8")),
        "trainable_parameters": training_summary.get("trainable_parameters", {}),
        "loop_eval": loop_eval,
        "decision": {
            "any_deeper_loop_val_ce_beats_loop1": any_deeper_beats,
            "best_val_ce_loop": best_loop,
            "best_val_ce": best_ce,
            "loop1_val_ce": loop1_ce,
            "next_step": (
                "run_debiased_forced_depth_benchmark_lora_rank0"
                if loop_eval and best_ce is not None
                else "review_training_failure"
            ),
        },
    }
    write_json(summary_path, payload)
    write_markdown_summary(summary_path, payload)
    backup_run_dir_to_drive(run_dir)
    publish(run_dir, summary_path)
    disconnect("unfreeze recurrent curriculum finished")
except Exception:
    disconnect("unfreeze recurrent curriculum errored")
    raise
