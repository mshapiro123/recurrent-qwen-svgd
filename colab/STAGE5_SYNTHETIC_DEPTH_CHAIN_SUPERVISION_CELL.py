"""Colab cell: train-split diagnostic and intermediate-chain supervision.

Phase A evaluates the failed staged-staircase checkpoint on the train split.
If train diagonal accuracy is still below the bar, Phase B resumes from the
proven N=16 primitive checkpoint and trains per-loop intermediate labels.
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


STAGE5_SYNTHETIC_DEPTH_CHAIN_SUPERVISION_CELL_VERSION = "synthetic_depth_chain_supervision_v1"
# Safety marker: Phase B uses loop_loss_mode=per_loop_labels and train_chain_label_sft.
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
        raise RuntimeError("Attach a GPU runtime first. L4/T4 is sufficient for chain supervision.")
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
        print("No chain-supervision changes to publish.", flush=True)
        return
    run(["git", "commit", "-m", message])
    push = run(["git", "push", "origin", "main"], check=False)
    if push.returncode == 0:
        return
    run(["git", "pull", "--rebase", "--autostash", "origin", "main"])
    run(["git", "push", "origin", "main"])


def restore_checkpoint(candidate_paths: list[str | None], dest: Path, *, label: str) -> Path:
    for raw in candidate_paths:
        if not raw:
            continue
        candidate = Path(raw)
        if candidate.exists():
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(candidate, dest)
            print(f"restored_{label}={dest}", flush=True)
            return dest
    raise FileNotFoundError(
        f"Could not restore {label}. Tried: " + ", ".join(str(item) for item in candidate_paths if item)
    )


def restore_failed_staircase_checkpoint(run_dir: Path, source_summary: Path) -> Path:
    payload = read_json(source_summary)
    stages = payload.get("stages") or []
    if not stages:
        raise RuntimeError(f"No stages in source summary: {source_summary}")
    stage = stages[-1]
    return restore_checkpoint(
        [stage.get("checkpoint_drive_backup"), stage.get("checkpoint")],
        run_dir / "restored" / "failed_staircase_checkpoint.pt",
        label="failed_staircase_checkpoint",
    )


def restore_primitive_checkpoint(run_dir: Path, *, n_symbols: int) -> Path:
    curve_summary_path = Path(
        os.environ.get(
            "STAGE5_SYNTH_CHAIN_PRIMITIVE_CURVE_SUMMARY",
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
    return restore_checkpoint(
        [run_summary.get("checkpoint_drive_backup"), run_summary.get("checkpoint"), chosen.get("checkpoint")],
        run_dir / "restored" / "primitive_checkpoint.pt",
        label="primitive_checkpoint",
    )


def filter_depth_jsonl(source: Path, dest: Path, *, max_depth: int) -> dict[str, Any]:
    rows = [json.loads(line) for line in source.read_text(encoding="utf-8").splitlines() if line.strip()]
    kept = [row for row in rows if int(row.get("depth", row.get("synthetic_depth", 0))) <= int(max_depth)]
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text("".join(json.dumps(row, ensure_ascii=True) + "\n" for row in kept), encoding="utf-8")
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


def diagonal_at_matching_loop(matrix: dict[str, Any], *, max_depth: int) -> dict[str, float]:
    values = {}
    cells = matrix.get("matrix", {})
    for depth in range(1, max_depth + 1):
        values[str(depth)] = float(cells[str(depth)][str(depth)]["accuracy"])
    return values


def diagonal_clears_bar(matrix: dict[str, Any], *, max_depth: int, threshold: float) -> bool:
    return all(value >= threshold for value in diagonal_at_matching_loop(matrix, max_depth=max_depth).values())


def eval_checkpoint(
    run_dir: Path,
    *,
    name: str,
    checkpoint: Path,
    data_jsonl: Path,
    loop_counts: str,
    threshold: float,
    score_target: str,
) -> dict[str, Any]:
    matrix_jsonl = run_dir / "eval" / f"{name}_matrix_rows.jsonl"
    matrix_summary = run_dir / "eval" / f"{name}_matrix_summary.json"
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
            score_target,
            "--split",
            os.environ.get("STAGE5_SYNTH_CHAIN_LAYER_SPLIT", "6,18"),
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
    eval_log = run_dir / "eval" / f"{name}_eval_synthetic_depth_matrix.log"
    eval_log.write_text(eval_proc.stdout or "", encoding="utf-8")
    matrix = read_json(matrix_summary)
    return {
        "name": name,
        "data_jsonl": path_for_cli(data_jsonl),
        "score_target": score_target,
        "matrix_rows": path_for_cli(matrix_jsonl),
        "matrix_summary": path_for_cli(matrix_summary),
        "eval_log": path_for_cli(eval_log),
        "matrix": matrix,
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
    prelude_multiplier: float,
) -> Path:
    cfg = {
        "model_name": os.environ.get("MODEL_NAME", "Qwen/Qwen2.5-0.5B-Instruct"),
        "dtype": os.environ.get("DTYPE", "bfloat16"),
        "adapter_dtype": os.environ.get("ADAPTER_DTYPE", "float32"),
        "attn_implementation": os.environ.get("ATTN_IMPLEMENTATION", "default"),
        "layer_split": os.environ.get("STAGE5_SYNTH_CHAIN_LAYER_SPLIT", "6,18"),
        "max_length": int(os.environ.get("STAGE5_SYNTH_CHAIN_MAX_LENGTH", "512")),
        "max_loops": max_loops,
        "loop_loss_mode": "per_loop_labels",
        "initial_halt_prob": float(os.environ.get("STAGE5_SYNTH_CHAIN_INITIAL_HALT_PROB", "0.15")),
        "beta": 0.0,
        "halt_target_nll_weight": 0.0,
        "loop_control_ce_weight": 0.0,
        "batch_size": int(os.environ.get("STAGE5_SYNTH_CHAIN_BATCH_SIZE", "1")),
        "optimizer": os.environ.get("STAGE5_SYNTH_CHAIN_OPTIMIZER", "adamw"),
        "learning_rate": lr,
        "adamw_lr": lr,
        "weight_decay": float(os.environ.get("STAGE5_SYNTH_CHAIN_WEIGHT_DECAY", "0.0")),
        "max_grad_norm": float(os.environ.get("STAGE5_SYNTH_CHAIN_MAX_GRAD_NORM", "0.5")),
        "max_steps": max_steps,
        "save_every": int(os.environ.get("STAGE5_SYNTH_CHAIN_SAVE_EVERY", "0")),
        "log_every": int(os.environ.get("STAGE5_SYNTH_CHAIN_LOG_EVERY", "25")),
        "bridge_prelude_grad_multiplier": prelude_multiplier,
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
        "synthetic_train_format": "chain_label",
        "synthetic_phase": "intermediate_chain_supervision",
        "synthetic_stage": stage_name,
    }
    config_path = run_dir / f"{stage_name}_train_config.yaml"
    write_yaml(config_path, cfg)
    return config_path


def train_stage(
    run_dir: Path,
    *,
    stage_name: str,
    train_jsonl: Path,
    resume_from: Path,
    max_loops: int,
    max_steps: int,
    lr: float,
    prelude_multiplier: float,
) -> dict[str, Any]:
    config_path = write_training_config(
        run_dir,
        stage_name=stage_name,
        train_jsonl=train_jsonl,
        resume_from=resume_from,
        max_loops=max_loops,
        max_steps=max_steps,
        lr=lr,
        prelude_multiplier=prelude_multiplier,
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
    return {
        "stage_name": stage_name,
        "max_loops": max_loops,
        "max_steps": max_steps,
        "learning_rate": lr,
        "bridge_prelude_grad_multiplier": prelude_multiplier,
        "train_jsonl": path_for_cli(train_jsonl),
        "train_config": path_for_cli(config_path),
        "train_log": path_for_cli(train_log),
        "training_summary": path_for_cli(run_dir / "train" / stage_name / "train_unfrozen_recurrent_summary.json"),
        "checkpoint": path_for_cli(checkpoint),
        "checkpoint_abs": str(checkpoint),
    }


def maybe_backup_checkpoint_to_drive(checkpoint: Path, *, run_id: str, stage_name: str) -> str | None:
    if not env_flag("STAGE5_SYNTH_CHAIN_BACKUP_CHECKPOINTS_TO_DRIVE", "1"):
        return None
    dest_dir = DRIVE_CHECKPOINT_ROOT / run_id / stage_name
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / checkpoint.name
    shutil.copy2(checkpoint, dest)
    print(f"checkpoint_drive_backup={dest}", flush=True)
    return str(dest)


def write_summary(run_dir: Path, payload: dict[str, Any]) -> None:
    write_json(run_dir / "summary.json", payload)
    lines = [
        f"# Synthetic Depth Chain Supervision - {payload['run_id']}",
        "",
        f"- Status: `{payload['status']}`",
        f"- N symbols: `{payload['n_symbols']}`",
        f"- Max depth: `{payload['max_depth']}`",
        f"- Train diagnostic diagonal: `{payload.get('train_split_diagnostic', {}).get('diagonal_accuracy')}`",
        f"- Chain run executed: `{payload.get('chain_run_executed')}`",
        "",
    ]
    for stage in payload.get("chain_stages", []):
        lines.extend(
            [
                f"## {stage['stage_name']}",
                "",
                f"- Checkpoint: `{stage.get('checkpoint')}`",
                f"- Drive backup: `{stage.get('checkpoint_drive_backup')}`",
                f"- Train diagonal: `{stage.get('train_eval', {}).get('diagonal_accuracy')}`",
                f"- Test diagonal: `{stage.get('test_eval', {}).get('diagonal_accuracy')}`",
                f"- Test frontier: `{stage.get('test_eval', {}).get('matrix', {}).get('frontier_by_loop')}`",
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
        "STAGE5_SYNTHETIC_DEPTH_CHAIN_SUPERVISION_CELL_VERSION="
        f"{STAGE5_SYNTHETIC_DEPTH_CHAIN_SUPERVISION_CELL_VERSION}",
        flush=True,
    )
    run(["python", "-m", "pip", "install", "-q", "-r", "requirements.txt"])
    run(
        [
            sys.executable,
            "-m",
            "pytest",
            "-q",
            "tests/test_recurrent_wrapper_tiny.py::test_per_loop_label_loss_mode_uses_active_intermediate_labels_on_tiny_model",
            "tests/test_causal_dataset_loop_targets.py::test_jsonl_dataset_builds_and_collates_loop_labels",
            "tests/test_synthetic_depth_task.py::test_chain_label_sft_row_has_intermediate_loop_targets_in_choices",
            "tests/test_eval_synthetic_depth_matrix.py",
        ]
    )

    run_id = os.environ.get("STAGE5_SYNTH_CHAIN_RUN_ID") or time.strftime(
        "stage5_synthetic_depth_chain_supervision_%Y%m%d_%H%M%S"
    )
    run_dir = ROOT / "outputs" / "stage5" / run_id
    source_summary = Path(
        os.environ.get(
            "STAGE5_SYNTH_CHAIN_SOURCE_SUMMARY",
            "outputs/stage5/stage5_synthetic_depth_staged_staircase_20260701_180040/summary.json",
        )
    )
    source_payload = read_json(source_summary)
    n_symbols = int(os.environ.get("STAGE5_SYNTH_CHAIN_N_SYMBOLS", str(source_payload.get("n_symbols", 16))))
    max_depth = int(os.environ.get("STAGE5_SYNTH_CHAIN_MAX_DEPTH", str(source_payload.get("max_depth", 4))))
    rows_per_depth = int(
        os.environ.get("STAGE5_SYNTH_CHAIN_ROWS_PER_DEPTH", str(source_payload.get("rows_per_depth", 256)))
    )
    seed = int(os.environ.get("STAGE5_SYNTH_CHAIN_SEED", str(source_payload.get("dataset", {}).get("seed", 20260702))))
    threshold = float(os.environ.get("STAGE5_SYNTH_CHAIN_THRESHOLD", "0.71"))
    eval_loops = os.environ.get("STAGE5_SYNTH_CHAIN_EVAL_LOOPS", "1,2,3,4")

    failed_checkpoint = restore_failed_staircase_checkpoint(run_dir, source_summary)
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
            os.environ.get("STAGE5_SYNTH_CHAIN_NUM_CHOICES", "4"),
            "--max_target_loops",
            str(max_depth),
        ]
    )
    train_diag = eval_checkpoint(
        run_dir,
        name="phase_a_failed_checkpoint_train_split",
        checkpoint=failed_checkpoint,
        data_jsonl=data_dir / "train_mcq.jsonl",
        loop_counts=eval_loops,
        threshold=threshold,
        score_target="option_text",
    )
    train_diag["diagonal_accuracy"] = diagonal_at_matching_loop(train_diag["matrix"], max_depth=max_depth)
    train_diag["diagonal_clears_bar"] = diagonal_clears_bar(
        train_diag["matrix"],
        max_depth=max_depth,
        threshold=threshold,
    )
    mode = os.environ.get("STAGE5_SYNTH_CHAIN_RUN_AFTER_TRAIN_DIAGNOSTIC", "auto").strip().lower()
    should_run_chain = mode == "always" or (mode == "auto" and not train_diag["diagonal_clears_bar"])

    summary: dict[str, Any] = {
        "kind": "stage5_synthetic_depth_chain_supervision",
        "run_id": run_id,
        "cell_version": STAGE5_SYNTHETIC_DEPTH_CHAIN_SUPERVISION_CELL_VERSION,
        "status": "phase_a_train_split_diagnostic_finished",
        "source_summary": path_for_cli(source_summary),
        "n_symbols": n_symbols,
        "max_depth": max_depth,
        "rows_per_depth": rows_per_depth,
        "seed": seed,
        "threshold": threshold,
        "failed_checkpoint": path_for_cli(failed_checkpoint),
        "primitive_checkpoint": path_for_cli(primitive_checkpoint),
        "data_summary": path_for_cli(data_dir / "summary.json"),
        "train_split_diagnostic": train_diag,
        "chain_run_executed": False,
        "chain_run_decision": {
            "mode": mode,
            "should_run_chain": should_run_chain,
            "reason": "train diagonal did not clear bar" if should_run_chain else "train diagonal cleared bar or mode=never",
        },
        "chain_stages": [],
    }
    write_summary(run_dir, summary)
    publish(
        [
            run_dir / "summary.json",
            run_dir / "summary.md",
            data_dir / "summary.json",
            Path(train_diag["matrix_summary"]),
            Path(train_diag["matrix_rows"]),
            Path(train_diag["eval_log"]),
        ],
        message=f"Record synthetic depth train-split diagnostic {run_id} [skip ci]",
    )

    if should_run_chain:
        train_le2 = data_dir / "train_chain_label_depth_le2_sft.jsonl"
        train_le4 = data_dir / "train_chain_label_depth_le4_sft.jsonl"
        le2_filter = filter_depth_jsonl(data_dir / "train_chain_label_sft.jsonl", train_le2, max_depth=2)
        le4_filter = filter_depth_jsonl(data_dir / "train_chain_label_sft.jsonl", train_le4, max_depth=max_depth)
        lr = float(os.environ.get("STAGE5_SYNTH_CHAIN_LR", "1e-5"))
        prelude_multiplier = float(os.environ.get("STAGE5_SYNTH_CHAIN_BRIDGE_PRELUDE_GRAD_MULTIPLIER", "8.0"))
        stage12_steps = int(os.environ.get("STAGE5_SYNTH_CHAIN_STAGE12_STEPS", "500"))
        stage1234_steps = int(os.environ.get("STAGE5_SYNTH_CHAIN_STAGE1234_STEPS", "1000"))
        chain_stages: list[dict[str, Any]] = []

        stage12 = train_stage(
            run_dir,
            stage_name="chain_depth_le2",
            train_jsonl=train_le2,
            resume_from=primitive_checkpoint,
            max_loops=2,
            max_steps=stage12_steps,
            lr=lr,
            prelude_multiplier=prelude_multiplier,
        )
        stage12_checkpoint = Path(stage12["checkpoint_abs"])
        stage12["checkpoint_drive_backup"] = maybe_backup_checkpoint_to_drive(
            stage12_checkpoint,
            run_id=run_id,
            stage_name="chain_depth_le2",
        )
        stage12["train_eval"] = eval_checkpoint(
            run_dir,
            name="chain_depth_le2_train",
            checkpoint=stage12_checkpoint,
            data_jsonl=data_dir / "train_chain_mcq.jsonl",
            loop_counts=eval_loops,
            threshold=threshold,
            score_target="label",
        )
        stage12["test_eval"] = eval_checkpoint(
            run_dir,
            name="chain_depth_le2_test",
            checkpoint=stage12_checkpoint,
            data_jsonl=data_dir / "test_chain_mcq.jsonl",
            loop_counts=eval_loops,
            threshold=threshold,
            score_target="label",
        )
        for key in ("train_eval", "test_eval"):
            stage12[key]["diagonal_accuracy"] = diagonal_at_matching_loop(stage12[key]["matrix"], max_depth=max_depth)
        stage12.pop("checkpoint_abs", None)
        chain_stages.append(stage12)
        summary.update(
            {
                "status": "chain_depth_le2_finished",
                "chain_run_executed": True,
                "chain_train_filters": {"depth_le2": le2_filter, "depth_le4": le4_filter},
                "chain_stages": chain_stages,
            }
        )
        write_summary(run_dir, summary)
        publish(
            [
                run_dir / "summary.json",
                run_dir / "summary.md",
                train_le2,
                train_le4,
                Path(stage12["train_config"]),
                Path(stage12["training_summary"]),
                Path(stage12["train_log"]),
                Path(stage12["train_eval"]["matrix_summary"]),
                Path(stage12["train_eval"]["matrix_rows"]),
                Path(stage12["train_eval"]["eval_log"]),
                Path(stage12["test_eval"]["matrix_summary"]),
                Path(stage12["test_eval"]["matrix_rows"]),
                Path(stage12["test_eval"]["eval_log"]),
            ],
            message=f"Record synthetic chain supervision depth<=2 {run_id} [skip ci]",
        )

        stage1234 = train_stage(
            run_dir,
            stage_name="chain_depth_le4",
            train_jsonl=train_le4,
            resume_from=stage12_checkpoint,
            max_loops=max_depth,
            max_steps=stage1234_steps,
            lr=lr,
            prelude_multiplier=prelude_multiplier,
        )
        stage1234_checkpoint = Path(stage1234["checkpoint_abs"])
        stage1234["checkpoint_drive_backup"] = maybe_backup_checkpoint_to_drive(
            stage1234_checkpoint,
            run_id=run_id,
            stage_name="chain_depth_le4",
        )
        stage1234["train_eval"] = eval_checkpoint(
            run_dir,
            name="chain_depth_le4_train",
            checkpoint=stage1234_checkpoint,
            data_jsonl=data_dir / "train_chain_mcq.jsonl",
            loop_counts=eval_loops,
            threshold=threshold,
            score_target="label",
        )
        stage1234["test_eval"] = eval_checkpoint(
            run_dir,
            name="chain_depth_le4_test",
            checkpoint=stage1234_checkpoint,
            data_jsonl=data_dir / "test_chain_mcq.jsonl",
            loop_counts=eval_loops,
            threshold=threshold,
            score_target="label",
        )
        for key in ("train_eval", "test_eval"):
            stage1234[key]["diagonal_accuracy"] = diagonal_at_matching_loop(
                stage1234[key]["matrix"],
                max_depth=max_depth,
            )
        stage1234.pop("checkpoint_abs", None)
        chain_stages.append(stage1234)
        summary.update(
            {
                "status": "finished",
                "chain_run_executed": True,
                "chain_stages": chain_stages,
                "interpretation_gate": {
                    "success_signature": "chain_depth_le4 test diagonal clears threshold through depth 4",
                    "train_diagonal": stage1234["train_eval"]["diagonal_accuracy"],
                    "test_diagonal": stage1234["test_eval"]["diagonal_accuracy"],
                    "test_frontier_by_loop": stage1234["test_eval"]["matrix"].get("frontier_by_loop"),
                    "test_diagonal_clears_bar": diagonal_clears_bar(
                        stage1234["test_eval"]["matrix"],
                        max_depth=max_depth,
                        threshold=threshold,
                    ),
                },
            }
        )
        write_summary(run_dir, summary)
        publish(
            [
                run_dir / "summary.json",
                run_dir / "summary.md",
                data_dir / "summary.json",
                data_dir / "train_chain_mcq.jsonl",
                data_dir / "test_chain_mcq.jsonl",
                train_le2,
                train_le4,
                Path(stage1234["train_config"]),
                Path(stage1234["training_summary"]),
                Path(stage1234["train_log"]),
                Path(stage1234["train_eval"]["matrix_summary"]),
                Path(stage1234["train_eval"]["matrix_rows"]),
                Path(stage1234["train_eval"]["eval_log"]),
                Path(stage1234["test_eval"]["matrix_summary"]),
                Path(stage1234["test_eval"]["matrix_rows"]),
                Path(stage1234["test_eval"]["eval_log"]),
            ],
            message=f"Record Stage 5 synthetic chain supervision {run_id} [skip ci]",
        )

    if env_flag("STAGE5_SYNTH_CHAIN_DISCONNECT", "0"):
        print("Disconnecting Colab runtime after synthetic-depth chain supervision.", flush=True)
        runtime.unassign()
except Exception:
    print("Synthetic-depth chain supervision errored; leaving runtime connected.", flush=True)
    raise
