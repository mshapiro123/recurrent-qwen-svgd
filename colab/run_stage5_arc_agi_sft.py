"""Fine-tune recurrent Phase1 on ARC-AGI-format SFT rows and evaluate grids.

This is a small, guarded Stage 5D run. It trains only the recurrent wrapper's
trainable components from the Stage 4 Phase1 checkpoint on public ARC-AGI
training tasks, then evaluates exact-grid generation on ARC-AGI evaluation
tasks.

Default settings are deliberately modest. Increase limits and steps only after
the smoke run confirms parsing, checkpoint restore, and exact-grid reporting.
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


ROOT = Path(__file__).resolve().parents[1]
RUN_ID = os.environ.get("STAGE5_ARC_AGI_SFT_RUN_ID") or time.strftime("stage5_arc_agi_sft_%Y%m%d_%H%M%S")
RUN_DIR = ROOT / "outputs" / "stage5" / RUN_ID
RUN_DIR.mkdir(parents=True, exist_ok=True)

BASE_RUN_ID = os.environ.get("STAGE5_BASE_RUN_ID", "stage4_opus_a100_20260620")
BASE_RUN_DIR = ROOT / "outputs" / "stage4" / BASE_RUN_ID
PHASE1_CKPT = Path(os.environ.get("STAGE5_PHASE1_CKPT", str(BASE_RUN_DIR / "phase1" / "phase1_step_500.pt")))
if not PHASE1_CKPT.is_absolute():
    PHASE1_CKPT = ROOT / PHASE1_CKPT

MODEL_NAME = os.environ.get("MODEL_NAME", "Qwen/Qwen2.5-0.5B-Instruct")
DATA_ROOT = ROOT / "data" / "arc_agi"
ARC_AGI_1_REPO = os.environ.get("ARC_AGI_1_REPO", "https://github.com/fchollet/ARC-AGI.git")
ARC_AGI_2_REPO = os.environ.get("ARC_AGI_2_REPO", "https://github.com/arcprize/ARC-AGI-2.git")
ARC_VERSION = os.environ.get("STAGE5_ARC_AGI_VERSION", "1")
TRAIN_SPLIT = os.environ.get("STAGE5_ARC_AGI_TRAIN_SPLIT", "training")
EVAL_SPLIT = os.environ.get("STAGE5_ARC_AGI_EVAL_SPLIT", "evaluation")
TRAIN_TASK_LIMIT = int(os.environ.get("STAGE5_ARC_AGI_TRAIN_TASK_LIMIT", "100"))
EVAL_TASK_LIMIT = int(os.environ.get("STAGE5_ARC_AGI_EVAL_TASK_LIMIT", "10"))
COLOR_AUGS = int(os.environ.get("STAGE5_ARC_AGI_COLOR_AUGS", "2"))
GEOMETRY_AUGS = os.environ.get("STAGE5_ARC_AGI_GEOMETRY_AUGS", "all")
TRACE_MODE = os.environ.get("STAGE5_ARC_AGI_TRACE_MODE", "none")
TRACE_FILTER = os.environ.get("STAGE5_ARC_AGI_TRACE_FILTER", "all")
SYNTHETIC_TASKS = int(os.environ.get("STAGE5_ARC_AGI_SYNTHETIC_TASKS", "0"))
SYNTHETIC_SEED = int(os.environ.get("STAGE5_ARC_AGI_SYNTHETIC_SEED", "101"))
SYNTHETIC_MODES = os.environ.get("STAGE5_ARC_AGI_SYNTHETIC_MODES", "all")
MAX_LENGTH = int(os.environ.get("STAGE5_ARC_AGI_MAX_LENGTH", "1024"))
MAX_TOTAL_TOKENS = int(os.environ.get("STAGE5_ARC_AGI_MAX_TOTAL_TOKENS", "2048"))
TRAIN_STEPS = int(os.environ.get("STAGE5_ARC_AGI_TRAIN_STEPS", "300"))
SAVE_EVERY = int(os.environ.get("STAGE5_ARC_AGI_SAVE_EVERY", "150"))
EVAL_CHECKPOINT_LADDER = os.environ.get("STAGE5_ARC_AGI_EVAL_CHECKPOINT_LADDER", "0").strip().lower() in {
    "1",
    "true",
    "yes",
    "y",
}
LEARNING_RATE = float(os.environ.get("STAGE5_ARC_AGI_LR", "8e-6"))
BETA = float(os.environ.get("STAGE5_ARC_AGI_BETA", "0.08"))
DISTILL_ENABLED = os.environ.get("STAGE5_ARC_AGI_DISTILL", "0").strip().lower() in {"1", "true", "yes", "y"}
DISTILL_WEIGHT = float(os.environ.get("STAGE5_ARC_AGI_DISTILL_WEIGHT", "0.1"))
DISTILL_TEMPERATURE = float(os.environ.get("STAGE5_ARC_AGI_DISTILL_TEMPERATURE", "1.0"))
DISTILL_ON = os.environ.get("STAGE5_ARC_AGI_DISTILL_ON", "response")
MAX_NEW_TOKENS = int(os.environ.get("STAGE5_ARC_AGI_MAX_NEW_TOKENS", "512"))
GRID_FORMAT = os.environ.get("STAGE5_ARC_AGI_GRID_FORMAT", "compact")
PROGRAM_PARSE_MODE = os.environ.get("STAGE5_ARC_AGI_PROGRAM_PARSE_MODE", "fallback")
SELECTION_STRATEGY = os.environ.get("STAGE5_ARC_AGI_SELECTION_STRATEGY", "heuristic")
PROGRAM_ONLY_EVAL = os.environ.get("STAGE5_ARC_AGI_PROGRAM_ONLY_EVAL", "0").strip().lower() in {
    "1",
    "true",
    "yes",
    "y",
}
PROGRAM_ONLY_CHECKPOINT_LADDER = os.environ.get(
    "STAGE5_ARC_AGI_PROGRAM_ONLY_CHECKPOINT_LADDER",
    "0",
).strip().lower() in {
    "1",
    "true",
    "yes",
    "y",
}
INCLUDE_SYMBOLIC = os.environ.get("STAGE5_ARC_AGI_INCLUDE_SYMBOLIC", "0").strip().lower() in {
    "1",
    "true",
    "yes",
    "y",
}
SYMBOLIC_POSITION = os.environ.get("STAGE5_ARC_AGI_SYMBOLIC_POSITION", "after_model")
SYMBOLIC_CANDIDATE_FORMAT = os.environ.get("STAGE5_ARC_AGI_SYMBOLIC_CANDIDATE_FORMAT", "grid")
DTYPE = os.environ.get("DTYPE", "bfloat16")
ADAPTER_DTYPE = os.environ.get("ADAPTER_DTYPE", "float32")
DEVICE = os.environ.get("DEVICE", "cuda")
PUSH_RESULTS = os.environ.get("STAGE5_ARC_AGI_SFT_PUSH", "1").strip().lower() in {"1", "true", "yes", "y"}

TRAIN_JSONL = ROOT / "data" / f"{RUN_ID}_train.jsonl"
VAL_JSONL = ROOT / "data" / f"{RUN_ID}_val.jsonl"
SYNTHETIC_TASKS_JSON = RUN_DIR / "synthetic_arc_tasks.json"
SYNTHETIC_JSONL = ROOT / "data" / f"{RUN_ID}_synthetic_train.jsonl"


def run(cmd: list[str], *, check: bool = True, log_name: str | None = None) -> subprocess.CompletedProcess[str]:
    print("$", " ".join(map(str, cmd)))
    process = subprocess.Popen(
        cmd,
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        bufsize=1,
    )
    chunks: list[str] = []
    assert process.stdout is not None
    for line in process.stdout:
        print(line, end="", flush=True)
        chunks.append(line)
    stdout = "".join(chunks)
    proc = subprocess.CompletedProcess(cmd, process.wait(), stdout=stdout, stderr=None)
    if log_name:
        (RUN_DIR / log_name).write_text(stdout, encoding="utf-8")
    if check and proc.returncode:
        raise RuntimeError(f"command failed: {' '.join(map(str, cmd))}")
    return proc


def path_for_cli(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def mount_drive_if_possible() -> None:
    if Path("/content/drive/MyDrive").exists():
        return
    try:
        from google.colab import drive  # type: ignore

        drive.mount("/content/drive")
    except Exception as exc:  # pragma: no cover - Colab only
        print(f"Drive mount skipped/failed: {exc}")


def restore_phase1_checkpoint() -> None:
    if PHASE1_CKPT.exists():
        return
    mount_drive_if_possible()
    drive_root = Path(os.environ.get("DRIVE_BACKUP_DIR", "/content/drive/MyDrive/recurrent-qwen-svgd-artifacts"))
    candidates = sorted(drive_root.rglob("phase1_step_500.pt")) if drive_root.exists() else []
    for candidate in candidates:
        if BASE_RUN_ID in str(candidate):
            PHASE1_CKPT.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(candidate, PHASE1_CKPT)
            print(f"restored_phase1_checkpoint={candidate} -> {PHASE1_CKPT}")
            return
    if candidates:
        PHASE1_CKPT.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(candidates[0], PHASE1_CKPT)
        print(f"restored_phase1_checkpoint={candidates[0]} -> {PHASE1_CKPT}")
        return
    raise FileNotFoundError(f"Missing Phase1 checkpoint: {PHASE1_CKPT}")


def clone_or_update(repo_url: str, target: Path) -> None:
    if target.exists() and (target / ".git").exists():
        run(["git", "-C", str(target), "pull", "--ff-only"], check=False)
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    run(["git", "clone", "--depth", "1", repo_url, str(target)])


def resolve_repo_dir() -> Path:
    if ARC_VERSION == "2":
        repo_dir = DATA_ROOT / "ARC-AGI-2"
        clone_or_update(ARC_AGI_2_REPO, repo_dir)
    else:
        repo_dir = DATA_ROOT / "ARC-AGI"
        clone_or_update(ARC_AGI_1_REPO, repo_dir)
    return repo_dir


def resolve_split_path(repo_dir: Path, split: str) -> Path:
    if user_path := os.environ.get(f"STAGE5_ARC_AGI_{split.upper()}_PATH"):
        return Path(user_path)
    candidates = [
        repo_dir / "data" / split,
        repo_dir / split,
        repo_dir / "data" / f"{split}_challenges",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"Could not find ARC-AGI split {split!r} under {repo_dir}")


def write_yaml(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")


def read_summary(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def count_jsonl(path: Path) -> int:
    if not path.exists():
        return 0
    return sum(1 for line in path.read_text(encoding="utf-8").splitlines() if line.strip())


def append_jsonl(target: Path, source: Path) -> None:
    if not source.exists() or source.stat().st_size == 0:
        return
    with target.open("a", encoding="utf-8") as out, source.open("r", encoding="utf-8") as inp:
        for line in inp:
            if line.strip():
                out.write(line if line.endswith("\n") else line + "\n")


def add_symbolic_args(cmd: list[str]) -> list[str]:
    if INCLUDE_SYMBOLIC:
        cmd += [
            "--include_symbolic_candidates",
            "--symbolic_position",
            SYMBOLIC_POSITION,
            "--symbolic_candidate_format",
            SYMBOLIC_CANDIDATE_FORMAT,
        ]
    return cmd


def prepare_sft(train_path: Path) -> None:
    run(
        [
            sys.executable,
            "training/prepare_arc_agi_sft_jsonl.py",
            "--tasks_path",
            str(train_path),
            "--tokenizer_name",
            MODEL_NAME,
            "--output_jsonl",
            path_for_cli(TRAIN_JSONL),
            "--val_jsonl",
            path_for_cli(VAL_JSONL),
            "--limit",
            str(TRAIN_TASK_LIMIT),
            "--augment_color_permutations",
            str(COLOR_AUGS),
            "--augment_geometries",
            GEOMETRY_AUGS,
            "--grid_format",
            GRID_FORMAT,
            "--trace_mode",
            TRACE_MODE,
            "--trace_filter",
            TRACE_FILTER,
            "--max_total_tokens",
            str(MAX_TOTAL_TOKENS),
        ],
        log_name="prepare_arc_agi_sft.log",
    )
    if SYNTHETIC_TASKS:
        synthetic_trace_mode = TRACE_MODE if TRACE_MODE in {"symbolic", "symbolic_program"} else "symbolic"
        run(
            [
                sys.executable,
                "training/generate_arc_agi_synthetic_tasks.py",
                "--output_json",
                path_for_cli(SYNTHETIC_TASKS_JSON),
                "--num_tasks",
                str(SYNTHETIC_TASKS),
                "--seed",
                str(SYNTHETIC_SEED),
                "--modes",
                SYNTHETIC_MODES,
            ],
            log_name="generate_synthetic_arc_tasks.log",
        )
        run(
            [
                sys.executable,
                "training/prepare_arc_agi_sft_jsonl.py",
                "--tasks_path",
                path_for_cli(SYNTHETIC_TASKS_JSON),
                "--tokenizer_name",
                MODEL_NAME,
                "--output_jsonl",
                path_for_cli(SYNTHETIC_JSONL),
                "--augment_color_permutations",
                "0",
                "--augment_geometries",
                "none",
                "--grid_format",
                GRID_FORMAT,
                "--trace_mode",
                synthetic_trace_mode,
                "--trace_filter",
                "covered",
                "--max_total_tokens",
                str(MAX_TOTAL_TOKENS),
                "--no-append_eos",
            ],
            log_name="prepare_synthetic_arc_sft.log",
        )
        before = count_jsonl(TRAIN_JSONL)
        append_jsonl(TRAIN_JSONL, SYNTHETIC_JSONL)
        after = count_jsonl(TRAIN_JSONL)
        print(f"synthetic_jsonl_rows={count_jsonl(SYNTHETIC_JSONL)}")
        print(f"train_rows_after_synthetic={after} added={after - before}")


def eval_arc_agi_payload(
    label: str,
    mode: str,
    tasks_path: Path,
    checkpoint: Path | None = None,
    *,
    program_parse_mode: str | None = None,
) -> dict[str, Any]:
    summary_json = RUN_DIR / f"{label}_summary.json"
    parse_mode = program_parse_mode or PROGRAM_PARSE_MODE
    cmd = [
        sys.executable,
        "eval/eval_arc_agi.py",
        "--tasks_path",
        str(tasks_path),
        "--limit",
        str(EVAL_TASK_LIMIT),
        "--mode",
        mode,
        "--max_new_tokens",
        str(MAX_NEW_TOKENS),
        "--grid_format",
        GRID_FORMAT,
        "--program_parse_mode",
        parse_mode,
        "--selection_strategy",
        SELECTION_STRATEGY,
        "--dtype",
        DTYPE,
        "--adapter_dtype",
        ADAPTER_DTYPE,
        "--device",
        DEVICE,
        "--output_jsonl",
        path_for_cli(RUN_DIR / f"{label}_candidates.jsonl"),
        "--summary_json",
        path_for_cli(summary_json),
        "--summary_md",
        path_for_cli(RUN_DIR / f"{label}_summary.md"),
    ]
    if mode != "base":
        assert checkpoint is not None
        cmd += [
            "--checkpoint",
            path_for_cli(checkpoint),
            "--max_loops",
            "4",
            "--num_candidates",
            "1",
        ]
    add_symbolic_args(cmd)
    run(cmd, log_name=f"{label}_eval.log")
    return read_summary(summary_json)


def eval_arc_agi(label: str, mode: str, tasks_path: Path, checkpoint: Path | None = None) -> dict[str, Any]:
    return eval_arc_agi_payload(label, mode, tasks_path, checkpoint)["summary"]


def eval_diagnostics(payload: dict[str, Any]) -> dict[str, Any]:
    return {
        "candidate_source_summary": payload.get("candidate_source_summary", {}),
        "task_family_summary": payload.get("task_family_summary", {}),
        "parse_method_summary": payload.get("parse_method_summary", {}),
        "program_verifier_summary": payload.get("program_verifier_summary", {}),
    }


def program_verifier_line(label: str, diagnostics: dict[str, Any]) -> str:
    verifier = diagnostics.get("program_verifier_summary", {})
    return (
        f"- {label} program verifier: candidates `{verifier.get('candidates_with_program', 0)}`, "
        f"fits_train `{verifier.get('candidates_program_fits_train', 0)}`, "
        f"fit_selected_exact `{verifier.get('program_fit_selected_exact', 0)}`"
    )


def compact_eval_payload(payload: dict[str, Any]) -> dict[str, Any]:
    return {
        "summary": payload["summary"],
        "eval_diagnostics": eval_diagnostics(payload),
    }


def eval_program_only_payloads(
    *,
    eval_path: Path,
    tuned_ckpt: Path,
    checkpoint_ladder: list[dict[str, Any]],
) -> dict[str, Any]:
    if not PROGRAM_ONLY_EVAL:
        return {}
    rows: dict[str, Any] = {
        "base": compact_eval_payload(eval_arc_agi_payload("base_program_only", "base", eval_path, program_parse_mode="program_only")),
        "phase1_start": compact_eval_payload(
            eval_arc_agi_payload(
                "phase1_start_program_only",
                "phase1",
                eval_path,
                PHASE1_CKPT,
                program_parse_mode="program_only",
            )
        ),
        "phase1_arc_agi_tuned": compact_eval_payload(
            eval_arc_agi_payload(
                "phase1_arc_agi_tuned_program_only",
                "phase1",
                eval_path,
                tuned_ckpt,
                program_parse_mode="program_only",
            )
        ),
    }
    if PROGRAM_ONLY_CHECKPOINT_LADDER:
        rows["checkpoint_ladder"] = []
        for row in checkpoint_ladder:
            checkpoint = ROOT / row["checkpoint"]
            payload = eval_arc_agi_payload(
                f"phase1_arc_agi_step_{row['step']}_program_only",
                "phase1",
                eval_path,
                checkpoint,
                program_parse_mode="program_only",
            )
            rows["checkpoint_ladder"].append(
                {
                    "step": row["step"],
                    "checkpoint": row["checkpoint"],
                    **compact_eval_payload(payload),
                }
            )
    return rows


def checkpoint_step(path: Path) -> int:
    try:
        return int(path.stem.rsplit("_", 1)[1])
    except (IndexError, ValueError):
        return -1


def checkpoint_delta(candidate: dict[str, Any], reference: dict[str, Any]) -> dict[str, Any]:
    return {
        "first_exact_delta": int(candidate.get("first_exact", 0)) - int(reference.get("first_exact", 0)),
        "selected_exact_delta": int(candidate.get("selected_exact", 0)) - int(reference.get("selected_exact", 0)),
        "best_of_k_exact_delta": int(candidate.get("best_of_k_exact", 0)) - int(reference.get("best_of_k_exact", 0)),
        "valid_candidate_rate_delta": float(candidate.get("valid_candidate_rate", 0.0))
        - float(reference.get("valid_candidate_rate", 0.0)),
    }


def eval_checkpoint_ladder(
    *,
    eval_path: Path,
    start_summary: dict[str, Any],
    base_summary: dict[str, Any],
) -> list[dict[str, Any]]:
    checkpoint_dir = RUN_DIR / "phase1_arc_agi"
    rows: list[dict[str, Any]] = []
    for checkpoint in sorted(checkpoint_dir.glob("phase1_step_*.pt"), key=checkpoint_step):
        step = checkpoint_step(checkpoint)
        if step < 0:
            continue
        payload = eval_arc_agi_payload(f"phase1_arc_agi_step_{step}", "phase1", eval_path, checkpoint)
        summary = payload["summary"]
        rows.append(
            {
                "step": step,
                "checkpoint": path_for_cli(checkpoint),
                "summary": summary,
                "eval_diagnostics": eval_diagnostics(payload),
                "delta_vs_start": checkpoint_delta(summary, start_summary),
                "delta_vs_base": checkpoint_delta(summary, base_summary),
            }
        )
    return rows


def best_ladder_row(rows: list[dict[str, Any]]) -> dict[str, Any] | None:
    if not rows:
        return None
    return max(
        rows,
        key=lambda row: (
            int(row["summary"].get("best_of_k_exact", 0)),
            int(row["summary"].get("selected_exact", 0)),
            float(row["summary"].get("valid_candidate_rate", 0.0)),
            int(row["step"]),
        ),
    )


def train_phase1() -> Path:
    cfg = {
        "model_name": MODEL_NAME,
        "dtype": DTYPE,
        "adapter_dtype": ADAPTER_DTYPE,
        "layer_split": "6,18",
        "max_length": MAX_LENGTH,
        "max_loops": 4,
        "initial_halt_prob": 0.15,
        "beta": BETA,
        "batch_size": 1,
        "learning_rate": LEARNING_RATE,
        "weight_decay": 0.0,
        "max_grad_norm": 0.3,
        "max_steps": TRAIN_STEPS,
        "save_every": SAVE_EVERY,
        "log_every": 25,
        "train_on_prompt": False,
        "output_dir": path_for_cli(RUN_DIR / "phase1_arc_agi"),
        "resume_from": path_for_cli(PHASE1_CKPT),
        "lora": {"enabled": True, "rank": 8, "alpha": 16, "dropout": 0.0},
        "distillation": {
            "enabled": DISTILL_ENABLED,
            "weight": DISTILL_WEIGHT,
            "temperature": DISTILL_TEMPERATURE,
            "on": DISTILL_ON,
        },
    }
    cfg_path = RUN_DIR / "phase1_arc_agi.yaml"
    write_yaml(cfg_path, cfg)
    run(
        [
            sys.executable,
            "training/train_phase1_ponder.py",
            "--config",
            path_for_cli(cfg_path),
            "--train_jsonl",
            path_for_cli(TRAIN_JSONL),
            "--device",
            DEVICE,
        ],
        log_name="phase1_arc_agi_train.log",
    )
    checkpoint = RUN_DIR / "phase1_arc_agi" / f"phase1_step_{TRAIN_STEPS}.pt"
    if not checkpoint.exists():
        raise FileNotFoundError(checkpoint)
    return checkpoint


def backup_to_drive() -> None:
    mount_drive_if_possible()
    drive_root = Path(os.environ.get("DRIVE_BACKUP_DIR", "/content/drive/MyDrive/recurrent-qwen-svgd-artifacts"))
    if not drive_root.exists():
        print(f"Drive backup skipped; missing {drive_root}")
        return
    backup = drive_root / RUN_ID
    backup.mkdir(parents=True, exist_ok=True)
    shutil.copytree(RUN_DIR, backup / "run_dir", dirs_exist_ok=True)
    for source in [TRAIN_JSONL, VAL_JSONL, SYNTHETIC_JSONL, SYNTHETIC_TASKS_JSON]:
        if source.exists():
            target = backup / "data" / source.name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
    print(f"backed_up_to={backup}")


def git_commit_results() -> None:
    for pattern in ["*.yaml", "*.json", "*.md", "*.log", "*.jsonl"]:
        run(["git", "add", "-f", str((RUN_DIR / pattern).relative_to(ROOT))], check=False)
    if run(["git", "diff", "--cached", "--quiet"], check=False).returncode == 0:
        print("No ARC-AGI SFT outputs changed.")
        return
    run(["git", "commit", "-m", "Record Stage 5 ARC-AGI SFT smoke"])
    run(["git", "push", "origin", "main"], check=False)


def main() -> int:
    if any(arg in {"-h", "--help"} for arg in sys.argv[1:]):
        print("Run from Colab. Configure with STAGE5_ARC_AGI_* environment variables.")
        return 0

    repo_dir = resolve_repo_dir()
    train_path = resolve_split_path(repo_dir, TRAIN_SPLIT)
    eval_path = resolve_split_path(repo_dir, EVAL_SPLIT)
    restore_phase1_checkpoint()

    metadata = {
        "run_id": RUN_ID,
        "arc_version": ARC_VERSION,
        "train_split": TRAIN_SPLIT,
        "eval_split": EVAL_SPLIT,
        "train_task_limit": TRAIN_TASK_LIMIT,
        "eval_task_limit": EVAL_TASK_LIMIT,
        "color_augmentations": COLOR_AUGS,
        "geometry_augmentations": GEOMETRY_AUGS,
        "trace_mode": TRACE_MODE,
        "trace_filter": TRACE_FILTER,
        "synthetic_tasks": SYNTHETIC_TASKS,
        "synthetic_seed": SYNTHETIC_SEED,
        "synthetic_modes": SYNTHETIC_MODES,
        "train_steps": TRAIN_STEPS,
        "save_every": SAVE_EVERY,
        "learning_rate": LEARNING_RATE,
        "grid_format": GRID_FORMAT,
        "program_parse_mode": PROGRAM_PARSE_MODE,
        "selection_strategy": SELECTION_STRATEGY,
        "program_only_eval": PROGRAM_ONLY_EVAL,
        "program_only_checkpoint_ladder": PROGRAM_ONLY_CHECKPOINT_LADDER,
        "eval_checkpoint_ladder": EVAL_CHECKPOINT_LADDER,
        "distillation": {
            "enabled": DISTILL_ENABLED,
            "weight": DISTILL_WEIGHT,
            "temperature": DISTILL_TEMPERATURE,
            "on": DISTILL_ON,
        },
        "include_symbolic_candidates": INCLUDE_SYMBOLIC,
        "symbolic_position": SYMBOLIC_POSITION if INCLUDE_SYMBOLIC else None,
        "symbolic_candidate_format": SYMBOLIC_CANDIDATE_FORMAT if INCLUDE_SYMBOLIC else None,
        "phase1_checkpoint": path_for_cli(PHASE1_CKPT),
        "train_path": str(train_path),
        "eval_path": str(eval_path),
    }
    (RUN_DIR / "run_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps(metadata, indent=2))

    prepare_sft(train_path)
    base_payload = eval_arc_agi_payload("base", "base", eval_path)
    base_summary = base_payload["summary"]
    start_payload = eval_arc_agi_payload("phase1_start", "phase1", eval_path, PHASE1_CKPT)
    start_summary = start_payload["summary"]
    tuned_ckpt = train_phase1()
    tuned_payload = eval_arc_agi_payload("phase1_arc_agi_tuned", "phase1", eval_path, tuned_ckpt)
    tuned_summary = tuned_payload["summary"]
    checkpoint_ladder = (
        eval_checkpoint_ladder(eval_path=eval_path, start_summary=start_summary, base_summary=base_summary)
        if EVAL_CHECKPOINT_LADDER
        else []
    )
    best_checkpoint = best_ladder_row(checkpoint_ladder)
    program_only_eval = eval_program_only_payloads(
        eval_path=eval_path,
        tuned_ckpt=tuned_ckpt,
        checkpoint_ladder=checkpoint_ladder,
    )

    summary = {
        "metadata": metadata,
        "base": base_summary,
        "phase1_start": start_summary,
        "phase1_arc_agi_tuned": tuned_summary,
        "eval_diagnostics": {
            "base": eval_diagnostics(base_payload),
            "phase1_start": eval_diagnostics(start_payload),
            "phase1_arc_agi_tuned": eval_diagnostics(tuned_payload),
        },
        "program_only_eval": program_only_eval,
        "tuned_checkpoint": path_for_cli(tuned_ckpt),
        "checkpoint_ladder": checkpoint_ladder,
        "best_checkpoint": best_checkpoint,
    }
    diagnostics = summary["eval_diagnostics"]
    (RUN_DIR / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    lines = [
        f"# Stage 5 ARC-AGI SFT Smoke - {RUN_ID}",
        "",
        f"- ARC version: `{ARC_VERSION}`",
        f"- Train split: `{train_path}`",
        f"- Eval split: `{eval_path}`",
        f"- Train task limit: `{TRAIN_TASK_LIMIT}`",
        f"- Eval task limit: `{EVAL_TASK_LIMIT}`",
        f"- Save every: `{SAVE_EVERY}`",
        f"- Eval checkpoint ladder: `{EVAL_CHECKPOINT_LADDER}`",
        f"- Geometry augmentations: `{GEOMETRY_AUGS}`",
        f"- Trace mode: `{TRACE_MODE}`",
        f"- Trace filter: `{TRACE_FILTER}`",
        f"- Synthetic tasks: `{SYNTHETIC_TASKS}` modes `{SYNTHETIC_MODES}`",
        f"- Distillation: `{DISTILL_ENABLED}` weight `{DISTILL_WEIGHT}` temperature `{DISTILL_TEMPERATURE}` on `{DISTILL_ON}`",
        f"- Symbolic candidates: `{INCLUDE_SYMBOLIC}`",
        f"- Symbolic position: `{SYMBOLIC_POSITION if INCLUDE_SYMBOLIC else 'n/a'}`",
        f"- Symbolic candidate format: `{SYMBOLIC_CANDIDATE_FORMAT if INCLUDE_SYMBOLIC else 'n/a'}`",
        f"- Tuned checkpoint: `{path_for_cli(tuned_ckpt)}`",
        f"- Grid format: `{GRID_FORMAT}`",
        f"- Program parse mode: `{PROGRAM_PARSE_MODE}`",
        f"- Selection strategy: `{SELECTION_STRATEGY}`",
        f"- Program-only eval: `{PROGRAM_ONLY_EVAL}`",
        "",
        "## Exact-Grid Results",
        f"- Base: `{base_summary}`",
        f"- Phase1 start: `{start_summary}`",
        f"- Phase1 ARC-AGI tuned: `{tuned_summary}`",
        "",
        "## Program Verifier Diagnostics",
        program_verifier_line("Base", diagnostics["base"]),
        program_verifier_line("Phase1 start", diagnostics["phase1_start"]),
        program_verifier_line("Phase1 tuned", diagnostics["phase1_arc_agi_tuned"]),
        "",
    ]
    if program_only_eval:
        lines.extend(
            [
                "## Program-Only Evaluation",
                f"- Base: `{program_only_eval['base']['summary']}`",
                f"- Phase1 start: `{program_only_eval['phase1_start']['summary']}`",
                f"- Phase1 tuned: `{program_only_eval['phase1_arc_agi_tuned']['summary']}`",
                program_verifier_line("Program-only tuned", program_only_eval["phase1_arc_agi_tuned"]["eval_diagnostics"]),
                "",
            ]
        )
    if checkpoint_ladder:
        lines.extend(["## Checkpoint Ladder", ""])
        for row in checkpoint_ladder:
            lines.append(
                f"- Step `{row['step']}`: summary `{row['summary']}`, "
                f"delta_vs_start `{row['delta_vs_start']}`, delta_vs_base `{row['delta_vs_base']}`, "
                f"program_verifier `{row.get('eval_diagnostics', {}).get('program_verifier_summary', {})}`"
            )
        lines.extend(["", f"Best checkpoint: `{best_checkpoint}`", ""])
    lines.append(
        "This is still a smoke run. Use it to validate whether ARC-format SFT improves valid-grid and exact-grid rates before scaling."
    )
    (RUN_DIR / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print((RUN_DIR / "summary.md").read_text(encoding="utf-8"))
    backup_to_drive()
    if PUSH_RESULTS:
        git_commit_results()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
