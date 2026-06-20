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
MAX_LENGTH = int(os.environ.get("STAGE5_ARC_AGI_MAX_LENGTH", "1024"))
MAX_TOTAL_TOKENS = int(os.environ.get("STAGE5_ARC_AGI_MAX_TOTAL_TOKENS", "2048"))
TRAIN_STEPS = int(os.environ.get("STAGE5_ARC_AGI_TRAIN_STEPS", "300"))
SAVE_EVERY = int(os.environ.get("STAGE5_ARC_AGI_SAVE_EVERY", "150"))
LEARNING_RATE = float(os.environ.get("STAGE5_ARC_AGI_LR", "8e-6"))
BETA = float(os.environ.get("STAGE5_ARC_AGI_BETA", "0.08"))
MAX_NEW_TOKENS = int(os.environ.get("STAGE5_ARC_AGI_MAX_NEW_TOKENS", "512"))
DTYPE = os.environ.get("DTYPE", "bfloat16")
ADAPTER_DTYPE = os.environ.get("ADAPTER_DTYPE", "float32")
DEVICE = os.environ.get("DEVICE", "cuda")
PUSH_RESULTS = os.environ.get("STAGE5_ARC_AGI_SFT_PUSH", "1").strip().lower() in {"1", "true", "yes", "y"}

TRAIN_JSONL = ROOT / "data" / f"{RUN_ID}_train.jsonl"
VAL_JSONL = ROOT / "data" / f"{RUN_ID}_val.jsonl"


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
            "--max_total_tokens",
            str(MAX_TOTAL_TOKENS),
        ],
        log_name="prepare_arc_agi_sft.log",
    )


def eval_arc_agi(label: str, mode: str, tasks_path: Path, checkpoint: Path | None = None) -> dict[str, Any]:
    summary_json = RUN_DIR / f"{label}_summary.json"
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
    run(cmd, log_name=f"{label}_eval.log")
    return read_summary(summary_json)["summary"]


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
    for source in [TRAIN_JSONL, VAL_JSONL]:
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
        "train_steps": TRAIN_STEPS,
        "learning_rate": LEARNING_RATE,
        "phase1_checkpoint": path_for_cli(PHASE1_CKPT),
        "train_path": str(train_path),
        "eval_path": str(eval_path),
    }
    (RUN_DIR / "run_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps(metadata, indent=2))

    prepare_sft(train_path)
    base_summary = eval_arc_agi("base", "base", eval_path)
    start_summary = eval_arc_agi("phase1_start", "phase1", eval_path, PHASE1_CKPT)
    tuned_ckpt = train_phase1()
    tuned_summary = eval_arc_agi("phase1_arc_agi_tuned", "phase1", eval_path, tuned_ckpt)

    summary = {
        "metadata": metadata,
        "base": base_summary,
        "phase1_start": start_summary,
        "phase1_arc_agi_tuned": tuned_summary,
        "tuned_checkpoint": path_for_cli(tuned_ckpt),
    }
    (RUN_DIR / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    lines = [
        f"# Stage 5 ARC-AGI SFT Smoke - {RUN_ID}",
        "",
        f"- ARC version: `{ARC_VERSION}`",
        f"- Train split: `{train_path}`",
        f"- Eval split: `{eval_path}`",
        f"- Train task limit: `{TRAIN_TASK_LIMIT}`",
        f"- Eval task limit: `{EVAL_TASK_LIMIT}`",
        f"- Tuned checkpoint: `{path_for_cli(tuned_ckpt)}`",
        "",
        "## Exact-Grid Results",
        f"- Base: `{base_summary}`",
        f"- Phase1 start: `{start_summary}`",
        f"- Phase1 ARC-AGI tuned: `{tuned_summary}`",
        "",
        "This is still a smoke run. Use it to validate whether ARC-format SFT improves valid-grid and exact-grid rates before scaling.",
    ]
    (RUN_DIR / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print((RUN_DIR / "summary.md").read_text(encoding="utf-8"))
    backup_to_drive()
    if PUSH_RESULTS:
        git_commit_results()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
