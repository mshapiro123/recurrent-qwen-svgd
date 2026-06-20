"""Run a small exact-grid ARC-AGI smoke evaluation.

This script is intentionally smoke-sized by default. It verifies that the
project can load official ARC-AGI-format tasks, prompt the model, parse output
grids, and report exact-grid accuracy. It is the first bridge from
ARC-Challenge proxy work to true ARC-AGI measurement.
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


ROOT = Path(__file__).resolve().parents[1]
RUN_ID = os.environ.get("STAGE5_ARC_AGI_RUN_ID") or time.strftime("stage5_arc_agi_smoke_%Y%m%d_%H%M%S")
RUN_DIR = ROOT / "outputs" / "stage5" / RUN_ID
RUN_DIR.mkdir(parents=True, exist_ok=True)

BASE_RUN_ID = os.environ.get("STAGE5_BASE_RUN_ID", "stage4_opus_a100_20260620")
BASE_RUN_DIR = ROOT / "outputs" / "stage4" / BASE_RUN_ID
PHASE1_CKPT = Path(os.environ.get("STAGE5_PHASE1_CKPT", str(BASE_RUN_DIR / "phase1" / "phase1_step_500.pt")))
if not PHASE1_CKPT.is_absolute():
    PHASE1_CKPT = ROOT / PHASE1_CKPT

DATA_ROOT = ROOT / "data" / "arc_agi"
ARC_AGI_1_REPO = os.environ.get("ARC_AGI_1_REPO", "https://github.com/fchollet/ARC-AGI.git")
ARC_AGI_2_REPO = os.environ.get("ARC_AGI_2_REPO", "https://github.com/arcprize/ARC-AGI-2.git")
ARC_VERSION = os.environ.get("STAGE5_ARC_AGI_VERSION", "1")
ARC_SPLIT = os.environ.get("STAGE5_ARC_AGI_SPLIT", "evaluation")
LIMIT = int(os.environ.get("STAGE5_ARC_AGI_LIMIT", "5"))
MAX_NEW_TOKENS = int(os.environ.get("STAGE5_ARC_AGI_MAX_NEW_TOKENS", "512"))
GRID_FORMAT = os.environ.get("STAGE5_ARC_AGI_GRID_FORMAT", "compact")
SELECTION_STRATEGY = os.environ.get("STAGE5_ARC_AGI_SELECTION_STRATEGY", "heuristic")
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
PUSH_RESULTS = os.environ.get("STAGE5_ARC_AGI_PUSH", "1").strip().lower() in {"1", "true", "yes", "y"}


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


def resolve_tasks_path() -> Path:
    if user_path := os.environ.get("STAGE5_ARC_AGI_TASKS_PATH"):
        return Path(user_path)
    if ARC_VERSION == "2":
        repo_dir = DATA_ROOT / "ARC-AGI-2"
        clone_or_update(ARC_AGI_2_REPO, repo_dir)
    else:
        repo_dir = DATA_ROOT / "ARC-AGI"
        clone_or_update(ARC_AGI_1_REPO, repo_dir)

    candidates = [
        repo_dir / "data" / ARC_SPLIT,
        repo_dir / ARC_SPLIT,
        repo_dir / "data" / f"{ARC_SPLIT}_challenges",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"Could not find ARC-AGI split {ARC_SPLIT!r} under {repo_dir}")


def read_summary(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


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


def backup_to_drive() -> None:
    mount_drive_if_possible()
    drive_root = Path(os.environ.get("DRIVE_BACKUP_DIR", "/content/drive/MyDrive/recurrent-qwen-svgd-artifacts"))
    if not drive_root.exists():
        print(f"Drive backup skipped; missing {drive_root}")
        return
    backup = drive_root / RUN_ID
    backup.mkdir(parents=True, exist_ok=True)
    shutil.copytree(RUN_DIR, backup / "run_dir", dirs_exist_ok=True)
    print(f"backed_up_to={backup}")


def git_commit_results() -> None:
    for pattern in ["*.json", "*.md", "*.log", "*.jsonl"]:
        run(["git", "add", "-f", str((RUN_DIR / pattern).relative_to(ROOT))], check=False)
    if run(["git", "diff", "--cached", "--quiet"], check=False).returncode == 0:
        print("No ARC-AGI smoke outputs changed.")
        return
    run(["git", "commit", "-m", "Record Stage 5 ARC-AGI smoke eval"])
    run(["git", "push", "origin", "main"], check=False)


def main() -> int:
    if any(arg in {"-h", "--help"} for arg in sys.argv[1:]):
        print(
            "Run this script from Colab to execute a small ARC-AGI exact-grid smoke eval.\n"
            "Configuration is via environment variables, e.g. STAGE5_ARC_AGI_LIMIT=5 "
            "and STAGE5_ARC_AGI_VERSION=1."
        )
        return 0

    tasks_path = resolve_tasks_path()
    restore_phase1_checkpoint()
    metadata = {
        "run_id": RUN_ID,
        "arc_version": ARC_VERSION,
        "arc_split": ARC_SPLIT,
        "limit": LIMIT,
        "tasks_path": str(tasks_path),
        "phase1_checkpoint": str(PHASE1_CKPT.relative_to(ROOT) if PHASE1_CKPT.is_relative_to(ROOT) else PHASE1_CKPT),
        "grid_format": GRID_FORMAT,
        "selection_strategy": SELECTION_STRATEGY,
        "include_symbolic_candidates": INCLUDE_SYMBOLIC,
        "symbolic_position": SYMBOLIC_POSITION if INCLUDE_SYMBOLIC else None,
        "symbolic_candidate_format": SYMBOLIC_CANDIDATE_FORMAT if INCLUDE_SYMBOLIC else None,
        "dtype": DTYPE,
        "adapter_dtype": ADAPTER_DTYPE,
    }
    (RUN_DIR / "run_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps(metadata, indent=2))

    base_summary_json = RUN_DIR / "base_summary.json"
    phase1_summary_json = RUN_DIR / "phase1_summary.json"
    run(
        add_symbolic_args(
            [
            sys.executable,
            "eval/eval_arc_agi.py",
            "--tasks_path",
            str(tasks_path),
            "--limit",
            str(LIMIT),
            "--mode",
            "base",
            "--max_new_tokens",
            str(MAX_NEW_TOKENS),
            "--grid_format",
            GRID_FORMAT,
            "--selection_strategy",
            SELECTION_STRATEGY,
            "--dtype",
            DTYPE,
            "--adapter_dtype",
            ADAPTER_DTYPE,
            "--device",
            DEVICE,
            "--output_jsonl",
            str((RUN_DIR / "base_candidates.jsonl").relative_to(ROOT)),
            "--summary_json",
            str(base_summary_json.relative_to(ROOT)),
            "--summary_md",
            str((RUN_DIR / "base_summary.md").relative_to(ROOT)),
            ]
        ),
        log_name="base_eval.log",
    )
    run(
        add_symbolic_args(
            [
            sys.executable,
            "eval/eval_arc_agi.py",
            "--tasks_path",
            str(tasks_path),
            "--limit",
            str(LIMIT),
            "--mode",
            "phase1",
            "--checkpoint",
            str(PHASE1_CKPT.relative_to(ROOT) if PHASE1_CKPT.is_relative_to(ROOT) else PHASE1_CKPT),
            "--max_loops",
            "4",
            "--num_candidates",
            "1",
            "--max_new_tokens",
            str(MAX_NEW_TOKENS),
            "--grid_format",
            GRID_FORMAT,
            "--selection_strategy",
            SELECTION_STRATEGY,
            "--dtype",
            DTYPE,
            "--adapter_dtype",
            ADAPTER_DTYPE,
            "--device",
            DEVICE,
            "--output_jsonl",
            str((RUN_DIR / "phase1_candidates.jsonl").relative_to(ROOT)),
            "--summary_json",
            str(phase1_summary_json.relative_to(ROOT)),
            "--summary_md",
            str((RUN_DIR / "phase1_summary.md").relative_to(ROOT)),
            ]
        ),
        log_name="phase1_eval.log",
    )
    summary = {
        "metadata": metadata,
        "base": read_summary(base_summary_json)["summary"],
        "phase1": read_summary(phase1_summary_json)["summary"],
    }
    (RUN_DIR / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    lines = [
        f"# Stage 5 ARC-AGI Smoke - {RUN_ID}",
        "",
        f"- ARC version: `{ARC_VERSION}`",
        f"- Split: `{ARC_SPLIT}`",
        f"- Limit: `{LIMIT}`",
        f"- Tasks path: `{tasks_path}`",
        f"- Grid format: `{GRID_FORMAT}`",
        f"- Selection strategy: `{SELECTION_STRATEGY}`",
        f"- Symbolic candidates: `{INCLUDE_SYMBOLIC}`",
        f"- Symbolic position: `{SYMBOLIC_POSITION if INCLUDE_SYMBOLIC else 'n/a'}`",
        f"- Symbolic candidate format: `{SYMBOLIC_CANDIDATE_FORMAT if INCLUDE_SYMBOLIC else 'n/a'}`",
        "",
        "## Exact-Grid Results",
        f"- Base: `{summary['base']}`",
        f"- Phase1 recurrent: `{summary['phase1']}`",
        "",
        "This is a smoke eval. Increase `STAGE5_ARC_AGI_LIMIT` after parsing and generation are stable.",
    ]
    (RUN_DIR / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print((RUN_DIR / "summary.md").read_text(encoding="utf-8"))
    backup_to_drive()
    if PUSH_RESULTS:
        git_commit_results()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
