"""Gate whether exact candidate distillation improves recurrent ARC SFT.

This runner first generates candidate outputs on the public ARC training split,
extracts exact candidates into SFT rows, then runs two matched child SFT jobs:

1. baseline ARC SFT
2. the same ARC SFT plus exact candidate-distillation rows

The held-out evaluation split is still used only by the child SFT evaluations.
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
RUN_ID = os.environ.get("STAGE5_ARC_AGI_CANDIDATE_DISTILL_GATE_RUN_ID") or time.strftime(
    "stage5_arc_agi_candidate_distill_gate_%Y%m%d_%H%M%S"
)
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
TRAIN_SPLIT = os.environ.get("STAGE5_ARC_AGI_TRAIN_SPLIT", "training")
TRAIN_TASK_LIMIT = int(os.environ.get("STAGE5_ARC_AGI_TRAIN_TASK_LIMIT", "100"))
SOURCE_TASK_LIMIT = int(os.environ.get("STAGE5_ARC_AGI_CANDIDATE_DISTILL_TASK_LIMIT", str(TRAIN_TASK_LIMIT)))
GRID_FORMAT = os.environ.get("STAGE5_ARC_AGI_GRID_FORMAT", "compact")
SOURCE_GEOMETRY_TTA = os.environ.get("STAGE5_ARC_AGI_CANDIDATE_DISTILL_GEOMETRY_TTA", "all")
SOURCE_PROGRAM_PARSE_MODE = os.environ.get("STAGE5_ARC_AGI_CANDIDATE_DISTILL_PROGRAM_PARSE_MODE", "fallback")
SOURCE_SELECTION_STRATEGY = os.environ.get("STAGE5_ARC_AGI_CANDIDATE_DISTILL_SELECTION_STRATEGY", "symbolic_priority")
SOURCE_INCLUDE_SYMBOLIC = os.environ.get(
    "STAGE5_ARC_AGI_CANDIDATE_DISTILL_INCLUDE_SYMBOLIC",
    "1",
).strip().lower() in {"1", "true", "yes", "y"}
SOURCE_SYMBOLIC_POSITION = os.environ.get("STAGE5_ARC_AGI_CANDIDATE_DISTILL_SYMBOLIC_POSITION", "after_model")
SOURCE_SYMBOLIC_FORMAT = os.environ.get("STAGE5_ARC_AGI_CANDIDATE_DISTILL_SYMBOLIC_FORMAT", "grid")
SOURCE_MODE = os.environ.get("STAGE5_ARC_AGI_CANDIDATE_DISTILL_SOURCE_MODE", "phase1")
SOURCE_CHECKPOINT = os.environ.get("STAGE5_ARC_AGI_CANDIDATE_DISTILL_SOURCE_CHECKPOINT", "")
DISTILL_CHOICE = os.environ.get("STAGE5_ARC_AGI_CANDIDATE_DISTILL_CHOICE", "best_exact")
DISTILL_COMPLETION_SOURCE = os.environ.get("STAGE5_ARC_AGI_CANDIDATE_DISTILL_COMPLETION_SOURCE", "candidate_text")
MAX_NEW_TOKENS = int(os.environ.get("STAGE5_ARC_AGI_MAX_NEW_TOKENS", "512"))
DTYPE = os.environ.get("DTYPE", "bfloat16")
ADAPTER_DTYPE = os.environ.get("ADAPTER_DTYPE", "float32")
DEVICE = os.environ.get("DEVICE", "cuda")
PUSH_RESULTS = os.environ.get("STAGE5_ARC_AGI_CANDIDATE_DISTILL_GATE_PUSH", "1").strip().lower() in {
    "1",
    "true",
    "yes",
    "y",
}


def run(
    cmd: list[str],
    *,
    env: dict[str, str] | None = None,
    check: bool = True,
    log_name: str | None = None,
) -> subprocess.CompletedProcess[str]:
    print("$", " ".join(map(str, cmd)))
    process = subprocess.Popen(
        cmd,
        cwd=ROOT,
        env=env,
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


def resolve_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def read_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def child_run_id(label: str) -> str:
    return f"{RUN_ID}_{label}"


def child_summary(label: str) -> dict[str, Any]:
    return read_json(ROOT / "outputs" / "stage5" / child_run_id(label) / "summary.json")


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
        return resolve_path(user_path)
    candidates = [
        repo_dir / "data" / split,
        repo_dir / split,
        repo_dir / "data" / f"{split}_challenges",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"Could not find ARC-AGI split {split!r} under {repo_dir}")


def source_checkpoint() -> Path | None:
    if SOURCE_MODE == "base":
        return None
    if SOURCE_CHECKPOINT:
        return resolve_path(SOURCE_CHECKPOINT)
    return PHASE1_CKPT


def generate_candidate_source(train_path: Path) -> dict[str, Any]:
    checkpoint = source_checkpoint()
    if SOURCE_MODE != "base" and checkpoint is None:
        raise ValueError("non-base source mode requires a checkpoint")
    if checkpoint is not None and not checkpoint.exists():
        raise FileNotFoundError(checkpoint)

    summary_json = RUN_DIR / "candidate_source_summary.json"
    output_jsonl = RUN_DIR / "candidate_source_candidates.jsonl"
    cmd = [
        sys.executable,
        "eval/eval_arc_agi.py",
        "--tasks_path",
        str(train_path),
        "--limit",
        str(SOURCE_TASK_LIMIT),
        "--mode",
        SOURCE_MODE,
        "--max_new_tokens",
        str(MAX_NEW_TOKENS),
        "--grid_format",
        GRID_FORMAT,
        "--geometry_tta",
        SOURCE_GEOMETRY_TTA,
        "--program_parse_mode",
        SOURCE_PROGRAM_PARSE_MODE,
        "--selection_strategy",
        SOURCE_SELECTION_STRATEGY,
        "--dtype",
        DTYPE,
        "--adapter_dtype",
        ADAPTER_DTYPE,
        "--device",
        DEVICE,
        "--output_jsonl",
        path_for_cli(output_jsonl),
        "--summary_json",
        path_for_cli(summary_json),
        "--summary_md",
        path_for_cli(RUN_DIR / "candidate_source_summary.md"),
    ]
    if SOURCE_MODE != "base":
        assert checkpoint is not None
        cmd += ["--checkpoint", path_for_cli(checkpoint), "--max_loops", "4", "--num_candidates", "1"]
    if SOURCE_INCLUDE_SYMBOLIC:
        cmd += [
            "--include_symbolic_candidates",
            "--symbolic_position",
            SOURCE_SYMBOLIC_POSITION,
            "--symbolic_candidate_format",
            SOURCE_SYMBOLIC_FORMAT,
        ]
    run(cmd, log_name="candidate_source_eval.log")
    payload = read_json(summary_json)
    payload["candidate_jsonl"] = path_for_cli(output_jsonl)
    return payload


def run_child(label: str, *, candidate_jsonl: Path | None = None) -> dict[str, Any]:
    env = os.environ.copy()
    env["STAGE5_ARC_AGI_SFT_RUN_ID"] = child_run_id(label)
    env["STAGE5_ARC_AGI_SFT_PUSH"] = "0"
    env.pop("STAGE5_ARC_AGI_CANDIDATE_DISTILL_JSONLS", None)
    if candidate_jsonl is not None:
        env["STAGE5_ARC_AGI_CANDIDATE_DISTILL_JSONLS"] = path_for_cli(candidate_jsonl)
        env["STAGE5_ARC_AGI_CANDIDATE_DISTILL_CHOICE"] = DISTILL_CHOICE
        env["STAGE5_ARC_AGI_CANDIDATE_DISTILL_COMPLETION_SOURCE"] = DISTILL_COMPLETION_SOURCE
    run([sys.executable, "colab/run_stage5_arc_agi_sft.py"], env=env, log_name=f"{label}.log")
    return child_summary(label)


def compact(summary: dict[str, Any]) -> dict[str, Any]:
    tuned = summary["phase1_arc_agi_tuned"]
    start = summary["phase1_start"]
    best_checkpoint = summary.get("best_checkpoint") or {}
    best_summary = best_checkpoint.get("summary") or tuned
    candidate_distill_rows = sum(int(item.get("rows", 0)) for item in summary.get("candidate_distill_info", []))
    return {
        "base_selected": summary["base"]["selected_exact"],
        "base_best": summary["base"]["best_of_k_exact"],
        "start_selected": start["selected_exact"],
        "start_best": start["best_of_k_exact"],
        "tuned_selected": tuned["selected_exact"],
        "tuned_best": tuned["best_of_k_exact"],
        "tuned_valid_rate": tuned["valid_candidate_rate"],
        "best_step": best_checkpoint.get("step"),
        "best_selected": best_summary["selected_exact"],
        "best_best": best_summary["best_of_k_exact"],
        "best_valid_rate": best_summary["valid_candidate_rate"],
        "examples": tuned["examples_with_targets"],
        "tasks_solved_best": best_summary["tasks_solved_best_of_k"],
        "tasks": best_summary["tasks_with_targets"],
        "candidate_distill_rows": candidate_distill_rows,
    }


def metric_delta(candidate: dict[str, Any], reference: dict[str, Any]) -> dict[str, Any]:
    return {
        "tuned_selected_delta": int(candidate["tuned_selected"]) - int(reference["tuned_selected"]),
        "tuned_best_delta": int(candidate["tuned_best"]) - int(reference["tuned_best"]),
        "best_selected_delta": int(candidate["best_selected"]) - int(reference["best_selected"]),
        "best_best_delta": int(candidate["best_best"]) - int(reference["best_best"]),
        "tasks_solved_best_delta": int(candidate["tasks_solved_best"]) - int(reference["tasks_solved_best"]),
        "best_valid_rate_delta": float(candidate["best_valid_rate"]) - float(reference["best_valid_rate"]),
    }


def write_report(payload: dict[str, Any]) -> None:
    (RUN_DIR / "summary.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    rows = payload["comparison"]
    lines = [
        f"# Stage 5 ARC Candidate Distillation Gate - {RUN_ID}",
        "",
        f"- Candidate source mode: `{SOURCE_MODE}`",
        f"- Candidate source checkpoint: `{payload['metadata']['source_checkpoint']}`",
        f"- Candidate source train split: `{payload['metadata']['train_path']}`",
        f"- Candidate source limit: `{SOURCE_TASK_LIMIT}`",
        f"- Candidate source geometry TTA: `{SOURCE_GEOMETRY_TTA}`",
        f"- Candidate source include symbolic: `{SOURCE_INCLUDE_SYMBOLIC}`",
        f"- Candidate distill choice: `{DISTILL_CHOICE}`",
        f"- Candidate completion source: `{DISTILL_COMPLETION_SOURCE}`",
        "",
        "## Candidate Source",
        "",
        f"- Summary: `{payload['candidate_source']['summary']}`",
        f"- Candidate JSONL: `{payload['candidate_source']['candidate_jsonl']}`",
        "",
        "## SFT Comparison",
        "",
        "| Arm | Candidate rows | Final selected | Final best | Best step | Best selected | Best best | Tasks best | Best valid rate |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for label, row in rows.items():
        best_step = row["best_step"] if row["best_step"] is not None else "final"
        lines.append(
            f"| `{label}` | {row['candidate_distill_rows']} | "
            f"{row['tuned_selected']}/{row['examples']} | {row['tuned_best']}/{row['examples']} | "
            f"{best_step} | {row['best_selected']}/{row['examples']} | {row['best_best']}/{row['examples']} | "
            f"{row['tasks_solved_best']}/{row['tasks']} | {row['best_valid_rate']:.4f} |"
        )
    lines += [
        "",
        "## Delta: candidate_distill - baseline",
        "",
        f"`{payload['delta_candidate_distill_vs_baseline']}`",
        "",
        "Gate: candidate distillation should improve selected or best exact-grid score without lowering valid-grid rate materially.",
    ]
    (RUN_DIR / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print((RUN_DIR / "summary.md").read_text(encoding="utf-8"))


def backup_to_drive() -> None:
    mount_drive_if_possible()
    drive_root = Path(os.environ.get("DRIVE_BACKUP_DIR", "/content/drive/MyDrive/recurrent-qwen-svgd-artifacts"))
    if not drive_root.exists():
        print(f"Drive backup skipped; missing {drive_root}")
        return
    backup = drive_root / RUN_ID
    backup.mkdir(parents=True, exist_ok=True)
    shutil.copytree(RUN_DIR, backup / "run_dir", dirs_exist_ok=True)
    for label in ["baseline", "candidate_distill"]:
        child_dir = ROOT / "outputs" / "stage5" / child_run_id(label)
        if child_dir.exists():
            shutil.copytree(child_dir, backup / label, dirs_exist_ok=True)
    print(f"backed_up_to={backup}")


def git_commit_results() -> None:
    if not PUSH_RESULTS:
        return
    for pattern in ["*.json", "*.md", "*.log", "*.jsonl"]:
        run(["git", "add", "-f", str((RUN_DIR / pattern).relative_to(ROOT))], check=False)
    if run(["git", "diff", "--cached", "--quiet"], check=False).returncode == 0:
        print("No ARC candidate distillation gate outputs changed.")
        return
    run(["git", "commit", "-m", "Record ARC candidate distillation gate"])
    run(["git", "push", "origin", "main"], check=False)


def main() -> int:
    if any(arg in {"-h", "--help"} for arg in sys.argv[1:]):
        print("Run from Colab to compare ARC SFT with and without exact candidate distillation rows.")
        return 0

    restore_phase1_checkpoint()
    repo_dir = resolve_repo_dir()
    train_path = resolve_split_path(repo_dir, TRAIN_SPLIT)
    candidate_source = generate_candidate_source(train_path)
    candidate_jsonl = ROOT / candidate_source["candidate_jsonl"]
    baseline = run_child("baseline")
    candidate_distill = run_child("candidate_distill", candidate_jsonl=candidate_jsonl)
    comparison = {
        "baseline": compact(baseline),
        "candidate_distill": compact(candidate_distill),
    }
    payload = {
        "run_id": RUN_ID,
        "metadata": {
            "arc_version": ARC_VERSION,
            "train_split": TRAIN_SPLIT,
            "train_path": str(train_path),
            "phase1_checkpoint": path_for_cli(PHASE1_CKPT),
            "source_mode": SOURCE_MODE,
            "source_checkpoint": path_for_cli(source_checkpoint()) if source_checkpoint() is not None else None,
            "source_task_limit": SOURCE_TASK_LIMIT,
            "source_geometry_tta": SOURCE_GEOMETRY_TTA,
            "source_program_parse_mode": SOURCE_PROGRAM_PARSE_MODE,
            "source_selection_strategy": SOURCE_SELECTION_STRATEGY,
            "source_include_symbolic": SOURCE_INCLUDE_SYMBOLIC,
            "source_symbolic_position": SOURCE_SYMBOLIC_POSITION if SOURCE_INCLUDE_SYMBOLIC else None,
            "source_symbolic_format": SOURCE_SYMBOLIC_FORMAT if SOURCE_INCLUDE_SYMBOLIC else None,
            "distill_choice": DISTILL_CHOICE,
            "distill_completion_source": DISTILL_COMPLETION_SOURCE,
        },
        "candidate_source": candidate_source,
        "baseline": baseline,
        "candidate_distill": candidate_distill,
        "comparison": comparison,
        "delta_candidate_distill_vs_baseline": metric_delta(comparison["candidate_distill"], comparison["baseline"]),
    }
    write_report(payload)
    backup_to_drive()
    git_commit_results()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
