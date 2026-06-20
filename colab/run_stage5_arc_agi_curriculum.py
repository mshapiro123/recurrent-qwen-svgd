"""Run a staged ARC-style curriculum for recurrent Phase1 recovery.

Each stage delegates to ``run_stage5_arc_agi_sft.py`` with a different
synthetic-task mixture, then carries that stage's best checkpoint into the next
stage. This makes the recovery experiment less brittle than one large mixed SFT
run: we can see which task families transfer, which regress, and where the
best checkpoint should be taken from before trying particle/SVGD inference.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
RUN_ID = os.environ.get("STAGE5_ARC_AGI_CURRICULUM_RUN_ID") or time.strftime(
    "stage5_arc_agi_curriculum_%Y%m%d_%H%M%S"
)
RUN_DIR = ROOT / "outputs" / "stage5" / RUN_ID
RUN_DIR.mkdir(parents=True, exist_ok=True)

BASE_RUN_ID = os.environ.get("STAGE5_BASE_RUN_ID", "stage4_opus_a100_20260620")
BASE_RUN_DIR = ROOT / "outputs" / "stage4" / BASE_RUN_ID
INITIAL_PHASE1_CKPT = Path(
    os.environ.get("STAGE5_PHASE1_CKPT", str(BASE_RUN_DIR / "phase1" / "phase1_step_500.pt"))
)
if not INITIAL_PHASE1_CKPT.is_absolute():
    INITIAL_PHASE1_CKPT = ROOT / INITIAL_PHASE1_CKPT

PUSH_RESULTS = os.environ.get("STAGE5_ARC_AGI_CURRICULUM_PUSH", "1").strip().lower() in {
    "1",
    "true",
    "yes",
    "y",
}

DEFAULT_STAGES = (
    "warmup:constant_output,geometry_color:120:150;"
    "crop:crop_non_background,crop_recolor,crop_transform_recolor:180:200;"
    "object:move_recolor,frame_object:180:200;"
    "mixed:all:240:250"
)
STAGE_SPEC = os.environ.get("STAGE5_ARC_AGI_CURRICULUM_STAGES", DEFAULT_STAGES)
SYNTHETIC_SEED_BASE = int(os.environ.get("STAGE5_ARC_AGI_CURRICULUM_SYNTHETIC_SEED_BASE", "3000"))
TRAIN_TASK_LIMIT = os.environ.get("STAGE5_ARC_AGI_TRAIN_TASK_LIMIT", "80")
EVAL_TASK_LIMIT = os.environ.get("STAGE5_ARC_AGI_EVAL_TASK_LIMIT", "20")
SAVE_EVERY = os.environ.get("STAGE5_ARC_AGI_SAVE_EVERY", "0")
PROGRAM_PARSE_MODE = os.environ.get("STAGE5_ARC_AGI_PROGRAM_PARSE_MODE", "prefer")
TRACE_MODE = os.environ.get("STAGE5_ARC_AGI_RECOVERY_TRACE_MODE", "symbolic_program")
TRACE_FILTER = os.environ.get("STAGE5_ARC_AGI_RECOVERY_TRACE_FILTER", "covered")
DTYPE = os.environ.get("DTYPE", "bfloat16")
ADAPTER_DTYPE = os.environ.get("ADAPTER_DTYPE", "float32")
DEVICE = os.environ.get("DEVICE", "cuda")


@dataclass(frozen=True)
class CurriculumStage:
    name: str
    synthetic_modes: str
    synthetic_tasks: int
    train_steps: int


def parse_curriculum_stages(value: str) -> list[CurriculumStage]:
    stages: list[CurriculumStage] = []
    for item in value.split(";"):
        item = item.strip()
        if not item:
            continue
        parts = item.split(":")
        if len(parts) != 4:
            raise ValueError(
                "Curriculum stages must be semicolon-separated "
                "name:modes:synthetic_tasks:train_steps items. "
                f"Got {item!r}."
            )
        name, modes, tasks, steps = parts
        stages.append(CurriculumStage(name=name, synthetic_modes=modes, synthetic_tasks=int(tasks), train_steps=int(steps)))
    if not stages:
        raise ValueError("At least one curriculum stage is required.")
    return stages


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


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def compact_family_summary(summary: dict[str, Any]) -> dict[str, Any]:
    rows = {}
    for family, stats in sorted(summary.items()):
        rows[family] = {
            "selected_exact": stats.get("selected_exact", 0),
            "best_of_k_exact": stats.get("best_of_k_exact", 0),
            "examples_with_targets": stats.get("examples_with_targets", 0),
            "tasks_solved_best_of_k": stats.get("tasks_solved_best_of_k", 0),
            "tasks_with_targets": stats.get("tasks_with_targets", 0),
            "valid_candidate_rate": stats.get("valid_candidate_rate", 0.0),
        }
    return rows


def stage_best_checkpoint(summary: dict[str, Any]) -> dict[str, Any]:
    best = summary.get("best_checkpoint") or {}
    if best.get("checkpoint") and best.get("summary"):
        return {
            "source": "best_checkpoint",
            "checkpoint": best["checkpoint"],
            "step": best.get("step"),
            "summary": best["summary"],
            "eval_diagnostics": best.get("eval_diagnostics", {}),
        }
    return {
        "source": "final_checkpoint",
        "checkpoint": summary["tuned_checkpoint"],
        "step": None,
        "summary": summary["phase1_arc_agi_tuned"],
        "eval_diagnostics": (summary.get("eval_diagnostics") or {}).get("phase1_arc_agi_tuned", {}),
    }


def run_stage(stage: CurriculumStage, stage_index: int, resume_checkpoint: Path) -> dict[str, Any]:
    child_run_id = f"{RUN_ID}_{stage_index:02d}_{stage.name}"
    env = os.environ.copy()
    env.update(
        {
            "STAGE5_ARC_AGI_SFT_RUN_ID": child_run_id,
            "STAGE5_ARC_AGI_SFT_PUSH": "0",
            "STAGE5_PHASE1_CKPT": path_for_cli(resume_checkpoint),
            "STAGE5_ARC_AGI_SYNTHETIC_MODES": stage.synthetic_modes,
            "STAGE5_ARC_AGI_SYNTHETIC_TASKS": str(stage.synthetic_tasks),
            "STAGE5_ARC_AGI_SYNTHETIC_SEED": str(SYNTHETIC_SEED_BASE + stage_index),
            "STAGE5_ARC_AGI_TRAIN_STEPS": str(stage.train_steps),
            "STAGE5_ARC_AGI_SAVE_EVERY": SAVE_EVERY,
            "STAGE5_ARC_AGI_EVAL_CHECKPOINT_LADDER": "1",
            "STAGE5_ARC_AGI_TRAIN_TASK_LIMIT": TRAIN_TASK_LIMIT,
            "STAGE5_ARC_AGI_EVAL_TASK_LIMIT": EVAL_TASK_LIMIT,
            "STAGE5_ARC_AGI_PROGRAM_PARSE_MODE": PROGRAM_PARSE_MODE,
            "STAGE5_ARC_AGI_TRACE_MODE": TRACE_MODE,
            "STAGE5_ARC_AGI_TRACE_FILTER": TRACE_FILTER,
            "DTYPE": DTYPE,
            "ADAPTER_DTYPE": ADAPTER_DTYPE,
            "DEVICE": DEVICE,
        }
    )
    run([sys.executable, "colab/run_stage5_arc_agi_sft.py"], env=env, log_name=f"{stage_index:02d}_{stage.name}.log")
    child_dir = ROOT / "outputs" / "stage5" / child_run_id
    summary = read_json(child_dir / "summary.json")
    best = stage_best_checkpoint(summary)
    best_checkpoint = resolve_path(best["checkpoint"])
    if not best_checkpoint.exists():
        raise FileNotFoundError(best_checkpoint)
    return {
        "stage": stage.__dict__,
        "child_run_id": child_run_id,
        "child_run_dir": path_for_cli(child_dir),
        "resume_checkpoint": path_for_cli(resume_checkpoint),
        "selected_checkpoint": best,
        "base": summary.get("base", {}),
        "phase1_start": summary.get("phase1_start", {}),
        "phase1_arc_agi_tuned": summary.get("phase1_arc_agi_tuned", {}),
        "eval_diagnostics": summary.get("eval_diagnostics", {}),
        "best_checkpoint": summary.get("best_checkpoint"),
        "program_only_eval": summary.get("program_only_eval", {}),
    }


def backup_to_drive(stage_rows: list[dict[str, Any]]) -> None:
    if not Path("/content/drive/MyDrive").exists():
        try:
            from google.colab import drive  # type: ignore

            drive.mount("/content/drive")
        except Exception as exc:  # pragma: no cover - Colab only
            print(f"Drive mount skipped/failed: {exc}")
            return
    drive_root = Path(os.environ.get("DRIVE_BACKUP_DIR", "/content/drive/MyDrive/recurrent-qwen-svgd-artifacts"))
    if not drive_root.exists():
        print(f"Drive backup skipped; missing {drive_root}")
        return
    backup = drive_root / RUN_ID
    backup.mkdir(parents=True, exist_ok=True)
    shutil.copytree(RUN_DIR, backup / "run_dir", dirs_exist_ok=True)
    for row in stage_rows:
        child_dir = resolve_path(row["child_run_dir"])
        if child_dir.exists():
            shutil.copytree(child_dir, backup / row["child_run_id"], dirs_exist_ok=True)
    print(f"backed_up_to={backup}")


def git_commit_results() -> None:
    for pattern in ["*.json", "*.md", "*.log"]:
        run(["git", "add", "-f", str((RUN_DIR / pattern).relative_to(ROOT))], check=False)
    if run(["git", "diff", "--cached", "--quiet"], check=False).returncode == 0:
        print("No ARC-AGI curriculum outputs changed.")
        return
    run(["git", "commit", "-m", "Record Stage 5 ARC-AGI curriculum run"])
    run(["git", "push", "origin", "main"], check=False)


def write_report(payload: dict[str, Any]) -> None:
    (RUN_DIR / "summary.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    lines = [
        f"# Stage 5 ARC-AGI Curriculum - {RUN_ID}",
        "",
        f"- Initial checkpoint: `{payload['initial_checkpoint']}`",
        f"- Final selected checkpoint: `{payload['final_checkpoint']}`",
        f"- Program parse mode: `{PROGRAM_PARSE_MODE}`",
        f"- Trace mode/filter: `{TRACE_MODE}` / `{TRACE_FILTER}`",
        "",
        "## Stages",
        "",
    ]
    for row in payload["stages"]:
        tuned_family = compact_family_summary(
            ((row.get("eval_diagnostics") or {}).get("phase1_arc_agi_tuned") or {}).get("task_family_summary", {})
        )
        best_family = compact_family_summary((row.get("selected_checkpoint", {}).get("eval_diagnostics") or {}).get("task_family_summary", {}))
        lines.extend(
            [
                f"### {row['stage']['name']}",
                "",
                f"- Synthetic modes: `{row['stage']['synthetic_modes']}`",
                f"- Synthetic tasks: `{row['stage']['synthetic_tasks']}`",
                f"- Train steps: `{row['stage']['train_steps']}`",
                f"- Resume checkpoint: `{row['resume_checkpoint']}`",
                f"- Selected checkpoint: `{row['selected_checkpoint']}`",
                f"- Base: `{row['base']}`",
                f"- Phase1 start: `{row['phase1_start']}`",
                f"- Tuned: `{row['phase1_arc_agi_tuned']}`",
                f"- Tuned family summary: `{tuned_family}`",
                f"- Selected checkpoint family summary: `{best_family}`",
                "",
            ]
        )
    lines.append("Use this as the deterministic recovery ladder before running replicated particle/SVGD gates on the final selected checkpoint.")
    (RUN_DIR / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print((RUN_DIR / "summary.md").read_text(encoding="utf-8"))


def main() -> int:
    if any(arg in {"-h", "--help"} for arg in sys.argv[1:]):
        print("Run from Colab. Configure stages with STAGE5_ARC_AGI_CURRICULUM_STAGES.")
        return 0

    stages = parse_curriculum_stages(STAGE_SPEC)
    resume_checkpoint = INITIAL_PHASE1_CKPT
    if not resume_checkpoint.exists():
        # Let the child SFT runner restore the first checkpoint from Drive; after
        # that it must exist for subsequent child stages.
        print(f"initial_checkpoint_missing_locally={resume_checkpoint}; child runner will attempt restore")
    stage_rows = []
    for idx, stage in enumerate(stages):
        row = run_stage(stage, idx, resume_checkpoint)
        stage_rows.append(row)
        resume_checkpoint = resolve_path(row["selected_checkpoint"]["checkpoint"])

    payload = {
        "run_id": RUN_ID,
        "settings": {
            "stage_spec": STAGE_SPEC,
            "synthetic_seed_base": SYNTHETIC_SEED_BASE,
            "train_task_limit": TRAIN_TASK_LIMIT,
            "eval_task_limit": EVAL_TASK_LIMIT,
            "save_every": SAVE_EVERY,
            "program_parse_mode": PROGRAM_PARSE_MODE,
            "trace_mode": TRACE_MODE,
            "trace_filter": TRACE_FILTER,
        },
        "initial_checkpoint": path_for_cli(INITIAL_PHASE1_CKPT),
        "final_checkpoint": path_for_cli(resume_checkpoint),
        "stages": stage_rows,
    }
    write_report(payload)
    backup_to_drive(stage_rows)
    if PUSH_RESULTS:
        git_commit_results()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
