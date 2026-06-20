"""Recover and summarize a Stage 4 run after interruption.

This is intentionally lightweight: it does not rerun training or benchmarks.
It writes a recovery summary, backs the run directory up to Drive when mounted,
and optionally commits non-checkpoint evidence files.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
RUN_ID = os.environ.get("STAGE4_RUN_ID")
PUSH_RESULTS = os.environ.get("STAGE4_RECOVERY_PUSH", "1").strip().lower() in {"1", "true", "yes", "y"}
AUTO_MOUNT_DRIVE = os.environ.get("STAGE4_AUTO_MOUNT_DRIVE", "1").strip().lower() in {"1", "true", "yes", "y"}


def run(cmd: list[str], *, check: bool = True) -> subprocess.CompletedProcess[str]:
    print("$", " ".join(map(str, cmd)))
    proc = subprocess.run(cmd, cwd=ROOT, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    print(proc.stdout)
    if check and proc.returncode:
        raise RuntimeError(f"command failed: {' '.join(map(str, cmd))}")
    return proc


def latest_stage4_dir() -> Path:
    stage_root = ROOT / "outputs" / "stage4"
    candidates = [path for path in stage_root.glob("*") if path.is_dir()]
    if not candidates:
        raise FileNotFoundError(f"No Stage 4 runs found under {stage_root}")
    return max(candidates, key=lambda path: path.stat().st_mtime)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            rows.append({"_malformed": True, "_raw_prefix": line[:200]})
    return rows


def summarize_mcq(path: Path) -> dict[str, Any]:
    rows = [row for row in read_jsonl(path) if not row.get("_malformed")]
    by_aggregate: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_aggregate.setdefault(str(row.get("aggregate", "mean")), []).append(row)
    return {
        aggregate: {
            "correct": sum(1 for row in aggregate_rows if row.get("hit")),
            "total": len(aggregate_rows),
            "accuracy": sum(1 for row in aggregate_rows if row.get("hit")) / max(len(aggregate_rows), 1),
        }
        for aggregate, aggregate_rows in sorted(by_aggregate.items())
    }


def accuracy(summary: dict[str, Any], label: str = "mean") -> float | None:
    metric = summary.get(label)
    if not metric:
        return None
    return float(metric["accuracy"])


def diff(left: float | None, right: float | None) -> float | None:
    if left is None or right is None:
        return None
    return left - right


def summarize_ladder(base: dict[str, Any], phase1: dict[str, Any], phase2: dict[str, Any]) -> dict[str, Any]:
    base_mean = accuracy(base)
    phase1_mean = accuracy(phase1)
    phase2_values = [accuracy(phase2, key) for key in ("mean", "max", "vote")]
    phase2_best = max([value for value in phase2_values if value is not None], default=None)
    return {
        "base_mean_accuracy": base_mean,
        "phase1_mean_accuracy": phase1_mean,
        "phase2_mean_accuracy": accuracy(phase2, "mean"),
        "phase2_max_accuracy": accuracy(phase2, "max"),
        "phase2_vote_accuracy": accuracy(phase2, "vote"),
        "phase2_best_accuracy": phase2_best,
        "phase1_gap_to_base": diff(phase1_mean, base_mean),
        "phase2_best_lift_over_phase1": diff(phase2_best, phase1_mean),
        "phase2_best_gap_to_base": diff(phase2_best, base_mean),
    }


def backup_to_drive(run_dir: Path, run_id: str) -> str | None:
    drive_root = Path(os.environ.get("DRIVE_BACKUP_DIR", "/content/drive/MyDrive/recurrent-qwen-svgd-artifacts"))
    if AUTO_MOUNT_DRIVE and not drive_root.parent.exists():
        try:
            from google.colab import drive  # type: ignore

            drive.mount("/content/drive")
        except Exception as exc:  # pragma: no cover - only available in Colab
            print(f"Drive auto-mount failed: {exc}")
    if not drive_root.exists():
        print(f"Drive backup skipped; missing {drive_root}")
        return None
    backup = drive_root / run_id / "run_dir"
    backup.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(run_dir, backup, dirs_exist_ok=True)
    print(f"backed_up_to={backup}")
    return str(backup)


def git_commit_recovery(run_dir: Path) -> None:
    patterns = [
        "*.yaml",
        "*.json",
        "*.md",
        "*.log",
        "arc_*.jsonl",
        "exact_phase1_vs_phase2.jsonl",
    ]
    for pattern in patterns:
        run(["git", "add", "-f", str((run_dir / pattern).relative_to(ROOT))], check=False)
    if run(["git", "diff", "--cached", "--quiet"], check=False).returncode == 0:
        print("No staged recovery files to commit.")
        return
    run(["git", "commit", "-m", "Record Stage 4 recovery summary"])
    run(["git", "push", "origin", "main"], check=False)


def main() -> int:
    run_dir = ROOT / "outputs" / "stage4" / RUN_ID if RUN_ID else latest_stage4_dir()
    run_id = run_dir.name
    if not run_dir.exists():
        raise FileNotFoundError(run_dir)

    arc = {
        "base": summarize_mcq(run_dir / "arc_base_label.jsonl"),
        "phase1_recurrent_baseline": summarize_mcq(run_dir / "arc_phase1_label.jsonl"),
        "phase2_recurrent_candidate": summarize_mcq(run_dir / "arc_phase2_svgd_label.jsonl"),
    }
    ladder = summarize_ladder(
        arc["base"],
        arc["phase1_recurrent_baseline"],
        arc["phase2_recurrent_candidate"],
    )
    summary = {
        "run_id": run_id,
        "run_dir": str(run_dir.relative_to(ROOT)),
        "artifact_exists": {
            "phase1_step_500": (run_dir / "phase1" / "phase1_step_500.pt").exists(),
            "phase2_step_100": (run_dir / "phase2" / "phase2_step_100.pt").exists(),
            "projection": (run_dir / "within_group_projection.pt").exists(),
        },
        "arc": arc,
        "arc_ladder": ladder,
        "exact_rows": len(read_jsonl(run_dir / "exact_phase1_vs_phase2.jsonl")),
    }
    backup_path = backup_to_drive(run_dir, run_id)
    summary["drive_backup"] = backup_path

    (run_dir / "recovery_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    lines = [
        f"# Stage 4 Recovery Summary - {run_id}",
        "",
        "## ARC Baseline Ladder",
        f"- Base mean accuracy: `{ladder['base_mean_accuracy']}`",
        f"- Phase1 recurrent mean accuracy: `{ladder['phase1_mean_accuracy']}`",
        f"- Phase2 recurrent best accuracy: `{ladder['phase2_best_accuracy']}`",
        f"- Phase1 gap to base: `{ladder['phase1_gap_to_base']}`",
        f"- Phase2 best lift over Phase1: `{ladder['phase2_best_lift_over_phase1']}`",
        f"- Phase2 best gap to base: `{ladder['phase2_best_gap_to_base']}`",
        "",
        "## Artifact Backup",
        f"- Drive backup: `{backup_path}`",
        f"- Phase1 checkpoint exists: `{summary['artifact_exists']['phase1_step_500']}`",
        f"- Phase2 checkpoint exists: `{summary['artifact_exists']['phase2_step_100']}`",
        f"- Projection exists: `{summary['artifact_exists']['projection']}`",
    ]
    (run_dir / "recovery_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print((run_dir / "recovery_summary.md").read_text(encoding="utf-8"))

    if PUSH_RESULTS:
        git_commit_recovery(run_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
