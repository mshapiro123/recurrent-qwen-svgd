"""Preflight checks for the current Stage 5 Colab recovery run.

This is a lightweight diagnostic script: it does not load Qwen and it does not
train. Run it before the balanced recovery autopilot to confirm that Git, Drive,
the selected checkpoint, and any resumable child summaries are visible.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from colab.run_stage5_recovered_phase1_arc_gate import (  # noqa: E402
    candidate_drive_checkpoints,
    drive_diagnostics,
    path_for_cli,
)


DEFAULT_SOURCE_SUMMARY = ROOT / "outputs" / "stage5" / "stage5_balanced_mcq_current" / "summary.json"
DEFAULT_AUTOPILOT_RUN_ID = "stage5_balanced_recovery_autopilot_current"


def run_git(args: list[str]) -> str:
    proc = subprocess.run(
        ["git", *args],
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    return proc.stdout.strip()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def infer_stage5_run_id(path: str | Path) -> str | None:
    parts = Path(path).parts
    for idx, part in enumerate(parts):
        if part == "stage5" and idx + 1 < len(parts):
            return parts[idx + 1]
    return None


def selected_checkpoint(source_summary: Path) -> Path | None:
    if not source_summary.exists():
        return None
    payload = read_json(source_summary)
    checkpoint = (payload.get("best_checkpoint") or {}).get("checkpoint")
    if not checkpoint:
        return None
    path = Path(str(checkpoint))
    return path if path.is_absolute() else ROOT / path


def child_summary(run_id: str, suffix: str) -> Path:
    return ROOT / "outputs" / "stage5" / f"{run_id}_{suffix}" / "summary.json"


def summary_status(path: Path) -> str:
    if not path.exists():
        return "missing"
    try:
        payload = read_json(path)
    except Exception as exc:  # pragma: no cover - corrupted JSON is rare and environment-specific
        return f"unreadable: {exc}"
    status = payload.get("status", "unknown")
    passed = payload.get("passed")
    return f"{status} passed={passed}"


def checkpoint_restore_status(checkpoint: Path | None) -> list[str]:
    if checkpoint is None:
        return ["selected_checkpoint=missing_from_source_summary"]
    run_id = infer_stage5_run_id(checkpoint)
    lines = [
        f"selected_checkpoint={path_for_cli(checkpoint)}",
        f"checkpoint_exists={checkpoint.exists()}",
        f"checkpoint_run_id={run_id or 'unknown'}",
    ]
    if checkpoint.exists() or not run_id:
        return lines
    candidates = candidate_drive_checkpoints(run_id, checkpoint.name)
    existing = [path for path in candidates if path.exists()]
    lines.append(f"drive_candidates_checked={min(len(candidates), 12)}")
    lines.append(f"drive_candidate_exists={bool(existing)}")
    if existing:
        lines.append(f"first_existing_drive_candidate={existing[0]}")
    else:
        lines.append("first_drive_candidates:")
        lines.extend(f"  {path}" for path in candidates[:12])
    return lines


def main() -> int:
    source_summary = Path(os.environ.get("STAGE5_BALANCED_RECOVERY_SOURCE_SUMMARY", str(DEFAULT_SOURCE_SUMMARY)))
    if not source_summary.is_absolute():
        source_summary = ROOT / source_summary
    run_id = os.environ.get("STAGE5_BALANCED_RECOVERY_RUN_ID", DEFAULT_AUTOPILOT_RUN_ID)

    print(f"root={ROOT}")
    print(f"git_head={run_git(['rev-parse', '--short', 'HEAD'])}")
    print("git_status:")
    status = run_git(["status", "--short"])
    print(status or "  clean")
    print()
    print(drive_diagnostics())
    print()

    print(f"source_summary={path_for_cli(source_summary)} exists={source_summary.exists()}")
    if source_summary.exists():
        source_payload = read_json(source_summary)
        print(f"source_status={source_payload.get('status')}")
    checkpoint = selected_checkpoint(source_summary)
    for line in checkpoint_restore_status(checkpoint):
        print(line)
    print()

    distill_summary = child_summary(run_id, "distill")
    arc_mix_summary = child_summary(run_id, "arc_mix")
    parent_summary = ROOT / "outputs" / "stage5" / run_id / "summary.json"
    print(f"autopilot_run_id={run_id}")
    print(f"parent_summary={path_for_cli(parent_summary)} status={summary_status(parent_summary)}")
    print(f"distill_summary={path_for_cli(distill_summary)} status={summary_status(distill_summary)}")
    print(f"arc_mix_summary={path_for_cli(arc_mix_summary)} status={summary_status(arc_mix_summary)}")
    print()

    checkpoint_run_id = infer_stage5_run_id(checkpoint) if checkpoint else None
    checkpoint_available = bool(
        checkpoint
        and (
            checkpoint.exists()
            or (
                checkpoint_run_id is not None
                and any(path.exists() for path in candidate_drive_checkpoints(checkpoint_run_id, checkpoint.name))
            )
        )
    )
    if parent_summary.exists():
        print("next_action=Review committed autopilot summary or rerun only if it failed before push.")
    elif distill_summary.exists() or arc_mix_summary.exists():
        print("next_action=Rerun colab/run_stage5_balanced_recovery_autopilot.py with the same run id to resume.")
    elif checkpoint_available:
        print("next_action=Run colab/run_stage5_balanced_recovery_autopilot.py.")
    else:
        print("next_action=Fix Drive mount/checkpoint restore before starting the A100 job.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
