"""Print the current Stage 5 source summary and top next action.

This is a cheap Colab sanity check before spending GPU.  It does not train or
download models; it reads the current source-summary pointer and delegates to
the planner so stale notebooks are easier to catch.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
POINTER = ROOT / "config" / "stage5_current_source_summary.txt"
PROGRAM_PHASE = "Phase 0: loop-closure re-entry, then Phase 1 deterministic depth recovery/control"
PREFERRED_STATUS_TARGET = "master_sequence_status"


def path_for_cli(path: Path) -> str:
    try:
        return path.relative_to(ROOT).as_posix()
    except ValueError:
        return str(path).replace("\\", "/")


def read_pointer() -> Path | None:
    if not POINTER.exists():
        return None
    raw = POINTER.read_text(encoding="utf-8").strip()
    if not raw:
        return None
    path = Path(raw.replace("\\", "/"))
    return path if path.is_absolute() else ROOT / path


def main() -> int:
    print(f"program_phase={PROGRAM_PHASE}", flush=True)
    print(f"preferred_status_target={PREFERRED_STATUS_TARGET}", flush=True)
    print("recent_commits:", flush=True)
    subprocess.run(["git", "log", "--oneline", "-5"], cwd=ROOT, check=False)
    source = read_pointer()
    if source is None:
        print(f"current_source_summary_pointer_missing={path_for_cli(POINTER)}", flush=True)
        return 0
    print(f"current_source_summary={path_for_cli(source)}", flush=True)
    if source.exists():
        payload = json.loads(source.read_text(encoding="utf-8"))
        print(f"source_kind={payload.get('kind') or payload.get('gate')}", flush=True)
        print(f"source_status={payload.get('status')}", flush=True)
        print(f"source_passed={payload.get('passed')}", flush=True)
    else:
        print("source_summary_missing=true", flush=True)
        return 0
    print("planner_top_action:", flush=True)
    subprocess.run(
        [sys.executable, "colab/plan_stage5_next_run.py", "--source-summary", path_for_cli(source)],
        cwd=ROOT,
        check=False,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
