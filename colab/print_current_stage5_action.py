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
PREFERRED_TARGET = "traced_sft_score_alignment_repair"


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
    print(f"preferred_target={PREFERRED_TARGET}")
    print("recent_commits:")
    subprocess.run(["git", "log", "--oneline", "-5"], cwd=ROOT, check=False)
    source = read_pointer()
    if source is None:
        print(f"current_source_summary_pointer_missing={path_for_cli(POINTER)}")
        return 0
    print(f"current_source_summary={path_for_cli(source)}")
    if source.exists():
        payload = json.loads(source.read_text(encoding="utf-8"))
        print(f"source_kind={payload.get('kind') or payload.get('gate')}")
        print(f"source_status={payload.get('status')}")
        print(f"source_passed={payload.get('passed')}")
    else:
        print("source_summary_missing=true")
        return 0
    print("planner_top_action:")
    subprocess.run(
        [sys.executable, "colab/plan_stage5_next_run.py", "--source-summary", path_for_cli(source)],
        cwd=ROOT,
        check=False,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
