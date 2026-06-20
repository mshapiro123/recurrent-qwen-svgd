"""Compare grid-only ARC-AGI SFT against symbolic-trace ARC-AGI SFT.

This is the training counterpart to the candidate-source gate. By default it
runs two matched `run_stage5_arc_agi_sft.py` child jobs:

1. `STAGE5_ARC_AGI_TRACE_MODE=none`
2. `STAGE5_ARC_AGI_TRACE_MODE=symbolic`, `STAGE5_ARC_AGI_TRACE_FILTER=covered`

The goal is to determine whether explicit compact transformation traces are a
better training target than final-grid-only supervision before scaling the ARC
recipe.
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
RUN_ID = os.environ.get("STAGE5_ARC_AGI_TRACE_SFT_GATE_RUN_ID") or time.strftime(
    "stage5_arc_agi_trace_sft_gate_%Y%m%d_%H%M%S"
)
RUN_DIR = ROOT / "outputs" / "stage5" / RUN_ID
RUN_DIR.mkdir(parents=True, exist_ok=True)
PUSH_RESULTS = os.environ.get("STAGE5_ARC_AGI_TRACE_SFT_GATE_PUSH", "1").strip().lower() in {
    "1",
    "true",
    "yes",
    "y",
}
DEFAULT_ARMS = "grid_only,symbolic_trace_covered"


def run(cmd: list[str], *, env: dict[str, str] | None = None, check: bool = True, log_name: str | None = None):
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


def child_run_id(label: str) -> str:
    return f"{RUN_ID}_{label}"


def child_summary(label: str) -> dict[str, Any]:
    path = ROOT / "outputs" / "stage5" / child_run_id(label) / "summary.json"
    return json.loads(path.read_text(encoding="utf-8"))


def available_arms() -> dict[str, tuple[str, str]]:
    return {
        "grid_only": ("none", "all"),
        "symbolic_trace_all": ("symbolic", "all"),
        "symbolic_trace_covered": ("symbolic", "covered"),
    }


def requested_arms() -> list[tuple[str, str, str]]:
    names = [item.strip() for item in os.environ.get("STAGE5_ARC_AGI_TRACE_SFT_GATE_ARMS", DEFAULT_ARMS).split(",")]
    names = [name for name in names if name]
    arms = available_arms()
    unknown = set(names) - set(arms)
    if unknown:
        raise ValueError(f"Unknown trace SFT gate arms: {sorted(unknown)}")
    return [(name, *arms[name]) for name in names]


def run_child(label: str, trace_mode: str, trace_filter: str) -> dict[str, Any]:
    env = os.environ.copy()
    env["STAGE5_ARC_AGI_SFT_RUN_ID"] = child_run_id(label)
    env["STAGE5_ARC_AGI_TRACE_MODE"] = trace_mode
    env["STAGE5_ARC_AGI_TRACE_FILTER"] = trace_filter
    env["STAGE5_ARC_AGI_SFT_PUSH"] = "0"
    run([sys.executable, "colab/run_stage5_arc_agi_sft.py"], env=env, log_name=f"{label}.log")
    return child_summary(label)


def compact(summary: dict[str, Any]) -> dict[str, Any]:
    tuned = summary["phase1_arc_agi_tuned"]
    start = summary["phase1_start"]
    return {
        "base_selected": summary["base"]["selected_exact"],
        "base_best": summary["base"]["best_of_k_exact"],
        "start_selected": start["selected_exact"],
        "start_best": start["best_of_k_exact"],
        "tuned_selected": tuned["selected_exact"],
        "tuned_best": tuned["best_of_k_exact"],
        "tuned_valid_rate": tuned["valid_candidate_rate"],
        "examples": tuned["examples_with_targets"],
        "tasks_solved_best": tuned["tasks_solved_best_of_k"],
        "tasks": tuned["tasks_with_targets"],
    }


def backup_to_drive() -> None:
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
    for label, _, _ in requested_arms():
        child_dir = ROOT / "outputs" / "stage5" / child_run_id(label)
        if child_dir.exists():
            shutil.copytree(child_dir, backup / label, dirs_exist_ok=True)
    print(f"backed_up_to={backup}")


def git_commit_results() -> None:
    for pattern in ["*.json", "*.md", "*.log"]:
        run(["git", "add", "-f", str((RUN_DIR / pattern).relative_to(ROOT))], check=False)
    if run(["git", "diff", "--cached", "--quiet"], check=False).returncode == 0:
        print("No ARC-AGI trace SFT gate outputs changed.")
        return
    run(["git", "commit", "-m", "Record Stage 5 ARC-AGI trace SFT gate"])
    run(["git", "push", "origin", "main"], check=False)


def main() -> int:
    if any(arg in {"-h", "--help"} for arg in sys.argv[1:]):
        print("Run from Colab after Stage 5 checkpoints are available.")
        return 0

    arms = requested_arms()
    results = {label: run_child(label, trace_mode, trace_filter) for label, trace_mode, trace_filter in arms}
    payload = {
        "run_id": RUN_ID,
        "arms": [{"label": label, "trace_mode": mode, "trace_filter": trace_filter} for label, mode, trace_filter in arms],
        "results": results,
        "comparison": {label: compact(summary) for label, summary in results.items()},
    }
    (RUN_DIR / "summary.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    rows = payload["comparison"]
    lines = [
        f"# Stage 5 ARC-AGI Trace SFT Gate - {RUN_ID}",
        "",
        "| Arm | Tuned selected | Tuned best | Tasks best | Valid rate |",
        "|---|---:|---:|---:|---:|",
    ]
    for label, row in rows.items():
        lines.append(
            f"| `{label}` | {row['tuned_selected']}/{row['examples']} | "
            f"{row['tuned_best']}/{row['examples']} | "
            f"{row['tasks_solved_best']}/{row['tasks']} | {row['tuned_valid_rate']:.4f} |"
        )
    lines += [
        "",
        "Gate: symbolic-trace SFT should beat or match grid-only SFT before scaling this recipe. "
        "The default symbolic arm keeps only examples covered by exact symbolic traces.",
    ]
    (RUN_DIR / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print((RUN_DIR / "summary.md").read_text(encoding="utf-8"))
    backup_to_drive()
    if PUSH_RESULTS:
        git_commit_results()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
