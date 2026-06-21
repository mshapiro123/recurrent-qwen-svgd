"""Run Stage 5 ARC-AGI gates with thresholded branching.

This keeps one Colab runtime moving without hiding the experimental decisions.
It runs:

1. candidate-source gate;
2. trace-SFT gate, only if symbolic/hybrid candidates clear thresholds;
3. distillation SFT gate, only if symbolic-trace SFT clears thresholds.

All child gates write their own outputs under `outputs/stage5/<run_id>_*`.
This script writes one aggregate decision report.
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
RUN_ID = os.environ.get("STAGE5_ARC_AGI_AUTOPILOT_RUN_ID") or time.strftime("stage5_arc_agi_autopilot_%Y%m%d_%H%M%S")
RUN_DIR = ROOT / "outputs" / "stage5" / RUN_ID
RUN_DIR.mkdir(parents=True, exist_ok=True)
PUSH_RESULTS = os.environ.get("STAGE5_ARC_AGI_AUTOPILOT_PUSH", "1").strip().lower() in {"1", "true", "yes", "y"}

MIN_SYMBOLIC_EXACT = int(os.environ.get("STAGE5_ARC_AGI_AUTOPILOT_MIN_SYMBOLIC_EXACT", "1"))
MIN_HYBRID_BEST_DELTA = int(os.environ.get("STAGE5_ARC_AGI_AUTOPILOT_MIN_HYBRID_BEST_DELTA", "0"))
MIN_TRACE_BEST_DELTA = int(os.environ.get("STAGE5_ARC_AGI_AUTOPILOT_MIN_TRACE_BEST_DELTA", "0"))
DEFAULT_TRACE_SFT_GATE_ARMS = os.environ.get(
    "STAGE5_ARC_AGI_AUTOPILOT_TRACE_SFT_GATE_ARMS",
    "grid_only,symbolic_program_trace_covered,symbolic_state_trace_covered",
)


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


def run_child(label: str, script: str, env_updates: dict[str, str], *, log_name: str) -> dict[str, Any]:
    summary_path = ROOT / "outputs" / "stage5" / child_run_id(label) / "summary.json"
    if summary_path.exists() and os.environ.get("STAGE5_ARC_AGI_AUTOPILOT_RESUME", "1") in {"1", "true", "yes"}:
        print(f"reusing_child_summary={summary_path}")
        return child_summary(label)
    env = os.environ.copy()
    env.update(env_updates)
    run([sys.executable, script], env=env, log_name=log_name)
    return child_summary(label)


def rows_by_variant(candidate_gate: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {row["variant"]: row for row in candidate_gate.get("rows", [])}


def decide_trace_gate(candidate_gate: dict[str, Any]) -> tuple[bool, dict[str, Any]]:
    coverage = candidate_gate.get("symbolic_coverage", {})
    rows = rows_by_variant(candidate_gate)
    phase1_model = rows.get("phase1_model_only", {})
    phase1_hybrid = rows.get("phase1_hybrid_symbolic_first", {})
    base_model = rows.get("base_model_only", {})
    base_hybrid = rows.get("base_hybrid_symbolic_first", {})
    symbolic_exact = int(coverage.get("exact_symbolic", 0))
    phase1_delta = int(phase1_hybrid.get("best", 0)) - int(phase1_model.get("best", 0))
    base_delta = int(base_hybrid.get("best", 0)) - int(base_model.get("best", 0))
    best_delta = max(phase1_delta, base_delta)
    decision = symbolic_exact >= MIN_SYMBOLIC_EXACT and best_delta >= MIN_HYBRID_BEST_DELTA
    evidence = {
        "symbolic_exact": symbolic_exact,
        "min_symbolic_exact": MIN_SYMBOLIC_EXACT,
        "phase1_hybrid_best_delta": phase1_delta,
        "base_hybrid_best_delta": base_delta,
        "best_hybrid_delta": best_delta,
        "min_hybrid_best_delta": MIN_HYBRID_BEST_DELTA,
        "run_trace_sft_gate": decision,
    }
    return decision, evidence


def trace_arm_settings(arm: str) -> tuple[str | None, str | None]:
    if arm == "grid_only":
        return "none", "all"
    if arm.endswith("_covered"):
        trace_filter = "covered"
        prefix = arm[: -len("_covered")]
    elif arm.endswith("_all"):
        trace_filter = "all"
        prefix = arm[: -len("_all")]
    else:
        return None, None

    mode_by_prefix = {
        "symbolic_trace": "symbolic",
        "symbolic_program_trace": "symbolic_program",
        "symbolic_state_trace": "symbolic_state_trace",
    }
    return mode_by_prefix.get(prefix), trace_filter


def best_trace_arm(comparison: dict[str, Any]) -> tuple[str | None, dict[str, Any]]:
    grid = comparison.get("grid_only", {})
    candidates: list[tuple[tuple[int, int, int, str], str, dict[str, Any]]] = []
    for arm, row in comparison.items():
        if arm == "grid_only":
            continue
        mode, trace_filter = trace_arm_settings(arm)
        if mode is None or trace_filter is None:
            continue
        best_delta = int(row.get("tuned_best", 0)) - int(grid.get("tuned_best", 0))
        selected_delta = int(row.get("tuned_selected", 0)) - int(grid.get("tuned_selected", 0))
        checkpoint_delta = int(row.get("best_best", row.get("tuned_best", 0))) - int(
            grid.get("best_best", grid.get("tuned_best", 0))
        )
        candidates.append(((best_delta, selected_delta, checkpoint_delta, arm), arm, row))
    if not candidates:
        return None, {}
    _, arm, row = max(candidates, key=lambda item: item[0])
    return arm, row


def decide_distill_gate(trace_gate: dict[str, Any]) -> tuple[bool, dict[str, Any]]:
    comparison = trace_gate.get("comparison", {})
    grid = comparison.get("grid_only", {})
    arm, trace = best_trace_arm(comparison)
    trace_mode, trace_filter = trace_arm_settings(arm or "")
    best_delta = int(trace.get("tuned_best", 0)) - int(grid.get("tuned_best", 0))
    selected_delta = int(trace.get("tuned_selected", 0)) - int(grid.get("tuned_selected", 0))
    decision = best_delta >= MIN_TRACE_BEST_DELTA
    evidence = {
        "best_trace_arm": arm,
        "best_trace_mode": trace_mode,
        "best_trace_filter": trace_filter,
        "trace_best_delta": best_delta,
        "trace_selected_delta": selected_delta,
        "min_trace_best_delta": MIN_TRACE_BEST_DELTA,
        "run_distill_gate": decision,
    }
    return decision, evidence


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
    for label in ["candidate_gate", "trace_sft_gate", "distill_sft_gate"]:
        child_dir = ROOT / "outputs" / "stage5" / child_run_id(label)
        if child_dir.exists():
            shutil.copytree(child_dir, backup / label, dirs_exist_ok=True)
    print(f"backed_up_to={backup}")


def git_commit_results() -> None:
    for pattern in ["*.json", "*.md", "*.log"]:
        run(["git", "add", "-f", str((RUN_DIR / pattern).relative_to(ROOT))], check=False)
    if run(["git", "diff", "--cached", "--quiet"], check=False).returncode == 0:
        print("No ARC-AGI autopilot outputs changed.")
        return
    run(["git", "commit", "-m", "Record Stage 5 ARC-AGI autopilot"])
    run(["git", "push", "origin", "main"], check=False)


def write_report(payload: dict[str, Any]) -> None:
    (RUN_DIR / "summary.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    lines = [
        f"# Stage 5 ARC-AGI Autopilot - {RUN_ID}",
        "",
        "## Decisions",
        "",
    ]
    for item in payload["decisions"]:
        lines.extend(
            [
                f"### {item['name']}",
                "",
                f"- Ran: `{item['ran']}`",
                f"- Reason: {item['reason']}",
                f"- Evidence: `{item.get('evidence', {})}`",
                "",
            ]
        )
    lines.extend(
        [
            "## Thresholds",
            "",
            f"- Min symbolic exact: `{MIN_SYMBOLIC_EXACT}`",
            f"- Min hybrid best delta: `{MIN_HYBRID_BEST_DELTA}`",
            f"- Min trace best delta: `{MIN_TRACE_BEST_DELTA}`",
            f"- Trace SFT gate arms: `{DEFAULT_TRACE_SFT_GATE_ARMS}`",
        ]
    )
    (RUN_DIR / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print((RUN_DIR / "summary.md").read_text(encoding="utf-8"))


def main() -> int:
    if any(arg in {"-h", "--help"} for arg in sys.argv[1:]):
        print("Run from Colab to execute Stage 5 ARC-AGI gates with thresholded branching.")
        return 0

    decisions: list[dict[str, Any]] = []
    candidate_gate = run_child(
        "candidate_gate",
        "colab/run_stage5_arc_agi_candidate_gate.py",
        {
            "STAGE5_ARC_AGI_GATE_RUN_ID": child_run_id("candidate_gate"),
            "STAGE5_ARC_AGI_GATE_PUSH": "0",
        },
        log_name="candidate_gate.log",
    )
    run_trace, trace_evidence = decide_trace_gate(candidate_gate)
    decisions.append(
        {
            "name": "candidate_gate",
            "ran": True,
            "reason": "candidate source value gate always runs first",
            "evidence": trace_evidence,
            "summary_run_id": child_run_id("candidate_gate"),
        }
    )

    trace_gate = None
    if run_trace:
        trace_gate = run_child(
            "trace_sft_gate",
            "colab/run_stage5_arc_agi_trace_sft_gate.py",
            {
                "STAGE5_ARC_AGI_TRACE_SFT_GATE_RUN_ID": child_run_id("trace_sft_gate"),
                "STAGE5_ARC_AGI_TRACE_SFT_GATE_ARMS": DEFAULT_TRACE_SFT_GATE_ARMS,
                "STAGE5_ARC_AGI_TRACE_SFT_GATE_PUSH": "0",
            },
            log_name="trace_sft_gate.log",
        )
        run_distill, distill_evidence = decide_distill_gate(trace_gate)
        decisions.append(
            {
                "name": "trace_sft_gate",
                "ran": True,
                "reason": "candidate gate cleared symbolic/hybrid thresholds",
                "evidence": distill_evidence,
                "summary_run_id": child_run_id("trace_sft_gate"),
            }
        )
    else:
        run_distill = False
        distill_evidence = {}
        decisions.append(
            {
                "name": "trace_sft_gate",
                "ran": False,
                "reason": "candidate gate did not clear symbolic/hybrid thresholds",
                "evidence": trace_evidence,
            }
        )

    distill_gate = None
    if run_distill:
        distill_gate = run_child(
            "distill_sft_gate",
            "colab/run_stage5_arc_agi_distill_sft_gate.py",
            {
                "STAGE5_ARC_AGI_DISTILL_GATE_RUN_ID": child_run_id("distill_sft_gate"),
                "STAGE5_ARC_AGI_TRACE_MODE": str(distill_evidence.get("best_trace_mode") or "symbolic_program"),
                "STAGE5_ARC_AGI_TRACE_FILTER": str(distill_evidence.get("best_trace_filter") or "covered"),
                "STAGE5_ARC_AGI_DISTILL_GATE_PUSH": "0",
            },
            log_name="distill_sft_gate.log",
        )
        decisions.append(
            {
                "name": "distill_sft_gate",
                "ran": True,
                "reason": "trace-SFT gate cleared threshold",
                "summary_run_id": child_run_id("distill_sft_gate"),
            }
        )
    else:
        decisions.append(
            {
                "name": "distill_sft_gate",
                "ran": False,
                "reason": "trace-SFT gate did not run or did not clear threshold",
                "evidence": distill_evidence,
            }
        )

    payload = {
        "run_id": RUN_ID,
        "thresholds": {
            "min_symbolic_exact": MIN_SYMBOLIC_EXACT,
            "min_hybrid_best_delta": MIN_HYBRID_BEST_DELTA,
            "min_trace_best_delta": MIN_TRACE_BEST_DELTA,
            "trace_sft_gate_arms": DEFAULT_TRACE_SFT_GATE_ARMS,
        },
        "decisions": decisions,
        "candidate_gate": candidate_gate,
        "trace_sft_gate": trace_gate,
        "distill_sft_gate": distill_gate,
    }
    write_report(payload)
    backup_to_drive()
    if PUSH_RESULTS:
        git_commit_results()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
