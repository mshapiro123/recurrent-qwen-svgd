"""Run full balanced MCQ assessment for a passed Stage 5 recovery gate.

Input is normally the fixed-run autopilot summary produced by
``run_stage5_balanced_recovery_autopilot.py``. When that recovery gate passes,
this script extracts the selected child checkpoint, runs full ARC-Easy and full
ARC-Challenge through the benchmark suite, then runs the balanced MCQ
assessment over the fresh benchmark summary.
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
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from colab.run_stage5_recovered_phase1_arc_gate import (  # noqa: E402
    path_for_cli,
    restore_checkpoint_if_needed,
)


RUN_ID = os.environ.get("STAGE5_RECOVERY_FULL_ASSESS_RUN_ID") or time.strftime(
    "stage5_recovery_full_assessment_%Y%m%d_%H%M%S"
)
RUN_DIR = ROOT / "outputs" / "stage5" / RUN_ID
SOURCE_SUMMARY = Path(
    os.environ.get(
        "STAGE5_RECOVERY_FULL_ASSESS_SOURCE_SUMMARY",
        "outputs/stage5/stage5_balanced_recovery_autopilot_current/summary.json",
    )
)
if not SOURCE_SUMMARY.is_absolute():
    SOURCE_SUMMARY = ROOT / SOURCE_SUMMARY

PUSH_RESULTS = os.environ.get("STAGE5_RECOVERY_FULL_ASSESS_PUSH", "1").strip().lower() in {
    "1",
    "true",
    "yes",
    "y",
}


def read_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def run(
    cmd: list[str],
    *,
    env: dict[str, str] | None = None,
    check: bool = True,
    log_name: str | None = None,
) -> subprocess.CompletedProcess[str]:
    print("$", " ".join(map(str, cmd)), flush=True)
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
    proc = subprocess.CompletedProcess(cmd, process.wait(), stdout, None)
    if log_name:
        RUN_DIR.mkdir(parents=True, exist_ok=True)
        (RUN_DIR / log_name).write_text(stdout, encoding="utf-8")
    if check and proc.returncode:
        raise RuntimeError(f"command failed: {' '.join(map(str, cmd))}")
    return proc


def infer_stage5_run_id(path: str | Path) -> str:
    parts = Path(path).parts
    for idx, part in enumerate(parts):
        if part == "stage5" and idx + 1 < len(parts):
            return parts[idx + 1]
    raise ValueError(f"Could not infer Stage 5 run id from checkpoint path: {path}")


def resolve_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def checkpoint_from_gate_payload(payload: dict[str, Any]) -> str | None:
    best = payload.get("best_arm") or payload.get("best_checkpoint") or {}
    if isinstance(best, dict):
        checkpoint = best.get("checkpoint")
        if checkpoint:
            return str(checkpoint)
        nested = best.get("best_checkpoint")
        if isinstance(nested, dict) and nested.get("checkpoint"):
            return str(nested["checkpoint"])
    checkpoint = payload.get("checkpoint")
    return str(checkpoint) if checkpoint else None


def selected_gate_payload(source_payload: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    kind = source_payload.get("kind")
    if kind != "stage5_balanced_recovery_autopilot":
        if source_payload.get("passed"):
            return ("source", source_payload)
        raise ValueError(f"Source summary is not a passed recovery gate: kind={kind!r}")
    status = source_payload.get("status")
    if status == "distill_gate_passed":
        payload = source_payload.get("distill")
        if isinstance(payload, dict):
            return ("distill", payload)
    if status == "arc_mix_gate_passed":
        payload = source_payload.get("arc_mix")
        if isinstance(payload, dict):
            return ("arc_mix", payload)
    raise ValueError(f"Recovery autopilot did not pass a gate: status={status!r}")


def selected_checkpoint(source_payload: dict[str, Any]) -> tuple[str, Path, dict[str, Any]]:
    gate_name, gate_payload = selected_gate_payload(source_payload)
    checkpoint = checkpoint_from_gate_payload(gate_payload)
    if not checkpoint:
        raise ValueError(f"Could not find checkpoint in selected {gate_name} payload")
    return gate_name, resolve_path(checkpoint), gate_payload


def run_full_benchmark(checkpoint: Path) -> Path:
    benchmark_run_id = f"{RUN_ID}_balanced_full"
    benchmark_summary = ROOT / "outputs" / "stage5" / benchmark_run_id / "summary.json"
    if benchmark_summary.exists():
        print(f"reusing_benchmark_summary={path_for_cli(benchmark_summary)}")
        return benchmark_summary
    env = os.environ.copy()
    env.update(
        {
            "STAGE5_BENCHMARK_SUITE_RUN_ID": benchmark_run_id,
            "STAGE5_BENCHMARK_CHECKPOINT": path_for_cli(checkpoint),
            "STAGE5_BENCHMARK_SOURCE_SUMMARY": path_for_cli(SOURCE_SUMMARY),
            "STAGE5_BENCHMARKS": "arc_easy,arc_challenge",
            "STAGE5_BENCHMARK_ARC_EASY_LIMIT": "full",
            "STAGE5_BENCHMARK_ARC_CHALLENGE_LIMIT": "full",
            "STAGE5_BENCHMARK_SCORE_TARGETS": "label",
            "STAGE5_BENCHMARK_AGGREGATES": "mean",
            "STAGE5_BENCHMARK_RECURRENT_MODE": "phase1",
            "STAGE5_BENCHMARK_NUM_TRAJECTORIES": "1",
            "STAGE5_BENCHMARK_CONTINUE_ON_FAILURE": "0",
            "STAGE5_BENCHMARK_PUSH": "0",
        }
    )
    run([sys.executable, "colab/run_stage5_benchmark_suite.py"], env=env, log_name="benchmark_suite.log")
    return benchmark_summary


def run_balanced_assessment(benchmark_summary: Path) -> Path:
    assessment_dir = RUN_DIR / "balanced_assessment"
    run(
        [
            sys.executable,
            "colab/assess_stage5_balanced_mcq.py",
            "--arc_easy_sweep",
            path_for_cli(benchmark_summary),
            "--arc_challenge_summary",
            path_for_cli(benchmark_summary),
            "--output_dir",
            path_for_cli(assessment_dir),
        ],
        log_name="balanced_assessment.log",
    )
    return assessment_dir / "summary.json"


def write_report(payload: dict[str, Any]) -> None:
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    (RUN_DIR / "summary.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    lines = [
        f"# Stage 5 Recovery Full Assessment - {RUN_ID}",
        "",
        f"- Status: `{payload['status']}`",
        f"- Passed: `{payload['passed']}`",
        f"- Source summary: `{payload['source_summary']}`",
        f"- Selected gate: `{payload['selected_gate']}`",
        f"- Selected checkpoint: `{payload['selected_checkpoint']}`",
        f"- Benchmark summary: `{payload['benchmark_summary']}`",
        f"- Balanced assessment: `{payload['balanced_assessment_summary']}`",
        f"- Next step: {payload['next_step']}",
    ]
    (RUN_DIR / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print((RUN_DIR / "summary.md").read_text(encoding="utf-8"))


def backup_to_drive(paths: list[Path]) -> None:
    try:
        from google.colab import drive  # type: ignore

        if not Path("/content/drive/MyDrive").exists():
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
    for path in paths:
        if not path.exists():
            continue
        target = backup / "outputs" / "stage5" / path.name
        if path.is_dir():
            shutil.copytree(path, target, dirs_exist_ok=True)
        else:
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, target)
    print(f"backed_up_to={backup}")


def commit_results(paths: list[Path]) -> None:
    if not PUSH_RESULTS:
        return
    for path in paths:
        if path.exists():
            run(["git", "add", "-f", path_for_cli(path)], check=False)
    if run(["git", "diff", "--cached", "--quiet"], check=False).returncode == 0:
        print("No recovery full assessment outputs changed.")
        return
    run(["git", "commit", "-m", f"Record Stage 5 recovery full assessment {RUN_ID}"])
    run(["git", "push", "origin", "main"], check=False)


def main() -> int:
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    if not SOURCE_SUMMARY.exists():
        raise FileNotFoundError(f"Missing source summary: {SOURCE_SUMMARY}")
    source_payload = read_json(SOURCE_SUMMARY)
    selected_gate, checkpoint, gate_payload = selected_checkpoint(source_payload)
    restore_checkpoint_if_needed(checkpoint, run_id=infer_stage5_run_id(checkpoint))

    benchmark_summary = run_full_benchmark(checkpoint)
    assessment_summary = run_balanced_assessment(benchmark_summary)
    assessment_payload = read_json(assessment_summary)
    payload = {
        "run_id": RUN_ID,
        "kind": "stage5_recovery_full_assessment",
        "source_summary": path_for_cli(SOURCE_SUMMARY),
        "source_status": source_payload.get("status"),
        "selected_gate": selected_gate,
        "selected_gate_status": gate_payload.get("status"),
        "selected_checkpoint": path_for_cli(checkpoint),
        "benchmark_summary": path_for_cli(benchmark_summary),
        "balanced_assessment_summary": path_for_cli(assessment_summary),
        "status": assessment_payload.get("status"),
        "passed": assessment_payload.get("passed"),
        "next_step": assessment_payload.get("next_step"),
        "best_checkpoint": assessment_payload.get("best_checkpoint"),
        "balanced_assessment": assessment_payload,
    }
    write_report(payload)
    result_paths = [RUN_DIR, benchmark_summary.parent]
    backup_to_drive(result_paths)
    commit_results(result_paths)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
