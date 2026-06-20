"""Run safe follow-up evaluations from a Stage 5 autopilot summary.

The curriculum-particle autopilot writes a decision report. This runner turns
that report into the next contained Colab step:

* if candidate distillation failed, it records the failure and optionally starts
  a baseline curriculum branch only when explicitly allowed;
* if a recovered checkpoint exists, it runs the recovered-vs-base ARC benchmark;
* it then runs the geometry TTA / selector sweep against the same checkpoint.

This script is evaluation-first by default. It does not start more training
unless ``STAGE5_ARC_AGI_FOLLOWUP_ALLOW_BASELINE_CURRICULUM=1`` is set.
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
RUN_ID = os.environ.get("STAGE5_ARC_AGI_FOLLOWUP_RUN_ID") or time.strftime(
    "stage5_arc_agi_autopilot_followup_%Y%m%d_%H%M%S"
)
RUN_DIR = ROOT / "outputs" / "stage5" / RUN_ID
RUN_DIR.mkdir(parents=True, exist_ok=True)

AUTOPILOT_SUMMARY = os.environ.get("STAGE5_ARC_AGI_AUTOPILOT_SUMMARY", "")
LIMIT = int(os.environ.get("STAGE5_ARC_AGI_FOLLOWUP_LIMIT", os.environ.get("STAGE5_ARC_AGI_LIMIT", "50")))
RUN_RECOVERED_BENCHMARK = os.environ.get("STAGE5_ARC_AGI_FOLLOWUP_RUN_RECOVERED_BENCHMARK", "1").strip().lower() in {
    "1",
    "true",
    "yes",
    "y",
}
RUN_TTA_SWEEP = os.environ.get("STAGE5_ARC_AGI_FOLLOWUP_RUN_TTA_SWEEP", "1").strip().lower() in {
    "1",
    "true",
    "yes",
    "y",
}
ALLOW_BASELINE_CURRICULUM = os.environ.get(
    "STAGE5_ARC_AGI_FOLLOWUP_ALLOW_BASELINE_CURRICULUM",
    "0",
).strip().lower() in {"1", "true", "yes", "y"}
PUSH_RESULTS = os.environ.get("STAGE5_ARC_AGI_FOLLOWUP_PUSH", "1").strip().lower() in {"1", "true", "yes", "y"}


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


def resolve_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def path_for_cli(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def read_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def latest_autopilot_summary() -> Path:
    candidates: list[Path] = []
    for path in ROOT.glob("outputs/stage5/*/summary.json"):
        try:
            payload = read_json(path)
        except Exception:
            continue
        if "compact" in payload and (
            "candidate_distillation_passed" in payload["compact"] or "particle_passed" in payload["compact"]
        ):
            candidates.append(path)
    if not candidates:
        raise FileNotFoundError(
            "No Stage 5 autopilot summary found. Set STAGE5_ARC_AGI_AUTOPILOT_SUMMARY."
        )
    return sorted(candidates, key=lambda path: path.stat().st_mtime, reverse=True)[0]


def resolve_autopilot_summary() -> Path:
    return resolve_path(AUTOPILOT_SUMMARY) if AUTOPILOT_SUMMARY else latest_autopilot_summary()


def curriculum_summary_path(autopilot: dict[str, Any]) -> Path | None:
    child_run_id = (autopilot.get("child_run_ids") or {}).get("curriculum")
    if child_run_id:
        return ROOT / "outputs" / "stage5" / child_run_id / "summary.json"
    action_commands = [
        action.get("command", "")
        for action in (autopilot.get("compact") or {}).get("recommended_next_actions", [])
    ]
    for command in action_commands:
        prefix = "STAGE5_ARC_AGI_CURRICULUM_SUMMARY="
        if prefix in command:
            value = command.split(prefix, 1)[1].split(" ", 1)[0]
            return resolve_path(value)
    return None


def should_run_eval_followups(autopilot: dict[str, Any]) -> tuple[bool, str]:
    compact = autopilot.get("compact") or {}
    if not compact.get("candidate_distillation_passed"):
        return False, "candidate distillation gate did not pass"
    if not compact.get("final_checkpoint"):
        return False, "autopilot did not record a final checkpoint"
    curriculum_summary = curriculum_summary_path(autopilot)
    if curriculum_summary is None:
        return False, "autopilot did not record a curriculum summary path"
    if not curriculum_summary.exists():
        return False, f"missing curriculum summary: {curriculum_summary}"
    return True, "recovered checkpoint available"


def run_child(label: str, script: str, env_updates: dict[str, str]) -> dict[str, Any]:
    child_summary = ROOT / "outputs" / "stage5" / env_updates[f"STAGE5_ARC_AGI_{label}_RUN_ID"] / "summary.json"
    if child_summary.exists() and os.environ.get("STAGE5_ARC_AGI_FOLLOWUP_RESUME", "1").strip().lower() in {
        "1",
        "true",
        "yes",
        "y",
    }:
        print(f"reusing_child_summary={child_summary}")
        return read_json(child_summary)
    env = os.environ.copy()
    env.update(env_updates)
    run([sys.executable, script], env=env, log_name=f"{label.lower()}.log")
    return read_json(child_summary)


def run_recovered_benchmark(curriculum_summary: Path) -> dict[str, Any]:
    return run_child(
        "RECOVERED_BENCHMARK",
        "colab/run_stage5_arc_agi_recovered_benchmark.py",
        {
            "STAGE5_ARC_AGI_RECOVERED_BENCHMARK_RUN_ID": f"{RUN_ID}_recovered_benchmark_limit{LIMIT}",
            "STAGE5_ARC_AGI_RECOVERED_BENCHMARK_PUSH": "0",
            "STAGE5_ARC_AGI_CURRICULUM_SUMMARY": path_for_cli(curriculum_summary),
            "STAGE5_ARC_AGI_LIMIT": str(LIMIT),
        },
    )


def run_tta_sweep(curriculum_summary: Path) -> dict[str, Any]:
    return run_child(
        "TTA_SWEEP",
        "colab/run_stage5_arc_agi_tta_sweep.py",
        {
            "STAGE5_ARC_AGI_TTA_SWEEP_RUN_ID": f"{RUN_ID}_tta_sweep_limit{LIMIT}",
            "STAGE5_ARC_AGI_TTA_SWEEP_PUSH": "0",
            "STAGE5_ARC_AGI_CURRICULUM_SUMMARY": path_for_cli(curriculum_summary),
            "STAGE5_ARC_AGI_LIMIT": str(LIMIT),
        },
    )


def run_baseline_curriculum() -> dict[str, Any]:
    child_id = f"{RUN_ID}_baseline_no_candidate_distill"
    env = os.environ.copy()
    env.update(
        {
            "STAGE5_ARC_AGI_CURRICULUM_PARTICLE_AUTOPILOT_RUN_ID": child_id,
            "STAGE5_ARC_AGI_CURRICULUM_PARTICLE_AUTOPILOT_RUN_CANDIDATE_DISTILL_GATE": "0",
            "STAGE5_ARC_AGI_CURRICULUM_PARTICLE_AUTOPILOT_PUSH": "0",
        }
    )
    run(
        [sys.executable, "colab/run_stage5_arc_agi_curriculum_particle_autopilot.py"],
        env=env,
        log_name="baseline_no_candidate_distill.log",
    )
    return read_json(ROOT / "outputs" / "stage5" / child_id / "summary.json")


def compare_recovered_to_base(benchmark: dict[str, Any] | None) -> dict[str, Any] | None:
    if not benchmark:
        return None
    deltas = benchmark.get("deltas") or {}
    return deltas.get("recovered_vs_base")


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
    print(f"backed_up_to={backup}")


def git_commit_results() -> None:
    for pattern in ["*.json", "*.md", "*.log"]:
        run(["git", "add", "-f", str((RUN_DIR / pattern).relative_to(ROOT))], check=False)
    if run(["git", "diff", "--cached", "--quiet"], check=False).returncode == 0:
        print("No autopilot follow-up outputs changed.")
        return
    run(["git", "commit", "-m", "Record Stage 5 autopilot follow-up"])
    run(["git", "push", "origin", "main"], check=False)


def write_report(payload: dict[str, Any]) -> None:
    (RUN_DIR / "summary.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    lines = [
        f"# Stage 5 Autopilot Follow-Up - {RUN_ID}",
        "",
        f"- Autopilot summary: `{payload['metadata']['autopilot_summary']}`",
        f"- Follow-up limit: `{payload['metadata']['limit']}`",
        f"- Evaluation follow-ups ran: `{payload['decision']['run_eval_followups']}`",
        f"- Decision reason: `{payload['decision']['reason']}`",
        f"- Baseline curriculum branch ran: `{payload['decision']['ran_baseline_curriculum']}`",
        "",
        "## Benchmark",
        "",
        f"- Recovered vs base delta: `{compare_recovered_to_base(payload.get('recovered_benchmark'))}`",
        "",
        "## TTA Sweep",
        "",
        f"- Rows: `{len((payload.get('tta_sweep') or {}).get('rows', []))}`",
        f"- Deltas: `{(payload.get('tta_sweep') or {}).get('deltas')}`",
        "",
        "## Next Read",
        "",
        "Use this report to decide whether to increase the ARC limit, tune the curriculum, or defer particle/SVGD.",
    ]
    (RUN_DIR / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print((RUN_DIR / "summary.md").read_text(encoding="utf-8"))


def main() -> int:
    if any(arg in {"-h", "--help"} for arg in sys.argv[1:]):
        print("Run after Stage 5 curriculum-particle autopilot to launch safe benchmark/TTA follow-ups.")
        return 0

    summary_path = resolve_autopilot_summary()
    autopilot = read_json(summary_path)
    run_eval, reason = should_run_eval_followups(autopilot)
    curriculum_summary = curriculum_summary_path(autopilot)

    recovered_benchmark = None
    tta_sweep = None
    baseline_curriculum = None

    if run_eval and curriculum_summary is not None:
        if RUN_RECOVERED_BENCHMARK:
            recovered_benchmark = run_recovered_benchmark(curriculum_summary)
        if RUN_TTA_SWEEP:
            tta_sweep = run_tta_sweep(curriculum_summary)
    elif ALLOW_BASELINE_CURRICULUM:
        baseline_curriculum = run_baseline_curriculum()

    payload = {
        "run_id": RUN_ID,
        "metadata": {
            "autopilot_summary": path_for_cli(summary_path),
            "limit": LIMIT,
            "curriculum_summary": path_for_cli(curriculum_summary) if curriculum_summary else None,
            "run_recovered_benchmark": RUN_RECOVERED_BENCHMARK,
            "run_tta_sweep": RUN_TTA_SWEEP,
            "allow_baseline_curriculum": ALLOW_BASELINE_CURRICULUM,
        },
        "decision": {
            "run_eval_followups": run_eval,
            "reason": reason,
            "ran_baseline_curriculum": baseline_curriculum is not None,
        },
        "autopilot_compact": autopilot.get("compact"),
        "recovered_benchmark": recovered_benchmark,
        "tta_sweep": tta_sweep,
        "baseline_curriculum": baseline_curriculum,
    }
    write_report(payload)
    backup_to_drive()
    if PUSH_RESULTS:
        git_commit_results()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
