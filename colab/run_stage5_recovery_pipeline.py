"""One-command Stage 5 balanced recovery pipeline for Colab.

The pipeline is deliberately thin: it runs the existing preflight, the
balanced recovery autopilot, and, if the autopilot passes a gate, the full
balanced ARC-Easy + ARC-Challenge assessment. It writes one parent summary so
the notebook can be rerun with fixed run ids and resume from completed child
summaries.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from colab.run_stage5_recovered_phase1_arc_gate import path_for_cli  # noqa: E402


RUN_ID = os.environ.get("STAGE5_RECOVERY_PIPELINE_RUN_ID") or time.strftime(
    "stage5_recovery_pipeline_%Y%m%d_%H%M%S"
)
RUN_DIR = ROOT / "outputs" / "stage5" / RUN_ID
AUTOPILOT_RUN_ID = os.environ.get(
    "STAGE5_BALANCED_RECOVERY_RUN_ID",
    "stage5_balanced_recovery_autopilot_current",
)
FULL_ASSESS_RUN_ID = os.environ.get(
    "STAGE5_RECOVERY_FULL_ASSESS_RUN_ID",
    "stage5_recovery_full_assessment_current",
)
PUSH_RESULTS = os.environ.get("STAGE5_RECOVERY_PIPELINE_PUSH", "1").strip().lower() in {
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


def autopilot_summary_path(run_id: str = AUTOPILOT_RUN_ID) -> Path:
    return ROOT / "outputs" / "stage5" / run_id / "summary.json"


def full_assessment_summary_path(run_id: str = FULL_ASSESS_RUN_ID) -> Path:
    return ROOT / "outputs" / "stage5" / run_id / "summary.json"


def autopilot_passed(payload: dict[str, Any]) -> bool:
    return payload.get("status") in {"distill_gate_passed", "arc_mix_gate_passed"}


def child_env() -> dict[str, str]:
    env = os.environ.copy()
    env.setdefault("STAGE5_BALANCED_RECOVERY_RUN_ID", AUTOPILOT_RUN_ID)
    env.setdefault("STAGE5_RECOVERY_FULL_ASSESS_RUN_ID", FULL_ASSESS_RUN_ID)
    env.setdefault(
        "STAGE5_RECOVERY_FULL_ASSESS_SOURCE_SUMMARY",
        f"outputs/stage5/{AUTOPILOT_RUN_ID}/summary.json",
    )
    env.setdefault("STAGE5_BALANCED_RECOVERY_PUSH", "1")
    env.setdefault("STAGE5_RECOVERY_FULL_ASSESS_PUSH", "1")
    env.setdefault("STAGE5_BALANCED_DISTILL_ARMS", "response_w005_lr3e6")
    env.setdefault("STAGE5_ARC_MIX_ARMS", "arc_mix_nodistill_lr3e6")
    return env


def commit_results() -> None:
    if not PUSH_RESULTS:
        return
    run(["git", "add", "-f", path_for_cli(RUN_DIR)], check=False)
    if run(["git", "diff", "--cached", "--quiet"], check=False).returncode == 0:
        print("No recovery pipeline outputs changed.")
        return
    run(["git", "commit", "-m", f"Record Stage 5 recovery pipeline {RUN_ID}"])
    run(["git", "push", "origin", "main"], check=False)


def write_report(payload: dict[str, Any]) -> None:
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    (RUN_DIR / "summary.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    lines = [
        f"# Stage 5 Recovery Pipeline - {RUN_ID}",
        "",
        f"- Status: `{payload['status']}`",
        f"- Autopilot summary: `{payload['autopilot_summary']}`",
        f"- Full assessment summary: `{payload.get('full_assessment_summary') or 'not_run'}`",
        f"- Next step: {payload['next_step']}",
    ]
    (RUN_DIR / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print((RUN_DIR / "summary.md").read_text(encoding="utf-8"))


def build_summary(
    *,
    autopilot_payload: dict[str, Any] | None,
    full_payload: dict[str, Any] | None,
) -> dict[str, Any]:
    if autopilot_payload is None:
        status = "autopilot_missing"
        next_step = "Inspect preflight/autopilot logs; no recovery gate summary was produced."
    elif not autopilot_passed(autopilot_payload):
        status = "recovery_gate_not_passed"
        next_step = autopilot_payload.get("next_step") or "Revise recovery recipe before full assessment."
    elif full_payload is None:
        status = "full_assessment_missing"
        next_step = "Rerun this pipeline with the same run ids to resume full balanced assessment."
    else:
        status = f"full_assessment_{full_payload.get('status', 'unknown')}"
        next_step = full_payload.get("next_step") or "Review full assessment summary."

    return {
        "run_id": RUN_ID,
        "kind": "stage5_recovery_pipeline",
        "status": status,
        "next_step": next_step,
        "autopilot_run_id": AUTOPILOT_RUN_ID,
        "autopilot_summary": path_for_cli(autopilot_summary_path()),
        "autopilot_status": autopilot_payload.get("status") if autopilot_payload else None,
        "full_assessment_run_id": FULL_ASSESS_RUN_ID,
        "full_assessment_summary": path_for_cli(full_assessment_summary_path()) if full_payload else None,
        "full_assessment_status": full_payload.get("status") if full_payload else None,
        "autopilot": autopilot_payload,
        "full_assessment": full_payload,
    }


def main() -> int:
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    env = child_env()

    run([sys.executable, "colab/check_stage5_colab_preflight.py"], env=env, log_name="preflight.log")

    autopilot_summary = autopilot_summary_path(env["STAGE5_BALANCED_RECOVERY_RUN_ID"])
    if autopilot_summary.exists():
        print(f"reusing_autopilot_summary={path_for_cli(autopilot_summary)}")
    else:
        run([sys.executable, "colab/run_stage5_balanced_recovery_autopilot.py"], env=env, log_name="autopilot.log")
    autopilot_payload = read_json(autopilot_summary) if autopilot_summary.exists() else None

    full_payload: dict[str, Any] | None = None
    if autopilot_payload and autopilot_passed(autopilot_payload):
        full_summary = full_assessment_summary_path(env["STAGE5_RECOVERY_FULL_ASSESS_RUN_ID"])
        if full_summary.exists():
            print(f"reusing_full_assessment_summary={path_for_cli(full_summary)}")
        else:
            run([sys.executable, "colab/run_stage5_recovery_full_assessment.py"], env=env, log_name="full_assessment.log")
        full_payload = read_json(full_summary) if full_summary.exists() else None

    payload = build_summary(autopilot_payload=autopilot_payload, full_payload=full_payload)
    write_report(payload)
    commit_results()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
