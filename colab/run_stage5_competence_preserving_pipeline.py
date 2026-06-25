"""Run the next Stage 5 competence-preserving mixed objective gate.

This is the follow-up to the latest full balanced ARC assessment. That
assessment found that the proxy-selected recurrent checkpoint still trails base
and exhibits answer-calibration drift. This runner therefore resumes from the
selected recurrent checkpoint and runs ARC-train mixed SFT with stronger
base-logit response distillation before deciding whether to spend time on a full
balanced assessment.
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

from colab.run_stage5_balanced_arc_mix_gate import checkpoint_run_id, selected_checkpoint  # noqa: E402
from colab.run_stage5_recovered_phase1_arc_gate import restore_checkpoint_if_needed  # noqa: E402

RUN_ID = os.environ.get("STAGE5_COMPETENCE_PIPELINE_RUN_ID") or time.strftime(
    "stage5_competence_preserving_%Y%m%d_%H%M%S"
)
RUN_DIR = ROOT / "outputs" / "stage5" / RUN_ID
DEFAULT_SOURCE_SUMMARY = "outputs/stage5/stage5_debiased_benchmark_assessment_20260625_121302/summary.json"
SOURCE_SUMMARY = Path(
    os.environ.get(
        "STAGE5_COMPETENCE_SOURCE_SUMMARY",
        DEFAULT_SOURCE_SUMMARY,
    )
)
if not SOURCE_SUMMARY.is_absolute():
    SOURCE_SUMMARY = ROOT / SOURCE_SUMMARY

ARC_MIX_RUN_ID = os.environ.get("STAGE5_COMPETENCE_ARC_MIX_RUN_ID", f"{RUN_ID}_arc_mix")
FULL_ASSESS_RUN_ID = os.environ.get("STAGE5_COMPETENCE_FULL_ASSESS_RUN_ID", f"{RUN_ID}_full_assessment")
PUSH_RESULTS = os.environ.get("STAGE5_COMPETENCE_PIPELINE_PUSH", "1").strip().lower() in {
    "1",
    "true",
    "yes",
    "y",
}
PREFLIGHT_CHECKPOINT_RESTORE = os.environ.get(
    "STAGE5_COMPETENCE_PREFLIGHT_CHECKPOINT_RESTORE",
    "1",
).strip().lower() in {"1", "true", "yes", "y"}


def path_for_cli(path: Path) -> str:
    try:
        return path.relative_to(ROOT).as_posix()
    except ValueError:
        return str(path)


def current_source_summary_file() -> Path:
    return ROOT / "config" / "stage5_current_source_summary.txt"


def update_current_source_summary(summary_path: Path) -> Path:
    pointer = current_source_summary_file()
    pointer.parent.mkdir(parents=True, exist_ok=True)
    pointer.write_text(path_for_cli(summary_path) + "\n", encoding="utf-8")
    return pointer


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


def arc_mix_summary_path() -> Path:
    return ROOT / "outputs" / "stage5" / ARC_MIX_RUN_ID / "summary.json"


def full_assessment_summary_path() -> Path:
    return ROOT / "outputs" / "stage5" / FULL_ASSESS_RUN_ID / "summary.json"


def arc_mix_passed(payload: dict[str, Any] | None) -> bool:
    return bool(payload and payload.get("status") in {"proxy_lift", "proxy_matches_base"})


def child_env() -> dict[str, str]:
    env = os.environ.copy()
    env.setdefault("STAGE5_ARC_MIX_RUN_ID", ARC_MIX_RUN_ID)
    env.setdefault("STAGE5_ARC_MIX_SOURCE_SUMMARY", path_for_cli(SOURCE_SUMMARY))
    env.setdefault(
        "STAGE5_ARC_MIX_ARMS",
        "arc_mix_response_w01_lr2e6,arc_mix_response_w02_lr2e6",
    )
    env.setdefault("STAGE5_ARC_MIX_ARC_CHALLENGE_REPEAT", "2")
    env.setdefault("STAGE5_ARC_MIX_ARC_EASY_REPEAT", "4")
    env.setdefault("STAGE5_ARC_MIX_ARC_EVAL_LIMIT", "128")
    env.setdefault("STAGE5_ARC_MIX_PUSH", "1")
    env.setdefault("STAGE5_RECOVERY_FULL_ASSESS_RUN_ID", FULL_ASSESS_RUN_ID)
    env.setdefault("STAGE5_RECOVERY_FULL_ASSESS_SOURCE_SUMMARY", f"outputs/stage5/{ARC_MIX_RUN_ID}/summary.json")
    env.setdefault("STAGE5_RECOVERY_FULL_ASSESS_PUSH", "1")
    return env


def preflight_checkpoint_restore(source_payload: dict[str, Any]) -> Path | None:
    if not PREFLIGHT_CHECKPOINT_RESTORE:
        print("checkpoint_restore_preflight=disabled", flush=True)
        return None
    checkpoint = selected_checkpoint(source_payload)
    restore_checkpoint_if_needed(checkpoint, run_id=checkpoint_run_id(checkpoint))
    print(f"checkpoint_restore_preflight=ok checkpoint={path_for_cli(checkpoint)}", flush=True)
    return checkpoint


def commit_results() -> None:
    if not PUSH_RESULTS:
        return
    run(["git", "add", "-f", path_for_cli(RUN_DIR)], check=False)
    pointer = current_source_summary_file()
    if pointer.exists():
        run(["git", "add", "-f", path_for_cli(pointer)], check=False)
    if run(["git", "diff", "--cached", "--quiet"], check=False).returncode == 0:
        print("No competence-preserving pipeline outputs changed.")
        return
    run(["git", "commit", "-m", f"Record Stage 5 competence-preserving pipeline {RUN_ID} [skip ci]"])
    pushed = run(["git", "push", "origin", "main"], check=False)
    if pushed.returncode == 0:
        return
    print("Initial competence pipeline push failed; attempting one autostash rebase and retry.", flush=True)
    run(["git", "pull", "--rebase", "--autostash", "origin", "main"])
    run(["git", "push", "origin", "main"])


def try_commit_results(*, context: str) -> None:
    """Commit text summaries when possible without hiding the actual run result."""

    try:
        commit_results()
    except Exception as exc:
        print(f"commit_results_failed context={context}: {exc}", flush=True)


def write_report(payload: dict[str, Any]) -> None:
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    summary_path = RUN_DIR / "summary.json"
    summary_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    update_current_source_summary(summary_path)
    lines = [
        f"# Stage 5 Competence-Preserving Pipeline - {RUN_ID}",
        "",
        f"- Status: `{payload['status']}`",
        f"- Source summary: `{payload['source_summary']}`",
        f"- ARC-mix summary: `{payload.get('arc_mix_summary') or 'not_run'}`",
        f"- Full assessment summary: `{payload.get('full_assessment_summary') or 'not_run'}`",
        f"- Next step: {payload['next_step']}",
    ]
    if payload.get("failure_diagnosis"):
        lines.insert(-1, f"- Failure diagnosis: `{payload['failure_diagnosis']}`")
    if payload.get("child_log_tail"):
        lines.extend(
            [
                "",
                "## Child Log Tail",
                "",
                "```text",
                str(payload["child_log_tail"]),
                "```",
            ]
        )
    (RUN_DIR / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print((RUN_DIR / "summary.md").read_text(encoding="utf-8"))


def build_summary(
    *,
    source_payload: dict[str, Any],
    arc_payload: dict[str, Any] | None,
    full_payload: dict[str, Any] | None,
) -> dict[str, Any]:
    if arc_payload is None:
        status = "arc_mix_missing"
        next_step = "Inspect ARC-mix logs; no competence-preserving gate summary was produced."
    elif not arc_mix_passed(arc_payload):
        status = "arc_mix_not_passed"
        next_step = arc_payload.get("next_step") or "Revise competence-preserving mix before full assessment."
    elif full_payload is None:
        status = "full_assessment_missing"
        next_step = "Rerun with the same run ids to resume full balanced assessment."
    else:
        status = f"full_assessment_{full_payload.get('status', 'unknown')}"
        next_step = full_payload.get("next_step") or "Review full assessment summary."
    return {
        "run_id": RUN_ID,
        "kind": "stage5_competence_preserving_pipeline",
        "source_summary": path_for_cli(SOURCE_SUMMARY),
        "source_status": source_payload.get("status"),
        "status": status,
        "next_step": next_step,
        "arc_mix_run_id": ARC_MIX_RUN_ID,
        "arc_mix_summary": path_for_cli(arc_mix_summary_path()) if arc_payload else None,
        "arc_mix_status": arc_payload.get("status") if arc_payload else None,
        "full_assessment_run_id": FULL_ASSESS_RUN_ID,
        "full_assessment_summary": path_for_cli(full_assessment_summary_path()) if full_payload else None,
        "full_assessment_status": full_payload.get("status") if full_payload else None,
        "arc_mix": arc_payload,
        "full_assessment": full_payload,
    }


def child_log_tail(stage: str, *, max_lines: int = 80) -> str:
    log_name = {
        "arc_mix": "arc_mix.log",
        "full_assessment": "full_assessment.log",
    }.get(stage)
    if not log_name:
        return ""
    path = RUN_DIR / log_name
    if not path.exists():
        return ""
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    return "\n".join(lines[-max_lines:])


def failure_diagnosis(stage: str, evidence: str) -> tuple[str | None, str | None]:
    if (
        stage in {"checkpoint_restore", "arc_mix", "full_assessment"}
        and "Missing recovered checkpoint" in evidence
        and "Could not restore it from Drive" in evidence
    ):
        return (
            "checkpoint_restore_or_drive_mount_failed",
            (
                "Mount Google Drive in the top-level Colab cell, keep the same run ids, "
                "and rerun so the child stage can restore the required checkpoint."
            ),
        )
    return None, None


def failure_summary(*, stage: str, error: str, source_payload: dict[str, Any] | None) -> dict[str, Any]:
    log_tail = child_log_tail(stage)
    diagnosis, diagnosed_next_step = failure_diagnosis(stage, "\n".join([error, log_tail]))
    return {
        "run_id": RUN_ID,
        "kind": "stage5_competence_preserving_pipeline",
        "source_summary": path_for_cli(SOURCE_SUMMARY),
        "source_status": source_payload.get("status") if source_payload else None,
        "status": "pipeline_failed",
        "failed_stage": stage,
        "failure_diagnosis": diagnosis,
        "error": error,
        "child_log_tail": log_tail,
        "next_step": diagnosed_next_step
        or f"Inspect {stage} logs under {path_for_cli(RUN_DIR)} and rerun with the same run ids.",
        "arc_mix_run_id": ARC_MIX_RUN_ID,
        "arc_mix_summary": path_for_cli(arc_mix_summary_path()),
        "arc_mix_status": None,
        "full_assessment_run_id": FULL_ASSESS_RUN_ID,
        "full_assessment_summary": None,
        "full_assessment_status": None,
        "arc_mix": None,
        "full_assessment": None,
    }


def main() -> int:
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    source_payload: dict[str, Any] | None = None
    current_stage = "source"
    try:
        if not SOURCE_SUMMARY.exists():
            raise FileNotFoundError(f"Missing source summary: {SOURCE_SUMMARY}")
        source_payload = read_json(SOURCE_SUMMARY)
        env = child_env()
        current_stage = "checkpoint_restore"
        preflight_checkpoint_restore(source_payload)

        arc_summary = arc_mix_summary_path()
        if arc_summary.exists():
            print(f"reusing_arc_mix_summary={path_for_cli(arc_summary)}")
        else:
            current_stage = "arc_mix"
            run([sys.executable, "colab/run_stage5_balanced_arc_mix_gate.py"], env=env, log_name="arc_mix.log")
        current_stage = "arc_mix"
        arc_payload = read_json(arc_summary) if arc_summary.exists() else None

        full_payload: dict[str, Any] | None = None
        if arc_mix_passed(arc_payload):
            full_summary = full_assessment_summary_path()
            if full_summary.exists():
                print(f"reusing_full_assessment_summary={path_for_cli(full_summary)}")
            else:
                current_stage = "full_assessment"
                run(
                    [sys.executable, "colab/run_stage5_recovery_full_assessment.py"],
                    env=env,
                    log_name="full_assessment.log",
                )
            current_stage = "full_assessment"
            full_payload = read_json(full_summary) if full_summary.exists() else None

        payload = build_summary(source_payload=source_payload, arc_payload=arc_payload, full_payload=full_payload)
        write_report(payload)
        try_commit_results(context="success")
        return 0
    except Exception as exc:
        payload = failure_summary(stage=current_stage, error=str(exc), source_payload=source_payload)
        write_report(payload)
        try_commit_results(context="failure")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
