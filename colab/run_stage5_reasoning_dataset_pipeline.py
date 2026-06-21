"""Run dataset audit, then optionally execute the planner-selected next action.

This is the single-cell Colab entrypoint for the "which HF traces should feed
the recurrent model?" branch. It keeps the expensive work evidence-led:

1. audit the registry-selected reasoning datasets;
2. write a dataset-audit summary;
3. ask the Stage 5 planner what to do from that summary;
4. optionally execute exactly that allowlisted action.
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
RUN_ID = os.environ.get("STAGE5_REASONING_DATASET_PIPELINE_RUN_ID") or time.strftime(
    "stage5_reasoning_dataset_pipeline_%Y%m%d_%H%M%S"
)
RUN_DIR = ROOT / "outputs" / "stage5" / RUN_ID
AUDIT_RUN_ID = os.environ.get("STAGE5_REASONING_DATASET_PIPELINE_AUDIT_RUN_ID", f"{RUN_ID}_audit")
NEXT_ACTION_RUN_ID = os.environ.get("STAGE5_REASONING_DATASET_PIPELINE_NEXT_ACTION_RUN_ID", f"{RUN_ID}_next")
EXECUTE_NEXT = os.environ.get("STAGE5_REASONING_DATASET_PIPELINE_EXECUTE_NEXT", "1").strip().lower() in {
    "1",
    "true",
    "yes",
    "y",
}
MAX_ACTIONS = int(os.environ.get("STAGE5_REASONING_DATASET_PIPELINE_MAX_ACTIONS", "1"))
PUSH_RESULTS = os.environ.get("STAGE5_REASONING_DATASET_PIPELINE_PUSH", "1").strip().lower() in {
    "1",
    "true",
    "yes",
    "y",
}


def path_for_cli(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT)).replace("\\", "/")
    except ValueError:
        return str(path).replace("\\", "/")


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


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


def audit_summary_path() -> Path:
    return ROOT / "outputs" / "stage5" / AUDIT_RUN_ID / "summary.json"


def next_action_summary_path() -> Path:
    return ROOT / "outputs" / "stage5" / NEXT_ACTION_RUN_ID / "summary.json"


def audit_env() -> dict[str, str]:
    env = os.environ.copy()
    env.setdefault("STAGE5_DATASET_AUDIT_RUN_ID", AUDIT_RUN_ID)
    env.setdefault(
        "STAGE5_DATASET_AUDIT_KEYS",
        "opus47_sft,opus47_raw,fable5_pi_agent,fable5_flat,jackrong_opus47_trace_inversion",
    )
    env.setdefault("STAGE5_DATASET_AUDIT_LIMIT", "1000")
    env.setdefault("STAGE5_DATASET_AUDIT_PUSH", "1" if PUSH_RESULTS else "0")
    return env


def next_action_env() -> dict[str, str]:
    env = os.environ.copy()
    env.setdefault("STAGE5_ARC_AGI_NEXT_ACTION_RUN_ID", NEXT_ACTION_RUN_ID)
    env["STAGE5_ARC_AGI_NEXT_ACTION_SOURCE_SUMMARY"] = path_for_cli(audit_summary_path())
    env.setdefault("STAGE5_ARC_AGI_NEXT_ACTION_EXECUTE", "1" if EXECUTE_NEXT else "0")
    env.setdefault("STAGE5_ARC_AGI_NEXT_ACTION_MAX_ACTIONS", str(MAX_ACTIONS))
    env.setdefault("STAGE5_ARC_AGI_NEXT_ACTION_ALLOW_REPEAT", "0")
    return env


def commit_results() -> None:
    if not PUSH_RESULTS:
        return
    run(["git", "add", "-f", path_for_cli(RUN_DIR)], check=False)
    audit_dir = audit_summary_path().parent
    next_dir = next_action_summary_path().parent
    if audit_dir.exists():
        run(["git", "add", "-f", path_for_cli(audit_dir)], check=False)
    if next_dir.exists():
        run(["git", "add", "-f", path_for_cli(next_dir)], check=False)
    if run(["git", "diff", "--cached", "--quiet"], check=False).returncode == 0:
        print("No reasoning dataset pipeline outputs changed.")
        return
    run(["git", "commit", "-m", f"Record Stage 5 reasoning dataset pipeline {RUN_ID}"])
    run(["git", "push", "origin", "main"], check=False)


def write_report(payload: dict[str, Any]) -> None:
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    (RUN_DIR / "summary.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    lines = [
        f"# Stage 5 Reasoning Dataset Pipeline - {RUN_ID}",
        "",
        f"- Status: `{payload['status']}`",
        f"- Audit summary: `{payload.get('audit_summary')}`",
        f"- Next-action summary: `{payload.get('next_action_summary') or 'not_run'}`",
        f"- Next executed: `{payload['next_executed']}`",
        f"- Next step: {payload['next_step']}",
    ]
    (RUN_DIR / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print((RUN_DIR / "summary.md").read_text(encoding="utf-8"))


def build_summary(
    *,
    audit_payload: dict[str, Any] | None,
    next_payload: dict[str, Any] | None,
    error: str | None = None,
) -> dict[str, Any]:
    if error:
        status = "pipeline_failed"
        next_step = "Inspect pipeline logs and rerun with the same run IDs."
    elif audit_payload is None:
        status = "audit_missing"
        next_step = "Rerun the dataset audit."
    elif next_payload is None:
        status = "audit_complete"
        next_step = audit_payload.get("next_step") or "Review dataset audit summary."
    else:
        status = "next_action_complete" if EXECUTE_NEXT else "next_action_planned"
        steps = next_payload.get("steps")
        selected = steps[0].get("action", {}) if isinstance(steps, list) and steps else {}
        next_step = selected.get("name") or audit_payload.get("next_step") or "Review next-action summary."
    return {
        "run_id": RUN_ID,
        "kind": "stage5_reasoning_dataset_pipeline",
        "status": status,
        "error": error,
        "audit_run_id": AUDIT_RUN_ID,
        "audit_summary": path_for_cli(audit_summary_path()) if audit_payload else None,
        "audit_status": audit_payload.get("status") if audit_payload else None,
        "next_action_run_id": NEXT_ACTION_RUN_ID,
        "next_action_summary": path_for_cli(next_action_summary_path()) if next_payload else None,
        "next_executed": EXECUTE_NEXT and next_payload is not None,
        "next_step": next_step,
        "audit": audit_payload,
        "next_action": next_payload,
    }


def main() -> int:
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    audit_payload: dict[str, Any] | None = None
    next_payload: dict[str, Any] | None = None
    try:
        run([sys.executable, "colab/run_stage5_reasoning_dataset_audit.py"], env=audit_env(), log_name="audit.log")
        audit_payload = read_json(audit_summary_path())
        run([sys.executable, "colab/run_stage5_next_action.py"], env=next_action_env(), log_name="next_action.log")
        if next_action_summary_path().exists():
            next_payload = read_json(next_action_summary_path())
        payload = build_summary(audit_payload=audit_payload, next_payload=next_payload)
        write_report(payload)
        commit_results()
        return 0
    except Exception as exc:  # noqa: BLE001 - Colab pipeline should leave a report.
        payload = build_summary(audit_payload=audit_payload, next_payload=next_payload, error=str(exc))
        write_report(payload)
        commit_results()
        raise


if __name__ == "__main__":
    raise SystemExit(main())
