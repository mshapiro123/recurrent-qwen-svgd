"""Run the next balanced Stage 5 competence-recovery gate.

This wrapper is intentionally conservative. It first runs the lightweight
response-distillation gate from the current balanced checkpoint. If that gate
does not show proxy lift, it automatically runs the ARC-train mixed supervision
gate. This keeps a Colab A100 moving while preserving the decision order from
the Stage 5 recovery plan.
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

from colab.colab_auth import ensure_hf_token_from_colab  # noqa: E402
from colab.run_stage5_recovered_phase1_arc_gate import path_for_cli  # noqa: E402


RUN_ID = os.environ.get("STAGE5_BALANCED_RECOVERY_RUN_ID") or time.strftime(
    "stage5_balanced_recovery_autopilot_%Y%m%d_%H%M%S"
)
RUN_DIR = ROOT / "outputs" / "stage5" / RUN_ID

SOURCE_SUMMARY = Path(
    os.environ.get(
        "STAGE5_BALANCED_RECOVERY_SOURCE_SUMMARY",
        "outputs/stage5/stage5_balanced_mcq_current/summary.json",
    )
)
if not SOURCE_SUMMARY.is_absolute():
    SOURCE_SUMMARY = ROOT / SOURCE_SUMMARY

PUSH_RESULTS = os.environ.get("STAGE5_BALANCED_RECOVERY_PUSH", "1").strip().lower() in {
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


def child_env(prefix: str, run_id: str, source_summary: Path) -> dict[str, str]:
    env = os.environ.copy()
    env[f"{prefix}_RUN_ID"] = run_id
    env[f"{prefix}_SOURCE_SUMMARY"] = path_for_cli(source_summary)
    env[f"{prefix}_PUSH"] = "0"
    return env


def child_summary_path(run_id: str) -> Path:
    return ROOT / "outputs" / "stage5" / run_id / "summary.json"


def should_run_arc_mix(distill_payload: dict[str, Any]) -> bool:
    return distill_payload.get("status") not in {"proxy_lift", "proxy_matches_base"}


def stageable_result_paths(payload: dict[str, Any] | None) -> list[Path]:
    """Return run directories referenced by a child gate payload.

    The response-distillation gate can itself launch nested recovery-ladder
    runs. Those checkpoints are not inside the distill gate directory, so the
    autopilot has to carry them along when it is responsible for committing the
    result bundle.
    """

    if not payload:
        return []
    paths: list[Path] = []
    run_id = payload.get("run_id")
    if isinstance(run_id, str) and run_id:
        paths.append(ROOT / "outputs" / "stage5" / run_id)
    for row in payload.get("arms") or []:
        child_run_id = row.get("child_run_id")
        if isinstance(child_run_id, str) and child_run_id:
            paths.append(ROOT / "outputs" / "stage5" / child_run_id)
        checkpoint = row.get("checkpoint") or (row.get("best_checkpoint") or {}).get("checkpoint")
        if checkpoint:
            checkpoint_path = Path(str(checkpoint))
            if not checkpoint_path.is_absolute():
                checkpoint_path = ROOT / checkpoint_path
            try:
                stage5_idx = checkpoint_path.parts.index("stage5")
            except ValueError:
                continue
            if stage5_idx + 1 < len(checkpoint_path.parts):
                paths.append(Path(*checkpoint_path.parts[: stage5_idx + 2]))
    unique: list[Path] = []
    seen: set[str] = set()
    for path in paths:
        key = str(path)
        if key not in seen:
            seen.add(key)
            unique.append(path)
    return unique


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
        print("No balanced recovery autopilot outputs changed.")
        return
    run(["git", "commit", "-m", f"Record balanced Stage 5 recovery autopilot {RUN_ID} [skip ci]"])
    run(["git", "push", "origin", "main"], check=False)


def write_report(payload: dict[str, Any]) -> None:
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    (RUN_DIR / "summary.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    lines = [
        f"# Stage 5 Balanced Recovery Autopilot - {RUN_ID}",
        "",
        f"- Status: `{payload['status']}`",
        f"- Source summary: `{payload['source_summary']}`",
        f"- Distill summary: `{payload['distill_summary']}`",
        f"- ARC-mix summary: `{payload.get('arc_mix_summary') or 'not_run'}`",
        f"- Next step: {payload['next_step']}",
        "",
        "## Child Gate Status",
        "",
        f"- Distill: `{payload['distill_status']}`",
        f"- ARC-mix: `{payload.get('arc_mix_status') or 'not_run'}`",
    ]
    (RUN_DIR / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print((RUN_DIR / "summary.md").read_text(encoding="utf-8"))


def build_summary(
    *,
    distill_run_id: str,
    distill_payload: dict[str, Any],
    arc_mix_run_id: str | None,
    arc_mix_payload: dict[str, Any] | None,
) -> dict[str, Any]:
    if distill_payload.get("passed"):
        status = "distill_gate_passed"
        next_step = "Run the full balanced MCQ assessment on the best distillation checkpoint."
    elif arc_mix_payload and arc_mix_payload.get("passed"):
        status = "arc_mix_gate_passed"
        next_step = "Run the full balanced MCQ assessment on the best ARC-mix checkpoint."
    elif arc_mix_payload:
        status = "no_recovery_gate_lift"
        next_step = (
            "Do not continue these small recovery arms. Inspect failure cases or change the "
            "competence-recovery data recipe before returning to particles."
        )
    else:
        status = "distill_gate_failed"
        next_step = "Inspect the distillation gate failure before launching ARC-mix."

    return {
        "run_id": RUN_ID,
        "kind": "stage5_balanced_recovery_autopilot",
        "source_summary": path_for_cli(SOURCE_SUMMARY),
        "status": status,
        "next_step": next_step,
        "distill_run_id": distill_run_id,
        "distill_summary": f"outputs/stage5/{distill_run_id}/summary.json",
        "distill_status": distill_payload.get("status"),
        "distill_passed": distill_payload.get("passed"),
        "arc_mix_run_id": arc_mix_run_id,
        "arc_mix_summary": f"outputs/stage5/{arc_mix_run_id}/summary.json" if arc_mix_run_id else None,
        "arc_mix_status": arc_mix_payload.get("status") if arc_mix_payload else None,
        "arc_mix_passed": arc_mix_payload.get("passed") if arc_mix_payload else None,
        "distill": distill_payload,
        "arc_mix": arc_mix_payload,
    }


def main() -> int:
    ensure_hf_token_from_colab()
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    if not SOURCE_SUMMARY.exists():
        raise FileNotFoundError(f"Missing source summary: {SOURCE_SUMMARY}")

    distill_run_id = f"{RUN_ID}_distill"
    distill_summary = child_summary_path(distill_run_id)
    if distill_summary.exists():
        print(f"reusing_distill_summary={path_for_cli(distill_summary)}")
    else:
        distill_env = child_env("STAGE5_BALANCED_DISTILL", distill_run_id, SOURCE_SUMMARY)
        run(
            [sys.executable, "colab/run_stage5_balanced_distill_gate.py"],
            env=distill_env,
            log_name="distill_gate.log",
        )
    distill_payload = read_json(distill_summary)

    arc_mix_run_id: str | None = None
    arc_mix_payload: dict[str, Any] | None = None
    if should_run_arc_mix(distill_payload):
        arc_mix_run_id = f"{RUN_ID}_arc_mix"
        arc_mix_summary = child_summary_path(arc_mix_run_id)
        if arc_mix_summary.exists():
            print(f"reusing_arc_mix_summary={path_for_cli(arc_mix_summary)}")
        else:
            arc_mix_env = child_env("STAGE5_ARC_MIX", arc_mix_run_id, SOURCE_SUMMARY)
            run(
                [sys.executable, "colab/run_stage5_balanced_arc_mix_gate.py"],
                env=arc_mix_env,
                log_name="arc_mix_gate.log",
            )
        arc_mix_payload = read_json(arc_mix_summary)

    payload = build_summary(
        distill_run_id=distill_run_id,
        distill_payload=distill_payload,
        arc_mix_run_id=arc_mix_run_id,
        arc_mix_payload=arc_mix_payload,
    )
    write_report(payload)
    result_paths = [RUN_DIR]
    result_paths.extend(stageable_result_paths(distill_payload))
    result_paths.extend(stageable_result_paths(arc_mix_payload))
    backup_to_drive(result_paths)
    commit_results(result_paths)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
