"""Build capability-ladder strong-trace jobs from the latest probe summary.

This is a no-GPU Colab/CPU runner. It consumes a
``stage5_capability_ladder_mcq_probe`` summary, restores the private scored rows
from Drive if needed, builds provider-neutral trace-generation jobs, backs them
up to Drive, and optionally commits safe run metadata to GitHub.
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


RUN_ID = os.environ.get("STAGE5_CAPABILITY_LADDER_TRACE_RUN_ID") or time.strftime(
    "stage5_capability_ladder_trace_jobs_%Y%m%d_%H%M%S"
)
RUN_DIR = ROOT / "outputs" / "stage5" / RUN_ID
SOURCE_SUMMARY = os.environ.get("STAGE5_CAPABILITY_LADDER_TRACE_SOURCE_SUMMARY", "").strip()
MODELS = os.environ.get("STAGE5_CAPABILITY_LADDER_TRACE_MODELS", "opus-strong,glm-strong")
MAX_ROWS = os.environ.get("STAGE5_CAPABILITY_LADDER_TRACE_MAX_ROWS", "").strip()
MAX_PER_TIER = os.environ.get("STAGE5_CAPABILITY_LADDER_TRACE_MAX_PER_TIER", "").strip()
PUSH_RESULTS = os.environ.get("STAGE5_CAPABILITY_LADDER_TRACE_PUSH", "1").strip().lower() in {
    "1",
    "true",
    "yes",
    "y",
}
BACKUP_DRIVE = os.environ.get("STAGE5_CAPABILITY_LADDER_TRACE_BACKUP_DRIVE", "1").strip().lower() in {
    "1",
    "true",
    "yes",
    "y",
}
REFUSE_GPU_RUNTIME = os.environ.get("STAGE5_CAPABILITY_LADDER_TRACE_REFUSE_GPU", "1").strip().lower() in {
    "1",
    "true",
    "yes",
    "y",
}
ALLOW_GPU_RUNTIME_FOR_CPU_WORK = os.environ.get(
    "STAGE5_CAPABILITY_LADDER_TRACE_ALLOW_GPU",
    "0",
).strip().lower() in {"1", "true", "yes", "y"}


def path_for_cli(path: Path) -> str:
    try:
        return path.relative_to(ROOT).as_posix()
    except ValueError:
        return str(path)


def read_json(path: str | Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} is not a JSON object.")
    return payload


def write_json(path: str | Path, payload: dict[str, Any]) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def run(cmd: list[str], *, check: bool = True, log_name: str | None = None) -> subprocess.CompletedProcess[str]:
    printable = " ".join(map(str, cmd))
    print("$", printable, flush=True)
    proc = subprocess.run(
        cmd,
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    print(proc.stdout, flush=True)
    if log_name:
        RUN_DIR.mkdir(parents=True, exist_ok=True)
        (RUN_DIR / log_name).write_text(proc.stdout, encoding="utf-8")
    if check and proc.returncode:
        raise RuntimeError(f"command failed: {printable}")
    return proc


def attached_gpu_names() -> list[str]:
    if shutil.which("nvidia-smi") is None:
        return []
    try:
        proc = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            check=False,
        )
    except (FileNotFoundError, OSError):
        return []
    if proc.returncode:
        return []
    return [line.strip() for line in proc.stdout.splitlines() if line.strip()]


def refuse_gpu_runtime_for_cpu_work() -> None:
    gpus = attached_gpu_names()
    if REFUSE_GPU_RUNTIME and gpus and not ALLOW_GPU_RUNTIME_FOR_CPU_WORK:
        raise RuntimeError(
            "Refusing to build capability-ladder trace jobs on an attached GPU runtime: "
            + "; ".join(gpus)
            + ". Switch to CPU runtime, or set STAGE5_CAPABILITY_LADDER_TRACE_ALLOW_GPU=1 deliberately."
        )
    if gpus:
        print("GPU attached but CPU-work override is enabled:", gpus, flush=True)
    else:
        print("No GPU attached; good for capability-ladder trace-job building.", flush=True)


def current_source_summary_file() -> Path:
    return ROOT / "config" / "stage5_current_source_summary.txt"


def resolve_source_summary() -> Path:
    raw = SOURCE_SUMMARY
    if not raw:
        pointer = current_source_summary_file()
        if pointer.exists():
            raw = pointer.read_text(encoding="utf-8").strip()
    if not raw:
        raise FileNotFoundError(
            "Missing source summary. Set STAGE5_CAPABILITY_LADDER_TRACE_SOURCE_SUMMARY "
            "or update config/stage5_current_source_summary.txt."
        )
    path = Path(raw)
    if not path.is_absolute():
        path = ROOT / path
    if not path.exists():
        raise FileNotFoundError(path)
    return path


def scored_path_from_summary(source_summary: Path) -> Path:
    payload = read_json(source_summary)
    artifacts = payload.get("artifacts") if isinstance(payload.get("artifacts"), dict) else {}
    raw = str(artifacts.get("scored_capability_rows") or "").strip()
    if not raw:
        raise ValueError(f"{source_summary} does not contain artifacts.scored_capability_rows.")
    path = Path(raw)
    if not path.is_absolute():
        path = ROOT / path
    return path


def drive_search_roots() -> list[Path]:
    drive = Path("/content/drive/MyDrive")
    if not drive.exists():
        return []
    return [
        drive / "recurrent-qwen-svgd" / "stage5_capability_ladder",
        drive / "recurrent-qwen-svgd",
        drive / "recurrent-qwen-svgd-artifacts",
        drive / "recurrent-qwen-svgd-fresh",
    ]


def restore_scored_rows_if_needed(source_summary: Path) -> dict[str, Any]:
    expected = scored_path_from_summary(source_summary)
    if expected.exists():
        return {"restored": False, "path": path_for_cli(expected), "source": "local"}

    run_id = str(read_json(source_summary).get("run_id") or source_summary.parent.name)
    candidates: list[Path] = []
    for root in drive_search_roots():
        if root.exists():
            candidates.extend(root.rglob("scored_capability_rows.jsonl"))
    candidates = sorted(
        candidates,
        key=lambda path: (run_id not in str(path), len(str(path))),
    )
    if not candidates:
        raise FileNotFoundError(
            f"Missing scored rows at {expected} and no Drive backup scored_capability_rows.jsonl was found."
        )

    source = candidates[0]
    expected.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, expected)
    return {
        "restored": True,
        "path": path_for_cli(expected),
        "source": str(source),
        "candidates": len(candidates),
    }


def backup_to_drive(paths: list[Path]) -> dict[str, Any]:
    if not BACKUP_DRIVE:
        return {"enabled": False}
    drive_root = Path("/content/drive/MyDrive")
    if not drive_root.exists():
        return {"enabled": True, "available": False}
    dest_root = drive_root / "recurrent-qwen-svgd" / "stage5_capability_ladder_trace_jobs" / RUN_ID
    copied: list[str] = []
    for path in paths:
        if not path.exists():
            continue
        dest = dest_root / path.name
        if dest.exists():
            if dest.is_dir():
                shutil.rmtree(dest)
            else:
                dest.unlink()
        if path.is_dir():
            shutil.copytree(path, dest)
        else:
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, dest)
        copied.append(str(dest))
    return {"enabled": True, "available": True, "dest_root": str(dest_root), "copied": copied}


def update_current_source_summary(summary_path: Path) -> Path:
    pointer = current_source_summary_file()
    pointer.parent.mkdir(parents=True, exist_ok=True)
    pointer.write_text(path_for_cli(summary_path) + "\n", encoding="utf-8")
    return pointer


def safe_commit(summary_path: Path) -> None:
    if not PUSH_RESULTS:
        return
    pointer = update_current_source_summary(summary_path)
    run(["git", "add", path_for_cli(RUN_DIR), path_for_cli(pointer)], check=False, log_name="git_add.log")
    diff = run(["git", "diff", "--cached", "--quiet"], check=False)
    if diff.returncode == 0:
        print("No safe trace-job result changes to commit.", flush=True)
        return
    run(["git", "commit", "-m", "Record capability-ladder trace jobs"], check=True, log_name="git_commit.log")
    run(["git", "push", "origin", "main"], check=True, log_name="git_push.log")


def build_jobs(source_summary: Path) -> tuple[Path, Path, dict[str, Any]]:
    jobs_jsonl = RUN_DIR / "capability_ladder_trace_jobs.jsonl"
    report_json = RUN_DIR / "capability_ladder_trace_jobs_report.json"
    cmd = [
        sys.executable,
        "training/build_capability_ladder_trace_jobs.py",
        "--summary_json",
        path_for_cli(source_summary),
        "--models",
        MODELS,
        "--output_jsonl",
        path_for_cli(jobs_jsonl),
        "--report_json",
        path_for_cli(report_json),
    ]
    if MAX_ROWS:
        cmd.extend(["--max_rows", MAX_ROWS])
    if MAX_PER_TIER:
        cmd.extend(["--max_per_tier", MAX_PER_TIER])
    run(cmd, log_name="build_capability_ladder_trace_jobs.log")
    return jobs_jsonl, report_json, read_json(report_json)


def write_summary(
    *,
    source_summary: Path,
    restore_report: dict[str, Any],
    jobs_jsonl: Path,
    report_json: Path,
    trace_report: dict[str, Any],
    drive_backup: dict[str, Any],
) -> Path:
    payload = {
        "run_id": RUN_ID,
        "kind": "stage5_capability_ladder_trace_jobs",
        "status": "ready" if int(trace_report.get("jobs") or 0) > 0 else "empty",
        "source_summary": path_for_cli(source_summary),
        "models": [item.strip() for item in MODELS.split(",") if item.strip()],
        "restore": restore_report,
        "trace_jobs": {
            "jobs": int(trace_report.get("jobs") or 0),
            "selected_rows": int(trace_report.get("selected_rows") or 0),
            "tier_counts": trace_report.get("tier_counts", {}),
            "by_target_loop": trace_report.get("by_target_loop", {}),
            "by_model": trace_report.get("by_model", {}),
        },
        "artifacts": {
            "jobs_jsonl": path_for_cli(jobs_jsonl),
            "report_json": path_for_cli(report_json),
            "run_dir": path_for_cli(RUN_DIR),
        },
        "drive_backup": drive_backup,
        "next_action": (
            "Run training/run_curriculum_job_responses.py with a provider/model map to generate trace responses, "
            "then training/collect_capability_ladder_trace_outputs.py to build traced scored rows."
        ),
    }
    summary = RUN_DIR / "summary.json"
    write_json(summary, payload)
    (RUN_DIR / "summary.md").write_text(markdown_summary(payload), encoding="utf-8")
    return summary


def markdown_summary(payload: dict[str, Any]) -> str:
    jobs = payload["trace_jobs"]
    return "\n".join(
        [
            f"# Capability-Ladder Trace Jobs: {payload['run_id']}",
            "",
            f"- Status: `{payload['status']}`",
            f"- Source summary: `{payload['source_summary']}`",
            f"- Jobs: `{jobs['jobs']}`",
            f"- Selected rows: `{jobs['selected_rows']}`",
            f"- Target loops: `{jobs['by_target_loop']}`",
            f"- Tiers: `{jobs['tier_counts']}`",
            "",
            "## Next Action",
            "",
            payload["next_action"],
            "",
        ]
    )


def main() -> int:
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    refuse_gpu_runtime_for_cpu_work()
    source_summary = resolve_source_summary()
    restore_report = restore_scored_rows_if_needed(source_summary)
    jobs_jsonl, report_json, trace_report = build_jobs(source_summary)
    drive_backup = backup_to_drive([RUN_DIR, jobs_jsonl, report_json])
    summary = write_summary(
        source_summary=source_summary,
        restore_report=restore_report,
        jobs_jsonl=jobs_jsonl,
        report_json=report_json,
        trace_report=trace_report,
        drive_backup=drive_backup,
    )
    print(f"summary_json={path_for_cli(summary)}", flush=True)
    print(f"status={read_json(summary)['status']}", flush=True)
    safe_commit(summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
