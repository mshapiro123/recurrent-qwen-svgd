"""Run provider responses for capability-ladder trace jobs.

This CPU/network runner follows a ``stage5_capability_ladder_trace_jobs``
summary, restores job artifacts from Drive if needed, runs
``training/run_curriculum_job_responses.py`` with explicit provider opt-in,
backs raw responses up to Drive, and can commit raw responses for recovery in
the private project repo. It refuses visible GPU runtimes by default.
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


RUN_ID = os.environ.get("STAGE5_CAPABILITY_LADDER_TRACE_RESPONSE_RUN_ID") or time.strftime(
    "stage5_capability_ladder_trace_responses_%Y%m%d_%H%M%S"
)
RUN_DIR = ROOT / "outputs" / "stage5" / RUN_ID
SOURCE_SUMMARY = os.environ.get("STAGE5_CAPABILITY_LADDER_TRACE_RESPONSE_SOURCE_SUMMARY", "").strip()
RUN_PROVIDER = os.environ.get("STAGE5_CAPABILITY_LADDER_TRACE_RESPONSE_RUN_PROVIDER", "0").strip().lower() in {
    "1",
    "true",
    "yes",
    "y",
}
BACKEND = os.environ.get(
    "STAGE5_CAPABILITY_LADDER_TRACE_RESPONSE_BACKEND",
    "openai_compatible" if RUN_PROVIDER else "dry_run",
)
API_KEY_ENV = os.environ.get("STAGE5_CAPABILITY_LADDER_TRACE_RESPONSE_API_KEY_ENV", "OPENAI_API_KEY")
BASE_URL = os.environ.get("STAGE5_CAPABILITY_LADDER_TRACE_RESPONSE_BASE_URL", "https://api.openai.com/v1")
MODEL_OVERRIDE = os.environ.get("STAGE5_CAPABILITY_LADDER_TRACE_RESPONSE_MODEL_OVERRIDE", "").strip()
MODEL_MAP_JSON = os.environ.get("STAGE5_CAPABILITY_LADDER_TRACE_RESPONSE_MODEL_MAP_JSON", "").strip()
MODEL_MAP_JSON_INLINE = os.environ.get(
    "STAGE5_CAPABILITY_LADDER_TRACE_RESPONSE_MODEL_MAP_JSON_INLINE",
    "",
).strip()
COMMAND = os.environ.get("STAGE5_CAPABILITY_LADDER_TRACE_RESPONSE_COMMAND", "").strip()
LIMIT = os.environ.get("STAGE5_CAPABILITY_LADDER_TRACE_RESPONSE_LIMIT", "").strip()
MAX_TOKENS = os.environ.get("STAGE5_CAPABILITY_LADDER_TRACE_RESPONSE_MAX_TOKENS", "2048")
TEMPERATURE = os.environ.get("STAGE5_CAPABILITY_LADDER_TRACE_RESPONSE_TEMPERATURE", "0.2")
TIMEOUT_SEC = os.environ.get("STAGE5_CAPABILITY_LADDER_TRACE_RESPONSE_TIMEOUT_SEC", "180")
SLEEP_SEC = os.environ.get("STAGE5_CAPABILITY_LADDER_TRACE_RESPONSE_SLEEP_SEC", "0")
MAX_RETRIES = os.environ.get("STAGE5_CAPABILITY_LADDER_TRACE_RESPONSE_MAX_RETRIES", "2")
RETRY_SLEEP_SEC = os.environ.get("STAGE5_CAPABILITY_LADDER_TRACE_RESPONSE_RETRY_SLEEP_SEC", "3")
RETRY_BACKOFF = os.environ.get("STAGE5_CAPABILITY_LADDER_TRACE_RESPONSE_RETRY_BACKOFF", "2")
FAIL_FAST = os.environ.get("STAGE5_CAPABILITY_LADDER_TRACE_RESPONSE_FAIL_FAST", "1").strip().lower() in {
    "1",
    "true",
    "yes",
    "y",
}
RESUME = os.environ.get("STAGE5_CAPABILITY_LADDER_TRACE_RESPONSE_RESUME", "1").strip().lower() in {
    "1",
    "true",
    "yes",
    "y",
}
PUSH_RESULTS = os.environ.get("STAGE5_CAPABILITY_LADDER_TRACE_RESPONSE_PUSH", "1").strip().lower() in {
    "1",
    "true",
    "yes",
    "y",
}
COMMIT_RESPONSES = os.environ.get(
    "STAGE5_CAPABILITY_LADDER_TRACE_RESPONSE_COMMIT_RESPONSES",
    "1",
).strip().lower() in {"1", "true", "yes", "y"}
BACKUP_DRIVE = os.environ.get("STAGE5_CAPABILITY_LADDER_TRACE_RESPONSE_BACKUP_DRIVE", "1").strip().lower() in {
    "1",
    "true",
    "yes",
    "y",
}
REFUSE_GPU_RUNTIME = os.environ.get("STAGE5_CAPABILITY_LADDER_TRACE_RESPONSE_REFUSE_GPU", "1").strip().lower() in {
    "1",
    "true",
    "yes",
    "y",
}
ALLOW_GPU_RUNTIME_FOR_CPU_WORK = os.environ.get(
    "STAGE5_CAPABILITY_LADDER_TRACE_RESPONSE_ALLOW_GPU",
    "0",
).strip().lower() in {"1", "true", "yes", "y"}


def path_for_cli(path: Path) -> str:
    try:
        return path.relative_to(ROOT).as_posix()
    except ValueError:
        return str(path)


def resolve_path(raw: str | Path) -> Path:
    path = Path(str(raw).replace("\\", "/"))
    return path if path.is_absolute() else ROOT / path


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
            "Refusing to run capability-ladder trace responses on an attached GPU runtime: "
            + "; ".join(gpus)
            + ". Switch to CPU runtime, or set STAGE5_CAPABILITY_LADDER_TRACE_RESPONSE_ALLOW_GPU=1 deliberately."
        )
    if gpus:
        print("GPU attached but CPU-work override is enabled:", gpus, flush=True)
    else:
        print("No GPU attached; good for provider/network trace-response collection.", flush=True)


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
            "Missing source summary. Set STAGE5_CAPABILITY_LADDER_TRACE_RESPONSE_SOURCE_SUMMARY "
            "or update config/stage5_current_source_summary.txt."
        )
    path = resolve_path(raw)
    if not path.exists():
        raise FileNotFoundError(path)
    return path


def artifact_path(payload: dict[str, Any], name: str) -> Path:
    artifacts = payload.get("artifacts") if isinstance(payload.get("artifacts"), dict) else {}
    raw = str(artifacts.get(name) or "").strip()
    if not raw:
        raise ValueError(f"Source summary does not contain artifacts.{name}.")
    return resolve_path(raw)


def drive_search_roots() -> list[Path]:
    drive = Path("/content/drive/MyDrive")
    if not drive.exists():
        return []
    return [
        drive / "recurrent-qwen-svgd" / "stage5_capability_ladder_trace_jobs",
        drive / "recurrent-qwen-svgd",
        drive / "recurrent-qwen-svgd-artifacts",
        drive / "recurrent-qwen-svgd-fresh",
    ]


def restore_if_missing(path: Path, *, filename: str, run_id_hint: str) -> dict[str, Any]:
    if path.exists():
        return {"restored": False, "path": path_for_cli(path), "source": "local"}
    candidates: list[Path] = []
    for root in drive_search_roots():
        if root.exists():
            candidates.extend(root.rglob(filename))
    candidates = sorted(candidates, key=lambda item: (run_id_hint not in str(item), len(str(item))))
    if not candidates:
        raise FileNotFoundError(f"Missing {path} and no Drive backup {filename} was found.")
    source = candidates[0]
    path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, path)
    return {"restored": True, "path": path_for_cli(path), "source": str(source), "candidates": len(candidates)}


def provider_config_ready() -> tuple[bool, str]:
    if BACKEND == "dry_run":
        return True, "dry_run"
    if not RUN_PROVIDER:
        return False, "Set STAGE5_CAPABILITY_LADDER_TRACE_RESPONSE_RUN_PROVIDER=1 before provider/API spend."
    if BACKEND == "openai_compatible" and not (MODEL_OVERRIDE or MODEL_MAP_JSON or MODEL_MAP_JSON_INLINE):
        return False, "Set STAGE5_CAPABILITY_LADDER_TRACE_RESPONSE_MODEL_OVERRIDE or MODEL_MAP_JSON."
    if BACKEND == "command" and not COMMAND:
        return False, "Set STAGE5_CAPABILITY_LADDER_TRACE_RESPONSE_COMMAND for backend=command."
    return True, "ready"


def inline_model_map_path() -> Path | None:
    if not MODEL_MAP_JSON_INLINE:
        return None
    try:
        payload = json.loads(MODEL_MAP_JSON_INLINE)
    except json.JSONDecodeError as exc:
        raise ValueError("STAGE5_CAPABILITY_LADDER_TRACE_RESPONSE_MODEL_MAP_JSON_INLINE is invalid JSON.") from exc
    if not isinstance(payload, dict) or not payload:
        raise ValueError("STAGE5_CAPABILITY_LADDER_TRACE_RESPONSE_MODEL_MAP_JSON_INLINE must be a non-empty JSON object.")
    if any(not isinstance(key, str) or not isinstance(value, str) for key, value in payload.items()):
        raise ValueError("Inline model map must map string logical model names to string provider model ids.")
    path = RUN_DIR / "model_map.json"
    write_json(path, payload)
    return path


def run_responses(jobs_jsonl: Path) -> tuple[Path, Path, dict[str, Any]]:
    responses_jsonl = RUN_DIR / "trace_responses.jsonl"
    report_json = RUN_DIR / "trace_response_report.json"
    model_map_path = resolve_path(MODEL_MAP_JSON) if MODEL_MAP_JSON else inline_model_map_path()
    cmd = [
        sys.executable,
        "training/run_curriculum_job_responses.py",
        "--jobs_jsonl",
        path_for_cli(jobs_jsonl),
        "--output_jsonl",
        path_for_cli(responses_jsonl),
        "--report_json",
        path_for_cli(report_json),
        "--backend",
        BACKEND,
        "--max_tokens",
        MAX_TOKENS,
        "--temperature",
        TEMPERATURE,
        "--timeout_sec",
        TIMEOUT_SEC,
        "--sleep_sec",
        SLEEP_SEC,
    ]
    if LIMIT:
        cmd.extend(["--limit", LIMIT])
    if RESUME:
        cmd.append("--resume")
    if FAIL_FAST:
        cmd.append("--fail_fast")
    if BACKEND == "openai_compatible":
        cmd.extend(
            [
                "--api_key_env",
                API_KEY_ENV,
                "--base_url",
                BASE_URL,
                "--max_retries",
                MAX_RETRIES,
                "--retry_sleep_sec",
                RETRY_SLEEP_SEC,
                "--retry_backoff",
                RETRY_BACKOFF,
            ]
        )
        if MODEL_OVERRIDE:
            cmd.extend(["--model_override", MODEL_OVERRIDE])
        if model_map_path is not None:
            cmd.extend(["--model_map_json", path_for_cli(model_map_path)])
    elif BACKEND == "command":
        cmd.extend(["--command", COMMAND])
        if MODEL_OVERRIDE:
            cmd.extend(["--model_override", MODEL_OVERRIDE])
        if model_map_path is not None:
            cmd.extend(["--model_map_json", path_for_cli(model_map_path)])
    run(cmd, check=False, log_name="run_curriculum_job_responses.log")
    return responses_jsonl, report_json, read_json(report_json)


def backup_to_drive(paths: list[Path]) -> dict[str, Any]:
    if not BACKUP_DRIVE:
        return {"enabled": False}
    drive_root = Path("/content/drive/MyDrive")
    if not drive_root.exists():
        return {"enabled": True, "available": False}
    dest_root = drive_root / "recurrent-qwen-svgd" / "stage5_capability_ladder_trace_responses" / RUN_ID
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


def safe_commit(summary_path: Path, *, update_pointer: bool, include_responses: bool) -> None:
    if not PUSH_RESULTS:
        return
    pointer = update_current_source_summary(summary_path) if update_pointer else None
    safe_paths = [
        summary_path,
        RUN_DIR / "summary.md",
        RUN_DIR / "trace_response_report.json",
        RUN_DIR / "model_map.json",
        pointer,
    ]
    if include_responses and COMMIT_RESPONSES:
        safe_paths.append(RUN_DIR / "trace_responses.jsonl")
    for path in safe_paths:
        if path is not None and path.exists():
            run(["git", "add", "-f", path_for_cli(path)], check=False)
    diff = run(["git", "diff", "--cached", "--quiet"], check=False)
    if diff.returncode == 0:
        print("No safe trace-response result changes to commit.", flush=True)
        return
    run(["git", "commit", "-m", "Record capability-ladder trace responses"], check=True)
    push = run(["git", "push", "origin", "main"], check=False)
    if push.returncode == 0:
        return
    run(["git", "pull", "--rebase", "origin", "main"], check=True)
    run(["git", "push", "origin", "main"], check=True)


def response_status(report: dict[str, Any]) -> str:
    if BACKEND == "dry_run":
        return "dry_run"
    if int(report.get("errors") or 0) or int(report.get("timeouts") or 0):
        return "responses_need_review"
    if int(report.get("written") or 0) or int(report.get("skipped") or 0):
        return "responses_ready"
    return "responses_empty"


def write_summary(
    *,
    source_summary: Path,
    jobs_jsonl: Path,
    jobs_report: Path,
    restore_report: dict[str, Any],
    responses_jsonl: Path,
    response_report_json: Path,
    response_report: dict[str, Any],
    drive_backup: dict[str, Any],
) -> Path:
    status = response_status(response_report)
    payload = {
        "run_id": RUN_ID,
        "kind": "stage5_capability_ladder_trace_responses",
        "status": status,
        "source_summary": path_for_cli(source_summary),
        "backend": BACKEND,
        "run_provider": RUN_PROVIDER,
        "restore": restore_report,
        "response_report": response_report,
        "artifacts": {
            "jobs_jsonl": path_for_cli(jobs_jsonl),
            "jobs_report": path_for_cli(jobs_report),
            "responses_jsonl": path_for_cli(responses_jsonl),
            "response_report_json": path_for_cli(response_report_json),
            "run_dir": path_for_cli(RUN_DIR),
        },
        "drive_backup": drive_backup,
        "next_action": (
            "Run capability_ladder_trace_collect_cpu to verify final answers and build traced curriculum rows."
            if status == "responses_ready"
            else "Inspect provider response report before trace collection."
        ),
    }
    summary = RUN_DIR / "summary.json"
    write_json(summary, payload)
    (RUN_DIR / "summary.md").write_text(markdown_summary(payload), encoding="utf-8")
    return summary


def markdown_summary(payload: dict[str, Any]) -> str:
    report = payload["response_report"]
    return "\n".join(
        [
            f"# Capability-Ladder Trace Responses: {payload['run_id']}",
            "",
            f"- Status: `{payload['status']}`",
            f"- Backend: `{payload['backend']}`",
            f"- Selected jobs: `{report.get('selected')}`",
            f"- Written: `{report.get('written')}`",
            f"- Skipped: `{report.get('skipped')}`",
            f"- Errors: `{report.get('errors')}`",
            f"- Timeouts: `{report.get('timeouts')}`",
            f"- Responses JSONL: `{payload['artifacts']['responses_jsonl']}`",
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
    source_payload = read_json(source_summary)
    run_id_hint = str(source_payload.get("run_id") or source_summary.parent.name)
    jobs_jsonl = artifact_path(source_payload, "jobs_jsonl")
    jobs_report = artifact_path(source_payload, "report_json")
    restore_report = {
        "jobs": restore_if_missing(jobs_jsonl, filename=jobs_jsonl.name, run_id_hint=run_id_hint),
        "report": restore_if_missing(jobs_report, filename=jobs_report.name, run_id_hint=run_id_hint),
    }
    ready, reason = provider_config_ready()
    if not ready:
        raise RuntimeError(reason)
    responses_jsonl, response_report_json, response_report = run_responses(jobs_jsonl)
    drive_backup = backup_to_drive([RUN_DIR, responses_jsonl, response_report_json])
    summary = write_summary(
        source_summary=source_summary,
        jobs_jsonl=jobs_jsonl,
        jobs_report=jobs_report,
        restore_report=restore_report,
        responses_jsonl=responses_jsonl,
        response_report_json=response_report_json,
        response_report=response_report,
        drive_backup=drive_backup,
    )
    print(f"summary_json={path_for_cli(summary)}", flush=True)
    status = str(read_json(summary)["status"])
    print(f"status={status}", flush=True)
    safe_commit(summary, update_pointer=status != "dry_run", include_responses=status == "responses_ready")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
